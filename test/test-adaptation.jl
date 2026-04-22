@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Adaptation" begin

    @testset "WelfordBoomerangStats basic" begin
        d = 3
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        @test ws.total_time == 0.0
        @test !ws.initialized
        @test length(ws.sum_x_dt) == d

        # Before data, stats should return defaults
        μ = PDMPSamplers.stats_mean(ws)
        @test all(μ .== 0.0)
        v = PDMPSamplers.stats_var(ws)
        @test all(v .== 1.0)
    end

    @testset "WelfordBoomerangStats diagonal updates" begin
        d = 3
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        θ0 = zeros(d)

        # First call initializes
        x1 = [1.0, 2.0, 3.0]
        PDMPSamplers.welford_update!(ws, x1, θ0, 0.0, flow)
        @test ws.initialized
        @test ws.total_time == 0.0

        # Second call accumulates
        x2 = [1.5, 2.5, 3.5]
        PDMPSamplers.welford_update!(ws, x2, θ0, 1.0, flow)
        @test ws.total_time == 1.0

        μ = PDMPSamplers.stats_mean(ws)
        @test all(isfinite, μ)

        v = PDMPSamplers.stats_var(ws)
        @test all(v .>= 0)

        # dt = 0 → no change
        old_time = ws.total_time
        PDMPSamplers.welford_update!(ws, x2, θ0, 1.0, flow)
        @test ws.total_time == old_time
    end

    @testset "WelfordBoomerangStats fullrank mode" begin
        d = 4
        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        @test ws.sum_xy_dt !== nothing
        flow = AdaptiveBoomerang(d; scheme=:fullrank)

        Random.seed!(42)
        t = 0.0
        for _ in 1:100
            x = randn(d)
            θ = randn(d)
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, θ, t, flow)
        end

        C = PDMPSamplers.stats_cov(ws)
        @test size(C) == (d, d)
        @test issymmetric(C)
        @test all(diag(C) .>= 0)
    end

    @testset "WelfordBoomerangStats stats_cov errors without fullrank" begin
        ws = PDMPSamplers.WelfordBoomerangStats(3)
        @test_throws ErrorException PDMPSamplers.stats_cov(ws)
    end

    @testset "WelfordBoomerangStats stats_std" begin
        d = 3
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        θ0 = zeros(d)
        @test all(PDMPSamplers.stats_std(ws) .== 1.0)

        PDMPSamplers.welford_update!(ws, [2.0, 4.0, 6.0], θ0, 0.0, flow)
        PDMPSamplers.welford_update!(ws, [2.0, 4.0, 6.0], θ0, 1.0, flow)
        σ = PDMPSamplers.stats_std(ws)
        @test all(σ .>= 0)
        @test σ ≈ sqrt.(PDMPSamplers.stats_var(ws))
    end

    @testset "_coord_time for WelfordBoomerangStats" begin
        d = 3
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        θ0 = zeros(d)
        @test PDMPSamplers._coord_time(ws, 1) == 0.0

        PDMPSamplers.welford_update!(ws, ones(d), θ0, 0.0, flow)
        PDMPSamplers.welford_update!(ws, ones(d), θ0, 5.0, flow)
        @test PDMPSamplers._coord_time(ws, 1) == 5.0
        @test PDMPSamplers._coord_time(ws, 2) == 5.0
    end

    @testset "adapt_interval geometric growth" begin
        base_dt = 1.0
        @test PDMPSamplers.adapt_interval(0, base_dt) == 1.0
        @test PDMPSamplers.adapt_interval(3, base_dt) == 2.0
        @test PDMPSamplers.adapt_interval(6, base_dt) == 4.0
        # Capped at 32x
        @test PDMPSamplers.adapt_interval(100, base_dt) == 32.0
    end

    @testset "update_boomerang! diagonal with WelfordBoomerangStats" begin
        d = 4
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        ws = PDMPSamplers.WelfordBoomerangStats(d)

        # No data → no-op
        PDMPSamplers.update_boomerang!(flow, ws, Val(:diagonal), nothing)
        @test all(flow.μ .== 0.0)

        # Feed data
        Random.seed!(42)
        t = 0.0
        θ0 = zeros(d)
        for _ in 1:200
            x = randn(d) .+ [1.0, 2.0, 3.0, 4.0]
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, θ0, t, flow)
        end

        PDMPSamplers.update_boomerang!(flow, ws, Val(:diagonal), nothing)
        @test flow.μ ≈ [1.0, 2.0, 3.0, 4.0] atol = 0.5
        @test all(diag(flow.Γ) .> 0)
    end

    @testset "update_boomerang! fullrank with WelfordBoomerangStats" begin
        d = 4
        flow = AdaptiveBoomerang(d; scheme=:fullrank)
        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)

        Random.seed!(42)
        μ_true = [1.0, 2.0, 3.0, 4.0]
        t = 0.0
        θ0 = zeros(d)
        for _ in 1:500
            x = randn(d) .+ μ_true
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, θ0, t, flow)
        end

        PDMPSamplers.update_boomerang!(flow, ws, Val(:fullrank), PDMPSamplers.FullrankWorkspace(d))
        @test flow.μ ≈ μ_true atol = 0.5
        @test all(diag(Matrix(flow.Γ)) .> 0)
    end

    @testset "update_boomerang! lowrank" begin
        d = 5
        flow = AdaptiveBoomerang(d; scheme=:lowrank, rank=2)
        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)

        Random.seed!(42)
        t = 0.0
        θ0 = zeros(d)
        for _ in 1:500
            x = randn(d)
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, θ0, t, flow)
        end

        PDMPSamplers.update_boomerang!(flow, ws, Val(:lowrank), PDMPSamplers.FullrankWorkspace(d))
        lrp = flow.Γ::PDMPSamplers.LowRankPrecision
        @test all(lrp.Λ .> 0)
        @test all(lrp.D .> 0)
    end

    @testset "_fullrank_diagonal_fallback!" begin
        d = 3
        flow = AdaptiveBoomerang(d; scheme=:fullrank)
        σ2_diag = [0.5, 1.0, 2.0]

        PDMPSamplers._fullrank_diagonal_fallback!(flow, σ2_diag)
        for i in 1:d
            @test flow.Γ[i, i] ≈ 1.0 / σ2_diag[i]
        end
    end

    @testset "_has_data and _shrinkage_time" begin
        ws = PDMPSamplers.WelfordBoomerangStats(3)
        flow = AdaptiveBoomerang(3; scheme=:diagonal)
        θ0 = zeros(3)
        @test !PDMPSamplers._has_data(ws)
        @test PDMPSamplers._shrinkage_time(ws) == 0.0

        PDMPSamplers.welford_update!(ws, [1.0, 2.0, 3.0], θ0, 0.0, flow)
        PDMPSamplers.welford_update!(ws, [1.0, 2.0, 3.0], θ0, 1.0, flow)
        @test PDMPSamplers._has_data(ws)
        @test PDMPSamplers._shrinkage_time(ws) == 1.0
    end

    @testset "did_dynamics_adapt" begin
        @test PDMPSamplers.did_dynamics_adapt(PDMPSamplers.NoAdaptation()) == false

        d = 3
        ba = PDMPSamplers.BoomerangAdapter(1.0, 0.0, d; scheme=:diagonal)
        @test PDMPSamplers.did_dynamics_adapt(ba) == false

        ba.did_update = true
        @test PDMPSamplers.did_dynamics_adapt(ba) == true

        seq = PDMPSamplers.SequenceAdapter((PDMPSamplers.NoAdaptation(), ba))
        @test PDMPSamplers.did_dynamics_adapt(seq) == true
    end

    @testset "NoAdaptation" begin
        na = PDMPSamplers.NoAdaptation()
        @test PDMPSamplers.adapt!(na) === nothing
    end

    @testset "default_adapter fallbacks" begin
        zz = ZigZag(3)
        grad = FullGradient(x -> x)

        result = PDMPSamplers.default_adapter(zz, grad)
        @test result isa PDMPSamplers.NoAdaptation
    end

    @testset "default_adapter for MutableBoomerang + SubsampledGradient" begin
        d = 3
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        grad_sub = SubsampledGradient(
            (out, x) -> (out .= x),
            n -> nothing,
            tr -> nothing,
            (out, x) -> (out .= x),
            5,
            2,
            false;
            resample_dt=0.25,
        )

        adapter = PDMPSamplers.default_adapter(flow, grad_sub, 6.0, 40.0, 1.5)
        @test adapter isa PDMPSamplers.SequenceAdapter

        ad_flow, ad_grad = adapter.adapters
        @test ad_flow isa PDMPSamplers.BoomerangAdapter
        @test ad_flow.base_dt == 6.0
        @test ad_flow.last_update == 1.5
        @test ad_grad isa PDMPSamplers.SequenceAdapter
    end

    @testset "default_dynamics_adapter for PreconditionedDynamics" begin
        zz = ZigZag(3)
        precond = PDMPSamplers.PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(), zz)
        ad = PDMPSamplers.default_dynamics_adapter(precond, 10.0, 0.0)
        @test ad isa PDMPSamplers.PreconditionerAdapter
        @test ad.dt == 10.0
    end

    @testset "default_dynamics_adapter for MutableBoomerang" begin
        flow_diag = AdaptiveBoomerang(3; scheme=:diagonal)
        ad_diag = PDMPSamplers.default_dynamics_adapter(flow_diag, 5.0, 0.0)
        @test ad_diag isa PDMPSamplers.BoomerangAdapter
        @test ad_diag.scheme == :diagonal

        flow_full = AdaptiveBoomerang(4; scheme=:fullrank)
        ad_full = PDMPSamplers.default_dynamics_adapter(flow_full, 5.0, 0.0)
        @test ad_full.scheme == :fullrank

        flow_lr = AdaptiveBoomerang(5; scheme=:lowrank, rank=2)
        ad_lr = PDMPSamplers.default_dynamics_adapter(flow_lr, 5.0, 0.0)
        @test ad_lr.scheme == :lowrank
    end

    @testset "BoomerangAdapter construction" begin
        d = 4
        ad = PDMPSamplers.BoomerangAdapter(2.0, 1.0, d; scheme=:fullrank)
        @test ad.base_dt == 2.0
        @test ad.last_update == 1.0
        @test ad.no_updates_done == 0
        @test ad.scheme == :fullrank
        @test ad.stats isa PDMPSamplers.WelfordBoomerangStats
        @test ad.stats.sum_xy_dt !== nothing
    end

    @testset "Allocation-free stats helpers" begin
        d = 3

        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        μ = zeros(d)
        C = zeros(d, d)
        for _ in 1:5
            PDMPSamplers.stats_mean!(μ, ws)
            PDMPSamplers.stats_cov!(C, ws)
        end

        alloc_mean = minimum((@allocated PDMPSamplers.stats_mean!(μ, ws)) for _ in 1:10)
        alloc_cov = minimum((@allocated PDMPSamplers.stats_cov!(C, ws)) for _ in 1:10)
        @test alloc_mean == 0
        @test alloc_cov == 0
    end

    @testset "BoomerangAdapter fallback for non-MutableBoomerang" begin
        d = 3
        ba = PDMPSamplers.BoomerangAdapter(1.0, 0.0, d)
        boom = Boomerang(d)
        # adapt! on non-MutableBoomerang should be a no-op
        PDMPSamplers.adapt!(ba, nothing, boom, nothing, nothing)
        @test ba.did_update == false
    end

    @testset "End-to-end adaptive Boomerang diagonal" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Σ_true = cov(target.D)

        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        m = mean(trace)
        @test maximum(abs.(m .- μ_true)) ≤ 1.0
    end

    @testset "End-to-end adaptive Boomerang fullrank" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)

        flow = AdaptiveBoomerang(d; scheme=:fullrank)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        m = mean(trace)
        @test maximum(abs.(m .- μ_true)) ≤ 1.0
    end

    @testset "End-to-end adaptive Boomerang lowrank" begin
        d = 5
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)

        flow = AdaptiveBoomerang(d; scheme=:lowrank, rank=2)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        m = mean(trace)
        @test maximum(abs.(m .- μ_true)) ≤ 3.0
    end

    @testset "SequenceAdapter adapt!" begin
        na1 = PDMPSamplers.NoAdaptation()
        na2 = PDMPSamplers.NoAdaptation()
        seq = PDMPSamplers.SequenceAdapter((na1, na2))
        PDMPSamplers.adapt!(seq, nothing, nothing, nothing, nothing)
    end

    @testset "AnchorBankAdapter adapt! — warmup phase triggers update" begin
        d = 3
        x = [1.0, 2.0, 3.0]
        state = PDMPState(10.0, SkeletonPoint(x, zeros(d)))
        trace_mgr = PDMPSamplers.TraceManager(nothing, nothing, 100.0)

        select_called = Ref(0)
        update_trace = Ref{Any}(nothing)
        ad = AnchorBankAdapter(
            _ -> (select_called[] += 1),
            tr -> (update_trace[] = tr),
            5.0, 0.0, true,
        )

        PDMPSamplers.adapt!(ad, state, nothing, nothing, trace_mgr; phase=:warmup)

        @test select_called[] == 1
        @test update_trace[] === nothing   # get_warmup_trace returns nothing
        @test ad.last_update == 0.0
    end

    @testset "AnchorBankAdapter adapt! — main phase with warmup_only=false triggers update" begin
        d = 3
        x = [1.0, 2.0, 3.0]
        state = PDMPState(10.0, SkeletonPoint(x, zeros(d)))
        trace_mgr = PDMPSamplers.TraceManager(nothing, nothing, 100.0)

        update_called = Ref(0)
        ad = AnchorBankAdapter(
            _ -> nothing,
            _ -> (update_called[] += 1),
            5.0, 0.0, false,
        )

        PDMPSamplers.adapt!(ad, state, nothing, nothing, trace_mgr; phase=:main)

        @test update_called[] == 0
        @test ad.last_update == 0.0
    end

    @testset "AnchorBankAdapter adapt! — main phase with warmup_only=true skips update" begin
        d = 3
        x = [1.0, 2.0, 3.0]
        state = PDMPState(10.0, SkeletonPoint(x, zeros(d)))
        trace_mgr = PDMPSamplers.TraceManager(nothing, nothing, 100.0)

        update_called = Ref(0)
        ad = AnchorBankAdapter(
            _ -> nothing,
            _ -> (update_called[] += 1),
            5.0, 0.0, true,
        )

        PDMPSamplers.adapt!(ad, state, nothing, nothing, trace_mgr; phase=:main)

        @test update_called[] == 0
        @test ad.last_update == 0.0
    end

    @testset "AnchorBankAdapter adapt! — update not triggered when interval not elapsed" begin
        d = 3
        x = [1.0, 2.0, 3.0]
        state = PDMPState(10.0, SkeletonPoint(x, zeros(d)))
        trace_mgr = PDMPSamplers.TraceManager(nothing, nothing, 100.0)

        select_called = Ref(0)
        update_called = Ref(0)
        ad = AnchorBankAdapter(
            _ -> (select_called[] += 1),
            _ -> (update_called[] += 1),
            5.0, 8.0, true,   # 10.0 - 8.0 = 2.0 < 5.0
        )

        PDMPSamplers.adapt!(ad, state, nothing, nothing, trace_mgr; phase=:warmup)

        @test select_called[] == 1   # select always runs
        @test update_called[] == 0   # update not triggered
        @test ad.last_update == 8.0  # unchanged
    end

    @testset "AnchorUpdater adapt! — warmup triggers update_anchor!" begin
        d = 3
        state = PDMPState(10.0, SkeletonPoint(ones(d), zeros(d)))
        fake_trace = [1, 2]  # iterable with ≥2 elements
        trace_mgr = PDMPSamplers.TraceManager(fake_trace, fake_trace, 100.0)

        anchor_called = Ref(0)
        grad = (; update_anchor! = _ -> (anchor_called[] += 1))
        ad = PDMPSamplers.AnchorUpdater(5.0, 0.0)

        PDMPSamplers.adapt!(ad, state, nothing, grad, trace_mgr; phase=:warmup)
        @test anchor_called[] == 1
        @test ad.last_update == 10.0
    end

    @testset "AnchorUpdater adapt! — not triggered when interval not elapsed" begin
        d = 3
        state = PDMPState(10.0, SkeletonPoint(ones(d), zeros(d)))
        fake_trace = [1, 2]
        trace_mgr = PDMPSamplers.TraceManager(fake_trace, fake_trace, 100.0)

        anchor_called = Ref(0)
        grad = (; update_anchor! = _ -> (anchor_called[] += 1))
        ad = PDMPSamplers.AnchorUpdater(5.0, 8.0)  # 10.0 - 8.0 = 2.0 < 5.0

        PDMPSamplers.adapt!(ad, state, nothing, grad, trace_mgr; phase=:warmup)
        @test anchor_called[] == 0
        @test ad.last_update == 8.0
    end

    @testset "AnchorUpdater adapt! — main phase with warmup_only=false" begin
        d = 3
        state = PDMPState(10.0, SkeletonPoint(ones(d), zeros(d)))
        fake_trace = [1, 2]
        trace_mgr = PDMPSamplers.TraceManager(fake_trace, fake_trace, 100.0)

        anchor_called = Ref(0)
        grad = (; update_anchor! = _ -> (anchor_called[] += 1))
        ad = PDMPSamplers.AnchorUpdater(5.0, 0.0, false)

        PDMPSamplers.adapt!(ad, state, nothing, grad, trace_mgr; phase=:main)
        @test anchor_called[] == 1
        @test ad.last_update == 10.0
    end

    @testset "AnchorUpdater adapt! — main phase skipped when warmup_only=true" begin
        d = 3
        state = PDMPState(10.0, SkeletonPoint(ones(d), zeros(d)))
        fake_trace = [1, 2]
        trace_mgr = PDMPSamplers.TraceManager(fake_trace, fake_trace, 100.0)

        anchor_called = Ref(0)
        grad = (; update_anchor! = _ -> (anchor_called[] += 1))
        ad = PDMPSamplers.AnchorUpdater(5.0, 0.0, true)

        PDMPSamplers.adapt!(ad, state, nothing, grad, trace_mgr; phase=:main)
        @test anchor_called[] == 0
        @test ad.last_update == 0.0
    end

    @testset "default_adapter for MutableBoomerang + SubsampledGradient + AnchorBankAdapter" begin
        d = 3
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        grad_sub = SubsampledGradient(
            (out, x) -> (out .= x),
            n -> nothing,
            tr -> nothing,
            (out, x) -> (out .= x),
            5,
            2,
            false;
            resample_dt=0.25,
        )
        bank_adapter = AnchorBankAdapter(_ -> nothing, _ -> nothing, 0.0, 0.0, true)

        adapter = PDMPSamplers.default_adapter(flow, grad_sub, bank_adapter, 6.0, 40.0, 1.5)

        @test adapter isa PDMPSamplers.SequenceAdapter
        ad_flow, ad_resampler, ad_bank = adapter.adapters
        @test ad_flow isa PDMPSamplers.BoomerangAdapter
        @test ad_resampler isa PDMPSamplers.GradientResampler
        @test ad_bank === bank_adapter
        @test bank_adapter.update_dt ≈ 40.0 / 2   # t_warmup / no_anchor_updates
        @test bank_adapter.last_update == 1.5
    end
end
