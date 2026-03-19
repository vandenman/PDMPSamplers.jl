@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Adaptation" begin

    @testset "WelfordBoomerangStats basic" begin
        d = 3
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        @test ws.total_time == 0.0
        @test !ws.initialized
        @test length(ws.sum_x_lin) == d

        # Before data, stats should return defaults
        μ = PDMPSamplers.stats_mean(ws)
        @test all(μ .== 0.0)
        v = PDMPSamplers.stats_var(ws)
        @test all(v .== 1.0)
    end

    @testset "WelfordBoomerangStats diagonal updates" begin
        d = 3
        ws = PDMPSamplers.WelfordBoomerangStats(d)

        # First call initializes
        x1 = [1.0, 2.0, 3.0]
        PDMPSamplers.welford_update!(ws, x1, 0.0)
        @test ws.initialized
        @test ws.total_time == 0.0

        # Second call accumulates
        x2 = [1.5, 2.5, 3.5]
        PDMPSamplers.welford_update!(ws, x2, 1.0)
        @test ws.total_time == 1.0

        μ = PDMPSamplers.stats_mean(ws)
        @test all(isfinite, μ)

        v = PDMPSamplers.stats_var(ws)
        @test all(v .>= 0)

        # dt = 0 → no change
        old_time = ws.total_time
        PDMPSamplers.welford_update!(ws, x2, 1.0)
        @test ws.total_time == old_time
    end

    @testset "WelfordBoomerangStats fullrank mode" begin
        d = 4
        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        @test ws.sum_xy !== nothing

        Random.seed!(42)
        t = 0.0
        for _ in 1:100
            x = randn(d)
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, t)
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

    @testset "BoomerangWarmupStats basic" begin
        d = 3
        stats = PDMPSamplers.BoomerangWarmupStats(d)
        @test length(stats.coord_time) == d
        @test stats.cursor == 0

        μ = PDMPSamplers.stats_mean(stats)
        @test all(μ .== 0.0)

        v = PDMPSamplers.stats_var(stats)
        @test all(v .== 1.0)

        σ = PDMPSamplers.stats_std(stats)
        @test all(σ .== 1.0)
    end

    @testset "BoomerangWarmupStats fullrank stats_cov" begin
        d = 3
        stats = PDMPSamplers.BoomerangWarmupStats(d; fullrank=true)
        @test stats.sum_xy !== nothing

        # Simulate some data
        stats.coord_time .= 10.0
        stats.sum_x .= [5.0, 10.0, 15.0]
        stats.sum_x2 .= [30.0, 110.0, 230.0]
        stats.sum_xy .= [30.0 55.0 80.0; 55.0 110.0 165.0; 80.0 165.0 230.0]

        C = PDMPSamplers.stats_cov(stats)
        @test size(C) == (d, d)
        @test issymmetric(C)
    end

    @testset "BoomerangWarmupStats stats_cov errors without fullrank" begin
        stats = PDMPSamplers.BoomerangWarmupStats(3)
        @test_throws ErrorException PDMPSamplers.stats_cov(stats)
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
        for _ in 1:200
            x = randn(d) .+ [1.0, 2.0, 3.0, 4.0]
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, t)
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
        for _ in 1:500
            x = randn(d) .+ μ_true
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, t)
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
        for _ in 1:500
            x = randn(d)
            t += 0.1
            PDMPSamplers.welford_update!(ws, x, t)
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
        @test !PDMPSamplers._has_data(ws)
        @test PDMPSamplers._shrinkage_time(ws) == 0.0

        PDMPSamplers.welford_update!(ws, [1.0, 2.0, 3.0], 0.0)
        PDMPSamplers.welford_update!(ws, [1.0, 2.0, 3.0], 1.0)
        @test PDMPSamplers._has_data(ws)
        @test PDMPSamplers._shrinkage_time(ws) == 1.0

        bws = PDMPSamplers.BoomerangWarmupStats(3)
        @test !PDMPSamplers._has_data(bws)
        @test PDMPSamplers._shrinkage_time(bws) == 0.0

        bws.coord_time .= [1.0, 2.0, 3.0]
        @test PDMPSamplers._has_data(bws)
        @test PDMPSamplers._shrinkage_time(bws) == 3.0
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
        @test ad.stats.sum_xy !== nothing
    end

    @testset "Allocation-free stats helpers" begin
        d = 3

        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        μ = zeros(d)
        C = zeros(d, d)
        PDMPSamplers.stats_mean!(μ, ws)
        PDMPSamplers.stats_cov!(C, ws)

        alloc_mean_welford = @allocated PDMPSamplers.stats_mean!(μ, ws)
        alloc_cov_welford = @allocated PDMPSamplers.stats_cov!(C, ws)
        @test alloc_mean_welford == 0
        @test alloc_cov_welford == 0

        warm = PDMPSamplers.BoomerangWarmupStats(d; fullrank=true)
        μ2 = zeros(d)
        C2 = zeros(d, d)
        PDMPSamplers.stats_mean!(μ2, warm)
        PDMPSamplers.stats_cov!(C2, warm)

        alloc_mean_warm = @allocated PDMPSamplers.stats_mean!(μ2, warm)
        alloc_cov_warm = @allocated PDMPSamplers.stats_cov!(C2, warm)
        @test alloc_mean_warm == 0
        @test alloc_cov_warm == 0
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
end
