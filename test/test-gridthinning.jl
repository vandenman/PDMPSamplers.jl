@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

import DifferentiationInterface as DI

@testset "Grid thinning" begin

    @testset "PiecewiseConstantBound functor" begin
        t_grid = [0.0, 1.0, 2.0, 3.0]
        Λ_vals = [1.5, 2.0, 0.5]
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, Λ_vals)

        @test pcb(0.5) == 1.5
        @test pcb(1.5) == 2.0
        @test pcb(2.5) == 0.5
        # Outside bounds
        @test pcb(-0.5) == 0.0
        @test pcb(3.5) == 0.0
    end

    @testset "recompute_time_grid!" begin
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [1.0])
        PDMPSamplers.recompute_time_grid!(pcb, 5.0, 10)
        @test length(pcb.t_grid) == 11
        @test pcb.t_grid[1] ≈ 0.0
        @test pcb.t_grid[end] ≈ 5.0
        @test length(pcb.Λ_vals) == 10
    end

    @testset "propose_event_time" begin
        t_grid = [0.0, 1.0, 2.0, 3.0]
        Λ_vals = [2.0, 3.0, 1.0]
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, Λ_vals)

        # Small u → event in first segment
        τ, lb = PDMPSamplers.propose_event_time(pcb, 0.5, 0.0)
        @test 0.0 ≤ τ ≤ 1.0
        @test lb == 2.0

        # Large u → event in late segments or beyond
        τ2, lb2 = PDMPSamplers.propose_event_time(pcb, 100.0, 0.0)
        @test isinf(τ2)

        # With refresh_rate: cell 0 has Λ=2.0, total rate = 3.0
        # u=0.5 should give τ = 0.5 / 3.0 ≈ 0.1667
        τ3, lb3 = PDMPSamplers.propose_event_time(pcb, 0.5, 1.0)
        @test τ3 ≈ 0.5 / 3.0 atol=1e-14
        @test lb3 ≈ 3.0

        # refresh_rate with zero Λ_val: avoid divide-by-zero
        pcb_zero = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [0.0])
        τ4, lb4 = PDMPSamplers.propose_event_time(pcb_zero, 0.25, 1.0)
        @test τ4 ≈ 0.25 / 1.0 atol=1e-14
        @test lb4 ≈ 1.0
    end

    @testset "_compute_cell_bound!" begin
        t_grid = [0.0, 1.0]
        y_vals = [2.0, 3.0]
        d_vals = [1.0, -1.0]
        Λ_vals = [0.0]

        PDMPSamplers._compute_cell_bound!(Λ_vals, t_grid, y_vals, d_vals, 1)
        @test Λ_vals[1] >= max(y_vals[1], y_vals[2])

        # Equal derivatives → use yᵢ directly
        d_vals2 = [1.0, 1.0]
        Λ_vals2 = [0.0]
        PDMPSamplers._compute_cell_bound!(Λ_vals2, t_grid, y_vals, d_vals2, 1)
        @test Λ_vals2[1] >= max(y_vals[1], y_vals[2])
    end

    @testset "get_rate_and_deriv with Tuple{G,Nothing} (no HVP)" begin
        d = 3
        target = gen_data(Distributions.MvNormal, d, 2.0)
        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end

        rate, deriv = PDMPSamplers.get_rate_and_deriv(state, flow, (grad_func, nothing))
        @test rate >= 0
        @test deriv == 0.0  # no HVP → zero derivative
    end

    @testset "get_rate_and_deriv with FiniteDiffHVP" begin
        d = 3
        target = gen_data(Distributions.MvNormal, d, 2.0)
        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end

        fd_hvp = PDMPSamplers.FiniteDiffHVP(grad_func, zeros(d))
        rate_fd, deriv_fd = PDMPSamplers.get_rate_and_deriv(state, flow, fd_hvp)
        @test rate_fd >= 0
        @test isfinite(deriv_fd)

        # Compare with exact HVP
        hvp_func = (x, v) -> begin
            out = similar(x)
            neg_hvp!(target, out, x, v)
            out
        end
        rate_exact, deriv_exact = PDMPSamplers.get_rate_and_deriv(state, flow, (grad_func, hvp_func))
        @test rate_fd ≈ rate_exact
        @test isapprox(deriv_fd, deriv_exact; rtol=0.01)
    end

    @testset "FiniteDiffVHV with gridthinning fallback" begin
        d = 3
        target = gen_data(Distributions.MvNormal, d, 2.0)
        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end

        fd_vhv = PDMPSamplers.FiniteDiffVHV(grad_func, zeros(d))
        rate_fd, deriv_fd = PDMPSamplers.get_rate_and_deriv(state, flow, fd_vhv)
        @test rate_fd >= 0
        @test isfinite(deriv_fd)

        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP/VHV so gridthinning uses finite-diff fallback
        alg = GridThinningStrategy(; use_fd_hvp=true, N=16, t_max=1.5)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 500.0; progress=show_progress, statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test length(trace) > 20
        @test all(isfinite, mean(trace))
    end

    @testset "∂λ∂t for different flow types" begin
        d = 3
        Random.seed!(42)

        ξ = SkeletonPoint(randn(d), randn(d))
        state = PDMPState(0.0, ξ)
        ∇U = randn(d)
        Hv = randn(d)

        # Generic ContinuousDynamics (BouncyParticle)
        bps = BouncyParticle(d, 1.0)
        result_bps = PDMPSamplers.∂λ∂t(state, ∇U, Hv, bps)
        @test result_bps ≈ dot(ξ.θ, Hv)

        # ZigZag
        zz = ZigZag(d)
        result_zz = PDMPSamplers.∂λ∂t(state, ∇U, Hv, zz)
        @test isfinite(result_zz)

        # Boomerang
        boom = Boomerang(d)
        result_boom = PDMPSamplers.∂λ∂t(state, ∇U, Hv, boom)
        @test isfinite(result_boom)

        # MutableBoomerang
        aboom = AdaptiveBoomerang(d)
        result_aboom = PDMPSamplers.∂λ∂t(state, ∇U, Hv, aboom)
        @test isfinite(result_aboom)

        # LowRankMutableBoomerang
        lr_boom = AdaptiveBoomerang(d; scheme=:lowrank, rank=2)
        result_lr = PDMPSamplers.∂λ∂t(state, ∇U, Hv, lr_boom)
        @test isfinite(result_lr)
    end

    @testset "∂λ∂t Boomerang with StickyPDMPState" begin
        d = 3
        boom = Boomerang(d)
        ξ = SkeletonPoint(randn(d), randn(d))
        free = BitVector([true, false, true])
        state = StickyPDMPState(0.0, ξ, free, randn(d))
        ∇U = randn(d)
        Hv = randn(d)
        result = PDMPSamplers.∂λ∂t(state, ∇U, Hv, boom)
        @test isfinite(result)
    end

    @testset "∂λ∂t LowRankMutableBoomerang with StickyPDMPState" begin
        d = 3
        lr_boom = AdaptiveBoomerang(d; scheme=:lowrank, rank=2)
        ξ = SkeletonPoint(randn(d), randn(d))
        free = BitVector([true, false, true])
        state = StickyPDMPState(0.0, ξ, free, randn(d))
        ∇U = randn(d)
        Hv = randn(d)
        result = PDMPSamplers.∂λ∂t(state, ∇U, Hv, lr_boom)
        @test isfinite(result)
    end

    @testset "min_grid_cells and max_grid_horizon" begin
        zz = ZigZag(3)
        bps = BouncyParticle(3, 1.0)
        boom = Boomerang(3)

        @test PDMPSamplers.min_grid_cells(zz, 5, 20) == 5
        @test PDMPSamplers.min_grid_cells(boom, 5, 20) == 5
        @test PDMPSamplers.min_grid_cells(boom, 15, 20) == 15

        @test PDMPSamplers.max_grid_horizon(zz) == 1e10
        @test PDMPSamplers.max_grid_horizon(boom) ≈ 8π
    end

    @testset "_adapt_grid_N! and _increase_grid_N!" begin
        d = 3
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, N_min=5, t_max=2.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state_cache, 20, 5, Inf)

        # Tight bound → shrink N
        PDMPSamplers._adapt_grid_N!(alg, 0.8)
        @test alg.N[] < 20

        # Reset
        alg.N[] = 20
        PDMPSamplers.recompute_time_grid!(alg)

        # Loose bound → increase N (when below max)
        alg.N[] = 10
        PDMPSamplers._adapt_grid_N!(alg, 0.05)
        @test alg.N[] > 10

        # Already at max → no change
        alg.N[] = 20
        PDMPSamplers._adapt_grid_N!(alg, 0.05)
        @test alg.N[] == 20

        # Already at min → no change
        alg.N[] = 5
        PDMPSamplers._adapt_grid_N!(alg, 0.8)
        @test alg.N[] == 5
    end

    @testset "_increase_grid_N!" begin
        d = 3
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, N_min=5, t_max=2.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state_cache, 16, 5, Inf)

        PDMPSamplers._increase_grid_N!(alg)
        @test alg.N[] == 16

        # Already at max
        PDMPSamplers._increase_grid_N!(alg)
        @test alg.N[] == 16
    end

    @testset "lazy fallback disables pathological lazy searches until reset" begin
        function x7_grad!(out, x)
            out[1] = 7 * x[1]^6
            return out
        end

        model = PDMPModel(1, FullGradient(x7_grad!))
        flow = BouncyParticle(1, 0.0)
        strat = GridThinningStrategy(; N=20, t_max=5.0, safety_limit=10, lazy=true)
        ξ0 = SkeletonPoint([0.0], [2.0])
        rng = Xoshiro(1004)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model, strat, 0.0, ξ0)

        @test alg_.lazy_enabled[]

        τ, event_type, _ = PDMPSamplers.next_event_time(rng, model_, flow, alg_, state, cache, stats, Inf, true)

        @test isfinite(τ)
        @test event_type == :reflect
        @test !alg_.lazy_enabled[]

        PDMPSamplers.reset_grid_scale!(alg_, strat.t_max)
        @test alg_.lazy_enabled[]
    end

    @testset "reset_grid_scale!" begin
        d = 3
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, N_min=5, t_max=5.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state_cache, 10, 5, Inf)

        PDMPSamplers.reset_grid_scale!(alg, 3.0)
        @test alg.t_max[] ≈ 3.0
        @test alg.N[] == 10
        @test length(alg.pcb.t_grid) == 11
    end

    @testset "GridThinningStrategy construction with use_fd_hvp" begin
        strat = GridThinningStrategy(; use_fd_hvp=true, N=30)
        @test strat.use_fd_hvp
        @test strat.N == 30

        strat_hybrid = GridThinningStrategy(; envelope=:hybrid_linear, lazy=false)
        @test strat_hybrid.envelope === :hybrid_linear

        strat_inflated = GridThinningStrategy(; envelope=:inflated_linear, lazy=false,
            curvature_bound=(args...) -> PDMPSamplers.CertifiedUpperCurvature(0.0))
        @test strat_inflated.envelope === :inflated_linear
        @test strat_inflated.curvature_bound !== nothing
        @test strat_inflated.certification === :required
        @test strat_inflated.inflated_affine_threshold == 0.95
        @test strat_inflated.inflated_affine_min_area_gain == 0.0

        auto_preset = certified_auto_scalar_bps_grid(;
            curvature_bound=PDMPSamplers.GlobalCertifiedUpperCurvature(0.0))
        @test auto_preset.N == 1
        @test auto_preset.envelope === :certified_auto
        @test auto_preset.certification === :required
        @test auto_preset.inflated_affine_threshold == 0.9
        @test auto_preset.inflated_affine_min_area_gain == 0.0
        @test auto_preset.certified_auto_probe_interval == 20

        scalar_preset = certified_scalar_bps_grid(;
            curvature_bound=PDMPSamplers.GlobalCertifiedUpperCurvature(0.0))
        @test scalar_preset.N == 1
        @test scalar_preset.envelope === :inflated_linear
        @test scalar_preset.certification === :required
        @test scalar_preset.inflated_affine_threshold == 0.9
        @test scalar_preset.inflated_affine_min_area_gain == 0.0

        flat_preset = certified_flat_scalar_bps_grid(;
            curvature_bound=PDMPSamplers.GlobalCertifiedUpperCurvature(0.0))
        @test flat_preset.N == 1
        @test flat_preset.envelope === :inflated_constant
        @test flat_preset.certification === :required
    end

    @testset "certified_auto forced affine probe is non-sticky" begin
        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        strat = GridThinningStrategy(; envelope=:certified_auto, lazy=false,
            curvature_bound=PDMPSamplers.GlobalCertifiedUpperCurvature(0.0),
            certified_auto_probe_interval=1)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state, 1, 1, 2, 1.0)
        stats = PDMPSamplers.DevelStatisticCounter()

        alg.certified_auto_prefer_affine_next[] = false
        alg.certified_auto_flat_grids_since_probe[] = 1
        PDMPSamplers._begin_certified_auto_grid!(alg)
        @test alg.certified_auto_prefer_affine_current[]

        stats.affine_area_constant_equiv = 10.0
        stats.certified_auto_affine_cells = 1
        stats.certified_auto_area_saved = 0.1
        PDMPSamplers._record_certified_auto_grid_choice!(stats, alg, 0, 0, 0.0, 0.0)
        @test stats.certified_auto_forced_probe_grids == 1
        @test stats.certified_auto_affine_preferred_grids == 1
        @test !alg.certified_auto_prefer_affine_next[]
        @test alg.certified_auto_flat_grids_since_probe[] == 1
    end

    @testset "Early stopping in construct_upper_bound_grad_and_hess!" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        pcb = PDMPSamplers.PiecewiseConstantBound(collect(range(0.0, 2.0, 21)), zeros(20))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end
        hvp_func = (x, v) -> begin
            out = similar(x)
            neg_hvp!(target, out, x, v)
            out
        end

        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.construct_upper_bound_grad_and_hess!(pcb, state, flow, (grad_func, hvp_func);
            early_stop_threshold=0.001, stats=stats)
        # With a very low threshold, early stopping should trigger
        @test stats.grid_builds == 1
    end

    @testset "constant BPS grid uses batched signed jets when available" begin
        flow = BouncyParticle(1, 0.0)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 0.5, 1.0], zeros(2))
        stats = PDMPSamplers.DevelStatisticCounter()
        provider = TestSignedGridJets(Ref(0))

        n = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            pcb, state, flow, provider, false;
            stats,
        )

        @test n == 2
        @test provider.calls[] == 1
        @test stats.grid_endpoint_jet_calls == 1
        @test stats.grid_endpoint_evaluations == 0
        @test stats.grid_endpoint_gradient_calls == 0
        @test stats.grid_endpoint_hessian_calls == 0
        @test pcb.y_vals[1:3] ≈ [0.0, 0.0, 0.5]
        @test pcb.d_vals[1:3] ≈ [0.0, 0.0, 1.0]
        @test pcb.Λ_vals ≈ [0.0, 0.5]
    end

    @testset "append-only constant budget extension matches full construction" begin
        flow = BouncyParticle(1, 0.0)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        provider_full = TestSignedGridJets(Ref(0))
        provider_append = TestSignedGridJets(Ref(0))
        t_grid = collect(range(0.0, 1.0, 11))

        full = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        n_full = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            full, state, flow, provider_full, false; early_stop_threshold=0.08)

        appended = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        n_first = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            appended, state, flow, provider_append, false; early_stop_threshold=0.01)
        prefix_Λ = copy(appended.Λ_vals[1:n_first])
        prefix_y = copy(appended.y_vals[1:(n_first + 1)])
        prefix_d = copy(appended.d_vals[1:(n_first + 1)])
        built_area = PDMPSamplers._piecewise_constant_area(appended)

        n_appended = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            appended, state, flow, provider_append, false;
            early_stop_threshold=0.08,
            start_cell=n_first + 1,
            initial_integral=built_area)

        @test n_appended == n_full
        @test appended.Λ_vals[1:n_full] ≈ full.Λ_vals[1:n_full]
        @test appended.y_vals[1:(n_full + 1)] ≈ full.y_vals[1:(n_full + 1)]
        @test appended.d_vals[1:(n_full + 1)] ≈ full.d_vals[1:(n_full + 1)]
        @test appended.Λ_vals[1:n_first] == prefix_Λ
        @test appended.y_vals[1:(n_first + 1)] == prefix_y
        @test appended.d_vals[1:(n_first + 1)] == prefix_d

        for budget in (0.01, 0.04, 0.08)
            τ_full, lb_full = PDMPSamplers.propose_event_time(full, budget)
            τ_app, lb_app = PDMPSamplers.propose_event_time(appended, budget)
            @test τ_app ≈ τ_full
            @test lb_app ≈ lb_full
        end
    end

    @testset "append-only inflated affine budget extension preserves prefix" begin
        flow = BouncyParticle(1, 0.0)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        provider_full = TestSignedGridJets(Ref(0))
        provider_append = TestSignedGridJets(Ref(0))
        cert = PDMPSamplers.GlobalCertifiedUpperCurvature(0.0)
        t_grid = collect(range(0.0, 1.0, 11))

        full_pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        full_pab = PDMPSamplers.PiecewiseAffineBound(32)
        n_full = PDMPSamplers.construct_signed_inflated_grid!(
            full_pab, full_pcb, state, flow, provider_full, cert;
            early_stop_threshold=0.08,
            build_affine=true)

        app_pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        app_pab = PDMPSamplers.PiecewiseAffineBound(32)
        n_first = PDMPSamplers.construct_signed_inflated_grid!(
            app_pab, app_pcb, state, flow, provider_append, cert;
            early_stop_threshold=0.01,
            build_affine=true)
        prefix_segments = app_pab.n_segments
        prefix_breaks = copy(app_pab.t_breaks[1:(prefix_segments + 1)])
        prefix_y_left = copy(app_pab.y_left[1:prefix_segments])
        prefix_slopes = copy(app_pab.slopes[1:prefix_segments])
        prefix_cum_area = copy(app_pab.cum_area[1:(prefix_segments + 1)])
        built_area = PDMPSamplers.total_area(app_pab)

        n_appended = PDMPSamplers.construct_signed_inflated_grid!(
            app_pab, app_pcb, state, flow, provider_append, cert;
            early_stop_threshold=0.08,
            build_affine=true,
            start_cell=n_first + 1,
            initial_integral=built_area,
            append=true)

        @test n_appended == n_full
        @test app_pcb.Λ_vals[1:n_full] ≈ full_pcb.Λ_vals[1:n_full]
        @test app_pcb.y_vals[1:(n_full + 1)] ≈ full_pcb.y_vals[1:(n_full + 1)]
        @test app_pcb.d_vals[1:(n_full + 1)] ≈ full_pcb.d_vals[1:(n_full + 1)]
        @test app_pab.t_breaks[1:(prefix_segments + 1)] == prefix_breaks
        @test app_pab.y_left[1:prefix_segments] == prefix_y_left
        @test app_pab.slopes[1:prefix_segments] == prefix_slopes
        @test app_pab.cum_area[1:(prefix_segments + 1)] == prefix_cum_area
        @test PDMPSamplers.total_area(app_pab) ≈ PDMPSamplers.total_area(full_pab)

        for budget in (0.01, 0.04, 0.08)
            τ_full, lb_full = PDMPSamplers.propose_event_time(full_pab, budget)
            τ_app, lb_app = PDMPSamplers.propose_event_time(app_pab, budget)
            @test τ_app ≈ τ_full
            @test lb_app ≈ lb_full
        end
    end

    @testset "construct_upper_bound_grad_and_hess! with cached values" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))
        pcb = PDMPSamplers.PiecewiseConstantBound(collect(range(0.0, 2.0, 11)), zeros(10))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end
        hvp_func = (x, v) -> begin
            out = similar(x)
            neg_hvp!(target, out, x, v)
            out
        end

        stats = PDMPSamplers.DevelStatisticCounter()
        # With cached_y0 and cached_d0
        PDMPSamplers.construct_upper_bound_grad_and_hess!(pcb, state, flow, (grad_func, hvp_func);
            cached_y0=1.0, cached_d0=0.5, stats=stats)
        @test pcb.y_vals[1] == 1.0
        @test pcb.d_vals[1] == 0.5
        @test stats.grid_builds == 1
    end

    @testset "End-to-end with FD-HVP" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP
        alg = GridThinningStrategy(; use_fd_hvp=true)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 10_000.0; progress=show_progress, statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test length(trace) > 50
        @test all(isfinite, mean(trace))
    end

    @testset "End-to-end without HVP (grad-only)" begin
        d = 3
        Random.seed!(43)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP
        alg = GridThinningStrategy()  # no FD-HVP either

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 10_000.0; progress=show_progress, statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test length(trace) > 50
        @test all(isfinite, mean(trace))
    end

    @testset "Strategy interface defaults and roots acceptance" begin
        roots = PDMPSamplers.RootsPoissonTimeStrategy()
        stats = PDMPSamplers.DevelStatisticCounter()

        @test PDMPSamplers._reset_inner_grid!(roots) === nothing
        @test PDMPSamplers._maybe_activate_constant_bound!(roots, stats) === nothing
        @test PDMPSamplers.accept_reflection_event(roots, :dummy) === true
    end

    @testset "Post-warmup constant-bound activation" begin
        d = 3
        state = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, t_max=2.0, post_warmup_simplify=true)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state, 20, 5, 5.0)

        stats = PDMPSamplers.DevelStatisticCounter()
        stats.reflections_accepted = 2
        stats.refreshment_events = 18
        alg.max_observed_rate[] = 3.5
        PDMPSamplers._maybe_activate_constant_bound!(alg, stats)
        @test alg.constant_bound_rate[] ≈ 7.0

        alg.constant_bound_rate[] = NaN
        stats.reflections_accepted = 8
        stats.refreshment_events = 2
        PDMPSamplers._maybe_activate_constant_bound!(alg, stats)
        @test isnan(alg.constant_bound_rate[])
    end

    @testset "Subsampled gradient grid adaptation branches" begin
        d = 3
        state = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, t_max=2.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state, 20, 5, 5.0)

        grad_sub = SubsampledGradient(
            (out, x) -> (out .= x),
            n -> nothing,
            tr -> nothing,
            (out, x) -> (out .= x),
            5,
            0,
            false;
            resample_dt=0.1,
        )

        alg.t_max[] = 10.0
        PDMPSamplers._adapt_grid_t_max!(alg, 0.2, grad_sub)
        @test alg.t_max[] ≈ 4.0

        old_tmax = alg.t_max[]
        PDMPSamplers._shrink_t_max_on_rejection!(alg, alg.pcb, 0.01, grad_sub)
        @test alg.t_max[] == old_tmax
    end

    @testset "End-to-end with joint directional curvature" begin
        d = 2
        f_logdensity(x) = -0.5 * sum(abs2, x)

        flow = BouncyParticle(d, 1.0)
        model = PDMPModel(d, LogDensity(f_logdensity), DI.AutoForwardDiff(), true)
        alg = GridThinningStrategy(; N=16, t_max=1.5)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 5_000.0; progress=show_progress, statistic_counter=PDMPSamplers.DevelStatisticCounter)

        @test length(trace) > 20
        @test stats.∇²f_calls > 0
    end

    @testset "get_rate_and_deriv FiniteDiffVHV with BouncyParticle (ContinuousDynamics dispatch)" begin
        d = 3
        Random.seed!(44)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = BouncyParticle(d, 1.0)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP
        alg = GridThinningStrategy(; use_fd_hvp=true, N=16, t_max=1.5)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 500.0; progress=show_progress, statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test length(trace) > 10
        @test all(isfinite, mean(trace))
    end

    @testset "_constant_bound_event_time direct call" begin
        d = 3
        Random.seed!(45)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = BouncyParticle(d, 1.0)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        state = PDMPState(0.0, ξ0)

        strat = GridThinningStrategy(; N=16, t_max=2.0, post_warmup_simplify=true)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state, 16, 5, 5.0)
        cache = (; z=similar(ξ0.x), ∇ϕx=similar(ξ0.x))
        stats = PDMPSamplers.DevelStatisticCounter()

        # Arm the constant bound to a deliberately high value so thinning always accepts
        alg.constant_bound_rate[] = 1e6

        τ, event_type, meta = PDMPSamplers._constant_bound_event_time(
            model, flow, alg, state, cache, stats, Inf, false)
        @test isfinite(τ)
        @test τ > 0.0
        @test event_type === :reflect

        # With include_refresh=true and high refresh rate, should sometimes return :refresh
        flow_refresh = BouncyParticle(d, 1e6)
        alg2 = PDMPSamplers._build_grid_adaptive_state(strat, state, 16, 5, 5.0)
        alg2.constant_bound_rate[] = 1.0
        τ2, event_type2, _ = PDMPSamplers._constant_bound_event_time(
            model, flow_refresh, alg2, state, cache, stats, Inf, true)
        @test isfinite(τ2)
        @test event_type2 === :refresh

        # Scenario 3: actual rate exceeds bound → fallback, constant_bound_rate → NaN
        # ZeroMeanIsoNormal: gradient(x) = x, so rate = max(0, dot(x, θ))
        # With x=[10,0,...] and θ=[1,0,...], rate = 10 + τ at proposal time τ.
        # Setting λ_bound=10 guarantees l_actual = 10+τ > 10 for any τ > 0.
        target3 = gen_data(Distributions.ZeroMeanIsoNormal, d)
        grad3 = FullGradient(Base.Fix1(neg_gradient!, target3))
        model3 = PDMPModel(d, grad3)
        x3 = zeros(d); x3[1] = 10.0
        θ3 = zeros(d); θ3[1] = 1.0
        ξ3 = SkeletonPoint(x3, θ3)
        state3 = PDMPState(0.0, ξ3)
        alg3 = PDMPSamplers._build_grid_adaptive_state(strat, state3, 16, 5, 5.0)
        alg3.constant_bound_rate[] = 10.0
        τ3, _, _ = PDMPSamplers._constant_bound_event_time(
            model3, flow, alg3, state3, cache, stats, Inf, false)
        @test isnan(alg3.constant_bound_rate[])
        @test isfinite(τ3)
    end
end
