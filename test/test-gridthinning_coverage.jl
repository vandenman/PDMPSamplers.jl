@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Grid thinning coverage" begin

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

        # With refresh_rate
        τ3, lb3 = PDMPSamplers.propose_event_time(pcb, 0.5, 1.0)
        @test τ3 ≤ τ  # higher total rate → earlier event
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

    @testset "FiniteDiffHVP with shared-buffer grad (aliasing regression)" begin
        d = 5
        Random.seed!(44)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        for flow_type in (ZigZag, BouncyParticle, Boomerang)
            flow = flow_type(d)
            x0 = randn(d)
            θ0 = PDMPSamplers.initialize_velocity(flow, d)
            state = PDMPState(0.0, SkeletonPoint(x0, θ0))
            move_forward_time!(state, 0.3, flow)

            grad = FullGradient(Base.Fix1(neg_gradient!, target))
            cache = (; z = zeros(d), ∇ϕx = zeros(d))
            aliasing_grad = PDMPSamplers.make_grad_U_func(state, flow, grad, cache)

            fd = PDMPSamplers.FiniteDiffHVP(aliasing_grad, zeros(d))
            rate_fd, deriv_fd = PDMPSamplers.get_rate_and_deriv(state, flow, fd, false)

            hvp_func = (x, v) -> begin
                out = similar(x)
                neg_hvp!(target, out, x, v)
                out
            end
            fresh_grad = x -> begin
                out = similar(x)
                neg_gradient!(target, out, x)
                out
            end
            rate_ref, deriv_ref = PDMPSamplers.get_rate_and_deriv(state, flow, (fresh_grad, hvp_func), false)

            @test rate_fd ≈ rate_ref
            @test isapprox(deriv_fd, deriv_ref; rtol=0.02) || (iszero(deriv_ref) && abs(deriv_fd) < 1e-6)
        end
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
        @test PDMPSamplers.min_grid_cells(boom, 5, 20) == 10
        @test PDMPSamplers.min_grid_cells(boom, 15, 20) == 15

        @test PDMPSamplers.max_grid_horizon(zz) == 1e10
        @test PDMPSamplers.max_grid_horizon(boom) ≈ 8π
    end

    @testset "_adapt_grid_N! and _increase_grid_N!" begin
        d = 3
        pcb = PDMPSamplers.PiecewiseConstantBound(collect(range(0.0, 2.0, 21)), zeros(20))
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        alg = PDMPSamplers.GridAdaptiveState(
            pcb, Ref(20), Ref(2.0), 1.5, 0.5, 500, 5, 20, Inf,
            copy(state_cache), copy(state_cache), Float64[], false, zeros(d)
        )

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
        pcb = PDMPSamplers.PiecewiseConstantBound(collect(range(0.0, 2.0, 21)), zeros(20))
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        alg = PDMPSamplers.GridAdaptiveState(
            pcb, Ref(16), Ref(2.0), 1.5, 0.5, 500, 5, 20, Inf,
            copy(state_cache), copy(state_cache), Float64[], false, zeros(d)
        )

        PDMPSamplers._increase_grid_N!(alg)
        @test alg.N[] == 20

        # Already at max
        PDMPSamplers._increase_grid_N!(alg)
        @test alg.N[] == 20
    end

    @testset "reset_grid_scale!" begin
        d = 3
        pcb = PDMPSamplers.PiecewiseConstantBound(collect(range(0.0, 2.0, 21)), zeros(20))
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        alg = PDMPSamplers.GridAdaptiveState(
            pcb, Ref(10), Ref(5.0), 1.5, 0.5, 500, 5, 20, Inf,
            copy(state_cache), copy(state_cache), Float64[], false, zeros(d)
        )

        PDMPSamplers.reset_grid_scale!(alg, 3.0)
        @test alg.t_max[] ≈ 3.0
        @test alg.N[] == 20
        @test length(alg.pcb.t_grid) == 21
    end

    @testset "GridThinningStrategy construction with use_fd_hvp" begin
        strat = GridThinningStrategy(; use_fd_hvp=true, N=30)
        @test strat.use_fd_hvp
        @test strat.N == 30
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

        stats = PDMPSamplers.StatisticCounter()
        PDMPSamplers.construct_upper_bound_grad_and_hess!(pcb, state, flow, (grad_func, hvp_func);
            early_stop_threshold=0.001, stats=stats)
        # With a very low threshold, early stopping should trigger
        @test stats.grid_builds == 1
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

        stats = PDMPSamplers.StatisticCounter()
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
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 10_000.0; progress=show_progress)
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
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 10_000.0; progress=show_progress)
        @test length(trace) > 50
        @test all(isfinite, mean(trace))
    end
end
