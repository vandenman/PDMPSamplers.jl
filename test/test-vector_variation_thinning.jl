@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "VectorVariationThinningStrategy" begin
    @testset "constructor, scalar inversion, and reset invariants" begin
        constructor = PDMPSamplers.VectorVariationThinningStrategy
        @test_throws ArgumentError constructor(; N=0)
        @test_throws ArgumentError constructor(; N_min=0)
        @test_throws ArgumentError constructor(; t_max=0.0)
        @test_throws ArgumentError constructor(; max_refinement_depth=-1)
        @test_throws ArgumentError constructor(; validation_rtol=-1.0)
        @test_throws ArgumentError constructor(; validation_atol=-1.0)
        @test_throws ArgumentError constructor(; min_cell_width=0.0)
        @test_throws ArgumentError constructor(; max_skip_width=0.0)
        @test_throws ArgumentError constructor(; derivative_hermite_trigger_scale=-1.0)
        @test occursin("VectorVariationThinningStrategy", sprint(show, constructor()))

        fa = [-1.0, 2.0, -0.5]
        fb = [2.0, -1.0, -0.25]
        h = 1.5
        rate(t) = sum(max(fa[i] + (fb[i] - fa[i]) * t / h, 0.0) for i in eachindex(fa))
        expected, _ = PDMPSamplers.QuadGK.quadgk(rate, 0.0, h; rtol=1e-12)
        area = PDMPSamplers._vv_linear_positive_area(fa, fb, h)
        @test area ≈ expected atol=1e-12
        target = 0.41area
        τ = PDMPSamplers._vv_linear_positive_area_time(fa, fb, h, target)
        partial, _ = PDMPSamplers.QuadGK.quadgk(rate, 0.0, τ; rtol=1e-12)
        @test partial ≈ target atol=1e-10
        @test PDMPSamplers._vv_linear_positive_value(fa, fb, h, τ) ≈ rate(τ)
        @test PDMPSamplers._vv_linear_positive_area(fa, fb, 0.0) == 0.0
        @test PDMPSamplers._vv_linear_positive_area_time(fa, fb, h, 0.0) == 0.0

        model = PDMPModel(1, FullGradient((out, x) -> copyto!(out, x)))
        strategy = constructor(; N=3, N_min=2, t_max=0.5)
        state, _, alg, _, _ = PDMPSamplers.initialize_state(
            Xoshiro(20260729), ZigZag(1), model, strategy, 0.0,
            SkeletonPoint([1.0], [1.0]))
        alg.has_cached_gradient[] = true
        alg.cached_U[] = 2.0
        PDMPSamplers.reset_grid_scale!(alg, 0.75)
        @test alg.t_max[] == 0.75
        @test alg.N[] >= alg.N_min
        @test !alg.has_cached_gradient[]
        @test isnan(alg.cached_U[])
        @test state.ξ.x == [1.0]
    end

    @testset "linear Gaussian event reuses accepted candidate gradient" begin
        function gaussian_grad!(out, x)
            out[1] = x[1]
            return out
        end

        model = PDMPModel(1, FullGradient(gaussian_grad!))
        flow = ZigZag(1)
        alg = PDMPSamplers.VectorVariationThinningStrategy(;
            N=4,
            N_min=1,
            t_max=2.0,
            validation_rtol=0.0,
            validation_atol=1e-12,
            fallback=GridThinningStrategy(; N=4, N_min=1, t_max=2.0, lazy=false, bound=:flat),
        )
        ξ0 = SkeletonPoint([1.0], [1.0])
        rng = Xoshiro(20260716)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)

        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, Inf, false)

        @test event_type === :reflect
        @test isfinite(τ)
        @test τ > 0
        @test meta isa PDMPSamplers.GradientMeta
        @test meta.∇ϕx[1] ≈ 1.0 + τ
        @test alg_.has_cached_gradient[]
        @test alg_.cached_gradient[1] ≈ meta.∇ϕx[1]
        @test isnan(alg_.cached_U[])
        @test stats.positive_variation_accepts == 1
        @test stats.positive_variation_fallbacks == 0
        @test stats.grid_bound_violations == 0

        state.ξ.x[1] += τ * state.ξ.θ[1]
        state.t[] += τ
        state.ξ.θ[1] = -state.ξ.θ[1]
        before_reuses = stats.grid_cached_endpoint_reuses
        τ2, event_type2, _ = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, 0.1, false)

        @test event_type2 === :horizon_hit
        @test τ2 ≈ 0.1
        @test stats.grid_cached_endpoint_reuses == before_reuses + 1
    end

    @testset "dense preconditioned ZigZag channels match package rate" begin
        d = 3
        flow = DensePreconditionedZigZag(d)
        L = [1.0 0.0 0.0; 0.4 1.0 0.0; -0.2 0.3 1.0]
        set_dense_preconditioner!(flow.metric, L)
        signs = [1.0, -1.0, 1.0]

        x = [0.2, -0.3, 0.5]
        grad = [0.7, -1.2, 0.4]
        state = PDMPState(0.0, SkeletonPoint(x, flow.metric.L * signs))
        out = similar(grad)

        PDMPSamplers._vv_signed_channels!(out, state, grad, flow, (; z=similar(x)))

        expected = signs .* (transpose(flow.metric.L) * grad)
        @test out ≈ expected
        @test sum(max.(out, 0.0)) ≈ PDMPSamplers.λ(state, grad, flow)
    end

    @testset "bound violation routes through fallback instead of accepting proposal" begin
        function bowed_grad!(out, x)
            out[1] = 10.0 + 100.0 * x[1] * (1.0 - x[1])
            return out
        end

        model = PDMPModel(1, FullGradient(bowed_grad!))
        flow = ZigZag(1)
        alg = PDMPSamplers.VectorVariationThinningStrategy(;
            N=1,
            N_min=1,
            t_max=1.0,
            validation_rtol=0.0,
            validation_atol=0.0,
            safety_limit=5,
            fallback=GridThinningStrategy(; N=2, N_min=1, t_max=1.0, lazy=false, bound=:flat),
        )
        ξ0 = SkeletonPoint([0.0], [1.0])
        rng = Xoshiro(8)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)

        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, 1.0, false)

        @test event_type in (:reflect, :horizon_hit)
        @test τ >= 0.0
        @test meta isa PDMPSamplers.GradientMeta
        @test stats.grid_bound_violations >= 1
        @test stats.positive_variation_fallbacks >= 1
    end

    @testset "preconditioned path and unsupported-flow fallback" begin
        function positive_grad!(out, x)
            out .= 5.0 .* x
            return out
        end

        model = PDMPModel(2, FullGradient(positive_grad!))
        strategy = PDMPSamplers.VectorVariationThinningStrategy(;
            N=2, N_min=1, t_max=0.4,
            fallback=GridThinningStrategy(;
                N=2, N_min=1, t_max=0.4, lazy=false, bound=:flat),
        )

        flow = PreconditionedZigZag(2; scale=[2.0, 0.5])
        rng = Xoshiro(20260730)
        state, model_, alg, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, strategy, 0.0,
            SkeletonPoint([1.0, 0.5], [2.0, -0.5]);
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg, state, cache, stats, 0.4, false)
        @test 0.0 <= τ <= 0.4
        @test event_type in (:reflect, :horizon_hit)
        @test meta isa PDMPSamplers.GradientMeta
        @test stats.positive_variation_cells > 0

        zero_τ, zero_event, _ = PDMPSamplers.next_event_time(
            rng, model_, flow, alg, state, cache, stats, 0.0, false, :support_hit)
        @test zero_τ == 0.0
        @test zero_event === :support_hit

        bps = BouncyParticle(2, 0.0)
        bps_rng = Xoshiro(20260731)
        bps_state, bps_model, bps_alg, bps_cache, bps_stats =
            PDMPSamplers.initialize_state(
                bps_rng, bps, model, strategy, 0.0,
                SkeletonPoint([1.0, 0.5], [1.0, -1.0]);
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
        fallback_τ, fallback_event, _ = PDMPSamplers.next_event_time(
            bps_rng, bps_model, bps, bps_alg, bps_state, bps_cache, bps_stats, 0.2, false)
        @test 0.0 <= fallback_τ <= 0.2
        @test fallback_event in (:reflect, :horizon_hit)
        @test bps_stats.positive_variation_cells == 0
    end
end
