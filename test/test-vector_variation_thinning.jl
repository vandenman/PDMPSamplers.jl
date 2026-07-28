@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "VectorVariationThinningStrategy" begin
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
        flow.metric.L .= L
        flow.metric.Linv .= inv(LowerTriangular(L))
        flow.metric.v_canonical .= [1.0, -1.0, 1.0]

        x = [0.2, -0.3, 0.5]
        grad = [0.7, -1.2, 0.4]
        state = PDMPState(0.0, SkeletonPoint(x, flow.metric.L * flow.metric.v_canonical))
        out = similar(grad)

        PDMPSamplers._vv_signed_channels!(out, state, grad, flow, (; z=similar(x)))

        expected = flow.metric.v_canonical .* (transpose(flow.metric.L) * grad)
        @test out ≈ expected
        @test sum(max.(out, 0.0)) ≈ PDMPSamplers.λ(state.ξ, grad, flow)
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
end
