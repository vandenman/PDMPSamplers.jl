@isdefined(PDMPSamplers) || begin
    include(joinpath(@__DIR__, "testsetup.jl"))
    import DifferentiationInterface as DI
end

@testset "Directional curvature (vhv) parity" begin

    @testset "extract_vhv helper" begin
        v = [1.0, 2.0, 3.0]
        Hv = [0.5, 1.0, 1.5]
        @test PDMPSamplers.extract_vhv(v, Hv) ≈ dot(v, Hv)
        @test PDMPSamplers.extract_vhv(v, 42.0) == 42.0
    end

    @testset "∂λ∂t scalar vs vector parity: $pdmp_type" for pdmp_type in (BouncyParticle, Boomerang, ZigZag)
        d = 5
        target = gen_data(MvNormal, d, 1.0)
        flow = pdmp_type(d)
        x = randn(d)
        θ = PDMPSamplers.initialize_velocity(flow, d)
        state = PDMPState(0.0, SkeletonPoint(x, θ))
        move_forward_time!(state, 0.5, flow)

        out = similar(x)
        ∇U_xt = neg_gradient!(target, similar(x), state.ξ.x)
        Hxt_vt = neg_hvp!(target, similar(x), state.ξ.x, state.ξ.θ)

        deriv_vec = PDMPSamplers.∂λ∂t(state, ∇U_xt, Hxt_vt, flow)
        vhv_scalar = dot(state.ξ.θ, Hxt_vt)
        deriv_scalar = PDMPSamplers.∂λ∂t(state, ∇U_xt, vhv_scalar, flow)

        if pdmp_type == ZigZag
            @test isfinite(deriv_vec)
            @test isfinite(deriv_scalar)
        else
            @test deriv_scalar ≈ deriv_vec
        end
    end

    @testset "get_rate_and_deriv: VHVProvider vs Tuple for $pdmp_type" for pdmp_type in (BouncyParticle, Boomerang)
        d = 5
        target = gen_data(MvNormal, d, 1.0)
        flow = pdmp_type(d)
        x = randn(d)
        θ = PDMPSamplers.initialize_velocity(flow, d)
        state = PDMPState(0.0, SkeletonPoint(x, θ))
        move_forward_time!(state, 0.3, flow)

        grad_func = x_arg -> begin
            out = similar(x_arg)
            neg_gradient!(target, out, x_arg)
            out
        end
        hvp_func = (x_arg, v) -> begin
            out = similar(x_arg)
            neg_hvp!(target, out, x_arg, v)
            out
        end
        vhv_func = (x_arg, v, w) -> begin
            Hv = similar(x_arg)
            neg_hvp!(target, Hv, x_arg, v)
            dot(w, Hv)
        end

        rate_vec, deriv_vec = PDMPSamplers.get_rate_and_deriv(state, flow, (grad_func, hvp_func), false)
        provider = PDMPSamplers.VHVProvider(grad_func, vhv_func)
        rate_vhv, deriv_vhv = PDMPSamplers.get_rate_and_deriv(state, flow, provider, false)

        @test rate_vec ≈ rate_vhv
        @test deriv_vec ≈ deriv_vhv
    end

    @testset "get_rate_and_deriv: VHVProvider for ZigZag (bilinear)" begin
        d = 5
        target = gen_data(MvNormal, d, 1.0)
        flow = ZigZag(d)
        x = randn(d)
        θ = PDMPSamplers.initialize_velocity(flow, d)
        state = PDMPState(0.0, SkeletonPoint(x, θ))
        move_forward_time!(state, 0.3, flow)

        grad_func = x_arg -> begin
            out = similar(x_arg)
            neg_gradient!(target, out, x_arg)
            out
        end
        hvp_func = (x_arg, v) -> begin
            out = similar(x_arg)
            neg_hvp!(target, out, x_arg, v)
            out
        end
        vhv_func = (x_arg, v, w) -> begin
            Hv = similar(x_arg)
            neg_hvp!(target, Hv, x_arg, v)
            dot(w, Hv)
        end

        rate_vec, deriv_vec = PDMPSamplers.get_rate_and_deriv(state, flow, (grad_func, hvp_func), false)
        provider = PDMPSamplers.VHVProvider(grad_func, vhv_func)
        rate_vhv, deriv_vhv = PDMPSamplers.get_rate_and_deriv(state, flow, provider, false)

        @test rate_vec ≈ rate_vhv
        @test deriv_vec ≈ deriv_vhv
    end

    @testset "PDMPModel with vhv field" begin
        d = 3
        f_logdensity(x) = -0.5 * sum(x .^ 2)
        backend = DI.AutoForwardDiff()

        model = PDMPModel(d, LogDensity(f_logdensity), backend, true)
        @test model.vhv !== nothing
        @test model.hvp !== nothing

        x_test = [1.0, 2.0, 3.0]
        v_test = [0.5, -0.3, 0.8]

        hvp_result = model.hvp(x_test, v_test)
        expected_vhv = dot(v_test, hvp_result)
        actual_vhv = model.vhv(x_test, v_test, v_test)
        @test actual_vhv ≈ expected_vhv atol=1e-6

        w_test = [1.0, 0.0, -1.0]
        expected_whv = dot(w_test, hvp_result)
        actual_whv = model.vhv(x_test, v_test, w_test)
        @test actual_whv ≈ expected_whv atol=1e-6
    end

    @testset "PDMPModel from FullGradient + backend" begin
        d = 3
        f_grad!(out, x) = (out .= x)
        grad = FullGradient(f_grad!)
        backend = DI.AutoForwardDiff()

        model = PDMPModel(d, grad, backend, true)
        @test model.vhv !== nothing

        x_test = [1.0, 2.0, 3.0]
        v_test = [0.5, -0.3, 0.8]
        w_test = v_test

        expected = dot(w_test, v_test)
        actual = model.vhv(x_test, v_test, w_test)
        @test actual ≈ expected atol=1e-6
    end

    @testset "with_stats wraps vhv" begin
        d = 3
        f_logdensity(x) = -0.5 * sum(x .^ 2)
        backend = DI.AutoForwardDiff()
        model = PDMPModel(d, LogDensity(f_logdensity), backend, true)

        stats = PDMPSamplers.StatisticCounter()
        model_stats = PDMPSamplers.with_stats(model, stats)

        @test model_stats.vhv isa PDMPSamplers.WithStatsVHV
        x_test = [1.0, 2.0, 3.0]
        v_test = [0.5, -0.3, 0.8]
        model_stats.vhv(x_test, v_test, v_test)
        @test stats.∇²f_calls == 1
    end

    @testset "Sampling smoke test with vhv: $pdmp_type" for pdmp_type in (BouncyParticle, Boomerang)
        d = 3
        f_logdensity(x) = -0.5 * sum(x .^ 2)
        backend = DI.AutoForwardDiff()
        model = PDMPModel(d, LogDensity(f_logdensity), backend, true)
        @test model.vhv !== nothing

        flow = pdmp_type(d)
        alg = GridThinningStrategy(; N=10, t_max=2.0)
        x0 = randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 500.0)
        @test length(trace) > 10
        @test all(isfinite, mean(trace))
    end
end
