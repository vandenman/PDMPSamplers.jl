@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

import ADTypes
import DifferentiationInterface as DI
import ForwardDiff

@testset "Miscellaneous" begin

    @testset "HVP sign for FullGradient path" begin
        d = 3
        A = [2.0 0.5 0.1; 0.5 3.0 0.2; 0.1 0.2 1.5]
        f_grad!(out, x) = (mul!(out, A, x); out)
        grad = FullGradient(f_grad!)
        backend = DI.AutoForwardDiff()
        model = PDMPModel(d, grad, backend, true)

        x_test = [1.0, -0.5, 2.0]
        v_test = [0.3, 0.7, -0.2]
        result = model.hvp(x_test, v_test)
        expected = A * v_test
        @test result ≈ expected
    end

    @testset "_integrate on small traces" begin
        flow = ZigZag(3)
        trace_empty = PDMPTrace(Vector{PDMPEvent{Float64, Vector{Float64}, Vector{Float64}}}(), flow)
        @test_throws ErrorException("Cannot compute statistics on an empty trace") mean(trace_empty)

        x0 = [1.0, 2.0, 3.0]
        θ0 = [1.0, -1.0, 1.0]
        trace_single = PDMPTrace([PDMPEvent(0.0, x0, θ0)], flow)
        @test_throws ErrorException("Cannot compute statistics on a trace with fewer than 2 events") mean(trace_single)
    end

    @testset "Boomerang freezing_time" begin
        flow_zero_mu = Boomerang(1)

        @testset "μ=0, θ=0, x>0 → π/2" begin
            ξ = SkeletonPoint([1.0], [0.0])
            @test PDMPSamplers.freezing_time(ξ, flow_zero_mu, 1) ≈ π / 2
        end

        @testset "μ=0, θ=0, x<0 → π/2" begin
            ξ = SkeletonPoint([-1.0], [0.0])
            @test PDMPSamplers.freezing_time(ξ, flow_zero_mu, 1) ≈ π / 2
        end

        @testset "μ=0, θ=0, x=0 → Inf" begin
            ξ = SkeletonPoint([0.0], [0.0])
            @test PDMPSamplers.freezing_time(ξ, flow_zero_mu, 1) == Inf
        end

        @testset "x=2μ singularity → finite positive" begin
            μ_val = 1.5
            flow_nonzero = Boomerang(Diagonal([1.0]), [μ_val])
            ξ = SkeletonPoint([2μ_val], [1.0])
            t = PDMPSamplers.freezing_time(ξ, flow_nonzero, 1)
            @test isfinite(t)
            @test t > 0
        end

        @testset "General cases: compare to brute-force root finding" begin
            for (x, θ, μ) in [(2.0, 1.0, 0.0), (0.5, -1.5, 0.0),
                               (1.0, 0.5, 0.3), (3.0, -1.0, 1.0)]
                flow_i = Boomerang(Diagonal([1.0]), [μ])
                ξ = SkeletonPoint([x], [θ])
                t_computed = PDMPSamplers.freezing_time(ξ, flow_i, 1)
                trajectory(t) = (x - μ) * cos(t) + θ * sin(t) + μ
                if isfinite(t_computed)
                    @test abs(trajectory(t_computed)) < 1e-10
                    @test t_computed > 0
                end
            end
        end
    end

    @testset "ZigZag reflect! coordinate selection distribution" begin
        d = 4
        flow = ZigZag(d)
        x = [1.0, 2.0, 3.0, 4.0]
        θ = ones(d)
        ∇ϕ = [1.0, 2.0, 4.0, 8.0]

        grad = FullGradient((out, x) -> out .= x)
        ξ = SkeletonPoint(copy(x), copy(θ))
        cache = PDMPSamplers.add_gradient_to_cache(
            PDMPSamplers.initialize_cache(flow, grad, ThinningStrategy(GlobalBounds(1.0, d)), 0.0, ξ),
            ξ
        )

        n_samples = 50_000
        counts = zeros(Int, d)
        for _ in 1:n_samples
            ξ_test = SkeletonPoint(copy(x), ones(d))
            PDMPSamplers.reflect!(ξ_test, copy(∇ϕ), flow, cache)
            for i in 1:d
                if ξ_test.θ[i] != 1.0
                    counts[i] += 1
                    break
                end
            end
        end

        rates = [max(0.0, θ[i] * ∇ϕ[i]) for i in 1:d]
        expected_probs = rates ./ sum(rates)
        empirical_probs = counts ./ n_samples
        for i in 1:d
            @test abs(empirical_probs[i] - expected_probs[i]) < 0.02
        end
    end
end
