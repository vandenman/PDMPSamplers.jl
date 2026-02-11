using Test
using PDMPSamplers
using Random
using LinearAlgebra
using DynamicPPL
using Distributions
using Statistics
import ADTypes
import ForwardDiff

@testset "DynamicPPL Extension" begin

    # Define a simple multivariate Gaussian model
    @model function mvnormal_model(d)
        μ = zeros(d)
        Σ = Matrix(1.0I(d))
        x ~ MvNormal(μ, Σ)
        return x
    end

    d = 3

    @testset "PDMPModel construction from DynamicPPL model" begin
        dpppl_model = mvnormal_model(d)
        backend = ADTypes.AutoForwardDiff()

        # Test without HVP
        model_no_hvp = PDMPModel(dpppl_model, backend; needs_hvp=false)
        @test model_no_hvp.d == d
        @test model_no_hvp.grad isa FullGradient
        @test model_no_hvp.hvp === nothing
    end

    @testset "Sampling with DynamicPPL model" begin
        Random.seed!(456)

        dpppl_model = mvnormal_model(d)
        backend = ADTypes.AutoForwardDiff()
        model = PDMPModel(dpppl_model, backend; needs_hvp=false)

        # Set up sampler
        μ = zeros(d)
        Σ = Matrix(1.0I(d))
        flow = ZigZag(inv(Σ), μ)
        alg = ThinningStrategy(LocalBounds(fill(1.5, d)))

        x0 = randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        T = 1000.0

        # Run sampler
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T, progress=false)

        # Basic checks
        @test length(trace.events) > 10
        @test stats.∇f_calls > 0
        @test stats.reflections_accepted > 0
        @test stats.reflections_accepted / stats.reflections_events > 0.1

        # Check that samples are approximately correct
        samples = Matrix(PDMPDiscretize(trace, 1.0))
        sample_mean = vec(mean(samples, dims=1))
        sample_cov = cov(samples)

        # Very loose checks (just that it's working)
        @test norm(sample_mean - μ) < 1.0  # Within 1.0 of true mean
        @test norm(sample_cov - Σ) < 2.0   # Within 2.0 of true cov
    end
end
