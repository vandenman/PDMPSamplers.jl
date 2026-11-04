using Test
using PDMPSamplers
using Random
using LinearAlgebra
using DynamicPPL
using Distributions
using Statistics
import ADTypes
import ForwardDiff
import Mooncake
import DifferentiationInterface as DI

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
        θ0 = PDMPSamplers.initialize_velocity(Random.default_rng(), flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        T = 1000.0

        # Run sampler
        trace, stats = only(pdmp_sample(ξ0, flow, model, alg, 0.0, T, progress=false))

        # Basic checks
        @test length(trace) > 10
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

    @testset "GridThinningStrategy with needs_hvp=true" begin
        Random.seed!(789)

        dpppl_model = mvnormal_model(d)
        backend = ADTypes.AutoForwardDiff()
        model = PDMPModel(dpppl_model, backend; needs_hvp=true)

        @test model.vhv !== nothing
        @test model.joint !== nothing

        μ = zeros(d)
        Σ = Matrix(1.0I(d))

        for FlowType in (ZigZag, BouncyParticle)
            flow = FlowType == ZigZag ? ZigZag(inv(Σ), μ) : BouncyParticle(inv(Σ), μ)
            alg = GridThinningStrategy()

            x0 = randn(d)
            θ0 = PDMPSamplers.initialize_velocity(Random.default_rng(), flow, d)
            ξ0 = SkeletonPoint(x0, θ0)

            trace, stats = only(pdmp_sample(ξ0, flow, model, alg, 0.0, 1000.0; progress=false))

            @test length(trace) > 10
            @test stats.∇f_calls > 0
            @test stats.∇²f_calls > 0

            samples = Matrix(PDMPDiscretize(trace, 1.0))
            sample_mean = vec(mean(samples, dims=1))
            @test norm(sample_mean - μ) < 1.0
        end
    end

    # Bounded parameters must be sampled on the unconstrained scale.
    # λ ~ Gamma(2, 1) and y | λ ~ Poisson(λ) give λ | y ~ Gamma(2 + Σy, rate 1 + n).
    # On η = log λ the negative log density (with Jacobian) is
    # -(2 + Σy) η + (1 + n) exp(η).
    @model function poisson_gamma(y)
        λ ~ Gamma(2.0, 1.0)
        for i in eachindex(y)
            y[i] ~ Poisson(λ)
        end
    end
    y_counts = [3, 1, 4, 1, 5, 2]
    a_post = 2 + sum(y_counts)
    b_post = 1 + length(y_counts)
    posterior = Gamma(a_post, 1 / b_post)
    neg_grad_η(η) = -a_post + b_post * exp(η)
    neg_hess_η(η) = b_post * exp(η)

    @testset "Constrained parameters are sampled unconstrained" begin
        model = PDMPModel(poisson_gamma(y_counts), ADTypes.AutoForwardDiff(); needs_hvp=true)
        @test model.d == 1

        # η = -3 is valid on the unconstrained scale but outside the support of λ.
        g = zeros(1)
        PDMPSamplers.compute_gradient!(model.grad, [-3.0], g)
        @test g[1] ≈ neg_grad_η(-3.0)
        @test model.hvp([-3.0], [1.0])[1] ≈ neg_hess_η(-3.0)

        chains = pdmp_sample([0.0], ZigZag(1), model, GridThinningStrategy(), 0.0, 20_000.0;
            seed=1, progress=false)
        λ_draws = exp.(Matrix(PDMPDiscretize(chains, 0.5)))
        @test mean(λ_draws) ≈ mean(posterior) rtol=0.05
        @test std(λ_draws) ≈ std(posterior) rtol=0.1
    end

    @testset "Mooncake (reverse mode) without HVP" begin
        model = PDMPModel(poisson_gamma(y_counts), ADTypes.AutoMooncake(); needs_hvp=false)
        g = zeros(1)
        PDMPSamplers.compute_gradient!(model.grad, [0.5], g)
        @test g[1] ≈ neg_grad_η(0.5)
    end

    @testset "SecondOrder backend for HVPs" begin
        backend = DI.SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff())
        model = PDMPModel(poisson_gamma(y_counts), backend; needs_hvp=true)
        g = zeros(1)
        PDMPSamplers.compute_gradient!(model.grad, [0.5], g)
        @test g[1] ≈ neg_grad_η(0.5)
        @test model.hvp([0.5], [1.0])[1] ≈ neg_hess_η(0.5)
        @test model.vhv([0.5], [1.0], [1.0]) ≈ neg_hess_η(0.5)
    end
end
