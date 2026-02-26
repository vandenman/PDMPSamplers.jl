using Test
using PDMPSamplers
using Random
using LinearAlgebra
using BridgeStan
using Statistics
import JSON

@testset "BridgeStan Extension" begin

    # Set up test problem - multivariate Gaussian
    d = 3
    μ = zeros(d)
    Σ = Matrix(1.0I(d))

    # Create temporary directory for Stan files
    model_dir = mktempdir()

    try
        # Copy or create Stan model
        stan_code = """
        data {
            int<lower=1> N;
            vector[N] mu;
            matrix[N, N] sigma;
        }
        parameters {
            vector[N] x;
        }
        model {
            x ~ multi_normal(mu, sigma);
        }
        """

        stan_file = joinpath(model_dir, "mvnormal.stan")
        data_file = joinpath(model_dir, "mvnormal_data.json")

        write(stan_file, stan_code)

        # Create Stan data
        data_dict = Dict(
            "N" => d,
            "mu" => μ,
            "sigma" => collect(eachrow(Σ))
        )

        JSON.json(data_file, data_dict)

        model = PDMPModel(stan_file, data_file)

        alg = GridThinningStrategy()

        flow = ZigZag(d)
        x0 = randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        T = 1000.0

        # Run sampler
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T, progress=false)

        # Basic checks
        @test length(trace) > 10
        @test stats.∇f_calls > 0
        @test stats.reflections_accepted > 0
        @test stats.reflections_accepted / stats.reflections_events > 0.1

        # Check that samples are approximately correct
        sample_mean = mean(trace)
        sample_cov = cov(trace)

        # Very loose checks (just that it's working)
        @test norm(sample_mean - μ) < 0.5  # Within 0.5 of true mean
        @test norm(sample_cov - Σ) < 0.75   # Within 0.75 of true cov

        # Note: This test requires a compiled Stan model
        # In practice, you would compile the model beforehand
        # For now, we skip if the model isn't compiled
        # @info "BridgeStan test requires pre-compiled Stan model, skipping automatic compilation"
        # @test_skip true  # Skip until we have a pre-compiled model in CI

    finally
        rm(model_dir, recursive=true, force=true)
    end
end
