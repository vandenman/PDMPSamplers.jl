using Test
using PDMPSamplers
using Random
using LinearAlgebra
using BridgeStan
import JSON3 as JSON

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
            "sigma" => Σ
        )

        open(data_file, "w") do io
            JSON.write(io, data_dict)
        end

        # Note: This test requires a compiled Stan model
        # In practice, you would compile the model beforehand
        # For now, we skip if the model isn't compiled
        @info "BridgeStan test requires pre-compiled Stan model, skipping automatic compilation"
        @test_skip true  # Skip until we have a pre-compiled model in CI

    finally
        rm(model_dir, recursive=true, force=true)
    end
end
