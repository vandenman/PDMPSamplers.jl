using Test
using PDMPSamplers
using Random
using LinearAlgebra
using Statistics
import JSON

const _skip_bridgestan_extension_test = Sys.isapple() && Sys.ARCH == :aarch64 && get(ENV, "CI", "") == "true"

if _skip_bridgestan_extension_test
    @testset "BridgeStan Extension" begin
        @test true
    end
else
    @eval using BridgeStan

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

            # the same tests but with HVP disabled (should still work, just less efficiently)
            model = PDMPModel(stan_file, data_file; hvp=false)

            flow = ZigZag(d)
            x0 = randn(d)
            θ0 = PDMPSamplers.initialize_velocity(flow, d)
            ξ0 = SkeletonPoint(x0, θ0)

            T = 500.0

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

            # ── Unit tests for FastBridgeStanModel internals ──────────────────────────

            # N(0, I_d): ∇ log p(x) = -x
            x_test = [1.0, -2.0, 0.5]
            v_test = [3.0, -1.0, 2.0]

            @testset "fast_log_density_gradient!" begin
                sm_unit = BridgeStan.StanModel(stan_file, data_file)
                model_unit = PDMPModel(sm_unit; hvp=false)
                out = zeros(d)
                PDMPSamplers.compute_gradient!(model_unit.grad, x_test, out)
                @test norm(out - x_test) < 1e-8
            end

            @testset "fast_log_density_hvp!" begin
                sm_unit_hvp = BridgeStan.StanModel(stan_file, data_file)
                model_unit_hvp = PDMPModel(sm_unit_hvp; hvp=true)
                result = model_unit_hvp.hvp(x_test, v_test)
                @test norm(result - v_test) < 1e-8
            end

            @testset "PDMPModel(sm; hvp=true)" begin
                sm_hvp = BridgeStan.StanModel(stan_file, data_file)
                model_hvp = PDMPModel(sm_hvp; hvp=true)
                flow_hvp = ZigZag(d)
                x0_hvp = randn(d)
                θ0_hvp = PDMPSamplers.initialize_velocity(flow_hvp, d)
                ξ0_hvp = SkeletonPoint(x0_hvp, θ0_hvp)
                trace_hvp, stats_hvp = pdmp_sample(ξ0_hvp, flow_hvp, model_hvp, alg, 0.0, 1000.0; progress=false)
                @test length(trace_hvp) > 10
                @test stats_hvp.∇f_calls > 0
                @test stats_hvp.reflections_accepted > 0
                sample_mean_hvp = mean(trace_hvp)
                sample_cov_hvp = cov(trace_hvp)
                @test norm(sample_mean_hvp - μ) < 0.5
                @test norm(sample_cov_hvp - Σ) < 0.75
            end

        finally
            rm(model_dir, recursive=true, force=true)
        end
    end
end
