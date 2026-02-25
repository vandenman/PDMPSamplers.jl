@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Trace estimators and ESS" begin

    d = 5
    target = gen_data(MvNormal, d, 2.0)
    D = target.D
    Γ = inv(Symmetric(cov(D)))
    μ = mean(D)
    grad = FullGradient(Base.Fix1(neg_gradient!, target))
    model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
    alg = GridThinningStrategy()
    T = 100_000.0

    @testset "$pdmp_type" for pdmp_type in (ZigZag, BouncyParticle, Boomerang)

        flow = pdmp_type(Γ, μ)
        ξ0 = SkeletonPoint(μ .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

        # --- Discretized vs analytic estimators ---
        dt = mean(diff(PDMPSamplers.event_times(trace)))

        samples0 = collect(PDMPDiscretize(trace, dt))
        @test all(isassigned(samples0, i) for i in eachindex(samples0))
        time_range = PDMPSamplers._to_range(PDMPDiscretize(trace, dt))
        @test first.(samples0) == time_range

        samples = Matrix(PDMPDiscretize(trace, dt))

        min_ess = minimum(MCMCDiagnosticTools.ess.(eachcol(samples)))
        se_factor = 2.0 / sqrt(min_ess)

        rtol = max(0.1, se_factor * 0.5)
        atol = max(0.1, se_factor * 0.1)

        @test vec(mean(samples, dims=1)) ≈ mean(trace) rtol = rtol atol = atol
        @test vec(std(samples, dims=1)) ≈ std(trace) rtol = rtol atol = atol
        @test vec(var(samples, dims=1)) ≈ var(trace) rtol = rtol atol = atol
        @test cov(samples) ≈ cov(trace) rtol = rtol atol = atol
        @test cor(samples) ≈ cor(trace) rtol = rtol atol = atol

        # --- ESS ---
        e = ess(trace)
        @test length(e) == d
        @test all(e .> 0)

        # batch-means ESS vs discretized ESS should agree within an order of magnitude
        # (Boomerang's curved dynamics cause higher autocorrelation in discretized samples)
        if !(pdmp_type === Boomerang)
            e_disc = MCMCDiagnosticTools.ess.(eachcol(samples))
            ratios = e ./ e_disc
            @test all(r -> 0.1 < r < 10.0, ratios)
        end

        # different n_batches should give similar results (within factor 3)
        e1 = ess(trace; n_batches=30)
        e2 = ess(trace; n_batches=100)
        @test length(e1) == d
        @test length(e2) == d
        @test all(e1 .> 0)
        @test all(e2 .> 0)
        ratios_batches = e1 ./ e2
        @test all(r -> 0.3 < r < 3.0, ratios_batches)

        # --- CDF and quantile ---
        marginal_σ = sqrt.(diag(cov(D)))

        for j in 1:d
            marginal_dist = Normal(μ[j], marginal_σ[j])

            # CDF at the true mean should be ≈ 0.5 (by symmetry)
            @test PDMPSamplers.cdf(trace, μ[j]; coordinate=j) ≈ 0.5 atol = 0.1

            # CDF at the true median ± 2σ should bracket the distribution
            @test PDMPSamplers.cdf(trace, μ[j] - 3marginal_σ[j]; coordinate=j) < 0.05
            @test PDMPSamplers.cdf(trace, μ[j] + 3marginal_σ[j]; coordinate=j) > 0.95

            # CDF should match discretized empirical CDF
            disc_cdf_at_mean = mean(samples[:, j] .≤ μ[j])
            @test PDMPSamplers.cdf(trace, μ[j]; coordinate=j) ≈ disc_cdf_at_mean atol = atol

            # CDF should match the true CDF
            for q_val in [μ[j] - marginal_σ[j], μ[j], μ[j] + marginal_σ[j]]
                true_cdf = Distributions.cdf(marginal_dist, q_val)
                @test PDMPSamplers.cdf(trace, q_val; coordinate=j) ≈ true_cdf atol = 0.1
            end
        end

        # Quantile: compare against true quantiles and discretized quantiles
        ps = [0.1, 0.25, 0.5, 0.75, 0.9]
        scalar_q = Dict{Tuple{Float64,Int}, Float64}()
        for p in ps
            q_trace = quantile(trace, p)
            @test length(q_trace) == d

            for j in 1:d
                marginal_dist = Normal(μ[j], marginal_σ[j])
                q_true = quantile(marginal_dist, p)
                q_disc = quantile(samples[:, j], p)
                q_j = quantile(trace, p; coordinate=j)
                scalar_q[(p, j)] = q_j

                @test q_j ≈ q_trace[j]
                @test q_j ≈ q_true atol = max(0.5, 2 * se_factor * marginal_σ[j])
                @test q_j ≈ q_disc atol = atol * marginal_σ[j]
            end
        end

        # Vector-of-quantiles: single sweep should match scalar calls
        for j in 1:d
            q_vec = quantile(trace, ps; coordinate=j)
            @test length(q_vec) == length(ps)
            for (k, p) in enumerate(ps)
                @test q_vec[k] ≈ scalar_q[(p, j)]
            end
        end

        # Median should equal the 0.5 quantile
        q_median = median(trace)
        @test q_median ≈ [scalar_q[(0.5, j)] for j in 1:d]
        for j in 1:d
            @test median(trace; coordinate=j) ≈ scalar_q[(0.5, j)]
        end

    end
end
