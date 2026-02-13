@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Trace estimators and ESS" begin

    d = 5
    D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(MvNormal, d, 2.0)
    Γ = inv(Symmetric(cov(D)))
    μ = mean(D)
    grad = FullGradient(∇f!)
    model = PDMPModel(d, grad, ∇²f!)
    alg = GridThinningStrategy()
    T = 100_000.0

    @testset "$pdmp_type" for pdmp_type in (ZigZag, BouncyParticle, Boomerang)

        flow = pdmp_type(Γ, μ)
        ξ0 = SkeletonPoint(μ .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

        # --- Discretized vs analytic estimators ---
        dt = mean(diff([event.time for event in trace.events]))

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
            for i in 1:d
                ratio = e[i] / e_disc[i]
                @test 0.1 < ratio < 10.0
            end
        end

        # different n_batches should give similar results (within factor 3)
        e1 = ess(trace; n_batches=30)
        e2 = ess(trace; n_batches=100)
        @test length(e1) == d
        @test length(e2) == d
        @test all(e1 .> 0)
        @test all(e2 .> 0)
        for i in 1:d
            @test 0.3 < e1[i] / e2[i] < 3.0
        end

    end
end
