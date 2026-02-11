@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Discretized and analytic estimators agree" begin

    d = 10
    D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(MvNormal, d, 2.0, 1:d, 1:d)
    grad = FullGradient(∇f!)
    model = PDMPModel(d, grad, ∇²f!)
    Γ, μ = inv(cov(D)), mean(D)

    pdmp_types = (ZigZag, BouncyParticle, Boomerang)
    alg = GridThinningStrategy()

    @testset "flow: $(pdmp_type)" for pdmp_type in pdmp_types

        flow = pdmp_type(Γ, μ)
        ξ0 = SkeletonPoint(μ .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        T = 50_000.0
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

        dt = mean(diff([event.time for event in trace.events]))

        samples0 = collect(PDMPDiscretize(trace, dt))
        @test all(isassigned(samples0, i) for i in eachindex(samples0))
        time_range = PDMPSamplers._to_range(PDMPDiscretize(trace, dt))
        @test first.(samples0) == time_range

        samples = Matrix(PDMPDiscretize(trace, dt))

        min_ess = minimum(MCMCDiagnosticTools.ess.(eachcol(samples)))
        se_factor = 2.0 / sqrt(min_ess)

        base_rtol = 0.1
        base_atol = 0.1

        rtol = max(base_rtol, se_factor * 0.5)
        atol = max(base_atol, se_factor * 0.1)

        @test vec(mean(samples, dims=1)) ≈ mean(trace) rtol = rtol atol = atol
        @test vec(std(samples, dims=1)) ≈ std(trace) rtol = rtol atol = atol
        @test vec(var(samples, dims=1)) ≈ var(trace) rtol = rtol atol = atol
        @test cov(samples) ≈ cov(trace) rtol = rtol atol = atol
        @test cor(samples) ≈ cor(trace) rtol = rtol atol = atol

    end
end
