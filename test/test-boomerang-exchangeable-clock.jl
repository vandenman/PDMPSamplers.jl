# Exact thinning release clock for exchangeable Gaussian slabs on a diagonal
# Boomerang orbit, checked against the generic clock (state movement, quadrature
# hazard and bisection) that it replaces.
using PDMPSamplers, Test, Random, Statistics, LinearAlgebra

@testset "Boomerang exchangeable-slab release clock" begin
    rng = Random.Xoshiro(20260929)
    nβ = 12                       # selectable coefficients (as in the RI-CLPM)
    d = nβ + 3                    # plus three non-selectable coordinates
    beta_idx = collect(1:nβ)
    u, v = 0.35^2 * 0.75, 0.35^2 * 0.25
    free = trues(d)
    free[[2, 3, 5, 8, 9, 10, 12]] .= false        # five active coefficients
    x = randn(rng, d) .* 0.3
    θ = randn(rng, d) .* 0.4
    x[.!free] .= 0.0
    θ[.!free] .= 0.0
    stored = zeros(d)
    stored[.!free] .= 0.2 .+ rand(rng, count(.!free))  # unequal stored speeds
    can_stick = falses(d)
    can_stick[beta_idx] .= true
    n = 20_000

    providers = (ZeroMeanExchangeableGaussianSlab(beta_idx, u, v),
                 ExchangeableGaussianSlab(beta_idx, 0.1, u, v))
    means = (zeros(d), [fill(0.05, nβ); 0.4; -0.2; 0.1])
    # (state scale, horizon): a gentle case over more than half a period, and a
    # case where the conditional mean, and so the rate, varies strongly along
    # the orbit and the horizon leaves substantial mass without a release
    scenarios = ((1.0, 4.0), (5.0, 0.6))
    for provider in providers, μ in means, (scale, H) in scenarios
        flow = MutableBoomerang(Diagonal(fill(2.0, d)), μ, 0.1)
        prior = BetaBernoulliModelPrior(nβ, 1.0, 4.0)
        clock = default_aggregate_unstick_clock(provider, prior, flow)
        @test clock isa PDMPSamplers.LinearGaussianAggregateClock
        state = StickyPDMPState(0.0, SkeletonPoint(scale .* x, scale .* θ), copy(free),
                                copy(stored))
        generic = clock.fallback

        # 1. rates along the orbit agree with the generic (state-moving) rate
        for τ in (0.0, 0.3, 1.1, 2.7, 3.9)
            @test PDMPSamplers.rate(clock, flow, state, τ, can_stick) ≈
                  PDMPSamplers.rate(generic, flow, state, τ, can_stick) rtol = 1e-10
        end

        # 2. release-time distribution equals 1 - exp(-Λ(t)) with the generic
        #    quadrature hazard Λ, including the mass beyond the horizon
        draws = [PDMPSamplers.sample_time(rng, clock, flow, state, H, can_stick)
                 for _ in 1:n]
        grid = range(0, H; length = 81)
        cdf_exact = [1 - exp(-PDMPSamplers.cumulative_hazard(generic, flow, state,
                                                             0.0, t, can_stick))
                     for t in grid]
        cdf_emp = [mean(draws .<= t) for t in grid]
        @test maximum(abs.(cdf_emp .- cdf_exact)) < 1.95 / sqrt(n)   # KS, 0.1%
        @test mean(isinf.(draws)) ≈ 1 - cdf_exact[end] atol = 3 * sqrt(0.25 / n)
        @test all(t -> isinf(t) || 0 <= t < H, draws)
    end
end
