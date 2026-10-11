@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

mutable struct GridRestartRNG <: Random.AbstractRNG
    exps::Vector{Float64}
    uniforms::Vector{Float64}
end
Random.randexp(rng::GridRestartRNG) = popfirst!(rng.exps)
Random.rand(rng::GridRestartRNG) = popfirst!(rng.uniforms)

@testset "Grid rejection restarts retain elapsed time and competing events" begin
    model = PDMPModel(1, FullGradient((out, x) -> copyto!(out, x)),
        (out, x, v) -> copyto!(out, v))
    strategy = GridThinningStrategy(N=1, N_min=1, t_max=1.0,
        bound=:constant, curvature_backend=:exact, bound_violation=:throw,
        max_rejections_before_tail_restart=1)
    for (refresh, horizon, exps, expected) in (
            (0.0, Inf, [1.0, 0.12, 0.12], :reflect),
            (0.0, 0.55, [1.0, 100.0, 100.0], :horizon_hit),
            (1.0, Inf, [0.55, 1.0, 100.0, 100.0], :refresh))
        flow = BouncyParticle(1, refresh)
        state, model_, alg, cache, stats = PDMPSamplers.initialize_state(
            Random.Xoshiro(1), flow, model, strategy, 0.0,
            SkeletonPoint([1.0], [1.0]); statistic_counter=PDMPSamplers.DevelStatisticCounter)
        time, event, _ = PDMPSamplers.next_event_time(
            GridRestartRNG(exps, [0.9, 0.1]), model_, flow, alg, state,
            cache, stats, horizon, refresh > 0)
        @test event === expected
        @test time > 0.5
        expected === :reflect || @test time ≈ 0.55
        @test stats.grid_budget_tail_restarts == 1
        @test stats.grid_bound_violations == 0
        @test state.ξ.x == [1.0]
    end
end
