@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

mutable struct GridRestartRNG <: Random.AbstractRNG
    exps::Vector{Float64}
    uniforms::Vector{Float64}
end
Random.randexp(rng::GridRestartRNG) = popfirst!(rng.exps)
Random.rand(rng::GridRestartRNG) = popfirst!(rng.uniforms)
Random.randn!(::GridRestartRNG, values::Array{Float64}) = fill!(values, -1.0)

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
@testset "Rejection cap never substitutes a pending refresh" begin
    model = PDMPModel(1, FullGradient((out, x) -> copyto!(out, x)),
        (out, x, v) -> copyto!(out, v))
    flow = BouncyParticle(1, 1.0)
    for limit in (1, 100)
        strategy = GridThinningStrategy(N=1, N_min=1, t_max=1.0,
            bound=:constant, curvature_backend=:exact, bound_violation=:throw,
            safety_limit=limit, max_rejections_before_tail_restart=100)
        state, model_, alg, cache, stats = PDMPSamplers.initialize_state(
            Random.Xoshiro(1), flow, model, strategy, 0.0,
            SkeletonPoint([1.0], [1.0]); statistic_counter=PDMPSamplers.DevelStatisticCounter)
        # Roof 2: refresh at .8, reject at .5, then reflect at .56.
        rng = GridRestartRNG([0.8, 1.0, 0.12], [0.9, 0.1])
        if limit == 1
            @test_throws PDMPSamplers._GridSafetyLimitException PDMPSamplers.next_event_time(
                rng, model_, flow, alg, state, cache, stats)
        else
            time, event, _ = PDMPSamplers.next_event_time(
                rng, model_, flow, alg, state, cache, stats)
            @test event === :reflect
            @test time ≈ 0.56
        end
        @test stats.grid_bound_violations == 0
        @test stats.grid_budget_tail_restarts == 0
    end
end

@testset "Sampler step propagates rejection caps with boundary recovery enabled" begin
    model = PDMPModel(1, FullGradient((out, x) -> copyto!(out, x)),
        (out, x, v) -> copyto!(out, v))
    flow = BouncyParticle(1, 1.0)
    for mode in (:error, :line_search, :line_search_truncated_refresh), limit in (1, 100)
        strategy = GridThinningStrategy(N=1, N_min=1, t_max=1.0,
            bound=:constant, curvature_backend=:exact, bound_violation=:throw,
            safety_limit=limit, max_rejections_before_tail_restart=100)
        state, model_, alg, cache, stats = PDMPSamplers.initialize_state(
            Random.Xoshiro(1), flow, model, strategy, 0.0,
            SkeletonPoint([1.0], [1.0]); statistic_counter=PDMPSamplers.DevelStatisticCounter)
        traces = PDMPSamplers.TraceManager(state, flow, alg, 0.0)
        PDMPSamplers.begin_trace_phase!(traces, state, flow, :main)
        policy = PDMPSamplers._boundary_policy(SupportBoundaryOptions(;
            detect_boundaries=true, mode, max_refresh_attempts=1, refresh_probe_time=0.0))
        rng = GridRestartRNG([0.8, 1.0, 0.12], [0.9, 0.1])
        if limit == 1
            @test_throws PDMPSamplers._GridSafetyLimitException PDMPSamplers._step!(
                rng, state, model_, flow, alg, cache, stats, traces, policy, :main)
            @test state.t[] == 0.0
            @test state.ξ.x == [1.0]
            @test state.ξ.θ == [1.0]
            @test traces.main_trace.times == [0.0]
        else
            event = PDMPSamplers._step!(rng, state, model_, flow, alg,
                cache, stats, traces, policy, :main)
            @test event === :reflect
            @test state.t[] ≈ 0.56
            @test state.ξ.x ≈ [1.56]
            @test state.ξ.θ ≈ [-1.0]
            @test traces.main_trace.times ≈ [0.0, 0.56]
        end
        @test stats.refreshment_events == 0
        @test stats.support_boundary_events == 0
        @test stats.support_boundary_refresh_attempts == 0
    end
end
