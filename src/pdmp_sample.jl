const SeedSpec = Union{
    Nothing,
    Integer,
    AbstractVector{<:Integer},
    Random.AbstractRNG,
    AbstractVector{<:Random.AbstractRNG},
}

function pdmp_sample(d::Integer, args...; seed::SeedSpec=nothing, kwargs...)
    n_chains = _infer_n_chains(args, kwargs)
    x₀ = randn(_make_initial_rng(seed, n_chains), d)
    return pdmp_sample(x₀, args...; seed, kwargs...)
end

pdmp_sample(x₀::AbstractVector, θ₀::AbstractVector, flow::ContinuousDynamics, args...; kwargs...) = pdmp_sample(SkeletonPoint(x₀, θ₀), flow, args...; kwargs...)
function pdmp_sample(x₀::AbstractVector, flow::ContinuousDynamics, args...; seed::SeedSpec=nothing, kwargs...)
    n_chains = _infer_n_chains(args, kwargs)
    θ₀ = initialize_velocity(_make_initial_rng(seed, n_chains), flow, length(x₀))
    return pdmp_sample(SkeletonPoint(x₀, θ₀), flow, args...; seed, kwargs...)
end

_make_rng(::Nothing) = Random.default_rng()
_make_rng(seed::Integer) = Random.Xoshiro(seed)
_make_rng(seed::Random.AbstractRNG) = Random.Xoshiro(_rng_seed(seed, 1))

function _infer_n_chains(args, kwargs)
    if haskey(kwargs, :n_chains)
        return Int(kwargs[:n_chains])
    elseif !isempty(args) && args[1] isa AbstractVector{<:PDMPModel}
        return length(args[1])
    end
    return 1
end

_validate_seed_spec(::Nothing, n_chains::Int) = nothing
_validate_seed_spec(::Integer, n_chains::Int) = nothing
_validate_seed_spec(::Random.AbstractRNG, n_chains::Int) = nothing

function _validate_seed_spec(seeds::AbstractVector{<:Integer}, n_chains::Int)
    length(seeds) == n_chains || throw(ArgumentError("seed vector length must match n_chains ($n_chains), got $(length(seeds))"))
    return nothing
end

function _validate_seed_spec(rngs::AbstractVector{<:Random.AbstractRNG}, n_chains::Int)
    length(rngs) == n_chains || throw(ArgumentError("seed RNG vector length must match n_chains ($n_chains), got $(length(rngs))"))
    return nothing
end

function _make_initial_rng(seed::SeedSpec, n_chains::Int)
    _validate_seed_spec(seed, n_chains)
    if seed isa Nothing
        return _make_rng(seed)
    end
    return _make_chain_rng(seed, 1)
end

function _rng_seed(rng::Random.AbstractRNG, draw_i::Int)
    rng_copy = Random.copy(rng)
    seed = zero(UInt)
    for _ in 1:draw_i
        seed = rand(rng_copy, UInt)
    end
    return seed
end

"""
    pdmp_sample(ξ₀, flow, model, alg, t₀=0.0, T=10_000, t_warmup=0.0;
                stop=nothing, warmup_stop=nothing, n_chains=1, threaded=false,
                progress=true, adapter=NoAdaptation())

Run PDMP sampling with optional stopping criteria for warmup and main sampling phases.

If `warmup_stop === nothing`, warmup uses `FixedTimeCriterion(t₀ + t_warmup)`.
If `stop === nothing`, sampling uses `FixedTimeCriterion(T)`.
Provided criteria take precedence over time arguments.

Warmup runs first with adaptation enabled and writes to the warmup trace.
Main sampling runs second with adaptation disabled and writes to the main trace.
Criteria are initialized per phase, so mutable criteria (e.g. `WallTimeCriterion`, ESS counters)
are phase-local. An exception is `TotalWallTimeCriterion`, whose timer starts once globally
and is not re-initialized per phase.
"""

function pdmp_sample(
    ξ₀::SkeletonPoint, flow::ContinuousDynamics, model::PDMPModel,
    alg::PoissonTimeStrategy,
    t₀::Real=0.0, T::Real=10_000, t_warmup::Real=0.0;
    stop::Union{StoppingCriterion,Nothing}=nothing,
    warmup_stop::Union{StoppingCriterion,Nothing}=nothing,
    n_chains::Int=1, threaded::Bool=false,
    progress::Bool=true,
    adapter::AbstractAdapter=NoAdaptation(),
    seed::SeedSpec=nothing,
    support_boundary_options::SupportBoundaryOptions=SupportBoundaryOptions(),
    statistic_counter=StatisticCounter,
)
    n_chains >= 1 || throw(ArgumentError("n_chains must be >= 1, got $n_chains"))
    _validate_seed_spec(seed, n_chains)
    support_boundary_options = _validate_support_boundary_options(support_boundary_options)
    if isone(n_chains)
        rng = _make_initial_rng(seed, n_chains)
        trace, stats = _pdmp_sample_single(rng, ξ₀, flow, model, alg, t₀, T, t_warmup,
            progress, adapter, stop, warmup_stop, support_boundary_options, model,
            statistic_counter)
        return PDMPChains([trace], [stats])
    end
    models = [_copy_model(model) for _ in 1:n_chains]
    return pdmp_sample(ξ₀, flow, models, alg, t₀, T, t_warmup;
        stop, warmup_stop, threaded, progress, adapter, seed,
        support_boundary_options, statistic_counter)
end

_make_chain_rng(::Nothing, chain_i::Int) = Random.Xoshiro()
_make_chain_rng(seed::Integer, chain_i::Int) = Random.Xoshiro(seed + chain_i - 1)
_make_chain_rng(seeds::AbstractVector{<:Integer}, chain_i::Int) = Random.Xoshiro(seeds[chain_i])
_make_chain_rng(rng::Random.AbstractRNG, chain_i::Int) = Random.Xoshiro(_rng_seed(rng, chain_i))
_make_chain_rng(rngs::AbstractVector{<:Random.AbstractRNG}, chain_i::Int) = Random.Xoshiro(_rng_seed(rngs[chain_i], 1))

function pdmp_sample(
    ξ₀::SkeletonPoint, flow::ContinuousDynamics, models::AbstractVector{<:PDMPModel},
    alg::PoissonTimeStrategy,
    t₀::Real=0.0, T::Real=10_000, t_warmup::Real=0.0;
    stop::Union{StoppingCriterion,Nothing}=nothing,
    warmup_stop::Union{StoppingCriterion,Nothing}=nothing,
    threaded::Bool=false,
    progress::Bool=true,
    adapter::AbstractAdapter=NoAdaptation(),
    seed::SeedSpec=nothing,
    support_boundary_options::SupportBoundaryOptions=SupportBoundaryOptions(),
    statistic_counter=StatisticCounter,
)
    n_chains = length(models)
    n_chains >= 1 || throw(ArgumentError("models must be non-empty"))
    _validate_seed_spec(seed, n_chains)
    support_boundary_options = _validate_support_boundary_options(support_boundary_options)

    if isone(n_chains)
        rng = _make_initial_rng(seed, n_chains)
        trace, stats = _pdmp_sample_single(rng, ξ₀, flow, models[1], alg, t₀, T, t_warmup,
            progress, adapter, stop, warmup_stop, support_boundary_options, models[1],
            statistic_counter)
        return PDMPChains([trace], [stats])
    end

    if threaded
        tasks = map(1:n_chains) do i
            Threads.@spawn begin
                rng_i        = _make_chain_rng(seed, i)
                flow_i        = _copy_flow(flow)
                stop_i        = _maybe_copy_criterion(stop)
                warmup_stop_i = _maybe_copy_criterion(warmup_stop)
                _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, models[i], alg, t₀, T, t_warmup,
                    false, adapter, stop_i, warmup_stop_i, support_boundary_options, models[i],
                    statistic_counter)
            end
        end
        results = fetch.(tasks)
    else
        results = map(1:n_chains) do i
            rng_i        = _make_chain_rng(seed, i)
            flow_i        = _copy_flow(flow)
            stop_i        = _maybe_copy_criterion(stop)
            warmup_stop_i = _maybe_copy_criterion(warmup_stop)
            _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, models[i], alg, t₀, T, t_warmup,
                false, adapter, stop_i, warmup_stop_i, support_boundary_options, models[i],
                statistic_counter)
        end
    end

    traces    = [r[1] for r in results]
    all_stats = [r[2] for r in results]
    return PDMPChains(traces, all_stats)
end

_maybe_copy_criterion(::Nothing) = nothing
_maybe_copy_criterion(c::StoppingCriterion) = copy(c)

_copy_flow(flow::ContinuousDynamics) = flow
_copy_flow(flow::MutableBoomerang) = copy(flow)
_copy_flow(pd::PreconditionedDynamics) = PreconditionedDynamics(deepcopy(pd.metric), _copy_flow(pd.dynamics))

initialize_flow_state!(::AbstractPDMPState, ::ContinuousDynamics) = nothing

function _copy_model(model::PDMPModel)
    grad_new = copy(model.grad)
    hvp_new = model.hvp === nothing ? nothing : _copy_callable(model.hvp)
    vhv_new = model.vhv === nothing ? nothing : _copy_callable(model.vhv)
    joint_new = model.joint === nothing ? nothing : _copy_callable(model.joint)
    return PDMPModel(model.d, grad_new, hvp_new, vhv_new, false, false, joint_new)
end

function _update_progress!(progress::Bool, prg, tstop::Base.RefValue{Float64}, T::Float64, progress_stops::Int, state::AbstractPDMPState)
    if progress && state.t[] > tstop[]
        tstop[] += T / progress_stops
        ProgressMeter.next!(prg)
    end
    return nothing
end

function _record_phase_stats!(
    stats::AbstractStatisticCounter,
    phase::Symbol,
    events_start::Int,
    grad_start::Int,
    hess_start::Int,
    time_start::UInt64,
)
    elapsed = (time_ns() - time_start) / 1e9
    if phase === :warmup
        _inc_counter_warmup_events(stats, _get_counter_reflections_events(stats) + _get_counter_refreshment_events(stats) + _get_counter_sticky_events(stats) - events_start)
        _inc_counter_warmup_gradient_calls(stats, _get_counter_∇f_calls(stats) - grad_start)
        _inc_counter_warmup_hessian_calls(stats, _get_counter_∇²f_calls(stats) - hess_start)
        _inc_counter_warmup_elapsed_time(stats, elapsed)
    elseif phase === :main
        _inc_counter_main_events(stats, _get_counter_reflections_events(stats) + _get_counter_refreshment_events(stats) + _get_counter_sticky_events(stats) - events_start)
        _inc_counter_main_gradient_calls(stats, _get_counter_∇f_calls(stats) - grad_start)
        _inc_counter_main_hessian_calls(stats, _get_counter_∇²f_calls(stats) - hess_start)
        _inc_counter_main_elapsed_time(stats, elapsed)
    end
    return nothing
end

function _run_phase!(
    rng::Random.AbstractRNG,
    criterion::StoppingCriterion,
    state::AbstractPDMPState,
    model_::PDMPModel,
    flow::FL,
    alg_::PoissonTimeStrategy,
    cache::NamedTuple,
    trace_manager::TraceManager,
    stats::AbstractStatisticCounter,
    health::HealthMonitor,
    phase::Symbol,
    adapter::AbstractAdapter,
    progress::Bool,
    prg,
    tstop::Base.RefValue{Float64},
    T::Float64,
    progress_stops::Int,
    boundary_policy::BoundaryPolicy
) where {FL<:ContinuousDynamics}
    initialize!(criterion, state, trace_manager, stats)
    phase === :main && record_event!(trace_manager, state, flow, nothing, phase)

    _maybe_simplify_counter = 0
    phase_events_start = _get_counter_reflections_events(stats) + _get_counter_refreshment_events(stats) + _get_counter_sticky_events(stats)
    phase_grad_start = _get_counter_∇f_calls(stats)
    phase_hess_start = _get_counter_∇²f_calls(stats)
    phase_time_start = time_ns()

    while true
        if is_satisfied(criterion, state, trace_manager, stats)
            _set_counter_stop_reason(stats, stop_reason(criterion, state, trace_manager, stats))
            _record_phase_stats!(
                stats, phase, phase_events_start, phase_grad_start, phase_hess_start, phase_time_start)
            return nothing
        end

        event_type = _step!(rng, state, model_, flow, alg_, cache, stats, trace_manager, boundary_policy, phase)
        update!(criterion, state, trace_manager, stats, event_type)

        adapt!(rng, adapter, state, flow, model_.grad, trace_manager; phase, stats)
        model_.grad isa SubsampledGradient && _invalidate_cached_gradient!(alg_)
        _handle_dynamics_adaptation!(rng, adapter, alg_, state, flow, stats)

        if phase === :main
            _maybe_simplify_counter += 1
            if _maybe_simplify_counter >= 100
                _maybe_activate_constant_bound!(alg_, stats)
                _maybe_simplify_counter = 0
            end
        end

        check_health!(health, stats)
        _update_progress!(progress, prg, tstop, T, progress_stops, state)
    end
end

function _handle_dynamics_adaptation!(
    rng::Random.AbstractRNG,
    adapter::AbstractAdapter,
    alg_::PoissonTimeStrategy,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    stats::AbstractStatisticCounter,
)

    !did_dynamics_adapt(adapter) && return nothing

    _reset_inner_grid!(alg_)
    _inc_counter_grid_resets_from_dynamics_adaptation(stats)

    alg_ isa StickyLoopState && update_all_stick_times!(rng, alg_, state, flow)

    return nothing
end

function _run_phase_with_boundary_policy!(
    rng::Random.AbstractRNG,
    criterion::StoppingCriterion,
    state::AbstractPDMPState,
    model_::PDMPModel,
    flow::ContinuousDynamics,
    alg_::PoissonTimeStrategy,
    cache::NamedTuple,
    trace_manager::TraceManager,
    stats::AbstractStatisticCounter,
    health::HealthMonitor,
    phase::Symbol,
    adapter::AbstractAdapter,
    progress::Bool,
    prg,
    tstop::Base.RefValue{Float64},
    T::Float64,
    progress_stops::Int,
    boundary_policy::NoBoundaryHandling,
    original_model::PDMPModel,
    support_boundary_options::SupportBoundaryOptions,
)
    return _run_phase!(rng, criterion, state, model_, flow, alg_, cache, trace_manager, stats, health,
        phase, adapter, progress, prg, tstop, T, progress_stops, boundary_policy)
end

function _run_phase_with_boundary_policy!(
    rng::Random.AbstractRNG,
    criterion::StoppingCriterion,
    state::AbstractPDMPState,
    model_::PDMPModel,
    flow::ContinuousDynamics,
    alg_::PoissonTimeStrategy,
    cache::NamedTuple,
    trace_manager::TraceManager,
    stats::AbstractStatisticCounter,
    health::HealthMonitor,
    phase::Symbol,
    adapter::AbstractAdapter,
    progress::Bool,
    prg,
    tstop::Base.RefValue{Float64},
    T::Float64,
    progress_stops::Int,
    boundary_policy::BoundaryHandling,
    original_model::PDMPModel,
    support_boundary_options::SupportBoundaryOptions,
)
    try
        return _run_phase!(rng, criterion, state, model_, flow, alg_, cache, trace_manager, stats, health,
            phase, adapter, progress, prg, tstop, T, progress_stops, boundary_policy)
    catch err
        if err isa _ProbeFailureException
            _handle_boundary!(original_model, err.ctx, support_boundary_options)
        else
            rethrow()
        end
    end
end

function _pdmp_sample_single(
    rng::Random.AbstractRNG,
    ξ₀::SkeletonPoint, flow::FL, model::PDMPModel,
    alg::PoissonTimeStrategy,
    t₀::Real, T::Real, t_warmup::Real,
    progress::Bool, adapter::AbstractAdapter,
    stop::Union{StoppingCriterion,Nothing}, warmup_stop::Union{StoppingCriterion,Nothing},
    support_boundary_options::SupportBoundaryOptions, original_model::PDMPModel,
    statistic_counter
) where {FL<:ContinuousDynamics}

    # TODO: it's possible to sample to have t_warmup < T...
    # we should always sample T + t_warmup!

    # TODO: this is a better design!
    # stats = StatisticCounter()
    # p = PDMPSampler(flow, with_stats(grad), alg, t₀, ξ₀)
    # (state, grad_, alg_, cache) = iterate(p)

    # this part should be a new function, so that the top part only does setup/ resuming is easy
    # s = with_progress(with_stopping_criterion(p, <conditions>))
    # trace = PDMPTrace()

    # burnin
    # Iterators.advance(trace, 0)

    # aa = 1:2:11
    # it = Iterators.drop(aa, 3)
    # first(it)

    if isnothing(warmup_stop)
        t_warmup < (T - t₀) || throw(ArgumentError("t_warmup ($t_warmup) is larger than T - t₀ ($T - $t₀ = $(T - t₀)), which implies storing nothing at all. This is probably an error."))
    end

    t_start = time_ns()

    state, model_, alg_, cache, stats = initialize_state(rng, flow, model, alg, t₀, ξ₀;
        statistic_counter)

    validate_state(state, flow, "at initialization")

    t_warmup_abs = t₀ + t_warmup
    trace_manager = TraceManager(state, flow, alg, t_warmup_abs)
    health = HealthMonitor()
    adapter = adapter isa NoAdaptation ? default_adapter(flow, model_.grad, t_warmup ÷ 10, t_warmup, t₀) : adapter

    warmup_criterion = isnothing(warmup_stop) ? FixedTimeCriterion(t_warmup_abs) : warmup_stop
    stop_criterion = isnothing(stop) ? FixedTimeCriterion(T) : stop
    boundary_policy = _boundary_policy(support_boundary_options)

    # progressmanager = ProgressManager(progress, T, t₀, t_warmup, progress_stops)
    progress_stops = 300
    if progress
        prg = ProgressMeter.Progress(progress_stops, dt=1)
        tstop = Ref(T / progress_stops)
    else
        prg = nothing
        tstop = Ref(Inf)
    end

    T_float = Float64(T)
    _run_phase_with_boundary_policy!(rng, warmup_criterion, state, model_, flow, alg_, cache,
        trace_manager, stats, health, :warmup, adapter, progress, prg, tstop, T_float,
        progress_stops, boundary_policy, original_model, support_boundary_options)

    _maybe_activate_constant_bound!(alg_, stats)

    _run_phase_with_boundary_policy!(rng, stop_criterion, state, model_, flow, alg_, cache,
        trace_manager, stats, health, :main, adapter, progress, prg, tstop, T_float,
        progress_stops, boundary_policy, original_model, support_boundary_options)

    _set_counter_elapsed_time(stats, (time_ns() - t_start) / 1e9)

    return compact(get_main_trace(trace_manager)), stats

end

function initialize_state(rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint; statistic_counter=StatisticCounter)
    ξ = copy(ξ₀)
    t = t₀
    stats = statistic_counter()
    state = alg isa Sticky ? StickyPDMPState(t, ξ) : PDMPState(t, ξ)
    initialize_flow_state!(state, flow)
    cache = add_gradient_to_cache(initialize_cache(rng, flow, model.grad, alg, t, ξ), ξ)
    alg_ = _to_internal(alg, rng, flow, model, state, cache, stats)
    model_ = with_stats(model, stats)
    model_.grad isa SubsampledGradient && model_.grad.resample_indices!(model_.grad.nsub)
    return state, model_, alg_, cache, stats
end
initialize_state(flow::ContinuousDynamics, model::PDMPModel, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint) = initialize_state(Random.default_rng(), flow, model, alg, t₀, ξ₀)
