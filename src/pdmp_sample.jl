const SeedSpec = Union{
    Nothing,
    Integer,
    AbstractVector{<:Integer},
    Random.AbstractRNG,
    AbstractVector{<:Random.AbstractRNG},
}

function pdmp_sample(d::Integer, args...; seed::SeedSpec=nothing, kwargs...)
    n_chains = _infer_n_chains(args, kwargs)
    x₀ = randn(make_initial_rng(seed, n_chains), d)
    return pdmp_sample(x₀, args...; seed, kwargs...)
end

pdmp_sample(x₀::AbstractVector, θ₀::AbstractVector, flow::ContinuousDynamics, args...; kwargs...) = pdmp_sample(SkeletonPoint(x₀, θ₀), flow, args...; kwargs...)
function pdmp_sample(x₀::AbstractVector, flow::ContinuousDynamics, args...; seed::SeedSpec=nothing, kwargs...)
    n_chains = _infer_n_chains(args, kwargs)
    θ₀ = initialize_velocity(make_initial_rng(seed, n_chains), flow, length(x₀))
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

"""
    make_initial_rng(seed, n_chains::Int) -> AbstractRNG

Return the random number generator `pdmp_sample` uses for the first chain
given the same `seed` and `n_chains`, so callers can initialize a state with
the draws the sampler would make. Part of the extension interface.
"""
function make_initial_rng(seed::SeedSpec, n_chains::Int)
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
    initial_free::Union{Nothing,AbstractVector{Bool}}=nothing,
    initial_stored_velocity::Union{Nothing,AbstractVector{<:Real}}=nothing,
    trace_storage::Union{Nothing,StreamingTraceStorage}=nothing,
)
    n_chains >= 1 || throw(ArgumentError("n_chains must be >= 1, got $n_chains"))
    _validate_seed_spec(seed, n_chains)
    support_boundary_options = _validate_support_boundary_options(support_boundary_options)
    if isone(n_chains)
        rng = make_initial_rng(seed, n_chains)
        trace, stats, installed, retained_initial, endpoint = _pdmp_sample_single(rng, ξ₀, flow, model, alg, t₀, T, t_warmup,
            progress, adapter, stop, warmup_stop, support_boundary_options, model,
            statistic_counter; initial_free, initial_stored_velocity, trace_storage)
        return PDMPChains([trace], [stats], [installed], [retained_initial], [endpoint])
    end
    models = [copy(model) for _ in 1:n_chains]
    return pdmp_sample(ξ₀, flow, models, alg, t₀, T, t_warmup;
        stop, warmup_stop, threaded, progress, adapter, seed,
        support_boundary_options, statistic_counter, initial_free,
        initial_stored_velocity, trace_storage)
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
    initial_free::Union{Nothing,AbstractVector{Bool}}=nothing,
    initial_stored_velocity::Union{Nothing,AbstractVector{<:Real}}=nothing,
    trace_storage::Union{Nothing,StreamingTraceStorage}=nothing,
)
    n_chains = length(models)
    n_chains >= 1 || throw(ArgumentError("models must be non-empty"))
    _validate_seed_spec(seed, n_chains)
    support_boundary_options = _validate_support_boundary_options(support_boundary_options)

    if isone(n_chains)
        rng = make_initial_rng(seed, n_chains)
        trace, stats, installed, retained_initial, endpoint = _pdmp_sample_single(rng, ξ₀, flow, models[1], alg, t₀, T, t_warmup,
            progress, adapter, stop, warmup_stop, support_boundary_options, models[1],
            statistic_counter; initial_free, initial_stored_velocity,
            trace_storage)
        return PDMPChains([trace], [stats], [installed], [retained_initial], [endpoint])
    end

    if threaded
        tasks = map(1:n_chains) do i
            Threads.@spawn begin
                rng_i        = _make_chain_rng(seed, i)
                flow_i        = _copy_flow(flow)
                alg_i         = _copy_algorithm(alg)
                adapter_i     = _copy_adapter(adapter)
                stop_i        = _maybe_copy_criterion(stop)
                warmup_stop_i = _maybe_copy_criterion(warmup_stop)
                _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, models[i], alg_i, t₀, T, t_warmup,
                    false, adapter_i, stop_i, warmup_stop_i, support_boundary_options, models[i],
                    statistic_counter; initial_free, initial_stored_velocity,
                    trace_storage=_chain_trace_storage(trace_storage, i))
            end
        end
        results = fetch.(tasks)
    else
        results = map(1:n_chains) do i
            rng_i        = _make_chain_rng(seed, i)
            flow_i        = _copy_flow(flow)
            alg_i         = _copy_algorithm(alg)
            adapter_i     = _copy_adapter(adapter)
            stop_i        = _maybe_copy_criterion(stop)
            warmup_stop_i = _maybe_copy_criterion(warmup_stop)
            _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, models[i], alg_i, t₀, T, t_warmup,
                false, adapter_i, stop_i, warmup_stop_i, support_boundary_options, models[i],
                statistic_counter; initial_free, initial_stored_velocity,
                trace_storage=_chain_trace_storage(trace_storage, i))
        end
    end

    traces    = [r[1] for r in results]
    all_stats = [r[2] for r in results]
    installed = [r[3] for r in results]
    retained_initial = [r[4] for r in results]
    endpoints = [r[5] for r in results]
    return PDMPChains(traces, all_stats, installed, retained_initial, endpoints)
end

_chain_trace_storage(::Nothing, chain::Integer) = nothing
function _chain_trace_storage(storage::StreamingTraceStorage, chain::Integer)
    return StreamingTraceStorage(joinpath(storage.directory,
        "chain_" * lpad(string(chain), 4, '0'));
        buffer_events=storage.buffer_events,
        write_events=storage.write_events,
        online_batches=storage.online_batches,
        online_end_time=storage.online_end_time,
        online_grid_spacing=storage.online_grid_spacing,
        online_warmup_batches=storage.online_warmup_batches)
end

_maybe_copy_criterion(::Nothing) = nothing
_maybe_copy_criterion(c::StoppingCriterion) = copy(c)

_copy_flow(flow::ContinuousDynamics) = flow
_copy_flow(flow::MutableBoomerang) = copy(flow)
_copy_flow(pd::PreconditionedDynamics) = PreconditionedDynamics(deepcopy(pd.metric), _copy_flow(pd.dynamics))
_copy_algorithm(alg::PoissonTimeStrategy) = deepcopy(alg)
_copy_adapter(adapter::AbstractAdapter) = deepcopy(adapter)

_adaptation_can_stick(::PoissonTimeStrategy) = nothing
_adaptation_can_stick(alg::Union{Sticky,AggregateSticky}) = alg.can_stick

"""
    initialize_flow_state!(state, flow)

Bring flow-specific parts of a freshly built `state` in line with `flow`
(for example the dense preconditioned Zig-Zag velocity). Call it after
constructing a state by hand. Part of the extension interface.
"""
initialize_flow_state!(::AbstractPDMPState, ::ContinuousDynamics) = nothing

function Base.copy(model::PDMPModel)
    grad_new = copy(model.grad)
    hvp_new = _copy_model_hvp(model, grad_new)
    vhv_new = model.vhv === nothing ? nothing : _copy_callable(model.vhv)
    joint_new = model.joint === nothing ? nothing : _copy_callable(model.joint)
    return PDMPModel(model.d, grad_new, hvp_new, vhv_new, false, false, joint_new)
end

_copy_model_hvp(model::PDMPModel, grad) =
    model.hvp === nothing ? nothing : _copy_callable(model.hvp)
_copy_model_hvp(model::PDMPModel{<:SubsampledControlVariate}, grad) =
    grad.deterministic_hvp! === nothing ? nothing :
        InplaceHVP(grad.deterministic_hvp!, zeros(model.d))

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
    full_grad_start::Int,
    prior_grad_start::Int,
    fd_curvature_grad_start::Int,
    potential_start::Int,
    grid_start::NamedTuple,
    pipeline_start::NamedTuple,
    time_start::UInt64,
)
    elapsed = (time_ns() - time_start) / 1e9
    if phase === :warmup
        _inc_counter_warmup_events(stats, _get_counter_reflections_events(stats) + _get_counter_refreshment_events(stats) + _get_counter_sticky_events(stats) - events_start)
        _inc_counter_warmup_gradient_calls(stats, _get_counter_∇f_calls(stats) - grad_start)
        _inc_counter_warmup_hessian_calls(stats, _get_counter_∇²f_calls(stats) - hess_start)
        _inc_counter_warmup_full_gradient_calls(stats, _get_counter_full_gradient_calls(stats) - full_grad_start)
        _inc_counter_warmup_prior_gradient_calls(stats, _get_counter_prior_gradient_calls(stats) - prior_grad_start)
        _inc_counter_warmup_fd_curvature_gradient_calls(stats, _get_counter_fd_curvature_gradient_calls(stats) - fd_curvature_grad_start)
        _inc_counter_warmup_potential_calls(stats, _get_counter_potential_calls(stats) - potential_start)
        _inc_counter_warmup_exact_curvature_calls(stats, _get_counter_∇²f_calls(stats) - hess_start)
        _inc_counter_warmup_grid_endpoint_evaluations(stats, _get_counter_grid_endpoint_evaluations(stats) - grid_start.endpoint_evaluations)
        _inc_counter_warmup_grid_endpoint_gradient_calls(stats, _get_counter_grid_endpoint_gradient_calls(stats) - grid_start.endpoint_gradient_calls)
        _inc_counter_warmup_grid_endpoint_hessian_calls(stats, _get_counter_grid_endpoint_hessian_calls(stats) - grid_start.endpoint_hessian_calls)
        _inc_counter_warmup_grid_endpoint_derivative_calls(stats, _get_counter_grid_endpoint_derivative_calls(stats) - grid_start.endpoint_derivative_calls)
        _inc_counter_warmup_grid_acceptance_gradient_calls(stats, _get_counter_grid_acceptance_gradient_calls(stats) - grid_start.acceptance_gradient_calls)
        _inc_counter_warmup_grid_acceptance_tests(stats, _get_counter_grid_acceptance_tests(stats) - grid_start.acceptance_tests)
        _inc_counter_warmup_grid_cached_endpoint_reuses(stats, _get_counter_grid_cached_endpoint_reuses(stats) - grid_start.cached_endpoint_reuses)
        _inc_counter_warmup_grid_points_evaluated(stats, _get_counter_grid_points_evaluated(stats) - grid_start.points_evaluated)
        _inc_counter_warmup_grid_endpoint_derivative_points_loaded(stats, _get_counter_grid_endpoint_derivative_points_loaded(stats) - grid_start.endpoint_derivative_points_loaded)
        _inc_counter_warmup_subsampling_cell_roof_proposals(stats,
            _get_counter_subsampling_cell_roof_proposals(stats) - grid_start.subsampling_cell_roof_proposals)
        _inc_counter_warmup_subsampling_aggregate_accepts(stats,
            _get_counter_subsampling_aggregate_accepts(stats) - grid_start.subsampling_aggregate_accepts)
        _inc_counter_warmup_subsampling_subset_evaluations(stats,
            _get_counter_subsampling_subset_evaluations(stats) - grid_start.subsampling_subset_evaluations)
        _inc_counter_warmup_subsampling_final_reflections(stats,
            _get_counter_subsampling_final_reflections(stats) - grid_start.subsampling_final_reflections)
        _inc_counter_warmup_elapsed_time(stats, elapsed)
    elseif phase === :main
        _inc_counter_main_events(stats, _get_counter_reflections_events(stats) + _get_counter_refreshment_events(stats) + _get_counter_sticky_events(stats) - events_start)
        _inc_counter_main_gradient_calls(stats, _get_counter_∇f_calls(stats) - grad_start)
        _inc_counter_main_hessian_calls(stats, _get_counter_∇²f_calls(stats) - hess_start)
        _inc_counter_main_full_gradient_calls(stats, _get_counter_full_gradient_calls(stats) - full_grad_start)
        _inc_counter_main_prior_gradient_calls(stats, _get_counter_prior_gradient_calls(stats) - prior_grad_start)
        _inc_counter_main_fd_curvature_gradient_calls(stats, _get_counter_fd_curvature_gradient_calls(stats) - fd_curvature_grad_start)
        _inc_counter_main_potential_calls(stats, _get_counter_potential_calls(stats) - potential_start)
        _inc_counter_main_exact_curvature_calls(stats, _get_counter_∇²f_calls(stats) - hess_start)
        _inc_counter_main_grid_endpoint_evaluations(stats, _get_counter_grid_endpoint_evaluations(stats) - grid_start.endpoint_evaluations)
        _inc_counter_main_grid_endpoint_gradient_calls(stats, _get_counter_grid_endpoint_gradient_calls(stats) - grid_start.endpoint_gradient_calls)
        _inc_counter_main_grid_endpoint_hessian_calls(stats, _get_counter_grid_endpoint_hessian_calls(stats) - grid_start.endpoint_hessian_calls)
        _inc_counter_main_grid_endpoint_derivative_calls(stats, _get_counter_grid_endpoint_derivative_calls(stats) - grid_start.endpoint_derivative_calls)
        _inc_counter_main_grid_acceptance_gradient_calls(stats, _get_counter_grid_acceptance_gradient_calls(stats) - grid_start.acceptance_gradient_calls)
        _inc_counter_main_grid_acceptance_tests(stats, _get_counter_grid_acceptance_tests(stats) - grid_start.acceptance_tests)
        _inc_counter_main_grid_cached_endpoint_reuses(stats, _get_counter_grid_cached_endpoint_reuses(stats) - grid_start.cached_endpoint_reuses)
        _inc_counter_main_grid_points_evaluated(stats, _get_counter_grid_points_evaluated(stats) - grid_start.points_evaluated)
        _inc_counter_main_grid_endpoint_derivative_points_loaded(stats, _get_counter_grid_endpoint_derivative_points_loaded(stats) - grid_start.endpoint_derivative_points_loaded)
        _inc_counter_main_subsampling_cell_roof_proposals(stats,
            _get_counter_subsampling_cell_roof_proposals(stats) - grid_start.subsampling_cell_roof_proposals)
        _inc_counter_main_subsampling_aggregate_accepts(stats,
            _get_counter_subsampling_aggregate_accepts(stats) - grid_start.subsampling_aggregate_accepts)
        _inc_counter_main_subsampling_subset_evaluations(stats,
            _get_counter_subsampling_subset_evaluations(stats) - grid_start.subsampling_subset_evaluations)
        _inc_counter_main_subsampling_final_reflections(stats,
            _get_counter_subsampling_final_reflections(stats) - grid_start.subsampling_final_reflections)
        _inc_counter_main_elapsed_time(stats, elapsed)
    end
    _record_subsampling_pipeline_phase!(stats, phase, pipeline_start)
    return nothing
end

function _record_phase_stats!(
    stats::AbstractStatisticCounter,
    phase::Symbol,
    events_start::Int,
    grad_start::Int,
    hess_start::Int,
    full_grad_start::Int,
    prior_grad_start::Int,
    fd_curvature_grad_start::Int,
    potential_start::Int,
    pipeline_start::NamedTuple,
    time_start::UInt64,
)
    grid_start = (;
        endpoint_evaluations=0,
        endpoint_gradient_calls=0,
        endpoint_hessian_calls=0,
        endpoint_derivative_calls=0,
        acceptance_gradient_calls=0,
        acceptance_tests=0,
        cached_endpoint_reuses=0,
        points_evaluated=0,
        endpoint_derivative_points_loaded=0,
        subsampling_cell_roof_proposals=0,
        subsampling_aggregate_accepts=0,
        subsampling_subset_evaluations=0,
        subsampling_final_reflections=0,
    )
    return _record_phase_stats!(
        stats, phase, events_start, grad_start, hess_start, full_grad_start, prior_grad_start,
        fd_curvature_grad_start, potential_start, grid_start, pipeline_start,
        time_start)
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
    boundary_policy::BoundaryPolicy;
    adaptation_grad=model_.grad,
) where {FL<:ContinuousDynamics}
    initialize!(criterion, state, trace_manager, stats)
    begin_trace_phase!(trace_manager, state, flow, phase)
    if phase === :main && !(get_main_trace(trace_manager) isa StreamingPDMPTrace)
        record_event!(trace_manager, state, flow, nothing, phase)
    end

    _maybe_simplify_counter = 0
    phase_events_start = _get_counter_reflections_events(stats) + _get_counter_refreshment_events(stats) + _get_counter_sticky_events(stats)
    phase_grad_start = _get_counter_∇f_calls(stats)
    phase_hess_start = _get_counter_∇²f_calls(stats)
    phase_full_grad_start = _get_counter_full_gradient_calls(stats)
    phase_prior_grad_start = _get_counter_prior_gradient_calls(stats)
    phase_fd_curvature_grad_start = _get_counter_fd_curvature_gradient_calls(stats)
    phase_potential_start = _get_counter_potential_calls(stats)
    phase_grid_start = (;
        endpoint_evaluations=_get_counter_grid_endpoint_evaluations(stats),
        endpoint_gradient_calls=_get_counter_grid_endpoint_gradient_calls(stats),
        endpoint_hessian_calls=_get_counter_grid_endpoint_hessian_calls(stats),
        endpoint_derivative_calls=_get_counter_grid_endpoint_derivative_calls(stats),
        acceptance_gradient_calls=_get_counter_grid_acceptance_gradient_calls(stats),
        acceptance_tests=_get_counter_grid_acceptance_tests(stats),
        cached_endpoint_reuses=_get_counter_grid_cached_endpoint_reuses(stats),
        points_evaluated=_get_counter_grid_points_evaluated(stats),
        endpoint_derivative_points_loaded=_get_counter_grid_endpoint_derivative_points_loaded(stats),
        subsampling_cell_roof_proposals=_get_counter_subsampling_cell_roof_proposals(stats),
        subsampling_aggregate_accepts=_get_counter_subsampling_aggregate_accepts(stats),
        subsampling_subset_evaluations=_get_counter_subsampling_subset_evaluations(stats),
        subsampling_final_reflections=_get_counter_subsampling_final_reflections(stats),
    )
    phase_pipeline_start = _subsampling_pipeline_snapshot(stats)
    phase_time_start = time_ns()

    while true
        if is_satisfied(criterion, state, trace_manager, stats)
            _set_counter_stop_reason(stats, stop_reason(criterion, state, trace_manager, stats))
            _record_phase_stats!(
                stats, phase, phase_events_start, phase_grad_start, phase_hess_start,
                phase_full_grad_start, phase_prior_grad_start, phase_fd_curvature_grad_start, phase_potential_start,
                phase_grid_start, phase_pipeline_start, phase_time_start)
            return nothing
        end

        criterion_horizon = _step_horizon(criterion, state)
        adapter_horizon = adaptation_horizon(
            adapter, state, flow, adaptation_grad, phase, stats)
        adapter_owns_horizon = isfinite(adapter_horizon) &&
            adapter_horizon <= criterion_horizon
        prepare_trace_storage_boundary!(trace_manager, phase)
        event_type = _step!(rng, state, model_, flow, alg_, cache, stats,
            trace_manager, boundary_policy, phase,
            min(criterion_horizon, adapter_horizon),
            adapter_owns_horizon ? :anchor_selection_boundary : :horizon_hit)
        event_type === :anchor_selection_boundary ||
            update!(criterion, state, trace_manager, stats, event_type)

        adapt!(rng, adapter, state, flow, adaptation_grad, trace_manager;
            phase, stats, event_type,
            anchor_boundary_won=event_type === :anchor_selection_boundary)
        _handle_gradient_adaptation!(adapter, alg_)
        _handle_dynamics_adaptation!(rng, adapter, alg_, state, flow, stats,
            trace_manager, phase)

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

function _handle_gradient_adaptation!(
    adapter::AbstractAdapter,
    alg_::PoissonTimeStrategy,
)
    did_gradient_adapt(adapter) || return nothing
    _invalidate_cached_gradient!(alg_)
    return nothing
end

function _handle_dynamics_adaptation!(
    rng::Random.AbstractRNG,
    adapter::AbstractAdapter,
    alg_::PoissonTimeStrategy,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    stats::AbstractStatisticCounter,
    trace_manager::TraceManager,
    phase::Symbol,
)

    !did_dynamics_adapt(adapter) && return nothing

    prepare_dynamics_adaptation_storage_boundary!(trace_manager, phase,
        state)
    record_dynamics_adaptation!(trace_manager, state, flow, phase)

    _reset_inner_grid!(alg_)
    _inc_counter_grid_resets_from_dynamics_adaptation(stats)

    state isa StickyPDMPState && _invalidate_boundary_velocity_cache!(state)
    alg_ isa Union{StickyLoopState,AggregateStickyLoopState} && rebuild_sticky_schedule!(rng, alg_, state, flow)

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
    support_boundary_options::SupportBoundaryOptions;
    adaptation_grad=model_.grad,
)
    return _run_phase!(rng, criterion, state, model_, flow, alg_, cache, trace_manager, stats, health,
        phase, adapter, progress, prg, tstop, T, progress_stops, boundary_policy;
        adaptation_grad)
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
    support_boundary_options::SupportBoundaryOptions;
    adaptation_grad=model_.grad,
)
    try
        return _run_phase!(rng, criterion, state, model_, flow, alg_, cache, trace_manager, stats, health,
            phase, adapter, progress, prg, tstop, T, progress_stops, boundary_policy;
            adaptation_grad)
    catch err
        if err isa _ProbeFailureException
            _handle_boundary!(original_model, err.ctx, support_boundary_options)
        else
            rethrow()
        end
    end
end

function _run_phase_for_policy!(
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
    support_boundary_options::SupportBoundaryOptions;
    adaptation_grad=model_.grad,
)
    return _run_phase!(rng, criterion, state, model_, flow, alg_, cache, trace_manager, stats,
        health, phase, adapter, progress, prg, tstop, T, progress_stops, boundary_policy;
        adaptation_grad)
end

function _run_phase_for_policy!(
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
    support_boundary_options::SupportBoundaryOptions;
    adaptation_grad=model_.grad,
)
    return _run_phase_with_boundary_policy!(rng, criterion, state, model_, flow, alg_, cache,
        trace_manager, stats, health, phase, adapter, progress, prg, tstop, T, progress_stops,
        boundary_policy, original_model, support_boundary_options; adaptation_grad)
end

_phase_criterion(::Nothing, T::Real) = _CommonPhaseCriterion(; T)
_phase_criterion(c::FixedTimeCriterion, ::Real) = _CommonPhaseCriterion(; T=c.T)
_phase_criterion(c::EventCountCriterion, ::Real) =
    _CommonPhaseCriterion(; max_events=c.max_events, event_source=c)
_phase_criterion(c::StoppingCriterion, ::Real) = c

_step_horizon(::StoppingCriterion, state) = Inf
_step_horizon(c::FixedTimeCriterion, state) = max(0.0, c.T - state.t[])
_step_horizon(c::_CommonPhaseCriterion, state) = c.check_time ? max(0.0, c.T - state.t[]) : Inf
_step_horizon(c::AdaptiveWarmupCriterion, state) =
    max(0.0, c.start_time + c.max_time - state.t[])
_step_horizon(c::AnyCriterion, state) = minimum(child -> _step_horizon(child, state), c.criteria)

function _pdmp_sample_single(
    rng::Random.AbstractRNG,
    ξ₀::SkeletonPoint, flow::FL, model::PDMPModel,
    alg::PoissonTimeStrategy,
    t₀::Real, T::Real, t_warmup::Real,
    progress::Bool, adapter::AbstractAdapter,
    stop::Union{StoppingCriterion,Nothing}, warmup_stop::Union{StoppingCriterion,Nothing},
    support_boundary_options::SupportBoundaryOptions, original_model::PDMPModel,
    statistic_counter;
    initial_free::Union{Nothing,AbstractVector{Bool}}=nothing,
    initial_stored_velocity::Union{Nothing,AbstractVector{<:Real}}=nothing,
    trace_storage::Union{Nothing,StreamingTraceStorage}=nothing,
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
    initialization_allocated_start = Base.gc_bytes()

    state, phase_model, phase_alg, phase_cache, stats = initialize_state(
        rng, flow, model, alg, t₀, ξ₀;
        statistic_counter, initial_free, initial_stored_velocity)
    installed_initial_state = PDMPTerminalState(state, flow)

    validate_state(state, flow, "at initialization")

    t_warmup_abs = t₀ + t_warmup
    trace_manager = trace_storage === nothing ?
        TraceManager(state, flow, alg, t_warmup_abs) :
        TraceManager(state, flow, alg, t_warmup_abs, trace_storage)
    health = HealthMonitor()
    # Adapt through the statistics-wrapped gradient that actually drives the
    # warmup phase. For a subsampled warmup it is the authoritative live CV
    # whose envelope was copied by `with_stats`. Passing the unwrapped
    # construction-time CV here lets anchor callbacks update shared providers
    # while leaving the live phase envelope stale.
    warmup_grad = phase_model.grad
    adapter = adapter isa NoAdaptation ? default_warmup_adapter(
        flow, warmup_grad, t_warmup, t₀;
        can_stick=_adaptation_can_stick(alg)) : adapter

    warmup_criterion = _phase_criterion(warmup_stop, t_warmup_abs)
    stop_criterion = _phase_criterion(stop, T)
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
    _set_counter_initialization_elapsed_time(stats, (time_ns() - t_start) / 1e9)
    _set_counter_initialization_allocated_bytes(
        stats, Float64(Base.gc_bytes() - initialization_allocated_start))

    T_float = Float64(T)
    if !(isnothing(warmup_stop) && t_warmup_abs <= t₀)
        warmup_phase_start = time_ns()
        _run_phase_for_policy!(rng, warmup_criterion, state, phase_model, flow, phase_alg, phase_cache,
            trace_manager, stats, health, :warmup, adapter, progress, prg, tstop, T_float,
            progress_stops, boundary_policy, model, support_boundary_options;
            adaptation_grad=warmup_grad)
        finish_trace_phase!(trace_manager, state, flow, :warmup)
        _set_counter_warmup_phase_elapsed_time(
            stats, (time_ns() - warmup_phase_start) / 1e9)
    end

    transition_start = time_ns()
    adapter_finish_start = time_ns()
    did_adapt = finish_warmup!(adapter, state, flow, warmup_grad, trace_manager, stats)
    scale_min, scale_max = _metric_scale_extrema(flow)
    free_time_min, free_time_max = _adaptation_free_time_extrema(adapter)
    _set_counter_final_metric_scale_min(stats, scale_min)
    _set_counter_final_metric_scale_max(stats, scale_max)
    _set_counter_adaptation_updates(stats, _adaptation_update_count(adapter))
    _set_counter_preconditioner_updates(
        stats, _preconditioner_update_count(adapter))
    _set_counter_boomerang_updates(stats, _boomerang_update_count(adapter))
    _set_counter_adaptation_free_time_min(stats, free_time_min)
    _set_counter_adaptation_free_time_max(stats, free_time_max)
    _set_counter_warmup_adapter_finish_elapsed_time(
        stats, (time_ns() - adapter_finish_start) / 1e9)
    main_sampler_initialization_start = time_ns()
    main_sampler_initialization_allocated_start = Base.gc_bytes()
    did_adapt && _reset_inner_grid!(phase_alg)
    _set_counter_main_sampler_initialization_elapsed_time(
        stats, (time_ns() - main_sampler_initialization_start) / 1e9)
    _set_counter_main_sampler_initialization_allocated_bytes(stats,
        Float64(Base.gc_bytes() - main_sampler_initialization_allocated_start))
    algorithm_finish_start = time_ns()
    finish_warmup!(phase_alg, stats, flow)
    _maybe_activate_constant_bound!(phase_alg, stats)
    _set_counter_algorithm_warmup_finish_elapsed_time(
        stats, (time_ns() - algorithm_finish_start) / 1e9)
    _set_counter_transition_elapsed_time(stats, (time_ns() - transition_start) / 1e9)

    retained_initial_state = PDMPTerminalState(state, flow)
    main_phase_start = time_ns()
    main_phase_allocated_start = Base.gc_bytes()
    _run_phase_for_policy!(rng, stop_criterion, state, phase_model, flow,
        phase_alg, phase_cache, trace_manager, stats, health, :main,
        adapter, progress, prg, tstop, T_float, progress_stops,
        boundary_policy, original_model, support_boundary_options)
    finish_trace_phase!(trace_manager, state, flow, :main)
    _set_counter_main_phase_elapsed_time(stats, (time_ns() - main_phase_start) / 1e9)
    _set_counter_main_phase_allocated_bytes(
        stats, Float64(Base.gc_bytes() - main_phase_allocated_start))

    finalization_start = time_ns()
    _record_final_grid_state!(phase_alg, stats)
    trace = compact(get_main_trace(trace_manager))
    _set_counter_finalization_elapsed_time(stats, (time_ns() - finalization_start) / 1e9)
    _set_counter_elapsed_time(stats, (time_ns() - t_start) / 1e9)

    endpoint = PDMPTerminalState(state, flow)
    return trace, stats, installed_initial_state, retained_initial_state, endpoint

end

function initialize_state(rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint;
        statistic_counter=StatisticCounter, initial_free=nothing,
        initial_stored_velocity=nothing)
    ξ = copy(ξ₀)
    t = t₀
    stats = statistic_counter()
    state = if requires_sticky_state(alg)
        free = isnothing(initial_free) ?
            .!(iszero.(ξ.x) .&& iszero.(ξ.θ)) : BitVector(initial_free)
        stored = isnothing(initial_stored_velocity) ? zeros(length(ξ)) :
            Float64.(initial_stored_velocity)
        length(free) == length(ξ) || throw(DimensionMismatch(
            "initial_free length does not match state dimension"))
        length(stored) == length(ξ) || throw(DimensionMismatch(
            "initial_stored_velocity length does not match state dimension"))
        StickyPDMPState(Ref(float(t)), ξ, free, stored)
    else
        (isnothing(initial_free) && isnothing(initial_stored_velocity)) ||
            throw(ArgumentError("sticky continuation state supplied to a non-sticky sampler"))
        PDMPState(t, ξ)
    end
    initialize_flow_state!(state, flow)
    cache = add_gradient_to_cache(initialize_cache(rng, flow, model.grad, alg, t, ξ), ξ)
    model_ = with_stats(model, stats)
    alg_ = _to_internal(alg, rng, flow, model_, state, cache, stats)
    return state, model_, alg_, cache, stats
end
initialize_state(flow::ContinuousDynamics, model::PDMPModel, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint) = initialize_state(Random.default_rng(), flow, model, alg, t₀, ξ₀)
