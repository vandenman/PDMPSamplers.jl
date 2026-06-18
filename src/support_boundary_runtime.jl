abstract type BoundaryPolicy end
struct NoBoundaryHandling <: BoundaryPolicy end
struct BoundaryHandling{M} <: BoundaryPolicy
    opts::SupportBoundaryOptions
end

_boundary_policy(opts::SupportBoundaryOptions) = opts.detect_boundaries ? BoundaryHandling{opts.mode}(opts) : NoBoundaryHandling()

_next_event_time_for_step(
    rng::Random.AbstractRNG,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    state::AbstractPDMPState,
    cache::NamedTuple,
    stats::StatisticCounter,
    ::NoBoundaryHandling,
) = next_event_time(rng, model, flow, alg, state, cache, stats)

_next_event_time_for_step(
    rng::Random.AbstractRNG,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    state::AbstractPDMPState,
    cache::NamedTuple,
    stats::StatisticCounter,
    ::BoundaryHandling,
) = next_event_time(rng, model, flow, alg, state, cache, stats)

_next_event_time_for_step(
    rng::Random.AbstractRNG,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::GridAdaptiveState,
    state::AbstractPDMPState,
    cache::NamedTuple,
    stats::StatisticCounter,
    ::BoundaryHandling,
) = next_event_time(rng, model, flow, alg, state, cache, stats, Inf, true, :horizon_hit, true)

# The default hot path deliberately has no support-boundary try/catch or option
# plumbing; boundary recovery is handled by the typed BoundaryHandling method.
function _step!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model_::PDMPModel,
    flow::FL,
    alg_::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    ::NoBoundaryHandling,
    phase::Symbol
) where {FL<:ContinuousDynamics}
    τ, event_type, meta = _next_event_time_for_step(rng, model_, flow, alg_, state, cache, stats, NoBoundaryHandling())
    @assert ispositive(τ) "Proposed event time τ ($τ) is non-positive. Sampler is stuck!"

    needs_saving, saving_args = _handle_event_no_boundary!(rng, τ, model_.grad, flow, alg_, state, cache, event_type, meta, stats)
    needs_saving && record_event!(trace_manager, state, flow, saving_args, phase)

    return event_type
end

function _step!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model_::PDMPModel,
    flow::FL,
    alg_::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    boundary_policy::BoundaryHandling,
    phase::Symbol
) where {FL<:ContinuousDynamics}
    support_boundary_options = boundary_policy.opts
    try
        τ, event_type, meta = _next_event_time_for_step(rng, model_, flow, alg_, state, cache, stats, boundary_policy)

        @assert ispositive(τ) "Proposed event time τ ($τ) is non-positive. Sampler is stuck!"

        needs_saving, saving_args = handle_event!(rng, τ, model_.grad, flow, alg_, state, cache, event_type, meta, stats)
        needs_saving && record_event!(trace_manager, state, flow, saving_args, phase)

        return event_type
    catch err
        if err isa _ProbeFailureException
            return _handle_step_boundary!(rng, state, model_, flow, alg_, cache, stats, trace_manager, err.ctx, support_boundary_options; phase)
        elseif err isa _GridSafetyLimitException
            return _handle_grid_safety_limit!(rng, state, model_, flow, alg_, cache, stats, trace_manager, err.ctx, support_boundary_options; phase)
        elseif err isa SupportBoundaryError
            rethrow(err)
        else
            rethrow()
        end
    end
end

function _handle_step_boundary!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    ctx::BoundaryContext,
    opts::SupportBoundaryOptions;
    phase::Symbol
)
    opts.detect_boundaries || throw(ctx.original_error)
    if opts.mode === :line_search_truncated_refresh
        return _line_search_truncated_refresh_boundary!(rng, state, model, flow, alg, cache, stats, trace_manager, ctx, opts; phase)
    end
    if opts.mode === :line_search && !_linear_boundary_flow(flow)
        _handle_boundary!(model, ctx, SupportBoundaryOptions(; mode=:error))
    else
        _handle_boundary!(model, ctx, opts)
    end
end

function _throw_grid_safety_support_error(model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions; message::String)
    if !_support_boundary_probe_is_valid(model, ctx, ctx.t_invalid)
        localization = localize_support_boundary!(model, ctx, opts)
        throw(_build_boundary_error(ctx, opts; localized=true, localization,
            message="Grid thinning reached its safety limit after entering an invalid support region."))
    end

    throw(_build_boundary_error(ctx, opts; message))
end

function _line_search_truncated_refresh_from_current_state!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    ctx::BoundaryContext,
    opts::SupportBoundaryOptions;
    phase::Symbol
)
    if !_supports_line_search_truncated_refresh_state(state, flow)
        _throw_grid_safety_support_error(model, ctx, opts;
            message="Grid thinning reached its safety limit; :line_search_truncated_refresh is only implemented for BPS-family flows.")
    end

    x_current = copy(state.ξ.x)
    θ_current = copy(state.ξ.θ)
    t_current = state.t[]
    state_probe = _boundary_probe_state(alg, state)

    for _ in 1:opts.max_refresh_attempts
        copyto!(state.ξ.x, x_current)
        copyto!(state.ξ.θ, θ_current)
        state.t[] = t_current

        try
            _check_gradient_probe_finite(compute_gradient!(state, model.grad, flow, cache))
        catch
            _throw_grid_safety_support_error(model, ctx, opts;
                message="Grid thinning reached its safety limit, and the current state is not a valid refresh point.")
        end

        refresh_velocity!(rng, state, flow)
        stats.support_boundary_refresh_attempts += 1

        if _short_forward_probe_is_valid!(state_probe, state, model, flow, cache, opts)
            stats.support_boundary_events += 1
            stats.refreshment_events += 1
            stats.last_rejected = false
            _invalidate_cached_gradient!(alg)
            record_event!(trace_manager, state, flow, nothing, phase)
            return :refresh
        end

        stats.support_boundary_refresh_failures += 1
    end

    copyto!(state.ξ.x, x_current)
    copyto!(state.ξ.θ, θ_current)
    state.t[] = t_current
    _throw_grid_safety_support_error(model, ctx, opts;
        message="Grid thinning reached its safety limit, and no valid refreshed direction was found.")
end

function _handle_grid_safety_limit!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    ctx::BoundaryContext,
    opts::SupportBoundaryOptions;
    phase::Symbol
)
    opts.detect_boundaries || throw(ctx.original_error)
    if opts.mode === :error
        throw(ctx.original_error)
    elseif opts.mode === :line_search_truncated_refresh
        return _line_search_truncated_refresh_from_current_state!(rng, state, model, flow, alg, cache, stats, trace_manager, ctx, opts; phase)
    elseif opts.mode === :line_search
        if _linear_boundary_flow(flow)
            _throw_grid_safety_support_error(model, ctx, opts;
                message="Grid thinning reached its safety limit while support-boundary diagnostics were enabled.")
        else
            throw(ctx.original_error)
        end
    end
    throw(ArgumentError("Unknown support-boundary mode: $(opts.mode)"))
end

_public_algorithm_type(alg::PoissonTimeStrategy) = typeof(alg)
_public_algorithm_type(::GridAdaptiveState) = GridThinningStrategy
_public_algorithm_type(::StickyLoopState) = Sticky

_linear_boundary_flow(::ContinuousDynamics) = false
_linear_boundary_flow(::BouncyParticle) = true
_linear_boundary_flow(::ZigZag) = true
_linear_boundary_flow(flow::PreconditionedDynamics) = _linear_boundary_flow(flow.dynamics)

function _rewind_linear_boundary_position!(x::AbstractVector, θ::AbstractVector, τ::Real, ::Union{BouncyParticle,ZigZag})
    @. x = x - τ * θ
    return x
end
_rewind_linear_boundary_position!(x::AbstractVector, θ::AbstractVector, τ::Real, flow::PreconditionedDynamics) =
    _rewind_linear_boundary_position!(x, θ, τ, flow.dynamics)

function _boundary_context_after_forward_move(
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    τ::Real,
    original_error::Any
)
    _linear_boundary_flow(flow) || return nothing
    x0 = copy(state.ξ.x)
    v = copy(state.ξ.θ)
    _rewind_linear_boundary_position!(x0, v, τ, flow)
    τ_float = Float64(τ)
    t0 = Float64(state.t[] - τ)
    return BoundaryContext(x0, v, t0, 0.0, max(τ_float, eps(Float64)), original_error, typeof(flow), _public_algorithm_type(alg))
end

_supports_line_search_truncated_refresh(::ContinuousDynamics) = false
_supports_line_search_truncated_refresh(::BouncyParticle) = true
_supports_line_search_truncated_refresh(flow::PreconditionedDynamics) = _supports_line_search_truncated_refresh(flow.dynamics)
_supports_line_search_truncated_refresh_state(::AbstractPDMPState, flow::ContinuousDynamics) = _supports_line_search_truncated_refresh(flow)
_supports_line_search_truncated_refresh_state(::StickyPDMPState, ::ContinuousDynamics) = false
_supports_capped_boundary_search(::PoissonTimeStrategy) = false
_supports_capped_boundary_search(::GridAdaptiveState) = true

function _check_gradient_probe_finite(gradient)
    _gradient_probe_is_finite(gradient) || throw(_NonfiniteGradientProbe())
    return gradient
end

_boundary_probe_state(::PoissonTimeStrategy, state::AbstractPDMPState) = copy(state)
_boundary_probe_state(alg::GridAdaptiveState, state::AbstractPDMPState) = (copyto!(alg.state_cache, state); alg.state_cache)

function _restore_boundary_start!(state::AbstractPDMPState, ctx::BoundaryContext)
    copyto!(state.ξ.x, ctx.x0)
    copyto!(state.ξ.θ, ctx.v)
    state.t[] = ctx.time0
    return state
end

function _move_to_boundary_safe_point!(state::AbstractPDMPState, flow::ContinuousDynamics, ctx::BoundaryContext, safe_time::Float64)
    _restore_boundary_start!(state, ctx)
    safe_time > 0.0 && move_forward_time!(state, safe_time, flow)
    return state
end

function _short_forward_probe_is_valid(
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    probe_time::Float64
)
    state_probe = copy(state)
    return _short_forward_probe_is_valid!(state_probe, state, model, flow, cache, probe_time)
end

function _short_forward_probe_is_valid!(
    state_probe::AbstractPDMPState,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    probe_time::Float64
)
    probe_time == 0.0 && return true
    copyto!(state_probe, state)
    try
        move_forward_time!(state_probe, probe_time, flow)
        _check_gradient_probe_finite(compute_gradient!(state_probe, model.grad, flow, cache))
        return true
    catch
        return false
    end
end

function _short_forward_probe_is_valid(
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    opts::SupportBoundaryOptions,
)
    state_probe = copy(state)
    return _short_forward_probe_is_valid!(state_probe, state, model, flow, cache, opts)
end

function _short_forward_probe_is_valid!(
    state_probe::AbstractPDMPState,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    opts::SupportBoundaryOptions,
)
    probe_time = opts.refresh_probe_time
    for _ in 0:opts.max_bisection_steps
        _short_forward_probe_is_valid!(state_probe, state, model, flow, cache, probe_time) && return true
        probe_time *= 0.5
        probe_time > 0.0 || break
    end
    return false
end

function _try_move_to_valid_boundary_safe_point!(
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    ctx::BoundaryContext,
    safe_time::Float64
)
    _move_to_boundary_safe_point!(state, flow, ctx, safe_time)
    try
        _check_gradient_probe_finite(compute_gradient!(state, model.grad, flow, cache))
        return true
    catch
        return false
    end
end

function _try_move_to_valid_localization_time!(
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    ctx::BoundaryContext,
    localization::SupportBoundaryLocalization,
    opts::SupportBoundaryOptions,
)
    if _try_move_to_valid_boundary_safe_point!(state, model, flow, cache, ctx, localization.safe_time)
        return true, localization.safe_time
    end

    if localization.last_valid_time != localization.safe_time &&
       _try_move_to_valid_boundary_safe_point!(state, model, flow, cache, ctx, localization.last_valid_time)
        return true, localization.last_valid_time
    end

    candidate_time = localization.safe_time
    for _ in 1:opts.max_bisection_steps
        candidate_time *= 0.5
        candidate_time > 0.0 || break
        if _try_move_to_valid_boundary_safe_point!(state, model, flow, cache, ctx, candidate_time)
            return true, candidate_time
        end
    end

    if localization.safe_time != 0.0 && _try_move_to_valid_boundary_safe_point!(state, model, flow, cache, ctx, 0.0)
        return true, 0.0
    end

    return false, 0.0
end

function _with_safe_time(localization::SupportBoundaryLocalization, safe_time::Float64)
    return SupportBoundaryLocalization(
        localization.last_valid_time,
        localization.first_invalid_time,
        localization.estimated_boundary_time,
        safe_time,
    )
end

function _truncated_refresh_localization(localization::SupportBoundaryLocalization, opts::SupportBoundaryOptions)
    refresh_clip = min(opts.clip_fraction, 0.8)
    safe_time = min(localization.safe_time, refresh_clip * localization.estimated_boundary_time)
    return _with_safe_time(localization, max(0.0, safe_time))
end

function _boundary_refresh_probe_time(localization::SupportBoundaryLocalization, opts::SupportBoundaryOptions)
    opts.refresh_probe_time == 0.0 && return 0.0
    probe_scale = 1e-3 * max(localization.safe_time, localization.estimated_boundary_time)
    return max(opts.refresh_probe_time, min(1e-4, probe_scale))
end

function _try_backtrack_valid_boundary_time!(
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    ctx::BoundaryContext,
    safe_time::Float64,
    opts::SupportBoundaryOptions,
)
    candidate_time = safe_time
    for _ in 1:opts.max_bisection_steps
        candidate_time *= 0.5
        candidate_time > 0.0 || break
        if _try_move_to_valid_boundary_safe_point!(state, model, flow, cache, ctx, candidate_time)
            return true, candidate_time
        end
    end

    if safe_time != 0.0 && _try_move_to_valid_boundary_safe_point!(state, model, flow, cache, ctx, 0.0)
        return true, 0.0
    end

    return false, safe_time
end

function _last_trace_event_before(trace::PDMPTrace, current_time::Float64)
    for k in length(trace.times):-1:1
        trace.times[k] < current_time || continue
        return PDMPEvent(trace.times[k], copy(trace.positions[:, k]), copy(trace.velocities[:, k]))
    end
    return nothing
end

function _last_trace_event_before(trace::FactorizedTrace, current_time::Float64)
    return _last_trace_event_before(PDMPTrace(trace), current_time)
end

function _last_trace_event_before(trace_manager::TraceManager, phase::Symbol, current_time::Float64)
    trace = phase === :warmup ? get_warmup_trace(trace_manager) : get_main_trace(trace_manager)
    event = _last_trace_event_before(trace, current_time)
    event !== nothing && return event

    if phase === :main
        return _last_trace_event_before(get_warmup_trace(trace_manager), current_time)
    end
    return nothing
end

function _trace_event_gradient_is_valid(
    event::PDMPEvent,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple
)
    state_probe = copy(state)
    copyto!(state_probe.ξ.x, event.position)
    copyto!(state_probe.ξ.θ, event.velocity)
    state_probe.t[] = event.time
    try
        _check_gradient_probe_finite(compute_gradient!(state_probe, model.grad, flow, cache))
        return true
    catch
        return false
    end
end

function _fallback_boundary_context_from_trace(
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    trace_manager::TraceManager,
    ctx::BoundaryContext,
    phase::Symbol
)
    current_time = Float64(ctx.time0 + ctx.t_invalid)
    event = _last_trace_event_before(trace_manager, phase, current_time)
    event === nothing && return nothing
    _trace_event_gradient_is_valid(event, state, model, flow, cache) || return nothing

    t_invalid = current_time - Float64(event.time)
    t_invalid > 0.0 || return nothing
    return BoundaryContext(
        copy(event.position), copy(event.velocity), Float64(event.time),
        0.0, t_invalid, ctx.original_error, ctx.flow_type, ctx.algorithm_type,
    )
end

function _validated_refresh_context!(
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    cache::NamedTuple,
    trace_manager::TraceManager,
    ctx::BoundaryContext,
    opts::SupportBoundaryOptions,
    phase::Symbol
)
    localization = localize_support_boundary!(model, ctx, opts)
    localization = _truncated_refresh_localization(localization, opts)
    found_valid_time, safe_time = _try_move_to_valid_localization_time!(state, model, flow, cache, ctx, localization, opts)
    if found_valid_time
        return ctx, _with_safe_time(localization, safe_time)
    end

    fallback_ctx = _fallback_boundary_context_from_trace(state, model, flow, cache, trace_manager, ctx, phase)
    if fallback_ctx !== nothing
        fallback_localization = localize_support_boundary!(model, fallback_ctx, opts)
        fallback_localization = _truncated_refresh_localization(fallback_localization, opts)
        found_valid_time, safe_time = _try_move_to_valid_localization_time!(
            state, model, flow, cache, fallback_ctx, fallback_localization, opts,
        )
        if found_valid_time
            return fallback_ctx, _with_safe_time(fallback_localization, safe_time)
        end
    end

    throw(_build_boundary_error(ctx, opts; localized=true, localization,
        message="Support boundary localized, but no valid interior refresh point was found."))
end

function _handle_capped_boundary_event!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    τ::Float64,
    event_type::Symbol,
    meta,
    phase::Symbol,
    support_boundary_options::SupportBoundaryOptions,
)
    needs_saving, saving_args = try
        handle_event!(rng, τ, model.grad, flow, alg, state, cache, event_type, meta, stats)
    catch err
        if err isa _ProbeFailureException
            return _handle_step_boundary!(rng, state, model, flow, alg, cache, stats, trace_manager, err.ctx, support_boundary_options; phase)
        elseif err isa _GridSafetyLimitException
            return _handle_grid_safety_limit!(rng, state, model, flow, alg, cache, stats, trace_manager, err.ctx, support_boundary_options; phase)
        elseif err isa SupportBoundaryError
            rethrow(err)
        else
            rethrow()
        end
    end
    needs_saving && record_event!(trace_manager, state, flow, saving_args, phase)
    return event_type
end

function _boundary_refresh_from_localization!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    ctx::BoundaryContext,
    localization::SupportBoundaryLocalization,
    opts::SupportBoundaryOptions;
    phase::Symbol
)
    safe_time = localization.safe_time
    probe_time = _boundary_refresh_probe_time(localization, opts)
    state_probe = _boundary_probe_state(alg, state)

    for _ in 1:opts.max_refresh_attempts
        _move_to_boundary_safe_point!(state, flow, ctx, safe_time)

        refresh_velocity!(rng, state, flow)
        stats.support_boundary_refresh_attempts += 1

        if _short_forward_probe_is_valid!(state_probe, state, model, flow, cache, probe_time)
            stats.support_boundary_events += 1
            stats.refreshment_events += 1
            stats.last_rejected = false
            _invalidate_cached_gradient!(alg)
            record_event!(trace_manager, state, flow, nothing, phase)
            return :refresh
        end

        stats.support_boundary_refresh_failures += 1

        found_valid_time, backtracked_time = _try_backtrack_valid_boundary_time!(
            state, model, flow, cache, ctx, safe_time, opts,
        )
        if found_valid_time
            safe_time = backtracked_time
            localization = _with_safe_time(localization, safe_time)
        end
    end

    _restore_boundary_start!(state, ctx)
    throw(_build_boundary_error(ctx, opts; localized=true, localization,
        message="Support boundary localized, but no valid refreshed direction was found."))
end

function _line_search_truncated_refresh_boundary!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model::PDMPModel,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager,
    ctx::BoundaryContext,
    opts::SupportBoundaryOptions;
    phase::Symbol
)
    if !_supports_line_search_truncated_refresh_state(state, flow)
        throw(_build_boundary_error(ctx, opts; message="Support-boundary mode :line_search_truncated_refresh is only implemented for BPS-family flows."))
    end

    ctx, localization = _validated_refresh_context!(state, model, flow, cache, trace_manager, ctx, opts, phase)
    safe_time = localization.safe_time

    if safe_time > 0.0 && _supports_capped_boundary_search(alg)
        _restore_boundary_start!(state, ctx)
        _invalidate_cached_gradient!(alg)

        τ, event_type, meta = try
            next_event_time(rng, model, flow, alg, state, cache, stats, safe_time, true, :support_boundary, true)
        catch err
            if err isa _ProbeFailureException || err isa _GridSafetyLimitException
                0.0, :support_boundary, GradientMeta(cache.∇ϕx)
            else
                rethrow()
            end
        end

        if event_type !== :support_boundary
            _restore_boundary_start!(state, ctx)
            return _handle_capped_boundary_event!(rng, state, model, flow, alg, cache, stats, trace_manager,
                Float64(τ), event_type, meta, phase, opts)
        end
    end

    return _boundary_refresh_from_localization!(rng, state, model, flow, alg, cache, stats, trace_manager, ctx, localization, opts; phase)
end
