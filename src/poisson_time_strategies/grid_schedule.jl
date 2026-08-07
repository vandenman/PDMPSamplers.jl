finish_warmup!(::PoissonTimeStrategy, ::AbstractStatisticCounter) = nothing
finish_warmup!(alg::PoissonTimeStrategy, stats::AbstractStatisticCounter, ::ContinuousDynamics) =
    finish_warmup!(alg, stats)

_record_final_grid_state!(::PoissonTimeStrategy, ::AbstractStatisticCounter) = nothing

# These grid lifecycle hooks compile very large method instances because
# GridAdaptiveState carries the full provider stack. They are cold, or called
# only periodically, so avoiding specialization reduces TTFX without a measured
# runtime benefit from per-provider specialization here.
function _record_final_grid_state!(
    @nospecialize(alg::GridAdaptiveState),
    @nospecialize(stats::AbstractStatisticCounter),
)
    _set_counter_grid_final_N(stats, alg.N[])
    _set_counter_grid_final_tmax(stats, alg.t_max[])
    _set_counter_grid_final_h(stats, alg.t_max[] / alg.N[])
    _set_counter_grid_schedule_frozen(stats, alg.schedule_frozen[])
    return nothing
end

function finish_warmup!(@nospecialize(alg::GridAdaptiveState), ::AbstractStatisticCounter)
    alg.curvature_bound isa WarmupCurvatureBound &&
        _finish_warmup_curvature_bound!(alg.curvature_bound)
    return nothing
end

function _required_grid_counter_value(stats::AbstractStatisticCounter, name::Symbol)
    hasproperty(stats, name) || throw(ArgumentError(
        "GridWarmupTuning requires statistic counter field $(name)"))
    value = getproperty(stats, name)
    value isa Real || throw(ArgumentError(
        "GridWarmupTuning requires numeric statistic counter field $(name)"))
    return Float64(value)
end

function _maybe_tune_grid_after_warmup!(alg::GridAdaptiveState, stats::AbstractStatisticCounter, flow::ContinuousDynamics)
    tuning = alg.warmup_tuning
    tuning === nothing && return nothing
    events = max(_required_grid_counter_value(stats, :warmup_events), 1.0)
    endpoint_grad = _required_grid_counter_value(stats, :warmup_grid_endpoint_gradient_calls)
    acceptance_grad = _required_grid_counter_value(stats, :warmup_grid_acceptance_gradient_calls)
    gradients_per_event = (endpoint_grad + acceptance_grad) / events
    horizon_hits = _required_grid_counter_value(stats, :grid_horizon_hits)
    rejections = _required_grid_counter_value(stats, :lazy_proposal_rejections)
    horizon_rate = horizon_hits / events
    rejection_rate = rejections / events

    _set_counter_grid_warmup_objective_events(stats, events)
    _set_counter_grid_warmup_objective_endpoint_gradients(stats, endpoint_grad)
    _set_counter_grid_warmup_objective_acceptance_gradients(stats, acceptance_grad)
    _set_counter_grid_warmup_objective_gradients_per_event(stats, gradients_per_event)
    _set_counter_grid_warmup_objective_horizon_hits(stats, horizon_hits)
    _set_counter_grid_warmup_objective_horizon_rate(stats, horizon_rate)
    _set_counter_grid_warmup_objective_rejections(stats, rejections)
    _set_counter_grid_warmup_objective_rejection_rate(stats, rejection_rate)

    changed = false
    if gradients_per_event > tuning.target_gradients_per_event && alg.N[] > alg.N_min
        alg.N[] = max(alg.N_min, alg.N[] - tuning.n_step)
        changed = true
    elseif horizon_rate > tuning.horizon_rate_high && alg.N[] < alg.N_max
        alg.N[] = min(alg.N_max, alg.N[] + tuning.n_step)
        changed = true
    end

    old_tmax = alg.t_max[]
    max_tmax = min(tuning.max_tmax, max_grid_horizon(flow))
    if horizon_rate > tuning.horizon_rate_high
        alg.t_max[] = min(max_tmax, old_tmax * tuning.tmax_grow)
    elseif rejection_rate > tuning.rejection_rate_high
        alg.t_max[] = max(tuning.min_tmax, old_tmax * tuning.tmax_shrink)
    end
    changed |= alg.t_max[] != old_tmax

    if changed
        recompute_time_grid!(alg)
        _set_counter_grid_N_current(stats, alg.N[])
    end
    alg.schedule_frozen[] = true
    _set_counter_grid_final_N(stats, alg.N[])
    _set_counter_grid_final_tmax(stats, alg.t_max[])
    _set_counter_grid_final_h(stats, alg.t_max[] / alg.N[])
    _set_counter_grid_schedule_frozen(stats, true)
    return nothing
end

function finish_warmup!(
    @nospecialize(alg::GridAdaptiveState),
    @nospecialize(stats::AbstractStatisticCounter),
    @nospecialize(flow::ContinuousDynamics),
)
    finish_warmup!(alg, stats)
    _maybe_tune_grid_after_warmup!(alg, stats, flow)
    return nothing
end

function reset_grid_scale!(alg::GridAdaptiveState, t_max::Float64=2.0)
    alg.t_max[] = t_max
    alg.N[] = alg.N_max
    alg.lazy_enabled[] = alg.lazy
    alg.has_cached_gradient[] = false
    alg.has_cached_rate_derivative[] = false
    recompute_time_grid!(alg)
end

function _shrink_grid_after_bound_violation!(alg::GridAdaptiveState, stats::AbstractStatisticCounter)
    new_t_max = max(alg.t_max[] * alg.α⁻, sqrt(eps(Float64)))
    reset_grid_scale!(alg, new_t_max)
    _inc_counter_grid_shrinks(stats)
    return nothing
end

function _invalidate_cached_gradient!(alg::GridAdaptiveState)
    alg.has_cached_gradient[] = false
    alg.has_cached_rate_derivative[] = false
    return nothing
end

function _cache_endpoint_rate_derivative!(alg::GridAdaptiveState, y::Real, d::Real, horizon_event::Symbol)
    if horizon_event === :horizon_hit && isfinite(y) && isfinite(d)
        alg.cached_rate[] = Float64(y)
        alg.cached_rate_derivative[] = Float64(d)
        alg.has_cached_rate_derivative[] = true
    else
        alg.has_cached_rate_derivative[] = false
    end
    return nothing
end

function _adapt_grid_N!(alg::GridAdaptiveState, tightness::Float64)
    alg.schedule_frozen[] && return nothing
    N = alg.N[]
    if tightness > 0.5 && N > alg.N_min
        # Bounds are tight enough, try fewer grid cells
        new_N = max(alg.N_min, N - 2)
        if new_N != N
            alg.N[] = new_N
            recompute_time_grid!(alg)
        end
    elseif tightness < 0.1 && N < alg.N_max
        _increase_grid_N!(alg)
    end
end

function _increase_grid_N!(alg::GridAdaptiveState)
    alg.schedule_frozen[] && return nothing
    N = alg.N[]
    new_N = min(alg.N_max, N + 4)
    if new_N != N
        alg.N[] = new_N
        recompute_time_grid!(alg)
    end
end

function _adapt_grid_t_max!(alg::GridAdaptiveState, τ_accepted::Float64, ::GradientStrategy)
    alg.schedule_frozen[] && return nothing
    t_max = alg.t_max[]
    if τ_accepted < 0.25 * t_max
        new_t_max = max(4.0 * τ_accepted, 0.1)
        if new_t_max < t_max
            alg.t_max[] = new_t_max
            recompute_time_grid!(alg)
        end
    end
end
function _shrink_t_max_on_rejection!(alg::GridAdaptiveState, pcb::PiecewiseConstantBound, cumulative_exp::Float64, ::GradientStrategy)
    alg.schedule_frozen[] && return nothing
    total_integral = _piecewise_constant_area(pcb, alg.N[])
    if total_integral > 0 && cumulative_exp < 0.1 * total_integral
        alg.t_max[] = max(alg.t_max[] * alg.α⁻, 0.1)
        recompute_time_grid!(alg)
    end
end
_reset_inner_grid!(alg::GridAdaptiveState) = reset_grid_scale!(alg)

function _maybe_activate_constant_bound!(
    @nospecialize(alg::GridAdaptiveState),
    @nospecialize(stats::AbstractStatisticCounter),
)
    alg.post_warmup_simplify || return nothing
    isfinite(alg.constant_bound_rate[]) && return nothing
    total_events = _get_counter_reflections_accepted(stats) + _get_counter_refreshment_events(stats)
    total_events < 10 && return nothing
    reflection_ratio = _get_counter_reflections_accepted(stats) / total_events
    reflection_ratio > 0.3 && return nothing
    max_rate = alg.max_observed_rate[]
    !ispositive(max_rate) && return nothing
    alg.constant_bound_rate[] = max_rate * 2.0
    return nothing
end
