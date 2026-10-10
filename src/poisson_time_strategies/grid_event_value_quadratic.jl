_can_use_value_quadratic_grid(flow::ContinuousDynamics, curvature_bound) =
    _rate_aggregation(flow) === :scalar && curvature_bound !== nothing

# The full-gradient value-quadratic search below needs only the total rate at
# the grid points and the acceptance test at the candidate, so it also applies
# to componentwise (Zig-Zag) rates: the bound then concerns the total rate
# sum_i (theta_i d_i U)_+, as for the scalar rates of the other dynamics, and
# the grid points need gradients only (no rate derivatives).
_can_use_global_value_quadratic_grid(flow::ContinuousDynamics, curvature_bound) =
    _rate_aggregation(flow) in (:scalar, :componentwise) && curvature_bound !== nothing

@inline function _value_quadratic_cell_bound(y_left::Real, y_right::Real, cell_width::Real, residual_bound::Real)
    residual = max(Float64(residual_bound), 0.0) * Float64(cell_width)^2 / 8
    return max(Float64(y_left), Float64(y_right), 0.0) + residual
end

function _value_rate_at_state!(
    probe_failure_handler::GridBoundaryProbe,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    grad_provider,
    t_valid::Float64,
    t_invalid::Float64,
)
    rate, _ = _get_rate_and_deriv_or_throw(
        probe_failure_handler, state, flow, GradientOnlyProvider(grad_provider), false;
        t_valid, t_invalid)
    return rate
end


_maybe_probe_warmup_curvature_bound!(args...) = nothing

function _maybe_probe_warmup_curvature_bound!(
    bound::WarmupCurvatureBound,
    probe_failure_handler::GridBoundaryProbe,
    state_probe::AbstractPDMPState,
    state_start::AbstractPDMPState,
    flow::ContinuousDynamics,
    grad_provider,
    t_left::Float64,
    t_right::Float64,
    y_left::Float64,
    y_right::Float64,
    stats::AbstractStatisticCounter,
)
    bound.active || return nothing
    bound.cells_seen += 1
    bound.observations < bound.max_observations || return nothing
    rem(bound.cells_seen, bound.probe_stride) == 0 || return nothing
    h = t_right - t_left
    ispositive(h) || return nothing
    copyto!(state_probe, state_start)
    t_probe = t_left + bound.probe_fraction * h
    !iszero(t_probe) && move_forward_time!(state_probe, t_probe, flow)
    _inc_counter_grid_certificate_calls(stats)
    _inc_counter_grid_points_evaluated(stats, 1)
    y_probe = _value_rate_at_state!(
        probe_failure_handler, state_probe, flow, grad_provider, t_left, t_right)
    required = 8 * max(0.0, y_probe - max(y_left, y_right, 0.0)) / (h * h)
    _update_warmup_curvature_bound!(bound, required)
    return nothing
end

function _next_event_time_value_quadratic!(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    )::GridEvent where {FL<:ContinuousDynamics}

    _can_use_global_value_quadratic_grid(flow, alg.curvature_bound) ||
        return _next_event_time_lazy!(rng, _grid_event_provider(model, flow, alg, stats), model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)

    state_ = alg.state_cache
    state2_ = alg.state_cache2
    copyto!(state_, state)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    default_return = alg.empty_gradient_meta

    τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf

    N = alg.N[]
    t_max = alg.t_max[]
    effective_horizon, horizon_event = _effective_grid_horizon(model.grad, t_max, τ_refresh, max_horizon, max_horizon_event)
    Δt = effective_horizon / N
    max_t_max = max_grid_horizon(flow)

    if alg.has_cached_gradient[]
        _inc_counter_grid_cached_endpoint_reuses(stats)
        y_left = pos(λ(state_, alg.cached_gradient, flow))
        alg.has_cached_gradient[] = false
        alg.has_cached_rate_derivative[] = false
    else
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        y_left = _value_rate_at_state!(probe_failure_handler, state_, flow, alg.grad_provider, 0.0, 0.0)
    end
    t_left = 0.0

    cumulative_area = 0.0
    exp_target = Random.randexp(rng)
    safety_limit = alg.safety_limit
    default_max_rejections = min(25, max(10, alg.safety_limit ÷ 2))
    max_rejections = alg.lazy_max_rejections > 0 ? alg.lazy_max_rejections : default_max_rejections
    low_tightness_rejections = 0
    low_tightness_threshold = alg.lazy_low_tightness_threshold
    max_low_tightness_rejections = alg.lazy_max_low_tightness_rejections
    proposal_attempts = 0
    proposal_rejections = 0
    tightness_sum = 0.0
    min_tightness = Inf
    max_tightness = -Inf

    _inc_counter_grid_builds(stats)
    _inc_counter_grid_points_evaluated(stats, 1)
    _record_grid_schedule!(stats, alg)

    while safety_limit > 0
        safety_limit -= 1

        t_right = t_left + Δt
        t_right > effective_horizon && (t_right = effective_horizon)
        Δt_cell = t_right - t_left
        if !ispositive(Δt_cell)
            return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event, max_t_max, default_return)
        end

        move_forward_time!(state_, Δt_cell, flow)
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        y_right = _value_rate_at_state!(probe_failure_handler, state_, flow, alg.grad_provider, t_left, t_right)
        _inc_counter_grid_points_evaluated(stats, 1)

        residual_bound = _evaluate_curvature_bound(alg.curvature_bound, state, flow, t_left, t_right, stats)
        residual_bound === nothing && return _next_event_time_lazy!(rng, _grid_event_provider(model, flow, alg, stats), model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
        _maybe_probe_warmup_curvature_bound!(
            alg.curvature_bound, probe_failure_handler, state2_, state, flow,
            alg.grad_provider, t_left, t_right, Float64(y_left), Float64(y_right), stats)

        Λ_cell = _value_quadratic_cell_bound(y_left, y_right, Δt_cell, residual_bound)
        area_cell = pos(Λ_cell) * Δt_cell

        if !ispositive(area_cell)
            t_left = t_right
            y_left = y_right
            if t_right >= effective_horizon
                alg.has_cached_rate_derivative[] = false
                return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                    max_t_max, default_return)
            end
            continue
        end

        if cumulative_area + area_cell < exp_target
            cumulative_area += area_cell
            t_left = t_right
            y_left = y_right
            if t_right >= effective_horizon
                alg.has_cached_rate_derivative[] = false
                return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                    max_t_max, default_return)
            end
            continue
        end

        while true
            lb_proposal = pos(Λ_cell)
            τ_proposal = t_left + (exp_target - cumulative_area) / lb_proposal
            if τ_proposal >= t_right || !isfinite(τ_proposal)
                t_left = t_right
                y_left = y_right
                cumulative_area = 0.0
                exp_target = Random.randexp(rng)
                if t_right >= effective_horizon
                    alg.has_cached_rate_derivative[] = false
                    return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                        max_t_max, default_return)
                end
                break
            end

            if τ_refresh < τ_proposal
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                _invalidate_cached_gradient!(alg)
                return τ_refresh, :refresh, default_return
            end

            copyto!(state2_, state)
            move_forward_time!(state2_, τ_proposal, flow)
            _inc_counter_grid_acceptance_gradient_calls(stats)
            ∇ϕx = _compute_grid_gradient_or_throw!(
                state2_, state, flow, model, cache, t_left, τ_proposal, probe_failure_handler)

            l_actual = λ(state2_, ∇ϕx, flow)
            signed_actual = Float64(l_actual)
            _inc_counter_grid_acceptance_tests(stats)
            proposal_attempts += 1

            tightness = _safe_tightness(l_actual, lb_proposal)
            tightness_sum += tightness
            min_tightness = min(min_tightness, tightness)
            max_tightness = max(max_tightness, tightness)

            if _bound_violated(l_actual, lb_proposal)
                _inc_counter_lazy_fallback_bound_violation(stats)
                _inc_counter_grid_bound_violations(stats)
                if alg.bound_violation === :throw
                    throw(ErrorException(_grid_bound_violation_message(
                        alg, stats, state2_, flow, τ_proposal, NaN, l_actual,
                        lb_proposal, exp_target, λ_refresh)))
                elseif alg.bound_violation === :shrink
                    _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                    alg.has_cached_gradient[] = false
                    alg.has_cached_rate_derivative[] = false
                    _shrink_grid_after_bound_violation!(alg, stats)
                    return _next_event_time_lazy!(rng, _grid_event_provider(model, flow, alg, stats), model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
                end
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                return _next_event_time_lazy!(rng, _grid_event_provider(model, flow, alg, stats), model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            if rand(rng) * lb_proposal <= l_actual
                copyto!(alg.cached_gradient, ∇ϕx)
                alg.has_cached_gradient[] = true
                alg.has_cached_rate_derivative[] = false
                _adapt_grid_N!(alg, tightness)
                _adapt_grid_t_max!(alg, τ_proposal, model.grad)
                _set_counter_grid_N_current(stats, alg.N[])
                alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return τ_proposal, :reflect, GradientMeta(∇ϕx)
            end

            proposal_rejections += 1
            if tightness < low_tightness_threshold
                low_tightness_rejections += 1
            else
                low_tightness_rejections = 0
            end

            if low_tightness_rejections >= max_low_tightness_rejections || proposal_rejections >= max_rejections
                low_tightness_rejections >= max_low_tightness_rejections &&
                    _inc_counter_lazy_fallback_low_tightness(stats)
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_lazy!(rng, _grid_event_provider(model, flow, alg, stats), model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            cumulative_area += lb_proposal * (τ_proposal - t_left)
            t_left = τ_proposal
            y_left = signed_actual
            Δt_tail = t_right - t_left
            if !ispositive(Δt_tail)
                t_left = t_right
                y_left = y_right
                cumulative_area = 0.0
                exp_target = Random.randexp(rng)
                if t_right >= effective_horizon
                    alg.has_cached_rate_derivative[] = false
                    return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                        max_t_max, default_return)
                end
                break
            end
            Λ_tail = _value_quadratic_cell_bound(y_left, y_right, Δt_tail, residual_bound)
            isfinite(Λ_tail) && (Λ_cell = min(Λ_cell, Λ_tail))
            exp_target += Random.randexp(rng)
        end
    end

    if isfinite(τ_refresh) && τ_refresh <= min(t_max, max_horizon)
        alg.has_cached_gradient[] = false
        alg.has_cached_rate_derivative[] = false
        return τ_refresh, :refresh, default_return
    end

    _throw_grid_safety_limit_error(state, flow, model; t_invalid=effective_horizon,
        message="Safety limit reached in value-quadratic Grid algorithm")
end
