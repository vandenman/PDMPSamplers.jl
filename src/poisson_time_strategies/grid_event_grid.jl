function _next_event_time_grid!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe())::GridEvent where {P, FL<:ContinuousDynamics}

    time_offset = 0.0
    tail_restart_count = 0
    default_return = GradientMeta(alg.empty_∇ϕx)

    while true
        pcb = alg.pcb
        state_ = alg.state_cache
        copyto!(state_, state)

        λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))

        # Draw refresh time first so we can cap grid construction.
        τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
        effective_horizon, horizon_event =
            _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)

        # Budget-first grid construction: draw the first exponential budget
        # before building the grid, then construct only enough bound area to
        # cover it. Rejections append budget and extend/rebuild only as needed.
        cumulative_exp = Random.randexp(rng)

        modes = _grid_bound_modes(alg, state, flow, grad_and_hvp)
        had_cached_gradient = alg.has_cached_gradient[]
        alg.has_cached_rate_derivative[] = false
        if modes.use_single_pass_signed
            alg.has_cached_gradient[] = false
            cached_gradient = had_cached_gradient ? alg.cached_gradient : nothing
            cached_y0 = cached_d0 = cached_g0 = cached_dg0 = NaN
        elseif had_cached_gradient && !modes.use_constant_batched_signed
            cached_y0, cached_d0 = _get_rate_and_deriv_or_throw(
                probe_failure_handler, state_, flow, grad_and_hvp, false, alg.cached_gradient;
                t_valid=0.0, t_invalid=0.0)
            cached_g0, cached_dg0 = if alg.bound === :linear
                rate_and_derivative(state_, flow, grad_and_hvp, alg.cached_gradient)
            else
                (NaN, NaN)
            end
            alg.has_cached_gradient[] = false
            alg.has_cached_rate_derivative[] = false
            cached_gradient = nothing
        else
            cached_y0, cached_d0 = NaN, NaN
            cached_g0, cached_dg0 = NaN, NaN
            cached_gradient = nothing
            if modes.use_constant_batched_signed
                _invalidate_cached_gradient!(alg)
            end
        end
        n_cells_bounded, built_area = _build_grid_bound_prefix!(
            pcb, state, flow, grad_and_hvp, alg, stats, state_, effective_horizon,
            cumulative_exp, probe_failure_handler, modes;
            cached_gradient, cached_y0, cached_d0, cached_g0, cached_dg0)
        _set_counter_grid_N_current(stats, alg.N[])

        # Budget-first exactness invariant:
        #   * once proposal times have been generated from the built dominating
        #     bound prefix, that prefix must never be changed;
        #   * after a rejection, a larger exponential budget may only append
        #     later cells/segments to the existing prefix;
        #   * if many rejections force a safety rebuild, restart only after the
        #     last rejected time, then offset returned times back to this call.
        rejection_count = 0
        max_rejections = alg.max_rejections_before_tail_restart
        last_rejected_time = 0.0
        max_t_max = max_grid_horizon(flow)
        tail_restart = false

        safety_limit = alg.safety_limit
        while safety_limit > 0
            τ_reflection, lb_reflection = if modes.use_linear
                propose_event_time(rng, alg.affine_bound, cumulative_exp)
            else
                propose_event_time(rng, pcb, cumulative_exp)
            end

            if τ_reflection >= effective_horizon

                τ, event_type, meta = _return_grid_horizon!(
                    alg, stats, flow, alg.t_max[], effective_horizon, horizon_event, max_t_max, default_return)
                return time_offset + τ, event_type, meta
            end

            if τ_refresh < τ_reflection
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _restore_lazy_after_grid!(alg, flow)
                return time_offset + τ_refresh, :refresh, default_return
            end

            # Move from original position to proposed time for acceptance test.
            copyto!(state_, state)
            move_forward_time!(state_, τ_reflection, flow)
            _inc_counter_grid_acceptance_gradient_calls(stats)
            ∇ϕx = _compute_grid_gradient_or_throw!(
                state_, state, flow, model, cache, 0.0, τ_reflection, probe_failure_handler)

            l_reflection = λ(state_.ξ, ∇ϕx, flow)
            _inc_counter_grid_acceptance_tests(stats)
            if l_reflection > lb_reflection * (1 + 1e-10) + 1e-12
                _inc_counter_grid_bound_violations(stats)
                signed_reflection = flow isa BouncyParticle ? dot(∇ϕx, state_.ξ.θ) : NaN
                msg = _grid_bound_violation_message(
                    alg, stats, state_, flow, τ_reflection, signed_reflection, l_reflection,
                    lb_reflection, cumulative_exp, λ_refresh, modes.use_linear)
                if modes.use_linear
                    _inc_counter_affine_bound_violations(stats)
                end
                if alg.bound_violation === :throw
                    throw(ErrorException(msg))
                elseif alg.bound_violation === :shrink
                    _shrink_grid_after_bound_violation!(alg, stats)
                    return _next_event_time_grid!(
                        rng, grad_and_hvp, model, flow, alg, state, cache, stats,
                        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
                end
            end

            if rand(rng) * lb_reflection <= l_reflection
                tightness = l_reflection / lb_reflection
                _adapt_grid_N!(alg, tightness)
                _adapt_grid_t_max!(alg, τ_reflection, model.grad)
                _set_counter_grid_N_current(stats, alg.N[])
                alg.max_observed_rate[] = max(alg.max_observed_rate[], l_reflection)
                copyto!(alg.cached_gradient, ∇ϕx)
                alg.has_cached_gradient[] = true
                alg.has_cached_rate_derivative[] = false
                _restore_lazy_after_grid!(alg, flow)
                return time_offset + τ_reflection, :reflect, GradientMeta(∇ϕx)
            end

            # Rejection: cumulative_exp has advanced, next proposal will be later.
            rejection_count += 1
            last_rejected_time = τ_reflection
            cumulative_exp += Random.randexp(rng)
            if cumulative_exp > built_area * (1 + 64eps(Float64)) + 64eps(Float64)
                start_cell = n_cells_bounded + 1
                effective_horizon, horizon_event =
                    _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)
                modes = _grid_bound_modes(alg, state, flow, grad_and_hvp)
                if modes.use_constant_batched_signed
                    alg.has_cached_gradient[] = false
                    alg.has_cached_rate_derivative[] = false
                end
                n_cells_bounded, built_area = _build_grid_bound_prefix!(
                    pcb, state, flow, grad_and_hvp, alg, stats, state_, effective_horizon,
                    cumulative_exp, probe_failure_handler, modes;
                    start_cell, initial_integral=built_area, append=true)
            end
            if rejection_count >= max_rejections && !alg.schedule_frozen[]
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                _shrink_t_max_on_rejection!(alg, pcb, cumulative_exp, model.grad)

                _inc_counter_grid_shrinks(stats)
                _inc_counter_grid_budget_tail_restarts(stats)
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false

                remaining_horizon = effective_horizon - last_rejected_time
                if !ispositive(remaining_horizon)
                    return time_offset + last_rejected_time, horizon_event, default_return
                end

                tail_state = alg.state_cache2
                copyto!(tail_state, state)
                move_forward_time!(tail_state, last_rejected_time, flow)
                time_offset += last_rejected_time
                state = copy(tail_state)
                max_horizon = remaining_horizon
                include_refresh = false
                max_horizon_event = horizon_event
                tail_restart_count += 1
                if tail_restart_count > alg.safety_limit
                    _throw_grid_safety_limit_error(state, flow, model;
                        t_invalid=remaining_horizon,
                        message="Tail restart limit reached in Grid algorithm")
                end
                tail_restart = true
                break
            end
            safety_limit -= 1
        end

        tail_restart && continue

        if isfinite(τ_refresh) && τ_refresh <= min(alg.t_max[], max_horizon)
            alg.has_cached_gradient[] = false
            alg.has_cached_rate_derivative[] = false
            _restore_lazy_after_grid!(alg, flow)
            return time_offset + τ_refresh, :refresh, default_return
        end

        _throw_grid_safety_limit_error(state, flow, model;
            t_invalid=effective_horizon, message="Safety limit reached")
    end
end

_metric_scale_extrema(::ContinuousDynamics) = (NaN, NaN)

function _metric_scale_extrema(flow::PreconditionedDynamics{<:DiagonalPreconditioner})
    scales = flow.metric.scale
    return minimum(scales), maximum(scales)
end

function _metric_scale_extrema(flow::DensePreconditionedBPS)
    diag_entries = diag(flow.metric.L)
    return minimum(diag_entries), maximum(diag_entries)
end

function _metric_scale_extrema(flow::DensePreconditionedZigZag)
    diag_entries = diag(flow.metric.L)
    return minimum(diag_entries), maximum(diag_entries)
end

_safe_tightness(l_actual::Real, Λ_cell::Real) = ispositive(Λ_cell) ? l_actual / pos(Λ_cell) : NaN

function _record_lazy_search_stats!(stats::AbstractStatisticCounter, proposal_attempts::Int, proposal_rejections::Int)
    _inc_counter_lazy_proposal_attempts(stats, proposal_attempts)
    _inc_counter_lazy_proposal_rejections(stats, proposal_rejections)
    return nothing
end

function _lazy_grid_failure_message(; alg::GridAdaptiveState, flow::ContinuousDynamics,
    state_time::Float64, effective_horizon::Float64, Δt::Float64, no_event_cells::Int, proposal_attempts::Int,
    proposal_rejections::Int, tightness_sum::Float64, min_tightness::Float64, max_tightness::Float64,
    last_t_left::Float64, last_t_right::Float64, last_y_left::Float64, last_y_right::Float64,
    last_d_left::Float64, last_d_right::Float64, last_Λ_cell::Float64, last_exp_target::Float64,
    last_cumulative_area::Float64, last_τ_proposal::Float64, last_l_actual::Float64)

    mean_tightness = proposal_attempts == 0 ? NaN : tightness_sum / proposal_attempts
    last_tightness = _safe_tightness(last_l_actual, last_Λ_cell)
    scale_min, scale_max = _metric_scale_extrema(flow)
    return string(
        "Safety limit reached in lazy grid",
        " | N=", alg.N[],
        " t_max=", alg.t_max[],
        " state_time=", state_time,
        " effective_horizon=", effective_horizon,
        " Δt=", Δt,
        " | no_event_cells=", no_event_cells,
        " proposal_attempts=", proposal_attempts,
        " proposal_rejections=", proposal_rejections,
        " | last_t=[", last_t_left, ", ", last_t_right, "]",
        " τ=", last_τ_proposal,
        " | last_rate=[", last_y_left, ", ", last_y_right, "]",
        " last_deriv=[", last_d_left, ", ", last_d_right, "]",
        " | last_bound=", last_Λ_cell,
        " last_actual=", last_l_actual,
        " last_tightness=", last_tightness,
        " | tightness[min/mean/max]=[", min_tightness, ", ", mean_tightness, ", ", max_tightness, "]",
        " | last_exp_target=", last_exp_target,
        " last_cumulative_area=", last_cumulative_area,
        " | metric_scale[min,max]=[", scale_min, ", ", scale_max, "]"
    )
end

# ── Lazy grid evaluation (Phase 2) ──────────────────────────────────────────
# Interleaves grid point evaluation with proposal generation so that for
# well-adapted samplers only the first few intervals are evaluated.

function _next_event_time_lazy!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    )::GridEvent where {P, FL<:ContinuousDynamics}

    state_ = alg.state_cache
    state2_ = alg.state_cache2
    copyto!(state_, state)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    default_return = GradientMeta(alg.empty_∇ϕx)

    τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf

    N = alg.N[]
    t_max = alg.t_max[]
    effective_horizon, horizon_event = _effective_grid_horizon(model.grad, t_max, τ_refresh, max_horizon, max_horizon_event)
    Δt = effective_horizon / N
    max_t_max = max_grid_horizon(flow)

    # Evaluate initial grid point (k=0)
    if alg.has_cached_gradient[]
        # Reuse cached gradient from previous event (2C): skip one gradient call.
        _inc_counter_grid_cached_endpoint_reuses(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_left, d_left = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false, alg.cached_gradient;
            t_valid=0.0, t_invalid=0.0)
        alg.has_cached_gradient[] = false
        alg.has_cached_rate_derivative[] = false
    elseif alg.has_cached_rate_derivative[]
        _inc_counter_grid_cached_endpoint_reuses(stats)
        y_left = alg.cached_rate[]
        d_left = alg.cached_rate_derivative[]
        alg.has_cached_rate_derivative[] = false
    else
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_left, d_left = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false;
            t_valid=0.0, t_invalid=0.0)
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
    no_event_cells = 0
    proposal_attempts = 0
    proposal_rejections = 0
    tightness_sum = 0.0
    min_tightness = Inf
    max_tightness = -Inf
    last_t_left = NaN
    last_t_right = NaN
    last_y_left = NaN
    last_y_right = NaN
    last_d_left = NaN
    last_d_right = NaN
    last_Λ_cell = NaN
    last_τ_proposal = NaN
    last_l_actual = NaN
    last_exp_target = NaN
    last_cumulative_area = NaN

    _inc_counter_grid_builds(stats)
    _inc_counter_grid_points_evaluated(stats, 1)
    _record_grid_schedule!(stats, alg)

    while safety_limit > 0
        safety_limit -= 1

        # Advance state to next grid point
        t_right = t_left + Δt
        if t_right > effective_horizon
            t_right = effective_horizon
        end
        Δt_cell = t_right - t_left

        if !ispositive(Δt_cell)
            # Reached the horizon
            return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event, max_t_max, default_return)
        end

        # Move state_cache forward by Δt_cell to evaluate the right endpoint
        move_forward_time!(state_, Δt_cell, flow)
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_right, d_right = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false;
            t_valid=t_left, t_invalid=t_right)
        _inc_counter_grid_points_evaluated(stats, 1)

        # Compute piecewise constant bound for this interval
        Λ_cell = _tangent_intersection_bound(t_left, t_right, y_left, y_right, d_left, d_right)
        area_cell = pos(Λ_cell) * Δt_cell
        last_t_left = t_left
        last_t_right = t_right
        last_y_left = y_left
        last_y_right = y_right
        last_d_left = d_left
        last_d_right = d_right
        last_Λ_cell = Λ_cell
        last_exp_target = exp_target
        last_cumulative_area = cumulative_area

        if cumulative_area + area_cell < exp_target
            # No event in this interval — advance
            no_event_cells += 1
            cumulative_area += area_cell
            t_left = t_right
            y_left = y_right
            d_left = d_right

            # Check if we've exhausted the horizon
            if t_right >= effective_horizon
                _cache_endpoint_rate_derivative!(alg, y_right, d_right, horizon_event)
                return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event, max_t_max, default_return)
            end
            continue
        end

        # Event is in this interval.  If a proposal is rejected, keep using the
        # same constant cell bound for the remainder of the cell instead of
        # rebuilding a new tangent bound at the rejected time.
        while true
            lb_proposal = pos(Λ_cell)
            τ_proposal = t_left + (exp_target - cumulative_area) / lb_proposal
            if τ_proposal >= t_right || !isfinite(τ_proposal)
                no_event_cells += 1
                t_left = t_right
                y_left = y_right
                d_left = d_right
                cumulative_area = 0.0
                exp_target = Random.randexp(rng)
                if t_right >= effective_horizon
                    _cache_endpoint_rate_derivative!(alg, y_right, d_right, horizon_event)
                    return _return_grid_horizon!(
                        alg, stats, flow, t_max, effective_horizon, horizon_event,
                        max_t_max, default_return)
                end
                break
            end

            if τ_refresh < τ_proposal
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                _invalidate_cached_gradient!(alg)
                return τ_refresh, :refresh, default_return
            end

            state2_.t[] = state.t[]
            copyto!(state2_.ξ, state.ξ)
            move_forward_time!(state2_, τ_proposal, flow)
            _inc_counter_grid_acceptance_gradient_calls(stats)
            ∇ϕx = _compute_grid_gradient_or_throw!(
                state2_, state, flow, model, cache, t_left, τ_proposal, probe_failure_handler)

            l_actual = λ(state2_.ξ, ∇ϕx, flow)
            _inc_counter_grid_acceptance_tests(stats)
            proposal_attempts += 1
            last_τ_proposal = τ_proposal
            last_l_actual = l_actual

            tightness = _safe_tightness(l_actual, lb_proposal)
            tightness_sum += tightness
            min_tightness = min(min_tightness, tightness)
            max_tightness = max(max_tightness, tightness)

            if l_actual > lb_proposal
                _inc_counter_lazy_fallback_bound_violation(stats)
                _inc_counter_grid_bound_violations(stats)
                if alg.bound_violation === :throw
                    throw(ErrorException("lazy constant GridThinning bound violated at proposal time"))
                elseif alg.bound_violation === :shrink
                    _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                    alg.lazy_enabled[] = false
                    alg.has_cached_gradient[] = false
                    alg.has_cached_rate_derivative[] = false
                    _shrink_grid_after_bound_violation!(alg, stats)
                    return _next_event_time_grid!(
                        rng, grad_and_hvp, model, flow, alg, state, cache, stats,
                        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
                end
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.lazy_enabled[] = false
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
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

            if low_tightness_rejections >= max_low_tightness_rejections
                _inc_counter_lazy_fallback_low_tightness(stats)
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.lazy_enabled[] = false
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            if proposal_rejections >= max_rejections
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.lazy_enabled[] = false
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            exp_target += Random.randexp(rng)
        end
    end

    if isfinite(τ_refresh) && τ_refresh <= min(t_max, max_horizon)
        return τ_refresh, :refresh, default_return
    end

    message = _lazy_grid_failure_message(; alg, flow, state_time=state.t[], effective_horizon, Δt, no_event_cells,
        proposal_attempts, proposal_rejections, tightness_sum, min_tightness, max_tightness,
        last_t_left, last_t_right, last_y_left, last_y_right, last_d_left, last_d_right,
        last_Λ_cell, last_exp_target, last_cumulative_area, last_τ_proposal, last_l_actual)
    _throw_grid_safety_limit_error(state, flow, model; t_invalid=effective_horizon, message)
end
