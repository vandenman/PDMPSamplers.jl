function _prepare_tail_restart_state!(tail_state::AbstractPDMPState,
        state::AbstractPDMPState, elapsed::Real, flow::ContinuousDynamics)
    copyto!(tail_state, state)
    move_forward_time!(tail_state, elapsed, flow)
    return tail_state
end

function _next_event_time_grid!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe())::GridEvent where {P, FL<:ContinuousDynamics}

    time_offset = 0.0
    tail_restart_count = 0
    default_return = alg.empty_gradient_meta

    while true
        pcb = alg.pcb
        state_ = alg.state_cache
        copyto!(state_, state)

        λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))

        # Draw refresh time first so we can cap grid construction.
        τ_refresh = _ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
        effective_horizon, horizon_event =
            _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)

        if grad_and_hvp isa GradientOnlyProvider
            recompute_time_grid!(pcb, effective_horizon, alg.N[])
        end

        # Budget-first grid construction: draw the first exponential budget
        # before building the grid, then construct only enough bound area to
        # cover it. Rejections append budget and extend/rebuild only as needed.
        cumulative_exp = Random.randexp(rng)

        modes = _grid_bound_modes(alg, state, flow, grad_and_hvp, stats)
        had_cached_gradient = alg.has_cached_gradient[]
        if modes.use_single_pass_signed || modes.use_value_only
            alg.has_cached_gradient[] = false
            cached_gradient = had_cached_gradient ? alg.cached_gradient : nothing
            cached_y0 = cached_d0 = NaN
        elseif had_cached_gradient && !modes.use_constant_batched_signed
            cached_y0, cached_d0 = _get_rate_and_deriv_or_throw(
                probe_failure_handler, state_, flow, grad_and_hvp, false, alg.cached_gradient;
                t_valid=0.0, t_invalid=0.0)
            alg.has_cached_gradient[] = false
            cached_gradient = nothing
        else
            cached_y0, cached_d0 = NaN, NaN
            cached_gradient = nothing
            if modes.use_constant_batched_signed
                _invalidate_cached_gradient!(alg)
            end
        end
        n_cells_bounded, built_area = _build_grid_bound_prefix!(
            pcb, state, flow, grad_and_hvp, alg, stats, state_, effective_horizon,
            cumulative_exp, probe_failure_handler, modes;
            cached_gradient, cached_y0, cached_d0)
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
            τ_reflection, lb_reflection = propose_event_time(rng, pcb, cumulative_exp)

            if τ_reflection >= effective_horizon

                τ, event_type, meta = _return_grid_horizon!(
                    alg, stats, flow, alg.t_max[], effective_horizon, horizon_event, max_t_max, default_return)
                return time_offset + τ, event_type, meta
            end

            if τ_refresh < τ_reflection
                alg.has_cached_gradient[] = false
                return time_offset + τ_refresh, :refresh, default_return
            end

            # Move from original position to proposed time for acceptance test.
            copyto!(state_, state)
            move_forward_time!(state_, τ_reflection, flow)
            _inc_counter_grid_acceptance_gradient_calls(stats)
            ∇ϕx = _compute_grid_gradient_or_throw!(
                state_, state, flow, model, cache, 0.0, τ_reflection, probe_failure_handler)

            l_reflection = λ(state_, ∇ϕx, flow)
            _inc_counter_grid_acceptance_tests(stats)
            if _bound_violated(l_reflection, lb_reflection)
                _inc_counter_grid_bound_violations(stats)
                signed_reflection = flow isa BouncyParticle ? dot(∇ϕx, state_.ξ.θ) : NaN
                msg = _grid_bound_violation_message(
                    alg, stats, state_, flow, τ_reflection, signed_reflection, l_reflection,
                    lb_reflection, cumulative_exp, λ_refresh)
                if alg.bound_violation === :throw
                    throw(ErrorException(msg))
                elseif alg.bound_violation === :shrink
                    _shrink_grid_after_bound_violation!(alg, stats)
                    τ, event, meta = _next_event_time_grid!(
                        rng, grad_and_hvp, model, flow, alg, state, cache, stats,
                        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
                    return time_offset + τ, event, meta
                end
            end

            if rand(rng) * lb_reflection <= l_reflection
                tightness = l_reflection / lb_reflection
                _adapt_grid_N!(alg, tightness)
                _adapt_grid_t_max!(alg, τ_reflection, model.grad)
                _set_counter_grid_N_current(stats, alg.N[])
                copyto!(alg.cached_gradient, ∇ϕx)
                alg.has_cached_gradient[] = true
                return time_offset + τ_reflection, :reflect, GradientMeta(∇ϕx)
            end

            # Rejection: cumulative_exp has advanced, next proposal will be later.
            rejection_count += 1
            last_rejected_time = τ_reflection
            _inc_counter_grid_proposal_rejections(stats)
            cell = searchsortedlast(pcb.t_grid, τ_reflection)
            if modes.use_value_only && isfinite(pcb.d_vals[cell])
                # Only change the unvisited tail after a rejection. Discard
                # spent hazard, retain physical times and the right endpoint.
                right = pcb.t_grid[cell + 1]
                tail_bound = _value_quadratic_cell_bound(l_reflection,
                    pcb.y_vals[cell + 1], right - τ_reflection, pcb.d_vals[cell])
                fill!(view(pcb.Λ_vals, 1:(cell - 1)), 0.0)
                pcb.t_grid[cell] = τ_reflection
                pcb.y_vals[cell] = l_reflection
                pcb.Λ_vals[cell] = min(pcb.Λ_vals[cell], tail_bound)
                built_area = _piecewise_constant_area(pcb)
                cumulative_exp = Random.randexp(rng)
            else
                cumulative_exp += Random.randexp(rng)
            end
            if cumulative_exp > built_area * (1 + 64eps(Float64)) + 64eps(Float64)
                start_cell = n_cells_bounded + 1
                effective_horizon, horizon_event =
                    _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)
                modes = _grid_bound_modes(alg, state, flow, grad_and_hvp, stats)
                if modes.use_constant_batched_signed
                    alg.has_cached_gradient[] = false
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

                remaining_horizon = effective_horizon - last_rejected_time
                if !_ispositive(remaining_horizon)
                    return time_offset + last_rejected_time, horizon_event, default_return
                end

                tail_state = _prepare_tail_restart_state!(alg.state_cache2,
                    state, last_rejected_time, flow)
                time_offset += last_rejected_time
                # `state_cache` remains the proposal scratch state, so the
                # preallocated second cache can safely become the new tail
                # origin without mutating the caller or aliasing proposal work.
                state = tail_state
                if probe_failure_handler isa GridBoundaryProbeHandler
                    probe_failure_handler = _grid_probe_failure_handler(
                        state, flow, model, GridThinningStrategy)
                end
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

# Adaptive Boomerang stores the adapted Gaussian metric through the Cholesky
# factor L of Γ.  Expose its diagonal range in the same summary field used by
# the preconditioned linear flows.  This is queried only after warmup, so the
# reduction is outside the sampling loop.
function _metric_scale_extrema(flow::Union{Boomerang, MutableBoomerang})
    flow.L === nothing && return (NaN, NaN)
    diag_entries = diag(flow.L)
    return minimum(diag_entries), maximum(diag_entries)
end
