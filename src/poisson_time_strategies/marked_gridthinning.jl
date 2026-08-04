# Exact marked-control-variate GridThinning for ordinary linear trajectories.

function _draw_uniform_without_replacement!(rng::Random.AbstractRNG, out::AbstractVector,
    N::Int, scratch::Dict{Int,Int}; excluded::Int=0)
    empty!(scratch)
    population = iszero(excluded) ? N : N - 1
    length(out) <= population || throw(ArgumentError("sample exceeds available population"))
    for j in eachindex(out)
        remaining = population - j + 1
        slot = rand(rng, 1:remaining)
        logical_index = get(scratch, slot, slot)
        scratch[slot] = get(scratch, remaining, remaining)
        out[j] = iszero(excluded) || logical_index < excluded ? logical_index : logical_index + 1
    end
    return out
end

function _draw_component(rng::Random.AbstractRNG, envelope::SeparableResidualEnvelope, B::Real)
    u = rand(rng) * B
    cumulative = 0.0
    chosen = 0
    for r in eachindex(envelope.totals)
        cumulative += envelope.scales[r] * envelope.totals[r]
        if u < cumulative
            chosen = r
            break
        end
    end
    if iszero(chosen)
        chosen = findlast(r -> envelope.scales[r] * envelope.totals[r] > 0,
            eachindex(envelope.totals))
    end
    return chosen
end

"""
    draw_subset!(rng, cv, state, t, D, B) -> M_S

Draw a fresh size-biased marked subset into `cv.subset` and return its
subset-specific envelope. `B` must be the total residual bound returned by
`total_residual_bound` at the same `(state, t)`; the evaluated component
scales are reused without scanning observations.
"""
function draw_subset!(rng::Random.AbstractRNG, cv::MarkedControlVariate,
    state::AbstractPDMPState, t::Real, D::Real, B::Real)
    envelope = cv.envelope
    N = size(envelope.weights, 2)
    m = cv.m
    D >= 0 && B >= 0 || throw(ArgumentError("marked envelope parts must be nonnegative"))
    total = D + B

    if iszero(B) || (!iszero(D) && rand(rng) * total <= D)
        _draw_uniform_without_replacement!(rng, cv.subset, N, cv.sampling_map)
    else
        component = _draw_component(rng, envelope, B)
        table = envelope.alias_tables[component]
        table === nothing && error("positive residual component has no alias table")
        distinguished = rand(rng, table)
        cv.subset[1] = distinguished
        if m > 1
            _draw_uniform_without_replacement!(rng, view(cv.subset, 2:m), N,
                cv.sampling_map; excluded=distinguished)
        end
    end

    subset_weight = 0.0
    for r in axes(envelope.weights, 1)
        row_sum = 0.0
        for i in cv.subset
            row_sum += envelope.weights[r, i]
        end
        subset_weight += envelope.scales[r] * row_sum
    end
    return D + (N / m) * subset_weight
end

function _append_marked_segment!(combined::PiecewiseAffineBound,
    envelope::SeparableResidualEnvelope, state::AbstractPDMPState,
    left::Float64, right::Float64, deterministic_left::Float64,
    deterministic_slope::Float64)
    right > left || return combined
    B_left = total_residual_bound(envelope, state, left)
    B_right = total_residual_bound(envelope, state, right)
    residual_slope = (B_right - B_left) / (right - left)
    append_affine_segment!(combined, left, right,
        max(0.0, deterministic_left) + B_left,
        deterministic_slope + residual_slope)
    return combined
end

function _append_marked_prefix!(combined::PiecewiseAffineBound,
    cv::MarkedControlVariate, alg::GridAdaptiveState, state::AbstractPDMPState,
    effective_horizon::Float64, modes, first_cell::Int, last_cell::Int,
    first_deterministic_segment::Int)
    if modes.use_linear
        deterministic = alg.affine_bound
        for j in first_deterministic_segment:deterministic.n_segments
            left = deterministic.t_breaks[j]
            left >= effective_horizon && break
            right = min(deterministic.t_breaks[j + 1], effective_horizon)
            _append_marked_segment!(combined, cv.envelope, state, left, right,
                deterministic.y_left[j], deterministic.slopes[j])
        end
    else
        pcb = alg.pcb
        for j in first_cell:last_cell
            left = pcb.t_grid[j]
            left >= effective_horizon && break
            right = min(pcb.t_grid[j + 1], effective_horizon)
            _append_marked_segment!(combined, cv.envelope, state, left, right,
                max(0.0, pcb.Λ_vals[j]), 0.0)
        end
    end
    return combined
end

function _marked_bound_violation(alg, stats, actual, bound, kind)
    actual <= bound * (1 + 1e-10) + 1e-12 && return false
    _inc_counter_grid_bound_violations(stats)
    message = "marked GridThinning $(kind) bound violated: actual=$(actual), bound=$(bound)"
    kind === :subset && throw(ErrorException(message *
        "; shrinking the deterministic grid cannot repair a residual envelope"))
    alg.bound_violation === :shrink && return true
    throw(ErrorException(message * "; the invalid proposal was discarded"))
end

function _marked_horizon_cells(pcb::PiecewiseConstantBound, effective_horizon::Float64)
    return max(0, min(length(pcb.Λ_vals),
        searchsortedfirst(pcb.t_grid, effective_horizon) - 1))
end

function _deterministic_prefix_area(alg::GridAdaptiveState, modes, n_cells::Int)
    modes.use_linear && return total_area(alg.affine_bound)
    area = 0.0
    for j in 1:n_cells
        area += max(0.0, alg.pcb.Λ_vals[j]) *
            (alg.pcb.t_grid[j + 1] - alg.pcb.t_grid[j])
    end
    return area
end


function _extend_marked_bound_to_budget!(cv::MarkedControlVariate,
    alg::GridAdaptiveState, state::AbstractPDMPState, flow::ContinuousDynamics,
    provider, modes, stats::AbstractStatisticCounter,
    effective_horizon::Float64, target_area::Float64,
    n_cells_bounded::Int, deterministic_area::Float64)
    combined = alg.marked_bound
    n_horizon = _marked_horizon_cells(alg.pcb, effective_horizon)
    while total_area(combined) <= target_area && n_cells_bounded < n_horizon
        first_cell = n_cells_bounded + 1
        cell_horizon = min(alg.pcb.t_grid[first_cell + 1], effective_horizon)
        first_segment = alg.affine_bound.n_segments + 1
        n_cells_bounded, _ = _build_grid_bound_prefix!(alg.pcb, state, flow,
            provider, alg, stats, alg.state_cache, cell_horizon, Inf,
            NoGridBoundaryProbe(), modes;
            start_cell=first_cell, initial_integral=deterministic_area,
            append=first_cell > 1)
        _append_marked_prefix!(combined, cv, alg, state, effective_horizon,
            modes, first_cell, n_cells_bounded, first_segment)
        deterministic_area = _deterministic_prefix_area(alg, modes, n_cells_bounded)
    end
    return n_cells_bounded, deterministic_area
end

const OrdinaryMarkedLinearFlow = Union{BouncyParticle,ZigZag}

function next_event_time(rng::Random.AbstractRNG,
    model::PDMPModel{<:MarkedControlVariate}, flow::OrdinaryMarkedLinearFlow,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64=Inf,
    include_refresh::Bool=true, max_horizon_event::Symbol=:horizon_hit)::GridEvent

    cv = model.grad
    begin_search!(cv, state)
    default_return = GradientMeta(alg.empty_∇ϕx)

    # A shrink restart discards the invalid proposal and rebuilds from the
    # unchanged physical state while retaining the frozen anchor.
    while true
        _invalidate_cached_gradient!(alg)
        reset_affine_bound!(alg.marked_bound)
        reset_affine_bound!(alg.affine_bound)
        fill!(alg.pcb.Λ_vals, 0.0)

        λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
        τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
        effective_horizon, horizon_event = _effective_grid_horizon(
            cv, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)
        cumulative_exp = Random.randexp(rng)
        provider = _grid_event_provider(model, flow, alg, stats)
        modes = _grid_bound_modes(alg, state, flow, provider)
        n_cells_bounded = 0
        deterministic_area = 0.0
        n_cells_bounded, deterministic_area = _extend_marked_bound_to_budget!(
            cv, alg, state, flow, provider, modes, stats, effective_horizon,
            cumulative_exp, n_cells_bounded, deterministic_area)

        restart_after_violation = false
        for _ in 1:alg.safety_limit
            τ_proposal, bar_M = propose_event_time(rng, alg.marked_bound, cumulative_exp)
            if τ_proposal >= effective_horizon || !isfinite(τ_proposal)
                max_t_max = max_grid_horizon(flow)
                return _return_grid_horizon!(alg, stats, flow, alg.t_max[],
                    effective_horizon, horizon_event, max_t_max, default_return)
            end
            if τ_refresh < τ_proposal
                return τ_refresh, :refresh, default_return
            end

            candidate = alg.state_cache2
            copyto!(candidate, state)
            move_forward_time!(candidate, τ_proposal, flow)
            D = modes.use_linear ? max(0.0, alg.affine_bound(τ_proposal)) :
                max(0.0, alg.pcb(τ_proposal))
            B = total_residual_bound(cv.envelope, state, τ_proposal)
            if _marked_bound_violation(alg, stats, D + B, bar_M, :aggregate) &&
                alg.bound_violation === :shrink
                _shrink_grid_after_bound_violation!(alg, stats)
                restart_after_violation = true
                break
            end

            _inc_counter_grid_acceptance_gradient_calls(stats)
            G = compute_gradient!(candidate, cv, flow, cache)
            deterministic_actual = λ(candidate.ξ, G, flow)
            if _marked_bound_violation(alg, stats, deterministic_actual, D, :deterministic) &&
                alg.bound_violation === :shrink
                _shrink_grid_after_bound_violation!(alg, stats)
                restart_after_violation = true
                break
            end

            M_subset = draw_subset!(rng, cv, state, τ_proposal, D, B)
            fill!(cv.residual_buffer, 0.0)
            residual_gradient!(cv.residual_buffer, cv.residual_oracle,
                candidate.ξ.x, cv.subset, cv.active_anchor)
            G .+= (size(cv.envelope.weights, 2) / cv.m) .* cv.residual_buffer
            actual = λ(candidate.ξ, G, flow)
            _inc_counter_grid_acceptance_tests(stats)
            _marked_bound_violation(alg, stats, actual, M_subset, :subset)

            if rand(rng) * M_subset <= actual
                if ispositive(D)
                    _adapt_grid_N!(alg, min(deterministic_actual / D, 1.0))
                end
                _adapt_grid_t_max!(alg, τ_proposal, cv)
                alg.max_observed_rate[] = max(alg.max_observed_rate[], actual)
                return τ_proposal, :reflect, GradientMeta(G)
            end
            cumulative_exp += Random.randexp(rng)
            n_cells_bounded, deterministic_area = _extend_marked_bound_to_budget!(
                cv, alg, state, flow, provider, modes, stats, effective_horizon,
                cumulative_exp, n_cells_bounded, deterministic_area)
        end
        restart_after_violation && continue
        _throw_grid_safety_limit_error(state, flow, model;
            t_invalid=effective_horizon, message="Safety limit reached in marked GridThinning")
    end
end

function next_event_time(rng::Random.AbstractRNG,
    model::PDMPModel{<:MarkedControlVariate}, flow::OrdinaryMarkedLinearFlow,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, detect_boundaries::Bool)::GridEvent
    detect_boundaries && throw(ArgumentError(
        "support-boundary detection is not yet supported by MarkedControlVariate GridThinning"))
    return next_event_time(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event)
end
