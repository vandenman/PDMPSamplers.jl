# Exact subsampling-control-variate GridThinning for certified trajectory envelopes.

function _draw_uniform_without_replacement!(rng::Random.AbstractRNG, out::AbstractVector,
    N::Int, scratch::Dict{Int,Int}; excluded::Int=0)
    empty!(scratch)
    population = iszero(excluded) ? N : N - 1
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
    cumulative = zero(eltype(envelope.totals))
    chosen = 0
    scales = envelope.scales
    totals = envelope.totals
    for r in eachindex(totals)
        mass = scales[r] * totals[r]
        ispositive(mass) && (chosen = r)
        cumulative += mass
        if u < cumulative
            return r
        end
    end
    return chosen
end

"""
    draw_subset!(rng, cv, D, B) -> M_S

Draw a fresh size-biased subsampling subset into `cv.subset` and return its
subset-specific envelope. `B` must be the total residual bound returned by
`total_residual_bound`; the component scales evaluated immediately beforehand
are reused without scanning observations.
"""
function draw_subset!(rng::Random.AbstractRNG, cv::SubsampledControlVariate,
    D::Real, B::Real)
    envelope = cv.envelope
    weights = envelope.weights
    scales = envelope.scales
    N = size(weights, 2)
    m = cv.m
    total = D + B

    if iszero(B) || (!iszero(D) && rand(rng) * total <= D)
        _draw_uniform_without_replacement!(rng, cv.subset, N, cv.sampling_map)
    else
        component = _draw_component(rng, envelope, B)
        table = envelope.alias_tables[component]
        distinguished = rand(rng, table)
        cv.subset[1] = distinguished
        if m > 1
            _draw_uniform_without_replacement!(rng, view(cv.subset, 2:m), N,
                cv.sampling_map; excluded=distinguished)
        end
    end

    subset_weight = 0.0
    @inbounds for i in cv.subset, r in axes(weights, 1)
        subset_weight += scales[r] * weights[r, i]
    end
    return D + (N / m) * subset_weight
end

"""Evaluate one aggregate-accepted subsampling proposal without advancing the live state."""
function _evaluate_subsampling_candidate!(rng::Random.AbstractRNG,
        cv::SubsampledControlVariate, flow::ContinuousDynamics,
        state::AbstractPDMPState, candidate::AbstractPDMPState, cache,
        stats::AbstractStatisticCounter, violation_policy,
        τ::Real, D::Real, B::Real)
    copyto!(candidate, state)
    move_forward_time!(candidate, τ, flow)
    _inc_counter_grid_acceptance_gradient_calls(stats)
    G = compute_gradient!(candidate, cv, flow, cache)
    deterministic_actual = λ(candidate, G, flow)
    if _subsampling_bound_violation(
            violation_policy, stats, deterministic_actual, D, :deterministic)
        return nothing
    end

    M_subset = draw_subset!(rng, cv, D, B)
    _inc_counter_subsampling_subset_evaluations(stats)
    fill!(cv.residual_buffer, 0.0)
    cv.residual_oracle(cv.residual_buffer, candidate.ξ.x, cv.subset, cv.anchor)
    axpy!(size(cv.envelope.weights, 2) / cv.m, cv.residual_buffer, G)
    actual = λ(candidate, G, flow)
    _inc_counter_grid_acceptance_tests(stats)
    if _subsampling_bound_violation(
            violation_policy, stats, actual, M_subset, :subset)
        return nothing
    end
    accepted = ispositive(M_subset) && rand(rng) * M_subset <= actual
    return (; accepted, G, deterministic_actual, actual)
end

function _append_subsampling_segment!(combined::PiecewiseAffineBound,
    envelope::SeparableResidualEnvelope, state::AbstractPDMPState,
    flow::ContinuousDynamics,
    left::Float64, right::Float64, deterministic_left::Float64,
    deterministic_slope::Float64)
    right > left || return combined
    scales = component_cell_scales!(envelope.scales, envelope, state, flow,
        left, right)
    B_cell = dot(scales, envelope.totals)
    append_affine_segment!(combined, left, right,
        pos(deterministic_left) + B_cell, deterministic_slope)
    return combined
end

function _append_subsampling_prefix!(combined::PiecewiseAffineBound,
    cv::SubsampledControlVariate, alg::GridAdaptiveState, state::AbstractPDMPState,
    flow::ContinuousDynamics,
    effective_horizon::Float64, modes, first_cell::Int, last_cell::Int,
    first_deterministic_segment::Int)
    if modes.use_linear
        deterministic = alg.affine_bound
        for j in first_deterministic_segment:deterministic.n_segments
            left = deterministic.t_breaks[j]
            left >= effective_horizon && break
            right = min(deterministic.t_breaks[j + 1], effective_horizon)
            _append_subsampling_segment!(combined, cv.envelope, state, flow, left, right,
                deterministic.y_left[j], deterministic.slopes[j])
        end
    else
        pcb = alg.pcb
        for j in first_cell:last_cell
            left = pcb.t_grid[j]
            left >= effective_horizon && break
            right = min(pcb.t_grid[j + 1], effective_horizon)
            _append_subsampling_segment!(combined, cv.envelope, state, flow, left, right,
                pos(pcb.Λ_vals[j]), zero(pcb.Λ_vals[j]))
        end
    end
    return combined
end

function _subsampling_bound_violation(alg::GridAdaptiveState, stats, actual, bound, kind)
    _bound_violated(actual, bound) || return false
    _inc_counter_grid_bound_violations(stats)
    message = "subsampling GridThinning $(kind) bound violated: actual=$(actual), bound=$(bound)"
    kind === :subset && throw(ErrorException(message *
        "; shrinking the deterministic grid cannot repair a residual envelope"))
    alg.bound_violation === :shrink && return true
    throw(ErrorException(message * "; the invalid proposal was discarded"))
end

function _extend_subsampling_bound_to_budget!(cv::SubsampledControlVariate,
    alg::GridAdaptiveState, state::AbstractPDMPState, flow::ContinuousDynamics,
    provider, modes, stats::AbstractStatisticCounter,
    effective_horizon::Float64, target_area::Float64,
    n_cells_bounded::Int, deterministic_area::Float64)
    combined = alg.subsampling_bound
    n_horizon = _grid_cell_count(
        alg.pcb.t_grid, length(alg.pcb.Λ_vals), effective_horizon)
    while total_area(combined) <= target_area && n_cells_bounded < n_horizon
        first_cell = n_cells_bounded + 1
        cell_horizon = min(alg.pcb.t_grid[first_cell + 1], effective_horizon)
        first_segment = alg.affine_bound.n_segments + 1
        n_cells_bounded, deterministic_area = _build_grid_bound_prefix!(alg.pcb, state, flow,
            provider, alg, stats, alg.state_cache, cell_horizon, Inf,
            NoGridBoundaryProbe(), modes;
            start_cell=first_cell, initial_integral=deterministic_area,
            append=first_cell > 1)
        _append_subsampling_prefix!(combined, cv, alg, state, flow, effective_horizon,
            modes, first_cell, n_cells_bounded, first_segment)
    end
    return n_cells_bounded, deterministic_area
end

function next_event_time(rng::Random.AbstractRNG,
    model::PDMPModel{<:SubsampledControlVariate}, flow::ContinuousDynamics,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64=Inf,
    include_refresh::Bool=true, max_horizon_event::Symbol=:horizon_hit)::GridEvent

    cv = model.grad
    default_return = GradientMeta(alg.empty_∇ϕx)

    # A shrink restart discards the invalid proposal and rebuilds from the
    # unchanged physical state while retaining the frozen anchor.
    while true
        _invalidate_cached_gradient!(alg)
        reset_affine_bound!(alg.subsampling_bound)
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
        n_cells_bounded, deterministic_area = _extend_subsampling_bound_to_budget!(
            cv, alg, state, flow, provider, modes, stats, effective_horizon,
            cumulative_exp, n_cells_bounded, deterministic_area)

        # A valid subsampling envelope may produce arbitrarily many genuine
        # rejections before the next event. A proposal-count cap would turn
        # envelope looseness into a correctness-affecting runtime failure;
        # the finite search horizon and refresh clock provide termination.
        while true
            τ_proposal, bar_M = propose_event_time(rng, alg.subsampling_bound, cumulative_exp)
            if τ_proposal >= effective_horizon || !isfinite(τ_proposal)
                max_t_max = max_grid_horizon(flow)
                return _return_grid_horizon!(alg, stats, flow, alg.t_max[],
                    effective_horizon, horizon_event, max_t_max, default_return)
            end
            if τ_refresh < τ_proposal
                return τ_refresh, :refresh, default_return
            end

            _inc_counter_subsampling_cell_roof_proposals(stats)
            D = modes.use_linear ? pos(alg.affine_bound(τ_proposal)) :
                pos(alg.pcb(τ_proposal))
            B = total_residual_bound(cv.envelope, state, flow, τ_proposal)
            if _subsampling_bound_violation(alg, stats, D + B, bar_M, :aggregate)
                _shrink_grid_after_bound_violation!(alg, stats)
                break
            end

            # The proposal clock uses the cellwise residual roof, whereas the
            # subsampling mixture below is normalized by the tighter pointwise
            # bound. Thin from the roof to that pointwise aggregate before any
            # gradient or residual work.
            aggregate = D + B
            if ispositive(aggregate) && rand(rng) * bar_M <= aggregate
                _inc_counter_subsampling_aggregate_accepts(stats)
                result = _evaluate_subsampling_candidate!(rng, cv, flow, state,
                    alg.state_cache2, cache, stats, alg, τ_proposal, D, B)
                if result === nothing
                    _shrink_grid_after_bound_violation!(alg, stats)
                    break
                end
                if result.accepted
                    if ispositive(D)
                        _adapt_grid_N!(alg,
                            min(result.deterministic_actual / D, 1.0))
                    end
                    _adapt_grid_t_max!(alg, τ_proposal, cv)
                    alg.max_observed_rate[] = max(
                        alg.max_observed_rate[], result.actual)
                    _inc_counter_subsampling_final_reflections(stats)
                    return τ_proposal, :reflect, GradientMeta(result.G)
                end
            end

            # Every valid rejection consumes one further exponential budget.
            cumulative_exp += Random.randexp(rng)
            n_cells_bounded, deterministic_area = _extend_subsampling_bound_to_budget!(
                cv, alg, state, flow, provider, modes, stats, effective_horizon,
                cumulative_exp, n_cells_bounded, deterministic_area)
        end
    end
end

function next_event_time(rng::Random.AbstractRNG,
    model::PDMPModel{<:SubsampledControlVariate}, flow::ContinuousDynamics,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, detect_boundaries::Bool)::GridEvent
    detect_boundaries && throw(ArgumentError(
        "support-boundary detection is not yet supported by SubsampledControlVariate GridThinning"))
    return next_event_time(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event)
end
