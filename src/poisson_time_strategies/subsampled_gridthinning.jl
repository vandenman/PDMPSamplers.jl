# Exact subsampling-control-variate GridThinning for certified trajectory envelopes.

# For both ordinary equal-inclusion sampling and balanced equal-inclusion
# sampling, a total batch of m contributions from a population of N terms has
# Horvitz--Thompson multiplier N/m.  In the balanced case this is also
# (N/n_strata)/(m/n_strata); the stratum count cancels.
@inline function _subsampling_scale(cv::SubsampledControlVariate)
    return n_observations(cv.envelope) / cv.m
end

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

function _draw_uniform_without_replacement_range!(rng::Random.AbstractRNG,
        out::AbstractVector, first::Int, count::Int, N::Int,
        scratch::Dict{Int,Int}; excluded::Int=0)
    empty!(scratch)
    population = iszero(excluded) ? N : N - 1
    @inbounds for offset in 0:(count - 1)
        remaining = population - offset
        slot = rand(rng, 1:remaining)
        logical_index = get(scratch, slot, slot)
        scratch[slot] = get(scratch, remaining, remaining)
        out[first + offset] = iszero(excluded) || logical_index < excluded ?
            logical_index : logical_index + 1
    end
    return out
end

function _draw_component(rng::Random.AbstractRNG, envelope::AbstractResidualEnvelope, B::Real)
    u = rand(rng) * B
    cumulative = envelope.cumulative_masses
    component = min(searchsortedlast(cumulative, u) + 1,
        lastindex(cumulative))
    return component
end

_draw_distinguished(rng, envelope::SeparableResidualEnvelope, component) =
    rand(rng, envelope.alias_tables[component])
function _draw_distinguished(rng, envelope::BlockSeparableResidualEnvelope,
        component)
    local_index = rand(rng, envelope.alias_tables[component])
    return (envelope.component_blocks[component] - 1) *
        size(envelope.weights, 2) + local_index
end
_draw_distinguished(rng, envelope::GroupedResidualEnvelope, component) =
    rand(rng, envelope.members[component])

function _draw_base_subset!(rng, cv::SubsampledControlVariate,
        ::UniformSubsamplingDesign)
    N = n_observations(cv.envelope)
    return _draw_uniform_without_replacement!(
        rng, cv.subset, N, cv.sampling_map)
end

function _draw_base_subset!(rng, cv::SubsampledControlVariate,
        design::BalancedStratifiedSubsamplingDesign)
    offset = 0
    destination = 1
    for _ in 1:design.n_strata
        selected = @view cv.subset[
            destination:(destination + design.per_stratum - 1)]
        _draw_uniform_without_replacement!(rng, selected,
            design.stratum_size, cv.sampling_map)
        @inbounds for j in eachindex(selected)
            selected[j] += offset
        end
        destination += design.per_stratum
        offset += design.stratum_size
    end
    return cv.subset
end

function _draw_base_subset_conditional!(rng, cv::SubsampledControlVariate,
        ::UniformSubsamplingDesign, distinguished::Int)
    cv.subset[1] = distinguished
    if cv.m > 1
        _draw_uniform_without_replacement_range!(rng, cv.subset, 2,
            cv.m - 1, n_observations(cv.envelope), cv.sampling_map;
            excluded=distinguished)
    end
    return cv.subset
end

function _draw_base_subset_conditional!(rng, cv::SubsampledControlVariate,
        design::BalancedStratifiedSubsamplingDesign, distinguished::Int)
    distinguished_stratum, local0 = divrem(
        distinguished - 1, design.stratum_size)
    distinguished_stratum += 1
    distinguished_local = local0 + 1
    destination = 1
    for stratum in 1:design.n_strata
        offset = (stratum - 1) * design.stratum_size
        if stratum == distinguished_stratum
            cv.subset[destination] = distinguished
            destination += 1
            count = design.per_stratum - 1
            if count > 0
                selected = @view cv.subset[destination:(destination + count - 1)]
                _draw_uniform_without_replacement!(rng, selected,
                    design.stratum_size, cv.sampling_map;
                    excluded=distinguished_local)
                @inbounds for j in eachindex(selected)
                    selected[j] += offset
                end
                destination += count
            end
        else
            selected = @view cv.subset[
                destination:(destination + design.per_stratum - 1)]
            _draw_uniform_without_replacement!(rng, selected,
                design.stratum_size, cv.sampling_map)
            @inbounds for j in eachindex(selected)
                selected[j] += offset
            end
            destination += design.per_stratum
        end
    end
    return cv.subset
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
    scales = envelope.scales
    N = n_observations(envelope)
    m = cv.m
    total = D + B

    residual_source = !(iszero(B) ||
        (!iszero(D) && rand(rng) * total <= D))
    if !residual_source
        _draw_base_subset!(rng, cv, cv.subset_design)
    else
        component = _draw_component(rng, envelope, B)
        distinguished = _draw_distinguished(rng, envelope, component)
        _draw_base_subset_conditional!(
            rng, cv, cv.subset_design, distinguished)
    end
    record_subsampling_mark_source!(cv.residual_oracle, residual_source)

    subset_weight = 0.0
    @inbounds for i in cv.subset
        subset_weight += observation_residual_bound(envelope, i)
    end
    return D + _subsampling_scale(cv) * subset_weight
end

@inline function _uniform_subset_probability(N::Int, m::Int)
    probability = 1.0
    @inbounds for j in 1:m
        probability *= j / (N - m + j)
    end
    return probability
end

@inline function _base_subset_probability(cv::SubsampledControlVariate,
        ::UniformSubsamplingDesign)
    return _uniform_subset_probability(n_observations(cv.envelope), cv.m)
end

@inline function _base_subset_probability(cv::SubsampledControlVariate,
        design::BalancedStratifiedSubsamplingDesign)
    return _uniform_subset_probability(
        design.stratum_size, design.per_stratum)^design.n_strata
end

@inline function _selected_subset_probability(cv::SubsampledControlVariate,
        D::Real, B::Real, selected_bound::Real)
    total = D + B
    ispositive(total) || return 0.0
    return _base_subset_probability(cv, cv.subset_design) *
        selected_bound / total
end

_signed_subset_bound(state, gradient, ::ContinuousDynamics, D, M) = M
function _signed_subset_bound(state, gradient,
        ::Union{BouncyParticle,AnyBoomerang}, D, M)
    residual_bound = max(0.0, M - D)
    signed_deterministic = dot(state.ξ.θ, gradient)
    return max(0.0, signed_deterministic + residual_bound)
end
function _signed_subset_bound(state, gradient,
        flow::PreconditionedDynamics, D, M)
    dynamics = flow.dynamics
    if dynamics isa BouncyParticle || dynamics isa AnyBoomerang
        residual_bound = max(0.0, M - D)
        signed_deterministic = dot(state.ξ.θ, gradient)
        return max(0.0, signed_deterministic + residual_bound)
    end
    return M
end

struct SubsamplingCandidateResult
    accepted::Bool
    deterministic_actual::Float64
    actual::Float64
end

# Research-only immutable capture of the candidate immediately before the
# selected residual oracle is invoked. It is disabled by default so the
# production hot path retains its existing allocation behaviour.
const _CAPTURE_SUBSAMPLING_FAILURE_FIXTURE = Ref(false)
const _PENDING_SUBSAMPLING_FAILURE_FIXTURE = Ref{Any}(nothing)

enable_subsampling_failure_fixture_capture!(enabled::Bool=true) =
    (_CAPTURE_SUBSAMPLING_FAILURE_FIXTURE[] = enabled;
     _PENDING_SUBSAMPLING_FAILURE_FIXTURE[] = nothing; nothing)

subsampling_failure_context(oracle, envelope, subset) = NamedTuple()

function _capture_subsampling_candidate_fixture!(candidate, flow, cv, alg,
        τ, D, B, cell_roof, M_subset, residual_subset_bound,
        tight_subset_bound, deterministic_actual, deterministic_gradient)
    _CAPTURE_SUBSAMPLING_FAILURE_FIXTURE[] || return nothing
    resolved_flow = _underlying_flow(flow)
    mean = resolved_flow isa AnyBoomerang ? copy(resolved_flow.μ) : Float64[]
    covariance = if hasproperty(resolved_flow, :ΣL)
        factor = Matrix(resolved_flow.ΣL)
        factor * transpose(factor)
    else
        Matrix{Float64}(undef, 0, 0)
    end
    free = hasproperty(candidate, :free) ? copy(candidate.free) : BitVector()
    context = subsampling_failure_context(
        cv.residual_oracle, cv.envelope, cv.subset)
    _PENDING_SUBSAMPLING_FAILURE_FIXTURE[] = merge((
        candidate_position=copy(candidate.ξ.x),
        physical_velocity=copy(candidate.ξ.θ),
        free=free,
        flow_mean=mean,
        flow_covariance=covariance,
        physical_time=float(candidate.t[]),
        proposal_elapsed=float(τ),
        subset=copy(cv.subset),
        factor_population=n_observations(cv.envelope),
        batch_size=cv.m,
        scaling=_subsampling_scale(cv),
        deterministic_gradient=copy(deterministic_gradient),
        deterministic_actual=float(deterministic_actual),
        deterministic_roof=float(D),
        residual_cell_mass=float(B),
        complete_cell_roof=float(cell_roof),
        selected_cell_bound=float(M_subset),
        residual_subset_bound=float(residual_subset_bound),
        tight_subset_bound=float(tight_subset_bound),
        control_variate_anchor=copy(cv.anchor),
        control_variate_anchor_identity=objectid(cv.anchor),
        flow_type=string(typeof(flow)),
        resolved_flow_type=string(typeof(resolved_flow)),
        bound_strategy=string(alg.bound)), context)
    return nothing
end

"""Evaluate one aggregate-accepted subsampling proposal without advancing the live state."""
function _evaluate_subsampling_candidate!(rng::Random.AbstractRNG,
        cv::SubsampledControlVariate, flow::ContinuousDynamics,
        state::AbstractPDMPState, candidate::AbstractPDMPState, cache,
        stats::AbstractStatisticCounter, violation_policy,
        τ::Real, D::Real, B::Real, cell_roof::Real, provider=nothing)
    _copy_t0 = time_ns()
    copyto!(candidate, state)
    _inc_counter_candidate_state_copies(stats)
    _inc_counter_candidate_state_copy_seconds(stats,
        (time_ns() - _copy_t0) * 1.0e-9)
    _move_t0 = time_ns()
    move_forward_time!(candidate, τ, flow)
    _inc_counter_candidate_move_forwards(stats)
    _inc_counter_candidate_move_forward_seconds(stats,
        (time_ns() - _move_t0) * 1.0e-9)
    scale = _subsampling_scale(cv)
    screening_token = residual_sampling_generation(cv.envelope, candidate)
    reused_screen = reuse_screened_residual_sampling!(cv.envelope,
        candidate, flow, 0.0, screening_token)
    reused_screen || prepare_residual_sampling!(cv.envelope, candidate, flow, 0.0)
    _inc_counter_subset_draws(stats)
    _subset_t0 = time_ns()
    M_subset = draw_subset!(rng, cv, D, B)
    selected_probability = _selected_subset_probability(cv, D, B, M_subset)
    _inc_counter_subset_draw_seconds(stats,
        (time_ns() - _subset_t0) * 1.0e-9)
    if needs_subsampling_residual_gradient(cv.residual_oracle)
        deterministic_gradient!(cache.∇ϕx, cv, candidate.ξ.x)
    end
    _refine_t0 = time_ns()
    _inc_counter_selected_subset_bound_evaluations(stats)
    residual_subset_bound = subsampling_residual_subset_bound(
        cv.residual_oracle, candidate, cache.∇ϕx, flow, D, M_subset,
        cv.subset, scale)
    if iszero(residual_subset_bound) ||
            (residual_subset_bound < M_subset &&
             rand(rng) * M_subset > residual_subset_bound)
        _inc_counter_selected_subset_refinement_seconds(stats,
            (time_ns() - _refine_t0) * 1.0e-9)
        record_subsampling_candidate!(cv.residual_oracle,
            cell_roof, D + B, true, selected_probability, residual_subset_bound,
            NaN, false)
        return SubsamplingCandidateResult(false, 0.0, 0.0)
    end

    _inc_counter_grid_acceptance_gradient_calls(stats)
    G = compute_gradient!(candidate, cv, flow, cache)
    deterministic_actual = λ(candidate, G, flow)
    if _bound_violated(deterministic_actual, D)
        _record_subsampling_bound_violation!(:deterministic,
            deterministic_actual, D, τ, violation_policy, candidate, flow, cv,
            G, provider)
    end
    if _subsampling_bound_violation(
            violation_policy, stats, deterministic_actual, D, :deterministic)
        return nothing
    end

    _inc_counter_selected_subset_bound_evaluations(stats)
    tight_subset_bound = subsampling_subset_bound(
        cv.residual_oracle, candidate, G, flow, D,
        residual_subset_bound, cv.subset, scale)
    _inc_counter_selected_subset_refinement_seconds(stats,
        (time_ns() - _refine_t0) * 1.0e-9)
    if iszero(tight_subset_bound) ||
            (tight_subset_bound < residual_subset_bound &&
             rand(rng) * residual_subset_bound > tight_subset_bound)
        record_subsampling_candidate!(cv.residual_oracle,
            cell_roof, D + B, true, selected_probability, tight_subset_bound,
            NaN, false)
        return SubsamplingCandidateResult(false, deterministic_actual, 0.0)
    end
    _inc_counter_selected_subset_bound_passes(stats)
    _inc_counter_subsampling_subset_evaluations(stats)
    _capture_subsampling_candidate_fixture!(candidate, flow, cv,
        violation_policy, τ, D, B, cell_roof, M_subset,
        residual_subset_bound, tight_subset_bound, deterministic_actual, G)
    fill!(cv.residual_buffer, 0.0)
    _residual_t0 = time_ns()
    cached = subsampling_cached_residual!(cv.residual_oracle,
        cv.residual_buffer, candidate, cv.subset, cv.anchor)
    cached || cv.residual_oracle(
        cv.residual_buffer, candidate.ξ.x, cv.subset, cv.anchor)
    _inc_counter_residual_oracle_seconds(stats,
        (time_ns() - _residual_t0) * 1.0e-9)
    actual = subsampling_candidate_rate!(cv.residual_oracle, candidate, G,
        cv.residual_buffer, scale, flow,
        deterministic_actual, cv.subset)
    _inc_counter_residual_oracle_evaluations(stats)
    _inc_counter_grid_acceptance_tests(stats)
    if _bound_violated(actual, tight_subset_bound)
        if _CAPTURE_SUBSAMPLING_FAILURE_FIXTURE[] &&
                _PENDING_SUBSAMPLING_FAILURE_FIXTURE[] !== nothing
            _PENDING_SUBSAMPLING_FAILURE_FIXTURE[] = merge(
                _PENDING_SUBSAMPLING_FAILURE_FIXTURE[], (
                    selected_residual=copy(cv.residual_buffer),
                    final_candidate_gradient=copy(G),
                    actual_rate=float(actual)))
        end
        _record_subsampling_bound_violation!(:subset, actual,
            tight_subset_bound, τ, violation_policy, candidate, flow, cv,
            G, provider)
    end
    if _subsampling_bound_violation(
            violation_policy, stats, actual, tight_subset_bound, :subset)
        return nothing
    end
    _final_t0 = time_ns()
    accepted = rand(rng) * tight_subset_bound <= actual
    _inc_counter_final_thinning_seconds(stats,
        (time_ns() - _final_t0) * 1.0e-9)
    accepted && _inc_counter_final_thinning_acceptances(stats)
    record_subsampling_candidate!(cv.residual_oracle,
        cell_roof, D + B, true, selected_probability, tight_subset_bound,
        actual, accepted)
    positive_residual_rate = λ(candidate, cv.residual_buffer, flow)
    rmul!(cv.residual_buffer, -1)
    negative_residual_rate = λ(candidate, cv.residual_buffer, flow)
    rmul!(cv.residual_buffer, -1)
    residual_rate = scale *
        (positive_residual_rate + negative_residual_rate)
    _record_t0 = time_ns()
    record_subsampling_proposal!(cv.residual_oracle,
        D, B, tight_subset_bound, deterministic_actual, residual_rate,
        actual, accepted)
    _inc_counter_statistic_recordings(stats)
    _inc_counter_statistic_recording_seconds(stats,
        (time_ns() - _record_t0) * 1.0e-9)
    return SubsamplingCandidateResult(accepted, deterministic_actual, actual)
end

function _append_subsampling_segment!(combined::PiecewiseAffineBound,
    envelope::AbstractResidualEnvelope, state::AbstractPDMPState,
    flow::ContinuousDynamics,
    left::Float64, right::Float64, deterministic_left::Float64,
    deterministic_slope::Float64)
    right > left || return combined
    scales = component_cell_scales!(envelope.scales, envelope, state, flow,
        left, right)
    residual_cell = residual_affine_cell_bound(
        envelope, state, flow, left, right)
    if residual_cell === nothing
        B_cell = dot(scales, envelope.totals)
        append_affine_segment!(combined, left, right,
            pos(deterministic_left) + B_cell, deterministic_slope)
    else
        # The residual hook is itself certified on the closed cell.  The
        # generic constant path above remains the fallback for every other
        # envelope and flow combination.
        append_affine_segment!(combined, left, right,
            pos(deterministic_left) + residual_cell.left,
            deterministic_slope + residual_cell.slope)
    end
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
            _append_subsampling_segment!(combined, cv.envelope, state, flow,
                left, right, deterministic.y_left[j], deterministic.slopes[j])
        end
    else
        pcb = alg.pcb
        for j in first_cell:last_cell
            left = pcb.t_grid[j]
            left >= effective_horizon && break
            right = min(pcb.t_grid[j + 1], effective_horizon)
            _append_subsampling_segment!(combined, cv.envelope, state, flow,
                left, right, pos(pcb.Λ_vals[j]), zero(pcb.Λ_vals[j]))
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

# Research diagnostic hook.  The ordinary path pays only the existing failed
# comparison; snapshots are allocated solely after a violation has occurred.
const _LAST_SUBSAMPLING_BOUND_VIOLATION = Ref{Any}(nothing)

clear_subsampling_bound_violation_diagnostic!() =
    (_LAST_SUBSAMPLING_BOUND_VIOLATION[] = nothing)
last_subsampling_bound_violation_diagnostic() =
    _LAST_SUBSAMPLING_BOUND_VIOLATION[]

function _record_subsampling_bound_violation!(kind, actual, bound, tau,
        alg, candidate, flow, cv, gradient=nothing, provider=nothing)
    grid = alg.pcb.t_grid
    cell = clamp(searchsortedlast(grid, tau), 1, length(grid) - 1)
    contributions = gradient === nothing ? Float64[] :
        max.(0.0, candidate.ξ.θ .* gradient)
    coordinate = isempty(contributions) ? 0 : argmax(contributions)
    metric = hasproperty(flow, :metric) && hasproperty(flow.metric, :scale) ?
        copy(flow.metric.scale) : Float64[]
    free = hasproperty(candidate, :free) ? copy(candidate.free) : BitVector()
    anchor = copy(cv.anchor)
    position = copy(candidate.ξ.x)
    rate_grid = Float64[]
    derivative_grid = Float64[]
    finite_difference_derivative_grid = Float64[]
    diagnostic_times = Float64[]
    captured = _PENDING_SUBSAMPLING_FAILURE_FIXTURE[]
    if captured === nothing && provider !== nothing && flow isa Union{AnyBoomerang,
            PreconditionedDynamics{<:AbstractPreconditioner,<:AnyBoomerang}}
        # Research-only failure path: independently replay the complete cell.
        # The ordinary event path never enters this allocation-heavy block.
        diagnostic_times = collect(range(grid[cell], grid[cell + 1], length=401))
        rate_grid = similar(diagnostic_times)
        derivative_grid = similar(diagnostic_times)
        finite_difference_derivative_grid = similar(diagnostic_times)
        for (k, t) in pairs(diagnostic_times)
            state_t = copy(candidate)
            move_forward_time!(state_t, t - tau, flow)
            rate_grid[k], derivative_grid[k] =
                rate_and_derivative(state_t, flow, provider)
            eps_t = min(1.0e-5,
                max(1.0e-8, 0.2 * min(t - grid[cell], grid[cell + 1] - t)))
            if t == grid[cell]
                left = copy(state_t); right = copy(state_t)
                move_forward_time!(right, eps_t, flow)
                q_left, _ = rate_and_derivative(left, flow, provider)
                q_right, _ = rate_and_derivative(right, flow, provider)
                finite_difference_derivative_grid[k] = (q_right - q_left) / eps_t
            elseif t == grid[cell + 1]
                left = copy(state_t); right = copy(state_t)
                move_forward_time!(left, -eps_t, flow)
                q_left, _ = rate_and_derivative(left, flow, provider)
                q_right, _ = rate_and_derivative(right, flow, provider)
                finite_difference_derivative_grid[k] = (q_right - q_left) / eps_t
            else
                left = copy(state_t); right = copy(state_t)
                move_forward_time!(left, -eps_t, flow)
                move_forward_time!(right, eps_t, flow)
                q_left, _ = rate_and_derivative(left, flow, provider)
                q_right, _ = rate_and_derivative(right, flow, provider)
                finite_difference_derivative_grid[k] =
                    (q_right - q_left) / (2eps_t)
            end
        end
    end
    resolved_flow = _underlying_flow(flow)
    reference_mean = resolved_flow isa AnyBoomerang ? copy(resolved_flow.μ) : Float64[]
    reference_precision = resolved_flow isa AnyBoomerang ? Matrix(resolved_flow.Γ) :
        Matrix{Float64}(undef, 0, 0)
    base = (
        stage=kind,
        physical_time=candidate.t[],
        proposal_time=float(tau),
        cell_left=grid[cell],
        cell_right=grid[cell + 1],
        coordinate=coordinate,
        subset=copy(cv.subset),
        actual=float(actual),
        bound=float(bound),
        absolute_excess=float(actual - bound),
        relative_excess=float(actual / bound - 1),
        anchor_identity=objectid(cv.anchor),
        anchor_distance=norm(position - anchor),
        position=position,
        velocity=copy(candidate.ξ.θ),
        free=free,
        metric=metric,
        anchor=anchor,
        coordinate_contributions=contributions,
        deterministic_gradient_provider_type=string(typeof(cv.deterministic_gradient!)),
        deterministic_hvp_provider_type=string(typeof(cv.deterministic_hvp!)),
        rate_derivative_provider_type=string(typeof(provider)),
        bound_strategy=string(alg.bound),
        curvature_bound=alg.curvature_bound isa Real ? float(alg.curvature_bound) : NaN,
        resolved_flow_type=string(typeof(flow)),
        resolved_inner_flow_type=string(typeof(resolved_flow)),
        reference_mean=reference_mean,
        reference_precision=reference_precision,
        diagnostic_times=diagnostic_times,
        signed_rate=rate_grid,
        analytic_rate_derivative=derivative_grid,
        finite_difference_rate_derivative=finite_difference_derivative_grid,
        recorded_cell_roof=alg.pcb.Λ_vals[cell])
    _LAST_SUBSAMPLING_BOUND_VIOLATION[] = captured === nothing ? base :
        merge(base, (immutable_candidate=captured,))
    _PENDING_SUBSAMPLING_FAILURE_FIXTURE[] = nothing
    return nothing
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

function _next_subsampled_event_time_with_provider!(rng::Random.AbstractRNG,
    provider,
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
        Λ_vals::Vector{Float64} = alg.pcb.Λ_vals
        @inbounds for i in eachindex(Λ_vals)
            Λ_vals[i] = 0.0
        end

        λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
        τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
        effective_horizon, horizon_event = _effective_grid_horizon(
            cv, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)
        prepare_residual_horizon!(
            cv.envelope, state, flow, effective_horizon)
        cumulative_exp = Random.randexp(rng)
        modes = _grid_bound_modes(alg, state, flow, provider)
        n_cells_bounded = 0
        deterministic_area = 0.0
        _schedule_t0 = time_ns()
        _inc_counter_grid_schedule_builds(stats)
        n_cells_bounded, deterministic_area = _extend_subsampling_bound_to_budget!(
            cv, alg, state, flow, provider, modes, stats, effective_horizon,
            cumulative_exp, n_cells_bounded, deterministic_area)
        _inc_counter_grid_schedule_build_seconds(stats,
            (time_ns() - _schedule_t0) * 1.0e-9)

        # A valid subsampling envelope may produce arbitrarily many genuine
        # rejections before the next event. A proposal-count cap would turn
        # envelope looseness into a correctness-affecting runtime failure;
        # the finite search horizon and refresh clock provide termination.
        while true
            _loop_t0 = time_ns()
            _search_t0 = time_ns()
            _poisson_t0 = time_ns()
            τ_proposal, bar_M = propose_event_time(rng, alg.subsampling_bound, cumulative_exp)
            _inc_counter_poisson_time_generations(stats)
            _inc_counter_poisson_time_generation_seconds(stats,
                (time_ns() - _poisson_t0) * 1.0e-9)
            _inc_counter_grid_schedule_searches(stats)
            _inc_counter_grid_schedule_search_seconds(stats,
                (time_ns() - _search_t0) * 1.0e-9)
            if τ_proposal >= effective_horizon || !isfinite(τ_proposal)
                _inc_counter_candidate_loop_iterations(stats)
                _inc_counter_candidate_loop_overhead_seconds(stats,
                    (time_ns() - _loop_t0) * 1.0e-9)
                max_t_max = max_grid_horizon(flow)
                return _return_grid_horizon!(alg, stats, flow, alg.t_max[],
                    effective_horizon, horizon_event, max_t_max, default_return)
            end
            if τ_refresh < τ_proposal
                _inc_counter_candidate_loop_iterations(stats)
                _inc_counter_candidate_loop_overhead_seconds(stats,
                    (time_ns() - _loop_t0) * 1.0e-9)
                return τ_refresh, :refresh, default_return
            end

            _inc_counter_subsampling_cell_roof_proposals(stats)
            _inc_counter_clock_candidates(stats)
            D = modes.use_linear ? pos(alg.affine_bound(τ_proposal)) :
                pos(alg.pcb(τ_proposal))
            _inc_counter_pointwise_screen_evaluations(stats)
            _screen_t0 = time_ns()
            B = screening_residual_bound(
                cv.envelope, state, flow, τ_proposal)
            _inc_counter_pointwise_screen_seconds(stats,
                (time_ns() - _screen_t0) * 1.0e-9)
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
                _inc_counter_pointwise_screen_passes(stats)
                result = _evaluate_subsampling_candidate!(rng, cv, flow, state,
                    alg.state_cache2, cache, stats, alg, τ_proposal, D, B,
                    bar_M, provider)
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
                    _inc_counter_reflection_events(stats)
                    _inc_counter_candidate_loop_iterations(stats)
                    _inc_counter_candidate_loop_overhead_seconds(stats,
                        (time_ns() - _loop_t0) * 1.0e-9)
                    return τ_proposal, :reflect, GradientMeta(cache.∇ϕx)
                end
            else
                record_subsampling_candidate!(cv.residual_oracle,
                    bar_M, aggregate, false, NaN, NaN, NaN, false)
            end

            # Every valid rejection consumes one further exponential budget.
            cumulative_exp += Random.randexp(rng)
            n_cells_bounded, deterministic_area = _extend_subsampling_bound_to_budget!(
                cv, alg, state, flow, provider, modes, stats, effective_horizon,
                cumulative_exp, n_cells_bounded, deterministic_area)
            _inc_counter_candidate_loop_iterations(stats)
            _inc_counter_candidate_loop_overhead_seconds(stats,
                (time_ns() - _loop_t0) * 1.0e-9)
        end
    end
end

function next_event_time(rng::Random.AbstractRNG,
    model::PDMPModel{<:SubsampledControlVariate}, flow::ContinuousDynamics,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64=Inf,
    include_refresh::Bool=true, max_horizon_event::Symbol=:horizon_hit)::GridEvent
    return _next_subsampled_event_time_with_provider!(rng,
        _grid_event_provider(model, flow, alg, stats), model, flow, alg, state,
        cache, stats, max_horizon, include_refresh, max_horizon_event)
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
