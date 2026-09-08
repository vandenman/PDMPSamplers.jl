function _clock_state_at(state::StickyPDMPState, flow::ContinuousDynamics, τ::Real)
    state_at = _shallow_copy_sticky_state(state)
    move_forward_time!(state_at, τ, flow)
    return state_at
end

function _clock_active_and_stickable(clock::SummedRateClock, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    cache = clock.cache
    return _active_beta_from_free!(cache.active_beta, provider, state.free), _stickable_beta_from_can_stick!(cache.stickable_beta, provider, can_stick)
end

_positive_logweight(logw::Real) = logw > -Inf

function _has_inactive_stickable_beta(clock::AbstractAggregateUnstickClock, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    active_beta = _active_beta_from_free(provider, state.free)
    for (j, i) in enumerate(beta_indices(provider))
        if can_stick[i] && !state.free[i] &&
           log_model_add_odds(clock.model_prior, active_beta, j) > -Inf
            return true
        end
    end
    return false
end

function _has_inactive_stickable_beta(clock::SummedRateClock,
        state::StickyPDMPState, can_stick::BitVector)
    active_beta, stickable_beta = _clock_active_and_stickable(
        clock, state, can_stick)
    @inbounds for j in eachindex(active_beta, stickable_beta)
        if stickable_beta[j] && !active_beta[j] &&
           log_model_add_odds(clock.model_prior, active_beta, j) > -Inf
            return true
        end
    end
    return false
end

function _boundary_logweights_with_velocity!(weights::AbstractVector{Float64}, clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    active_beta, stickable_beta = _clock_active_and_stickable(clock, state, can_stick)
    boundary_logweights!(weights, provider, clock.model_prior, state.ξ.x, active_beta, stickable_beta)
    @inbounds for j in eachindex(indices)
        isfinite(weights[j]) && (weights[j] += log(
            _boundary_proposal_clock_constant(flow, state, indices[j])))
    end
    return weights
end

_uses_coordinate_velocity_constants(::ContinuousDynamics) = false
_uses_coordinate_velocity_constants(::AnyBoomerang) = true
_uses_coordinate_velocity_constants(::PreconditionedDynamics) = true

"""
    rate(clock, flow, state, τ, can_stick)

Instantaneous aggregate unstick rate at elapsed time `τ` from `state`, summing
over inactive stickable beta coordinates.
"""
function rate(clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    τ < 0 && throw(ArgumentError("τ must be non-negative"))
    diagnostics = clock.diagnostics
    diagnostics.enabled && (diagnostics.rate_evaluations += 1)
    movement_started = diagnostics.enabled ? time_ns() : UInt64(0)
    state_at = iszero(τ) ? state : _clock_state_at(state, flow, τ)
    diagnostics.enabled &&
        (diagnostics.state_movement_ns += time_ns() - movement_started)
    cache = clock.cache
    boundary_started = diagnostics.enabled ? time_ns() : UInt64(0)
    if !_uses_coordinate_velocity_constants(flow)
        active_beta, stickable_beta = _clock_active_and_stickable(clock, state_at, can_stick)
        lograte = aggregate_lograte!(cache.log_weights, clock.slab_provider, clock.model_prior, log(unstick_rate_constant(flow, 1)), state_at.ξ.x, active_beta, stickable_beta)
        lograte == -Inf && return 0.0
        lograte == Inf && return Inf
        result = exp(lograte)
        diagnostics.enabled &&
            (diagnostics.boundary_weight_ns += time_ns() - boundary_started)
        return result
    end
    lograte = LogExpFunctions.logsumexp(_boundary_logweights_with_velocity!(cache.log_weights, clock, flow, state_at, can_stick))
    lograte == -Inf && return 0.0
    lograte == Inf && return Inf
    result = exp(lograte)
    diagnostics.enabled &&
        (diagnostics.boundary_weight_ns += time_ns() - boundary_started)
    return result
end

"""
    cumulative_hazard(clock, flow, state, t0, t1, can_stick)

Aggregate unstick cumulative hazard over elapsed-time interval `[t0, t1]`.
"""
function cumulative_hazard(clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    0 <= t0 <= t1 || throw(ArgumentError("expected 0 <= t0 <= t1, got t0=$t0, t1=$t1"))
    t0 == t1 && return 0.0
    rate(clock, flow, state, 0.5 * (Float64(t0) + Float64(t1)), can_stick) == Inf && return Inf
    diagnostics = clock.diagnostics
    diagnostics.enabled &&
        (diagnostics.cumulative_hazard_evaluations += 1;
         diagnostics.quadrature_evaluations += 1)
    started = diagnostics.enabled ? time_ns() : UInt64(0)
    value, _ = QuadGK.quadgk(τ -> rate(clock, flow, state, τ, can_stick), Float64(t0), Float64(t1); rtol=clock.rtol, atol=clock.atol)
    diagnostics.enabled && (diagnostics.quadrature_ns += time_ns() - started)
    return max(0.0, value)
end

"""
    sample_time(rng, clock, flow, state, horizon, can_stick)

Sample the next aggregate unstick waiting time, capped at `horizon`. Returns
`Inf` when no aggregate unstick occurs before the horizon.
"""
function sample_time(rng::Random.AbstractRNG, clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    diagnostics = clock.diagnostics
    diagnostics.enabled && (diagnostics.aggregate_calls += 1)
    aggregate_started = diagnostics.enabled ? time_ns() : UInt64(0)
    if !_has_inactive_stickable_beta(clock, state, can_stick)
        diagnostics.enabled &&
            (diagnostics.aggregate_clock_ns += time_ns() - aggregate_started)
        return Inf
    end
    if rate(clock, flow, state, 0.0, can_stick) == Inf
        diagnostics.enabled &&
            (diagnostics.aggregate_clock_ns += time_ns() - aggregate_started)
        return 0.0
    end
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = cumulative_hazard(clock, flow, state, 0.0, Float64(horizon), can_stick)
        if H < threshold
            diagnostics.enabled &&
                (diagnostics.aggregate_clock_ns += time_ns() - aggregate_started)
            return Inf
        end
        f = τ -> begin
            diagnostics.enabled && (diagnostics.root_iterations += 1)
            cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
        end
        root_started = diagnostics.enabled ? time_ns() : UInt64(0)
        result = Roots.find_zero(f, (0.0, Float64(horizon)), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
        if diagnostics.enabled
            diagnostics.root_inversion_ns += time_ns() - root_started
            diagnostics.aggregate_clock_ns += time_ns() - aggregate_started
        end
        return result
    end

    lo = 0.0
    hi = Float64(clock.initial_bracket)
    H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
    while H_hi < threshold
        lo = hi
        next_hi = hi * clock.bracket_multiplier
        (!isfinite(next_hi) || next_hi <= hi) && return Inf
        hi = next_hi
        H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
    end
    f = τ -> begin
        diagnostics.enabled && (diagnostics.root_iterations += 1)
        cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
    end
    root_started = diagnostics.enabled ? time_ns() : UInt64(0)
    result = Roots.find_zero(f, (lo, hi), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
    if diagnostics.enabled
        diagnostics.root_inversion_ns += time_ns() - root_started
        diagnostics.aggregate_clock_ns += time_ns() - aggregate_started
    end
    return result
end

function reset_thinning_diagnostics!(clock::SummedRateClock)
    d = clock.diagnostics
    d.aggregate_calls = 0
    d.rate_evaluations = 0
    d.quadrature_evaluations = 0
    d.cumulative_hazard_evaluations = 0
    d.root_iterations = 0
    d.aggregate_clock_ns = 0
    d.state_movement_ns = 0
    d.boundary_weight_ns = 0
    d.quadrature_ns = 0
    d.root_inversion_ns = 0
    return clock
end

thinning_diagnostics(clock::SummedRateClock) = clock.diagnostics

"""
    sample_label(rng, clock, flow, state, can_stick)
    sample_label(rng, clock, flow, state, τ, can_stick)

Sample the coordinate label for an aggregate unstick event, either at `state` or
at elapsed time `τ` from `state`.
"""
function sample_label(rng::Random.AbstractRNG, clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, can_stick::BitVector)
    cache = clock.cache
    if !_uses_coordinate_velocity_constants(flow)
        active_beta, stickable_beta = _clock_active_and_stickable(clock, state, can_stick)
        boundary_logweights!(cache.log_weights, clock.slab_provider, clock.model_prior, state.ξ.x, active_beta, stickable_beta)
        return _sample_from_logweights(rng, beta_indices(clock.slab_provider), cache.log_weights)
    end
    _boundary_logweights_with_velocity!(cache.log_weights, clock, flow, state, can_stick)
    return _sample_from_logweights(rng, beta_indices(clock.slab_provider), cache.log_weights)
end

_fallback_clock(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}) = clock.fallback
_summed_fallback_clock(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}) = clock.fallback

rate(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector) =
    rate(_fallback_clock(clock), flow, state, τ, can_stick)

cumulative_hazard(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector) =
    cumulative_hazard(_fallback_clock(clock), flow, state, t0, t1, can_stick)

sample_time(rng::Random.AbstractRNG, clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, horizon::Real, can_stick::BitVector) =
    _sample_time_exact_fallback(rng, clock, flow, state, horizon, can_stick)

sample_label(rng::Random.AbstractRNG, clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, can_stick::BitVector) =
    sample_label(rng, _fallback_clock(clock), flow, state, can_stick)

function _sample_time_exact_fallback(rng::Random.AbstractRNG, clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock},
                                     flow::ContinuousDynamics, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    if !clock.allow_slow_fallback
        throw(ArgumentError(
            "$(nameof(typeof(clock))) has no certified residual sampler for provider " *
            "$(nameof(typeof(clock.slab_provider))) and flow $(nameof(typeof(flow))). " *
            "Use a provider/flow pair with a concrete certified method or construct the clock with allow_slow_fallback=true."
        ))
    end
    clock.diagnostics.fallbacks += 1
    clock.diagnostics.fallback_calls += 1
    clock.diagnostics.last_fallback_provider = nameof(typeof(clock.slab_provider))
    clock.diagnostics.last_fallback_dynamics = nameof(typeof(flow))
    return sample_time(rng, clock.fallback, flow, state, horizon, can_stick)
end

rate(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector) =
    rate(_summed_fallback_clock(clock), flow, state, τ, can_stick)

cumulative_hazard(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector) =
    cumulative_hazard(_summed_fallback_clock(clock), flow, state, t0, t1, can_stick)

sample_time(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, horizon::Real, can_stick::BitVector) =
    sample_time(rng, _summed_fallback_clock(clock), flow, state, horizon, can_stick)

sample_label(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, can_stick::BitVector) =
    sample_label(rng, _summed_fallback_clock(clock), flow, state, can_stick)

sample_label(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector) =
    sample_label(rng, _summed_fallback_clock(clock), flow, state, τ, can_stick)
