function _clock_state_at(state::StickyPDMPState, flow::ContinuousDynamics, τ::Real)
    state_at = copy(state)
    move_forward_time!(state_at, τ, flow)
    return state_at
end

function _clock_active_and_stickable(clock::SummedRateClock, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    return _active_beta_from_free(provider, state.free), _stickable_beta_from_can_stick(provider, can_stick)
end

_positive_logweight(logw::Real) = logw > -Inf

function _sample_from_logweights(rng::Random.AbstractRNG, indices::AbstractVector{Int}, logweights::AbstractVector{Float64})
    maxv = maximum(logweights)
    maxv == -Inf && throw(ArgumentError("cannot sample an unstick label because all inactive stickable rates are zero"))
    if maxv == Inf
        candidates = Int[]
        @inbounds for j in eachindex(indices)
            logweights[j] == Inf && push!(candidates, indices[j])
        end
        return rand(rng, candidates)
    end
    total = 0.0
    @inbounds for j in eachindex(indices)
        total += isfinite(logweights[j]) ? exp(logweights[j] - maxv) : 0.0
    end
    draw = rand(rng) * total
    last_candidate = 0
    @inbounds for j in eachindex(indices)
        if isfinite(logweights[j])
            last_candidate = j
            draw -= exp(logweights[j] - maxv)
            draw <= 0 && return indices[j]
        end
    end
    return indices[last_candidate]
end

function _has_inactive_stickable_beta(clock::AbstractAggregateUnstickClock, state::StickyPDMPState, can_stick::BitVector)
    for i in beta_indices(clock.slab_provider)
        if can_stick[i] && !state.free[i]
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
        isfinite(weights[j]) && (weights[j] += log(_unstick_rate_constant(flow, state, indices[j])))
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
    state_at = iszero(τ) ? state : _clock_state_at(state, flow, τ)
    if !_uses_coordinate_velocity_constants(flow)
        active_beta, stickable_beta = _clock_active_and_stickable(clock, state_at, can_stick)
        lograte = aggregate_lograte(clock.slab_provider, clock.model_prior, log(unstick_rate_constant(flow, 1)), state_at.ξ.x, active_beta, stickable_beta)
        lograte == -Inf && return 0.0
        lograte == Inf && return Inf
        return exp(lograte)
    end
    weights = Vector{Float64}(undef, length(beta_indices(clock.slab_provider)))
    lograte = LogExpFunctions.logsumexp(_boundary_logweights_with_velocity!(weights, clock, flow, state_at, can_stick))
    lograte == -Inf && return 0.0
    lograte == Inf && return Inf
    return exp(lograte)
end

"""
    cumulative_hazard(clock, flow, state, t0, t1, can_stick)

Aggregate unstick cumulative hazard over elapsed-time interval `[t0, t1]`.
"""
function cumulative_hazard(clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    0 <= t0 <= t1 || throw(ArgumentError("expected 0 <= t0 <= t1, got t0=$t0, t1=$t1"))
    t0 == t1 && return 0.0
    rate(clock, flow, state, 0.5 * (Float64(t0) + Float64(t1)), can_stick) == Inf && return Inf
    value, _ = QuadGK.quadgk(τ -> rate(clock, flow, state, τ, can_stick), Float64(t0), Float64(t1); rtol=clock.rtol, atol=clock.atol)
    return max(0.0, value)
end

"""
    sample_time(rng, clock, flow, state, horizon, can_stick)

Sample the next aggregate unstick waiting time, capped at `horizon`. Returns
`Inf` when no aggregate unstick occurs before the horizon.
"""
function sample_time(rng::Random.AbstractRNG, clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    _has_inactive_stickable_beta(clock, state, can_stick) || return Inf
    rate(clock, flow, state, 0.0, can_stick) == Inf && return 0.0
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = cumulative_hazard(clock, flow, state, 0.0, Float64(horizon), can_stick)
        H < threshold && return Inf
        f = τ -> cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
        return Roots.find_zero(f, (0.0, Float64(horizon)), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
    end

    lo = 0.0
    hi = Float64(clock.initial_bracket)
    H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
    iterations = 0
    while H_hi < threshold && iterations < 60
        lo = hi
        hi *= clock.bracket_multiplier
        H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
        iterations += 1
    end
    H_hi < threshold && return Inf
    f = τ -> cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
    return Roots.find_zero(f, (lo, hi), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
end

"""
    sample_label(rng, clock, flow, state, can_stick)
    sample_label(rng, clock, flow, state, τ, can_stick)

Sample the coordinate label for an aggregate unstick event, either at `state` or
at elapsed time `τ` from `state`.
"""
function sample_label(rng::Random.AbstractRNG, clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, can_stick::BitVector)
    if !_uses_coordinate_velocity_constants(flow)
        active_beta, stickable_beta = _clock_active_and_stickable(clock, state, can_stick)
        return sample_unstick_label(rng, clock.slab_provider, clock.model_prior, state.ξ.x, active_beta, stickable_beta)
    end
    weights = Vector{Float64}(undef, length(beta_indices(clock.slab_provider)))
    _boundary_logweights_with_velocity!(weights, clock, flow, state, can_stick)
    return _sample_from_logweights(rng, beta_indices(clock.slab_provider), weights)
end

_fallback_clock(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}) = clock.fallback
_summed_fallback_clock(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}) = SummedRateClock(clock.slab_provider, clock.model_prior; rtol=clock.rtol, atol=clock.atol)

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
    return sample_time(rng, clock.fallback, flow, state, horizon, can_stick)
end

rate(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::AnyBoomerang, state::StickyPDMPState, τ::Real, can_stick::BitVector) =
    rate(_summed_fallback_clock(clock), flow, state, τ, can_stick)

rate(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::PreconditionedDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector) =
    rate(_summed_fallback_clock(clock), flow, state, τ, can_stick)

cumulative_hazard(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::AnyBoomerang, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector) =
    cumulative_hazard(_summed_fallback_clock(clock), flow, state, t0, t1, can_stick)

cumulative_hazard(clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::PreconditionedDynamics, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector) =
    cumulative_hazard(_summed_fallback_clock(clock), flow, state, t0, t1, can_stick)

sample_time(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::AnyBoomerang, state::StickyPDMPState, horizon::Real, can_stick::BitVector) =
    sample_time(rng, _summed_fallback_clock(clock), flow, state, horizon, can_stick)

sample_time(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::PreconditionedDynamics, state::StickyPDMPState, horizon::Real, can_stick::BitVector) =
    sample_time(rng, _summed_fallback_clock(clock), flow, state, horizon, can_stick)

sample_label(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector) =
    sample_label(rng, _summed_fallback_clock(clock), flow, state, can_stick)

sample_label(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::PreconditionedDynamics, state::StickyPDMPState, can_stick::BitVector) =
    sample_label(rng, _summed_fallback_clock(clock), flow, state, can_stick)

sample_label(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::AnyBoomerang, state::StickyPDMPState, τ::Real, can_stick::BitVector) =
    sample_label(rng, _summed_fallback_clock(clock), flow, state, τ, can_stick)

sample_label(rng::Random.AbstractRNG, clock::Union{LinearGaussianAggregateClock,ExponentialSumAggregateClock}, flow::PreconditionedDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector) =
    sample_label(rng, _summed_fallback_clock(clock), flow, state, τ, can_stick)
