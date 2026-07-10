"""
A simple wrapper function so that the Sticky strategy knows whether a function
returns the waiting time, or the rate of an inhomogeneous Poisson process.

The rate_function must accept the same arguments as κ(i, x, free, θ)

"""
struct RateFunction{F<:Function}
    rate_function::F
end

"""


There are three options for the unfreezing time, κ.

- `AbstractVector`: For independent model and parameter priors, the rates are fixed and can be known in advance.
- `Function`: For dependent model priors, the rates only depend on whether the parameters are (non)zero.
- `RateFunction`: For dependent parameters priors, the rates depend on the specific values of the parameters, and the unfreeze times become inhomogeneous Poisson processes.

The first two should return the rate of an homogeneous Poisson process, such that

```julia
λ = κ[i] or κ(i, x, γ, θ)
rand(Exponential(inv(κ * abs(θf))))
```
equals the unfreezing time.

The third should return the rate of an inhomogeneous Poisson process, which is sampled through the algorithm specified by `alg`.
"""
struct Sticky{T<:PoissonTimeStrategy,U<:Union{Function,RateFunction,AbstractVector}} <: PoissonTimeStrategy
    alg::T
    κ::U
    can_stick::BitVector
end
Sticky(alg::PoissonTimeStrategy, κ::AbstractVector) = Sticky(alg, κ, .!isinf.(κ))
Sticky(::PoissonTimeStrategy, ::Function) = throw(ArgumentError("When κ is a function, a can_stick vector must be provided explicitly."))

"""
    AggregateSticky(alg, clock, can_stick)

Sticky strategy for dependent slab priors. The wrapped inner algorithm `alg`
handles reflection events, while `clock` samples a single aggregate unstick time
over inactive stickable beta coordinates. `can_stick` is a full-state Boolean
mask. The current implementation supports `ZigZag` and `BouncyParticle` flows
with a `GridThinningStrategy` inner algorithm.
"""
struct AggregateSticky{T<:PoissonTimeStrategy,C<:AbstractAggregateUnstickClock} <: PoissonTimeStrategy
    alg::T
    clock::C
    can_stick::BitVector
end

AggregateSticky(alg::PoissonTimeStrategy, clock::AbstractAggregateUnstickClock, can_stick::AbstractVector{Bool}) =
    AggregateSticky(alg, clock, BitVector(can_stick))
AggregateSticky(::PoissonTimeStrategy, ::AbstractAggregateUnstickClock) =
    throw(ArgumentError("AggregateSticky requires an explicit can_stick vector."))

requires_sticky_state(::Sticky) = true
requires_sticky_state(::AggregateSticky) = true

_supports_aggregate_sticky_flow(::ZigZag) = true
_supports_aggregate_sticky_flow(::BouncyParticle) = true
_supports_aggregate_sticky_flow(::ContinuousDynamics) = false

struct StickyLoopState{T<:PoissonTimeStrategy,U<:Union{Function,RateFunction,AbstractVector},V<:AbstractVector} <: PoissonTimeStrategy
    # A' could be the internal version of the wrapped algorithm
    inner_alg_state::T # this should perhaps be the more generic, i.e., _to_internal(Sticky.alg, ...)!
    κ::U
    can_stick::BitVector
    sticky_times::Vector{Float64}  # Absolute times of next freeze/unfreeze event
    stickable_indices::Vector{Int}
    sticky_pq::PriorityQueue{Int,Float64}
    empty_∇ϕx::V
end

mutable struct AggregateStickyLoopState{T<:PoissonTimeStrategy,C<:AbstractAggregateUnstickClock,V<:AbstractVector} <: PoissonTimeStrategy
    inner_alg_state::T
    clock::C
    can_stick::BitVector
    sticky_times::Vector{Float64}
    stickable_indices::Vector{Int}
    sticky_pq::PriorityQueue{Int,Float64}
    aggregate_unstick_time::Float64
    empty_∇ϕx::V
end

accept_reflection_event(rng::Random.AbstractRNG, alg::StickyLoopState, args...) = accept_reflection_event(rng, alg.inner_alg_state, args...)
accept_reflection_event(alg::StickyLoopState, args...) = accept_reflection_event(alg.inner_alg_state, args...)
accept_reflection_event(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, args...) = accept_reflection_event(rng, alg.inner_alg_state, args...)
accept_reflection_event(alg::AggregateStickyLoopState, args...) = accept_reflection_event(alg.inner_alg_state, args...)

_is_sticky_loop_state(::StickyLoopState) = true
_is_sticky_loop_state(::AggregateStickyLoopState) = true

# this could use less memory by looking at
function _to_internal(strat::Sticky, rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)

    d = length(state.ξ)
    sticky_times = fill(Inf, d)
    stickable_indices = findall(strat.can_stick)
    sticky_pq = PriorityQueue{Int,Float64}()

    internal_alg_ = _to_internal(strat.alg, rng, flow, model, state, cache, stats)
    state isa StickyPDMPState && set_active_set!(model, state.free)

    # old_velocity = copy(state.ξ.θ)
    # # zero is problematic because the unfreeze time divides by abs(θf[i]), so divide by zero
    # if any(iszero, old_velocity)
    #     old_velocity2 = initialize_velocity(flow, d)
    #     for i in eachindex(old_velocity, old_velocity2)
    #         if iszero(old_velocity[i])
    #             old_velocity[i] = old_velocity2[i]
    #         end
    #     end
    # end
    alg = StickyLoopState(internal_alg_, strat.κ, strat.can_stick, sticky_times, stickable_indices, sticky_pq, similar(state.ξ.x, 0))
    update_all_stick_times!(rng, alg, state, flow)
    # @show alg.sticky_times
    any(isnan, alg.sticky_times) && error("sticky_times contains NaN: $(alg.sticky_times)")

    return alg

end

function _to_internal(strat::AggregateSticky, rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    state isa StickyPDMPState || throw(ArgumentError("AggregateSticky requires StickyPDMPState; initialize_state must use requires_sticky_state"))
    _supports_aggregate_sticky_flow(flow) ||
        throw(ArgumentError("AggregateSticky currently supports only ZigZag and BouncyParticle flows; Boomerang and preconditioned flows need flow-specific boundary velocity laws"))
    d = length(state.ξ)
    length(strat.can_stick) == d || throw(DimensionMismatch("can_stick length $(length(strat.can_stick)) does not match dimension $d"))
    sticky_times = fill(Inf, d)
    stickable_indices = findall(strat.can_stick)
    sticky_pq = PriorityQueue{Int,Float64}()
    internal_alg_ = _to_internal(strat.alg, rng, flow, model, state, cache, stats)
    internal_alg_ isa GridAdaptiveState ||
        throw(ArgumentError("AggregateSticky requires an inner strategy with bounded event search; use GridThinningStrategy for now"))
    set_active_set!(model, state.free)
    alg = AggregateStickyLoopState(internal_alg_, copy(strat.clock), copy(strat.can_stick), sticky_times, stickable_indices, sticky_pq, Inf, similar(state.ξ.x, 0))
    update_all_stick_times!(rng, alg, state, flow)
    any(isnan, alg.sticky_times) && error("sticky_times contains NaN: $(alg.sticky_times)")
    isnan(alg.aggregate_unstick_time) && error("aggregate_unstick_time is NaN")
    return alg
end

function _set_sticky_time!(alg::StickyLoopState, i::Int, t::Float64)
    alg.sticky_times[i] = t
    alg.sticky_pq[i] = t
    return t
end

function _set_sticky_time!(alg::AggregateStickyLoopState, i::Int, t::Float64)
    alg.sticky_times[i] = t
    if isfinite(t)
        alg.sticky_pq[i] = t
    elseif haskey(alg.sticky_pq, i)
        delete!(alg.sticky_pq, i)
    end
    return t
end

"""
    unstick_rate_constant(flow, i)

Return the boundary velocity normalizing constant used by aggregate sticky
clocks for coordinate `i`. Currently implemented for `ZigZag` and
`BouncyParticle`; Boomerang and preconditioned flows throw until their
flow-specific boundary velocity laws are implemented.
"""
unstick_rate_constant(::ZigZag, ::Integer) = 1.0
unstick_rate_constant(::BouncyParticle, ::Integer) = sqrt(2 / π)
unstick_rate_constant(::AnyBoomerang, ::Integer) =
    throw(ArgumentError("aggregate sticky Boomerang boundary constants need the flow-specific velocity covariance and are not implemented yet"))
unstick_rate_constant(::PreconditionedDynamics, ::Integer) =
    throw(ArgumentError("aggregate sticky preconditioned boundary constants need the transformed boundary velocity law and are not implemented yet"))

function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, ::ZigZag, i::Integer)
    state.ξ.θ[i] = rand(rng, (-1.0, 1.0))
    state.old_velocity[i] = 0.0
    return state.ξ.θ[i]
end

function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, ::BouncyParticle, i::Integer)
    magnitude = sqrt(rand(rng, Exponential(2.0)))
    state.ξ.θ[i] = rand(rng, Bool) ? magnitude : -magnitude
    state.old_velocity[i] = 0.0
    return state.ξ.θ[i]
end

draw_boundary_velocity!(::Random.AbstractRNG, ::StickyPDMPState, ::AnyBoomerang, ::Integer) =
    throw(ArgumentError("aggregate sticky Boomerang boundary velocity draws are not implemented yet"))
draw_boundary_velocity!(::Random.AbstractRNG, ::StickyPDMPState, ::PreconditionedDynamics, ::Integer) =
    throw(ArgumentError("aggregate sticky preconditioned boundary velocity draws are not implemented yet"))

function _clock_state_at(state::StickyPDMPState, flow::ContinuousDynamics, τ::Real)
    state_at = copy(state)
    move_forward_time!(state_at, τ, flow)
    return state_at
end

function _clock_active_and_stickable(clock::SummedRateClock, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    return _active_beta_from_free(provider, state.free), _stickable_beta_from_can_stick(provider, can_stick)
end

function _has_inactive_stickable_beta(clock::AbstractAggregateUnstickClock, state::StickyPDMPState, can_stick::BitVector)
    for i in beta_indices(clock.slab_provider)
        if can_stick[i] && !state.free[i]
            return true
        end
    end
    return false
end

"""
    rate(clock, flow, state, τ, can_stick)

Instantaneous aggregate unstick rate at elapsed time `τ` from `state`, summing
over inactive stickable beta coordinates.
"""
function rate(clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    τ < 0 && throw(ArgumentError("τ must be non-negative"))
    state_at = iszero(τ) ? state : _clock_state_at(state, flow, τ)
    active_beta, stickable_beta = _clock_active_and_stickable(clock, state_at, can_stick)
    lograte = aggregate_lograte(
        clock.slab_provider,
        clock.model_prior_odds,
        log(unstick_rate_constant(flow, 1)),
        state_at.ξ.x,
        active_beta,
        stickable_beta,
    )
    return isfinite(lograte) ? exp(lograte) : 0.0
end

"""
    cumulative_hazard(clock, flow, state, t0, t1, can_stick)

Aggregate unstick cumulative hazard over elapsed-time interval `[t0, t1]`.
"""
function cumulative_hazard(clock::SummedRateClock, flow::ContinuousDynamics, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    0 <= t0 <= t1 || throw(ArgumentError("expected 0 <= t0 <= t1, got t0=$t0, t1=$t1"))
    t0 == t1 && return 0.0
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
function sample_label(rng::Random.AbstractRNG, clock::SummedRateClock, ::ContinuousDynamics, state::StickyPDMPState, can_stick::BitVector)
    active_beta, stickable_beta = _clock_active_and_stickable(clock, state, can_stick)
    return sample_unstick_label(rng, clock.slab_provider, clock.model_prior_odds, state.ξ.x, active_beta, stickable_beta)
end

_fallback_clock(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}) = clock.fallback

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
    clock.diagnostics.fallbacks += 1
    return sample_time(rng, clock.fallback, flow, state, horizon, can_stick)
end

function sample_label(rng::Random.AbstractRNG, clock::AbstractAggregateUnstickClock, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    state_at = _clock_state_at(state, flow, τ)
    return sample_label(rng, clock, flow, state_at, can_stick)
end

_log_model_add_odds_with_count(prior::AbstractModelPriorOdds, active::BitVector, j::Integer, ::Integer) =
    log_model_add_odds(prior, active, j)
function _log_model_add_odds_with_count(prior::BetaBernoulliModelPriorOdds, active::BitVector, j::Integer, k::Integer)
    active[j] && throw(ArgumentError("log_model_add_odds is defined for adding an inactive coordinate; coordinate $j is already active"))
    denom = prior.b + prior.n - k - 1
    denom <= 0 && return Inf
    return log(prior.a + k) - log(denom)
end
function _log_model_add_odds_with_count(prior::ExchangeableModelSizePrior, active::BitVector, j::Integer, k::Integer)
    p = length(prior)
    active[j] && throw(ArgumentError("log_model_add_odds is defined for adding an inactive coordinate; coordinate $j is already active"))
    k < p || return -Inf
    return prior.log_omega[k + 2] - prior.log_omega[k + 1] + log(k + 1) - log(p - k)
end

"""
    _prepare_linear_gaussian_cache!(clock, state, can_stick, τ=0)

Populate the reusable linear-clock cache with beta values at elapsed time `τ`,
active/stickable beta masks, and active beta positions. Returns `(cache,
nactive)`.
"""
function _prepare_linear_gaussian_cache!(clock::LinearGaussianAggregateClock, state::StickyPDMPState, can_stick::BitVector, τ::Real=0.0)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    cache = clock.cache
    gaussian_slab!(provider, cache.mean, cache.cov, state.ξ.x)
    nactive = 0
    @inbounds for j in eachindex(indices)
        i = indices[j]
        active = state.free[i]
        cache.active_beta[j] = active
        cache.stickable_beta[j] = can_stick[i]
        cache.beta_values[j] = state.ξ.x[i] + τ * state.ξ.θ[i]
        cache.beta_velocity[j] = state.ξ.θ[i]
        if active
            nactive += 1
            cache.active_positions[nactive] = j
        end
    end
    return cache, nactive
end

"""
    _factor_linear_gaussian_active_block!(cache, nactive)

Factor the active covariance block in-place and overwrite `delta_A` and
`velocity_A` by the corresponding covariance solves.
"""
function _factor_linear_gaussian_active_block!(cache::LinearGaussianAggregateCache, nactive::Integer)
    iszero(nactive) && return cache
    @inbounds for aidx in 1:nactive
        ia = cache.active_positions[aidx]
        cache.delta_A[aidx] = cache.beta_values[ia] - cache.mean[ia]
        cache.velocity_A[aidx] = cache.beta_velocity[ia]
        for bidx in 1:nactive
            ib = cache.active_positions[bidx]
            cache.cov_AA[aidx, bidx] = cache.cov[ia, ib]
        end
    end
    _cholesky_factor_prefix!(cache.cov_AA, nactive)
    _cholesky_solve_with_factor_prefix!(cache.cov_AA, cache.delta_A, nactive)
    _cholesky_solve_with_factor_prefix!(cache.cov_AA, cache.velocity_A, nactive)
    return cache
end

"""
    _linear_gaussian_component_params!(cache, j, nactive)

Return `(a, b, s)` for inactive coordinate `j`, where the conditional boundary
mean along the segment is `a + b*t` and `s` is the conditional standard
deviation.
"""
function _linear_gaussian_component_params!(cache::LinearGaussianAggregateCache, j::Integer, nactive::Integer)
    if iszero(nactive)
        s2 = cache.cov[j, j]
        @assert s2 > 0
        return cache.mean[j], 0.0, sqrt(s2)
    end

    @inbounds for aidx in 1:nactive
        ia = cache.active_positions[aidx]
        value = cache.cov[j, ia]
        cache.cov_jA[aidx] = value
        cache.solved_cross[aidx] = value
    end
    _cholesky_solve_with_factor_prefix!(cache.cov_AA, cache.solved_cross, nactive)

    cross_delta = 0.0
    cross_velocity = 0.0
    cross_solve = 0.0
    @inbounds for aidx in 1:nactive
        c = cache.cov_jA[aidx]
        cross_delta += c * cache.delta_A[aidx]
        cross_velocity += c * cache.velocity_A[aidx]
        cross_solve += c * cache.solved_cross[aidx]
    end
    s2 = cache.cov[j, j] - cross_solve
    @assert s2 > 0
    return cache.mean[j] + cross_delta, cross_velocity, sqrt(s2)
end

function _linear_gaussian_has_component(clock::LinearGaussianAggregateClock, state::StickyPDMPState, can_stick::BitVector)
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    @inbounds for j in eachindex(cache.active_beta)
        cache.stickable_beta[j] && !cache.active_beta[j] &&
            isfinite(_log_model_add_odds_with_count(clock.model_prior_odds, cache.active_beta, j, nactive)) && return true
    end
    return false
end

function _linear_gaussian_available_hazard(clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    _factor_linear_gaussian_active_block!(cache, nactive)
    Cv = unstick_rate_constant(flow, 1)
    total = 0.0
    @inbounds for j in eachindex(cache.active_beta)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior_odds, cache.active_beta, j, nactive)
            isfinite(logρ) || continue
            a, b, s = _linear_gaussian_component_params!(cache, j, nactive)
            total += Cv * _linear_gaussian_component_total_hazard(a, b, s, logρ)
        end
    end
    return total
end

"""
    _linear_gaussian_component_hazard(a, b, s, log_weight, T)

Closed-form integral from `0` to `T` of a weighted Gaussian boundary density
with linear conditional mean `a + b*t`.
"""
function _linear_gaussian_component_hazard(a::Real, b::Real, s::Real, log_weight::Real, T::Real)
    T <= 0 && return 0.0
    w = exp(log_weight)
    if iszero(b)
        z = a / s
        return w * exp(-0.5 * abs2(z)) / sqrt(2π) / s * T
    end
    z0 = a / s
    zT = (a + b * T) / s
    return w / abs(b) * abs(Distributions.normcdf(zT) - Distributions.normcdf(z0))
end

"""
    _linear_gaussian_component_total_hazard(a, b, s, log_weight)

Closed-form total available future hazard for a single linear Gaussian boundary
component.
"""
function _linear_gaussian_component_total_hazard(a::Real, b::Real, s::Real, log_weight::Real)
    w = exp(log_weight)
    if iszero(b)
        return iszero(w) ? 0.0 : Inf
    end
    z0 = a / s
    if b > 0
        return w / b * (1 - Distributions.normcdf(z0))
    else
        return w / abs(b) * Distributions.normcdf(z0)
    end
end

function rate(clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    _factor_linear_gaussian_active_block!(cache, nactive)
    Cv = unstick_rate_constant(flow, 1)
    total = 0.0
    @inbounds for j in eachindex(cache.active_beta)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior_odds, cache.active_beta, j, nactive)
            isfinite(logρ) || continue
            a, b, s = _linear_gaussian_component_params!(cache, j, nactive)
            z = (a + b * τ) / s
            total += Cv * exp(logρ) * exp(-0.5 * abs2(z)) / sqrt(2π) / s
        end
    end
    return total
end

function cumulative_hazard(clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    t0 == t1 && return 0.0
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    _factor_linear_gaussian_active_block!(cache, nactive)
    Cv = unstick_rate_constant(flow, 1)
    total = 0.0
    @inbounds for j in eachindex(cache.active_beta)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior_odds, cache.active_beta, j, nactive)
            isfinite(logρ) || continue
            a, b, s = _linear_gaussian_component_params!(cache, j, nactive)
            total += Cv * (
                _linear_gaussian_component_hazard(a, b, s, logρ, Float64(t1)) -
                _linear_gaussian_component_hazard(a, b, s, logρ, Float64(t0))
            )
        end
    end
    return max(0.0, total)
end

function sample_time(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    _linear_gaussian_has_component(clock, state, can_stick) || return Inf
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = cumulative_hazard(clock, flow, state, 0.0, Float64(horizon), can_stick)
        H < threshold && return Inf
        f = τ -> cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
        return Roots.find_zero(f, (0.0, Float64(horizon)), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
    end

    total_available = _linear_gaussian_available_hazard(clock, flow, state, can_stick)
    total_available < threshold && return Inf

    lo = 0.0
    hi = 1.0
    H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
    iterations = 0
    while H_hi < threshold && iterations < 60
        lo = hi
        hi *= 2.0
        H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
        iterations += 1
    end
    H_hi < threshold && return Inf
    f = τ -> cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
    return Roots.find_zero(f, (lo, hi), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
end

function _linear_gaussian_label_weights!(clock::LinearGaussianAggregateClock, state::StickyPDMPState, can_stick::BitVector, τ::Real)
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick, τ)
    _factor_linear_gaussian_active_block!(cache, nactive)
    indices = beta_indices(clock.slab_provider)
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior_odds, cache.active_beta, j, nactive)
            if isfinite(logρ)
                a, _, s = _linear_gaussian_component_params!(cache, j, nactive)
                z = a / s
                logw = logρ - 0.5 * (log(2π) + 2log(s) + abs2(z))
            else
                logw = -Inf
            end
            cache.log_weights[j] = logw
            max_logw = max(max_logw, logw)
        else
            cache.log_weights[j] = -Inf
        end
    end
    return cache, max_logw
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock, ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, max_logw = _linear_gaussian_label_weights!(clock, state, can_stick, 0.0)
    indices = beta_indices(clock.slab_provider)
    isfinite(max_logw) || throw(ArgumentError("cannot sample an unstick label because all inactive stickable rates are zero"))
    total = 0.0
    @inbounds for j in eachindex(indices)
        total += isfinite(cache.log_weights[j]) ? exp(cache.log_weights[j] - max_logw) : 0.0
    end

    draw = rand(rng) * total
    last_candidate = 0
    @inbounds for j in eachindex(indices)
        if isfinite(cache.log_weights[j])
            last_candidate = j
            draw -= exp(cache.log_weights[j] - max_logw)
            draw <= 0 && return indices[j]
        end
    end
    return indices[last_candidate]
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    cache, max_logw = _linear_gaussian_label_weights!(clock, state, can_stick, τ)
    indices = beta_indices(clock.slab_provider)
    isfinite(max_logw) || throw(ArgumentError("cannot sample an unstick label because all inactive stickable rates are zero"))
    total = 0.0
    @inbounds for j in eachindex(indices)
        total += isfinite(cache.log_weights[j]) ? exp(cache.log_weights[j] - max_logw) : 0.0
    end

    draw = rand(rng) * total
    last_candidate = 0
    @inbounds for j in eachindex(indices)
        if isfinite(cache.log_weights[j])
            last_candidate = j
            draw -= exp(cache.log_weights[j] - max_logw)
            draw <= 0 && return indices[j]
        end
    end
    return indices[last_candidate]
end


"""
    unfreeze_time(alg::StickyLoopState, state::StickyPDMPState, i::Integer)

Simulate the time a stuck/ frozen particle takes to unfreeze/ unstick
"""
function unfreeze_time(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, i::Integer)
    validate_state(state, nothing, "in unfreeze_time")
    κ = get_κ(alg, i, state.ξ.x, state.free, state.ξ.θ)

    if κ isa Distribution

        retval = rand(rng, κ)
        if isnegative(retval)# || isinf(retval)
            @show κ, i, state.ξ.x, state.free, state.ξ.θ
            throw(ArgumentError("κ must be non-negative and finite!"))
        end
        # @show κ, i, state.ξ.x, state.free
        return retval
    else
        θf = state.old_velocity[i]
        if isnegative(κ)
            @show κ, i, state.ξ.x, state.free
            throw(ArgumentError("κ must be non-negative!"))
        end

        # return -log(rand()) / (κ * abs(θf)) # old approach
        return rand(rng, Exponential(inv(κ * abs(θf))))
    end
end

# κ = 1.23
# θf = 20.4
# e1 = [-log(rand()) / (κ * abs(θf)) for _ in 1:10000]
# D2 = Exponential((κ * abs(θf)))
# qprobs = .01:.01:.99
# q1 = quantile(e1, qprobs)
# q2 = quantile(D2, qprobs)
# f, ax, _ = scatter(q1, q2)
# ablines!(ax, 0, 1, color = :grey, linestyle = :dash)
# f

# f0(κ, θf) = -log(rand()) / (κ * abs(θf))
# f1(κ, θf) = rand(Exponential(κ * abs(θf)))
# @benchmark f0($κ,$θf)
# @benchmark f1($κ,$θf)

"""
    τ = freezing_time(ξ::SkeletonPoint, flow::ContinuousDynamics, i::Integer)

computes the hitting time of the particle to hit 0 given the position `ξ.x[i]` and the velocity `ξ.θ[i]`.
"""
function freezing_time(ξ::SkeletonPoint, ::Union{BouncyParticle,ZigZag}, i::Integer)
    x = ξ.x[i]
    θ = ξ.θ[i]
    if θ * x >= 0
        return Inf
    else
        return -x / θ
    end
end

get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy,<:AbstractVector}, i, args...) = sticky_strat.κ[i]
get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy,<:Function}, i, args...) = sticky_strat.κ(i, args...)
get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy,<:RateFunction}, i, args...) = sticky_strat.κ(i, args...)

get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, i, args...) = sticky_state.κ[i]
get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy,<:Function}, i, args...) = sticky_state.κ(i, args...)
get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy,<:RateFunction}, i, args...) = sticky_state.κ(i, args...)

function update_all_stick_times!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)

    t = state.t[]
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        else # stuck/ frozen
            _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
        end
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    end
end

function _update_aggregate_unstick_time!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, horizon::Real=Inf)
    t = state.t[]
    τ = sample_time(rng, alg.clock, flow, state, horizon, alg.can_stick)
    alg.aggregate_unstick_time = t + τ
    return alg.aggregate_unstick_time
end

function update_all_stick_times!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    empty!(alg.sticky_pq)
    fill!(alg.sticky_times, Inf)
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        end
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    end
    _, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
    _update_aggregate_unstick_time!(rng, alg, state, flow, max(0.0, t_freeze - t))
    return nothing
end

function update_all_freeze_times!(alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
            isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
        end
    end
end

function update_all_freeze_times!(alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    empty!(alg.sticky_pq)
    fill!(alg.sticky_times, Inf)
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
            isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
        end
    end
    return nothing
end

function update_all_unfreeze_times!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    for i in alg.stickable_indices
        if !state.free[i]
            _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
            isinf(alg.sticky_times[i]) && error("sticky_times[$i] is Inf but it's stuck with x[i]=$(state.ξ.x[i])")
        end
    end
end

function update_all_unfreeze_times!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    _, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
    _update_aggregate_unstick_time!(rng, alg, state, flow, max(0.0, t_freeze - t))
    isinf(alg.aggregate_unstick_time) && _has_inactive_stickable_beta(alg.clock, state, alg.can_stick) &&
        isinf(t_freeze) && error("aggregate_unstick_time is Inf but at least one stickable coordinate is stuck")
    return nothing
end

_sticky_coordinate_index(meta::CoordinateMeta) = meta.i
_sticky_coordinate_index(i::Integer) = Int(i)

function _update_sticky_time_at_index!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, i::Int)
    if !alg.can_stick[i]
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        return nothing
    end
    t = state.t[]
    if state.free[i]
        _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    else
        _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN after unfreezing")
    end
    return nothing
end

function _update_sticky_time_at_index!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, i::Int)
    if !alg.can_stick[i]
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        return nothing
    end
    t = state.t[]
    if state.free[i]
        _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    else
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        _, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
        _update_aggregate_unstick_time!(rng, alg, state, flow, max(0.0, t_freeze - t))
    end
    return nothing
end

function _update_sticky_schedule_after_reflect!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, meta)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_reflect!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, meta)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_reflect!(rng::Random.AbstractRNG, alg::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, state::StickyPDMPState, flow::ZigZag, meta::Union{CoordinateMeta,Integer})
    _update_sticky_time_at_index!(rng, alg, state, flow, _sticky_coordinate_index(meta))
    return nothing
end

function _update_sticky_schedule_after_refresh!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_refresh!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_refresh!(::Random.AbstractRNG, ::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, ::StickyPDMPState, ::ZigZag)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(::Random.AbstractRNG, ::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, ::StickyPDMPState, ::ZigZag)
    return nothing
end

function stick_or_unstick!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::ContinuousDynamics, alg::StickyLoopState, i::Int)

    t = state.t[]
    ξ = state.ξ
    sticky_times = alg.sticky_times
    θf = state.old_velocity
    tol = sqrt(eps(eltype(ξ.x))) # tolerance for floating point errors in move_forward_time!, could also depend on the flow?
    if state.free[i] # if free -> stuck

        # deterministic process should have move x[i] to exactly zero, but perhaps this needs a tolerance
        abs(ξ.x[i]) < tol || error("freezing but not frozen: x[i] = $(ξ.x[i]) !≈ 0 at $(sticky_times[i]) with tol = $(tol)")

        θf[i] = ξ.θ[i] # store speed
        ξ.θ[i] = 0.0 # freeze speed
        ξ.x[i] = 0.0 # freeze position, set to 0 exactly to avoid floating point errors
        state.free[i] = false # mark as stuck

        # κᵢ = get_κ(alg, i, state.ξ.x)
        # sticky_times[i] = t - log(rand()) / (κᵢ * abs(θf[i])) # sticky time

        if alg.κ isa AbstractVector
            _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
            @assert !isnan(alg.sticky_times[i]) "sticky_times[$i] is NaN after unfreezing"
        else
            #= TODO: not sure about this design... there are a few cases:

                1. prior inclusion probabilities are independent and fixed: γᵢ ~ Bernoulli(pᵢ)
                    -> κ isa Vector
                2. prior inclusion probabilities depend on whether other parameters "stick": γᵢ ~ BetaBernoulli(n, a, b)
                    -> κ isa Function
                3. prior inclusion probabilities depend on hyperparameters: γᵢ ~ Bernoulli(θ); θ ~ Beta(1, 1))
                    -> κ isa Function

                alternatively for case 2:

                - all unfreezing times are exponentials
                - sample the first unfreezing time from the joint distribution of independent not identically distributed Exponentials.
                - tᶠ is min(first unfreezing time, first freezing time).
                We must (re)compute all freezing times since they are deterministic.

                case 3 still needs to be studied. No idea if the current approach even works.

            =#
            update_all_stick_times!(rng, alg, state, flow)
            # update_all_unfreeze_times!(alg, state, flow)
        end
        # tfrez[i] = t - log(rand()) # option 2 # TODO: this is independent of the prior!?

        # TODO: maybe we need to update other freezing times here as well


    else # stuck -> not stuck

        # deterministic process should have move x[i] to exactly zero and left it there
        # velocity should be at exactly zero at this point.
        (abs(ξ.x[i]) < tol && iszero(ξ.θ[i])) || error("unfreezing but not frozen: x[i] = $(ξ.x[i]) ≉ 0 or θ[i] = $(ξ.θ[i]) ≉ 0 at $(sticky_times[i]) with tol = $(tol)")# isfrozen

        ξ.θ[i] = θf[i] # restore speed
        θf[i] = zero(eltype(θf[i])) # perhaps not necessary?
        state.free[i] = true # mark as not stuck

        # update_all_stick_times!(alg, state, flow)
        _set_sticky_time!(alg, i, t + freezing_time(ξ, flow, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(sticky_times[i])) after freezing (θ[i] = $(ξ.θ[i]))")
        if !(alg.κ isa AbstractVector)
            update_all_stick_times!(rng, alg, state, flow)
            # update_all_unfreeze_times!(alg, state, flow)
        end
        # TODO: maybe we need to update other freezing times here as well

    end
    validate_state(state, flow, "after stick_or_unstick! at index $i")
end

function stick_or_unstick!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::ContinuousDynamics, alg::AggregateStickyLoopState, i::Int)
    t = state.t[]
    ξ = state.ξ
    tol = sqrt(eps(eltype(ξ.x)))
    alg.can_stick[i] || throw(ArgumentError("coordinate $i is not stickable"))

    if state.free[i]
        abs(ξ.x[i]) < tol || error("freezing but not frozen: x[$i] = $(ξ.x[i]) !≈ 0 at $(t) with tol = $(tol)")
        state.old_velocity[i] = ξ.θ[i]
        ξ.θ[i] = 0.0
        ξ.x[i] = 0.0
        state.free[i] = false
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        update_all_stick_times!(rng, alg, state, flow)
    else
        (abs(ξ.x[i]) < tol && iszero(ξ.θ[i])) ||
            error("unfreezing but not frozen: x[$i] = $(ξ.x[i]) ≉ 0 or θ[$i] = $(ξ.θ[i]) ≉ 0 at $(t) with tol = $(tol)")
        draw_boundary_velocity!(rng, state, flow, i)
        state.free[i] = true
        _set_sticky_time!(alg, i, t + freezing_time(ξ, flow, i))
        update_all_stick_times!(rng, alg, state, flow)
    end
    validate_state(state, flow, "after aggregate stick_or_unstick! at index $i")
end

function _bounded_inner_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
        inner_alg_state::GridAdaptiveState, state::StickyPDMPState, cache, stats::AbstractStatisticCounter, max_horizon::Float64)
    return next_event_time(rng, model, flow, inner_alg_state, state, cache, stats, max_horizon, false, :horizon_hit)
end

function _bounded_inner_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
        inner_alg_state::PoissonTimeStrategy, state::StickyPDMPState, cache, stats::AbstractStatisticCounter, max_horizon::Float64)
    τ, event_type, meta = next_event_time(rng, model, flow, inner_alg_state, state, cache, stats)
    if isfinite(max_horizon) && τ > max_horizon
        return max_horizon, :horizon_hit, EmptyMeta()
    end
    return τ, event_type, meta
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::AggregateStickyLoopState, state::StickyPDMPState, cache, stats::AbstractStatisticCounter)
    t = state.t[]
    inner_alg_state = alg.inner_alg_state

    i_freeze, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
    t_unstick = alg.aggregate_unstick_time
    t_sticky = min(t_freeze, t_unstick)
    τ_sticky = max(0.0, t_sticky - t)

    if any(state.free)
        τ_refresh = rand_refresh_time(rng, flow)
        max_horizon = min(τ_sticky, τ_refresh)
        _inc_counter_sticky_inner_searches(stats)
        τ_inner, event_type, meta = _bounded_inner_event_time(rng, model, flow, inner_alg_state, state, cache, stats, max_horizon)

        if τ_sticky <= τ_inner && τ_sticky <= τ_refresh
            _inc_counter_sticky_inner_wasted_by_sticky(stats)
            if t_unstick <= t_freeze
                i_unstick = sample_label(rng, alg.clock, flow, state, τ_sticky, alg.can_stick)
                return τ_sticky, :sticky, CoordinateMeta(i_unstick)
            end
            return τ_sticky, :sticky, CoordinateMeta(i_freeze)
        elseif τ_refresh <= τ_inner
            _inc_counter_sticky_inner_wasted_by_refresh(stats)
            return τ_refresh, :refresh, GradientMeta(alg.empty_∇ϕx)
        else
            _inc_counter_sticky_inner_wins(stats)
            return τ_inner, event_type, meta
        end
    else
        _inc_counter_sticky_all_frozen_events(stats)
        isfinite(t_unstick) || return Inf, :sticky, CoordinateMeta(0)
        i_unstick = sample_label(rng, alg.clock, flow, state, t_unstick - t, alg.can_stick)
        return t_unstick - t, :sticky, CoordinateMeta(i_unstick)
    end
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::StickyLoopState, state::StickyPDMPState, cache, stats::AbstractStatisticCounter)

    t = state.t[]
    inner_alg_state = alg.inner_alg_state

    if isempty(alg.sticky_pq)
        i = 0
        tᶠ = Inf
    else
        i, tᶠ = first(alg.sticky_pq)
    end
    # non_sticky_state = PDMPState(state.t, state.ξ) # or substate, but that messes with dimensionality


    if any(state.free)
        # only propose reflection/ refreshment times when at least one parameter is free
        # max_horizon = tᶠ - t
        # if iszero(max_horizon)
        #     # sticky event happens now
        #     return 0.0, :sticky, i
        # end

        # TODO: this could be written in a cleaner way!
        # τ, event_type, meta = if inner_alg_state isa GridAdaptiveState
        #     # Pass max_horizon to GridThinning

        #     if iszero(max_horizon)
        #         @show max_horizon, state, sticky_time, tᶠ, t
        #         max_horizon = Inf
        #     end
        #     next_event_time(model, flow, inner_alg_state, state, cache, stats, 5 * max_horizon)
        # else
        #     next_event_time(model, flow, inner_alg_state, state, cache, stats)
        # end
        # @show τ

        # Sample refresh time independently at the sticky level
        τ_refresh = rand_refresh_time(rng, flow)
        tʳ = t + τ_refresh

        _inc_counter_sticky_inner_searches(stats)
        τ, event_type, meta = next_event_time(rng, model, flow, inner_alg_state, state, cache, stats, Inf, false)

        t′ = t + τ

        if tᶠ < t′ && tᶠ < tʳ #  sticky event happens first
            _inc_counter_sticky_inner_wasted_by_sticky(stats)
            Δt = tᶠ - t
            return Δt, :sticky, CoordinateMeta(i)
        elseif tʳ < t′
            _inc_counter_sticky_inner_wasted_by_refresh(stats)
            return τ_refresh, :refresh, GradientMeta(alg.empty_∇ϕx)
        else
            _inc_counter_sticky_inner_wins(stats)
            return τ, event_type, meta
        end

        # original
        # if tᶠ < t′
        #     Δt = tᶠ - t
        #     # iszero(Δt) && @warn "Sticky event time equals current time t = $t. This may lead to infinite loops."
        #     return Δt, :sticky, i
        # else
        #     return τ, event_type, meta
        # end
    else
        _inc_counter_sticky_all_frozen_events(stats)
        Δt = tᶠ - t
        return Δt, :sticky, CoordinateMeta(i)
    end
end

_reset_inner_grid!(alg::StickyLoopState) = _reset_inner_grid!(alg.inner_alg_state)
_invalidate_cached_gradient!(alg::StickyLoopState) = _invalidate_cached_gradient!(alg.inner_alg_state)

function _maybe_activate_constant_bound!(alg::StickyLoopState, stats::AbstractStatisticCounter)
    _maybe_activate_constant_bound!(alg.inner_alg_state, stats)
end
