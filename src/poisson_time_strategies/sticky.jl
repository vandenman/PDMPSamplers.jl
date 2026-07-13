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
    if !clock.allow_slow_fallback
        capability = certified_residual_capability(clock.slab_provider, flow)
        if capability isa NoCertifiedResidualCapability
            throw(ArgumentError(
                "$(nameof(typeof(clock))) cannot build a certified residual envelope for provider " *
                "$(nameof(typeof(clock.slab_provider))) and flow $(nameof(typeof(flow))). " *
                "Provide a structured Gaussian scale-mixture provider with certified interval/trajectory bounds, " *
                "or construct the clock with allow_slow_fallback=true to use the exact SummedRateClock fallback."
            ))
        end
        throw(ArgumentError(
            "$(nameof(typeof(clock))) found a residual-envelope capability for provider " *
            "$(nameof(typeof(clock.slab_provider))), but certified residual sampling is not implemented for this capability yet."
        ))
    end
    clock.diagnostics.fallbacks += 1
    return sample_time(rng, clock.fallback, flow, state, horizon, can_stick)
end

certified_residual_capability(::GlobalLogscaleExchangeableGaussianSlab, ::Union{ZigZag,BouncyParticle}) =
    FloatingPointScalarResidualCapability()

struct _ChebyshevResidualCell
    lo::Float64
    hi::Float64
    R::Float64
    residual_area::Float64
end

struct _ChebyshevResidualEnvelope
    segment::ScalarLogscaleGaussianLineSegment
    coeffs::Vector{Float64}
    power_coeffs::Vector{Float64}
    cells::Vector{_ChebyshevResidualCell}
    edges::Vector{Float64}
    residual_prefix::Vector{Float64}
    Hbar_horizon::Float64
end

_fp_pad(x::Real; rel::Float64=1024eps(Float64), abs_pad::Float64=1024eps(Float64)) =
    Float64(rel * max(1.0, abs(Float64(x))) + abs_pad)

function _scalar_logscale_gaussian_line_rate(seg::ScalarLogscaleGaussianLineSegment, t::Real)
    ell = seg.ell0 + seg.r * Float64(t)
    s = seg.s0 * exp(ell)
    z = (seg.a + seg.b * Float64(t)) / s
    return exp(seg.log_total_weight - log(seg.s0) - ell - 0.5 * abs2(z) - 0.5 * log(2π))
end

function _scalar_logscale_gaussian_line_upper(seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    m0 = seg.a + seg.b * lo_f
    m1 = seg.a + seg.b * hi_f
    m_lo = min(m0, m1)
    m_hi = max(m0, m1)
    e0 = seg.ell0 + seg.r * lo_f
    e1 = seg.ell0 + seg.r * hi_f
    ell_lo = min(e0, e1)
    ell_hi = max(e0, e1)
    logs_lo = log(seg.s0) + ell_lo
    logs_hi = log(seg.s0) + ell_hi
    s_lo = exp(logs_lo)
    s_hi = exp(logs_hi)
    z1 = m_lo / s_lo
    z2 = m_lo / s_hi
    z3 = m_hi / s_lo
    z4 = m_hi / s_hi
    z_lo = min(z1, z2, z3, z4)
    z_hi = max(z1, z2, z3, z4)
    z2_lo = z_lo <= 0 <= z_hi ? 0.0 : min(abs2(z_lo), abs2(z_hi))
    logλ_hi = seg.log_total_weight - logs_lo - 0.5 * z2_lo - 0.5 * log(2π)
    upper = exp(logλ_hi)
    return upper + _fp_pad(upper)
end

function _chebyshev_fit_coeffs(f, T::Real, degree::Integer)
    n = max(4 * Int(degree) + 1, 65)
    coeffs = zeros(Float64, Int(degree) + 1)
    for k in 0:n-1
        θ = π * (k + 0.5) / n
        x = cos(θ)
        t = 0.5 * Float64(T) * (x + 1)
        y = Float64(f(t))
        coeffs[1] += y
        for j in 1:Int(degree)
            coeffs[j + 1] += y * cos(j * θ)
        end
    end
    coeffs[1] /= n
    for j in 2:length(coeffs)
        coeffs[j] *= 2 / n
    end
    return coeffs
end

function _chebyshev_eval(coeffs::AbstractVector{Float64}, t::Real, T::Real)
    x = 2 * Float64(t) / Float64(T) - 1
    b1 = 0.0
    b2 = 0.0
    for j in length(coeffs):-1:2
        b0 = 2x * b1 - b2 + coeffs[j]
        b2 = b1
        b1 = b0
    end
    return x * b1 - b2 + coeffs[1]
end

function _chebyshev_to_power(coeffs::AbstractVector{Float64})
    K = length(coeffs) - 1
    out = zeros(Float64, K + 1)
    Tprev = zeros(Float64, K + 1)
    Tcurr = zeros(Float64, K + 1)
    Tprev[1] = 1.0
    out .+= coeffs[1] .* Tprev
    K == 0 && return out
    Tcurr[2] = 1.0
    out .+= coeffs[2] .* Tcurr
    for n in 2:K
        Tnext = zeros(Float64, K + 1)
        for i in 1:K
            Tnext[i + 1] += 2 * Tcurr[i]
        end
        Tnext .-= Tprev
        out .+= coeffs[n + 1] .* Tnext
        Tprev, Tcurr = Tcurr, Tnext
    end
    return out
end

function _poly_integral_x(power_coeffs::AbstractVector{Float64}, x::Real)
    xf = Float64(x)
    total = 0.0
    xpow = xf
    for i in eachindex(power_coeffs)
        total += power_coeffs[i] * xpow / i
        xpow *= xf
    end
    return total
end

function _chebyshev_primitive(power_coeffs::AbstractVector{Float64}, t::Real, T::Real)
    x = 2 * Float64(t) / Float64(T) - 1
    return 0.5 * Float64(T) * (_poly_integral_x(power_coeffs, x) - _poly_integral_x(power_coeffs, -1.0))
end

_iadd(a, b) = (a[1] + b[1], a[2] + b[2])
_isub(a, b) = (a[1] - b[2], a[2] - b[1])
function _imul(a, b)
    vals = (a[1] * b[1], a[1] * b[2], a[2] * b[1], a[2] * b[2])
    return (minimum(vals), maximum(vals))
end
_iscale(c::Real, a) = c >= 0 ? (Float64(c) * a[1], Float64(c) * a[2]) : (Float64(c) * a[2], Float64(c) * a[1])

function _chebyshev_interval(coeffs::AbstractVector{Float64}, lo::Real, hi::Real, T::Real)
    x = (2 * Float64(lo) / Float64(T) - 1, 2 * Float64(hi) / Float64(T) - 1)
    total = (coeffs[1], coeffs[1])
    length(coeffs) == 1 && return total
    Tprev = (1.0, 1.0)
    Tcurr = x
    total = _iadd(total, _iscale(coeffs[2], Tcurr))
    for n in 2:length(coeffs)-1
        Tnext = _isub(_iscale(2.0, _imul(x, Tcurr)), Tprev)
        total = _iadd(total, _iscale(coeffs[n + 1], Tnext))
        Tprev, Tcurr = Tcurr, Tnext
    end
    pad = _fp_pad(max(abs(total[1]), abs(total[2])))
    return (total[1] - pad, total[2] + pad)
end

function _certify_scalar_cell(seg::ScalarLogscaleGaussianLineSegment, coeffs::Vector{Float64}, lo::Float64, hi::Float64, T::Float64)
    L_hi = _scalar_logscale_gaussian_line_upper(seg, lo, hi)
    P_lo, _ = _chebyshev_interval(coeffs, lo, hi, T)
    R = max(0.0, L_hi - P_lo)
    R += _fp_pad(max(L_hi, abs(P_lo), R))
    return _ChebyshevResidualCell(lo, hi, R, R * (hi - lo))
end

function _build_scalar_residual_envelope(clock::ChebyshevResidualAggregateClock, seg::ScalarLogscaleGaussianLineSegment)
    T = seg.horizon
    coeffs = _chebyshev_fit_coeffs(t -> _scalar_logscale_gaussian_line_rate(seg, t), T, clock.order)
    power = _chebyshev_to_power(coeffs)
    initial_cells = min(max(4, ceil(Int, sqrt(clock.max_cells))), clock.max_cells)
    heap = _ChebyshevResidualCell[]
    for i in 0:initial_cells-1
        lo = T * i / initial_cells
        hi = T * (i + 1) / initial_cells
        push!(heap, _certify_scalar_cell(seg, coeffs, lo, hi, T))
    end
    grid_n = max(129, 8 * clock.order + 1)
    target_area = T / grid_n * sum(_scalar_logscale_gaussian_line_rate(seg, T * (i + 0.5) / grid_n) for i in 0:grid_n-1)
    target_residual = clock.residual_budget * max(target_area, eps(Float64))
    total_residual = sum(c.residual_area for c in heap)
    while total_residual > target_residual && length(heap) < clock.max_cells
        idx = 1
        best_area = heap[1].residual_area
        for i in 2:length(heap)
            if heap[i].residual_area > best_area
                idx = i
                best_area = heap[i].residual_area
            end
        end
        parent = heap[idx]
        mid = 0.5 * (parent.lo + parent.hi)
        left = _certify_scalar_cell(seg, coeffs, parent.lo, mid, T)
        right = _certify_scalar_cell(seg, coeffs, mid, parent.hi, T)
        total_residual += left.residual_area + right.residual_area - parent.residual_area
        heap[idx] = left
        push!(heap, right)
    end
    sort!(heap, by = c -> c.lo)
    edges = [c.lo for c in heap]
    push!(edges, heap[end].hi)
    prefix = zeros(Float64, length(heap) + 1)
    for i in eachindex(heap)
        prefix[i + 1] = prefix[i] + heap[i].residual_area
    end
    Hbar_T = _chebyshev_primitive(power, T, T) + prefix[end]
    return _ChebyshevResidualEnvelope(seg, coeffs, power, heap, edges, prefix, max(0.0, Hbar_T))
end

function _residual_cell_index(env::_ChebyshevResidualEnvelope, t::Real)
    tf = Float64(t)
    tf <= 0 && return 1
    tf >= env.segment.horizon && return length(env.cells)
    return clamp(searchsortedlast(env.edges, tf), 1, length(env.cells))
end

function _residual_primitive(env::_ChebyshevResidualEnvelope, t::Real)
    tf = clamp(Float64(t), 0.0, env.segment.horizon)
    idx = _residual_cell_index(env, tf)
    return env.residual_prefix[idx] + env.cells[idx].R * (tf - env.cells[idx].lo)
end

function _envelope_hazard(env::_ChebyshevResidualEnvelope, t::Real)
    return _chebyshev_primitive(env.power_coeffs, t, env.segment.horizon) + _residual_primitive(env, t)
end

function _envelope_rate(env::_ChebyshevResidualEnvelope, t::Real)
    idx = _residual_cell_index(env, t)
    return _chebyshev_eval(env.coeffs, t, env.segment.horizon) + env.cells[idx].R
end

function rate(clock::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    τ < 0 && throw(ArgumentError("τ must be non-negative"))
    seg = scalar_logscale_gaussian_line_segment(clock.slab_provider, clock.model_prior_odds, flow, state, can_stick, max(Float64(τ), eps(Float64)))
    return _scalar_logscale_gaussian_line_rate(seg, τ)
end

function sample_time(rng::Random.AbstractRNG, clock::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
                     flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    if !isfinite(horizon)
        clock.allow_slow_fallback && return _sample_time_exact_fallback(rng, clock, flow, state, horizon, can_stick)
        throw(ArgumentError("ChebyshevResidualAggregateClock for GlobalLogscaleExchangeableGaussianSlab requires a finite horizon; rolling finite windows are future work"))
    end
    horizon <= 0 && return Inf
    seg = scalar_logscale_gaussian_line_segment(clock.slab_provider, clock.model_prior_odds, flow, state, can_stick, Float64(horizon))
    env = _build_scalar_residual_envelope(clock, seg)
    d = clock.diagnostics
    d.last_cells = length(env.cells)
    d.residual_area += env.residual_prefix[end]
    d.envelope_hazard += env.Hbar_horizon
    d.true_hazard += max(0.0, QuadGK.quadgk(t -> _scalar_logscale_gaussian_line_rate(seg, t), 0.0, Float64(horizon); rtol=clock.fallback.rtol, atol=clock.fallback.atol)[1])
    d.max_residual = max(d.max_residual, maximum(c.R for c in env.cells))
    d.min_envelope = min(d.min_envelope, minimum(_envelope_rate(env, 0.5 * (c.lo + c.hi)) for c in env.cells))
    d.rate_evaluations += max(4 * clock.order + 1, 65)

    t = 0.0
    while t < horizon
        target = _envelope_hazard(env, t) + rand(rng, Exponential())
        if target > env.Hbar_horizon
            return Inf
        end
        f = τ -> _envelope_hazard(env, τ) - target
        T = Roots.find_zero(f, (t, Float64(horizon)), Roots.Bisection(); atol=clock.fallback.atol, rtol=clock.fallback.rtol)
        d.proposals += 1
        λ = _scalar_logscale_gaussian_line_rate(seg, T)
        λbar = _envelope_rate(env, T)
        λbar <= 0 && throw(ArgumentError("certified residual envelope produced a non-positive proposal rate"))
        ratio = λ / λbar
        d.max_envelope_ratio = max(d.max_envelope_ratio, ratio)
        if rand(rng) <= min(1.0, ratio)
            d.accepted += 1
            return T
        end
        d.rejected += 1
        t = T
    end
    return Inf
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

_exchangeable_linear_mean(provider::ExchangeableGaussianSlab) = provider.mean
_exchangeable_linear_mean(::ZeroMeanExchangeableGaussianSlab) = 0.0

function _exchangeable_linear_params(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    active = clock.cache.active_beta
    stickable = clock.cache.stickable_beta
    μ = _exchangeable_linear_mean(provider)
    k = 0
    nU = 0
    sum_centered = 0.0
    sum_velocity = 0.0
    @inbounds for j in eachindex(indices)
        i = indices[j]
        is_active = state.free[i]
        active[j] = is_active
        stickable[j] = can_stick[i]
        if is_active
            k += 1
            sum_centered += state.ξ.x[i] - μ
            sum_velocity += state.ξ.θ[i]
        elseif can_stick[i]
            nU += 1
        end
    end
    denom = provider.u + k * provider.v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    c = provider.v / denom
    a = μ + c * sum_centered
    b = c * sum_velocity
    s2 = provider.u * (provider.u + (k + 1) * provider.v) / denom
    s2 > 0 || throw(ArgumentError("conditional variance must be positive, got $s2"))
    return active, stickable, k, nU, a, b, sqrt(s2)
end

function _exchangeable_log_weight_sum(prior::AbstractModelPriorOdds, active::BitVector, stickable::BitVector, k::Integer)
    max_logw = -Inf
    @inbounds for j in eachindex(active)
        if stickable[j] && !active[j]
            max_logw = max(max_logw, _log_model_add_odds_with_count(prior, active, j, k))
        end
    end
    isfinite(max_logw) || return -Inf
    total = 0.0
    @inbounds for j in eachindex(active)
        if stickable[j] && !active[j]
            logw = _log_model_add_odds_with_count(prior, active, j, k)
            total += isfinite(logw) ? exp(logw - max_logw) : 0.0
        end
    end
    return max_logw + log(total)
end

function _exchangeable_log_weight_sum(prior::Union{ExchangeableModelSizePrior,BetaBernoulliModelPriorOdds}, active::BitVector, stickable::BitVector, k::Integer)
    nU = count(j -> stickable[j] && !active[j], eachindex(active))
    iszero(nU) && return -Inf
    logρ = _log_model_add_odds_with_count(prior, active, findfirst(j -> stickable[j] && !active[j], eachindex(active)), k)
    isfinite(logρ) || return -Inf
    return log(nU) + logρ
end

function _exchangeable_log_total_weight(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, active::BitVector, stickable::BitVector, k::Integer)
    log_sum = _exchangeable_log_weight_sum(clock.model_prior_odds, active, stickable, k)
    isfinite(log_sum) || return -Inf
    return log(unstick_rate_constant(flow, 1)) + log_sum
end

function rate(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    active, stickable, k, nU, a, b, s = _exchangeable_linear_params(clock, state, can_stick)
    iszero(nU) && return 0.0
    logW = _exchangeable_log_total_weight(clock, flow, active, stickable, k)
    isfinite(logW) || return 0.0
    z = (a + b * τ) / s
    return exp(logW) * exp(-0.5 * abs2(z)) / sqrt(2π) / s
end

function cumulative_hazard(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    t0 == t1 && return 0.0
    active, stickable, k, nU, a, b, s = _exchangeable_linear_params(clock, state, can_stick)
    iszero(nU) && return 0.0
    logW = _exchangeable_log_total_weight(clock, flow, active, stickable, k)
    isfinite(logW) || return 0.0
    return max(0.0, _linear_gaussian_component_hazard(a, b, s, logW, Float64(t1)) -
                    _linear_gaussian_component_hazard(a, b, s, logW, Float64(t0)))
end

function _exchangeable_gaussian_line_total_hazard(a::Real, b::Real, s::Real, logW::Real)
    return _linear_gaussian_component_total_hazard(a, b, s, logW)
end

function _exchangeable_invert_gaussian_line(a::Real, b::Real, s::Real, W::Real, threshold::Real)
    if iszero(b)
        λ = W * exp(-0.5 * abs2(a / s)) / sqrt(2π) / s
        return iszero(λ) ? Inf : Float64(threshold) / λ
    end
    z0 = a / s
    Φ0 = Distributions.normcdf(z0)
    if b > 0
        target = Φ0 + b * Float64(threshold) / W
    else
        target = Φ0 - abs(b) * Float64(threshold) / W
    end
    0.0 < target < 1.0 || return Inf
    z = quantile(Normal(), target)
    τ = (s * z - a) / b
    return τ >= 0 ? τ : 0.0
end

function sample_time(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    active, stickable, k, nU, a, b, s = _exchangeable_linear_params(clock, state, can_stick)
    iszero(nU) && return Inf
    logW = _exchangeable_log_total_weight(clock, flow, active, stickable, k)
    isfinite(logW) || return Inf
    W = exp(logW)
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = _linear_gaussian_component_hazard(a, b, s, logW, Float64(horizon))
        H < threshold && return Inf
        return _exchangeable_invert_gaussian_line(a, b, s, W, threshold)
    end
    total_available = _exchangeable_gaussian_line_total_hazard(a, b, s, logW)
    total_available < threshold && return Inf
    return _exchangeable_invert_gaussian_line(a, b, s, W, threshold)
end

function _sample_uniform_inactive_stickable(rng::Random.AbstractRNG, indices::AbstractVector{Int}, active::BitVector, stickable::BitVector, nU::Integer)
    nU > 0 || throw(ArgumentError("cannot sample an unstick label because there are no inactive stickable coordinates"))
    draw = rand(rng, 1:nU)
    seen = 0
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            seen += 1
            seen == draw && return indices[j]
        end
    end
    return indices[lastindex(indices)]
end

function _sample_static_model_prior_label(rng::Random.AbstractRNG, indices::AbstractVector{Int}, prior::AbstractModelPriorOdds, active::BitVector, stickable::BitVector, k::Integer)
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            max_logw = max(max_logw, _log_model_add_odds_with_count(prior, active, j, k))
        end
    end
    isfinite(max_logw) || throw(ArgumentError("cannot sample an unstick label because all inactive stickable rates are zero"))
    total = 0.0
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            logw = _log_model_add_odds_with_count(prior, active, j, k)
            total += isfinite(logw) ? exp(logw - max_logw) : 0.0
        end
    end
    draw = rand(rng) * total
    last_candidate = 0
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            logw = _log_model_add_odds_with_count(prior, active, j, k)
            if isfinite(logw)
                last_candidate = j
                draw -= exp(logw - max_logw)
                draw <= 0 && return indices[j]
            end
        end
    end
    return indices[last_candidate]
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    active, stickable, k, nU, _, _, _ = _exchangeable_linear_params(clock, state, can_stick)
    indices = beta_indices(clock.slab_provider)
    return _sample_static_model_prior_label(rng, indices, clock.model_prior_odds, active, stickable, k)
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab,<:Union{ExchangeableModelSizePrior,BetaBernoulliModelPriorOdds}}, ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    active, stickable, _, nU, _, _, _ = _exchangeable_linear_params(clock, state, can_stick)
    return _sample_uniform_inactive_stickable(rng, beta_indices(clock.slab_provider), active, stickable, nU)
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    return sample_label(rng, clock, flow, state, can_stick)
end

function scalar_logscale_gaussian_line_segment(
    provider::GlobalLogscaleExchangeableGaussianSlab,
    model_prior::AbstractModelPriorOdds,
    flow::Union{ZigZag,BouncyParticle},
    state::StickyPDMPState,
    can_stick::BitVector,
    horizon::Real,
)
    isfinite(horizon) || throw(ArgumentError("scalar logscale residual segments require a finite horizon"))
    clock = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(provider.beta_indices, provider.u, provider.v), model_prior)
    active, stickable, k, nU, a0, b, s0 = _exchangeable_linear_params(clock, state, can_stick)
    iszero(nU) && throw(ArgumentError("cannot build a scalar logscale segment because there are no inactive stickable coordinates"))
    logW = _exchangeable_log_total_weight(clock, flow, active, stickable, k)
    isfinite(logW) || throw(ArgumentError("cannot build a scalar logscale segment because all model-prior odds are zero"))
    if !iszero(provider.mean)
        μ = provider.mean
        sum_centered = 0.0
        sum_velocity = 0.0
        indices = beta_indices(provider)
        @inbounds for j in eachindex(indices)
            if state.free[indices[j]]
                sum_centered += state.ξ.x[indices[j]] - μ
                sum_velocity += state.ξ.θ[indices[j]]
            end
        end
        denom = provider.u + k * provider.v
        c = provider.v / denom
        a0 = μ + c * sum_centered
        b = c * sum_velocity
    end
    ell0 = provider.logscale_offset + state.ξ.x[provider.logscale_index]
    r = state.ξ.θ[provider.logscale_index]
    return ScalarLogscaleGaussianLineSegment(Float64(a0), Float64(b), Float64(s0), Float64(ell0), Float64(r), Float64(logW), Float64(horizon))
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

function _independent_fixed_logweights!(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    cache = clock.cache
    @inbounds for j in eachindex(indices)
        i = indices[j]
        active = state.free[i]
        cache.active_beta[j] = active
        cache.stickable_beta[j] = can_stick[i]
    end
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            cache.log_weights[j] = log_model_add_odds(clock.model_prior_odds, cache.active_beta, j) + provider.log_q_zero[j]
            max_logw = max(max_logw, cache.log_weights[j])
        else
            cache.log_weights[j] = -Inf
        end
    end
    return cache, max_logw
end

function _independent_fixed_rate(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, max_logw = _independent_fixed_logweights!(clock, state, can_stick)
    isfinite(max_logw) || return 0.0
    total = 0.0
    @inbounds for j in eachindex(cache.log_weights)
        total += isfinite(cache.log_weights[j]) ? exp(cache.log_weights[j] - max_logw) : 0.0
    end
    return unstick_rate_constant(flow, 1) * exp(max_logw) * total
end

function rate(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    return _independent_fixed_rate(clock, flow, state, can_stick)
end

function cumulative_hazard(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    return (Float64(t1) - Float64(t0)) * _independent_fixed_rate(clock, flow, state, can_stick)
end

function sample_time(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    λ = _independent_fixed_rate(clock, flow, state, can_stick)
    iszero(λ) && return Inf
    τ = rand(rng, Exponential()) / λ
    return τ <= horizon ? τ : Inf
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, max_logw = _independent_fixed_logweights!(clock, state, can_stick)
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

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    return sample_label(rng, clock, flow, state, can_stick)
end

function _prepare_exponential_sum_cache!(clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    cache = clock.cache
    Cv = unstick_rate_constant(flow, 1)
    logCv_phi0 = log(Cv) - 0.5 * log(2π)
    @inbounds for j in eachindex(indices)
        i = indices[j]
        cache.active_beta[j] = state.free[i]
        cache.stickable_beta[j] = can_stick[i]
        cache.slopes[j] = state.ξ.θ[provider.logscale_indices[j]]
        cache.logc[j] = -Inf
    end
    @inbounds for j in eachindex(indices)
        i = indices[j]
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = log_model_add_odds(clock.model_prior_odds, cache.active_beta, j)
            log_s0 = provider.log_base_scales[j] + state.ξ.x[provider.logscale_indices[j]]
            cache.logc[j] = logCv_phi0 + logρ - log_s0
        end
    end
    return cache
end

function _exponential_component_hazard_from_logc(logc::Real, r::Real, T::Real)
    T <= 0 && return 0.0
    isfinite(logc) || return isinf(logc) && logc > 0 ? Inf : 0.0
    c = exp(logc)
    iszero(r) && return c * T
    return -c * expm1(-r * T) / r
end

function _exponential_sum_hazard_from_cache(cache::ExponentialSumAggregateCache, T::Real)
    total = 0.0
    @inbounds for j in eachindex(cache.logc)
        total += _exponential_component_hazard_from_logc(cache.logc[j], cache.slopes[j], T)
    end
    return max(0.0, total)
end

function rate(clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    max_logλ = -Inf
    @inbounds for j in eachindex(cache.logc)
        isfinite(cache.logc[j]) && (max_logλ = max(max_logλ, cache.logc[j] - cache.slopes[j] * τ))
    end
    isfinite(max_logλ) || return 0.0
    total = 0.0
    @inbounds for j in eachindex(cache.logc)
        if isfinite(cache.logc[j])
            total += exp(cache.logc[j] - cache.slopes[j] * τ - max_logλ)
        end
    end
    return exp(max_logλ) * total
end

function cumulative_hazard(clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    t0 == t1 && return 0.0
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    return max(0.0, _exponential_sum_hazard_from_cache(cache, Float64(t1)) -
                    _exponential_sum_hazard_from_cache(cache, Float64(t0)))
end

function _exponential_sum_available_hazard(cache::ExponentialSumAggregateCache)
    total = 0.0
    @inbounds for j in eachindex(cache.logc)
        if isfinite(cache.logc[j])
            c = exp(cache.logc[j])
            r = cache.slopes[j]
            if r <= 0
                return Inf
            else
                total += c / r
            end
        end
    end
    return total
end

function sample_time(rng::Random.AbstractRNG, clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    any(isfinite, cache.logc) || return Inf
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = _exponential_sum_hazard_from_cache(cache, Float64(horizon))
        H < threshold && return Inf
        f = τ -> _exponential_sum_hazard_from_cache(cache, τ) - threshold
        return Roots.find_zero(f, (0.0, Float64(horizon)), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
    end

    available = _exponential_sum_available_hazard(cache)
    available < threshold && return Inf
    lo = 0.0
    hi = 1.0
    H_hi = _exponential_sum_hazard_from_cache(cache, hi)
    iterations = 0
    while H_hi < threshold && iterations < 80
        lo = hi
        hi *= 2.0
        H_hi = _exponential_sum_hazard_from_cache(cache, hi)
        iterations += 1
    end
    H_hi < threshold && return Inf
    f = τ -> _exponential_sum_hazard_from_cache(cache, τ) - threshold
    return Roots.find_zero(f, (lo, hi), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
end

function sample_label(rng::Random.AbstractRNG, clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    return sample_label(rng, clock, flow, state, 0.0, can_stick)
end

function sample_label(rng::Random.AbstractRNG, clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    indices = beta_indices(clock.slab_provider)
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        cache.log_weights[j] = isfinite(cache.logc[j]) ? cache.logc[j] - cache.slopes[j] * τ : -Inf
        max_logw = max(max_logw, cache.log_weights[j])
    end
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
