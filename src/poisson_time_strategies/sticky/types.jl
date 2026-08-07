"""


There are two supported ways to provide the scalar sticky rate κ.

- `AbstractVector`: For independent model and parameter priors, the rates are fixed and can be known in advance.
- `Function`: For dependent model priors, the rates only depend on whether the parameters are (non)zero.

Both forms must return a non-negative scalar. For a frozen coordinate `i`, the
proposal clock has rate `κ * Cᵢ`, where `Cᵢ` is the exact flux normalizer for
Gaussian/product dynamics or the state-dependent dominating constant for dense
ZigZag. Distribution-valued return values are unsupported.

At every accepted freeze or unfreeze transition, the complete velocity on the
new active coordinate stratum is redrawn from that stratum's invariant law.
No saved pre-freeze velocity is restored.
"""
struct Sticky{T<:PoissonTimeStrategy,U<:Union{Function,AbstractVector}} <: PoissonTimeStrategy
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
mask. The current implementation supports `ZigZag`, `BouncyParticle`,
Boomerang-family flows, and supported preconditioned variants with a
`GridThinningStrategy` inner algorithm.
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
_supports_aggregate_sticky_flow(::AnyBoomerang) = true
_supports_aggregate_sticky_flow(flow::PreconditionedDynamics) = _supports_aggregate_sticky_flow(flow.dynamics)
_supports_aggregate_sticky_flow(::DensePreconditionedZigZag) = true
_supports_aggregate_sticky_flow(::ContinuousDynamics) = false
