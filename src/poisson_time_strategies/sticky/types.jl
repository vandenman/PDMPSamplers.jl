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
_supports_aggregate_sticky_flow(::DensePreconditionedZigZag) = false
_supports_aggregate_sticky_flow(::ContinuousDynamics) = false
