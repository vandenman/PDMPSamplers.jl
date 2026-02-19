# Strategy abstract types
abstract type PoissonTimeStrategy end
abstract type BoundStrategy end

abstract type GradientStrategy end
abstract type GlobalGradientStrategy         <: GradientStrategy end
abstract type CoordinateWiseGradientStrategy <: GradientStrategy end # factorized

# Dynamics abstract types
abstract type ContinuousDynamics end
abstract type FactorizedDynamics    <: ContinuousDynamics end
abstract type NonFactorizedDynamics <: ContinuousDynamics end

isfactorized(flow::ContinuousDynamics) = flow isa FactorizedDynamics
isfactorized(::Type{T}) where {T<:ContinuousDynamics} = T <: FactorizedDynamics

# Preconditioner abstract type
abstract type AbstractPreconditioner end

# pdmp trace & events
abstract type AbstractPDMPTrace end
abstract type AbstractPDMPEvent end

# adaptation
abstract type AbstractAdapter end

# Event metadata types — carried as the third element of the
# (τ, event_type, meta) tuple returned by `next_event_time`.
"""
    abstract type PDMPEventMeta end

Supertype for metadata values returned by `next_event_time` as the third
element of the `(τ, event_type, meta)` tuple.

Concrete subtypes:
- [`BoundsMeta`](@ref)      — thinning acceptance bounds (a, b).
- [`GradientMeta`](@ref)    — precomputed gradient ∇ϕ(x).
- [`CoordinateMeta`](@ref)  — coordinate index.
- [`EmptyMeta`](@ref)       — no metadata.
"""
abstract type PDMPEventMeta end

"""Thinning bounds `a` and `b` such that the Poisson rate upper bound is `a + b * t`."""
struct BoundsMeta <: PDMPEventMeta
    a::Float64
    b::Float64
end

"""Precomputed gradient, carried to avoid recomputation in `handle_event!`."""
struct GradientMeta <: PDMPEventMeta
    ∇ϕx::Vector{Float64}
end

"""Coordinate index (used by sticky events and coordinate-wise thinning)."""
struct CoordinateMeta <: PDMPEventMeta
    i::Int
end

"""Singleton — no metadata (used by `RootsPoissonTimeStrategy`)."""
struct EmptyMeta <: PDMPEventMeta end

"""
    wrap_meta(meta) -> PDMPEventMeta

Convert a raw value returned by a user-supplied `ExactStrategy` function into
a `PDMPEventMeta` subtype. Values that are already `PDMPEventMeta` are returned
as-is.
"""
wrap_meta(m::PDMPEventMeta) = m
wrap_meta(m::Integer) = CoordinateMeta(Int(m))
wrap_meta(::Nothing) = EmptyMeta()
