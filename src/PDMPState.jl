abstract type AbstractPDMPState end

# unclear if we need this type, can also work directly with PDMPState instead?
struct SkeletonPoint{TX<:AbstractVector{<:Real}, TΘ<:AbstractVector{<:Real}}
    x::TX
    θ::TΘ
    function SkeletonPoint(x::TX, θ::TΘ) where {
        TX<:AbstractVector{<:Real},
        TΘ<:AbstractVector{<:Real}
    }
        length(x) == length(θ) || throw(ArgumentError("x and θ must have equal length"))
        new{TX,TΘ}(x, θ)
    end
end
Base.length(ξ::SkeletonPoint) = length(ξ.x)
Base.copy(ξ::SkeletonPoint) = SkeletonPoint(copy(ξ.x), copy(ξ.θ))
function Base.copyto!(dest::SkeletonPoint, src::SkeletonPoint)
    copyto!(dest.x, src.x)
    copyto!(dest.θ, src.θ)
    return dest
end


mutable struct BoundaryVelocityScratch
    active::Vector{Int}
    ΣAA::Matrix{Float64}
    θA::Vector{Float64}
    solved_θ::Vector{Float64}
    solved_cross::Vector{Float64}
    cached_free::BitVector
    factor_source::Any
    active_count::Int
    active_factor_valid::Bool
    canonical_signs::Vector{Float64}
    proposal_signs::Vector{Float64}
    factor_generation::UInt
end

# the U<: Real is so we can put a Dual number in there for ForwardDiff but it's not ideal though
struct PDMPState{T<:SkeletonPoint, U<:Real} <: AbstractPDMPState
    t::Base.RefValue{U}
    ξ::T
    boundary_scratch::BoundaryVelocityScratch
end
PDMPState(t::Real, ξ::SkeletonPoint) =
    PDMPState(Ref(float(t)), ξ, BoundaryVelocityScratch())
PDMPState(t::Base.RefValue{<:Real}, ξ::SkeletonPoint) =
    PDMPState(t, ξ, BoundaryVelocityScratch())

function BoundaryVelocityScratch(d::Integer)
    return BoundaryVelocityScratch(
        Vector{Int}(undef, d),
        Matrix{Float64}(undef, d, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        falses(d),
        nothing,
        0,
        false,
        zeros(d),
        zeros(d),
        zero(UInt),
    )
end
BoundaryVelocityScratch() = BoundaryVelocityScratch(0)

function Base.copy(s::BoundaryVelocityScratch)
    return BoundaryVelocityScratch(
        copy(s.active),
        copy(s.ΣAA),
        copy(s.θA),
        copy(s.solved_θ),
        copy(s.solved_cross),
        copy(s.cached_free),
        s.factor_source,
        s.active_count,
        s.active_factor_valid,
        copy(s.canonical_signs),
        copy(s.proposal_signs),
        s.factor_generation,
    )
end

function _ensure_boundary_scratch!(s::BoundaryVelocityScratch, d::Integer)
    length(s.active) >= d && size(s.ΣAA, 1) >= d && return s
    resize!(s.active, d)
    s.ΣAA = Matrix{Float64}(undef, d, d)
    resize!(s.θA, d)
    resize!(s.solved_θ, d)
    resize!(s.solved_cross, d)
    resize!(s.cached_free, d)
    resize!(s.canonical_signs, d)
    resize!(s.proposal_signs, d)
    s.active_factor_valid = false
    return s
end

function _ensure_product_boundary_scratch!(s::BoundaryVelocityScratch, d::Integer)
    length(s.active) >= d && return s
    resize!(s.active, d)
    resize!(s.cached_free, d)
    resize!(s.θA, d)
    resize!(s.solved_θ, d)
    resize!(s.solved_cross, d)
    resize!(s.canonical_signs, d)
    resize!(s.proposal_signs, d)
    return s
end

struct StickyPDMPState{T<:SkeletonPoint, U<:Real} <: AbstractPDMPState
    t::Base.RefValue{U}
    ξ::T
    free::BitVector
    boundary_scratch::BoundaryVelocityScratch
end
StickyPDMPState(t::Real, args...) = StickyPDMPState(Ref(float(t)), args...)
StickyPDMPState(t::Base.RefValue{<:Real}, ξ::SkeletonPoint) =
    StickyPDMPState(t, ξ, .!(iszero.(ξ.x) .&& iszero.(ξ.θ)),
        BoundaryVelocityScratch())
StickyPDMPState(t::Base.RefValue{<:Real}, ξ::SkeletonPoint, free::BitVector) =
    StickyPDMPState(t, ξ, free, BoundaryVelocityScratch())

# default method
subflow(flow::ContinuousDynamics, ::BitVector) = flow

Base.copy(state::PDMPState) =
    PDMPState(Ref(state.t[]), copy(state.ξ), copy(state.boundary_scratch))
Base.copy(state::StickyPDMPState) =
    StickyPDMPState(Ref(state.t[]), copy(state.ξ), copy(state.free),
                    copy(state.boundary_scratch))

_shallow_copy_sticky_state(state::StickyPDMPState) =
    StickyPDMPState(Ref(state.t[]), copy(state.ξ), copy(state.free),
                    state.boundary_scratch)

function Base.copyto!(dest::PDMPState, src::PDMPState)
    dest.t[] = src.t[]
    copyto!(dest.ξ, src.ξ)
    _copy_compatible_boundary_cache!(dest.boundary_scratch, src.boundary_scratch)
    return dest
end

function Base.copyto!(dest::StickyPDMPState, src::StickyPDMPState)
    dest.t[] = src.t[]
    copyto!(dest.ξ, src.ξ)
    copyto!(dest.free, src.free)
    _copy_compatible_boundary_cache!(dest.boundary_scratch, src.boundary_scratch)
    return dest
end

function reflect!(rng::Random.AbstractRNG, state::AbstractPDMPState, ∇ϕ::AbstractVector, flow::ContinuousDynamics, cache)
    reflect!(rng, state.ξ, ∇ϕ, flow, cache)
end

function reflect!(state::AbstractPDMPState, ∇ϕ::Real, i::Integer, flow::ContinuousDynamics)
    # assumes that the algorithm always suggest a valid non-sticking i
    reflect!(state.ξ, ∇ϕ, i, flow)
end

refresh_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState,
    flow::ContinuousDynamics) = draw_stratum_velocity!(rng, state, flow)
refresh_velocity!(rng::Random.AbstractRNG, state::PDMPState, flow::ContinuousDynamics) = refresh_velocity!(rng, state.ξ, flow)

# backward-compatible wrappers (no rng argument → default_rng)
reflect!(ξ_or_state, ∇ϕ::AbstractVector, flow::ContinuousDynamics, cache) = reflect!(Random.default_rng(), ξ_or_state, ∇ϕ, flow, cache)
refresh_velocity!(ξ_or_state, flow::ContinuousDynamics) = refresh_velocity!(Random.default_rng(), ξ_or_state, flow)

move_forward_time(state::AbstractPDMPState, τ::Real, flow::ContinuousDynamics) = move_forward_time!(copy(state), τ, flow)

function validate_state(state::PDMPState, flow::Union{Nothing, ContinuousDynamics}, msg::AbstractString = "")
    ξ = state.ξ
    all(isfinite, ξ.x) || error("state.ξ.x contains non-finite values $(msg): $(ξ.x)")
    all(isfinite, ξ.θ) || error("state.ξ.θ contains non-finite values $(msg): $(ξ.θ)")
end

function validate_state(state::StickyPDMPState, flow::Union{Nothing, ContinuousDynamics}, msg::AbstractString = "")
    ξ = state.ξ
    all(isfinite, ξ.x) || error("state.ξ.x contains non-finite values $(msg): $(ξ.x)")
    all(isfinite, ξ.θ) || error("state.ξ.θ contains non-finite values $(msg): $(ξ.θ)")

    free = state.free
    for i in eachindex(free)
        if !free[i]
            iszero(ξ.x[i]) || error("state.ξ.x[$i] is frozen but not zero! $(msg): $(ξ.x)")
            iszero(ξ.θ[i]) || error("state.ξ.θ[$i] is frozen but not zero! $(msg): $(ξ.θ)")
        end
    end
end
