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


# the U<: Real is so we can put a Dual number in there for ForwardDiff but it's not ideal though
struct PDMPState{T<:SkeletonPoint, U<:Real} <: AbstractPDMPState
    t::Base.RefValue{U}
    ξ::T
end
PDMPState(t::Real, ξ::SkeletonPoint) = PDMPState(Ref(float(t)), ξ)

struct StickyPDMPState{T<:SkeletonPoint, U<:Real} <: AbstractPDMPState
    t::Base.RefValue{U}
    ξ::T
    free::BitVector
    old_velocity::Vector{Float64} # Old velocity at the time of freezing
end
StickyPDMPState(t::Real, args...) = StickyPDMPState(Ref(float(t)), args...)
StickyPDMPState(t::Base.RefValue{<:Real}, ξ::SkeletonPoint) = StickyPDMPState(t, ξ, .!(iszero.(ξ.x) .&& iszero.(ξ.θ)), similar(ξ.θ))

substate(state::StickyPDMPState) = PDMPState(state.t, SkeletonPoint(view(state.ξ.x, state.free), view(state.ξ.θ, state.free)))

# default method
subflow(flow::ContinuousDynamics, ::BitVector) = flow

Base.copy(state::PDMPState) = PDMPState(Ref(state.t[]), copy(state.ξ))
Base.copy(state::StickyPDMPState) = StickyPDMPState(Ref(state.t[]), copy(state.ξ), copy(state.free), copy(state.old_velocity))

function reflect!(state::AbstractPDMPState, ∇ϕ::AbstractVector, flow::ContinuousDynamics, cache)
    reflect!(state.ξ, ∇ϕ, flow, cache)
end

# TODO: this breaks ZigZag, but fixes BouncyParticle & Boomerang?
# the logic for these kinds of subflow/ substate needs to be rethought properly...
# there could be a generic fallback, but for the best performance each flow should perhaps implement something custom
# a generic fallback would also need "subflow", i.e., for the boomerang..., so this is nontrivial.
# function reflect!(state::StickyPDMPState, ∇ϕ::AbstractVector, flow::ContinuousDynamics, cache)
#     # this does not work in general! we'd need some kind of sub-cache here as well...
#     reflect!(substate(state), view(∇ϕ, state.free), flow, cache)
# end

function reflect!(state::AbstractPDMPState, ∇ϕ::Real, i::Integer, flow::ContinuousDynamics)
    # assumes that the algorithm always suggest a valid non-sticking i
    reflect!(state.ξ, ∇ϕ, i, flow)
end

refresh_velocity!(state::StickyPDMPState, flow::ContinuousDynamics) = refresh_velocity!(substate(state).ξ, subflow(flow, state.free))
refresh_velocity!(state::PDMPState, flow::ContinuousDynamics) = refresh_velocity!(state.ξ, flow)

move_forward_time(state::AbstractPDMPState, τ::Real, flow::ContinuousDynamics) = move_forward_time!(copy(state), τ, flow)

function validate_state(state::PDMPState, flow::Union{Nothing, ContinuousDynamics}, msg::AbstractString = "")
    ξ = state.ξ
    @assert all(isfinite, ξ.x) "state.ξ.x contains non-finite values $(msg): $(ξ.x)"
    @assert all(isfinite, ξ.θ) "state.ξ.θ contains non-finite values $(msg): $(ξ.θ)"
end

function validate_state(state::StickyPDMPState, flow::Union{Nothing, ContinuousDynamics}, msg::AbstractString = "")
    ξ = state.ξ
    @assert all(isfinite, ξ.x) "state.ξ.x contains non-finite values $(msg): $(ξ.x)"
    @assert all(isfinite, ξ.θ) "state.ξ.θ contains non-finite values $(msg): $(ξ.θ)"

    free = state.free
    for i in eachindex(free)
        if !free[i]
            @assert iszero(ξ.x[i]) "state.ξ.x[$i] is frozen but not zero! contains non-finite values $(msg): $(ξ.x)"
            @assert iszero(ξ.θ[i]) "state.ξ.θ[$i] is frozen but not zero! contains non-finite values $(msg): $(ξ.θ)"
        end
    end
end
