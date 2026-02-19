# redundant?
struct IdentityPreconditioner <: AbstractPreconditioner end

struct DiagonalPreconditioner{T<:AbstractVector{<:Real}} <: AbstractPreconditioner
    # could also be PDMats.PDiagMat?
    scale::T # The 'a' vector
end

# Not yet implemented
# struct FullRankPreconditioner{T} <: AbstractPreconditioner
#     cov::Matrix{T}   # The 'A' matrix
#     chol::Cholesky{T, Matrix{T}} # Cache factorizations if needed
# end

subpreconditioner(pd::IdentityPreconditioner, ::BitVector) = pd
subpreconditioner(pd::DiagonalPreconditioner, free::BitVector) = DiagonalPreconditioner(view(pd.scale, free))


struct PreconditionedDynamics{P <: AbstractPreconditioner, D <: ContinuousDynamics} <: ContinuousDynamics
    metric::P
    dynamics::D
end

update_preconditioner!(flow::ContinuousDynamics, ::AbstractPDMPTrace, state::AbstractPDMPState) = flow
# function update_preconditioner!(flow::PreconditionedDynamics{M<:FullRankPreconditioner}, trace::AbstractPDMPTrace)
# end

function update_preconditioner!(flow::PreconditionedDynamics{<:DiagonalPreconditioner}, trace::AbstractPDMPTrace, state::AbstractPDMPState, first_update::Bool = false)
    sigmas = Statistics.std(trace)
    # old_scales = copy(flow.metric.scale)
    # old_velocity = copy(state.ξ.θ)
    for i in eachindex(sigmas)

        old_scale = flow.metric.scale[i]
        # safety check
        if iszero(sigmas[i])
            new_scale = old_scale / 2.0
        else
            new_scale = sigmas[i]
        end

        flow.metric.scale[i] = new_scale

        # ratio = iszero(old_scale) ? 1.0 : (new_scale / old_scale)
        ratio = first_update ? new_scale : new_scale / old_scale

        # B. Apply to the Particle
        if !(state isa StickyPDMPState) || state.free[i]
            # Case 1: Particle is moving. Update physical velocity.
            state.ξ.θ[i] *= ratio
        else
            # Case 2: Particle is stuck. Update the stored velocity.
            state.old_velocity[i] *= ratio
        end
    end
    # @info "preconditioner adapt: old state.ξ.θ = $old_velocity, new state.ξ.θ = $(state.ξ.θ), old_scales = $old_scales new_scales = $(flow.metric.scale)"
    flow
end

isfactorized(::PreconditionedDynamics{<:Any, T}) where {T<:ContinuousDynamics} = isfactorized(T)

# not needed?
# transform_velocity!(v, flow::ContinuousDynamics) = nothing # No-op
# transform_velocity!(v, flow::PreconditionedDynamics) = transform_velocity!(v, flow.metric)

"""
Transform velocity vector `v` according to preconditioner `M`.
Usually equivalent to v := M * v
"""
transform_velocity!(v, ::IdentityPreconditioner) = nothing # No-op
transform_velocity!(v, M::DiagonalPreconditioner) = (v .*= M.scale)
# transform_velocity!(v, M::FullRankPreconditioner) = (v .= M.L * v)  # Not yet implemented


# 1. Trajectory Movement (Kinematics are invariant)
move_forward_time!(ξ::SkeletonPoint, τ, pd::PreconditionedDynamics) = move_forward_time!(ξ, τ, pd.dynamics)
move_forward_time!(state::AbstractPDMPState, τ, pd::PreconditionedDynamics) = move_forward_time!(state, τ, pd.dynamics)

# 2. Event Rates (Dot products are invariant)
λ(ξ::SkeletonPoint, ∇ϕ::AbstractVector, pd::PreconditionedDynamics) = λ(ξ, ∇ϕ, pd.dynamics)

# 3. Reflection Logic (Mirroring is invariant)
reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, pd::PreconditionedDynamics, cache) = reflect!(ξ, ∇ϕ, pd.dynamics, cache)
reflect!(state::AbstractPDMPState, ∇ϕ::AbstractVector, pd::PreconditionedDynamics, cache) = reflect!(state, ∇ϕ, pd.dynamics, cache)


# 4. Hitting Times (Geometry is invariant)
freezing_time(ξ::SkeletonPoint, pd::PreconditionedDynamics, i::Integer) = freezing_time(ξ, pd.dynamics, i)

"""
Get the refreshment rate for the given continuous dynamics.

    Fallback implementation assumes `flow` has a field `λref` and returns `flow.λref`.
"""
refresh_rate(flow::PreconditionedDynamics) = refresh_rate(flow.dynamics)


# 5. Initialization
function initialize_velocity(pd::PreconditionedDynamics, d::Integer)
    # 1. Ask inner dynamics for a "canonical" velocity (e.g., {-1, 1})
    v = initialize_velocity(pd.dynamics, d)
    # 2. Stretch it to physical space (e.g., {-a_i, a_i})
    transform_velocity!(v, pd.metric)
    return v
end

# 6. Refreshment
function refresh_velocity!(ξ::SkeletonPoint, pd::PreconditionedDynamics)
    # 1. Let the inner dynamics refresh to a canonical state
    # (Note: This assumes the inner dynamics resets v to something standard, like N(0,I))
    refresh_velocity!(ξ, pd.dynamics)
    # 2. Apply the preconditioner again
    transform_velocity!(ξ.θ, pd.metric)
end

subflow(pd::PreconditionedDynamics, free::BitVector) = PreconditionedDynamics(subpreconditioner(pd.metric, free), subflow(pd.dynamics, free))

# Type aliases for common combinations
const PreconditionedZigZag{T} = PreconditionedDynamics{DiagonalPreconditioner{T}, ZigZag}
const PreconditionedBPS{T} = PreconditionedDynamics{DiagonalPreconditioner{T}, BouncyParticle}

# Convenience constructors
"""
    PreconditionedZigZag(d::Integer; scale=ones(d))

Create a preconditioned Zig-Zag sampler with `d` dimensions.
"""
function PreconditionedZigZag(d::Integer; scale::AbstractVector{T}=ones(d)) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), ZigZag(d))
end

"""
    PreconditionedBPS(d::Integer; refresh_rate=1.0, scale=ones(d))

Create a preconditioned BPS sampler with `d` dimensions.
"""
function PreconditionedBPS(d::Integer; refresh_rate::Real=1.0, scale::AbstractVector{T}=ones(d)) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), BouncyParticle(d, refresh_rate))
end

function PreconditionedBPS(Γ::AbstractMatrix, μ::AbstractVector; refresh_rate::Real=1.0, scale::AbstractVector{T}=ones(length(μ))) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), BouncyParticle(Γ, μ, refresh_rate))
end

function PreconditionedZigZag(Γ::AbstractMatrix, μ::AbstractVector; scale::AbstractVector{T}=ones(length(μ))) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), ZigZag(Γ, μ))
end

# Forwarding ab methods
ab(ξ::SkeletonPoint, c::AbstractVector, pd::PreconditionedDynamics, cache) = ab(ξ, c, pd.dynamics, cache)
ab_i(i::Integer, ξ::SkeletonPoint, c::AbstractVector, pd::PreconditionedDynamics, cache) = ab_i(i, ξ, c, pd.dynamics, cache)