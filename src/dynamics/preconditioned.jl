# redundant?
struct IdentityPreconditioner <: AbstractPreconditioner end

struct DiagonalPreconditioner{T<:AbstractVector{<:Real}} <: AbstractPreconditioner
    scale::T
end

mutable struct DensePreconditioner <: AbstractPreconditioner
    L::Matrix{Float64}
    Linv::Matrix{Float64}
    v_canonical::Vector{Float64}
end

function DensePreconditioner(d::Integer)
    L = Matrix{Float64}(I, d, d)
    Linv = Matrix{Float64}(I, d, d)
    v_canonical = zeros(d)
    DensePreconditioner(L, Linv, v_canonical)
end

subpreconditioner(pd::IdentityPreconditioner, ::BitVector) = pd
subpreconditioner(pd::DiagonalPreconditioner, free::BitVector) = DiagonalPreconditioner(view(pd.scale, free))
function subpreconditioner(pd::DensePreconditioner, free::BitVector)
    DensePreconditioner(pd.L[free, free], pd.Linv[free, free], view(pd.v_canonical, free))
end


struct PreconditionedDynamics{P <: AbstractPreconditioner, D <: ContinuousDynamics} <: ContinuousDynamics
    metric::P
    dynamics::D
end

update_preconditioner!(flow::ContinuousDynamics, ::AbstractPDMPTrace, state::AbstractPDMPState) = flow

function update_preconditioner!(flow::PreconditionedDynamics{<:DiagonalPreconditioner}, trace::AbstractPDMPTrace, state::AbstractPDMPState, first_update::Bool = false)
    sigmas = Statistics.std(trace)
    for i in eachindex(sigmas)

        old_scale = flow.metric.scale[i]
        if iszero(sigmas[i])
            new_scale = old_scale / 2.0
        else
            new_scale = sigmas[i]
        end

        flow.metric.scale[i] = new_scale
        ratio = first_update ? new_scale : new_scale / old_scale

        if !(state isa StickyPDMPState) || state.free[i]
            state.ξ.θ[i] *= ratio
        else
            state.old_velocity[i] *= ratio
        end
    end
    flow
end

function update_preconditioner!(flow::PreconditionedDynamics{DensePreconditioner}, trace::AbstractPDMPTrace, state::AbstractPDMPState, first_update::Bool = false)
    M = flow.metric
    Σ_est = Statistics.cov(trace)
    d = size(Σ_est, 1)

    for i in 1:d
        Σ_est[i, i] = max(Σ_est[i, i], 1e-8)
    end
    Σ_sym = Symmetric((Σ_est + Σ_est') / 2)

    L_new = try
        cholesky(Σ_sym).L
    catch e
        e isa PosDefException || rethrow()
        return flow
    end

    M.L .= L_new
    M.Linv .= inv(LowerTriangular(L_new))

    # Re-draw canonical velocity and set physical velocity accordingly
    if flow.dynamics isa ZigZag
        rand!(M.v_canonical, (-1.0, 1.0))
    else
        randn!(M.v_canonical)
    end
    mul!(state.ξ.θ, M.L, M.v_canonical)
    flow
end

isfactorized(::PreconditionedDynamics{<:Any, T}) where {T<:ContinuousDynamics} = isfactorized(T)
isfactorized(::PreconditionedDynamics{DensePreconditioner, <:ZigZag}) = false

# not needed?
# transform_velocity!(v, flow::ContinuousDynamics) = nothing # No-op
# transform_velocity!(v, flow::PreconditionedDynamics) = transform_velocity!(v, flow.metric)

"""
Transform velocity vector `v` according to preconditioner `M`.
Usually equivalent to v := M * v
"""
transform_velocity!(v, ::IdentityPreconditioner) = nothing # No-op
transform_velocity!(v, M::DiagonalPreconditioner) = (v .*= M.scale)
function transform_velocity!(v, M::DensePreconditioner)
    copyto!(M.v_canonical, v)
    mul!(v, M.L, M.v_canonical)
end


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

# --- Dense-preconditioned ZigZag overrides ---
# In canonical space z = L⁻¹x, the rate decomposes coordinate-wise.
# Canonical gradient: ∇z = L' ∇x. Rate: Σ pos(v_i * ∇z_i).

const DensePreconditionedZigZag = PreconditionedDynamics{DensePreconditioner, <:ZigZag}
const DensePreconditionedBPS = PreconditionedDynamics{DensePreconditioner, <:BouncyParticle}

function λ(ξ::SkeletonPoint, ∇ϕ::AbstractVector, pd::DensePreconditionedZigZag)
    M = pd.metric
    v = M.v_canonical
    L = M.L
    rate = zero(eltype(∇ϕ))
    d = length(v)
    @inbounds for i in 1:d
        grad_z_i = zero(eltype(∇ϕ))
        for j in 1:d
            grad_z_i += L[j, i] * ∇ϕ[j]
        end
        rate += pos(v[i] * grad_z_i)
    end
    return rate
end

function reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, pd::DensePreconditionedZigZag, cache)
    M = pd.metric
    v = M.v_canonical
    L = M.L
    d = length(v)

    # Compute canonical gradient L'∇ϕ and weighted rates
    z = cache.z
    mul!(z, L', ∇ϕ)

    total_rate = zero(eltype(∇ϕ))
    @inbounds for i in 1:d
        total_rate += pos(v[i] * z[i])
    end

    i₀ = 1
    if ispositive(total_rate)
        u = rand() * total_rate
        cumsum = zero(total_rate)
        @inbounds for i in 1:d
            cumsum += pos(v[i] * z[i])
            if cumsum >= u
                i₀ = i
                break
            end
        end
    else
        i₀ = rand(1:d)
    end

    # Flip canonical velocity and update physical velocity
    old_vi = v[i₀]
    v[i₀] = -old_vi
    @inbounds for j in 1:d
        ξ.θ[j] -= 2.0 * old_vi * L[j, i₀]
    end
    return nothing
end

function reflect!(state::AbstractPDMPState, ∇ϕ::AbstractVector, pd::DensePreconditionedZigZag, cache)
    reflect!(state.ξ, ∇ϕ, pd, cache)
end

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


# Convenience constructors for dense-preconditioned dynamics
function DensePreconditionedZigZag(d::Integer)
    PreconditionedDynamics(DensePreconditioner(d), ZigZag(d))
end

function DensePreconditionedZigZag(Γ::AbstractMatrix, μ::AbstractVector)
    PreconditionedDynamics(DensePreconditioner(length(μ)), ZigZag(Γ, μ))
end

function DensePreconditionedBPS(d::Integer; refresh_rate::Real=1.0)
    PreconditionedDynamics(DensePreconditioner(d), BouncyParticle(d, refresh_rate))
end

function DensePreconditionedBPS(Γ::AbstractMatrix, μ::AbstractVector; refresh_rate::Real=1.0)
    PreconditionedDynamics(DensePreconditioner(length(μ)), BouncyParticle(Γ, μ, refresh_rate))
end