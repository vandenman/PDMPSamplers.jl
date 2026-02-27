struct Boomerang{U, T, S, LT} <: NonFactorizedDynamics
    Γ::U
    μ::T
    λref::S
    ρ::S  # reserved for partial-refreshment features
    L::LT
    ΣL::LT  # cholesky(Symmetric(Γ)).L
end
function Boomerang(Γ, μ, λ = 0.1; ρ=0.0)
    if Γ isa Diagonal
        return Boomerang(Γ, μ, λ, ρ, cholesky(Symmetric(Γ)).L, cholesky(inv(Γ)).L)
    elseif Γ isa PDMats.PDMat
        invΓ = inv(Γ)
        return Boomerang(Γ, μ, λ, ρ, cholesky(Γ).L, cholesky(invΓ).L)
    else
        Γs = Symmetric(Γ)
        return Boomerang(Γs, μ, λ, ρ, cholesky(Γs).L, cholesky(inv(Γs)).L)
    end
end
Boomerang(d::Integer) = Boomerang(I(d), zeros(d), 0.1)
Boomerang(d::Integer, λref::Real) = Boomerang(I(d), zeros(d), λref)

"""
    MutableBoomerang{U, T, S, LT, ET} <: NonFactorizedDynamics

Mutable version of `Boomerang` for adaptation during warmup.
Same fields as `Boomerang` plus `eigen_cache` for dense/low-rank Γ (Phase 2+).
Created via the `AdaptiveBoomerang` convenience constructors.
"""
mutable struct MutableBoomerang{U, T, S, LT, ET} <: NonFactorizedDynamics
    Γ::U
    μ::T
    const λref::S
    const ρ::S
    L::LT
    ΣL::LT
    eigen_cache::ET  # nothing for diagonal Γ; reserved for Phase 2+
end

"""
    AnyBoomerang

Union type matching both `Boomerang` and `MutableBoomerang`.
All Boomerang dynamics methods dispatch on this type.
"""
const AnyBoomerang = Union{Boomerang, MutableBoomerang}

function MutableBoomerang(Γ, μ, λ = 0.1; ρ=0.0)
    if Γ isa Diagonal
        return MutableBoomerang(Γ, μ, λ, ρ, cholesky(Symmetric(Γ)).L, cholesky(inv(Γ)).L, nothing)
    else
        Γs = Symmetric(Γ)
        return MutableBoomerang(Γs, μ, λ, ρ, cholesky(Γs).L, cholesky(inv(Γs)).L, nothing)
    end
end

"""
    AdaptiveBoomerang(d::Integer; λref=0.1, ρ=0.0)

Convenience constructor for an adaptive Boomerang sampler with dimension `d`.
Starts from identity precision and zero mean; learns μ and diag(Γ) during warmup.
Returns a `MutableBoomerang`.
"""
function AdaptiveBoomerang(d::Integer; λref=0.1, ρ=0.0)
    MutableBoomerang(Diagonal(ones(d)), zeros(d), λref; ρ=ρ)
end

"""
    AdaptiveBoomerang(Γ, μ; λref=0.1, ρ=0.0)

Convenience constructor for an adaptive Boomerang sampler with initial guess `(Γ, μ)`.
Returns a `MutableBoomerang`.
"""
function AdaptiveBoomerang(Γ, μ; λref=0.1, ρ=0.0)
    MutableBoomerang(Γ, μ, λref; ρ=ρ)
end

initialize_velocity(flow::AnyBoomerang, d::Integer) = refresh_velocity!(Vector{Float64}(undef, d), flow)

# Boomerang-specific velocity refreshment (complete refresh from Gaussian)
refresh_rate(flow::AnyBoomerang) = flow.λref
refresh_velocity!(ξ::SkeletonPoint, flow::AnyBoomerang) = refresh_velocity!(ξ.θ, flow)
refresh_velocity!(θ::AbstractVector, flow::AnyBoomerang) = ldiv!(flow.L', randn!(θ))
function refresh_velocity!(state::StickyPDMPState, flow::AnyBoomerang)
    ΣL = flow.ΣL
    # ΣL = cholesky(Symmetric(L*L')).L  # could cache this if many stickies
    ΣLs = view(ΣL, state.free, :)
    θ = state.ξ.θ
    randn!(θ)
    u = similar(θ, sum(state.free))
    mul!(u, ΣLs, θ)
    j = 1
    for i in eachindex(θ)
        if state.free[i]
            θ[i] = u[j]
            j += 1
        else
            θ[i] = zero(eltype(θ))
        end
    end
    return θ
end

function reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::AnyBoomerang, cache)
    θ = ξ.θ
    z = cache.z
    copyto!(z, ∇ϕ)
    ldiv!(flow.L, z)
    ldiv!(flow.L', z)
    reflection_coeff = 2 * dot(θ, ∇ϕ) / dot(∇ϕ, z)
    θ .-= reflection_coeff .* z
    return nothing
end

function reflect!(state::StickyPDMPState, ∇ϕ::AbstractVector, flow::AnyBoomerang, cache)
    reflect!(state.ξ, ∇ϕ, flow, cache)
    # Reflection may set non-zero velocity for frozen coordinates; restore invariant
    for i in eachindex(state.free)
        if !state.free[i]
            state.ξ.θ[i] = 0.0
        end
    end
end


function move_forward_time!(state::AbstractPDMPState, τ::Real, flow::AnyBoomerang)
    state.t[] += τ
    move_forward_time!(state.ξ, τ, flow)
    state
end

function move_forward_time!(state::StickyPDMPState, τ::Real, flow::AnyBoomerang)
    state.t[] += τ
    move_forward_time!(state.ξ, τ, flow, state.free)
    state
end

function move_forward_time!(ξ::SkeletonPoint, τ::Real, flow::AnyBoomerang, free::BitVector)
    x, θ = ξ.x, ξ.θ
    μ = flow.μ
    s, c = sincos(τ)
    for i in eachindex(x)
        free[i] || continue
        Δ = x[i] - μ[i]
        x[i] =  Δ*c + θ[i]*s + μ[i]
        θ[i] = -Δ*s + θ[i]*c
    end
end

function move_forward_time!(ξ::SkeletonPoint, τ::Real, flow::AnyBoomerang)
    x, θ = ξ.x, ξ.θ
    μ = flow.μ
    s, c = sincos(τ)
    for i in eachindex(x)
        Δ = x[i] - μ[i]
        x[i] =  Δ*c + θ[i]*s + μ[i]
        θ[i] = -Δ*s + θ[i]*c
    end
end

λ(ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::AnyBoomerang) = pos(dot(∇ϕ, ξ.θ))

function freezing_time(ξ::SkeletonPoint, flow::AnyBoomerang, i::Integer)
    x = ξ.x[i]
    θ = ξ.θ[i]
    μ = flow.μ[i]
    if iszero(μ)
        iszero(x) && iszero(θ) && return Inf
        iszero(θ) && return oftype(float(x), π / 2)
        if θ * x >= 0.0
            return π - atan(x / θ)
        else
            return atan(-x / θ)
        end
    else
        u = x^2 - 2μ * x + θ^2
        u < 0 && return Inf
        denom = 2μ - x
        sqrtu = sqrt(u)
        t1 = mod(2atan(sqrtu - θ, denom), 2π)
        t2 = mod(-2atan(sqrtu + θ, denom), 2π)
        # t = 0 means "now", not a future crossing; replace with 2π (full period)
        iszero(t1) && (t1 = oftype(t1, 2π))
        iszero(t2) && (t2 = oftype(t2, 2π))
        iszero(x) && return max(t1, t2)
        return min(t1, t2)
    end
end

function ab(ξ::SkeletonPoint, c::AbstractVector, flow::AnyBoomerang, cache)
    # Boomerang sampler with reference Gaussian N(μ, Γ⁻¹).
    #
    # The CORRECTED gradient is: ∇ϕ_corrected = ∇ϕ_target - Γ(x - μ)
    # The corrected rate is: λ(t) = max(0, ⟨∇ϕ_corrected(x(t)), θ(t)⟩)
    #
    # The user-provided c_val has two interpretations:
    #
    # 1. c_val = 0: The target matches the reference Gaussian exactly.
    #    In this case, ∇ϕ_corrected = 0, so the bound should be 0.
    #    Only refreshments drive exploration (correct and efficient).
    #
    # 2. c_val > 0: The target differs from the reference by at most c_val.
    #    Specifically, c_val bounds max_t |⟨∇ϕ_target(x(t)) - Γ(x(t)-μ), θ(t)⟩|.
    #    This is the bound on the CORRECTED gradient's dot product with velocity.
    #
    # Note: Unlike BPS/ZigZag, c_val for Boomerang should bound the deviation
    # from the reference, not the raw target gradient.

    c_val = maximum(c)

    # When c_val = 0, user asserts target = reference, so no bounces needed
    # When c_val > 0, it directly bounds the corrected rate
    a = c_val
    b = 0.0  # No linear growth for Boomerang (periodic dynamics)

    return (a, b, nothing)
end

function correct_gradient!(∇ϕ::AbstractVector, x::AbstractVector, ::AbstractVector, flow::AnyBoomerang, cache)
    # For Boomerang dynamics, the corrected gradient is: ∇U(x) - Γ(x - μ)
    # This ensures sampling from the correct distribution with precision Γ
    z = cache.z
    z .= x .- flow.μ
    # mul!(∇ϕ, flow.Γ, z, one(eltype(∇ϕ)), one(eltype(∇ϕ)))  # ∇ϕ += Γ(x-μ)
    mul!(∇ϕ, flow.Γ, z, -one(eltype(∇ϕ)), one(eltype(∇ϕ)))  # ∇ϕ -= Γ(x-μ)
    return ∇ϕ
end
