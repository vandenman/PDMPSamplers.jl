struct Boomerang{U, T, S, LT} <: NonFactorizedDynamics
    Γ::U
    μ::T
    λref::S
    ρ::S  # reserved for partial-refreshment features
    L::LT
    ΣL::LT  # cholesky(Symmetric(Γ)).L
end

"""
    LowRankPrecision{T<:Real}

Represents a precision matrix Γ = Σ⁻¹ via a low-rank covariance decomposition:
    Σ = D + V Λ V'
where D is diagonal (d), V is d×r (eigenvectors), Λ is r (eigenvalues).
All Γ-related operations (mul, solve, sample) use the Woodbury identity at O(dr) cost.
"""
mutable struct LowRankPrecision{T<:Real}
    D::Vector{T}           # d, diagonal part of Σ
    V::Matrix{T}           # d × r, eigenvectors of off-diagonal Σ
    Λ::Vector{T}           # r, eigenvalues (positive)
    # Precomputed for Woodbury: Γz = D⁻¹z - D⁻¹V M⁻¹ V'D⁻¹z, M = Λ⁻¹ + V'D⁻¹V
    Dinv::Vector{T}        # 1/D, length d
    DinvV::Matrix{T}       # D⁻¹V, d × r
    M_chol::Matrix{T}      # cholesky(M).L stored as Matrix for mutability, r × r
    # For sampling from N(0, Σ) = D^{1/2}ε₁ + V Λ^{1/2} ε₂
    Dsqrt::Vector{T}       # √D, length d
    Λsqrt::Vector{T}       # √Λ, length r
    # Buffers
    buf_r1::Vector{T}      # length r
    buf_r2::Vector{T}      # length r
    buf_d::Vector{T}       # length d
end

function LowRankPrecision(d::Int, r::Int)
    T = Float64
    LowRankPrecision{T}(
        ones(T, d), zeros(T, d, r), ones(T, r),
        ones(T, d), zeros(T, d, r), Matrix{T}(I, r, r),
        ones(T, d), ones(T, r),
        zeros(T, r), zeros(T, r), zeros(T, d),
    )
end

"""
    lowrank_precompute!(lrp::LowRankPrecision)

Recompute derived quantities after D, V, Λ are updated.
"""
function lowrank_precompute!(lrp::LowRankPrecision)
    d = length(lrp.D)
    r = length(lrp.Λ)

    lrp.Dinv .= 1.0 ./ lrp.D
    lrp.Dsqrt .= sqrt.(lrp.D)
    lrp.Λsqrt .= sqrt.(lrp.Λ)

    # DinvV = D⁻¹V
    for j in 1:r, i in 1:d
        lrp.DinvV[i, j] = lrp.Dinv[i] * lrp.V[i, j]
    end

    # M = Λ⁻¹ + V'D⁻¹V (r × r)
    M = zeros(r, r)
    for i in 1:r
        M[i, i] = 1.0 / lrp.Λ[i]
    end
    mul!(M, lrp.V', lrp.DinvV, 1.0, 1.0)  # M += V' * DinvV

    # Store Cholesky factor
    lrp.M_chol .= cholesky(Symmetric(M)).L
    return lrp
end

"""
    lowrank_mul!(y, lrp, x, α, β)

Compute y = α * Γ x + β * y where Γ = (D + V Λ V')⁻¹.
Uses Woodbury identity: Γx = D⁻¹x - D⁻¹V M⁻¹ V'D⁻¹x.
"""
function lowrank_mul!(y::AbstractVector, lrp::LowRankPrecision, x::AbstractVector, α::Real, β::Real)
    # Step 1: D⁻¹x
    buf_d = lrp.buf_d
    buf_d .= lrp.Dinv .* x

    # Step 2: V'D⁻¹x (r-vector)
    buf_r1 = lrp.buf_r1
    mul!(buf_r1, lrp.DinvV', x)

    # Step 3: M⁻¹ (V'D⁻¹x) via Cholesky solve
    L_M = LowerTriangular(lrp.M_chol)
    ldiv!(L_M, buf_r1)
    ldiv!(L_M', buf_r1)

    # Step 4: D⁻¹V * (M⁻¹ V'D⁻¹x) → subtract from D⁻¹x
    mul!(buf_d, lrp.DinvV, buf_r1, -1.0, 1.0)  # buf_d = D⁻¹x - D⁻¹V M⁻¹ V'D⁻¹x

    # Step 5: y = α * Γx + β * y
    if iszero(β)
        y .= α .* buf_d
    else
        y .= α .* buf_d .+ β .* y
    end
    return y
end

"""
    lowrank_solve!(z, lrp)

In-place solve z ← Γ⁻¹z = Σz = (D + VΛV')z.
"""
function lowrank_solve!(z::AbstractVector, lrp::LowRankPrecision)
    # Σz = Dz + V(Λ(V'z))
    buf_r1 = lrp.buf_r1
    mul!(buf_r1, lrp.V', z)         # V'z
    buf_r1 .*= lrp.Λ                # Λ V'z
    z .*= lrp.D                      # Dz
    mul!(z, lrp.V, buf_r1, 1.0, 1.0)  # z = Dz + V Λ V'z
    return z
end

"""
    lowrank_quadform(lrp, v)

Compute v' Γ v where Γ = Σ⁻¹, using Woodbury:
v'Γv = v'D⁻¹v - (V'D⁻¹v)' M⁻¹ (V'D⁻¹v).
"""
function lowrank_quadform(lrp::LowRankPrecision, v::AbstractVector)
    # v'D⁻¹v
    q = zero(eltype(v))
    for i in eachindex(v)
        q += v[i]^2 * lrp.Dinv[i]
    end

    # V'D⁻¹v
    buf_r1 = lrp.buf_r1
    for j in eachindex(buf_r1)
        s = zero(eltype(v))
        for i in eachindex(v)
            s += lrp.DinvV[i, j] * v[i]
        end
        buf_r1[j] = s
    end

    # M⁻¹ (V'D⁻¹v) via Cholesky
    L_M = LowerTriangular(lrp.M_chol)
    ldiv!(L_M, buf_r1)

    # Subtract: q -= |L_M⁻¹ V'D⁻¹v|²
    for j in eachindex(buf_r1)
        q -= buf_r1[j]^2
    end

    return q
end

"""
    lowrank_sample!(θ, lrp)

Sample θ ~ N(0, Σ) where Σ = D + VΛV'.
Uses θ = D^{1/2}ε₁ + V Λ^{1/2}ε₂ with ε₁ ~ N(0,I_d), ε₂ ~ N(0,I_r).
"""
function lowrank_sample!(θ::AbstractVector, lrp::LowRankPrecision)
    randn!(θ)
    θ .*= lrp.Dsqrt  # D^{1/2} ε₁

    buf_r1 = lrp.buf_r1
    randn!(buf_r1)
    buf_r1 .*= lrp.Λsqrt  # Λ^{1/2} ε₂

    mul!(θ, lrp.V, buf_r1, 1.0, 1.0)  # θ += V Λ^{1/2} ε₂
    return θ
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
    AdaptiveBoomerang(d::Integer; λref=0.1, ρ=0.0, scheme=:diagonal)

Convenience constructor for an adaptive Boomerang sampler with dimension `d`.
Starts from identity precision and zero mean; learns μ and Γ during warmup.

- `scheme=:diagonal`: adapts only diag(Γ) and μ. O(d) per update and per step.
- `scheme=:fullrank`: adapts full Γ and μ. O(d³) per update, O(d) per step.
- `scheme=:lowrank`: adapts low-rank Γ and μ. O(d³) per update, O(dr) per step.
  Use `rank` keyword to set the rank (default: `min(d-1, 5)`).

Returns a `MutableBoomerang`.
"""
function AdaptiveBoomerang(d::Integer; λref=0.1, ρ=0.0, scheme=:diagonal, rank::Int=min(d - 1, 5))
    if scheme == :diagonal
        MutableBoomerang(Diagonal(ones(d)), zeros(d), λref; ρ=ρ)
    elseif scheme == :fullrank
        Γ = Symmetric(Matrix{Float64}(I, d, d))
        μ = zeros(d)
        L = LowerTriangular(Matrix{Float64}(I, d, d))
        ΣL = LowerTriangular(Matrix{Float64}(I, d, d))
        MutableBoomerang(Γ, μ, Float64(λref), Float64(ρ), L, ΣL, nothing)
    elseif scheme == :lowrank
        r = min(rank, d - 1)
        r < 1 && error("Low-rank scheme requires d ≥ 2 and rank ≥ 1.")
        lrp = LowRankPrecision(d, r)
        lowrank_precompute!(lrp)
        μ = zeros(d)
        MutableBoomerang(lrp, μ, Float64(λref), Float64(ρ), nothing, nothing, nothing)
    else
        error("Unknown adaptation scheme: $scheme. Use :diagonal, :fullrank, or :lowrank.")
    end
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
function refresh_velocity!(θ::AbstractVector, flow::AnyBoomerang)
    if iszero(flow.ρ)
        ldiv!(flow.L', randn!(θ))
    else
        θ_fresh = similar(θ)
        ldiv!(flow.L', randn!(θ_fresh))
        ρ = flow.ρ
        θ .= ρ .* θ .+ sqrt(1 - ρ^2) .* θ_fresh
    end
    return θ
end
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

# ─── Low-rank Boomerang method overrides ──────────────────────────────────────
# These override AnyBoomerang methods that use flow.L, flow.ΣL, or flow.Γ
# with direct operations on the LowRankPrecision structure.

"""
    LowRankMutableBoomerang

Type alias for `MutableBoomerang` with a `LowRankPrecision` Γ.
Used for method dispatch on low-rank Boomerang dynamics.
"""
const LowRankMutableBoomerang = MutableBoomerang{<:LowRankPrecision}

function refresh_velocity!(θ::AbstractVector, flow::LowRankMutableBoomerang)
    if iszero(flow.ρ)
        lowrank_sample!(θ, flow.Γ)
    else
        θ_fresh = similar(θ)
        lowrank_sample!(θ_fresh, flow.Γ)
        ρ = flow.ρ
        θ .= ρ .* θ .+ sqrt(1 - ρ^2) .* θ_fresh
    end
    return θ
end

function refresh_velocity!(state::StickyPDMPState, flow::LowRankMutableBoomerang)
    lrp = flow.Γ
    θ = state.ξ.θ
    # Sample θ_free ~ N(0, Σ[free,free]) where Σ = D + VΛV'
    buf_r = lrp.buf_r1
    randn!(buf_r)
    buf_r .*= lrp.Λsqrt  # Λ^{1/2} ε₂
    for i in eachindex(θ)
        if state.free[i]
            val = lrp.Dsqrt[i] * randn()
            for k in eachindex(buf_r)
                val += lrp.V[i, k] * buf_r[k]
            end
            θ[i] = val
        else
            θ[i] = zero(eltype(θ))
        end
    end
    return θ
end

function reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::LowRankMutableBoomerang, cache)
    θ = ξ.θ
    z = cache.z
    copyto!(z, ∇ϕ)
    lowrank_solve!(z, flow.Γ)  # z = Γ⁻¹ ∇ϕ = Σ ∇ϕ
    reflection_coeff = 2 * dot(θ, ∇ϕ) / dot(∇ϕ, z)
    θ .-= reflection_coeff .* z
    return nothing
end

function correct_gradient!(∇ϕ::AbstractVector, x::AbstractVector, ::AbstractVector, flow::LowRankMutableBoomerang, cache)
    z = cache.z
    z .= x .- flow.μ
    lowrank_mul!(∇ϕ, flow.Γ, z, -one(eltype(∇ϕ)), one(eltype(∇ϕ)))
    return ∇ϕ
end
