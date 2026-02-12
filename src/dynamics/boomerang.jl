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

initialize_velocity(flow::Boomerang, d::Integer) = refresh_velocity!(Vector{Float64}(undef, d), flow)

# Boomerang-specific velocity refreshment (complete refresh from Gaussian)
refresh_rate(flow::Boomerang) = flow.λref
refresh_velocity!(ξ::SkeletonPoint, flow::Boomerang) = refresh_velocity!(ξ.θ, flow)
refresh_velocity!(θ::AbstractVector, flow::Boomerang) = ldiv!(flow.L', randn!(θ))
function refresh_velocity!(state::StickyPDMPState, flow::Boomerang)
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

function reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::Boomerang, cache)
    θ = ξ.θ
    z = cache.z
    copyto!(z, ∇ϕ)
    ldiv!(flow.L, z)
    ldiv!(flow.L', z)
    reflection_coeff = 2 * dot(θ, ∇ϕ) / dot(∇ϕ, z)
    θ .-= reflection_coeff .* z
    return nothing
end

function reflect!(state::StickyPDMPState, ∇ϕ::AbstractVector, flow::Boomerang, cache)
    # this does not work in general! we'd need some kind of sub-cache here as well...
    reflect!(substate(state), view(∇ϕ, state.free), flow, cache)
end


function move_forward_time!(state::AbstractPDMPState, τ::Real, flow::Boomerang)
    state.t[] += τ
    move_forward_time!(state.ξ, τ, flow)
    state
end

function move_forward_time!(ξ::SkeletonPoint, τ::Real, flow::Boomerang)
    x, θ = ξ.x, ξ.θ
    μ = flow.μ
    s, c = sincos(τ)
    for i in eachindex(x)
        Δ = x[i] - μ[i]
        x[i] =  Δ*c + θ[i]*s + μ[i]
        θ[i] = -Δ*s + θ[i]*c
    end
end

λ(ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::Boomerang) = pos(dot(∇ϕ, ξ.θ))

function freezing_time(ξ::SkeletonPoint, flow::Boomerang, i::Integer)
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

function ab(ξ::SkeletonPoint, c::AbstractVector, flow::Boomerang, cache)
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

function correct_gradient!(∇ϕ::AbstractVector, x::AbstractVector, ::AbstractVector, flow::Boomerang, cache)
    # For Boomerang dynamics, the corrected gradient is: ∇U(x) - Γ(x - μ)
    # This ensures sampling from the correct distribution with precision Γ
    z = cache.z
    z .= x .- flow.μ
    # mul!(∇ϕ, flow.Γ, z, one(eltype(∇ϕ)), one(eltype(∇ϕ)))  # ∇ϕ += Γ(x-μ)
    mul!(∇ϕ, flow.Γ, z, -one(eltype(∇ϕ)), one(eltype(∇ϕ)))  # ∇ϕ -= Γ(x-μ)
    return ∇ϕ
end
