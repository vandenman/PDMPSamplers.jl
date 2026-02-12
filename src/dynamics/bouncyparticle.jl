"""
    BouncyParticle(λ) <: ContinuousDynamics
Input: argument `Γ`, a sparse precision matrix approximating target precision.
Bouncy particle sampler,  `λ` is the refreshment rate, which has to be
strictly positive.
"""
struct BouncyParticle{T, S, R, V, LT} <: NonFactorizedDynamics
    Γ::T
    μ::S
    λref::R
    ρ::R  # reserved for partial-refreshment features
    U::V  # reserved for partial-refreshment features
    L::LT
end

function BouncyParticle(Γ, μ, λref = 0.1; ρ=0.0)
    L = cholesky(Symmetric(Γ)).L
    σ = Vector(diag(Γ)).^(-0.5)
    return BouncyParticle(Γ, μ, λref, ρ, σ, L)
end

BouncyParticle(d::Integer) = BouncyParticle(I(d), zeros(d), 0.1)
BouncyParticle(d::Integer, λref::Real) = BouncyParticle(I(d), zeros(d), λref)

function move_forward_time!(ξ::SkeletonPoint, τ::Real, ::BouncyParticle)
    # LinearAlgebra.axpy!(τ, ξ.θ, ξ.x)
    ξ.x .+= τ .* ξ.θ
end

function move_forward_time!(state::PDMPState, τ::Real, flow::BouncyParticle)
    state.t[] += τ
    move_forward_time!(state.ξ, τ, flow)
    # state.ξ.x .+= τ .* state.ξ.θ
    # LinearAlgebra.axpy!(τ, state.ξ.θ, state.ξ.x)
    state
end
function move_forward_time!(state::StickyPDMPState, τ::Real, flow::BouncyParticle)
    state.t[] += τ
    move_forward_time!(state.ξ, τ, flow)
    # LinearAlgebra.axpy!(τ, state.ξ.θ, state.ξ.x)
    state
end

initialize_velocity(::BouncyParticle, d::Integer) = randn(d)
refresh_velocity!(ξ::SkeletonPoint, ::BouncyParticle) = randn!(ξ.θ)
refresh_rate(flow::BouncyParticle) = flow.λref


# BPS reflection: bounce against the gradient hyperplane
function reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, ::BouncyParticle, cache)
    θ = ξ.θ

    z = cache.z
    copyto!(z, ∇ϕ)
    LinearAlgebra.normalize!(z)
    coeff = 2 * dot(θ, z)
    θ .-= coeff .* z

    return nothing
end

function reflect!(state::StickyPDMPState, ∇ϕ::AbstractVector, flow::BouncyParticle, cache)
    # this does not work in general! we'd need some kind of sub-cache here as well...
    subcache = (; z = view(cache.z, 1:sum(state.free)))
    reflect!(substate(state), view(∇ϕ, state.free), flow, subcache)
end

λ(ξ::SkeletonPoint, ∇ϕx::AbstractVector, flow::BouncyParticle) = pos(dot(∇ϕx, ξ.θ))


# The canonical freezing_time for BouncyParticle is defined in
# src/poisson_time_strategies/sticky.jl, dispatching on Union{BouncyParticle,ZigZag}.

# Bounds computation for BPS
function ab(ξ::SkeletonPoint, c::AbstractVector, flow::BouncyParticle, cache)

    x, θ = ξ.x, ξ.θ
    c_val = maximum(c)

    # TODO: should use cache to avoid allocating x - flow.μ

    # Exact analysis: λ(t) = max(0, A + Bt) where:
    z = cache.z
    z .= x .- flow.μ  # Centered position
    A = dot(z, flow.Γ, θ) # Linear coefficient
    B = dot(θ, flow.Γ, θ) # Quadratic coefficient (≥ 0)

    a = c_val + A
    b = pos(B)

    refresh_time = rand_refresh_time(flow)
    # refresh_time = ispositive(refresh_rate(flow)) ? rand(Exponential(inv(refresh_rate(flow)))) : Inf
    # refresh_time = flow.λref > 0 ? flow.λref : Inf
    return (a, b, refresh_time)
end
