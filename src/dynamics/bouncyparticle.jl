"""
    BouncyParticle(Γ, μ, λref=0.1) <: ContinuousDynamics

Bouncy particle sampler with reference precision `Γ`, reference mean `μ`,
and refreshment rate `λref`.
"""
struct BouncyParticle{T, S, R} <: NonFactorizedDynamics
    Γ::T
    μ::S
    λref::R
    function BouncyParticle(Γ::T, μ::S, λref::R) where {T,S,R}
        cholesky(Symmetric(Γ)) # Validate the precision used by analytic bounds.
        return new{T,S,R}(Γ, μ, λref)
    end
end

BouncyParticle(Γ, μ) = BouncyParticle(Γ, μ, 0.1)
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
    # Frozen sticky coordinates are boundary states, not points moving with
    # their retained velocity. Advancing the complete SkeletonPoint here
    # transports a frozen coordinate away from x = θ = 0 and makes the next
    # state validation fail. Boomerang's sticky flow uses the same rule.
    @inbounds for i in eachindex(state.free, state.ξ.x, state.ξ.θ)
        state.free[i] && (state.ξ.x[i] += τ * state.ξ.θ[i])
    end
    # LinearAlgebra.axpy!(τ, state.ξ.θ, state.ξ.x)
    state
end

initialize_velocity(rng::Random.AbstractRNG, ::BouncyParticle, d::Integer) = randn(rng, d)
refresh_velocity!(rng::Random.AbstractRNG, ξ::SkeletonPoint, ::BouncyParticle) = randn!(rng, ξ.θ)
refresh_rate(flow::BouncyParticle) = flow.λref

function initialize_cache(::Random.AbstractRNG, ::BouncyParticle, ::GlobalGradientStrategy, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z=similar(ξ.x))
end


# BPS reflection: bounce against the gradient hyperplane
reflect!(::Random.AbstractRNG, ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::BouncyParticle, cache) = reflect!(ξ, ∇ϕ, flow, cache)
function reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, ::BouncyParticle, cache)
    θ = ξ.θ

    z = cache.z
    copyto!(z, ∇ϕ)
    LinearAlgebra.normalize!(z)
    coeff = 2 * dot(θ, z)
    θ .-= coeff .* z

    return nothing
end

reflect!(::Random.AbstractRNG, state::StickyPDMPState, ∇ϕ::AbstractVector, flow::BouncyParticle, cache) = reflect!(state, ∇ϕ, flow, cache)
function reflect!(state::StickyPDMPState, ∇ϕ::AbstractVector, flow::BouncyParticle, cache)
    numerator = zero(eltype(state.ξ.θ))
    denominator = zero(eltype(state.ξ.θ))
    @inbounds for i in eachindex(state.free, state.ξ.θ, ∇ϕ)
        if state.free[i]
            numerator += state.ξ.θ[i] * ∇ϕ[i]
            denominator += abs2(∇ϕ[i])
        end
    end
    iszero(denominator) && return nothing
    coefficient = 2 * numerator / denominator
    @inbounds for i in eachindex(state.free, state.ξ.θ, ∇ϕ)
        state.free[i] && (state.ξ.θ[i] -= coefficient * ∇ϕ[i])
    end
    return nothing
end

λ(ξ::SkeletonPoint, ∇ϕx::AbstractVector, flow::BouncyParticle) = pos(dot(∇ϕx, ξ.θ))

function rate_and_derivative(
    state::AbstractPDMPState,
    ::BouncyParticle,
    provider::GradHVPProvider,
)
    x = state.ξ.x
    θ = state.ξ.θ
    grad = _provider_grad(provider)
    hvp = _provider_hvp(provider)
    ∇U = grad(x)
    Hθ = hvp(x, θ)
    return dot(∇U, θ), extract_vhv(θ, Hθ)
end

function rate_and_derivative(
    state::AbstractPDMPState,
    ::BouncyParticle,
    provider::GradHVPProvider,
    cached_gradient::AbstractVector,
)
    θ = state.ξ.θ
    hvp = _provider_hvp(provider)
    Hθ = hvp(state.ξ.x, θ)
    return dot(cached_gradient, θ), extract_vhv(θ, Hθ)
end

function rate_and_derivative(
    state::AbstractPDMPState,
    flow::BouncyParticle,
    provider::VHVProvider,
)
    x = state.ξ.x
    θ = state.ξ.θ
    ∇U = provider.grad(x)
    return dot(∇U, θ), _compute_vhv_scalar(provider, state, ∇U, flow)
end

function rate_and_derivative(
    state::AbstractPDMPState,
    flow::BouncyParticle,
    provider::VHVProvider,
    cached_gradient::AbstractVector,
)
    return (
        dot(cached_gradient, state.ξ.θ),
        _compute_vhv_scalar(provider, state, cached_gradient, flow),
    )
end

function rate_and_derivative(
    state::AbstractPDMPState,
    ::BouncyParticle,
    provider::WithStatsJoint,
)
    return provider(state.ξ.x, state.ξ.θ)
end

function rate_and_derivative(
    state::AbstractPDMPState,
    ::BouncyParticle,
    provider::GradientOnlyProvider,
)
    ∇U = provider.grad(state.ξ.x)
    return dot(∇U, state.ξ.θ), 0.0
end

function rate_and_derivative(
    state::AbstractPDMPState,
    ::BouncyParticle,
    ::GradientOnlyProvider,
    cached_gradient::AbstractVector,
)
    return dot(cached_gradient, state.ξ.θ), 0.0
end

function rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, fd::FiniteDiffVHV)
    x = state.ξ.x
    θ = state.ξ.θ
    ∇U = fd.grad(x)
    copyto!(fd.grad_buf, ∇U)
    vhv = _restore_reference_vhv(_fd_vhv_scalar(fd, x, θ, θ), θ, flow)
    return dot(fd.grad_buf, θ), vhv
end

function rate_and_derivative(
    state::AbstractPDMPState,
    flow::BouncyParticle,
    fd::FiniteDiffVHV,
    cached_gradient::AbstractVector,
)
    copyto!(fd.grad_buf, cached_gradient)
    x = state.ξ.x
    θ = state.ξ.θ
    vhv = _restore_reference_vhv(_fd_vhv_scalar(fd, x, θ, θ), θ, flow)
    return dot(fd.grad_buf, θ), vhv
end

# The canonical sticking_time for BouncyParticle is defined in
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

    refresh_time = rand_refresh_time(Random.default_rng(), flow)
    # refresh_time = _ispositive(refresh_rate(flow)) ? rand(Exponential(inv(refresh_rate(flow)))) : Inf
    # refresh_time = flow.λref > 0 ? flow.λref : Inf
    return (a, b, refresh_time)
end
