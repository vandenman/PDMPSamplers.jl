# TODO: these fields are not necessarily part of the dynamics, but rather used in the thinning algorithm...
# maybe they belong there?

struct ZigZag{T,S,S2,R} <: FactorizedDynamics
    Γ::T
    μ::S
    σ::S2       # reserved for partial-refreshment features
    λref::Float64
    ρ::Float64  # reserved for partial-refreshment features
    ρ̄::R        # reserved for partial-refreshment features
end
ZigZag(Γ, μ, σ=(Vector(diag(Γ))).^(-0.5); λref=0.0, ρ=0.0) = ZigZag(Γ, μ, σ, λref, ρ, sqrt(1-ρ^2))
ZigZag(d::Integer) = ZigZag(I(d), zeros(d))

refresh_rate(::ZigZag) = 0.0
initialize_velocity(::ZigZag, d::Integer) = rand((-1., 1.), d)
refresh_velocity!(::SkeletonPoint, ::ZigZag) = nothing
function reflect!(ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::ZigZag, cache)

    θ = ξ.θ
    # Single-pass weighted sampling: compute cumulative sum on-the-fly
    total_rate = zero(eltype(∇ϕ))
    for i in eachindex(θ)
        total_rate += λ_i(i, ξ, ∇ϕ[i], flow)
    end
    if ispositive(total_rate)
        u = rand() * total_rate
        cumsum = zero(total_rate)
        i₀ = firstindex(θ)
        for i in eachindex(θ)
            cumsum += λ_i(i, ξ, ∇ϕ[i], flow)
            if cumsum >= u
                i₀ = i
                break
            end
        end
    else
        i₀ = rand(eachindex(θ))
    end
    θ[i₀] = -θ[i₀]
    return i₀
end

λ(ξ::SkeletonPoint, ∇ϕ::AbstractVector, flow::ZigZag)   = sum(i->λ_i(i, ξ, ∇ϕ[i], flow), eachindex(ξ.θ))
λ_i(i::Integer, ξ::SkeletonPoint, ∇ϕ_i::Real, ::ZigZag) = pos(ξ.θ[i] * ∇ϕ_i)


function move_forward_time!(state::AbstractPDMPState, τ::Real, flow::ZigZag)
    state.t[] += τ
    move_forward_time!(state.ξ, τ, flow)
    state
end

# this signature is a bit ugly, but avoids some errors...
function move_forward_time!(ξ::SkeletonPoint, τ::Real, ::ZigZag)
    # LinearAlgebra.axpy!(τ, ξ.θ, ξ.x)
    ξ.x .+= τ .* ξ.θ
end

# fallbacks for AD (mostly ForwardDiff) that allocates new arrays
# move_forward_time(ξ::SkeletonPoint, τ::Real, ::ZigZag) = SkeletonPoint(ξ.x .+ τ .* ξ.θ, ξ.θ)


# factorized
function reflect!(ξ::SkeletonPoint, ∇ϕ::Real, i::Integer, flow::ZigZag)
    ξ.θ[i] = -ξ.θ[i]
    return i
end

# The canonical freezing_time for ZigZag is defined in
# src/poisson_time_strategies/sticky.jl, dispatching on Union{BouncyParticle,ZigZag}.

# actually part of thinning
"""
    ab(x, θ, c, flow)

Compute the `(a,b)` parameters for the upper bound on the rate of the flow.
"""
function ab(ξ::SkeletonPoint, c::AbstractVector, flow::ZigZag, cache)
    a, b = mapreduce(
        i -> ab_i(i, ξ, c, flow, cache),
        (acc, val) -> (acc[1] + val[1], acc[2] + val[2]),
        eachindex(ξ.x)
    )
    return (a, b, Inf)
end

"""
    ab_i(i, x, θ, c, flow)

Compute the `(a,b)` parameters for the upper bound on the rate of the
`i`-th coordinate for the ZigZag flow.
"""
function ab_i(i::Int, ξ::SkeletonPoint, c::AbstractVector, flow::ZigZag, cache)

    x, θ = ξ.x, ξ.θ

    # # This is the i-th component of the sum in the global `ab` function
    # a_i = pos((idot(flow.Γ, i, x) - idot(flow.Γ, i, flow.μ))' * θ[i]) + c[i]
    # b_i = pos(θ[i]' * idot(flow.Γ, i, θ)) + c[i]/100 # `pos` for safety, b should be non-negative

    grad_term = θ[i] * (idot(flow.Γ, i, x) - idot(flow.Γ, i, flow.μ))
    hess_term = θ[i] * idot(flow.Γ, i, θ)

    a_i = pos(grad_term) + c[i]
    b_i = pos(hess_term)# + c[i]  # Remove /100, add abs()

    return (a_i, b_i)
end
