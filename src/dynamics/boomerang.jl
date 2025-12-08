struct Boomerang{U, T, S, LT} <: NonFactorizedDynamics
    Γ::U
    μ::T
    λref::S
    ρ::S
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

    # own implementation:
    # TODO: can reduce allocations here, but ldiv! is only in 1.12
    # θ = ξ.θ
    # θ .-= (2 * dot(∇ϕ, θ) / normsq(flow.L \ ∇ϕ)) * (flow.L' \ (flow.L \ ∇ϕ))
    # but right now it ignores flow.L!

    # claude
    θ = ξ.θ

    # θ = copy(θ0)
    # Reflect: θ_new = θ - 2(θ·∇ϕ)∇ϕ/||∇ϕ||²_Γ⁻¹
    Γ_inv_grad = flow.Γ \ ∇ϕ
    reflection_coeff = 2 * dot(θ, ∇ϕ) / dot(∇ϕ, Γ_inv_grad)
    θ .-= reflection_coeff .* Γ_inv_grad
    θ

    # these are equal
    # θ1 = θ .- reflection_coeff .* Γ_inv_grad
    # θ2 = θ .- (2*dot(∇ϕ, θ)/normsq(flow.L\∇ϕ))*(flow.L'\(flow.L\∇ϕ))
    # θ3 = θ .- (2 * dot(∇ϕ, θ) / normsq(flow.L \ ∇ϕ)) * (flow.L' \ (flow.L \ ∇ϕ))


    # TODO: this is not entirely accurate, it should implement
    # θ = ξ.θ
    # # Reflect: v_new = v - 2(v·n̂)n̂ where n̂ = ∇ϕ/||∇ϕ||
    # grad_norm_sq = sum(abs2, ∇ϕ)
    # if grad_norm_sq > 0 # isn't this always true? unless ∇ϕ is zero?? and there is probability zero to be there?
    #     reflection_coeff = 2 * dot(θ, ∇ϕ) / grad_norm_sq
    #     θ .-= reflection_coeff .* ∇ϕ
    # end
    # θ
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
    if μ == 0
        if θ*x >= 0.0
            return π - atan(x/θ)
        else
            return atan(-x/θ)
        end
    else
        u = x^2 - 2μ*x + θ^2
        u < 0 && return Inf
        t1 = mod(2atan((sqrt(u) - θ)/(2μ - x)), 2pi)
        t2 = mod(-2atan((sqrt(u) + θ)/(2μ - x)), 2pi)
        x == 0 && return max(t1, t2)
        return min(t1, t2)
    end
end

# Bounds computation for Boomerang
# function ab(ξ::SkeletonPoint, c::AbstractVector, flow::Boomerang, cache)

#     x, θ = ξ.x, ξ.θ
#     cc = maximum(c)
#     return (sqrt(normsq(θ) + normsq((x - flow.μ)))*cc, 0.0, Inf)

#     x, θ = ξ.x, ξ.θ
#     μ = flow.μ
#     Γ = flow.Γ

#     c_val = maximum(c)

#     # For Boomerang with Gaussian target, we can compute an exact bound
#     # The rate is λ(t) = max(0, (Γ(x(t)-μ))·θ(t))

#     # With x(t) = (x₀-μ)cos(t) + θ₀sin(t) + μ
#     #      θ(t) = -(x₀-μ)sin(t) + θ₀cos(t)

#     x_centered = x - μ

#     # The dot product becomes:
#     # (Γ((x₀-μ)cos(t) + θ₀sin(t))) · (-(x₀-μ)sin(t) + θ₀cos(t))
#     # = (Γ(x₀-μ))·θ₀ cos²(t) + (Γθ₀)·(x₀-μ) sin²(t) +
#     #   ((Γ(x₀-μ))·(x₀-μ) - (Γθ₀)·θ₀) cos(t)sin(t)

#     Γx = Γ * x_centered
#     Γθ = Γ * θ

#     # Coefficients of the trigonometric expansion
#     # A = dot(Γx, θ)      # coefficient of cos²(t)
#     # B = dot(Γθ, x_centered)  # coefficient of sin²(t)
#     # C = dot(Γx, x_centered) - dot(Γθ, θ)  # coefficient of cos(t)sin(t)

#     # CORRECTED Coefficients of the trigonometric expansion
#     A = dot(Γx, θ)
#     B = -dot(Γθ, x_centered)  # Fixed: Added minus sign
#     C = dot(Γθ, θ) - dot(Γx, x_centered) # Fixed: Swapped order

#     # The rate function is λ(t) = max(0, A cos²(t) + B sin²(t) + C cos(t)sin(t))
#     # This can be rewritten as λ(t) = max(0, P + Q cos(2t) + R sin(2t))
#     # where P = (A+B)/2, Q = (A-B)/2, R = C/2

#     P = (A + B) / 2
#     Q = (A - B) / 2
#     R = C / 2

#     # Maximum of P + Q cos(2t) + R sin(2t) is P + sqrt(Q² + R²)
#     max_rate = P + sqrt(Q^2 + R^2)

#     # The bound is max(0, max_rate)
#     a = c_val + max(0.0, max_rate)
#     b = 0.0  # No linear growth in time

#     refresh_time = flow.λref > 0 ? rand(Exponential(inv(flow.λref))) : Inf

#     return (a, b, refresh_time)
# end

function ab(ξ::SkeletonPoint, c::AbstractVector, flow::Boomerang, cache)

    x, θ = ξ.x, ξ.θ
    μ = flow.μ
    Γ = flow.Γ

    c_val = maximum(c)

    x_centered = cache.z
    Γx = cache.Γx
    Γθ = cache.Γθ


    # For Boomerang with Gaussian target, we can compute an exact bound
    # The rate is λ(t) = max(0, (Γ(x(t)-μ))·θ(t))

    # With x(t) = (x₀-μ)cos(t) + θ₀sin(t) + μ
    #      θ(t) = -(x₀-μ)sin(t) + θ₀cos(t)

    x_centered .= x .- μ


    # The dot product becomes:
    # (Γ((x₀-μ)cos(t) + θ₀sin(t))) · (-(x₀-μ)sin(t) + θ₀cos(t))
    # = (Γ(x₀-μ))·θ₀ cos²(t) + (Γθ₀)·(x₀-μ) sin²(t) +
    #   ((Γ(x₀-μ))·(x₀-μ) - (Γθ₀)·θ₀) cos(t)sin(t)

    # Γx = Γ * x_centered
    # Γθ = Γ * θ
    mul!(Γx, Γ, x_centered)
    mul!(Γθ, Γ, θ)

    # Coefficients of the trigonometric expansion
    # A = dot(Γx, θ)      # coefficient of cos²(t)
    # B = dot(Γθ, x_centered)  # coefficient of sin²(t)
    # C = dot(Γx, x_centered) - dot(Γθ, θ)  # coefficient of cos(t)sin(t)

    # CORRECTED Coefficients of the trigonometric expansion
    A = dot(Γx, θ)
    B = -dot(Γθ, x_centered)  # Fixed: Added minus sign
    C = dot(Γθ, θ) - dot(Γx, x_centered) # Fixed: Swapped order

    # The rate function is λ(t) = max(0, A cos²(t) + B sin²(t) + C cos(t)sin(t))
    # This can be rewritten as λ(t) = max(0, P + Q cos(2t) + R sin(2t))
    # where P = (A+B)/2, Q = (A-B)/2, R = C/2

    P = (A + B) / 2
    Q = (A - B) / 2
    R = C / 2

    # Maximum of P + Q cos(2t) + R sin(2t) is P + sqrt(Q² + R²)
    max_rate = P + sqrt(Q^2 + R^2)

    # The bound is max(0, max_rate)
    a = c_val + pos(max_rate)
    b = 0.0  # No linear growth in time

    # Infiltrator.@infiltrate

    refresh_time = ispositive(flow.λref) ? rand(Exponential(inv(flow.λref))) : Inf

    return (a, b, refresh_time)
end

function correct_gradient!(∇ϕ::AbstractVector, x::AbstractVector, ::AbstractVector, flow::Boomerang, cache)
    # For Boomerang dynamics, the corrected gradient is: ∇U(x) + Γ(x - μ)
    # This ensures sampling from the correct distribution with precision Γ
    z = cache.z
    z .= x .- flow.μ
    # mul!(∇ϕ, flow.Γ, z, one(eltype(∇ϕ)), one(eltype(∇ϕ)))  # ∇ϕ += Γ(x-μ)
    mul!(∇ϕ, flow.Γ, z, -one(eltype(∇ϕ)), one(eltype(∇ϕ)))  # ∇ϕ -= Γ(x-μ)
    ∇ϕ

    # z = cache.z
    # z .= x .- flow.μ
    # ldiv!(flow.L, z)
    # ldiv!(flow.L', z)
    # ∇ϕ .+= z

    # mul!(∇ϕ, flow.Γ, z, -one(eltype(∇ϕ)), one(eltype(∇ϕ)))
    # mul!(∇ϕ, flow.Γ, z, one(eltype(∇ϕ)), one(eltype(∇ϕ)))

    # ∇ϕ0 = randn(length(x))
    # ∇ϕ = copy(∇ϕ0)
    # y = copy(∇ϕ)

    # y .-= (F.L'\(F.L\(x - F.μ)))
    # y

    ∇ϕ
end
