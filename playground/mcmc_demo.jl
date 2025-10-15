using LinearAlgebra, Distributions, QuadGK
import SpecialFunctions
# ------------------------------------------------------------------
#   Donut distribution (Chi-Feng MCMC demo)
# ------------------------------------------------------------------
struct Donut <: ContinuousMultivariateDistribution
    radius::Float64
    sigma2::Float64
end
Donut() = Donut(3.0, 1.0)
Base.size(::Donut) = (2,)

# --- logpdf and pdf ---
function Distributions.logpdf(d::Donut, x::AbstractVector)
    @assert length(x) == 2
    r = sqrt(x[1]^2 + x[2]^2)
    return -((r - d.radius)^2) / d.sigma2
end


# --- gradient of logpdf ---
function Distributions.gradlogpdf(d::Donut, x::AbstractVector)
    @assert length(x) == 2
    r = sqrt(x[1]^2 + x[2]^2)
    if iszero(r)
        return zeros(2)
    end
    scale = (2 / d.sigma2) * (d.radius / r - 1)
    return scale .* x
end

# Hessian-vector product for Donut: H·v
"""
    hessian_vector_product(d::Donut, x, v)

Compute (∇^2 log p(x)) * v efficiently without forming the full Hessian.
"""
function hessian_vector_product(d::Donut, x::AbstractVector, v::AbstractVector)
    @assert length(x)==2 && length(v)==2
    r = sqrt(x[1]^2 + x[2]^2)
    if r == 0.0
        return zeros(2)
    end
    # f'(r), f''(r) for f(r) = - (r-R)^2 / sigma2
    fprime = -2*(r - d.radius) / d.sigma2
    fpp = -2.0 / d.sigma2

    # Hv = (f'(r)/r) * v + (f''(r) - f'(r)/r) * x * ((x⋅v)/r^2)
    alpha = fprime / r
    beta = (fpp - fprime / r) / (r^2)   # multiply by (x * (x·v))
    dot_x_v = dot(x, v)
    return alpha .* v .+ (beta * dot_x_v) .* x
end


# --- numeric marginals (unnormalized) ---
# function marginal_x(d::Donut, x::Real; lim=6)
#     f(y) = pdf(d, [x, y])
#     return quadgk(f, -lim, lim; rtol=1e-6)[1]
# end

# function marginal_y(d::Donut, y::Real; lim=6)
#     f(x) = pdf(d, [x, y])
#     return quadgk(f, -lim, lim; rtol=1e-6)[1]
# end

# normalization constant Z (exact)
function normalization_constant(d::Donut)
    r, s = d.radius, d.sigma2
    I = 0.5*s*exp(-r^2/s) + 0.5*r*sqrt(pi*s)*(1 + SpecialFunctions.erf(r/sqrt(s)))
    return 2π * I   # = Z from the derivation above
end

# normalized marginal via cosh parameterization (B)
function marginal_x(d::Donut, x::Real)
    Z = normalization_constant(d)
    r, s = d.radius, d.sigma2
    ax = abs(x)
    iszero(x) || ax <= eps(x) && return oftype(x, √(π * s) * (1 + SpecialFunctions.erf(r/sqrt(s))))

    # @show x
    integrand(ρ) = ρ * exp(-((ρ - r)^2) / s) / sqrt(ρ^2 - ax^2)
    T = typeof(x)
    val1, _ = quadgk(integrand, T(ax + 1e-8), T(r + 10*sqrt(s)))
    val2, _ = quadgk(integrand, T(r + 10*sqrt(s)), T(Inf))
    return 2*(val1 + val2) / Z
end

# d1 = 5.880000153450621
# f1 = 5.88f0
# d1 > f1
# d1^2 > Float64(f1)^2
# d1^2 > f1^2
# 🤮

marginal_y(d::Donut, y::Real) = marginal_x(d, y)


# ------------------------------------------------------------------
#   Banana distribution (Chi-Feng MCMC demo)
# ------------------------------------------------------------------
struct Banana <: ContinuousMultivariateDistribution
    a::Float64
    b::Float64
    base::MvNormal
end

function Banana(; a=2.0, b=0.2)
    μ = [0.0, 4.0]
    Σ = [1.0 0.5; 0.5 1.0]
    base = MvNormal(μ, Σ)
    return Banana(a, b, base)
end

Base.size(::Banana) = (2,)

# --- logpdf and pdf ---
function Distributions.logpdf(bn::Banana, x::AbstractVector)
    @assert length(x) == 2
    a, b, base = bn.a, bn.b, bn.base
    y₁ = x[1] / a
    y₂ = a * x[2] + a * b * (abs2(x[1]) + abs2(a))
    return Distributions.logpdf(base, [y₁, y₂])
end

# --- gradient of logpdf ---
function Distributions.gradlogpdf(bn::Banana, x::AbstractVector)
    @assert length(x) == 2
    a, b, base = bn.a, bn.b, bn.base

    y₁ = x[1] / a
    y₂ = a * x[2] + a * b * (x[1]^2 + a^2)

    # gradient of log p(y) w.r.t. y
    grad_y = gradlogpdf(base, [y₁, y₂])

    # chain rule
    ∂x1 = grad_y[1] / a + grad_y[2] * a * b * 2 * x[1]
    ∂x2 = grad_y[2] * a

    return [∂x1, ∂x2]
end

# Hessian-vector product for banana
"""
    hessian_vector_product(bn::Banana, x, v)

Compute (∇^2_x log p(x)) * v using chain rule:
  H_x = J^T H_y J + sum_k g_k ∇^2_x y_k
Then H_x * v = J^T (H_y * (J*v)) + sum_k g_k (∇^2_x y_k * v)
For our transformation, only y2 has nonzero Hessian: ∇^2_x y2 = [2ab 0; 0 0].
"""
function hessian_vector_product(bn::Banana, x::AbstractVector, v::AbstractVector)
    @assert length(x)==2 && length(v)==2
    a, b = bn.a, bn.b
    # Jacobian J
    J = [
        1.0/a  0.0;
        2.0*a*b*x[1]  a
    ]

    # compute J*v
    t = J * v   # 2-vector

    # For base Gaussian: H_y = -Σ^{-1}
    Σ = bn.base.Σ
    Σinv = inv(Σ)          # 2x2; cheap for 2x2
    H_y = -Σinv

    # term1 = J^T * (H_y * t)
    term1 = J' * (H_y * t)

    # term2 = sum_k g_k (H_x(y_k) * v). Only k=2 contributes:
    # H_x(y2) = [2ab 0; 0 0]  => H_x(y2)*v = [2ab*v1, 0]
    # g = grad_y log p(y)
    a1 = x[1] / a
    a2 = a * x[2] + a * b * (x[1]^2 + a^2)
    g = gradlogpdf(bn.base, [a1, a2])
    g2 = g[2]
    term2 = g2 .* [2.0*a*b*v[1], 0.0]

    return term1 .+ term2
end

# --- numeric marginals (unnormalized) ---
function marginal_x(bn::Banana, x::Real; lim=6)
    f(y) = pdf(bn, [x, y])
    return quadgk(f, -lim, lim; rtol=1e-6)[1]
end

function marginal_y(bn::Banana, y::Real; lim=6)
    f(x) = pdf(bn, [x, y])
    return quadgk(f, -lim, lim; rtol=1e-6)[1]
end
