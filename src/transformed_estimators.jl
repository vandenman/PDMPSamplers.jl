# ──────────────────────────────────────────────────────────────────────────────
# Coordinate interpolation (used by GL-5 fallback)
# ──────────────────────────────────────────────────────────────────────────────

_interpolate_coord(::Union{ZigZag,BouncyParticle}, x0j::Float64, θ0j::Float64, s::Float64, ::Float64) = x0j + θ0j * s

function _interpolate_coord(::AnyBoomerang, x0j::Float64, θ0j::Float64, s::Float64, μj::Float64)
    μj + (x0j - μj) * cos(s) + θ0j * sin(s)
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-coordinate transformed mean: segment integrals
# ──────────────────────────────────────────────────────────────────────────────

# GL-5 fallback (any transform × any dynamics)
function _transformed_mean_segment(
    transform::ParameterTransform, base::ContinuousDynamics,
    x0j::Float64, θ0j::Float64, dt::Float64, μj::Float64
)
    _gl5_integrate_eval(transform, base, x0j, θ0j, dt, μj)
end

# Identity × Linear: closed-form
function _transformed_mean_segment(
    ::IdentityTransform, ::Union{ZigZag,BouncyParticle},
    x0j::Float64, θ0j::Float64, dt::Float64, ::Float64
)
    x0j * dt + θ0j * dt^2 / 2
end

# Identity × Boomerang: closed-form
function _transformed_mean_segment(
    ::IdentityTransform, ::AnyBoomerang,
    x0j::Float64, θ0j::Float64, dt::Float64, μj::Float64
)
    Δ = x0j - μj
    s, c = sincos(dt)
    Δ * s + θ0j * (1 - c) + μj * dt
end

# LowerBound × Linear: closed-form
function _transformed_mean_segment(
    transform::LowerBoundTransform, ::Union{ZigZag,BouncyParticle},
    x0j::Float64, θ0j::Float64, dt::Float64, ::Float64
)
    L = transform.lower
    if iszero(θ0j)
        return (L + exp(x0j)) * dt
    end
    L * dt + exp(x0j) * expm1(θ0j * dt) / θ0j
end

# UpperBound × Linear: closed-form
function _transformed_mean_segment(
    transform::UpperBoundTransform, ::Union{ZigZag,BouncyParticle},
    x0j::Float64, θ0j::Float64, dt::Float64, ::Float64
)
    U = transform.upper
    if iszero(θ0j)
        return (U - exp(x0j)) * dt
    end
    U * dt - exp(x0j) * expm1(θ0j * dt) / θ0j
end

# DoubleBound × Linear: closed-form via log1pexp
function _transformed_mean_segment(
    transform::DoubleBoundTransform, ::Union{ZigZag,BouncyParticle},
    x0j::Float64, θ0j::Float64, dt::Float64, ::Float64
)
    L, U = transform.lower, transform.upper
    if iszero(θ0j)
        return (L + (U - L) * LogExpFunctions.logistic(x0j)) * dt
    end
    L * dt + (U - L) / θ0j * (LogExpFunctions.log1pexp(x0j + θ0j * dt) - LogExpFunctions.log1pexp(x0j))
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-coordinate transformed variance: segment integrals
# ──────────────────────────────────────────────────────────────────────────────

# GL-5 fallback (any transform × any dynamics)
function _transformed_var_segment(
    transform::ParameterTransform, base::ContinuousDynamics,
    x0j::Float64, θ0j::Float64, dt::Float64, μj::Float64, μfj::Float64
)
    _gl5_integrate_var_eval(transform, base, x0j, θ0j, dt, μj, μfj)
end

# Identity × Linear: closed-form
function _transformed_var_segment(
    ::IdentityTransform, ::Union{ZigZag,BouncyParticle},
    x0j::Float64, θ0j::Float64, dt::Float64, ::Float64, μfj::Float64
)
    y0 = x0j - μfj
    dt * (y0^2 + y0 * θ0j * dt + θ0j^2 * dt^2 / 3)
end

# Identity × Boomerang: closed-form
function _transformed_var_segment(
    ::IdentityTransform, ::AnyBoomerang,
    x0j::Float64, θ0j::Float64, dt::Float64, μj::Float64, μfj::Float64
)
    a = x0j - μj
    b = θ0j
    c0 = μj - μfj
    s, c = sincos(dt)
    s2, _ = sincos(2dt)
    a^2 * (dt / 2 + s2 / 4) + b^2 * (dt / 2 - s2 / 4) + c0^2 * dt + a * b * s^2 + 2 * a * c0 * s + 2 * b * c0 * (1 - c)
end

# LowerBound × Linear: closed-form
# (L + exp(y) - μf)² = (L-μf)² + 2(L-μf)exp(y) + exp(2y)
function _transformed_var_segment(
    transform::LowerBoundTransform, ::Union{ZigZag,BouncyParticle},
    x0j::Float64, θ0j::Float64, dt::Float64, ::Float64, μfj::Float64
)
    L = transform.lower
    c = L - μfj
    if iszero(θ0j)
        ey = exp(x0j)
        return (c^2 + 2c * ey + ey^2) * dt
    end
    v = θ0j
    c^2 * dt + 2c * exp(x0j) * expm1(v * dt) / v + exp(2x0j) * expm1(2v * dt) / (2v)
end

# UpperBound × Linear: closed-form
# (U - exp(y) - μf)² = (U-μf)² - 2(U-μf)exp(y) + exp(2y)
function _transformed_var_segment(
    transform::UpperBoundTransform, ::Union{ZigZag,BouncyParticle},
    x0j::Float64, θ0j::Float64, dt::Float64, ::Float64, μfj::Float64
)
    U = transform.upper
    c = U - μfj
    if iszero(θ0j)
        ey = exp(x0j)
        return (c^2 - 2c * ey + ey^2) * dt
    end
    v = θ0j
    c^2 * dt - 2c * exp(x0j) * expm1(v * dt) / v + exp(2x0j) * expm1(2v * dt) / (2v)
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    Statistics.mean(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform})

Compute ``\\mathrm{E}[f_j(y_j)]`` for each coordinate using exact piecewise integration.
Uses closed-form formulas for common transforms on linear dynamics (ZigZag/BPS) and
5-point Gauss-Legendre quadrature elsewhere.
"""
function Statistics.mean(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform})
    d = length(first(trace).position)
    length(transforms) == d || throw(DimensionMismatch("transforms length $(length(transforms)) ≠ dimension $d"))
    return _integrate(trace, _transformed_mean_segment, transforms)
end

"""
    Statistics.var(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform})

Compute the variance of each transformed coordinate using the trace's sticky masks.
"""
function Statistics.var(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform})
    centers = Statistics.mean(trace, transforms)
    return _integrate(trace, _transformed_var_segment, transforms, centers)
end

function _integrate_segment(
    f::Union{typeof(_transformed_mean_segment),typeof(_transformed_var_segment)},
    flow::Union{ZigZag,BouncyParticle,AnyBoomerang}, x0, x1, v0, v1, t0, t1, args...)
    integral = zeros(length(x0))
    return _integrate_segment!(integral, f, flow, x0, x1, v0, v1, t0, t1, args...)
end

function _integrate_segment!(integral,
    f::Union{typeof(_transformed_mean_segment),typeof(_transformed_var_segment)},
    flow::Union{ZigZag,BouncyParticle,AnyBoomerang}, x0, x1, v0, v1, t0, t1,
    transforms::AbstractVector{<:ParameterTransform}, args...)
    return _integrate_segment!(integral, f, flow, x0, x1, v0, v1, t0, t1,
        nothing, transforms, args...)
end

function _integrate_segment!(integral,
    f::Union{typeof(_transformed_mean_segment),typeof(_transformed_var_segment)},
    flow::Union{ZigZag,BouncyParticle,AnyBoomerang}, x0, x1, v0, v1, t0, t1,
    free::Union{Nothing,AbstractVector{Bool}}, transforms, centers=nothing)
    base = underlying_flow(flow)
    dt = Float64(t1 - t0)
    for j in eachindex(integral)
        if free !== nothing && !free[j]
            value = transforms[j](x0[j])
            integral[j] += (centers === nothing ? value : (value - centers[j])^2) * dt
        else
            reference = base isa AnyBoomerang ? Float64(base.μ[j]) : 0.0
            integral[j] += centers === nothing ?
                f(transforms[j], base, Float64(x0[j]), Float64(v0[j]), dt, reference) :
                f(transforms[j], base, Float64(x0[j]), Float64(v0[j]), dt, reference, centers[j])
        end
    end
    return integral
end

"""
    Statistics.std(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform})

Compute standard deviation of ``f_j(y_j)`` for each coordinate.
"""
Statistics.std(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform}) = sqrt.(Statistics.var(trace, transforms))

"""
    Statistics.quantile(trace::AbstractPDMPTrace, p::Real, transforms::AbstractVector{<:ParameterTransform};
                        coordinate::Integer=-1)

Compute the `p`-th quantile of ``f(y)`` using monotonicity of the transform.
When `coordinate=-1`, returns a vector of quantiles for all coordinates.
"""
function Statistics.quantile(
    trace::AbstractPDMPTrace, p::Real, transforms::AbstractVector{<:ParameterTransform};
    coordinate::Integer=-1
)
    (0 < p < 1) || throw(DomainError(p, "Quantile probability must be in (0, 1)"))
    if coordinate == -1
        d = length(first(trace).position)
        return [_constrained_quantile_scalar(trace, p, j, transforms[j]) for j in 1:d]
    end
    return _constrained_quantile_scalar(trace, p, coordinate, transforms[coordinate])
end

function _constrained_quantile_scalar(trace::AbstractPDMPTrace, p::Real, j::Integer, transform::ParameterTransform)
    p_adj = is_increasing(transform) ? p : 1 - p
    raw = Statistics.quantile(trace, p_adj; coordinate=j)
    return transform(raw)
end

"""
    Statistics.median(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform};
                      coordinate::Integer=-1)

Compute the median of ``f(y)`` for each coordinate.
"""
function Statistics.median(
    trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform};
    coordinate::Integer=-1
)
    Statistics.quantile(trace, 0.5, transforms; coordinate)
end

"""
    cdf(trace::AbstractPDMPTrace, q::Real, transforms::AbstractVector{<:ParameterTransform};
        coordinate::Integer)

Compute ``\\Pr[f_j(y_j) \\le q]`` for coordinate `coordinate`.
"""
function cdf(
    trace::AbstractPDMPTrace, q::Real, transforms::AbstractVector{<:ParameterTransform};
    coordinate::Integer
)
    transform = transforms[coordinate]
    y_q = inv_transform(transform, q)
    c = cdf(trace, y_q; coordinate)
    return is_increasing(transform) ? c : 1 - c
end
