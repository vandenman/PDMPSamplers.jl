# ──────────────────────────────────────────────────────────────────────────────
# Coordinate interpolation (used by GL-5 fallback)
# ──────────────────────────────────────────────────────────────────────────────

_interpolate_coord(::Union{ZigZag,BouncyParticle}, x0j::Float64, θ0j::Float64, s::Float64, ::Float64) = x0j + θ0j * s

function _interpolate_coord(::Boomerang, x0j::Float64, θ0j::Float64, s::Float64, μj::Float64)
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
    _gl5_integrate(dt) do s
        transform(_interpolate_coord(base, x0j, θ0j, s, μj))
    end
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
    ::IdentityTransform, ::Boomerang,
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
    _gl5_integrate(dt) do s
        (transform(_interpolate_coord(base, x0j, θ0j, s, μj)) - μfj)^2
    end
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
    ::IdentityTransform, ::Boomerang,
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
    flow = trace.flow
    base = _underlying_flow(flow)

    iter = trace
    next = iterate(iter)
    isnothing(next) && error("Cannot compute statistics on an empty trace")

    t₀, x_state, θ_state, _ = next[2]
    xt₀ = copy(x_state)
    θt₀ = copy(θ_state)

    d = length(xt₀)
    length(transforms) == d || throw(DimensionMismatch("transforms length $(length(transforms)) ≠ dimension $d"))

    start_time = t₀
    integral = zeros(d)

    next = iterate(iter, next[2])
    isnothing(next) && error("Cannot compute statistics on a trace with fewer than 2 events")

    is_boomerang = base isa Boomerang
    while next !== nothing
        t₁, x_state, θ_state, _ = next[2]
        dt = t₁ - t₀

        for j in 1:d
            μj = is_boomerang ? base.μ[j] : 0.0
            integral[j] += _transformed_mean_segment(transforms[j], base, Float64(xt₀[j]), Float64(θt₀[j]), dt, μj)
        end

        t₀ = t₁
        copyto!(xt₀, x_state)
        copyto!(θt₀, θ_state)
        next = iterate(iter, next[2])
    end

    total_time = t₀ - start_time
    return integral / total_time
end

"""
    Statistics.var(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform})

Compute ``\\mathrm{Var}[f_j(y_j)]`` for each coordinate.
"""
function Statistics.var(trace::AbstractPDMPTrace, transforms::AbstractVector{<:ParameterTransform})
    μf = Statistics.mean(trace, transforms)

    flow = trace.flow
    base = _underlying_flow(flow)

    iter = trace
    next = iterate(iter)
    isnothing(next) && error("Cannot compute statistics on an empty trace")

    t₀, x_state, θ_state, _ = next[2]
    xt₀ = copy(x_state)
    θt₀ = copy(θ_state)

    d = length(xt₀)
    start_time = t₀
    integral = zeros(d)

    next = iterate(iter, next[2])
    isnothing(next) && error("Cannot compute statistics on a trace with fewer than 2 events")

    is_boomerang = base isa Boomerang
    while next !== nothing
        t₁, x_state, θ_state, _ = next[2]
        dt = t₁ - t₀

        for j in 1:d
            μj = is_boomerang ? base.μ[j] : 0.0
            integral[j] += _transformed_var_segment(transforms[j], base, Float64(xt₀[j]), Float64(θt₀[j]), dt, μj, μf[j])
        end

        t₀ = t₁
        copyto!(xt₀, x_state)
        copyto!(θt₀, θ_state)
        next = iterate(iter, next[2])
    end

    total_time = t₀ - start_time
    return integral / total_time
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
