"""
    ParameterTransform

Abstract supertype for scalar parameter transforms mapping unconstrained ℝ to constrained space.
"""
abstract type ParameterTransform end

"""
    IdentityTransform <: ParameterTransform

No transformation; constrained and unconstrained spaces are identical.
"""
struct IdentityTransform <: ParameterTransform end

"""
    LowerBoundTransform <: ParameterTransform

Transform for lower-bounded parameters: ``f(y) = L + \\exp(y)``.
"""
struct LowerBoundTransform <: ParameterTransform
    lower::Float64
end

"""
    UpperBoundTransform <: ParameterTransform

Transform for upper-bounded parameters: ``f(y) = U - \\exp(y)``.
"""
struct UpperBoundTransform <: ParameterTransform
    upper::Float64
end

"""
    DoubleBoundTransform <: ParameterTransform

Transform for double-bounded parameters: ``f(y) = L + (U - L) \\cdot \\text{logistic}(y)``.
"""
struct DoubleBoundTransform <: ParameterTransform
    lower::Float64
    upper::Float64
    function DoubleBoundTransform(lower::Real, upper::Real)
        lower < upper || throw(ArgumentError("lower ($lower) must be less than upper ($upper)"))
        new(Float64(lower), Float64(upper))
    end
end

# Forward transforms (unconstrained → constrained)
(::IdentityTransform)(y::Real)     = Float64(y)
(t::LowerBoundTransform)(y::Real)  = t.lower + exp(y)
(t::UpperBoundTransform)(y::Real)  = t.upper - exp(y)
(t::DoubleBoundTransform)(y::Real) = t.lower + (t.upper - t.lower) * LogExpFunctions.logistic(y)

"""
    inv_transform(t::ParameterTransform, x::Real)

Apply the inverse transform (constrained → unconstrained).
"""
inv_transform(::IdentityTransform, x::Real)     = Float64(x)
inv_transform(t::LowerBoundTransform, x::Real)  = log(x - t.lower)
inv_transform(t::UpperBoundTransform, x::Real)   = log(t.upper - x)
inv_transform(t::DoubleBoundTransform, x::Real)  = LogExpFunctions.logit((x - t.lower) / (t.upper - t.lower))

"""
    is_increasing(t::ParameterTransform)

Return `true` if the transform is monotonically increasing, `false` if decreasing.
"""
is_increasing(::IdentityTransform)    = true
is_increasing(::LowerBoundTransform)  = true
is_increasing(::UpperBoundTransform)  = false
is_increasing(::DoubleBoundTransform) = true

# ──────────────────────────────────────────────────────────────────────────────
# 5-point Gauss-Legendre quadrature on [-1, 1]
# ──────────────────────────────────────────────────────────────────────────────

const _GL5_NODES = (
    -0.9061798459386640,
    -0.5384693101056831,
     0.0,
     0.5384693101056831,
     0.9061798459386640,
)

const _GL5_WEIGHTS = (
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
)

"""
    _gl5_integrate(f, dt::Float64)

Integrate `f(s)` over `[0, dt]` using 5-point Gauss-Legendre quadrature.
"""
function _gl5_integrate(f, dt::Float64)
    half = dt / 2
    s = 0.0
    @inbounds for k in 1:5
        s += _GL5_WEIGHTS[k] * f(half + half * _GL5_NODES[k])
    end
    return s * half
end
