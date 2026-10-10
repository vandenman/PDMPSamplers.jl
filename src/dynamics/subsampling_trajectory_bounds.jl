# Likelihood-agnostic trajectory geometry for subsampling residual envelopes.

function _linear_displacement(state, anchor, t)
    value = zero(promote_type(eltype(state.ξ.x), eltype(anchor), typeof(t)))
    @inbounds for j in eachindex(state.ξ.x, state.ξ.θ, anchor)
        δ = state.ξ.x[j] + t * state.ξ.θ[j] - anchor[j]
        value += δ * δ
    end
    return sqrt(value)
end

function _linear_displacement_cell(state, anchor, left, right)
    # The Euclidean norm of an affine path is convex.
    return max(_linear_displacement(state, anchor, left),
        _linear_displacement(state, anchor, right))
end

function _harmonic_abs_max(c, a, b, left, right)
    result = max(abs(c + a * cos(left) + b * sin(left)),
        abs(c + a * cos(right) + b * sin(right)))
    phase = atan(b, a)
    k_first = ceil(Int, (left - phase) / π)
    k_last = floor(Int, (right - phase) / π)
    for k in k_first:k_last
        t = phase + k * π
        result = max(result, abs(c + a * cos(t) + b * sin(t)))
    end
    return result
end

@inline _coordinate_is_free(::AbstractPDMPState, i) = true
@inline _coordinate_is_free(state::StickyPDMPState, i) = state.free[i]

function _boomerang_geometry(flow, state, anchor, t)
    s, c = sincos(t)
    displacement2 = zero(promote_type(eltype(state.ξ.x), eltype(anchor), typeof(t)))
    velocity2 = zero(displacement2)
    @inbounds for j in eachindex(state.ξ.x, state.ξ.θ, anchor, flow.μ)
        δ = if !_coordinate_is_free(state, j)
            state.ξ.x[j] - anchor[j]
        else
            flow.μ[j] - anchor[j] +
                (state.ξ.x[j] - flow.μ[j]) * c + state.ξ.θ[j] * s
        end
        displacement2 += δ * δ
        if _coordinate_is_free(state, j)
            velocity = -(state.ξ.x[j] - flow.μ[j]) * s + state.ξ.θ[j] * c
            velocity2 += velocity * velocity
        end
    end
    return sqrt(displacement2), sqrt(velocity2)
end

function _boomerang_geometry_cell(flow, state, anchor, left, right)
    displacement2 = zero(promote_type(eltype(state.ξ.x), eltype(anchor),
        typeof(left), typeof(right)))
    velocity2 = zero(displacement2)
    @inbounds for j in eachindex(state.ξ.x, state.ξ.θ, anchor, flow.μ)
        coordinate = if !_coordinate_is_free(state, j)
            abs(state.ξ.x[j] - anchor[j])
        else
            _harmonic_abs_max(flow.μ[j] - anchor[j],
                state.ξ.x[j] - flow.μ[j], state.ξ.θ[j], left, right)
        end
        displacement2 += coordinate * coordinate
        if _coordinate_is_free(state, j)
            velocity_coordinate = _harmonic_abs_max(0.0, state.ξ.θ[j],
                -(state.ξ.x[j] - flow.μ[j]), left, right)
            velocity2 += velocity_coordinate * velocity_coordinate
        end
    end
    return sqrt(displacement2), sqrt(velocity2)
end

trajectory_displacement_bound(::Union{BouncyParticle,ZigZag}, state, anchor, t) =
    _linear_displacement(state, anchor, t)
trajectory_displacement_bound(flow::AnyBoomerang, state, anchor, t) =
    first(_boomerang_geometry(flow, state, anchor, t))
trajectory_displacement_bound(flow::PreconditionedDynamics, state, anchor, t) =
    trajectory_displacement_bound(flow.dynamics, state, anchor, t)
trajectory_displacement_bound(flow::ContinuousDynamics, state, anchor, t) =
    throw(ArgumentError("no subsampling trajectory geometry for $(typeof(flow))"))

trajectory_displacement_cell_bound(::Union{BouncyParticle,ZigZag}, state,
        anchor, left, right) = _linear_displacement_cell(state, anchor, left, right)
trajectory_displacement_cell_bound(flow::AnyBoomerang, state, anchor, left, right) =
    first(_boomerang_geometry_cell(flow, state, anchor, left, right))
trajectory_displacement_cell_bound(flow::PreconditionedDynamics, state,
        anchor, left, right) = trajectory_displacement_cell_bound(
            flow.dynamics, state, anchor, left, right)
trajectory_displacement_cell_bound(flow::ContinuousDynamics, state,
        anchor, left, right) = throw(ArgumentError(
            "no subsampling trajectory geometry for $(typeof(flow))"))

_linear_rate_dual_bound(::ContinuousDynamics, state) = norm(state.ξ.θ)
_linear_rate_dual_bound(flow::DensePreconditionedZigZag, state) =
    sqrt(count(_coordinate_active(state, i, 0) for i in eachindex(state.ξ.x))) *
    flow.metric.L_operator_norm

residual_rate_dual_bound(flow::Union{BouncyParticle,ZigZag}, state, t) =
    _linear_rate_dual_bound(flow, state)
residual_rate_dual_bound(flow::AnyBoomerang, state, t) =
    last(_boomerang_geometry(flow, state, flow.μ, t))
residual_rate_dual_bound(flow::PreconditionedDynamics, state, t) =
    _residual_rate_dual_bound(flow, flow.dynamics, state, t)
_residual_rate_dual_bound(flow, ::Union{BouncyParticle,ZigZag}, state, t) =
    _linear_rate_dual_bound(flow, state)
_residual_rate_dual_bound(flow, dynamics::AnyBoomerang, state, t) =
    last(_boomerang_geometry(dynamics, state, dynamics.μ, t))
residual_rate_dual_bound(flow::ContinuousDynamics, state, t) = throw(ArgumentError(
    "no subsampling residual-rate geometry for $(typeof(flow))"))

residual_rate_dual_cell_bound(flow::Union{BouncyParticle,ZigZag}, state, left, right) =
    _linear_rate_dual_bound(flow, state)
residual_rate_dual_cell_bound(flow::AnyBoomerang, state, left, right) =
    last(_boomerang_geometry_cell(flow, state, flow.μ, left, right))
residual_rate_dual_cell_bound(flow::PreconditionedDynamics, state, left, right) =
    _residual_rate_dual_cell_bound(flow, flow.dynamics, state, left, right)
_residual_rate_dual_cell_bound(flow, ::Union{BouncyParticle,ZigZag}, state, left, right) =
    _linear_rate_dual_bound(flow, state)
_residual_rate_dual_cell_bound(flow, dynamics::AnyBoomerang, state, left, right) =
    last(_boomerang_geometry_cell(dynamics, state, dynamics.μ, left, right))
residual_rate_dual_cell_bound(flow::ContinuousDynamics, state, left, right) =
    throw(ArgumentError("no subsampling residual-rate geometry for $(typeof(flow))"))

trajectory_geometry_bounds(flow::ContinuousDynamics, state, anchor, t) =
    (trajectory_displacement_bound(flow, state, anchor, t),
        residual_rate_dual_bound(flow, state, t))
trajectory_geometry_bounds(flow::AnyBoomerang, state, anchor, t) =
    _boomerang_geometry(flow, state, anchor, t)

trajectory_geometry_cell_bounds(flow::ContinuousDynamics, state, anchor, left, right) =
    (trajectory_displacement_cell_bound(flow, state, anchor, left, right),
        residual_rate_dual_cell_bound(flow, state, left, right))
trajectory_geometry_cell_bounds(flow::AnyBoomerang, state, anchor, left, right) =
    _boomerang_geometry_cell(flow, state, anchor, left, right)

function _fill_trajectory_scales!(out, growth_rates, displacement, velocity)
    base = velocity * displacement
    @inbounds for r in eachindex(out, growth_rates)
        out[r] = base * exp(growth_rates[r] * displacement)
    end
    return out
end

function (scales::TrajectoryComponentScales)(out, state, flow::ContinuousDynamics, t)
    displacement, velocity = trajectory_geometry_bounds(flow, state, scales.anchor, t)
    return _fill_trajectory_scales!(out, scales.growth_rates, displacement, velocity)
end

function (scales::TrajectoryComponentScales)(out, state, flow::ContinuousDynamics,
        left, right)
    displacement, velocity = trajectory_geometry_cell_bounds(
        flow, state, scales.anchor, left, right)
    return _fill_trajectory_scales!(out, scales.growth_rates, displacement, velocity)
end

function _damped_hcv_scales!(out, velocity, displacement, damping)
    displacement2 = displacement * displacement
    α = damping / (damping + displacement2)
    out[1] = velocity * displacement * (1 - α)
    out[2] = velocity * displacement2 * α
    return out
end

function (scales::DampedHCVComponentScales)(out, state,
        flow::ContinuousDynamics, t)
    displacement, velocity = trajectory_geometry_bounds(
        flow, state, scales.anchor, t)
    return _damped_hcv_scales!(out, velocity, displacement, scales.damping)
end

function (scales::DampedHCVComponentScales)(out, state,
        flow::ContinuousDynamics, left, right)
    displacement, velocity = trajectory_geometry_cell_bounds(
        flow, state, scales.anchor, left, right)
    return _damped_hcv_scales!(out, velocity, displacement, scales.damping)
end
