# The full-gradient value-quadratic construction needs only the total rate at
# the grid points and the acceptance test at the candidate, so it also applies
# to componentwise (Zig-Zag) rates: the bound then concerns the total rate
# sum_i (theta_i d_i U)_+, as for the scalar rates of the other dynamics, and
# the grid points need gradients only (no rate derivatives).
_can_use_global_value_quadratic_grid(flow::ContinuousDynamics, curvature_bound) =
    _rate_aggregation(flow) in (:scalar, :componentwise) && curvature_bound !== nothing

@inline function _value_quadratic_cell_bound(y_left::Real, y_right::Real, cell_width::Real, residual_bound::Real)
    residual = max(Float64(residual_bound), 0.0) * Float64(cell_width)^2 / 8
    return max(Float64(y_left), Float64(y_right), 0.0) + residual
end

function _value_rate_at_state!(
    probe_failure_handler::GridBoundaryProbe,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    grad_provider,
    t_valid::Float64,
    t_invalid::Float64,
)
    rate, _ = _get_rate_and_deriv_or_throw(
        probe_failure_handler, state, flow, GradientOnlyProvider(grad_provider), false;
        t_valid, t_invalid)
    return rate
end


# A callable certificate may decline one cell. Use the ordinary signed
# tangent builder on that cell; no proposal has yet used its bound.
function _tangent_value_cell_bound(state, flow, alg, stats, left, right, probe)
    cell_state = copy(state)
    move_forward_time!(cell_state, left, flow)
    cell_bound = PiecewiseConstantBound([0.0, right - left], [0.0])
    provider = alg.curvature_backend === :finite_difference || alg.exact_provider === nothing ?
        _grid_fd_provider(alg, stats) : alg.exact_provider
    construct_rate_bound_grid!(cell_bound, cell_state, flow, provider, nothing;
        stats, probe_failure_handler=probe)
    return cell_bound.Λ_vals[1]
end
