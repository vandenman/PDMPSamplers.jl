function _grid_bound_violation_message(alg, stats::AbstractStatisticCounter, state::AbstractPDMPState, flow::ContinuousDynamics,
    τ::Real, signed_actual::Real, l_actual::Real, bound_actual::Real, cumulative_exp::Real, λ_refresh::Real, use_linear::Bool)
    ratio = bound_actual == 0 ? Inf : l_actual / bound_actual
    cell_index = if isempty(alg.pcb.t_grid)
        0
    else
        clamp(searchsortedlast(alg.pcb.t_grid, τ), 1, max(length(alg.pcb.Λ_vals), 1))
    end
    a = 0.0
    b = 0.0
    y_left = NaN
    y_right = NaN
    d_left = NaN
    d_right = NaN
    if 1 <= cell_index <= length(alg.pcb.Λ_vals)
        a = alg.pcb.t_grid[cell_index]
        b = alg.pcb.t_grid[cell_index + 1]
        y_left = alg.pcb.y_vals[cell_index]
        y_right = alg.pcb.y_vals[cell_index + 1]
        d_left = alg.pcb.d_vals[cell_index]
        d_right = alg.pcb.d_vals[cell_index + 1]
    end

    segment_index = 0
    seg_left = NaN
    seg_right = NaN
    seg_y_left = NaN
    seg_slope = NaN
    if use_linear && alg.affine_bound.n_segments > 0
        segment_index = τ == alg.affine_bound.t_breaks[alg.affine_bound.n_segments + 1] ?
            alg.affine_bound.n_segments :
            clamp(searchsortedlast(@view(alg.affine_bound.t_breaks[1:(alg.affine_bound.n_segments + 1)]), τ),
                  1, alg.affine_bound.n_segments)
        seg_left = alg.affine_bound.t_breaks[segment_index]
        seg_right = alg.affine_bound.t_breaks[segment_index + 1]
        seg_y_left = alg.affine_bound.y_left[segment_index]
        seg_slope = alg.affine_bound.slopes[segment_index]
    end

    x = state.ξ.x
    v = state.ξ.θ
    return string(
        alg.bound, " GridThinning bound violated at proposal time",
        " | acceptance_test_index=", _get_counter_grid_acceptance_tests(stats),
        " cell_index=", cell_index,
        " cell=[", a, ", ", b, "]",
        " tau=", τ,
        " x=", collect(x),
        " v=", collect(v),
        " g_tau=", signed_actual,
        " lambda_tau=", l_actual,
        " bound_tau=", bound_actual,
        " ratio=", ratio,
        " refresh=", λ_refresh,
        " cumulative_hazard_before_tau=", cumulative_exp,
        " | cell_y=[", y_left, ", ", y_right, "]",
        " cell_d=[", d_left, ", ", d_right, "]",
        " | segment_index=", segment_index,
        " segment=[", seg_left, ", ", seg_right, "]",
        " segment_y_left=", seg_y_left,
        " segment_slope=", seg_slope
    )
end

function _record_grid_schedule!(stats::AbstractStatisticCounter, alg)
    _inc_counter_grid_schedule_samples(stats)
    _inc_counter_grid_N_sum(stats, alg.N[])
    _inc_counter_grid_tmax_sum(stats, alg.t_max[])
    _inc_counter_grid_h_sum(stats, alg.t_max[] / alg.N[])
    return nothing
end

@inline _bound_violated(actual, bound) =
    actual > bound * (1 + 1e-10) + 1e-12

function _piecewise_constant_area(pcb::PiecewiseConstantBound,
        n_cells::Integer=length(pcb.Λ_vals))
    area = zero(eltype(pcb.Λ_vals))
    @inbounds for i in 1:n_cells
        area += pos(pcb.Λ_vals[i]) * (pcb.t_grid[i + 1] - pcb.t_grid[i])
    end
    return area
end

_grid_built_area(pcb::PiecewiseConstantBound, bound::PiecewiseAffineBound, use_linear::Bool) =
    use_linear ? total_area(bound) : _piecewise_constant_area(pcb)

function _record_budget_grid_build!(stats::AbstractStatisticCounter, n_cells::Integer, built_area::Real,
    exponential_budget::Real, is_extension::Bool)
    is_extension && (_inc_counter_grid_budget_extensions(stats))
    _inc_counter_grid_budget_cells_built(stats, max(Int(n_cells), 0))
    _inc_counter_grid_budget_area_built(stats, float(built_area))
    _inc_counter_grid_budget_exponential_sum(stats, float(exponential_budget))
    return nothing
end
