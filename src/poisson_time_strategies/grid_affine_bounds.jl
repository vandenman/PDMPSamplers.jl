_auto_policy(bound::Symbol) = bound === :auto

_use_linear_bound(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider) =
    (alg.bound === :linear && _can_use_signed_grid_bound(alg, state, flow, provider)) ||
    (_auto_policy(alg.bound) && _can_use_signed_grid_bound(alg, state, flow, provider))

_use_flat_bound(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider) =
    (alg.bound === :flat) &&
    _can_use_signed_grid_bound(alg, state, flow, provider)

_use_signed_grid_bound(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider) =
    alg.bound in (:linear, :flat, :auto) &&
    _can_use_signed_grid_bound(alg, state, flow, provider)

function _can_use_signed_grid_bound(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider)
    return _can_use_signed_grid(state, flow, provider)
end

function _curvature_bound_value(value)
    value === nothing && return nothing
    value isa Real || throw(ArgumentError(
        "curvature_bound must be nothing, a finite number, or return one; got $(typeof(value))"))
    isfinite(value) || throw(ArgumentError("curvature_bound must be finite, got $(value)"))
    return Float64(value)
end

function _evaluate_curvature_bound(curvature_bound, state::AbstractPDMPState, flow::ContinuousDynamics,
    a::Real, b::Real, stats::Union{AbstractStatisticCounter,Nothing})
    curvature_bound === nothing && return nothing
    curvature_bound isa Real && return _curvature_bound_value(curvature_bound)
    stats !== nothing && (_inc_counter_grid_certificate_calls(stats))
    value = _curvature_bound_value(curvature_bound(state, flow, a, b))
    value === nothing && stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, 1))
    return value
end

function _prepare_grid_curvature_bound(curvature_bound, state::AbstractPDMPState, flow::ContinuousDynamics,
    t_grid::AbstractVector, n_cells::Integer, stats::Union{AbstractStatisticCounter,Nothing})
    if curvature_bound === nothing
        stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, n_cells))
        return (global_value=nothing, first_value=nothing, has_first=true, cell_values=nothing)
    elseif curvature_bound isa Real
        return (global_value=Float64(curvature_bound), first_value=nothing,
            has_first=false, cell_values=nothing)
    end

    value = _evaluate_curvature_bound(curvature_bound, state, flow, t_grid[1], t_grid[n_cells + 1], stats)
    return (global_value=nothing, first_value=value, has_first=true, cell_values=nothing)
end

function _prepared_or_cell_curvature_value(prepared, cell::Integer, curvature_bound,
    state::AbstractPDMPState, flow::ContinuousDynamics, a::Real, b::Real,
    stats::Union{AbstractStatisticCounter,Nothing})
    prepared.global_value !== nothing && return prepared.global_value
    curvature_bound === nothing && return nothing
    if prepared.cell_values !== nothing
        return prepared.cell_values[cell]
    end
    if cell == 1 && prepared.has_first
        return prepared.first_value
    end
    return _evaluate_curvature_bound(curvature_bound, state, flow, a, b, stats)
end

function _channel_curvature_matrix(
    curvature_bound,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    t_grid::AbstractVector,
    n_channels::Integer,
    n_cells::Integer,
    stats::Union{AbstractStatisticCounter,Nothing},
)
    if curvature_bound === nothing
        stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, n_channels * n_cells))
        return zeros(Float64, n_channels, n_cells)
    elseif curvature_bound isa Real
        return fill(Float64(curvature_bound), n_channels, n_cells)
    end
    prepared = _prepare_grid_curvature_bound(
        curvature_bound, state, flow, t_grid, n_cells, stats)
    L = Matrix{Float64}(undef, n_channels, n_cells)
    for cell in 1:n_cells
        a = t_grid[cell]
        b = t_grid[cell + 1]
        value = _prepared_or_cell_curvature_value(
            prepared, cell, curvature_bound, state, flow, a, b, stats)
        L[:, cell] .= value === nothing ? 0.0 : Float64(value)
    end
    return L
end

function _affine_cell_tolerances(a::Real, b::Real, y_a::Real, y_b::Real, d_a::Real, d_b::Real, M::Real)
    h = b - a
    rate_scale = max(1.0, abs(float(y_a)), abs(float(y_b)), abs(float(M)))
    slope_scale = max(1.0, abs(float(d_a)), abs(float(d_b)), rate_scale / max(abs(float(h)), eps(Float64)))
    time_scale = max(1.0, abs(float(a)), abs(float(b)), abs(float(h)))
    rate_tol = 256 * eps(Float64) * rate_scale
    slope_tol = 256 * eps(Float64) * slope_scale
    time_tol = 256 * eps(Float64) * time_scale
    area_tol = 256 * eps(Float64) * rate_scale * max(abs(float(h)), eps(Float64))
    return rate_tol, slope_tol, time_tol, area_tol
end

function _append_constant_affine_cell!(bound::PiecewiseAffineBound, stats::Union{AbstractStatisticCounter,Nothing},
    a::Real, b::Real, M::Real; auto::Bool=false)
    M_pos = pos(M)
    append_affine_segment!(bound, a, b, M_pos, zero(M_pos))
    h = b - a
    area = M_pos * h
    if stats !== nothing
        _inc_counter_affine_constant_cells(stats)
        _inc_counter_affine_area_constant_equiv(stats, area)
        _inc_counter_affine_area_hybrid(stats, area)
        _inc_counter_affine_segments_added(stats, 1)
        if auto
            _inc_counter_auto_flat_cells(stats)
            total = _get_counter_auto_flat_cells(stats) + _get_counter_auto_affine_cells(stats)
            _set_counter_auto_affine_fraction(stats, total == 0 ? 0.0 : _get_counter_auto_affine_cells(stats) / total)
        end
    end
    return false
end

function _append_hybrid_affine_cell!(bound::PiecewiseAffineBound, stats::Union{AbstractStatisticCounter,Nothing},
    a::Real, b::Real, y_a::Real, y_b::Real, d_a::Real, d_b::Real, M::Real)

    h = b - a
    if !(isfinite(a) && isfinite(b) && isfinite(y_a) && isfinite(y_b) &&
         isfinite(d_a) && isfinite(d_b) && isfinite(M) && h > 0)
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    rate_tol, slope_tol, time_tol, area_tol =
        _affine_cell_tolerances(a, b, y_a, y_b, d_a, d_b, M)

    if !(d_a > slope_tol && d_b < -slope_tol && y_a > rate_tol && y_b > rate_tol)
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    denom = d_a - d_b
    denom > slope_tol || return _append_constant_affine_cell!(bound, stats, a, b, M)

    s_star = (y_b - y_a - d_b * h) / denom
    if !(s_star > time_tol && s_star < h - time_tol)
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    m_left = y_a + d_a * s_star
    m_right = y_b + d_b * (s_star - h)
    height_tol = 512 * eps(Float64) * max(1.0, abs(float(m_left)), abs(float(m_right)), abs(float(M)))
    if abs(m_left - m_right) > height_tol
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    m = (m_left + m_right) / 2
    left_area = _affine_segment_area(y_a, d_a, s_star)
    right_h = h - s_star
    right_area = _affine_segment_area(m, d_b, right_h)
    const_area = pos(M) * h
    roof_area = left_area + right_area

    if !(m >= -rate_tol && left_area >= -area_tol && right_area >= -area_tol &&
         roof_area >= -area_tol && roof_area <= const_area + max(area_tol, 1e-12 * max(1.0, abs(const_area))))
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    append_affine_segment!(bound, a, a + s_star, y_a, d_a)
    append_affine_segment!(bound, a + s_star, b, m, d_b)
    if stats !== nothing
        _inc_counter_affine_roof_cells(stats)
        _inc_counter_affine_area_constant_equiv(stats, const_area)
        _inc_counter_affine_area_hybrid(stats, roof_area)
        _inc_counter_affine_area_saved(stats, max(const_area - roof_area, 0.0))
        _inc_counter_affine_segments_added(stats, 2)
    end
    return true
end

function _append_line_piece!(
    bound::PiecewiseAffineBound,
    a::Real,
    s_left::Real,
    s_right::Real,
    alpha_at_a::Real,
    beta::Real,
)
    s_right <= s_left && return bound
    t_left = a + s_left
    t_right = a + s_right
    alpha_at_left = alpha_at_a + beta * s_left
    return append_affine_segment!(bound, t_left, t_right, alpha_at_left, beta)
end

function _append_positive_part_line_piece!(
    bound::PiecewiseAffineBound,
    a::Real,
    s_left::Real,
    s_right::Real,
    alpha_at_a::Real,
    beta::Real,
)
    s_right <= s_left && return bound
    y_left = alpha_at_a + beta * s_left
    y_right = alpha_at_a + beta * s_right
    tol = _affine_nonnegative_tolerance(y_left, y_right)

    if y_left <= tol && y_right <= tol
        return _append_line_piece!(bound, a, s_left, s_right, 0.0, 0.0)
    elseif y_left >= -tol && y_right >= -tol
        alpha = max(y_left, 0.0)
        return append_affine_segment!(bound, a + s_left, a + s_right, alpha, beta)
    end

    iszero(beta) && return _append_line_piece!(bound, a, s_left, s_right, max(y_left, 0.0), 0.0)
    s_zero = clamp(-alpha_at_a / beta, s_left, s_right)
    if y_left <= 0 && y_right > 0
        _append_line_piece!(bound, a, s_left, s_zero, 0.0, 0.0)
        return append_affine_segment!(bound, a + s_zero, a + s_right, 0.0, beta)
    else
        append_affine_segment!(bound, a + s_left, a + s_zero, max(y_left, 0.0), beta)
        return _append_line_piece!(bound, a, s_zero, s_right, 0.0, 0.0)
    end
end

function _append_linear_cell!(bound::PiecewiseAffineBound, stats::Union{AbstractStatisticCounter,Nothing},
    a::Real, b::Real, y_a::Real, y_b::Real, d_a::Real, d_b::Real, M::Real, L)

    L === nothing && return _append_constant_affine_cell!(bound, stats, a, b, M)
    h = b - a
    if !(isfinite(a) && isfinite(b) && isfinite(y_a) && isfinite(y_b) &&
         isfinite(d_a) && isfinite(d_b) && isfinite(M) && isfinite(L) && h > 0)
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    rate_tol, slope_tol, time_tol, area_tol =
        _affine_cell_tolerances(a, b, y_a, y_b, d_a, d_b, M)

    # The scalar curvature bound applies to the smooth signed rate. GridThinning's
    # cached endpoint values are for the positive-part rate; when an endpoint is
    # clamped at zero we no longer have the signed tangent needed to certify a
    # linearized roof across possible zero crossings.  Keep this first-stage
    # bound conservative and fall back to the existing constant cell.
    if !(y_a > rate_tol && y_b > rate_tol)
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    Lbar = max(float(L), 0.0)
    beta_a = d_a + 0.5 * Lbar * h
    beta_b = d_b - 0.5 * Lbar * h
    alpha_a = y_a
    alpha_b = y_b - beta_b * h

    # B_L is min(line a, line b). Split at the intersection when it is
    # materially inside the cell; otherwise one line dominates over the cell.
    denom = beta_a - beta_b
    split_points = Float64[0.0, h]
    if abs(denom) > slope_tol
        s_star = (alpha_b - alpha_a) / denom
        if s_star > time_tol && s_star < h - time_tol
            push!(split_points, s_star)
            sort!(split_points)
        end
    end

    const_area = pos(M) * h
    hybrid_area = 0.0
    start_segments = bound.n_segments
    try
        for k in 1:(length(split_points) - 1)
            s_left = split_points[k]
            s_right = split_points[k + 1]
            s_mid = (s_left + s_right) / 2
            val_a = alpha_a + beta_a * s_mid
            val_b = alpha_b + beta_b * s_mid
            if val_a <= val_b
                _append_line_piece!(bound, a, s_left, s_right, alpha_a, beta_a)
            else
                _append_line_piece!(bound, a, s_left, s_right, alpha_b, beta_b)
            end
            j = bound.n_segments
            seg_h = bound.t_breaks[j + 1] - bound.t_breaks[j]
            hybrid_area += _affine_segment_area(bound.y_left[j], bound.slopes[j], seg_h)
        end
    catch err
        bound.n_segments = start_segments
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    if !(hybrid_area >= -area_tol)
        bound.n_segments = start_segments
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    if stats !== nothing
        _inc_counter_affine_inflated_cells(stats)
        _inc_counter_affine_area_constant_equiv(stats, const_area)
        _inc_counter_affine_area_hybrid(stats, max(hybrid_area, 0.0))
        _inc_counter_affine_area_saved(stats, max(const_area - max(hybrid_area, 0.0), 0.0))
        _inc_counter_affine_segments_added(stats, bound.n_segments - start_segments)
    end
    return true
end

function _append_rate_linear_cell!(bound::PiecewiseAffineBound, stats::Union{AbstractStatisticCounter,Nothing},
    a::Real, b::Real, g_a::Real, g_b::Real, dg_a::Real, dg_b::Real, M::Real, L;
    linear_area_threshold::Real=1.0,
    linear_min_area_gain::Real=0.0,
    auto::Bool=false)

    L === nothing && return _append_constant_affine_cell!(bound, stats, a, b, M; auto)
    h = b - a
    if !(isfinite(a) && isfinite(b) && isfinite(g_a) && isfinite(g_b) &&
         isfinite(dg_a) && isfinite(dg_b) && isfinite(M) && isfinite(L) && h > 0)
        return _append_constant_affine_cell!(bound, stats, a, b, M; auto)
    end

    _, slope_tol, time_tol, area_tol =
        _affine_cell_tolerances(a, b, g_a, g_b, dg_a, dg_b, M)

    Lbar = max(float(L), 0.0)
    beta_a = dg_a + 0.5 * Lbar * h
    beta_b = dg_b - 0.5 * Lbar * h
    alpha_a = g_a
    alpha_b = g_b - beta_b * h

    denom = beta_a - beta_b
    split_points = Float64[0.0, h]
    if abs(denom) > slope_tol
        s_star = (alpha_b - alpha_a) / denom
        if s_star > time_tol && s_star < h - time_tol
            push!(split_points, s_star)
        end
    end

    for (alpha, beta) in ((alpha_a, beta_a), (alpha_b, beta_b))
        if abs(beta) > slope_tol
            s_zero = -alpha / beta
            if s_zero > time_tol && s_zero < h - time_tol
                push!(split_points, s_zero)
            end
        end
    end
    sort!(unique!(split_points))

    const_area = pos(M) * h
    clipped_area = 0.0
    start_segments = bound.n_segments
    try
        for k in 1:(length(split_points) - 1)
            s_left = split_points[k]
            s_right = split_points[k + 1]
            s_mid = (s_left + s_right) / 2
            val_a = alpha_a + beta_a * s_mid
            val_b = alpha_b + beta_b * s_mid
            if val_a <= val_b
                _append_positive_part_line_piece!(bound, a, s_left, s_right, alpha_a, beta_a)
            else
                _append_positive_part_line_piece!(bound, a, s_left, s_right, alpha_b, beta_b)
            end
        end
        for j in (start_segments + 1):bound.n_segments
            seg_h = bound.t_breaks[j + 1] - bound.t_breaks[j]
            clipped_area += _affine_segment_area(bound.y_left[j], bound.slopes[j], seg_h)
        end
    catch err
        bound.n_segments = start_segments
        return _append_constant_affine_cell!(bound, stats, a, b, M; auto)
    end

    if !(clipped_area >= -area_tol)
        bound.n_segments = start_segments
        return _append_constant_affine_cell!(bound, stats, a, b, M; auto)
    end

    flat_area = pos(M) * h
    area_gain = flat_area - max(clipped_area, 0.0)
    min_area_gain = max(float(linear_min_area_gain), 0.0)
    if area_gain <= min_area_gain
        bound.n_segments = start_segments
        stats !== nothing && (_inc_counter_affine_cells_skipped_by_min_gain(stats))
        return _append_constant_affine_cell!(bound, stats, a, b, M; auto)
    end

    threshold = float(linear_area_threshold)
    if threshold < 1.0 && flat_area > area_tol && clipped_area / flat_area >= threshold
        bound.n_segments = start_segments
        return _append_constant_affine_cell!(bound, stats, a, b, M; auto)
    end

    if stats !== nothing
        _inc_counter_affine_inflated_cells(stats)
        _inc_counter_affine_area_constant_equiv(stats, flat_area)
        _inc_counter_affine_area_hybrid(stats, max(clipped_area, 0.0))
        _inc_counter_affine_area_saved(stats, max(area_gain, 0.0))
        _inc_counter_affine_segments_added(stats, bound.n_segments - start_segments)
        if auto
            _inc_counter_auto_affine_cells(stats)
            _inc_counter_auto_area_saved(stats, max(area_gain, 0.0))
            total = _get_counter_auto_flat_cells(stats) + _get_counter_auto_affine_cells(stats)
            _set_counter_auto_affine_fraction(stats, total == 0 ? 0.0 : _get_counter_auto_affine_cells(stats) / total)
        end
    end
    return true
end

function _rate_flat_upper(a::Real, b::Real, g_a::Real, g_b::Real, dg_a::Real, dg_b::Real, L)
    h = b - a
    if !(isfinite(a) && isfinite(b) && isfinite(g_a) && isfinite(g_b) &&
         isfinite(dg_a) && isfinite(dg_b) && isfinite(L) && h > 0)
        return Inf
    end
    _, slope_tol, time_tol, _ =
        _affine_cell_tolerances(a, b, g_a, g_b, dg_a, dg_b, max(abs(g_a), abs(g_b)))

    Lbar = max(float(L), 0.0)
    beta_a = dg_a + 0.5 * Lbar * h
    beta_b = dg_b - 0.5 * Lbar * h
    alpha_a = g_a
    alpha_b = g_b - beta_b * h

    candidates = Float64[0.0, h]
    denom = beta_a - beta_b
    if abs(denom) > slope_tol
        s_star = (alpha_b - alpha_a) / denom
        if s_star > time_tol && s_star < h - time_tol
            push!(candidates, s_star)
        end
    end

    M = 0.0
    for s in candidates
        M = max(M, min(alpha_a + beta_a * s, alpha_b + beta_b * s))
    end
    return max(M, 0.0)
end

function _signed_inflated_roof_lines(
    a::Real,
    b::Real,
    g_a::Real,
    g_b::Real,
    dg_a::Real,
    dg_b::Real,
    L,
)
    h = b - a
    Lbar = max(float(L), 0.0)
    beta_a = dg_a + 0.5 * Lbar * h
    beta_b = dg_b - 0.5 * Lbar * h
    alpha_a = float(g_a)
    alpha_b = float(g_b) - beta_b * h
    return alpha_a, beta_a, alpha_b, beta_b
end

function _push_componentwise_breakpoint!(
    breakpoints::Vector{Float64},
    s::Real,
    h::Real,
    time_tol::Real,
)
    if s > time_tol && s < h - time_tol
        push!(breakpoints, Float64(s))
    end
    return breakpoints
end

function _merge_componentwise_breakpoints!(breakpoints::Vector{Float64}, time_tol::Real)
    sort!(breakpoints)
    n_in = length(breakpoints)
    n_in <= 1 && return n_in
    write = 1
    for read in 2:n_in
        if abs(breakpoints[read] - breakpoints[write]) > time_tol
            write += 1
            breakpoints[write] = breakpoints[read]
        end
    end
    resize!(breakpoints, write)
    return n_in - write
end

function _add_signed_roof_breakpoints!(
    breakpoints::Vector{Float64},
    a::Real,
    b::Real,
    g_a::Real,
    g_b::Real,
    dg_a::Real,
    dg_b::Real,
    L,
    slope_tol::Real,
    time_tol::Real,
)
    h = b - a
    alpha_a, beta_a, alpha_b, beta_b =
        _signed_inflated_roof_lines(a, b, g_a, g_b, dg_a, dg_b, L)
    denom = beta_a - beta_b
    if abs(denom) > slope_tol
        _push_componentwise_breakpoint!(
            breakpoints, (alpha_b - alpha_a) / denom, h, time_tol)
    end
    for (alpha, beta) in ((alpha_a, beta_a), (alpha_b, beta_b))
        abs(beta) > slope_tol &&
            _push_componentwise_breakpoint!(breakpoints, -alpha / beta, h, time_tol)
    end
    return breakpoints
end

function _signed_roof_line_on_subinterval(
    a::Real,
    b::Real,
    g_a::Real,
    g_b::Real,
    dg_a::Real,
    dg_b::Real,
    L,
    s_mid::Real,
)
    alpha_a, beta_a, alpha_b, beta_b =
        _signed_inflated_roof_lines(a, b, g_a, g_b, dg_a, dg_b, L)
    val_a = alpha_a + beta_a * s_mid
    val_b = alpha_b + beta_b * s_mid
    return val_a <= val_b ? (alpha_a, beta_a) : (alpha_b, beta_b)
end

function _append_componentwise_flat_cell!(
    bound::PiecewiseAffineBound,
    stats::Union{AbstractStatisticCounter,Nothing},
    a::Real,
    b::Real,
    M::Real;
    auto::Bool=false,
    reason::Symbol=:other,
)
    if stats !== nothing
        _inc_counter_componentwise_flat_fallback_cells(stats)
        if reason === :segment_cap
            _inc_counter_componentwise_flat_fallback_segment_cap(stats)
        elseif reason === :numerical
            _inc_counter_componentwise_flat_fallback_numerical(stats)
        elseif reason === :area_gate
            _inc_counter_componentwise_affine_skipped_by_area_gate(stats)
            _inc_counter_componentwise_flat_fallback_area_gate(stats)
        elseif reason === :policy
            _inc_counter_componentwise_affine_skipped_by_policy(stats)
            _inc_counter_componentwise_flat_fallback_policy(stats)
        else
            _inc_counter_componentwise_flat_fallback_other(stats)
        end
    end
    return _append_constant_affine_cell!(bound, stats, a, b, M; auto)
end

function _componentwise_zero_crossing_count(
    G::AbstractMatrix,
    left::Integer,
)
    count = 0
    for channel in axes(G, 1)
        g_left = G[channel, left]
        g_right = G[channel, left + 1]
        if iszero(g_left) || iszero(g_right) || signbit(g_left) != signbit(g_right)
            count += 1
        end
    end
    return count
end

function _append_componentwise_signed_affine_cell!(
    bound::PiecewiseAffineBound,
    stats::Union{AbstractStatisticCounter,Nothing},
    a::Real,
    b::Real,
    G::AbstractMatrix,
    dG::AbstractMatrix,
    L::AbstractMatrix,
    left::Integer,
    cell::Integer,
    M::Real;
    linear_area_threshold::Real=1.0,
    linear_min_area_gain::Real=0.0,
    auto::Bool=false,
    max_segments::Integer=64,
)
    h = b - a
    n_channels = size(G, 1)
    if !(isfinite(a) && isfinite(b) && isfinite(M) && h > 0)
        return _append_componentwise_flat_cell!(
            bound, stats, a, b, M; auto, reason=:numerical)
    end

    _, slope_tol, time_tol, area_tol =
        _affine_cell_tolerances(a, b, 0.0, M, 0.0, 0.0, M)
    breakpoints = Float64[0.0, h]
    zero_crossings = _componentwise_zero_crossing_count(G, left)
    for channel in 1:n_channels
        values = (
            G[channel, left],
            G[channel, left + 1],
            dG[channel, left],
            dG[channel, left + 1],
            L[channel, cell],
        )
        all(isfinite, values) ||
            return _append_componentwise_flat_cell!(
                bound, stats, a, b, M; auto, reason=:numerical)
        _add_signed_roof_breakpoints!(
            breakpoints, a, b, values[1], values[2], values[3], values[4],
            values[5], slope_tol, time_tol)
    end
    proposed_breakpoints = max(length(breakpoints) - 2, 0)
    merged = _merge_componentwise_breakpoints!(breakpoints, time_tol)
    if length(breakpoints) - 1 > max_segments
        _record_counter_componentwise_cell_diagnostics!(
            stats, proposed_breakpoints, length(breakpoints) - 1, zero_crossings, 0.0, 0.0)
        return _append_componentwise_flat_cell!(
            bound, stats, a, b, M; auto, reason=:segment_cap)
    end

    start_segments = bound.n_segments
    affine_area = 0.0
    try
        for k in 1:(length(breakpoints) - 1)
            s_left = breakpoints[k]
            s_right = breakpoints[k + 1]
            s_mid = (s_left + s_right) / 2
            total_alpha = 0.0
            total_beta = 0.0
            for channel in 1:n_channels
                alpha, beta = _signed_roof_line_on_subinterval(
                    a, b, G[channel, left], G[channel, left + 1],
                    dG[channel, left], dG[channel, left + 1], L[channel, cell], s_mid)
                if alpha + beta * s_mid > area_tol
                    total_alpha += alpha + beta * s_left
                    total_beta += beta
                end
            end
            append_affine_segment!(bound, a + s_left, a + s_right, max(total_alpha, 0.0), total_beta)
            affine_area += _affine_segment_area(total_alpha, total_beta, s_right - s_left)
        end
    catch err
        bound.n_segments = start_segments
        return _append_componentwise_flat_cell!(
            bound, stats, a, b, M; auto, reason=:numerical)
    end

    flat_area = pos(M) * h
    if affine_area < -area_tol || affine_area > flat_area + max(area_tol, 1e-10 * max(1.0, flat_area))
        bound.n_segments = start_segments
        return _append_componentwise_flat_cell!(
            bound, stats, a, b, M; auto, reason=:numerical)
    end

    area_gain = flat_area - max(affine_area, 0.0)
    saved_fraction = flat_area <= area_tol ? 0.0 : max(area_gain, 0.0) / flat_area
    segments_added = bound.n_segments - start_segments
    if area_gain <= max(float(linear_min_area_gain), 0.0)
        bound.n_segments = start_segments
        stats !== nothing && (_inc_counter_affine_cells_skipped_by_min_gain(stats))
        _record_counter_componentwise_cell_diagnostics!(
            stats, proposed_breakpoints, segments_added, zero_crossings, area_gain, saved_fraction)
        return _append_componentwise_flat_cell!(
            bound, stats, a, b, M; auto, reason=:area_gate)
    end

    threshold = float(linear_area_threshold)
    if threshold < 1.0 && flat_area > area_tol && affine_area / flat_area >= threshold
        bound.n_segments = start_segments
        _record_counter_componentwise_cell_diagnostics!(
            stats, proposed_breakpoints, segments_added, zero_crossings, area_gain, saved_fraction)
        return _append_componentwise_flat_cell!(
            bound, stats, a, b, M; auto, reason=:area_gate)
    end

    if stats !== nothing
        _record_counter_componentwise_cell_diagnostics!(
            stats, proposed_breakpoints, segments_added, zero_crossings,
            max(area_gain, 0.0), saved_fraction)
        _inc_counter_affine_inflated_cells(stats)
        _inc_counter_affine_area_constant_equiv(stats, flat_area)
        _inc_counter_affine_area_hybrid(stats, max(affine_area, 0.0))
        _inc_counter_affine_area_saved(stats, max(area_gain, 0.0))
        _inc_counter_affine_segments_added(stats, segments_added)
        _inc_counter_componentwise_affine_cells(stats)
        _inc_counter_componentwise_affine_segments_added(stats, segments_added)
        _inc_counter_componentwise_breakpoints_merged(stats, merged)
        _inc_counter_componentwise_area_saved(stats, max(area_gain, 0.0))
        if auto
            _inc_counter_auto_affine_cells(stats)
            _inc_counter_auto_area_saved(stats, max(area_gain, 0.0))
            total = _get_counter_auto_flat_cells(stats) +
                _get_counter_auto_affine_cells(stats)
            fraction = total == 0 ? 0.0 :
                _get_counter_auto_affine_cells(stats) / total
            _set_counter_auto_affine_fraction(stats, fraction)
        end
    end
    return true
end

function _affine_added_area(bound::PiecewiseAffineBound, start_segments::Integer)
    area = 0.0
    for j in (start_segments + 1):bound.n_segments
        h = bound.t_breaks[j + 1] - bound.t_breaks[j]
        area += _affine_segment_area(bound.y_left[j], bound.slopes[j], h)
    end
    return max(area, 0.0)
end

function build_hybrid_affine_bound!(bound::PiecewiseAffineBound, pcb::PiecewiseConstantBound,
    n_cells::Integer, stats::Union{AbstractStatisticCounter,Nothing}=nothing)
    reset_affine_bound!(bound)
    n = min(n_cells, length(pcb.Λ_vals))
    for i in 1:n
        _append_hybrid_affine_cell!(bound, stats,
            pcb.t_grid[i], pcb.t_grid[i + 1],
            pcb.y_vals[i], pcb.y_vals[i + 1],
            pcb.d_vals[i], pcb.d_vals[i + 1],
            pcb.Λ_vals[i])
    end
    return bound
end

function build_linear_bound!(bound::PiecewiseAffineBound, pcb::PiecewiseConstantBound,
    n_cells::Integer, original_state::AbstractPDMPState, flow::ContinuousDynamics, curvature_bound,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing)
    reset_affine_bound!(bound)
    n = min(n_cells, length(pcb.Λ_vals))
    if n <= 0
        return bound
    end
    prepared = _prepare_grid_curvature_bound(
        curvature_bound, original_state, flow, pcb.t_grid, n, stats)
    for i in 1:n
        a = pcb.t_grid[i]
        b = pcb.t_grid[i + 1]
        L_value = _prepared_or_cell_curvature_value(
            prepared, i, curvature_bound, original_state, flow, a, b, stats)
        _append_linear_cell!(bound, stats,
            a, b, pcb.y_vals[i], pcb.y_vals[i + 1], pcb.d_vals[i], pcb.d_vals[i + 1], pcb.Λ_vals[i], L_value)
    end
    return bound
end

function build_rate_linear_bound!(bound::PiecewiseAffineBound, pcb::PiecewiseConstantBound,
    n_cells::Integer, original_state::AbstractPDMPState, flow::ContinuousDynamics, curvature_bound,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing; linear_area_threshold::Real=1.0,
    linear_min_area_gain::Real=0.0)
    reset_affine_bound!(bound)
    n = min(n_cells, length(pcb.Λ_vals))
    if n <= 0
        return bound
    end
    prepared = _prepare_grid_curvature_bound(
        curvature_bound, original_state, flow, pcb.t_grid, n, stats)
    for i in 1:n
        a = pcb.t_grid[i]
        b = pcb.t_grid[i + 1]
        L_value = _prepared_or_cell_curvature_value(
            prepared, i, curvature_bound, original_state, flow, a, b, stats)
        L_cell = L_value === nothing ? 0.0 : Float64(L_value)
        M = max(pcb.Λ_vals[i], _rate_flat_upper(
            a, b, pcb.y_vals[i], pcb.y_vals[i + 1], pcb.d_vals[i], pcb.d_vals[i + 1], L_cell))
        pcb.Λ_vals[i] = M
        _append_rate_linear_cell!(bound, stats,
            a, b, pcb.y_vals[i], pcb.y_vals[i + 1], pcb.d_vals[i], pcb.d_vals[i + 1], M, L_cell;
            linear_area_threshold,
            linear_min_area_gain)
    end
    return bound
end

rate_and_derivative(state::AbstractPDMPState, flow::ContinuousDynamics, provider, ::AbstractVector) =
    rate_and_derivative(state, flow, provider)

function _rate_and_derivative_or_throw(
    ::NoGridBoundaryProbe,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    provider,
    args...;
    t_valid::Float64,
    t_invalid::Float64,
)
    return rate_and_derivative(state, flow, provider, args...)
end

function _rate_and_derivative_or_throw(
    probe_failure_handler::GridBoundaryProbeHandler,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    provider,
    args...;
    t_valid::Float64,
    t_invalid::Float64,
)
    try
        return rate_and_derivative(state, flow, provider, args...)
    catch err
        _throw_grid_boundary_error(probe_failure_handler, state, err; t_valid, t_invalid)
    end
end

function construct_rate_grid!(
    pcb::PiecewiseConstantBound,
    state::AbstractPDMPState,
    flow::BouncyParticle,
    provider,
    n_cells::Integer;
    cached_g0::Float64=NaN,
    cached_dg0::Float64=NaN,
    state_cache::Union{AbstractPDMPState,Nothing}=nothing,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing,
)
    n = min(n_cells, length(pcb.Λ_vals))
    n <= 0 && return n
    state_t = state_cache === nothing ? copy(state) : (copyto!(state_cache, state); state_cache)
    if isnan(cached_g0)
        if stats !== nothing
            _inc_counter_grid_endpoint_evaluations(stats)
            _inc_counter_grid_endpoint_gradient_calls(stats)
            _inc_counter_grid_endpoint_hessian_calls(stats)
        end
        pcb.y_vals[1], pcb.d_vals[1] = rate_and_derivative(state_t, flow, provider)
    else
        !isnothing(stats) && (_inc_counter_grid_cached_endpoint_reuses(stats))
        pcb.y_vals[1] = cached_g0
        pcb.d_vals[1] = cached_dg0
    end
    for i in 2:(n + 1)
        Δt = pcb.t_grid[i] - pcb.t_grid[i - 1]
        move_forward_time!(state_t, Δt, flow)
        if stats !== nothing
            _inc_counter_grid_endpoint_evaluations(stats)
            _inc_counter_grid_endpoint_gradient_calls(stats)
            _inc_counter_grid_endpoint_hessian_calls(stats)
        end
        pcb.y_vals[i], pcb.d_vals[i] = rate_and_derivative(state_t, flow, provider)
    end
    return n
end

function construct_rate_bound_grid!(
    bound::PiecewiseAffineBound,
    pcb::PiecewiseConstantBound,
    state::AbstractPDMPState,
    flow::Union{
        BouncyParticle,
        AnyBoomerang,
        PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle},
    },
    provider,
    curvature_bound;
    cached_gradient::Union{AbstractVector,Nothing}=nothing,
    early_stop_threshold::Float64=Inf,
    state_cache::Union{AbstractPDMPState,Nothing}=nothing,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing,
    max_time::Float64=Inf,
    build_affine::Bool=true,
    linear_area_threshold::Real=1.0,
    linear_min_area_gain::Real=0.0,
    auto::Bool=false,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    start_cell::Integer=1,
    initial_integral::Float64=0.0,
    append::Bool=false,
    max_componentwise_affine_segments_per_cell::Integer=64,
    rate_value_buf::Union{Vector{Float64},Nothing}=nothing,
    rate_derivative_buf::Union{Vector{Float64},Nothing}=nothing,
)
    build_affine && !append && reset_affine_bound!(bound)
    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    y_vals = pcb.y_vals
    d_vals = pcb.d_vals
    N = length(Λ_vals)
    state_t = state_cache === nothing ? copy(state) : (copyto!(state_cache, state); state_cache)
    iszero(t_grid[1]) || error("t_grid[1] must be zero, got $(t_grid[1])")

    n_time_cells = isfinite(max_time) ? max(0, min(N, searchsortedfirst(t_grid, max_time) - 1)) : N
    start_cell = clamp(Int(start_cell), 1, N + 1)
    start_cell > n_time_cells && return start_cell - 1
    used_batched_derivatives = _supports_rate_derivatives(provider, flow) && n_time_cells > 0
    loaded_batched_points = start_cell == 1 ? 0 : start_cell
    if start_cell > 1
        move_forward_time!(state_t, t_grid[start_cell], flow)
    elseif used_batched_derivatives
        loaded_batched_points = _load_rate_derivatives!(
            pcb, provider, state, flow, 1, n_time_cells + 1, loaded_batched_points, stats)
    elseif cached_gradient === nothing
        if stats !== nothing
            _inc_counter_grid_endpoint_evaluations(stats)
            _inc_counter_grid_endpoint_gradient_calls(stats)
            _inc_counter_grid_endpoint_hessian_calls(stats)
        end
        y_vals[1], d_vals[1] = _rate_and_derivative_or_throw(
            probe_failure_handler, state_t, flow, provider; t_valid=0.0, t_invalid=0.0)
    else
        if stats !== nothing
            _inc_counter_grid_cached_endpoint_reuses(stats)
            _inc_counter_grid_endpoint_hessian_calls(stats)
        end
        y_vals[1], d_vals[1] = _rate_and_derivative_or_throw(
            probe_failure_handler, state_t, flow, provider, cached_gradient; t_valid=0.0, t_invalid=0.0)
    end

    cumulative_integral = initial_integral
    N_evaluated = N
    prepared = _prepare_grid_curvature_bound(
        curvature_bound, state, flow, t_grid, N, stats)
    for i in (start_cell + 1):(N + 1)
        if t_grid[i - 1] >= max_time
            for j in (i - 1):N
                Λ_vals[j] = 0.0
            end
            N_evaluated = i - 2
            if stats !== nothing
                _inc_counter_grid_points_skipped(stats, N - N_evaluated)
            end
            break
        end

        Δt = t_grid[i] - t_grid[i - 1]
        if used_batched_derivatives
            loaded_batched_points = _load_rate_derivatives!(
                pcb, provider, state, flow, i, n_time_cells + 1, loaded_batched_points, stats)
        else
            move_forward_time!(state_t, Δt, flow)
            if stats !== nothing
                _inc_counter_grid_endpoint_evaluations(stats)
                _inc_counter_grid_endpoint_gradient_calls(stats)
                _inc_counter_grid_endpoint_hessian_calls(stats)
            end
            y_vals[i], d_vals[i] = _rate_and_derivative_or_throw(
                probe_failure_handler, state_t, flow, provider; t_valid=t_grid[i - 1], t_invalid=t_grid[i])
        end

        cell = i - 1
        a = t_grid[cell]
        b = t_grid[cell + 1]
        L_value = _prepared_or_cell_curvature_value(
            prepared, cell, curvature_bound, state, flow, a, b, stats)
        L_cell = L_value === nothing ? 0.0 : Float64(L_value)
        M = _rate_flat_upper(
            a, b, y_vals[cell], y_vals[cell + 1], d_vals[cell], d_vals[cell + 1], L_cell)
        Λ_vals[cell] = M

        cell_area = M * Δt
        if build_affine
            start_segments = bound.n_segments
            _append_rate_linear_cell!(bound, stats,
                a, b, y_vals[cell], y_vals[cell + 1], d_vals[cell], d_vals[cell + 1], M, L_cell;
                linear_area_threshold,
                linear_min_area_gain,
                auto)
            cell_area = _affine_added_area(bound, start_segments)
        elseif stats !== nothing
            _inc_counter_affine_constant_cells(stats)
            _inc_counter_affine_area_constant_equiv(stats, cell_area)
            _inc_counter_affine_area_hybrid(stats, cell_area)
            if auto
                _inc_counter_auto_flat_cells(stats)
                total = _get_counter_auto_flat_cells(stats) + _get_counter_auto_affine_cells(stats)
                _set_counter_auto_affine_fraction(stats, total == 0 ? 0.0 : _get_counter_auto_affine_cells(stats) / total)
            end
        end

        cumulative_integral += cell_area
        if cumulative_integral >= early_stop_threshold && i <= N
            N_evaluated = i - 1
            for j in i:N
                Λ_vals[j] = 0.0
            end
            if stats !== nothing
                _inc_counter_grid_early_stops(stats)
                _inc_counter_grid_points_skipped(stats, N - N_evaluated)
            end
            break
        end
    end

    if stats !== nothing
        _inc_counter_grid_builds(stats)
        _inc_counter_grid_points_evaluated(stats, start_cell == 1 ?
            (N_evaluated > 1 ? N_evaluated - 1 : N_evaluated) : N_evaluated)
    end
    return N_evaluated
end

function construct_rate_bound_grid!(
    bound::PiecewiseAffineBound,
    pcb::PiecewiseConstantBound,
    state::AbstractPDMPState,
    flow::Union{
        ZigZag,
        PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag},
        PreconditionedDynamics{DensePreconditioner,<:ZigZag},
    },
    provider,
    curvature_bound;
    cached_gradient::Union{AbstractVector,Nothing}=nothing,
    early_stop_threshold::Float64=Inf,
    state_cache::Union{AbstractPDMPState,Nothing}=nothing,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing,
    max_time::Float64=Inf,
    build_affine::Bool=true,
    linear_area_threshold::Real=1.0,
    linear_min_area_gain::Real=0.0,
    auto::Bool=false,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    start_cell::Integer=1,
    initial_integral::Float64=0.0,
    append::Bool=false,
    max_componentwise_affine_segments_per_cell::Integer=64,
    rate_value_buf::Union{Vector{Float64},Nothing}=nothing,
    rate_derivative_buf::Union{Vector{Float64},Nothing}=nothing,
)
    build_affine && !append && reset_affine_bound!(bound)
    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    y_vals = pcb.y_vals
    d_vals = pcb.d_vals
    N = length(Λ_vals)
    iszero(t_grid[1]) || error("t_grid[1] must be zero, got $(t_grid[1])")

    n_time_cells = isfinite(max_time) ? max(0, min(N, searchsortedfirst(t_grid, max_time) - 1)) : N
    start_cell = clamp(Int(start_cell), 1, N + 1)
    start_cell > n_time_cells && return start_cell - 1

    n_channels = _rate_channel_count(state, flow)
    chunk_size = max(2, _grid_rate_derivative_chunk_points())
    loaded_start = 0
    loaded_stop = -1
    G = Matrix{Float64}(undef, n_channels, 0)
    dG = similar(G)

    L = _channel_curvature_matrix(
        curvature_bound, state, flow, t_grid, n_channels, N, stats)
    cumulative_integral = initial_integral
    N_evaluated = start_cell - 1
    for cell in start_cell:n_time_cells
        if !(loaded_start <= cell && cell + 1 <= loaded_stop)
            start_point = cell
            stop_point = min(n_time_cells + 1, max(cell + 1, cell + chunk_size - 1))
            n_points = stop_point - start_point + 1
            if rate_value_buf === nothing || rate_derivative_buf === nothing
                G = Matrix{Float64}(undef, n_channels, n_points)
                dG = similar(G)
            else
                G, dG = _rate_derivative_scratch!(
                    rate_value_buf, rate_derivative_buf, n_channels, n_points)
            end
            stats !== nothing && (_inc_counter_grid_endpoint_derivative_calls(stats))
            stats !== nothing && (_inc_counter_grid_endpoint_derivative_points_loaded(stats, n_points))
            if state_cache === nothing
                _fill_rate_derivatives!(
                    G, dG, provider, state, flow, @view(t_grid[start_point:stop_point]), n_points)
            else
                _fill_rate_derivatives!(
                    G, dG, provider, state, flow, @view(t_grid[start_point:stop_point]), n_points, state_cache)
            end
            stats !== nothing && (_inc_counter_componentwise_channels(stats, n_channels))
            stats !== nothing && (_inc_counter_componentwise_channel_point_evaluations(
                stats, n_channels * n_points))
            for point in start_point:stop_point
                offset = point - start_point + 1
                y_vals[point] = sum(pos(G[j, offset]) for j in 1:n_channels)
                d_vals[point] = sum(ispositive(G[j, offset]) ? dG[j, offset] : 0.0 for j in 1:n_channels)
            end
            loaded_start = start_point
            loaded_stop = stop_point
        end
        a = t_grid[cell]
        b = t_grid[cell + 1]
        M = 0.0
        left = cell - loaded_start + 1
        right = left + 1
        for channel in 1:n_channels
            M += _rate_flat_upper(
                a, b, G[channel, left], G[channel, right],
                dG[channel, left], dG[channel, right], L[channel, cell])
        end
        Λ_vals[cell] = M
        cell_area = M * (b - a)
        if build_affine
            start_segments = bound.n_segments
            _append_componentwise_signed_affine_cell!(
                bound, stats, a, b, G, dG, L, left, cell, M;
                linear_area_threshold,
                linear_min_area_gain,
                auto,
                max_segments=max_componentwise_affine_segments_per_cell)
            cell_area = _affine_added_area(bound, start_segments)
        elseif stats !== nothing
            _inc_counter_affine_constant_cells(stats)
            _inc_counter_affine_area_constant_equiv(stats, cell_area)
            _inc_counter_affine_area_hybrid(stats, cell_area)
            _inc_counter_componentwise_flat_fallback_cells(stats)
            _inc_counter_componentwise_affine_skipped_by_policy(stats)
            _inc_counter_componentwise_flat_fallback_policy(stats)
            if auto
                _inc_counter_auto_flat_cells(stats)
                total = _get_counter_auto_flat_cells(stats) +
                    _get_counter_auto_affine_cells(stats)
                fraction = total == 0 ? 0.0 :
                    _get_counter_auto_affine_cells(stats) / total
                _set_counter_auto_affine_fraction(stats, fraction)
            end
        end
        cumulative_integral += cell_area
        N_evaluated = cell
        if cumulative_integral >= early_stop_threshold && cell <= N
            for j in (cell + 1):N
                Λ_vals[j] = 0.0
            end
            stats !== nothing && (_inc_counter_grid_early_stops(stats))
            stats !== nothing && (_inc_counter_grid_points_skipped(stats, N - N_evaluated))
            break
        end
    end

    if stats !== nothing
        _inc_counter_grid_builds(stats)
        _inc_counter_grid_points_evaluated(stats, max(N_evaluated - start_cell + 1, 0))
    end
    return N_evaluated
end
