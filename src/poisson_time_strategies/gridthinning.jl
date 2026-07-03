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

    start_point = start_cell
    stop_point = n_time_cells + 1
    n_points = stop_point - start_point + 1
    n_channels = _rate_channel_count(state, flow)
    if rate_value_buf === nothing || rate_derivative_buf === nothing
        G = Matrix{Float64}(undef, n_channels, n_points)
        dG = similar(G)
    else
        G, dG = _rate_derivative_scratch!(
            rate_value_buf, rate_derivative_buf, n_channels, n_points)
    end
    stats !== nothing && (_inc_counter_grid_endpoint_derivative_calls(stats))
    stats !== nothing && (_inc_counter_grid_endpoint_derivative_points_loaded(stats, n_points))
    _fill_rate_derivatives!(
        G, dG, provider, state, flow, @view(t_grid[start_point:stop_point]), n_points)
    stats !== nothing && (_inc_counter_componentwise_channels(stats, n_channels))
    stats !== nothing && (_inc_counter_componentwise_channel_point_evaluations(
        stats, n_channels * n_points))

    for point in start_point:stop_point
        offset = point - start_point + 1
        y_vals[point] = sum(pos(G[j, offset]) for j in 1:n_channels)
        d_vals[point] = sum(ispositive(G[j, offset]) ? dG[j, offset] : 0.0 for j in 1:n_channels)
    end

    L = _channel_curvature_matrix(
        curvature_bound, state, flow, t_grid, n_channels, N, stats)
    cumulative_integral = initial_integral
    N_evaluated = N
    for cell in start_cell:n_time_cells
        a = t_grid[cell]
        b = t_grid[cell + 1]
        M = 0.0
        for channel in 1:n_channels
            left = cell - start_point + 1
            right = left + 1
            M += _rate_flat_upper(
                a, b, G[channel, left], G[channel, right],
                dG[channel, left], dG[channel, right], L[channel, cell])
        end
        Λ_vals[cell] = M
        cell_area = M * (b - a)
        if build_affine
            start_segments = bound.n_segments
            left = cell - start_point + 1
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
        if cumulative_integral >= early_stop_threshold && cell <= N
            N_evaluated = cell
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

function _grid_bound_violation_message(
    alg,
    stats::AbstractStatisticCounter,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    τ::Real,
    signed_actual::Real,
    l_actual::Real,
    bound_actual::Real,
    cumulative_exp::Real,
    λ_refresh::Real,
    use_linear::Bool,
)
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

function _piecewise_constant_area(pcb::PiecewiseConstantBound)
    area = 0.0
    @inbounds for i in eachindex(pcb.Λ_vals)
        area += pos(pcb.Λ_vals[i]) * (pcb.t_grid[i + 1] - pcb.t_grid[i])
    end
    return area
end

_grid_built_area(pcb::PiecewiseConstantBound, bound::PiecewiseAffineBound, use_linear::Bool) =
    use_linear ? total_area(bound) : _piecewise_constant_area(pcb)

function _record_budget_grid_build!(
    stats::AbstractStatisticCounter,
    n_cells::Integer,
    built_area::Real,
    exponential_budget::Real,
    is_extension::Bool,
)
    is_extension && (_inc_counter_grid_budget_extensions(stats))
    _inc_counter_grid_budget_cells_built(stats, max(Int(n_cells), 0))
    _inc_counter_grid_budget_area_built(stats, float(built_area))
    _inc_counter_grid_budget_exponential_sum(stats, float(exponential_budget))
    return nothing
end

"""
    GridThinningStrategy(; bound=:constant, kwargs...)

Adaptive GridThinning configuration.

Preferred user-facing bounds are `:constant`, `:flat`, `:linear`, and `:auto`.
The older `bound` keyword remains accepted as a compatibility alias.
Passing neither keyword keeps the historical constant GridThinning behavior
used by downstream packages.
"""
struct GridThinningStrategy <: PoissonTimeStrategy
    N::Int
    N_min::Int
    t_max::Float64
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    early_stop_threshold::Float64
    use_fd_hvp::Bool
    post_warmup_simplify::Bool
    lazy::Bool
    bound::Symbol
    curvature_bound
    bound_violation::Symbol
    linear_area_threshold::Float64
    linear_min_area_gain::Float64
    max_rejections_before_tail_restart::Int
    max_componentwise_affine_segments_per_cell::Int
end

_normalize_grid_bound(::Nothing) = :constant

function _normalize_grid_bound(bound::Symbol)
    bound === :constant && return :constant
    bound === :flat && return :flat
    bound === :linear && return :linear
    bound === :auto && return :auto
    bound === :sticky_auto && return :sticky_auto
    throw(ArgumentError("unknown GridThinning bound $(bound)"))
end

function GridThinningStrategy(;
    N::Int=20,
    N_min::Int=5,
    t_max::Real=2.0,
    α⁺::Real=1.5,
    α⁻::Real=0.5,
    safety_limit::Int=500,
    early_stop_threshold::Real=5.0,
    use_fd_hvp::Bool=false,
    post_warmup_simplify::Bool=false,
    lazy::Bool=true,
    bound=nothing,
    curvature_bound=nothing,
    bound_violation=nothing,
    linear_area_threshold::Real=0.95,
    linear_min_area_gain::Real=0.0,
    max_rejections_before_tail_restart::Int=100,
    max_componentwise_affine_segments_per_cell::Int=64,
)
    bound_symbol = _normalize_grid_bound(bound)
    bound_violation_symbol = bound_violation === nothing ?
        (bound_symbol === :constant ? :count : :shrink) : Symbol(bound_violation)
    return GridThinningStrategy(
        N, N_min, Float64(t_max), Float64(α⁺), Float64(α⁻), safety_limit,
        Float64(early_stop_threshold), use_fd_hvp, post_warmup_simplify,
        lazy, bound_symbol, curvature_bound, bound_violation_symbol,
        Float64(linear_area_threshold),
        Float64(linear_min_area_gain),
        max_rejections_before_tail_restart, max_componentwise_affine_segments_per_cell)
end

function Base.show(io::IO, strat::GridThinningStrategy)
    print(io, "GridThinningStrategy(")
    print(io, "N=", strat.N, ", N_min=", strat.N_min, ", t_max=", strat.t_max)
    print(io, ", bound=", strat.bound)
    strat.bound in (:linear, :auto) && print(io, ", linear_area_threshold=", strat.linear_area_threshold,
        ", linear_min_area_gain=", strat.linear_min_area_gain)
    print(io, ")")
end

_default_early_stop(::ContinuousDynamics, est::Float64) = est
_default_early_stop(pd::PreconditionedDynamics, est::Float64) = _default_early_stop(pd.dynamics, est)

function _to_internal(strat::GridThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    T = typeof(strat.t_max)
    0.0 <= strat.linear_area_threshold || throw(ArgumentError("linear_area_threshold must be nonnegative"))
    0.0 <= strat.linear_min_area_gain || throw(ArgumentError("linear_min_area_gain must be nonnegative"))
    strat.max_rejections_before_tail_restart > 0 || throw(ArgumentError("max_rejections_before_tail_restart must be positive"))
    strat.max_componentwise_affine_segments_per_cell > 0 ||
        throw(ArgumentError("max_componentwise_affine_segments_per_cell must be positive"))
    # Derivative info is always available: either via HVP, VHV, joint, or FD fallback.
    N_base = strat.N
    N_min = min_grid_cells(flow, strat.N_min, N_base)
    est = _default_early_stop(flow, strat.early_stop_threshold)
    est = _adjust_early_stop(model.grad, est)
    N_max = max(N_base + 4, 2 * N_base)
    _build_grid_adaptive_state(strat, state, flow, model, cache, N_base, N_min, N_max, est)
end

_adjust_early_stop(::GradientStrategy, est::Float64) = est
_adjust_early_stop(::SubsampledGradient, ::Float64) = Inf

function _effective_grid_horizon(
    ::GradientStrategy,
    t_max::Float64,
    τ_refresh::Float64,
    max_horizon::Float64,
    max_horizon_event::Symbol=:horizon_hit,
)
    if τ_refresh <= t_max && τ_refresh <= max_horizon
        return τ_refresh, :refresh
    elseif max_horizon <= t_max
        return max_horizon, max_horizon_event
    end
    return t_max, :horizon_hit
end

function _return_grid_horizon!(alg, stats::AbstractStatisticCounter, t_max::Float64, effective_horizon::Float64, horizon_event::Symbol, max_t_max::Float64, default_return)
    _inc_counter_grid_horizon_hits(stats)
    alg.has_cached_gradient[] = false
    if horizon_event === :horizon_hit
        alg.t_max[] = min(t_max * alg.α⁺, max_t_max)
        recompute_time_grid!(alg)
        _inc_counter_grid_grows(stats)
        return t_max, :horizon_hit, default_return
    end
    return effective_horizon, horizon_event, default_return
end

function _build_grid_adaptive_state(strat::GridThinningStrategy, state::S, N_base::Int, N_min::Int, est) where S<:AbstractPDMPState
    flow = BouncyParticle(length(state.ξ))
    model = PDMPModel(length(state.ξ), FullGradient((out, x) -> copyto!(out, x)))
    cache = (; ∇ϕx=similar(state.ξ.x))
    return _build_grid_adaptive_state(strat, state, flow, model, cache, N_base, N_min, N_base, est)
end

function _build_grid_adaptive_state(strat::GridThinningStrategy, state::S, flow::ContinuousDynamics, model::PDMPModel, cache, N_base::Int, N_min::Int, est) where S<:AbstractPDMPState
    return _build_grid_adaptive_state(strat, state, flow, model, cache, N_base, N_min, N_base, est)
end

function _build_grid_adaptive_state(strat::GridThinningStrategy, state::S, flow::ContinuousDynamics, model::PDMPModel, cache, N_base::Int, N_min::Int, N_max::Int, est) where S<:AbstractPDMPState
    T = typeof(strat.t_max)
    state_cache = copy(state)
    state_cache2 = copy(state)
    grad_provider = GradientProvider(state_cache.ξ.θ, flow, model.grad, cache)
    GridAdaptiveState(
        PiecewiseConstantBound(collect(range(0.0, strat.t_max, N_base + 1)), zeros(T, N_base)),
        PiecewiseAffineBound(2N_base),
        Base.RefValue{Int}(N_base),
        Base.RefValue{Float64}(strat.t_max),
        strat.α⁺,
        strat.α⁻,
        strat.safety_limit,
        N_min,
        N_max,
        est,
        state_cache,
        state_cache2,
        similar(state.ξ.x, 0),
        strat.use_fd_hvp,
        similar(state.ξ.x),
        similar(state.ξ.x),
        similar(state.ξ.x),
        Float64[],
        Float64[],
        Ref(NaN),
        Ref(0.0),
        strat.post_warmup_simplify,
        strat.lazy,
        Ref(strat.lazy),
        similar(state.ξ.x),
        Ref(false),
        grad_provider,
        GradHVPProvider(grad_provider, model.hvp),
        VHVProvider(grad_provider, model.vhv, similar(state.ξ.x)),
        FiniteDiffVHV(grad_provider, similar(state.ξ.x), similar(state.ξ.x), similar(state.ξ.x)),
        strat.bound,
        strat.curvature_bound,
        strat.bound_violation,
        strat.linear_area_threshold,
        strat.linear_min_area_gain,
        strat.max_rejections_before_tail_restart,
        strat.max_componentwise_affine_segments_per_cell,
    )
end

struct GridAdaptiveState{S<:AbstractPDMPState,V<:AbstractVector,P,GH,VP,FD} <: PoissonTimeStrategy
    pcb::PiecewiseConstantBound{Float64}
    affine_bound::PiecewiseAffineBound{Float64}
    N::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    N_min::Int
    N_max::Int
    early_stop_threshold::Float64
    state_cache::S
    state_cache2::S
    empty_∇ϕx::V
    use_fd_hvp::Bool
    fd_buf::Vector{Float64}
    fd_grad_buf::Vector{Float64}
    fd_w_buf::Vector{Float64}
    rate_value_buf::Vector{Float64}
    rate_derivative_buf::Vector{Float64}
    constant_bound_rate::Base.RefValue{Float64}
    max_observed_rate::Base.RefValue{Float64}
    post_warmup_simplify::Bool
    lazy::Bool
    lazy_enabled::Base.RefValue{Bool}
    cached_gradient::Vector{Float64}
    has_cached_gradient::Base.RefValue{Bool}
    grad_provider::P
    grad_hvp_provider::GH
    vhv_provider::VP
    fd_vhv_provider::FD
    bound::Symbol
    curvature_bound
    bound_violation::Symbol
    linear_area_threshold::Float64
    linear_min_area_gain::Float64
    max_rejections_before_tail_restart::Int
    max_componentwise_affine_segments_per_cell::Int
end

accept_reflection_event(::Random.AbstractRNG, ::GridAdaptiveState, args...) = true
accept_reflection_event(::GridAdaptiveState, args...) = true

recompute_time_grid!(alg::GridAdaptiveState) = recompute_time_grid!(alg.pcb, alg.t_max[], alg.N[])

function reset_grid_scale!(alg::GridAdaptiveState, t_max::Float64=2.0)
    alg.t_max[] = t_max
    alg.N[] = alg.N_max
    alg.lazy_enabled[] = alg.lazy
    alg.has_cached_gradient[] = false
    recompute_time_grid!(alg)
end

function _shrink_grid_after_bound_violation!(alg::GridAdaptiveState, stats::AbstractStatisticCounter)
    new_t_max = max(alg.t_max[] * alg.α⁻, sqrt(eps(Float64)))
    reset_grid_scale!(alg, new_t_max)
    _inc_counter_grid_shrinks(stats)
    return nothing
end

function _invalidate_cached_gradient!(alg::GridAdaptiveState)
    alg.has_cached_gradient[] = false
    return nothing
end

function _constant_bound_event_time(
    rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
)
    λ_bound = alg.constant_bound_rate[]
    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    τ_refresh = ispositive(λ_refresh) ? rand(rng, Exponential(inv(λ_refresh))) : Inf
    default_return = GradientMeta(alg.empty_∇ϕx)

    state_ = alg.state_cache
    copyto!(state_, state)
    t_max, horizon_event = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)

    cumulative_exp = 0.0
    for _ in 1:alg.safety_limit
        cumulative_exp += rand(rng, Exponential())
        τ_proposal = cumulative_exp / λ_bound

        if τ_proposal >= t_max
            _inc_counter_grid_horizon_hits(stats)
            alg.has_cached_gradient[] = false
            return t_max, horizon_event, default_return
        end

        if τ_refresh < τ_proposal
            alg.has_cached_gradient[] = false
            return τ_refresh, :refresh, default_return
        end

        _inc_counter_constant_bound_attempts(stats)
        copyto!(state_, state)
        move_forward_time!(state_, τ_proposal, flow)
        _inc_counter_grid_acceptance_gradient_calls(stats)
        ∇ϕx = _compute_grid_gradient_or_throw!(
            state_, state, flow, model, cache, 0.0, τ_proposal, probe_failure_handler)
        l_actual = λ(state_.ξ, ∇ϕx, flow)
        _inc_counter_grid_acceptance_tests(stats)

        if l_actual > λ_bound
            _inc_counter_constant_bound_violations(stats)
            _inc_counter_grid_bound_violations(stats)
            if alg.bound_violation === :throw
                signed_actual = flow isa BouncyParticle ? dot(∇ϕx, state_.ξ.θ) : NaN
                throw(ErrorException(_grid_bound_violation_message(
                    alg, stats, state_, flow, τ_proposal, signed_actual, l_actual,
                    λ_bound, cumulative_exp, λ_refresh, false)))
            elseif alg.bound_violation === :shrink
                alg.constant_bound_rate[] = NaN
                _shrink_grid_after_bound_violation!(alg, stats)
                return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
                    max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end
            alg.constant_bound_rate[] = NaN
            return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
                max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
        end

        if rand(rng) * λ_bound <= l_actual
            alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
            _inc_counter_constant_bound_accepts(stats)
            return τ_proposal, :reflect, GradientMeta(∇ϕx)
        end
        _inc_counter_constant_bound_rejections(stats)
    end

    _inc_counter_constant_bound_safety_fallbacks(stats)
    alg.constant_bound_rate[] = NaN
    return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _constant_bound_event_time(
    model::PDMPModel{<:GlobalGradientStrategy},
    flow::ContinuousDynamics,
    alg::GridAdaptiveState,
    state::AbstractPDMPState,
    cache,
    stats::AbstractStatisticCounter,
    max_horizon::Float64,
    include_refresh::Bool,
    max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
)
    return _constant_bound_event_time(
        Random.default_rng(), model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler,
    )
end


function _make_grad_provider(grad_func, model::PDMPModel, flow::ContinuousDynamics, alg::GridAdaptiveState)
    joint_func = model.joint
    if joint_func !== nothing && _joint_compatible(flow)
        return joint_func
    end
    vhv_func = model.vhv
    if vhv_func !== nothing
        return alg.vhv_provider
    end
    hvp_func = model.hvp
    if hvp_func === nothing
        # Always fall back to finite-diff curvature when no HVP is available.
        # This gives much tighter bounds than gradient-only mode.
        return alg.fd_vhv_provider
    end
    return alg.grad_hvp_provider
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64=Inf, include_refresh::Bool=true, max_horizon_event::Symbol=:horizon_hit) where {FL<:ContinuousDynamics}
    return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, NoGridBoundaryProbe())
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol,
    detect_boundaries::Bool) where {FL<:ContinuousDynamics}
    detect_boundaries || return next_event_time(rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event)
    probe_failure_handler = _grid_probe_failure_handler(state, flow, model, GridThinningStrategy)
    return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _next_event_time_with_probe(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol,
    probe_failure_handler::GridBoundaryProbe) where {FL<:ContinuousDynamics}
    if isfinite(alg.constant_bound_rate[])
        return _constant_bound_event_time(rng, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end

    state_ = alg.state_cache
    copyto!(state_, state)

    grad_and_hvp = _make_grad_provider(alg.grad_provider, model, flow, alg)

    # Function barrier: specialized on the concrete type of grad_and_hvp
    if alg.lazy_enabled[]
        return _next_event_time_lazy!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end
    return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _next_event_time_grid!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe()) where {P, FL<:ContinuousDynamics}

    pcb = alg.pcb
    state_ = alg.state_cache
    copyto!(state_, state)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))

    default_return = GradientMeta(alg.empty_∇ϕx)

    # Draw refresh time FIRST so we can cap grid construction (Phase 1A)
    τ_refresh = ispositive(λ_refresh) ? rand(rng, Exponential(inv(λ_refresh))) : Inf
    effective_horizon, horizon_event = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)

    # Budget-first grid construction: draw the first exponential budget before
    # building the grid, then construct only enough bound area to cover it.
    # Rejections add exponential increments and extend/rebuild the grid only
    # when the cumulative budget exceeds the already-built area.
    cumulative_exp = rand(rng, Exponential())

    # Build grid once for this event, capped at effective horizon
    use_linear = _use_linear_bound(alg, state, flow, grad_and_hvp)
    use_flat = _use_flat_bound(alg, state, flow, grad_and_hvp)
    use_single_pass_signed = _use_signed_grid_bound(alg, state, flow, grad_and_hvp)
    use_constant_batched_signed = !use_single_pass_signed && _supports_constant_grid_rate_derivatives(flow, grad_and_hvp)
    had_cached_gradient = alg.has_cached_gradient[]
    if use_single_pass_signed
        cached_gradient = had_cached_gradient ? alg.cached_gradient : nothing
        alg.has_cached_gradient[] = false
        n_cells_bounded = construct_rate_bound_grid!(
            alg.affine_bound, pcb, state, flow, grad_and_hvp, alg.curvature_bound;
            cached_gradient,
            early_stop_threshold=cumulative_exp,
            state_cache=state_,
            stats,
            max_time=effective_horizon,
            build_affine=use_linear,
            linear_area_threshold=alg.linear_area_threshold,
            linear_min_area_gain=alg.linear_min_area_gain,
            auto=_auto_policy(alg.bound),
            max_componentwise_affine_segments_per_cell=
                alg.max_componentwise_affine_segments_per_cell,
            rate_value_buf=alg.rate_value_buf,
            rate_derivative_buf=alg.rate_derivative_buf,
            probe_failure_handler,
        )
    elseif had_cached_gradient && !use_constant_batched_signed
        cached_y0, cached_d0 = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false, alg.cached_gradient;
            t_valid=0.0, t_invalid=0.0)
        cached_g0, cached_dg0 = if alg.bound === :linear
            rate_and_derivative(state_, flow, grad_and_hvp, alg.cached_gradient)
        else
            (NaN, NaN)
        end
        alg.has_cached_gradient[] = false
    else
        cached_y0, cached_d0 = NaN, NaN
        cached_g0, cached_dg0 = NaN, NaN
        use_constant_batched_signed && (alg.has_cached_gradient[] = false)
    end
    if !use_single_pass_signed
        n_cells_bounded = construct_upper_bound_grad_and_hess!(pcb, state_, flow, grad_and_hvp, false;
            cached_y0, cached_d0,
            early_stop_threshold=cumulative_exp, stats, state_cache=state_,
            max_time=effective_horizon, probe_failure_handler)
    end
    if use_linear && !use_single_pass_signed
        if alg.bound === :linear
            construct_rate_grid!(pcb, state, flow, grad_and_hvp, n_cells_bounded;
                cached_g0, cached_dg0, state_cache=state_, stats)
            build_rate_linear_bound!(alg.affine_bound, pcb, n_cells_bounded, state, flow, alg.curvature_bound, stats;
                linear_area_threshold=alg.linear_area_threshold,
                linear_min_area_gain=alg.linear_min_area_gain)
        else
            build_hybrid_affine_bound!(alg.affine_bound, pcb, n_cells_bounded, stats)
        end
    end
    _record_grid_schedule!(stats, alg)
    _set_counter_grid_N_current(stats, alg.N[])
    built_area = _grid_built_area(pcb, alg.affine_bound, use_linear)
    _record_budget_grid_build!(stats, n_cells_bounded, built_area, cumulative_exp, false)

    # Budget-first exactness invariant:
    #   * once proposal times have been generated from the built dominating
    #     bound prefix, that prefix must never be changed;
    #   * after a rejection, a larger exponential budget may only append later
    #     cells/segments to the existing prefix;
    #   * if many rejections force a safety rebuild, the old local origin is
    #     abandoned only after the last rejected time.  The restarted problem
    #     begins at that time with a fresh exponential budget and all returned
    #     times are offset back to the original local origin.
    rejection_count = 0
    max_rejections = alg.max_rejections_before_tail_restart
    last_rejected_time = 0.0

    max_t_max = max_grid_horizon(flow)

    safety_limit = alg.safety_limit
    while safety_limit > 0
        τ_reflection, lb_reflection = if use_linear
            propose_event_time(rng, alg.affine_bound, cumulative_exp)
        else
            propose_event_time(rng, pcb, cumulative_exp)
        end

        if τ_reflection >= effective_horizon

            return _return_grid_horizon!(alg, stats, alg.t_max[], effective_horizon, horizon_event, max_t_max, default_return)
        end

        if τ_refresh < τ_reflection
            alg.has_cached_gradient[] = false
            return τ_refresh, :refresh, default_return
        end

        # Move from original position to proposed time for acceptance test
        copyto!(state_, state)
        move_forward_time!(state_, τ_reflection, flow)
        _inc_counter_grid_acceptance_gradient_calls(stats)
        ∇ϕx = _compute_grid_gradient_or_throw!(
            state_, state, flow, model, cache, 0.0, τ_reflection, probe_failure_handler)

        l_reflection = λ(state_.ξ, ∇ϕx, flow)
        _inc_counter_grid_acceptance_tests(stats)
        if l_reflection > lb_reflection * (1 + 1e-10) + 1e-12
            _inc_counter_grid_bound_violations(stats)
            signed_reflection = flow isa BouncyParticle ? dot(∇ϕx, state_.ξ.θ) : NaN
            msg = _grid_bound_violation_message(
                alg, stats, state_, flow, τ_reflection, signed_reflection, l_reflection,
                lb_reflection, cumulative_exp, λ_refresh, use_linear)
            if use_linear
                _inc_counter_affine_bound_violations(stats)
            end
            if alg.bound_violation === :throw
                throw(ErrorException(msg))
            elseif alg.bound_violation === :shrink
                _shrink_grid_after_bound_violation!(alg, stats)
                return _next_event_time_grid!(
                    rng, grad_and_hvp, model, flow, alg, state, cache, stats,
                    max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end
        end

        if rand(rng) * lb_reflection <= l_reflection
            tightness = l_reflection / lb_reflection
            _adapt_grid_N!(alg, tightness)
            _adapt_grid_t_max!(alg, τ_reflection, model.grad)
            _set_counter_grid_N_current(stats, alg.N[])
            alg.max_observed_rate[] = max(alg.max_observed_rate[], l_reflection)
            copyto!(alg.cached_gradient, ∇ϕx)
            alg.has_cached_gradient[] = true
            return τ_reflection, :reflect, GradientMeta(∇ϕx)
        end

        # Rejection: cumulative_exp has advanced, next proposal will be at a later time
        rejection_count += 1
        last_rejected_time = τ_reflection
        cumulative_exp += rand(rng, Exponential())
        if cumulative_exp > built_area * (1 + 64eps(Float64)) + 64eps(Float64)
            start_cell = n_cells_bounded + 1
            effective_horizon, horizon_event = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)
            use_linear = _use_linear_bound(alg, state, flow, grad_and_hvp)
            use_flat = _use_flat_bound(alg, state, flow, grad_and_hvp)
            use_single_pass_signed = _use_signed_grid_bound(alg, state, flow, grad_and_hvp)
            use_constant_batched_signed = !use_single_pass_signed && _supports_constant_grid_rate_derivatives(flow, grad_and_hvp)
            use_constant_batched_signed && (alg.has_cached_gradient[] = false)
            if use_single_pass_signed
                                                n_cells_bounded = construct_rate_bound_grid!(
                    alg.affine_bound, pcb, state, flow, grad_and_hvp, alg.curvature_bound;
                    early_stop_threshold=cumulative_exp,
                    state_cache=state_,
                    stats,
                    max_time=effective_horizon,
                    build_affine=use_linear,
                    linear_area_threshold=alg.linear_area_threshold,
                    linear_min_area_gain=alg.linear_min_area_gain,
                    auto=_auto_policy(alg.bound),
                    max_componentwise_affine_segments_per_cell=
                        alg.max_componentwise_affine_segments_per_cell,
                    rate_value_buf=alg.rate_value_buf,
                    rate_derivative_buf=alg.rate_derivative_buf,
                    probe_failure_handler,
                    start_cell,
                    initial_integral=built_area,
                    append=true,
                )
            else
                n_cells_bounded = construct_upper_bound_grad_and_hess!(pcb, state, flow, grad_and_hvp, false;
                    early_stop_threshold=cumulative_exp, stats, state_cache=state_,
                    max_time=effective_horizon, probe_failure_handler,
                    start_cell, initial_integral=built_area)
            end
            if use_linear && !use_single_pass_signed
                if alg.bound === :linear
                    construct_rate_grid!(pcb, state, flow, grad_and_hvp, n_cells_bounded;
                        state_cache=state_, stats)
                    build_rate_linear_bound!(alg.affine_bound, pcb, n_cells_bounded, state, flow, alg.curvature_bound, stats;
                        linear_area_threshold=alg.linear_area_threshold,
                        linear_min_area_gain=alg.linear_min_area_gain)
                else
                    build_hybrid_affine_bound!(alg.affine_bound, pcb, n_cells_bounded, stats)
                end
            end
            _record_grid_schedule!(stats, alg)
            built_area = _grid_built_area(pcb, alg.affine_bound, use_linear)
            _record_budget_grid_build!(stats, n_cells_bounded, built_area, cumulative_exp, true)
        end
        if rejection_count >= max_rejections
            # Too many rejections.  Do not restart the dominating process at
            # the original time: proposals before the last rejected time have
            # already been consumed.  Instead restart a fresh local thinning
            # problem from the last rejected time and offset the returned time.
            _increase_grid_N!(alg)
            recompute_time_grid!(alg)

            # Also shrink t_max when the grid integral greatly exceeds
            # cumulative_exp, indicating most of the domain has zero rate.
            _shrink_t_max_on_rejection!(alg, pcb, cumulative_exp, model.grad)

            _inc_counter_grid_shrinks(stats)
            _inc_counter_grid_budget_tail_restarts(stats)
            alg.has_cached_gradient[] = false

            remaining_horizon = effective_horizon - last_rejected_time
            if remaining_horizon <= 0
                return last_rejected_time, horizon_event, default_return
            end

            tail_state = alg.state_cache2
            copyto!(tail_state, state)
            move_forward_time!(tail_state, last_rejected_time, flow)
            τ_tail, event_type, meta = _next_event_time_with_probe(
                rng, model, flow, alg, tail_state, cache, stats,
                remaining_horizon, false, horizon_event, probe_failure_handler)
            return last_rejected_time + τ_tail, event_type, meta
        end
        safety_limit -= 1
    end

    if isfinite(τ_refresh) && τ_refresh <= min(alg.t_max[], max_horizon)
        alg.has_cached_gradient[] = false
        return τ_refresh, :refresh, default_return
    end

    _throw_grid_safety_limit_error(state, flow, model;
        t_invalid=effective_horizon, message="Safety limit reached")
end

_metric_scale_extrema(::ContinuousDynamics) = (NaN, NaN)

function _metric_scale_extrema(flow::PreconditionedDynamics{<:DiagonalPreconditioner})
    scales = flow.metric.scale
    return minimum(scales), maximum(scales)
end

function _metric_scale_extrema(flow::DensePreconditionedBPS)
    diag_entries = diag(flow.metric.L)
    return minimum(diag_entries), maximum(diag_entries)
end

function _metric_scale_extrema(flow::DensePreconditionedZigZag)
    diag_entries = diag(flow.metric.L)
    return minimum(diag_entries), maximum(diag_entries)
end

_safe_tightness(l_actual::Real, Λ_cell::Real) = ispositive(Λ_cell) ? l_actual / pos(Λ_cell) : NaN

function _record_lazy_search_stats!(stats::AbstractStatisticCounter, proposal_attempts::Int, proposal_rejections::Int)
    _inc_counter_lazy_proposal_attempts(stats, proposal_attempts)
    _inc_counter_lazy_proposal_rejections(stats, proposal_rejections)
    return nothing
end

function _lazy_grid_failure_message(; alg::GridAdaptiveState, flow::ContinuousDynamics,
    state_time::Float64, effective_horizon::Float64, Δt::Float64, no_event_cells::Int, proposal_attempts::Int,
    proposal_rejections::Int, tightness_sum::Float64, min_tightness::Float64, max_tightness::Float64,
    last_t_left::Float64, last_t_right::Float64, last_y_left::Float64, last_y_right::Float64,
    last_d_left::Float64, last_d_right::Float64, last_Λ_cell::Float64, last_exp_target::Float64,
    last_cumulative_area::Float64, last_τ_proposal::Float64, last_l_actual::Float64)

    mean_tightness = proposal_attempts == 0 ? NaN : tightness_sum / proposal_attempts
    last_tightness = _safe_tightness(last_l_actual, last_Λ_cell)
    scale_min, scale_max = _metric_scale_extrema(flow)
    return string(
        "Safety limit reached in lazy grid",
        " | N=", alg.N[],
        " t_max=", alg.t_max[],
        " state_time=", state_time,
        " effective_horizon=", effective_horizon,
        " Δt=", Δt,
        " | no_event_cells=", no_event_cells,
        " proposal_attempts=", proposal_attempts,
        " proposal_rejections=", proposal_rejections,
        " | last_t=[", last_t_left, ", ", last_t_right, "]",
        " τ=", last_τ_proposal,
        " | last_rate=[", last_y_left, ", ", last_y_right, "]",
        " last_deriv=[", last_d_left, ", ", last_d_right, "]",
        " | last_bound=", last_Λ_cell,
        " last_actual=", last_l_actual,
        " last_tightness=", last_tightness,
        " | tightness[min/mean/max]=[", min_tightness, ", ", mean_tightness, ", ", max_tightness, "]",
        " | last_exp_target=", last_exp_target,
        " last_cumulative_area=", last_cumulative_area,
        " | metric_scale[min,max]=[", scale_min, ", ", scale_max, "]"
    )
end

# ── Lazy grid evaluation (Phase 2) ──────────────────────────────────────────
# Interleaves grid point evaluation with proposal generation so that for
# well-adapted samplers only the first few intervals are evaluated.

function _next_event_time_lazy!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    ) where {P, FL<:ContinuousDynamics}

    state_ = alg.state_cache
    state2_ = alg.state_cache2
    copyto!(state_, state)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    default_return = GradientMeta(alg.empty_∇ϕx)

    τ_refresh = ispositive(λ_refresh) ? rand(rng, Exponential(inv(λ_refresh))) : Inf

    N = alg.N[]
    t_max = alg.t_max[]
    effective_horizon, horizon_event = _effective_grid_horizon(model.grad, t_max, τ_refresh, max_horizon, max_horizon_event)
    Δt = effective_horizon / N
    max_t_max = max_grid_horizon(flow)

    # Evaluate initial grid point (k=0)
    if alg.has_cached_gradient[]
        # Reuse cached gradient from previous event (2C): skip one gradient call.
        _inc_counter_grid_cached_endpoint_reuses(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_left, d_left = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false, alg.cached_gradient;
            t_valid=0.0, t_invalid=0.0)
        alg.has_cached_gradient[] = false
    else
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_left, d_left = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false;
            t_valid=0.0, t_invalid=0.0)
    end
    t_left = 0.0

    cumulative_area = 0.0
    exp_target = rand(rng, Exponential())
    safety_limit = alg.safety_limit
    max_rejections = min(25, max(10, alg.safety_limit ÷ 2))
    low_tightness_rejections = 0
    low_tightness_threshold = 0.1
    max_low_tightness_rejections = 3
    no_event_cells = 0
    proposal_attempts = 0
    proposal_rejections = 0
    tightness_sum = 0.0
    min_tightness = Inf
    max_tightness = -Inf
    last_t_left = NaN
    last_t_right = NaN
    last_y_left = NaN
    last_y_right = NaN
    last_d_left = NaN
    last_d_right = NaN
    last_Λ_cell = NaN
    last_τ_proposal = NaN
    last_l_actual = NaN
    last_exp_target = NaN
    last_cumulative_area = NaN

    _inc_counter_grid_builds(stats)
    _inc_counter_grid_points_evaluated(stats, 1)
    _record_grid_schedule!(stats, alg)

    while safety_limit > 0
        safety_limit -= 1

        # Advance state to next grid point
        t_right = t_left + Δt
        if t_right > effective_horizon
            t_right = effective_horizon
        end
        Δt_cell = t_right - t_left

        if Δt_cell <= 0.0
            # Reached the horizon
            return _return_grid_horizon!(alg, stats, t_max, effective_horizon, horizon_event, max_t_max, default_return)
        end

        # Move state_cache forward by Δt_cell to evaluate the right endpoint
        move_forward_time!(state_, Δt_cell, flow)
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_right, d_right = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false;
            t_valid=t_left, t_invalid=t_right)
        _inc_counter_grid_points_evaluated(stats, 1)

        # Compute piecewise constant bound for this interval
        Λ_cell = _tangent_intersection_bound(t_left, t_right, y_left, y_right, d_left, d_right)
        area_cell = pos(Λ_cell) * Δt_cell
        last_t_left = t_left
        last_t_right = t_right
        last_y_left = y_left
        last_y_right = y_right
        last_d_left = d_left
        last_d_right = d_right
        last_Λ_cell = Λ_cell
        last_exp_target = exp_target
        last_cumulative_area = cumulative_area

        if cumulative_area + area_cell < exp_target
            # No event in this interval — advance
            no_event_cells += 1
            cumulative_area += area_cell
            t_left = t_right
            y_left = y_right
            d_left = d_right

            # Check if we've exhausted the horizon
            if t_right >= effective_horizon
                return _return_grid_horizon!(alg, stats, t_max, effective_horizon, horizon_event, max_t_max, default_return)
            end
            continue
        end

        # Event is in this interval — propose time via inverse CDF
        u_remaining = exp_target - cumulative_area
        time_in_cell = u_remaining / pos(Λ_cell)
        τ_proposal = t_left + time_in_cell

        # Check refresh
        if τ_refresh < τ_proposal
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            return τ_refresh, :refresh, default_return
        end

        # Acceptance test: move from original state to proposed time
        state2_.t[] = state.t[]
        copyto!(state2_.ξ, state.ξ)
        move_forward_time!(state2_, τ_proposal, flow)
        _inc_counter_grid_acceptance_gradient_calls(stats)
        ∇ϕx = _compute_grid_gradient_or_throw!(
            state2_, state, flow, model, cache, t_left, τ_proposal, probe_failure_handler)

        l_actual = λ(state2_.ξ, ∇ϕx, flow)
        _inc_counter_grid_acceptance_tests(stats)
        proposal_attempts += 1
        last_τ_proposal = τ_proposal
        last_l_actual = l_actual

        tightness = _safe_tightness(l_actual, Λ_cell)
        tightness_sum += tightness
        min_tightness = min(min_tightness, tightness)
        max_tightness = max(max_tightness, tightness)

        if l_actual > pos(Λ_cell)
            # Safety violation — fall back to eager grid with finer N
            _inc_counter_lazy_fallback_bound_violation(stats)
            _inc_counter_grid_bound_violations(stats)
            if alg.bound_violation === :throw
                throw(ErrorException("lazy constant GridThinning bound violated at proposal time"))
            elseif alg.bound_violation === :shrink
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.lazy_enabled[] = false
                alg.has_cached_gradient[] = false
                _shrink_grid_after_bound_violation!(alg, stats)
                return _next_event_time_grid!(
                    rng, grad_and_hvp, model, flow, alg, state, cache, stats,
                    max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            alg.lazy_enabled[] = false
            alg.has_cached_gradient[] = false
            _increase_grid_N!(alg)
            recompute_time_grid!(alg)
            return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
        end

        if rand(rng) * pos(Λ_cell) <= l_actual
            # Accepted — cache gradient for next call (2C)
            copyto!(alg.cached_gradient, ∇ϕx)
            alg.has_cached_gradient[] = true

            _adapt_grid_N!(alg, tightness)
            _adapt_grid_t_max!(alg, τ_proposal, model.grad)
            _set_counter_grid_N_current(stats, alg.N[])
            alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            return τ_proposal, :reflect, GradientMeta(∇ϕx)
        end

        # Rejected — recycle gradient (2B).
        # The gradient ∇ϕx from compute_gradient! is still valid at τ_proposal.
        # Use it as cached gradient to save one gradient call in get_rate_and_deriv.
        proposal_rejections += 1
        if tightness < low_tightness_threshold
            low_tightness_rejections += 1
        else
            low_tightness_rejections = 0
        end

        if low_tightness_rejections >= max_low_tightness_rejections
            _inc_counter_lazy_fallback_low_tightness(stats)
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            alg.lazy_enabled[] = false
            alg.has_cached_gradient[] = false
            _increase_grid_N!(alg)
            recompute_time_grid!(alg)
            return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
        end

        if proposal_rejections >= max_rejections
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            alg.lazy_enabled[] = false
            alg.has_cached_gradient[] = false
            _increase_grid_N!(alg)
            recompute_time_grid!(alg)
            return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
        end
        copyto!(state_, state2_)
        _inc_counter_grid_cached_endpoint_reuses(stats)
        y_left, d_left = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false, ∇ϕx;
            t_valid=τ_proposal, t_invalid=τ_proposal + eps(Float64))
        t_left = τ_proposal
        cumulative_area = 0.0
        exp_target = rand(rng, Exponential())
    end

    if isfinite(τ_refresh) && τ_refresh <= min(t_max, max_horizon)
        return τ_refresh, :refresh, default_return
    end

    message = _lazy_grid_failure_message(; alg, flow, state_time=state.t[], effective_horizon, Δt, no_event_cells,
        proposal_attempts, proposal_rejections, tightness_sum, min_tightness, max_tightness,
        last_t_left, last_t_right, last_y_left, last_y_right, last_d_left, last_d_right,
        last_Λ_cell, last_exp_target, last_cumulative_area, last_τ_proposal, last_l_actual)
    _throw_grid_safety_limit_error(state, flow, model; t_invalid=effective_horizon, message)
end

function _adapt_grid_N!(alg::GridAdaptiveState, tightness::Float64)
    N = alg.N[]
    if tightness > 0.5 && N > alg.N_min
        # Bounds are tight enough, try fewer grid cells
        new_N = max(alg.N_min, N - 2)
        if new_N != N
            alg.N[] = new_N
            recompute_time_grid!(alg)
        end
    elseif tightness < 0.1 && N < alg.N_max
        _increase_grid_N!(alg)
    end
end

function _increase_grid_N!(alg::GridAdaptiveState)
    N = alg.N[]
    new_N = min(alg.N_max, N + 4)
    if new_N != N
        alg.N[] = new_N
        recompute_time_grid!(alg)
    end
end

function _adapt_grid_t_max!(alg::GridAdaptiveState, τ_accepted::Float64, ::GradientStrategy)
    t_max = alg.t_max[]
    if τ_accepted < 0.25 * t_max
        new_t_max = max(4.0 * τ_accepted, 0.1)
        if new_t_max < t_max
            alg.t_max[] = new_t_max
            recompute_time_grid!(alg)
        end
    end
end
function _adapt_grid_t_max!(alg::GridAdaptiveState, τ_accepted::Float64, ::SubsampledGradient)
    t_max = alg.t_max[]
    if τ_accepted < 0.05 * t_max
        new_t_max = max(20.0 * τ_accepted, 0.5)
        if new_t_max < t_max
            alg.t_max[] = new_t_max
            recompute_time_grid!(alg)
        end
    end
end

function _shrink_t_max_on_rejection!(alg::GridAdaptiveState, pcb::PiecewiseConstantBound, cumulative_exp::Float64, ::GradientStrategy)
    total_integral = sum(i -> pos(pcb.Λ_vals[i]) * (pcb.t_grid[i+1] - pcb.t_grid[i]), 1:alg.N[])
    if total_integral > 0 && cumulative_exp < 0.1 * total_integral
        alg.t_max[] = max(alg.t_max[] * alg.α⁻, 0.1)
        recompute_time_grid!(alg)
    end
end
_shrink_t_max_on_rejection!(::GridAdaptiveState, ::PiecewiseConstantBound, ::Float64, ::SubsampledGradient) = nothing

_reset_inner_grid!(alg::GridAdaptiveState) = reset_grid_scale!(alg)

function _maybe_activate_constant_bound!(alg::GridAdaptiveState, stats::AbstractStatisticCounter)
    alg.post_warmup_simplify || return nothing
    isfinite(alg.constant_bound_rate[]) && return nothing
    total_events = _get_counter_reflections_accepted(stats) + _get_counter_refreshment_events(stats)
    total_events < 10 && return nothing
    reflection_ratio = _get_counter_reflections_accepted(stats) / total_events
    reflection_ratio > 0.3 && return nothing
    max_rate = alg.max_observed_rate[]
    max_rate <= 0.0 && return nothing
    alg.constant_bound_rate[] = max_rate * 2.0
    return nothing
end

# ── Support-boundary helpers for grid thinning ───────────────────────────────

function _grid_probe_failure_handler(
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    algorithm_type::Type
)
    return GridBoundaryProbeHandler(original_state, flow, model, algorithm_type)
end

function _throw_grid_boundary_error(
    probe::GridBoundaryProbeHandler{S,F,M,A},
    current_state::AbstractPDMPState,
    err::Exception;
    t_valid::Float64=0.0,
    t_invalid::Float64=current_state.t[] - probe.original_state.t[],
) where {S,F,M,A}
    return _throw_grid_boundary_error(
        current_state, probe.original_state, probe.flow, probe.model, err;
        t_valid, t_invalid, algorithm_type=A)
end

function _throw_grid_boundary_error(
    current_state::AbstractPDMPState,
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    err::Exception;
    t_valid::Float64=0.0,
    t_invalid::Float64=current_state.t[] - original_state.t[],
    algorithm_type::Type=GridThinningStrategy
)
    x0 = copy(original_state.ξ.x)
    v = copy(original_state.ξ.θ)
    t_valid = max(t_valid, 0.0)
    t_invalid = max(t_invalid, t_valid + eps(Float64))
    ctx = BoundaryContext(
        x0, v, Float64(original_state.t[]), t_valid, t_invalid,
        err, typeof(flow), algorithm_type,
    )
    if _is_bridgestan_probe_error(err)
        throw(_ProbeFailureException(ctx))
    end
    if _support_boundary_probe_is_valid(model, ctx, t_invalid)
        if err isa ErrorException && occursin("Outside support", err.msg)
            throw(_ProbeFailureException(ctx))
        end
        throw(MethodError(_throw_grid_boundary_error, (current_state, original_state, flow, model, err)))
    end
    throw(_ProbeFailureException(ctx))
end

function _throw_grid_boundary_error(
    current_state::AbstractPDMPState,
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    err::_ProbeFailureException;
    kwargs...
)
    rethrow(err)
end

function _throw_grid_safety_limit_error(
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel;
    t_invalid::Float64,
    message::String,
    algorithm_type::Type=GridThinningStrategy
)
    t_invalid = max(t_invalid, eps(Float64))
    ctx = BoundaryContext(
        copy(original_state.ξ.x), copy(original_state.ξ.θ), Float64(original_state.t[]),
        0.0, t_invalid, ErrorException(message), typeof(flow), algorithm_type,
    )
    throw(_GridSafetyLimitException(ctx))
end
