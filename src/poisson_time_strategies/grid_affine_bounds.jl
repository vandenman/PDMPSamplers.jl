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


function construct_rate_bound_grid!(
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
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    start_cell::Integer=1,
    initial_integral::Float64=0.0,
    append::Bool=false,
    rate_value_buf::Union{Vector{Float64},Nothing}=nothing,
    rate_derivative_buf::Union{Vector{Float64},Nothing}=nothing,
)
    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    y_vals = pcb.y_vals
    d_vals = pcb.d_vals
    N = length(Λ_vals)
    state_t = state_cache === nothing ? copy(state) : (copyto!(state_cache, state); state_cache)
    iszero(t_grid[1]) || error("t_grid[1] must be zero, got $(t_grid[1])")

    n_time_cells = _grid_cell_count(t_grid, N, max_time)
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
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    start_cell::Integer=1,
    initial_integral::Float64=0.0,
    append::Bool=false,
    rate_value_buf::Union{Vector{Float64},Nothing}=nothing,
    rate_derivative_buf::Union{Vector{Float64},Nothing}=nothing,
)
    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    y_vals = pcb.y_vals
    d_vals = pcb.d_vals
    N = length(Λ_vals)
    iszero(t_grid[1]) || error("t_grid[1] must be zero, got $(t_grid[1])")

    n_time_cells = _grid_cell_count(t_grid, N, max_time)
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
