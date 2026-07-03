
"""
A structure to hold the piecewise constant upper bound.
Contains the grid points and the constant rate values on each segment.
"""
struct PiecewiseConstantBound{T<:Real}
    t_grid::Vector{T} # Grid points [t_0, t_1, ..., t_N]
    Λ_vals::Vector{T} # Bound values [Λ_0, Λ_1, ..., Λ_{N-1}] on each interval
    y_vals::Vector{T} # y values [y_0, y_1, ..., y_{N-1}] on each interval
    d_vals::Vector{T} # d values [d_0, d_1, ..., d_{N-1}] on each interval
end
function PiecewiseConstantBound(t_grid::AbstractVector, Λ_vals::AbstractVector)
    return PiecewiseConstantBound(collect(t_grid), collect(Λ_vals), similar(Λ_vals, length(Λ_vals) + 1), similar(Λ_vals, length(Λ_vals) + 1))
end

"""
    (bound::PiecewiseConstantBound)(t::Real)

Functor to evaluate the piecewise constant bound Λ(t) at a given time t.
"""
function (bound::PiecewiseConstantBound)(t::Real)
    # Check if t is outside the horizon
    if t < bound.t_grid[1] || t >= bound.t_grid[end]
        return 0.0 # Or handle as an error
    end

    # Find which segment t falls into. `searchsortedlast` is efficient for this.
    i = searchsortedlast(bound.t_grid, t)
    return bound.Λ_vals[i]
end

"""
    construct_upper_bound(x₀, v₀, flow, U, t_max, N)

Constructs a piecewise-constant upper bound for the event rate λ(t)
using the grid-based method from Andral & Kamatani (2024).

# Arguments
- `x₀`, `v₀`: Initial position and velocity.
- `flow`: The deterministic dynamics (e.g., BouncyParticle, Boomerang).
- `U`: The potential energy function `U(x)`.
- `t_max`: The time horizon for the bound.
- `N`: The number of grid segments.

# Returns
- A `PiecewiseConstantBound` object.
"""
function construct_upper_bound(ξ::SkeletonPoint, flow, ∇U!::Function, t_max::Real, N::Int)

    pcb = PiecewiseConstantBound(Vector{eltype(ξ.x)}(undef, N + 1), Vector{eltype(ξ.x)}(undef, N))
    recompute_time_grid!(pcb, t_max, N)
    construct_upper_bound!(pcb, ξ, flow, ∇U!)
    return pcb
end

function recompute_time_grid!(pcb::PiecewiseConstantBound, t_max::Real, N::Integer)
    resize!(pcb.t_grid, N + 1)
    resize!(pcb.Λ_vals, N)
    resize!(pcb.y_vals, N + 1)
    resize!(pcb.d_vals, N + 1)
    pcb.t_grid .= range(0.0, t_max, N + 1)
end

function make_grad_U_func(θ::AbstractVector, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return GradientProvider(θ, flow, gradient_strategy, cache)
end
function make_grad_U_func(state::AbstractPDMPState, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return make_grad_U_func(state.ξ.θ, flow, gradient_strategy, cache)
end

function make_hvp_func(flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return function (x, θ)
        dot(compute_gradient!(x, θ, gradient_strategy, flow, cache), θ)
        # dot(compute_gradient_uncorrected!(x, θ, gradient_strategy, flow, cache), θ)
    end
end
function make_hvp_func(::AbstractPDMPState, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return make_hvp_func(flow, gradient_strategy, cache)
end


function construct_upper_bound_grad_and_hess!(pcb::PiecewiseConstantBound, state::AbstractPDMPState, flow::FL,
    grad_and_hess_or_grad_and_hvp, add_rate::Bool=true;
    cached_y0::Float64=NaN, cached_d0::Float64=NaN,
    early_stop_threshold::Float64=Inf, stats::Union{AbstractStatisticCounter,Nothing}=nothing,
    state_cache::Union{AbstractPDMPState,Nothing}=nothing,
    max_time::Float64=Inf,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    start_cell::Integer=1,
    initial_integral::Float64=0.0) where {FL<:ContinuousDynamics}

    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    N = length(Λ_vals)
    y_vals = pcb.y_vals
    d_vals = pcb.d_vals

    state_t = state_cache === nothing ? copy(state) : (copyto!(state_cache, state); state_cache)
    iszero(t_grid[1]) || error("t_grid[1] must be zero, got $(t_grid[1])")
    n_time_cells = isfinite(max_time) ? max(0, min(N, searchsortedfirst(t_grid, max_time) - 1)) : N
    start_cell = clamp(Int(start_cell), 1, N + 1)
    start_cell > n_time_cells && return start_cell - 1
    used_batched_rate_derivatives = _supports_constant_grid_rate_derivatives(flow, grad_and_hess_or_grad_and_hvp) &&
        !_uses_builtin_grid_provider(grad_and_hess_or_grad_and_hvp) && n_time_cells > 0

    loaded_batched_points = start_cell == 1 ? 0 : start_cell
    λ_refresh = add_rate ? refresh_rate(flow) : 0.0
    if start_cell > 1
        move_forward_time!(state_t, t_grid[start_cell], flow)
    elseif used_batched_rate_derivatives
        loaded_batched_points = _load_constant_rate_derivatives!(
            pcb, grad_and_hess_or_grad_and_hvp, state, flow, 1, n_time_cells + 1,
            loaded_batched_points, λ_refresh, stats)
    elseif isnan(cached_y0)
        if stats !== nothing
            _inc_counter_grid_endpoint_evaluations(stats)
            _inc_counter_grid_endpoint_gradient_calls(stats)
            _inc_counter_grid_endpoint_hessian_calls(stats)
        end
        y_vals[1], d_vals[1] = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_t, flow, grad_and_hess_or_grad_and_hvp, add_rate;
            t_valid=0.0, t_invalid=0.0)
    else
        !isnothing(stats) && (_inc_counter_grid_cached_endpoint_reuses(stats))
        y_vals[1] = cached_y0
        d_vals[1] = cached_d0
    end

    # Early termination: stop evaluating grid points once cumulative integral is large enough
    cumulative_integral = initial_integral
    N_evaluated = N  # how many cells we actually computed

    for i in (start_cell + 1):(N + 1)
        # Horizon cap: skip evaluation beyond effective time horizon (Phase 1A)
        if t_grid[i - 1] >= max_time
            for j in (i-1):N
                Λ_vals[j] = 0.0
            end
            N_evaluated = i - 2
            if !isnothing(stats)
                _inc_counter_grid_points_skipped(stats, N - N_evaluated)
            end
            break
        end

        Δt = t_grid[i] - t_grid[i-1]
        if used_batched_rate_derivatives
            loaded_batched_points = _load_constant_rate_derivatives!(
                pcb, grad_and_hess_or_grad_and_hvp, state, flow, i, n_time_cells + 1,
                loaded_batched_points, λ_refresh, stats)
        else
            move_forward_time!(state_t, Δt, flow)
            if stats !== nothing
                _inc_counter_grid_endpoint_evaluations(stats)
                _inc_counter_grid_endpoint_gradient_calls(stats)
                _inc_counter_grid_endpoint_hessian_calls(stats)
            end
            y_vals[i], d_vals[i] = _get_rate_and_deriv_or_throw(
                probe_failure_handler, state_t, flow, grad_and_hess_or_grad_and_hvp, add_rate;
                t_valid=t_grid[i-1], t_invalid=t_grid[i])
        end

        # Compute bound for interval [i-1] immediately so we can track cumulative integral
        _compute_cell_bound!(Λ_vals, t_grid, y_vals, d_vals, i - 1)
        cumulative_integral += pos(Λ_vals[i-1]) * Δt

        if cumulative_integral >= early_stop_threshold && i <= N
            # Enough integrated rate; zero out remaining cells
            N_evaluated = i - 1
            for j in i:N
                Λ_vals[j] = 0.0
            end
            if !isnothing(stats)
                _inc_counter_grid_early_stops(stats)
                _inc_counter_grid_points_skipped(stats, N - N_evaluated)
            end
            break
        end
    end

    validate_state(state_t, flow, "after grid construction in Grid algorithm")

    if !isnothing(stats)
        _inc_counter_grid_builds(stats)
        _inc_counter_grid_points_evaluated(stats, start_cell == 1 ?
            (N_evaluated > 1 ? N_evaluated - 1 : N_evaluated) : N_evaluated)
    end
    return N_evaluated
end

_supports_constant_grid_rate_derivatives(::ContinuousDynamics, provider) = false
_supports_constant_grid_rate_derivatives(flow::BouncyParticle, provider) =      _supports_rate_derivatives(provider, flow)
_supports_constant_grid_rate_derivatives(pd::PreconditionedDynamics, provider) =_supports_constant_grid_rate_derivatives(pd.dynamics, provider)

function _constant_rate_derivative_from_signed(g::Real, dg::Real, λ_refresh::Real)
    return pos(g) + λ_refresh, ispositive(g) ? dg : zero(dg)
end

_grid_rate_derivative_chunk_points() = 16

function _load_rate_derivative_chunk!(
    pcb::PiecewiseConstantBound,
    provider,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    target_point::Integer,
    max_points::Integer,
    loaded_points::Integer,
    stats::Union{AbstractStatisticCounter,Nothing},
    transform,
)
    target_point <= loaded_points && return loaded_points
    start_point = loaded_points + 1
    stop_point = min(max_points, max(target_point, loaded_points + _grid_rate_derivative_chunk_points()))
    n_points = stop_point - start_point + 1
    if stats !== nothing
        _inc_counter_grid_endpoint_derivative_calls(stats)
        _inc_counter_grid_endpoint_derivative_points_loaded(stats, n_points)
    end
    values = reshape(@view(pcb.y_vals[start_point:stop_point]), 1, n_points)
    derivatives = reshape(@view(pcb.d_vals[start_point:stop_point]), 1, n_points)
    _fill_rate_derivatives!(
        values, derivatives, provider, state, flow, @view(pcb.t_grid[start_point:stop_point]), n_points)
    for (offset, point) in enumerate(start_point:stop_point)
        pcb.y_vals[point], pcb.d_vals[point] = transform(values[1, offset], derivatives[1, offset])
    end
    return stop_point
end

function _load_constant_rate_derivatives!(
    pcb::PiecewiseConstantBound,
    provider,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    target_point::Integer,
    max_points::Integer,
    loaded_points::Integer,
    λ_refresh::Real,
    stats::Union{AbstractStatisticCounter,Nothing},
)
    transform = (g, dg) -> _constant_rate_derivative_from_signed(g, dg, λ_refresh)
    return _load_rate_derivative_chunk!(
        pcb, provider, state, flow, target_point, max_points, loaded_points, stats, transform)
end

function _load_rate_derivatives!(
    pcb::PiecewiseConstantBound,
    provider,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    target_point::Integer,
    max_points::Integer,
    loaded_points::Integer,
    stats::Union{AbstractStatisticCounter,Nothing},
)
    return _load_rate_derivative_chunk!(
        pcb, provider, state, flow, target_point, max_points, loaded_points, stats,
        (g, dg) -> (g, dg))
end

function _compute_cell_bound!(Λ_vals::Vector, t_grid::Vector, y_vals::Vector, d_vals::Vector, i::Int)
    Λ_vals[i] = _tangent_intersection_bound(t_grid[i], t_grid[i+1], y_vals[i], y_vals[i+1], d_vals[i], d_vals[i+1])
end

function _tangent_intersection_bound(tᵢ::Float64, tᵢ₊₁::Float64, yᵢ::Float64, yᵢ₊₁::Float64, dᵢ::Float64, dᵢ₊₁::Float64)
    if abs(dᵢ - dᵢ₊₁) < 1e-9
        mᵢ = yᵢ
    else
        x_intersect = (yᵢ₊₁ - yᵢ + dᵢ * tᵢ - dᵢ₊₁ * tᵢ₊₁) / (dᵢ - dᵢ₊₁)
        x_clipped = clamp(x_intersect, tᵢ, tᵢ₊₁)
        mᵢ = dᵢ * x_clipped + yᵢ - dᵢ * tᵢ
    end
    return max(yᵢ, yᵢ₊₁, mᵢ)
end

function construct_upper_bound!(pcb::PiecewiseConstantBound, ξ::SkeletonPoint, flow::ContinuousDynamics, ∇U!::Function, use_hvp::Bool=true)
    construct_upper_bound!(pcb, PDMPState(0.0, ξ), flow, ∇U!, use_hvp)
end
function construct_upper_bound!(pcb::PiecewiseConstantBound, state::PDMPState, flow::ContinuousDynamics, ∇U!::Function,
    use_hvp::Bool=true)

    use_hvp || error("Only HVP mode is supported")

    g = similar(state.ξ.x)
    out = similar(g)
    ∇U = Base.Fix1(∇U!, out)
    f = (x, v) -> dot(∇U(x), v)
    prep = DI.prepare_gradient(f, DI.AutoMooncake(), g, DI.Constant(copy(g)))

    # Avoid boxing prep by wrapping it in a struct or passing it explicitly
    hvp = let prep = prep, g = g, f = f
        (x, v) -> DI.gradient!(f, g, prep, DI.AutoMooncake(), x, DI.Constant(v))
    end

    return construct_upper_bound_grad_and_hess!(pcb, state, flow, (∇U, hvp))
end


function propose_event_time(rng::Random.AbstractRNG, pcb::PiecewiseConstantBound, u::Real=rand(rng, Exponential()), refresh_rate::Real=0.0)

    area_before = zero(eltype(pcb.Λ_vals))
    segment_idx = 0
    integral = zero(eltype(pcb.Λ_vals))
    for i in eachindex(pcb.Λ_vals)
        integral += pos(pcb.Λ_vals[i] + refresh_rate) * (pcb.t_grid[i+1] - pcb.t_grid[i])
        if integral >= u
            segment_idx = i
            break
        end
        area_before = integral
    end

    if iszero(segment_idx)
        return (Inf, 0.0)
    end

    # Get the properties of this segment
    t_start = pcb.t_grid[segment_idx]
    Λ_val = pos(pcb.Λ_vals[segment_idx] + refresh_rate)
    # This is how much of the random draw `u` we need to "spend" inside this segment
    u_remaining = u - area_before

    # Calculate the time into the segment: time = distance / speed
    time_in_segment = u_remaining / Λ_val

    τ_proposed = t_start + time_in_segment

    return τ_proposed, Λ_val

end

propose_event_time(pcb::PiecewiseConstantBound, u::Real, refresh_rate::Real=0.0) = propose_event_time(Random.default_rng(), pcb, u, refresh_rate)
