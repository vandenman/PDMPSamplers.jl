
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

abstract type GridBoundaryProbe end
struct NoGridBoundaryProbe <: GridBoundaryProbe end
struct GridBoundaryProbeHandler{S,F,M,A} <: GridBoundaryProbe
    original_state::S
    flow::F
    model::M
end
GridBoundaryProbeHandler(original_state::S, flow::F, model::M, ::Type{A}) where {S,F,M,A} =
    GridBoundaryProbeHandler{S,F,M,A}(original_state, flow, model)

_is_bridgestan_probe_error(err) = err isa ErrorException && startswith(err.msg, "BridgeStan gradient failed")

function _get_rate_and_deriv_or_throw(
    ::NoGridBoundaryProbe,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    grad_and_hess_or_grad_and_hvp,
    add_rate::Bool,
    args...;
    t_valid::Float64,
    t_invalid::Float64
)
    return get_rate_and_deriv(state, flow, grad_and_hess_or_grad_and_hvp, add_rate, args...)
end

function _get_rate_and_deriv_or_throw(
    probe_failure_handler::GridBoundaryProbeHandler,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    grad_and_hess_or_grad_and_hvp,
    add_rate::Bool,
    args...;
    t_valid::Float64,
    t_invalid::Float64
)
    try
        return get_rate_and_deriv(state, flow, grad_and_hess_or_grad_and_hvp, add_rate, args...)
    catch err
        err isa _ProbeFailureException && rethrow()
        if err isa ErrorException && err.msg == "bad hvp"
            throw(MethodError(get_rate_and_deriv, (state, flow, grad_and_hess_or_grad_and_hvp, add_rate, args...)))
        end
        if t_valid == t_invalid
            # If the current state itself is already invalid, keep routing the
            # failure through support-boundary recovery with a tiny forward
            # bracket so truncated-refresh can fall back to the last valid
            # trace event instead of leaking the raw model error.
            t_invalid = max(t_valid + eps(Float64), eps(Float64))
            try
                _throw_grid_boundary_error(probe_failure_handler, state, err; t_valid, t_invalid)
            catch boundary_err
                if boundary_err isa MethodError && boundary_err.f === _throw_grid_boundary_error
                    if _is_bridgestan_probe_error(err)
                        x0 = copy(probe_failure_handler.original_state.ξ.x)
                        v = copy(probe_failure_handler.original_state.ξ.θ)
                        ctx = BoundaryContext(
                            x0, v, Float64(probe_failure_handler.original_state.t[]),
                            max(t_valid, 0.0), max(t_invalid, eps(Float64)),
                            err, typeof(flow), typeof(probe_failure_handler).parameters[4],
                        )
                        throw(_ProbeFailureException(ctx))
                    end
                    throw(err)
                end
                rethrow()
            end
        end
        _throw_grid_boundary_error(probe_failure_handler, state, err; t_valid, t_invalid)
    end
end

function _compute_grid_gradient_or_throw!(
    state::AbstractPDMPState,
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    cache,
    t_valid::Float64,
    t_invalid::Float64,
    ::NoGridBoundaryProbe,
)
    return compute_gradient!(state, model.grad, flow, cache)
end

function _compute_grid_gradient_or_throw!(
    state::AbstractPDMPState,
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    cache,
    t_valid::Float64,
    t_invalid::Float64,
    probe_failure_handler::GridBoundaryProbeHandler,
)
    try
        return compute_gradient!(state, model.grad, flow, cache)
    catch err
        _throw_grid_boundary_error(probe_failure_handler, state, err; t_valid, t_invalid)
    end
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

# --- Grid parameter caps, dispatched on flow type ---

_joint_compatible(::BouncyParticle) = true
_joint_compatible(pd::PreconditionedDynamics) = _joint_compatible(pd.dynamics)
_joint_compatible(::ContinuousDynamics) = false

min_grid_cells(::ContinuousDynamics, N_min::Int, ::Int) = N_min
min_grid_cells(::AnyBoomerang, N_min::Int, ::Int) = max(N_min, 5)
min_grid_cells(pd::PreconditionedDynamics, N_min::Int, N::Int) = min_grid_cells(pd.dynamics, N_min, N)

max_grid_horizon(::ContinuousDynamics) = 1e10
max_grid_horizon(::AnyBoomerang) = 8π
max_grid_horizon(pd::PreconditionedDynamics) = max_grid_horizon(pd.dynamics)

# --- helper functions for λ(t) and λ'(t) ---
# Generic fallback: ignore cached gradient for providers that don't support it
get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider, add_rate::Bool, ::AbstractVector) =
    get_rate_and_deriv(state, flow, provider, add_rate)

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::Union{Tuple,GradHVPProvider}, add_rate::Bool=true)

    xt, vt = state.ξ.x, state.ξ.θ  # state already moved to time t

    grad = _provider_grad(provider)
    hvp = _provider_hvp(provider)
    ∇U_xt = grad(xt)
    Hxt_vt = hvp(xt, vt)  # Hessian-vector product

    # base rate (before positive-part)
    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)

    f_prime_t = ∂λ∂t(state, ∇U_xt, Hxt_vt, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv

end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::Union{Tuple,GradHVPProvider},
    add_rate::Bool, cached_gradient::AbstractVector)
    xt, vt = state.ξ.x, state.ξ.θ
    hvp = _provider_hvp(provider)
    Hxt_vt = hvp(xt, vt)
    f_t = λ(state.ξ, cached_gradient, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, cached_gradient, Hxt_vt, flow)
    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function _compute_vhv_scalar(provider::VHVProvider, state::AbstractPDMPState, ∇U_xt::AbstractVector, ::ContinuousDynamics)
    xt, vt = state.ξ.x, state.ξ.θ
    return provider.vhv(xt, vt, vt)
end

function _compute_vhv_scalar(provider::VHVProvider, state::AbstractPDMPState, ∇U_xt::AbstractVector, ::ZigZag)
    xt, vt = state.ξ.x, state.ξ.θ
    w = provider.w_buf === nothing ? similar(vt) : provider.w_buf
    for i in eachindex(vt)
        w[i] = ispositive(vt[i] * ∇U_xt[i]) ? vt[i] : zero(eltype(vt))
    end
    return provider.vhv(xt, vt, w)
end

function _compute_vhv_scalar(provider::VHVProvider, state::AbstractPDMPState, ∇U_xt::AbstractVector, pd::PreconditionedDynamics)
    return _compute_vhv_scalar(provider, state, ∇U_xt, pd.dynamics)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::VHVProvider, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ

    ∇U_xt = provider.grad(xt)

    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)

    curvature_scalar = _compute_vhv_scalar(provider, state, ∇U_xt, flow)
    f_prime_t = ∂λ∂t(state, ∇U_xt, curvature_scalar, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::VHVProvider,
    add_rate::Bool, cached_gradient::AbstractVector)
    f_t = λ(state.ξ, cached_gradient, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    curvature_scalar = _compute_vhv_scalar(provider, state, cached_gradient, flow)
    f_prime_t = ∂λ∂t(state, cached_gradient, curvature_scalar, flow)
    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::WithStatsJoint, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    dphi, d2phi = provider(xt, vt)

    f_t = pos(dphi) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = d2phi

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv
end

function get_rate_and_deriv(
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    provider::WithStatsJoint,
    add_rate::Bool,
    ::AbstractVector,
)
    return get_rate_and_deriv(state, flow, provider, add_rate)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, grad_and_nothing::Tuple{G,Nothing}, add_rate::Bool=true) where {G}
    grad = grad_and_nothing[1]
    xt = state.ξ.x
    ∇U_xt = grad(xt)
    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    return rate, zero(rate)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, grad_and_nothing::Tuple{G,Nothing},
    add_rate::Bool, cached_gradient::AbstractVector) where {G}
    f_t = λ(state.ξ, cached_gradient, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    return rate, zero(rate)
end

struct FiniteDiffHVP{G}
    grad::G
    buf::Vector{Float64}
    grad_buf::Vector{Float64}
    hvp_buf::Vector{Float64}
end
FiniteDiffHVP(grad, buf::Vector{Float64}) = FiniteDiffHVP(grad, buf, similar(buf), similar(buf))

function _fd_step_size(xt::AbstractVector, vt::AbstractVector)
    vnorm = norm(vt)
    iszero(vnorm) && return vnorm
    return oftype(vnorm, 1e-5) * max(one(vnorm), norm(xt) / vnorm)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffHVP, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    h = _fd_step_size(xt, vt)
    if iszero(h)
        fill!(fd.hvp_buf, 0.0)
    else
        fd.buf .= xt .+ h .* vt
        ∇U_shifted = fd.grad(fd.buf)
        fd.hvp_buf .= (∇U_shifted .- fd.grad_buf) ./ h
    end

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, fd.hvp_buf, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffHVP,
    add_rate::Bool, cached_gradient::AbstractVector)
    copyto!(fd.grad_buf, cached_gradient)

    xt, vt = state.ξ.x, state.ξ.θ
    h = _fd_step_size(xt, vt)
    if iszero(h)
        fill!(fd.hvp_buf, 0.0)
    else
        fd.buf .= xt .+ h .* vt
        ∇U_shifted = fd.grad(fd.buf)
        fd.hvp_buf .= (∇U_shifted .- fd.grad_buf) ./ h
    end

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, fd.hvp_buf, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function _fd_vhv_scalar(fd::FiniteDiffVHV, xt::AbstractVector, vt::AbstractVector, wt::AbstractVector)
    h = _fd_step_size(xt, vt)
    iszero(h) && return h
    fd.buf .= xt .+ h .* vt
    ∇U_shifted = fd.grad(fd.buf)
    return (dot(wt, ∇U_shifted) - dot(wt, fd.grad_buf)) / h
end

_restore_reference_vhv(vhv::Real, ::AbstractVector, ::ContinuousDynamics) = vhv
function _restore_reference_vhv(vhv::Real, vt::AbstractVector, flow::AnyBoomerang)
    # `fd.grad` calls compute_gradient!, so for Boomerang it differentiates
    # ∇U - Γ(x-μ). The Boomerang ∂λ∂t method expects curvature of the raw
    # target gradient ∇U and subtracts the reference contribution itself.
    return vhv + dot(vt, flow.Γ, vt)
end
function _restore_reference_vhv(vhv::Real, vt::AbstractVector, flow::LowRankMutableBoomerang)
    return vhv + lowrank_quadform(flow.Γ, vt)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffVHV, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    vhv_scalar = _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, vhv_scalar, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffVHV,
    add_rate::Bool, cached_gradient::AbstractVector)
    copyto!(fd.grad_buf, cached_gradient)

    xt, vt = state.ξ.x, state.ξ.θ
    vhv_scalar = _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, vhv_scalar, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ZigZag, fd::FiniteDiffVHV, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    w = fd.w_buf
    for i in eachindex(vt)
        w[i] = ispositive(vt[i] * fd.grad_buf[i]) ? vt[i] : zero(eltype(vt))
    end
    whv_scalar = _fd_vhv_scalar(fd, xt, vt, w)

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = whv_scalar

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ZigZag, fd::FiniteDiffVHV,
    add_rate::Bool, cached_gradient::AbstractVector)
    copyto!(fd.grad_buf, cached_gradient)

    vt = state.ξ.θ
    w = fd.w_buf
    for i in eachindex(vt)
        w[i] = ispositive(vt[i] * fd.grad_buf[i]) ? vt[i] : zero(eltype(vt))
    end
    xt = state.ξ.x
    whv_scalar = _fd_vhv_scalar(fd, xt, vt, w)

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = whv_scalar

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
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
