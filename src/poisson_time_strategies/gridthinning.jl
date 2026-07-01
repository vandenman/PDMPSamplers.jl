
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
    return function (x)
        return compute_gradient!(x, θ, gradient_strategy, flow, cache)
    end
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
    used_batched_signed_jets = _supports_constant_grid_signed_jets(flow, grad_and_hess_or_grad_and_hvp) && n_time_cells > 0

    loaded_batched_points = start_cell == 1 ? 0 : start_cell
    λ_refresh = add_rate ? refresh_rate(flow) : 0.0
    if start_cell > 1
        move_forward_time!(state_t, t_grid[start_cell], flow)
    elseif used_batched_signed_jets
        loaded_batched_points = _load_constant_signed_rate_jets!(
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
        if used_batched_signed_jets
            loaded_batched_points = _load_constant_signed_rate_jets!(
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

_supports_constant_grid_signed_jets(::ContinuousDynamics, provider) = false
_supports_constant_grid_signed_jets(::BouncyParticle, provider) = supports_grid_signed_rate_jets(provider)
_supports_constant_grid_signed_jets(pd::PreconditionedDynamics, provider) =
    _supports_constant_grid_signed_jets(pd.dynamics, provider)

function _constant_rate_jet_from_signed(g::Real, dg::Real, λ_refresh::Real)
    return pos(g) + λ_refresh, ispositive(g) ? dg : zero(dg)
end

_grid_signed_jet_chunk_points() = 16

function _load_constant_signed_rate_jets!(
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
    target_point <= loaded_points && return loaded_points
    start_point = loaded_points + 1
    stop_point = min(max_points, max(target_point, loaded_points + _grid_signed_jet_chunk_points()))
    n_points = stop_point - start_point + 1
    if stats !== nothing
        _inc_counter_grid_endpoint_jet_calls(stats)
        _inc_counter_grid_endpoint_jet_points_loaded(stats, n_points)
    end
    g_values, dg_values = signed_rate_jets_for_grid(
        provider, state, flow, @view(pcb.t_grid[start_point:stop_point]), n_points)
    length(g_values) >= n_points || throw(ArgumentError(
        "signed_rate_jets_for_grid returned $(length(g_values)) g-values for $(n_points) grid points"))
    length(dg_values) >= n_points || throw(ArgumentError(
        "signed_rate_jets_for_grid returned $(length(dg_values)) derivative values for $(n_points) grid points"))
    for (offset, point) in enumerate(start_point:stop_point)
        pcb.y_vals[point], pcb.d_vals[point] =
            _constant_rate_jet_from_signed(g_values[offset], dg_values[offset], λ_refresh)
    end
    return stop_point
end

function _load_signed_rate_jets!(
    pcb::PiecewiseConstantBound,
    provider,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    target_point::Integer,
    max_points::Integer,
    loaded_points::Integer,
    stats::Union{AbstractStatisticCounter,Nothing},
)
    target_point <= loaded_points && return loaded_points
    start_point = loaded_points + 1
    stop_point = min(max_points, max(target_point, loaded_points + _grid_signed_jet_chunk_points()))
    n_points = stop_point - start_point + 1
    if stats !== nothing
        _inc_counter_grid_endpoint_jet_calls(stats)
        _inc_counter_grid_endpoint_jet_points_loaded(stats, n_points)
    end
    g_values, dg_values = signed_rate_jets_for_grid(
        provider, state, flow, @view(pcb.t_grid[start_point:stop_point]), n_points)
    length(g_values) >= n_points || throw(ArgumentError(
        "signed_rate_jets_for_grid returned $(length(g_values)) g-values for $(n_points) grid points"))
    length(dg_values) >= n_points || throw(ArgumentError(
        "signed_rate_jets_for_grid returned $(length(dg_values)) derivative values for $(n_points) grid points"))
    for (offset, point) in enumerate(start_point:stop_point)
        pcb.y_vals[point] = g_values[offset]
        pcb.d_vals[point] = dg_values[offset]
    end
    return stop_point
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

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, (grad, hvp)::Tuple{G,H}, add_rate::Bool=true) where {G,H}

    xt, vt = state.ξ.x, state.ξ.θ  # state already moved to time t

    ∇U_xt = grad(xt)
    Hxt_vt = hvp(xt, vt)  # Hessian-vector product

    # base rate (before positive-part)
    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)

    f_prime_t = ∂λ∂t(state, ∇U_xt, Hxt_vt, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv

end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, (grad, hvp)::Tuple{G,H},
    add_rate::Bool, cached_gradient::AbstractVector) where {G,H}
    xt, vt = state.ξ.x, state.ξ.θ
    Hxt_vt = hvp(xt, vt)
    f_t = λ(state.ξ, cached_gradient, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, cached_gradient, Hxt_vt, flow)
    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

struct VHVProvider{G,V,W<:Union{Nothing,AbstractVector}}
    grad::G
    vhv::V
    w_buf::W
end
VHVProvider(grad, vhv) = VHVProvider(grad, vhv, nothing)

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

struct JointProvider{J}
    joint::J
end

supports_grid_signed_rate_jets(provider::JointProvider) = supports_grid_signed_rate_jets(provider.joint)
signed_rate_jets_for_grid(provider::JointProvider, state::AbstractPDMPState, flow::ContinuousDynamics,
    t_grid::AbstractVector, n_points::Integer) =
    signed_rate_jets_for_grid(provider.joint, state, flow, t_grid, n_points)

supports_grid_signed_rate_jets(provider::WithStatsJoint) = supports_grid_signed_rate_jets(provider.f)
signed_rate_jets_for_grid(provider::WithStatsJoint, state::AbstractPDMPState, flow::ContinuousDynamics,
    t_grid::AbstractVector, n_points::Integer) =
    signed_rate_jets_for_grid(provider.f, state, flow, t_grid, n_points)

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::JointProvider, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    dphi, d2phi = provider.joint(xt, vt)

    f_t = pos(dphi) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = d2phi

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv
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

struct FiniteDiffVHV{G}
    grad::G
    buf::Vector{Float64}
    grad_buf::Vector{Float64}
    w_buf::Vector{Float64}
end
FiniteDiffVHV(grad, buf::Vector{Float64}) = FiniteDiffVHV(grad, buf, similar(buf), similar(buf))
FiniteDiffVHV(grad, buf::Vector{Float64}, w_buf::Vector{Float64}) = FiniteDiffVHV(grad, buf, similar(buf), w_buf)

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

abstract type AbstractCurvatureCertificate end

struct CertifiedUpperCurvature{T<:Real} <: AbstractCurvatureCertificate
    value::T
end

struct GlobalCertifiedUpperCurvature{T<:Real} <: AbstractCurvatureCertificate
    value::T
end

struct NoCertificate <: AbstractCurvatureCertificate end

supports_certified_roof(::AbstractPDMPState, ::BouncyParticle, provider) = true
supports_certified_roof(state::AbstractPDMPState, pd::PreconditionedDynamics, provider) =
    supports_certified_roof(state, pd.dynamics, provider)
supports_certified_roof(::AbstractPDMPState, ::ContinuousDynamics, provider) = false

supports_signed_certified_roof(::AbstractPDMPState, ::BouncyParticle, provider) = true
supports_signed_certified_roof(::AbstractPDMPState, ::ContinuousDynamics, provider) = false
supports_signed_certified_roof(::AbstractPDMPState, ::BouncyParticle, ::FiniteDiffVHV) = false
supports_signed_certified_roof(::AbstractPDMPState, ::BouncyParticle, ::FiniteDiffHVP) = false
supports_signed_certified_roof(::AbstractPDMPState, ::BouncyParticle, ::Tuple{G,Nothing}) where {G} = false

_certified_auto_envelope(envelope::Symbol) =
    envelope === :certified_auto || envelope === :certified_auto_affine_sticky
_certified_auto_affine_sticky(envelope::Symbol) = envelope === :certified_auto_affine_sticky

_certified_auto_currently_prefers_affine(alg) =
    hasproperty(alg, :certified_auto_prefer_affine_current) && alg.certified_auto_prefer_affine_current[]

_use_affine_envelope(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider) =
    (alg.envelope === :hybrid_linear && supports_certified_roof(state, flow, provider)) ||
    (alg.envelope === :inflated_linear && _can_use_signed_certified_roof(alg, state, flow, provider)) ||
    (_certified_auto_envelope(alg.envelope) && _certified_auto_currently_prefers_affine(alg) &&
        _can_use_signed_certified_roof(alg, state, flow, provider))

_use_inflated_constant_envelope(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider) =
    ((alg.envelope === :inflated_constant) ||
        (_certified_auto_envelope(alg.envelope) && !_certified_auto_currently_prefers_affine(alg))) &&
    _can_use_signed_certified_roof(alg, state, flow, provider)

_use_signed_certified_grid(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider) =
    alg.envelope in (:inflated_linear, :inflated_constant, :certified_auto, :certified_auto_affine_sticky) &&
    alg.certification === :required &&
    _can_use_signed_certified_roof(alg, state, flow, provider)

function _validate_certification_mode(certification::Symbol)
    certification === :required && return certification
    certification === :opportunistic && return certification
    throw(ArgumentError("certification must be :required or :opportunistic, got $(certification)"))
end

function _can_use_signed_certified_roof(alg, state::AbstractPDMPState, flow::ContinuousDynamics, provider)
    mode = _validate_certification_mode(alg.certification)
    supported = supports_signed_certified_roof(state, flow, provider)
    if !supported
        mode === :required && throw(ArgumentError(
            "envelope=$(alg.envelope) with certification=:required needs an exact signed derivative provider; " *
            "finite-difference or gradient-only curvature is not a certificate"))
        return false
    end
    if alg.curvature_bound === nothing
        mode === :required && throw(ArgumentError(
            "envelope=$(alg.envelope) with certification=:required needs curvature_bound returning CertifiedUpperCurvature"))
        return false
    end
    return true
end

_normalize_curvature_certificate(cert::CertifiedUpperCurvature) = cert
_normalize_curvature_certificate(cert::GlobalCertifiedUpperCurvature) = cert
_normalize_curvature_certificate(::NoCertificate) = NoCertificate()
_normalize_curvature_certificate(::Nothing) = NoCertificate()
_normalize_curvature_certificate(cert) = cert

rate_curvature_upper_bound(::Nothing, state::AbstractPDMPState, flow::ContinuousDynamics, a::Real, b::Real) = NoCertificate()
rate_curvature_upper_bound(cert::AbstractCurvatureCertificate, state::AbstractPDMPState, flow::ContinuousDynamics, a::Real, b::Real) =
    _normalize_curvature_certificate(cert)
rate_curvature_upper_bound(provider, state::AbstractPDMPState, flow::ContinuousDynamics, a::Real, b::Real) =
    _normalize_curvature_certificate(provider(state, flow, a, b))

supports_grid_curvature_bounds(provider) = false
curvature_bounds_for_grid(provider, state::AbstractPDMPState, flow::ContinuousDynamics,
    t_grid::AbstractVector, n_cells::Integer) = NoCertificate()

supports_grid_signed_rate_jets(provider) = false
signed_rate_jets_for_grid(provider, state::AbstractPDMPState, flow::ContinuousDynamics,
    t_grid::AbstractVector, n_points::Integer) =
    throw(MethodError(signed_rate_jets_for_grid, (provider, state, flow, t_grid, n_points)))

function _curvature_certificate_value(cert, certification::Symbol)
    mode = _validate_certification_mode(certification)
    if cert isa Union{CertifiedUpperCurvature,GlobalCertifiedUpperCurvature}
        L = cert.value
        isfinite(L) || throw(ArgumentError("curvature certificate value must be finite, got $(L)"))
        return L
    elseif cert isa NoCertificate
        mode === :required && throw(ArgumentError("required curvature certificate was not supplied"))
        return nothing
    elseif cert isa Real
        throw(ArgumentError(
            "curvature_bound must return CertifiedUpperCurvature(L), not a raw number; " *
            "use NoCertificate() for opportunistic fallback"))
    else
        throw(ArgumentError("unsupported curvature certificate return type $(typeof(cert))"))
    end
end

function _evaluate_curvature_provider(curvature_bound, state::AbstractPDMPState, flow::ContinuousDynamics,
    a::Real, b::Real, stats::Union{AbstractStatisticCounter,Nothing}, certification::Symbol)
    !(curvature_bound isa AbstractCurvatureCertificate) && stats !== nothing && (_inc_counter_grid_certificate_calls(stats))
    cert = rate_curvature_upper_bound(curvature_bound, state, flow, a, b)
    value = _curvature_certificate_value(cert, certification)
    value === nothing && stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, 1))
    return cert, value
end

function _normalize_grid_curvature_values(certs, certification::Symbol, stats::Union{AbstractStatisticCounter,Nothing}, n_cells::Integer)
    certs = _normalize_curvature_certificate(certs)
    if certs isa GlobalCertifiedUpperCurvature
        return (global_value=_curvature_certificate_value(certs, certification),
            first_value=nothing, has_first=false, cell_values=nothing)
    elseif certs isa AbstractCurvatureCertificate
        value = _curvature_certificate_value(certs, certification)
        value === nothing && stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, n_cells))
        return (global_value=nothing, first_value=value, has_first=true, cell_values=nothing)
    elseif certs isa AbstractVector
        length(certs) >= n_cells || throw(ArgumentError(
            "curvature_bounds_for_grid returned $(length(certs)) certificates for $(n_cells) cells"))
        values = Vector{Union{Nothing,Float64}}(undef, n_cells)
        for i in 1:n_cells
            value = _curvature_certificate_value(_normalize_curvature_certificate(certs[i]), certification)
            value === nothing && stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, 1))
            values[i] = value === nothing ? nothing : Float64(value)
        end
        return (global_value=nothing, first_value=nothing, has_first=false, cell_values=values)
    else
        throw(ArgumentError("unsupported grid curvature certificate return type $(typeof(certs))"))
    end
end

function _prepare_grid_curvature_certificate(curvature_bound, state::AbstractPDMPState, flow::ContinuousDynamics,
    t_grid::AbstractVector, n_cells::Integer, stats::Union{AbstractStatisticCounter,Nothing}, certification::Symbol)
    if curvature_bound isa GlobalCertifiedUpperCurvature
        return (global_value=_curvature_certificate_value(curvature_bound, certification),
            first_value=nothing, has_first=false, cell_values=nothing)
    elseif curvature_bound isa CertifiedUpperCurvature || curvature_bound isa NoCertificate
        value = _curvature_certificate_value(curvature_bound, certification)
        value === nothing && stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, n_cells))
        return (global_value=nothing, first_value=value, has_first=true, cell_values=nothing)
    elseif curvature_bound === nothing
        value = _curvature_certificate_value(NoCertificate(), certification)
        value === nothing && stats !== nothing && (_inc_counter_grid_certificate_fallbacks(stats, n_cells))
        return (global_value=nothing, first_value=value, has_first=true, cell_values=nothing)
    elseif supports_grid_curvature_bounds(curvature_bound)
        stats !== nothing && (_inc_counter_grid_certificate_calls(stats))
        certs = curvature_bounds_for_grid(curvature_bound, state, flow, t_grid, n_cells)
        return _normalize_grid_curvature_values(certs, certification, stats, n_cells)
    end

    cert, value = _evaluate_curvature_provider(curvature_bound, state, flow, t_grid[1], t_grid[n_cells + 1], stats, certification)
    if cert isa GlobalCertifiedUpperCurvature
        return (global_value=value, first_value=nothing, has_first=false, cell_values=nothing)
    end
    return (global_value=nothing, first_value=value, has_first=true, cell_values=nothing)
end

function _prepared_or_cell_curvature_value(prepared, cell::Integer, curvature_bound,
    state::AbstractPDMPState, flow::ContinuousDynamics, a::Real, b::Real,
    stats::Union{AbstractStatisticCounter,Nothing}, certification::Symbol)
    prepared.global_value !== nothing && return prepared.global_value
    if prepared.cell_values !== nothing
        return prepared.cell_values[cell]
    end
    if cell == 1 && prepared.has_first
        return prepared.first_value
    end
    _, value = _evaluate_curvature_provider(curvature_bound, state, flow, a, b, stats, certification)
    return value
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
    a::Real, b::Real, M::Real; certified_auto::Bool=false)
    M_pos = pos(M)
    append_affine_segment!(bound, a, b, M_pos, zero(M_pos))
    h = b - a
    area = M_pos * h
    if stats !== nothing
        _inc_counter_affine_constant_cells(stats)
        _inc_counter_affine_area_constant_equiv(stats, area)
        _inc_counter_affine_area_hybrid(stats, area)
        _inc_counter_affine_segments_added(stats, 1)
        if certified_auto
            _inc_counter_certified_auto_flat_cells(stats)
            total = _get_counter_certified_auto_flat_cells(stats) + _get_counter_certified_auto_affine_cells(stats)
            _set_counter_certified_auto_affine_fraction(stats, total == 0 ? 0.0 : _get_counter_certified_auto_affine_cells(stats) / total)
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

function _append_inflated_affine_cell!(bound::PiecewiseAffineBound, stats::Union{AbstractStatisticCounter,Nothing},
    a::Real, b::Real, y_a::Real, y_b::Real, d_a::Real, d_b::Real, M::Real, L)

    L === nothing && return _append_constant_affine_cell!(bound, stats, a, b, M)
    h = b - a
    if !(isfinite(a) && isfinite(b) && isfinite(y_a) && isfinite(y_b) &&
         isfinite(d_a) && isfinite(d_b) && isfinite(M) && isfinite(L) && h > 0)
        return _append_constant_affine_cell!(bound, stats, a, b, M)
    end

    rate_tol, slope_tol, time_tol, area_tol =
        _affine_cell_tolerances(a, b, y_a, y_b, d_a, d_b, M)

    # The scalar certificate applies to the smooth signed rate.  GridThinning's
    # cached endpoint values are for the positive-part rate; when an endpoint is
    # clamped at zero we no longer have the signed tangent needed to certify a
    # linearized roof across possible zero crossings.  Keep this first-stage
    # envelope conservative and fall back to the existing constant cell.
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

function _append_signed_inflated_affine_cell!(bound::PiecewiseAffineBound, stats::Union{AbstractStatisticCounter,Nothing},
    a::Real, b::Real, g_a::Real, g_b::Real, dg_a::Real, dg_b::Real, M::Real, L;
    affine_area_threshold::Real=1.0,
    affine_min_area_gain::Real=0.0,
    certified_auto::Bool=false)

    L === nothing && return _append_constant_affine_cell!(bound, stats, a, b, M; certified_auto)
    h = b - a
    if !(isfinite(a) && isfinite(b) && isfinite(g_a) && isfinite(g_b) &&
         isfinite(dg_a) && isfinite(dg_b) && isfinite(M) && isfinite(L) && h > 0)
        return _append_constant_affine_cell!(bound, stats, a, b, M; certified_auto)
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
        return _append_constant_affine_cell!(bound, stats, a, b, M; certified_auto)
    end

    if !(clipped_area >= -area_tol)
        bound.n_segments = start_segments
        return _append_constant_affine_cell!(bound, stats, a, b, M; certified_auto)
    end

    flat_area = pos(M) * h
    area_gain = flat_area - max(clipped_area, 0.0)
    min_area_gain = max(float(affine_min_area_gain), 0.0)
    if area_gain <= min_area_gain
        bound.n_segments = start_segments
        stats !== nothing && (_inc_counter_affine_cells_skipped_by_min_gain(stats))
        return _append_constant_affine_cell!(bound, stats, a, b, M; certified_auto)
    end

    threshold = float(affine_area_threshold)
    if threshold < 1.0 && flat_area > area_tol && clipped_area / flat_area >= threshold
        bound.n_segments = start_segments
        return _append_constant_affine_cell!(bound, stats, a, b, M; certified_auto)
    end

    if stats !== nothing
        _inc_counter_affine_inflated_cells(stats)
        _inc_counter_affine_area_constant_equiv(stats, flat_area)
        _inc_counter_affine_area_hybrid(stats, max(clipped_area, 0.0))
        _inc_counter_affine_area_saved(stats, max(area_gain, 0.0))
        _inc_counter_affine_segments_added(stats, bound.n_segments - start_segments)
        if certified_auto
            _inc_counter_certified_auto_affine_cells(stats)
            _inc_counter_certified_auto_area_saved(stats, max(area_gain, 0.0))
            total = _get_counter_certified_auto_flat_cells(stats) + _get_counter_certified_auto_affine_cells(stats)
            _set_counter_certified_auto_affine_fraction(stats, total == 0 ? 0.0 : _get_counter_certified_auto_affine_cells(stats) / total)
        end
    end
    return true
end

function _signed_inflated_flat_upper(a::Real, b::Real, g_a::Real, g_b::Real, dg_a::Real, dg_b::Real, L)
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

function build_inflated_affine_bound!(bound::PiecewiseAffineBound, pcb::PiecewiseConstantBound,
    n_cells::Integer, original_state::AbstractPDMPState, flow::ContinuousDynamics, curvature_bound,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing; certification::Symbol=:opportunistic)
    reset_affine_bound!(bound)
    n = min(n_cells, length(pcb.Λ_vals))
    if n <= 0
        return bound
    end
    prepared = _prepare_grid_curvature_certificate(
        curvature_bound, original_state, flow, pcb.t_grid, n, stats, certification)
    for i in 1:n
        a = pcb.t_grid[i]
        b = pcb.t_grid[i + 1]
        L_value = _prepared_or_cell_curvature_value(
            prepared, i, curvature_bound, original_state, flow, a, b, stats, certification)
        _append_inflated_affine_cell!(bound, stats,
            a, b, pcb.y_vals[i], pcb.y_vals[i + 1], pcb.d_vals[i], pcb.d_vals[i + 1], pcb.Λ_vals[i], L_value)
    end
    return bound
end

function build_signed_inflated_affine_bound!(bound::PiecewiseAffineBound, pcb::PiecewiseConstantBound,
    n_cells::Integer, original_state::AbstractPDMPState, flow::ContinuousDynamics, curvature_bound,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing; certification::Symbol=:required,
    affine_area_threshold::Real=1.0,
    affine_min_area_gain::Real=0.0)
    reset_affine_bound!(bound)
    n = min(n_cells, length(pcb.Λ_vals))
    if n <= 0
        return bound
    end
    prepared = _prepare_grid_curvature_certificate(
        curvature_bound, original_state, flow, pcb.t_grid, n, stats, certification)
    for i in 1:n
        a = pcb.t_grid[i]
        b = pcb.t_grid[i + 1]
        L_value = _prepared_or_cell_curvature_value(
            prepared, i, curvature_bound, original_state, flow, a, b, stats, certification)
        M = L_value === nothing ? pcb.Λ_vals[i] : max(pcb.Λ_vals[i], _signed_inflated_flat_upper(
            a, b, pcb.y_vals[i], pcb.y_vals[i + 1], pcb.d_vals[i], pcb.d_vals[i + 1], L_value))
        pcb.Λ_vals[i] = M
        _append_signed_inflated_affine_cell!(bound, stats,
            a, b, pcb.y_vals[i], pcb.y_vals[i + 1], pcb.d_vals[i], pcb.d_vals[i + 1], M, L_value;
            affine_area_threshold,
            affine_min_area_gain)
    end
    return bound
end

signed_rate_and_derivative(state::AbstractPDMPState, flow::ContinuousDynamics, provider, ::AbstractVector) =
    signed_rate_and_derivative(state, flow, provider)

function _signed_rate_and_derivative_or_throw(
    ::NoGridBoundaryProbe,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    provider,
    args...;
    t_valid::Float64,
    t_invalid::Float64,
)
    return signed_rate_and_derivative(state, flow, provider, args...)
end

function _signed_rate_and_derivative_or_throw(
    probe_failure_handler::GridBoundaryProbeHandler,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    provider,
    args...;
    t_valid::Float64,
    t_invalid::Float64,
)
    try
        return signed_rate_and_derivative(state, flow, provider, args...)
    catch err
        _throw_grid_boundary_error(probe_failure_handler, state, err; t_valid, t_invalid)
    end
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, (grad, hvp)::Tuple{G,H}) where {G,H}
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = grad(xt)
    Hxt_vt = hvp(xt, vt)
    return dot(∇U_xt, vt), extract_vhv(vt, Hxt_vt)
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, (grad, hvp)::Tuple{G,H},
    cached_gradient::AbstractVector) where {G,H}
    vt = state.ξ.θ
    Hxt_vt = hvp(state.ξ.x, vt)
    return dot(cached_gradient, vt), extract_vhv(vt, Hxt_vt)
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, provider::VHVProvider)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = provider.grad(xt)
    return dot(∇U_xt, vt), _compute_vhv_scalar(provider, state, ∇U_xt, flow)
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, provider::VHVProvider,
    cached_gradient::AbstractVector)
    return dot(cached_gradient, state.ξ.θ), _compute_vhv_scalar(provider, state, cached_gradient, flow)
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, provider::JointProvider)
    return provider.joint(state.ξ.x, state.ξ.θ)
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, grad_and_nothing::Tuple{G,Nothing}) where {G}
    ∇U_xt = grad_and_nothing[1](state.ξ.x)
    return dot(∇U_xt, state.ξ.θ), 0.0
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, grad_and_nothing::Tuple{G,Nothing},
    cached_gradient::AbstractVector) where {G}
    return dot(cached_gradient, state.ξ.θ), 0.0
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, fd::FiniteDiffVHV)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)
    return dot(fd.grad_buf, vt), _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
end

function signed_rate_and_derivative(state::AbstractPDMPState, flow::BouncyParticle, fd::FiniteDiffVHV,
    cached_gradient::AbstractVector)
    copyto!(fd.grad_buf, cached_gradient)
    xt, vt = state.ξ.x, state.ξ.θ
    return dot(fd.grad_buf, vt), _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
end

function construct_signed_rate_grid!(
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
        pcb.y_vals[1], pcb.d_vals[1] = signed_rate_and_derivative(state_t, flow, provider)
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
        pcb.y_vals[i], pcb.d_vals[i] = signed_rate_and_derivative(state_t, flow, provider)
    end
    return n
end

function construct_signed_inflated_grid!(
    bound::PiecewiseAffineBound,
    pcb::PiecewiseConstantBound,
    state::AbstractPDMPState,
    flow::BouncyParticle,
    provider,
    curvature_bound;
    cached_gradient::Union{AbstractVector,Nothing}=nothing,
    early_stop_threshold::Float64=Inf,
    state_cache::Union{AbstractPDMPState,Nothing}=nothing,
    stats::Union{AbstractStatisticCounter,Nothing}=nothing,
    max_time::Float64=Inf,
    certification::Symbol=:required,
    build_affine::Bool=true,
    affine_area_threshold::Real=1.0,
    affine_min_area_gain::Real=0.0,
    certified_auto::Bool=false,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    start_cell::Integer=1,
    initial_integral::Float64=0.0,
    append::Bool=false,
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
    used_batched_jets = supports_grid_signed_rate_jets(provider) && n_time_cells > 0
    loaded_batched_points = start_cell == 1 ? 0 : start_cell
    if start_cell > 1
        move_forward_time!(state_t, t_grid[start_cell], flow)
    elseif used_batched_jets
        loaded_batched_points = _load_signed_rate_jets!(
            pcb, provider, state, flow, 1, n_time_cells + 1, loaded_batched_points, stats)
    elseif cached_gradient === nothing
        if stats !== nothing
            _inc_counter_grid_endpoint_evaluations(stats)
            _inc_counter_grid_endpoint_gradient_calls(stats)
            _inc_counter_grid_endpoint_hessian_calls(stats)
        end
        y_vals[1], d_vals[1] = _signed_rate_and_derivative_or_throw(
            probe_failure_handler, state_t, flow, provider; t_valid=0.0, t_invalid=0.0)
    else
        if stats !== nothing
            _inc_counter_grid_cached_endpoint_reuses(stats)
            _inc_counter_grid_endpoint_hessian_calls(stats)
        end
        y_vals[1], d_vals[1] = _signed_rate_and_derivative_or_throw(
            probe_failure_handler, state_t, flow, provider, cached_gradient; t_valid=0.0, t_invalid=0.0)
    end

    cumulative_integral = initial_integral
    N_evaluated = N
    prepared = _prepare_grid_curvature_certificate(
        curvature_bound, state, flow, t_grid, N, stats, certification)
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
        if used_batched_jets
            loaded_batched_points = _load_signed_rate_jets!(
                pcb, provider, state, flow, i, n_time_cells + 1, loaded_batched_points, stats)
        else
            move_forward_time!(state_t, Δt, flow)
            if stats !== nothing
                _inc_counter_grid_endpoint_evaluations(stats)
                _inc_counter_grid_endpoint_gradient_calls(stats)
                _inc_counter_grid_endpoint_hessian_calls(stats)
            end
            y_vals[i], d_vals[i] = _signed_rate_and_derivative_or_throw(
                probe_failure_handler, state_t, flow, provider; t_valid=t_grid[i - 1], t_invalid=t_grid[i])
        end

        cell = i - 1
        a = t_grid[cell]
        b = t_grid[cell + 1]
        L_value = _prepared_or_cell_curvature_value(
            prepared, cell, curvature_bound, state, flow, a, b, stats, certification)
        M = L_value === nothing ? max(pos(y_vals[cell]), pos(y_vals[cell + 1])) :
            _signed_inflated_flat_upper(a, b, y_vals[cell], y_vals[cell + 1], d_vals[cell], d_vals[cell + 1], L_value)
        Λ_vals[cell] = M

        cell_area = M * Δt
        if build_affine
            start_segments = bound.n_segments
            _append_signed_inflated_affine_cell!(bound, stats,
                a, b, y_vals[cell], y_vals[cell + 1], d_vals[cell], d_vals[cell + 1], M, L_value;
                affine_area_threshold,
                affine_min_area_gain,
                certified_auto)
            cell_area = _affine_added_area(bound, start_segments)
        elseif stats !== nothing
            _inc_counter_affine_constant_cells(stats)
            _inc_counter_affine_area_constant_equiv(stats, cell_area)
            _inc_counter_affine_area_hybrid(stats, cell_area)
            if certified_auto
                _inc_counter_certified_auto_flat_cells(stats)
                total = _get_counter_certified_auto_flat_cells(stats) + _get_counter_certified_auto_affine_cells(stats)
                _set_counter_certified_auto_affine_fraction(stats, total == 0 ? 0.0 : _get_counter_certified_auto_affine_cells(stats) / total)
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
    use_affine::Bool,
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
    if use_affine && alg.affine_bound.n_segments > 0
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
        alg.envelope, " GridThinning envelope violated at proposal time",
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

_grid_built_area(pcb::PiecewiseConstantBound, bound::PiecewiseAffineBound, use_affine::Bool) =
    use_affine ? total_area(bound) : _piecewise_constant_area(pcb)

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

Base.@kwdef struct GridThinningStrategy <: PoissonTimeStrategy
    N::Int = 20
    N_min::Int = 5
    t_max::Float64 = 2.0
    α⁺::Float64 = 1.5
    α⁻::Float64 = 0.5
    safety_limit::Int = 500
    early_stop_threshold::Float64 = 5.0
    use_fd_hvp::Bool = false
    post_warmup_simplify::Bool = false
    lazy::Bool = true
    envelope::Symbol = :constant
    curvature_bound = nothing
    bound_violation::Symbol = :count
    certification::Symbol = :required
    inflated_affine_threshold::Float64 = 0.95
    inflated_affine_min_area_gain::Float64 = 0.0
    max_rejections_before_tail_restart::Int = 100
    certified_auto_probe_interval::Int = 20
end

"""
    certified_auto_scalar_bps_grid(; N=1, inflated_affine_threshold=0.9,
        inflated_affine_min_area_gain=0.0, certified_auto_probe_interval=20,
        kwargs...)

Return the recommended adaptive certified scalar-BPS `GridThinningStrategy`
preset.  It uses signed endpoint jets plus a certified signed-rate curvature
bound and lets GridThinning choose between certified flat and certified affine
cells/grids.

This is intended for scalar BPS targets with `certification=:required`, where
the user should not have to know in advance whether the flat or affine
representation is faster.
"""
certified_auto_scalar_bps_grid(; N::Int=1, inflated_affine_threshold::Real=0.9,
    inflated_affine_min_area_gain::Real=0.0, certified_auto_probe_interval::Int=20,
    kwargs...) = GridThinningStrategy(;
        N,
        envelope=:certified_auto,
        certification=:required,
        inflated_affine_threshold=Float64(inflated_affine_threshold),
        inflated_affine_min_area_gain=Float64(inflated_affine_min_area_gain),
        certified_auto_probe_interval,
        kwargs...)

"""
    certified_scalar_bps_grid(; N=1, inflated_affine_threshold=0.9,
        inflated_affine_min_area_gain=0.0, kwargs...)

Return the recommended certified scalar-BPS `GridThinningStrategy` preset for
expensive targets with analytic signed endpoint jets and a certified signed-rate
curvature bound.

This is deliberately an affine preset, not a package default.  Prefer
`certified_auto_scalar_bps_grid` unless you specifically want to force the
affine representation.  Supply
`curvature_bound=...` and, when available, a model-level signed-jet provider.
"""
certified_scalar_bps_grid(; N::Int=1, inflated_affine_threshold::Real=0.9,
    inflated_affine_min_area_gain::Real=0.0, kwargs...) = GridThinningStrategy(;
        N,
        envelope=:inflated_linear,
        certification=:required,
        inflated_affine_threshold=Float64(inflated_affine_threshold),
        inflated_affine_min_area_gain=Float64(inflated_affine_min_area_gain),
        kwargs...)

"""
    certified_flat_scalar_bps_grid(; N=1, kwargs...)

Return the cheaper certified-flat scalar-BPS `GridThinningStrategy` preset.
This uses the same signed endpoint/certified-curvature construction as the
affine preset, then flattens each cell.  It is useful when target/rate
evaluation is cheap or affine-segment overhead dominates.
"""
certified_flat_scalar_bps_grid(; N::Int=1, kwargs...) = GridThinningStrategy(;
    N,
    envelope=:inflated_constant,
    certification=:required,
    kwargs...)

function Base.show(io::IO, strat::GridThinningStrategy)
    print(io, "GridThinningStrategy(")
    print(io, "N=", strat.N, ", N_min=", strat.N_min, ", t_max=", strat.t_max)
    print(io, ", envelope=", strat.envelope)
    strat.envelope in (:inflated_linear, :inflated_constant, :certified_auto, :certified_auto_affine_sticky) && print(io, ", certification=", strat.certification)
    strat.envelope in (:inflated_linear, :certified_auto, :certified_auto_affine_sticky) && print(io, ", inflated_affine_threshold=", strat.inflated_affine_threshold,
        ", inflated_affine_min_area_gain=", strat.inflated_affine_min_area_gain)
    print(io, ")")
end

_default_early_stop(::ContinuousDynamics, est::Float64) = est
_default_early_stop(pd::PreconditionedDynamics, est::Float64) = _default_early_stop(pd.dynamics, est)

function _to_internal(strat::GridThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    T = typeof(strat.t_max)
    _validate_certification_mode(strat.certification)
    if _certified_auto_envelope(strat.envelope) && strat.certification !== :required
        throw(ArgumentError("envelope=$(strat.envelope) currently requires certification=:required"))
    end
    0.0 <= strat.inflated_affine_threshold || throw(ArgumentError("inflated_affine_threshold must be nonnegative"))
    0.0 <= strat.inflated_affine_min_area_gain || throw(ArgumentError("inflated_affine_min_area_gain must be nonnegative"))
    strat.max_rejections_before_tail_restart > 0 || throw(ArgumentError("max_rejections_before_tail_restart must be positive"))
    strat.certified_auto_probe_interval > 0 || throw(ArgumentError("certified_auto_probe_interval must be positive"))
    # Derivative info is always available: either via HVP, VHV, joint, or FD fallback.
    N_base = strat.N
    N_min = min_grid_cells(flow, strat.N_min, N_base)
    est = _default_early_stop(flow, strat.early_stop_threshold)
    est = _adjust_early_stop(model.grad, est)
    N_max = max(N_base + 4, 2 * N_base)
    _build_grid_adaptive_state(strat, state, N_base, N_min, N_max, est)
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
    return _build_grid_adaptive_state(strat, state, N_base, N_min, N_base, est)
end

function _build_grid_adaptive_state(strat::GridThinningStrategy, state::S, N_base::Int, N_min::Int, N_max::Int, est) where S<:AbstractPDMPState
    T = typeof(strat.t_max)
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
        copy(state),
        copy(state),
        similar(state.ξ.x, 0),
        strat.use_fd_hvp,
        similar(state.ξ.x),
        similar(state.ξ.x),
        similar(state.ξ.x),
        Ref(NaN),
        Ref(0.0),
        strat.post_warmup_simplify,
        strat.lazy,
        Ref(strat.lazy),
        similar(state.ξ.x),
        Ref(false),
        strat.envelope,
        strat.curvature_bound,
        strat.bound_violation,
        strat.certification,
        strat.inflated_affine_threshold,
        strat.inflated_affine_min_area_gain,
        strat.max_rejections_before_tail_restart,
        strat.certified_auto_probe_interval,
        Ref(_certified_auto_affine_sticky(strat.envelope) ? true : true),
        Ref(_certified_auto_affine_sticky(strat.envelope) ? true : true),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
    )
end

struct GridAdaptiveState{S<:AbstractPDMPState,V<:AbstractVector} <: PoissonTimeStrategy
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
    constant_bound_rate::Base.RefValue{Float64}
    max_observed_rate::Base.RefValue{Float64}
    post_warmup_simplify::Bool
    lazy::Bool
    lazy_enabled::Base.RefValue{Bool}
    cached_gradient::Vector{Float64}
    has_cached_gradient::Base.RefValue{Bool}
    envelope::Symbol
    curvature_bound
    bound_violation::Symbol
    certification::Symbol
    inflated_affine_threshold::Float64
    inflated_affine_min_area_gain::Float64
    max_rejections_before_tail_restart::Int
    certified_auto_probe_interval::Int
    certified_auto_prefer_affine_next::Base.RefValue{Bool}
    certified_auto_prefer_affine_current::Base.RefValue{Bool}
    certified_auto_flat_grids_since_probe::Base.RefValue{Int}
    certified_auto_low_saving_streak::Base.RefValue{Int}
    certified_auto_flat_streak_current::Base.RefValue{Int}
    certified_auto_affine_streak_current::Base.RefValue{Int}
    certified_auto_current_grid_extensions::Base.RefValue{Int}
    certified_auto_current_grid_rejections::Base.RefValue{Int}
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

function _begin_certified_auto_grid!(alg::GridAdaptiveState)
    _certified_auto_envelope(alg.envelope) || return nothing
    alg.certified_auto_current_grid_extensions[] = 0
    alg.certified_auto_current_grid_rejections[] = 0
    prefer_affine = alg.certified_auto_prefer_affine_next[]
    if !prefer_affine && alg.certified_auto_flat_grids_since_probe[] >= alg.certified_auto_probe_interval
        prefer_affine = true
        alg.certified_auto_flat_grids_since_probe[] = 0
    end
    alg.certified_auto_prefer_affine_current[] = prefer_affine
    return nothing
end

function _set_certified_auto_next_preference!(
    stats::AbstractStatisticCounter,
    alg::GridAdaptiveState,
    prefer_affine::Bool,
)
    old = alg.certified_auto_prefer_affine_next[]
    if old != prefer_affine && _certified_auto_affine_sticky(alg.envelope)
        prefer_affine ?
            _inc_counter_certified_auto_switched_to_affine(stats) :
            _inc_counter_certified_auto_switched_to_flat(stats)
    end
    alg.certified_auto_prefer_affine_next[] = prefer_affine
    return nothing
end

function _force_certified_auto_affine_next!(stats::AbstractStatisticCounter, alg::GridAdaptiveState)
    _certified_auto_envelope(alg.envelope) || return nothing
    _set_certified_auto_next_preference!(stats, alg, true)
    alg.certified_auto_low_saving_streak[] = 0
    alg.certified_auto_flat_grids_since_probe[] = 0
    return nothing
end

function _record_certified_auto_streaks!(stats::AbstractStatisticCounter, alg::GridAdaptiveState)
    if alg.certified_auto_prefer_affine_current[]
        alg.certified_auto_affine_streak_current[] += 1
        alg.certified_auto_flat_streak_current[] = 0
        _set_counter_certified_auto_affine_streak_grids(
            stats,
            max(_get_counter_certified_auto_affine_streak_grids(stats),
                alg.certified_auto_affine_streak_current[]),
        )
    else
        alg.certified_auto_flat_streak_current[] += 1
        alg.certified_auto_affine_streak_current[] = 0
        _set_counter_certified_auto_flat_streak_grids(
            stats,
            max(_get_counter_certified_auto_flat_streak_grids(stats),
                alg.certified_auto_flat_streak_current[]),
        )
    end
    return nothing
end

function _record_certified_auto_grid_choice!(
    stats::AbstractStatisticCounter,
    alg::GridAdaptiveState,
    flat_before::Int,
    affine_before::Int,
    area_saved_before::Float64,
    flat_area_before::Float64,
    new_grid::Bool=true,
)
    _certified_auto_envelope(alg.envelope) || return nothing
    forced_probe = !alg.certified_auto_prefer_affine_next[] && alg.certified_auto_prefer_affine_current[]
    flat_cells = _get_counter_certified_auto_flat_cells(stats) - flat_before
    affine_cells = _get_counter_certified_auto_affine_cells(stats) - affine_before
    area_saved = _get_counter_certified_auto_area_saved(stats) - area_saved_before
    flat_area = _get_counter_affine_area_constant_equiv(stats) - flat_area_before
    saving_ratio = flat_area > 0 ? area_saved / flat_area : 0.0

    if new_grid
        _certified_auto_affine_sticky(alg.envelope) && _inc_counter_certified_auto_mode_affine_sticky(stats)
        forced_probe && (_inc_counter_certified_auto_forced_probe_grids(stats))
        alg.certified_auto_prefer_affine_current[] ?
            (_inc_counter_certified_auto_affine_preferred_grids(stats)) :
            (_inc_counter_certified_auto_flat_preferred_grids(stats))
        _certified_auto_affine_sticky(alg.envelope) && _record_certified_auto_streaks!(stats, alg)
    end

    if new_grid
        if affine_cells > 0
            _inc_counter_certified_auto_used_affine_grids(stats)
        elseif flat_cells > 0
            _inc_counter_certified_auto_used_flat_grids(stats)
        end
    end

    if _certified_auto_affine_sticky(alg.envelope)
        if alg.certified_auto_prefer_affine_current[] && saving_ratio < 0.02
            alg.certified_auto_low_saving_streak[] += 1
        elseif saving_ratio >= 0.02
            alg.certified_auto_low_saving_streak[] = 0
        end
        _set_counter_certified_auto_low_saving_streak_max(
            stats,
            max(_get_counter_certified_auto_low_saving_streak_max(stats),
                alg.certified_auto_low_saving_streak[]),
        )

        if forced_probe && saving_ratio > 0.05
            _force_certified_auto_affine_next!(stats, alg)
        elseif alg.certified_auto_current_grid_extensions[] > 2 || alg.certified_auto_current_grid_rejections[] > 2
            _force_certified_auto_affine_next!(stats, alg)
        else
            _set_certified_auto_next_preference!(stats, alg,
                alg.certified_auto_low_saving_streak[] < 5)
        end
    else
        _set_certified_auto_next_preference!(stats, alg, flat_area > 0 && saving_ratio >= 0.05)
    end

    if new_grid
        if alg.certified_auto_prefer_affine_next[]
            alg.certified_auto_flat_grids_since_probe[] = 0
        else
            alg.certified_auto_flat_grids_since_probe[] += 1
        end
    end
    total = _get_counter_certified_auto_flat_cells(stats) + _get_counter_certified_auto_affine_cells(stats)
    _set_counter_certified_auto_affine_fraction(stats, total == 0 ? 0.0 : _get_counter_certified_auto_affine_cells(stats) / total)
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

function _constant_bound_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit, probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe())
    return _constant_bound_event_time(Random.default_rng(), model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end


function _make_grad_provider(grad_func, model::PDMPModel, flow::ContinuousDynamics, alg::GridAdaptiveState)
    joint_func = model.joint
    if joint_func !== nothing && _joint_compatible(flow)
        return JointProvider(joint_func)
    end
    vhv_func = model.vhv
    if vhv_func !== nothing
        return VHVProvider(grad_func, vhv_func, alg.fd_buf)
    end
    hvp_func = model.hvp
    if hvp_func === nothing
        # Always fall back to finite-diff curvature when no HVP is available.
        # This gives much tighter bounds than gradient-only mode.
        return FiniteDiffVHV(grad_func, alg.fd_buf, alg.fd_grad_buf, alg.fd_w_buf)
    end
    return (grad_func, hvp_func)
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

    grad_func = make_grad_U_func(state_, flow, model.grad, cache)
    grad_and_hvp = _make_grad_provider(grad_func, model, flow, alg)

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
    # building the grid, then construct only enough envelope area to cover it.
    # Rejections add exponential increments and extend/rebuild the grid only
    # when the cumulative budget exceeds the already-built area.
    _begin_certified_auto_grid!(alg)
    cumulative_exp = rand(rng, Exponential())

    # Build grid once for this event, capped at effective horizon
    use_affine = _use_affine_envelope(alg, state, flow, grad_and_hvp)
    use_inflated_constant = _use_inflated_constant_envelope(alg, state, flow, grad_and_hvp)
    use_single_pass_signed = _use_signed_certified_grid(alg, state, flow, grad_and_hvp)
    use_constant_batched_signed = !use_single_pass_signed && _supports_constant_grid_signed_jets(flow, grad_and_hvp)
    had_cached_gradient = alg.has_cached_gradient[]
    if use_single_pass_signed
        cached_gradient = had_cached_gradient ? alg.cached_gradient : nothing
        alg.has_cached_gradient[] = false
        auto_flat_before = _get_counter_certified_auto_flat_cells(stats)
        auto_affine_before = _get_counter_certified_auto_affine_cells(stats)
        auto_saved_before = _get_counter_certified_auto_area_saved(stats)
        auto_flat_area_before = _get_counter_affine_area_constant_equiv(stats)
        n_cells_bounded = construct_signed_inflated_grid!(
            alg.affine_bound, pcb, state, flow, grad_and_hvp, alg.curvature_bound;
            cached_gradient,
            early_stop_threshold=cumulative_exp,
            state_cache=state_,
            stats,
            max_time=effective_horizon,
            certification=alg.certification,
            build_affine=use_affine,
            affine_area_threshold=alg.inflated_affine_threshold,
            affine_min_area_gain=alg.inflated_affine_min_area_gain,
            certified_auto=_certified_auto_envelope(alg.envelope),
            probe_failure_handler,
        )
        _record_certified_auto_grid_choice!(
            stats, alg, auto_flat_before, auto_affine_before, auto_saved_before, auto_flat_area_before)
    elseif had_cached_gradient && !use_constant_batched_signed
        cached_y0, cached_d0 = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false, alg.cached_gradient;
            t_valid=0.0, t_invalid=0.0)
        cached_g0, cached_dg0 = if alg.envelope === :inflated_linear
            signed_rate_and_derivative(state_, flow, grad_and_hvp, alg.cached_gradient)
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
    if use_affine && !use_single_pass_signed
        if alg.envelope === :inflated_linear
            construct_signed_rate_grid!(pcb, state, flow, grad_and_hvp, n_cells_bounded;
                cached_g0, cached_dg0, state_cache=state_, stats)
            build_signed_inflated_affine_bound!(alg.affine_bound, pcb, n_cells_bounded, state, flow, alg.curvature_bound, stats;
                certification=alg.certification,
                affine_area_threshold=alg.inflated_affine_threshold,
                affine_min_area_gain=alg.inflated_affine_min_area_gain)
        else
            build_hybrid_affine_bound!(alg.affine_bound, pcb, n_cells_bounded, stats)
        end
    end
    _record_grid_schedule!(stats, alg)
    _set_counter_grid_N_current(stats, alg.N[])
    built_area = _grid_built_area(pcb, alg.affine_bound, use_affine)
    _record_budget_grid_build!(stats, n_cells_bounded, built_area, cumulative_exp, false)

    # Budget-first exactness invariant:
    #   * once proposal times have been generated from the built dominating
    #     envelope prefix, that prefix must never be changed;
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
        τ_reflection, lb_reflection = if use_affine
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
                lb_reflection, cumulative_exp, λ_refresh, use_affine)
            if use_affine
                _inc_counter_affine_bound_violations(stats)
                throw(ErrorException(msg))
            elseif alg.bound_violation === :throw
                throw(ErrorException(msg))
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
        alg.certified_auto_current_grid_rejections[] = rejection_count
        rejection_count > 2 && _force_certified_auto_affine_next!(stats, alg)
        last_rejected_time = τ_reflection
        cumulative_exp += rand(rng, Exponential())
        if cumulative_exp > built_area * (1 + 64eps(Float64)) + 64eps(Float64)
            start_cell = n_cells_bounded + 1
            effective_horizon, horizon_event = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)
            use_affine = _use_affine_envelope(alg, state, flow, grad_and_hvp)
            use_inflated_constant = _use_inflated_constant_envelope(alg, state, flow, grad_and_hvp)
            use_single_pass_signed = _use_signed_certified_grid(alg, state, flow, grad_and_hvp)
            use_constant_batched_signed = !use_single_pass_signed && _supports_constant_grid_signed_jets(flow, grad_and_hvp)
            use_constant_batched_signed && (alg.has_cached_gradient[] = false)
            if use_single_pass_signed
                auto_flat_before = _get_counter_certified_auto_flat_cells(stats)
                auto_affine_before = _get_counter_certified_auto_affine_cells(stats)
                auto_saved_before = _get_counter_certified_auto_area_saved(stats)
                auto_flat_area_before = _get_counter_affine_area_constant_equiv(stats)
                n_cells_bounded = construct_signed_inflated_grid!(
                    alg.affine_bound, pcb, state, flow, grad_and_hvp, alg.curvature_bound;
                    early_stop_threshold=cumulative_exp,
                    state_cache=state_,
                    stats,
                    max_time=effective_horizon,
                    certification=alg.certification,
                    build_affine=use_affine,
                    affine_area_threshold=alg.inflated_affine_threshold,
                    affine_min_area_gain=alg.inflated_affine_min_area_gain,
                    certified_auto=_certified_auto_envelope(alg.envelope),
                    probe_failure_handler,
                    start_cell,
                    initial_integral=built_area,
                    append=true,
                )
                _record_certified_auto_grid_choice!(
                    stats, alg, auto_flat_before, auto_affine_before, auto_saved_before, auto_flat_area_before, false)
            else
                n_cells_bounded = construct_upper_bound_grad_and_hess!(pcb, state, flow, grad_and_hvp, false;
                    early_stop_threshold=cumulative_exp, stats, state_cache=state_,
                    max_time=effective_horizon, probe_failure_handler,
                    start_cell, initial_integral=built_area)
            end
            if use_affine && !use_single_pass_signed
                if alg.envelope === :inflated_linear
                    construct_signed_rate_grid!(pcb, state, flow, grad_and_hvp, n_cells_bounded;
                        state_cache=state_, stats)
                    build_signed_inflated_affine_bound!(alg.affine_bound, pcb, n_cells_bounded, state, flow, alg.curvature_bound, stats;
                        certification=alg.certification,
                        affine_area_threshold=alg.inflated_affine_threshold,
                        affine_min_area_gain=alg.inflated_affine_min_area_gain)
                else
                    build_hybrid_affine_bound!(alg.affine_bound, pcb, n_cells_bounded, stats)
                end
            end
            _record_grid_schedule!(stats, alg)
            built_area = _grid_built_area(pcb, alg.affine_bound, use_affine)
            _record_budget_grid_build!(stats, n_cells_bounded, built_area, cumulative_exp, true)
            alg.certified_auto_current_grid_extensions[] += 1
            alg.certified_auto_current_grid_extensions[] > 2 && _force_certified_auto_affine_next!(stats, alg)
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
                throw(ErrorException("lazy constant GridThinning envelope violated at proposal time"))
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
