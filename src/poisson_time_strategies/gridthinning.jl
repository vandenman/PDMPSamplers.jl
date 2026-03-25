
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
        compute_gradient!(x, θ, gradient_strategy, flow, cache)
        # compute_gradient_uncorrected!(x, θ, gradient_strategy, flow, cache)
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



function construct_upper_bound_grad_and_hess!(pcb::PiecewiseConstantBound, state::AbstractPDMPState, flow::FL,
    grad_and_hess_or_grad_and_hvp, add_rate::Bool=true;
    cached_y0::Float64=NaN, cached_d0::Float64=NaN,
    early_stop_threshold::Float64=Inf, stats::Union{StatisticCounter,Nothing}=nothing,
    state_cache::Union{AbstractPDMPState,Nothing}=nothing,
    max_time::Float64=Inf) where {FL<:ContinuousDynamics}

    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    N = length(Λ_vals)
    y_vals = pcb.y_vals
    d_vals = pcb.d_vals

    state_t = state_cache === nothing ? copy(state) : (copyto!(state_cache, state); state_cache)
    @assert iszero(t_grid[1])
    if isnan(cached_y0)
        y_vals[1], d_vals[1] = get_rate_and_deriv(state_t, flow, grad_and_hess_or_grad_and_hvp, add_rate)
    else
        y_vals[1] = cached_y0
        d_vals[1] = cached_d0
    end

    # Early termination: stop evaluating grid points once cumulative integral is large enough
    cumulative_integral = 0.0
    N_evaluated = N  # how many cells we actually computed

    for i in 2:N+1
        # Horizon cap: skip evaluation beyond effective time horizon (Phase 1A)
        if t_grid[i] > max_time
            for j in (i-1):N
                Λ_vals[j] = 0.0
            end
            N_evaluated = i - 2
            if !isnothing(stats)
                stats.grid_points_skipped += N - N_evaluated
            end
            break
        end

        Δt = t_grid[i] - t_grid[i-1]
        move_forward_time!(state_t, Δt, flow)
        y_vals[i], d_vals[i] = get_rate_and_deriv(state_t, flow, grad_and_hess_or_grad_and_hvp, add_rate)

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
                stats.grid_early_stops += 1
                stats.grid_points_skipped += N - N_evaluated
            end
            break
        end
    end

    validate_state(state_t, flow, "after grid construction in Grid algorithm")

    if !isnothing(stats)
        stats.grid_builds += 1
        stats.grid_points_evaluated += N_evaluated + 1  # +1 for the initial point
    end
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

    @assert use_hvp "Only HVP mode is supported"

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

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffHVP, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    h = 1e-5 * max(1.0, norm(xt) / norm(vt))
    fd.buf .= xt .+ h .* vt
    ∇U_shifted = fd.grad(fd.buf)
    fd.hvp_buf .= (∇U_shifted .- fd.grad_buf) ./ h

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
    h = 1e-5 * max(1.0, norm(xt) / norm(vt))
    fd.buf .= xt .+ h .* vt
    ∇U_shifted = fd.grad(fd.buf)
    fd.hvp_buf .= (∇U_shifted .- fd.grad_buf) ./ h

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
    h = 1e-5 * max(1.0, norm(xt) / norm(vt))
    fd.buf .= xt .+ h .* vt
    ∇U_shifted = fd.grad(fd.buf)
    return (dot(wt, ∇U_shifted) - dot(wt, fd.grad_buf)) / h
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffVHV, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    vhv_scalar = _fd_vhv_scalar(fd, xt, vt, vt)
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
    vhv_scalar = _fd_vhv_scalar(fd, xt, vt, vt)
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
    Λ_val = pos(pcb.Λ_vals[segment_idx])

    # This is how much of the random draw `u` we need to "spend" inside this segment
    u_remaining = u - area_before

    # Calculate the time into the segment: time = distance / speed
    time_in_segment = u_remaining / Λ_val

    τ_proposed = t_start + time_in_segment

    return τ_proposed, Λ_val

end

propose_event_time(pcb::PiecewiseConstantBound, u::Real, refresh_rate::Real=0.0) = propose_event_time(Random.default_rng(), pcb, u, refresh_rate)

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
end

_default_early_stop(::ContinuousDynamics, est::Float64) = est
_default_early_stop(pd::PreconditionedDynamics, est::Float64) = _default_early_stop(pd.dynamics, est)

function _to_internal(strat::GridThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::StatisticCounter)
    T = typeof(strat.t_max)
    # Derivative info is always available: either via HVP, VHV, joint, or FD fallback.
    N_base = strat.N
    N_min = min_grid_cells(flow, strat.N_min, N_base)
    est = _default_early_stop(flow, strat.early_stop_threshold)
    est = _adjust_early_stop(model.grad, est)
    _build_grid_adaptive_state(strat, state, N_base, N_min, est)
end

_adjust_early_stop(::GradientStrategy, est::Float64) = est
_adjust_early_stop(::SubsampledGradient, ::Float64) = Inf

_effective_grid_horizon(::GradientStrategy, t_max::Float64, τ_refresh::Float64, max_horizon::Float64) = min(t_max, τ_refresh, max_horizon)

function _build_grid_adaptive_state(strat::GridThinningStrategy, state::S, N_base::Int, N_min::Int, est) where S<:AbstractPDMPState
    T = typeof(strat.t_max)
    GridAdaptiveState(
        PiecewiseConstantBound(collect(range(0.0, strat.t_max, N_base + 1)), zeros(T, N_base)),
        Base.RefValue{Int}(N_base),
        Base.RefValue{Float64}(strat.t_max),
        strat.α⁺,
        strat.α⁻,
        strat.safety_limit,
        N_min,
        N_base,
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
        similar(state.ξ.x),
        Ref(false),
    )
end

struct GridAdaptiveState{S<:AbstractPDMPState,V<:AbstractVector} <: PoissonTimeStrategy
    pcb::PiecewiseConstantBound{Float64}
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
    cached_gradient::Vector{Float64}
    has_cached_gradient::Base.RefValue{Bool}
end

accept_reflection_event(::Random.AbstractRNG, ::GridAdaptiveState, args...) = true
accept_reflection_event(::GridAdaptiveState, args...) = true

recompute_time_grid!(alg::GridAdaptiveState) = recompute_time_grid!(alg.pcb, alg.t_max[], alg.N[])

function reset_grid_scale!(alg::GridAdaptiveState, t_max::Float64=2.0)
    alg.t_max[] = t_max
    alg.N[] = alg.N_max
    recompute_time_grid!(alg)
end

function _constant_bound_event_time(
    rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::StatisticCounter, max_horizon::Float64, include_refresh::Bool
)
    λ_bound = alg.constant_bound_rate[]
    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    τ_refresh = ispositive(λ_refresh) ? rand(rng, Exponential(inv(λ_refresh))) : Inf
    default_return = GradientMeta(alg.empty_∇ϕx)

    state_ = alg.state_cache
    copyto!(state_, state)
    t_max = min(alg.t_max[], max_horizon)

    cumulative_exp = 0.0
    for _ in 1:alg.safety_limit
        cumulative_exp += rand(rng, Exponential())
        τ_proposal = cumulative_exp / λ_bound

        if τ_proposal >= t_max
            if τ_refresh < t_max
                return τ_refresh, :refresh, default_return
            end
            return t_max, :horizon_hit, default_return
        end

        if τ_refresh < τ_proposal
            return τ_refresh, :refresh, default_return
        end

        copyto!(state_, state)
        move_forward_time!(state_, τ_proposal, flow)
        ∇ϕx = compute_gradient!(state_, model.grad, flow, cache)
        l_actual = λ(state_.ξ, ∇ϕx, flow)

        if l_actual > λ_bound
            alg.constant_bound_rate[] = NaN
            return next_event_time(rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh)
        end

        if rand(rng) * λ_bound <= l_actual
            alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
            return τ_proposal, :reflect, GradientMeta(∇ϕx)
        end
    end

    alg.constant_bound_rate[] = NaN
    return next_event_time(rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh)
end

function _constant_bound_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::StatisticCounter, max_horizon::Float64, include_refresh::Bool)
    return _constant_bound_event_time(Random.default_rng(), model, flow, alg, state, cache, stats, max_horizon, include_refresh)
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

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::StatisticCounter,
    max_horizon::Float64=Inf, include_refresh::Bool=true) where {FL<:ContinuousDynamics}

    if isfinite(alg.constant_bound_rate[])
        return _constant_bound_event_time(rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh)
    end

    state_ = alg.state_cache
    copyto!(state_, state)

    grad_func = make_grad_U_func(state_, flow, model.grad, cache)
    grad_and_hvp = _make_grad_provider(grad_func, model, flow, alg)

    # Function barrier: specialized on the concrete type of grad_and_hvp
    if alg.lazy
        return _next_event_time_lazy!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh)
    end
    return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh)
end

function _next_event_time_grid!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::StatisticCounter,
    max_horizon::Float64, include_refresh::Bool) where {P, FL<:ContinuousDynamics}

    pcb = alg.pcb
    state_ = alg.state_cache
    state2_ = alg.state_cache2
    copyto!(state_, state)
    copyto!(state2_, state)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))

    default_return = GradientMeta(alg.empty_∇ϕx)

    # Draw refresh time FIRST so we can cap grid construction (Phase 1A)
    τ_refresh = ispositive(λ_refresh) ? rand(rng, Exponential(inv(λ_refresh))) : Inf
    effective_horizon = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon)

    # Build grid once for this event, capped at effective horizon
    if alg.has_cached_gradient[]
        cached_y0, cached_d0 = get_rate_and_deriv(state_, flow, grad_and_hvp, false, alg.cached_gradient)
        alg.has_cached_gradient[] = false
    else
        cached_y0, cached_d0 = NaN, NaN
    end
    construct_upper_bound_grad_and_hess!(pcb, state_, flow, grad_and_hvp, false;
        cached_y0, cached_d0,
        early_stop_threshold=alg.early_stop_threshold, stats, state_cache=state_,
        max_time=effective_horizon)
    stats.grid_N_current = alg.N[]

    # Cumulative exponential sum for correct sequential thinning.
    # After rejection at time τ_k, the next proposal continues from τ_k
    # (not from t=0), eliminating the bias from restart-from-zero thinning.
    cumulative_exp = 0.0
    rejection_count = 0
    max_rejections = 100

    max_t_max = max_grid_horizon(flow)

    safety_limit = alg.safety_limit
    while safety_limit > 0

        cumulative_exp += rand(rng, Exponential())
        τ_reflection, lb_reflection = propose_event_time(rng, pcb, cumulative_exp)

        if τ_reflection >= effective_horizon

            if effective_horizon < alg.t_max[]
                # Grid was capped by refresh or max_horizon: return refresh
                return τ_refresh, :refresh, default_return
            end

            t_max = alg.t_max[]
            new_t_max = alg.t_max[] * alg.α⁺
            alg.t_max[] = min(new_t_max, max_t_max)
            recompute_time_grid!(alg)
            stats.grid_grows += 1

            return t_max, :horizon_hit, default_return
        end

        if τ_refresh < τ_reflection
            return τ_refresh, :refresh, default_return
        end

        # Move from original position to proposed time for acceptance test
        state_.t[] = state2_.t[]
        copyto!(state_.ξ, state2_.ξ)
        move_forward_time!(state_, τ_reflection, flow)
        ∇ϕx = compute_gradient!(state_, model.grad, flow, cache)

        l_reflection = λ(state_.ξ, ∇ϕx, flow)

        if rand(rng) * lb_reflection <= l_reflection
            tightness = l_reflection / lb_reflection
            _adapt_grid_N!(alg, tightness)
            _adapt_grid_t_max!(alg, τ_reflection, model.grad)
            stats.grid_N_current = alg.N[]
            alg.max_observed_rate[] = max(alg.max_observed_rate[], l_reflection)
            copyto!(alg.cached_gradient, ∇ϕx)
            alg.has_cached_gradient[] = true
            return τ_reflection, :reflect, GradientMeta(∇ϕx)
        end

        # Rejection: cumulative_exp has advanced, next proposal will be at a later time
        rejection_count += 1
        if rejection_count >= max_rejections
            # Too many rejections — rebuild with a finer grid and restart the
            # thinning. The cumulative_exp is reset because the new grid has
            # different cell integrals.
            _increase_grid_N!(alg)
            recompute_time_grid!(alg)

            # Also shrink t_max when the grid integral greatly exceeds
            # cumulative_exp, indicating most of the domain has zero rate.
            _shrink_t_max_on_rejection!(alg, pcb, cumulative_exp, model.grad)

            effective_horizon = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon)
            construct_upper_bound_grad_and_hess!(pcb, state2_, flow, grad_and_hvp, false;
                early_stop_threshold=alg.early_stop_threshold, stats, state_cache=state_,
                max_time=effective_horizon)
            cumulative_exp = 0.0
            rejection_count = 0
            max_rejections = min(max_rejections * 2, alg.safety_limit)
            stats.grid_shrinks += 1
        end
        safety_limit -= 1
    end

    error("Safety limit reached")
end

# ── Lazy grid evaluation (Phase 2) ──────────────────────────────────────────
# Interleaves grid point evaluation with proposal generation so that for
# well-adapted samplers only the first few intervals are evaluated.

function _next_event_time_lazy!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::StatisticCounter,
    max_horizon::Float64, include_refresh::Bool) where {P, FL<:ContinuousDynamics}

    state_ = alg.state_cache
    state2_ = alg.state_cache2
    copyto!(state_, state)
    copyto!(state2_, state)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    default_return = GradientMeta(alg.empty_∇ϕx)

    τ_refresh = ispositive(λ_refresh) ? rand(rng, Exponential(inv(λ_refresh))) : Inf

    N = alg.N[]
    t_max = alg.t_max[]
    effective_horizon = _effective_grid_horizon(model.grad, t_max, τ_refresh, max_horizon)
    Δt = effective_horizon / N
    max_t_max = max_grid_horizon(flow)

    # Evaluate initial grid point (k=0)
    if alg.has_cached_gradient[]
        # Reuse cached gradient from previous event (2C): skip one gradient call.
        y_left, d_left = get_rate_and_deriv(state_, flow, grad_and_hvp, false, alg.cached_gradient)
        alg.has_cached_gradient[] = false
    else
        y_left, d_left = get_rate_and_deriv(state_, flow, grad_and_hvp, false)
    end
    t_left = 0.0

    cumulative_area = 0.0
    exp_target = rand(rng, Exponential())
    safety_limit = alg.safety_limit

    stats.grid_builds += 1
    stats.grid_points_evaluated += 1

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
            if effective_horizon < t_max
                return τ_refresh, :refresh, default_return
            end
            alg.t_max[] = min(t_max * alg.α⁺, max_t_max)
            recompute_time_grid!(alg)
            stats.grid_grows += 1
            return t_max, :horizon_hit, default_return
        end

        # Move state_cache forward by Δt_cell to evaluate the right endpoint
        move_forward_time!(state_, Δt_cell, flow)
        y_right, d_right = get_rate_and_deriv(state_, flow, grad_and_hvp, false)
        stats.grid_points_evaluated += 1

        # Compute piecewise constant bound for this interval
        Λ_cell = _tangent_intersection_bound(t_left, t_right, y_left, y_right, d_left, d_right)
        area_cell = pos(Λ_cell) * Δt_cell

        if cumulative_area + area_cell < exp_target
            # No event in this interval — advance
            cumulative_area += area_cell
            t_left = t_right
            y_left = y_right
            d_left = d_right

            # Check if we've exhausted the horizon
            if t_right >= effective_horizon
                if effective_horizon < t_max
                    return τ_refresh, :refresh, default_return
                end
                alg.t_max[] = min(t_max * alg.α⁺, max_t_max)
                recompute_time_grid!(alg)
                stats.grid_grows += 1
                return t_max, :horizon_hit, default_return
            end
            continue
        end

        # Event is in this interval — propose time via inverse CDF
        u_remaining = exp_target - cumulative_area
        time_in_cell = u_remaining / pos(Λ_cell)
        τ_proposal = t_left + time_in_cell

        # Check refresh
        if τ_refresh < τ_proposal
            return τ_refresh, :refresh, default_return
        end

        # Acceptance test: move from original state to proposed time
        state2_.t[] = state.t[]
        copyto!(state2_.ξ, state.ξ)
        move_forward_time!(state2_, τ_proposal, flow)
        ∇ϕx = compute_gradient!(state2_, model.grad, flow, cache)

        l_actual = λ(state2_.ξ, ∇ϕx, flow)

        if l_actual > pos(Λ_cell)
            # Safety violation — fall back to eager grid with finer N
            alg.has_cached_gradient[] = false
            _increase_grid_N!(alg)
            recompute_time_grid!(alg)
            return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh)
        end

        if rand(rng) * pos(Λ_cell) <= l_actual
            # Accepted — cache gradient for next call (2C)
            copyto!(alg.cached_gradient, ∇ϕx)
            alg.has_cached_gradient[] = true

            tightness = l_actual / pos(Λ_cell)
            _adapt_grid_N!(alg, tightness)
            _adapt_grid_t_max!(alg, τ_proposal, model.grad)
            stats.grid_N_current = alg.N[]
            alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
            return τ_proposal, :reflect, GradientMeta(∇ϕx)
        end

        # Rejected — recycle gradient (2B).
        # The gradient ∇ϕx from compute_gradient! is still valid at τ_proposal.
        # Use it as cached gradient to save one gradient call in get_rate_and_deriv.
        copyto!(state_, state2_)
        y_left, d_left = get_rate_and_deriv(state_, flow, grad_and_hvp, false, ∇ϕx)
        t_left = τ_proposal
        cumulative_area = 0.0
        exp_target = rand(rng, Exponential())
    end

    error("Safety limit reached in lazy grid")
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

function _maybe_activate_constant_bound!(alg::GridAdaptiveState, stats::StatisticCounter)
    alg.post_warmup_simplify || return nothing
    isfinite(alg.constant_bound_rate[]) && return nothing
    total_events = stats.reflections_accepted + stats.refreshment_events
    total_events < 10 && return nothing
    reflection_ratio = stats.reflections_accepted / total_events
    reflection_ratio > 0.3 && return nothing
    max_rate = alg.max_observed_rate[]
    max_rate <= 0.0 && return nothing
    alg.constant_bound_rate[] = max_rate * 2.0
    return nothing
end