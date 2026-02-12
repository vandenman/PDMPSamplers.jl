
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

    pcb = PiecewiseConstantBound{eltype(ξ.x)}(Vector{eltype(ξ.x)}(undef, N + 1), Vector{eltype(ξ.x)}(undef, N))
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



function construct_upper_bound_grad_and_hess!(pcb::PiecewiseConstantBound, state::AbstractPDMPState, flow::ContinuousDynamics,
    grad_and_hess_or_grad_and_hvp::Union{Function,NTuple{2,Function}}, add_rate::Bool=true;
    cached_y0::Float64=NaN, cached_d0::Float64=NaN,
    early_stop_threshold::Float64=Inf, stats::Union{StatisticCounter,Nothing}=nothing)

    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    N = length(Λ_vals)
    y_vals = pcb.y_vals
    d_vals = pcb.d_vals

    state_t = copy(state)
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
        Δt = t_grid[i] - t_grid[i-1]
        move_forward_time!(state_t, Δt, flow)
        validate_state(state_t, flow, "after moving forward in time in Grid algorithm")
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

    if !isnothing(stats)
        stats.grid_builds += 1
        stats.grid_points_evaluated += N_evaluated + 1  # +1 for the initial point
    end
end

function _compute_cell_bound!(Λ_vals::Vector, t_grid::Vector, y_vals::Vector, d_vals::Vector, i::Int)
    tᵢ, tᵢ₊₁ = t_grid[i], t_grid[i+1]
    yᵢ, yᵢ₊₁ = y_vals[i], y_vals[i+1]
    dᵢ, dᵢ₊₁ = d_vals[i], d_vals[i+1]

    if abs(dᵢ - dᵢ₊₁) < 1e-9
        mᵢ = yᵢ
    else
        x_intersect = (yᵢ₊₁ - yᵢ + dᵢ * tᵢ - dᵢ₊₁ * tᵢ₊₁) / (dᵢ - dᵢ₊₁)
        x_clipped = clamp(x_intersect, tᵢ, tᵢ₊₁)
        mᵢ = dᵢ * x_clipped + yᵢ - dᵢ * tᵢ
    end

    Λ_vals[i] = max(yᵢ, yᵢ₊₁, mᵢ)
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

# helper functions for λ(t) and λ'(t) ---
function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, (grad, hvp)::NTuple{2,Function}, add_rate::Bool=true)

    xt, vt = state.ξ.x, state.ξ.θ  # state already moved to time t

    ∇U_xt = grad(xt)
    Hxt_vt = hvp(xt, vt)  # Hessian-vector product

    # base rate (before positive-part)
    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)

    # @show ∇U_xt,  Hxt_vt, f_t

    # derivative of base rate
    # TODO: perhap define ∂λ∂t(state, ∇U_xt, ???, flow)
    if flow isa ZigZag
        f_prime_t = zero(eltype(vt))
        for i in eachindex(∇U_xt)
            if ispositive(vt[i] * ∇U_xt[i])
                f_prime_t += vt[i] * Hxt_vt[i]
            end
        end
    else
        f_prime_t = dot(vt, Hxt_vt)
    end

    if flow isa Boomerang

        # 1. Subtract the reference Hessian part from f_prime_t
        # H_Φ*v = Γ*v for the reference potential
        # @show f_prime_t, dot(vt, flow.Γ, vt)
        f_prime_t -= dot(vt, flow.Γ, vt)

        # 2. Add the velocity drift part
        # dv/dt = -∇Φ = Γ(μ - xt)
        vdot = flow.Γ * (flow.μ - xt)

        # We use `∇U_xt` directly because it IS the corrected gradient.
        # @show f_prime_t, vdot, dot(vdot, ∇U_xt)
        f_prime_t += dot(vdot, ∇U_xt)

    elseif !(flow isa ZigZag || flow isa BouncyParticle || flow isa PreconditionedDynamics{<:Any,<:Union{BouncyParticle,ZigZag}})
        throw(ArgumentError("Unsupported flow type!"))
    end

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv

end

function propose_event_time(pcb::PiecewiseConstantBound, u::Real=rand(Exponential()), refresh_rate::Real=0.0)

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


Base.@kwdef struct GridThinningStrategy <: PoissonTimeStrategy
    N::Int = 30
    N_min::Int = 5
    t_max::Float64 = 2.0
    α⁺::Float64 = 1.5
    α⁻::Float64 = 0.5
    safety_limit::Int = 500
    early_stop_threshold::Float64 = Inf
end

function _to_internal(strat::GridThinningStrategy, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::StatisticCounter)
    T = typeof(strat.t_max)
    GridAdaptiveState(
        PiecewiseConstantBound(collect(range(0.0, strat.t_max, strat.N + 1)), zeros(T, strat.N)),
        Base.RefValue{Int}(strat.N),
        Base.RefValue{Float64}(strat.t_max),
        strat.α⁺,
        strat.α⁻,
        strat.safety_limit,
        strat.N_min,
        strat.N,
        strat.early_stop_threshold,
    )
end

struct GridAdaptiveState <: PoissonTimeStrategy
    pcb::PiecewiseConstantBound{Float64}
    N::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    N_min::Int
    N_max::Int
    early_stop_threshold::Float64
end

recompute_time_grid!(alg::GridAdaptiveState) = recompute_time_grid!(alg.pcb, alg.t_max[], alg.N[])


function next_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::StatisticCounter,
    max_horizon::Float64=Inf, include_refresh::Bool=true)

    pcb = alg.pcb
    state_ = copy(state)
    state2_ = copy(state)

    grad_func = make_grad_U_func(state_, flow, model.grad, cache)
    hvp_func = model.hvp
    grad_and_hvp = (grad_func, hvp_func)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))

    default_return = (; ∇ϕx=similar(state.ξ.x, 0))

    # Build grid once for this event
    construct_upper_bound_grad_and_hess!(pcb, state_, flow, grad_and_hvp, false;
        early_stop_threshold=alg.early_stop_threshold, stats)
    stats.grid_N_current = alg.N[]

    # Draw refresh time once (separate Poisson process)
    τ_refresh = ispositive(λ_refresh) ? rand(Exponential(inv(λ_refresh))) : Inf

    # Cumulative exponential sum for correct sequential thinning.
    # After rejection at time τ_k, the next proposal continues from τ_k
    # (not from t=0), eliminating the bias from restart-from-zero thinning.
    cumulative_exp = 0.0
    rejection_count = 0
    max_rejections = 100

    safety_limit = alg.safety_limit
    while safety_limit > 0

        cumulative_exp += rand(Exponential())
        τ_reflection, lb_reflection = propose_event_time(pcb, cumulative_exp)

        if τ_reflection >= alg.t_max[]

            if τ_refresh < alg.t_max[]
                return τ_refresh, :refresh, default_return
            end

            t_max = alg.t_max[]
            new_t_max = alg.t_max[] * alg.α⁺
            alg.t_max[] = min(new_t_max, 1e10)
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

        if rand() * lb_reflection <= l_reflection
            tightness = l_reflection / lb_reflection
            _adapt_grid_N!(alg, tightness)
            stats.grid_N_current = alg.N[]
            return τ_reflection, :reflect, (; ∇ϕx)
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
            total_integral = sum(i -> pos(pcb.Λ_vals[i]) * (pcb.t_grid[i+1] - pcb.t_grid[i]), 1:alg.N[])
            if total_integral > 0 && cumulative_exp < 0.1 * total_integral
                alg.t_max[] = max(alg.t_max[] * alg.α⁻, 0.1)
                recompute_time_grid!(alg)
            end

            construct_upper_bound_grad_and_hess!(pcb, state2_, flow, grad_and_hvp, false;
                early_stop_threshold=alg.early_stop_threshold, stats)
            cumulative_exp = 0.0
            rejection_count = 0
            max_rejections = min(max_rejections * 2, alg.safety_limit)
            stats.grid_shrinks += 1
        end
        safety_limit -= 1
    end

    error("Safety limit reached")
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