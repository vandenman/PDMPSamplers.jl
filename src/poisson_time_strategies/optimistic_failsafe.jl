# COV_EXCL_START
# Optimistic failsafe Poisson time strategy (Phase 3A+3B)
#
# Uses piecewise linear interpolation through gradient-only rate evaluations
# at grid points. The bound may underestimate the true rate (optimistic).
# When a bound violation is detected at proposal time, the grid point is
# inserted and the thinning proposal is restarted.

# ─── Piecewise linear bound ──────────────────────────────────────────────────

struct PiecewiseLinearBound
    t_grid::Vector{Float64}
    y_vals::Vector{Float64}   # rate values at grid points (reflection rate only)
    Λ_cum::Vector{Float64}    # cumulative integral at each grid point
end

function PiecewiseLinearBound(N::Int)
    PiecewiseLinearBound(
        Vector{Float64}(undef, N + 1),
        Vector{Float64}(undef, N + 1),
        Vector{Float64}(undef, N + 1),
    )
end

function reinitialize_grid!(plb::PiecewiseLinearBound, t_max::Real, N::Int)
    resize!(plb.t_grid, N + 1)
    resize!(plb.y_vals, N + 1)
    resize!(plb.Λ_cum, N + 1)
    plb.t_grid .= range(0.0, t_max, N + 1)
end

function compute_cumulative_integral!(plb::PiecewiseLinearBound)
    plb.Λ_cum[1] = 0.0
    for i in 1:length(plb.y_vals) - 1
        Δt = plb.t_grid[i + 1] - plb.t_grid[i]
        plb.Λ_cum[i + 1] = plb.Λ_cum[i] + (plb.y_vals[i] + plb.y_vals[i + 1]) / 2 * Δt
    end
end

function insert_grid_point!(plb::PiecewiseLinearBound, t::Float64, y::Float64)
    i = searchsortedlast(plb.t_grid, t)
    insert!(plb.t_grid, i + 1, t)
    insert!(plb.y_vals, i + 1, y)
    push!(plb.Λ_cum, 0.0)
    compute_cumulative_integral!(plb)
end

"""
    propose_from_linear_bound(plb, u)

Given cumulative exponential `u`, find τ such that ∫₀^τ λ̃(s) ds = u
where λ̃ is the piecewise linear interpolation. Returns `(τ, λ̃(τ))`.
"""
function propose_from_linear_bound(plb::PiecewiseLinearBound, u::Real)
    total_integral = plb.Λ_cum[end]
    if u >= total_integral
        return (Inf, 0.0)
    end

    # Find segment: Λ_cum[i] ≤ u < Λ_cum[i+1]
    i = searchsortedlast(plb.Λ_cum, u)
    N_segments = length(plb.y_vals) - 1
    i = clamp(i, 1, N_segments)

    u_rem = u - plb.Λ_cum[i]
    Δt = plb.t_grid[i + 1] - plb.t_grid[i]
    a = plb.y_vals[i]
    b = plb.y_vals[i + 1]
    slope = (b - a) / Δt

    # Solve: a*δ + slope/2 * δ² = u_rem  for δ ≥ 0
    if abs(slope) < 1e-12
        if a < 1e-15
            return (Inf, 0.0)
        end
        δ = u_rem / a
    else
        discriminant = a^2 + 2 * slope * u_rem
        if discriminant < 0
            return (Inf, 0.0)
        end
        δ = (-a + sqrt(discriminant)) / slope
    end

    τ = plb.t_grid[i] + δ
    α = δ / Δt
    λ_tilde = (1 - α) * a + α * b

    return (τ, λ_tilde)
end

# ─── User-facing and internal types ──────────────────────────────────────────

Base.@kwdef struct OptimisticStrategy <: PoissonTimeStrategy
    N::Int = 20
    t_max::Float64 = 2.0
    α⁺::Float64 = 1.5
    safety_limit::Int = 500
    max_rewinds::Int = 30
end

struct OptimisticState{S<:AbstractPDMPState} <: PoissonTimeStrategy
    plb::PiecewiseLinearBound
    N_base::Int
    N_current::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64
    safety_limit::Int
    max_rewinds_per_rebuild::Int
    state_cache::S
    state_cache2::S
    empty_∇ϕx::Vector{Float64}
    bound_violations::Base.RefValue{Int}
    total_proposals::Base.RefValue{Int}
end

accept_reflection_event(::OptimisticState, args...) = true

function _to_internal(strat::OptimisticStrategy, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::StatisticCounter)
    OptimisticState(
        PiecewiseLinearBound(strat.N),
        strat.N,
        Ref(strat.N),
        Ref(strat.t_max),
        strat.α⁺,
        strat.safety_limit,
        strat.max_rewinds,
        copy(state),
        copy(state),
        similar(state.ξ.x, 0),
        Ref(0),
        Ref(0),
    )
end

# ─── Grid rate evaluation (gradient-only, no HVP) ───────────────────────────

function evaluate_grid_rates!(plb::PiecewiseLinearBound, state_orig::AbstractPDMPState,
    state_scratch::AbstractPDMPState, flow::ContinuousDynamics, grad_func)

    copyto!(state_scratch, state_orig)
    ∇U_xt = grad_func(state_scratch.ξ.x)
    plb.y_vals[1] = pos(λ(state_scratch.ξ, ∇U_xt, flow))

    for i in 2:length(plb.y_vals)
        Δt = plb.t_grid[i] - plb.t_grid[i - 1]
        move_forward_time!(state_scratch, Δt, flow)
        ∇U_xt = grad_func(state_scratch.ξ.x)
        plb.y_vals[i] = pos(λ(state_scratch.ξ, ∇U_xt, flow))
    end
end

# ─── t_max adaptation ───────────────────────────────────────────────────────

function _adapt_t_max_optimistic!(alg::OptimisticState, τ_accepted::Float64)
    t_max = alg.t_max[]
    if τ_accepted < 0.25 * t_max
        new_t_max = max(4.0 * τ_accepted, 0.1)
        if new_t_max < t_max
            alg.t_max[] = new_t_max
        end
    end
end

# ─── Main event time proposal ────────────────────────────────────────────────

function next_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::OptimisticState, state::AbstractPDMPState, cache, stats::StatisticCounter,
    max_horizon::Float64=Inf, include_refresh::Bool=true) where {FL<:ContinuousDynamics}

    plb = alg.plb
    state_ = alg.state_cache
    state2_ = alg.state_cache2
    copyto!(state2_, state)

    grad_func = make_grad_U_func(state2_, flow, model.grad, cache)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    default_return = GradientMeta(alg.empty_∇ϕx)

    τ_refresh = ispositive(λ_refresh) ? rand(Exponential(inv(λ_refresh))) : Inf
    effective_horizon = min(alg.t_max[], τ_refresh, max_horizon)

    N_active = alg.N_current[]
    max_N = 4 * alg.N_base  # cap grid refinement at 4× base

    # Build grid and evaluate gradient-only rates
    reinitialize_grid!(plb, effective_horizon, N_active)
    evaluate_grid_rates!(plb, state2_, state_, flow, grad_func)
    compute_cumulative_integral!(plb)
    stats.grid_builds += 1
    stats.grid_points_evaluated += N_active + 1

    max_t_max = max_grid_horizon(flow)
    rewind_count = 0
    cumulative_exp = 0.0

    for _ in 1:alg.safety_limit
        cumulative_exp += rand(Exponential())
        τ, λ_tilde = propose_from_linear_bound(plb, cumulative_exp)
        alg.total_proposals[] += 1

        if !isfinite(τ) || τ >= effective_horizon
            if effective_horizon < alg.t_max[]
                return τ_refresh, :refresh, default_return
            end
            t_max = alg.t_max[]
            alg.t_max[] = min(t_max * alg.α⁺, max_t_max)
            stats.grid_grows += 1
            return t_max, :horizon_hit, default_return
        end

        if τ_refresh < τ
            return τ_refresh, :refresh, default_return
        end

        # Acceptance test: move to proposed time and compute true rate
        copyto!(state_, state2_)
        move_forward_time!(state_, τ, flow)
        ∇ϕx = compute_gradient!(state_, model.grad, flow, cache)
        l_true = λ(state_.ξ, ∇ϕx, flow)

        if l_true > λ_tilde
            alg.bound_violations[] += 1
            rewind_count += 1

            if rewind_count > alg.max_rewinds_per_rebuild && N_active < max_N
                # Too many violations — double grid resolution and rebuild
                N_active = min(2 * N_active, max_N)
                alg.N_current[] = N_active
                reinitialize_grid!(plb, effective_horizon, N_active)
                evaluate_grid_rates!(plb, state2_, state_, flow, grad_func)
                compute_cumulative_integral!(plb)
                stats.grid_builds += 1
                stats.grid_points_evaluated += N_active + 1
                rewind_count = 0
                cumulative_exp = 0.0
                continue
            end

            insert_grid_point!(plb, τ, l_true)
            stats.grid_points_evaluated += 1
            cumulative_exp = 0.0
            continue
        end

        if rand() * λ_tilde <= l_true
            _adapt_t_max_optimistic!(alg, τ)
            # Decay N_current back toward N_base after successful acceptance
            if N_active > alg.N_base
                alg.N_current[] = max(alg.N_base, N_active - 2)
            end
            return τ, :reflect, GradientMeta(∇ϕx)
        end

        # Rejection: cumulative_exp already advanced, next proposal will be later
    end

    error("OptimisticStrategy: safety limit reached after $(alg.safety_limit) iterations")
end

# COV_EXCL_STOP
