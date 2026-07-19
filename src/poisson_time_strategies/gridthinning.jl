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

function _piecewise_constant_area(pcb::PiecewiseConstantBound)
    area = 0.0
    @inbounds for i in eachindex(pcb.Λ_vals)
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

"""
    GridThinningStrategy(; bound=:constant, kwargs...)

Adaptive GridThinning configuration.

`curvature_backend` controls directional curvature evaluation explicitly:
`:exact` requires a compatible joint, VHV, or HVP provider;
`:finite_difference` evaluates curvature from shifted gradients; and `:auto`
uses an exact provider when available, otherwise finite differences. Finite
differences approximate the directional derivative and therefore have
step-size/truncation error even when the underlying gradients are exact.
`use_fd_hvp` remains a compatibility alias for
`curvature_backend=:finite_difference`.

Preferred user-facing bounds are `:constant`, `:flat`, `:linear`, `:auto`,
and `:value_quadratic`. The explicit experimental `:shared_node` mode uses
corrected signed rates at consecutive nodes, quadratic interpolation, and
empirical secant-disagreement inflation. It is a numerical clock, not a
certified thinning envelope. Passing neither `bound` keyword keeps the historical
constant GridThinning behavior used by downstream packages.

For `bound=:value_quadratic`, `curvature_bound` is a certified upper bound on
the scalar rate curvature over each grid cell. If the cell width is `h`, the
envelope is `max(r_left, r_right, 0) + curvature_bound*h^2/8`. The value may be
a finite number or a callable `(state, flow, a, b) -> M_cell`. A callable may
return `nothing` to fall back to ordinary GridThinning for that cell. Supplying
a bound that is too small invalidates the envelope; proposal-time checks do not
certify behavior between proposals.
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
    curvature_backend::Symbol
    post_warmup_simplify::Bool
    lazy::Bool
    bound::Symbol
    curvature_bound
    bound_violation::Symbol
    linear_area_threshold::Float64
    linear_min_area_gain::Float64
    max_rejections_before_tail_restart::Int
    max_componentwise_affine_segments_per_cell::Int
    lazy_low_tightness_threshold::Float64
    lazy_max_low_tightness_rejections::Int
    lazy_max_rejections::Int
    allow_small_boomerang::Bool
    warmup_tuning
end

"""
    GridWarmupTuning(; target_gradients_per_event=7.0,
        horizon_rate_high=0.35, rejection_rate_high=1.25,
        min_tmax=0.1, max_tmax=Inf, n_step=1,
        tmax_grow=1.25, tmax_shrink=0.75)

Optional post-warmup grid schedule tuning for `GridThinningStrategy`.
Pass an instance as `warmup_tuning=` to enable it explicitly.
"""
struct GridWarmupTuning
    target_gradients_per_event::Float64
    horizon_rate_high::Float64
    rejection_rate_high::Float64
    min_tmax::Float64
    max_tmax::Float64
    n_step::Int
    tmax_grow::Float64
    tmax_shrink::Float64
end

function GridWarmupTuning(; target_gradients_per_event::Real=7.0,
    horizon_rate_high::Real=0.35, rejection_rate_high::Real=1.25,
    min_tmax::Real=0.1, max_tmax::Real=Inf, n_step::Integer=1,
    tmax_grow::Real=1.25, tmax_shrink::Real=0.75)
    target = float(target_gradients_per_event)
    horizon_hi = float(horizon_rate_high)
    rejection_hi = float(rejection_rate_high)
    min_t = float(min_tmax)
    max_t = float(max_tmax)
    grow = float(tmax_grow)
    shrink = float(tmax_shrink)
    ispositive(target) || throw(ArgumentError("target_gradients_per_event must be positive"))
    ispositive(horizon_hi) || throw(ArgumentError("horizon_rate_high must be positive"))
    ispositive(rejection_hi) || throw(ArgumentError("rejection_rate_high must be positive"))
    min_t >= 0 || throw(ArgumentError("min_tmax must be nonnegative"))
    max_t > min_t || throw(ArgumentError("max_tmax must be greater than min_tmax"))
    n_step > 0 || throw(ArgumentError("n_step must be positive"))
    grow > 1 || throw(ArgumentError("tmax_grow must be greater than 1"))
    0 < shrink < 1 || throw(ArgumentError("tmax_shrink must lie between 0 and 1"))
    return GridWarmupTuning(target, horizon_hi, rejection_hi, min_t, max_t,
        Int(n_step), grow, shrink)
end

"""
    WarmupCurvatureBound(; initial_bound, min_bound=0.0, safety_factor=2.0,
        probe_fraction=0.5, probe_stride=100, max_observations=1000,
        apply=true)

Experimental empirical curvature provider for `GridThinningStrategy(;
bound=:value_quadratic)`.

During warmup, midpoint probes estimate the smallest observed `M` needed by the
implemented constant endpoint envelope. After warmup, `finish_warmup!` freezes
the provider to `max(min_bound, safety_factor * observed_max)` when
`apply=true`; with `apply=false` the initial bound is retained and the observed
suggestion is only recorded.

This provider is a tuning aid, not a certificate. Use a finite number or a
problem-specific callable when exactness depends on an analytical bound.
"""
mutable struct WarmupCurvatureBound
    initial_bound::Float64
    current_bound::Float64
    min_bound::Float64
    safety_factor::Float64
    probe_fraction::Float64
    probe_stride::Int
    max_observations::Int
    cells_seen::Int
    observed_max::Float64
    observations::Int
    active::Bool
    apply::Bool
end

function WarmupCurvatureBound(; initial_bound::Real, min_bound::Real=0.0,
    safety_factor::Real=2.0, probe_fraction::Real=0.5,
    probe_stride::Integer=100, max_observations::Integer=1000,
    apply::Bool=true)
    initial = _curvature_bound_value(initial_bound)
    min_value = _curvature_bound_value(min_bound)
    factor = Float64(safety_factor)
    isfinite(factor) && factor >= 1.0 ||
        throw(ArgumentError("safety_factor must be finite and at least 1"))
    probe = Float64(probe_fraction)
    0.0 < probe < 1.0 ||
        throw(ArgumentError("probe_fraction must lie strictly between 0 and 1"))
    probe_stride > 0 || throw(ArgumentError("probe_stride must be positive"))
    max_observations > 0 || throw(ArgumentError("max_observations must be positive"))
    return WarmupCurvatureBound(initial, initial, min_value, factor, probe,
        Int(probe_stride), Int(max_observations), 0, 0.0, 0, true, apply)
end

(bound::WarmupCurvatureBound)(state::AbstractPDMPState, flow::ContinuousDynamics, a::Real, b::Real) =
    bound.current_bound

function _update_warmup_curvature_bound!(bound::WarmupCurvatureBound, required::Real)
    bound.active || return bound
    value = max(Float64(required), 0.0)
    bound.observed_max = max(bound.observed_max, value)
    bound.observations += 1
    return bound
end

function _finish_warmup_curvature_bound!(bound::WarmupCurvatureBound)
    bound.active || return bound
    suggested = max(bound.min_bound, bound.safety_factor * bound.observed_max)
    if bound.apply
        bound.current_bound = suggested
    end
    bound.active = false
    return bound
end

_normalize_grid_bound(::Nothing) = :constant

function _normalize_curvature_backend(curvature_backend, use_fd_hvp::Bool)
    if curvature_backend === nothing
        return use_fd_hvp ? :finite_difference : :auto
    end
    backend = Symbol(curvature_backend)
    backend in (:exact, :finite_difference, :auto) || throw(ArgumentError(
        "curvature_backend must be :exact, :finite_difference, or :auto"))
    use_fd_hvp && backend !== :finite_difference && throw(ArgumentError(
        "use_fd_hvp=true conflicts with curvature_backend=$(backend); use curvature_backend=:finite_difference"))
    return backend
end

function _normalize_grid_bound(bound::Symbol)
    bound === :constant && return :constant
    bound === :flat && return :flat
    bound === :linear && return :linear
    bound === :value_quadratic && return :value_quadratic
    bound === :shared_node && return :shared_node
    bound === :auto && return :auto
    bound === :sticky_auto && return :sticky_auto
    throw(ArgumentError("unknown GridThinning bound $(bound)"))
end

function GridThinningStrategy(; N::Int=20, N_min::Int=5, t_max::Real=2.0, α⁺::Real=1.5, α⁻::Real=0.5,
    safety_limit::Int=500, early_stop_threshold::Real=5.0, use_fd_hvp::Bool=false,
    curvature_backend=nothing, post_warmup_simplify::Bool=false,
    lazy::Bool=true, bound=nothing, curvature_bound=nothing, bound_violation=nothing, linear_area_threshold::Real=0.95,
    linear_min_area_gain::Real=0.0, max_rejections_before_tail_restart::Int=100, max_componentwise_affine_segments_per_cell::Int=64,
    lazy_low_tightness_threshold::Real=0.1, lazy_max_low_tightness_rejections::Int=3,
    lazy_max_rejections::Int=0, allow_small_boomerang::Bool=false, warmup_tuning=nothing)
    bound_symbol = _normalize_grid_bound(bound)
    if bound_symbol === :shared_node
        curvature_bound isa Real && isfinite(curvature_bound) && curvature_bound >= 0 ||
            throw(ArgumentError("bound=:shared_node requires a finite nonnegative numerical curvature_bound used as empirical inflation"))
    end
    bound_violation_symbol = bound_violation === nothing ?
        (bound_symbol === :constant ? :count : bound_symbol === :shared_node ? :throw : :shrink) : Symbol(bound_violation)
    lazy_low_tightness_threshold >= 0 ||
        throw(ArgumentError("lazy_low_tightness_threshold must be nonnegative"))
    lazy_max_low_tightness_rejections > 0 ||
        throw(ArgumentError("lazy_max_low_tightness_rejections must be positive"))
    lazy_max_rejections >= 0 ||
        throw(ArgumentError("lazy_max_rejections must be nonnegative; use 0 for the default derived cap"))
    backend = _normalize_curvature_backend(curvature_backend, use_fd_hvp)
    return GridThinningStrategy(
        N, N_min, Float64(t_max), Float64(α⁺), Float64(α⁻), safety_limit,
        Float64(early_stop_threshold), backend === :finite_difference, backend,
        post_warmup_simplify,
        lazy, bound_symbol, curvature_bound, bound_violation_symbol,
        Float64(linear_area_threshold),
        Float64(linear_min_area_gain),
        max_rejections_before_tail_restart, max_componentwise_affine_segments_per_cell,
        Float64(lazy_low_tightness_threshold), lazy_max_low_tightness_rejections,
        lazy_max_rejections, allow_small_boomerang, warmup_tuning)
end

function Base.show(io::IO, strat::GridThinningStrategy)
    print(io, "GridThinningStrategy(")
    print(io, "N=", strat.N, ", N_min=", strat.N_min, ", t_max=", strat.t_max)
    print(io, ", bound=", strat.bound)
    strat.bound in (:linear, :auto) && print(io, ", linear_area_threshold=", strat.linear_area_threshold,
        ", linear_min_area_gain=", strat.linear_min_area_gain)
    strat.bound in (:value_quadratic, :shared_node) && print(io, ", curvature_bound=", strat.curvature_bound)
    print(io, ")")
end

_default_early_stop(::ContinuousDynamics, est::Float64) = est
_default_early_stop(pd::PreconditionedDynamics, est::Float64) = _default_early_stop(pd.dynamics, est)

function _grid_min_cells(strat::GridThinningStrategy, flow::ContinuousDynamics, N_base::Int)
    strat.bound in (:value_quadratic, :shared_node) && return strat.N_min
    if flow isa AnyBoomerang && strat.allow_small_boomerang
        return strat.N_min
    end
    return min_grid_cells(flow, strat.N_min, N_base)
end

function _to_internal(strat::GridThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    T = typeof(strat.t_max)
    0.0 <= strat.linear_area_threshold || throw(ArgumentError("linear_area_threshold must be nonnegative"))
    0.0 <= strat.linear_min_area_gain || throw(ArgumentError("linear_min_area_gain must be nonnegative"))
    strat.max_rejections_before_tail_restart > 0 || throw(ArgumentError("max_rejections_before_tail_restart must be positive"))
    strat.max_componentwise_affine_segments_per_cell > 0 ||
        throw(ArgumentError("max_componentwise_affine_segments_per_cell must be positive"))
    # Derivative info is always available: either via HVP, VHV, joint, or FD fallback.
    N_base = strat.N
    N_min = _grid_min_cells(strat, flow, N_base)
    est = _default_early_stop(flow, strat.early_stop_threshold)
    est = _adjust_early_stop(model.grad, est)
    N_max = max(N_base + 4, 2 * N_base)
    alg = _build_grid_adaptive_state(strat, state, flow, model, cache, N_base, N_min, N_max, est, stats)
    _set_counter_grid_initial_N(stats, alg.N[])
    _set_counter_grid_final_N(stats, alg.N[])
    _set_counter_grid_initial_tmax(stats, alg.t_max[])
    _set_counter_grid_final_tmax(stats, alg.t_max[])
    _set_counter_grid_initial_h(stats, alg.t_max[] / alg.N[])
    _set_counter_grid_final_h(stats, alg.t_max[] / alg.N[])
    _set_counter_grid_schedule_frozen(stats, false)
    selected_backend = _selected_curvature_backend(strat.curvature_backend, model, flow)
    _set_counter_curvature_backend(stats, selected_backend)
    return alg
end

_adjust_early_stop(::GradientStrategy, est::Float64) = est
_adjust_early_stop(grad::SubsampledGradient, est::Float64) =
    grad.fixed_batch_within_event ? est : Inf

function _effective_grid_horizon(::GradientStrategy, t_max::Float64, τ_refresh::Float64, max_horizon::Float64,
    max_horizon_event::Symbol=:horizon_hit)
    if τ_refresh <= t_max && τ_refresh <= max_horizon
        return τ_refresh, :refresh
    elseif max_horizon <= t_max
        return max_horizon, max_horizon_event
    end
    return t_max, :horizon_hit
end

_adapt_grid_on_horizon!(alg, ::ContinuousDynamics) = nothing

function _adapt_grid_on_horizon!(alg, ::PreconditionedDynamics{<:AbstractPreconditioner,<:ZigZag})
    N = alg.N[]
    new_N = max(alg.N_min, cld(N, 2))
    new_N == N && return nothing
    alg.N[] = new_N
    return nothing
end

_restore_lazy_after_grid!(alg, ::ContinuousDynamics) = nothing

function _restore_lazy_after_grid!(alg, ::AnyBoomerang)
    alg.lazy_enabled[] = alg.lazy
    return nothing
end

function _return_grid_horizon!(alg, stats::AbstractStatisticCounter, flow::ContinuousDynamics, t_max::Float64, effective_horizon::Float64, horizon_event::Symbol, max_t_max::Float64, default_return)
    _inc_counter_grid_horizon_hits(stats)
    alg.has_cached_gradient[] = false
    horizon_event === :horizon_hit || (alg.has_cached_rate_derivative[] = false)
    if horizon_event === :horizon_hit
        if !alg.schedule_frozen[]
            _adapt_grid_on_horizon!(alg, flow)
            alg.t_max[] = min(t_max * alg.α⁺, max_t_max)
            recompute_time_grid!(alg)
            _inc_counter_grid_grows(stats)
        end
        _restore_lazy_after_grid!(alg, flow)
        return t_max, :horizon_hit, default_return
    end
    _restore_lazy_after_grid!(alg, flow)
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

function _build_grid_adaptive_state(strat::GridThinningStrategy, state::S, flow::ContinuousDynamics, model::PDMPModel, cache, N_base::Int, N_min::Int, N_max::Int, est, stats=nothing) where S<:AbstractPDMPState
    T = typeof(strat.t_max)
    state_cache = copy(state)
    state_cache2 = copy(state)
    grad_provider = GradientProvider(state_cache.ξ.θ, flow, model.grad, cache)
    fd_grad_provider = grad_provider
    grad_hvp_provider = GradHVPProvider(grad_provider, model.hvp)
    vhv_provider = VHVProvider(grad_provider, model.vhv, similar(state.ξ.x))
    fd_vhv_provider = FiniteDiffVHV(fd_grad_provider, similar(state.ξ.x), similar(state.ξ.x), similar(state.ξ.x), stats)
    event_provider = _initial_grid_event_provider(strat.curvature_backend, model, flow,
        grad_hvp_provider, vhv_provider, fd_vhv_provider)
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
        Ref(false),
        est,
        state_cache,
        state_cache2,
        similar(state.ξ.x, 0),
        strat.curvature_backend,
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
        Ref(NaN),
        Ref(NaN),
        Ref(false),
        grad_provider,
        grad_hvp_provider,
        vhv_provider,
        fd_vhv_provider,
        event_provider,
        strat.bound,
        strat.curvature_bound,
        strat.bound_violation,
        strat.linear_area_threshold,
        strat.linear_min_area_gain,
        strat.max_rejections_before_tail_restart,
        strat.max_componentwise_affine_segments_per_cell,
        strat.lazy_low_tightness_threshold,
        strat.lazy_max_low_tightness_rejections,
        strat.lazy_max_rejections,
        strat.warmup_tuning,
    )
end

struct GridAdaptiveState{S<:AbstractPDMPState,V<:AbstractVector,P,GH,VP,FD,EP} <: PoissonTimeStrategy
    pcb::PiecewiseConstantBound{Float64}
    affine_bound::PiecewiseAffineBound{Float64}
    N::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    N_min::Int
    N_max::Int
    schedule_frozen::Base.RefValue{Bool}
    early_stop_threshold::Float64
    state_cache::S
    state_cache2::S
    empty_∇ϕx::V
    curvature_backend::Symbol
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
    cached_rate::Base.RefValue{Float64}
    cached_rate_derivative::Base.RefValue{Float64}
    has_cached_rate_derivative::Base.RefValue{Bool}
    grad_provider::P
    grad_hvp_provider::GH
    vhv_provider::VP
    fd_vhv_provider::FD
    event_provider::EP
    bound::Symbol
    curvature_bound
    bound_violation::Symbol
    linear_area_threshold::Float64
    linear_min_area_gain::Float64
    max_rejections_before_tail_restart::Int
    max_componentwise_affine_segments_per_cell::Int
    lazy_low_tightness_threshold::Float64
    lazy_max_low_tightness_rejections::Int
    lazy_max_rejections::Int
    warmup_tuning
end

accept_reflection_event(::Random.AbstractRNG, ::GridAdaptiveState, args...) = true
accept_reflection_event(::GridAdaptiveState, args...) = true

recompute_time_grid!(alg::GridAdaptiveState) = recompute_time_grid!(alg.pcb, alg.t_max[], alg.N[])

finish_warmup!(::PoissonTimeStrategy, ::AbstractStatisticCounter) = nothing
finish_warmup!(alg::PoissonTimeStrategy, stats::AbstractStatisticCounter, ::ContinuousDynamics) =
    finish_warmup!(alg, stats)

_record_final_grid_state!(::PoissonTimeStrategy, ::AbstractStatisticCounter) = nothing

function _record_final_grid_state!(alg::GridAdaptiveState, stats::AbstractStatisticCounter)
    _set_counter_grid_final_N(stats, alg.N[])
    _set_counter_grid_final_tmax(stats, alg.t_max[])
    _set_counter_grid_final_h(stats, alg.t_max[] / alg.N[])
    _set_counter_grid_schedule_frozen(stats, alg.schedule_frozen[])
    return nothing
end

function finish_warmup!(alg::GridAdaptiveState, ::AbstractStatisticCounter)
    alg.curvature_bound isa WarmupCurvatureBound &&
        _finish_warmup_curvature_bound!(alg.curvature_bound)
    return nothing
end

function _required_grid_counter_value(stats::AbstractStatisticCounter, name::Symbol)
    hasproperty(stats, name) || throw(ArgumentError(
        "GridWarmupTuning requires statistic counter field $(name)"))
    value = getproperty(stats, name)
    value isa Real || throw(ArgumentError(
        "GridWarmupTuning requires numeric statistic counter field $(name)"))
    return Float64(value)
end

function _maybe_tune_grid_after_warmup!(alg::GridAdaptiveState, stats::AbstractStatisticCounter, flow::ContinuousDynamics)
    tuning = alg.warmup_tuning
    tuning === nothing && return nothing
    events = max(_required_grid_counter_value(stats, :warmup_events), 1.0)
    endpoint_grad = _required_grid_counter_value(stats, :warmup_grid_endpoint_gradient_calls)
    acceptance_grad = _required_grid_counter_value(stats, :warmup_grid_acceptance_gradient_calls)
    gradients_per_event = (endpoint_grad + acceptance_grad) / events
    horizon_hits = _required_grid_counter_value(stats, :grid_horizon_hits)
    rejections = _required_grid_counter_value(stats, :lazy_proposal_rejections)
    horizon_rate = horizon_hits / events
    rejection_rate = rejections / events

    _set_counter_grid_warmup_objective_events(stats, events)
    _set_counter_grid_warmup_objective_endpoint_gradients(stats, endpoint_grad)
    _set_counter_grid_warmup_objective_acceptance_gradients(stats, acceptance_grad)
    _set_counter_grid_warmup_objective_gradients_per_event(stats, gradients_per_event)
    _set_counter_grid_warmup_objective_horizon_hits(stats, horizon_hits)
    _set_counter_grid_warmup_objective_horizon_rate(stats, horizon_rate)
    _set_counter_grid_warmup_objective_rejections(stats, rejections)
    _set_counter_grid_warmup_objective_rejection_rate(stats, rejection_rate)

    changed = false
    if gradients_per_event > tuning.target_gradients_per_event && alg.N[] > alg.N_min
        alg.N[] = max(alg.N_min, alg.N[] - tuning.n_step)
        changed = true
    elseif horizon_rate > tuning.horizon_rate_high && alg.N[] < alg.N_max
        alg.N[] = min(alg.N_max, alg.N[] + tuning.n_step)
        changed = true
    end

    old_tmax = alg.t_max[]
    max_tmax = min(tuning.max_tmax, max_grid_horizon(flow))
    if horizon_rate > tuning.horizon_rate_high
        alg.t_max[] = min(max_tmax, old_tmax * tuning.tmax_grow)
    elseif rejection_rate > tuning.rejection_rate_high
        alg.t_max[] = max(tuning.min_tmax, old_tmax * tuning.tmax_shrink)
    end
    changed |= alg.t_max[] != old_tmax

    if changed
        recompute_time_grid!(alg)
        _set_counter_grid_N_current(stats, alg.N[])
    end
    alg.schedule_frozen[] = true
    _set_counter_grid_final_N(stats, alg.N[])
    _set_counter_grid_final_tmax(stats, alg.t_max[])
    _set_counter_grid_final_h(stats, alg.t_max[] / alg.N[])
    _set_counter_grid_schedule_frozen(stats, true)
    return nothing
end

function finish_warmup!(alg::GridAdaptiveState, stats::AbstractStatisticCounter, flow::ContinuousDynamics)
    finish_warmup!(alg, stats)
    _maybe_tune_grid_after_warmup!(alg, stats, flow)
    return nothing
end

function reset_grid_scale!(alg::GridAdaptiveState, t_max::Float64=2.0)
    alg.t_max[] = t_max
    alg.N[] = alg.N_max
    alg.lazy_enabled[] = alg.lazy
    alg.has_cached_gradient[] = false
    alg.has_cached_rate_derivative[] = false
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
    alg.has_cached_rate_derivative[] = false
    return nothing
end

function _cache_endpoint_rate_derivative!(alg::GridAdaptiveState, y::Real, d::Real, horizon_event::Symbol)
    if horizon_event === :horizon_hit && isfinite(y) && isfinite(d)
        alg.cached_rate[] = Float64(y)
        alg.cached_rate_derivative[] = Float64(d)
        alg.has_cached_rate_derivative[] = true
    else
        alg.has_cached_rate_derivative[] = false
    end
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
    alg.has_cached_rate_derivative[] = false
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
            alg.has_cached_rate_derivative[] = false
            return t_max, horizon_event, default_return
        end

        if τ_refresh < τ_proposal
            alg.has_cached_gradient[] = false
            alg.has_cached_rate_derivative[] = false
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

function _constant_bound_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::GridAdaptiveState,
    state::AbstractPDMPState, cache, stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol=:horizon_hit, probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe())
    return _constant_bound_event_time(
        Random.default_rng(), model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler,
    )
end

_can_use_value_quadratic_grid(flow::ContinuousDynamics, curvature_bound) =
    _rate_aggregation(flow) === :scalar && curvature_bound !== nothing

function _shared_node_cell_bound(
    t_previous::Real,
    y_previous::Real,
    t_left::Real,
    y_left::Real,
    t_right::Real,
    y_right::Real,
    inflation::Real,
)
    scale = max(Float64(inflation), 0.0)
    endpoint_max = max(Float64(y_left), Float64(y_right), 0.0)
    if !isfinite(t_previous) || !(t_previous < t_left < t_right)
        return endpoint_max + scale * abs(Float64(y_right) - Float64(y_left))
    end
    slope_left = (Float64(y_left) - Float64(y_previous)) / (Float64(t_left) - Float64(t_previous))
    slope_right = (Float64(y_right) - Float64(y_left)) / (Float64(t_right) - Float64(t_left))
    quadratic = (slope_right - slope_left) / (Float64(t_right) - Float64(t_previous))
    linear = slope_left - quadratic * (Float64(t_previous) + Float64(t_left))
    predicted_max = endpoint_max
    if quadratic < 0.0
        vertex = -linear / (2 * quadratic)
        if t_left < vertex < t_right
            constant = Float64(y_left) - quadratic * Float64(t_left)^2 - linear * Float64(t_left)
            predicted_max = max(predicted_max, quadratic * vertex^2 + linear * vertex + constant)
        end
    end
    defect = abs(slope_right - slope_left) * (Float64(t_right) - Float64(t_left))
    return max(predicted_max, 0.0) + scale * defect
end

@inline function _value_quadratic_cell_bound(y_left::Real, y_right::Real, cell_width::Real, residual_bound::Real)
    residual = max(Float64(residual_bound), 0.0) * Float64(cell_width)^2 / 8
    return max(Float64(y_left), Float64(y_right), 0.0) + residual
end

function _value_rate_at_state!(
    probe_failure_handler::GridBoundaryProbe,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    grad_provider,
    t_valid::Float64,
    t_invalid::Float64,
)
    rate, _ = _get_rate_and_deriv_or_throw(
        probe_failure_handler, state, flow, (grad_provider, nothing), false;
        t_valid, t_invalid)
    return rate
end

function _signed_rate_at_state!(
    ::NoGridBoundaryProbe,
    state::AbstractPDMPState,
    grad_provider,
    ::Float64,
    ::Float64,
)
    return Float64(dot(grad_provider(state.ξ.x), state.ξ.θ))
end

function _signed_rate_at_state!(
    probe::GridBoundaryProbeHandler,
    state::AbstractPDMPState,
    grad_provider,
    t_valid::Float64,
    t_invalid::Float64,
)
    try
        return Float64(dot(grad_provider(state.ξ.x), state.ξ.θ))
    catch err
        _throw_grid_boundary_error(probe, state, err; t_valid, t_invalid)
    end
end

_maybe_probe_warmup_curvature_bound!(args...) = nothing

function _maybe_probe_warmup_curvature_bound!(
    bound::WarmupCurvatureBound,
    probe_failure_handler::GridBoundaryProbe,
    state_probe::AbstractPDMPState,
    state_start::AbstractPDMPState,
    flow::ContinuousDynamics,
    grad_provider,
    t_left::Float64,
    t_right::Float64,
    y_left::Float64,
    y_right::Float64,
    stats::AbstractStatisticCounter,
)
    bound.active || return nothing
    bound.cells_seen += 1
    bound.observations < bound.max_observations || return nothing
    rem(bound.cells_seen, bound.probe_stride) == 0 || return nothing
    h = t_right - t_left
    ispositive(h) || return nothing
    copyto!(state_probe, state_start)
    t_probe = t_left + bound.probe_fraction * h
    t_probe != 0.0 && move_forward_time!(state_probe, t_probe, flow)
    _inc_counter_grid_certificate_calls(stats)
    _inc_counter_grid_points_evaluated(stats, 1)
    y_probe = _value_rate_at_state!(
        probe_failure_handler, state_probe, flow, grad_provider, t_left, t_right)
    required = 8 * max(0.0, y_probe - max(y_left, y_right, 0.0)) / (h * h)
    _update_warmup_curvature_bound!(bound, required)
    return nothing
end

function _next_event_time_value_quadratic!(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
    ) where {FL<:ContinuousDynamics}

    shared_node = alg.bound === :shared_node
    _can_use_value_quadratic_grid(flow, alg.curvature_bound) ||
        return _next_event_time_lazy!(rng, _make_grad_provider(alg.grad_provider, model, flow, alg),
            model, flow, alg, state, cache, stats, max_horizon, include_refresh,
            max_horizon_event, probe_failure_handler)

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

    if alg.has_cached_gradient[]
        _inc_counter_grid_cached_endpoint_reuses(stats)
        y_left = shared_node ? Float64(dot(alg.cached_gradient, state_.ξ.θ)) :
            pos(λ(state_.ξ, alg.cached_gradient, flow))
        alg.has_cached_gradient[] = false
        alg.has_cached_rate_derivative[] = false
    else
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        y_left = shared_node ?
            _signed_rate_at_state!(probe_failure_handler, state_, alg.grad_provider, 0.0, 0.0) :
            _value_rate_at_state!(probe_failure_handler, state_, flow, alg.grad_provider, 0.0, 0.0)
    end
    t_left = 0.0
    t_previous = NaN
    y_previous = NaN

    cumulative_area = 0.0
    exp_target = rand(rng, Exponential())
    safety_limit = alg.safety_limit
    default_max_rejections = min(25, max(10, alg.safety_limit ÷ 2))
    max_rejections = alg.lazy_max_rejections > 0 ? alg.lazy_max_rejections : default_max_rejections
    low_tightness_rejections = 0
    low_tightness_threshold = alg.lazy_low_tightness_threshold
    max_low_tightness_rejections = alg.lazy_max_low_tightness_rejections
    proposal_attempts = 0
    proposal_rejections = 0
    tightness_sum = 0.0
    min_tightness = Inf
    max_tightness = -Inf

    _inc_counter_grid_builds(stats)
    _inc_counter_grid_points_evaluated(stats, 1)
    _record_grid_schedule!(stats, alg)

    while safety_limit > 0
        safety_limit -= 1

        t_right = t_left + Δt
        t_right > effective_horizon && (t_right = effective_horizon)
        Δt_cell = t_right - t_left
        if Δt_cell <= 0.0
            return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event, max_t_max, default_return)
        end

        move_forward_time!(state_, Δt_cell, flow)
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        y_right = shared_node ?
            _signed_rate_at_state!(probe_failure_handler, state_, alg.grad_provider, t_left, t_right) :
            _value_rate_at_state!(probe_failure_handler, state_, flow, alg.grad_provider, t_left, t_right)
        _inc_counter_grid_points_evaluated(stats, 1)

        residual_bound = if shared_node
            Float64(alg.curvature_bound)
        else
            value = _evaluate_curvature_bound(alg.curvature_bound, state, flow, t_left, t_right, stats)
            value === nothing && return _next_event_time_lazy!(
                rng, _make_grad_provider(alg.grad_provider, model, flow, alg),
                model, flow, alg, state, cache, stats, max_horizon, include_refresh,
                max_horizon_event, probe_failure_handler)
            _maybe_probe_warmup_curvature_bound!(
                alg.curvature_bound, probe_failure_handler, state2_, state, flow,
                alg.grad_provider, t_left, t_right, Float64(y_left), Float64(y_right), stats)
            value
        end

        if shared_node
            _inc_counter_shared_node_cells(stats)
            isfinite(t_previous) ? _inc_counter_shared_node_three_point_cells(stats) :
                _inc_counter_shared_node_two_point_cells(stats)
        end
        Λ_cell = shared_node ?
            _shared_node_cell_bound(t_previous, y_previous, t_left, y_left, t_right, y_right, residual_bound) :
            _value_quadratic_cell_bound(y_left, y_right, Δt_cell, residual_bound)
        area_cell = pos(Λ_cell) * Δt_cell

        if area_cell <= 0.0
            t_previous = t_left
            y_previous = y_left
            t_left = t_right
            y_left = y_right
            if t_right >= effective_horizon
                alg.has_cached_rate_derivative[] = false
                return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                    max_t_max, default_return)
            end
            continue
        end

        if cumulative_area + area_cell < exp_target
            cumulative_area += area_cell
            t_previous = t_left
            y_previous = y_left
            t_left = t_right
            y_left = y_right
            if t_right >= effective_horizon
                alg.has_cached_rate_derivative[] = false
                return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                    max_t_max, default_return)
            end
            continue
        end

        while true
            lb_proposal = pos(Λ_cell)
            τ_proposal = t_left + (exp_target - cumulative_area) / lb_proposal
            if τ_proposal >= t_right || !isfinite(τ_proposal)
                t_previous = t_left
                y_previous = y_left
                t_left = t_right
                y_left = y_right
                cumulative_area = 0.0
                exp_target = rand(rng, Exponential())
                if t_right >= effective_horizon
                    alg.has_cached_rate_derivative[] = false
                    return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                        max_t_max, default_return)
                end
                break
            end

            if τ_refresh < τ_proposal
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                _invalidate_cached_gradient!(alg)
                return τ_refresh, :refresh, default_return
            end

            copyto!(state2_, state)
            move_forward_time!(state2_, τ_proposal, flow)
            _inc_counter_grid_acceptance_gradient_calls(stats)
            ∇ϕx = _compute_grid_gradient_or_throw!(
                state2_, state, flow, model, cache, t_left, τ_proposal, probe_failure_handler)

            l_actual = λ(state2_.ξ, ∇ϕx, flow)
            signed_actual = shared_node ? Float64(dot(∇ϕx, state2_.ξ.θ)) : Float64(l_actual)
            _inc_counter_grid_acceptance_tests(stats)
            proposal_attempts += 1

            tightness = _safe_tightness(l_actual, lb_proposal)
            tightness_sum += tightness
            min_tightness = min(min_tightness, tightness)
            max_tightness = max(max_tightness, tightness)

            if l_actual > lb_proposal * (1 + 1e-10) + 1e-12
                _inc_counter_lazy_fallback_bound_violation(stats)
                _inc_counter_grid_bound_violations(stats)
                if alg.bound_violation === :throw
                    throw(ErrorException(_grid_bound_violation_message(
                        alg, stats, state2_, flow, τ_proposal, NaN, l_actual,
                        lb_proposal, exp_target, λ_refresh, false)))
                elseif alg.bound_violation === :shrink
                    _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                    alg.has_cached_gradient[] = false
                    alg.has_cached_rate_derivative[] = false
                    _shrink_grid_after_bound_violation!(alg, stats)
                    return _next_event_time_lazy!(
                        rng, _make_grad_provider(alg.grad_provider, model, flow, alg),
                        model, flow, alg, state, cache, stats, max_horizon, include_refresh,
                        max_horizon_event, probe_failure_handler)
                end
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                return _next_event_time_lazy!(
                    rng, _make_grad_provider(alg.grad_provider, model, flow, alg),
                    model, flow, alg, state, cache, stats, max_horizon, include_refresh,
                    max_horizon_event, probe_failure_handler)
            end

            if rand(rng) * lb_proposal <= l_actual
                copyto!(alg.cached_gradient, ∇ϕx)
                alg.has_cached_gradient[] = true
                alg.has_cached_rate_derivative[] = false
                _adapt_grid_N!(alg, tightness)
                _adapt_grid_t_max!(alg, τ_proposal, model.grad)
                _set_counter_grid_N_current(stats, alg.N[])
                alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return τ_proposal, :reflect, GradientMeta(∇ϕx)
            end

            proposal_rejections += 1
            if tightness < low_tightness_threshold
                low_tightness_rejections += 1
            else
                low_tightness_rejections = 0
            end

            if low_tightness_rejections >= max_low_tightness_rejections || proposal_rejections >= max_rejections
                low_tightness_rejections >= max_low_tightness_rejections &&
                    _inc_counter_lazy_fallback_low_tightness(stats)
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_lazy!(
                    rng, _make_grad_provider(alg.grad_provider, model, flow, alg),
                    model, flow, alg, state, cache, stats, max_horizon, include_refresh,
                    max_horizon_event, probe_failure_handler)
            end

            cumulative_area += lb_proposal * (τ_proposal - t_left)
            t_previous = t_left
            y_previous = y_left
            t_left = τ_proposal
            y_left = signed_actual
            Δt_tail = t_right - t_left
            if Δt_tail <= 0.0
                t_left = t_right
                y_left = y_right
                cumulative_area = 0.0
                exp_target = rand(rng, Exponential())
                if t_right >= effective_horizon
                    alg.has_cached_rate_derivative[] = false
                    return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event,
                        max_t_max, default_return)
                end
                break
            end
            Λ_tail = shared_node ?
                _shared_node_cell_bound(t_previous, y_previous, t_left, y_left, t_right, y_right, residual_bound) :
                _value_quadratic_cell_bound(y_left, y_right, Δt_tail, residual_bound)
            isfinite(Λ_tail) && (Λ_cell = min(Λ_cell, Λ_tail))
            exp_target += rand(rng, Exponential())
        end
    end

    if isfinite(τ_refresh) && τ_refresh <= min(t_max, max_horizon)
        alg.has_cached_gradient[] = false
        alg.has_cached_rate_derivative[] = false
        return τ_refresh, :refresh, default_return
    end

    _throw_grid_safety_limit_error(state, flow, model; t_invalid=effective_horizon,
        message=shared_node ? "Safety limit reached in experimental shared-node Grid algorithm" :
            "Safety limit reached in value-quadratic Grid algorithm")
end

function _make_grad_provider(grad_func, model::PDMPModel, flow::ContinuousDynamics, alg::GridAdaptiveState)
    backend = _selected_curvature_backend(alg.curvature_backend, model, flow)
    backend === :finite_difference && return alg.fd_vhv_provider
    joint_func = model.joint
    if joint_func !== nothing && _joint_compatible(flow)
        return joint_func
    end
    vhv_func = model.vhv
    if vhv_func !== nothing
        return alg.vhv_provider
    end
    hvp_func = model.hvp
    hvp_func === nothing && throw(ArgumentError(
        "curvature_backend=:exact requires a compatible joint, VHV, or HVP provider"))
    return alg.grad_hvp_provider
end

function _initial_grid_event_provider(
    backend::Symbol,
    model::PDMPModel,
    flow::ContinuousDynamics,
    grad_hvp_provider,
    vhv_provider,
    fd_vhv_provider,
)
    selected = _selected_curvature_backend(backend, model, flow)
    selected === :finite_difference && return fd_vhv_provider
    joint_func = model.joint
    if joint_func !== nothing && _joint_compatible(flow)
        return joint_func
    end
    vhv_func = model.vhv
    vhv_func !== nothing && return vhv_provider
    hvp_func = model.hvp
    hvp_func === nothing && throw(ArgumentError(
        "curvature_backend=:exact requires a compatible joint, VHV, or HVP provider"))
    return grad_hvp_provider
end

function _next_event_time_with_provider!(
    rng::Random.AbstractRNG,
    grad_and_hvp,
    model::PDMPModel{<:GlobalGradientStrategy},
    flow::FL,
    alg::GridAdaptiveState,
    state::AbstractPDMPState,
    cache,
    stats::AbstractStatisticCounter,
    max_horizon::Float64,
    include_refresh::Bool,
    max_horizon_event::Symbol,
    probe_failure_handler::GridBoundaryProbe,
) where {FL<:ContinuousDynamics}
    if alg.lazy_enabled[]
        return _next_event_time_lazy!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end
    return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

@inline function _typed_grid_event(result::Tuple)
    τ = result[1]::Real
    τ_float = Float64(τ)::Float64
    return (τ_float, result[2]::Symbol, result[3]::GradientMeta)
end

function _next_event_time_with_selected_provider!(
    rng::Random.AbstractRNG,
    model::PDMPModel{<:GlobalGradientStrategy},
    flow::FL,
    alg::GridAdaptiveState,
    state::AbstractPDMPState,
    cache,
    stats::AbstractStatisticCounter,
    max_horizon::Float64,
    include_refresh::Bool,
    max_horizon_event::Symbol,
    probe_failure_handler::GridBoundaryProbe,
) where {FL<:ContinuousDynamics}
    backend = _selected_curvature_backend(alg.curvature_backend, model, flow)
    if backend === :finite_difference
        return _next_event_time_with_provider!(rng, alg.fd_vhv_provider, model, flow, alg, state,
            cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end
    joint_func = model.joint
    if joint_func !== nothing && _joint_compatible(flow)
        return _next_event_time_with_provider!(rng, joint_func, model, flow, alg, state,
            cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end
    vhv_func = model.vhv
    if vhv_func !== nothing
        return _next_event_time_with_provider!(rng, alg.vhv_provider, model, flow, alg, state,
            cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end
    hvp_func = model.hvp
    hvp_func === nothing && throw(ArgumentError(
        "curvature_backend=:exact requires a compatible joint, VHV, or HVP provider"))
    return _next_event_time_with_provider!(rng, alg.grad_hvp_provider, model, flow, alg, state,
        cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

@inline function _has_exact_curvature_provider(model::PDMPModel, flow::ContinuousDynamics)
    return (model.joint !== nothing && _joint_compatible(flow)) ||
        model.vhv !== nothing || model.hvp !== nothing
end

function _selected_curvature_backend(backend::Symbol, model::PDMPModel, flow::ContinuousDynamics)
    backend === :finite_difference && return :finite_difference
    has_exact = _has_exact_curvature_provider(model, flow)
    backend === :exact && !has_exact && throw(ArgumentError(
        "curvature_backend=:exact requires a compatible joint, VHV, or HVP provider"))
    return has_exact ? :exact : :finite_difference
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
        return _typed_grid_event(_constant_bound_event_time(rng, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler))
    end

    if alg.bound in (:value_quadratic, :shared_node)
        return _typed_grid_event(_next_event_time_value_quadratic!(
            rng, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler))
    end

    state_ = alg.state_cache
    copyto!(state_, state)

    return _next_event_time_with_provider!(rng, alg.event_provider, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _grid_bound_modes(alg::GridAdaptiveState, state::AbstractPDMPState, flow::ContinuousDynamics, provider)
    use_linear = _use_linear_bound(alg, state, flow, provider)
    use_single_pass_signed = _use_signed_grid_bound(alg, state, flow, provider)
    return (;
        use_linear,
        use_single_pass_signed,
        use_constant_batched_signed=!use_single_pass_signed && _supports_constant_grid_rate_derivatives(flow, provider),
    )
end

function _build_grid_bound_prefix!(pcb::PiecewiseConstantBound, state::AbstractPDMPState, flow::ContinuousDynamics, provider, alg::GridAdaptiveState,
    stats::AbstractStatisticCounter, state_cache::AbstractPDMPState, effective_horizon::Float64, cumulative_exp::Float64,
    probe_failure_handler::GridBoundaryProbe, modes; cached_gradient=nothing, cached_y0::Float64=NaN, cached_d0::Float64=NaN,
    cached_g0::Float64=NaN, cached_dg0::Float64=NaN, start_cell::Integer=1, initial_integral::Float64=0.0, append::Bool=false)

    if modes.use_single_pass_signed
        n_cells_bounded = construct_rate_bound_grid!(alg.affine_bound, pcb, state, flow, provider, alg.curvature_bound;
            cached_gradient, early_stop_threshold=cumulative_exp, state_cache, stats, max_time=effective_horizon,
            build_affine=modes.use_linear, linear_area_threshold=alg.linear_area_threshold, linear_min_area_gain=alg.linear_min_area_gain,
            auto=_auto_policy(alg.bound), max_componentwise_affine_segments_per_cell=alg.max_componentwise_affine_segments_per_cell,
            rate_value_buf=alg.rate_value_buf, rate_derivative_buf=alg.rate_derivative_buf, probe_failure_handler, start_cell, initial_integral, append)
    else
        n_cells_bounded = construct_upper_bound_grad_and_hess!(pcb, state, flow, provider, false;
            cached_y0, cached_d0, early_stop_threshold=cumulative_exp, stats, state_cache, max_time=effective_horizon,
            probe_failure_handler, start_cell, initial_integral)
    end
    if modes.use_linear && !modes.use_single_pass_signed
        if alg.bound === :linear
            construct_rate_grid!(pcb, state, flow, provider, n_cells_bounded; cached_g0, cached_dg0, state_cache, stats)
            build_rate_linear_bound!(alg.affine_bound, pcb, n_cells_bounded, state, flow, alg.curvature_bound, stats;
                linear_area_threshold=alg.linear_area_threshold, linear_min_area_gain=alg.linear_min_area_gain)
        else
            build_hybrid_affine_bound!(alg.affine_bound, pcb, n_cells_bounded, stats)
        end
    end
    _record_grid_schedule!(stats, alg)
    built_area = _grid_built_area(pcb, alg.affine_bound, modes.use_linear)
    _record_budget_grid_build!(stats, n_cells_bounded, built_area, cumulative_exp, append)
    return n_cells_bounded, built_area
end

function _next_event_time_grid!(rng::Random.AbstractRNG, grad_and_hvp::P, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe()) where {P, FL<:ContinuousDynamics}

    time_offset = 0.0
    tail_restart_count = 0

    @label restart_from_tail

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

    modes = _grid_bound_modes(alg, state, flow, grad_and_hvp)
    had_cached_gradient = alg.has_cached_gradient[]
    alg.has_cached_rate_derivative[] = false
    if modes.use_single_pass_signed
        alg.has_cached_gradient[] = false
        cached_gradient = had_cached_gradient ? alg.cached_gradient : nothing
        cached_y0 = cached_d0 = cached_g0 = cached_dg0 = NaN
    elseif had_cached_gradient && !modes.use_constant_batched_signed
        cached_y0, cached_d0 = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_, flow, grad_and_hvp, false, alg.cached_gradient;
            t_valid=0.0, t_invalid=0.0)
        cached_g0, cached_dg0 = if alg.bound === :linear
            rate_and_derivative(state_, flow, grad_and_hvp, alg.cached_gradient)
        else
            (NaN, NaN)
        end
        alg.has_cached_gradient[] = false
        alg.has_cached_rate_derivative[] = false
        cached_gradient = nothing
    else
        cached_y0, cached_d0 = NaN, NaN
        cached_g0, cached_dg0 = NaN, NaN
        cached_gradient = nothing
        if modes.use_constant_batched_signed
            _invalidate_cached_gradient!(alg)
        end
    end
    n_cells_bounded, built_area = _build_grid_bound_prefix!(pcb, state, flow, grad_and_hvp, alg, stats, state_, effective_horizon,
        cumulative_exp, probe_failure_handler, modes; cached_gradient, cached_y0, cached_d0, cached_g0, cached_dg0)
    _set_counter_grid_N_current(stats, alg.N[])

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
            τ_reflection, lb_reflection = if modes.use_linear
            propose_event_time(rng, alg.affine_bound, cumulative_exp)
        else
            propose_event_time(rng, pcb, cumulative_exp)
        end

        if τ_reflection >= effective_horizon

            τ, event_type, meta = _return_grid_horizon!(
                alg, stats, flow, alg.t_max[], effective_horizon, horizon_event, max_t_max, default_return)
            return time_offset + τ, event_type, meta
        end

        if τ_refresh < τ_reflection
            alg.has_cached_gradient[] = false
            alg.has_cached_rate_derivative[] = false
            _restore_lazy_after_grid!(alg, flow)
            return time_offset + τ_refresh, :refresh, default_return
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
                lb_reflection, cumulative_exp, λ_refresh, modes.use_linear)
            if modes.use_linear
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
            alg.has_cached_rate_derivative[] = false
            _restore_lazy_after_grid!(alg, flow)
            return time_offset + τ_reflection, :reflect, GradientMeta(∇ϕx)
        end

        # Rejection: cumulative_exp has advanced, next proposal will be at a later time
        rejection_count += 1
        last_rejected_time = τ_reflection
        cumulative_exp += rand(rng, Exponential())
        if cumulative_exp > built_area * (1 + 64eps(Float64)) + 64eps(Float64)
            start_cell = n_cells_bounded + 1
            effective_horizon, horizon_event = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)
            modes = _grid_bound_modes(alg, state, flow, grad_and_hvp)
            if modes.use_constant_batched_signed
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
            end
            n_cells_bounded, built_area = _build_grid_bound_prefix!(pcb, state, flow, grad_and_hvp, alg, stats, state_, effective_horizon,
                cumulative_exp, probe_failure_handler, modes; start_cell, initial_integral=built_area, append=true)
        end
        if rejection_count >= max_rejections && !alg.schedule_frozen[]
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
            alg.has_cached_rate_derivative[] = false

            remaining_horizon = effective_horizon - last_rejected_time
            if remaining_horizon <= 0
                return time_offset + last_rejected_time, horizon_event, default_return
            end

            tail_state = alg.state_cache2
            copyto!(tail_state, state)
            move_forward_time!(tail_state, last_rejected_time, flow)
            time_offset += last_rejected_time
            state = copy(tail_state)
            max_horizon = remaining_horizon
            include_refresh = false
            max_horizon_event = horizon_event
            tail_restart_count += 1
            if tail_restart_count > alg.safety_limit
                _throw_grid_safety_limit_error(state, flow, model;
                    t_invalid=remaining_horizon,
                    message="Tail restart limit reached in Grid algorithm")
            end
            @goto restart_from_tail
        end
        safety_limit -= 1
    end

    if isfinite(τ_refresh) && τ_refresh <= min(alg.t_max[], max_horizon)
        alg.has_cached_gradient[] = false
        alg.has_cached_rate_derivative[] = false
        _restore_lazy_after_grid!(alg, flow)
        return time_offset + τ_refresh, :refresh, default_return
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
        alg.has_cached_rate_derivative[] = false
    elseif alg.has_cached_rate_derivative[]
        _inc_counter_grid_cached_endpoint_reuses(stats)
        y_left = alg.cached_rate[]
        d_left = alg.cached_rate_derivative[]
        alg.has_cached_rate_derivative[] = false
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
    default_max_rejections = min(25, max(10, alg.safety_limit ÷ 2))
    max_rejections = alg.lazy_max_rejections > 0 ? alg.lazy_max_rejections : default_max_rejections
    low_tightness_rejections = 0
    low_tightness_threshold = alg.lazy_low_tightness_threshold
    max_low_tightness_rejections = alg.lazy_max_low_tightness_rejections
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
            return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event, max_t_max, default_return)
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
                _cache_endpoint_rate_derivative!(alg, y_right, d_right, horizon_event)
                return _return_grid_horizon!(alg, stats, flow, t_max, effective_horizon, horizon_event, max_t_max, default_return)
            end
            continue
        end

        # Event is in this interval.  If a proposal is rejected, keep using the
        # same constant cell bound for the remainder of the cell instead of
        # rebuilding a new tangent bound at the rejected time.
        while true
            lb_proposal = pos(Λ_cell)
            τ_proposal = t_left + (exp_target - cumulative_area) / lb_proposal
            if τ_proposal >= t_right || !isfinite(τ_proposal)
                no_event_cells += 1
                t_left = t_right
                y_left = y_right
                d_left = d_right
                cumulative_area = 0.0
                exp_target = rand(rng, Exponential())
                if t_right >= effective_horizon
                    _cache_endpoint_rate_derivative!(alg, y_right, d_right, horizon_event)
                    return _return_grid_horizon!(
                        alg, stats, flow, t_max, effective_horizon, horizon_event,
                        max_t_max, default_return)
                end
                break
            end

            if τ_refresh < τ_proposal
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                _invalidate_cached_gradient!(alg)
                return τ_refresh, :refresh, default_return
            end

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

            tightness = _safe_tightness(l_actual, lb_proposal)
            tightness_sum += tightness
            min_tightness = min(min_tightness, tightness)
            max_tightness = max(max_tightness, tightness)

            if l_actual > lb_proposal
                _inc_counter_lazy_fallback_bound_violation(stats)
                _inc_counter_grid_bound_violations(stats)
                if alg.bound_violation === :throw
                    throw(ErrorException("lazy constant GridThinning bound violated at proposal time"))
                elseif alg.bound_violation === :shrink
                    _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                    alg.lazy_enabled[] = false
                    alg.has_cached_gradient[] = false
                    alg.has_cached_rate_derivative[] = false
                    _shrink_grid_after_bound_violation!(alg, stats)
                    return _next_event_time_grid!(
                        rng, grad_and_hvp, model, flow, alg, state, cache, stats,
                        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
                end
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.lazy_enabled[] = false
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            if rand(rng) * lb_proposal <= l_actual
                copyto!(alg.cached_gradient, ∇ϕx)
                alg.has_cached_gradient[] = true
                alg.has_cached_rate_derivative[] = false

                _adapt_grid_N!(alg, tightness)
                _adapt_grid_t_max!(alg, τ_proposal, model.grad)
                _set_counter_grid_N_current(stats, alg.N[])
                alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return τ_proposal, :reflect, GradientMeta(∇ϕx)
            end

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
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            if proposal_rejections >= max_rejections
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                alg.lazy_enabled[] = false
                alg.has_cached_gradient[] = false
                alg.has_cached_rate_derivative[] = false
                _increase_grid_N!(alg)
                recompute_time_grid!(alg)
                return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            exp_target += rand(rng, Exponential())
        end
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
    alg.schedule_frozen[] && return nothing
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
    alg.schedule_frozen[] && return nothing
    N = alg.N[]
    new_N = min(alg.N_max, N + 4)
    if new_N != N
        alg.N[] = new_N
        recompute_time_grid!(alg)
    end
end

function _adapt_grid_t_max!(alg::GridAdaptiveState, τ_accepted::Float64, ::GradientStrategy)
    alg.schedule_frozen[] && return nothing
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
    alg.schedule_frozen[] && return nothing
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
    alg.schedule_frozen[] && return nothing
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
    !ispositive(max_rate) && return nothing
    alg.constant_bound_rate[] = max_rate * 2.0
    return nothing
end
