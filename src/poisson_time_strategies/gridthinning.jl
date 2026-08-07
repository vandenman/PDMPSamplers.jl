
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
and `:value_quadratic`. Passing neither `bound` keyword keeps the historical
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

_validate_grid_model(::ContinuousDynamics, ::PDMPModel) = nothing

_validate_marked_grid_envelope(::ContinuousDynamics, ::SeparableResidualEnvelope) = nothing
_validate_marked_grid_envelope(flow::PreconditionedDynamics,
        envelope::SeparableResidualEnvelope) =
    _validate_marked_grid_envelope(flow.dynamics, envelope)
function _validate_marked_grid_envelope(::AnyBoomerang,
        envelope::SeparableResidualEnvelope)
    envelope.component_cell_scales! === nothing && throw(ArgumentError(
        "MarkedControlVariate GridThinning requires an explicit certified " *
        "component_cell_scales! callback for Boomerang trajectories"))
    return nothing
end

_validate_grid_model(flow::ContinuousDynamics,
    model::PDMPModel{<:MarkedControlVariate}) =
        _validate_marked_grid_envelope(flow, model.grad.envelope)

function _to_internal(strat::GridThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    _validate_grid_model(flow, model)
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
function _effective_grid_horizon(::GradientStrategy, t_max::Float64, τ_refresh::Float64, max_horizon::Float64,
    max_horizon_event::Symbol=:horizon_hit)
    if τ_refresh <= t_max && τ_refresh <= max_horizon
        return τ_refresh, :refresh
    elseif max_horizon <= t_max
        return max_horizon, max_horizon_event
    end
    return t_max, :horizon_hit
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
    GridAdaptiveState(
        PiecewiseConstantBound(collect(range(0.0, strat.t_max, N_base + 1)), zeros(T, N_base)),
        PiecewiseAffineBound(2N_base),
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

struct GridAdaptiveState{S<:AbstractPDMPState,V<:AbstractVector,P} <: PoissonTimeStrategy
    pcb::PiecewiseConstantBound{Float64}
    affine_bound::PiecewiseAffineBound{Float64}
    marked_bound::PiecewiseAffineBound{Float64}
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
