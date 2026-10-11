
"""
    GridThinningStrategy(; bound=:value_quadratic, kwargs...)

Adaptive GridThinning configuration.

`curvature_backend` controls directional curvature evaluation explicitly:
`:exact` requires a compatible joint, VHV, or HVP provider;
`:finite_difference` evaluates curvature from shifted gradients; and `:auto`
uses an exact provider when available, otherwise finite differences. Finite
differences approximate the directional derivative and therefore have
step-size/truncation error even when the underlying gradients are exact.

`bound` is `:value_quadratic` (default) or `:constant`. `:value_quadratic`
needs a `curvature_bound`; without one, the cells use the ordinary GridThinning
construction.

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
    curvature_backend::Symbol
    lazy::Bool
    bound::Symbol
    curvature_bound
    bound_violation::Symbol
    max_rejections_before_tail_restart::Int
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

_normalize_grid_bound(::Nothing) = :value_quadratic

function _normalize_curvature_backend(curvature_backend)
    if curvature_backend === nothing
        return :auto
    end
    backend = Symbol(curvature_backend)
    backend in (:exact, :finite_difference, :auto) || throw(ArgumentError(
        "curvature_backend must be :exact, :finite_difference, or :auto"))
    return backend
end

function _normalize_grid_bound(bound::Symbol)
    bound in (:constant, :value_quadratic) || throw(ArgumentError(
        "GridThinning bound must be :value_quadratic or :constant; got $(bound)"))
    return bound
end

function GridThinningStrategy(; N::Int=20, N_min::Int=5, t_max::Real=2.0, α⁺::Real=1.5, α⁻::Real=0.5,
    safety_limit::Int=500,
    curvature_backend=nothing,
    lazy::Bool=true, bound=nothing, curvature_bound=nothing, bound_violation=nothing,
    max_rejections_before_tail_restart::Int=100,
    lazy_low_tightness_threshold::Real=0.1, lazy_max_low_tightness_rejections::Int=3,
    lazy_max_rejections::Int=0, allow_small_boomerang::Bool=false, warmup_tuning=nothing)
    bound_symbol = _normalize_grid_bound(bound)
    bound_violation_symbol = bound_violation === nothing ?
        (bound_symbol === :constant ? :count : :shrink) : Symbol(bound_violation)
    lazy_low_tightness_threshold >= 0 ||
        throw(ArgumentError("lazy_low_tightness_threshold must be nonnegative"))
    lazy_max_low_tightness_rejections > 0 ||
        throw(ArgumentError("lazy_max_low_tightness_rejections must be positive"))
    lazy_max_rejections >= 0 ||
        throw(ArgumentError("lazy_max_rejections must be nonnegative; use 0 for the default derived cap"))
    backend = _normalize_curvature_backend(curvature_backend)
    return GridThinningStrategy(
        N, N_min, Float64(t_max), Float64(α⁺), Float64(α⁻), safety_limit,
        backend,
        lazy, bound_symbol, curvature_bound, bound_violation_symbol,
        max_rejections_before_tail_restart,
        Float64(lazy_low_tightness_threshold), lazy_max_low_tightness_rejections,
        lazy_max_rejections, allow_small_boomerang, warmup_tuning)
end

function Base.show(io::IO, strat::GridThinningStrategy)
    print(io, "GridThinningStrategy(")
    print(io, "N=", strat.N, ", N_min=", strat.N_min, ", t_max=", strat.t_max)
    print(io, ", bound=", strat.bound)
    strat.bound === :value_quadratic && print(io, ", curvature_bound=", strat.curvature_bound)
    print(io, ")")
end


function _grid_min_cells(strat::GridThinningStrategy, flow::ContinuousDynamics, N_base::Int)
    strat.bound === :value_quadratic && return strat.N_min
    if flow isa AnyBoomerang && strat.allow_small_boomerang
        return strat.N_min
    end
    return min_grid_cells(flow, strat.N_min, N_base)
end

_validate_grid_model(::ContinuousDynamics, ::PDMPModel) = nothing

"""
    validate_subsampling_grid_envelope(flow, envelope)

Throw an `ArgumentError` if `envelope` cannot certify GridThinning cells for
`flow`; return `nothing` otherwise. Called when a subsampled model is set up
for GridThinning. Envelope types extend it to state their requirements. Part
of the extension interface.
"""
validate_subsampling_grid_envelope(::ContinuousDynamics, ::AbstractResidualEnvelope) = nothing
validate_subsampling_grid_envelope(flow::PreconditionedDynamics,
        envelope::AbstractResidualEnvelope) =
    validate_subsampling_grid_envelope(flow.dynamics, envelope)
function validate_subsampling_grid_envelope(::AnyBoomerang,
        envelope::AbstractResidualEnvelope)
    envelope.component_cell_scales! === nothing && throw(ArgumentError(
        "SubsampledControlVariate GridThinning requires an explicit certified " *
        "component_cell_scales! callback for Boomerang trajectories"))
    return nothing
end

_validate_grid_model(flow::ContinuousDynamics,
    model::PDMPModel{<:SubsampledControlVariate}) =
        validate_subsampling_grid_envelope(flow, model.grad.envelope)

function _to_internal(strat::GridThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    _validate_grid_model(flow, model)
    T = typeof(strat.t_max)
    strat.max_rejections_before_tail_restart > 0 || throw(ArgumentError("max_rejections_before_tail_restart must be positive"))
    # Derivative info is always available: either via HVP, VHV, joint, or FD fallback.
    N_base = strat.N
    N_min = _grid_min_cells(strat, flow, N_base)
    N_max = max(N_base + 4, 2 * N_base)
    alg = _build_grid_adaptive_state(strat, state, flow, model, cache, N_base, N_min, N_max, stats)
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

function _build_grid_adaptive_state(strat::GridThinningStrategy, state::S, flow::ContinuousDynamics, model::PDMPModel, cache, N_base::Int, N_min::Int, N_max::Int, stats=nothing) where S<:AbstractPDMPState
    T = typeof(strat.t_max)
    state_cache = copy(state)
    state_cache2 = copy(state)
    grad_provider = GradientProvider(state_cache.ξ.θ, flow, model.grad, cache)
    fd_buf = similar(state.ξ.x)
    fd_grad_buf = similar(state.ξ.x)
    fd_w_buf = similar(state.ξ.x)
    exact_provider = if model.joint !== nothing && _joint_compatible(flow)
        model.joint
    elseif model.vhv !== nothing
        VHVProvider(grad_provider, model.vhv, fd_w_buf)
    elseif model.hvp !== nothing
        GradHVPProvider(grad_provider, model.hvp)
    else
        nothing
    end
    pcb = PiecewiseConstantBound(
        collect(range(0.0, strat.t_max, N_base + 1)), zeros(T, N_base))
    sizehint!(pcb.t_grid, N_max + 1)
    sizehint!(pcb.Λ_vals, N_max)
    sizehint!(pcb.y_vals, N_max + 1)
    sizehint!(pcb.d_vals, N_max + 1)
    empty_gradient = similar(state.ξ.x, 0)
    GridAdaptiveState(
        pcb,
        PiecewiseAffineBound(2N_max),
        Base.RefValue{Int}(N_base),
        Base.RefValue{Float64}(strat.t_max),
        strat.α⁺,
        strat.α⁻,
        strat.safety_limit,
        N_min,
        N_max,
        Ref(false),
        state_cache,
        state_cache2,
        GradientMeta(empty_gradient),
        strat.curvature_backend,
        fd_buf,
        fd_grad_buf,
        fd_w_buf,
        Float64[],
        Float64[],
        strat.lazy,
        Ref(strat.lazy),
        similar(state.ξ.x),
        Ref(false),
        Ref(NaN),
        Ref(NaN),
        Ref(false),
        grad_provider,
        exact_provider,
        strat.bound,
        strat.curvature_bound,
        strat.bound_violation,
        strat.max_rejections_before_tail_restart,
        strat.lazy_low_tightness_threshold,
        strat.lazy_max_low_tightness_rejections,
        strat.lazy_max_rejections,
        strat.warmup_tuning,
    )
end

struct GridAdaptiveState{S<:AbstractPDMPState,P,Q,B} <: PoissonTimeStrategy
    pcb::PiecewiseConstantBound{Float64}
    subsampling_bound::PiecewiseAffineBound{Float64}
    N::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    N_min::Int
    N_max::Int
    schedule_frozen::Base.RefValue{Bool}
    state_cache::S
    state_cache2::S
    empty_gradient_meta::GradientMeta
    curvature_backend::Symbol
    fd_buf::Vector{Float64}
    fd_grad_buf::Vector{Float64}
    fd_w_buf::Vector{Float64}
    rate_value_buf::Vector{Float64}
    rate_derivative_buf::Vector{Float64}
    lazy::Bool
    lazy_enabled::Base.RefValue{Bool}
    cached_gradient::Vector{Float64}
    has_cached_gradient::Base.RefValue{Bool}
    cached_rate::Base.RefValue{Float64}
    cached_rate_derivative::Base.RefValue{Float64}
    has_cached_rate_derivative::Base.RefValue{Bool}
    grad_provider::P
    exact_provider::Q
    bound::Symbol
    curvature_bound::B
    bound_violation::Symbol
    max_rejections_before_tail_restart::Int
    lazy_low_tightness_threshold::Float64
    lazy_max_low_tightness_rejections::Int
    lazy_max_rejections::Int
    warmup_tuning
end

accept_reflection_event(::Random.AbstractRNG, ::GridAdaptiveState, args...) = true
accept_reflection_event(::GridAdaptiveState, args...) = true

recompute_time_grid!(alg::GridAdaptiveState) = recompute_time_grid!(alg.pcb, alg.t_max[], alg.N[])
