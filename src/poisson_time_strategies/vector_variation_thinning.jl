"""
    VectorVariationThinningStrategy(; kwargs...)

EXPERIMENTAL, intentionally unexported and still in development.

Experimental ZigZag/PreconditionedZigZag event search that uses the full
signed channel vector returned by each gradient evaluation.  This first
implementation uses linear channel interpolation with midpoint refinement and
falls back to `GridThinningStrategy` when the local vector model is unreliable.
"""
struct VectorVariationThinningStrategy <: PoissonTimeStrategy
    N::Int
    N_min::Int
    t_max::Float64
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    max_refinement_depth::Int
    validation_rtol::Float64
    validation_atol::Float64
    min_cell_width::Float64
    max_skip_width::Float64
    use_derivative_hermite::Bool
    derivative_hermite_on_demand::Bool
    derivative_hermite_trigger_scale::Float64
    fallback::GridThinningStrategy
end

function VectorVariationThinningStrategy(; N::Int=20, N_min::Int=5, t_max::Real=2.0,
    α⁺::Real=1.5, α⁻::Real=0.5, safety_limit::Int=500, max_refinement_depth::Int=8,
    validation_rtol::Real=0.05, validation_atol::Real=1e-8, min_cell_width::Real=1e-8,
    max_skip_width::Real=0.25, use_derivative_hermite::Bool=false,
    derivative_hermite_on_demand::Bool=false, derivative_hermite_trigger_scale::Real=10.0,
    fallback::GridThinningStrategy=GridThinningStrategy(; N, N_min, t_max, α⁺, α⁻, safety_limit,
        use_fd_hvp=true, bound=:flat))

    N > 0 || throw(ArgumentError("N must be positive"))
    N_min > 0 || throw(ArgumentError("N_min must be positive"))
    t_max > 0 || throw(ArgumentError("t_max must be positive"))
    max_refinement_depth >= 0 || throw(ArgumentError("max_refinement_depth must be nonnegative"))
    validation_rtol >= 0 || throw(ArgumentError("validation_rtol must be nonnegative"))
    validation_atol >= 0 || throw(ArgumentError("validation_atol must be nonnegative"))
    min_cell_width > 0 || throw(ArgumentError("min_cell_width must be positive"))
    max_skip_width > 0 || throw(ArgumentError("max_skip_width must be positive"))
    derivative_hermite_trigger_scale >= 0 ||
        throw(ArgumentError("derivative_hermite_trigger_scale must be nonnegative"))
    return VectorVariationThinningStrategy(
        N, N_min, Float64(t_max), Float64(α⁺), Float64(α⁻), safety_limit,
        max_refinement_depth, Float64(validation_rtol), Float64(validation_atol),
        Float64(min_cell_width), Float64(max_skip_width), use_derivative_hermite,
        derivative_hermite_on_demand, Float64(derivative_hermite_trigger_scale), fallback)
end

function Base.show(io::IO, strat::VectorVariationThinningStrategy)
    print(io, "VectorVariationThinningStrategy(")
    print(io, "N=", strat.N, ", t_max=", strat.t_max)
    print(io, ", max_refinement_depth=", strat.max_refinement_depth)
    print(io, ")")
end

struct VectorVariationAdaptiveState{S<:AbstractPDMPState,V<:AbstractVector,F<:GridAdaptiveState} <: PoissonTimeStrategy
    N::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    N_min::Int
    validation_rtol::Float64
    validation_atol::Float64
    state_cache::S
    state_cache2::S
    empty_∇ϕx::V
    cached_gradient::V
    has_cached_gradient::Base.RefValue{Bool}
    cached_U::Base.RefValue{Float64}
    signed_left::V
    signed_right::V
    signed_candidate::V
    fallback::F
end

function _to_internal(strat::VectorVariationThinningStrategy, rng::Random.AbstractRNG,
    flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter)

    fallback = _to_internal(strat.fallback, rng, flow, model, state, cache, stats)
    state_cache = copy(state)
    state_cache2 = copy(state)
    return VectorVariationAdaptiveState(
        Ref(strat.N), Ref(strat.t_max), strat.α⁺, strat.α⁻, strat.safety_limit,
        min_grid_cells(flow, strat.N_min, strat.N),
        strat.validation_rtol, strat.validation_atol,
        state_cache, state_cache2, similar(state.ξ.x, 0), similar(state.ξ.x), Ref(false),
        Ref(NaN), similar(state.ξ.x), similar(state.ξ.x), similar(state.ξ.x),
        fallback)
end

accept_reflection_event(::Random.AbstractRNG, ::VectorVariationAdaptiveState, args...) = true
accept_reflection_event(::VectorVariationAdaptiveState, args...) = true

function reset_grid_scale!(alg::VectorVariationAdaptiveState, t_max::Float64=2.0)
    alg.t_max[] = t_max
    alg.N[] = max(alg.N[], alg.N_min)
    alg.has_cached_gradient[] = false
    alg.cached_U[] = NaN
    reset_grid_scale!(alg.fallback, t_max)
    return nothing
end

function _vv_shrink_grid_N!(alg::VectorVariationAdaptiveState)
    if alg.N[] > alg.N_min
        alg.N[] = max(alg.N_min, alg.N[] - 1)
    end
    return nothing
end

@inline function _vv_tol(alg::VectorVariationAdaptiveState, scale::Float64)
    return alg.validation_atol + alg.validation_rtol * max(scale, 1.0)
end

function _vv_signed_channels!(out::AbstractVector, state::AbstractPDMPState,
    grad::AbstractVector, ::ContinuousDynamics, cache)

    θ = state.ξ.θ
    @inbounds for i in eachindex(out, θ, grad)
        out[i] = θ[i] * grad[i]
    end
    return out
end

function _vv_signed_channels!(out::AbstractVector, state::AbstractPDMPState,
    grad::AbstractVector, flow::DensePreconditionedZigZag, cache)

    mul!(out, transpose(flow.metric.L), grad)
    v = flow.metric.v_canonical
    @inbounds for i in eachindex(out, v)
        out[i] *= v[i]
    end
    return out
end

function _vv_linear_positive_area(fa::AbstractVector, fb::AbstractVector, h::Float64)
    h <= 0.0 && return 0.0
    total = 0.0
    @inbounds for i in eachindex(fa, fb)
        total += _linear_positive_area(Float64(fa[i]), Float64(fb[i]), h)
    end
    return total
end

function _vv_linear_positive_value(fa::AbstractVector, fb::AbstractVector, h::Float64, t::Float64)
    h <= 0.0 && return 0.0
    u = t / h
    total = 0.0
    @inbounds for i in eachindex(fa, fb)
        total += pos(Float64(fa[i]) + (Float64(fb[i]) - Float64(fa[i])) * u)
    end
    return total
end

function _vv_linear_positive_area_time(fa::AbstractVector, fb::AbstractVector, h::Float64, area::Float64)
    area <= 0.0 && return 0.0
    total = _vv_linear_positive_area(fa, fb, h)
    area >= total && return h
    lo = 0.0
    hi = h
    for _ in 1:40
        mid = 0.5 * (lo + hi)
        amid = 0.0
        @inbounds for i in eachindex(fa, fb)
            fmid = Float64(fa[i]) + (Float64(fb[i]) - Float64(fa[i])) * (mid / h)
            amid += _linear_positive_area(Float64(fa[i]), fmid, mid)
        end
        if amid < area
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function _vv_observe_endpoint!(out::AbstractVector, alg::VectorVariationAdaptiveState,
    model::PDMPModel, flow::ContinuousDynamics, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, t::Float64, probe_failure_handler::GridBoundaryProbe)

    state_t = alg.state_cache
    copyto!(state_t, state)
    t != 0.0 && move_forward_time!(state_t, t, flow)
    _inc_counter_grid_endpoint_evaluations(stats)
    _inc_counter_grid_endpoint_gradient_calls(stats)
    grad = _compute_grid_gradient_or_throw!(
        state_t, state, flow, model, cache, max(0.0, prevfloat(t)), t, probe_failure_handler)
    _inc_counter_grid_points_evaluated(stats, 1)
    _vv_signed_channels!(out, state_t, grad, flow, cache)
    U = _last_gradient_potential(model)
    return Float64(t), U === nothing ? NaN : Float64(U)
end

function _vv_observe_candidate!(out::AbstractVector, alg::VectorVariationAdaptiveState,
    model::PDMPModel, flow::ContinuousDynamics, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, t::Float64, probe_failure_handler::GridBoundaryProbe)

    state_t = alg.state_cache2
    copyto!(state_t, state)
    t != 0.0 && move_forward_time!(state_t, t, flow)
    _inc_counter_grid_acceptance_gradient_calls(stats)
    grad = _compute_grid_gradient_or_throw!(
        state_t, state, flow, model, cache, max(0.0, prevfloat(t)), t, probe_failure_handler)
    _inc_counter_grid_acceptance_tests(stats)
    _vv_signed_channels!(out, state_t, grad, flow, cache)
    U = _last_gradient_potential(model)
    return Float64(t), U === nothing ? NaN : Float64(U), grad
end

function _vv_fallback!(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy},
    flow::ContinuousDynamics, alg::VectorVariationAdaptiveState, state::AbstractPDMPState,
    cache, stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, probe_failure_handler::GridBoundaryProbe)

    alg.has_cached_gradient[] = false
    alg.cached_U[] = NaN
    return _next_event_time_with_probe(rng, model, flow, alg.fallback, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy},
    flow::FL, alg::VectorVariationAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64=Inf, include_refresh::Bool=true,
    max_horizon_event::Symbol=:horizon_hit) where {FL<:ContinuousDynamics}

    return _next_vector_variation_event_time!(
        rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh,
        max_horizon_event, NoGridBoundaryProbe())
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy},
    flow::FL, alg::VectorVariationAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, detect_boundaries::Bool) where {FL<:ContinuousDynamics}

    detect_boundaries || return next_event_time(
        rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh,
        max_horizon_event)
    probe_failure_handler = _grid_probe_failure_handler(state, flow, model, VectorVariationThinningStrategy)
    return _next_vector_variation_event_time!(
        rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh,
        max_horizon_event, probe_failure_handler)
end

function _next_vector_variation_event_time!(rng::Random.AbstractRNG,
    model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
    alg::VectorVariationAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, probe_failure_handler::GridBoundaryProbe)

    return _vv_fallback!(rng, model, flow, alg, state, cache, stats, max_horizon,
        include_refresh, max_horizon_event, probe_failure_handler)
end

_vv_supported_flow(::ZigZag) = true
_vv_supported_flow(::PreconditionedDynamics{<:AbstractPreconditioner,<:ZigZag}) = true
_vv_supported_flow(::ContinuousDynamics) = false

function _next_vector_variation_event_time!(rng::Random.AbstractRNG,
    model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::VectorVariationAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, probe_failure_handler::GridBoundaryProbe) where {FL<:Union{ZigZag,PreconditionedDynamics}}

    _vv_supported_flow(flow) || return _vv_fallback!(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
    hard_horizon = min(max_horizon, max_grid_horizon(flow))
    search_horizon = min(hard_horizon, τ_refresh)
    horizon_event = if τ_refresh <= hard_horizon
        :refresh
    else
        max_horizon <= max_grid_horizon(flow) ? max_horizon_event : :horizon_hit
    end
    default_return = GradientMeta(alg.empty_∇ϕx)
    if search_horizon <= 0.0
        return 0.0, horizon_event, default_return
    end

    _inc_counter_grid_builds(stats)
    _record_grid_schedule!(stats, alg)
    _set_counter_grid_N_current(stats, alg.N[])

    exp_target = Random.randexp(rng)
    cumulative_area = 0.0
    proposal_attempts = 0
    proposal_rejections = 0
    h = alg.t_max[] / alg.N[]
    t_left, U_left = if alg.has_cached_gradient[]
        _inc_counter_grid_cached_endpoint_reuses(stats)
        alg.has_cached_gradient[] = false
        _vv_signed_channels!(alg.signed_left, state, alg.cached_gradient, flow, cache)
        0.0, alg.cached_U[]
    else
        _vv_observe_endpoint!(alg.signed_left, alg, model, flow, state, cache, stats, 0.0, probe_failure_handler)
    end

    safety = alg.safety_limit
    while safety > 0
        safety -= 1
        t_right = min(t_left + h, search_horizon)
        if t_right <= t_left
            alg.has_cached_gradient[] = false
            alg.cached_U[] = NaN
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            return search_horizon, horizon_event, default_return
        end
        _, U_right = _vv_observe_endpoint!(alg.signed_right, alg, model, flow, state, cache, stats, t_right, probe_failure_handler)
        _inc_counter_positive_variation_cells(stats)
        h_cell = t_right - t_left
        area_cell = _vv_linear_positive_area(alg.signed_left, alg.signed_right, h_cell)

        if cumulative_area + area_cell < exp_target
            _inc_counter_positive_variation_skipped_cells(stats, 1)
            cumulative_area += area_cell
            t_left = t_right
            U_left = U_right
            copyto!(alg.signed_left, alg.signed_right)
            if t_left >= search_horizon
                alg.has_cached_gradient[] = false
                alg.cached_U[] = NaN
                _vv_shrink_grid_N!(alg)
                _set_counter_grid_N_current(stats, alg.N[])
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return search_horizon, horizon_event, default_return
            end
            if area_cell <= alg.validation_atol
                h = min(h * alg.α⁺, alg.t_max[])
            end
            continue
        end

        while true
            area_target = exp_target - cumulative_area
            if area_target <= 0.0
                area_target = eps(Float64)
            end
            τ_local = _vv_linear_positive_area_time(alg.signed_left, alg.signed_right, h_cell, area_target)
            τ_proposal = t_left + τ_local
            if τ_proposal >= t_right || !isfinite(τ_proposal)
                cumulative_area += area_cell
                break
            end
            if τ_refresh < τ_proposal
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return τ_refresh, :refresh, default_return
            end

            _, Uτ, gradτ = _vv_observe_candidate!(
                alg.signed_candidate, alg, model, flow, state, cache, stats, τ_proposal, probe_failure_handler)
            λ_env = _vv_linear_positive_value(alg.signed_left, alg.signed_right, h_cell, τ_local)
            λ_actual = sum(pos, alg.signed_candidate)
            proposal_attempts += 1

            tol = _vv_tol(alg, max(λ_env, λ_actual))
            if λ_actual > λ_env + tol
                _inc_counter_grid_bound_violations(stats)
                _inc_counter_positive_variation_fallbacks(stats)
                alg.has_cached_gradient[] = false
                alg.cached_U[] = NaN
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return _vv_fallback!(rng, model, flow, alg, state, cache, stats,
                    max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            if λ_env > 0.0 && rand(rng) * λ_env <= λ_actual
                copyto!(alg.cached_gradient, gradτ)
                alg.cached_U[] = Uτ
                alg.has_cached_gradient[] = true
                tightness = λ_env <= 0.0 ? 0.0 : λ_actual / λ_env
                _adapt_grid_N!(alg.fallback, tightness)
                alg.N[] = max(alg.N_min, alg.fallback.N[])
                _vv_shrink_grid_N!(alg)
                alg.t_max[] = min(max_grid_horizon(flow), max(alg.t_max[] * alg.α⁻, τ_proposal * alg.α⁺))
                _set_counter_grid_N_current(stats, alg.N[])
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                _inc_counter_positive_variation_accepts(stats)
                return τ_proposal, :reflect, GradientMeta(alg.cached_gradient)
            end

            proposal_rejections += 1
            exp_target += Random.randexp(rng)
            if cumulative_area + area_cell < exp_target
                _inc_counter_positive_variation_skipped_cells(stats, 1)
                cumulative_area += area_cell
                break
            end
        end

        t_left = t_right
        U_left = U_right
        copyto!(alg.signed_left, alg.signed_right)
        if t_left >= search_horizon
            alg.has_cached_gradient[] = false
            alg.cached_U[] = NaN
            _vv_shrink_grid_N!(alg)
            _set_counter_grid_N_current(stats, alg.N[])
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            return search_horizon, horizon_event, default_return
        else
            continue
        end
    end

    _inc_counter_positive_variation_fallbacks(stats)
    _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
    return _vv_fallback!(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end
