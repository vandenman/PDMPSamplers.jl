_grid_hvp_provider(alg::GridAdaptiveState, model::PDMPModel) = GradHVPProvider(alg.grad_provider, model.hvp)
_grid_vhv_provider(alg::GridAdaptiveState, model::PDMPModel) = VHVProvider(alg.grad_provider, model.vhv, alg.fd_w_buf)
_grid_fd_provider(alg::GridAdaptiveState, stats) = FiniteDiffVHV(alg.grad_provider, alg.fd_buf, alg.fd_grad_buf, alg.fd_w_buf, stats)

function _grid_event_provider(model::PDMPModel, flow::ContinuousDynamics, alg::GridAdaptiveState, stats)
    backend = _selected_curvature_backend(alg.curvature_backend, model, flow)
    backend === :finite_difference && return _grid_fd_provider(alg, stats)
    joint_func = model.joint
    if joint_func !== nothing && _joint_compatible(flow)
        return joint_func
    end
    vhv_func = model.vhv
    if vhv_func !== nothing
        return _grid_vhv_provider(alg, model)
    end
    hvp_func = model.hvp
    hvp_func === nothing && throw(ArgumentError(
        "curvature_backend=:exact requires a compatible joint, VHV, or HVP provider"))
    return _grid_hvp_provider(alg, model)
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
)::GridEvent where {FL<:ContinuousDynamics}
    if alg.lazy_enabled[]
        return _next_event_time_lazy!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end
    return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
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
    max_horizon::Float64=Inf, include_refresh::Bool=true, max_horizon_event::Symbol=:horizon_hit)::GridEvent where {FL<:ContinuousDynamics}
    return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, NoGridBoundaryProbe())
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol,
    detect_boundaries::Bool)::GridEvent where {FL<:ContinuousDynamics}
    detect_boundaries || return next_event_time(rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event)
    probe_failure_handler = _grid_probe_failure_handler(state, flow, model, GridThinningStrategy)
    return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _next_event_time_with_probe(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    max_horizon::Float64, include_refresh::Bool, max_horizon_event::Symbol,
    probe_failure_handler::GridBoundaryProbe)::GridEvent where {FL<:ContinuousDynamics}
    if isfinite(alg.constant_bound_rate[])
        return _constant_bound_event_time(rng, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end

    if alg.bound in (:value_quadratic, :shared_node)
        return _next_event_time_value_quadratic!(
            rng, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end

    state_ = alg.state_cache
    copyto!(state_, state)

    provider = _grid_event_provider(model, flow, alg, stats)
    return _next_event_time_with_provider!(rng, provider, model, flow, alg, state, cache, stats, max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
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
