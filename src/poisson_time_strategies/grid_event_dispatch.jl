_grid_hvp_provider(alg::GridAdaptiveState, model::PDMPModel) = GradHVPProvider(alg.grad_provider, model.hvp)
_grid_vhv_provider(alg::GridAdaptiveState, model::PDMPModel) = VHVProvider(alg.grad_provider, model.vhv, alg.fd_w_buf)
_grid_fd_provider(alg::GridAdaptiveState, stats) = FiniteDiffVHV(alg.grad_provider, alg.fd_buf, alg.fd_grad_buf, alg.fd_w_buf, stats)

function _grid_event_provider(model::PDMPModel, flow::ContinuousDynamics, alg::GridAdaptiveState, stats)
    backend = _selected_curvature_backend(alg.curvature_backend, model, flow)
    backend === :finite_difference && return _grid_fd_provider(alg, stats)
    alg.exact_provider === nothing && throw(ArgumentError(
        "curvature_backend=:exact requires a compatible joint, VHV, or HVP provider"))
    return alg.exact_provider
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
    # A provider-owned interval certificate is already the complete cell
    # construction.  Sending it through the derivative-based lazy builder
    # silently discards that certificate (and was the normal full-gradient
    # OMRF path).  Lazy endpoint construction is only applicable when no
    # direct certificate is available.
    if alg.lazy_enabled[] &&
            !has_direct_deterministic_rate_cell_bound(grad_and_hvp)
        return _next_event_time_lazy!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
    end
    return _next_event_time_grid!(rng, grad_and_hvp, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _next_event_time_resolved_provider!(rng::Random.AbstractRNG,
    provider, model::PDMPModel{<:GlobalGradientStrategy}, flow::FL,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64,
    include_refresh::Bool, max_horizon_event::Symbol,
    probe_failure_handler::GridBoundaryProbe)::GridEvent where {FL<:ContinuousDynamics}
    # This function barrier is important: `GridAdaptiveState.exact_provider` is
    # intentionally type-erased, whereas the direct-certificate trait must be
    # resolved on the concrete provider without boxing in the proposal loop.
    if has_direct_deterministic_rate_cell_bound(provider)
        return _next_event_time_with_provider!(rng, provider, model, flow, alg,
            state, cache, stats, max_horizon, include_refresh,
            max_horizon_event, probe_failure_handler)
    end
    if alg.bound in (:value_quadratic, :shared_node)
        return _next_event_time_value_quadratic!(
            rng, model, flow, alg, state, cache, stats,
            max_horizon, include_refresh, max_horizon_event,
            probe_failure_handler)
    end
    state_ = alg.state_cache
    copyto!(state_, state)
    return _next_event_time_with_provider!(rng, provider, model, flow, alg,
        state, cache, stats, max_horizon, include_refresh,
        max_horizon_event, probe_failure_handler)
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


    # A provider-owned direct interval certificate supersedes derivative-based
    # value-quadratic construction for this event search.  Probe only the first
    # deterministic cell; `_grid_bound_modes` repeats the call and carries the
    # certified value into the ordinary grid builder.
    provider = _grid_event_provider(model, flow, alg, stats)
    return _next_event_time_resolved_provider!(rng, provider, model, flow, alg,
        state, cache, stats, max_horizon, include_refresh, max_horizon_event,
        probe_failure_handler)
end

function _grid_bound_modes(alg::GridAdaptiveState, state::AbstractPDMPState,
        flow::ContinuousDynamics, provider, stats::Union{Nothing,AbstractStatisticCounter}=nothing)
    use_direct = has_direct_deterministic_rate_cell_bound(provider)
    direct_started = use_direct && stats !== nothing ? time_ns() : UInt64(0)
    direct_first = use_direct ? deterministic_rate_cell_bound(provider, state,
        flow, alg.pcb.t_grid[1], alg.pcb.t_grid[2]) : nothing
    use_direct && stats !== nothing && _inc_counter_grid_bound_seconds(
        stats, (time_ns() - direct_started) * 1.0e-9)
    use_linear = !use_direct && _use_linear_bound(alg, state, flow, provider)
    # Subsampled GridThinning also reaches this dispatcher.  Its historical
    # value-quadratic path accidentally fell through to the endpoint-tangent
    # construction, which ignores `curvature_bound`.  Use the same signed-rate
    # flat cell construction as the global value-quadratic path for scalar-rate
    # flows.  Correctness still requires the caller-supplied curvature value or
    # callback to be a genuine uniform cell certificate.
    use_value_quadratic = alg.bound === :value_quadratic &&
        _can_use_value_quadratic_grid(flow, alg.curvature_bound) &&
        _can_use_signed_grid_bound(alg, state, flow, provider)
    use_single_pass_signed = !use_direct &&
        (_use_signed_grid_bound(alg, state, flow, provider) || use_value_quadratic)
    return (;
        use_direct,
        direct_first,
        use_linear,
        use_single_pass_signed,
        use_constant_batched_signed=!use_single_pass_signed && _supports_constant_grid_rate_derivatives(flow, provider),
    )
end

function _build_grid_bound_prefix!(pcb::PiecewiseConstantBound, state::AbstractPDMPState, flow::ContinuousDynamics, provider, alg::GridAdaptiveState,
    stats::AbstractStatisticCounter, state_cache::AbstractPDMPState, effective_horizon::Float64, cumulative_exp::Float64,
    probe_failure_handler::GridBoundaryProbe, modes; cached_gradient=nothing, cached_y0::Float64=NaN, cached_d0::Float64=NaN,
    cached_g0::Float64=NaN, cached_dg0::Float64=NaN, start_cell::Integer=1, initial_integral::Float64=0.0, append::Bool=false)

    _inc_counter_grid_bound_evaluations(stats)
    _grid_t0 = time_ns()
    if modes.use_direct
        cumulative = initial_integral
        n_cells_bounded = start_cell - 1
        for cell in start_cell:_grid_cell_count(
                pcb.t_grid, length(pcb.Λ_vals), effective_horizon)
            left = pcb.t_grid[cell]
            right = min(pcb.t_grid[cell + 1], effective_horizon)
            value = cell == 1 ? modes.direct_first :
                deterministic_rate_cell_bound(provider, state, flow, left, right)
            value === nothing && throw(ArgumentError(
                "direct deterministic cell certificate disappeared within one grid"))
            isfinite(value) && value >= 0 || throw(DomainError(value,
                "direct deterministic cell certificate must be finite and nonnegative"))
            pcb.Λ_vals[cell] = nextfloat(Float64(value))
            pcb.y_vals[cell] = NaN
            pcb.d_vals[cell] = NaN
            cumulative += pcb.Λ_vals[cell] * (right - left)
            n_cells_bounded = cell
            cumulative >= cumulative_exp && break
        end
        if n_cells_bounded < length(pcb.Λ_vals)
            for cell in (n_cells_bounded + 1):length(pcb.Λ_vals)
                pcb.Λ_vals[cell] = 0.0
            end
        end
        _inc_counter_grid_builds(stats)
        _inc_counter_grid_points_evaluated(stats,
            max(0, n_cells_bounded - start_cell + 1))
    elseif modes.use_single_pass_signed
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
    _inc_counter_grid_bound_seconds(stats, (time_ns() - _grid_t0) * 1.0e-9)
    return n_cells_bounded, built_area
end
