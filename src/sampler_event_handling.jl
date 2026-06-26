function _check_sticky_times!(alg::PoissonTimeStrategy, state::AbstractPDMPState)
    return nothing
end

function _check_sticky_times!(alg::StickyLoopState, state::AbstractPDMPState)
    all(>=(state.t[]), alg.sticky_times) || error("some sticky_times are negative!")
    return nothing
end

function _compute_exact_reflection_gradient!(
    state::AbstractPDMPState,
    gradient_strategy::GlobalGradientStrategy,
    flow::ContinuousDynamics,
    cache,
    alg::PoissonTimeStrategy,
    τ::Real,
    wrap_boundary::Bool,
)
    if !wrap_boundary
        return compute_gradient!(state, gradient_strategy, flow, cache)
    end

    return try
        compute_gradient!(state, gradient_strategy, flow, cache)
    catch err
        ctx = _boundary_context_after_forward_move(state, flow, alg, τ, err)
        ctx === nothing && rethrow()
        throw(_ProbeFailureException(ctx))
    end
end

function _compute_reflection_gradient!(
    state::AbstractPDMPState,
    gradient_strategy::GlobalGradientStrategy,
    flow::ContinuousDynamics,
    cache,
    meta,
    alg::PoissonTimeStrategy,
    τ::Real,
    wrap_boundary::Bool,
)
    if meta isa GradientMeta && length(meta.∇ϕx) == length(state.ξ.x)
        return meta.∇ϕx
    end
    if !wrap_boundary
        return compute_gradient_for_reflection!(state, gradient_strategy, flow, cache)
    end

    return try
        compute_gradient_for_reflection!(state, gradient_strategy, flow, cache)
    catch err
        ctx = _boundary_context_after_forward_move(state, flow, alg, τ, err)
        ctx === nothing && rethrow()
        throw(_ProbeFailureException(ctx))
    end
end

function _handle_global_event_impl!(
    rng::Random.AbstractRNG,
    τ::Real,
    gradient_strategy::GlobalGradientStrategy,
    flow::ContinuousDynamics,
    alg::PoissonTimeStrategy,
    state::AbstractPDMPState,
    cache,
    event_type::Symbol,
    meta,
    stats::StatisticCounter,
    wrap_boundary::Bool,
)
    move_forward_time!(state, τ, flow)
    validate_state(state, flow, "after moving forward in time")

    needs_saving = false
    saving_args = nothing
    stats.last_rejected = false

    if event_type == :reflect
        stats.reflections_events += 1

        if alg isa ExactStrategy
            stats.reflections_accepted += 1

            if flow isa ZigZag
                i = meta.i
                reflect!(state.ξ, zero(eltype(cache.∇ϕx)), i, flow)
            else
                ∇ϕx = _compute_exact_reflection_gradient!(state, gradient_strategy, flow, cache, alg, τ, wrap_boundary)
                saving_args = reflect!(rng, state, ∇ϕx, flow, cache)
            end
            needs_saving = true
            (alg isa StickyLoopState && state isa StickyPDMPState) && _update_sticky_schedule_after_reflect!(rng, alg, state, flow, meta)
        else
            ∇ϕx = _compute_reflection_gradient!(state, gradient_strategy, flow, cache, meta, alg, τ, wrap_boundary)

            if accept_reflection_event(rng, alg, state.ξ, ∇ϕx, flow, τ, cache, meta)
                stats.reflections_accepted += 1
                saving_args = reflect!(rng, state, ∇ϕx, flow, cache)
                needs_saving = true
                (alg isa StickyLoopState && state isa StickyPDMPState) && _update_sticky_schedule_after_reflect!(rng, alg, state, flow, saving_args)
            else
                stats.last_rejected = true
            end
        end

    elseif event_type == :refresh
        refresh_velocity!(rng, state, flow)
        needs_saving = true
        stats.refreshment_events += 1
        (alg isa StickyLoopState && state isa StickyPDMPState) && _update_sticky_schedule_after_refresh!(rng, alg, state, flow)

    elseif event_type == :sticky
        stats.sticky_events += 1
        i = meta.i
        stick_or_unstick!(rng, state::StickyPDMPState, flow, alg, i)
        set_active_set!(gradient_strategy, state.free)
        validate_state(state, flow, "after stick_or_unstick!")
        needs_saving = true
        if isfactorized(flow)
            saving_args = i
        end

    elseif event_type == :horizon_hit
        (alg isa StickyLoopState && state isa StickyPDMPState) && _update_sticky_schedule_after_horizon_hit!(rng, alg, state, flow)
        stats.last_rejected = true
    end

    _check_sticky_times!(alg, state)
    return needs_saving, saving_args
end

function _handle_event_no_boundary!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::GlobalGradientStrategy, flow::ContinuousDynamics, alg::PoissonTimeStrategy, state::AbstractPDMPState, cache, event_type::Symbol, meta, stats::StatisticCounter)
    return _handle_global_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, event_type, meta, stats, false)
end

function handle_event!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::GlobalGradientStrategy, flow::ContinuousDynamics, alg::PoissonTimeStrategy, state::AbstractPDMPState, cache, event_type::Symbol, meta, stats::StatisticCounter)
    return _handle_global_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, event_type, meta, stats, true)
end

function _coordinate_boundary_context(
    state::PDMPState,
    flow::ZigZag,
    alg::ThinningStrategy,
    τ::Real,
    original_error,
)
    x0 = copy(state.ξ.x)
    v = copy(state.ξ.θ)
    _rewind_linear_boundary_position!(x0, v, τ, flow)
    return BoundaryContext(
        x0, v, Float64(state.t[] - τ), 0.0, max(Float64(τ), eps(Float64)),
        original_error, typeof(flow), _public_algorithm_type(alg))
end

function _handle_coordinatewise_event_impl!(
    rng::Random.AbstractRNG,
    τ::Real,
    gradient_strategy::CoordinateWiseGradient,
    flow::ZigZag,
    alg::ThinningStrategy,
    state::PDMPState,
    cache,
    meta,
    stats::StatisticCounter,
    wrap_boundary::Bool,
)
    stats.last_rejected = false
    pq = cache.pq
    i₀ = meta.i
    ξ = state.ξ
    abc_i₀_old = ab_i(i₀, ξ, alg, flow, cache)
    move_forward_time!(state, τ, flow)

    ∇ϕ_i₀ = if !wrap_boundary
        compute_gradient!(gradient_strategy, ξ.x, i₀, cache)
    else
        try
            compute_gradient!(gradient_strategy, ξ.x, i₀, cache)
        catch err
            throw(_ProbeFailureException(_coordinate_boundary_context(state, flow, alg, τ, err)))
        end
    end

    l_i₀ = λ_i(i₀, ξ, ∇ϕ_i₀, flow)
    l_bound_i₀ = pos(abc_i₀_old[1] + abc_i₀_old[2] * τ)

    stats.reflections_events += 1

    needs_saving = false
    saving_args = nothing

    if rand(rng) * l_bound_i₀ <= l_i₀
        if l_i₀ >= l_bound_i₀
            error("Tuning parameter `c` too small: l_i₀=$l_i₀, l_bound_i₀=$l_bound_i₀")
        end

        saving_args = reflect!(state, ∇ϕ_i₀, i₀, flow)
        needs_saving = true
        stats.reflections_accepted += 1

        empty!(pq)
        for i in eachindex(ξ.x)
            abc_i_new = ab_i(i, ξ, alg, flow, nothing)
            t_event = state.t[] + poisson_time(abc_i_new[1], abc_i_new[2], rand(rng))
            push!(pq, i => t_event)
        end
    else
        abc_i₀_new = ab_i(i₀, ξ, alg, flow, nothing)
        t_event = state.t[] + poisson_time(abc_i₀_new[1], abc_i₀_new[2], rand(rng))
        push!(pq, i₀ => t_event)
        stats.last_rejected = true
    end

    return needs_saving, saving_args
end

function _handle_event_no_boundary!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::CoordinateWiseGradient, flow::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, event_type, meta, stats::StatisticCounter)
    return _handle_coordinatewise_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, meta, stats, false)
end

function handle_event!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::CoordinateWiseGradient, flow::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, event_type, meta, stats::StatisticCounter)
    return _handle_coordinatewise_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, meta, stats, true)
end
