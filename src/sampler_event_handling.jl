function _check_sticky_times!(alg::PoissonTimeStrategy, state::AbstractPDMPState)
    return nothing
end

function _check_sticky_times!(alg::StickyLoopState, state::AbstractPDMPState)
    all(>=(state.t[]), alg.sticky_times) || error("some sticky_times are negative!")
    return nothing
end

function _check_sticky_times!(alg::AggregateStickyLoopState, state::AbstractPDMPState)
    all(>=(state.t[]), alg.sticky_times) || error("some sticky_times are negative!")
    alg.aggregate_unstick_time >= state.t[] || error("aggregate_unstick_time is in the past")
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

const _BOOMERANG_INTERFERENCE_TARGET_INDICES = Int[]
_boomerang_interference_target_indices() = _BOOMERANG_INTERFERENCE_TARGET_INDICES

@inline function _diag_metric_quad_subset(Γ::Diagonal, z::AbstractVector, indices::AbstractVector{Int})
    q = 0.0
    diagΓ = Γ.diag
    @inbounds for i in indices
        if 1 <= i <= length(z)
            q += Float64(diagΓ[i]) * Float64(z[i])^2
        end
    end
    return q
end

@inline function _diag_metric_quad_subset(Γ, z::AbstractVector, indices::AbstractVector{Int})
    q = 0.0
    @inbounds for j in indices
        1 <= j <= length(z) || continue
        zj = Float64(z[j])
        row_sum = 0.0
        for k in indices
            1 <= k <= length(z) || continue
            row_sum += Float64(Γ[j, k]) * Float64(z[k])
        end
        q += zj * row_sum
    end
    return q
end

@inline function _boomerang_metric_quad(flow::AnyBoomerang, z::AbstractVector, ∇ϕ::AbstractVector)
    return Float64(dot(z, ∇ϕ))
end

@inline function _boomerang_metric_quad(flow::LowRankMutableBoomerang, z::AbstractVector, ∇ϕ::AbstractVector)
    return Float64(lowrank_quadform(flow.Γ, z))
end

function _record_boomerang_interference!(
    stats::AbstractStatisticCounter,
    state::AbstractPDMPState,
    ∇ϕ::AbstractVector,
    flow::AnyBoomerang,
    cache,
    phase::Symbol,
)
    phase === :main || return nothing
    target_indices = _boomerang_interference_target_indices()
    isempty(target_indices) && return nothing

    θ = state.ξ.θ
    d = length(θ)
    c_total = 0.0
    c_target = 0.0
    @inbounds for i in 1:d
        ci = abs(Float64(θ[i]) * Float64(∇ϕ[i]))
        c_total += ci
    end
    @inbounds for i in target_indices
        if 1 <= i <= d
            c_target += abs(Float64(θ[i]) * Float64(∇ϕ[i]))
        end
    end
    c_share = ispositive(c_total) ? c_target / c_total : 0.0

    z = cache.z
    copyto!(z, ∇ϕ)
    if flow isa LowRankMutableBoomerang
        lowrank_solve!(z, flow.Γ)
    else
        ldiv!(flow.L, z)
        ldiv!(flow.L', z)
    end
    d_total = max(_boomerang_metric_quad(flow, z, ∇ϕ), 0.0)
    d_target = if flow.Γ isa LowRankPrecision
        # Exact target-subblock metric for low-rank Γ is not cheap without an
        # extra work vector; skip low-rank target-energy diagnostics for now.
        0.0
    else
        max(_diag_metric_quad_subset(flow.Γ, z, target_indices), 0.0)
    end
    d_share = ispositive(d_total) ? min(d_target / d_total, 1.0) : 0.0

    _inc_counter_boomerang_interference_events(stats)
    _inc_counter_boomerang_target_c_share_sum(stats, c_share)
    _inc_counter_boomerang_target_d_share_sum(stats, d_share)

    c_threshold = 0.1
    d_threshold = 0.25
    if c_share <= c_threshold && d_share >= d_threshold
        _inc_counter_boomerang_nuisance_driven_target_disturbances(stats)
    end
    return nothing
end

@inline _record_boomerang_interference!(stats, state, ∇ϕ, flow, cache, phase) = nothing

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
    stats::AbstractStatisticCounter,
    wrap_boundary::Bool,
    phase::Symbol,
)
    move_forward_time!(state, τ, flow)
    validate_state(state, flow, "after moving forward in time")

    needs_saving = false
    saving_args = nothing
    _set_counter_last_rejected(stats, false)

    if event_type == :reflect
        _inc_counter_reflections_events(stats)

        if alg isa ExactStrategy
            _inc_counter_reflections_accepted(stats)

            if flow isa ZigZag
                i = meta.i
                reflect!(state.ξ, zero(eltype(cache.∇ϕx)), i, flow)
            else
                ∇ϕx = _compute_exact_reflection_gradient!(state, gradient_strategy, flow, cache, alg, τ, wrap_boundary)
                _record_boomerang_interference!(stats, state, ∇ϕx, flow, cache, phase)
                saving_args = reflect!(rng, state, ∇ϕx, flow, cache)
            end
            needs_saving = true
            (_is_sticky_loop_state(alg) && state isa StickyPDMPState) && _update_sticky_schedule_after_reflect!(rng, alg, state, flow, meta)
        else
            ∇ϕx = _compute_reflection_gradient!(state, gradient_strategy, flow, cache, meta, alg, τ, wrap_boundary)

            if accept_reflection_event(rng, alg, state.ξ, ∇ϕx, flow, τ, cache, meta)
                _inc_counter_reflections_accepted(stats)
                _record_boomerang_interference!(stats, state, ∇ϕx, flow, cache, phase)
                saving_args = reflect!(rng, state, ∇ϕx, flow, cache)
                needs_saving = true
                (_is_sticky_loop_state(alg) && state isa StickyPDMPState) && _update_sticky_schedule_after_reflect!(rng, alg, state, flow, saving_args)
            else
                _set_counter_last_rejected(stats, true)
            end
        end

    elseif event_type == :refresh
        refresh_velocity!(rng, state, flow)
        needs_saving = true
        _inc_counter_refreshment_events(stats)
        (_is_sticky_loop_state(alg) && state isa StickyPDMPState) && _update_sticky_schedule_after_refresh!(rng, alg, state, flow)

    elseif event_type == :sticky
        _inc_counter_sticky_events(stats)
        i = meta.i
        position_will_snap = state.free[i]
        stick_or_unstick!(rng, state::StickyPDMPState, flow, alg, i)
        if position_will_snap &&
            alg isa Union{StickyLoopState,AggregateStickyLoopState} &&
            alg.inner_alg_state isa GridAdaptiveState
            _invalidate_cached_gradient!(alg.inner_alg_state)
        end
        set_active_set!(gradient_strategy, state.free)
        validate_state(state, flow, "after stick_or_unstick!")
        needs_saving = true
        if isfactorized(flow)
            saving_args = i
        end

    elseif event_type == :horizon_hit
        (_is_sticky_loop_state(alg) && state isa StickyPDMPState) && _update_sticky_schedule_after_horizon_hit!(rng, alg, state, flow)
        _set_counter_last_rejected(stats, true)
    end

    _check_sticky_times!(alg, state)
    return needs_saving, saving_args
end

function _handle_event_no_boundary!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::GlobalGradientStrategy, flow::ContinuousDynamics, alg::PoissonTimeStrategy, state::AbstractPDMPState, cache, event_type::Symbol, meta, stats::AbstractStatisticCounter, phase::Symbol=:unknown)
    return _handle_global_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, event_type, meta, stats, false, phase)
end

function handle_event!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::GlobalGradientStrategy, flow::ContinuousDynamics, alg::PoissonTimeStrategy, state::AbstractPDMPState, cache, event_type::Symbol, meta, stats::AbstractStatisticCounter, phase::Symbol=:unknown)
    return _handle_global_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, event_type, meta, stats, true, phase)
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
    stats::AbstractStatisticCounter,
    wrap_boundary::Bool,
)
    _set_counter_last_rejected(stats, false)
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

    _inc_counter_reflections_events(stats)

    needs_saving = false
    saving_args = nothing

    if rand(rng) * l_bound_i₀ <= l_i₀
        if l_i₀ >= l_bound_i₀
            error("Tuning parameter `c` too small: l_i₀=$l_i₀, l_bound_i₀=$l_bound_i₀")
        end

        saving_args = reflect!(state, ∇ϕ_i₀, i₀, flow)
        needs_saving = true
        _inc_counter_reflections_accepted(stats)

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
        _set_counter_last_rejected(stats, true)
    end

    return needs_saving, saving_args
end

function _handle_event_no_boundary!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::CoordinateWiseGradient, flow::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, event_type, meta, stats::AbstractStatisticCounter, phase::Symbol=:unknown)
    return _handle_coordinatewise_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, meta, stats, false)
end

function handle_event!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::CoordinateWiseGradient, flow::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, event_type, meta, stats::AbstractStatisticCounter, phase::Symbol=:unknown)
    return _handle_coordinatewise_event_impl!(rng, τ, gradient_strategy, flow, alg, state, cache, meta, stats, true)
end
