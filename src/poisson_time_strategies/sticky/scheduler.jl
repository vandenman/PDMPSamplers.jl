# θf = 20.4
# e1 = [-log(rand()) / (κ * abs(θf)) for _ in 1:10000]
# D2 = Exponential((κ * abs(θf)))
# qprobs = .01:.01:.99
# q1 = quantile(e1, qprobs)
# q2 = quantile(D2, qprobs)
# f, ax, _ = scatter(q1, q2)
# ablines!(ax, 0, 1, color = :grey, linestyle = :dash)
# f

# f0(κ, θf) = -log(rand()) / (κ * abs(θf))
# f1(κ, θf) = rand(Exponential(κ * abs(θf)))
# @benchmark f0($κ,$θf)
# @benchmark f1($κ,$θf)

"""
    τ = freezing_time(ξ::SkeletonPoint, flow::ContinuousDynamics, i::Integer)

computes the hitting time of the particle to hit 0 given the position `ξ.x[i]` and the velocity `ξ.θ[i]`.
"""
function freezing_time(ξ::SkeletonPoint, ::Union{BouncyParticle,ZigZag}, i::Integer)
    x = ξ.x[i]
    θ = ξ.θ[i]
    if θ * x >= 0
        return Inf
    else
        return -x / θ
    end
end

get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy,<:AbstractVector}, i, args...) = sticky_strat.κ[i]
get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy,<:Function}, i, args...) = sticky_strat.κ(i, args...)
get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy,<:RateFunction}, i, args...) = sticky_strat.κ(i, args...)

get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, i, args...) = sticky_state.κ[i]
get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy,<:Function}, i, args...) = sticky_state.κ(i, args...)
get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy,<:RateFunction}, i, args...) = sticky_state.κ(i, args...)

function update_all_stick_times!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)

    t = state.t[]
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        else # stuck/ frozen
            _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
        end
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    end
end

function _update_aggregate_unstick_time!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, horizon::Real=Inf)
    t = state.t[]
    τ = sample_time(rng, alg.clock, flow, state, horizon, alg.can_stick)
    alg.aggregate_unstick_time = t + τ
    return alg.aggregate_unstick_time
end

function update_all_stick_times!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    empty!(alg.sticky_pq)
    fill!(alg.sticky_times, Inf)
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        end
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    end
    _, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
    _update_aggregate_unstick_time!(rng, alg, state, flow, max(0.0, t_freeze - t))
    return nothing
end

function update_all_freeze_times!(alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
            isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
        end
    end
end

function update_all_freeze_times!(alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    empty!(alg.sticky_pq)
    fill!(alg.sticky_times, Inf)
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
            isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
        end
    end
    return nothing
end

function update_all_unfreeze_times!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    for i in alg.stickable_indices
        if !state.free[i]
            _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
            isinf(alg.sticky_times[i]) && error("sticky_times[$i] is Inf but it's stuck with x[i]=$(state.ξ.x[i])")
        end
    end
end

function update_all_unfreeze_times!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    _, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
    _update_aggregate_unstick_time!(rng, alg, state, flow, max(0.0, t_freeze - t))
    return nothing
end

_sticky_coordinate_index(meta::CoordinateMeta) = meta.i
_sticky_coordinate_index(i::Integer) = Int(i)

function _update_sticky_time_at_index!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, i::Int)
    if !alg.can_stick[i]
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        return nothing
    end
    t = state.t[]
    if state.free[i]
        _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    else
        _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN after unfreezing")
    end
    return nothing
end

function _update_sticky_time_at_index!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, i::Int)
    if !alg.can_stick[i]
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        return nothing
    end
    t = state.t[]
    if state.free[i]
        _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
    else
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        _, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
        _update_aggregate_unstick_time!(rng, alg, state, flow, max(0.0, t_freeze - t))
    end
    return nothing
end

function _update_sticky_schedule_after_reflect!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, meta)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_reflect!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, meta)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_reflect!(rng::Random.AbstractRNG, alg::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, state::StickyPDMPState, flow::ZigZag, meta::Union{CoordinateMeta,Integer})
    _update_sticky_time_at_index!(rng, alg, state, flow, _sticky_coordinate_index(meta))
    return nothing
end

function _update_sticky_schedule_after_refresh!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_refresh!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_refresh!(::Random.AbstractRNG, ::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, ::StickyPDMPState, ::ZigZag)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(::Random.AbstractRNG, alg::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_freeze_times!(alg, state, flow)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(::Random.AbstractRNG, ::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, ::StickyPDMPState, ::ZigZag)
    return nothing
end

function stick_or_unstick!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::ContinuousDynamics, alg::StickyLoopState, i::Int)

    t = state.t[]
    ξ = state.ξ
    sticky_times = alg.sticky_times
    θf = state.old_velocity
    tol = sqrt(eps(eltype(ξ.x))) # tolerance for floating point errors in move_forward_time!, could also depend on the flow?
    if state.free[i] # if free -> stuck

        # deterministic process should have move x[i] to exactly zero, but perhaps this needs a tolerance
        abs(ξ.x[i]) < tol || error("freezing but not frozen: x[i] = $(ξ.x[i]) !≈ 0 at $(sticky_times[i]) with tol = $(tol)")

        θf[i] = ξ.θ[i] # store speed
        ξ.θ[i] = 0.0 # freeze speed
        ξ.x[i] = 0.0 # freeze position, set to 0 exactly to avoid floating point errors
        state.free[i] = false # mark as stuck

        # κᵢ = get_κ(alg, i, state.ξ.x)
        # sticky_times[i] = t - log(rand()) / (κᵢ * abs(θf[i])) # sticky time

        if alg.κ isa AbstractVector
            _set_sticky_time!(alg, i, t + unfreeze_time(rng, alg, state, i))
            @assert !isnan(alg.sticky_times[i]) "sticky_times[$i] is NaN after unfreezing"
        else
            #= TODO: not sure about this design... there are a few cases:

                1. prior inclusion probabilities are independent and fixed: γᵢ ~ Bernoulli(pᵢ)
                    -> κ isa Vector
                2. prior inclusion probabilities depend on whether other parameters "stick": γᵢ ~ BetaBernoulli(n, a, b)
                    -> κ isa Function
                3. prior inclusion probabilities depend on hyperparameters: γᵢ ~ Bernoulli(θ); θ ~ Beta(1, 1))
                    -> κ isa Function

                alternatively for case 2:

                - all unfreezing times are exponentials
                - sample the first unfreezing time from the joint distribution of independent not identically distributed Exponentials.
                - tᶠ is min(first unfreezing time, first freezing time).
                We must (re)compute all freezing times since they are deterministic.

                case 3 still needs to be studied. No idea if the current approach even works.

            =#
            update_all_stick_times!(rng, alg, state, flow)
            # update_all_unfreeze_times!(alg, state, flow)
        end
        # tfrez[i] = t - log(rand()) # option 2 # TODO: this is independent of the prior!?

        # TODO: maybe we need to update other freezing times here as well


    else # stuck -> not stuck

        # deterministic process should have move x[i] to exactly zero and left it there
        # velocity should be at exactly zero at this point.
        (abs(ξ.x[i]) < tol && iszero(ξ.θ[i])) || error("unfreezing but not frozen: x[i] = $(ξ.x[i]) ≉ 0 or θ[i] = $(ξ.θ[i]) ≉ 0 at $(sticky_times[i]) with tol = $(tol)")# isfrozen

        ξ.θ[i] = θf[i] # restore speed
        θf[i] = zero(eltype(θf[i])) # perhaps not necessary?
        state.free[i] = true # mark as not stuck

        # update_all_stick_times!(alg, state, flow)
        _set_sticky_time!(alg, i, t + freezing_time(ξ, flow, i))
        isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(sticky_times[i])) after freezing (θ[i] = $(ξ.θ[i]))")
        if !(alg.κ isa AbstractVector)
            update_all_stick_times!(rng, alg, state, flow)
            # update_all_unfreeze_times!(alg, state, flow)
        end
        # TODO: maybe we need to update other freezing times here as well

    end
    validate_state(state, flow, "after stick_or_unstick! at index $i")
end

function stick_or_unstick!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::ContinuousDynamics, alg::AggregateStickyLoopState, i::Int)
    t = state.t[]
    ξ = state.ξ
    tol = sqrt(eps(eltype(ξ.x)))
    alg.can_stick[i] || throw(ArgumentError("coordinate $i is not stickable"))

    if state.free[i]
        abs(ξ.x[i]) < tol || error("freezing but not frozen: x[$i] = $(ξ.x[i]) !≈ 0 at $(t) with tol = $(tol)")
        state.old_velocity[i] = ξ.θ[i]
        ξ.θ[i] = 0.0
        ξ.x[i] = 0.0
        state.free[i] = false
        alg.sticky_times[i] = Inf
        haskey(alg.sticky_pq, i) && delete!(alg.sticky_pq, i)
        update_all_stick_times!(rng, alg, state, flow)
    else
        (abs(ξ.x[i]) < tol && iszero(ξ.θ[i])) ||
            error("unfreezing but not frozen: x[$i] = $(ξ.x[i]) ≉ 0 or θ[$i] = $(ξ.θ[i]) ≉ 0 at $(t) with tol = $(tol)")
        draw_boundary_velocity!(rng, state, flow, i)
        state.free[i] = true
        _set_sticky_time!(alg, i, t + freezing_time(ξ, flow, i))
        update_all_stick_times!(rng, alg, state, flow)
    end
    validate_state(state, flow, "after aggregate stick_or_unstick! at index $i")
end

function _bounded_inner_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
        inner_alg_state::GridAdaptiveState, state::StickyPDMPState, cache, stats::AbstractStatisticCounter,
        max_horizon::Float64, detect_boundaries::Bool=false)
    return next_event_time(rng, model, flow, inner_alg_state, state, cache, stats,
        max_horizon, false, :sticky_horizon_hit, detect_boundaries)
end

function _bounded_inner_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
        inner_alg_state::PoissonTimeStrategy, state::StickyPDMPState, cache, stats::AbstractStatisticCounter,
        max_horizon::Float64, detect_boundaries::Bool=false)
    τ, event_type, meta = next_event_time(rng, model, flow, inner_alg_state, state, cache, stats)
    if isfinite(max_horizon) && τ > max_horizon
        return max_horizon, :horizon_hit, EmptyMeta()
    end
    return τ, event_type, meta
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
        alg::AggregateStickyLoopState, state::StickyPDMPState, cache, stats::AbstractStatisticCounter,
        detect_boundaries::Bool=false)
    t = state.t[]
    inner_alg_state = alg.inner_alg_state

    i_freeze, t_freeze = isempty(alg.sticky_pq) ? (0, Inf) : first(alg.sticky_pq)
    t_unstick = alg.aggregate_unstick_time
    t_sticky = min(t_freeze, t_unstick)
    τ_sticky = max(0.0, t_sticky - t)

    if any(state.free)
        τ_refresh = rand_refresh_time(rng, flow)
        max_horizon = min(τ_sticky, τ_refresh)
        _inc_counter_sticky_inner_searches(stats)
        τ_inner, event_type, meta = _bounded_inner_event_time(rng, model, flow, inner_alg_state, state, cache, stats,
            max_horizon, detect_boundaries)

        if τ_sticky <= τ_inner && τ_sticky <= τ_refresh
            _inc_counter_sticky_inner_wasted_by_sticky(stats)
            if t_unstick <= t_freeze
                i_unstick = sample_label(rng, alg.clock, flow, state, τ_sticky, alg.can_stick)
                return τ_sticky, :sticky, CoordinateMeta(i_unstick)
            end
            return τ_sticky, :sticky, CoordinateMeta(i_freeze)
        elseif τ_refresh <= τ_inner
            _inc_counter_sticky_inner_wasted_by_refresh(stats)
            return τ_refresh, :refresh, GradientMeta(alg.empty_∇ϕx)
        else
            _inc_counter_sticky_inner_wins(stats)
            return τ_inner, event_type, meta
        end
    else
        _inc_counter_sticky_all_frozen_events(stats)
        isfinite(t_unstick) || return Inf, :sticky, CoordinateMeta(0)
        i_unstick = sample_label(rng, alg.clock, flow, state, t_unstick - t, alg.can_stick)
        return t_unstick - t, :sticky, CoordinateMeta(i_unstick)
    end
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
        alg::StickyLoopState, state::StickyPDMPState, cache, stats::AbstractStatisticCounter,
        detect_boundaries::Bool=false)

    t = state.t[]
    inner_alg_state = alg.inner_alg_state

    if isempty(alg.sticky_pq)
        i = 0
        tᶠ = Inf
    else
        i, tᶠ = first(alg.sticky_pq)
    end
    # non_sticky_state = PDMPState(state.t, state.ξ) # or substate, but that messes with dimensionality


    if any(state.free)
        # only propose reflection/ refreshment times when at least one parameter is free
        # max_horizon = tᶠ - t
        # if iszero(max_horizon)
        #     # sticky event happens now
        #     return 0.0, :sticky, i
        # end

        # TODO: this could be written in a cleaner way!
        # τ, event_type, meta = if inner_alg_state isa GridAdaptiveState
        #     # Pass max_horizon to GridThinning

        #     if iszero(max_horizon)
        #         @show max_horizon, state, sticky_time, tᶠ, t
        #         max_horizon = Inf
        #     end
        #     next_event_time(model, flow, inner_alg_state, state, cache, stats, 5 * max_horizon)
        # else
        #     next_event_time(model, flow, inner_alg_state, state, cache, stats)
        # end
        # @show τ

        # Sample refresh time independently at the sticky level and cap the
        # inner search.  GridThinning mutates its grid state while searching;
        # letting it search past a sticky/refresh boundary wastes gradients and
        # may adapt the proposal using a segment that the sticky wrapper will
        # discard.
        τ_refresh = rand_refresh_time(rng, flow)
        τ_sticky = max(0.0, tᶠ - t)
        max_horizon = min(τ_sticky, τ_refresh)

        _inc_counter_sticky_inner_searches(stats)
        τ, event_type, meta = _bounded_inner_event_time(
            rng, model, flow, inner_alg_state, state, cache, stats, max_horizon, detect_boundaries)

        if τ_sticky <= τ && τ_sticky <= τ_refresh # sticky event happens first
            _inc_counter_sticky_inner_wasted_by_sticky(stats)
            return τ_sticky, :sticky, CoordinateMeta(i)
        elseif τ_refresh <= τ
            _inc_counter_sticky_inner_wasted_by_refresh(stats)
            return τ_refresh, :refresh, GradientMeta(alg.empty_∇ϕx)
        else
            _inc_counter_sticky_inner_wins(stats)
            return τ, event_type, meta
        end

        # original
        # if tᶠ < t′
        #     Δt = tᶠ - t
        #     # iszero(Δt) && @warn "Sticky event time equals current time t = $t. This may lead to infinite loops."
        #     return Δt, :sticky, i
        # else
        #     return τ, event_type, meta
        # end
    else
        _inc_counter_sticky_all_frozen_events(stats)
        Δt = tᶠ - t
        return Δt, :sticky, CoordinateMeta(i)
    end
end

_reset_inner_grid!(alg::StickyLoopState) = _reset_inner_grid!(alg.inner_alg_state)
_reset_inner_grid!(alg::AggregateStickyLoopState) = _reset_inner_grid!(alg.inner_alg_state)
finish_warmup!(alg::StickyLoopState, stats::AbstractStatisticCounter, flow::ContinuousDynamics) =
    finish_warmup!(alg.inner_alg_state, stats, flow)
finish_warmup!(alg::AggregateStickyLoopState, stats::AbstractStatisticCounter, flow::ContinuousDynamics) =
    finish_warmup!(alg.inner_alg_state, stats, flow)
_record_final_grid_state!(alg::StickyLoopState, stats::AbstractStatisticCounter) =
    _record_final_grid_state!(alg.inner_alg_state, stats)
_record_final_grid_state!(alg::AggregateStickyLoopState, stats::AbstractStatisticCounter) =
    _record_final_grid_state!(alg.inner_alg_state, stats)
_invalidate_cached_gradient!(alg::StickyLoopState) = _invalidate_cached_gradient!(alg.inner_alg_state)
_invalidate_cached_gradient!(alg::AggregateStickyLoopState) = _invalidate_cached_gradient!(alg.inner_alg_state)

function _maybe_activate_constant_bound!(alg::StickyLoopState, stats::AbstractStatisticCounter)
    _maybe_activate_constant_bound!(alg.inner_alg_state, stats)
end
function _maybe_activate_constant_bound!(alg::AggregateStickyLoopState, stats::AbstractStatisticCounter)
    _maybe_activate_constant_bound!(alg.inner_alg_state, stats)
end
