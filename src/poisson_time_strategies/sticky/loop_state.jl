struct StickyLoopState{T<:PoissonTimeStrategy,U<:Union{Function,RateFunction,AbstractVector},V<:AbstractVector} <: PoissonTimeStrategy
    # A' could be the internal version of the wrapped algorithm
    inner_alg_state::T # this should perhaps be the more generic, i.e., _to_internal(Sticky.alg, ...)!
    κ::U
    can_stick::BitVector
    sticky_times::Vector{Float64}  # Absolute times of next freeze/unfreeze event
    stickable_indices::Vector{Int}
    sticky_pq::PriorityQueue{Int,Float64}
    empty_∇ϕx::V
end

mutable struct AggregateStickyLoopState{T<:PoissonTimeStrategy,C<:AbstractAggregateUnstickClock,V<:AbstractVector} <: PoissonTimeStrategy
    inner_alg_state::T
    clock::C
    can_stick::BitVector
    sticky_times::Vector{Float64}
    stickable_indices::Vector{Int}
    sticky_pq::PriorityQueue{Int,Float64}
    aggregate_unstick_time::Float64
    empty_∇ϕx::V
end

accept_reflection_event(rng::Random.AbstractRNG, alg::StickyLoopState, args...) = accept_reflection_event(rng, alg.inner_alg_state, args...)
accept_reflection_event(alg::StickyLoopState, args...) = accept_reflection_event(alg.inner_alg_state, args...)
accept_reflection_event(rng::Random.AbstractRNG, alg::AggregateStickyLoopState, args...) = accept_reflection_event(rng, alg.inner_alg_state, args...)
accept_reflection_event(alg::AggregateStickyLoopState, args...) = accept_reflection_event(alg.inner_alg_state, args...)

_is_sticky_loop_state(::StickyLoopState) = true
_is_sticky_loop_state(::AggregateStickyLoopState) = true

function _enforce_nonstickable_coordinates_free!(state::StickyPDMPState, can_stick::AbstractVector{Bool})
    length(can_stick) == length(state.free) ||
        throw(DimensionMismatch("can_stick length $(length(can_stick)) does not match dimension $(length(state.free))"))
    @inbounds for i in eachindex(state.free, can_stick)
        can_stick[i] || (state.free[i] = true)
    end
    return state
end

# this could use less memory by looking at
function _to_internal(strat::Sticky, rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)

    d = length(state.ξ)
    state isa StickyPDMPState && _enforce_nonstickable_coordinates_free!(state, strat.can_stick)
    sticky_times = fill(Inf, d)
    stickable_indices = findall(strat.can_stick)
    sticky_pq = PriorityQueue{Int,Float64}()

    internal_alg_ = _to_internal(strat.alg, rng, flow, model, state, cache, stats)
    state isa StickyPDMPState && set_active_set!(model, state.free)

    # old_velocity = copy(state.ξ.θ)
    # # zero is problematic because the unfreeze time divides by abs(θf[i]), so divide by zero
    # if any(iszero, old_velocity)
    #     old_velocity2 = initialize_velocity(flow, d)
    #     for i in eachindex(old_velocity, old_velocity2)
    #         if iszero(old_velocity[i])
    #             old_velocity[i] = old_velocity2[i]
    #         end
    #     end
    # end
    alg = StickyLoopState(internal_alg_, strat.κ, strat.can_stick, sticky_times, stickable_indices, sticky_pq, similar(state.ξ.x, 0))
    update_all_stick_times!(rng, alg, state, flow)
    # @show alg.sticky_times
    any(isnan, alg.sticky_times) && error("sticky_times contains NaN: $(alg.sticky_times)")

    return alg

end

function _to_internal(strat::AggregateSticky, rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    state isa StickyPDMPState || throw(ArgumentError("AggregateSticky requires StickyPDMPState; initialize_state must use requires_sticky_state"))
    _supports_aggregate_sticky_flow(flow) ||
        throw(ArgumentError("AggregateSticky does not support this flow; dense-preconditioned ZigZag needs a separate coordinate boundary velocity law"))
    d = length(state.ξ)
    length(strat.can_stick) == d || throw(DimensionMismatch("can_stick length $(length(strat.can_stick)) does not match dimension $d"))
    _enforce_nonstickable_coordinates_free!(state, strat.can_stick)
    sticky_times = fill(Inf, d)
    stickable_indices = findall(strat.can_stick)
    sticky_pq = PriorityQueue{Int,Float64}()
    internal_alg_ = _to_internal(strat.alg, rng, flow, model, state, cache, stats)
    internal_alg_ isa GridAdaptiveState ||
        throw(ArgumentError("AggregateSticky requires an inner strategy with bounded event search; use GridThinningStrategy for now"))
    set_active_set!(model, state.free)
    alg = AggregateStickyLoopState(internal_alg_, copy(strat.clock), copy(strat.can_stick), sticky_times, stickable_indices, sticky_pq, Inf, similar(state.ξ.x, 0))
    update_all_stick_times!(rng, alg, state, flow)
    any(isnan, alg.sticky_times) && error("sticky_times contains NaN: $(alg.sticky_times)")
    isnan(alg.aggregate_unstick_time) && error("aggregate_unstick_time is NaN")
    return alg
end

function _set_sticky_time!(alg::StickyLoopState, i::Int, t::Float64)
    alg.sticky_times[i] = t
    alg.sticky_pq[i] = t
    return t
end

function _set_sticky_time!(alg::AggregateStickyLoopState, i::Int, t::Float64)
    alg.sticky_times[i] = t
    if isfinite(t)
        alg.sticky_pq[i] = t
    elseif haskey(alg.sticky_pq, i)
        delete!(alg.sticky_pq, i)
    end
    return t
end
