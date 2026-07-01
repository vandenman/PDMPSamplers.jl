"""
A simple wrapper function so that the Sticky strategy knows whether a function
returns the waiting time, or the rate of an inhomogeneous Poisson process.

The rate_function must accept the same arguments as κ(i, x, free, θ)

"""
struct RateFunction{F<:Function}
    rate_function::F
end

"""


There are three options for the unfreezing time, κ.

- `AbstractVector`: For independent model and parameter priors, the rates are fixed and can be known in advance.
- `Function`: For dependent model priors, the rates only depend on whether the parameters are (non)zero.
- `RateFunction`: For dependent parameters priors, the rates depend on the specific values of the parameters, and the unfreeze times become inhomogeneous Poisson processes.

The first two should return the rate of an homogeneous Poisson process, such that

```julia
λ = κ[i] or κ(i, x, γ, θ)
rand(Exponential(inv(κ * abs(θf))))
```
equals the unfreezing time.

The third should return the rate of an inhomogeneous Poisson process, which is sampled through the algorithm specified by `alg`.
"""
struct Sticky{T<:PoissonTimeStrategy,U<:Union{Function,RateFunction,AbstractVector}} <: PoissonTimeStrategy
    alg::T
    κ::U
    can_stick::BitVector
end
Sticky(alg::PoissonTimeStrategy, κ::AbstractVector) = Sticky(alg, κ, .!isinf.(κ))
Sticky(::PoissonTimeStrategy, ::Function) = throw(ArgumentError("When κ is a function, a can_stick vector must be provided explicitly."))

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

accept_reflection_event(rng::Random.AbstractRNG, alg::StickyLoopState, args...) = accept_reflection_event(rng, alg.inner_alg_state, args...)
accept_reflection_event(alg::StickyLoopState, args...) = accept_reflection_event(alg.inner_alg_state, args...)

# this could use less memory by looking at
function _to_internal(strat::Sticky, rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)

    d = length(state.ξ)
    sticky_times = fill(Inf, d)
    stickable_indices = findall(strat.can_stick)
    sticky_pq = PriorityQueue{Int,Float64}()

    internal_alg_ = _to_internal(strat.alg, rng, flow, model, state, cache, stats)

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

function _set_sticky_time!(alg::StickyLoopState, i::Int, t::Float64)
    alg.sticky_times[i] = t
    alg.sticky_pq[i] = t
    return t
end


"""
    unfreeze_time(alg::StickyLoopState, state::StickyPDMPState, i::Integer)

Simulate the time a stuck/ frozen particle takes to unfreeze/ unstick
"""
function unfreeze_time(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, i::Integer)
    validate_state(state, nothing, "in unfreeze_time")
    κ = get_κ(alg, i, state.ξ.x, state.free, state.ξ.θ)

    if κ isa Distribution

        retval = rand(rng, κ)
        if isnegative(retval)# || isinf(retval)
            @show κ, i, state.ξ.x, state.free, state.ξ.θ
            throw(ArgumentError("κ must be non-negative and finite!"))
        end
        # @show κ, i, state.ξ.x, state.free
        return retval
    else
        θf = state.old_velocity[i]
        if isnegative(κ)
            @show κ, i, state.ξ.x, state.free
            throw(ArgumentError("κ must be non-negative!"))
        end

        # return -log(rand()) / (κ * abs(θf)) # old approach
        return rand(rng, Exponential(inv(κ * abs(θf))))
    end
end

# κ = 1.23
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

function update_all_freeze_times!(alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    for i in alg.stickable_indices
        if state.free[i]
            _set_sticky_time!(alg, i, t + freezing_time(state.ξ, flow, i))
            isnan(alg.sticky_times[i]) && error("sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))")
        end
    end
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

function _update_sticky_schedule_after_reflect!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics, meta)
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

function _update_sticky_schedule_after_refresh!(::Random.AbstractRNG, ::StickyLoopState{<:PoissonTimeStrategy,<:AbstractVector}, ::StickyPDMPState, ::ZigZag)
    return nothing
end

function _update_sticky_schedule_after_horizon_hit!(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    update_all_stick_times!(rng, alg, state, flow)
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

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::StickyLoopState, state::StickyPDMPState, cache, stats::AbstractStatisticCounter)

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

        # Sample refresh time independently at the sticky level
        τ_refresh = rand_refresh_time(rng, flow)
        tʳ = t + τ_refresh

        _inc_counter_sticky_inner_searches(stats)
        τ, event_type, meta = next_event_time(rng, model, flow, inner_alg_state, state, cache, stats, Inf, false)

        t′ = t + τ

        if tᶠ < t′ && tᶠ < tʳ #  sticky event happens first
            _inc_counter_sticky_inner_wasted_by_sticky(stats)
            Δt = tᶠ - t
            return Δt, :sticky, CoordinateMeta(i)
        elseif tʳ < t′
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
_invalidate_cached_gradient!(alg::StickyLoopState) = _invalidate_cached_gradient!(alg.inner_alg_state)

function _maybe_activate_constant_bound!(alg::StickyLoopState, stats::AbstractStatisticCounter)
    _maybe_activate_constant_bound!(alg.inner_alg_state, stats)
end
