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
struct Sticky{T<:PoissonTimeStrategy, U<:Union{Function, RateFunction, AbstractVector}} <: PoissonTimeStrategy
    alg::T
    κ::U
    can_stick::BitVector
end
Sticky(alg::PoissonTimeStrategy, κ::AbstractVector) = Sticky(alg, κ, .!isinf.(κ))
Sticky(::PoissonTimeStrategy, ::Function) = throw(ArgumentError("When κ is a function, a can_stick vector must be provided explicitly."))

struct StickyLoopState{T <: PoissonTimeStrategy, U<:Union{Function, RateFunction, AbstractVector}} <: PoissonTimeStrategy
    # A' could be the internal version of the wrapped algorithm
    inner_alg_state::T # this should perhaps be the more generic, i.e., _to_internal(Sticky.alg, ...)!
    κ::U
    can_stick::BitVector
    sticky_times::Vector{Float64}  # Absolute times of next freeze/unfreeze event
    old_velocity::Vector{Float64} # Old velocity at the time of freezing
end

# this could use less memory by looking at
function _to_internal(strat::Sticky, flow::ContinuousDynamics, grad::GradientStrategy, state::AbstractPDMPState, cache, stats::StatisticCounter)

    d = length(state.ξ)
    sticky_times = Vector{Float64}(undef, d)

    internal_alg_ = _to_internal(strat.alg, flow, grad, state, cache, stats)

    old_velocity = copy(state.ξ.θ)
    # zero is problematic because the unfreeze time divides by abs(θf[i]), so divide by zero
    if any(iszero, old_velocity)
        old_velocity2 = initialize_velocity(flow, d)
        for i in eachindex(old_velocity, old_velocity2)
            if iszero(old_velocity[i])
                old_velocity[i] = old_velocity2[i]
            end
        end
    end
    alg = StickyLoopState(internal_alg_, strat.κ, strat.can_stick, sticky_times, old_velocity)
    update_all_stick_times!(alg, state, flow)

    # do this once
    for i in eachindex(alg.sticky_times)
        if !alg.can_stick[i]
            alg.sticky_times[i] = Inf
        end
    end
    # @show alg.sticky_times
    @assert !any(isnan, alg.sticky_times) "sticky_times contains NaN: $(alg.sticky_times)"

    return alg

end


"""
    unfreeze_time(alg::StickyLoopState, state::StickyPDMPState, i::Integer)

Simulate the time a stuck/ frozen particle takes to unfreeze/ unstick
"""
function unfreeze_time(alg::StickyLoopState, state::StickyPDMPState, i::Integer)
    validate_state(state, nothing, "in unfreeze_time")
    κ = get_κ(alg, i, state.ξ.x, state.free, state.ξ.θ)

    if κ isa Distribution

        retval = rand(κ)
        if isnegative(retval)# || isinf(retval)
            @show κ, i, state.ξ.x, state.free, state.ξ.θ
            throw(ArgumentError("κ must be non-negative and finite!"))
        end
        # @show κ, i, state.ξ.x, state.free
        return retval
    else
        θf = alg.old_velocity[i]
        if isnegative(κ)
            @show κ, i, state.ξ.x, state.free
            throw(ArgumentError("κ must be non-negative!"))
        end

    # return -log(rand()) / (κ * abs(θf)) # old approach
        return rand(Exponential(inv(κ * abs(θf))))
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
function freezing_time(ξ::SkeletonPoint, ::Union{BouncyParticle, ZigZag}, i::Integer)
    x = ξ.x[i]
    θ = ξ.θ[i]
    if θ * x >= 0
        return Inf
    else
        return -x / θ
    end
end

get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy, <:AbstractVector}, i, args...)     = sticky_strat.κ[i]
get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy, <:Function},       i, args...)     = sticky_strat.κ(i, args...)
get_κ(sticky_strat::Sticky{<:PoissonTimeStrategy, <:RateFunction},       i, args...) = sticky_strat.κ(i, args...)

get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy, <:AbstractVector}, i, args...)     = sticky_state.κ[i]
get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy, <:Function},       i, args...)     = sticky_state.κ(i, args...)
get_κ(sticky_state::StickyLoopState{<:PoissonTimeStrategy, <:RateFunction},       i, args...) = sticky_state.κ(i, args...)

function update_all_stick_times!(alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)

    t = state.t[]
    # TODO: instead of 1:d we should precompute a vector of things that may stick and loop over that
    # then we don't need to check if alg.can_stick[i]
    for i in eachindex(alg.can_stick)
        if alg.can_stick[i]
            if state.free[i]
                alg.sticky_times[i] = t + freezing_time(state.ξ, flow, i)
                # @show "freezing_time", alg.sticky_times[i]
            else # stuck/ frozen
                alg.sticky_times[i] = t + unfreeze_time(alg, state, i)
                # @assert !isinf(alg.sticky_times[i]) "sticky_times[$i] is Inf but it's stuck with x[i]=$(state.ξ.x[i])"

                # @show "unsticking_time", alg.sticky_times[i]
            end
            @assert !isnan(alg.sticky_times[i]) "sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))"
        end
    end
end

function update_all_freeze_times!(alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    for i in eachindex(alg.can_stick)
        if alg.can_stick[i]
            if state.free[i]
                alg.sticky_times[i] = t + freezing_time(state.ξ, flow, i)
                @assert !isnan(alg.sticky_times[i]) "sticky_times[$i] is NaN ($(alg.sticky_times[i])) after freezing (θ[i] = $(state.ξ.θ[i]))"
            end
        end
    end
end

function update_all_unfreeze_times!(alg::StickyLoopState, state::StickyPDMPState, flow::ContinuousDynamics)
    t = state.t[]
    for i in eachindex(alg.can_stick)
        if alg.can_stick[i]
            if !state.free[i]
                alg.sticky_times[i] = t + unfreeze_time(alg, state, i)
                @assert !isinf(alg.sticky_times[i]) "sticky_times[$i] is Inf but it's stuck with x[i]=$(state.ξ.x[i])"
            end
        end
    end
end

function stick_or_unstick!(state::StickyPDMPState, flow::ContinuousDynamics, alg::StickyLoopState, i::Int)

    t = state.t[]
    ξ = state.ξ
    sticky_times = alg.sticky_times
    θf = alg.old_velocity
    tol = sqrt(eps(eltype(ξ.x))) # tolerance for floating point errors in move_forward_time!, could also depend on the flow?
    if state.free[i] # if free -> stuck

        # deterministic process should have move x[i] to exactly zero, but perhaps this needs a tolerance
        @assert abs(ξ.x[i]) < tol "freezing but not frozen: x[i] = $(ξ.x[i]) !≈ 0 at $(sticky_times[i]) with tol = $(tol)"

        θf[i] = ξ.θ[i] # store speed
        ξ.θ[i] = 0.0 # freeze speed
        ξ.x[i] = 0.0 # freeze position, set to 0 exactly to avoid floating point errors
        state.free[i] = false # mark as stuck

        # κᵢ = get_κ(alg, i, state.ξ.x)
        # sticky_times[i] = t - log(rand()) / (κᵢ * abs(θf[i])) # sticky time

        if alg.κ isa AbstractVector
            sticky_times[i] = t + unfreeze_time(alg, state, i)
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
            update_all_stick_times!(alg, state, flow)
            # update_all_unfreeze_times!(alg, state, flow)
        end
        # tfrez[i] = t - log(rand()) # option 2 # TODO: this is independent of the prior!?

        # TODO: maybe we need to update other freezing times here as well


    else # stuck -> not stuck

        # deterministic process should have move x[i] to exactly zero and left it there
        # velocity should be at exactly zero at this point.
        @assert abs(ξ.x[i]) < tol && iszero(ξ.θ[i]) "unfreezing but not frozen: x[i] = $(ξ.x[i]) ≉ 0 or θ[i] = $(ξ.θ[i]) ≉ 0 at $(sticky_times[i]) with tol = $(tol)"# isfrozen

        ξ.θ[i] = θf[i] # restore speed
        θf[i]  = zero(eltype(θf[i])) # perhaps not necessary?
        state.free[i] = true # mark as not stuck

        # update_all_stick_times!(alg, state, flow)
        sticky_times[i] = t + freezing_time(ξ, flow, i)
        @assert !isnan(alg.sticky_times[i]) "sticky_times[$i] is NaN ($(sticky_times[i])) after freezing (θ[i] = $(ξ.θ[i]))"
        if !(alg.κ isa AbstractVector)
            update_all_stick_times!(alg, state, flow)
            # update_all_unfreeze_times!(alg, state, flow)
        end
        # TODO: maybe we need to update other freezing times here as well

    end
end

function next_event_time(grad::GlobalGradientStrategy, flow::ContinuousDynamics, alg::StickyLoopState, state::StickyPDMPState, cache, stats::StatisticCounter)

    t = state.t[]
    sticky_time = alg.sticky_times
    inner_alg_state = alg.inner_alg_state

    tᶠ, i = findmin(sticky_time) # could be implemented with a queue or a MinHeap
    # non_sticky_state = PDMPState(state.t, state.ξ) # or substate, but that messes with dimensionality
    if any(state.free)
        # only propose reflection/ refreshment times when at least one parameter is free
        τ, event_type, meta = next_event_time(grad, flow, inner_alg_state, state, cache, stats)
        t′ = t + τ
        if tᶠ < t′
            Δt = tᶠ - t
            return Δt, :sticky, i
        else
            return τ, event_type, meta
        end
    else
        Δt = tᶠ - t
        return Δt, :sticky, i
    end
end
