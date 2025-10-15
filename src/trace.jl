_trace_type(flow::ContinuousDynamics, alg::PoissonTimeStrategy) = PDMPTrace
_trace_type(flow::FactorizedDynamics, alg::PoissonTimeStrategy) = FactorizedTrace
# _trace_type(flow::FactorizedDynamics, alg::PoissonTimeStrategy) = PDMPTrace
# _trace_type(flow::FactorizedDynamics, alg::Sticky)              = FactorizedTrace


abstract type AbstractPDMPEvent end

struct PDMPEvent{T,X,V} <: AbstractPDMPEvent
    time::T
    position::X
    velocity::V
end
PDMPEvent(t::Real, x::AbstractVector, θ::AbstractVector) = PDMPEvent{float(typeof(t)), typeof(x), typeof(θ)}(float(t), x, θ)
PDMPEvent(state::AbstractPDMPState) = PDMPEvent(state.t[], copy(state.ξ.x), copy(state.ξ.θ))

struct FactorizedEvent{T,U,V} <: AbstractPDMPEvent
    index::Int
    time::T
    position::U
    velocity::V
end

abstract type AbstractPDMPTrace end

struct PDMPTrace{T<:PDMPEvent,U<:ContinuousDynamics} <: AbstractPDMPTrace
    events::Vector{T}
    flow::U
end
PDMPTrace(state::AbstractPDMPState, flow::ContinuousDynamics) = PDMPTrace([PDMPEvent(state)], flow)

struct FactorizedTrace{T<:FactorizedEvent, U<:FactorizedDynamics, V<:PDMPEvent} <: AbstractPDMPTrace
    events::Vector{T}
    flow::U
    initial_state::V
end
function FactorizedTrace(state::AbstractPDMPState, flow::ContinuousDynamics)
    initial_state = PDMPEvent(state)
    events = Vector{FactorizedEvent{typeof(state.t[]), eltype(state.ξ.x), eltype(state.ξ.θ)}}()
    FactorizedTrace(events, flow, initial_state)
end

# TODO: use this to test that PDMPTrace and FactorizedTrace give the same results!
# in general we could also go the other way around, but only if trace.flow <: FactorizedDynamics
function PDMPTrace(trace::FactorizedTrace)
    events = [trace.initial_state]
    sizehint!(events, length(trace))
    position = copy(trace.initial_state.position)
    velocity = copy(trace.initial_state.velocity)
    ξ = SkeletonPoint(position, velocity)
    t0 = trace.initial_state.time
    for i in eachindex(trace.events)
        Δt = trace.events[i].time - t0
        move_forward_time!(ξ, Δt, trace.flow)
        t0 = trace.events[i].time
        _to_next_event!(position, velocity, trace.events[i])
        push!(events, PDMPEvent(trace.events[i].time, copy(position), copy(velocity)))
    end
    PDMPTrace(events, trace.flow)
end

Base.first(trace::PDMPTrace)       = trace.events[1]
Base.first(trace::FactorizedTrace) = trace.initial_state

Base.push!(trace::PDMPTrace,       event::PDMPEvent)         = push!(trace.events, event)
Base.push!(trace::PDMPTrace,       event::AbstractPDMPState) = push!(trace.events, PDMPEvent(event))
Base.push!(trace::FactorizedTrace, event::FactorizedEvent)   = push!(trace.events, event)

function Base.push!(trace::FactorizedTrace, event::AbstractPDMPState, i::Integer)
    # TODO: this needs to figure out which index i changed... but it's much easier to have the caller provide it...
    push!(trace.events, FactorizedEvent(i, event.t[], event.ξ.x[i], event.ξ.θ[i]))
end

function _to_next_event!(x::AbstractVector, θ::AbstractVector, event::PDMPEvent)
    x .= event.position
    θ .= event.velocity
    event.time
end
function _to_next_event!(x::AbstractVector, θ::AbstractVector, event::FactorizedEvent)
    x[event.index] = event.position
    θ[event.index] = event.velocity
    event.time
end

Base.length(trace::PDMPTrace)       = length(trace.events)
Base.length(trace::FactorizedTrace) = length(trace.events) + 1

function Base.iterate(trace::AbstractPDMPTrace)

    isempty(trace.events) && return nothing

    # Start with first event
    e1 = first(trace)
    t, x, θ = e1.time, copy(e1.position), copy(e1.velocity)
    k = (trace.flow isa FactorizedDynamics ? 1 : 2)
    return t => copy(x), (t, x, θ, k)
end

function Base.iterate(trace::AbstractPDMPTrace, (t, x, θ, k))
    k > length(trace.events) && return nothing
    k == length(trace.events) && return t => x, (t, x, θ, k + 1)

    Δt = trace.events[k].time - t
    move_forward_time!(SkeletonPoint(x, θ), Δt, trace.flow)
    t = _to_next_event!(x, θ, trace.events[k])
    return t => copy(x), (t, x, θ, k + 1)
end

Base.collect(FT::AbstractPDMPTrace) = collect(t=>x for (t, x) in FT)

Statistics.mean(trace::AbstractPDMPTrace) = _integrate(trace, Statistics.mean)
Statistics.var( trace::AbstractPDMPTrace) = _integrate(trace, Statistics.var, Statistics.mean(trace))
Statistics.std( trace::AbstractPDMPTrace) = sqrt.(Statistics.var(trace))
Statistics.cov( trace::AbstractPDMPTrace) = _integrate(trace, Statistics.cov, Statistics.mean(trace))
Statistics.cor( trace::AbstractPDMPTrace) = StatsBase.cov2cor!(Statistics.cov(trace))

# NEEDS TESTS!
inclusion_probs(trace::AbstractPDMPTrace) = _integrate(trace, inclusion_probs)
function _integrate_segment(::typeof(inclusion_probs), ::Union{ZigZag, BouncyParticle}, x0, x1, θ0, θ1, t0, t1)

    result = zeros(length(x0))
    for i in eachindex(x0)

        v = θ0[i]
        x = x0[i]

        #=

        Mathematica:

        g[x_] := If[x == 0, 1, 0]
        FullSimplify[Integrate[g[x + s*v], {s, 0, t1 - t0}],
            Assumptions -> {x \[Element] Reals, s > 0, v \[Element] Reals, {t1, t0} \[Element] PositivReals, t1 > t0}]

        -t0+t1  if v==0 && x==0
        0       True

            the code uses the negation, though it could perhaps multiply
        =#

        if !(iszero(x) && iszero(v))
            result[i] = t1 - t0
        end


    end
    return result
end

# model_probs(trace::PDMPTrace)     = Statistics.mean(trace) # for compatibility with other samplers

# _integrate_segment(::typeof(Statistics.mean), ::ZigZag,         x0, x1, θ0, θ1, t0, t1) = (x1 .- x0) * (t1 - t0) / 2
# _integrate_segment(::typeof(Statistics.mean), ::BouncyParticle, x0, x1, θ0, θ1, t0, t1) = (x1 .- x0) * (t1 - t0) / 2

# _integrate_segment(::typeof(Statistics.var), ::ZigZag,         x0, x1, θ0, θ1, t0, t1, μ) = ((x0 .- μ) .^ 2 + (x0 .- μ) .* (x1 .- μ) + (x1 .- μ) .^ 2) * (t1 - t0) / 3
# _integrate_segment(::typeof(Statistics.var), ::BouncyParticle, x0, x1, θ0, θ1, t0, t1, μ) = ((x0 .- μ) .^ 2 + (x0 .- μ) .* (x1 .- μ) + (x1 .- μ) .^ 2) * (t1 - t0) / 3
# NOTE: we only know about event at t0 + duration!
_integrate_segment(::typeof(Statistics.mean), ::ZigZag,         x0, x1, θ0, θ1, t0, t1) = x0 * (t1 - t0) + θ0 * (t1 - t0)^2 / 2
_integrate_segment(::typeof(Statistics.mean), ::BouncyParticle, x0, x1, θ0, θ1, t0, t1) = x0 * (t1 - t0) + θ0 * (t1 - t0)^2 / 2
# _integrate_segment(::typeof(Statistics.mean), ::Boomerang,      x0, x1, θ0, θ1, t0, t1) = x0 * (t1 - t0) + θ0 * (t1 - t0)^2 / 2

_integrate_segment(::typeof(Statistics.var), ::ZigZag,         x0, x1, θ0, θ1, t0, t1, μ) = (-(x0 .- μ) .^ 3 + (-t0 .* θ0 .+ t1 .* θ0 .+ x0 .- μ) .^ 3) ./ (3. * θ0)
_integrate_segment(::typeof(Statistics.var), ::BouncyParticle, x0, x1, θ0, θ1, t0, t1, μ) = (-(x0 .- μ) .^ 3 + (-t0 .* θ0 .+ t1 .* θ0 .+ x0 .- μ) .^ 3) ./ (3. * θ0)#(t0 * θ0 - t1 * θ0 - 2 * x0 + 2 * μ) * (t0 - t1) / 2


function _integrate_segment(::typeof(Statistics.cov), flow::Union{ZigZag, BouncyParticle}, x0, x1, θ0, θ1, t0, t1, μ)

    # define this
    d = length(x0)
    segment_integral = zeros(d, d)
    Δt = t1 - t0

    for j in 1:d, i in j:d

        xᵢ = x0[i]
        xⱼ = x0[j]
        θᵢ = θ0[i]
        θⱼ = θ0[j]
        μᵢ = μ[i]
        μⱼ = μ[j]
        Δᵢ = xᵢ - μᵢ
        Δⱼ = xⱼ - μⱼ

        segment_integral[i, j] = @evalpoly(Δt, 0.0, Δᵢ * Δⱼ, (θⱼ * Δᵢ + θᵢ * Δⱼ) / 2, θⱼ * θᵢ / 3)

        segment_integral[j, i] = segment_integral[i, j]

    end
    return segment_integral
end

function _integrate(trace::AbstractPDMPTrace, f::Function, args...)

    flow = trace.flow
    events = trace.events
    total_time = events[end].time - events[1].time

    # TODO: I think this should use a different iterator, e.g.,
    # dt => event
    # and then just loop?
    # integral = _integrate_segment(f, flow, event, dt args...)
    # not sure how to do this nicely, though,
    # ideally we only define _integrate_segment once, but the best way is to use in-place operations...
    # could also use a initialize function that is called once?

    # integral = sum(
    #     i -> _integrate_segment(
    #         f,
    #         flow,
    #         events[i].position, events[i+1].position,
    #         events[i].velocity, events[i+1].velocity,
    #         events[i].time,     events[i+1].time,
    #         args...
    #     ),
    #     1:length(events)-1
    # )

    # the stuff below works for both traces
    iter = trace
    next = iterate(iter)
    t₀, xt₀, θt₀, _ = next[2]
    next = iterate(iter, next[2])
    t₁, xt₁, θt₁, _ = next[2]
    integral = _integrate_segment(
        f,
        flow,
        xt₀, xt₁,
        θt₀, θt₁,
        t₀,  t₁,
        args...
    )
    t₀, xt₀, θt₀ = t₁, xt₁, θt₁
    next = iterate(iter, next[2])
    while next !== nothing
        t₁, xt₁, θt₁, _ = next[2]
        integral .+= _integrate_segment(
            f,
            flow,
            xt₀, xt₁,
            θt₀, θt₁,
            t₀,  t₁,
            args...
        )
        t₀, xt₀, θt₀ = t₁, xt₁, θt₁
        next = iterate(iter, next[2])
    end

    return integral / total_time
end


# Discretization support for PDMPTrace - ZigZag specific
struct PDMPDiscretize{T, S}
    trace::T
    dt::S
end

_to_range(D::PDMPDiscretize) = first(D.trace).time:D.dt:D.trace.events[end].time

# could technically figure this out?
Base.IteratorSize(::PDMPDiscretize) = Base.HasLength()
Base.length(D::PDMPDiscretize) = length(_to_range(D))

function Base.iterate(D::PDMPDiscretize)
    trace = D.trace
    isempty(trace.events) && return nothing

    # The first yielded state is exactly the first event
    e1 = first(trace)
    t_start, x, θ = e1.time, copy(e1.position), copy(e1.velocity)

    # --- Set up the time range for discretization ---
    t_stop = trace.events[end].time
    t_range = t_start:D.dt:t_stop
    # t_range = _to_range(D)
    # TODO: this needs to handle the case where D.dt > (t_stop - t_start)

    # Get the iterator for the range. We've already "produced" the value
    # at t_start, so we only need the state for the *next* step.
    range_iterator_state = iterate(t_range)[2]

    # The state for our iterator contains:
    # 1. The range object itself.
    # 2. The current state of the range's iterator.
    # 3. The position (x) and velocity (θ) at the previously yielded time step.
    # 4. The index (k) of the next continuous-time event to check.
    k = trace.flow isa FactorizedDynamics ? 1 : 2

    iterator_state = (t_range, range_iterator_state, x, θ, k)

    return (t_start => x), iterator_state
end

function Base.iterate(D::PDMPDiscretize, (t_range, range_state, x_last, θ_last, k))
    trace = D.trace

    # 1. Determine the next discrete time step from the range iterator
    next_range_item = iterate(t_range, range_state)
    isnothing(next_range_item) && return nothing # End of the range
    t_new, new_range_state = next_range_item

    # 2. Initialize our "moving" state from the state at the last discrete step
    # The time of the last step can be found from the new time and step size.
    # This is more robust than storing it.
    t_current = t_new - step(t_range)
    x_current = copy(x_last)
    θ_current = copy(θ_last)
    ξ = SkeletonPoint(x_current, θ_current)
    k_current = k

    # 3. Process all continuous-time events that occurred before t_new
    if D.trace isa FactorizedTrace
        # for the factorized case, we need to call _to_next_event! for every skipped event
        while k_current <= length(trace.events) && trace.events[k_current].time < t_new
            Δt = trace.events[k_current].time - t_current
            move_forward_time!(ξ, Δt, trace.flow)
            t_current = _to_next_event!(x_current, θ_current, trace.events[k_current])
            k_current += 1
        end
        k_current = max(1, k_current - 1) # step back to the last processed event
    else
        # for the other case, we can skip to the last one directly once we find it.
        # this could also be searchsortedlast or so? but not sure if that's faster
        # because we know that the next event is "close"
        while k_current <= length(trace.events) && trace.events[k_current].time < t_new
            k_current += 1
        end
        # k_current -= 1
        k_current = max(1, k_current - 1) # step back to the last event before t_new
        t_current = _to_next_event!(x_current, θ_current, trace.events[k_current])
    end
    # 4. Evolve the state from the time of the last processed event up to t_new
    Δt_final = t_new - t_current
    @assert Δt_final >= 0 "Should be impossible!"
    if Δt_final > 0
        move_forward_time!(ξ, Δt_final, trace.flow)
    end

    # 5. Return the calculated state and the new iterator state for the next call
    new_iterator_state = (t_range, new_range_state, x_current, θ_current, k_current)

    return (t_new => x_current), new_iterator_state
end

function Base.collect(D::PDMPDiscretize)
    collect(t => copy(x) for (t, x) in D)
end
Base.Matrix(D::PDMPDiscretize) = stack(last, D, dims = 1)
