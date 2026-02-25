# _trace_type(flow::ContinuousDynamics, alg::PoissonTimeStrategy) = PDMPTrace
# _trace_type(flow::FactorizedDynamics, alg::PoissonTimeStrategy) = FactorizedTrace
# is this type stable? isfactorized depends only on the type of flow though
_trace_type(flow::ContinuousDynamics, alg::PoissonTimeStrategy) = isfactorized(flow) ? FactorizedTrace : PDMPTrace
# _trace_type(flow::ContinuousDynamics, alg::PoissonTimeStrategy) = PDMPTrace
# _trace_type(flow::FactorizedDynamics, alg::PoissonTimeStrategy) = FactorizedTrace

# _trace_type(flow::FactorizedDynamics, alg::PoissonTimeStrategy) = PDMPTrace
# _trace_type(flow::FactorizedDynamics, alg::Sticky)              = FactorizedTrace




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

const GrowableMatrix{T} = ElasticArray{T, 2, 1, Vector{T}}

struct PDMPTrace{T, U<:ContinuousDynamics, M<:AbstractMatrix{T}} <: AbstractPDMPTrace
    times::Vector{T}
    positions::M    # d × N
    velocities::M   # d × N
    flow::U
end

function PDMPTrace(state::AbstractPDMPState, flow::ContinuousDynamics)
    T = float(typeof(state.t[]))
    d = length(state.ξ.x)
    times = T[state.t[]]
    positions = ElasticMatrix{T}(undef, d, 1)
    velocities = ElasticMatrix{T}(undef, d, 1)
    copyto!(view(positions, :, 1), state.ξ.x)
    copyto!(view(velocities, :, 1), state.ξ.θ)
    PDMPTrace(times, positions, velocities, flow)
end

function PDMPTrace(events::Vector{<:PDMPEvent}, flow::ContinuousDynamics)
    if isempty(events)
        error("Cannot construct PDMPTrace from empty event vector; use make_empty_trace instead")
    end
    T = float(typeof(events[1].time))
    d = length(events[1].position)
    times = T[e.time for e in events]
    positions = ElasticMatrix{T}(undef, d, length(events))
    velocities = ElasticMatrix{T}(undef, d, length(events))
    for (k, e) in enumerate(events)
        copyto!(view(positions, :, k), e.position)
        copyto!(view(velocities, :, k), e.velocity)
    end
    PDMPTrace(times, positions, velocities, flow)
end

function make_empty_trace(::Type{<:PDMPTrace}, state::AbstractPDMPState, flow::ContinuousDynamics)
    T = float(typeof(state.t[]))
    d = length(state.ξ.x)
    PDMPTrace(T[], ElasticMatrix{T}(undef, d, 0), ElasticMatrix{T}(undef, d, 0), flow)
end

# mutable struct FactorizedTrace{T<:FactorizedEvent, U<:FactorizedDynamics, V<:PDMPEvent} <: AbstractPDMPTrace
mutable struct FactorizedTrace{T<:FactorizedEvent, U<:ContinuousDynamics, V<:PDMPEvent} <: AbstractPDMPTrace
    const events::Vector{T}
    const flow::U
    initial_state::V
end
function FactorizedTrace(state::AbstractPDMPState, flow::ContinuousDynamics)
    initial_state = PDMPEvent(state)
    events = Vector{FactorizedEvent{typeof(state.t[]), eltype(state.ξ.x), eltype(state.ξ.θ)}}()
    # FactorizedTrace(events, flow isa PreconditionedDynamics ? flow.dynamics : flow, initial_state)
    FactorizedTrace(events, flow, initial_state)
end
function make_empty_trace(::Type{<:FactorizedTrace}, state::AbstractPDMPState, flow::ContinuousDynamics)
    initial_state = PDMPEvent(-one(state.t[]), state.ξ.x, state.ξ.θ)
    events = Vector{FactorizedEvent{typeof(state.t[]), eltype(state.ξ.x), eltype(state.ξ.θ)}}()
    # FactorizedTrace(events, flow isa PreconditionedDynamics ? flow.dynamics : flow, initial_state)
    FactorizedTrace(events, flow, initial_state)
end

# TODO: use this to test that PDMPTrace and FactorizedTrace give the same results!
# in general we could also go the other way around, but only if trace.flow <: FactorizedDynamics
function PDMPTrace(trace::FactorizedTrace)
    n = length(trace)
    T = float(typeof(trace.initial_state.time))
    d = length(trace.initial_state.position)

    times = Vector{T}(undef, n)
    positions = Matrix{T}(undef, d, n)
    velocities = Matrix{T}(undef, d, n)

    position = copy(trace.initial_state.position)
    velocity = copy(trace.initial_state.velocity)
    times[1] = trace.initial_state.time
    copyto!(view(positions, :, 1), position)
    copyto!(view(velocities, :, 1), velocity)

    ξ = SkeletonPoint(position, velocity)
    t0 = trace.initial_state.time
    for i in eachindex(trace.events)
        Δt = trace.events[i].time - t0
        move_forward_time!(ξ, Δt, trace.flow)
        t0 = trace.events[i].time
        _to_next_event!(position, velocity, trace.events[i])
        k = i + 1
        times[k] = t0
        copyto!(view(positions, :, k), position)
        copyto!(view(velocities, :, k), velocity)
    end
    PDMPTrace(times, positions, velocities, trace.flow)
end

function compact(trace::PDMPTrace{T, U, <:GrowableMatrix}) where {T, U}
    PDMPTrace(trace.times, Matrix(trace.positions), Matrix(trace.velocities), trace.flow)
end
compact(trace::PDMPTrace{T, U, <:Matrix}) where {T, U} = trace
compact(trace::FactorizedTrace) = trace

function Base.first(trace::PDMPTrace)
    PDMPEvent(trace.times[1], trace.positions[:, 1], trace.velocities[:, 1])
end
Base.first(trace::FactorizedTrace) = trace.initial_state

function Base.push!(trace::PDMPTrace{T,U,<:GrowableMatrix}, event::PDMPEvent) where {T,U}
    push!(trace.times, event.time)
    append!(trace.positions, event.position)
    append!(trace.velocities, event.velocity)
    trace
end
function Base.push!(trace::PDMPTrace{T,U,<:GrowableMatrix}, state::AbstractPDMPState) where {T,U}
    push!(trace.times, float(state.t[]))
    append!(trace.positions, state.ξ.x)
    append!(trace.velocities, state.ξ.θ)
    trace
end
Base.push!(trace::FactorizedTrace, event::FactorizedEvent)   = push!(trace.events, event)

function Base.push!(trace::FactorizedTrace, event::AbstractPDMPState, i::Integer)
    # TODO: this needs to figure out which index i changed... but it's much easier to have the caller provide it...
    if trace.initial_state.time >= zero(trace.initial_state.time)
        push!(trace.events, FactorizedEvent(i, event.t[], event.ξ.x[i], event.ξ.θ[i]))
    else
        # @info "setting initial event to " event
        trace.initial_state = PDMPEvent(event)
    end
    # avoid returning a type unstable value
    return trace
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

Base.length(trace::PDMPTrace)       = length(trace.times)
Base.length(trace::FactorizedTrace) = length(trace.events) + 1

function Base.iterate(trace::PDMPTrace)
    isempty(trace.times) && return nothing
    t = trace.times[1]
    x = trace.positions[:, 1]
    θ = trace.velocities[:, 1]
    return t => copy(x), (t, x, θ, 2)
end

function Base.iterate(trace::PDMPTrace, (t, x, θ, k))
    k > length(trace.times) && return nothing
    Δt = trace.times[k] - t
    move_forward_time!(SkeletonPoint(x, θ), Δt, trace.flow)
    copyto!(x, view(trace.positions, :, k))
    copyto!(θ, view(trace.velocities, :, k))
    t = trace.times[k]
    return t => copy(x), (t, x, θ, k + 1)
end

function Base.iterate(trace::FactorizedTrace)
    isempty(trace.events) && return nothing
    e1 = trace.initial_state
    t, x, θ = e1.time, copy(e1.position), copy(e1.velocity)
    return t => copy(x), (t, x, θ, 1)
end

function Base.iterate(trace::FactorizedTrace, (t, x, θ, k))
    k > length(trace.events) && return nothing
    Δt = trace.events[k].time - t
    move_forward_time!(SkeletonPoint(x, θ), Δt, trace.flow)
    t = _to_next_event!(x, θ, trace.events[k])
    return t => copy(x), (t, x, θ, k + 1)
end

Base.collect(trace::AbstractPDMPTrace) = collect(t => x for (t, x) in trace)

event_times(trace::PDMPTrace) = trace.times
event_times(trace::FactorizedTrace) = [trace.initial_state.time; [e.time for e in trace.events]]

# could be a separate file from here?
# shouldn't T1 and T2 always be identical?
struct TraceManager{T}
    main_trace::T
    warmup_trace::T
    t_warmup::Float64
end
function TraceManager(state::AbstractPDMPState, flow::ContinuousDynamics, alg::PoissonTimeStrategy, t_warmup::Real)
    TT = _trace_type(flow, alg)
    main_trace = make_empty_trace(TT, state, flow)
    warmup_trace = make_empty_trace(TT, state, flow)
    TraceManager(main_trace, warmup_trace, float(t_warmup))
end

get_warmup_trace(mgr::TraceManager) = mgr.warmup_trace
get_main_trace(mgr::TraceManager)   = mgr.main_trace

function record_event!(mgr::TraceManager, state, flow, args)
    trace = state.t[] < mgr.t_warmup ? get_warmup_trace(mgr) : get_main_trace(mgr)
    push_trace!(trace, state, flow, args)
    return nothing
end

# Helper to handle the "Factorized" vs "Standard" check cleanly
# function push_trace!(trace, state, flow, args)
#     # @show args, isnothing(args)
#     if isnothing(args)
#         # @show "here1"
#         push!(trace, state)
#     else
#         # @show "here2"
#         # The assertion logic moves here, closer to the data
#         @assert isfactorized(flow) "Flow/Trace mismatch"
#         # @show trace, state, args
#         # push!(trace, state)
#         push!(trace, state, args)
#     end
# end

# needs docs, at least for my future self
push_trace!(trace, state, flow, args::Nothing) = push!(trace, state)
push_trace!(trace::FactorizedTrace, state, flow, args::Integer) = push!(trace, state, args)


# Discretization support for PDMPTrace - ZigZag specific
struct PDMPDiscretize{T, S}
    trace::T
    dt::S
end

# Internal accessors to abstract over PDMPTrace vs FactorizedTrace storage
_isempty_trace(trace::PDMPTrace) = isempty(trace.times)
_isempty_trace(trace::FactorizedTrace) = isempty(trace.events)

_n_raw_events(trace::PDMPTrace) = length(trace.times)
_n_raw_events(trace::FactorizedTrace) = length(trace.events)

_event_time(trace::PDMPTrace, k::Int) = trace.times[k]
_event_time(trace::FactorizedTrace, k::Int) = trace.events[k].time

_last_event_time(trace::PDMPTrace) = trace.times[end]
_last_event_time(trace::FactorizedTrace) = trace.events[end].time

function _apply_event!(x::AbstractVector, θ::AbstractVector, trace::PDMPTrace, k::Int)
    copyto!(x, view(trace.positions, :, k))
    copyto!(θ, view(trace.velocities, :, k))
    trace.times[k]
end
function _apply_event!(x::AbstractVector, θ::AbstractVector, trace::FactorizedTrace, k::Int)
    _to_next_event!(x, θ, trace.events[k])
end

_to_range(D::PDMPDiscretize) = first(D.trace).time:D.dt:_last_event_time(D.trace)

# could technically figure this out?
Base.IteratorSize(::PDMPDiscretize) = Base.HasLength()
Base.length(D::PDMPDiscretize) = length(_to_range(D))

function Base.iterate(D::PDMPDiscretize)
    trace = D.trace
    _isempty_trace(trace) && return nothing

    # The first yielded state is exactly the first event
    e1 = first(trace)
    t_start, x, θ = e1.time, copy(e1.position), copy(e1.velocity)

    # --- Set up the time range for discretization ---
    t_stop = _last_event_time(trace)
    t_range = t_start:D.dt:t_stop

    # Get the iterator for the range. We've already "produced" the value
    # at t_start, so we only need the state for the *next* step.
    range_iterator_state = iterate(t_range)[2]

    k = trace isa FactorizedTrace ? 1 : 2

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
    t_current = t_new - step(t_range)
    x_current = copy(x_last)
    θ_current = copy(θ_last)
    ξ = SkeletonPoint(x_current, θ_current)
    k_current = k

    # 3. Process all continuous-time events that occurred before t_new
    if D.trace isa FactorizedTrace
        # for the factorized case, we need to call _to_next_event! for every skipped event
        while k_current <= _n_raw_events(trace) && _event_time(trace, k_current) < t_new
            Δt = _event_time(trace, k_current) - t_current
            move_forward_time!(ξ, Δt, trace.flow)
            t_current = _apply_event!(x_current, θ_current, trace, k_current)
            k_current += 1
        end
    else
        # For the non-factorized case, we can skip to the last event directly.
        found_event = false
        last_before = k_current
        while k_current <= _n_raw_events(trace) && _event_time(trace, k_current) < t_new
            last_before = k_current
            found_event = true
            k_current += 1
        end
        if found_event
            t_current = _apply_event!(x_current, θ_current, trace, last_before)
        end
    end
    # 4. Evolve the state from the time of the last processed event up to t_new
    Δt_final = t_new - t_current
    @assert Δt_final >= 0 "Should be impossible!"
    if ispositive(Δt_final)
        move_forward_time!(ξ, Δt_final, trace.flow)
    end

    # 5. Return the calculated state and the new iterator state for the next call
    new_iterator_state = (t_range, new_range_state, x_current, θ_current, k_current)

    return (t_new => x_current), new_iterator_state
end

function Base.collect(D::PDMPDiscretize)
    collect(t => x for (t, x) in D)
end
Base.Matrix(D::PDMPDiscretize) = stack(last, D, dims = 1)
