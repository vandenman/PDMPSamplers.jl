# _trace_type(flow::ContinuousDynamics, alg::PoissonTimeStrategy) = PDMPTrace
# _trace_type(flow::FactorizedDynamics, alg::PoissonTimeStrategy) = FactorizedTrace
# is this type stable? isfactorized depends only on the type of flow though
_trace_type(flow::ContinuousDynamics, alg::PoissonTimeStrategy) = isfactorized(flow) ? FactorizedTrace : PDMPTrace
_trace_type(flow::ContinuousDynamics, ::Union{Sticky,AggregateSticky}) =
    isfactorized(flow) ? FactorizedTrace : PDMPTrace
# Sticky ZigZag events still alter only one coordinate at a time.  Preserve
# that coordinate in the event handler and use the sparse factorized trace;
# forcing this case through PDMPTrace stores two dense d-vectors (and a mask)
# per event and then copies both matrices again in `compact`.
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

struct PDMPTrace{T, U<:ContinuousDynamics, M<:AbstractMatrix{T}, F} <: AbstractPDMPTrace
    times::Vector{T}
    positions::M    # d × N
    velocities::M   # d × N
    flow::U
    free_masks::F   # nothing for ordinary traces; d × N for sticky traces
end

PDMPTrace(times::Vector{T}, positions::M, velocities::M, flow::U) where {T,U<:ContinuousDynamics,M<:AbstractMatrix{T}} =
    PDMPTrace(times, positions, velocities, flow, nothing)

function PDMPTrace(state::AbstractPDMPState, flow::ContinuousDynamics)
    T = float(typeof(state.t[]))
    d = length(state.ξ.x)
    times = T[state.t[]]
    positions = ElasticMatrix{T}(undef, d, 1)
    velocities = ElasticMatrix{T}(undef, d, 1)
    copyto!(view(positions, :, 1), state.ξ.x)
    copyto!(view(velocities, :, 1), state.ξ.θ)
    free_masks = if state isa StickyPDMPState
        masks = ElasticMatrix{Bool}(undef, d, 1)
        copyto!(view(masks, :, 1), state.free)
        masks
    else
        nothing
    end
    PDMPTrace(times, positions, velocities, flow, free_masks)
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
    free_masks = state isa StickyPDMPState ? ElasticMatrix{Bool}(undef, d, 0) : nothing
    PDMPTrace(T[], ElasticMatrix{T}(undef, d, 0), ElasticMatrix{T}(undef, d, 0), flow, free_masks)
end

# mutable struct FactorizedTrace{T<:FactorizedEvent, U<:FactorizedDynamics, V<:PDMPEvent} <: AbstractPDMPTrace
mutable struct FactorizedTrace{T<:FactorizedEvent, U<:ContinuousDynamics, V<:PDMPEvent} <: AbstractPDMPTrace
    const events::Vector{T}
    const flow::U
    initial_state::V
    last_state::V
    last_state_valid::Bool
    bounds_cache::Dict{Int, Tuple{Float64, Float64}}
end

function FactorizedTrace(state::AbstractPDMPState, flow::ContinuousDynamics)
    initial_state = PDMPEvent(state)
    events = Vector{FactorizedEvent{typeof(state.t[]), eltype(state.ξ.x), eltype(state.ξ.θ)}}()
    FactorizedTrace(events, flow, initial_state, initial_state, true, Dict{Int, Tuple{Float64, Float64}}())
end
function make_empty_trace(::Type{<:FactorizedTrace}, state::AbstractPDMPState, flow::ContinuousDynamics)
    initial_state = PDMPEvent(-one(state.t[]), state.ξ.x, state.ξ.θ)
    events = Vector{FactorizedEvent{typeof(state.t[]), eltype(state.ξ.x), eltype(state.ξ.θ)}}()
    FactorizedTrace(events, flow, initial_state, initial_state, false, Dict{Int, Tuple{Float64, Float64}}())
end

function _invalidate_factorized_caches!(trace::FactorizedTrace)
    empty!(trace.bounds_cache)
    return trace
end

function _replay_last_state(trace::FactorizedTrace)
    e1 = trace.initial_state
    x, θ = copy(e1.position), copy(e1.velocity)
    t = e1.time
    for event in trace.events
        Δt = event.time - t
        move_forward_time!(SkeletonPoint(x, θ), Δt, trace.flow)
        _to_next_event!(x, θ, event)
        t = event.time
    end
    return PDMPEvent(t, x, θ)
end

function _advance_last_state(trace::FactorizedTrace, event::FactorizedEvent)
    trace.last_state_valid || return trace
    e = trace.last_state
    x, θ = copy(e.position), copy(e.velocity)
    Δt = event.time - e.time
    move_forward_time!(SkeletonPoint(x, θ), Δt, trace.flow)
    _to_next_event!(x, θ, event)
    trace.last_state = PDMPEvent(event.time, x, θ)
    return trace
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
    masks = trace.free_masks === nothing ? nothing : Matrix(trace.free_masks)
    PDMPTrace(trace.times, Matrix(trace.positions), Matrix(trace.velocities), trace.flow, masks)
end
compact(trace::PDMPTrace{T, U, <:Matrix}) where {T, U} = trace
compact(trace::FactorizedTrace) = trace

# ---------------------------------------------------------------------------
# Bounded, typed event storage
# ---------------------------------------------------------------------------

"""
    StreamingTraceStorage(directory; buffer_events=4096)

Opt-in bounded trace storage. Sampling writes immutable typed-event chunks to
`directory`; flushing a full buffer only performs file I/O. It never calls a
sampler restart, advances the flow, changes a clock/reference, or uses an RNG.

The format stores full effective velocities only for events which can change
all free velocities (BPS/Boomerang reflections, refreshment, and dynamics
adaptation). Coordinate events store one coordinate. Computational horizons
and position-anchor boundaries are observed for elapsed-time bookkeeping but
are not physical trace records.
"""
struct StreamingTraceStorage
    directory::String
    buffer_events::Int
    function StreamingTraceStorage(directory::AbstractString;
            buffer_events::Integer=4096)
        buffer_events >= 1 || throw(ArgumentError(
            "streaming trace buffer_events must be positive"))
        new(abspath(String(directory)), Int(buffer_events))
    end
end

const _STREAM_TRACE_MAGIC = UInt8[0x50, 0x44, 0x4d, 0x50, 0x54, 0x59, 0x50, 0x31]
const _STREAM_EVENT_REFLECT_FULL = UInt8(1)
const _STREAM_EVENT_REFLECT_COORD = UInt8(2)
const _STREAM_EVENT_REFRESH = UInt8(3)
const _STREAM_EVENT_FREEZE = UInt8(4)
const _STREAM_EVENT_RELEASE = UInt8(5)
const _STREAM_EVENT_DYNAMICS = UInt8(6)

struct StreamingTraceChunk
    path::String
    n_events::Int
    n_full::Int
    first_time::Float64
    last_time::Float64
    bytes::Int64
end

mutable struct StreamingChunkIndex <: AbstractVector{StreamingTraceChunk}
    directory::String
    count::Int
    total_bytes::Int64
end
Base.IndexStyle(::Type{StreamingChunkIndex}) = IndexLinear()
Base.size(index::StreamingChunkIndex) = (index.count,)
Base.length(index::StreamingChunkIndex) = index.count
function Base.getindex(index::StreamingChunkIndex, i::Int)
    checkbounds(index, i)
    path = joinpath(index.directory,
        "chunk_" * lpad(string(i), 8, '0') * ".bin")
    return _stream_chunk_metadata(path)
end
function Base.push!(index::StreamingChunkIndex, chunk::StreamingTraceChunk)
    expected = joinpath(index.directory,
        "chunk_" * lpad(string(index.count + 1), 8, '0') * ".bin")
    abspath(chunk.path) == abspath(expected) || throw(ArgumentError(
        "streaming chunk is not the next immutable chunk"))
    index.count += 1
    index.total_bytes += chunk.bytes
    return index
end

mutable struct StreamingEventBuffer
    times::Vector{Float64}
    kinds::Vector{UInt8}
    coordinates::Vector{Int32}
    positions::Vector{Float64}
    velocities::Vector{Float64}
    stored_velocities::Vector{Float64}
    full_slots::Vector{Int32}
    full_effective_velocities::Matrix{Float64}
    n_events::Int
    n_full::Int
end

function StreamingEventBuffer(d::Integer, capacity::Integer)
    StreamingEventBuffer(Vector{Float64}(undef, capacity),
        Vector{UInt8}(undef, capacity), Vector{Int32}(undef, capacity),
        Vector{Float64}(undef, capacity), Vector{Float64}(undef, capacity),
        Vector{Float64}(undef, capacity), Vector{Int32}(undef, capacity),
        Matrix{Float64}(undef, d, capacity), 0, 0)
end

mutable struct StreamingWarmupGrid
    times::Vector{Float64}
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    filled::BitVector
    next_index::Int
    last_time::Float64
    last_position::Vector{Float64}
    last_velocity::Vector{Float64}
    last_free::BitVector
end

mutable struct StreamingRawMoments
    total_time::Float64
    sum_x::Vector{Float64}
    sum_x2::Vector{Float64}
    free_time::Vector{Float64}
    last_time::Float64
    last_position::Vector{Float64}
    last_velocity::Vector{Float64}
    last_free::BitVector
end

function StreamingRawMoments(state::AbstractPDMPState)
    d = length(state.ξ.x)
    free = state isa StickyPDMPState ? copy(state.free) : trues(d)
    StreamingRawMoments(0.0, zeros(d), zeros(d), zeros(d), Float64(state.t[]),
        copy(state.ξ.x), copy(state.ξ.θ), free)
end

function StreamingWarmupGrid(state::AbstractPDMPState, start_time::Real,
        end_time::Real)
    # These are the two fixed grids consumed by the OMRF eight-anchor and
    # node-source warmup finalizers. Keeping their union is bounded in both
    # event count and physical dimension.
    times = unique!(sort!(vcat(
        collect(range(start_time + 0.5 * (end_time - start_time), end_time;
            length=64)),
        collect(range(start_time + 0.5 * (end_time - start_time), end_time;
            length=128)))))
    d = length(state.ξ.x)
    free = state isa StickyPDMPState ? copy(state.free) : trues(d)
    StreamingWarmupGrid(times, Matrix{Float64}(undef, d, length(times)),
        Matrix{Float64}(undef, d, length(times)), falses(length(times)), 1,
        Float64(state.t[]), copy(state.ξ.x), copy(state.ξ.θ), free)
end

mutable struct StreamingPDMPTrace{F<:ContinuousDynamics} <: AbstractPDMPTrace
    flow::F
    directory::String
    prefix::String
    buffer::StreamingEventBuffer
    chunks::StreamingChunkIndex
    initial_state::Any
    terminal_state::Any
    warmup_grid::Union{Nothing,StreamingWarmupGrid}
    moments::StreamingRawMoments
    physical_events::Int
    computational_boundaries::Int
    buffer_maximum::Int
    finalized::Bool
end

function StreamingPDMPTrace(state::AbstractPDMPState, flow::ContinuousDynamics,
        storage::StreamingTraceStorage, prefix::AbstractString;
        warmup_end::Union{Nothing,Real}=nothing)
    directory = joinpath(storage.directory, String(prefix))
    mkpath(directory)
    grid = warmup_end === nothing ? nothing :
        StreamingWarmupGrid(state, state.t[], warmup_end)
    trace = StreamingPDMPTrace(flow, directory, String(prefix),
        StreamingEventBuffer(length(state.ξ.x), storage.buffer_events),
        StreamingChunkIndex(directory, 0, 0), nothing, nothing, grid,
        StreamingRawMoments(state), 0, 0, 0, false)
    # A zero-event (or zero-duration) warmup is still a valid source for an
    # adaptation finalizer. Main starts later and is installed explicitly at
    # retained-phase entry, so it must not capture the pre-warmup state here.
    warmup_end === nothing || (trace.initial_state =
        _stream_terminal_state(state, flow))
    return trace
end

@inline function _stream_terminal_state(state::AbstractPDMPState,
        flow::ContinuousDynamics)
    return PDMPTerminalState(state, flow)
end

function _stream_move!(x, velocity, free, elapsed, flow)
    point = SkeletonPoint(x, velocity)
    if _underlying_flow(flow) isa AnyBoomerang && free !== nothing
        move_forward_time!(point, elapsed, _underlying_flow(flow), free)
    else
        move_forward_time!(point, elapsed, flow)
    end
    return nothing
end

function _stream_observe_warmup!(trace::StreamingPDMPTrace,
        state::AbstractPDMPState)
    grid = trace.warmup_grid
    grid === nothing && return nothing
    current_time = Float64(state.t[])
    current_time + 64eps(max(abs(current_time), 1.0)) >= grid.last_time ||
        throw(ArgumentError("streaming warmup observations are not chronological"))
    # Fill strict interior points from the previous outgoing physical state.
    while grid.next_index <= length(grid.times) &&
            grid.times[grid.next_index] < current_time
        target = grid.times[grid.next_index]
        x = copy(grid.last_position)
        velocity = copy(grid.last_velocity)
        _stream_move!(x, velocity, grid.last_free,
            target - grid.last_time, trace.flow)
        copyto!(view(grid.positions, :, grid.next_index), x)
        copyto!(view(grid.velocities, :, grid.next_index), velocity)
        grid.filled[grid.next_index] = true
        grid.next_index += 1
    end
    copyto!(grid.last_position, state.ξ.x)
    copyto!(grid.last_velocity, state.ξ.θ)
    if state isa StickyPDMPState
        copyto!(grid.last_free, state.free)
    else
        fill!(grid.last_free, true)
    end
    grid.last_time = current_time
    # A grid point exactly on an event uses its outgoing state, matching
    # searchsortedlast on the established dense trace.
    while grid.next_index <= length(grid.times) &&
            grid.times[grid.next_index] == current_time
        copyto!(view(grid.positions, :, grid.next_index), state.ξ.x)
        copyto!(view(grid.velocities, :, grid.next_index), state.ξ.θ)
        grid.filled[grid.next_index] = true
        grid.next_index += 1
    end
    return nothing
end

function _stream_accumulate_moments!(trace::StreamingPDMPTrace,
        state::AbstractPDMPState)
    moments = trace.moments
    current_time = Float64(state.t[])
    elapsed = current_time - moments.last_time
    elapsed >= -64eps(max(abs(current_time), 1.0)) || throw(ArgumentError(
        "streaming moment observations are not chronological"))
    if elapsed > 0
        base = _underlying_flow(trace.flow)
        if base isa AnyBoomerang
            sine, cosine = sincos(elapsed)
            sine2 = sine * sine
            sin2 = sin(2elapsed)
            @inbounds for i in eachindex(moments.sum_x)
                x0 = moments.last_position[i]
                if !moments.last_free[i]
                    moments.sum_x[i] += x0 * elapsed
                    moments.sum_x2[i] += x0 * x0 * elapsed
                    continue
                end
                velocity = moments.last_velocity[i]
                mean = base.μ[i]
                a = x0 - mean
                omc = 1 - cosine
                moments.sum_x[i] +=
                    a * sine + velocity * omc + mean * elapsed
                moments.sum_x2[i] +=
                    a * a * (elapsed / 2 + sin2 / 4) +
                    velocity * velocity * (elapsed / 2 - sin2 / 4) +
                    mean * mean * elapsed + a * velocity * sine2 +
                    2a * mean * sine + 2velocity * mean * omc
            end
        else
            elapsed2 = elapsed * elapsed
            elapsed3 = elapsed2 * elapsed
            @inbounds for i in eachindex(moments.sum_x)
                x0 = moments.last_position[i]
                velocity = moments.last_free[i] ? moments.last_velocity[i] : 0.0
                moments.sum_x[i] += x0 * elapsed + 0.5 * velocity * elapsed2
                moments.sum_x2[i] += x0 * x0 * elapsed +
                    x0 * velocity * elapsed2 +
                    velocity * velocity * elapsed3 / 3
            end
        end
        @inbounds for i in eachindex(moments.free_time)
            moments.last_free[i] && (moments.free_time[i] += elapsed)
        end
        moments.total_time += elapsed
    end
    moments.last_time = current_time
    copyto!(moments.last_position, state.ξ.x)
    copyto!(moments.last_velocity, state.ξ.θ)
    state isa StickyPDMPState ? copyto!(moments.last_free, state.free) :
        fill!(moments.last_free, true)
    return nothing
end

function _stream_chunk_path(trace::StreamingPDMPTrace, index::Integer)
    joinpath(trace.directory, "chunk_" * lpad(string(index), 8, '0') * ".bin")
end

function _flush_streaming_trace!(trace::StreamingPDMPTrace)
    buffer = trace.buffer
    n = buffer.n_events
    iszero(n) && return trace
    path = _stream_chunk_path(trace, length(trace.chunks) + 1)
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do io
        write(io, _STREAM_TRACE_MAGIC)
        write(io, Int32(size(buffer.full_effective_velocities, 1)))
        write(io, Int32(n))
        write(io, Int32(buffer.n_full))
        write(io, view(buffer.times, 1:n))
        write(io, view(buffer.kinds, 1:n))
        write(io, view(buffer.coordinates, 1:n))
        write(io, view(buffer.positions, 1:n))
        write(io, view(buffer.velocities, 1:n))
        write(io, view(buffer.stored_velocities, 1:n))
        write(io, view(buffer.full_slots, 1:n))
        write(io, view(buffer.full_effective_velocities, :, 1:buffer.n_full))
        flush(io)
    end
    mv(temporary, path; force=false)
    bytes = filesize(path)
    push!(trace.chunks, StreamingTraceChunk(path, n, buffer.n_full,
        buffer.times[1], buffer.times[n], bytes))
    buffer.n_events = 0
    buffer.n_full = 0
    return trace
end

@inline function _stream_effective_velocity(state::AbstractPDMPState, i)
    state isa StickyPDMPState && !state.free[i] ?
        state.stored_velocity[i] : state.ξ.θ[i]
end

function _stream_record_full!(trace::StreamingPDMPTrace,
        state::AbstractPDMPState, kind::UInt8)
    buffer = trace.buffer
    buffer.n_events == length(buffer.times) && _flush_streaming_trace!(trace)
    event = (buffer.n_events += 1)
    slot = (buffer.n_full += 1)
    buffer.times[event] = state.t[]
    buffer.kinds[event] = kind
    buffer.coordinates[event] = 0
    buffer.positions[event] = NaN
    buffer.velocities[event] = NaN
    buffer.stored_velocities[event] = NaN
    buffer.full_slots[event] = slot
    @inbounds for i in eachindex(state.ξ.θ)
        buffer.full_effective_velocities[i, slot] =
            _stream_effective_velocity(state, i)
    end
    trace.physical_events += 1
    trace.buffer_maximum = max(trace.buffer_maximum, buffer.n_events)
    return nothing
end

function _stream_record_coordinate!(trace::StreamingPDMPTrace,
        state::AbstractPDMPState, kind::UInt8, coordinate::Integer)
    buffer = trace.buffer
    buffer.n_events == length(buffer.times) && _flush_streaming_trace!(trace)
    event = (buffer.n_events += 1)
    buffer.times[event] = state.t[]
    buffer.kinds[event] = kind
    buffer.coordinates[event] = Int32(coordinate)
    buffer.positions[event] = state.ξ.x[coordinate]
    buffer.velocities[event] = state.ξ.θ[coordinate]
    buffer.stored_velocities[event] = state isa StickyPDMPState ?
        state.stored_velocity[coordinate] : 0.0
    buffer.full_slots[event] = 0
    trace.physical_events += 1
    trace.buffer_maximum = max(trace.buffer_maximum, buffer.n_events)
    return nothing
end

function begin_trace_phase!(trace::StreamingPDMPTrace,
        state::AbstractPDMPState, flow::ContinuousDynamics)
    trace.initial_state === nothing || return trace
    trace.initial_state = _stream_terminal_state(state, flow)
    trace.moments = StreamingRawMoments(state)
    _stream_observe_warmup!(trace, state)
    return trace
end
begin_trace_phase!(trace::AbstractPDMPTrace, state, flow) = trace

function finish_trace_phase!(trace::StreamingPDMPTrace,
        state::AbstractPDMPState, flow::ContinuousDynamics)
    _stream_accumulate_moments!(trace, state)
    _stream_observe_warmup!(trace, state)
    trace.terminal_state = _stream_terminal_state(state, flow)
    _flush_streaming_trace!(trace)
    trace.finalized = true
    return trace
end
finish_trace_phase!(trace::AbstractPDMPTrace, state, flow) = trace

function _record_streaming_event!(trace::StreamingPDMPTrace,
        state::AbstractPDMPState, flow::ContinuousDynamics, args,
        event_type::Symbol)
    _stream_accumulate_moments!(trace, state)
    _stream_observe_warmup!(trace, state)
    if event_type in (:horizon_hit, :anchor_selection_boundary)
        trace.computational_boundaries += 1
        return nothing
    elseif event_type === :refresh
        return _stream_record_full!(trace, state, _STREAM_EVENT_REFRESH)
    elseif event_type === :dynamics_adaptation
        return _stream_record_full!(trace, state, _STREAM_EVENT_DYNAMICS)
    elseif event_type === :sticky
        coordinate = Int(args)
        kind = state isa StickyPDMPState && state.free[coordinate] ?
            _STREAM_EVENT_RELEASE : _STREAM_EVENT_FREEZE
        return _stream_record_coordinate!(trace, state, kind, coordinate)
    elseif event_type === :reflect
        if args isa Integer
            return _stream_record_coordinate!(trace, state,
                _STREAM_EVENT_REFLECT_COORD, args)
        end
        return _stream_record_full!(trace, state, _STREAM_EVENT_REFLECT_FULL)
    elseif event_type === :initial
        return begin_trace_phase!(trace, state, flow)
    end
    throw(ArgumentError("unsupported streaming trace event type $event_type"))
end

function streaming_trace_manifest(trace::StreamingPDMPTrace)
    trace.finalized || _flush_streaming_trace!(trace)
    return (format="pdmpsamplers_typed_stream_v1",
        directory=trace.directory,
        chunk_count=length(trace.chunks),
        output_bytes=trace.chunks.total_bytes,
        initial_state=trace.initial_state,
        terminal_state=trace.terminal_state,
        physical_events=trace.physical_events,
        computational_boundaries=trace.computational_boundaries,
        buffer_maximum=trace.buffer_maximum)
end

compact(trace::StreamingPDMPTrace) = trace

function Statistics.mean(trace::StreamingPDMPTrace)
    trace.moments.total_time > 0 || error(
        "Cannot compute a mean on a streaming trace without elapsed time")
    return trace.moments.sum_x ./ trace.moments.total_time
end

function Statistics.mean!(out::AbstractVector, trace::StreamingPDMPTrace)
    copyto!(out, Statistics.mean(trace))
    return out
end

function Statistics.var(trace::StreamingPDMPTrace, means::AbstractVector)
    trace.moments.total_time > 0 || error(
        "Cannot compute a variance on a streaming trace without elapsed time")
    result = trace.moments.sum_x2 ./ trace.moments.total_time .- means .^ 2
    @. result = max(result, 0.0)
    return result
end

Statistics.var(trace::StreamingPDMPTrace) =
    Statistics.var(trace, Statistics.mean(trace))
Statistics.std(trace::StreamingPDMPTrace) = sqrt.(Statistics.var(trace))
Statistics.std(trace::StreamingPDMPTrace, means::AbstractVector) =
    sqrt.(Statistics.var(trace, means))

function inclusion_probs(trace::StreamingPDMPTrace)
    trace.moments.total_time > 0 || error(
        "Cannot compute inclusion probabilities without elapsed time")
    return trace.moments.free_time ./ trace.moments.total_time
end

struct LoadedStreamingChunk
    times::Vector{Float64}
    kinds::Vector{UInt8}
    coordinates::Vector{Int32}
    positions::Vector{Float64}
    velocities::Vector{Float64}
    stored_velocities::Vector{Float64}
    full_slots::Vector{Int32}
    full_effective_velocities::Matrix{Float64}
end

function _stream_chunk_metadata(path::AbstractString)
    open(path, "r") do io
        magic = Vector{UInt8}(undef, length(_STREAM_TRACE_MAGIC))
        read!(io, magic)
        magic == _STREAM_TRACE_MAGIC || throw(ArgumentError(
            "invalid streaming trace chunk header at $path"))
        read(io, Int32) # dimension; checked when the chunk is loaded
        n = Int(read(io, Int32))
        n_full = Int(read(io, Int32))
        header_bytes = length(_STREAM_TRACE_MAGIC) + 3sizeof(Int32)
        seek(io, header_bytes)
        first_time = n == 0 ? NaN : read(io, Float64)
        last_time = if n == 0
            NaN
        else
            seek(io, header_bytes + (n - 1)sizeof(Float64))
            read(io, Float64)
        end
        return StreamingTraceChunk(abspath(String(path)), n, n_full,
            first_time, last_time, filesize(path))
    end
end

function _restore_streaming_moments!(trace::StreamingPDMPTrace)
    initial = trace.initial_state
    d = length(initial.position)
    moments = StreamingRawMoments(0.0, zeros(d), zeros(d), zeros(d),
        initial.t, copy(initial.position), copy(initial.physical_velocity),
        copy(initial.free))
    base = _underlying_flow(trace.flow)
    _foreach_streaming_segment(trace) do time0, time1, x0, x1,
            velocity0, velocity1, free
        elapsed = time1 - time0
        if base isa AnyBoomerang
            sine, cosine = sincos(elapsed)
            sine2 = sine * sine
            sin2 = sin(2elapsed)
            @inbounds for i in 1:d
                if !free[i]
                    moments.sum_x[i] += x0[i] * elapsed
                    moments.sum_x2[i] += x0[i]^2 * elapsed
                    continue
                end
                a = x0[i] - base.μ[i]
                v = velocity0[i]
                omc = 1 - cosine
                moments.sum_x[i] += a * sine + v * omc + base.μ[i] * elapsed
                moments.sum_x2[i] +=
                    a^2 * (elapsed / 2 + sin2 / 4) +
                    v^2 * (elapsed / 2 - sin2 / 4) +
                    base.μ[i]^2 * elapsed + a * v * sine2 +
                    2a * base.μ[i] * sine + 2v * base.μ[i] * omc
            end
        else
            elapsed2 = elapsed^2
            elapsed3 = elapsed^3
            @inbounds for i in 1:d
                v = free[i] ? velocity0[i] : 0.0
                moments.sum_x[i] += x0[i] * elapsed + 0.5v * elapsed2
                moments.sum_x2[i] += x0[i]^2 * elapsed +
                    x0[i] * v * elapsed2 + v^2 * elapsed3 / 3
            end
        end
        @inbounds for i in 1:d
            free[i] && (moments.free_time[i] += elapsed)
        end
        moments.total_time += elapsed
    end
    terminal = trace.terminal_state
    moments.last_time = terminal.t
    copyto!(moments.last_position, terminal.position)
    copyto!(moments.last_velocity, terminal.physical_velocity)
    copyto!(moments.last_free, terminal.free)
    trace.moments = moments
    return trace
end

function _restore_streaming_trace(chunk_paths::AbstractVector,
        flow::ContinuousDynamics, initial_state, terminal_state;
        physical_events::Integer, computational_boundaries::Integer,
        buffer_maximum::Integer)
    paths = String.(chunk_paths)
    all(isfile, paths) || throw(ArgumentError(
        "one or more streaming trace chunks are unavailable"))
    if !isempty(paths)
        expected = [joinpath(dirname(first(paths)),
            "chunk_" * lpad(string(i), 8, '0') * ".bin")
            for i in eachindex(paths)]
        all(abspath.(paths) .== abspath.(expected)) || throw(ArgumentError(
            "streaming chunks are not one contiguous index"))
    end
    return _restore_streaming_trace(isempty(paths) ? "" :
        dirname(first(paths)), length(paths), flow, initial_state,
        terminal_state; physical_events, computational_boundaries,
        buffer_maximum)
end

function _restore_streaming_trace(directory::AbstractString,
        chunk_count::Integer, flow::ContinuousDynamics, initial_state,
        terminal_state; physical_events::Integer,
        computational_boundaries::Integer, buffer_maximum::Integer)
    chunk_count >= 0 || throw(ArgumentError("negative streaming chunk count"))
    resolved_directory = abspath(String(directory))
    total_bytes = Int64(0)
    for i in 1:chunk_count
        path = joinpath(resolved_directory,
            "chunk_" * lpad(string(i), 8, '0') * ".bin")
        isfile(path) || throw(ArgumentError("streaming chunk is unavailable: $path"))
        total_bytes += filesize(path)
    end
    d = length(initial_state.position)
    index = StreamingChunkIndex(resolved_directory, Int(chunk_count),
        total_bytes)
    trace = StreamingPDMPTrace(flow,
        resolved_directory, "restored",
        StreamingEventBuffer(d, 1), index,
        initial_state, terminal_state, nothing,
        StreamingRawMoments(0.0, zeros(d), zeros(d), zeros(d),
            initial_state.t, copy(initial_state.position),
            copy(initial_state.physical_velocity), copy(initial_state.free)),
        Int(physical_events), Int(computational_boundaries),
        Int(buffer_maximum), true)
    return _restore_streaming_moments!(trace)
end

function _load_streaming_chunk(chunk::StreamingTraceChunk)
    open(chunk.path, "r") do io
        magic = Vector{UInt8}(undef, length(_STREAM_TRACE_MAGIC))
        read!(io, magic)
        magic == _STREAM_TRACE_MAGIC || throw(ArgumentError(
            "invalid streaming trace chunk header at $(chunk.path)"))
        d = Int(read(io, Int32))
        n = Int(read(io, Int32))
        n_full = Int(read(io, Int32))
        n == chunk.n_events && n_full == chunk.n_full ||
            throw(ArgumentError("streaming trace chunk metadata mismatch"))
        times = Vector{Float64}(undef, n); read!(io, times)
        kinds = Vector{UInt8}(undef, n); read!(io, kinds)
        coordinates = Vector{Int32}(undef, n); read!(io, coordinates)
        positions = Vector{Float64}(undef, n); read!(io, positions)
        velocities = Vector{Float64}(undef, n); read!(io, velocities)
        stored = Vector{Float64}(undef, n); read!(io, stored)
        slots = Vector{Int32}(undef, n); read!(io, slots)
        effective = Matrix{Float64}(undef, d, n_full)
        read!(io, effective)
        eof(io) || throw(ArgumentError(
            "streaming trace chunk has trailing bytes at $(chunk.path)"))
        return LoadedStreamingChunk(times, kinds, coordinates, positions,
            velocities, stored, slots, effective)
    end
end

mutable struct StreamingTraceCursor
    x::Vector{Float64}
    velocity::Vector{Float64}
    stored_velocity::Vector{Float64}
    free::BitVector
    chunk_index::Int
    event_index::Int
    chunk::Union{Nothing,LoadedStreamingChunk}
    terminal_returned::Bool
end

function _stream_cursor(trace::StreamingPDMPTrace)
    initial = trace.initial_state
    initial === nothing && error("streaming trace has no initial state")
    return StreamingTraceCursor(copy(initial.position),
        copy(initial.physical_velocity), copy(initial.stored_frozen_velocity),
        copy(initial.free), 1, 0, nothing, false)
end

function _stream_next_chunk!(cursor::StreamingTraceCursor,
        trace::StreamingPDMPTrace)
    cursor.chunk_index > length(trace.chunks) && return false
    cursor.chunk = _load_streaming_chunk(trace.chunks[cursor.chunk_index])
    cursor.chunk_index += 1
    cursor.event_index = 0
    return true
end

function _stream_apply_event!(cursor::StreamingTraceCursor,
        chunk::LoadedStreamingChunk, event::Int)
    kind = chunk.kinds[event]
    coordinate = Int(chunk.coordinates[event])
    if kind in (_STREAM_EVENT_REFLECT_FULL, _STREAM_EVENT_REFRESH,
            _STREAM_EVENT_DYNAMICS)
        effective = view(chunk.full_effective_velocities, :,
            Int(chunk.full_slots[event]))
        @inbounds for i in eachindex(cursor.velocity)
            if cursor.free[i]
                cursor.velocity[i] = effective[i]
                cursor.stored_velocity[i] = 0.0
            else
                cursor.velocity[i] = 0.0
                cursor.stored_velocity[i] = effective[i]
            end
        end
    elseif kind in (_STREAM_EVENT_REFLECT_COORD, _STREAM_EVENT_FREEZE,
            _STREAM_EVENT_RELEASE)
        cursor.x[coordinate] = chunk.positions[event]
        cursor.velocity[coordinate] = chunk.velocities[event]
        cursor.stored_velocity[coordinate] = chunk.stored_velocities[event]
        kind === _STREAM_EVENT_FREEZE && (cursor.free[coordinate] = false)
        kind === _STREAM_EVENT_RELEASE && (cursor.free[coordinate] = true)
    else
        throw(ArgumentError("unknown streaming trace event kind $kind"))
    end
    return nothing
end

function Base.iterate(trace::StreamingPDMPTrace)
    trace.finalized || _flush_streaming_trace!(trace)
    cursor = _stream_cursor(trace)
    initial = trace.initial_state
    return initial.t => copy(cursor.x),
        (initial.t, cursor.x, cursor.velocity, cursor)
end

function Base.iterate(trace::StreamingPDMPTrace, iteration_state)
    time, x, velocity, cursor = iteration_state
    while cursor.chunk === nothing ||
            cursor.event_index >= length(cursor.chunk.times)
        _stream_next_chunk!(cursor, trace) || break
    end
    if cursor.chunk !== nothing &&
            cursor.event_index < length(cursor.chunk.times)
        cursor.event_index += 1
        event = cursor.event_index
        chunk = cursor.chunk
        event_time = chunk.times[event]
        _stream_move!(x, velocity, cursor.free, event_time - time, trace.flow)
        _stream_apply_event!(cursor, chunk, event)
        return event_time => copy(x),
            (event_time, x, velocity, cursor)
    end
    terminal = trace.terminal_state
    if !cursor.terminal_returned && terminal !== nothing && terminal.t >= time
        cursor.terminal_returned = true
        if terminal.t > time
            _stream_move!(x, velocity, cursor.free,
                terminal.t - time, trace.flow)
        end
        # Compare the independently replayed state before applying the saved
        # authoritative endpoint. Retained traces have a frozen flow and must
        # replay to that endpoint. A warmup trace can contain metric/flow
        # adaptation; its final flow alone is intentionally insufficient to
        # replay earlier harmonic intervals.
        if trace.prefix != "warmup"
            scale = max(1.0, maximum(abs, terminal.position),
                maximum(abs, terminal.physical_velocity),
                maximum(abs, terminal.stored_frozen_velocity))
            tolerance = 1e-8 * scale
            position_error = maximum(abs.(x .- terminal.position))
            velocity_error = maximum(abs.(velocity .-
                terminal.physical_velocity))
            stored_error = maximum(abs.(cursor.stored_velocity .-
                terminal.stored_frozen_velocity))
            cursor.free == terminal.free && position_error <= tolerance &&
                velocity_error <= tolerance && stored_error <= tolerance ||
                throw(ArgumentError("streaming retained replay disagrees " *
                    "with authoritative endpoint before replacement " *
                    "(position=$position_error, velocity=$velocity_error, " *
                    "stored_velocity=$stored_error, mask_equal=" *
                    "$(cursor.free == terminal.free))"))
        end
        # The saved endpoint remains authoritative after validation.
        copyto!(x, terminal.position)
        copyto!(velocity, terminal.physical_velocity)
        copyto!(cursor.stored_velocity, terminal.stored_frozen_velocity)
        copyto!(cursor.free, terminal.free)
        return terminal.t => copy(x),
            (terminal.t, x, velocity, cursor)
    end
    return nothing
end

Base.length(trace::StreamingPDMPTrace) = 1 + trace.physical_events +
    (trace.terminal_state === nothing ? 0 : 1)
Base.first(trace::StreamingPDMPTrace) = trace.initial_state
Base.last(trace::StreamingPDMPTrace) = trace.terminal_state
first_event_time(trace::StreamingPDMPTrace) = trace.initial_state.t
last_event_time(trace::StreamingPDMPTrace) = trace.terminal_state === nothing ?
    trace.moments.last_time : trace.terminal_state.t
event_times(trace::StreamingPDMPTrace) = vcat(trace.initial_state.t,
    (chunk.times for chunk in (_load_streaming_chunk(c) for c in trace.chunks))...,
    trace.terminal_state.t)

function streaming_time_uniform_states(trace::StreamingPDMPTrace,
        retained_fraction::Real, n_grid::Integer)
    grid = trace.warmup_grid
    grid === nothing && throw(ArgumentError(
        "streaming trace has no bounded warmup grid"))
    all(grid.filled) || throw(ArgumentError(
        "streaming warmup grid was not completely observed"))
    final_time = trace.terminal_state.t
    initial_time = trace.initial_state.t
    requested = collect(range(final_time - retained_fraction *
        (final_time - initial_time), final_time; length=n_grid))
    indices = Vector{Int}(undef, n_grid)
    @inbounds for i in eachindex(requested)
        index = searchsortedfirst(grid.times, requested[i])
        index <= length(grid.times) && isapprox(grid.times[index], requested[i];
            rtol=0.0, atol=64eps(max(abs(requested[i]), 1.0))) ||
            throw(ArgumentError("requested warmup grid was not pre-recorded"))
        indices[i] = index
    end
    return copy(grid.positions[:, indices]), copy(grid.velocities[:, indices])
end

"""Bounded replay at the established dense-trace warmup grid.

This uses the installed trace flow and outgoing event states, matching
`PDMPTrace` plus `searchsortedlast` without constructing a dense
coordinate-by-event matrix.
"""
function streaming_dense_time_uniform_states(trace::StreamingPDMPTrace,
        retained_fraction::Real, n_grid::Integer)
    0 < retained_fraction <= 1 || throw(ArgumentError(
        "retained fraction must lie in (0, 1]"))
    n_grid >= 1 || throw(ArgumentError("grid size must be positive"))
    final_time = last_event_time(trace)
    initial_iteration = iterate(trace)
    initial_iteration === nothing && throw(ArgumentError("empty streaming trace"))
    first_physical = iterate(trace, initial_iteration[2])
    first_physical === nothing && throw(ArgumentError(
        "streaming trace has no physical warmup event"))
    # The established dense warmup trace installs its first record at this
    # event, not at sampler initialization.
    initial_time = first_physical[2][1]
    requested = collect(range(final_time - retained_fraction *
        (final_time - initial_time), final_time; length=n_grid))
    positions = Matrix{Float64}(undef,
        length(trace.initial_state.position), n_grid)

    first_iteration = iterate(trace)
    first_iteration === nothing && throw(ArgumentError("empty streaming trace"))
    current_time = first_iteration[2][1]
    current_x = copy(first_iteration[2][2])
    current_velocity = copy(first_iteration[2][3])
    current_free = copy(first_iteration[2][4].free)
    iteration_state = first_iteration[2]
    following = iterate(trace, iteration_state)
    scratch_x = similar(current_x)
    scratch_velocity = similar(current_velocity)
    @inbounds for column in eachindex(requested)
        target = requested[column]
        while following !== nothing && following[2][1] <= target
            current_time = following[2][1]
            copyto!(current_x, following[2][2])
            copyto!(current_velocity, following[2][3])
            copyto!(current_free, following[2][4].free)
            iteration_state = following[2]
            following = iterate(trace, iteration_state)
        end
        copyto!(scratch_x, current_x)
        copyto!(scratch_velocity, current_velocity)
        _stream_move!(scratch_x, scratch_velocity, current_free,
            target - current_time, trace.flow)
        copyto!(view(positions, :, column), scratch_x)
    end
    return positions
end

function _foreach_streaming_segment(f, trace::StreamingPDMPTrace)
    first_iteration = iterate(trace)
    first_iteration === nothing && return nothing
    time0 = first_iteration[2][1]
    x0 = copy(first_iteration[2][2])
    velocity0 = copy(first_iteration[2][3])
    cursor = first_iteration[2][4]
    free0 = copy(cursor.free)
    iteration = iterate(trace, first_iteration[2])
    x1 = similar(x0)
    velocity1 = similar(velocity0)
    while iteration !== nothing
        time1 = iteration[2][1]
        copyto!(x1, iteration[2][2])
        copyto!(velocity1, iteration[2][3])
        time1 > time0 && f(time0, time1, x0, x1, velocity0, velocity1,
            free0)
        time0 = time1
        copyto!(x0, x1)
        copyto!(velocity0, velocity1)
        copyto!(free0, iteration[2][4].free)
        iteration = iterate(trace, iteration[2])
    end
    return nothing
end

function _integrate(trace::StreamingPDMPTrace, f, args...)
    integral = nothing
    _foreach_streaming_segment(trace) do time0, time1, x0, x1,
            velocity0, velocity1, free
        contribution = _integrate_segment(f, trace.flow, x0, x1,
            velocity0, velocity1, time0, time1, free, args...)
        if integral === nothing
            integral = contribution
        else
            integral .+= contribution
        end
    end
    integral === nothing && error(
        "Cannot compute statistics on a streaming trace without a segment")
    return integral ./ (last_event_time(trace) - first_event_time(trace))
end

# Match the established segment-wise centred variance used by warmup metric
# adaptation. The online raw moments remain the constant-memory output summary
# and are intentionally not allowed to perturb the physical sampler.
function _streaming_adaptation_integral(trace::StreamingPDMPTrace, statistic,
        args...)
    # The dense warmup trace begins at its first recorded event rather than at
    # sampler initialization. Reproduce that established adaptation view while
    # scanning chunks; the complete streaming output still retains t0.
    initial = iterate(trace)
    initial === nothing && error("empty streaming adaptation trace")
    current = iterate(trace, initial[2])
    current === nothing && error("streaming adaptation trace has no event")
    start_time = current[2][1]
    time0 = start_time
    x0 = copy(current[2][2])
    velocity0 = copy(current[2][3])
    free0 = copy(current[2][4].free)
    integral = zeros(length(x0))
    following = iterate(trace, current[2])
    following === nothing && error(
        "streaming adaptation trace has no integrable segment")
    while following !== nothing
        time1 = following[2][1]
        if time1 > time0
            _integrate_segment!(integral, statistic, trace.flow,
                x0, following[2][2], velocity0, following[2][3],
                time0, time1, free0, args...)
        end
        time0 = time1
        copyto!(x0, following[2][2])
        copyto!(velocity0, following[2][3])
        copyto!(free0, following[2][4].free)
        current = following
        following = iterate(trace, current[2])
    end
    return integral ./ (time0 - start_time)
end

function _adaptation_std(trace::StreamingPDMPTrace)
    if isfactorized(trace.flow)
        means = _streaming_factorized_adaptation_moment(trace, nothing)
        variances = _streaming_factorized_adaptation_moment(trace, means)
        return sqrt.(variances)
    end
    means = _streaming_adaptation_integral(trace, Statistics.mean)
    return sqrt.(_streaming_adaptation_integral(
        trace, Statistics.var, means))
end

function _streaming_factorized_adaptation_moment(
        trace::StreamingPDMPTrace, means::Union{Nothing,AbstractVector})
    trace.finalized || _flush_streaming_trace!(trace)
    cursor = _stream_cursor(trace)
    d = length(cursor.x)
    integral = zeros(d)
    last_x = zeros(d)
    last_velocity = zeros(d)
    last_time = zeros(d)
    cursor_time = trace.initial_state.t
    start_time = NaN
    end_time = NaN
    first_event = true
    for chunk_meta in trace.chunks
        chunk = _load_streaming_chunk(chunk_meta)
        @inbounds for event in eachindex(chunk.times)
            event_time = chunk.times[event]
            _stream_move!(cursor.x, cursor.velocity, cursor.free,
                event_time - cursor_time, trace.flow)
            _stream_apply_event!(cursor, chunk, event)
            cursor_time = event_time
            if first_event
                copyto!(last_x, cursor.x)
                copyto!(last_velocity, cursor.velocity)
                fill!(last_time, event_time)
                start_time = event_time
                end_time = event_time
                first_event = false
                continue
            end
            kind = chunk.kinds[event]
            coordinate = Int(chunk.coordinates[event])
            touched = kind in (_STREAM_EVENT_REFLECT_FULL,
                _STREAM_EVENT_REFRESH, _STREAM_EVENT_DYNAMICS) ? (1:d) :
                (coordinate:coordinate)
            for j in touched
                elapsed = event_time - last_time[j]
                if means === nothing
                    integral[j] += last_x[j] * elapsed +
                        last_velocity[j] * elapsed^2 / 2
                else
                    y = last_x[j] - means[j]
                    velocity = last_velocity[j]
                    integral[j] += elapsed * (y^2 + y * velocity * elapsed +
                        velocity^2 * elapsed^2 / 3)
                end
                last_x[j] = cursor.x[j]
                last_velocity[j] = cursor.velocity[j]
                last_time[j] = event_time
            end
            end_time = event_time
        end
    end
    first_event && error("streaming adaptation trace has no event")
    total_time = end_time - start_time
    total_time > 0 || error("streaming adaptation trace has no elapsed time")
    @inbounds for j in 1:d
        elapsed = end_time - last_time[j]
        if elapsed > 0
            if means === nothing
                integral[j] += last_x[j] * elapsed +
                    last_velocity[j] * elapsed^2 / 2
            else
                y = last_x[j] - means[j]
                velocity = last_velocity[j]
                integral[j] += elapsed * (y^2 + y * velocity * elapsed +
                    velocity^2 * elapsed^2 / 3)
            end
        end
    end
    return integral ./ total_time
end

function _integrate!(out::AbstractVector, trace::StreamingPDMPTrace, f,
        args...)
    copyto!(out, _integrate(trace, f, args...))
    return out
end

function Base.first(trace::PDMPTrace)
    PDMPEvent(trace.times[1], trace.positions[:, 1], trace.velocities[:, 1])
end
Base.first(trace::FactorizedTrace) = trace.initial_state

function Base.last(trace::PDMPTrace)
    n = length(trace.times)
    PDMPEvent(trace.times[n], trace.positions[:, n], trace.velocities[:, n])
end
function Base.last(trace::FactorizedTrace)
    if !trace.last_state_valid
        trace.last_state = _replay_last_state(trace)
        trace.last_state_valid = true
    end
    trace.last_state
end

function Base.push!(trace::PDMPTrace{T,U,<:GrowableMatrix}, event::PDMPEvent) where {T,U}
    trace.free_masks === nothing ||
        throw(ArgumentError("append a StickyPDMPState, not a bare PDMPEvent, to a sticky trace"))
    push!(trace.times, event.time)
    append!(trace.positions, event.position)
    append!(trace.velocities, event.velocity)
    trace
end
function Base.push!(trace::PDMPTrace{T,U,<:GrowableMatrix}, state::AbstractPDMPState) where {T,U}
    trace.free_masks === nothing || state isa StickyPDMPState ||
        throw(ArgumentError("cannot append a non-sticky state to a sticky trace"))
    push!(trace.times, float(state.t[]))
    append!(trace.positions, state.ξ.x)
    append!(trace.velocities, state.ξ.θ)
    if trace.free_masks !== nothing
        append!(trace.free_masks, state.free)
    end
    trace
end
function Base.push!(trace::FactorizedTrace, event::FactorizedEvent)
    push!(trace.events, event)
    _advance_last_state(trace, event)
    _invalidate_factorized_caches!(trace)
    return trace
end

function Base.push!(trace::FactorizedTrace, state::AbstractPDMPState)
    trace.initial_state.time < zero(trace.initial_state.time) || error("Cannot append a full state to an initialized FactorizedTrace without a changed coordinate index")
    event = PDMPEvent(state)
    trace.initial_state = event
    trace.last_state = event
    trace.last_state_valid = true
    _invalidate_factorized_caches!(trace)
    return trace
end

function Base.push!(trace::FactorizedTrace, event::AbstractPDMPState, i::Integer)
    # TODO: this needs to figure out which index i changed... but it's much easier to have the caller provide it...
    if trace.initial_state.time >= zero(trace.initial_state.time)
        push!(trace.events, FactorizedEvent(i, event.t[], event.ξ.x[i], event.ξ.θ[i]))
        trace.last_state = PDMPEvent(event)
        trace.last_state_valid = true
    else
        # @info "setting initial event to " event
        e = PDMPEvent(event)
        trace.initial_state = e
        trace.last_state = e
        trace.last_state_valid = true
    end
    _invalidate_factorized_caches!(trace)
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
Base.length(trace::FactorizedTrace) = trace.initial_state.time < zero(trace.initial_state.time) ? 0 : length(trace.events) + 1

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
    trace.initial_state.time < zero(trace.initial_state.time) && return nothing
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
event_times(trace::FactorizedTrace) = trace.initial_state.time < zero(trace.initial_state.time) ? typeof(trace.initial_state.time)[] : [trace.initial_state.time; [e.time for e in trace.events]]

first_event_time(trace::PDMPTrace) = trace.times[1]
first_event_time(trace::FactorizedTrace) = trace.initial_state.time

last_event_time(trace::PDMPTrace) = trace.times[end]
last_event_time(trace::FactorizedTrace) = isempty(trace.events) ? trace.initial_state.time : trace.events[end].time

# could be a separate file from here?
# shouldn't T1 and T2 always be identical?
struct TraceManager{T}
    main_trace::T
    warmup_trace::T
    t_warmup::Float64
end
function TraceManager(state::AbstractPDMPState, flow::ContinuousDynamics, alg::PoissonTimeStrategy, t_warmup::Real)
    TT = _trace_type(flow, alg)
    _build_trace_manager(TT, state, flow, t_warmup)
end

function TraceManager(state::AbstractPDMPState, flow::ContinuousDynamics,
        alg::PoissonTimeStrategy, t_warmup::Real,
        storage::StreamingTraceStorage)
    main_trace = StreamingPDMPTrace(state, flow, storage, "main")
    warmup_trace = StreamingPDMPTrace(state, flow, storage, "warmup";
        warmup_end=t_warmup)
    return TraceManager(main_trace, warmup_trace, float(t_warmup))
end

function _build_trace_manager(::Type{TT}, state::AbstractPDMPState, flow::ContinuousDynamics, t_warmup::Real) where TT
    main_trace = make_empty_trace(TT, state, flow)
    warmup_trace = make_empty_trace(TT, state, flow)
    TraceManager(main_trace, warmup_trace, float(t_warmup))
end

get_warmup_trace(mgr::TraceManager) = mgr.warmup_trace
get_main_trace(mgr::TraceManager)   = mgr.main_trace

function record_event!(mgr::TraceManager, state, flow, args, phase::Symbol,
        event_type::Symbol=:legacy)
    trace = if phase === :warmup
        get_warmup_trace(mgr)
    elseif phase === :main
        get_main_trace(mgr)
    else
        state.t[] < mgr.t_warmup ? get_warmup_trace(mgr) : get_main_trace(mgr)
    end
    if trace isa StreamingPDMPTrace
        _record_streaming_event!(trace, state, flow, args, event_type)
    else
        dense_args = event_type === :sticky && trace isa PDMPTrace ? nothing : args
        push_trace!(trace, state, flow, dense_args)
    end
    return nothing
end


function begin_trace_phase!(mgr::TraceManager, state, flow, phase::Symbol)
    trace = phase === :warmup ? get_warmup_trace(mgr) : get_main_trace(mgr)
    begin_trace_phase!(trace, state, flow)
    return nothing
end

function finish_trace_phase!(mgr::TraceManager, state, flow, phase::Symbol)
    trace = phase === :warmup ? get_warmup_trace(mgr) : get_main_trace(mgr)
    finish_trace_phase!(trace, state, flow)
    return nothing
end

# Dynamics adaptation may replace every velocity at an event time.  Store that
# zero-duration discontinuity explicitly; otherwise the next occupation-moment
# calculation integrates the following segment with the pre-adaptation
# velocity.  A factorized trace represents a full velocity replacement as one
# same-time event per coordinate.
function record_dynamics_adaptation!(mgr::TraceManager, state, flow,
        phase::Symbol)
    trace = phase === :warmup ? get_warmup_trace(mgr) : get_main_trace(mgr)
    if trace isa StreamingPDMPTrace
        _record_streaming_event!(trace, state, flow, nothing,
            :dynamics_adaptation)
    elseif trace isa FactorizedTrace
        for i in eachindex(state.ξ.x)
            push_trace!(trace, state, flow, i)
        end
    else
        push_trace!(trace, state, flow, nothing)
    end
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


function _apply_event!(x::AbstractVector, θ::AbstractVector, trace::PDMPTrace, k::Int)
    copyto!(x, view(trace.positions, :, k))
    copyto!(θ, view(trace.velocities, :, k))
    trace.times[k]
end
function _apply_event!(x::AbstractVector, θ::AbstractVector, trace::FactorizedTrace, k::Int)
    _to_next_event!(x, θ, trace.events[k])
end

_to_range(D::PDMPDiscretize) = first(D.trace).time:D.dt:last_event_time(D.trace)

# could technically figure this out?
Base.IteratorSize(::PDMPDiscretize) = Base.HasLength()
Base.length(D::PDMPDiscretize) = length(_to_range(D))

function _discretize_move_forward!(ξ::SkeletonPoint, τ::Real, flow::ContinuousDynamics, free)
    move_forward_time!(ξ, τ, flow)
    return ξ
end

function _discretize_move_forward!(ξ::SkeletonPoint, τ::Real, flow::AnyBoomerang, free::Nothing)
    move_forward_time!(ξ, τ, flow)
    return ξ
end

function _discretize_move_forward!(ξ::SkeletonPoint, τ::Real, flow::AnyBoomerang, free::BitVector)
    move_forward_time!(ξ, τ, flow, free)
    return ξ
end

function _discretize_move_forward!(ξ::SkeletonPoint, τ::Real, flow::PreconditionedDynamics, free)
    _discretize_move_forward!(ξ, τ, flow.dynamics, free)
    return ξ
end

function _discretize_apply_free_mask!(free::BitVector, trace::PDMPTrace, k::Int)
    trace.free_masks === nothing || copyto!(free, view(trace.free_masks, :, k))
    return free
end
_discretize_apply_free_mask!(free, trace, k::Int) = free

function Base.iterate(D::PDMPDiscretize)
    trace = D.trace
    _isempty_trace(trace) && return nothing

    # The first yielded state is exactly the first event
    e1 = first(trace)
    t_start, x, θ = e1.time, copy(e1.position), copy(e1.velocity)

    # --- Set up the time range for discretization ---
    t_stop = last_event_time(trace)
    t_range = t_start:D.dt:t_stop

    # Get the iterator for the range. We've already "produced" the value
    # at t_start, so we only need the state for the *next* step.
    range_iterator_state = iterate(t_range)[2]

    k = trace isa FactorizedTrace ? 1 : 2
    free = if trace isa PDMPTrace && trace.free_masks !== nothing
        BitVector(view(trace.free_masks, :, 1))
    else
        nothing
    end

    iterator_state = (t_range, range_iterator_state, x, θ, k, free)

    return (t_start => x), iterator_state
end

function Base.iterate(D::PDMPDiscretize, (t_range, range_state, x_last, θ_last, k, free))
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
            _discretize_move_forward!(ξ, Δt, trace.flow, free)
            t_current = _apply_event!(x_current, θ_current, trace, k_current)
            _discretize_apply_free_mask!(free, trace, k_current)
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
            _discretize_apply_free_mask!(free, trace, last_before)
        end
    end
    # 4. Evolve the state from the time of the last processed event up to t_new
    Δt_final = t_new - t_current
    @assert Δt_final >= 0 "Should be impossible!"
    if ispositive(Δt_final)
        _discretize_move_forward!(ξ, Δt_final, trace.flow, free)
    end

    # 5. Return the calculated state and the new iterator state for the next call
    new_iterator_state = (t_range, new_range_state, x_current, θ_current, k_current, free)

    return (t_new => x_current), new_iterator_state
end

function Base.collect(D::PDMPDiscretize)
    collect(t => x for (t, x) in D)
end
Base.Matrix(D::PDMPDiscretize) = stack(last, D, dims = 1)
