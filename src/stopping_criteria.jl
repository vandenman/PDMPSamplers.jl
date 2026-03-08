"""
    StoppingCriterion

Interface for sampler stopping criteria.

Required methods:
- `initialize!(criterion, state, trace_manager, stats)`
- `update!(criterion, state, trace_manager, stats, event_type)`
- `is_satisfied(criterion, state, trace_manager, stats)`
- `stop_reason(criterion)`
"""
abstract type StoppingCriterion end

initialize!(::StoppingCriterion, state, trace_manager, stats) = nothing
update!(::StoppingCriterion, state, trace_manager, stats, event_type) = nothing

function is_satisfied(criterion::StoppingCriterion, state, trace_manager, stats)::Bool
    throw(MethodError(is_satisfied, (criterion, state, trace_manager, stats)))
end

stop_reason(::StoppingCriterion)::Symbol = :none
stop_reason(criterion::StoppingCriterion, state, trace_manager, stats)::Symbol = stop_reason(criterion)

"""
    FixedTimeCriterion(T)

Stop when the sampler's internal time reaches `T`.
"""
struct FixedTimeCriterion <: StoppingCriterion
    T::Float64
end

FixedTimeCriterion(T::Real) = FixedTimeCriterion(Float64(T))
is_satisfied(c::FixedTimeCriterion, state, trace_manager, stats) = state.t[] >= c.T
stop_reason(::FixedTimeCriterion) = :reached_time

"""
    EventCountCriterion(max_events)

Stop when `max_events` events have occurred in the current phase.
Counts accepted reflections, refreshments, and sticky events from `StatisticCounter`.
The count is phase-local: a baseline is recorded at `initialize!`, and only events
since that point count toward the budget.
"""
mutable struct EventCountCriterion <: StoppingCriterion
    const max_events::Int
    baseline::Int
    function EventCountCriterion(max_events::Integer)
        max_events > 0 || throw(ArgumentError("max_events must be > 0, got $max_events"))
        return new(max_events, 0)
    end
end

function initialize!(c::EventCountCriterion, state, trace_manager, stats)
    c.baseline = stats.reflections_events + stats.refreshment_events + stats.sticky_events
    return nothing
end

function is_satisfied(c::EventCountCriterion, state, trace_manager, stats)
    total = stats.reflections_events + stats.refreshment_events + stats.sticky_events
    return (total - c.baseline) >= c.max_events
end

stop_reason(::EventCountCriterion) = :reached_event_budget

"""
    WallTimeCriterion(seconds)

Stop after `seconds` wall-clock seconds.
"""
mutable struct WallTimeCriterion <: StoppingCriterion
    max_seconds::Float64
    start_ns::UInt64
end

function WallTimeCriterion(seconds::Real)
    sec = Float64(seconds)
    isfinite(sec) && sec > 0 || throw(ArgumentError("seconds must be finite and > 0, got $seconds"))
    return WallTimeCriterion(sec, zero(UInt64))
end

function initialize!(c::WallTimeCriterion, state, trace_manager, stats)
    c.start_ns = time_ns()
    return nothing
end

function is_satisfied(c::WallTimeCriterion, state, trace_manager, stats)
    c.start_ns == zero(UInt64) && return false
    elapsed = (time_ns() - c.start_ns) / 1e9
    return elapsed >= c.max_seconds
end

stop_reason(::WallTimeCriterion) = :reached_wall_time

"""
    TotalWallTimeCriterion(seconds)

Stop after `seconds` wall-clock seconds in total, across phases when the same
criterion instance is reused.
"""
mutable struct TotalWallTimeCriterion <: StoppingCriterion
    max_seconds::Float64
    start_ns::Base.RefValue{UInt64}
end

function TotalWallTimeCriterion(seconds::Real)
    sec = Float64(seconds)
    isfinite(sec) && sec > 0 || throw(ArgumentError("seconds must be finite and > 0, got $seconds"))
    return TotalWallTimeCriterion(sec, Ref(zero(UInt64)))
end

function initialize!(c::TotalWallTimeCriterion, state, trace_manager, stats)
    if c.start_ns[] == zero(UInt64)
        c.start_ns[] = time_ns()
    end
    return nothing
end

function is_satisfied(c::TotalWallTimeCriterion, state, trace_manager, stats)
    c.start_ns[] == zero(UInt64) && return false
    elapsed = (time_ns() - c.start_ns[]) / 1e9
    return elapsed >= c.max_seconds
end

stop_reason(::TotalWallTimeCriterion) = :reached_wall_time

"""
    ESSCriterion(target_ess; check_every=500, min_trace_length=10, trace_selector=:main)

Stop when the minimum coordinate ESS reaches `target_ess`.
Computes `ess(trace)` (O(n) in trace length) every `check_every` events.
For long runs, prefer `OnlineESSCriterion` which uses O(1)-per-event online estimation.
"""
mutable struct ESSCriterion <: StoppingCriterion
    target_ess::Float64
    check_every::Int
    min_trace_length::Int
    events_since_check::Int
    satisfied::Bool
    trace_selector::Symbol
end

function ESSCriterion(
    target_ess::Real;
    check_every::Integer=500,
    min_trace_length::Integer=10,
    trace_selector::Symbol=:main
)
    target = Float64(target_ess)
    isfinite(target) && target > 0 || throw(ArgumentError("target_ess must be finite and > 0, got $target_ess"))
    check_every > 0 || throw(ArgumentError("check_every must be > 0, got $check_every"))
    min_trace_length >= 2 || throw(ArgumentError("min_trace_length must be >= 2, got $min_trace_length"))
    trace_selector in (:main, :warmup) || throw(ArgumentError("trace_selector must be :main or :warmup, got $trace_selector"))
    return ESSCriterion(target, check_every, min_trace_length, 0, false, trace_selector)
end

function initialize!(c::ESSCriterion, state, trace_manager, stats)
    c.events_since_check = 0
    c.satisfied = false
    return nothing
end

function update!(c::ESSCriterion, state, trace_manager, stats, event_type)
    c.events_since_check += 1
    return nothing
end

function _selected_trace(c::ESSCriterion, trace_manager)
    return c.trace_selector === :main ? get_main_trace(trace_manager) : get_warmup_trace(trace_manager)
end

function is_satisfied(c::ESSCriterion, state, trace_manager, stats)
    c.satisfied && return true
    c.events_since_check >= c.check_every || return false

    trace = _selected_trace(c, trace_manager)
    length(trace) >= c.min_trace_length || return false

    c.satisfied = minimum(ess(trace)) >= c.target_ess
    c.events_since_check = 0
    return c.satisfied
end

stop_reason(::ESSCriterion) = :reached_ess

"""
    OnlineESSCriterion(target_ess; check_every=500, min_samples=10, batch_size=50)

Stop when a resumable online ESS estimate reaches `target_ess`.
Uses Welford-based online variance with batch means, consuming O(1) per event.
The ESS estimate is based on unweighted skeleton positions (not time-weighted path integrals),
which is an approximation for continuous-time PDMP trajectories. The criterion is inherently
phase-local: it accumulates statistics only for events in the phase it is attached to.
"""
mutable struct OnlineESSCriterion <: StoppingCriterion
    target_ess::Float64
    check_every::Int
    min_samples::Int
    batch_size::Int
    events_since_check::Int
    satisfied::Bool
    n_samples::Int
    batch_fill::Int
    n_batches::Int
    batch_sum::Vector{Float64}
    mean_samples::Vector{Float64}
    m2_samples::Vector{Float64}
    mean_batches::Vector{Float64}
    m2_batches::Vector{Float64}
    scratch_ess::Vector{Float64}
end

function OnlineESSCriterion(
    target_ess::Real;
    check_every::Integer=500,
    min_samples::Integer=10,
    batch_size::Integer=50
)
    target = Float64(target_ess)
    isfinite(target) && target > 0 || throw(ArgumentError("target_ess must be finite and > 0, got $target_ess"))
    check_every > 0 || throw(ArgumentError("check_every must be > 0, got $check_every"))
    min_samples >= 2 || throw(ArgumentError("min_samples must be >= 2, got $min_samples"))
    batch_size > 0 || throw(ArgumentError("batch_size must be > 0, got $batch_size"))
    return OnlineESSCriterion(
        target,
        Int(check_every),
        Int(min_samples),
        Int(batch_size),
        0,
        false,
        0,
        0,
        0,
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
    )
end

function _ensure_online_buffers!(c::OnlineESSCriterion, d::Int)
    length(c.batch_sum) == d && return nothing
    c.batch_sum = zeros(d)
    c.mean_samples = zeros(d)
    c.m2_samples = zeros(d)
    c.mean_batches = zeros(d)
    c.m2_batches = zeros(d)
    c.scratch_ess = zeros(d)
    return nothing
end

function initialize!(c::OnlineESSCriterion, state, trace_manager, stats)
    d = length(state.ξ.x)
    _ensure_online_buffers!(c, d)
    c.events_since_check = 0
    c.satisfied = false
    c.n_samples = 0
    c.batch_fill = 0
    c.n_batches = 0
    fill!(c.batch_sum, 0.0)
    fill!(c.mean_samples, 0.0)
    fill!(c.m2_samples, 0.0)
    fill!(c.mean_batches, 0.0)
    fill!(c.m2_batches, 0.0)
    fill!(c.scratch_ess, 0.0)
    return nothing
end

function _update_online_sample!(c::OnlineESSCriterion, x::AbstractVector)
    c.n_samples += 1
    n = c.n_samples
    @inbounds for i in eachindex(c.mean_samples)
        xi = Float64(x[i])
        c.batch_sum[i] += xi
        δ = xi - c.mean_samples[i]
        c.mean_samples[i] += δ / n
        δ2 = xi - c.mean_samples[i]
        c.m2_samples[i] += δ * δ2
    end

    c.batch_fill += 1
    if c.batch_fill == c.batch_size
        c.n_batches += 1
        nb = c.n_batches
        inv_batch_size = inv(Float64(c.batch_size))
        @inbounds for i in eachindex(c.mean_batches)
            bm = c.batch_sum[i] * inv_batch_size
            c.batch_sum[i] = 0.0
            δb = bm - c.mean_batches[i]
            c.mean_batches[i] += δb / nb
            δb2 = bm - c.mean_batches[i]
            c.m2_batches[i] += δb * δb2
        end
        c.batch_fill = 0
    end
    return nothing
end

function update!(c::OnlineESSCriterion, state, trace_manager, stats, event_type)
    c.events_since_check += 1
    _update_online_sample!(c, state.ξ.x)
    return nothing
end

function is_satisfied(c::OnlineESSCriterion, state, trace_manager, stats)
    c.satisfied && return true
    c.events_since_check >= c.check_every || return false
    c.n_samples >= c.min_samples || return false
    c.n_batches >= 2 || return false

    sample_denom = c.n_samples - 1
    batch_denom = c.n_batches - 1
    @inbounds for i in eachindex(c.scratch_ess)
        overall_var = c.m2_samples[i] / sample_denom
        batch_var = c.m2_batches[i] / batch_denom
        if overall_var > 0 && batch_var > 0
            c.scratch_ess[i] = c.n_batches * overall_var / batch_var
        else
            c.scratch_ess[i] = Float64(c.n_batches)
        end
    end

    c.satisfied = minimum(c.scratch_ess) >= c.target_ess
    c.events_since_check = 0
    return c.satisfied
end

stop_reason(::OnlineESSCriterion) = :reached_ess

"""
    AnyCriterion(criteria...)

Stop when any child criterion is satisfied.
"""
struct AnyCriterion{Cs<:Tuple} <: StoppingCriterion
    criteria::Cs
end

"""
    AllCriteria(criteria...)

Stop when all child criteria are satisfied.
`stop_reason` returns `:all_criteria_satisfied` without forwarding to individual criteria.
"""
struct AllCriteria{Cs<:Tuple} <: StoppingCriterion
    criteria::Cs
end

function AnyCriterion(criteria::StoppingCriterion...)
    isempty(criteria) && throw(ArgumentError("AnyCriterion requires at least one child criterion"))
    return AnyCriterion(criteria)
end

function AllCriteria(criteria::StoppingCriterion...)
    isempty(criteria) && throw(ArgumentError("AllCriteria requires at least one child criterion"))
    return AllCriteria(criteria)
end

const _ComposedCriterion = Union{AnyCriterion, AllCriteria}

function initialize!(c::_ComposedCriterion, state, trace_manager, stats)
    for criterion in c.criteria
        initialize!(criterion, state, trace_manager, stats)
    end
    return nothing
end

function update!(c::_ComposedCriterion, state, trace_manager, stats, event_type)
    for criterion in c.criteria
        update!(criterion, state, trace_manager, stats, event_type)
    end
    return nothing
end

function is_satisfied(c::AnyCriterion, state, trace_manager, stats)
    for criterion in c.criteria
        is_satisfied(criterion, state, trace_manager, stats) && return true
    end
    return false
end

function is_satisfied(c::AllCriteria, state, trace_manager, stats)
    for criterion in c.criteria
        is_satisfied(criterion, state, trace_manager, stats) || return false
    end
    return true
end

function stop_reason(c::AnyCriterion, state, trace_manager, stats)
    for criterion in c.criteria
        is_satisfied(criterion, state, trace_manager, stats) && return stop_reason(criterion)
    end
    return :none
end

stop_reason(::AllCriteria) = :all_criteria_satisfied

# Value copy: start_ns is overwritten by initialize! at the start of each phase,
# unlike TotalWallTimeCriterion which uses a Ref to share a single global start time.
function Base.copy(c::WallTimeCriterion)
    return WallTimeCriterion(c.max_seconds, c.start_ns)
end

function Base.copy(c::TotalWallTimeCriterion)
    return TotalWallTimeCriterion(c.max_seconds, Ref(c.start_ns[]))
end

function Base.copy(c::ESSCriterion)
    return ESSCriterion(c.target_ess, c.check_every, c.min_trace_length, c.events_since_check, c.satisfied, c.trace_selector)
end

function Base.copy(c::OnlineESSCriterion)
    copied = OnlineESSCriterion(
        c.target_ess;
        check_every=c.check_every,
        min_samples=c.min_samples,
        batch_size=c.batch_size,
    )
    copied.events_since_check = c.events_since_check
    copied.satisfied = c.satisfied
    copied.n_samples = c.n_samples
    copied.batch_fill = c.batch_fill
    copied.n_batches = c.n_batches
    copied.batch_sum = copy(c.batch_sum)
    copied.mean_samples = copy(c.mean_samples)
    copied.m2_samples = copy(c.m2_samples)
    copied.mean_batches = copy(c.mean_batches)
    copied.m2_batches = copy(c.m2_batches)
    copied.scratch_ess = copy(c.scratch_ess)
    return copied
end

Base.copy(c::FixedTimeCriterion) = c
function Base.copy(c::EventCountCriterion)
    ec = EventCountCriterion(c.max_events)
    ec.baseline = c.baseline
    return ec
end

function Base.copy(c::AnyCriterion)
    return AnyCriterion(map(copy, c.criteria)...)
end

function Base.copy(c::AllCriteria)
    return AllCriteria(map(copy, c.criteria)...)
end

"""
    stop_after(; ess=nothing, events=nothing, wall_seconds=nothing, T=nothing,
                 ess_check_every=500, ess_min_trace_length=10, ess_trace_selector=:main,
                 ess_mode=:exact, ess_batch_size=50)

Construct a stopping criterion that is satisfied when any specified budget is reached.
"""
_combine_any(::Nothing, c::StoppingCriterion) = c
_combine_any(a::StoppingCriterion, b::StoppingCriterion) = AnyCriterion(a, b)

function stop_after(
    ;
    ess::Union{Nothing, Real}=nothing,
    events::Union{Nothing, Integer}=nothing,
    wall_seconds::Union{Nothing, Real}=nothing,
    T::Union{Nothing, Real}=nothing,
    ess_check_every::Integer=500,
    ess_min_trace_length::Integer=10,
    ess_trace_selector::Symbol=:main,
    ess_mode::Symbol=:exact,
    ess_batch_size::Integer=50
)::StoppingCriterion
    criterion::Union{Nothing, StoppingCriterion} = nothing

    if !isnothing(ess)
        ess_criterion = if ess_mode === :exact
            ESSCriterion(
                ess;
                check_every=ess_check_every,
                min_trace_length=ess_min_trace_length,
                trace_selector=ess_trace_selector,
            )
        elseif ess_mode === :online
            OnlineESSCriterion(
                ess;
                check_every=ess_check_every,
                min_samples=ess_min_trace_length,
                batch_size=ess_batch_size,
            )
        else
            throw(ArgumentError("ess_mode must be :exact or :online, got $ess_mode"))
        end
        criterion = _combine_any(criterion, ess_criterion)
    end
    !isnothing(events) && (criterion = _combine_any(criterion, EventCountCriterion(events)))
    !isnothing(wall_seconds) && (criterion = _combine_any(criterion, WallTimeCriterion(wall_seconds)))
    !isnothing(T) && (criterion = _combine_any(criterion, FixedTimeCriterion(T)))

    isnothing(criterion) && throw(ArgumentError("At least one of ess, events, wall_seconds, or T must be provided"))
    return criterion
end
