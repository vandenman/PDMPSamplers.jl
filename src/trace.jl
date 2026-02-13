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

struct PDMPTrace{T<:PDMPEvent,U<:ContinuousDynamics} <: AbstractPDMPTrace
    events::Vector{T}
    flow::U
end
PDMPTrace(state::AbstractPDMPState, flow::ContinuousDynamics) = PDMPTrace([PDMPEvent(state)], flow)
function make_empty_trace(::Type{<:PDMPTrace}, state::AbstractPDMPState, flow::ContinuousDynamics)
    Tevent = PDMPEvent{typeof(state.t[]), typeof(state.ξ.x), typeof(state.ξ.θ)}
    PDMPTrace(Vector{Tevent}(), flow)
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

Base.length(trace::PDMPTrace)       = length(trace.events)
Base.length(trace::FactorizedTrace) = length(trace.events) + 1

function Base.iterate(trace::AbstractPDMPTrace)

    isempty(trace.events) && return nothing

    # Start with first event
    e1 = first(trace)
    t, x, θ = e1.time, copy(e1.position), copy(e1.velocity)
    k = trace isa FactorizedTrace ? 1 : 2
    return t => copy(x), (t, x, θ, k)
end

function Base.iterate(trace::AbstractPDMPTrace, (t, x, θ, k))
    k > length(trace.events) && return nothing
    # k == length(trace.events) && return t => x, (t, x, θ, k + 1)

    Δt = trace.events[k].time - t
    move_forward_time!(SkeletonPoint(x, θ), Δt, trace.flow)
    t = _to_next_event!(x, θ, trace.events[k])
    return t => copy(x), (t, x, θ, k + 1)
end

Base.collect(trace::AbstractPDMPTrace) = collect(t => x for (t, x) in trace)

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

Statistics.mean(trace::AbstractPDMPTrace) = _integrate(trace, Statistics.mean)
Statistics.var( trace::AbstractPDMPTrace) = _integrate(trace, Statistics.var, Statistics.mean(trace))
Statistics.std( trace::AbstractPDMPTrace) = sqrt.(Statistics.var(trace))
Statistics.cov( trace::AbstractPDMPTrace) = _integrate(trace, Statistics.cov, Statistics.mean(trace))
function Statistics.cor(trace::AbstractPDMPTrace)
    C = Statistics.cov(trace)
    StatsBase.cov2cor!(C, sqrt.(diag(C)))
end


_underlying_flow(flow::ContinuousDynamics) = flow
_underlying_flow(flow::PreconditionedDynamics) = flow.dynamics

"""
    cdf(trace::AbstractPDMPTrace, q::Real; coordinate::Integer)

Compute the empirical CDF of marginal coordinate `coordinate` at threshold `q`
from the continuous occupation measure of a PDMP trace (no discretization).

Returns the fraction of total trajectory time spent with `x_j(t) ≤ q`.
"""
function cdf(trace::AbstractPDMPTrace, q::Real; coordinate::Integer)
    flow = trace.flow
    base = _underlying_flow(flow)
    j = coordinate

    iter = trace
    next = iterate(iter)
    isnothing(next) && error("Cannot compute CDF on an empty trace")

    t₀, x_state, θ_state, _ = next[2]
    x0j = x_state[j]
    θ0j = θ_state[j]
    start_time = t₀

    next = iterate(iter, next[2])
    isnothing(next) && error("Cannot compute CDF on a trace with fewer than 2 events")

    total_below = 0.0
    while next !== nothing
        t₁, x_state, θ_state, _ = next[2]
        τ = t₁ - t₀
        if base isa Boomerang
            total_below += _time_below_segment(flow, x0j, θ0j, τ, q, base.μ[j])
        else
            total_below += _time_below_segment(flow, x0j, θ0j, τ, q)
        end
        t₀ = t₁
        x0j = x_state[j]
        θ0j = θ_state[j]
        next = iterate(iter, next[2])
    end

    total_time = t₀ - start_time
    return total_below / total_time
end

function _trace_coordinate_bounds(trace::AbstractPDMPTrace, j::Integer)
    lo, hi = Inf, -Inf
    base = _underlying_flow(trace.flow)
    is_boom = base isa Boomerang

    for event in trace.events
        xj = event.position[j]
        lo = min(lo, xj)
        hi = max(hi, xj)
        if is_boom
            R = hypot(xj - base.μ[j], event.velocity[j])
            lo = min(lo, base.μ[j] - R)
            hi = max(hi, base.μ[j] + R)
        end
    end
    return lo, hi
end

# ──────────────────────────────────────────────────────────────────────────────
# Sweep-line quantile for linear dynamics (ZigZag & BPS)
# ──────────────────────────────────────────────────────────────────────────────

function _collect_sweep_events(trace::AbstractPDMPTrace, j::Integer)
    density_changes = Tuple{Float64, Float64}[]
    point_masses    = Tuple{Float64, Float64}[]

    iter = trace
    next = iterate(iter)
    isnothing(next) && error("Cannot compute quantile on an empty trace")
    t₀, x_state, θ_state, _ = next[2]
    x0j = Float64(x_state[j])
    θ0j = Float64(θ_state[j])
    start_time = t₀

    next = iterate(iter, next[2])
    isnothing(next) && error("Cannot compute quantile on a trace with fewer than 2 events")

    while next !== nothing
        t₁, x_state, θ_state, _ = next[2]
        τ = t₁ - t₀
        if iszero(θ0j)
            push!(point_masses, (x0j, τ))
        else
            x_end = x0j + θ0j * τ
            lo = min(x0j, x_end)
            hi = max(x0j, x_end)
            d_inv = inv(abs(θ0j))
            push!(density_changes, (lo,  d_inv))
            push!(density_changes, (hi, -d_inv))
        end
        t₀ = t₁
        x0j = Float64(x_state[j])
        θ0j = Float64(θ_state[j])
        next = iterate(iter, next[2])
    end

    total_time = t₀ - start_time
    return total_time, density_changes, point_masses
end

function _quantile_linear_sweep(
    total_time::Float64,
    density_changes::Vector{Tuple{Float64, Float64}},
    point_masses::Vector{Tuple{Float64, Float64}},
    sorted_targets::Vector{Float64},
)
    n_dc = length(density_changes)
    n_pm = length(point_masses)
    n_events = n_dc + n_pm
    # (x, density_delta, point_mass_weight)
    events = Vector{Tuple{Float64, Float64, Float64}}(undef, n_events)
    for i in 1:n_dc
        events[i] = (density_changes[i][1], density_changes[i][2], 0.0)
    end
    for i in 1:n_pm
        events[n_dc + i] = (point_masses[i][1], 0.0, point_masses[i][2])
    end
    sort!(events; by=first)

    n_q = length(sorted_targets)
    results = Vector{Float64}(undef, n_q)
    qi = 1  # pointer into sorted_targets

    cumulative = 0.0
    density = 0.0
    prev_x = NaN
    i = 1

    while i ≤ n_events && qi ≤ n_q
        x_k = events[i][1]

        # linear growth from prev_x to x_k
        if density > 0 && !isnan(prev_x)
            growth = density * (x_k - prev_x)
            while qi ≤ n_q && cumulative + growth ≥ sorted_targets[qi]
                needed = sorted_targets[qi] - cumulative
                results[qi] = prev_x + needed / density
                qi += 1
            end
            cumulative += growth
        end

        # process all events at x_k
        while i ≤ n_events && events[i][1] == x_k
            density += events[i][2]
            pm = events[i][3]
            if pm > 0
                cumulative += pm
                while qi ≤ n_q && cumulative ≥ sorted_targets[qi]
                    results[qi] = x_k
                    qi += 1
                end
            end
            i += 1
        end

        prev_x = x_k
    end

    qi ≤ n_q && error("Quantile sweep did not resolve all targets (resolved $(qi-1)/$n_q)")
    return results
end

# ──────────────────────────────────────────────────────────────────────────────
# Public quantile / median API
# ──────────────────────────────────────────────────────────────────────────────

"""
    Statistics.quantile(trace::AbstractPDMPTrace, p; coordinate::Integer)

Compute the `p`-th marginal quantile from the continuous occupation measure of a PDMP trace.
`p` can be a scalar or a vector of probabilities. When `coordinate` is omitted, returns
a vector of quantiles for all coordinates (only valid for scalar `p`).

For linear dynamics (Zig-Zag, BPS) this uses an exact O(N log N) sweep-line
algorithm over the piecewise-linear CDF. For Boomerang dynamics it falls back
to bisection root-finding on the exact CDF.
"""
function Statistics.quantile(trace::AbstractPDMPTrace, p::Real; coordinate::Integer=-1)
    (0 < p < 1) || throw(DomainError(p, "Quantile probability must be in (0, 1)"))
    if coordinate == -1
        d = length(first(trace).position)
        return [_quantile_scalar(trace, p, j) for j in 1:d]
    end
    return _quantile_scalar(trace, p, coordinate)
end

function Statistics.quantile(trace::AbstractPDMPTrace, p::AbstractVector{<:Real}; coordinate::Integer)
    all(x -> 0 < x < 1, p) || throw(DomainError(p, "All quantile probabilities must be in (0, 1)"))
    base = _underlying_flow(trace.flow)
    if base isa Boomerang
        return [_quantile_scalar(trace, pi, coordinate) for pi in p]
    end
    total_time, dc, pm = _collect_sweep_events(trace, coordinate)
    order = sortperm(collect(p))
    sorted_targets = [p[i] * total_time for i in order]
    sorted_results = _quantile_linear_sweep(total_time, dc, pm, sorted_targets)
    results = similar(sorted_results)
    for i in eachindex(order)
        results[order[i]] = sorted_results[i]
    end
    return results
end

function _quantile_scalar(trace::AbstractPDMPTrace, p::Real, coordinate::Integer)
    base = _underlying_flow(trace.flow)
    if base isa Boomerang
        lo, hi = _trace_coordinate_bounds(trace, coordinate)
        f(q) = cdf(trace, q; coordinate) - p
        return Roots.find_zero(f, (lo, hi), Roots.Bisection())
    end
    total_time, dc, pm = _collect_sweep_events(trace, coordinate)
    return _quantile_linear_sweep(total_time, dc, pm, [p * total_time])[1]
end

function Statistics.median(trace::AbstractPDMPTrace; coordinate::Integer=-1)
    return Statistics.quantile(trace, 0.5; coordinate)
end

"""
    ess(trace::AbstractPDMPTrace; n_batches::Integer=max(50, isqrt(length(trace))))

Estimate the effective sample size (ESS) per coordinate from a PDMP trace
using the batch-means method, without discretization.

Splits the trace into `n_batches` equal-time batches, computes the
time-weighted mean of each batch via exact piecewise integration, and
returns

    ESS_i = n_batches * Var_overall_i / Var(batch_means_i)

The result is a `Vector{Float64}` of length `d` (the dimension of the state).
"""
function ess(trace::AbstractPDMPTrace; n_batches::Integer=max(50, isqrt(length(trace))))

    flow = trace.flow

    iter = trace
    next = iterate(iter)
    isnothing(next) && error("Cannot compute ESS on an empty trace")

    t₀, x_state, θ_state, _ = next[2]
    xt = copy(x_state)
    θt = copy(θ_state)

    next = iterate(iter, next[2])
    isnothing(next) && error("Cannot compute ESS on a trace with fewer than 2 events")

    d = length(xt)
    t_start = t₀
    t_end = trace.events[end].time
    total_time = t_end - t_start

    batch_duration = total_time / n_batches
    batch_means = zeros(n_batches, d)

    batch_idx = 1
    batch_start = t_start
    batch_end = t_start + batch_duration
    batch_integral = zeros(d)

    xt_next = similar(xt)
    θt_next = similar(θt)

    while next !== nothing
        t₁, x_state, θ_state, _ = next[2]
        copyto!(xt_next, x_state)
        copyto!(θt_next, θ_state)

        seg_start = t₀
        seg_end = t₁

        while seg_start < seg_end && batch_idx <= n_batches
            chunk_end = min(seg_end, batch_end)

            if chunk_end > seg_start
                elapsed = seg_start - t₀
                xt_at_seg = copy(xt)
                θt_at_seg = copy(θt)
                if elapsed > 0
                    move_forward_time!(SkeletonPoint(xt_at_seg, θt_at_seg), elapsed, flow)
                end

                contribution = _integrate_segment(
                    Statistics.mean, flow,
                    xt_at_seg, xt_next,
                    θt_at_seg, θt_next,
                    seg_start, chunk_end
                )
                batch_integral .+= contribution
            end

            if chunk_end >= batch_end - eps(batch_end)
                batch_means[batch_idx, :] .= batch_integral ./ batch_duration
                fill!(batch_integral, 0.0)
                batch_idx += 1
                batch_start = batch_end
                batch_end = t_start + batch_idx * batch_duration
            end

            seg_start = chunk_end
        end

        t₀ = t₁
        copyto!(xt, xt_next)
        copyto!(θt, θt_next)
        next = iterate(iter, next[2])
    end

    if batch_idx == n_batches && any(!iszero, batch_integral)
        batch_means[batch_idx, :] .= batch_integral ./ batch_duration
    end

    n_complete = min(batch_idx, n_batches)
    n_complete < 3 && error("Too few complete batches ($n_complete) for ESS estimation")
    bm = view(batch_means, 1:n_complete, :)

    overall_var = var(trace)
    batch_var = vec(var(bm, dims=1))

    result = Vector{Float64}(undef, d)
    for i in 1:d
        if batch_var[i] > 0 && overall_var[i] > 0
            result[i] = n_complete * overall_var[i] / batch_var[i]
        else
            result[i] = Float64(n_complete)
        end
    end

    return result
end

"""
    inclusion_probs(trace::AbstractPDMPTrace)

Compute marginal inclusion probabilities from a PDMP trace.
For coordinate i, this equals the fraction of time the trajectory
spent away from zero.

For a spike-and-slab model with inclusion probability ``p_i`` and slab
mean ``\\mu_i``, the full-model mean satisfies:

    mean(trace)[i] ≈ p_i * μ_i

so the conditional slab mean can be recovered as:

    mean(trace) ./ inclusion_probs(trace) ≈ μ_slab
"""
inclusion_probs(trace::AbstractPDMPTrace) = _integrate(trace, inclusion_probs)

# TODO: this should actually use the flows M matrix!
_integrate_segment(f::Any, flow::PreconditionedDynamics, args...) = _integrate_segment(f, flow.dynamics, args...)

function _integrate_segment(::typeof(inclusion_probs), ::Union{ZigZag, BouncyParticle}, x0, x1, θ0, θ1, t0, t1)

    result = zeros(length(x0))
    _integrate_segment!(result, inclusion_probs, ZigZag(length(x0)), x0, x1, θ0, θ1, t0, t1)
    return result
end

function _integrate_segment!(buf::AbstractVector, ::typeof(inclusion_probs), ::Union{ZigZag, BouncyParticle}, x0, x1, θ0, θ1, t0, t1)
    # x_i(s) = x + v*s is zero only on a measure-zero set when not both are zero,
    # so inclusion_probs integral = dt when the trajectory is not identically zero.
    #
    # Mathematica: g[x_] := If[x == 0, 1, 0]
    #   FullSimplify[Integrate[g[x + s*v], {s, 0, t1 - t0}], ...]
    #   → -t0+t1  if v==0 && x==0;  0  otherwise
    # The code uses the negation of that condition.
    for i in eachindex(x0)
        v = θ0[i]
        x = x0[i]
        if !(iszero(x) && iszero(v))
            buf[i] += t1 - t0
        end
    end
    return buf
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

# _integrate_segment(::typeof(Statistics.var), ::ZigZag,         x0, x1, θ0, θ1, t0, t1, μ) = (-(x0 .- μ) .^ 3 + (-t0 .* θ0 .+ t1 .* θ0 .+ x0 .- μ) .^ 3) ./ (3. * θ0)
# _integrate_segment(::typeof(Statistics.var), ::BouncyParticle, x0, x1, θ0, θ1, t0, t1, μ) = (-(x0 .- μ) .^ 3 + (-t0 .* θ0 .+ t1 .* θ0 .+ x0 .- μ) .^ 3) ./ (3. * θ0)#(t0 * θ0 - t1 * θ0 - 2 * x0 + 2 * μ) * (t0 - t1) / 2
function _integrate_segment(::typeof(Statistics.var), ::Union{ZigZag, BouncyParticle}, x0, x1, θ0, θ1, t0, t1, μ)
    dt = t1 - t0

    # 1. Center the variables
    y0 = x0 .- μ

    # 2. Integrate (y0 + θ*t)^2 from 0 to dt
    # Expansion: y0^2 + 2*y0*θ*t + θ^2*t^2
    # Integral:  y0^2*t + y0*θ*t^2 + θ^2*t^3/3

    # Using @evalpoly for stability and speed (evaluates c0*t + c1*t^2 + ...)
    # Note: We factor out 'dt' to keep coefficients simple

    return @. dt * (y0^2 + y0 * θ0 * dt + (θ0^2 * dt^2) / 3)
end

function _integrate_segment(::typeof(Statistics.cov), flow::Union{ZigZag, BouncyParticle}, x0, x1, θ0, θ1, t0, t1, μ)
    d = length(x0)
    segment_integral = zeros(d, d)
    _integrate_segment!(segment_integral, Statistics.cov, flow, x0, x1, θ0, θ1, t0, t1, μ)
    return segment_integral
end

function _integrate_segment!(buf::AbstractMatrix, ::typeof(Statistics.cov), ::Union{ZigZag, BouncyParticle}, x0, x1, θ0, θ1, t0, t1, μ)
    d = length(x0)
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

        val = @evalpoly(Δt, 0.0, Δᵢ * Δⱼ, (θⱼ * Δᵢ + θᵢ * Δⱼ) / 2, θⱼ * θᵢ / 3)
        buf[i, j] += val
        i != j && (buf[j, i] += val)

    end
    return buf
end

# --- Boomerang integration methods ---
# Boomerang trajectory: x_i(s) = Δ_i cos(s) + θ_i sin(s) + μ_i, where Δ_i = x0_i - μ_i
# ∫₀^Δt x_i(s) ds = Δ_i sin(Δt) + θ_i (1 - cos(Δt)) + μ_i Δt
function _integrate_segment(::typeof(Statistics.mean), flow::Boomerang, x0, x1, θ0, θ1, t0, t1)
    dt = t1 - t0
    s, c = sincos(dt)
    μ = flow.μ
    return @. (x0 - μ) * s + θ0 * (1 - c) + μ * dt
end

function _integrate_segment(::typeof(Statistics.var), flow::Boomerang, x0, x1, θ0, θ1, t0, t1, μ_est)
    dt = t1 - t0
    μ = flow.μ
    s, c = sincos(dt)
    s2, c2 = sincos(2dt)

    result = similar(x0)
    for i in eachindex(x0)
        Δ = x0[i] - μ[i]
        v = θ0[i]
        m = μ_est[i]

        # x_i(s) = Δ cos(s) + v sin(s) + μ[i]
        # (x_i(s) - m)^2 = (Δ cos(s) + v sin(s) + μ[i] - m)^2
        # let a = Δ, b = v, c0 = μ[i] - m
        a = Δ
        b = v
        c0 = μ[i] - m

        # ∫₀^dt (a cos(s) + b sin(s) + c0)^2 ds
        # = a²/2 (dt + sin(2dt)/2) + b²/2 (dt - sin(2dt)/2) + c0² dt
        #   + 2ab/2 (1 - cos(2dt))/2    -- wait, let me be precise:
        #   - ab cos(2dt)/2  term from cross
        # Expand: a²cos²(s) + b²sin²(s) + c0² + 2ab cos(s)sin(s) + 2a c0 cos(s) + 2b c0 sin(s)
        # ∫cos²(s) ds = s/2 + sin(2s)/4
        # ∫sin²(s) ds = s/2 - sin(2s)/4
        # ∫cos(s)sin(s) ds = sin²(s)/2  = (1 - cos(2s))/4 ... actually = -cos(2s)/4 + const
        # more precisely: ∫₀^dt sin(s)cos(s) ds = sin²(dt)/2
        # ∫cos(s) ds = sin(s)
        # ∫sin(s) ds = 1 - cos(s)

        result[i] = (a^2 * (dt / 2 + s2 / 4) +
                      b^2 * (dt / 2 - s2 / 4) +
                      c0^2 * dt +
                      a * b * s^2 +
                      2 * a * c0 * s +
                      2 * b * c0 * (1 - c))
    end
    return result
end

function _integrate_segment(::typeof(Statistics.cov), flow::Boomerang, x0, x1, θ0, θ1, t0, t1, μ_est)
    d = length(x0)
    segment_integral = zeros(d, d)
    _integrate_segment!(segment_integral, Statistics.cov, flow, x0, x1, θ0, θ1, t0, t1, μ_est)
    return segment_integral
end

function _integrate_segment!(buf::AbstractMatrix, ::typeof(Statistics.cov), flow::Boomerang, x0, x1, θ0, θ1, t0, t1, μ_est)
    dt = t1 - t0
    μ = flow.μ
    d = length(x0)
    s, c = sincos(dt)
    s2, c2 = sincos(2dt)

    for j in 1:d, i in j:d
        aᵢ = x0[i] - μ[i]
        bᵢ = θ0[i]
        cᵢ = μ[i] - μ_est[i]
        aⱼ = x0[j] - μ[j]
        bⱼ = θ0[j]
        cⱼ = μ[j] - μ_est[j]

        # ∫₀^dt (aᵢ cos + bᵢ sin + cᵢ)(aⱼ cos + bⱼ sin + cⱼ) ds
        val = (aᵢ * aⱼ * (dt / 2 + s2 / 4) +
               bᵢ * bⱼ * (dt / 2 - s2 / 4) +
               (aᵢ * bⱼ + aⱼ * bᵢ) * s^2 / 2 +  # ∫cos sin = sin²/2, cross has two terms
               cᵢ * cⱼ * dt +
               (aᵢ * cⱼ + aⱼ * cᵢ) * s +
               (bᵢ * cⱼ + bⱼ * cᵢ) * (1 - c))

        buf[i, j] += val
        i != j && (buf[j, i] += val)
    end
    return buf
end

function _integrate_segment(::typeof(inclusion_probs), flow::Boomerang, x0, x1, θ0, θ1, t0, t1)
    result = zeros(length(x0))
    _integrate_segment!(result, inclusion_probs, flow, x0, x1, θ0, θ1, t0, t1)
    return result
end

function _integrate_segment!(buf::AbstractVector, ::typeof(inclusion_probs), flow::Boomerang, x0, x1, θ0, θ1, t0, t1)
    # x_i(s) = (x0[i] - μ[i]) cos(s) + θ0[i] sin(s) + μ[i]
    # x_i(s) == 0 only on a set of measure zero for most initial conditions,
    # so inclusion_probs integral = dt when the trajectory is not identically zero.
    μ = flow.μ
    dt = t1 - t0
    for i in eachindex(x0)
        Δ = x0[i] - μ[i]
        v = θ0[i]
        if !(iszero(Δ) && iszero(v) && iszero(μ[i]))
            buf[i] += dt
        end
    end
    return buf
end


# ──────────────────────────────────────────────────────────────────────────────
# Time-below-threshold for individual segments (used by `cdf` and `quantile`)
# ──────────────────────────────────────────────────────────────────────────────

"""
    _time_below_segment(flow, x0j, θ0j, τ, q[, μj])

Return the duration within a segment `[0, τ]` for which coordinate `j` satisfies `x_j(t) ≤ q`.
Linear dynamics: `x_j(t) = x0j + θ0j * t`.
Boomerang dynamics: `x_j(t) = μj + (x0j - μj) cos(t) + θ0j sin(t)`.
"""
function _time_below_segment(::Union{ZigZag, BouncyParticle}, x0j::Real, θ0j::Real, τ::Real, q::Real)
    if iszero(θ0j)
        return x0j ≤ q ? τ : zero(τ)
    end
    t_cross = (q - x0j) / θ0j
    t_clamped = clamp(t_cross, zero(τ), τ)
    return θ0j > 0 ? t_clamped : τ - t_clamped
end

function _time_below_segment(flow::Boomerang, x0j::Real, θ0j::Real, τ::Real, q::Real, μj::Real)
    a = x0j - μj
    b = θ0j
    C = q - μj
    R = hypot(a, b)

    (R ≤ 0 || C ≥ R)  && return τ
    C ≤ -R             && return zero(τ)

    φ = atan(b, a)
    α = acos(clamp(C / R, -one(R), one(R)))

    crossings = Float64[]
    for sign in (1, -1)
        base = φ + sign * α
        k_min = ceil(Int, -base / (2π))
        k_max = floor(Int, (τ - base) / (2π))
        for k in k_min:k_max
            t = base + 2π * k
            if 0 < t < τ
                push!(crossings, t)
            end
        end
    end

    sort!(crossings)
    pushfirst!(crossings, 0.0)
    push!(crossings, Float64(τ))

    time_below = 0.0
    for i in 1:length(crossings) - 1
        t_mid = (crossings[i] + crossings[i + 1]) / 2
        x_mid = μj + a * cos(t_mid) + b * sin(t_mid)
        if x_mid ≤ q
            time_below += crossings[i + 1] - crossings[i]
        end
    end
    return time_below
end

_time_below_segment(pd::PreconditionedDynamics, args...) = _time_below_segment(pd.dynamics, args...)

function _integrate(trace::AbstractPDMPTrace, f, args...)

    flow = trace.flow

    # the stuff below works for both traces, but has a few duplications
    # this should be easier to adapt to avoid allocations though

    iter = trace
    next = iterate(iter)
    isnothing(next) && error("Cannot compute statistics on an empty trace")

    t₀, x_state, θ_state, _ = next[2]
    xt₀ = copy(x_state)
    θt₀ = copy(θ_state)

    start_time = t₀
    next = iterate(iter, next[2])
    isnothing(next) && error("Cannot compute statistics on a trace with fewer than 2 events")
    t₁, x_state, θ_state, _ = next[2]
    xt₁ = copy(x_state) # Copy for safety, though strictly only xt0 needs to be frozen
    θt₁ = copy(θ_state)

    integral = _integrate_segment(
        f,
        flow,
        xt₀, xt₁,
        θt₀, θt₁,
        t₀,  t₁,
        args...
    )
    t₀ = t₁
    copyto!(xt₀, xt₁)
    copyto!(θt₀, θt₁)
    has_inplace = hasmethod(_integrate_segment!, Tuple{typeof(integral), typeof(f), typeof(flow), typeof(xt₀), typeof(xt₁), typeof(θt₀), typeof(θt₁), typeof(t₀), typeof(t₁), map(typeof, args)...})
    next = iterate(iter, next[2])
    while next !== nothing
        t₁, x_state, θ_state, _ = next[2]
        copyto!(xt₁, x_state)
        copyto!(θt₁, θ_state)
        if has_inplace
            _integrate_segment!(integral, f, flow, xt₀, xt₁, θt₀, θt₁, t₀, t₁, args...)
        else
            integral .+= _integrate_segment(f, flow, xt₀, xt₁, θt₀, θt₁, t₀, t₁, args...)
        end
        t₀ = t₁
        copyto!(xt₀, xt₁)
        copyto!(θt₀, θt₁)
        next = iterate(iter, next[2])
    end
    end_time = t₁
    total_time = end_time - start_time

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
    # k = trace.flow isa FactorizedDynamics ? 1 : 2
    # k = isfactorized(trace.flow) ? 1 : 2
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
        # do NOT step back for FactorizedTrace: each event was already applied via move_forward_time!
    else
        # For the non-factorized case, we can skip to the last event directly.
        # Keep k_current at its advanced position to avoid O(n*m) rescanning.
        found_event = false
        last_before = k_current
        while k_current <= length(trace.events) && trace.events[k_current].time < t_new
            last_before = k_current
            found_event = true
            k_current += 1
        end
        if found_event
            t_current = _to_next_event!(x_current, θ_current, trace.events[last_before])
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
