# Copy a callable: use copy for structs that define it, identity fallback for closures
_copy_callable(f) = _has_copy(f) ? copy(f) : f
_has_copy(f) = applicable(copy, f) && !(f isa Function)
_try_copy(x) = applicable(copy, x) ? copy(x) : x
_copy_callable(f::Base.Fix1) = Base.Fix1(_copy_callable(f.f), _try_copy(f.x))
_copy_callable(f::Base.Fix2) = Base.Fix2(_copy_callable(f.f), _try_copy(f.x))

"""
    set_active_set!(object, free::BitVector)

Synchronize stateful gradients or targets with the active/free coordinates of
a `StickyPDMPState`. The default method is a no-op.
"""
set_active_set!(object, free::BitVector) = nothing

function set_active_set!(f::Base.Fix1, free::BitVector)
    set_active_set!(f.f, free)
    set_active_set!(f.x, free)
    return nothing
end
function set_active_set!(f::Base.Fix2, free::BitVector)
    set_active_set!(f.f, free)
    set_active_set!(f.x, free)
    return nothing
end

# Concrete gradient strategies
struct FullGradient{F} <: GlobalGradientStrategy
    f::F
end

Base.copy(g::FullGradient) = FullGradient(_copy_callable(g.f))

"""
    SeparableResidualEnvelope(weights, component_scales!)

An additive residual-rate envelope with
`b_i(t) = sum(weights[r, i] * c_r(t), r)`. `component_scales!` is called as
`component_scales!(out, state, t)`. Along a linear GridThinning trajectory
each scale must be affine in `t`.

The resulting `b_i(t)` must dominate the observation's perturbation of the
chosen dynamics' event rate. For BPS this is
`abs(dot(v, r_i(x)))`; for ZigZag it is
`sum(abs(v[j] * r_i(x)[j]), j)`, which is the bound needed by its
coordinatewise flip rates.
"""
mutable struct SeparableResidualEnvelope{T<:AbstractMatrix,F}
    weights::T
    component_scales!::F
    totals::Vector{Float64}
    alias_tables::Vector{Union{Nothing,AliasTables.AliasTable{UInt64,Int}}}
    scales::Vector{Float64}
end

function SeparableResidualEnvelope(weights::AbstractMatrix, component_scales!)
    isempty(weights) && throw(ArgumentError("residual envelope weights must be nonempty"))
    any(x -> !isfinite(x) || x < 0, weights) &&
        throw(ArgumentError("residual envelope weights must be finite and nonnegative"))
    stored_weights = Matrix{Float64}(weights)
    totals = vec(sum(stored_weights; dims=2))
    tables = Union{Nothing,AliasTables.AliasTable{UInt64,Int}}[
        ispositive(totals[r]) ? AliasTables.AliasTable(view(stored_weights, r, :)) : nothing
        for r in axes(stored_weights, 1)
    ]
    return SeparableResidualEnvelope(stored_weights, component_scales!, totals,
        tables, zeros(Float64, size(stored_weights, 1)))
end

"""
    MarkedControlVariate(deterministic_gradient!, residual_oracle, envelope,
                         anchor, m; begin_search! = ..., deterministic_hvp! = nothing,
                         full_gradient! = nothing)

Exact marked-minibatch gradient strategy for GridThinning. The residual oracle
is called as `oracle(out, x, subset, frozen_anchor)` and returns the unscaled
sum of observation residual gradients for precisely `subset`. Julia owns the
`N/m` scaling. `begin_search!` runs once before the anchor is frozen.
The optional full-gradient callback is deliberately outside candidate and
reflection evaluation and is available only to explicit anchor-management
code through `full_gradient!`.
"""
mutable struct MarkedControlVariate{F,O,E<:SeparableResidualEnvelope,A,B,H,G} <: GlobalGradientStrategy
    deterministic_gradient!::F
    residual_oracle::O
    envelope::E
    anchor::A
    active_anchor::A
    begin_search_callback!::B
    deterministic_hvp!::H
    full_gradient!::G
    m::Int
    subset::Vector{Int}
    residual_buffer::Vector{Float64}
    sampling_map::Dict{Int,Int}
end

function MarkedControlVariate(deterministic_gradient!, residual_oracle,
    envelope::SeparableResidualEnvelope, anchor::AbstractVector, m::Integer;
    begin_search! = (cv, state) -> nothing, deterministic_hvp! = nothing,
    full_gradient! = nothing)
    N = size(envelope.weights, 2)
    1 <= m <= N || throw(ArgumentError("minibatch size m must lie in 1:N"))
    a = collect(Float64, anchor)
    return MarkedControlVariate(deterministic_gradient!, residual_oracle, envelope,
        a, copy(a), begin_search!, deterministic_hvp!, full_gradient!, Int(m),
        Vector{Int}(undef, m), zeros(Float64, length(a)), Dict{Int,Int}())
end

Base.copy(cv::MarkedControlVariate) = MarkedControlVariate(
    _copy_callable(cv.deterministic_gradient!), _copy_callable(cv.residual_oracle),
    deepcopy(cv.envelope), copy(cv.anchor), cv.m;
    begin_search! = _copy_callable(cv.begin_search_callback!),
    deterministic_hvp! = _copy_callable(cv.deterministic_hvp!),
    full_gradient! = _copy_callable(cv.full_gradient!))

function begin_search!(cv::MarkedControlVariate, state::AbstractPDMPState)
    cv.begin_search_callback!(cv, state)
    copyto!(cv.active_anchor, cv.anchor)
    return cv
end

function component_scales!(out, envelope::SeparableResidualEnvelope, state, t)
    envelope.component_scales!(out, state, t)
    length(out) == length(envelope.totals) || throw(DimensionMismatch("wrong number of component scales"))
    any(x -> !isfinite(x) || x < 0, out) &&
        throw(ArgumentError("residual-envelope component scales must be finite and nonnegative"))
    return out
end

function total_residual_bound(envelope::SeparableResidualEnvelope, state, t)
    scales = component_scales!(envelope.scales, envelope, state, t)
    return dot(scales, envelope.totals)
end

deterministic_gradient!(out, cv::MarkedControlVariate, x) = cv.deterministic_gradient!(out, x)
function deterministic_hvp!(out, cv::MarkedControlVariate, x, v)
    cv.deterministic_hvp! === nothing && throw(ArgumentError("no deterministic HVP callback is available"))
    cv.deterministic_hvp!(out, x, v)
end
function full_gradient!(out, cv::MarkedControlVariate, x)
    cv.full_gradient! === nothing && throw(ArgumentError(
        "no full-gradient callback is available; it is only needed for explicit anchor creation or updates"))
    cv.full_gradient!(out, x)
end
residual_gradient!(out, oracle, x, subset, anchor) = oracle(out, x, subset, anchor)

struct SubsampledGradient{F1, F2, F3, F4} <: GlobalGradientStrategy
    f::F1
    resample_indices!::F2
    update_anchor!::F3
    full::FullGradient{F4}
    nsub::Int
    no_anchor_updates::Int
    use_full_gradient_for_reflections::Bool
    resample_dt::Float64
    fixed_batch_within_event::Bool
end

Base.copy(g::SubsampledGradient) = SubsampledGradient(
    _copy_callable(g.f),
    _copy_callable(g.resample_indices!),
    _copy_callable(g.update_anchor!),
    copy(g.full),
    g.nsub,
    g.no_anchor_updates,
    g.use_full_gradient_for_reflections,
    g.resample_dt,
    g.fixed_batch_within_event,
)

SubsampledGradient(f::F1, resample_indices!::F2, update_anchor!::F3, full::FullGradient{F4},
                   nsub::Int, no_anchor_updates::Int, use_full_gradient_for_reflections::Bool,
                   resample_dt::Real) where {F1,F2,F3,F4} =
    SubsampledGradient(f, resample_indices!, update_anchor!, full, nsub, no_anchor_updates,
                       use_full_gradient_for_reflections, resample_dt, false)

# temporary backwards compatibility constructor for now
# SubsampledGradient(f::F1, resample_indices!::F2, nsub::Int) where {F1, F2} =
#     SubsampledGradient(f, resample_indices!, (trace) -> nothing, (args...) -> nothing, nsub, 0)
SubsampledGradient(f::Function, resample_indices!::Function, nsub::Int) =
    SubsampledGradient(f, resample_indices!, (trace) -> nothing, FullGradient((args...) -> nothing), nsub, 0, false, 0.0, false)

SubsampledGradient(f::Function, resample_indices!::Function, update_anchor!::Function, full::Function,
                   nsub::Int, no_anchor_updates::Int, use_full_gradient_for_reflections::Bool;
                   resample_dt::Float64=0.0, fixed_batch_within_event::Bool=false) =
    SubsampledGradient(f, resample_indices!, update_anchor!, FullGradient(full), nsub, no_anchor_updates,
                       use_full_gradient_for_reflections, resample_dt, fixed_batch_within_event)

struct CoordinateWiseGradient{F} <: CoordinateWiseGradientStrategy
    f::F
end

Base.copy(g::CoordinateWiseGradient) = CoordinateWiseGradient(_copy_callable(g.f))

with_stats(grad::FullGradient,       stats::AbstractStatisticCounter) =
    FullGradient(with_stats(grad.f, stats, Val(:ordinary_full_gradient)))
function with_stats(grad::SubsampledGradient, stats::AbstractStatisticCounter)
    SubsampledGradient(
        with_stats(grad.f, stats, Val(:stochastic_gradient)),
        grad.resample_indices!,
        grad.update_anchor!,
        FullGradient(with_stats(grad.full.f, stats, Val(:full_reflection_gradient))),
        grad.nsub,
        grad.no_anchor_updates,
        grad.use_full_gradient_for_reflections,
        grad.resample_dt,
        grad.fixed_batch_within_event,
    )
end

struct WithResidualStats{F,S}
    f::F
    stats::S
end
function (ws::WithResidualStats)(args...)
    _inc_counter_residual_oracle_calls(ws.stats)
    return ws.f(args...)
end

function with_stats(cv::MarkedControlVariate, stats::AbstractStatisticCounter)
    return MarkedControlVariate(
        WithStats(cv.deterministic_gradient!, stats, Val(:deterministic_gradient)),
        WithResidualStats(cv.residual_oracle, stats), deepcopy(cv.envelope),
        copy(cv.anchor), cv.m;
        begin_search! = cv.begin_search_callback!,
        deterministic_hvp! = cv.deterministic_hvp!,
        full_gradient! = cv.full_gradient! === nothing ? nothing :
            WithStats(cv.full_gradient!, stats, Val(:full_gradient)))
end
with_stats(grad::CoordinateWiseGradient, stats::AbstractStatisticCounter) =
    CoordinateWiseGradient(with_stats(grad.f, stats, Val(:ordinary_full_gradient)))

with_stats(f, stats::AbstractStatisticCounter) = WithStats(f, stats, Val(:ordinary_full_gradient))
with_stats(f, stats::AbstractStatisticCounter, purpose::Val) = WithStats(f, stats, purpose)

struct WithStats{F,S,P} <: Function
    f::F
    stats::S
end
WithStats(f::F, stats::S, ::Val{P}) where {F,S,P} = WithStats{F,S,P}(f, stats)

(ws::WithStats{F,S,P})(args...) where {F,S,P} = begin
    _inc_counter_∇f_calls(ws.stats)
    _inc_gradient_purpose!(ws.stats, Val(P))
    ws.f(args...)
end
set_active_set!(ws::WithStats, free::BitVector) = set_active_set!(ws.f, free)

struct WithFDCurvatureStats{F,S} <: Function
    f::F
    stats::S
end

(ws::WithFDCurvatureStats)(args...) = begin
    _inc_counter_fd_curvature_gradient_calls(ws.stats)
    ws.f(args...)
end
set_active_set!(ws::WithFDCurvatureStats, free::BitVector) = set_active_set!(ws.f, free)


# struct ControlVariateGradient{F} <: GradientStrategy
#     f::F
#     subsample_size::Int
#     reference_point::Vector{Float64}
#     cached_full_gradient::Vector{Float64}
#     control_frequency::Int
# end


# Gradient computation interface

set_active_set!(strategy::FullGradient, free::BitVector) = set_active_set!(strategy.f, free)
function set_active_set!(strategy::SubsampledGradient, free::BitVector)
    set_active_set!(strategy.f, free)
    set_active_set!(strategy.full, free)
    return nothing
end
set_active_set!(strategy::CoordinateWiseGradient, free::BitVector) = set_active_set!(strategy.f, free)
set_active_set!(::MarkedControlVariate, ::BitVector) = nothing

# Main entry point: compute gradient from state
function compute_gradient!(state::AbstractPDMPState, gradient_strategy::GradientStrategy, flow::ContinuousDynamics, cache)
    ∇ϕx = compute_gradient!(gradient_strategy, state.ξ.x, cache.∇ϕx)
    correct_gradient!(∇ϕx, state.ξ.x, state.ξ.θ, flow, cache)
    return ∇ϕx
end

# Compatibility: compute gradient from position and velocity vectors (used in gridthinning.jl)
function compute_gradient!(x::AbstractVector, θ::AbstractVector, gradient_strategy::GradientStrategy, flow::ContinuousDynamics, cache)
    ∇ϕx = compute_gradient!(gradient_strategy, x, cache.∇ϕx)
    correct_gradient!(∇ϕx, x, θ, flow, cache)
    return ∇ϕx
end

# For reflection events with subsampled gradients, may use full gradient
function compute_gradient_for_reflection!(state::AbstractPDMPState, gradient_strategy::GradientStrategy, flow::ContinuousDynamics, cache)
    ∇ϕx = compute_gradient_for_reflection!(gradient_strategy, state.ξ.x, cache.∇ϕx)
    correct_gradient!(∇ϕx, state.ξ.x, state.ξ.θ, flow, cache)
    return ∇ϕx
end

# Compute raw gradient (strategy-specific implementations)
function compute_gradient!(strategy::FullGradient, x, out)
    strategy.f(out, x)
    return out
end

function compute_gradient!(strategy::SubsampledGradient, x, out)
    strategy.f(out, x)
    return out
end

function compute_gradient!(strategy::MarkedControlVariate, x, out)
    deterministic_gradient!(out, strategy, x)
    return out
end

function compute_gradient!(strategy::CoordinateWiseGradient, x, i::Integer, cache)
    cache.∇ϕx[i] = strategy.f(x, i)
    return cache.∇ϕx[i]
end

# For subsampled gradients, optionally use full gradient for reflections
function compute_gradient_for_reflection!(strategy::FullGradient, x, out)
    strategy.f(out, x)
    return out
end

function compute_gradient_for_reflection!(strategy::SubsampledGradient, x, out)
    if strategy.use_full_gradient_for_reflections
        compute_gradient_for_reflection!(strategy.full, x, out)
    else
        strategy.f(out, x)
    end
    return out
end


compute_gradient_for_reflection!(strategy::MarkedControlVariate, x, out) =
    compute_gradient!(strategy, x, out)
