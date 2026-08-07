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
`component_scales!(out, state, flow, t)`. Along a linear GridThinning or marked
`ThinningStrategy` trajectory each scale must be affine in `t`.

The resulting `b_i(t)` must dominate the observation's perturbation of the
chosen dynamics' event rate. For BPS this is
`abs(dot(v, r_i(x)))`; for ZigZag it is
`sum(abs(v[j] * r_i(x)[j]), j)`, which is the bound needed by its
coordinatewise flip rates.
"""
struct SeparableResidualEnvelope{F,C}
    weights::Matrix{Float64}
    component_scales!::F
    component_cell_scales!::C
    totals::Vector{Float64}
    alias_tables::Vector{Union{Nothing,AliasTables.AliasTable{UInt64,Int}}}
    scales::Vector{Float64}
    cell_scales::Vector{Float64}
end

function SeparableResidualEnvelope(weights::AbstractMatrix, component_scales!;
        component_cell_scales! = nothing)
    isempty(weights) && throw(ArgumentError("residual envelope weights must be nonempty"))
    any(x -> !isfinite(x) || x < 0, weights) &&
        throw(ArgumentError("residual envelope weights must be finite and nonnegative"))
    stored_weights = Matrix{Float64}(weights)
    totals = vec(sum(stored_weights; dims=2))
    tables = Union{Nothing,AliasTables.AliasTable{UInt64,Int}}[
        ispositive(totals[r]) ? AliasTables.AliasTable(view(stored_weights, r, :)) : nothing
        for r in axes(stored_weights, 1)
    ]
    return SeparableResidualEnvelope(stored_weights, component_scales!,
        component_cell_scales!, totals,
        tables, zeros(Float64, size(stored_weights, 1)),
        zeros(Float64, size(stored_weights, 1)))
end

struct TrajectoryComponentScales
    anchor::Vector{Float64}
    growth_rates::Vector{Float64}
end

struct DampedHCVComponentScales
    anchor::Vector{Float64}
    damping::Float64
end

_stored_marked_anchor(anchor::Vector{Float64}) = anchor
_stored_marked_anchor(anchor::AbstractVector) = collect(Float64, anchor)

_trajectory_scale_anchor(::Any) = nothing
_trajectory_scale_anchor(scales::TrajectoryComponentScales) = scales.anchor
_trajectory_scale_anchor(scales::DampedHCVComponentScales) = scales.anchor
_trajectory_scale_anchor(envelope::SeparableResidualEnvelope) =
    _trajectory_scale_anchor(envelope.component_scales!)
function _marked_anchor_owner(envelope::SeparableResidualEnvelope, fallback)
    trajectory_anchor = _trajectory_scale_anchor(envelope)
    return trajectory_anchor === nothing ? fallback : trajectory_anchor
end

"""
    TrajectoryResidualEnvelope(weights, anchor; growth_rates=zeros(...))

Construct a separable residual envelope whose likelihood-specific rows are
combined with certified dynamics-specific trajectory geometry. A component
with growth rate `g` is scaled by `V(t) * δ(t) * exp(g * δ(t))`, where `δ`
dominates displacement from `anchor` and `V` is the appropriate dual event-rate
velocity norm. This covers globally Lipschitz likelihoods (`g = 0`) and binned
log-link likelihood bounds without embedding likelihood concepts in a flow.
"""
function TrajectoryResidualEnvelope(weights::AbstractMatrix,
        anchor::AbstractVector; growth_rates=zeros(size(weights, 1)))
    growth = collect(Float64, growth_rates)
    length(growth) == size(weights, 1) || throw(DimensionMismatch(
        "growth_rates must have one entry per envelope component"))
    any(x -> !isfinite(x) || x < 0, growth) && throw(ArgumentError(
        "growth_rates must be finite and nonnegative"))
    scales = TrajectoryComponentScales(_stored_marked_anchor(anchor), growth)
    return SeparableResidualEnvelope(weights, scales;
        component_cell_scales! = scales)
end

"""
    DampedHCVResidualEnvelope(first_order_weights, remainder_weights, anchor;
                              damping=length(anchor))

Construct the generic rank-two envelope for a damped analytic Hessian control
variate. If `δ(t)` bounds displacement from the anchor, `V(t)` is the
dynamics-specific dual velocity bound, and
`α(t) = damping / (damping + δ(t)^2)`, the component scales are
`V δ (1-α)` and `V δ^2 α`. The integration layer remains responsible for
supplying nonnegative first-order and Taylor-remainder weights.
"""
function DampedHCVResidualEnvelope(first_order_weights::AbstractVector,
        remainder_weights::AbstractVector, anchor::AbstractVector;
        damping::Real=length(anchor))
    length(first_order_weights) == length(remainder_weights) ||
        throw(DimensionMismatch("HCV envelope weight vectors must have equal length"))
    isfinite(damping) && damping > 0 || throw(ArgumentError(
        "HCV damping must be finite and positive"))
    weights = permutedims(hcat(first_order_weights, remainder_weights))
    scales = DampedHCVComponentScales(
        _stored_marked_anchor(anchor), Float64(damping))
    return SeparableResidualEnvelope(weights, scales;
        component_cell_scales! = scales)
end

"""
    MarkedControlVariate(deterministic_gradient!, residual_oracle, envelope,
                         anchor, m; deterministic_hvp! = nothing,
                         refresh_anchor! = nothing)

Exact marked-minibatch gradient strategy for `GridThinningStrategy` and
`ThinningStrategy`. The residual oracle is called as
`oracle(out, x, subset, frozen_anchor)` and returns the unscaled sum of
observation residual gradients for precisely `subset`. Julia owns the `N/m`
scaling. The accepted stochastic gradient is retained in the event metadata
and reused for reflection; it is not replaced by a full gradient. Anchors may
only be changed through the optional `refresh_anchor!` callback. The callback
receives the requested anchor and returns its complete prepared envelope;
`refresh_anchor!` then atomically installs both. It must return the same
concrete envelope type as the current envelope, so event-time dispatch remains
type stable. Changing envelope `weights`
requires reconstructing the envelope because `totals` and `alias_tables` are
derived from them.
"""
mutable struct MarkedControlVariate{F,O,E<:SeparableResidualEnvelope,H,R} <: GlobalGradientStrategy
    deterministic_gradient!::F
    residual_oracle::O
    envelope::E
    anchor::Vector{Float64}
    deterministic_hvp!::H
    refresh_anchor_callback!::R
    m::Int
    subset::Vector{Int}
    residual_buffer::Vector{Float64}
    sampling_map::Dict{Int,Int}
end

function MarkedControlVariate(deterministic_gradient!, residual_oracle,
    envelope::SeparableResidualEnvelope, anchor::AbstractVector, m::Integer;
    deterministic_hvp! = nothing, refresh_anchor! = nothing)
    N = size(envelope.weights, 2)
    1 <= m <= N || throw(ArgumentError("minibatch size m must lie in 1:N"))
    requested_anchor = collect(Float64, anchor)
    trajectory_anchor = _trajectory_scale_anchor(envelope)
    if trajectory_anchor !== nothing
        trajectory_anchor == requested_anchor || throw(ArgumentError(
            "TrajectoryResidualEnvelope and MarkedControlVariate anchors must match"))
    end
    active_anchor = _marked_anchor_owner(envelope, requested_anchor)
    return MarkedControlVariate(deterministic_gradient!, residual_oracle, envelope,
        active_anchor, deterministic_hvp!, refresh_anchor!, Int(m),
        Vector{Int}(undef, m), zeros(Float64, length(active_anchor)), Dict{Int,Int}())
end

function _validate_marked_anchor(envelope::SeparableResidualEnvelope, anchor)
    trajectory_anchor = _trajectory_scale_anchor(envelope)
    trajectory_anchor === nothing || trajectory_anchor == anchor || throw(ArgumentError(
        "marked residual envelope does not match the active anchor"))
    return nothing
end

function _validated_refreshed_envelope(cv::MarkedControlVariate,
        envelope::SeparableResidualEnvelope, requested)
    typeof(envelope) === typeof(cv.envelope) || throw(ArgumentError(
        "anchor-refresh provider must return the same concrete envelope type; " *
        "reconstruct the MarkedControlVariate to change callback types"))
    _validate_marked_anchor(envelope, requested)
    return envelope
end
_validated_refreshed_envelope(cv::MarkedControlVariate, value, requested) = throw(ArgumentError(
    "anchor-refresh provider must return a SeparableResidualEnvelope"))

function refresh_anchor!(cv::MarkedControlVariate, anchor::AbstractVector)
    callback = cv.refresh_anchor_callback!
    callback === nothing && throw(ArgumentError(
        "this MarkedControlVariate has no anchor-refresh provider"))
    length(anchor) == length(cv.anchor) || throw(DimensionMismatch(
        "new marked anchor has the wrong dimension"))
    requested = collect(Float64, anchor)
    new_envelope = _validated_refreshed_envelope(cv, callback(requested), requested)
    cv.envelope = new_envelope
    cv.anchor = _marked_anchor_owner(new_envelope, requested)
    return cv
end

function _reconstruct_marked(cv::MarkedControlVariate;
        deterministic_gradient! = cv.deterministic_gradient!,
        residual_oracle = cv.residual_oracle,
        envelope = deepcopy(cv.envelope),
        deterministic_hvp! = cv.deterministic_hvp!,
        refresh_anchor_callback! = cv.refresh_anchor_callback!)
    return MarkedControlVariate(deterministic_gradient!, residual_oracle,
        envelope, copy(cv.anchor), cv.m;
        deterministic_hvp! = deterministic_hvp!,
        refresh_anchor! = refresh_anchor_callback!)
end

function Base.copy(cv::MarkedControlVariate)
    cv.refresh_anchor_callback! === nothing || throw(ArgumentError(
        "an anchor-managed MarkedControlVariate cannot be copied safely; " *
        "construct one provider and model per chain"))
    return _reconstruct_marked(cv;
        deterministic_gradient! = _copy_callable(cv.deterministic_gradient!),
        residual_oracle = _copy_callable(cv.residual_oracle),
        deterministic_hvp! = _copy_callable(cv.deterministic_hvp!),
        refresh_anchor_callback! = _copy_callable(cv.refresh_anchor_callback!))
end

function component_scales!(out, envelope::SeparableResidualEnvelope, state, flow, t)
    envelope.component_scales!(out, state, flow, t)
    return _validate_component_scales(out, envelope.totals, "component")
end

function _validate_component_scales(out, totals, kind)
    length(out) == length(totals) || throw(DimensionMismatch(
        "wrong number of $(kind) scales"))
    any(x -> !isfinite(x) || x < 0, out) &&
        throw(ArgumentError("residual-envelope $(kind) scales must be finite and nonnegative"))
    return out
end

component_scales!(out, envelope::SeparableResidualEnvelope, state, t) =
    component_scales!(out, envelope, state, nothing, t)

function component_cell_scales!(out, envelope::SeparableResidualEnvelope,
        state, flow, left, right)
    callback = envelope.component_cell_scales!
    if callback === nothing
        # The original public constructor requires affine component scales.
        # Their maximum on a closed cell is attained at an endpoint.
        envelope.component_scales!(envelope.scales, state, flow, left)
        envelope.component_scales!(envelope.cell_scales, state, flow, right)
        @inbounds for r in eachindex(out, envelope.scales, envelope.cell_scales)
            out[r] = max(envelope.scales[r], envelope.cell_scales[r])
        end
    else
        callback(out, state, flow, left, right)
    end
    return _validate_component_scales(out, envelope.totals, "component cell")
end

function total_residual_bound(envelope::SeparableResidualEnvelope, state, flow, t)
    scales = component_scales!(envelope.scales, envelope, state, flow, t)
    return dot(scales, envelope.totals)
end

total_residual_bound(envelope::SeparableResidualEnvelope, state, t) =
    total_residual_bound(envelope, state, nothing, t)

deterministic_gradient!(out, cv::MarkedControlVariate, x) = cv.deterministic_gradient!(out, x)

struct CoordinateWiseGradient{F} <: CoordinateWiseGradientStrategy
    f::F
end

Base.copy(g::CoordinateWiseGradient) = CoordinateWiseGradient(_copy_callable(g.f))

with_stats(grad::FullGradient, stats::AbstractStatisticCounter) =
    FullGradient(with_stats(grad.f, stats, Val(:ordinary_full_gradient)))

struct WithResidualStats{F,S}
    f::F
    stats::S
end
function (ws::WithResidualStats)(args...)
    _inc_counter_residual_oracle_calls(ws.stats)
    return ws.f(args...)
end

function with_stats(cv::MarkedControlVariate, stats::AbstractStatisticCounter)
    return _reconstruct_marked(cv;
        deterministic_gradient! = WithStats(
            cv.deterministic_gradient!, stats, Val(:deterministic_gradient)),
        residual_oracle = WithResidualStats(cv.residual_oracle, stats))
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


# Gradient computation interface

set_active_set!(strategy::FullGradient, free::BitVector) = set_active_set!(strategy.f, free)
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

function compute_gradient!(strategy::MarkedControlVariate, x, out)
    deterministic_gradient!(out, strategy, x)
    return out
end

function compute_gradient!(strategy::CoordinateWiseGradient, x, i::Integer, cache)
    cache.∇ϕx[i] = strategy.f(x, i)
    return cache.∇ϕx[i]
end

function compute_gradient_for_reflection!(strategy::FullGradient, x, out)
    strategy.f(out, x)
    return out
end

compute_gradient_for_reflection!(strategy::MarkedControlVariate, x, out) =
    compute_gradient!(strategy, x, out)
