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
    SeparableResidualEnvelope(weights, component_scales!;
                              component_cell_scales! = nothing,
                              certified_affine = false)

An additive residual-rate envelope with
`b_i(t) = sum(weights[r, i] * c_r(t), r)`. `component_scales!` is called as
`component_scales!(out, state, flow, t)`. An arbitrary pointwise callback must
also provide `component_cell_scales!`, which certifies a bound over every closed
grid cell. Set `certified_affine=true` only when every component scale is known
to be affine in `t`; endpoint maxima are then a valid cell certificate and the
same affine callback may be used by `ThinningStrategy` on linear trajectories.

The resulting `b_i(t)` must dominate the observation's perturbation of the
chosen dynamics' event rate. For BPS this is
`abs(dot(v, r_i(x)))`; for ZigZag it is
`sum(abs(v[j] * r_i(x)[j]), j)`, which is the bound needed by its
coordinatewise flip rates.
"""
abstract type AbstractResidualEnvelope end

struct SeparableResidualEnvelope{F,C} <: AbstractResidualEnvelope
    weights::Matrix{Float64}
    component_scales!::F
    component_cell_scales!::C
    totals::Vector{Float64}
    alias_tables::Vector{Union{Nothing,AliasTables.AliasTable{UInt64,Int}}}
    scales::Vector{Float64}
    cell_scales::Vector{Float64}
    cumulative_masses::Vector{Float64}
end

"""
    GroupedResidualEnvelope(groups, component_scales!;
                            component_cell_scales!)

Compact separable envelope for one-hot component memberships. `groups[k, i]`
is the component containing observation `i` in partition `k`. This represents
the same bound as a zero-one component-by-observation matrix without storing
that dense matrix or one full alias table per component.
"""
struct GroupedResidualEnvelope{F,C} <: AbstractResidualEnvelope
    groups::Matrix{Int}
    members::Vector{Vector{Int}}
    component_scales!::F
    component_cell_scales!::C
    totals::Vector{Float64}
    scales::Vector{Float64}
    cell_scales::Vector{Float64}
    cumulative_masses::Vector{Float64}
end

function GroupedResidualEnvelope(groups::AbstractMatrix{<:Integer},
        n_components::Integer, component_scales!;
        component_cell_scales! = nothing)
    n_components >= 1 || throw(ArgumentError(
        "grouped residual envelope must have at least one component"))
    stored_groups = Matrix{Int}(groups)
    isempty(stored_groups) && throw(ArgumentError(
        "grouped residual envelope memberships must be nonempty"))
    all(group -> 1 <= group <= n_components, stored_groups) ||
        throw(ArgumentError("grouped residual envelope memberships are out of range"))
    members = [Int[] for _ in 1:n_components]
    totals = zeros(Float64, n_components)
    @inbounds for observation in axes(stored_groups, 2), partition in axes(stored_groups, 1)
        component = stored_groups[partition, observation]
        push!(members[component], observation)
        totals[component] += 1.0
    end
    all(member -> !isempty(member), members) || throw(ArgumentError(
        "every grouped residual-envelope component must contain an observation"))
    component_cell_scales! === nothing && throw(ArgumentError(
        "grouped residual envelopes require an explicit certified component_cell_scales! callback"))
    return GroupedResidualEnvelope(stored_groups, members, component_scales!,
        component_cell_scales!, totals, zeros(n_components), zeros(n_components),
        zeros(n_components))
end

"""Internal wrapper recording the caller's explicit affine certification."""
struct CertifiedAffineComponentScales{F}
    callback::F
end
(provider::CertifiedAffineComponentScales)(args...) = provider.callback(args...)

function SeparableResidualEnvelope(weights::AbstractMatrix, component_scales!;
        component_cell_scales! = nothing, certified_affine::Bool=false)
    isempty(weights) && throw(ArgumentError("residual envelope weights must be nonempty"))
    any(x -> !isfinite(x) || x < 0, weights) &&
        throw(ArgumentError("residual envelope weights must be finite and nonnegative"))
    stored_weights = Matrix{Float64}(weights)
    totals = vec(sum(stored_weights; dims=2))
    tables = Union{Nothing,AliasTables.AliasTable{UInt64,Int}}[
        ispositive(totals[r]) ? AliasTables.AliasTable(view(stored_weights, r, :)) : nothing
        for r in axes(stored_weights, 1)
    ]
    component_cell_scales! === nothing && !certified_affine && throw(ArgumentError(
        "arbitrary residual-envelope component scales require an explicit certified " *
        "component_cell_scales! callback; set certified_affine=true only for " *
        "component scales that are affine in time"))
    stored_scales = certified_affine ?
        CertifiedAffineComponentScales(component_scales!) : component_scales!
    return SeparableResidualEnvelope(stored_weights, stored_scales,
        component_cell_scales!, totals,
        tables, zeros(Float64, size(stored_weights, 1)),
        zeros(Float64, size(stored_weights, 1)),
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

_stored_subsampling_anchor(anchor::Vector{Float64}) = anchor
_stored_subsampling_anchor(anchor::AbstractVector) = collect(Float64, anchor)

_trajectory_scale_anchor(::Any) = nothing
_trajectory_scale_anchor(scales::TrajectoryComponentScales) = scales.anchor
_trajectory_scale_anchor(scales::DampedHCVComponentScales) = scales.anchor
_trajectory_scale_anchor(scales::CertifiedAffineComponentScales) =
    _trajectory_scale_anchor(scales.callback)
_trajectory_scale_anchor(envelope::AbstractResidualEnvelope) =
    _trajectory_scale_anchor(envelope.component_scales!)
function _subsampling_anchor_owner(envelope::AbstractResidualEnvelope, fallback)
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
    scales = TrajectoryComponentScales(_stored_subsampling_anchor(anchor), growth)
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
        _stored_subsampling_anchor(anchor), Float64(damping))
    return SeparableResidualEnvelope(weights, scales;
        component_cell_scales! = scales)
end

"""
    SubsampledControlVariate(deterministic_gradient!, residual_oracle, envelope,
                         anchor, m; deterministic_hvp! = nothing,
                         refresh_anchor! = nothing)

Exact subsampling-minibatch gradient strategy for `GridThinningStrategy` and
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
abstract type AbstractSubsamplingDesign end
struct UniformSubsamplingDesign <: AbstractSubsamplingDesign end

"""Internal balanced design for equal-size contiguous observation strata."""
struct BalancedStratifiedSubsamplingDesign <: AbstractSubsamplingDesign
    n_strata::Int
    stratum_size::Int
    per_stratum::Int
end

function balanced_stratified_subsampling_design(N::Integer, n_strata::Integer,
        m::Integer)
    n_strata >= 1 || throw(ArgumentError("number of subsampling strata must be positive"))
    N % n_strata == 0 || throw(ArgumentError(
        "balanced subsampling strata must have equal sizes"))
    m % n_strata == 0 || throw(ArgumentError(
        "subsample size must be divisible by the number of strata"))
    stratum_size = N ÷ n_strata
    per_stratum = m ÷ n_strata
    1 <= per_stratum <= stratum_size || throw(ArgumentError(
        "each subsampling stratum must contribute between one and all entries"))
    return BalancedStratifiedSubsamplingDesign(
        Int(n_strata), Int(stratum_size), Int(per_stratum))
end

mutable struct SubsampledControlVariate{F,O,E<:AbstractResidualEnvelope,H,R,D<:AbstractSubsamplingDesign} <: GlobalGradientStrategy
    deterministic_gradient!::F
    residual_oracle::O
    envelope::E
    anchor::Vector{Float64}
    deterministic_hvp!::H
    refresh_anchor_callback!::R
    subset_design::D
    m::Int
    subset::Vector{Int}
    residual_buffer::Vector{Float64}
    sampling_map::Dict{Int,Int}
end

function SubsampledControlVariate(deterministic_gradient!, residual_oracle,
    envelope::AbstractResidualEnvelope, anchor::AbstractVector, m::Integer;
    deterministic_hvp! = nothing, refresh_anchor! = nothing,
    subset_design::AbstractSubsamplingDesign=UniformSubsamplingDesign())
    N = n_observations(envelope)
    1 <= m <= N || throw(ArgumentError("minibatch size m must lie in 1:N"))
    if subset_design isa BalancedStratifiedSubsamplingDesign
        subset_design.n_strata * subset_design.stratum_size == N ||
            throw(DimensionMismatch("balanced subset design does not cover the envelope columns"))
        subset_design.n_strata * subset_design.per_stratum == m ||
            throw(DimensionMismatch("balanced subset design does not match minibatch size m"))
    end
    requested_anchor = collect(Float64, anchor)
    trajectory_anchor = _trajectory_scale_anchor(envelope)
    if trajectory_anchor !== nothing
        trajectory_anchor == requested_anchor || throw(ArgumentError(
            "TrajectoryResidualEnvelope and SubsampledControlVariate anchors must match"))
    end
    active_anchor = _subsampling_anchor_owner(envelope, requested_anchor)
    return SubsampledControlVariate(deterministic_gradient!, residual_oracle, envelope,
        active_anchor, deterministic_hvp!, refresh_anchor!, subset_design, Int(m),
        Vector{Int}(undef, m), zeros(Float64, length(active_anchor)), Dict{Int,Int}())
end

function _validate_subsampling_anchor(envelope::AbstractResidualEnvelope, anchor)
    trajectory_anchor = _trajectory_scale_anchor(envelope)
    trajectory_anchor === nothing || trajectory_anchor == anchor || throw(ArgumentError(
        "subsampling residual envelope does not match the active anchor"))
    return nothing
end

function _validated_refreshed_envelope(cv::SubsampledControlVariate,
        envelope::AbstractResidualEnvelope, requested)
    typeof(envelope) === typeof(cv.envelope) || throw(ArgumentError(
        "anchor-refresh provider must return the same concrete envelope type; " *
        "reconstruct the SubsampledControlVariate to change callback types"))
    _validate_subsampling_anchor(envelope, requested)
    return envelope
end
_validated_refreshed_envelope(cv::SubsampledControlVariate, value, requested) = throw(ArgumentError(
    "anchor-refresh provider must return an AbstractResidualEnvelope"))

function refresh_anchor!(cv::SubsampledControlVariate, anchor::AbstractVector)
    callback = cv.refresh_anchor_callback!
    callback === nothing && throw(ArgumentError(
        "this SubsampledControlVariate has no anchor-refresh provider"))
    length(anchor) == length(cv.anchor) || throw(DimensionMismatch(
        "new subsampling anchor has the wrong dimension"))
    requested = collect(Float64, anchor)
    new_envelope = _validated_refreshed_envelope(cv, callback(requested), requested)
    cv.envelope = new_envelope
    cv.anchor = _subsampling_anchor_owner(new_envelope, requested)
    return cv
end

function _reconstruct_subsampling(cv::SubsampledControlVariate;
        deterministic_gradient! = cv.deterministic_gradient!,
        residual_oracle = cv.residual_oracle,
        envelope = deepcopy(cv.envelope),
        deterministic_hvp! = cv.deterministic_hvp!,
        refresh_anchor_callback! = cv.refresh_anchor_callback!,
        subset_design = cv.subset_design)
    return SubsampledControlVariate(deterministic_gradient!, residual_oracle,
        envelope, copy(cv.anchor), cv.m;
        deterministic_hvp! = deterministic_hvp!,
        refresh_anchor! = refresh_anchor_callback!, subset_design)
end

function Base.copy(cv::SubsampledControlVariate)
    cv.refresh_anchor_callback! === nothing || throw(ArgumentError(
        "an anchor-managed SubsampledControlVariate cannot be copied safely; " *
        "construct one provider and model per chain"))
    return _reconstruct_subsampling(cv;
        deterministic_gradient! = _copy_callable(cv.deterministic_gradient!),
        residual_oracle = _copy_callable(cv.residual_oracle),
        deterministic_hvp! = _copy_callable(cv.deterministic_hvp!),
        refresh_anchor_callback! = _copy_callable(cv.refresh_anchor_callback!))
end

function component_scales!(out, envelope::AbstractResidualEnvelope, state, flow, t)
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

component_scales!(out, envelope::AbstractResidualEnvelope, state, t) =
    component_scales!(out, envelope, state, nothing, t)

function component_cell_scales!(out, envelope::AbstractResidualEnvelope,
        state, flow, left, right)
    callback = envelope.component_cell_scales!
    if callback === nothing
        envelope.component_scales! isa CertifiedAffineComponentScales ||
            throw(ArgumentError("endpoint cell bounds require certified affine component scales"))
        # The caller explicitly certified affine scales, whose maximum on a
        # closed cell is attained at an endpoint.
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

function total_residual_bound(envelope::AbstractResidualEnvelope, state, flow, t)
    scales = component_scales!(envelope.scales, envelope, state, flow, t)
    cumulative = 0.0
    @inbounds for component in eachindex(scales, envelope.totals)
        cumulative += scales[component] * envelope.totals[component]
        envelope.cumulative_masses[component] = cumulative
    end
    return cumulative
end

total_residual_bound(envelope::AbstractResidualEnvelope, state, t) =
    total_residual_bound(envelope, state, nothing, t)

n_observations(envelope::SeparableResidualEnvelope) = size(envelope.weights, 2)
n_observations(envelope::GroupedResidualEnvelope) = size(envelope.groups, 2)

function observation_residual_bound(envelope::SeparableResidualEnvelope,
        observation::Integer)
    result = 0.0
    @inbounds for component in axes(envelope.weights, 1)
        result += envelope.scales[component] *
            envelope.weights[component, observation]
    end
    return result
end

function observation_residual_bound(envelope::GroupedResidualEnvelope,
        observation::Integer)
    result = 0.0
    @inbounds for partition in axes(envelope.groups, 1)
        result += envelope.scales[envelope.groups[partition, observation]]
    end
    return result
end

"""Install a sampled residual in `gradient` and return its complete event rate."""
function subsampling_candidate_rate!(oracle, state, gradient, residual,
        scale, flow, deterministic_rate, subset)
    axpy!(scale, residual, gradient)
    return λ(state, gradient, flow)
end

"""Internal observation hook for proposal-weighted subsampling diagnostics."""
record_subsampling_proposal!(oracle, args...) = nothing

"""Internal hook for cheaply tightening a sampled subset's event-rate bound."""
subsampling_subset_bound(oracle, state, gradient, flow, D, M, subset, scale) =
    _signed_subset_bound(state, gradient, flow, D, M)

"""Internal pre-gradient hook for tightening a sampled residual bound."""
subsampling_residual_subset_bound(oracle, state, flow, D, M, subset, scale) = M

deterministic_gradient!(out, cv::SubsampledControlVariate, x) = cv.deterministic_gradient!(out, x)

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
subsampling_candidate_rate!(oracle::WithResidualStats, args...) =
    subsampling_candidate_rate!(oracle.f, args...)
record_subsampling_proposal!(oracle::WithResidualStats, args...) =
    record_subsampling_proposal!(oracle.f, args...)
subsampling_subset_bound(oracle::WithResidualStats, state, gradient, flow,
        D, M, subset, scale) = subsampling_subset_bound(
    oracle.f, state, gradient, flow, D, M, subset, scale)
subsampling_residual_subset_bound(oracle::WithResidualStats, state, flow,
        D, M, subset, scale) = subsampling_residual_subset_bound(
    oracle.f, state, flow, D, M, subset, scale)
function with_stats(cv::SubsampledControlVariate, stats::AbstractStatisticCounter)
    return _reconstruct_subsampling(cv;
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
function set_active_set!(strategy::SubsampledControlVariate, free::BitVector)
    set_active_set!(strategy.deterministic_gradient!, free)
    set_active_set!(strategy.residual_oracle, free)
    return nothing
end

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

function compute_gradient!(strategy::SubsampledControlVariate, x, out)
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

compute_gradient_for_reflection!(strategy::SubsampledControlVariate, x, out) =
    compute_gradient!(strategy, x, out)
