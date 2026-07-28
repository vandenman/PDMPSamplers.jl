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

_sync_active_set!(state::AbstractPDMPState, gradient_strategy::GradientStrategy) = nothing
_sync_active_set!(state::StickyPDMPState, gradient_strategy::GradientStrategy) = set_active_set!(gradient_strategy, state.free)

set_active_set!(strategy::FullGradient, free::BitVector) = set_active_set!(strategy.f, free)
function set_active_set!(strategy::SubsampledGradient, free::BitVector)
    set_active_set!(strategy.f, free)
    set_active_set!(strategy.full, free)
    return nothing
end
set_active_set!(strategy::CoordinateWiseGradient, free::BitVector) = set_active_set!(strategy.f, free)

# Main entry point: compute gradient from state
function compute_gradient!(state::AbstractPDMPState, gradient_strategy::GradientStrategy, flow::ContinuousDynamics, cache)
    _sync_active_set!(state, gradient_strategy)
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
    _sync_active_set!(state, gradient_strategy)
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
