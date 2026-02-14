# Copy a callable: use copy for structs that define it, identity fallback for closures
_copy_callable(f) = _has_copy(f) ? copy(f) : f
_has_copy(f) = applicable(copy, f) && !(f isa Function)
_copy_callable(f::Base.Fix1) = Base.Fix1(_copy_callable(f.f), copy(f.x))
_copy_callable(f::Base.Fix2) = Base.Fix2(_copy_callable(f.f), copy(f.x))

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
end

# temporary backwards compatibility constructor for now
# SubsampledGradient(f::F1, resample_indices!::F2, nsub::Int) where {F1, F2} =
#     SubsampledGradient(f, resample_indices!, (trace) -> nothing, (args...) -> nothing, nsub, 0)
SubsampledGradient(f::Function, resample_indices!::Function, nsub::Int) =
    SubsampledGradient(f, resample_indices!, (trace) -> nothing, (args...) -> nothing, nsub, 0, false)

SubsampledGradient(f::Function, resample_indices!::Function, update_anchor!::Function, full::Function, nsub::Int, no_anchor_updates::Int, use_full_gradient_for_reflections::Bool) =
    SubsampledGradient(f, resample_indices!, update_anchor!, FullGradient(full), nsub, no_anchor_updates, use_full_gradient_for_reflections)

struct CoordinateWiseGradient{F} <: CoordinateWiseGradientStrategy
    f::F
end

Base.copy(g::CoordinateWiseGradient) = CoordinateWiseGradient(_copy_callable(g.f))

with_stats(grad::FullGradient,       stats::StatisticCounter) = FullGradient(with_stats(grad.f, stats))
function with_stats(grad::SubsampledGradient, stats::StatisticCounter)
    SubsampledGradient(with_stats(grad.f, stats), grad.resample_indices!, grad.update_anchor!, grad.full, grad.nsub, grad.no_anchor_updates, grad.use_full_gradient_for_reflections)
end
with_stats(grad::CoordinateWiseGradient, stats::StatisticCounter) = CoordinateWiseGradient(with_stats(grad.f, stats))

with_stats(f, stats::StatisticCounter) = WithStats(f, stats)

struct WithStats{F,S} <: Function
    f::F
    stats::S
end
(ws::WithStats)(args...) = (ws.stats.∇f_calls += 1; ws.f(args...))


# struct ControlVariateGradient{F} <: GradientStrategy
#     f::F
#     subsample_size::Int
#     reference_point::Vector{Float64}
#     cached_full_gradient::Vector{Float64}
#     control_frequency::Int
# end


# Gradient computation interface

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
