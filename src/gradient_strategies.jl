# Concrete gradient strategies
struct FullGradient{F} <: GlobalGradientStrategy
    f::F
end


struct SubsampledGradient{F1, F2, F3} <: GlobalGradientStrategy
    f::F1
    resample_indices!::F2
    update_anchor!::F3
    nsub::Int
    no_anchor_updates::Int
end

# temporary backwards compatibility constructor for now
SubsampledGradient(f::F1, resample_indices!::F2, nsub::Int) where {F1, F2} =
    SubsampledGradient(f, resample_indices!, (trace) -> nothing, nsub, 0)

struct CoordinateWiseGradient{F} <: CoordinateWiseGradientStrategy
    f::F
end
with_stats(grad::FullGradient,       stats::StatisticCounter) = FullGradient(with_stats(grad.f, stats))
with_stats(grad::SubsampledGradient, stats::StatisticCounter) = SubsampledGradient(with_stats(grad.f, stats), grad.resample_indices!, grad.update_anchor!, grad.nsub, grad.no_anchor_updates)
with_stats(grad::CoordinateWiseGradient, stats::StatisticCounter) = CoordinateWiseGradient(with_stats(grad.f, stats))

function with_stats(f::Function, stats::StatisticCounter)
    return (args...) -> begin
        stats.∇f_calls += 1
        f(args...)
    end
end


# struct ControlVariateGradient{F} <: GradientStrategy
#     f::F
#     subsample_size::Int
#     reference_point::Vector{Float64}
#     cached_full_gradient::Vector{Float64}
#     control_frequency::Int
# end


# Gradient computation interface
function compute_gradient!(state::AbstractPDMPState, gradient_strategy::GradientStrategy, flow::ContinuousDynamics, cache)
    return compute_gradient!(state.ξ.x, state.ξ.θ, gradient_strategy, flow, cache)
end

function compute_gradient_uncorrected!(state::AbstractPDMPState, gradient_strategy::GradientStrategy, flow::ContinuousDynamics, cache)
    return compute_gradient_uncorrected!(state.ξ.x, state.ξ.θ, gradient_strategy, flow, cache)
end

# version with vectors for AD
function compute_gradient!(x::AbstractVector, θ::AbstractVector, gradient_strategy::GradientStrategy, flow::ContinuousDynamics, cache)
    ∇ϕx = compute_gradient!(gradient_strategy, x, cache.∇ϕx)
    correct_gradient!(∇ϕx, x, θ, flow, cache)
    return ∇ϕx
end

function compute_gradient_uncorrected!(x::AbstractVector, ::AbstractVector, gradient_strategy::GradientStrategy, ::ContinuousDynamics, cache)
    ∇ϕx = compute_gradient!(gradient_strategy, x, cache.∇ϕx)
    return ∇ϕx
end

function compute_gradient!(strategy::GradientStrategy, ∇ϕx, x, gradient_func)
    error("compute_gradient! not implemented for $(typeof(strategy))")
end

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
