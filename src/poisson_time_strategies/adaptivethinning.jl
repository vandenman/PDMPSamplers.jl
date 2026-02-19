struct AdaptiveThinningStrategy{T<:BoundStrategy} <: PoissonTimeStrategy
    acceptance_target::Float64
    c::T
end
AdaptiveThinningStrategy(c::BoundStrategy) = AdaptiveThinningStrategy(0.65, c)

struct AdaptiveBounds <: BoundStrategy
    c::Vector{Float64}
    adaptation_rate::Float64
    target_acceptance::Float64
    min_c::Float64
    max_c::Float64
end

AdaptiveBounds(c::Vector{Float64}; adaptation_rate=0.1, target_acceptance=0.3, min_c=0.1, max_c=1000.0) =
    AdaptiveBounds(copy(c), adaptation_rate, target_acceptance, min_c, max_c)
