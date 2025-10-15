# fallback implementations
correct_gradient!(∇ϕ::AbstractVector, x::AbstractVector, ::AbstractVector, ::ContinuousDynamics, cache) = ∇ϕ

rand_refresh_time(flow::ContinuousDynamics) = ispositive(flow.λref) ? rand(Exponential(inv(flow.λref))) : oftype(flow.λref, Inf)

# Component-wise bounds not applicable
function ab_i(i::Int, ξ::SkeletonPoint, c::AbstractVector, flow::ContinuousDynamics, cache)
    error("Component-wise bounds not supported for type $(typeof(flow))")
end

λ(state::AbstractPDMPState, ∇ϕ::AbstractVector, flow::ContinuousDynamics) = λ(state.ξ, ∇ϕ, flow)