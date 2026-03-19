# fallback implementations
correct_gradient!(∇ϕ::AbstractVector, x::AbstractVector, ::AbstractVector, ::ContinuousDynamics, cache) = ∇ϕ

rand_refresh_time(flow::ContinuousDynamics) = ispositive(refresh_rate(flow)) ? rand(Exponential(inv(refresh_rate(flow)))) : oftype(refresh_rate(flow), Inf)

# Component-wise bounds not applicable
function ab_i(i::Integer, ξ::SkeletonPoint, c::AbstractVector, flow::ContinuousDynamics, cache)
    error("Component-wise bounds not supported for type $(typeof(flow))")
end

λ(state::AbstractPDMPState, ∇ϕ::AbstractVector, flow::ContinuousDynamics) = λ(state.ξ, ∇ϕ, flow)

extract_vhv(vt::AbstractVector, Hxt_vt::AbstractVector) = dot(vt, Hxt_vt)
extract_vhv(::AbstractVector, vhv::Real) = vhv

function ∂λ∂t(state::AbstractPDMPState, ::AbstractVector, curvature_input, ::ContinuousDynamics)
    return extract_vhv(state.ξ.θ, curvature_input)
end