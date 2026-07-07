
if VERSION < v"1.13.0"
    ispositive(x::Real) = x > zero(x)
    isnegative(x::Real) = x < zero(x)
else
    # code for Julia ≥ 1.13
end

pos(x) = max(zero(x), x)

# TODO: remove these!
function idot(A, j, x)
    return dot((@view A[:, j]), x)
end

struct VHVProvider{G,V,W<:Union{Nothing,AbstractVector}}
    grad::G
    vhv::V
    w_buf::W
end
VHVProvider(grad, vhv) = VHVProvider(grad, vhv, nothing)

struct GradientProvider{V,F,G,C} <: Function
    θ::V
    flow::F
    gradient_strategy::G
    cache::C
end
(p::GradientProvider)(x::AbstractVector) =
    compute_gradient!(x, p.θ, p.gradient_strategy, p.flow, p.cache)

struct GradHVPProvider{G,H}
    grad::G
    hvp::H
end
_provider_grad((grad, hvp)::Tuple) = grad
_provider_hvp((grad, hvp)::Tuple) = hvp
_provider_grad(provider::GradHVPProvider) = provider.grad
_provider_hvp(provider::GradHVPProvider) = provider.hvp

struct FiniteDiffVHV{G}
    grad::G
    buf::Vector{Float64}
    grad_buf::Vector{Float64}
    w_buf::Vector{Float64}
end
FiniteDiffVHV(grad, buf::Vector{Float64}) =
    FiniteDiffVHV(grad, buf, similar(buf), similar(buf))
FiniteDiffVHV(grad, buf::Vector{Float64}, w_buf::Vector{Float64}) =
    FiniteDiffVHV(grad, buf, similar(buf), w_buf)
