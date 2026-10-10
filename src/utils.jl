
if VERSION < v"1.13.0"
    ispositive(x::Real) = x > zero(x)
    isnegative(x::Real) = x < zero(x)
else
    # code for Julia ≥ 1.13
end

pos(x) = max(zero(x), x)

@inline _grid_cell_count(t_grid, N, max_time) = isfinite(max_time) ?
    max(0, min(N, searchsortedfirst(t_grid, max_time) - 1)) : N

@inline function posdot(x, y)
    value = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for i in eachindex(x, y)
        value += pos(x[i] * y[i])
    end
    return value
end

function _rand_posdot_index(rng::Random.AbstractRNG, x, y)
    total = posdot(x, y)
    ispositive(total) || return rand(rng, eachindex(x))
    threshold = rand(rng) * total
    cumulative = zero(total)
    selected = firstindex(x)
    @inbounds for i in eachindex(x, y)
        cumulative += pos(x[i] * y[i])
        selected = i
        cumulative >= threshold && break
    end
    return selected
end

# TODO: remove these!
function idot(A, j, x)
    return dot((@view A[:, j]), x)
end

abstract type AbstractRateDerivativeProvider end

struct VHVProvider{G,V,W<:Union{Nothing,AbstractVector}} <: AbstractRateDerivativeProvider
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

struct GradHVPProvider{G,H} <: AbstractRateDerivativeProvider
    grad::G
    hvp::H
end

struct GradientOnlyProvider{G} <: AbstractRateDerivativeProvider
    grad::G
end

_provider_grad(provider::GradHVPProvider) = provider.grad
_provider_hvp(provider::GradHVPProvider) = provider.hvp
_provider_grad(provider::GradientOnlyProvider) = provider.grad

struct FiniteDiffVHV{G,S} <: AbstractRateDerivativeProvider
    grad::G
    buf::Vector{Float64}
    grad_buf::Vector{Float64}
    w_buf::Vector{Float64}
    stats::S
end
FiniteDiffVHV(grad, buf::Vector{Float64}) =
    FiniteDiffVHV(grad, buf, similar(buf), similar(buf))
FiniteDiffVHV(grad, buf::Vector{Float64}, w_buf::Vector{Float64}) =
    FiniteDiffVHV(grad, buf, similar(buf), w_buf)
FiniteDiffVHV(grad, buf::Vector{Float64}, grad_buf::Vector{Float64}, w_buf::Vector{Float64}) =
    FiniteDiffVHV(grad, buf, grad_buf, w_buf, nothing)
