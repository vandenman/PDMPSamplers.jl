"""
    DependentSlabTarget(d, posterior_grad!, prior_grad!, slab_provider, model_prior;
                        initial_free=trues(d))

Gradient target for dependent-slab sticky samplers. The returned callable writes
`posterior_grad - slab_prior_grad + active_slab_neggrad` into `out` and
synchronizes its active beta set through `set_active_set!`. The prior callback
must be the slab-prior negative gradient only, before active-set conditioning;
nuisance-prior terms must remain in `posterior_grad!`.
"""
mutable struct DependentSlabTarget{PG,RG,S<:AbstractSlabPrior,O<:AbstractModelPrior} <: Function
    d::Int
    posterior_grad!::PG
    prior_grad!::RG
    slab_provider::S
    model_prior::O
    free::BitVector
    post_buf::Vector{Float64}
    prior_buf::Vector{Float64}
    slab_buf::Vector{Float64}
    active_beta::BitVector
end

# Internal finite-difference curvature hook used when dependent-slab targets need HVPs.
struct _DependentSlabFiniteDiffHVP{T<:DependentSlabTarget} <: Function
    target::T
    x_plus::Vector{Float64}
    x_minus::Vector{Float64}
    grad_plus::Vector{Float64}
    grad_minus::Vector{Float64}
    step::Float64
end

struct _DependentSlabFiniteDiffVHV{T<:DependentSlabTarget} <: Function
    hvp!::_DependentSlabFiniteDiffHVP{T}
    hv_buf::Vector{Float64}
end

function DependentSlabTarget(
    d::Integer,
    posterior_grad!,
    prior_grad!,
    slab_provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior;
    initial_free::BitVector=trues(Int(d)),
)
    d_int = Int(d)
    ispositive(d_int) || throw(ArgumentError("d must be positive"))
    length(initial_free) == d_int || throw(DimensionMismatch("initial_free length $(length(initial_free)) does not match dimension $d_int"))
    indices = beta_indices(slab_provider)
    all(i -> 1 <= i <= d_int, indices) || throw(ArgumentError("slab beta indices must be valid coordinates in 1:$d_int"))
    length(model_prior) == length(indices) ||
        throw(DimensionMismatch("model-prior length $(length(model_prior)) does not match beta dimension $(length(indices))"))
    return DependentSlabTarget(
        d_int,
        posterior_grad!,
        prior_grad!,
        slab_provider,
        model_prior,
        copy(initial_free),
        zeros(d_int),
        zeros(d_int),
        zeros(d_int),
        falses(length(indices)),
    )
end

function Base.copy(target::DependentSlabTarget)
    copied = DependentSlabTarget(
        target.d,
        _copy_callable(target.posterior_grad!),
        _copy_callable(target.prior_grad!),
        _copy_callable(target.slab_provider),
        _copy_callable(target.model_prior);
        initial_free=copy(target.free),
    )
    return copied
end

function Base.copy(h::_DependentSlabFiniteDiffHVP)
    return _DependentSlabFiniteDiffHVP(copy(h.target), similar(h.x_plus), similar(h.x_minus), similar(h.grad_plus), similar(h.grad_minus), h.step)
end

function Base.copy(v::_DependentSlabFiniteDiffVHV)
    return _DependentSlabFiniteDiffVHV(copy(v.hvp!), similar(v.hv_buf))
end

_copy_callable(h::_DependentSlabFiniteDiffHVP) = copy(h)
_copy_callable(v::_DependentSlabFiniteDiffVHV) = copy(v)

function set_active_set!(target::DependentSlabTarget, free::BitVector)
    length(free) == target.d || throw(DimensionMismatch("active set length $(length(free)) does not match target dimension $(target.d)"))
    copyto!(target.free, free)
    set_active_set!(target.posterior_grad!, free)
    set_active_set!(target.prior_grad!, free)
    return nothing
end

set_active_set!(h::_DependentSlabFiniteDiffHVP, free::BitVector) = set_active_set!(h.target, free)
set_active_set!(v::_DependentSlabFiniteDiffVHV, free::BitVector) = set_active_set!(v.hvp!, free)

function _evaluate_negative_gradient!(f, out::AbstractVector, x::AbstractVector)
    f(out, x)
    return out
end
function _evaluate_negative_gradient!(strategy::GradientStrategy, out::AbstractVector, x::AbstractVector)
    compute_gradient!(strategy, x, out)
    return out
end

function (target::DependentSlabTarget)(out::AbstractVector, x::AbstractVector)
    length(out) == target.d || throw(DimensionMismatch("out length $(length(out)) does not match target dimension $(target.d)"))
    length(x) == target.d || throw(DimensionMismatch("x length $(length(x)) does not match target dimension $(target.d)"))

    indices = beta_indices(target.slab_provider)
    @inbounds for k in eachindex(indices)
        target.active_beta[k] = target.free[indices[k]]
    end

    _evaluate_negative_gradient!(target.posterior_grad!, target.post_buf, x)
    _evaluate_negative_gradient!(target.prior_grad!, target.prior_buf, x)
    active_prior_grad!(target.slab_provider, target.slab_buf, x, target.active_beta)
    @. out = target.post_buf - target.prior_buf + target.slab_buf
    return out
end

function (h::_DependentSlabFiniteDiffHVP)(out::AbstractVector, x::AbstractVector, v::AbstractVector)
    length(out) == h.target.d || throw(DimensionMismatch("out length $(length(out)) does not match target dimension $(h.target.d)"))
    length(x) == h.target.d || throw(DimensionMismatch("x length $(length(x)) does not match target dimension $(h.target.d)"))
    length(v) == h.target.d || throw(DimensionMismatch("v length $(length(v)) does not match target dimension $(h.target.d)"))
    hnorm = max(norm(v), one(Float64))
    hstep = h.step * max(norm(x), one(Float64)) / hnorm
    @. h.x_plus = x + hstep * v
    @. h.x_minus = x - hstep * v
    h.target(h.grad_plus, h.x_plus)
    h.target(h.grad_minus, h.x_minus)
    @. out = (h.grad_plus - h.grad_minus) / (2hstep)
    return out
end

function (v::_DependentSlabFiniteDiffVHV)(x::AbstractVector, direction::AbstractVector, weight::AbstractVector)
    v.hvp!(v.hv_buf, x, direction)
    return dot(weight, v.hv_buf)
end

function _dependent_slab_hvp(target::DependentSlabTarget; step::Real=sqrt(eps(Float64)))
    d = target.d
    return _DependentSlabFiniteDiffHVP(copy(target), zeros(d), zeros(d), zeros(d), zeros(d), Float64(step))
end

function PDMPModel(target::DependentSlabTarget; hvp::Bool=false)
    if hvp
        hvp! = _dependent_slab_hvp(target)
        vhv = _DependentSlabFiniteDiffVHV(copy(hvp!), zeros(target.d))
        return PDMPModel(target.d, FullGradient(target), hvp!, vhv, true, true)
    end
    return PDMPModel(target.d, FullGradient(target), nothing)
end
