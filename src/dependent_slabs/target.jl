"""
    DependentSlabTarget(d, posterior_grad!, prior_grad!, slab_provider, model_prior_odds;
                        initial_free=trues(d))

Gradient target for dependent-slab sticky samplers. The returned callable writes
`posterior_grad - slab_prior_grad + active_slab_neggrad` into `out` and
synchronizes its active beta set through `set_active_set!`. The prior callback
must be the slab-prior negative gradient only, before active-set conditioning;
nuisance-prior terms must remain in `posterior_grad!`.
"""
mutable struct DependentSlabTarget{PG,RG,S<:AbstractSlabBoundary,O<:AbstractModelPriorOdds} <: Function
    d::Int
    posterior_grad!::PG
    prior_grad!::RG
    slab_provider::S
    model_prior_odds::O
    free::BitVector
    post_buf::Vector{Float64}
    prior_buf::Vector{Float64}
    slab_buf::Vector{Float64}
    active_beta::BitVector
end

function DependentSlabTarget(
    d::Integer,
    posterior_grad!,
    prior_grad!,
    slab_provider::AbstractSlabBoundary,
    model_prior_odds::AbstractModelPriorOdds;
    initial_free::BitVector=trues(Int(d)),
)
    d_int = Int(d)
    d_int > 0 || throw(ArgumentError("d must be positive"))
    length(initial_free) == d_int || throw(DimensionMismatch("initial_free length $(length(initial_free)) does not match dimension $d_int"))
    indices = beta_indices(slab_provider)
    all(i -> 1 <= i <= d_int, indices) || throw(ArgumentError("slab beta indices must be valid coordinates in 1:$d_int"))
    length(model_prior_odds) == length(indices) ||
        throw(DimensionMismatch("model-prior odds length $(length(model_prior_odds)) does not match beta dimension $(length(indices))"))
    return DependentSlabTarget(
        d_int,
        posterior_grad!,
        prior_grad!,
        slab_provider,
        model_prior_odds,
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
        _copy_callable(target.model_prior_odds);
        initial_free=copy(target.free),
    )
    return copied
end

function set_active_set!(target::DependentSlabTarget, free::BitVector)
    length(free) == target.d || throw(DimensionMismatch("active set length $(length(free)) does not match target dimension $(target.d)"))
    copyto!(target.free, free)
    set_active_set!(target.posterior_grad!, free)
    set_active_set!(target.prior_grad!, free)
    return nothing
end

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

function PDMPModel(target::DependentSlabTarget; hvp::Bool=false)
    hvp && throw(ArgumentError("DependentSlabTarget does not yet provide HVP/VHV; use hvp=false"))
    return PDMPModel(target.d, FullGradient(target), nothing)
end
