abstract type AbstractModelPriorOdds end

struct BernoulliModelPriorOdds{P<:AbstractVector{<:Real}} <: AbstractModelPriorOdds
    prob::P
    function BernoulliModelPriorOdds(prob::P) where {P<:AbstractVector{<:Real}}
        isempty(prob) && throw(ArgumentError("prob must be non-empty"))
        all(p -> zero(p) <= p <= one(p), prob) || throw(ArgumentError("Bernoulli probabilities must be in [0, 1]"))
        new{P}(prob)
    end
end

Base.length(prior::BernoulliModelPriorOdds) = length(prior.prob)
Base.copy(prior::BernoulliModelPriorOdds) = BernoulliModelPriorOdds(copy(prior.prob))

function log_model_add_odds(prior::BernoulliModelPriorOdds, active::BitVector, j::Integer)
    1 <= j <= length(prior.prob) || throw(BoundsError(prior.prob, j))
    length(active) == length(prior.prob) || throw(DimensionMismatch("active set length $(length(active)) does not match prior length $(length(prior.prob))"))
    p = prior.prob[j]
    iszero(p) && return -Inf
    isone(p) && return Inf
    return log(p) - log1p(-p)
end

struct BetaBernoulliModelPriorOdds <: AbstractModelPriorOdds
    n::Int
    a::Float64
    b::Float64
    function BetaBernoulliModelPriorOdds(n::Integer, a::Real, b::Real)
        n > 0 || throw(ArgumentError("n must be positive"))
        a > 0 || throw(ArgumentError("a must be positive"))
        b > 0 || throw(ArgumentError("b must be positive"))
        new(Int(n), Float64(a), Float64(b))
    end
end

Base.length(prior::BetaBernoulliModelPriorOdds) = prior.n

function log_model_add_odds(prior::BetaBernoulliModelPriorOdds, active::BitVector, j::Integer)
    1 <= j <= prior.n || throw(BoundsError(active, j))
    length(active) == prior.n || throw(DimensionMismatch("active set length $(length(active)) does not match prior length $(prior.n)"))
    active[j] && throw(ArgumentError("log_model_add_odds is defined for adding an inactive coordinate; coordinate $j is already active"))
    k = count(active)
    denom = prior.b + prior.n - k - 1
    denom <= 0 && return Inf
    return log(prior.a + k) - log(denom)
end

abstract type AbstractGaussianSlabProvider end

const _LOG2PI = log(2π)

struct DenseGaussianSlab <: AbstractGaussianSlabProvider
    mean::Vector{Float64}
    cov::Matrix{Float64}
    beta_indices::Vector{Int}
    function DenseGaussianSlab(mean::AbstractVector{<:Real}, cov::AbstractMatrix{<:Real}, beta_indices::AbstractVector{<:Integer}=eachindex(mean))
        m = length(mean)
        size(cov) == (m, m) || throw(DimensionMismatch("covariance has size $(size(cov)), expected ($m, $m)"))
        length(beta_indices) == m || throw(DimensionMismatch("beta_indices length $(length(beta_indices)) does not match mean length $m"))
        all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
        length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
        cov_f = Matrix{Float64}(cov)
        isapprox(cov_f, cov_f'; rtol=sqrt(eps(Float64)), atol=sqrt(eps(Float64))) || throw(ArgumentError("covariance must be symmetric"))
        try
            cholesky(Symmetric(cov_f); check=true)
        catch err
            err isa PosDefException || rethrow()
            throw(ArgumentError("covariance must be positive definite"))
        end
        new(Vector{Float64}(mean), cov_f, Vector{Int}(beta_indices))
    end
end

Base.copy(provider::DenseGaussianSlab) = DenseGaussianSlab(copy(provider.mean), copy(provider.cov), copy(provider.beta_indices))
beta_indices(provider::AbstractGaussianSlabProvider) = provider.beta_indices

function gaussian_slab!(provider::DenseGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    length(mean_out) == length(provider.mean) || throw(DimensionMismatch("mean_out has length $(length(mean_out)), expected $(length(provider.mean))"))
    size(cov_out) == size(provider.cov) || throw(DimensionMismatch("cov_out has size $(size(cov_out)), expected $(size(provider.cov))"))
    copyto!(mean_out, provider.mean)
    copyto!(cov_out, provider.cov)
    return nothing
end

struct CallbackGaussianSlab{I<:AbstractVector{Int},F,G} <: AbstractGaussianSlabProvider
    beta_indices::I
    mean_cov!::F
    active_prior_grad!::G
end

function CallbackGaussianSlab(beta_indices::AbstractVector{<:Integer}; mean_cov!, active_prior_grad! = nothing)
    isempty(beta_indices) && throw(ArgumentError("beta_indices must be non-empty"))
    all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
    length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
    return CallbackGaussianSlab(Vector{Int}(beta_indices), mean_cov!, active_prior_grad!)
end

Base.copy(provider::CallbackGaussianSlab) =
    CallbackGaussianSlab(copy(provider.beta_indices); mean_cov! = provider.mean_cov!, active_prior_grad! = provider.active_prior_grad!)

function gaussian_slab!(provider::CallbackGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    provider.mean_cov!(mean_out, cov_out, x)
    return nothing
end

function _active_positions(active_beta::BitVector, m::Integer)
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    return findall(active_beta)
end

function _gaussian_logdensity(values::AbstractVector, mean::AbstractVector, cov::AbstractMatrix)
    k = length(values)
    k == length(mean) || throw(DimensionMismatch("values and mean lengths differ"))
    size(cov) == (k, k) || throw(DimensionMismatch("covariance has size $(size(cov)), expected ($k, $k)"))
    iszero(k) && return 0.0
    F = cholesky(Symmetric(Matrix{Float64}(cov)); check=true)
    delta = Vector{Float64}(values .- mean)
    solved = F \ delta
    logdet_cov = 2sum(log, diag(F.U))
    return -0.5 * (k * _LOG2PI + logdet_cov + dot(delta, solved))
end

function _current_mean_cov(provider::AbstractGaussianSlabProvider, x::AbstractVector)
    m = length(beta_indices(provider))
    mean = Vector{Float64}(undef, m)
    cov = Matrix{Float64}(undef, m, m)
    gaussian_slab!(provider, mean, cov, x)
    return mean, cov
end

function active_logdensity(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    mean, cov = _current_mean_cov(provider, x)
    A = _active_positions(active_beta, length(indices))
    return _gaussian_logdensity(x[indices[A]], mean[A], cov[A, A])
end

function _conditional_logdensity_zero(mean::AbstractVector, cov::AbstractMatrix, beta_values::AbstractVector, active_beta::BitVector, j_beta::Integer)
    m = length(mean)
    1 <= j_beta <= m || throw(BoundsError(mean, j_beta))
    length(beta_values) == m || throw(DimensionMismatch("beta_values length $(length(beta_values)) does not match beta dimension $m"))
    active_beta[j_beta] && throw(ArgumentError("conditional boundary density is defined for inactive coordinates; beta coordinate $j_beta is active"))

    A = _active_positions(active_beta, m)
    μj = mean[j_beta]
    σjj = cov[j_beta, j_beta]
    if isempty(A)
        var = σjj
        var > 0 || throw(ArgumentError("conditional variance must be positive, got $var"))
        return -0.5 * (_LOG2PI + log(var) + (0.0 - μj)^2 / var)
    end

    cov_AA = Matrix{Float64}(cov[A, A])
    F = cholesky(Symmetric(cov_AA); check=true)
    delta_A = Vector{Float64}(beta_values[A] .- mean[A])
    cov_jA = Vector{Float64}(cov[j_beta, A])
    solved_delta = F \ delta_A
    solved_cross = F \ cov_jA
    cond_mean = μj + dot(cov_jA, solved_delta)
    cond_var = σjj - dot(cov_jA, solved_cross)
    cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
    return -0.5 * (_LOG2PI + log(cond_var) + cond_mean^2 / cond_var)
end

function conditional_logdensity_zero(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    indices = beta_indices(provider)
    mean, cov = _current_mean_cov(provider, x)
    return _conditional_logdensity_zero(mean, cov, x[indices], active_beta, j_beta)
end

conditional_density_zero(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector, j_beta::Integer) =
    exp(conditional_logdensity_zero(provider, x, active_beta, j_beta))

function _write_active_gradient!(out::AbstractVector, full_indices::AbstractVector{Int}, active_positions::AbstractVector{Int}, grad_active::AbstractVector)
    fill!(out, 0.0)
    if length(out) == length(full_indices)
        @inbounds for (pos, value) in zip(active_positions, grad_active)
            out[pos] = value
        end
    elseif maximum(full_indices) <= length(out)
        @inbounds for (pos, value) in zip(active_positions, grad_active)
            out[full_indices[pos]] = value
        end
    else
        throw(DimensionMismatch("out must have length $(length(full_indices)) for beta-block gradients or at least $(maximum(full_indices)) for full-state gradients"))
    end
    return out
end

function active_prior_grad!(provider::DenseGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    A = _active_positions(active_beta, length(indices))
    if isempty(A)
        fill!(out, 0.0)
        return out
    end
    F = cholesky(Symmetric(Matrix{Float64}(provider.cov[A, A])); check=true)
    delta = Vector{Float64}(x[indices[A]] .- provider.mean[A])
    grad_active = F \ delta
    return _write_active_gradient!(out, indices, A, grad_active)
end

function active_prior_grad!(provider::CallbackGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    provider.active_prior_grad! === nothing &&
        throw(ArgumentError("CallbackGaussianSlab requires active_prior_grad! for active prior gradients"))
    provider.active_prior_grad!(out, x, active_beta)
    return out
end

mutable struct DependentSlabTarget{PG,RG,S<:AbstractGaussianSlabProvider,O<:AbstractModelPriorOdds} <: Function
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
    slab_provider::AbstractGaussianSlabProvider,
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
