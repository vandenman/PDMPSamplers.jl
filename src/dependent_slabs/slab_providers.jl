"""
    AbstractGaussianSlabProvider
    AbstractExchangeableGaussianSlab

Gaussian slab providers define a beta-block Gaussian slab through
`gaussian_slab!`. Exchangeable subtypes use the covariance form
`u * I + v * ones(p, p)`.
"""
abstract type AbstractGaussianSlabProvider <: AbstractSlabPrior end

"""
    AbstractExchangeableGaussianSlab

Gaussian slab provider with exchangeable covariance structure
`u * I + v * ones(p, p)`.
"""
abstract type AbstractExchangeableGaussianSlab <: AbstractGaussianSlabProvider end

mutable struct DenseGaussianSlabCache
    work_cov::Matrix{Float64}
    work_rhs::Vector{Float64}
    work_cross::Vector{Float64}
end

DenseGaussianSlabCache(m::Integer) = DenseGaussianSlabCache(Matrix{Float64}(undef, m, m), Vector{Float64}(undef, m), Vector{Float64}(undef, m))

"""
    DenseGaussianSlab(mean, cov, beta_indices=eachindex(mean))

Fixed multivariate Gaussian slab for the beta coordinates listed in
`beta_indices`. `mean` and `cov` are in beta-block order. This provider uses a
fixed-covariance cache and supports allocation-free active prior gradients for
dense slabs.
"""
struct DenseGaussianSlab <: AbstractGaussianSlabProvider
    mean::Vector{Float64}
    cov::Matrix{Float64}
    beta_indices::Vector{Int}
    cache::DenseGaussianSlabCache
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
        new(Vector{Float64}(mean), cov_f, Vector{Int}(beta_indices), DenseGaussianSlabCache(m))
    end
end

"""
    ExchangeableGaussianSlab(beta_indices, mean, u, v)

Gaussian slab with common mean and covariance `u * I + v * ones(p, p)` over the
listed beta coordinates.
"""
struct ExchangeableGaussianSlab <: AbstractExchangeableGaussianSlab
    beta_indices::Vector{Int}
    mean::Float64
    u::Float64
    v::Float64
    function ExchangeableGaussianSlab(beta_indices::AbstractVector{<:Integer}, mean::Real, u::Real, v::Real)
        isempty(beta_indices) && throw(ArgumentError("beta_indices must be non-empty"))
        all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
        length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
        p = length(beta_indices)
        u_f = Float64(u)
        v_f = Float64(v)
        u_f > 0 || throw(ArgumentError("u must be positive"))
        u_f + p * v_f > 0 || throw(ArgumentError("u + p*v must be positive"))
        new(Vector{Int}(beta_indices), Float64(mean), u_f, v_f)
    end
end

"""
    ZeroMeanExchangeableGaussianSlab(beta_indices, u, v)

Zero-mean exchangeable Gaussian slab with covariance `u * I + v * ones(p, p)`.
This has optimized boundary-density formulas for exchangeable active faces.
"""
struct ZeroMeanExchangeableGaussianSlab <: AbstractExchangeableGaussianSlab
    beta_indices::Vector{Int}
    u::Float64
    v::Float64
    function ZeroMeanExchangeableGaussianSlab(beta_indices::AbstractVector{<:Integer}, u::Real, v::Real)
        isempty(beta_indices) && throw(ArgumentError("beta_indices must be non-empty"))
        all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
        length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
        p = length(beta_indices)
        u_f = Float64(u)
        v_f = Float64(v)
        u_f > 0 || throw(ArgumentError("u must be positive"))
        u_f + p * v_f > 0 || throw(ArgumentError("u + p*v must be positive"))
        new(Vector{Int}(beta_indices), u_f, v_f)
    end
end

"""
    IndependentZeroMeanGaussianSlab(kappa, beta_indices=eachindex(kappa))

Independent zero-mean fixed Gaussian slab parameterized by its boundary
density at zero, `kappa[j] = q_j(0)`. This is the aggregate-clock equivalent of
the R `independent_slab_density()` specification.
"""
struct IndependentZeroMeanGaussianSlab <: AbstractGaussianSlabProvider
    beta_indices::Vector{Int}
    kappa::Vector{Float64}
    log_q_zero::Vector{Float64}
    precision::Vector{Float64}
    function IndependentZeroMeanGaussianSlab(kappa::AbstractVector{<:Real}, beta_indices::AbstractVector{<:Integer}=eachindex(kappa))
        isempty(kappa) && throw(ArgumentError("kappa must be non-empty"))
        length(beta_indices) == length(kappa) || throw(DimensionMismatch("beta_indices length $(length(beta_indices)) does not match kappa length $(length(kappa))"))
        all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
        length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
        κ = Vector{Float64}(kappa)
        all(>(0), κ) || throw(ArgumentError("kappa entries must be positive"))
        new(Vector{Int}(beta_indices), κ, log.(κ), @. 2π * κ^2)
    end
end

"""
    IndependentZeroMeanLogscaleGaussianSlab(beta_indices, logscale_indices, log_base_scales)

Independent zero-mean Gaussian slab with
`log(s_j(x)) = log_base_scales[j] + x[logscale_indices[j]]`. Repeated
`logscale_indices` represent shared scale coordinates.
"""
struct IndependentZeroMeanLogscaleGaussianSlab <: AbstractGaussianSlabProvider
    beta_indices::Vector{Int}
    logscale_indices::Vector{Int}
    log_base_scales::Vector{Float64}
    function IndependentZeroMeanLogscaleGaussianSlab(
        beta_indices::AbstractVector{<:Integer},
        logscale_indices::AbstractVector{<:Integer},
        log_base_scales::AbstractVector{<:Real},
    )
        isempty(beta_indices) && throw(ArgumentError("beta_indices must be non-empty"))
        length(logscale_indices) == length(beta_indices) ||
            throw(DimensionMismatch("logscale_indices length $(length(logscale_indices)) does not match beta dimension $(length(beta_indices))"))
        length(log_base_scales) == length(beta_indices) ||
            throw(DimensionMismatch("log_base_scales length $(length(log_base_scales)) does not match beta dimension $(length(beta_indices))"))
        all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
        all(>(0), logscale_indices) || throw(ArgumentError("logscale_indices must be positive"))
        length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
        isempty(intersect(Int.(beta_indices), Int.(logscale_indices))) ||
            throw(ArgumentError("logscale_indices must be disjoint from beta_indices"))
        new(Vector{Int}(beta_indices), Vector{Int}(logscale_indices), Vector{Float64}(log_base_scales))
    end
end

"""
    GlobalLogscaleExchangeableGaussianSlab(beta_indices, logscale_index, u0, v0; mean=0, logscale_offset=0)

Exchangeable Gaussian slab with covariance
`exp(2 * (logscale_offset + x[logscale_index])) * (u0 I + v0 11')`.
This provider has structured scalar residual-clock support for linear flows.
"""
struct GlobalLogscaleExchangeableGaussianSlab <: AbstractExchangeableGaussianSlab
    beta_indices::Vector{Int}
    logscale_index::Int
    mean::Float64
    u::Float64
    v::Float64
    logscale_offset::Float64
    function GlobalLogscaleExchangeableGaussianSlab(
        beta_indices::AbstractVector{<:Integer},
        logscale_index::Integer,
        u0::Real,
        v0::Real;
        mean::Real=0.0,
        logscale_offset::Real=0.0,
    )
        isempty(beta_indices) && throw(ArgumentError("beta_indices must be non-empty"))
        all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
        length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
        logscale_index > 0 || throw(ArgumentError("logscale_index must be positive"))
        Int(logscale_index) in Int.(beta_indices) &&
            throw(ArgumentError("logscale_index must be disjoint from beta_indices"))
        p = length(beta_indices)
        u_f = Float64(u0)
        v_f = Float64(v0)
        u_f > 0 || throw(ArgumentError("u0 must be positive"))
        u_f + p * v_f > 0 || throw(ArgumentError("u0 + p*v0 must be positive"))
        new(Vector{Int}(beta_indices), Int(logscale_index), Float64(mean), u_f, v_f, Float64(logscale_offset))
    end
end

Base.copy(provider::DenseGaussianSlab) = DenseGaussianSlab(copy(provider.mean), copy(provider.cov), copy(provider.beta_indices))
Base.copy(provider::ExchangeableGaussianSlab) = ExchangeableGaussianSlab(copy(provider.beta_indices), provider.mean, provider.u, provider.v)
Base.copy(provider::ZeroMeanExchangeableGaussianSlab) = ZeroMeanExchangeableGaussianSlab(copy(provider.beta_indices), provider.u, provider.v)
Base.copy(provider::IndependentZeroMeanGaussianSlab) = IndependentZeroMeanGaussianSlab(copy(provider.kappa), copy(provider.beta_indices))
Base.copy(provider::IndependentZeroMeanLogscaleGaussianSlab) =
    IndependentZeroMeanLogscaleGaussianSlab(copy(provider.beta_indices), copy(provider.logscale_indices), copy(provider.log_base_scales))
Base.copy(provider::GlobalLogscaleExchangeableGaussianSlab) =
    GlobalLogscaleExchangeableGaussianSlab(copy(provider.beta_indices), provider.logscale_index, provider.u, provider.v; mean=provider.mean, logscale_offset=provider.logscale_offset)

"""
    beta_indices(provider)

Return the full-state coordinate indices controlled by a slab boundary provider.
These indices define the beta-block order used by active sets and model priors.
"""
beta_indices(provider::AbstractGaussianSlabProvider) = provider.beta_indices

"""
    slab_cache_style(provider)

Return a cache trait for `provider`. Fixed Gaussian providers return
`FixedCovarianceCache`; state-dependent and arbitrary callback boundaries return
`NoSlabCache` by default.
"""
slab_cache_style(::AbstractSlabPrior) = NoSlabCache()

"""
    slab_cache_key(provider, x, active_beta)

Return a cache key for active-face quantities, or `nothing` when the provider
does not support active-set-only caching.
"""
slab_cache_key(::AbstractSlabPrior, ::AbstractVector, ::BitVector) = nothing
slab_cache_style(::DenseGaussianSlab) = FixedCovarianceCache()
slab_cache_style(::AbstractExchangeableGaussianSlab) = FixedCovarianceCache()
slab_cache_style(::IndependentZeroMeanGaussianSlab) = FixedCovarianceCache()
slab_cache_style(::IndependentZeroMeanLogscaleGaussianSlab) = NoSlabCache()
slab_cache_style(::GlobalLogscaleExchangeableGaussianSlab) = NoSlabCache()
slab_cache_key(::DenseGaussianSlab, ::AbstractVector, active_beta::BitVector) = copy(active_beta)
slab_cache_key(::AbstractExchangeableGaussianSlab, ::AbstractVector, active_beta::BitVector) = copy(active_beta)
slab_cache_key(::IndependentZeroMeanGaussianSlab, ::AbstractVector, active_beta::BitVector) = copy(active_beta)
slab_cache_key(::IndependentZeroMeanLogscaleGaussianSlab, ::AbstractVector, ::BitVector) = nothing
slab_cache_key(::GlobalLogscaleExchangeableGaussianSlab, ::AbstractVector, ::BitVector) = nothing

"""
    scalar_logscale_gaussian_line_segment(provider, model_prior, flow, state, can_stick, horizon)

Return scalar segment parameters for structured globally scaled exchangeable
Gaussian slabs. Methods are flow-specific and currently implemented with the
aggregate sticky linear-flow code.
"""
function scalar_logscale_gaussian_line_segment end

"""
    gaussian_slab!(provider, mean_out, cov_out, x)

Write the Gaussian slab mean and covariance, in beta-block order, into
`mean_out` and `cov_out` at state `x`. Fixed providers ignore `x`; callback
providers may compute state-dependent hyperparameters.
"""
function gaussian_slab!(provider::DenseGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    copyto!(mean_out, provider.mean)
    copyto!(cov_out, provider.cov)
    return nothing
end

function gaussian_slab!(provider::ExchangeableGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    fill!(mean_out, provider.mean)
    fill!(cov_out, provider.v)
    @inbounds for i in 1:m
        cov_out[i, i] = provider.u + provider.v
    end
    return nothing
end

function gaussian_slab!(provider::ZeroMeanExchangeableGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    fill!(mean_out, 0.0)
    fill!(cov_out, provider.v)
    @inbounds for i in 1:m
        cov_out[i, i] = provider.u + provider.v
    end
    return nothing
end

function gaussian_slab!(provider::IndependentZeroMeanGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    fill!(mean_out, 0.0)
    fill!(cov_out, 0.0)
    @inbounds for j in 1:m
        cov_out[j, j] = inv(provider.precision[j])
    end
    return nothing
end

function gaussian_slab!(provider::IndependentZeroMeanLogscaleGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    fill!(mean_out, 0.0)
    fill!(cov_out, 0.0)
    @inbounds for j in 1:m
        log_s = provider.log_base_scales[j] + x[provider.logscale_indices[j]]
        cov_out[j, j] = exp(2log_s)
    end
    return nothing
end

function gaussian_slab!(provider::GlobalLogscaleExchangeableGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    scale2 = exp(2 * (provider.logscale_offset + x[provider.logscale_index]))
    fill!(mean_out, provider.mean)
    fill!(cov_out, scale2 * provider.v)
    @inbounds for i in 1:m
        cov_out[i, i] = scale2 * (provider.u + provider.v)
    end
    return nothing
end

"""
    CallbackGaussianSlab(beta_indices; mean_cov!, active_prior_grad!=nothing)

Gaussian slab whose mean/covariance are supplied by `mean_cov!(mean, cov, x)`.
If the provider is used inside `DependentSlabTarget`, `active_prior_grad!` must
write the negative-gradient contribution for the active slab prior.
"""
struct CallbackGaussianSlab{I<:AbstractVector{Int},F,G} <: AbstractGaussianSlabProvider
    beta_indices::I
    mean_cov!::F
    active_prior_grad!::G
end

"""
    ArbitrarySlabBoundary(beta_indices; active_prior_neggrad!, log_q_zero!)

General active-face boundary provider. `active_prior_neggrad!(out, x,
active_beta)` writes the negative-gradient contribution for the active slab
prior. `log_q_zero!(x, active_beta, j)` returns the log boundary density for
adding inactive beta coordinate `j`.
"""
struct ArbitrarySlabBoundary{I<:AbstractVector{Int},G,Q} <: AbstractSlabPrior
    beta_indices::I
    active_prior_neggrad!::G
    log_q_zero!::Q
end

function ArbitrarySlabBoundary(beta_indices::AbstractVector{<:Integer}; active_prior_neggrad!, log_q_zero!)
    isempty(beta_indices) && throw(ArgumentError("beta_indices must be non-empty"))
    all(>(0), beta_indices) || throw(ArgumentError("beta_indices must be positive"))
    length(unique(beta_indices)) == length(beta_indices) || throw(ArgumentError("beta_indices must be unique"))
    return ArbitrarySlabBoundary(Vector{Int}(beta_indices), active_prior_neggrad!, log_q_zero!)
end

Base.copy(provider::ArbitrarySlabBoundary) =
    ArbitrarySlabBoundary(copy(provider.beta_indices); active_prior_neggrad! = provider.active_prior_neggrad!, log_q_zero! = provider.log_q_zero!)

beta_indices(provider::ArbitrarySlabBoundary) = provider.beta_indices

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

"""
    gaussian_slab(provider, x) -> mean, cov

Allocate and return the Gaussian slab mean and covariance for `provider` at
state `x`. Use `gaussian_slab!` when the caller owns output buffers.
"""
function gaussian_slab(provider::AbstractGaussianSlabProvider, x::AbstractVector)
    m = length(beta_indices(provider))
    mean = Vector{Float64}(undef, m)
    cov = Matrix{Float64}(undef, m, m)
    gaussian_slab!(provider, mean, cov, x)
    return mean, cov
end

function _active_positions(active_beta::BitVector, m::Integer)
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
    return -0.5 * (k * log2π + logdet_cov + dot(delta, solved))
end

function _current_mean_cov(provider::AbstractGaussianSlabProvider, x::AbstractVector)
    return gaussian_slab(provider, x)
end

"""
    active_logdensity(provider, x, active_beta)

Return the log density of the active beta coordinates under the Gaussian slab
induced by `provider` at state `x`.
"""
function active_logdensity(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    mean, cov = _current_mean_cov(provider, x)
    A = _active_positions(active_beta, length(indices))
    return _gaussian_logdensity(x[indices[A]], mean[A], cov[A, A])
end

function _conditional_logdensity_zero(mean::AbstractVector, cov::AbstractMatrix, beta_values::AbstractVector, active_beta::BitVector, j_beta::Integer)
    m = length(mean)
    A = _active_positions(active_beta, m)
    μj = mean[j_beta]
    σjj = cov[j_beta, j_beta]
    if isempty(A)
        var = σjj
        var > 0 || throw(ArgumentError("conditional variance must be positive, got $var"))
        return -0.5 * (log2π + log(var) + (0.0 - μj)^2 / var)
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
    return -0.5 * (log2π + log(cond_var) + cond_mean^2 / cond_var)
end

function _active_count(active_beta::BitVector)
    return count(active_beta)
end

function _write_active_gradient!(out::AbstractVector, full_indices::AbstractVector{Int}, active_beta::BitVector, grad_active::AbstractVector, k::Integer=_active_count(active_beta))
    fill!(out, 0.0)
    if length(out) == length(full_indices)
        a = 0
        @inbounds for j in eachindex(active_beta)
            if active_beta[j]
                a += 1
                out[j] = grad_active[a]
                a == k && break
            end
        end
    elseif maximum(full_indices) <= length(out)
        a = 0
        @inbounds for j in eachindex(active_beta)
            if active_beta[j]
                a += 1
                out[full_indices[j]] = grad_active[a]
                a == k && break
            end
        end
    else
        throw(DimensionMismatch("out must have length $(length(full_indices)) for beta-block gradients or at least $(maximum(full_indices)) for full-state gradients"))
    end
    return out
end

function _fill_dense_active_system!(work_cov::AbstractMatrix{Float64}, work_rhs::AbstractVector{Float64}, provider::DenseGaussianSlab, x::AbstractVector, active_beta::BitVector, k::Integer)
    a = 0
    indices = provider.beta_indices
    @inbounds for ia in eachindex(active_beta)
        if active_beta[ia]
            a += 1
            work_rhs[a] = x[indices[ia]] - provider.mean[ia]
            b = 0
            for ib in eachindex(active_beta)
                if active_beta[ib]
                    b += 1
                    work_cov[a, b] = provider.cov[ia, ib]
                    b == k && break
                end
            end
            a == k && break
        end
    end
    return nothing
end

function _conditional_logdensity_zero(provider::DenseGaussianSlab, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    μj = provider.mean[j_beta]
    σjj = provider.cov[j_beta, j_beta]
    k = _active_count(active_beta)
    if iszero(k)
        return -0.5 * (log2π + log(σjj) + μj^2 / σjj)
    end

    cache = provider.cache
    work_cov = cache.work_cov
    work_rhs = cache.work_rhs
    work_cross = cache.work_cross
    _fill_dense_active_system!(work_cov, work_rhs, provider, x, active_beta, k)
    _cholesky_factor_prefix!(work_cov, k)
    _cholesky_solve_with_factor_prefix!(work_cov, work_rhs, k)

    a = 0
    cond_shift = 0.0
    @inbounds for ia in eachindex(active_beta)
        if active_beta[ia]
            a += 1
            cov_jia = provider.cov[j_beta, ia]
            work_cross[a] = cov_jia
            cond_shift += cov_jia * work_rhs[a]
            a == k && break
        end
    end
    _cholesky_solve_with_factor_prefix!(work_cov, work_cross, k)

    a = 0
    var_shift = 0.0
    @inbounds for ia in eachindex(active_beta)
        if active_beta[ia]
            a += 1
            var_shift += provider.cov[j_beta, ia] * work_cross[a]
            a == k && break
        end
    end
    cond_mean = μj + cond_shift
    cond_var = σjj - var_shift
    return -0.5 * (log2π + log(cond_var) + cond_mean^2 / cond_var)
end

"""
    conditional_logdensity_zero(provider, x, active_beta, j_beta)

Return the Gaussian conditional log density of inactive beta coordinate
`j_beta` at zero, conditioning on the currently active beta coordinates.
"""
function conditional_logdensity_zero(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    indices = beta_indices(provider)
    mean, cov = _current_mean_cov(provider, x)
    1 <= j_beta <= length(indices) || throw(BoundsError(indices, j_beta))
    length(active_beta) == length(indices) || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    active_beta[j_beta] && throw(ArgumentError("conditional boundary density is defined for inactive coordinates; beta coordinate $j_beta is active"))
    return _conditional_logdensity_zero(mean, cov, x[indices], active_beta, j_beta)
end

function conditional_logdensity_zero(provider::DenseGaussianSlab, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    indices = beta_indices(provider)
    1 <= j_beta <= length(indices) || throw(BoundsError(indices, j_beta))
    length(active_beta) == length(indices) || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    active_beta[j_beta] && throw(ArgumentError("conditional boundary density is defined for inactive coordinates; beta coordinate $j_beta is active"))
    return _conditional_logdensity_zero(provider, x, active_beta, j_beta)
end

"""
    conditional_density_zero(provider, x, active_beta, j_beta)

Density version of `conditional_logdensity_zero`.
"""
conditional_density_zero(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector, j_beta::Integer) =
    exp(conditional_logdensity_zero(provider, x, active_beta, j_beta))

function conditional_logdensity_zero(provider::IndependentZeroMeanGaussianSlab, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    1 <= j_beta <= length(provider.beta_indices) || throw(BoundsError(provider.kappa, j_beta))
    active_beta[j_beta] && throw(ArgumentError("conditional boundary density is defined for inactive coordinates; beta coordinate $j_beta is active"))
    return provider.log_q_zero[j_beta]
end

function conditional_logdensity_zero(provider::IndependentZeroMeanLogscaleGaussianSlab, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    1 <= j_beta <= length(provider.beta_indices) || throw(BoundsError(provider.log_base_scales, j_beta))
    active_beta[j_beta] && throw(ArgumentError("conditional boundary density is defined for inactive coordinates; beta coordinate $j_beta is active"))
    log_s = provider.log_base_scales[j_beta] + x[provider.logscale_indices[j_beta]]
    return -0.5 * log2π - log_s
end

"""
    log_boundary_density_zero(provider, x, active_beta, j_beta)

Return the log boundary density at zero for adding inactive beta coordinate
`j_beta`. Gaussian providers use the conditional Gaussian density; arbitrary
providers dispatch to their `log_q_zero!` callback.
"""
log_boundary_density_zero(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector, j_beta::Integer) =
    conditional_logdensity_zero(provider, x, active_beta, j_beta)

function log_boundary_density_zero(provider::ArbitrarySlabBoundary, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    indices = beta_indices(provider)
    1 <= j_beta <= length(indices) || throw(BoundsError(indices, j_beta))
    length(active_beta) == length(indices) || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    active_beta[j_beta] && throw(ArgumentError("boundary density is defined for inactive coordinates; beta coordinate $j_beta is active"))
    return Float64(provider.log_q_zero!(x, active_beta, j_beta))
end

function _cholesky_factor_prefix!(A::AbstractMatrix{Float64}, k::Integer)
    @inbounds for j in 1:k
        s = A[j, j]
        for r in 1:j-1
            s -= abs2(A[j, r])
        end
        @assert s > 0
        Ajj = sqrt(s)
        A[j, j] = Ajj
        for i in j+1:k
            t = A[i, j]
            for r in 1:j-1
                t -= A[i, r] * A[j, r]
            end
            A[i, j] = t / Ajj
        end
    end
    return A
end

function _cholesky_solve_with_factor_prefix!(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, k::Integer)
    @inbounds for i in 1:k
        s = b[i]
        for j in 1:i-1
            s -= A[i, j] * b[j]
        end
        b[i] = s / A[i, i]
    end
    @inbounds for i in k:-1:1
        s = b[i]
        for j in i+1:k
            s -= A[j, i] * b[j]
        end
        b[i] = s / A[i, i]
    end
    return b
end

function _cholesky_solve_spd_prefix!(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, k::Integer)
    _cholesky_factor_prefix!(A, k)
    return _cholesky_solve_with_factor_prefix!(A, b, k)
end

"""
    active_prior_grad!(provider, out, x, active_beta)
    active_prior_neggrad!(provider, out, x, active_beta)

Write the active slab prior contribution into `out`. Despite the historical
`active_prior_grad!` name, the contract is the negative-gradient contribution
used by `DependentSlabTarget`: `posterior_grad - slab_prior_grad + active_slab_buf`.
The subtracted prior callback must contain only the slab component; nuisance
prior terms should remain in the posterior gradient.
For dense fixed Gaussian slabs this method is allocation-free after warmup.
"""
function active_prior_grad!(provider::DenseGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    length(active_beta) == length(indices) ||
        throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    cache = provider.cache
    k = _active_count(active_beta)
    if iszero(k)
        fill!(out, 0.0)
        return out
    end
    work_cov = cache.work_cov
    work_rhs = cache.work_rhs
    _fill_dense_active_system!(work_cov, work_rhs, provider, x, active_beta, k)
    _cholesky_solve_spd_prefix!(work_cov, work_rhs, k)
    return _write_active_gradient!(out, indices, active_beta, work_rhs, k)
end

function active_prior_grad!(provider::IndependentZeroMeanGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    length(active_beta) == length(indices) ||
        throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    fill!(out, 0.0)
    if length(out) == length(indices)
        @inbounds for j in eachindex(indices)
            active_beta[j] && (out[j] = provider.precision[j] * x[indices[j]])
        end
    elseif maximum(indices) <= length(out)
        @inbounds for j in eachindex(indices)
            active_beta[j] && (out[indices[j]] = provider.precision[j] * x[indices[j]])
        end
    else
        throw(DimensionMismatch("out must have length $(length(indices)) for beta-block gradients or at least $(maximum(indices)) for full-state gradients"))
    end
    return out
end

function active_prior_grad!(provider::IndependentZeroMeanLogscaleGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    length(active_beta) == length(indices) ||
        throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    fill!(out, 0.0)
    full_required = max(maximum(indices), maximum(provider.logscale_indices))
    if length(out) == length(indices)
        @inbounds for j in eachindex(indices)
            if active_beta[j]
                log_s = provider.log_base_scales[j] + x[provider.logscale_indices[j]]
                out[j] = x[indices[j]] / exp(2log_s)
            end
        end
        return out
    elseif full_required <= length(out)
        @inbounds for j in eachindex(indices)
            if active_beta[j]
                β = x[indices[j]]
                log_s = provider.log_base_scales[j] + x[provider.logscale_indices[j]]
                inv_s2 = exp(-2log_s)
                out[indices[j]] += β * inv_s2
                out[provider.logscale_indices[j]] += 1 - β^2 * inv_s2
            end
        end
        return out
    else
        throw(DimensionMismatch("out must have length $(length(indices)) for beta-block gradients or at least $full_required for full-state gradients"))
    end
end

function active_prior_grad!(provider::GlobalLogscaleExchangeableGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    length(active_beta) == length(indices) ||
        throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    fill!(out, 0.0)
    active_positions = _active_positions(active_beta, length(indices))
    isempty(active_positions) && return out
    full_required = max(maximum(indices), provider.logscale_index)
    length(out) >= full_required ||
        throw(DimensionMismatch("GlobalLogscaleExchangeableGaussianSlab active gradients require a full-state output of length at least $full_required"))
    μ = provider.mean
    k = length(active_positions)
    scale_log = provider.logscale_offset + x[provider.logscale_index]
    scale_inv2 = exp(-2scale_log)
    denom = provider.u + k * provider.v
    inv_base_diag = (provider.u + (k - 1) * provider.v) / (provider.u * denom)
    inv_base_off = -provider.v / (provider.u * denom)
    centered_sum = 0.0
    @inbounds for j in active_positions
        centered_sum += x[indices[j]] - μ
    end
    quad = 0.0
    @inbounds for j in active_positions
        centered = x[indices[j]] - μ
        base_solved = inv_base_diag * centered + inv_base_off * (centered_sum - centered)
        grad = scale_inv2 * base_solved
        out[indices[j]] += grad
        quad += centered * grad
    end
    out[provider.logscale_index] += k - quad
    return out
end

function active_prior_grad!(provider::AbstractGaussianSlabProvider, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    A = _active_positions(active_beta, length(indices))
    if isempty(A)
        fill!(out, 0.0)
        return out
    end
    mean, cov = _current_mean_cov(provider, x)
    F = cholesky(Symmetric(Matrix{Float64}(cov[A, A])); check=true)
    delta = Vector{Float64}(x[indices[A]] .- mean[A])
    grad_active = F \ delta
    return _write_active_gradient!(out, indices, active_beta, grad_active, length(A))
end

function active_prior_grad!(provider::CallbackGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    provider.active_prior_grad! === nothing &&
        throw(ArgumentError("CallbackGaussianSlab requires active_prior_grad! for active prior gradients"))
    provider.active_prior_grad!(out, x, active_beta)
    return out
end
