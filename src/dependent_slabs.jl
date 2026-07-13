"""
    AbstractModelPriorOdds

Interface for model-prior odds used by dependent slab clocks. Subtypes implement
`log_model_add_odds(prior, active, j)`, the log prior odds for adding inactive
beta coordinate `j` to the active model encoded by `active`.
"""
abstract type AbstractModelPriorOdds end

"""
    AbstractSlabBoundary

Interface for active-face slab boundary providers. A boundary provider supplies
the beta coordinate map, active-face negative gradients, and boundary densities
at zero for inactive coordinates.
"""
abstract type AbstractSlabBoundary end

"""
    AbstractUnstickClock
    AbstractAggregateUnstickClock

Clock interfaces for sticky unfreezing. Aggregate clocks sample a single
summed unsticking time over all inactive stickable beta coordinates and then
sample the coordinate label conditionally.
"""
abstract type AbstractUnstickClock end

"""
    AbstractAggregateUnstickClock

Subtype of `AbstractUnstickClock` for clocks that sample a single aggregate
unstick process over all inactive stickable beta coordinates.
"""
abstract type AbstractAggregateUnstickClock <: AbstractUnstickClock end

"""
    AbstractSlabCacheStyle
    NoSlabCache
    FixedCovarianceCache

Traits describing whether a slab boundary provider can be cached by active set.
`FixedCovarianceCache` means the Gaussian slab mean/covariance are fixed over a
linear-flow segment; state-dependent callback providers should use `NoSlabCache`.
"""
abstract type AbstractSlabCacheStyle end

"""
    AbstractCertifiedResidualCapability
    NoCertifiedResidualCapability

Trait objects for providers that can support certified residual proposal
envelopes. `NoCertifiedResidualCapability` means the provider has not supplied
enough structure or validated bounds for a Chebyshev/Fourier residual clock.
"""
abstract type AbstractCertifiedResidualCapability end

struct NoCertifiedResidualCapability <: AbstractCertifiedResidualCapability end
struct FloatingPointScalarResidualCapability <: AbstractCertifiedResidualCapability end

struct ScalarLogscaleGaussianLineSegment <: AbstractCertifiedResidualCapability
    a::Float64
    b::Float64
    s0::Float64
    ell0::Float64
    r::Float64
    log_total_weight::Float64
    horizon::Float64
end

"""
    NoSlabCache

Cache trait for slab boundary providers whose active-face quantities cannot be
cached by active set alone, for example state-dependent callbacks.
"""
struct NoSlabCache <: AbstractSlabCacheStyle end

"""
    FixedCovarianceCache

Cache trait for slab boundary providers whose Gaussian covariance is fixed over
the relevant trajectory segment.
"""
struct FixedCovarianceCache <: AbstractSlabCacheStyle end

"""
    BernoulliModelPriorOdds(prob)

Independent Bernoulli model prior for beta inclusion indicators. `prob[j]` is
the prior inclusion probability for beta coordinate `j`; `log_model_add_odds`
returns `log(prob[j] / (1 - prob[j]))`.
"""
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

"""
    log_model_add_odds(prior, active, j)

Return the log prior odds for adding inactive beta coordinate `j` to the active
model `active`. The index `j` is in beta-block coordinates, not full-state
coordinates.
"""
function log_model_add_odds(prior::BernoulliModelPriorOdds, active::BitVector, j::Integer)
    1 <= j <= length(prior.prob) || throw(BoundsError(prior.prob, j))
    length(active) == length(prior.prob) || throw(DimensionMismatch("active set length $(length(active)) does not match prior length $(length(prior.prob))"))
    p = prior.prob[j]
    iszero(p) && return -Inf
    isone(p) && return Inf
    return log(p) - log1p(-p)
end

"""
    BetaBernoulliModelPriorOdds(n, a, b)

Exchangeable beta-Bernoulli model prior for `n` beta coordinates with
inclusion-probability hyperprior `Beta(a, b)`.
"""
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

"""
    ExchangeableModelSizePrior(log_omega; normalize=false)

Exchangeable model-size prior over sizes `0:p`. `log_omega[k + 1]` stores the
log prior mass for model size `k`. If `normalize=true`, the entries are shifted
by `logsumexp(log_omega)`.
"""
struct ExchangeableModelSizePrior <: AbstractModelPriorOdds
    log_omega::Vector{Float64}
    function ExchangeableModelSizePrior(log_omega::AbstractVector{<:Real}; normalize::Bool=false)
        length(log_omega) >= 2 || throw(ArgumentError("log_omega must contain model-size probabilities for k=0:p"))
        vals = Vector{Float64}(log_omega)
        all(isfinite, vals) || throw(ArgumentError("log_omega entries must be finite"))
        if normalize
            lse = LogExpFunctions.logsumexp(vals)
            vals .-= lse
        end
        new(vals)
    end
end

Base.length(prior::ExchangeableModelSizePrior) = length(prior.log_omega) - 1
Base.copy(prior::ExchangeableModelSizePrior) = ExchangeableModelSizePrior(copy(prior.log_omega))

function log_model_add_odds(prior::ExchangeableModelSizePrior, active::BitVector, j::Integer)
    p = length(prior)
    1 <= j <= p || throw(BoundsError(active, j))
    length(active) == p || throw(DimensionMismatch("active set length $(length(active)) does not match prior length $p"))
    active[j] && throw(ArgumentError("log_model_add_odds is defined for adding an inactive coordinate; coordinate $j is already active"))
    k = count(active)
    k < p || return -Inf
    return prior.log_omega[k + 2] - prior.log_omega[k + 1] + log(k + 1) - log(p - k)
end

"""
    AbstractGaussianSlabProvider
    AbstractExchangeableGaussianSlab

Gaussian slab providers define a beta-block Gaussian slab through
`gaussian_slab!`. Exchangeable subtypes use the covariance form
`u * I + v * ones(p, p)`.
"""
abstract type AbstractGaussianSlabProvider <: AbstractSlabBoundary end

"""
    AbstractExchangeableGaussianSlab

Gaussian slab provider with exchangeable covariance structure
`u * I + v * ones(p, p)`.
"""
abstract type AbstractExchangeableGaussianSlab <: AbstractGaussianSlabProvider end

const _LOG2PI = log(2π)

mutable struct DenseGaussianSlabCache
    active_positions::Vector{Int}
    work_cov::Matrix{Float64}
    work_rhs::Vector{Float64}
end

DenseGaussianSlabCache(m::Integer) = DenseGaussianSlabCache(Vector{Int}(undef, m), Matrix{Float64}(undef, m, m), Vector{Float64}(undef, m))

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
        new(Vector{Int}(beta_indices), Vector{Int}(logscale_indices), Vector{Float64}(log_base_scales))
    end
end

"""
    GlobalLogscaleExchangeableGaussianSlab(beta_indices, logscale_index, u0, v0; mean=0, logscale_offset=0)

Exchangeable Gaussian slab with covariance
`exp(2 * (logscale_offset + x[logscale_index])) * (u0 I + v0 11')`.
This provider is structured for future scalar residual certificates.
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
slab_cache_style(::AbstractSlabBoundary) = NoSlabCache()

"""
    slab_cache_key(provider, x, active_beta)

Return a cache key for active-face quantities, or `nothing` when the provider
does not support active-set-only caching.
"""
slab_cache_key(::AbstractSlabBoundary, ::AbstractVector, ::BitVector) = nothing
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
    certified_residual_capability(provider, flow)

Return a provider/flow-specific capability object for certified residual
proposal envelopes, or `NoCertifiedResidualCapability()` when the provider is
opaque or unsupported. A real capability must include the validated structural
information needed to prove envelope domination over a trajectory segment; point
evaluations from callbacks are not sufficient.
"""
certified_residual_capability(::AbstractSlabBoundary, ::Any) = NoCertifiedResidualCapability()

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
    length(mean_out) == length(provider.mean) || throw(DimensionMismatch("mean_out has length $(length(mean_out)), expected $(length(provider.mean))"))
    size(cov_out) == size(provider.cov) || throw(DimensionMismatch("cov_out has size $(size(cov_out)), expected $(size(provider.cov))"))
    copyto!(mean_out, provider.mean)
    copyto!(cov_out, provider.cov)
    return nothing
end

function gaussian_slab!(provider::ExchangeableGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    length(mean_out) == m || throw(DimensionMismatch("mean_out has length $(length(mean_out)), expected $m"))
    size(cov_out) == (m, m) || throw(DimensionMismatch("cov_out has size $(size(cov_out)), expected ($m, $m)"))
    fill!(mean_out, provider.mean)
    fill!(cov_out, provider.v)
    @inbounds for i in 1:m
        cov_out[i, i] = provider.u + provider.v
    end
    return nothing
end

function gaussian_slab!(provider::ZeroMeanExchangeableGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    length(mean_out) == m || throw(DimensionMismatch("mean_out has length $(length(mean_out)), expected $m"))
    size(cov_out) == (m, m) || throw(DimensionMismatch("cov_out has size $(size(cov_out)), expected ($m, $m)"))
    fill!(mean_out, 0.0)
    fill!(cov_out, provider.v)
    @inbounds for i in 1:m
        cov_out[i, i] = provider.u + provider.v
    end
    return nothing
end

function gaussian_slab!(provider::IndependentZeroMeanGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    length(mean_out) == m || throw(DimensionMismatch("mean_out has length $(length(mean_out)), expected $m"))
    size(cov_out) == (m, m) || throw(DimensionMismatch("cov_out has size $(size(cov_out)), expected ($m, $m)"))
    fill!(mean_out, 0.0)
    fill!(cov_out, 0.0)
    @inbounds for j in 1:m
        cov_out[j, j] = inv(provider.precision[j])
    end
    return nothing
end

function gaussian_slab!(provider::IndependentZeroMeanLogscaleGaussianSlab, mean_out::AbstractVector, cov_out::AbstractMatrix, x::AbstractVector)
    m = length(provider.beta_indices)
    length(mean_out) == m || throw(DimensionMismatch("mean_out has length $(length(mean_out)), expected $m"))
    size(cov_out) == (m, m) || throw(DimensionMismatch("cov_out has size $(size(cov_out)), expected ($m, $m)"))
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
    length(mean_out) == m || throw(DimensionMismatch("mean_out has length $(length(mean_out)), expected $m"))
    size(cov_out) == (m, m) || throw(DimensionMismatch("cov_out has size $(size(cov_out)), expected ($m, $m)"))
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
struct ArbitrarySlabBoundary{I<:AbstractVector{Int},G,Q} <: AbstractSlabBoundary
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
    return -0.5 * (k * _LOG2PI + logdet_cov + dot(delta, solved))
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

"""
    conditional_logdensity_zero(provider, x, active_beta, j_beta)

Return the Gaussian conditional log density of inactive beta coordinate
`j_beta` at zero, conditioning on the currently active beta coordinates.
"""
function conditional_logdensity_zero(provider::AbstractGaussianSlabProvider, x::AbstractVector, active_beta::BitVector, j_beta::Integer)
    indices = beta_indices(provider)
    mean, cov = _current_mean_cov(provider, x)
    return _conditional_logdensity_zero(mean, cov, x[indices], active_beta, j_beta)
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
    return -0.5 * _LOG2PI - log_s
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

function _write_active_gradient!(out::AbstractVector, full_indices::AbstractVector{Int}, active_positions::AbstractVector{Int}, grad_active::AbstractVector, k::Integer=length(active_positions))
    fill!(out, 0.0)
    if length(out) == length(full_indices)
        @inbounds for a in 1:k
            pos = active_positions[a]
            value = grad_active[a]
            out[pos] = value
        end
    elseif maximum(full_indices) <= length(out)
        @inbounds for a in 1:k
            pos = active_positions[a]
            value = grad_active[a]
            out[full_indices[pos]] = value
        end
    else
        throw(DimensionMismatch("out must have length $(length(full_indices)) for beta-block gradients or at least $(maximum(full_indices)) for full-state gradients"))
    end
    return out
end

function _collect_active_positions!(positions::Vector{Int}, active_beta::BitVector)
    k = 0
    @inbounds for j in eachindex(active_beta)
        if active_beta[j]
            k += 1
            positions[k] = j
        end
    end
    return k
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
used by `DependentSlabTarget`: `posterior_grad - base_prior_grad + slab_buf`.
For dense fixed Gaussian slabs this method is allocation-free after warmup.
"""
function active_prior_grad!(provider::DenseGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    length(active_beta) == length(indices) ||
        throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $(length(indices))"))
    cache = provider.cache
    active_positions = cache.active_positions
    k = _collect_active_positions!(active_positions, active_beta)
    if iszero(k)
        fill!(out, 0.0)
        return out
    end
    work_cov = cache.work_cov
    work_rhs = cache.work_rhs
    @inbounds for a in 1:k
        ia = active_positions[a]
        work_rhs[a] = x[indices[ia]] - provider.mean[ia]
        for b in 1:k
            ib = active_positions[b]
            work_cov[a, b] = provider.cov[ia, ib]
        end
    end
    _cholesky_solve_spd_prefix!(work_cov, work_rhs, k)
    return _write_active_gradient!(out, indices, active_positions, work_rhs, k)
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
    return _write_active_gradient!(out, indices, A, grad_active)
end

function active_prior_grad!(provider::CallbackGaussianSlab, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    provider.active_prior_grad! === nothing &&
        throw(ArgumentError("CallbackGaussianSlab requires active_prior_grad! for active prior gradients"))
    provider.active_prior_grad!(out, x, active_beta)
    return out
end

"""
    active_prior_neggrad!(provider, out, x, active_beta)

Compatibility name for the active slab negative-gradient contract. New
providers should think of this as the canonical sign convention; historical
Gaussian providers also dispatch through `active_prior_grad!`.
"""
active_prior_neggrad!(provider::AbstractGaussianSlabProvider, out::AbstractVector, x::AbstractVector, active_beta::BitVector) =
    active_prior_grad!(provider, out, x, active_beta)

function active_prior_neggrad!(provider::ArbitrarySlabBoundary, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    provider.active_prior_neggrad!(out, x, active_beta)
    return out
end

active_prior_grad!(provider::ArbitrarySlabBoundary, out::AbstractVector, x::AbstractVector, active_beta::BitVector) =
    active_prior_neggrad!(provider, out, x, active_beta)

"""
    SummedRateClock(slab_provider, model_prior_odds; rtol=1e-8, atol=1e-10,
                    initial_bracket=1.0, bracket_multiplier=2.0)

Generic aggregate unstick clock. It evaluates the summed rate over all inactive
stickable beta coordinates by calling the provider's boundary densities and the
model-prior odds. This is the flexible arbitrary-prior baseline; it may allocate
and uses numerical quadrature/root finding for inhomogeneous clocks.
"""
struct SummedRateClock{P<:AbstractSlabBoundary,O<:AbstractModelPriorOdds,T<:Real} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior_odds::O
    rtol::T
    atol::T
    initial_bracket::T
    bracket_multiplier::T
end

"""
    AggregateClockDiagnostics

Mutable counters for residual-envelope aggregate clocks. `residual_area` is
reserved for certified proposal-envelope implementations; exact fallback
sampling leaves residual and envelope counters at zero and increments
`fallbacks`.
"""
mutable struct AggregateClockDiagnostics
    proposals::Int
    accepted::Int
    rejected::Int
    fallbacks::Int
    residual_area::Float64
    true_hazard::Float64
    envelope_hazard::Float64
    max_envelope_ratio::Float64
    max_residual::Float64
    min_envelope::Float64
    rate_evaluations::Int
    last_cells::Int
end

AggregateClockDiagnostics() = AggregateClockDiagnostics(0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0, Inf, 0, 0)

function reset_thinning_diagnostics!(diagnostics::AggregateClockDiagnostics)
    diagnostics.proposals = 0
    diagnostics.accepted = 0
    diagnostics.rejected = 0
    diagnostics.fallbacks = 0
    diagnostics.residual_area = 0.0
    diagnostics.true_hazard = 0.0
    diagnostics.envelope_hazard = 0.0
    diagnostics.max_envelope_ratio = 1.0
    diagnostics.max_residual = 0.0
    diagnostics.min_envelope = Inf
    diagnostics.rate_evaluations = 0
    diagnostics.last_cells = 0
    return diagnostics
end

"""
    ChebyshevResidualAggregateClock(slab_provider, model_prior_odds; ...)

Phase-6 moving-scale aggregate clock entry point for linear-flow residual
envelopes. Structured scalar residual paths use the direct interval-residual
construction with conservative floating-point safety inflation and assume
standard elementary functions such as `exp`, `log`, and `sqrt` are exact for
the certificate. This is a model-level certificate, not a theorem-level
machine-arithmetic certificate; an `IntervalArithmetic.jl` implementation with
outward rounding should be considered a future extension. Unsupported providers
route through the exact `SummedRateClock` fallback when
`allow_slow_fallback=true`. Set `allow_slow_fallback=false` to require a
provider/flow capability and error if none is available.
"""
struct ChebyshevResidualAggregateClock{P<:AbstractSlabBoundary,O<:AbstractModelPriorOdds,T<:Real,S<:SummedRateClock} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior_odds::O
    order::Int
    max_cells::Int
    residual_budget::T
    allow_slow_fallback::Bool
    diagnostics::AggregateClockDiagnostics
    fallback::S
end

"""
    FourierResidualAggregateClock(slab_provider, model_prior_odds; ...)

Phase-6 moving-scale aggregate clock entry point for Boomerang/Fourier residual
proposal envelopes. Until a provider supplies a genuinely certified residual
bound, sampling routes through the exact `SummedRateClock` fallback when
`allow_slow_fallback=true`. Set `allow_slow_fallback=false` to require a
certified provider/flow capability and error if none is available.
"""
struct FourierResidualAggregateClock{P<:AbstractSlabBoundary,O<:AbstractModelPriorOdds,T<:Real,S<:SummedRateClock} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior_odds::O
    order::Int
    cells::Int
    residual_budget::T
    allow_slow_fallback::Bool
    diagnostics::AggregateClockDiagnostics
    fallback::S
end

mutable struct LinearGaussianAggregateCache
    active_beta::BitVector
    stickable_beta::BitVector
    active_positions::Vector{Int}
    mean::Vector{Float64}
    cov::Matrix{Float64}
    beta_values::Vector{Float64}
    beta_velocity::Vector{Float64}
    cov_AA::Matrix{Float64}
    delta_A::Vector{Float64}
    velocity_A::Vector{Float64}
    cov_jA::Vector{Float64}
    solved_cross::Vector{Float64}
    log_weights::Vector{Float64}
end

function LinearGaussianAggregateCache(m::Integer)
    return LinearGaussianAggregateCache(
        BitVector(undef, m),
        BitVector(undef, m),
        Vector{Int}(undef, m),
        Vector{Float64}(undef, m),
        Matrix{Float64}(undef, m, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Matrix{Float64}(undef, m, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
    )
end

"""
    LinearGaussianAggregateClock(slab_provider, model_prior_odds; rtol=1e-8, atol=1e-10)

Optimized aggregate unstick clock for fixed-covariance Gaussian slabs under
linear flows. The provider must have `FixedCovarianceCache`; state-dependent
Gaussian callbacks should use `SummedRateClock`.
"""
struct LinearGaussianAggregateClock{P<:AbstractGaussianSlabProvider,O<:AbstractModelPriorOdds,T<:Real} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior_odds::O
    rtol::T
    atol::T
    cache::LinearGaussianAggregateCache
end

mutable struct ExponentialSumAggregateCache
    active_beta::BitVector
    stickable_beta::BitVector
    logc::Vector{Float64}
    slopes::Vector{Float64}
    log_weights::Vector{Float64}
end

function ExponentialSumAggregateCache(m::Integer)
    return ExponentialSumAggregateCache(BitVector(undef, m), BitVector(undef, m),
        Vector{Float64}(undef, m), Vector{Float64}(undef, m), Vector{Float64}(undef, m))
end

"""
    ExponentialSumAggregateClock(provider, model_prior_odds; rtol=1e-8, atol=1e-10)

Exact aggregate clock for independent zero-mean Gaussian slabs whose log
standard deviations are affine along linear-flow trajectories. The aggregate
rate has the form `sum_j c_j * exp(-r_j * t)`.
"""
struct ExponentialSumAggregateClock{P<:IndependentZeroMeanLogscaleGaussianSlab,O<:AbstractModelPriorOdds,T<:Real} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior_odds::O
    rtol::T
    atol::T
    cache::ExponentialSumAggregateCache
end

function _check_model_prior_length(model_prior_odds::AbstractModelPriorOdds, slab_provider::AbstractSlabBoundary)
    m = length(beta_indices(slab_provider))
    length(model_prior_odds) == m ||
        throw(DimensionMismatch("model-prior odds length $(length(model_prior_odds)) does not match beta dimension $m"))
    return nothing
end

function LinearGaussianAggregateClock(
    slab_provider::AbstractGaussianSlabProvider,
    model_prior_odds::AbstractModelPriorOdds;
    rtol::Real=1e-8,
    atol::Real=1e-10,
)
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    atol >= 0 || throw(ArgumentError("atol must be non-negative"))
    slab_cache_style(slab_provider) isa FixedCovarianceCache ||
        throw(ArgumentError("LinearGaussianAggregateClock requires a fixed-covariance slab provider; use SummedRateClock for state-dependent Gaussian callbacks"))
    _check_model_prior_length(model_prior_odds, slab_provider)
    return LinearGaussianAggregateClock(slab_provider, model_prior_odds, Float64(rtol), Float64(atol), LinearGaussianAggregateCache(length(beta_indices(slab_provider))))
end

function ExponentialSumAggregateClock(
    slab_provider::IndependentZeroMeanLogscaleGaussianSlab,
    model_prior_odds::AbstractModelPriorOdds;
    rtol::Real=1e-8,
    atol::Real=1e-10,
)
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    atol >= 0 || throw(ArgumentError("atol must be non-negative"))
    _check_model_prior_length(model_prior_odds, slab_provider)
    return ExponentialSumAggregateClock(slab_provider, model_prior_odds, Float64(rtol), Float64(atol), ExponentialSumAggregateCache(length(beta_indices(slab_provider))))
end

Base.copy(clock::LinearGaussianAggregateClock) = LinearGaussianAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior_odds);
    rtol=clock.rtol,
    atol=clock.atol,
)

Base.copy(clock::ExponentialSumAggregateClock) = ExponentialSumAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior_odds);
    rtol=clock.rtol,
    atol=clock.atol,
)

default_aggregate_unstick_clock(provider::IndependentZeroMeanLogscaleGaussianSlab, odds::AbstractModelPriorOdds) =
    ExponentialSumAggregateClock(provider, odds)
default_aggregate_unstick_clock(provider::AbstractGaussianSlabProvider, odds::AbstractModelPriorOdds) =
    slab_cache_style(provider) isa FixedCovarianceCache ? LinearGaussianAggregateClock(provider, odds) : SummedRateClock(provider, odds)
default_aggregate_unstick_clock(provider::AbstractSlabBoundary, odds::AbstractModelPriorOdds) =
    SummedRateClock(provider, odds)

function SummedRateClock(
    slab_provider::AbstractSlabBoundary,
    model_prior_odds::AbstractModelPriorOdds;
    rtol::Real=1e-8,
    atol::Real=1e-10,
    initial_bracket::Real=1.0,
    bracket_multiplier::Real=2.0,
)
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    atol >= 0 || throw(ArgumentError("atol must be non-negative"))
    initial_bracket > 0 || throw(ArgumentError("initial_bracket must be positive"))
    bracket_multiplier > 1 || throw(ArgumentError("bracket_multiplier must be greater than 1"))
    _check_model_prior_length(model_prior_odds, slab_provider)
    return SummedRateClock(slab_provider, model_prior_odds, Float64(rtol), Float64(atol), Float64(initial_bracket), Float64(bracket_multiplier))
end

Base.copy(clock::SummedRateClock) = SummedRateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior_odds);
    rtol=clock.rtol,
    atol=clock.atol,
    initial_bracket=clock.initial_bracket,
    bracket_multiplier=clock.bracket_multiplier,
)

function ChebyshevResidualAggregateClock(
    slab_provider::AbstractSlabBoundary,
    model_prior_odds::AbstractModelPriorOdds;
    order::Integer=16,
    max_cells::Integer=64,
    residual_budget::Real=1e-8,
    allow_slow_fallback::Bool=true,
    rtol::Real=1e-8,
    atol::Real=1e-10,
    initial_bracket::Real=1.0,
    bracket_multiplier::Real=2.0,
)
    order > 0 || throw(ArgumentError("order must be positive"))
    max_cells > 0 || throw(ArgumentError("max_cells must be positive"))
    residual_budget >= 0 || throw(ArgumentError("residual_budget must be non-negative"))
    fallback = SummedRateClock(slab_provider, model_prior_odds; rtol, atol, initial_bracket, bracket_multiplier)
    return ChebyshevResidualAggregateClock(slab_provider, model_prior_odds, Int(order), Int(max_cells), Float64(residual_budget), Bool(allow_slow_fallback), AggregateClockDiagnostics(), fallback)
end

Base.copy(clock::ChebyshevResidualAggregateClock) = ChebyshevResidualAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior_odds);
    order=clock.order,
    max_cells=clock.max_cells,
    residual_budget=clock.residual_budget,
    allow_slow_fallback=clock.allow_slow_fallback,
    rtol=clock.fallback.rtol,
    atol=clock.fallback.atol,
    initial_bracket=clock.fallback.initial_bracket,
    bracket_multiplier=clock.fallback.bracket_multiplier,
)

function FourierResidualAggregateClock(
    slab_provider::AbstractSlabBoundary,
    model_prior_odds::AbstractModelPriorOdds;
    order::Integer=16,
    cells::Integer=32,
    residual_budget::Real=1e-8,
    allow_slow_fallback::Bool=true,
    rtol::Real=1e-8,
    atol::Real=1e-10,
    initial_bracket::Real=1.0,
    bracket_multiplier::Real=2.0,
)
    order > 0 || throw(ArgumentError("order must be positive"))
    cells > 0 || throw(ArgumentError("cells must be positive"))
    residual_budget >= 0 || throw(ArgumentError("residual_budget must be non-negative"))
    fallback = SummedRateClock(slab_provider, model_prior_odds; rtol, atol, initial_bracket, bracket_multiplier)
    return FourierResidualAggregateClock(slab_provider, model_prior_odds, Int(order), Int(cells), Float64(residual_budget), Bool(allow_slow_fallback), AggregateClockDiagnostics(), fallback)
end

Base.copy(clock::FourierResidualAggregateClock) = FourierResidualAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior_odds);
    order=clock.order,
    cells=clock.cells,
    residual_budget=clock.residual_budget,
    allow_slow_fallback=clock.allow_slow_fallback,
    rtol=clock.fallback.rtol,
    atol=clock.fallback.atol,
    initial_bracket=clock.fallback.initial_bracket,
    bracket_multiplier=clock.fallback.bracket_multiplier,
)

reset_thinning_diagnostics!(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}) =
    reset_thinning_diagnostics!(clock.diagnostics)

function thinning_diagnostics(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock})
    d = clock.diagnostics
    proposal_count = max(d.proposals, 1)
    envelope_hazard = max(d.envelope_hazard, eps(Float64))
    return (
        proposals=d.proposals,
        accepted=d.accepted,
        rejected=d.rejected,
        fallbacks=d.fallbacks,
        residual_area=d.residual_area,
        true_hazard=d.true_hazard,
        envelope_hazard=d.envelope_hazard,
        empirical_acceptance=d.accepted / proposal_count,
        hazard_acceptance=d.true_hazard / envelope_hazard,
        max_envelope_ratio=d.max_envelope_ratio,
        max_residual=d.max_residual,
        min_envelope=d.min_envelope,
        rate_evaluations=d.rate_evaluations,
        last_cells=d.last_cells,
    )
end

function _active_beta_from_free(provider::AbstractSlabBoundary, free::BitVector)
    indices = beta_indices(provider)
    active_beta = BitVector(undef, length(indices))
    @inbounds for k in eachindex(indices)
        active_beta[k] = free[indices[k]]
    end
    return active_beta
end

function _stickable_beta_from_can_stick(provider::AbstractSlabBoundary, can_stick::BitVector)
    indices = beta_indices(provider)
    stickable_beta = BitVector(undef, length(indices))
    @inbounds for k in eachindex(indices)
        stickable_beta[k] = can_stick[indices[k]]
    end
    return stickable_beta
end

"""
    boundary_logweights!(out, provider, model_prior, x, active_beta, stickable_beta)

Write log unnormalized boundary weights for inactive stickable beta coordinates.
Entries that are active or not stickable are set to `-Inf`.
"""
function boundary_logweights!(
    out::AbstractVector,
    provider::AbstractSlabBoundary,
    model_prior::AbstractModelPriorOdds,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    m = length(beta_indices(provider))
    length(out) == m || throw(DimensionMismatch("out length $(length(out)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))
    fill!(out, -Inf)
    @inbounds for j in 1:m
        if stickable_beta[j] && !active_beta[j]
            out[j] = log_model_add_odds(model_prior, active_beta, j) +
                     log_boundary_density_zero(provider, x, active_beta, j)
        end
    end
    return out
end

_exchangeable_mean(provider::ExchangeableGaussianSlab) = provider.mean
_exchangeable_mean(::ZeroMeanExchangeableGaussianSlab) = 0.0

function _exchangeable_conditional_params(provider::AbstractExchangeableGaussianSlab, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    m = length(indices)
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    k = count(active_beta)
    μ = _exchangeable_mean(provider)
    u = provider.u
    v = provider.v
    denom = u + k * v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    sum_centered = 0.0
    @inbounds for j in 1:m
        active_beta[j] && (sum_centered += x[indices[j]] - μ)
    end
    c = v / denom
    cond_mean = μ + c * sum_centered
    cond_var = u * (u + (k + 1) * v) / denom
    cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
    return cond_mean, cond_var
end

function _exchangeable_conditional_params(provider::GlobalLogscaleExchangeableGaussianSlab, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    m = length(indices)
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    k = count(active_beta)
    μ = provider.mean
    denom = provider.u + k * provider.v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    sum_centered = 0.0
    @inbounds for j in 1:m
        active_beta[j] && (sum_centered += x[indices[j]] - μ)
    end
    c = provider.v / denom
    cond_mean = μ + c * sum_centered
    base_var = provider.u * (provider.u + (k + 1) * provider.v) / denom
    scale2 = exp(2 * (provider.logscale_offset + x[provider.logscale_index]))
    cond_var = scale2 * base_var
    cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
    return cond_mean, cond_var
end

function _exchangeable_log_q_zero(provider::AbstractExchangeableGaussianSlab, x::AbstractVector, active_beta::BitVector)
    cond_mean, cond_var = _exchangeable_conditional_params(provider, x, active_beta)
    return -0.5 * (_LOG2PI + log(cond_var) + cond_mean^2 / cond_var)
end

function boundary_logweights!(
    out::AbstractVector,
    provider::AbstractExchangeableGaussianSlab,
    model_prior::AbstractModelPriorOdds,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    m = length(beta_indices(provider))
    length(out) == m || throw(DimensionMismatch("out length $(length(out)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))
    fill!(out, -Inf)
    log_q = _exchangeable_log_q_zero(provider, x, active_beta)
    @inbounds for j in 1:m
        if stickable_beta[j] && !active_beta[j]
            out[j] = log_model_add_odds(model_prior, active_beta, j) + log_q
        end
    end
    return out
end

"""
    aggregate_lograte(provider, model_prior, log_Cv, x, active_beta, stickable_beta)

Return the log aggregate unstick rate, equal to `log_Cv` plus the log-sum of
inactive stickable boundary weights.
"""
function aggregate_lograte(
    provider::AbstractExchangeableGaussianSlab,
    model_prior::ExchangeableModelSizePrior,
    log_Cv::Real,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    m = length(beta_indices(provider))
    length(model_prior) == m ||
        throw(DimensionMismatch("model-prior length $(length(model_prior)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))
    n_inactive_stickable = count(j -> stickable_beta[j] && !active_beta[j], 1:m)
    iszero(n_inactive_stickable) && return -Inf
    k = count(active_beta)
    k < length(model_prior) || return -Inf
    log_q = _exchangeable_log_q_zero(provider, x, active_beta)
    log_rho = model_prior.log_omega[k + 2] - model_prior.log_omega[k + 1] + log(k + 1) - log(m - k)
    return Float64(log_Cv) + log_q + log(n_inactive_stickable) + log_rho
end

function boundary_logweights!(
    out::AbstractVector,
    provider::AbstractGaussianSlabProvider,
    model_prior::AbstractModelPriorOdds,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    indices = beta_indices(provider)
    m = length(indices)
    length(out) == m || throw(DimensionMismatch("out length $(length(out)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))

    fill!(out, -Inf)
    mean, cov = _current_mean_cov(provider, x)
    beta_values = x[indices]
    A = _active_positions(active_beta, m)

    if isempty(A)
        @inbounds for j in 1:m
            if stickable_beta[j] && !active_beta[j]
                var = cov[j, j]
                var > 0 || throw(ArgumentError("conditional variance must be positive, got $var"))
                log_q = -0.5 * (_LOG2PI + log(var) + mean[j]^2 / var)
                out[j] = log_model_add_odds(model_prior, active_beta, j) + log_q
            end
        end
        return out
    end

    cov_AA = Matrix{Float64}(cov[A, A])
    F = cholesky(Symmetric(cov_AA); check=true)
    delta_A = Vector{Float64}(beta_values[A] .- mean[A])
    solved_delta = F \ delta_A

    @inbounds for j in 1:m
        if stickable_beta[j] && !active_beta[j]
            cov_jA = Vector{Float64}(cov[j, A])
            solved_cross = F \ cov_jA
            cond_mean = mean[j] + dot(cov_jA, solved_delta)
            cond_var = cov[j, j] - dot(cov_jA, solved_cross)
            cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
            log_q = -0.5 * (_LOG2PI + log(cond_var) + cond_mean^2 / cond_var)
            out[j] = log_model_add_odds(model_prior, active_beta, j) + log_q
        end
    end
    return out
end

function aggregate_lograte(
    provider::AbstractSlabBoundary,
    model_prior::AbstractModelPriorOdds,
    log_Cv::Real,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    weights = Vector{Float64}(undef, length(beta_indices(provider)))
    boundary_logweights!(weights, provider, model_prior, x, active_beta, stickable_beta)
    lse = LogExpFunctions.logsumexp(weights)
    isfinite(lse) || return -Inf
    return Float64(log_Cv) + lse
end

"""
    sample_unstick_label(rng, provider, model_prior, x, active_beta, stickable_beta)

Sample a full-state coordinate label from the inactive stickable beta
coordinates, with probabilities proportional to the boundary weights.
"""
function sample_unstick_label(
    rng::Random.AbstractRNG,
    provider::AbstractSlabBoundary,
    model_prior::AbstractModelPriorOdds,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    weights = Vector{Float64}(undef, length(beta_indices(provider)))
    boundary_logweights!(weights, provider, model_prior, x, active_beta, stickable_beta)
    maxv = maximum(weights)
    isfinite(maxv) || throw(ArgumentError("cannot sample an unstick label because all inactive stickable rates are zero"))
    probs = exp.(weights .- maxv)
    j_beta = sample(rng, eachindex(probs), Weights(probs))
    return beta_indices(provider)[j_beta]
end

function sample_unstick_label(
    rng::Random.AbstractRNG,
    provider::AbstractExchangeableGaussianSlab,
    model_prior::ExchangeableModelSizePrior,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    indices = beta_indices(provider)
    candidates = Int[]
    for j in eachindex(indices)
        stickable_beta[j] && !active_beta[j] && push!(candidates, indices[j])
    end
    isempty(candidates) && throw(ArgumentError("cannot sample an unstick label because there are no inactive stickable coordinates"))
    return rand(rng, candidates)
end

"""
    DependentSlabTarget(d, posterior_grad!, prior_grad!, slab_provider, model_prior_odds;
                        initial_free=trues(d))

Gradient target for dependent-slab sticky samplers. The returned callable writes
`posterior_grad - prior_grad + active_slab_neggrad` into `out` and synchronizes
its active beta set through `set_active_set!`.
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
