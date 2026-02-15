# ──────────────────────────────────────────────────────────────────────────────
# Target structs: each bundles the reference distribution, precomputed
# quantities, and work buffers.  Explicit methods `neg_gradient!`,
# `neg_hvp!`, and `neg_partial` replace the former closures.
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# MvNormal target
# ──────────────────────────────────────────────────────────────────────────────

struct MvNormalTarget{T<:Distributions.AbstractMvNormal}
    D::T
    Σ_inv::Matrix{Float64}
    potential::Vector{Float64}   # Σ_inv * μ
    buffer::Vector{Float64}
end

function neg_gradient!(t::MvNormalTarget, out::AbstractVector, x::AbstractVector)
    mul!(t.buffer, t.Σ_inv, x)
    t.buffer .-= t.potential
    out .= t.buffer
end

function neg_hvp!(t::MvNormalTarget, out::AbstractVector, ::AbstractVector, v::AbstractVector)
    mul!(out, t.Σ_inv, v)
end

function neg_partial(t::MvNormalTarget, x::AbstractVector, i::Integer)
    dot(view(t.Σ_inv, :, i), x) - t.potential[i]
end

function gen_data(::Type{Distributions.MvNormal}, d, η,
                  μ = rand(Normal(0, 5), d),
                  σ = rand(LogNormal(0, 1), d),
                  R = rand(LKJ(d, η)))
    Σ = Symmetric(Diagonal(σ) * R * Diagonal(σ))
    Σ_inv = inv(Σ) # could do this in a safer way
    potential = Σ_inv * μ
    buffer = similar(potential)
    D = MvNormal(μ, Σ)
    return MvNormalTarget(D, Matrix(Σ_inv), potential, buffer)
end

# ──────────────────────────────────────────────────────────────────────────────
# ZeroMeanIsoNormal target
# ──────────────────────────────────────────────────────────────────────────────

struct ZeroMeanIsoNormalTarget{T<:Distributions.AbstractMvNormal}
    D::T
end

function neg_gradient!(::ZeroMeanIsoNormalTarget, out::AbstractVector, x::AbstractVector)
    copyto!(out, x)
end

function neg_hvp!(::ZeroMeanIsoNormalTarget, out::AbstractVector, ::AbstractVector, v::AbstractVector)
    copyto!(out, v) # == mul!(out, I, v)
end

function neg_partial(::ZeroMeanIsoNormalTarget, x::AbstractVector, i::Integer)
    x[i]
end

function gen_data(::Type{Distributions.ZeroMeanIsoNormal}, d)
    D = MvNormal(Diagonal(fill(1.0, d)))
    return ZeroMeanIsoNormalTarget(D)
end

# ──────────────────────────────────────────────────────────────────────────────
# MvTDist target
# ──────────────────────────────────────────────────────────────────────────────

struct MvTDistTarget{T<:Distributions.MvTDist}
    D::T
    Σ_inv::Matrix{Float64}
    μ::Vector{Float64}
    ν::Float64
    d::Int
    scalar_coeff::Float64       # (ν + d) / ν
    # work buffers
    x_centered::Vector{Float64}
    mahal_num::Vector{Float64}
end

function neg_gradient!(t::MvTDistTarget, out::AbstractVector, x::AbstractVector)
    t.x_centered .= x .- t.μ
    mul!(t.mahal_num, t.Σ_inv, t.x_centered)  # mahal_num = Σ⁻¹(x-μ)
    mahal_sq = dot(t.x_centered, t.mahal_num)  # (x-μ)ᵀΣ⁻¹(x-μ)
    denominator = 1 + mahal_sq / t.ν
    out .= (t.scalar_coeff / denominator) .* t.mahal_num
end

function neg_hvp!(t::MvTDistTarget, out::AbstractVector, x::AbstractVector, v::AbstractVector)
    t.x_centered .= x .- t.μ
    mul!(t.mahal_num, t.Σ_inv, t.x_centered)  # mahal_num = Σ⁻¹(x-μ)
    q = dot(t.x_centered, t.mahal_num)
    c1 = (t.ν + t.d) / (t.ν + q)

    # First term: c1 * Σ⁻¹ v
    mul!(out, t.Σ_inv, v, c1, zero(eltype(out)))

    # Second term: -2 c1 / (ν+q) * (uᵀΣ⁻¹v) * Σ⁻¹u, using dot(mahal_num, v) = uᵀΣ⁻¹v
    scalar = dot(t.mahal_num, v)
    axpy!(-2 * c1 / (t.ν + q) * scalar, t.mahal_num, out)

    return out
end

function neg_partial(t::MvTDistTarget, x::AbstractVector, i::Integer)
    # Note: For the t-distribution, calculating one component requires
    # almost all the work of calculating the full vector.
    t.x_centered .= x .- t.μ
    mul!(t.mahal_num, t.Σ_inv, t.x_centered)
    mahal_sq = dot(t.x_centered, t.mahal_num)
    denominator = 1 + mahal_sq / t.ν
    return (t.scalar_coeff / denominator) * t.mahal_num[i]
end

function gen_data(::Type{Distributions.MvTDist}, d, η,
                  μ = rand(Normal(0, 5), d),
                  σs = rand(LogNormal(0, 1), d);
                  ν=20.0)
    cholR = rand(LKJCholesky(d, η))
    lmul!(Diagonal(σs), cholR.L)
    Σ = PDMats.PDMat(cholR)

    D = MvTDist(ν, μ, Σ)

    Σ_inv = Matrix(inv(Σ))
    scalar_coeff = (ν + d) / ν

    x_centered = similar(μ)
    mahal_num = similar(μ)

    return MvTDistTarget(D, Σ_inv, μ, ν, d, scalar_coeff, x_centered, mahal_num)
end

# ──────────────────────────────────────────────────────────────────────────────
# GaussianMeanModel target
# ──────────────────────────────────────────────────────────────────────────────

struct GaussianMeanModel{T<:MvNormal}
    X::Matrix{Float64}    # n × p observations
    D::T
    prior_μ::Vector{Float64}    # Prior mean (usually zeros)
end

function GaussianMeanModel(n::Integer, μ, Σ, prior_μ = zero(μ))
    D = MvNormal(μ, Σ)
    X = permutedims(rand(D, n))
    return GaussianMeanModel(X, D, prior_μ)
end

# Analytic posterior (for comparison)
function analytic_posterior(obj::GaussianMeanModel)
    X = obj.X
    n = size(X, 1)
    μ₀ = obj.prior_μ
    Σ₀ = I
    Σ = cov(obj.D)
    x̄ = vec(mean(X, dims = 1))
    Σₙ = inv(inv(Σ₀) + n * inv(Σ))
    μₙ = Σₙ * (inv(Σ₀) * μ₀ + n * inv(Σ) * x̄)
    return MvNormal(μₙ, Σₙ)
end

struct GaussianMeanTarget{T<:MvNormal}
    D::T                       # analytic posterior (for comparison)
    obj::GaussianMeanModel
    Λ::Matrix{Float64}        # inv(cov(obj.D))
    Λ0::UniformScaling{Bool}  # inv(I) = I
    μ0::Vector{Float64}
    x̄::Vector{Float64}
    n::Int
    Σ_tot_inv::Matrix{Float64}
    # work buffers
    buffer::Vector{Float64}
    x̄_sub::Vector{Float64}
    # subsampling state
    indices::Vector{Int}
end

# Full negative gradient: ∇f(x) = Σ0⁻¹(x-μ0) + n * Σ⁻¹(x - x̄)
function neg_gradient!(t::GaussianMeanTarget, out::AbstractVector, x::AbstractVector)
    @. t.buffer = x - t.μ0
    mul!(out, t.Λ0, t.buffer)                 # prior part
    @. t.buffer = x - t.x̄
    mul!(out, t.Λ, t.buffer, t.n, 1.0)        # + n * Λ * (x - x̄)
end

# Subsampled negative gradient: replace x̄ with unbiased subsample mean
function neg_gradient_sub!(t::GaussianMeanTarget, out::AbstractVector, x::AbstractVector)
    mean!(t.x̄_sub', view(t.obj.X, t.indices, :))   # subsample mean
    @. t.buffer = x - t.μ0
    mul!(out, t.Λ0, t.buffer)                 # prior part
    @. t.buffer = x - t.x̄_sub
    mul!(out, t.Λ, t.buffer, t.n, 1.0)        # + n * Λ * (x - x̄_sub)
end

function neg_hvp!(t::GaussianMeanTarget, out::AbstractVector, ::AbstractVector, v::AbstractVector)
    mul!(out, t.Σ_tot_inv, v)
end

# Same Hessian form (no randomness needed), because Hessian doesn't depend on subsample
neg_hvp_sub!(t::GaussianMeanTarget, out, x, v) = neg_hvp!(t, out, x, v)

function neg_partial(t::GaussianMeanTarget, x::AbstractVector, i::Integer)
    dot(view(t.Σ_tot_inv, :, i), x) - t.μ0[i]
end

function resample_indices!(t::GaussianMeanTarget, n::Integer)
    length(t.indices) != n && resize!(t.indices, n)
    sample!(eachindex(t.indices), t.indices; replace = false)
end

function gen_data(::Type{GaussianMeanModel}, d, n, μ = zeros(d), Σ = I(d))
    obj = GaussianMeanModel(n, μ, Σ)
    D = analytic_posterior(obj)

    Λ = inv(cov(obj.D)) # could do this in a safer way
    Λ0 = inv(I)
    μ0 = obj.prior_μ
    x̄ = vec(mean(obj.X, dims=1))
    buffer = similar(x̄)
    x̄_sub  = similar(x̄)
    Σ_tot_inv = Λ0 + n .* Λ
    indices = Vector{Int}(undef, 0)

    return GaussianMeanTarget(D, obj, Matrix(Λ), Λ0, μ0, x̄, n, Matrix(Σ_tot_inv),
                              buffer, x̄_sub, indices)
end


# ──────────────────────────────────────────────────────────────────────────────
# LogisticRegressionModel target
# ──────────────────────────────────────────────────────────────────────────────

struct LogisticRegressionModel
    X::Matrix{Float64}
    y::Vector{Int}
    prior_μ::Vector{Float64}
    prior_Σ::Matrix{Float64}
    prior_Σ_inv::Matrix{Float64}
end

mutable struct LogisticRegressionTarget
    obj::LogisticRegressionModel
    β_true::Vector{Float64}
    # precomputed
    Λ0::Matrix{Float64}
    nobs::Int
    # work buffers
    buffer::Vector{Float64}
    η::Vector{Float64}          # length nobs
    p::Vector{Float64}          # length nobs
    # control variate anchor state
    β_anchor::Vector{Float64}
    η_anchor::Vector{Float64}   # η at anchor (length nobs)
    p_anchor::Vector{Float64}   # p = logistic(η_anchor)
    G_anchor::Vector{Float64}   # full data gradient data-part at anchor: X'*(p_anchor - y)
    # subsampling indices
    indices::Vector{Int}
end

# --- Full negative log posterior gradient: ∇f(β) = Σ0⁻¹(β-μ0) + X' (σ(Xβ) - y) ---
function neg_gradient!(t::LogisticRegressionTarget, out::AbstractVector, β::AbstractVector)
    X, y, μ0, Λ0 = t.obj.X, t.obj.y, t.obj.prior_μ, t.Λ0
    t.buffer .= β .- μ0
    mul!(out, Λ0, t.buffer)  # prior part
    # data part
    mul!(t.η, X, β)
    for i in eachindex(t.p)
        t.p[i] = LogExpFunctions.logistic(t.η[i]) - y[i]
    end
    mul!(out, X', t.p, 1.0, 1.0)
end

# --- Subsampled gradient with control variate ---
# estimator: prior + G_anchor + (n/m) * ( X_S'*(p_S(β)-y_S) - X_S'*(p_anchor_S - y_S) )
function neg_gradient_sub_cv!(t::LogisticRegressionTarget, out::AbstractVector, β::AbstractVector)
    X, y, μ0, Λ0 = t.obj.X, t.obj.y, t.obj.prior_μ, t.Λ0

    # prior
    t.buffer .= β .- μ0
    mul!(out, Λ0, t.buffer)

    # add full-data anchor data-part
    out .+= t.G_anchor

    # minibatch difference
    m = length(t.indices)
    iszero(m) && return out

    Xi = @view X[t.indices, :]
    ηc = view(t.η, eachindex(t.indices))   # temporary storage for η_i
    pc = view(t.p, eachindex(t.indices))
    # compute η_i(β) for batch
    mul!(ηc, Xi, β)                         # η_i = Xi * β

    pc .= LogExpFunctions.logistic.(ηc)     # p_i(β)

    # p_anchor for the batch (cheap indexed access)
    p0_batch = view(t.p_anchor, t.indices)

    # batch gradients: g_batch = Xi'*(p_i - y_i), g0_batch = Xi'*(p0_batch - y_i)
    # so diff = Xi'*( (p_i - y_i) - (p0_batch - y_i) ) = Xi'*(p_i - p0_batch)
    # compute diff = Xi'*(p_i - p0_batch)
    ηc .= pc .- p0_batch                  # reuse ηi storage for pi - p0
    mul!(t.buffer, Xi', ηc)               # buffer <- Xi' * (pi - p0_batch)

    scale = (t.nobs / m)
    out .+= scale .* t.buffer
end

# --- Full Hessian-vector product: ∇²f(β) v = Σ0⁻¹ v + X' ( w .* (Xv) ), w=p*(1-p) ---
function neg_hvp!(t::LogisticRegressionTarget, out::AbstractVector, β::AbstractVector, v::AbstractVector)
    X, Λ0 = t.obj.X, t.Λ0
    mul!(out, Λ0, v)   # prior part

    mul!(t.η, X, β)
    t.p .= LogExpFunctions.logistic.(t.η)
    t.p .= t.p .* (1.0 .- t.p)   # p now holds weights w

    mul!(t.η, X, v)          # η <- X * v
    t.η .*= t.p              # η <- w .* (X * v)
    mul!(t.buffer, X', t.η)
    out .+= t.buffer
end

# --- Subsampled Hessian-vector product (no CV here) ---
function neg_hvp_sub!(t::LogisticRegressionTarget, out::AbstractVector, β::AbstractVector, v::AbstractVector)
    X, Λ0 = t.obj.X, t.Λ0
    mul!(out, Λ0, v)

    m = length(t.indices)
    iszero(m) && return out

    Xi = @view X[t.indices, :]
    ηc = view(t.η, eachindex(t.indices))
    pc = view(t.p, eachindex(t.indices))

    mul!(ηc, Xi, β)
    pc .= LogExpFunctions.logistic.(ηc)
    pc .= pc .* (1.0 .- pc)    # now pi holds w_i
    Xv = ηc # rename for clarity
    mul!(Xv, Xi, v)                # X_i * v for batch
    Xv .*= pc
    mul!(t.buffer, Xi', Xv)        # buffer <- X_S' * ( w_S .* (X_S v) )
    scale = (t.nobs / m)
    out .+= scale .* t.buffer
end

function resample_indices!(t::LogisticRegressionTarget, m::Integer)
    length(t.indices) != m && resize!(t.indices, m)
    sample!(1:t.nobs, t.indices; replace = false)
end

# setter to (re)compute anchor; call whenever you want to update the anchor
function set_anchor!(t::LogisticRegressionTarget, β0::AbstractVector)
    X, y = t.obj.X, t.obj.y
    t.β_anchor .= β0
    mul!(t.η_anchor, X, t.β_anchor)
    @. t.p_anchor = LogExpFunctions.logistic(t.η_anchor)
    mul!(t.G_anchor, X', t.p_anchor .- y)
    return nothing
end

function anchor_info(t::LogisticRegressionTarget)
    return (β_anchor = copy(t.β_anchor),
            p_anchor = copy(t.p_anchor),
            G_anchor = copy(t.G_anchor))
end

function gen_data(::Type{LogisticRegressionModel}, d, n,
                  β_true = randn(d),
                  μ0 = zeros(d),
                  Σ0 = I(d))

    # simulate predictors
    X = randn(n, d-1)
    X .-= mean(X, dims=1) # center predictors
    X = hcat(ones(n), X)  # first column is the intercept
    η = X * β_true
    y = rand.(BernoulliLogit.(η))

    # prior precision
    Λ0 = inv(Σ0)

    obj = LogisticRegressionModel(X, y, μ0, Matrix{Float64}(Σ0), Matrix{Float64}(Λ0))

    nobs, d = size(X)
    buffer = zeros(d)
    p = similar(η)

    # --- control variate anchor state ---
    β_anchor = copy(μ0)
    η_anchor = similar(y, Float64)
    p_anchor = similar(y, Float64)
    G_anchor = zeros(d)

    indices = Vector{Int}(undef, 0)

    target = LogisticRegressionTarget(
        obj, β_true, Matrix{Float64}(Λ0), nobs,
        buffer, similar(η), p,
        β_anchor, η_anchor, p_anchor, G_anchor,
        indices)

    # initialize anchor
    set_anchor!(target, μ0)

    return target
end


# ──────────────────────────────────────────────────────────────────────────────
# Spike-and-slab distribution (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

struct SpikeAndSlabDist{D1, D2}
    spike_dist::D1
    slab_dist::D2
end

function marginal_pdfs_at_zero(D::Distributions.AbstractMvNormal)
    μ, Σ = mean(D), cov(D)
    [pdf(Normal(μ[i], sqrt(Σ[i, i])), 0.0) for i in 1:length(D)]
end

function marginal_pdfs_at_zero(D::Distributions.MvTDist)
    ν = D.df
    μ = mean(D)
    Σ = Matrix(D.Σ)  # scale matrix (not covariance)
    [pdf(μ[i] + sqrt(Σ[i, i]) * TDist(ν), 0.0) for i in 1:length(D)]
end

# The gen_data functions for spike-and-slab return (D, κ, slab_target).
# The slab_target provides the gradient/hvp for the continuous slab part;
# the test files combine it with the spike structure.

function gen_data(::Type{<:SpikeAndSlabDist{<:Bernoulli, <:Distributions.MvTDist}}, d, η)
    prob = rand(d)
    D1 = product_distribution([Bernoulli(prob[i]) for i in 1:d])
    # Zero mean and unit scales: ensures fast sticky mixing while
    # still exercising the position-dependent MvTDist gradient code.
    μ = zeros(d)
    σs = ones(d)
    slab_target = gen_data(Distributions.MvTDist, d, η, μ, σs)
    D = SpikeAndSlabDist(D1, slab_target.D)
    κ = prob ./ (1 .- prob) .* marginal_pdfs_at_zero(slab_target.D)
    return D, κ, slab_target
end

function gen_data(::Type{<:SpikeAndSlabDist{<:Bernoulli, T}}, d, args...) where T
    prob = rand(d)
    D1 = product_distribution([Bernoulli(prob[i]) for i in 1:d])
    slab_target = gen_data(T, d, args...)
    D = SpikeAndSlabDist(D1, slab_target.D)
    κ = prob ./ (1 .- prob) .* marginal_pdfs_at_zero(slab_target.D)
    return D, κ, slab_target
end

function gen_data2(::Type{<:SpikeAndSlabDist{<:Bernoulli, T}}, d, prob, args...) where T
    D1 = product_distribution([Bernoulli(prob[i]) for i in 1:d])
    slab_target = gen_data(T, d, args...)
    D = SpikeAndSlabDist(D1, slab_target.D)
    κ = prob ./ (1 .- prob) .* marginal_pdfs_at_zero(slab_target.D)
    return D, κ, slab_target
end

function gen_data(::Type{<:SpikeAndSlabDist{<:BetaBernoulli, T}}, d, args...) where T
    a, b = 2. + randexp(), 2 .+ randexp()
    D1 = BetaBernoulli(d, a, b)
    slab_target = gen_data(T, d, args...)
    D = SpikeAndSlabDist(D1, slab_target.D)
    mpdfs = marginal_pdfs_at_zero(slab_target.D)
    κ = BetaBernoulliKappa(a, b, mpdfs)
    return D, κ, slab_target
end


"""
    StickyTime(c, z0, vz)

Distribution of unfreeze times for hierarchical Beta–Bernoulli
(logit–θ parametrization) with slab density c=π_i(0), current logit z0,
and velocity vz of z.

- If vz == 0: this is just Exponential(c * exp(z0)).
- If vz != 0: τ = (1/vz) * log(1 + (vz/c*exp(z0)) * E), E ~ Exp(1).
"""
struct StickyTime <: ContinuousUnivariateDistribution
    c::Float64   # slab density at zero (π_i(0))
    z0::Float64  # current logit θ
    vz::Float64  # current velocity of z
end

function Base.rand(rng::Random.AbstractRNG, d::StickyTime)
    λ0 = d.c * exp(d.z0)
    if iszero(d.vz)
        return rand(rng, Exponential(λ0))
    else
        E = rand(rng, Exponential())
        arg = 1 + (d.vz * E) / λ0
        return arg > 0 ? log(arg) / d.vz : Inf
    end
end


"""
    BetaBernoulliHierarchical(n, a, b)

Hierarchical version of the BetaBernoulli distribution:
γᵢ ~ Bernouilli(p)
p  ~ Beta(a, b)
but _without_ integrating out p (which is what BetaBernoulli does).

primarily used for testing purposes.
"""
# type for testing that refers to
struct BetaBernoulliHierarchical{T<:BetaBernoulli} <: DiscreteMultivariateDistribution
    d::T
end
Base.length(d::BetaBernoulliHierarchical) = length(d.d)
Base.eltype(::Type{<:BetaBernoulliHierarchical}) = eltype(d.d)
Distributions.insupport(d::BetaBernoulliHierarchical, x::AbstractVector{<:Integer}) = Distributions.insupport(d.d, x)
Distributions._rand!(rng::Random.AbstractRNG, d::BetaBernoulliHierarchical, x::AbstractVector) = rand!(rng, d.d, x)
Distributions._logpdf(d::BetaBernoulliHierarchical, x::AbstractVector{<:Integer}) = Distributions._logpdf(d.d, x)
Statistics.mean(d::BetaBernoulliHierarchical) = mean(d.d)

# ──────────────────────────────────────────────────────────────────────────────
# BetaBernoulliHierarchical spike-and-slab target
#
# Extends the slab target with an extra logit-θ coordinate.
# Wraps a slab_target for the first d-1 coordinates, and adds the
# Beta(a,b) hyperprior gradient for the last (logit-θ) coordinate.
# ──────────────────────────────────────────────────────────────────────────────

struct BetaBernoulliHierarchicalTarget{ST}
    slab_target::ST
    a::Float64
    b::Float64
    mpdfs::Vector{Float64}      # marginal slab densities at zero
    d::Int                       # full dimension (d_slab + 1)
end

function neg_gradient!(t::BetaBernoulliHierarchicalTarget, out::AbstractVector, x::AbstractVector)
    d = t.d
    neg_gradient!(t.slab_target, view(out, 1:d-1), view(x, 1:d-1))
    z = x[d]
    θ = LogExpFunctions.logistic(z)
    out[d] = -t.a + (t.a + t.b) * θ
    return out
end

function neg_hvp!(t::BetaBernoulliHierarchicalTarget, out::AbstractVector, x::AbstractVector, v::AbstractVector)
    d = t.d
    neg_hvp!(t.slab_target, view(out, 1:d-1), view(x, 1:d-1), view(v, 1:d-1))
    z = x[d]
    θ = LogExpFunctions.logistic(z)
    out[d] = (t.a + t.b) * θ * (1 - θ) * v[d]
    return out
end

function neg_partial(t::BetaBernoulliHierarchicalTarget, x::AbstractVector, i::Integer)
    d = t.d
    if i <= d-1
        return neg_partial(t.slab_target, view(x, 1:d-1), i)
    elseif i == d
        z = x[d]
        θ = LogExpFunctions.logistic(z)
        return -t.a + (t.a + t.b) * θ
    else
        throw(ArgumentError("i out of bounds"))
    end
end

function gen_data(::Type{<:SpikeAndSlabDist{<:BetaBernoulliHierarchical, T}}, d, args...) where T

    # hyperprior for θ
    a, b = 2. + randexp(), 2. + randexp()

    # prior over indicators with explicit θ
    D1 = BetaBernoulliHierarchical(BetaBernoulli(d - 1, a, b))   # marker only

    # continuous slab part: first d-1 coordinates
    slab_target = gen_data(T, d - 1, args...)
    D = SpikeAndSlabDist(D1, slab_target.D)

    # precompute slab density at zero
    mpdfs = marginal_pdfs_at_zero(slab_target.D)

    # κ: conditional inclusion odds × slab density at 0
    κ = (i, x, γ, θ) -> begin
        @assert 1 <= i <= d-1 "κ only defined for slab coordinates"
        z0 = x[end]
        vz = θ[end]
        c  = mpdfs[i]
        return StickyTime(c, z0, vz)
    end

    target = BetaBernoulliHierarchicalTarget(slab_target, a, b, mpdfs, d)

    return D, κ, target
end


# ──────────────────────────────────────────────────────────────────────────────
# data_name: display names for test labels
# ──────────────────────────────────────────────────────────────────────────────

data_name(::Type{Distributions.ZeroMeanIsoNormal}, d) = "N(0, I($d))"
data_name(::Type{Distributions.MvNormal}, d) = "N(μ, Σ$d)"
data_name(::Type{Distributions.FullNormal}, d) = "N(μ, Σ$d)"
data_name(::Type{<:Distributions.MvTDist}, d) = "T(ν, μ, Σ$d)"
data_name(::Type{<:Distributions.AbstractMvNormal}, d) = "MvNormal($d)"

data_name(::Type{<:Distributions.Product{<:Any, T, <:Any}}, d) where T = data_name(T, d)
data_name(::Type{<:Distributions.Bernoulli}, d) = "Bern"
data_name(::Type{<:BetaBernoulli}, d) = "BetaBern"
data_name(::Type{<:BetaBernoulliHierarchical}, d) = "Bern(p), p ~ Beta"
data_name(::Type{SpikeAndSlabDist{D1, D2}}, d) where {D1, D2} = "δ₀ + (1 - δ₀)$(data_name(D2, d)), π₀ ~ $(data_name(D1, d))"

function _flow_name(trace)
    flow = trace.flow
    if flow isa PDMPSamplers.PreconditionedDynamics
        return "Precond($(nameof(typeof(flow.dynamics))))"
    else
        return string(nameof(typeof(flow)))
    end
end

function _f3(x)
    s = string(round(x; digits=3))
    i = findfirst('.', s)
    i === nothing && (s *= "."; i = length(s))
    return s * "0" ^ max(0, 3 - (length(s) - i))
end

function _format_elapsed(t::Real)
    if t < 1.0
        return @sprintf("%5.0fms", t * 1000)
    elseif t < 60.0
        return @sprintf("%5.1fs ", t)
    else
        return @sprintf("%4.1fmin", t / 60)
    end
end
_format_elapsed(::Nothing) = "      ?"

function _isapprox_closeness(a, b; rtol::Real=0.0, atol::Real=0.0)
    err = norm(a .- b)
    tol = max(atol, rtol * max(norm(a), norm(b)))
    return tol > 0 ? err / tol : (iszero(err) ? 0.0 : Inf)
end

# ──────────────────────────────────────────────────────────────────────────────
# test_approximation: compare trace estimators against known distribution
# ──────────────────────────────────────────────────────────────────────────────

function test_approximation(trace::PDMPSamplers.AbstractPDMPTrace, D::Distributions.AbstractMvNormal; elapsed::Union{Real,Nothing}=nothing)

    min_ess = minimum(ess(trace))
    if min_ess < 500
        show_test_diagnostics && @info "Skipping test_approximation (low ESS)" min_ess
        return nothing
    end

    mc = 1.0 / sqrt(min_ess)
    mean_rtol  = 0.2  + 3.0  * mc
    mean_atol  = 0.2  + 3.0  * mc
    cov_rtol   = 0.4  + 10.0 * mc
    quant_rtol = 0.2  + 3.0  * mc

    trace_mean = mean(trace)
    trace_cov = cov(trace)

    @test isapprox(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    cov_rtol < 1.0 && @test isapprox(trace_cov, cov(D); rtol=cov_rtol)

    # test quantiles in the first dimension
    probs = .1:.1:.99
    dist = Normal(mean(D)[1], sqrt(cov(D)[1, 1]))
    q_expected = map(Base.Fix1(quantile, dist), probs)
    q_observed = quantile(trace, collect(probs); coordinate=1)
    quant_rtol < 1.0 && @test isapprox(q_observed, q_expected; rtol=quant_rtol)

    c_mean  = _isapprox_closeness(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    c_cov   = _isapprox_closeness(trace_cov, cov(D); rtol=cov_rtol)
    c_quant = _isapprox_closeness(q_observed, q_expected; rtol=quant_rtol)
    failed = c_mean > 1.0 || (cov_rtol < 1.0 && c_cov > 1.0) || (quant_rtol < 1.0 && c_quant > 1.0)
    if failed || show_test_diagnostics
        d = length(D)
        label = failed ? "FAIL" : "ok  "
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(data_name(typeof(D), d), 16)) | ESS=$(lpad(round(Int, min_ess), 7)) | $(_format_elapsed(elapsed)) | c_mean=$(_f3(c_mean)) | c_cov=$(_f3(c_cov)) | c_quant=$(_f3(c_quant))")
    end
end

function test_approximation(trace::PDMPSamplers.AbstractPDMPTrace, D::Distributions.MvTDist; elapsed::Union{Real,Nothing}=nothing)

    ν_D, μ_D, Σ_D = Distributions.params(D)

    min_ess = minimum(ess(trace))
    if min_ess < 500
        show_test_diagnostics && @info "Skipping test_approximation (low ESS)" min_ess
        return nothing
    end

    mc = 1.0 / sqrt(min_ess)
    kurtosis_factor = ν_D > 4 ? sqrt((ν_D - 2) / (ν_D - 4)) : 2.0

    mean_rtol  = 0.2  + 3.0  * mc
    mean_atol  = 0.2  + 3.0  * mc
    cov_rtol   = 0.70 + 50.0 * kurtosis_factor * mc
    quant_rtol = 0.30 + 5.0  * kurtosis_factor * mc

    trace_mean = mean(trace)
    trace_cov = cov(trace)

    @test isapprox(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    cov_rtol < 1.0 && @test isapprox(trace_cov, cov(D); rtol=cov_rtol)

    # test quantiles in the first dimension
    qprobs = .1:.1:.99
    marginal_dist = μ_D[1] + sqrt(Σ_D[1, 1]) * Distributions.TDist(ν_D)
    q_expected = map(Base.Fix1(quantile, marginal_dist), qprobs)
    q_observed = quantile(trace, collect(qprobs); coordinate=1)
    quant_rtol < 1.0 && @test isapprox(q_observed, q_expected; rtol=quant_rtol)

    c_mean  = _isapprox_closeness(trace_mean, mean(D); rtol=mean_rtol, atol=mean_atol)
    c_cov   = _isapprox_closeness(trace_cov, cov(D); rtol=cov_rtol)
    c_quant = _isapprox_closeness(q_observed, q_expected; rtol=quant_rtol)
    failed = c_mean > 1.0 || (cov_rtol < 1.0 && c_cov > 1.0) || (quant_rtol < 1.0 && c_quant > 1.0)
    if failed || show_test_diagnostics
        d = length(D)
        label = failed ? "FAIL" : "ok  "
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(data_name(typeof(D), d), 16)) | ESS=$(lpad(round(Int, min_ess), 7)) | $(_format_elapsed(elapsed)) | c_mean=$(_f3(c_mean)) | c_cov=$(_f3(c_cov)) | c_quant=$(_f3(c_quant))")
    end
end

function test_approximation(
        samples,
        spike_dist::Union{
            Distributions.Product{<:Any, <:Bernoulli, <:Any},
            BetaBernoulli
        }
    )

    true_incl_probs = mean(spike_dist)

    # test using conjugate credible interval of Binomial model with Beta(1, 1) prior
    a, b = 1, 1
    cri_Δ = 1e-6#.001, 0.025 # <- would be better, but makes the test more flaky
    cri_bounds = [cri_Δ, 1 - cri_Δ]
    ind = falses(size(samples, 1))
    for i in axes(samples, 2)
        ind .= @view(samples[:, i]) .!= 0
        n_eff = MCMCDiagnosticTools.ess(ind)
        p̂ = mean(ind)
        k_eff = p̂ * n_eff

        # k = count(ind)
        # n = length(ind)
        # post = Beta(a + k, b + (n - k))
        post = Beta(a + k_eff, b + (n_eff - k_eff))
        lo, hi = quantile(post, cri_bounds)
        # increase the interval by 300 %
        delta = (hi - lo) * 3.0
        lo -= delta / 2
        hi += delta / 2
        # @show lo, hi
        @test lo <= true_incl_probs[i] <=  hi
    end

    d = size(samples, 2)

    # for d <= 6 we check all possible models
    enumerate_all = d <= 6
    obs_model_counts = StatsBase.countmap([.!iszero.(row) for row in eachrow(samples)])
    no_samples = size(samples, 1)
    obs_model_probs  = Dict{keytype(obs_model_counts), Float64}(k => v / no_samples for (k, v) in obs_model_counts)
    theoretical_probs = Dict{keytype(obs_model_counts), Float64}()
    if enumerate_all
        modelspace = Iterators.map(BitVector, Iterators.product((0:1 for _ in 1:d)...))
        for model in modelspace
            theoretical_probs[model] = pdf(spike_dist, model)
            get!(obs_model_probs, model, 0.0) # ensure all models are present
        end
    else
        for k in keys(obs_model_counts)
            theoretical_probs[k] = pdf(spike_dist, k)
        end
    end
    probs_observed = collect(values(obs_model_probs))
    probs_expected = [theoretical_probs[k] for k in keys(obs_model_probs)]

    obs_model_probs[falses(d)]
    theoretical_probs[falses(d)]
    obs_model_probs[trues(d)]
    theoretical_probs[trues(d)]

    # normalize to handle floating point artifacts
    probs_observed ./= sum(probs_observed)
    probs_expected ./= sum(probs_expected)

    # check if the two columns are close enough
    @test maximum(abs(probs_expected[i] - probs_observed[i]) for i in eachindex(probs_expected)) < 1e-2
    # check if the distribution is close enough
    @test isapprox(probs_expected, probs_observed, atol = 1e-2)

end

function test_approximation(samples, spike_dist::BetaBernoulliHierarchical)

    d = size(samples, 2)
    subsamples = view(samples, :, 1:d - 1)

    test_approximation(subsamples, spike_dist.d)

    # test θ
    θ = LogExpFunctions.logistic.(samples[:, end])
    θ_mean, θ_var = mean_and_var(θ)

    n_eff = MCMCDiagnosticTools.ess(θ)

    expected_dist = Beta(spike_dist.d.a, spike_dist.d.b)
    expected_mean = mean(expected_dist)
    expected_var  = var(expected_dist)

    @test isapprox(θ_mean, expected_mean, rtol=0.1, atol=0.1)
    @test isapprox(θ_var, expected_var,   rtol=0.2, atol=0.2)

    qprobs = .01:.01:.99
    k = length(qprobs)

    q_observed = quantile(θ, qprobs)
    q_expected = quantile(expected_dist, qprobs)

    n_eff = MCMCDiagnosticTools.ess(θ)

    if n_eff < 200
        isinteractive() && @info "Skipping θ (too few effective draws)" n_eff
        return nothing
    end

    dens = pdf.(expected_dist, q_expected)

    k = length(qprobs)
    Σ = zeros(k, k)
    for j in 1:k, r in j:k
        p, rprob = qprobs[j], qprobs[r]
        Σ[j,r] = (min(p,rprob) - p*rprob) / (n_eff * dens[j] * dens[r])
        Σ[r,j] = Σ[j,r]
    end

    # regularize
    ϵ = 1e-12 * maximum(view(Σ, diagind(Σ)))
    Σ += ϵ * I

    d = q_observed .- q_expected
    T = dot(d, Σ \ d)

    # compare to χ² quantile
    thresh = quantile(Chisq(k), 0.95)  # or 0.99
    @test T <= thresh


end

"""
    test_approximation(trace, D::SpikeAndSlabDist)

Test a sticky PDMP trace against a spike-and-slab target using analytic
trace estimators instead of discretization.

Uses the identity that for a spike-and-slab model:

    E[x_i]         = p_i * μ_i^slab       (full-model mean)
    mean(trace)[i] ≈ E[x_i]

so the conditional slab mean is recoverable as:

    mean(trace) ./ inclusion_probs(trace) ≈ μ_slab
"""
function test_approximation(trace, D::SpikeAndSlabDist; elapsed::Union{Real,Nothing}=nothing)

    d = length(D.slab_dist)

    min_ess = minimum(ess(trace))
    if min_ess < 500
        show_test_diagnostics && @info "Skipping test_approximation SpikeAndSlab (low ESS)" min_ess
        return nothing
    end

    mc = 1.0 / sqrt(min_ess)

    est_incl_probs = inclusion_probs(trace)
    true_incl_probs = mean(D.spike_dist)

    # Use a norm-based check: the per-coordinate `all(...)` test is too strict for d >= 5 because
    # the probability that at least one coordinate exceeds the tolerance grows with d.
    # A norm-based tolerance scales correctly: atol * sqrt(d).
    incl_atol_per = D.slab_dist isa Distributions.MvTDist ? (0.15 + 1.5 * mc) : (0.06 + 1.5 * mc)
    incl_atol = incl_atol_per * sqrt(d)

    @test isapprox(est_incl_probs, true_incl_probs; atol=incl_atol)

    est_mean = mean(trace)
    mean_rtol = 0.15 + 3.0 * mc
    mean_atol = 0.15 + 3.0 * mc

    true_full_mean = true_incl_probs .* mean(D.slab_dist)
    @test isapprox(est_mean[1:d], true_full_mean; rtol=mean_rtol, atol=mean_atol)

    if all(est_incl_probs .>= 0.1)
        est_slab_mean = est_mean[1:d] ./ est_incl_probs[1:d]
        @test isapprox(est_slab_mean, mean(D.slab_dist); rtol=mean_rtol, atol=mean_atol)
    end

    c_incl = _isapprox_closeness(est_incl_probs, true_incl_probs; atol=incl_atol)
    c_full = _isapprox_closeness(est_mean[1:d], true_full_mean; rtol=mean_rtol, atol=mean_atol)
    failed = c_incl > 1.0 || c_full > 1.0
    if failed || show_test_diagnostics
        label = failed ? "FAIL" : "ok  "
        dname = data_name(typeof(D), d)
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(dname, 40)) | ESS=$(lpad(round(Int, min_ess), 7)) | $(_format_elapsed(elapsed)) | c_incl=$(_f3(c_incl)) | c_full=$(_f3(c_full))")
    end
end
