import SpecialFunctions

"""
A Beta-Bernoulli distribution.

This is a multivariate binary distribution of length `n`.
The number of ones, `k`, in a sample follows a Beta-Binomial distribution
with parameters `n`, `a`, and `b`. The positions of the `k` ones are
uniformly distributed among the `n` positions.
"""
struct BetaBernoulli{T<:Real} <: DiscreteMultivariateDistribution
    n::Int
    a::T
    b::T

    function BetaBernoulli{T}(n::Int, a::T, b::T) where {T<:Real}
        if n <= 0
            throw(ArgumentError("n must be positive"))
        end
        if a <= 0 || b <= 0
            throw(ArgumentError("a and b must be positive"))
        end
        new{T}(n, a, b)
    end
end

BetaBernoulli(n::Int, a::Real, b::Real) = BetaBernoulli{promote_type(typeof(a), typeof(b))}(n, promote(a, b)...)

# Required methods for DiscreteMultivariateDistribution

Base.length(d::BetaBernoulli) = d.n
Base.eltype(::Type{<:BetaBernoulli}) = Int

Distributions.insupport(d::BetaBernoulli, x::AbstractVector{<:Integer}) = length(x) == length(d) && all(in((0, 1)), x)
Distributions.insupport(d::BetaBernoulli, x::BitVector) = length(x) == length(d)

function Distributions._rand!(rng::AbstractRNG, d::BetaBernoulli, x::AbstractVector{T}) where {T<:Integer}
    n = d.n
    k = rand(rng, BetaBinomial(n, d.a, d.b))
    fill!(x, zero(T))
    for i in 1:k
        x[i] = one(T)
    end
    shuffle!(rng, x)
    return x
end

function Distributions._logpdf(d::BetaBernoulli, x::AbstractVector{<:Integer})

    # should technically use oftype
    Distributions.insupport(d, x) || return -Inf

    k = sum(x)
    # The log-probability is the log-probability of getting k successes from the
    # BetaBinomial distribution, minus the log of the number of ways to arrange
    # these k successes among n trials.
    return logpdf(BetaBinomial(d.n, d.a, d.b), k) - SpecialFunctions.logabsbinomial(d.n, k)[1]
end

# Recommended statistical methods

function Distributions.mean(d::BetaBernoulli)
    p = d.a / (d.a + d.b)
    return fill(p, d.n)
end

function Distributions.var(d::BetaBernoulli)
    p = d.a / (d.a + d.b)
    # The variance of a single Bernoulli trial is p*(1-p)
    return fill(p * (1 - p), d.n)
end

"""
    BetaBernoulliKappa(a, b, mpdfs[, can_stick])

Callable struct for the sticky unfreezing rate κ under a BetaBernoulli
inclusion prior with parameters `a` and `b`.

`mpdfs[i]` is the marginal slab density at zero for coordinate `i`.

`can_stick` identifies the coordinates governed by the Beta--Bernoulli model
prior.  It defaults to every coordinate for backwards compatibility.  This
mask matters when a target also contains always-active parameters: those
parameters must not be counted as included model indicators.

Called as `κ(i, x, γ, θ...)` where `γ` is the free/frozen indicator.
"""
struct BetaBernoulliKappa{T<:AbstractVector{Float64},U<:AbstractVector{Bool}} <: Function
    a::Float64
    b::Float64
    mpdfs::T
    can_stick::U

    function BetaBernoulliKappa(a::Float64, b::Float64, mpdfs::T,
            can_stick::U) where {T<:AbstractVector{Float64},
                                U<:AbstractVector{Bool}}
        length(mpdfs) == length(can_stick) || throw(DimensionMismatch(
            "mpdfs and can_stick must have the same length"))
        new{T,U}(a, b, mpdfs, can_stick)
    end
end

function BetaBernoulliKappa(a::Real, b::Real,
        mpdfs::AbstractVector{<:Real}, can_stick::AbstractVector{Bool})
    length(mpdfs) == length(can_stick) || throw(DimensionMismatch(
        "mpdfs and can_stick must have the same length"))
    return BetaBernoulliKappa(Float64(a), Float64(b),
        Float64.(mpdfs), BitVector(can_stick))
end

BetaBernoulliKappa(a::Real, b::Real, mpdfs::AbstractVector{<:Real}) =
    BetaBernoulliKappa(a, b, mpdfs, trues(length(mpdfs)))

function (κ::BetaBernoulliKappa)(i::Integer, x, γ, args...)
    length(x) == length(κ.can_stick) || throw(DimensionMismatch(
        "state and can_stick vectors must have the same length"))
    length(γ) == length(κ.can_stick) || throw(DimensionMismatch(
        "free-state and can_stick vectors must have the same length"))
    κ.can_stick[i] || throw(ArgumentError(
        "coordinate $i is not governed by this BetaBernoulliKappa"))
    k_free = 0
    n_tot = 0
    @inbounds for j in eachindex(κ.can_stick, γ)
        if κ.can_stick[j]
            n_tot += 1
            k_free += γ[j]
        end
    end
    denom  = κ.b + n_tot - k_free - 1
    if denom <= 0
        return Inf
    end
    return (κ.a + k_free) / denom * κ.mpdfs[i]
end
