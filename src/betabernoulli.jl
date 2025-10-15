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
