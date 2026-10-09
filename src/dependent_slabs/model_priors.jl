"""
    BernoulliModelPrior(prob)

Independent Bernoulli model prior for beta inclusion indicators. `prob[j]` is
the prior inclusion probability for beta coordinate `j`; `log_model_add_odds`
returns `log(prob[j] / (1 - prob[j]))`.
"""
struct BernoulliModelPrior{P<:AbstractVector{<:Real}} <: AbstractModelPrior
    prob::P
    function BernoulliModelPrior(prob::P) where {P<:AbstractVector{<:Real}}
        isempty(prob) && throw(ArgumentError("prob must be non-empty"))
        all(p -> zero(p) <= p <= one(p), prob) || throw(ArgumentError("Bernoulli probabilities must be in [0, 1]"))
        new{P}(prob)
    end
end

Base.length(prior::BernoulliModelPrior) = length(prior.prob)
Base.copy(prior::BernoulliModelPrior) = BernoulliModelPrior(copy(prior.prob))

"""
    log_model_add_odds(prior, active, j)

Return the log prior odds for adding inactive beta coordinate `j` to the active
model `active`. The index `j` is in beta-block coordinates, not full-state
coordinates.
"""
function log_model_add_odds(prior::BernoulliModelPrior, active::BitVector, j::Integer)
    1 <= j <= length(prior.prob) || throw(BoundsError(prior.prob, j))
    length(active) == length(prior.prob) || throw(DimensionMismatch("active set length $(length(active)) does not match prior length $(length(prior.prob))"))
    p = prior.prob[j]
    iszero(p) && return -Inf
    isone(p) && return Inf
    return log(p) - log1p(-p)
end

"""
    BetaBernoulliModelPrior(n, a, b)

Exchangeable beta-Bernoulli model prior for `n` beta coordinates with
inclusion-probability hyperprior `Beta(a, b)`.
"""
struct BetaBernoulliModelPrior <: AbstractModelPrior
    n::Int
    a::Float64
    b::Float64
    function BetaBernoulliModelPrior(n::Integer, a::Real, b::Real)
        n > 0 || throw(ArgumentError("n must be positive"))
        a > 0 || throw(ArgumentError("a must be positive"))
        b > 0 || throw(ArgumentError("b must be positive"))
        new(Int(n), Float64(a), Float64(b))
    end
end

Base.length(prior::BetaBernoulliModelPrior) = prior.n

function log_model_add_odds(prior::BetaBernoulliModelPrior, active::BitVector, j::Integer)
    1 <= j <= prior.n || throw(BoundsError(active, j))
    length(active) == prior.n || throw(DimensionMismatch("active set length $(length(active)) does not match prior length $(prior.n)"))
    active[j] && throw(ArgumentError("log_model_add_odds is defined for adding an inactive coordinate; coordinate $j is already active"))
    k = count(active)
    denom = prior.b + prior.n - k - 1
    denom <= 0 && return Inf
    return log(prior.a + k) - log(denom)
end
