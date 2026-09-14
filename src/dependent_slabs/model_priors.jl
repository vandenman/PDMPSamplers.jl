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

"""
    ExchangeableModelSizePrior(log_omega; normalize=false)

Exchangeable model-size prior over sizes `0:p`. `log_omega[k + 1]` stores the
log prior mass for model size `k`. If `normalize=true`, the entries are shifted
by `logsumexp(log_omega)`.
"""
struct ExchangeableModelSizePrior <: AbstractModelPrior
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
