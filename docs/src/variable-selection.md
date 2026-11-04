# Variable selection with sticky samplers

This tutorial selects predictors in a linear regression with a spike-and-slab
prior. Each coefficient is either exactly zero (the spike) or drawn from a
normal distribution (the slab). The sampler explores which coefficients are
nonzero and what their values are in a single run.

## How sticky samplers work

A sticky PDMP moves like an ordinary PDMP until a coefficient reaches zero.
That coefficient then *sticks* at zero for an exponentially distributed time,
and while it is stuck the model excludes it. When it is released it continues
with the velocity it had. The rate of release, ``\kappa_i``, sets the balance
between the spike and the slab. For independent priors it is

```math
\kappa_i = \frac{p_i}{1 - p_i}\, \pi_i(0),
```

where ``p_i`` is the prior inclusion probability of coefficient ``i`` and
``\pi_i(0)`` is the slab density at zero. You don't propose models. The
sampler moves between them by freezing and releasing coefficients.

## Data

Six predictors, three of which have an effect.

```@example vs
using PDMPSamplers, Distributions, LinearAlgebra, Random

Random.seed!(2025)
n, d = 200, 6
X = randn(n, d)
β_true = [1.0, -0.8, 0.0, 0.0, 0.4, 0.0]
y = X * β_true + randn(n)
nothing # hide
```

## Model

The residual variance is fixed at 1 to keep the example short. Each
coefficient has slab ``\mathcal{N}(0, \tau^2)`` with ``\tau = 1`` and prior
inclusion probability ``0.5``.

The sampler needs the negative gradient of the log posterior *with every
coefficient included*: the likelihood times the slab densities. It handles
the spike itself. `GridThinningStrategy` also uses a Hessian-vector product.

```@example vs
τ = 1.0
p = fill(0.5, d)

XtX, Xty = X' * X, X' * y
H = XtX + I / τ^2                  # negative Hessian of the log posterior

neg_gradient!(out, β) = (out .= H * β .- Xty; out)
neg_hvp!(out, β, v)   = mul!(out, H, v)

model = PDMPModel(d, FullGradient(neg_gradient!), neg_hvp!)
nothing # hide
```

## Sampler

Wrap any Poisson-time strategy in `Sticky` and pass the release rates:

```@example vs
κ = p ./ (1 .- p) .* pdf(Normal(0, τ), 0.0)
alg = Sticky(GridThinningStrategy(), κ)
flow = ZigZag(d)

chains = pdmp_sample(zeros(d), flow, model, alg, 0.0, 20_000.0; progress=false)
nothing # hide
```

`Sticky(alg, κ)` lets every coordinate with finite ``\kappa_i`` stick. To keep
a coefficient always in the model (an intercept, say), set its ``\kappa_i`` to
`Inf`, or pass a mask: `Sticky(alg, κ, can_stick)`.

## Posterior inclusion probabilities

The posterior inclusion probability of a coefficient is the fraction of time it
spent away from zero. `inclusion_probs` integrates this along the trajectory:

```@example vs
round.(inclusion_probs(chains); digits=3)
```

Coefficients 1, 2 and 5 are always included. The others are included only part
of the time. The posterior mean averages over the models, so the zeros pull it
toward zero. Divide by the inclusion probability to get the mean of the
coefficient when it is in the model:

```@example vs
model_averaged = mean(chains)
conditional    = model_averaged ./ inclusion_probs(chains)
round.([model_averaged conditional]; digits=3)
```

## Reading the path

The result is a continuous path. To get draws at regular times, for
plotting or for code that expects samples, discretize it:

```@example vs
samples = Matrix(PDMPDiscretize(chains, 1.0))   # one row per time unit
size(samples)
```

A coefficient that is exactly zero in a row is excluded from the model at that
time. Tabulating the rows gives the posterior probabilities of the models:

```@example vs
models = [Tuple(Int.(row .!= 0)) for row in eachrow(samples)]
counts = Dict{eltype(models),Int}()
for m in models
    counts[m] = get(counts, m, 0) + 1
end
top = sort(collect(counts); by=last, rev=true)[1:4]
[(model=m, prob=round(c / length(models); digits=3)) for (m, c) in top]
```

The returned statistics count how often coordinates froze and were released:

```@example vs
trace, stats = chains[1]
(stats.sticky_freezes, stats.sticky_unfreezes)
```

## Checking the answer

With a known residual variance and normal slabs, the marginal likelihood of
every model has a closed form, and six predictors give only ``2^6 = 64``
models. Enumerating them gives the exact inclusion probabilities:

```@example vs
function log_marginal(γ)
    idx = findall(γ)
    isempty(idx) && return 0.0
    Hγ = XtX[idx, idx] + I / τ^2
    b = Xty[idx]
    return 0.5 * dot(b, Hγ \ b) - 0.5 * logdet(Hγ) - length(idx) * log(τ)
end

all_models = [BitVector(digits(m; base=2, pad=d)) for m in 0:2^d-1]
logw = [log_marginal(γ) + sum(log.(ifelse.(γ, p, 1 .- p))) for γ in all_models]
w = exp.(logw .- maximum(logw))
w ./= sum(w)
exact = sum(w .* all_models)

round.([exact inclusion_probs(chains)]; digits=3)
```

The two columns agree to within Monte Carlo error.

## Next steps

- Correlated slab priors and beta-binomial model priors are covered in
  [Dependent Slabs](dependent-slabs.md).
- Other dynamics work the same way; replace `ZigZag(d)` with, for example,
  `AdaptiveBoomerang(d)`. See [Choosing a sampler](choosing-a-sampler.md).
