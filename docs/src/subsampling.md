# Subsampling

The [ordinal MRF example](examples/omrf.md) pairs a custom Stan subsampling
workflow in R with a native Julia person-level control variate.

With ``N`` observations, the gradient of the log posterior is a sum of ``N``
terms, and most samplers evaluate all of them at every step. A PDMP only uses
the gradient to decide *whether* an event happens. That decision stays exact
if the gradient is replaced by an unbiased estimate from a small random batch
of observations, provided the rate bound used for thinning covers every
observation. Each event then costs ``m \ll N`` observations, and the sampler
still targets the exact posterior.

This page fits a logistic regression with 100 000 observations, using batches
of 10.

## Data

```@example sub
using PDMPSamplers, LinearAlgebra, Random, Statistics

logistic(η) = 1 / (1 + exp(-η))

Random.seed!(1)
N, d = 100_000, 5
X = [ones(N) randn(N, d - 1)]               # intercept and four predictors
β_true = [-0.5, 1.0, -0.5, 0.25, 0.0]
y = Float64.(rand(N) .< logistic.(X * β_true))
prior_precision = 0.01                       # β ~ N(0, 10²)
nothing # hide
```

## The control variate

Subsampling needs a reference point ``\hat\beta`` (the *anchor*) close to the
posterior mode. We find the mode with a few Newton steps:

```@example sub
function full_gradient!(out, β)               # negative log posterior gradient
    mul!(out, X', logistic.(X * β) .- y)
    out .+= prior_precision .* β
end

β̂ = zeros(d)
g = zeros(d)
for _ in 1:10
    full_gradient!(g, β̂)
    p = logistic.(X * β̂)
    H = X' * (p .* (1 .- p) .* X) + prior_precision * I
    β̂ .-= H \ g
end
β̂
```

The gradient is split into a part that is cheap to evaluate exactly and a sum
of small residuals, one per observation:

```math
\nabla U(\beta) = \underbrace{\nabla U_{\text{lik}}(\hat\beta) + \tau\beta}_{\text{deterministic}}
  + \sum_{i=1}^N \underbrace{\bigl(\sigma(x_i^\top\beta) - \sigma(x_i^\top\hat\beta)\bigr)\, x_i}_{r_i(\beta)},
```

where ``\sigma`` is the logistic function and ``\tau`` the prior precision.
The sampler estimates the sum from a random batch. Near the mode the
residuals are small, which keeps the estimate precise.

```@example sub
anchor = copy(β̂)
anchor_gradient = X' * (logistic.(X * anchor) .- y)

deterministic!(out, β) = (out .= anchor_gradient .+ prior_precision .* β)

# Unscaled sum of the residuals of the observations in `subset`
function residuals!(out, β, subset, anchor)
    fill!(out, 0.0)
    for i in subset
        xi = view(X, i, :)
        out .+= (logistic(dot(xi, β)) - logistic(dot(xi, anchor))) .* xi
    end
    return out
end
nothing # hide
```

## The envelope

For thinning, the sampler needs a bound on each observation's contribution to
the event rate along the path. The logistic function has slope at most
``1/4``, so for the Bouncy Particle sampler

```math
|\langle v, r_i(\beta)\rangle| \le \tfrac14 \|x_i\|^2 \cdot \|v\|\, \|\beta - \hat\beta\|.
```

Along the path ``\beta + vt``, the last factor is at most
``\|\beta - \hat\beta\| + \|v\| t``, which is affine in ``t``. A
`SeparableResidualEnvelope` takes the observation weights
``\tfrac14\|x_i\|^2`` and this common scale. It also draws the batch with
probabilities proportional to the weights.

```@example sub
weights = reshape([0.25 * sum(abs2, view(X, i, :)) for i in 1:N], 1, :)

envelope = SeparableResidualEnvelope(weights,
    (out, state, flow, t) -> begin
        speed = norm(state.ξ.θ)
        out[1] = speed * (norm(state.ξ.x - anchor) + speed * t)
    end; certified_affine=true)

cv = SubsampledControlVariate(deterministic!, residuals!, envelope, anchor, 10;
    deterministic_hvp! = (out, β, v) -> (out .= prior_precision .* v))
model = PDMPModel(d, cv)
nothing # hide
```

## Sampling

```@example sub
flow = BouncyParticle(d, 1.0)
chains = pdmp_sample(copy(anchor), flow, model, GridThinningStrategy(), 0.0, 50.0;
    progress=false)
round.(ess(chains))
```

The posterior is concentrated (standard deviations near 0.008), so 50 time
units are many crossings of it.

The sampler never evaluated the full gradient. It evaluated residuals for
batches of 10 observations:

```@example sub
trace, stats = chains[1]
(full_gradients = stats.full_gradient_calls,
 observations_per_batch = 10,
 residual_batches = stats.residual_oracle_calls)
```

## Checking the answer

With this many observations the posterior is close to normal, so the Laplace
approximation (the normal at the mode, with covariance the inverse of the
Hessian) is an accurate reference:

```@example sub
p = logistic.(X * β̂)
laplace_sd = sqrt.(diag(inv(X' * (p .* (1 .- p) .* X) + prior_precision * I)))
round.([β̂ mean(chains) laplace_sd sqrt.(var(chains))]; digits=4)
```

The columns are the mode, the posterior mean, and the standard deviations from
the Laplace approximation and from the sampler.

## How much faster is it?

The same model fitted three ways on one machine (a 14-core Linux desktop). The table shows the smallest effective sample size over
the five coefficients, per second of sampling:

| Sampler | Gradient | min ESS / second |
| --- | --- | ---: |
| Stan NUTS (CmdStan 2.39, `bernoulli_logit_glm`) | full | 14 |
| Bouncy Particle | full | 26 |
| Bouncy Particle | batches of 10 | 516 |

All three gave the same posterior means and standard deviations. The numbers
are from single runs and depend on the hardware. Stan's figure excludes
warmup; the PDMPs started at the mode found above, which took well under a
second. Stan and PDMPSamplers estimate the ESS differently, so treat the
comparison as indicative. The advantage grows with ``N``: the cost of a
full-gradient sampler grows linearly in ``N``, and that of the subsampled
sampler much more slowly.

::: details Stan model used for the comparison

```stan
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[K] beta;
}
model {
  beta ~ normal(0, 10);
  y ~ bernoulli_logit_glm(X, 0, beta);
}
```

1000 warmup and 1000 sampling iterations, default settings.

:::

## From R

In R, `brm_pdmp()` builds the control variate and envelope for you. It does
this for Bernoulli and binomial regressions with a logit link, Poisson with a
log link, Gaussian, and categorical models. Set `subsample_size`:

```r
library(PDMPSamplersR)
library(brms)

set.seed(1)
N <- 100000
dat <- data.frame(x1 = rnorm(N), x2 = rnorm(N), x3 = rnorm(N), x4 = rnorm(N))
dat$y <- rbinom(N, 1, plogis(-0.5 + dat$x1 - 0.5 * dat$x2 + 0.25 * dat$x3))

fit <- brm_pdmp(
  y ~ x1 + x2 + x3 + x4, data = dat, family = brms::bernoulli(),
  prior = prior(normal(0, 10), class = "b"),
  flow = "BouncyParticle",
  subsample_size = 10,
  T = 50, t_warmup = 10, n_anchor_updates = 5
)
fit
```

During warmup the anchor is moved `n_anchor_updates` times toward the region
the sampler visits.

## When to subsample

Subsampling pays off when ``N`` is large, the posterior is concentrated (so
the residuals stay small), and one observation's gradient is cheap. For small
data sets, the cost of the control variate and the looser rate bound outweigh
the savings. For models other than these regressions you write the residuals
and envelope yourself, as above; the [Extension interface](extension-interface.md)
lists the functions to implement for custom envelopes.
