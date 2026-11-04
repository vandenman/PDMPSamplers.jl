# Getting started

## Installation

:::tabs

== Julia

```julia
using Pkg
Pkg.add(url="https://github.com/vandenman/PDMPSamplers.jl")
# For the Stan example:
Pkg.add("BridgeStan")
```

== R

```r
install.packages("remotes") # or renv, pak, etc.
remotes::install_github("vandenman/PDMPSamplersR")
# For the brms example:
install.packages("brms")
```

PDMPSamplersR needs Julia; see [Getting started in R](r/getting-started.md)
for setup and an R-focused walkthrough.

:::

## A first sampler

The main information that PDMPSamplers needs from a model is the gradient of the log posterior distribution.
This can be quite cumbersome to derive manually, so PDMPSamplers integrates with various other packages to specify a model in a more natural way.

To illustrate this, let's use the bivariate normal distribution as an example and draw samples using the Zig-Zag sampler.

We can do this either by providing the negative gradient directly, by writing a Stan model, or by using brms code.
Each tab is a complete example that samples from the same bivariate normal distribution.

:::tabs

== Julia - manual

```@example getting-started
using PDMPSamplers, LinearAlgebra, Statistics

mu = [1.0, -1.0]
Sigma = [1.0 0.5; 0.5 2.0]
precision = inv(Sigma)

# Negative gradient of the log density.
neg_gradient!(out, x) = mul!(out, precision, x - mu)
d = length(mu)
model = PDMPModel(d, FullGradient(neg_gradient!))

flow = ZigZag(d)
alg = GridThinningStrategy()
chains = pdmp_sample(d, flow, model, alg, 0.0, 10_000.0;
                     seed=42, progress=false)

# Most statistics work out of the box: mean, std, var, cov, cor.
(mean = mean(chains), cov = cov(chains))
```

== Julia - Stan

```@example getting-started-stan
using PDMPSamplers, BridgeStan, Statistics

stan_code = """
data {
  vector[2] mu;
  matrix[2, 2] Sigma;
}
parameters {
  vector[2] x;
}
model {
  x ~ multi_normal(mu, Sigma);
}
"""

dir = mktempdir()
stan_file = joinpath(dir, "normal.stan")
data_file = joinpath(dir, "data.json")
write(stan_file, stan_code)
write(data_file, """{"mu":[1,-1],"Sigma":[[1,0.5],[0.5,2]]}""")

model = PDMPModel(StanModel(stan_file, data_file))
d = model.d
flow = ZigZag(d)
alg = GridThinningStrategy()
chains = pdmp_sample(d, flow, model, alg,
                     0.0, 10_000.0; seed=42, progress=false)

(mean = mean(chains), cov = cov(chains))
```

== R - manual

```r
library(PDMPSamplersR)
pdmpsamplers_setup()  # one-time Julia setup

mu <- c(1, -1)
Sigma <- matrix(c(1, 0.5, 0.5, 2), 2, 2)
precision <- solve(Sigma)

# Negative gradient of the log density.
neg_gradient <- function(x) c(precision %*% (x - mu))

result <- pdmp_sample(
  neg_gradient, d = 2,
  flow = "ZigZag", algorithm = "GridThinningStrategy",
  T = 10000, seed = 42, show_progress = FALSE
)

# Most statistics work out of the box: mean, sd, var, cov, cor.
mean(result)
cov(result)
```

```@eval
Main.r_example_output("manual")
```

== R - Stan

```r
library(PDMPSamplersR)
pdmpsamplers_setup()  # one-time Julia setup

stan_code <- "
data {
  vector[2] mu;
  matrix[2, 2] Sigma;
}
parameters {
  vector[2] x;
}
model {
  x ~ multi_normal(mu, Sigma);
}
"

stan_file <- tempfile(fileext = ".stan")
writeLines(stan_code, stan_file)
stan_data <- list(
  mu = c(1, -1),
  Sigma = matrix(c(1, 0.5, 0.5, 2), 2, 2)
)

result <- pdmp_sample_from_stanmodel(
  stan_file, stan_data,
  flow = "ZigZag", algorithm = "GridThinningStrategy",
  T = 10000, seed = 42, show_progress = FALSE
)

mean(result)
cov(result)
```

```@eval
Main.r_example_output("stan")
```

== R - brms

```r
library(PDMPSamplersR)
library(brms)
pdmpsamplers_setup()  # one-time Julia setup

mu <- c(1, -1)
Sigma <- matrix(c(1, 0.5, 0.5, 2), 2, 2)
data <- data.frame(
  y = mu,
  x1 = c(1, 0),
  x2 = c(0, 1)
)

# Known covariance, fixed residual scale, and flat coefficient priors.
formula <- bf(y ~ 0 + x1 + x2 + fcor(Sigma), sigma = 1)
fit <- brm_pdmp(
  formula, data = data, data2 = list(Sigma = Sigma),
  family = gaussian(),
  flow = "ZigZag", algorithm = "GridThinningStrategy",
  T = 10000,
  seed = 42, show_progress = FALSE
)

fixef(fit)[, "Estimate"]
vcov(fit)
```

```@eval
Main.r_example_output("brms")
```

:::

All five examples target a mean of $[1, -1]^T$ and covariance matrix with variances 1 and 2, and covariance 0.5.[^brmsnormal]
Their estimates should be close to these values but vary due to sampling error.

## The parts of a call

The function `pdmp_sample(start, flow, model, alg, t0, T)` has four ingredients.

- **The model** gives the sampler the gradient of the log posterior.
  In Julia, `LogDensity` takes the log density and gets gradients by automatic differentiation.
  You can also pass a manually derived gradient by wrapping it as `FullGradient(f!)`.
  In that case, the function `f!` is called as `f!(out, x)` and should write the *negative* gradient of the log density, ``-\nabla \log \pi(x)``, into `out`.
  The R package uses the same convention.
- **The flow** is the dynamics: `ZigZag`, `BouncyParticle`, `Boomerang` and their variants. See [Choosing a sampler](choosing-a-sampler.md).
- **The algorithm** simulates the times at which the particle changes direction. `GridThinningStrategy()` works for most models.
- **The initial values.** Pass either the dimension `d` to start at a random point, or a vector `x0`.

`t0` is the starting time, and `T` is the length of the main run after warmup.
With a warmup of `t_warmup`, the main run starts at `t0 + t_warmup` and ends at `t0 + t_warmup + T`.
A PDMP runs in continuous time, so `T` is not a number of iterations; see [How long to run](#How-long-to-run) below.

## Reading the result

In Julia, `pdmp_sample` returns a `PDMPChains` object. For R and brms results, see [Reading R results](r/results.md).
These summaries are available directly from the continuous path:

```@example getting-started
(mean = mean(chains), var = var(chains), ess = ess(chains))
```

However, this does not work with many tools or more advanced functions, which do work with MCMC draws.
For this purpose, `PDMPDiscretize` reads off the position at regular times.
In the example below, the continuous trace is discretized every 10 time units, giving 1001 draws:

```@example getting-started
draws = Matrix(PDMPDiscretize(chains, 10.0))
size(draws)
```
These can be treated as one would treat regular MCMC samples.
For example, `vec(mean(draws; dims=1))` would estimate the posterior means, albeit in a less efficient way than `mean(chains)` which is aware of the continuous nature of the `PDMPChains` object.

With several chains (`pdmp_sample(...; n_chains=4)`), these summaries pool the chains; pass `chain=i` for a single chain.
`PDMPDiscretize` reads off one path, so it needs `chain=i` when there are several.

`chains[1]` returns the path (`trace`) and run statistics (`stats`) of the first chain, for functions that need them:

```@example getting-started
trace, stats = chains[1]
stats.reflections_events     # number of direction changes
```

## How long to run

How far the particle moves in one time unit depends on the dynamics and on the scale of the posterior.
The Zig-Zag moves at speed 1 in every coordinate, so a posterior with standard deviations of about 1 takes a few time units to cross, and one with standard deviations of 0.01 takes a few hundredths.
Run a short pilot, look at the effective sample size `ess(chains)`, and scale `T` up until the ESS is large enough.

When the starting point is far from the posterior, run a warmup first.
The warmup time is the 7th argument. The warmup runs first and the main run of length `T` follows it: below, the warmup covers time 0 to 1 000 and the main run time 1 000 to 11 000.
Only the main run is used in the summaries.
Adaptive samplers such as `AdaptiveBoomerang` also tune themselves during the warmup time:

```@example getting-started
chains = pdmp_sample(d, flow, model, alg, 0.0, 10_000.0, 1_000.0; progress=false)
mean(chains)
```

## What's next

- [What are PDMPs?](what-are-pdmps.md): the idea behind the samplers, with pictures.
- [Variable selection](variable-selection.md): spike-and-slab priors with sticky samplers.
- [Subsampling](subsampling.md): large data sets with batches of observations.
- [Choosing a sampler](choosing-a-sampler.md): which dynamics, strategy and gradients to use.
- [Stan models](julia/stan.md) and [Turing.jl models](julia/turing.md):
  sample models written in either framework.
- [Using R](r.md): setup, Stan and brms models, and results.
- [API reference](api.md): all exported types and functions.

[^brmsnormal]: In the brms model, `x1` and `x2` each select one coefficient.
    With flat coefficient priors, the known covariance supplied by [`fcor`](https://paulbuerkner.com/brms/reference/fcor.html), and `sigma = 1`, the coefficient posterior is exactly this bivariate normal.
