# Getting started

## Installation

:::tabs

== Julia

```julia
using Pkg
Pkg.add(url="https://github.com/vandenman/PDMPSamplers.jl")
```

== R

```r
remotes::install_github("vandenman/PDMPSamplersR")
```

PDMPSamplersR needs Julia; see [Getting started in R](r/getting-started.md)
for setup and an R-focused walkthrough.

:::

## Your first sampler

This example samples from a bivariate normal distribution with the Zig-Zag
sampler.

:::tabs

== Julia

```@example getting-started
using PDMPSamplers, Distributions, ADTypes
import ForwardDiff

d = 2
target = MvNormal([1.0, -1.0], [1.0 0.5; 0.5 2.0])

# The model: the log density, differentiated with ForwardDiff.
# `true` also prepares Hessian-vector products, which GridThinningStrategy uses.
model = PDMPModel(d, LogDensity(x -> logpdf(target, x)), ADTypes.AutoForwardDiff(), true)

flow = ZigZag(d)                 # the dynamics
alg  = GridThinningStrategy()    # how event times are simulated

# Start at a random point and run from time 0 to time 10 000
chains = pdmp_sample(d, flow, model, alg, 0.0, 10_000.0; progress=false)

mean(chains)
```

== R

```r
library(PDMPSamplersR)
pdmpsamplers_setup()   # one time: sets up Julia and the Julia packages

# Negative gradient of the log density of the same bivariate normal
mu    <- c(1, -1)
Sigma <- matrix(c(1, 0.5, 0.5, 2), 2, 2)
Sigma_inv <- solve(Sigma)
neg_gradient <- function(x) drop(Sigma_inv %*% (x - mu))

result <- pdmp_sample(
    neg_gradient, d = 2,
    flow      = "ZigZag",
    algorithm = "GridThinningStrategy",
    T         = 10000
)
mean(result)
```

:::

## The parts of a call

`pdmp_sample(start, flow, model, alg, t0, T)` has four ingredients.

- **The model** gives the sampler the gradient of the log posterior.
  `LogDensity` takes the log density and gets gradients by automatic
  differentiation. You can also pass the gradient yourself with
  `FullGradient(f!)`. In that case `f!(out, x)` must write the *negative*
  gradient of the log density, ``-\nabla \log \pi(x)``, into `out`. The R
  package uses the same convention.
- **The flow** is the dynamics: `ZigZag`, `BouncyParticle`, `Boomerang` and
  their variants. See [Choosing a sampler](choosing-a-sampler.md).
- **The algorithm** simulates the times at which the particle changes
  direction. `GridThinningStrategy()` works for most models.
- **The start.** Pass the dimension `d` to start at a random point, or a
  vector `x0` to start there.

`t0` and `T` are the start and end *times* of the run. A PDMP runs in
continuous time, so `T` is not a number of iterations; see
[How long to run](#How-long-to-run) below.

## Reading the result

`pdmp_sample` returns a `PDMPChains` object. Summaries are computed exactly
from the continuous path:

```@example getting-started
(mean = mean(chains), var = var(chains), ess = ess(chains))
```

Many tools expect draws. `PDMPDiscretize` reads off the position at regular
times; here every 10 time units, giving 1001 draws:

```@example getting-started
draws = Matrix(PDMPDiscretize(chains, 10.0))
size(draws)
```

With several chains (`pdmp_sample(...; n_chains=4)`), these summaries pool
the chains; pass `chain=i` for a single chain. `PDMPDiscretize` reads off one
path, so it needs `chain=i` when there are several.

`chains[1]` returns the path (`trace`) and run statistics (`stats`) of the
first chain, for functions that need them:

```@example getting-started
trace, stats = chains[1]
stats.reflections_events     # number of direction changes
```

## How long to run

How far the particle moves in one time unit depends on the dynamics and on
the scale of the posterior. The Zig-Zag moves at speed 1 in every coordinate,
so a posterior with standard deviations of about 1 takes a few time units to
cross, and one with standard deviations of 0.01 takes a few hundredths. Run a
short pilot, look at the effective sample size `ess(chains)`, and scale `T` up
until the ESS is large enough.

When the starting point is far from the posterior, run a warmup first. The
warmup time is the 7th argument. The warmup runs first and the main run of
length `T` follows it: below, the warmup covers time 0 to 1 000 and the main
run time 1 000 to 11 000. Only the main run is used in the summaries. Adaptive samplers such as
`AdaptiveBoomerang` also tune themselves during it:

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
