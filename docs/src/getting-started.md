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

:::

## Your first sampler

This example samples from a bivariate normal distribution using
the ZigZag sampler.

:::tabs

== Julia

```@example getting-started
using PDMPSamplers, Distributions, LinearAlgebra, ADTypes
import ForwardDiff

# Define the target: a bivariate normal
d = 2
target = MvNormal(d, 1.0)

# Build the model from a log-density with automatic differentiation
model = PDMPModel(d, LogDensity(x -> logpdf(target, x)), ADTypes.AutoForwardDiff(), true)

# Choose dynamics and thinning strategy
flow = ZigZag(d)
alg = GridThinningStrategy()

# Sample for T = 10 000 time units
chains = pdmp_sample(d, flow, model, alg, 0.0, 10_000.0; progress=false)

# Posterior summary
mean(chains)
```

== R

```r
library(PDMPSamplersR)

result <- pdmp_sample(
    n_samples = 10000,
    flow      = "ZigZag",
    algorithm = "GridThinningStrategy"
)
mean(result)
```

:::

The `LogDensity` wrapper tells PDMPSamplers that you are providing a
log-density function (not a gradient). The package then uses
[DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)
to obtain gradients and Hessian-vector products automatically.

## What's next

- [API Reference](api.md) — full list of exported types and functions.
- [R Package](r.md) — R-specific documentation and vignettes.
