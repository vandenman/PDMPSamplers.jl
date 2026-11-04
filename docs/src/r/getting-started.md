# Getting started in R

PDMPSamplersR uses Julia to run the samplers. Install Julia first, then install
the R package from GitHub and prepare its Julia environment:

```r
install.packages("remotes")
remotes::install_github("vandenman/PDMPSamplersR")

library(PDMPSamplersR)
pdmpsamplers_setup()  # once per Julia environment; installs and precompiles dependencies
```

The first setup can take several minutes because it installs Julia packages.
Later R sessions can load `PDMPSamplersR` and sample without repeating setup.

## Your first sampler

This example targets a bivariate normal distribution. `pdmp_sample()` expects
a function that returns the **negative gradient of the log density**. For a
normal target with mean `mu` and covariance `Sigma`, that is
`solve(Sigma) %*% (x - mu)`.

```r
mu <- c(1, -1)
Sigma <- matrix(c(1, 0.5, 0.5, 2), 2, 2)
Sigma_inv <- solve(Sigma)
neg_gradient <- function(x) drop(Sigma_inv %*% (x - mu))

result <- pdmp_sample(
  neg_gradient, d = 2,
  flow = "ZigZag", algorithm = "GridThinningStrategy",
  T = 10000, seed = 42, show_progress = FALSE
)
mean(result)
```

`T` is the length of the main run in continuous-time units, not a number of
draws. Set `t_warmup` when you need a separate warmup interval; only the main
run contributes to the summaries. See [Reading R results](results.md) for
other estimators and discretized draws.

You can also provide a Stan model instead of an R gradient function. See
[Stan models](stan.md), or use [brms](brms.md) to work with a familiar
`brmsfit` result. The [sampler guide](../choosing-a-sampler.md) explains the
flow and algorithm choices.
