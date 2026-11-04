# brms models

`brm_pdmp()` uses PDMPSamplers.jl as a backend for a brms model. It returns a
`brmsfit`, so familiar brms summaries and plots work on its discretized draws.
Install brms and complete the [R setup](getting-started.md) before running it.

```r
library(PDMPSamplersR)
library(brms)

x <- seq(-2, 2, length.out = 100)
data <- data.frame(x = x, y = 1 + 0.8 * x + 0.5 * sin(seq_along(x)))
priors <- prior(normal(0, 5), class = "Intercept") +
  prior(normal(0, 5), class = "b") +
  prior(normal(0, 5), class = "sigma")

fit <- brm_pdmp(
  y ~ x, data = data, family = gaussian(),
  prior = priors,
  flow = "AdaptiveBoomerang", algorithm = "GridThinningStrategy",
  T = 10000, t_warmup = 2000, n_chains = 2,
  seed = 42, show_progress = FALSE
)
summary(fit)
```

The warmup lets `AdaptiveBoomerang` learn a reference distribution. `T` is
the length of the subsequent main run. `brm_pdmp()` discretizes the retained
trajectories to populate the `brmsfit`; the time step can be set with
`discretize_dt`.

The R result supports brms functions such as `fixef()`, `plot()`, and
`conditional_effects()`. PDMP sampling has different diagnostics from NUTS:
with one chain, `Rhat` is unavailable, and NUTS divergence and acceptance
diagnostics do not describe this sampler. Use multiple chains and review the
posterior summaries and plots. For Julia models, see the separate [Turing.jl guide](../julia/turing.md).

See [Choosing a sampler](../choosing-a-sampler.md) for the dynamics and
[Reading R results](results.md) for results returned by the lower-level R
functions.

```@raw html
<p>The <a href="../r-api/reference/brm_pdmp.html" target="_self">R function reference</a>
lists all brms backend options.</p>
```

The same help is available in R with
`?PDMPSamplersR::brm_pdmp`.
