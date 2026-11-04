# Reading R results

`pdmp_sample()` and `pdmp_sample_from_stanmodel()` return a `pdmp_result`. It
contains a continuous-time path for each chain. The summary functions
integrate over that path, rather than first choosing regularly spaced draws:

```r
mean(result)
var(result)
cov(result)
quantile(result, probs = c(0.025, 0.5, 0.975))
ess(result)
```

With several chains, these estimators pool their trajectories by run length by
default. Pass `chain = 1` (or another chain number) to inspect a single chain.
For a sticky sampler, `inclusion_probs(result)` reports the fraction of time
each coefficient spent active. See [Variable selection](../variable-selection.md)
for its interpretation.

## Draws for plotting and other tools

`discretize()` reads positions at regular times and returns a matrix with one
row per draw. A path belongs to one chain, so specify `chain` when the result
has multiple chains:

```r
draws <- discretize(result, dt = 1, chain = 1)
head(draws)
```

If `dt` is omitted, the package selects a spacing based on the
continuous-time effective sample size. Discretized draws are useful for plots
and software that expects a matrix; use the path-based estimators above for
posterior summaries.

## Constrained parameters

Stan samples on an unconstrained scale. For example, a positive parameter may
appear as its logarithm in the path. Supply the corresponding transform to
the summary functions, or transform discretized draws before interpreting
them. Use `stan_parameter_mapping(stan_file, standata)` to check the
unconstrained coordinate order before assigning names or transforms. The
[result function reference](reference/results.md) documents the arguments for
`mean.pdmp_result`, `discretize`, and the other summaries.

`brm_pdmp()` returns a `brmsfit` instead of a `pdmp_result`; use brms methods
such as `summary(fit)` and `fixef(fit)` for that result.
