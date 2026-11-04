# Stan models from R

`pdmp_sample_from_stanmodel()` takes Stan source or an already compiled Stan
model, plus data as a list or a JSON file. PDMPSamplersR compiles Stan source
through BridgeStan and runs the chosen PDMP sampler.

The example below fits a normal mean with known observation scale. It avoids
parameter constraints so the first result is easy to interpret:

```r
library(PDMPSamplersR)

stan_code <- "
data { int<lower=1> N; vector[N] y; }
parameters { real mu; }
model { mu ~ normal(0, 2); y ~ normal(mu, 1); }
"
stan_file <- tempfile(fileext = ".stan")
writeLines(stan_code, stan_file)
stan_data <- list(N = 5L, y = c(1.2, 1.5, 1.8, 1.1, 1.4))

result <- pdmp_sample_from_stanmodel(
  stan_file, stan_data,
  flow = "ZigZag", algorithm = "GridThinningStrategy",
  T = 10000, seed = 42, show_progress = FALSE
)
mean(result)
stan_parameter_mapping(stan_file, stan_data)
```

Stan runs on its unconstrained parameter scale. If a parameter has a
constraint, such as a positive standard deviation, use
`stan_parameter_mapping(stan_file, stan_data)` to check the coordinate order,
then apply the appropriate transform before interpreting its summary. See
[Reading R results](results.md) for transforms and draws.

For spike-and-slab selection, Stan models can be combined with `sticky = TRUE`
and a model prior. Correlated slabs use `slab_prior = dense_gaussian_slab(...)`.
Start with [Variable selection](../variable-selection.md) and
[Dependent slabs](../dependent-slabs.md) for the statistical setup.
For large data sets, see [Subsampling](../subsampling.md).

```@raw html
<p>The <a href="../r-api/reference/pdmp_sample_from_stanmodel.html" target="_self">R function reference</a>
lists all options for <code>pdmp_sample_from_stanmodel</code>, including custom Stan subsampling.</p>
```

The same help is available in R with
`?PDMPSamplersR::pdmp_sample_from_stanmodel`.

For paired R and Julia examples, continue with [Stan targets and variable
selection](../examples/stan-workflows.md), the [RI-CLPM](../examples/riclpm.md),
or [ordinal MRF subsampling](../examples/omrf.md).
