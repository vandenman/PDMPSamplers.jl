# Stan targets and variable selection

These examples pair the workflows from the original PDMPSamplersR Stan
vignette with Julia calls to PDMPSamplers.jl. They cover a multivariate normal,
two event-time strategies, independent and correlated spike-and-slab priors,
logistic regression, and a Bayesian two-sample test. The paired Stan examples use the same model and data in R and Julia.
The correlated-slab example uses the language-specific dependent-slab interfaces.

The examples assume [R setup](../r/getting-started.md) or [Julia
installation](../getting-started.md). A run length `T` is continuous time;
increase it and check `ess` for a substantive analysis.

## A multivariate normal: Zig-Zag and two event clocks

The model is a bivariate normal with known mean and covariance. `mean` and
`cov` integrate along the path. Changing the event-time strategy leaves the
target unchanged.

:::tabs

== R

```r
library(PDMPSamplersR)
model_file <- system.file("stan", "models", "mvnormal.stan",
                          package = "PDMPSamplersR")
mu <- c(1, -1)
Sigma <- matrix(c(1, 0.5, 0.5, 2), 2, 2)
data <- list(N = 2L, mu = mu, sigma = Sigma)

grid <- pdmp_sample_from_stanmodel(
  model_file, data, flow = "ZigZag", algorithm = "GridThinningStrategy",
  T = 10000, seed = 42, show_progress = FALSE
)
mean(grid)
cov(grid)

# An alternative clock with a user-supplied curvature bound.
bounded <- pdmp_sample_from_stanmodel(
  model_file, data, flow = "ZigZag", algorithm = "ThinningStrategy",
  c0 = 5, T = 10000, seed = 43, show_progress = FALSE
)
mean(bounded)
```

== Julia

```julia
using PDMPSamplers, BridgeStan, Statistics

stan_code = """
data { int<lower=1> N; vector[N] mu; matrix[N,N] sigma; }
parameters { vector[N] x; }
model { x ~ multi_normal(mu, sigma); }
"""
dir = mktempdir()
model_file = joinpath(dir, "mvnormal.stan")
data_file = joinpath(dir, "mvnormal.json")
write(model_file, stan_code)
write(data_file, """{"N":2,"mu":[1,-1],"sigma":[[1,0.5],[0.5,2]]}""")

model = PDMPModel(StanModel(model_file, data_file))
d = model.d
grid = pdmp_sample(d, ZigZag(d), model, GridThinningStrategy(),
                   0.0, 10_000.0; seed=42, progress=false)
mean(grid)
cov(grid)

bounded = pdmp_sample(d, ZigZag(d), model,
                      ThinningStrategy(GlobalBounds(5.0, d)),
                      0.0, 10_000.0; seed=43, progress=false)
mean(bounded)
```

:::

For a larger normal target, replace `mu` and `Sigma` with vectors and matrices
of the desired dimension. The same summary calls compare the estimate with
the known target; plots can put estimated means and covariance entries against
their true values.

## Independent spike-and-slab prior

Take independent normal slabs and Bernoulli inclusion indicators. If the slab
density at zero is ``s_i(0)`` and prior inclusion probability is ``p_i``, the
release rate is ``\kappa_i=p_i s_i(0)/(1-p_i)``. The Stan model supplies the
all-active normal density; the sticky sampler supplies the spikes.

```@raw html
<p>For Julia, download <a href="../stan/mvnormal.stan" download>the Stan model</a>
to your working directory. The R package already includes this model.</p>
```

:::tabs

== R

```r
library(PDMPSamplersR)
model_file <- system.file("stan", "models", "mvnormal.stan",
                          package = "PDMPSamplersR")
mu <- c(0.7, 0, -0.4)
slab_sd <- c(1, 1, 1)
data <- list(N = 3L, mu = mu, sigma = diag(slab_sd^2))
p <- c(0.7, 0.3, 0.5)

fit <- pdmp_sample_from_stanmodel(
  model_file, data, flow = "ZigZag", algorithm = "GridThinningStrategy",
  sticky = TRUE, can_stick = rep(TRUE, 3),
  model_prior = bernoulli(p),
  parameter_prior = dnorm(0, mean = mu, sd = slab_sd),
  T = 20000, seed = 42, show_progress = FALSE
)
inclusion_probs(fit)
mean(fit)
```

== Julia

```julia
using PDMPSamplers, BridgeStan, Distributions, Statistics
μ = [0.7, 0.0, -0.4]
model_file = "mvnormal.stan"
data_file = joinpath(mktempdir(), "independent.json")
write(data_file, """{"N":3,"mu":[0.7,0,-0.4],"sigma":[[1,0,0],[0,1,0],[0,0,1]]}""")
model = PDMPModel(StanModel(model_file, data_file))
p = [0.7, 0.3, 0.5]
κ = p ./ (1 .- p) .* [pdf(Normal(μ[i], 1), 0) for i in 1:3]
fit = pdmp_sample(3, ZigZag(3), model,
                  Sticky(GridThinningStrategy(), κ),
                  0.0, 20_000.0; seed=42, progress=false)
inclusion_probs(fit)
mean(fit)
```

:::

### Beta-binomial inclusion with independent slabs

Replacing separate Bernoulli indicators with a beta-binomial model prior
shares one latent inclusion probability across the coordinates. The sampler
therefore changes release rates with the current model size. This is the
independent-slab beta-binomial case from the R vignette; the correlated-slab
version follows below.

:::tabs

== R

```r
library(PDMPSamplersR)
model_file <- system.file("stan", "models", "mvnormal.stan",
                          package = "PDMPSamplersR")
mu <- c(0.7, 0, -0.4)
data <- list(N = 3L, mu = mu, sigma = diag(3))
fit <- pdmp_sample_from_stanmodel(
  model_file, data, flow = "ZigZag", algorithm = "GridThinningStrategy",
  sticky = TRUE, can_stick = rep(TRUE, 3),
  model_prior = betabernoulli(1, 1),
  parameter_prior = dnorm(0, mu, 1),
  T = 20000, seed = 43, show_progress = FALSE
)
inclusion_probs(fit)
draws <- discretize(fit, dt = 1)
table(rowSums(draws != 0)) / nrow(draws)
```

== Julia

```julia
using PDMPSamplers, BridgeStan, Distributions
μ = [0.7, 0.0, -0.4]
model_file = "mvnormal.stan"
data_file = joinpath(mktempdir(), "independent.json")
write(data_file, """{"N":3,"mu":[0.7,0,-0.4],"sigma":[[1,0,0],[0,1,0],[0,0,1]]}""")
model = PDMPModel(StanModel(model_file, data_file))
mpdfs = [pdf(Normal(μ[i], 1), 0) for i in 1:3]
κ = BetaBernoulliKappa(1.0, 1.0, mpdfs, trues(3))
fit = pdmp_sample(3, ZigZag(3), model,
                  Sticky(GridThinningStrategy(), κ, trues(3)),
                  0.0, 20_000.0; seed=43, progress=false)
inclusion_probs(fit)
draws = Matrix(PDMPDiscretize(fit, 1.0))
[count(==(k), vec(sum(draws .!= 0; dims=2))) / size(draws, 1) for k in 0:3]
```

:::

## Correlated slab and beta-binomial model prior

A diagonal slab can use the independent release rates above. A correlated
slab needs the active-face distribution when some coefficients are zero.
Use `dense_gaussian_slab()` in R or `DependentSlabTarget` in Julia; both
construct those active-face distributions. The earlier R vignette called a
non-diagonal covariance a general limitation because it combined a dense Stan
prior with independent slab release rates. That *combination* does not define
the intended spike-and-slab posterior; the dependent slab interface resolves
it.

The same construction accepts a beta-binomial model prior, which makes the
inclusion indicators dependent.

:::tabs

== R

```r
library(PDMPSamplersR)
model_file <- system.file("stan", "models", "mvnormal.stan",
                          package = "PDMPSamplersR")
mu <- c(0.4, -0.2)
Sigma <- matrix(c(1, 0.35, 0.35, 1.4), 2, 2)
data <- list(N = 2L, mu = mu, sigma = Sigma)
slab <- dense_gaussian_slab(mu, Sigma, coef = c("x.1", "x.2"))

fit <- pdmp_sample_from_stanmodel(
  model_file, data, flow = "ZigZag", algorithm = "GridThinningStrategy",
  sticky = TRUE, can_stick = c(TRUE, TRUE),
  model_prior = betabernoulli(1, 2), slab_prior = slab,
  T = 20000, seed = 42, show_progress = FALSE
)
inclusion_probs(fit)
draws <- discretize(fit, dt = 1)
table(rowSums(draws != 0)) / nrow(draws)  # posterior model-size probabilities
```

== Julia

```julia
using PDMPSamplers, LinearAlgebra, Statistics

μ = [0.4, -0.2]
Σ = [1.0 0.35; 0.35 1.4]
Q = inv(Σ)
slab = DenseGaussianSlab(μ, Σ, [1, 2])
prior = BetaBernoulliModelPrior(2, 1.0, 2.0)
function neggrad!(out, x)
    mul!(out, Q, x - μ)
end
target = DependentSlabTarget(2, neggrad!, slab, prior)
model = PDMPModel(target)
flow = ZigZag(2)
clock = default_aggregate_unstick_clock(slab, prior, flow)
alg = AggregateSticky(GridThinningStrategy(), clock, trues(2))
fit = pdmp_sample(2, flow, model, alg, 0.0, 20_000.0;
                  seed=42, progress=false)
inclusion_probs(fit)
draws = Matrix(PDMPDiscretize(fit, 1.0))
[count(==(k), vec(sum(draws .!= 0; dims=2))) / size(draws, 1) for k in 0:2]
```

:::

The [dependent slab guide](../dependent-slabs.md) explains the general
construction and supported dynamics.

## Logistic regression with variable selection

Here the intercept is always active and the regression coefficients can
freeze at zero. Both paths use the same Stan model and simulated data.

```@raw html
<p>Download <a href="../stan/logistic_regression.stan" download>the Stan model</a>
and <a href="../stan/logistic_regression_data.json" download>the data</a> to your
working directory.</p>
```

:::tabs

== R

```r
library(PDMPSamplersR)
model_file <- "logistic_regression.stan"
data_file <- "logistic_regression_data.json"

fit <- pdmp_sample_from_stanmodel(
  model_file, data_file, flow = "ZigZag", algorithm = "GridThinningStrategy",
  sticky = TRUE, can_stick = c(FALSE, rep(TRUE, 4)),
  model_prior = bernoulli(0.5),
  parameter_prior = rep(dnorm(0, 0, 10), 5),
  T = 20000, seed = 42, show_progress = FALSE
)
cbind(mean = mean(fit), inclusion = inclusion_probs(fit))
```

== Julia

```julia
using PDMPSamplers, BridgeStan, Distributions, Statistics
model_file = "logistic_regression.stan"
data_file = "logistic_regression_data.json"
model = PDMPModel(StanModel(model_file, data_file))
@assert model.d == 5
κ = [Inf; fill(pdf(Normal(0, 10), 0), 4)]
alg = Sticky(GridThinningStrategy(), κ, BitVector([false; trues(4)]))
fit = pdmp_sample(5, ZigZag(5), model, alg, 0.0, 20_000.0;
                  seed=42, progress=false)
hcat(mean(fit), inclusion_probs(fit))
```

:::

## A Bayesian two-sample test

The second coordinate is a standardized effect ``\delta`` with a Cauchy
slab. Its posterior inclusion probability yields the Bayes factor by dividing
posterior inclusion odds by prior inclusion odds. Both examples use prior
inclusion probability `0.5`, so the prior odds are `1`. They load the same
Stan model and data.

```@raw html
<p>Download <a href="../stan/ttest.stan" download>the Stan model</a>
and <a href="../stan/ttest_data.json" download>the data</a> to your
working directory.</p>
```

:::tabs

== R

```r
library(PDMPSamplersR)
model_file <- "ttest.stan"
data_file <- "ttest_data.json"

fit <- pdmp_sample_from_stanmodel(
  model_file, data_file, flow = "ZigZag", algorithm = "GridThinningStrategy",
  sticky = TRUE, can_stick = c(FALSE, TRUE, FALSE),
  model_prior = bernoulli(0.5),
  parameter_prior = rep(dcauchy(0, 0, 1), 3),
  T = 100000, seed = 42, show_progress = FALSE
)
p_delta <- inclusion_probs(fit)[2]
bayes_factor <- p_delta / (1 - p_delta)
bayes_factor
```

== Julia

```julia
using PDMPSamplers, BridgeStan, Distributions
model_file = "ttest.stan"
data_file = "ttest_data.json"
model = PDMPModel(StanModel(model_file, data_file))
@assert model.d == 3  # mu, delta, log(sigma) on the unconstrained scale
κ = [Inf, pdf(Cauchy(0, 1), 0), Inf]
alg = Sticky(GridThinningStrategy(), κ, BitVector([false, true, false]))
fit = pdmp_sample(3, ZigZag(3), model, alg, 0.0, 100_000.0;
                  seed=42, progress=false)
p_delta = inclusion_probs(fit)[2]
bayes_factor = p_delta / (1 - p_delta)
```

:::

The original R vignette also compares this estimate with
`BayesFactor::ttestBF(x, y, rscale = 1)`. The external calculation is an
optional check; the PDMP Bayes factor follows from the same inclusion odds in
either language.

```r
data <- jsonlite::fromJSON(data_file)
analytic <- BayesFactor::ttestBF(x = data$x, y = data$y, rscale = 1)
c(PDMP = bayes_factor,
  analytic = BayesFactor::extractBF(analytic, onlybf = TRUE))
```

## Saving the result

R materializes event data by default, so `saveRDS(fit, "fit.rds")` and
`readRDS("fit.rds")` work. `materialize = FALSE` retains a live Julia
reference; assign `fit <- materialize(fit)` before saving that result. Julia
can serialize a `PDMPChains` object in a compatible Julia environment:

:::tabs

== R

```r
saveRDS(fit, "fit.rds")
restored <- readRDS("fit.rds")
mean(restored)

# If the result was sampled with materialize = FALSE:
# fit <- materialize(fit)
# saveRDS(fit, "fit.rds")
```

== Julia

```julia
using Serialization, Statistics
serialize("fit.jls", fit)
restored = deserialize("fit.jls")
mean(restored)
```

:::
