# Subsampling an ordinal Markov random field

This example has 12 persons and three ordinal nodes with values `0`, `1`, or
`2`. Each person's pseudolikelihood contribution contains all three node
conditionals. A subsampling observation is therefore **one person**, not one
node. The target has six threshold parameters and three interaction
parameters.

The R package provides a custom Stan subset hook and a certified OMRF
residual envelope. The Julia package provides the general
`SubsampledControlVariate` interface. The examples below use those two routes
to the same all-active target. The R route also shows dependent slab variable
selection on the interactions. The native Julia subsampling example samples
the all-active model; see the [dependent slab guide](../dependent-slabs.md) for
sticky selection with full gradients.

## Data

The same fixed 12 × 3 matrix is used in both languages:

:::tabs

== R

```r
X <- matrix(c(
  0,1,2,  2,0,1,  1,2,0,  0,2,1,
  1,0,2,  2,1,0,  0,1,1,  1,2,2,
  2,0,0,  0,2,2,  1,1,0,  2,2,1
), nrow = 12, byrow = TRUE)
N <- nrow(X); P <- ncol(X); K <- 3L
```

== Julia

```julia
X = permutedims(reshape([
    0,1,2,  2,0,1,  1,2,0,  0,2,1,
    1,0,2,  2,1,0,  0,1,1,  1,2,2,
    2,0,0,  0,2,2,  1,1,0,  2,2,1
], 3, 12))
N, P = size(X)
K = 3
```

:::

## R: the custom Stan path

The bundled `omrf.stan` declares `pdmp_get_subsample_size()` and
`pdmp_get_subsample_index()`. `compile_pdmp_stan_model()` installs the package
header and compiler flags. Its one compiled library can evaluate the full
likelihood, prior alone, or an unscaled selected-person sum. The R bridge
applies the `N / m` factor once.

```r
library(PDMPSamplersR)
stan_file <- system.file("stan", "omrf", "omrf.stan",
                         package = "PDMPSamplersR")
stan_data <- list(
  N = N, P = P, K = K, X = X, seen = rep(K, P), prior_only = 0L,
  prior_interaction_sd = 0.5,
  prior_threshold_alpha = 2, prior_threshold_beta = 2
)
prior_data <- stan_data
prior_data$prior_only <- 1L

envelope <- omrf_residual_envelope(
  X = X, seen = stan_data$seen,
  thresholds = "thresholds_0", interactions = "interactions_0"
)
anchor <- rep(0, 9)
subsampling <- stan_subsampling(
  n_observations = N, subsample_size = 1L,
  prior_standata = prior_data,
  residual_envelope = envelope, anchor = anchor
)
compiled <- compile_pdmp_stan_model(stan_file)
stan_parameter_mapping(compiled, stan_data)

check <- stan_subsampling_diagnostics(
  compiled, stan_data, subsampling,
  position = seq(-0.2, 0.25, length.out = 9), subset = 5L,
  velocity = seq(0.4, 1.2, length.out = 9), flow = "ZigZag"
)
stopifnot(check$selected_scale == N,
          check$residual_rate <= check$envelope_rate * (1 + 1e-10) + 1e-10)
```

The envelope is certified for every grid cell; a finite diagnostic is useful
for inspecting one position and subset, but is not the proof of domination.
The selected likelihood gradient is retained for reflection, and each later
proposal draws a new person.

::: details Check the selected likelihood gradient

The R implementation below evaluates the person pseudolikelihood directly,
then compares a finite-difference gradient for person 5 with the Stan subset
hook. It also checks the decomposition and the anchor identity.

```r
edges <- rbind(c(1L, 2L), c(1L, 3L), c(2L, 3L))
person_loglik <- function(q, x) {
  thresholds <- matrix(q[1:6], nrow = 2L)
  interactions <- matrix(0, 3, 3)
  for (e in seq_len(nrow(edges))) {
    j <- edges[e, 1L]; k <- edges[e, 2L]
    interactions[j, k] <- q[6L + e]
    interactions[k, j] <- q[6L + e]
  }
  total <- 0
  for (j in 1:3) {
    if (x[j] > 0L) total <- total + thresholds[x[j], j]
    field <- sum(x * interactions[j, ])
    eta <- c(0, thresholds[, j] + (1:2) * field)
    m <- max(eta)
    total <- total - m - log(sum(exp(eta - m)))
  }
  total + sum(vapply(seq_len(nrow(edges)), function(e) {
    2 * q[6L + e] * x[edges[e, 1L]] * x[edges[e, 2L]]
  }, numeric(1)))
}

q <- seq(-0.2, 0.25, length.out = 9)
step <- 1e-6
selected <- vapply(seq_along(q), function(j) {
  plus <- minus <- q
  plus[j] <- plus[j] + step
  minus[j] <- minus[j] - step
  -(person_loglik(plus, X[5L, ]) - person_loglik(minus, X[5L, ])) /
    (2 * step)
}, numeric(1))
stopifnot(isTRUE(all.equal(check$selected_likelihood_gradient,
                           selected, tolerance = 2e-7)))
stopifnot(isTRUE(all.equal(
  check$subsampled_gradient,
  check$deterministic_gradient +
    check$selected_scale * check$residual_gradient,
  tolerance = 1e-12
)))
anchor_check <- stan_subsampling_diagnostics(
  compiled, stan_data, subsampling,
  position = anchor, subset = 5L,
  velocity = rep(1, 9), flow = "BouncyParticle"
)
stopifnot(isTRUE(all.equal(anchor_check$subsampled_gradient,
                           anchor_check$full_gradient, tolerance = 1e-10)))
```

:::

The original vignette runs three full and three subsampled chains and compares
means relative to Monte Carlo error. Here is the corresponding fit code:

```r
slab <- dense_gaussian_slab(
  rep(0, 3), diag(rep(0.5^2, 3)), coef = "interactions_0"
)
common <- list(
  path_to_stanmodel = compiled, standata = stan_data,
  flow = "ZigZag", algorithm = "GridThinningStrategy",
  T = 300, grid_n = 15L, grid_t_max = 0.5,
  sticky = TRUE, can_stick = "interactions_0",
  model_prior = betabernoulli(1, 4), slab_prior = slab,
  show_progress = FALSE, materialize = FALSE
)
seeds <- 901:903
full <- lapply(seeds, function(seed)
  do.call(pdmp_sample_from_stanmodel, c(common, list(seed = seed))))
subsampled <- lapply(seeds, function(seed)
  do.call(pdmp_sample_from_stanmodel,
          c(common, list(subsampling = subsampling, seed = seed))))

full_means <- do.call(cbind, lapply(full, mean))
subsampled_means <- do.call(cbind, lapply(subsampled, mean))
cbind(full = rowMeans(full_means), subsampled = rowMeans(subsampled_means))
```

::: details Compare replicated fits and inspect cost counters

These checks use the three fits in each list. The mean difference is scaled
by the larger of within-chain and between-chain Monte Carlo uncertainty.

```r
replicated_summary <- function(fits) {
  run_means <- do.call(cbind, lapply(fits, mean))
  run_variances <- do.call(cbind, lapply(fits, function(fit) diag(var(fit))))
  run_ess <- do.call(cbind, lapply(fits, ess))
  runs <- length(fits)
  within_mcse2 <- rowSums(run_variances / run_ess) / runs^2
  between_mcse2 <- apply(run_means, 1L, stats::var) / runs
  list(mean = rowMeans(run_means),
       mcse = sqrt(pmax(within_mcse2, between_mcse2)),
       minimum_ess = min(run_ess))
}
full_summary <- replicated_summary(full)
subsampled_summary <- replicated_summary(subsampled)
standardized_difference <- abs(full_summary$mean - subsampled_summary$mean) /
  sqrt(full_summary$mcse^2 + subsampled_summary$mcse^2)
stopifnot(full_summary$minimum_ess > 20,
          subsampled_summary$minimum_ess > 20,
          all(is.finite(standardized_difference)),
          max(standardized_difference) < 4.5)

for (fit in subsampled) {
  counters <- attr(fit, "subsampling_context_counters")[[1L]]
  if (is.environment(counters)) counters <- as.list(counters)
  stopifnot(
    counters$persons_evaluated ==
      subsampling$subsample_size * counters$analytic_residual_calls,
    counters$sampling_full_gradient_calls == 0L,
    counters$sampling_model_constructions == 0L,
    counters$sampling_data_constructions == 0L
  )
}
```

The full-data gradient calls recorded during initialization prepare the
anchor. The retained phase makes none; each residual oracle call evaluates
exactly one selected person here.

:::

## Julia: a native person oracle

Here the person pseudolikelihood is written in Julia. The six thresholds
occupy `q[1:6]`; the three edge interactions occupy `q[7:9]` in edge order
`(1,2)`, `(1,3)`, `(2,3)`. The prior matches the Stan model: logit-scale beta
priors for thresholds and `Normal(0, 0.5)` for interactions.

```julia
using PDMPSamplers, LinearAlgebra, ForwardDiff, Statistics

edges = ((1, 2), (1, 3), (2, 3))
function person_loglik(q, x)
    interactions = zeros(eltype(q), 3, 3)
    for (e, (j, k)) in enumerate(edges)
        interactions[j, k] = interactions[k, j] = q[6 + e]
    end
    total = zero(eltype(q))
    for j in 1:3
        field = sum(x[k] * interactions[j, k] for k in 1:3)
        η = [zero(eltype(q)), q[2j - 1] + field, q[2j] + 2field]
        m = maximum(η)
        total += x[j] == 0 ? zero(eltype(q)) : q[2j - 2 + x[j]]
        total -= m + log(sum(exp.(η .- m)))
    end
    total += sum(2q[6 + e] * x[j] * x[k]
                 for (e, (j, k)) in enumerate(edges))
    total
end

function prior_logdensity(q)
    thresholds = q[1:6]
    interactions = q[7:9]
    sum(2t - 4log1p(exp(t)) for t in thresholds) -
        sum(abs2, interactions) / (2 * 0.5^2)
end
```

To form the control variate, add the full selected-person gradient at the
anchor to the exact prior gradient. A selected batch contributes the change
in person gradient since the anchor. The per-person envelope below uses a
simple conservative bound: each node's softmax Hessian has operator norm at
most `33` for this data domain (`x[j] ∈ 0:2`), so the three node
conditionals together have norm at most `99`. The linear interaction term has
zero Hessian. A weight of `100` bounds each person's gradient change by
`100 * norm(q - anchor)`.

```julia
anchor = zeros(9)
person_grad(q, i) = ForwardDiff.gradient(z -> person_loglik(z, view(X, i, :)), q)
anchor_likelihood_grad = sum(person_grad(anchor, i) for i in 1:N)

function deterministic!(out, q)
    out .= .-anchor_likelihood_grad .-
           ForwardDiff.gradient(prior_logdensity, q)
end
function residuals!(out, q, subset, anchor)
    fill!(out, 0.0)
    for i in subset
        out .-= person_grad(q, i) .- person_grad(anchor, i)
    end
    out
end

weights = fill(100.0, 1, N)
envelope = SeparableResidualEnvelope(weights,
    (out, state, flow, t) -> begin
        speed = norm(state.ξ.θ)
        out[1] = speed * (norm(state.ξ.x - anchor) + speed * t)
    end; certified_affine=true)
cv = SubsampledControlVariate(
    deterministic!, residuals!, envelope, anchor, 1;
    deterministic_hvp! = (out, q, v) ->
        mul!(out, -ForwardDiff.hessian(prior_logdensity, q), v)
)
model = PDMPModel(9, cv)
flow = ZigZag(9)
fit = pdmp_sample(copy(anchor), flow, model, GridThinningStrategy(),
                  0.0, 300.0; seed=901, progress=false)
(mean = mean(fit), ess = ess(fit))
```

This Julia construction uses a first-order control variate; the R OMRF
envelope uses a tighter analytic Hessian control variate. Both target the same
all-active posterior. To add dependent slab selection in Julia, use the same
person likelihood with a full-gradient target:

```julia
using ADTypes
function full_logdensity(q)
    prior_logdensity(q) + sum(person_loglik(q, view(X, i, :)) for i in 1:N)
end
posterior = PDMPModel(9, LogDensity(full_logdensity),
                      ADTypes.AutoForwardDiff(), true)
beta = [7, 8, 9]
slab = DenseGaussianSlab(zeros(3), 0.25 * Matrix{Float64}(I, 3, 3), beta)
prior = BetaBernoulliModelPrior(3, 1.0, 4.0)
target = DependentSlabTarget(9, posterior.grad, slab, prior)
sticky_model = PDMPModel(target)
flow = ZigZag(9)
clock = default_aggregate_unstick_clock(slab, prior, flow)
can_stick = BitVector([falses(6); trues(3)])
alg = AggregateSticky(GridThinningStrategy(), clock, can_stick)
sticky_fit = pdmp_sample(9, flow, sticky_model, alg, 0.0, 300.0;
                         seed=901, progress=false)
inclusion_probs(sticky_fit)[beta]
```

The R example additionally combines subsampling with dependent slab selection
through its custom Stan bridge. There is currently no corresponding public
Julia composition for that combined workflow; use the R interface for it.
The [general subsampling guide](../subsampling.md) explains the envelope
requirements with a smaller Julia example.
