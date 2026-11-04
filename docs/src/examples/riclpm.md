# Dependent slab selection in a RI-CLPM

This example uses a bivariate, three-wave random-intercept cross-lagged panel
model. Two cross-lagged coefficients form the selectable block. Their slab is
a correlated Gaussian distribution, so the sampler must use the correct
conditional slab on every active model face. The R and Julia paths below run
the **same Stan model and data**.

```@raw html
<p>Download <a href="../stan/riclpm_sufficient.stan" download>the Stan model</a>
and <a href="../stan/riclpm_data.json" download>the example data</a> to your
working directory.</p>
```
The data file contains a deterministic 36-person simulation from the original
R vignette (seed `6104`): each observed trajectory is a person-specific random
intercept plus a within-person VAR(1) process. The likelihood uses `N`,
`z_sum`, and `z_crossprod` as sufficient statistics.

## Select the cross-lagged block

Both languages use a beta-binomial model prior with `a = 1`, `b = 2` and
the same 2 × 2 slab covariance. An ordinary sticky sampler with independent
release rates would give the wrong conditional prior when one cross-lagged
coefficient is frozen; the dependent slab interface handles that change.

:::tabs

== R

```r
library(PDMPSamplersR)
model_file <- "riclpm_sufficient.stan"
data_file <- "riclpm_data.json"
slab_mean <- c(0, 0)
slab_cov <- matrix(c(0.20, 0.04, 0.04, 0.16), 2, 2)
slab <- dense_gaussian_slab(slab_mean, slab_cov, coef = "cross_lagged")

flows <- c("ZigZag", "BouncyParticle", "AdaptiveBoomerang")
fits <- setNames(lapply(seq_along(flows), function(i) {
  flow <- flows[i]
  pdmp_sample_from_stanmodel(
    model_file, data_file, flow = flow,
    algorithm = "GridThinningStrategy", T = 2,
    t_warmup = if (flow == "AdaptiveBoomerang") 0.4 else 0,
    grid_n = 8L, grid_t_max = 0.15,
    sticky = TRUE, can_stick = "cross_lagged",
    model_prior = betabernoulli(1, 2), slab_prior = slab,
    seed = 610L + i, show_progress = FALSE
  )
}), flows)

lapply(fits, inclusion_probs)
```

== Julia

```julia
using PDMPSamplers, BridgeStan, Statistics

model_file = "riclpm_sufficient.stan"
data_file = "riclpm_data.json"
slab_mean = [0.0, 0.0]
slab_cov = [0.20 0.04; 0.04 0.16]
sm = StanModel(model_file, data_file)
names = param_unc_names(sm)
beta = findall(name -> startswith(name, "cross_lagged."), names)
@assert length(beta) == 2
d = param_unc_num(sm)
can_stick = falses(d)
can_stick[beta] .= true

function sample_flow(flow, seed, warmup)
    # Construct a fresh target for each run: its active-face cache is mutable.
    posterior = PDMPModel(StanModel(model_file, data_file))
    slab = DenseGaussianSlab(slab_mean, slab_cov, beta)
    prior = BetaBernoulliModelPrior(2, 1.0, 2.0)
    target = DependentSlabTarget(d, posterior.grad, slab, prior)
    model = PDMPModel(target)
    clock = default_aggregate_unstick_clock(slab, prior, flow)
    alg = AggregateSticky(GridThinningStrategy(N=8, t_max=0.15),
                          clock, can_stick)
    pdmp_sample(d, flow, model, alg, 0.0, 2.0, warmup;
                seed, progress=false)
end

fits = Dict(
    "ZigZag" => sample_flow(ZigZag(d), 611, 0.0),
    "BouncyParticle" => sample_flow(BouncyParticle(d), 612, 0.0),
    "AdaptiveBoomerang" => sample_flow(AdaptiveBoomerang(d), 613, 0.4),
)
Dict(name => inclusion_probs(chains)[beta] for (name, chains) in fits)
```

:::

These short runs check that all three dynamics can traverse the model. They
are too short for posterior conclusions. Increase `T` and check the effective
sample size for an analysis. The [dependent slab guide](../dependent-slabs.md)
explains the active-face target and release clocks in detail.
