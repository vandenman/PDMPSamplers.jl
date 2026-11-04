# Choosing a sampler

A sampler in PDMPSamplers.jl has three parts:

1. the **dynamics**: how the process moves between events (`ZigZag`,
   `BouncyParticle`, the Boomerang family);
2. the **Poisson-time strategy**: how the next event time is simulated
   (usually `GridThinningStrategy`), optionally wrapped in `Sticky` for
   variable selection;
3. the **gradient strategy**: full gradients, partial derivatives, or
   subsampled gradients.

This page gives rules of thumb for each choice. When in doubt, start with
`AdaptiveBoomerang(d)` and `GridThinningStrategy()`, and run a warmup.

## Dynamics

| Dynamics | Moves | Works well when | Watch out for |
| --- | --- | --- | --- |
| `ZigZag` | straight lines; one velocity component flips at an event | partial derivatives are cheap; coordinates on similar scales | very different scales or strong correlations |
| `BouncyParticle` | straight lines in any direction; velocity reflects off the gradient | correlated targets of moderate dimension | needs refreshment (`λref`) to be ergodic |
| `Boomerang` | ellipses around a Gaussian reference ``\mathcal{N}(\mu, \Gamma^{-1})`` | the posterior is close to a known Gaussian | a poor reference makes it slow |
| `AdaptiveBoomerang` | as `Boomerang`, but learns the reference during warmup | most models; a good default | needs a warmup period |

### Boomerang

The Boomerang moves along the elliptical orbits of a Gaussian reference
distribution. If the posterior equals that Gaussian, there are no events
other than refreshments; the closer the reference is to the posterior, the
fewer gradient evaluations each unit of time costs. That makes it usually the
most efficient choice, but only with a good reference. If you don't have one,
use `AdaptiveBoomerang`, which estimates the mean and precision during warmup:

```@example choose
using PDMPSamplers, Distributions, LinearAlgebra, ADTypes
import ForwardDiff

# A strongly correlated 10-dimensional normal with mean 2
d = 10
Σ = [0.9^abs(i - j) for i in 1:d, j in 1:d]
target = MvNormal(fill(2.0, d), Σ)
model = PDMPModel(d, LogDensity(x -> logpdf(target, x)), ADTypes.AutoForwardDiff(), true)

flow = AdaptiveBoomerang(d; scheme=:fullrank)       # dense precision, O(d³) per update
# flow = AdaptiveBoomerang(d)                       # diagonal precision
# flow = AdaptiveBoomerang(d; scheme=:lowrank, rank=5)

# the 7th argument is the warmup time
chains = pdmp_sample(d, flow, model, GridThinningStrategy(), 0.0, 10_000.0, 1_000.0; progress=false)
round.(mean(chains); digits=2)
```

The diagonal scheme is cheapest per event; the dense scheme pays off when the
posterior has strong correlations. On this target, the diagonal scheme needs
a run about ten times as long to reach the same accuracy.

### Zig-Zag

The Zig-Zag flips one velocity component at a time, and each component has
its own event rate. With `CoordinateWiseGradient` and `ThinningStrategy`, an
event needs only the partial derivative of one coordinate. This pays off when
partial derivatives are much cheaper than the full gradient, for example when
each parameter depends on only a few others. With full gradients the
advantage disappears.

Every coordinate moves at unit speed, so coordinates on very different scales
mix at very different rates. `PreconditionedZigZag(d; scale=σ)` rescales the
speeds; set `σ` to rough posterior standard deviations.

### Bouncy Particle

The Bouncy Particle sampler moves in any direction and reflects the velocity
off the gradient at an event. It needs refreshment of the velocity to explore
the whole space; `BouncyParticle(d, λref)` sets the rate. It is rotation
invariant, which helps with correlated targets, and `PreconditionedBPS`
rescales it like the preconditioned Zig-Zag.

## Poisson-time strategy

- `GridThinningStrategy()` is the default. It builds an upper bound on the
  event rate along the trajectory from gradient and curvature evaluations, so
  you don't supply bounds. It uses Hessian-vector products when the model
  provides them, and finite differences of gradients otherwise.
- `ThinningStrategy(GlobalBounds(c, d))` or
  `ThinningStrategy(LocalBounds(c))` thins against bounds you supply. Use
  it when you know good bounds, or with `CoordinateWiseGradient`.
- `Sticky(alg, κ)` and `AggregateSticky(alg, clock, can_stick)` wrap one of the
  above for spike-and-slab variable selection; see
  [Variable selection](variable-selection.md) and
  [Dependent Slabs](dependent-slabs.md).

## Subsampling

With many observations, most of the cost of an event is the gradient over all
of them. `SubsampledControlVariate` replaces it by an unbiased estimate from a
small batch of `m` observations, centred at a reference point (the anchor)
near the posterior mode. The sampler remains exact: subsampling changes the
cost, not the target.

Subsampling helps when

- the number of observations is large;
- the posterior is concentrated, so the anchor stays close to where the
  sampler spends its time;
- the gradient of a single observation is cheap.

For small data sets or diffuse posteriors the overhead of the control variate
and the looser rate bounds outweigh the savings; use full gradients. The
[Subsampling](subsampling.md) page works through an example. From R,
[PDMPSamplersR](r.md) sets up subsampling for regression models and Stan models.
