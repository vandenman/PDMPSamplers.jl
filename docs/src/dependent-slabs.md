# Dependent Slabs

This page assumes you know the basic sticky sampler from the
[Variable selection](variable-selection.md) tutorial, which uses independent
slabs.
The [RI-CLPM example](examples/riclpm.md) shows the same correlated slab
Stan model sampled from R and Julia.

Dependent slab support is intended for sticky variable-selection samplers where the active coefficients have a joint Gaussian slab prior.
Users should normally construct a model prior, a Gaussian slab prior, and an `AggregateSticky` strategy with a `GridThinningStrategy` inner clock.
Choose the clock with `default_aggregate_unstick_clock(slab, model_prior, flow)`;
the flow-aware form selects the clock that matches the dynamics (see the table
of clocks below).

## Support matrix

All listed dynamics are supported. They differ in how a frozen coordinate is
released (see below). Both release laws are exact and covered by correctness
tests. The boundary-proposal law is only needed for dynamics with correlated
velocities, so it has seen less use than the stored-velocity law.

| Dynamics | Aggregate sticky support | Release velocity |
| --- | --- | --- |
| `ZigZag` | supported | stored velocity |
| `BouncyParticle` | supported | stored velocity |
| `Boomerang`, `MutableBoomerang`, and `AdaptiveBoomerang` with a diagonal covariance | supported | stored velocity |
| `Boomerang`-family dynamics with a dense or low-rank covariance | supported | boundary proposal |
| Identity- or diagonal-preconditioned `ZigZag` and `BouncyParticle` | supported | stored velocity |
| Identity- or diagonal-preconditioned Boomerang-family dynamics | supported | as the underlying Boomerang |
| Dense-preconditioned `ZigZag` and `BouncyParticle` | supported | boundary proposal |

Two release laws are used:

- **Stored velocity.** When a coordinate sticks at zero, its velocity is
  stored, and it is released with that same velocity. The release rate of a
  frozen coordinate is proportional to the absolute value of its stored
  velocity.
- **Boundary proposal.** For dynamics with correlated velocities, a frozen
  coordinate is released with a velocity drawn from the conditional boundary
  law of the flow. Its release rate uses the corresponding expected speed.

For preconditioned Boomerang-family dynamics, log-linear Gaussian scale slabs
use `HarmonicLogLinearAggregateClock`; other slabs use the exact fallback.

The supported aggregate clocks are:

| Clock | Supported dynamics | Notes |
| --- | --- | --- |
| `SummedRateClock` | all supported dynamics/providers | Supported exact baseline; numerical integration is used for time-varying rates. |
| `LinearGaussianAggregateClock` | linear dynamics with fixed-covariance Gaussian slabs | Accelerated; other supported dynamics use exact fallback. |
| `ExponentialSumAggregateClock` | linear dynamics with log-linear independent Gaussian scales | Accelerated; other supported dynamics use exact fallback. |
| `HarmonicLogLinearAggregateClock` | Boomerang-family flows (also diagonally preconditioned) with log-linear Gaussian scale slabs | Certified piecewise-constant thinning bound on finite horizons; infinite horizons use the exact fallback. |

Every accelerated clock falls back to the exact `SummedRateClock` for
dynamics it has no specialized method for.

The public, user-facing constructors are the model priors `BernoulliModelPrior`
and `BetaBernoulliModelPrior`, and the slabs `DenseGaussianSlab`,
`ExchangeableGaussianSlab` (including mean zero), and
`LogLinearGaussianScaleSlab`.

## Complete example

This example has two selectable regression coefficients and one nuisance
coefficient. The posterior negative gradient passed to `DependentSlabTarget`
contains the likelihood, nuisance prior, and the coherent slab with **all beta
coefficients active**.

```@example dependent-slabs
using PDMPSamplers, LinearAlgebra
d = 3
beta = [1, 2]
X = [1.0 0.2 0.5; -0.3 1.1 0.4; 0.7 -0.6 1.0]
y = [0.4, -0.2, 0.8]

slab_cov = [1.0 0.35; 0.35 1.4]
slab_precision = inv(slab_cov)
slab = DenseGaussianSlab(zeros(2), slab_cov, beta)
model_prior = BernoulliModelPrior([0.3, 0.3])

# Negative gradient of the full, all-active posterior.
function posterior_neggrad!(out, x)
    mul!(out, X', X * x - y)                 # likelihood
    out[3] += x[3]                           # nuisance N(0, 1) prior
    out[beta] .+= slab_precision * x[beta]   # coherent all-active slab
    return out
end

target = DependentSlabTarget(d, posterior_neggrad!, slab, model_prior)
model = PDMPModel(target)
flow = ZigZag(d)

# This is a full-state mask. Coordinate 3 is nuisance and cannot freeze.
can_stick = BitVector([true, true, false])
clock = default_aggregate_unstick_clock(slab, model_prior, flow)
alg = AggregateSticky(GridThinningStrategy(), clock, can_stick)

initial = SkeletonPoint([0.1, -0.1, 0.0], [1.0, -1.0, 1.0])   # position and velocity
chains = pdmp_sample(initial, flow, model, alg, 0.0, 100.0; seed=42, progress=false)

# Posterior inclusion probabilities of the beta coefficients
inclusion_probs(chains)[beta]
```

To see which coefficients are active over time, discretize the path. A frozen
coefficient is exactly zero:

```@example dependent-slabs
draws = Matrix(PDMPDiscretize(chains, 1.0))
active = draws[:, beta] .!= 0     # active[n, j]: beta coefficient j is in the model at draw n
active[1:5, :]
```

Do not pass a gradient that simply omits the slab, and do not use an ordinary
posterior gradient directly after coefficients freeze. On an active face, a
dependent slab induces a conditional prior for the remaining coefficients.
`DependentSlabTarget` obtains that face target by subtracting the all-active
slab contribution and adding the correct active-face contribution. Skipping the
wrapper generally samples the wrong active-face distribution.

## Index and mask conventions

- `beta_indices(slab)` maps beta-block positions to full-state coordinates.
- `can_stick` and the sampler state's `free` mask always have length `d`.
- Active/stickable beta masks have length `length(beta_indices(slab))` and use
  beta-block order, not full-state order.
- `can_stick` may select any subset of `stickable_coordinates(clock)`, but may
  not select nuisance or other unsupported coordinates. This is checked during
  sampler initialization, before any event is scheduled.
- `stickable_coordinates(clock)` itself must return a vector of unique integer
  full-state coordinates in `1:d`; the entire vector is validated even when
  `can_stick` selects only a subset. Custom aggregate clocks must implement
  this extension method explicitly.
- For dynamics with a non-factorized trace (`BouncyParticle` and the Boomerang
  family), `trace.free_masks[i, n] == false` means coordinate `i` is frozen at
  zero at stored event `n`.

For fixed-covariance providers, active-face cache entries are keyed by the
beta-block active mask. Any active-set transition invalidates the old stratum
entry before rescheduling. State-dependent providers return no active-set-only
cache key because their boundary quantities must also follow the continuous
state.
