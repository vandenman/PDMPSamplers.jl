# Dependent Slabs

Dependent slab support is intended for sticky variable-selection samplers where the active coefficients have a joint Gaussian slab prior.
Users should normally construct a model prior, a Gaussian slab prior, and an `AggregateSticky` strategy with a `GridThinningStrategy` inner clock.
Choose the clock with `default_aggregate_unstick_clock(slab, model_prior, flow)`;
the flow-aware form selects the Fourier residual implementation needed by
Boomerang-family flows instead of the linear-flow Chebyshev clock.

## Support matrix

The labels below distinguish a certified accelerated path from the exact
`SummedRateClock` fallback. "Experimental" means the API is intentionally
unexported even though it is covered by correctness tests.

| Dynamics | Aggregate sticky support | Notes |
| --- | --- | --- |
| `ZigZag` | supported | Coordinate boundary velocities use the ZigZag law. |
| `BouncyParticle` | supported | Gaussian boundary velocities use the reference velocity marginal. |
| `Boomerang`, `MutableBoomerang`, and `AdaptiveBoomerang` | supported | Dense, diagonal, and low-rank covariance representations use conditional Gaussian boundary velocities. |
| Diagonal-preconditioned `ZigZag` | supported | Coordinate boundary velocities are scaled by the diagonal preconditioner. |
| Dense-preconditioned `ZigZag` | supported | The dense coordinate-boundary law and stationarity behavior are tested. |
| Preconditioned `BouncyParticle` | supported | Identity, diagonal, and dense preconditioners are supported. |
| Preconditioned Boomerang-family dynamics | supported | Boundary velocities use the transformed Gaussian velocity covariance; residual clocks currently use exact fallback. |

The supported aggregate clocks are:

| Clock | Supported dynamics | Notes |
| --- | --- | --- |
| `SummedRateClock` | all supported dynamics/providers | Supported exact baseline; numerical integration is used for time-varying rates. |
| `LinearGaussianAggregateClock` | linear dynamics with fixed-covariance Gaussian slabs | Accelerated; other supported dynamics use exact fallback. |
| `ExponentialSumAggregateClock` | linear dynamics with log-linear independent Gaussian scales | Accelerated; other supported dynamics use exact fallback. |
| `ChebyshevResidualAggregateClock` | `ZigZag`/`BouncyParticle` with global-logscale exchangeable slabs | Experimental certified finite-horizon envelope and exact analytic infinite-horizon sampler; all other combinations use exact fallback by default. |
| `FourierResidualAggregateClock` | unpreconditioned Boomerang-family flows with fixed-covariance or global-logscale Gaussian slabs | Experimental certified acceleration on finite horizons. Infinite horizons, preconditioned flows, callback/state-dependent Gaussian slabs, and `ArbitrarySlabBoundary` use exact fallback. |

Residual clocks default to `allow_slow_fallback=true`: every provider/dynamics
combination supported by `SummedRateClock` therefore remains usable. Setting it
to `false` is certified-only mode. A combination without a certified accelerated
implementation then raises one intentional capability error naming the provider
and dynamics; it never falls through to an internal dispatch error.

The public, user-facing constructors are the prior and slab types such as `BernoulliModelPrior`, `BetaBernoulliModelPrior`, `ExchangeableModelSizePrior`,
`DenseGaussianSlab`, `IndependentZeroMeanGaussianSlab`, `IndependentZeroMeanLogscaleGaussianSlab`, `ExchangeableGaussianSlab`,
`ZeroMeanExchangeableGaussianSlab`, and `GlobalLogscaleExchangeableGaussianSlab`.

## Complete example

This example has two selectable regression coefficients and one nuisance
coefficient. The posterior negative gradient passed to `DependentSlabTarget`
contains the likelihood, nuisance prior, and the coherent slab with **all beta
coefficients active**.

```julia
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

initial = SkeletonPoint([0.1, -0.1, 0.0], [1.0, -1.0, 1.0])
trace, stats = pdmp_sample(initial, flow, model, alg, 0.0, 100.0;
    seed=42, progress=false)

# free_masks[i, n] is true when coordinate i is active at stored event n.
active_beta_at_events = trace.free_masks[beta, :]
frozen_beta_at_events = .!active_beta_at_events
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
- A stored `free_masks[i, n] == false` means coordinate `i` is frozen at zero.

For fixed-covariance providers, active-face cache entries are keyed by the
beta-block active mask. Any active-set transition invalidates the old stratum
entry before rescheduling. State-dependent providers return no active-set-only
cache key because their boundary quantities must also follow the continuous
state.
