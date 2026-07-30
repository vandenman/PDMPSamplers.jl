# Dependent Slabs

Dependent slab support is intended for sticky variable-selection samplers where the active coefficients have a joint Gaussian slab prior.
Users should normally construct a model prior, a Gaussian slab prior, and an `AggregateSticky` strategy with a `GridThinningStrategy` inner clock.
Choose the clock with `default_aggregate_unstick_clock(slab, model_prior, flow)`;
the flow-aware form selects the Fourier residual implementation needed by
Boomerang-family flows instead of the linear-flow Chebyshev clock.

## Support Matrix

| Dynamics | Aggregate sticky support | Notes |
| --- | --- | --- |
| `ZigZag` | yes | Coordinate boundary velocities are sampled from the standard ZigZag law. |
| `BouncyParticle` | yes | Gaussian boundary velocities use the reference velocity marginal. |
| `Boomerang` and `AdaptiveBoomerang` | yes | Dense, diagonal, and low-rank Boomerang covariance representations use conditional Gaussian boundary velocities. |
| Diagonal-preconditioned `ZigZag` | yes | Coordinate boundary velocities are scaled by the diagonal preconditioner. |
| Dense-preconditioned `ZigZag` | no | A dense coordinate boundary law is not implemented. |
| Preconditioned `BouncyParticle` | yes | Identity, diagonal, and dense preconditioners are supported. |
| Preconditioned `Boomerang` and `AdaptiveBoomerang` | yes | Boundary velocities use the preconditioned Gaussian velocity covariance implied by the current preconditioner and Boomerang covariance. |

The supported aggregate clocks are:

| Clock | Supported dynamics | Notes |
| --- | --- | --- |
| `SummedRateClock` | all supported `AggregateSticky` dynamics | Exact by numerical integration for time-varying rates. |
| `LinearGaussianAggregateClock` | `ZigZag`, `BouncyParticle`, diagonal-preconditioned `ZigZag`, preconditioned `BouncyParticle` | Falls back to `SummedRateClock` for Boomerang-family flows. |
| `ExponentialSumAggregateClock` | global-logscale slab clocks with linear dynamics | Falls back to `SummedRateClock` for Boomerang-family flows. |
| `ChebyshevResidualAggregateClock` | `ZigZag`, `BouncyParticle` with global-logscale exchangeable slabs | Experimental and intentionally unexported. |
| `FourierResidualAggregateClock` | Boomerang-family flows with fixed-covariance or global-logscale Gaussian slabs | Experimental and intentionally unexported. |

The public, user-facing constructors are the prior and slab types such as `BernoulliModelPrior`, `BetaBernoulliModelPrior`, `ExchangeableModelSizePrior`,
`DenseGaussianSlab`, `IndependentZeroMeanGaussianSlab`, `IndependentZeroMeanLogscaleGaussianSlab`, `ExchangeableGaussianSlab`,
`ZeroMeanExchangeableGaussianSlab`, and `GlobalLogscaleExchangeableGaussianSlab`.
