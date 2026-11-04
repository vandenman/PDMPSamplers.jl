# PDMPSamplers.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://vandenman.github.io/PDMPSamplers.jl/dev/)
[![Build Status](https://github.com/vandenman/PDMPSamplers.jl/workflows/runtests/badge.svg)](https://github.com/vandenman/PDMPSamplers.jl/actions)
[![Coverage Status](https://codecov.io/gh/vandenman/PDMPSamplers.jl/graph/badge.svg?token=01KB2QQIMX)](https://codecov.io/gh/vandenman/PDMPSamplers.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

PDMPSamplers.jl does Bayesian inference with piecewise deterministic Markov
processes (PDMPs). It implements the Zig-Zag sampler, the Bouncy Particle
sampler and the Boomerang sampler (with adaptive and preconditioned variants).
Sticky PDMPs extend these samplers to spike-and-slab variable selection: a
coefficient sticks at zero for a random time, so the sampler visits both the
parameters and the models. Gradients can be subsampled, so one event costs a
small batch of observations instead of the full data set.

Version 0.1. The interface may change before 1.0.

## Why use it

- **Continuous-time output.** A sampler run is a continuous trajectory.
  Posterior means, variances and inclusion probabilities are integrals along
  that trajectory, so you don't need to thin it or pick a step size.
- **Model moves without proposals.** Sticky samplers switch between models by
  freezing and releasing coefficients at zero. You don't design or tune
  between-model proposals as in reversible jump.
- **Dependent priors.** Slab priors can be correlated (dense Gaussian,
  exchangeable, or log-linear scales), and the model prior can be Bernoulli or
  beta-binomial.
- **Use your existing models.** Stan models work through BridgeStan, and
  Turing.jl models work through DynamicPPL. From R, use
  [PDMPSamplersR](https://github.com/vandenman/PDMPSamplersR).

## Installation

PDMPSamplers.jl is not yet in the General registry. Install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/vandenman/PDMPSamplers.jl")
```

## Example

Sample from a bivariate normal with the Zig-Zag sampler:

```julia
using PDMPSamplers, Distributions, ADTypes
import ForwardDiff

d = 2
target = MvNormal(d, 1.0)

# The log density, differentiated with ForwardDiff. `true` also prepares
# Hessian-vector products, which GridThinningStrategy uses.
model = PDMPModel(d, LogDensity(x -> logpdf(target, x)), ADTypes.AutoForwardDiff(), true)

flow = ZigZag(d)
alg  = GridThinningStrategy()

# Start at a random point and run from time 0 to time 10 000
chains = pdmp_sample(d, flow, model, alg, 0.0, 10_000.0)

mean(chains)  # posterior mean, integrated along the trajectory
```

The documentation explains [what PDMPs are](https://vandenman.github.io/PDMPSamplers.jl/dev/what-are-pdmps),
has tutorials on
[variable selection](https://vandenman.github.io/PDMPSamplers.jl/dev/variable-selection) and
[subsampling](https://vandenman.github.io/PDMPSamplers.jl/dev/subsampling), and a guide to
[choosing a sampler](https://vandenman.github.io/PDMPSamplers.jl/dev/choosing-a-sampler).

## Links

- [Documentation](https://vandenman.github.io/PDMPSamplers.jl/dev/)
- [PDMPSamplersR](https://github.com/vandenman/PDMPSamplersR), the R interface
  ([getting started](https://vandenman.github.io/PDMPSamplers.jl/dev/r/getting-started)).
  It runs Stan models from R.

## Citation

If you use PDMPSamplers.jl, please cite it. The citation metadata is in
[`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button turns it
into BibTeX or APA. In BibTeX:

```bibtex
@software{PDMPSamplers_jl,
  author  = {van den Bergh, Don},
  title   = {{PDMPSamplers.jl}: Piecewise deterministic {M}arkov process samplers for {B}ayesian inference},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/vandenman/PDMPSamplers.jl}
}
```

## License

MIT. See [LICENSE.md](LICENSE.md).
