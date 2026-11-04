```@raw html
---
layout: home

hero:
  name: "PDMPSamplers"
  tagline: Bayesian inference with<br>Piecewise Deterministic Markov Processes
  actions:
    - theme: brand
      text: Getting Started
      link: /getting-started
    - theme: alt
      text: What are PDMPs?
      link: /what-are-pdmps
    - theme: alt
      text: View on GitHub
      link: https://github.com/vandenman/PDMPSamplers.jl

features:
  - title: Zig-Zag, Bouncy Particle & Boomerang
    details: Several PDMP dynamics, with adaptive and preconditioned variants. Event times are simulated by automatic Poisson thinning, so you don't supply rate bounds.
    link: /choosing-a-sampler
  - title: Variable selection
    details: Sticky samplers for spike-and-slab priors move between models without between-model proposals. Slab priors may be correlated.
    link: /variable-selection
  - title: Continuous-time output
    details: Posterior expectations are integrals along the trajectory. There is no step size, and no discretization error.
  - title: Turing.jl & Stan
    details: Sample models written in Turing.jl (DynamicPPL) or Stan (BridgeStan).
    link: /julia/turing
  - title: Subsampled gradients
    details: Control-variate subsampling lets an event use a small batch of observations while the sampler stays exact.
    link: /subsampling
  - title: Julia & R
    details: Native Julia package with an R interface, PDMPSamplersR.
    link: /r
---
```

```@raw html
<div class="vp-doc" style="width:80%; margin:auto">
```

PDMPSamplers implements piecewise deterministic Markov process (PDMP) samplers for Bayesian inference.
A PDMP is a continuous time algorithm that moves along deterministic paths and changes direction at random event times that depend on the gradient of the log posterior.
PDMPSamplers.jl is the Julia backend, and [PDMPSamplersR](r.md) uses JuliaCall to make the functionality available in R.
The sampler needs a function that calculates the negative gradient of the log posterior.
As this can be rather cumbersome, the software integrates with Stan, Turing, and brms (R only), so that it's possible to specify the models in more user friendly ways.

The PDMP approach has a few advantages over other MCMC samplers.
Notably, the sticky versions of the samplers can do spike-and-slab [variable selection](variable-selection.md).
Furthermore, the runtime of the algorithms can often be reduced using [subsampling](subsampling.md).

```@raw html
</div>
```
