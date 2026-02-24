```@raw html
---
layout: home

hero:
  name: "PDMPSamplers"
  tagline: Bayesian inference with Piecewise Deterministic Markov Processes
  actions:
    - theme: brand
      text: Getting Started
      link: /getting-started
    - theme: alt
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/vandenman/PDMPSamplers.jl

features:
  - title: ZigZag, BPS & Boomerang
    details: Multiple PDMP dynamics with automatic Poisson thinning.
  - title: Julia & R
    details: Native Julia package with an R bridge — use the language you prefer.
  - title: Turing.jl & Stan
    details: Integrations with popular modelling frameworks.
  - title: Continuous-time output
    details: Exact posterior expectations without discretization bias.
---
```

```@raw html
<div class="vp-doc" style="width:80%; margin:auto">
```

PDMPSamplers implements piecewise deterministic Markov process
samplers for Bayesian inference. These samplers produce
continuous-time output and can exploit subsampled gradients for
scalability.

Available for both **Julia** and **R**.

```@raw html
</div>
```
