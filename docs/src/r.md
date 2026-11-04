# Using PDMPSamplers from R

[PDMPSamplersR](https://github.com/vandenman/PDMPSamplersR) runs the Julia
samplers from R. This is the starting point for R users of these docs. The
samplers and the underlying ideas are shared across both languages; the pages
in this section explain the R interface.

## Start here

1. [Install and run a first sampler](r/getting-started.md) with an R gradient
   function.
2. [Sample a Stan model](r/stan.md) through BridgeStan, or [fit a brms
   model](r/brms.md) and get a `brmsfit` result.
3. [Read the result](r/results.md): continuous-time summaries, chains, and
   draws for plotting.

For the methods themselves, see [What are PDMPs?](what-are-pdmps.md) and
[Choosing a sampler](choosing-a-sampler.md). [Variable selection](variable-selection.md),
[dependent slabs](dependent-slabs.md), and [subsampling](subsampling.md) cover
features available through the R interface as well. Each R page links to the
relevant shared guide.

## Worked examples in both languages

- [Stan targets and variable selection](examples/stan-workflows.md): normal
  targets, Bernoulli and beta-binomial inclusion, correlated slabs, logistic
  regression, and a two-sample Bayes factor.
- [RI-CLPM with dependent slabs](examples/riclpm.md): the same Stan model and
  data with Zig-Zag, Bouncy Particle, and Adaptive Boomerang.
- [Ordinal MRF subsampling](examples/omrf.md): person-level subsampling from R
  and a native Julia control variate for the same all-active target.

## R function reference

The [function reference](r/reference.md) lists every exported R function and
its arguments, generated from the package help files. In an R session, the
same help is available with `?PDMPSamplersR::pdmp_sample` or
`?PDMPSamplersR::brm_pdmp`.
