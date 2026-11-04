# Documentation development

The Julia and R guides and the R function reference are published as one site.
Documenter/VitePress builds the guides; pkgdown builds the R reference from
the R package's help files.

## Setup

Install Julia, R, Quarto, and Pandoc, then run:

```sh
docs/instantiate.sh
```

This restores the R packages recorded in `docs/renv.lock` into
`docs/renv/library` and instantiates the Julia documentation environment.
The renv bootstrap also installs locally. No R packages are installed into
your user or system library.

The R package dependencies are listed in `docs/DESCRIPTION`. Interactive R
sessions started in `docs/` activate renv automatically; the R build and
validation scripts also activate it when called directly.

## Build and preview

Use a checkout of PDMPSamplersR as the reference source:

```sh
git clone https://github.com/vandenman/PDMPSamplersR.git /path/to/PDMPSamplersR
docs/preview.sh /path/to/PDMPSamplersR
```

This builds the reference and guides, checks the links, and starts VitePress.
Open the local URL printed by the server. To build without starting a server,
or preview an existing build:

```sh
docs/build.sh /path/to/PDMPSamplersR
docs/preview.sh --no-build
```

The scripts work from any working directory. Instead of passing the R source
path each time, set `PDMPSAMPLERS_R_SOURCE`. By default they use the
`r-package/` checkout at the repository root, as CI does.
Set `JULIA` or `RSCRIPT` to select a different executable if needed.

Rebuild after changing Julia examples, docstrings, or R help files. Generated
reference HTML is ignored by Git. The R package's `_pkgdown.yml` supplies its
reference grouping.

## Saved R example output

The Getting Started R tabs display output saved from actual runs. Local site
builds reuse it and check that it matches the example code. After changing an
R example, regenerate its output:

```sh
docs/update-examples.sh /path/to/PDMPSamplersR
```

This restores the separate `examples` renv profile, installs the checked-out
R interface into that profile's library, executes the examples with knitr,
and checks their estimated means and covariances. Commit the updated files
in `docs/examples/output/` with the example changes. Ordinary local builds
do not need the sampler runtime dependencies.

CI always runs this command before building, so the published site shows
output from the examples executed in that CI run. To update the examples
profile's dependency versions, start R in `docs/` with `RENV_PROFILE=examples`,
then run `renv::update()` and `renv::snapshot()`.

## Update dependencies

To intentionally update the R and Julia documentation dependencies:

```sh
docs/update.sh
```

Review and commit the changes to `docs/renv.lock` and any dependency
declarations. Normal setup restores the lockfile without updating it.

## Check examples

The Stan smoke script runs selected published examples with short
trajectories. The optional R mode also uses the docs renv library. To use it,
start R in `docs/` and install the R interface there with
`renv::install("/path/to/PDMPSamplersR")`. These sampler runtime dependencies
are not needed to build the site.

```sh
julia --project=docs docs/smoke/stan_workflows.jl --language julia
julia --project=docs docs/smoke/stan_workflows.jl --language r
```

## Deployment

CI uses `docs/instantiate.sh` and `docs/build.sh`, then runs
`docs/deploy.jl`. The deployment entry point also activates the R environment
and validates links before publishing. Building does not deploy.
