# Documentation development

The Documenter/VitePress site contains the Julia and R guides. Build it from
the PDMPSamplers.jl repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
python3 docs/serve.py
```

Open <http://localhost:8000/> to preview the built site.

The R function reference is generated from the R package's `man/*.Rd` files.
After updating those help files, regenerate and check the pages:

```sh
Rscript docs/generate_r_reference.R /path/to/PDMPSamplersR
Rscript docs/generate_r_reference.R /path/to/PDMPSamplersR --check
python3 docs/smoke/check_r_reference.py
```

The last check requires a fresh site build. The Stan smoke script runs the
published examples with short trajectories. The R check requires an installed
PDMPSamplersR package on the active R library path:

```sh
python3 docs/smoke/stan_workflows.py --language julia
python3 docs/smoke/stan_workflows.py --language r
```

The R CI workflow checks out the Julia documentation from its default branch,
so merge the Julia site PR before the R package PR.
