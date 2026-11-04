# Stan models from Julia

PDMPSamplers.jl samples Stan targets through BridgeStan. Load both packages to enable the integration. Stan parameters are sampled on the unconstrained scale, so transform positive parameters before interpreting them.

Write the Stan model and the data to files, and pass the paths to
`StanModel`. BridgeStan compiles the model the first time; this takes a
minute.

```@example stan
using PDMPSamplers, BridgeStan, Statistics

stan_code = """
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ normal(mu, sigma);
}
"""

dir = mktempdir()
model_file = joinpath(dir, "normal.stan")
data_file  = joinpath(dir, "normal.json")
write(model_file, stan_code)
write(data_file, """{"N": 5, "y": [1.2, 1.5, 1.8, 1.1, 1.4]}""")

stan_model = StanModel(model_file, data_file)
model = PDMPModel(stan_model)

d = model.d
chains = pdmp_sample(d, BouncyParticle(d), model, GridThinningStrategy(), 0.0, 10_000.0; progress=false)
nothing # hide
```

`mean(chains)` is the mean of `mu` and `log(sigma)`. For results on the
original scale, take draws from the path and constrain each with BridgeStan:

```@example stan
draws = Matrix(PDMPDiscretize(chains, 1.0))
constrained = reduce(vcat, (param_constrain(stan_model, collect(row))' for row in eachrow(draws)))

param_names(stan_model)          # ["mu", "sigma"]
mean(constrained; dims=1)
```

Gradients come from Stan's automatic differentiation. `GridThinningStrategy`
also needs curvature along the path; by default this is computed from two
gradients, which is usually fastest. Pass `PDMPModel(stan_model; hvp=true)` to
use Stan's Hessian-vector product instead.

To run Stan models from R, see [Stan models from R](../r/stan.md).
The [worked examples](../examples/stan-workflows.md) pair R and Julia code for
normal targets, variable selection, logistic regression, and a two-sample test.

