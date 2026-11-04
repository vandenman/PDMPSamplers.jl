# Turing.jl models

PDMPSamplers.jl samples Turing.jl models through DynamicPPL. Load both packages to enable the integration. Parameters are sampled on the unconstrained scale; transform constrained parameters before interpreting them.

```@example turing
using PDMPSamplers, DynamicPPL, Distributions, ADTypes, Statistics
import ForwardDiff

@model function normal_model(y)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0, 1), 0, Inf)
    for i in eachindex(y)
        y[i] ~ Normal(μ, σ)
    end
end

turing_model = normal_model([1.2, 1.5, 1.8, 1.1, 1.4])

# Gradients from ForwardDiff; needs_hvp=true also prepares the
# Hessian-vector products that GridThinningStrategy uses.
model = PDMPModel(turing_model, ADTypes.AutoForwardDiff(); needs_hvp=true)

d = model.d
chains = pdmp_sample(d, ZigZag(d), model, GridThinningStrategy(), 0.0, 10_000.0; progress=false)
nothing # hide
```

The parameters are in the order of the model, on the unconstrained scale:
here `μ` and `log(σ)`. Transform the draws back to interpret them:

```@example turing
draws = Matrix(PDMPDiscretize(chains, 1.0))
μ_draws = draws[:, 1]
σ_draws = exp.(draws[:, 2])
(mean(μ_draws), mean(σ_draws))
```

## Regression with adaptive dynamics

The same interface works with a regression model. Here
`AdaptiveBoomerang` learns a reference distribution during warmup, then
records two chains for the requested main run length.

```julia
using PDMPSamplers, DynamicPPL, Distributions, ADTypes, Statistics
import ForwardDiff

x = collect(range(-2.0, 2.0; length=100))
y = 1 .+ 0.8 .* x .+ 0.5 .* sin.(1:100)

@model function regression(x, y)
    α ~ Normal(0, 5)
    β ~ Normal(0, 5)
    σ ~ truncated(Normal(0, 5), 0, Inf)
    for i in eachindex(y)
        y[i] ~ Normal(α + β * x[i], σ)
    end
end

model = PDMPModel(regression(x, y), ADTypes.AutoForwardDiff(); needs_hvp=true)
d = model.d
chains = pdmp_sample(d, AdaptiveBoomerang(d), model, GridThinningStrategy(),
                     0.0, 10_000.0, 2_000.0;
                     n_chains=2, seed=42, progress=false)
mean(chains)  # α, β, and log(σ) on the unconstrained scale
```

This is a Julia workflow. For the separate R interface to brms, see
[brms models](../r/brms.md).

### Automatic differentiation backends

Any backend supported by
[DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)
works for the gradient. ForwardDiff is the simplest choice for models with
few parameters. For larger models a reverse-mode backend such as Mooncake is
faster, but it cannot compute Hessian-vector products on its own. Either
leave them out, in which case `GridThinningStrategy` computes the curvature
from two gradients:

```julia
import Mooncake
model = PDMPModel(turing_model, ADTypes.AutoMooncake())
```

or combine a forward-mode backend for the second derivative with the
reverse-mode gradient:

```julia
import Mooncake
import DifferentiationInterface as DI
backend = DI.SecondOrder(ADTypes.AutoMooncakeForward(), ADTypes.AutoMooncake())
model = PDMPModel(turing_model, backend; needs_hvp=true)
```
