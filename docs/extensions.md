# Package Extensions for PDMPSamplers.jl

This document describes the package extensions available for PDMPSamplers.jl.

## DynamicPPL Extension

The DynamicPPL extension allows you to directly use Turing.jl/DynamicPPL models with PDMP samplers.

### Installation

```julia
using Pkg
Pkg.add("PDMPSamplers")
Pkg.add("DynamicPPL")  # Triggers the extension
```

### Usage

```julia
using PDMPSamplers, DynamicPPL, Distributions
import Mooncake

# Define a DynamicPPL model
@model function normal_model(y)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0, 1), 0, Inf)
    for i in eachindex(y)
        y[i] ~ Normal(μ, σ)
    end
end

# Create model with data
data = [1.2, 1.5, 1.8, 1.1, 1.4]
model = normal_model(data)

# Construct PDMPModel automatically from DynamicPPL model
pdmp_model = PDMPModel(model, ADTypes.AutoMooncake(); needs_hvp=true)

# Sample using PDMP
flow = ZigZag(pdmp_model.d)
alg = GridThinningStrategy()
x0 = randn(pdmp_model.d)
trace, stats = pdmp_sample(x0, flow, pdmp_model, alg, 0.0, 10_000.0)
```

### Key Features
- Automatically extracts dimension from the model
- Works in the unconstrained parameter space
- Supports arbitrary AD backends via DifferentiationInterface
- Optionally generates HVP for GridThinningStrategy

## BridgeStan Extension

The BridgeStan extension allows you to use compiled Stan models directly with PDMP samplers.

### Installation

```julia
using Pkg
Pkg.add("PDMPSamplers")
Pkg.add("BridgeStan")  # Triggers the extension
```

### Usage

```julia
using PDMPSamplers, BridgeStan

# Load a compiled Stan model
sm = StanModel("path/to/model.so", "path/to/data.json")

# Construct PDMPModel from Stan model
pdmp_model = PDMPModel(sm; needs_hvp=true)

# Or construct directly from file paths
pdmp_model = PDMPModel(
    "path/to/model.so",
    "path/to/data.json";
    needs_hvp=true
)

# Sample using PDMP
flow = BouncyParticle(pdmp_model.d)
alg = GridThinningStrategy()
x0 = randn(pdmp_model.d)
trace, stats = pdmp_sample(x0, flow, pdmp_model, alg, 0.0, 10_000.0)
```

### Key Features
- Leverages Stan's efficient autodiff system
- Works in the unconstrained parameter space
- Automatically handles dimension extraction
- Finite-difference HVP approximation when needed

## Comparison

| Feature | DynamicPPL | BridgeStan |
|---------|------------|------------|
| Model definition | Julia DSL | Stan language |
| Gradient computation | User-chosen AD | Stan autodiff |
| HVP computation | Via DI/AD | Finite differences |
| Performance | Depends on AD backend | Typically very fast |
| Flexibility | High (pure Julia) | Medium (requires compilation) |

## Notes

- Both extensions work in the **unconstrained parameter space**
- For `GridThinningStrategy`, set `needs_hvp=true`
- For `ThinningStrategy`, HVP is not needed
- Initial conditions should be in unconstrained space
