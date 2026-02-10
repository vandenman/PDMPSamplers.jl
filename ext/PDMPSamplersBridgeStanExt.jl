module PDMPSamplersBridgeStanExt

using PDMPSamplers
using BridgeStan
import PDMPSamplers: PDMPModel, FullGradient

"""
    PDMPModel(sm::BridgeStan.StanModel)

Construct a `PDMPModel` from a BridgeStan StanModel.

BridgeStan provides compiled Stan models with efficient gradient computations.
This constructor wraps the model's log density gradient function for use with PDMP samplers.
The HVP (Hessian-vector product) is always enabled since it's compiled in Stan anyway.

# Arguments
- `sm::BridgeStan.StanModel`: A compiled Stan model

# Example
```julia
using BridgeStan, PDMPSamplers

# Assume you have a compiled Stan model
sm = StanModel("path/to/model.so", "path/to/data.json")
pdmp_model = PDMPModel(sm)

# Use with PDMP samplers
flow = ZigZag(...)
alg = GridThinningStrategy()
trace, stats = pdmp_sample(x0, flow, pdmp_model, alg, 0.0, 10_000.0)
```

# Notes
- BridgeStan works in the unconstrained parameter space
- The model automatically handles the appropriate transformations
- Gradients are computed efficiently through Stan's autodiff system
- HVP is always enabled for compatibility with GridThinningStrategy
"""
function PDMPModel(sm::BridgeStan.StanModel)

    # Get dimension from the Stan model
    d = BridgeStan.param_unc_num(sm)

    # Prepare buffers
    out = zeros(d)

    # Create gradient function that calls Stan's log_density_gradient
    # BridgeStan's log_density_gradient! computes the gradient of log p(x)
    grad_f! = (out, x) -> begin
        BridgeStan.log_density_gradient!(sm, x, out)
        out .= .-out
        return out
    end

    # Use Stan's compiled Hessian-vector product
    out_hvp = zeros(d)
    hvp_f = Base.Fix1((out, x, v) -> begin
        BridgeStan.log_density_hessian_vector_product!(sm, x, v, out)
        out .= .-out
        return out
    end, out_hvp)

    return PDMPModel(d, FullGradient(grad_f!), hvp_f, false, false)
end

"""
    PDMPModel(model_path::String, data_path::String=""; kwargs...)

Construct a `PDMPModel` by loading a Stan model from file paths.

This is a convenience constructor that first creates a BridgeStan.StanModel
and then constructs the PDMPModel from it. HVP is always enabled since
BridgeStan provides compiled HVP functions.

# Arguments
- `model_path::String`: Path to the compiled Stan model (.so file)
- `data_path::String=""`: Optional path to the data file (.json)
- `kwargs...`: Additional keyword arguments passed to `BridgeStan.StanModel`

# Example
```julia
using PDMPSamplers

pdmp_model = PDMPModel("model.stan", "data.json")
```
"""
function PDMPModel(model_path::String, data_path::String=""; kwargs...)
    sm = BridgeStan.StanModel(model_path, data_path; kwargs...)
    return PDMPModel(sm)
end

end # module
