module PDMPSamplersDynamicPPLExt

using PDMPSamplers
using DynamicPPL
import PDMPSamplers: PDMPModel, FullGradient
import DifferentiationInterface as DI
import ADTypes

"""
    PDMPModel(model::DynamicPPL.Model, backend::ADTypes.AbstractADType; needs_hvp::Bool=false)

Construct a `PDMPModel` from a DynamicPPL model.

# Arguments
- `model::DynamicPPL.Model`: A DynamicPPL model
- `backend::ADTypes.AbstractADType`: The automatic differentiation backend to use for gradients and HVPs
- `needs_hvp::Bool=false`: Whether to generate Hessian-vector product functions (required for GridThinningStrategy)

# Example
```julia
using DynamicPPL, Distributions, PDMPSamplers
import Mooncake

@model function normal_model(y)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0, 1), 0, Inf)
    y ~ Normal(μ, σ)
end

data = [1.2, 1.5, 1.8]
model = normal_model(data)
pdmp_model = PDMPModel(model, ADTypes.AutoMooncake(); needs_hvp=true)
```
"""
function PDMPModel(model::DynamicPPL.Model, backend::ADTypes.AbstractADType; needs_hvp::Bool=false)

    LDP = DynamicPPL.LogDensityProblems
    ldf = DynamicPPL.LogDensityFunction(model)
    d = LDP.dimension(ldf)

    # Wrap LogDensityFunction as a plain callable for DifferentiationInterface
    f = let ldf = ldf
        x -> LDP.logdensity(ldf, x)
    end

    # Prepare gradient function
    x = zeros(d)
    out = zeros(d)

    prep_grad = DI.prepare_gradient(f, backend, x)
    grad_f! = (out, x) -> begin
        DI.gradient!(f, out, prep_grad, backend, x)
        out .= .-out
    end

    # Prepare HVP if needed
    hvp_f = if needs_hvp
        backend == ADTypes.NoAutoDiff() && throw(ArgumentError(
            "Please provide a valid AD backend for Hessian-vector products when needs_hvp = true."
        ))

        out_hvp = zeros(d)
        θ = zeros(d)
        prep_hvp = DI.prepare_hvp(f, backend, x, (θ,))
        Base.Fix1((out, x, θ) -> begin
            DI.hvp!(f, (out,), prep_hvp, backend, x, (θ,))
            out .= .-out
        end, out_hvp)
    else
        nothing
    end

    return PDMPModel(d, FullGradient(grad_f!), hvp_f, false, false)
end

end # module
