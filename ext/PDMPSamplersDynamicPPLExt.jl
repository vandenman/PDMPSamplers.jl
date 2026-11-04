module PDMPSamplersDynamicPPLExt

using PDMPSamplers
using DynamicPPL
import PDMPSamplers: PDMPModel, FullGradient, _make_vhv_from_logdensity, _make_joint_callable,
    _gradient_backend
import DifferentiationInterface as DI
import ADTypes

const LDP = DynamicPPL.LogDensityProblems

# Log density of `model` on the unconstrained (linked) scale, including the
# log-Jacobian of the transforms.
function _linked_logdensity_function(model::DynamicPPL.Model)
    vi = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    return DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi)
end

# A callable struct rather than a closure: AD backends such as Mooncake
# deepcopy the function during preparation, which fails on captured modules.
struct LinkedLogDensity{L}
    ldf::L
end
(f::LinkedLogDensity)(x) = LDP.logdensity(f.ldf, x)

"""
    PDMPModel(model::DynamicPPL.Model, backend::ADTypes.AbstractADType; needs_hvp::Bool=false)

Construct a `PDMPModel` from a DynamicPPL model.

The model is sampled on the unconstrained scale, as Turing's samplers do: a
parameter with bounded support, such as a standard deviation, is transformed
to the real line (for example to its logarithm), and the log density includes
the log-Jacobian of that transform. The starting point and the trace are on
this scale.

# Arguments
- `model::DynamicPPL.Model`: A DynamicPPL model
- `backend::ADTypes.AbstractADType`: The automatic differentiation backend. Reverse-mode
  backends such as `AutoMooncake()` cannot compute Hessian-vector products on their own;
  use them with `needs_hvp=false`, or pass `DI.SecondOrder(outer, inner)` to differentiate
  the `inner` gradient with the `outer` backend.
- `needs_hvp::Bool=false`: Whether to prepare Hessian-vector products for
  `GridThinningStrategy`. Without them, it computes curvature from gradients.

# Example
```julia
using DynamicPPL, Distributions, PDMPSamplers, ADTypes
import ForwardDiff

@model function normal_model(y)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0, 1), 0, Inf)
    y ~ Normal(μ, σ)
end

data = [1.2, 1.5, 1.8]
model = normal_model(data)
pdmp_model = PDMPModel(model, ADTypes.AutoForwardDiff(); needs_hvp=true)  # samples (μ, log σ)
```
"""
function PDMPModel(model::DynamicPPL.Model, backend::ADTypes.AbstractADType; needs_hvp::Bool=false)

    ldf = _linked_logdensity_function(model)
    d = LDP.dimension(ldf)
    f = LinkedLogDensity(ldf)

    # Prepare gradient function
    x = zeros(d)
    out = zeros(d)

    grad_backend = _gradient_backend(backend)
    prep_grad = DI.prepare_gradient(f, grad_backend, x)
    grad_f! = (out, x) -> begin
        DI.gradient!(f, out, prep_grad, grad_backend, x)
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

    vhv_f = if needs_hvp
        _make_vhv_from_logdensity(f, d, backend)
    else
        nothing
    end

    joint_f = if needs_hvp
        _make_joint_callable(f, d, backend)
    else
        nothing
    end

    return PDMPModel(d, FullGradient(grad_f!), hvp_f, vhv_f, false, joint_f)
end

end # module
