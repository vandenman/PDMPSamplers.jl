module PDMPSamplersBridgeStanExt

using PDMPSamplers
using BridgeStan
using Base.Libc.Libdl: dlsym, dlpath
import PDMPSamplers: PDMPModel, FullGradient, _SupportBoundaryProbeError, _last_gradient_potential, _potential_available, _potential

# ── FastBridgeStanModel ──────────────────────────────────────────────────────
# Eliminates per-call overhead from BridgeStan: caches dlsym function pointers
# and pre-allocates Ref{Float64}/Ref{Cstring} buffers that are reused across
# millions of gradient/HVP evaluations.

struct BridgeStanCallError <: _SupportBoundaryProbeError
    operation::Symbol
    code::Cint
    message::Cstring
end

_bridgestan_error_message(err::Base.RefValue{Cstring}) =
    isassigned(err) && err[] != C_NULL ? err[] : Cstring(C_NULL)

function Base.showerror(io::IO, err::BridgeStanCallError)
    print(io, "BridgeStan ", err.operation, " failed (code ", err.code, ")")
    err.message == C_NULL && return
    msg = unsafe_string(err.message)
    isempty(msg) || print(io, ": ", msg)
end

struct FastBridgeStanModel
    # all the pointers are "owned" by the StanModel, so we keep it here to avoid it from going out of scope
    owner::BridgeStan.StanModel
    lib::Ptr{Nothing}
    stanmodel::Ptr{BridgeStan.StanModelStruct}
    log_density_fn::Ptr{Nothing}
    grad_fn::Ptr{Nothing}
    hvp_fn::Ptr{Nothing}
    lp::Base.RefValue{Float64}
    potential_lp::Base.RefValue{Float64}
    err::Base.RefValue{Cstring}
    d::Int
end

function FastBridgeStanModel(sm::BridgeStan.StanModel)
    d = BridgeStan.param_unc_num(sm)
    log_density_fn = dlsym(sm.lib, :bs_log_density)
    grad_fn = dlsym(sm.lib, :bs_log_density_gradient)
    hvp_fn = dlsym(sm.lib, :bs_log_density_hessian_vector_product)
    lp = Ref(0.0)
    potential_lp = Ref(0.0)
    err = Ref{Cstring}()
    FastBridgeStanModel(sm, sm.lib, sm.stanmodel, log_density_fn, grad_fn, hvp_fn, lp, potential_lp, err, d)
end

function fast_potential(m::FastBridgeStanModel, q::Vector{Float64})
    m.potential_lp[] = 0.0
    rc = @ccall $(m.log_density_fn)(
        m.stanmodel::Ptr{BridgeStan.StanModelStruct},
        true::Bool, true::Bool,
        q::Ref{Cdouble}, m.potential_lp::Ref{Cdouble},
        m.err::Ref{Cstring},
    )::Cint
    if rc != 0
        throw(BridgeStanCallError(:log_density, rc, _bridgestan_error_message(m.err))) # COV_EXCL_LINE
    end
    return -m.potential_lp[]
end

function Base.copy(m::FastBridgeStanModel)
    # BridgeStan.StanModel is explicitly not thread-safe. PDMP multi-chain
    # sampling copies the model before spawning chains, so make that copy own a
    # fresh C++ model instance rather than sharing m.stanmodel.
    sm = BridgeStan.StanModel(dlpath(m.owner.lib), m.owner.data, m.owner.seed; warn=false)
    return FastBridgeStanModel(sm)
end

function fast_log_density_gradient!(m::FastBridgeStanModel, q::Vector{Float64}, out::Vector{Float64})
    m.lp[] = 0.0
    rc = @ccall $(m.grad_fn)(
        m.stanmodel::Ptr{BridgeStan.StanModelStruct},
        true::Bool, true::Bool,
        q::Ref{Cdouble}, m.lp::Ref{Cdouble}, out::Ref{Cdouble},
        m.err::Ref{Cstring},
    )::Cint
    if rc != 0
        throw(BridgeStanCallError(:gradient, rc, _bridgestan_error_message(m.err))) # COV_EXCL_LINE
    end
    return out
end

function fast_log_density_hvp!(m::FastBridgeStanModel, q::Vector{Float64}, v::Vector{Float64}, out::Vector{Float64})
    m.lp[] = 0.0
    rc = @ccall $(m.hvp_fn)(
        m.stanmodel::Ptr{BridgeStan.StanModelStruct},
        true::Bool, true::Bool,
        q::Ref{Cdouble}, v::Ref{Cdouble},
        m.lp::Ref{Cdouble}, out::Ref{Cdouble},
        m.err::Ref{Cstring},
    )::Cint
    if rc != 0
        throw(BridgeStanCallError(:hvp, rc, _bridgestan_error_message(m.err))) # COV_EXCL_LINE
    end
    return out
end

struct BridgeStanGradient{M<:FastBridgeStanModel} <: Function
    model::M
end

function (g::BridgeStanGradient)(out::Vector{Float64}, x::Vector{Float64})
    fast_log_density_gradient!(g.model, x, out)
    out .= .-out
    return out
end

Base.copy(g::BridgeStanGradient) = BridgeStanGradient(copy(g.model))
PDMPSamplers._copy_callable(g::BridgeStanGradient) = copy(g)
_potential_available(::BridgeStanGradient) = true
_potential(g::BridgeStanGradient, x::Vector{Float64}) = fast_potential(g.model, x)
_last_gradient_potential(model::PDMPModel{<:FullGradient{<:BridgeStanGradient}}) =
    -model.grad.f.model.lp[]

struct BridgeStanHVP{M<:FastBridgeStanModel,V<:Vector{Float64}} <: Function
    model::M
    out::V
end

function (h::BridgeStanHVP)(x::Vector{Float64}, v::Vector{Float64})
    fast_log_density_hvp!(h.model, x, v, h.out)
    h.out .= .-h.out
    return h.out
end

Base.copy(h::BridgeStanHVP) = BridgeStanHVP(copy(h.model), similar(h.out))
PDMPSamplers._copy_callable(h::BridgeStanHVP) = copy(h)

# ── PDMPModel constructors ───────────────────────────────────────────────────

"""
    PDMPModel(sm::BridgeStan.StanModel; hvp::Bool=false)

Construct a `PDMPModel` from a BridgeStan StanModel.

BridgeStan provides compiled Stan models with efficient gradient computations.
This constructor wraps the model's log density gradient function for use with PDMP samplers.

Uses `FastBridgeStanModel` internally to cache function pointers and pre-allocate
buffers, eliminating per-call dlsym/allocation overhead.

When `hvp=false` (default), directional curvature is computed via scalar finite
differences (`FiniteDiffVHV`), which reuses the base gradient from the rate
computation and adds only one extra gradient call per grid point. When `hvp=true`,
Stan's compiled Hessian-vector product is used instead.

# Arguments
- `sm::BridgeStan.StanModel`: A compiled Stan model
- `hvp::Bool=false`: Whether to use Stan's compiled HVP for curvature

# Example
```julia
using BridgeStan, PDMPSamplers

# Assume you have a compiled Stan model
sm = StanModel("path/to/model.so", "path/to/data.json")
pdmp_model = PDMPModel(sm)

flow = ZigZag(...)
alg = GridThinningStrategy()
trace, stats = pdmp_sample(x0, flow, pdmp_model, alg, 0.0, 10_000.0)
```

# Notes
- BridgeStan works in the unconstrained parameter space
- The model automatically handles the appropriate transformations
- Gradients are computed efficiently through Stan's autodiff system
"""
function PDMPModel(sm::BridgeStan.StanModel; hvp::Bool=false)

    fsm = FastBridgeStanModel(sm)
    d = fsm.d

    grad_f! = BridgeStanGradient(fsm)

    if hvp
        hvp_f = BridgeStanHVP(fsm, zeros(d))
        return PDMPModel(d, FullGradient(grad_f!), hvp_f, false, false)
    end

    return PDMPModel(d, FullGradient(grad_f!))
end

"""
    PDMPModel(model_path::String, data_path::String=""; kwargs...)

Construct a `PDMPModel` by loading a Stan model from file paths.

This is a convenience constructor that first creates a BridgeStan.StanModel
and then constructs the PDMPModel from it. HVP is disabled by default since
finite-difference curvature is faster for BridgeStan models.

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
function PDMPModel(model_path::String, data_path::String=""; hvp::Bool=false, kwargs...)
    sm = BridgeStan.StanModel(model_path, data_path; kwargs...)
    return PDMPModel(sm; hvp=hvp)
end

# TODO: move these to the R package only
# Precompile entry-point signatures.
# We cannot invoke these (no .so on disk at precompile time), but recording
# the specialisations caches the inference work for these constructors.
# import PrecompileTools
# PrecompileTools.@compile_workload begin
#     precompile(PDMPModel, (BridgeStan.StanModel,))
#     precompile(PDMPModel, (String, String))
#     precompile(PDMPModel, (String,))
# end

end # module
