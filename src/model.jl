# TODO:
#
# this file needs some more thought.
# a lot of details work only for fullgradient
#
#




"""
    PDMPModel{G, H, V}

Encapsulates the geometry of the target distribution for samplers.

# Fields
- `grad::G`: The gradient strategy (e.g., `FullGradient` or `CoordinateWiseGradient`).
- `hvp::H`: The Hessian-Vector Product strategy (optional, for some algorithms).
- `vhv::V`: Directional curvature callable `(x, v, w) -> w'H(x)v` (optional, scalar fast path).

# Constructors

    PDMPModel(d::Int, grad::GradientStrategy, hvp::Union{Nothing,Function}=nothing)

Explicit construction.

    PDMPModel(d::LogDensity; backend, hvp = nothing)

Construct from a log-density `d`. Uses `ADTypes` backend to generate a `FullGradient` and optionally an HVP.

    PDMPModel(f::Function; backend, hvp = nothing)

Compatibility constructor. Wraps `f` in `FullGradient`.
"""
struct PDMPModel{G<:GradientStrategy,H,V,J}
    d::Int
    grad::G
    hvp::H
    vhv::V
    joint::J
    function PDMPModel(d::Integer, grad::GradientStrategy, hvp, vhv, grad_inplace::Bool, hvp_inplace::Bool, joint=nothing)

        hvp_new = if hvp === nothing
            nothing
        elseif hvp_inplace
            out_hvp = zeros(d)
            InplaceHVP(hvp, out_hvp)
        else
            hvp
        end

        return new{typeof(grad),typeof(hvp_new),typeof(vhv),typeof(joint)}(Int(d), grad, hvp_new, vhv, joint)
    end
end

function PDMPModel(d::Integer, grad::GradientStrategy, hvp, grad_inplace=true, hvp_inplace=true)
    PDMPModel(d, grad, hvp, nothing, grad_inplace, hvp_inplace)
end

function maybe_fix_grad(g::GradientStrategy, d::Integer)
    f = g.f
    f isa Base.Fix1 && return g

    f isa Function && return FullGradient(Base.Fix1((f), zeros(d)))
end


const GlobalGradientModel{H} = PDMPModel{<:GlobalGradientStrategy,H}
const CoordinateWiseGradientModel{H} = PDMPModel{<:CoordinateWiseGradientStrategy,H}

function PDMPModel(f::Function, args...; kwargs...)
    msg = "PDMPModel(f::Function, args..; kwargs...) is intentionally not implemented because it's unclear if this is a log density or gradient. " *
        "Use PDMPModel(LogDensity(f), args..; kwargs...) or PDMPModel(FullGradient(f), args..; kwargs...) instead."
    throw(ArgumentError(msg))
end

PDMPModel(d::Integer, grad::GradientStrategy) = PDMPModel(d, grad, nothing, nothing, true, true)

function PDMPModel(d::Integer, cv::SubsampledControlVariate)
    length(cv.anchor) == d || throw(DimensionMismatch("anchor length must match model dimension"))
    return PDMPModel(d, cv, cv.deterministic_hvp!, nothing, true, true)
end

PDMPModel(d::Integer, cv::SubsampledControlVariate, ::Nothing,
    grad_inplace=true, hvp_inplace=true) = PDMPModel(d, cv)

function PDMPModel(::Integer, ::SubsampledControlVariate, hvp,
    grad_inplace=true, hvp_inplace=true)
    throw(ArgumentError(
        "a subsampling model's deterministic HVP must be supplied through SubsampledControlVariate(deterministic_hvp! = ...), not as a separate PDMPModel HVP"))
end

function PDMPModel(d::Integer, grad::FullGradient, backend::ADTypes.AbstractADType, needs_hvp::Bool=false)

    hvp_f = if needs_hvp

        backend == ADTypes.NoAutoDiff() && throw(ArgumentError("Please provide a backend for Hessian-vector products when needs_hvp = true."))

        x = zeros(d)
        θ = zeros(d)
        primal_buf = zeros(d)
        f_scalar = (x, θ) -> begin
            buf = x isa AbstractVector{Float64} ? primal_buf : similar(x)
            grad.f(buf, x)
            dot(buf, θ)
        end
        prep = DI.prepare_gradient(f_scalar, backend, x, DI.Constant(θ))
        (out, x, v) -> begin
            DI.gradient!(f_scalar, out, prep, backend, x, DI.Constant(v))
        end
    else
        nothing
    end

    vhv_f = if needs_hvp
        _make_vhv_from_grad(grad.f, d, backend)
    else
        nothing
    end

    joint_f = if needs_hvp
        _make_joint_from_grad(grad.f, d, backend)
    else
        nothing
    end

    return PDMPModel(d, grad, hvp_f, vhv_f, true, true, joint_f)
end

"""
    LogDensity(f::Function)

Wrapper to indicate that the provided function `f` is a log-density function.
Used to instruct `PDMPModel` to compute gradients automatically.
"""
struct LogDensity{F}
    f::F
end
function PDMPModel(d::Integer, ldf::LogDensity, backend::ADTypes.AbstractADType=ADTypes.NoAutoDiff(), needs_hvp::Bool=false)

    # For now let's assume we create a closure that calls DI
    x = zeros(d)
    out = zeros(d)
    prep_grad = DI.prepare_gradient(ldf.f, backend, x)
    grad_f = (out, x) -> begin
        DI.gradient!(ldf.f, out, prep_grad, backend, x)
        out .= .-out
    end

    # If user wants HVP but didn't provide one, we can try to make one from d.f too?
    hvp_f = if needs_hvp
        out_hvp = zeros(d)
        θ = zeros(d)
        prep_hvp = DI.prepare_hvp(ldf.f, backend, x, (θ,))
        Base.Fix1((out, x, θ) -> begin
                DI.hvp!(ldf.f, (out,), prep_hvp, backend, x, (θ,))
                out .= .-out
            end, out_hvp)
    else
        nothing
    end

    vhv_f = if needs_hvp
        _make_vhv_from_logdensity(ldf.f, d, backend)
    else
        nothing
    end

    joint_f = if needs_hvp
        _make_joint_callable(ldf.f, d, backend)
    else
        nothing
    end

    return PDMPModel(d, FullGradient(grad_f), hvp_f, vhv_f, false, false, joint_f)
end

function with_stats(model::PDMPModel, stats::AbstractStatisticCounter)
    grad_new = with_stats(model.grad, stats)
    hvp_new = _model_hvp_with_stats(model, grad_new, stats)
    vhv_new = model.vhv === nothing ? nothing : WithStatsVHV(model.vhv, stats)
    joint_new = model.joint === nothing ? nothing : WithStatsJoint(model.joint, stats)
    PDMPModel(model.d, grad_new, hvp_new, vhv_new, false, false, joint_new)
end

_model_hvp_with_stats(model::PDMPModel, grad, stats) =
    model.hvp === nothing ? nothing : WithStatsHVP(model.hvp, stats)
_model_hvp_with_stats(model::PDMPModel{<:SubsampledControlVariate}, grad, stats) =
    grad.deterministic_hvp! === nothing ? nothing :
        WithStatsHVP(InplaceHVP(grad.deterministic_hvp!, zeros(model.d)), stats)

function set_active_set!(model::PDMPModel, free::BitVector)
    length(free) == model.d || throw(DimensionMismatch("active set length $(length(free)) does not match model dimension $(model.d)"))
    set_active_set!(model.grad, free)
    model.hvp !== nothing && set_active_set!(model.hvp, free)
    model.vhv !== nothing && set_active_set!(model.vhv, free)
    model.joint !== nothing && set_active_set!(model.joint, free)
    return nothing
end

_last_gradient_potential(::Any) = nothing
_last_gradient_potential(model::PDMPModel) = _last_gradient_potential(model.grad)
_last_gradient_potential(grad::FullGradient) = _last_gradient_potential(grad.f)
_last_gradient_potential(ws::WithStats) = _last_gradient_potential(ws.f)

_potential_available(::Any) = false
_potential_available(model::PDMPModel) = _potential_available(model.grad)
_potential_available(grad::FullGradient) = _potential_available(grad.f)
_potential_available(ws::WithStats) = _potential_available(ws.f)

function _potential(model::PDMPModel, x::Vector{Float64})
    _potential_available(model) || throw(ArgumentError("potential-only evaluation is unavailable for this model"))
    return _potential(model.grad, x)
end
_potential(grad::FullGradient, x::Vector{Float64}) = _potential(grad.f, x)
function _potential(ws::WithStats, x::Vector{Float64})
    _inc_counter_potential_calls(ws.stats)
    return _potential(ws.f, x)
end

struct InplaceHVP{F, O<:AbstractVector} <: Function
    f::F
    out::O
end
function (h::InplaceHVP)(x::AbstractVector, v::AbstractVector)
    h.f(h.out, x, v)
    return h.out
end
_copy_callable(h::InplaceHVP) = InplaceHVP(_copy_callable(h.f), copy(h.out))
set_active_set!(h::InplaceHVP, free::BitVector) = set_active_set!(h.f, free)

struct WithStatsHVP{F,S} <: Function
    f::F
    stats::S
end

(ws::WithStatsHVP)(x::AbstractVector, v::AbstractVector) = (_inc_counter_∇²f_calls(ws.stats); ws.f(x, v))
(ws::WithStatsHVP)(args...) = (_inc_counter_∇²f_calls(ws.stats); ws.f(args...))
set_active_set!(ws::WithStatsHVP, free::BitVector) = set_active_set!(ws.f, free)

struct WithStatsVHV{F,S} <: Function
    f::F
    stats::S
end

(ws::WithStatsVHV)(x::AbstractVector, v::AbstractVector, w::AbstractVector) = (_inc_counter_∇²f_calls(ws.stats); ws.f(x, v, w))
(ws::WithStatsVHV)(args...) = (_inc_counter_∇²f_calls(ws.stats); ws.f(args...))
set_active_set!(ws::WithStatsVHV, free::BitVector) = set_active_set!(ws.f, free)

struct WithStatsJoint{F,S} <: AbstractRateDerivativeProvider
    f::F
    stats::S
end
function (ws::WithStatsJoint)(x::AbstractVector, v::AbstractVector)
    _inc_counter_∇²f_calls(ws.stats)
    ws.f(x, v)
end
set_active_set!(ws::WithStatsJoint, free::BitVector) = set_active_set!(ws.f, free)

"""
    _make_vhv_from_grad(grad_f!, d, backend)

Build a scalar directional curvature callable `(x, v, w) -> w'H(x)v` from
an in-place gradient `grad_f!(out, x)` using a JVP/pushforward.
"""
function _make_vhv_from_grad(grad_f!, d::Integer, backend::ADTypes.AbstractADType)
    x_buf = zeros(d)
    v_buf = zeros(d)
    w_buf = zeros(d)
    phi = t -> begin
        xt = x_buf .+ t .* v_buf
        buf = similar(xt)
        grad_f!(buf, xt)
        dot(w_buf, buf)
    end
    prep = DI.prepare_derivative(phi, backend, zero(Float64))
    return _VHVCallable(grad_f!, phi, prep, backend, x_buf, v_buf, w_buf)
end

struct _VHVCallable{G,F,P,B<:ADTypes.AbstractADType}
    grad_f::G
    phi::F
    prep::P
    backend::B
    x_buf::Vector{Float64}
    v_buf::Vector{Float64}
    w_buf::Vector{Float64}
end

function (c::_VHVCallable)(x::AbstractVector, v::AbstractVector, w::AbstractVector)
    copyto!(c.x_buf, x)
    copyto!(c.v_buf, v)
    copyto!(c.w_buf, w)
    return DI.derivative(c.phi, c.prep, c.backend, zero(eltype(x)))
end

function _copy_callable(c::_VHVCallable)
    _make_vhv_from_grad(c.grad_f, length(c.x_buf), c.backend)
end

"""
    _make_vhv_from_logdensity(logp, d, backend)

Build a scalar directional curvature callable `(x, v, w) -> w'H(x)v` from
a log-density function using a JVP/pushforward. The sign is negated to match
the potential convention `U = -log p`.
"""
function _make_vhv_from_logdensity(logp, d::Integer, backend::ADTypes.AbstractADType)
    x_buf = zeros(d)
    v_buf = zeros(d)
    w_buf = zeros(d)
    phi = t -> begin
        xt = x_buf .+ t .* v_buf
        grad = DI.gradient(logp, backend, xt)
        -dot(w_buf, grad)
    end
    prep = DI.prepare_derivative(phi, backend, zero(Float64))
    return _VHVFromLogDensity(logp, phi, prep, backend, x_buf, v_buf, w_buf)
end

struct _VHVFromLogDensity{L,F,P,B<:ADTypes.AbstractADType}
    logp::L
    phi::F
    prep::P
    backend::B
    x_buf::Vector{Float64}
    v_buf::Vector{Float64}
    w_buf::Vector{Float64}
end

function (c::_VHVFromLogDensity)(x::AbstractVector, v::AbstractVector, w::AbstractVector)
    copyto!(c.x_buf, x)
    copyto!(c.v_buf, v)
    copyto!(c.w_buf, w)
    return DI.derivative(c.phi, c.prep, c.backend, zero(eltype(x)))
end

function _copy_callable(c::_VHVFromLogDensity)
    _make_vhv_from_logdensity(c.logp, length(c.x_buf), c.backend)
end

struct _JointCallable{L,F,P,B<:ADTypes.AbstractADType}
    logp::L
    phi::F
    prep::P
    backend::B
    x_buf::Vector{Float64}
    v_buf::Vector{Float64}
end

function _make_joint_callable(logp, d::Integer, backend::ADTypes.AbstractADType)
    x_buf = zeros(d)
    v_buf = zeros(d)
    phi = t -> -logp(x_buf .+ t .* v_buf)
    prep = DI.prepare_second_derivative(phi, backend, zero(Float64))
    return _JointCallable(logp, phi, prep, backend, x_buf, v_buf)
end

function (c::_JointCallable)(x::AbstractVector, v::AbstractVector)
    copyto!(c.x_buf, x)
    copyto!(c.v_buf, v)
    _, dphi, d2phi = DI.value_derivative_and_second_derivative(c.phi, c.prep, c.backend, zero(eltype(x)))
    return dphi, d2phi
end

function _copy_callable(c::_JointCallable)
    _make_joint_callable(c.logp, length(c.x_buf), c.backend)
end

struct _JointFromGradCallable{G,F,P,B<:ADTypes.AbstractADType}
    grad_f::G
    phi::F
    prep::P
    backend::B
    x_buf::Vector{Float64}
    v_buf::Vector{Float64}
end

function _make_joint_from_grad(grad_f!, d::Integer, backend::ADTypes.AbstractADType)
    x_buf = zeros(d)
    v_buf = zeros(d)
    phi = t -> begin
        xt = x_buf .+ t .* v_buf
        buf = similar(xt)
        grad_f!(buf, xt)
        dot(v_buf, buf)
    end
    prep = DI.prepare_derivative(phi, backend, zero(Float64))
    return _JointFromGradCallable(grad_f!, phi, prep, backend, x_buf, v_buf)
end

function (c::_JointFromGradCallable)(x::AbstractVector, v::AbstractVector)
    copyto!(c.x_buf, x)
    copyto!(c.v_buf, v)
    return DI.value_and_derivative(c.phi, c.prep, c.backend, zero(eltype(x)))
end

function _copy_callable(c::_JointFromGradCallable)
    _make_joint_from_grad(c.grad_f, length(c.x_buf), c.backend)
end
