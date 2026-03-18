# TODO:
#
# this file needs some more thought.
# a lot of details work only for fullgradient
# but we can make it more general, so it also works for subsampled gradients
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

    # TODO: needs similar handling for SubsampledGradient!
    f isa Function && return FullGradient(Base.Fix1((f), zeros(d)))
end


const GlobalGradientModel{H} = PDMPModel{<:GlobalGradientStrategy,H}
const CoordinateWiseGradientModel{H} = PDMPModel{<:CoordinateWiseGradientStrategy,H}

function PDMPModel(f::Function, args...; kwargs...)
    throw(ArgumentError("PDMPModel(f::Function, args..; kwargs...) is intentionally not implemented because it's unclear if this is a log density or gradient. Use PDMPModel(LogDensity(f), args..; kwargs...) or PDMPModel(FullGradient(f), args..; kwargs...) instead."))
end

PDMPModel(d::Integer, grad::GradientStrategy) = PDMPModel(d, grad, nothing, nothing, true, true)

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

    return PDMPModel(d, grad, hvp_f, vhv_f, true, true)
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

function with_stats(model::PDMPModel, stats::StatisticCounter)
    grad_new = with_stats(model.grad, stats)
    hvp_new = model.hvp === nothing ? nothing : WithStatsHVP(model.hvp, stats)
    vhv_new = model.vhv === nothing ? nothing : WithStatsVHV(model.vhv, stats)
    joint_new = model.joint === nothing ? nothing : WithStatsJoint(model.joint, stats)
    PDMPModel(model.d, grad_new, hvp_new, vhv_new, false, false, joint_new)
end

struct InplaceHVP{F, O<:AbstractVector} <: Function
    f::F
    out::O
end
(h::InplaceHVP)(x::AbstractVector, v::AbstractVector) = h.f(h.out, x, v)
_copy_callable(h::InplaceHVP) = InplaceHVP(_copy_callable(h.f), copy(h.out))

struct WithStatsHVP{F,S} <: Function
    f::F
    stats::S
end
(ws::WithStatsHVP)(x::AbstractVector, v::AbstractVector) = (ws.stats.∇²f_calls += 1; ws.f(x, v))
(ws::WithStatsHVP)(args...) = (ws.stats.∇²f_calls += 1; ws.f(args...))

struct WithStatsVHV{F,S} <: Function
    f::F
    stats::S
end
(ws::WithStatsVHV)(x::AbstractVector, v::AbstractVector, w::AbstractVector) = (ws.stats.∇²f_calls += 1; ws.f(x, v, w))
(ws::WithStatsVHV)(args...) = (ws.stats.∇²f_calls += 1; ws.f(args...))

struct WithStatsJoint{F,S} <: Function
    f::F
    stats::S
end
function (ws::WithStatsJoint)(x::AbstractVector, v::AbstractVector)
    ws.stats.∇²f_calls += 1
    ws.f(x, v)
end

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
