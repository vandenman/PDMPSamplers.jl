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
struct PDMPModel{G<:GradientStrategy,H,V}
    d::Int
    grad::G
    hvp::H
    vhv::V
    function PDMPModel(d::Integer, grad::GradientStrategy, hvp, vhv, grad_inplace::Bool, hvp_inplace::Bool)

        hvp_new = if hvp === nothing
            nothing
        elseif hvp_inplace
            out_hvp = zeros(d)
            InplaceHVP(hvp, out_hvp)
        else
            hvp
        end

        return new{typeof(grad),typeof(hvp_new),typeof(vhv)}(Int(d), grad, hvp_new, vhv)
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

    return PDMPModel(d, FullGradient(grad_f), hvp_f, vhv_f, false, false)
end

function with_stats(model::PDMPModel, stats::StatisticCounter)
    grad_new = with_stats(model.grad, stats)
    hvp_new = model.hvp === nothing ? nothing : WithStatsHVP(model.hvp, stats)
    vhv_new = model.vhv === nothing ? nothing : WithStatsVHV(model.vhv, stats)
    PDMPModel(model.d, grad_new, hvp_new, vhv_new, false, false)
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

"""
    _make_vhv_from_grad(grad_f!, d, backend)

Build a scalar directional curvature callable `(x, v, w) -> w'H(x)v` from
an in-place gradient `grad_f!(out, x)` using a JVP/pushforward.
"""
function _make_vhv_from_grad(grad_f!, d::Integer, backend::ADTypes.AbstractADType)
    x = zeros(d)
    v = zeros(d)
    w = zeros(d)
    primal_buf = zeros(d)
    g_scalar = (x, v, w) -> begin
        buf = x isa AbstractVector{Float64} ? primal_buf : similar(x)
        grad_f!(buf, x)
        dot(w, buf)
    end
    prep = DI.prepare_derivative(
        Base.Fix2(Base.Fix2(g_scalar, w), v),
        backend, zero(Float64)
    )
    return _VHVCallable(g_scalar, d, backend)
end

struct _VHVCallable{F,B<:ADTypes.AbstractADType}
    g_scalar::F
    d::Int
    backend::B
end

function (c::_VHVCallable)(x::AbstractVector, v::AbstractVector, w::AbstractVector)
    f_line = t -> c.g_scalar(x .+ t .* v, v, w)
    return DI.derivative(f_line, c.backend, zero(eltype(x)))
end

"""
    _make_vhv_from_logdensity(logp, d, backend)

Build a scalar directional curvature callable `(x, v, w) -> w'H(x)v` from
a log-density function using a JVP/pushforward. The sign is negated to match
the potential convention `U = -log p`.
"""
function _make_vhv_from_logdensity(logp, d::Integer, backend::ADTypes.AbstractADType)
    return _VHVFromLogDensity(logp, backend)
end

struct _VHVFromLogDensity{F,B<:ADTypes.AbstractADType}
    logp::F
    backend::B
end

function (c::_VHVFromLogDensity)(x::AbstractVector, v::AbstractVector, w::AbstractVector)
    g_scalar = t -> begin
        x_new = x .+ t .* v
        grad = DI.gradient(c.logp, c.backend, x_new)
        -dot(w, grad)
    end
    return DI.derivative(g_scalar, c.backend, zero(eltype(x)))
end
