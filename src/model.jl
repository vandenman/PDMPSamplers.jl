# TODO:
#
# this file needs some more thought.
# a lot of details work only for fullgradient
# but we can make it more general, so it also works for subsampled gradients
#
#




"""
    PDMPModel{G, H}

Encapsulates the geometry of the target distribution for samplers.

# Fields
- `grad::G`: The gradient strategy (e.g., `FullGradient` or `CoordinateWiseGradient`).
- `hvp::H`: The Hessian-Vector Product strategy (optional, for some algorithms).

# Constructors

    PDMPModel(d::Int, grad::GradientStrategy, hvp::Union{Nothing,Function}=nothing)

Explicit construction.

    PDMPModel(d::LogDensity; backend, hvp = nothing)

Construct from a log-density `d`. Uses `ADTypes` backend to generate a `FullGradient` and optionally an HVP.

    PDMPModel(f::Function; backend, hvp = nothing)

Compatibility constructor. Wraps `f` in `FullGradient`.
"""
struct PDMPModel{G<:GradientStrategy,H<:Union{Nothing,Function}}
    d::Int
    grad::G
    hvp::H
    function PDMPModel(d::Integer, grad::GradientStrategy, hvp, grad_inplace=true, hvp_inplace=true)

        # grad_new = if grad_inplace
        #     out_grad = zeros(d)
        #     Base.Fix1(grad, out_grad)
        # else
        #     grad.f
        # end
        hvp_new = if hvp === nothing
            nothing
        elseif hvp_inplace
            out_hvp = zeros(d)
            Base.Fix1(hvp, out_hvp)
        else
            hvp
        end

        # TODO: to use this form we'd need to adjust compute_gradient! everywhere as well!
        # that form is better though
        # return PDMPModel(d, maybe_fix_grad(grad, d), hvp_f)
        return new{typeof(grad),typeof(hvp_new)}(Int(d), grad, hvp_new)
    end
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

PDMPModel(d::Integer, grad::GradientStrategy) = PDMPModel(d, grad, nothing)

function PDMPModel(d::Integer, grad::FullGradient, backend::ADTypes.AbstractADType, needs_hvp::Bool=false)

    hvp_f = if needs_hvp

        backend == ADTypes.NoAutoDiff() && throw(ArgumentError("Please provide a backend for Hessian-vector products when needs_hvp = true."))

        x = zeros(d)
        θ = zeros(d)
        out_hvp = zeros(d)
        f! = (x, θ) -> dot(compute_gradient!(grad, x, out_hvp), θ)
        prep = DI.prepare_gradient(f!, backend, x, DI.Constant(θ))
        (out, x, v) -> begin
            DI.gradient!(f!, out, prep, backend, x, DI.Constant(v))
            out .= .-out
        end
    else
        nothing
    end


    return PDMPModel(d, grad, hvp_f)
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

    return PDMPModel(d, FullGradient(grad_f), hvp_f, false, false)
end

function with_stats(model::PDMPModel{<:GradientStrategy,<:Function}, stats::StatisticCounter)
    hvp = (x, θ) -> begin
        stats.∇²f_calls += 1
        model.hvp(x, θ)
    end
    PDMPModel(model.d, with_stats(model.grad, stats), hvp, false, false)
end
function with_stats(model::PDMPModel{<:GradientStrategy,<:Nothing}, stats::StatisticCounter)
    PDMPModel(model.d, with_stats(model.grad, stats), nothing, false, false)
end
