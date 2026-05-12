# Support-boundary handling for PDMP trajectories.
#
# When a forward gradient evaluation fails during event-time construction,
# this module provides:
#   - :error      — fail fast (default)
#   - :line_search — localize the first invalid time via bisection, then error
#
# The line-search mode is for diagnostics and boundary localization, not for
# silently continuing past invalid regions.

"""
    SupportBoundaryOptions(; max_bisection_steps=60, time_rtol=1e-8, time_atol=1e-10,
                           clip_fraction=0.999999999, record_boundary_state=true)

Configuration for support-boundary handling.

# Fields
- `max_bisection_steps::Int`: Maximum number of bisection iterations.
- `time_rtol::Float64`: Relative tolerance for time localization.
- `time_atol::Float64`: Absolute tolerance for time localization.
- `clip_fraction::Float64`: Fraction of the last-valid time to use as a safe interior point.
- `record_boundary_state::Bool`: Whether to record the boundary state in the error.
"""
Base.@kwdef struct SupportBoundaryOptions
    max_bisection_steps::Int = 60
    time_rtol::Float64 = 1e-8
    time_atol::Float64 = 1e-10
    clip_fraction::Float64 = 1 - 1e-10
    record_boundary_state::Bool = true
end

"""
    SupportBoundaryError <: Exception

Thrown when a PDMP trajectory reaches a point where the target or gradient is undefined.

# Fields
- `message::String`: Human-readable description.
- `original_error::Any`: The original exception that triggered the boundary detection.
- `flow_type::DataType`: The flow type (e.g., `ZigZag`, `BouncyParticle`).
- `algorithm_type::DataType`: The algorithm type (e.g., `GridThinningStrategy`).
- `last_valid_time::Float64`: The last known valid time on the ray.
- `first_invalid_time::Float64`: The first known invalid time on the ray.
- `estimated_boundary_time::Union{Nothing,Float64}`: Localized boundary time (only when `localized=true`).
- `localized::Bool`: Whether bisection localization was attempted.
"""
struct SupportBoundaryError <: Exception
    message::String
    original_error::Any
    flow_type::Type
    algorithm_type::Type
    last_valid_time::Float64
    first_invalid_time::Float64
    estimated_boundary_time::Union{Nothing,Float64}
    localized::Bool
end

function Base.showerror(io::IO, e::SupportBoundaryError)
    print(io, "SupportBoundaryError: ", e.message)
    if e.localized
        print(io, "\n  Estimated boundary time: ", e.estimated_boundary_time)
    end
    print(io, "\n  Last valid time: ", e.last_valid_time,
              "\n  First invalid time: ", e.first_invalid_time)
    print(io, "\n  Flow: ", e.flow_type, ", Algorithm: ", e.algorithm_type)
    if !e.localized
        print(io, "\n  Hint: use support_boundary_mode=:line_search to localize the first invalid time.")
    end
    print(io, "\n  Original error: ")
    showerror(io, e.original_error)
end

# ── Internal boundary context ───────────────────────────────────────────────

"""
    BoundaryContext

Internal record built when a forward probe fails on a linear ray.

# Fields
- `x0::Vector{Float64}`: Ray origin (valid state position).
- `v::Vector{Float64}`: Deterministic direction (velocity).
- `t_valid::Float64`: Last known valid time on the ray (relative to x0).
- `t_invalid::Float64`: First known invalid time on the ray (relative to x0).
- `original_error::Any`: The original exception.
- `flow_type::Type`: The flow type.
- `algorithm_type::Type`: The algorithm type.
"""
struct BoundaryContext
    x0::Vector{Float64}
    v::Vector{Float64}
    t_valid::Float64
    t_invalid::Float64
    original_error::Any
    flow_type::Type
    algorithm_type::Type
end

# Internal wrapper — thrown by grid thinning when a gradient probe fails.
# Carries a BoundaryContext so the _step! catch block can decide whether
# to localize or fail fast.
struct _ProbeFailureException <: Exception
    ctx::BoundaryContext
end

# ── Bisection line search ────────────────────────────────────────────────────

"""
    localize_support_boundary!(model, ctx, opts) -> (estimated_boundary_time, safe_time)

Localize the support boundary on the ray `x(t) = x0 + t * v` using bisection
between `ctx.t_valid` (known valid) and `ctx.t_invalid` (known invalid).

Returns `(estimated_boundary_time, safe_time)` where `safe_time = clip_fraction * t_lo`
is a safe interior point just before the boundary.
"""
function localize_support_boundary!(model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    return _localize_support_boundary!(model.grad, model, ctx, opts)
end

function _localize_support_boundary_global!(grad_f, d::Int, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    x0 = ctx.x0
    v = ctx.v
    t_lo = ctx.t_valid
    t_hi = ctx.t_invalid

    x_mid = Vector{Float64}(undef, d)
    grad_buf = Vector{Float64}(undef, d)

    for _ in 1:opts.max_bisection_steps
        if t_hi - t_lo <= opts.time_atol + opts.time_rtol * max(abs(t_lo), abs(t_hi))
            break
        end

        t_mid = (t_lo + t_hi) / 2
        @. x_mid = x0 + t_mid * v

        valid = try
            grad_f(grad_buf, x_mid)
            true
        catch
            false
        end

        if valid
            t_lo = t_mid
        else
            t_hi = t_mid
        end
    end

    estimated_boundary_time = t_hi
    safe_time = opts.clip_fraction * t_lo

    return estimated_boundary_time, safe_time
end

function _localize_support_boundary!(::FullGradient, model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    return _localize_support_boundary_global!(model.grad.f, model.d, ctx, opts)
end

function _localize_support_boundary!(::SubsampledGradient, model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    # SubsampledGradient: use the same calling convention as FullGradient
    return _localize_support_boundary_global!(model.grad.f, model.d, ctx, opts)
end

function _localize_support_boundary!(::CoordinateWiseGradient, model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    x0 = ctx.x0
    v = ctx.v
    t_lo = ctx.t_valid
    t_hi = ctx.t_invalid
    d = model.d
    grad_f = model.grad.f

    x_mid = Vector{Float64}(undef, d)

    for _ in 1:opts.max_bisection_steps
        if t_hi - t_lo <= opts.time_atol + opts.time_rtol * max(abs(t_lo), abs(t_hi))
            break
        end

        t_mid = (t_lo + t_hi) / 2
        @. x_mid = x0 + t_mid * v

        valid = try
            for i in 1:d
                grad_f(x_mid, i)
            end
            true
        catch
            false
        end

        if valid
            t_lo = t_mid
        else
            t_hi = t_mid
        end
    end

    estimated_boundary_time = t_hi
    safe_time = opts.clip_fraction * t_lo

    return estimated_boundary_time, safe_time
end

# ── Error construction helpers ───────────────────────────────────────────────

function _build_boundary_error(ctx::BoundaryContext, opts::SupportBoundaryOptions;
    estimated_boundary_time::Union{Nothing,Float64}=nothing,
    localized::Bool=false)

    msg = if localized
        "The trajectory appears to have left the valid support of the target. " *
        "Boundary localized via bisection."
    else
        "The trajectory appears to have left the valid support of the target. " *
        "The gradient or target density became undefined during forward probing."
    end

    return SupportBoundaryError(
        msg,
        ctx.original_error,
        ctx.flow_type,
        ctx.algorithm_type,
        ctx.t_valid,
        ctx.t_invalid,
        estimated_boundary_time,
        localized,
    )
end

function _handle_boundary!(model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions, mode::Symbol)
    if mode === :error
        throw(_build_boundary_error(ctx, opts))
    elseif mode === :line_search
        tau_star, tau_safe = localize_support_boundary!(model, ctx, opts)
        throw(_build_boundary_error(ctx, opts; estimated_boundary_time=tau_star, localized=true))
    else
        throw(ArgumentError("Unknown support_boundary_mode: $mode"))
    end
end


