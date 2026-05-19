# Support-boundary handling for PDMP trajectories.

"""
    SupportBoundaryOptions(; mode=:error, max_bisection_steps=60, time_rtol=1e-8,
                           time_atol=1e-10, clip_fraction=1 - 1e-10,
                           max_refresh_attempts=20, refresh_probe_time=1e-4,
                           min_safe_time=1e-12)

Configuration for support-boundary handling.

# Fields
- `mode::Symbol`: Boundary handling mode. One of `:error`, `:line_search`, or `:line_search_truncated_refresh`. The truncated-refresh mode
    localizes the support boundary, reruns event search up to a safe interior time, handles an ordinary event if one occurs first, and otherwise
    applies a BPS-family velocity refresh fallback.
- `max_bisection_steps::Int`: Maximum number of bisection iterations.
- `time_rtol::Float64`: Relative tolerance for time localization.
- `time_atol::Float64`: Absolute tolerance for time localization.
- `clip_fraction::Float64`: Fraction of the last-valid time to use as a safe interior point after localization. The truncated-refresh mode may
    apply an additional conservative cap before refreshing.
- `max_refresh_attempts::Int`: Maximum refreshed velocities to try if `:line_search_truncated_refresh` reaches the support boundary before an
    ordinary event.
- `refresh_probe_time::Float64`: Short forward probe time after a boundary refresh. A value of `0.0` disables the probe; otherwise the
    truncated-refresh mode may use a slightly longer scale-aware probe.
- `min_safe_time::Float64`: Minimum time gap used when clipping away from the localized boundary.
"""
Base.@kwdef struct SupportBoundaryOptions
    mode::Symbol = :error
    max_bisection_steps::Int = 60
    time_rtol::Float64 = 1e-8
    time_atol::Float64 = 1e-10
    clip_fraction::Float64 = 1 - 1e-10
    max_refresh_attempts::Int = 20
    refresh_probe_time::Float64 = 1e-4
    min_safe_time::Float64 = 1e-12
end

const _SUPPORT_BOUNDARY_MODES = (:error, :line_search, :line_search_truncated_refresh)

function _validate_support_boundary_options(opts::SupportBoundaryOptions)
    opts.mode in _SUPPORT_BOUNDARY_MODES || throw(ArgumentError("Unknown support-boundary mode: $(opts.mode)"))
    opts.max_bisection_steps >= 0 || throw(ArgumentError("max_bisection_steps must be non-negative"))
    opts.time_rtol >= 0.0 || throw(ArgumentError("time_rtol must be non-negative"))
    opts.time_atol >= 0.0 || throw(ArgumentError("time_atol must be non-negative"))
    0.0 < opts.clip_fraction <= 1.0 || throw(ArgumentError("clip_fraction must be in (0, 1]"))
    opts.max_refresh_attempts >= 0 || throw(ArgumentError("max_refresh_attempts must be non-negative"))
    opts.refresh_probe_time >= 0.0 || throw(ArgumentError("refresh_probe_time must be non-negative"))
    opts.min_safe_time >= 0.0 || throw(ArgumentError("min_safe_time must be non-negative"))
    return opts
end

struct SupportBoundaryLocalization
    last_valid_time::Float64
    first_invalid_time::Float64
    estimated_boundary_time::Float64
    safe_time::Float64
end

Base.iterate(loc::SupportBoundaryLocalization) = (loc.estimated_boundary_time, 2)
Base.iterate(loc::SupportBoundaryLocalization, state::Int) = state == 2 ? (loc.safe_time, 3) : nothing

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
    if !e.localized && occursin("The gradient or target density became undefined", e.message)
        print(io, "\n  Hint: use support_boundary_options=SupportBoundaryOptions(; mode=:line_search) to localize the first invalid time.")
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
    time0::Float64
    t_valid::Float64
    t_invalid::Float64
    original_error::Any
    flow_type::Type
    algorithm_type::Type
end

function BoundaryContext(
    x0::Vector{Float64},
    v::Vector{Float64},
    t_valid::Float64,
    t_invalid::Float64,
    original_error,
    flow_type::Type,
    algorithm_type::Type
)
    return BoundaryContext(x0, v, 0.0, t_valid, t_invalid, original_error, flow_type, algorithm_type)
end

# Internal wrapper — thrown by grid thinning when a gradient probe fails.
# Carries a BoundaryContext so the _step! catch block can decide whether
# to localize or fail fast.
struct _ProbeFailureException <: Exception
    ctx::BoundaryContext
end

struct _GridSafetyLimitException <: Exception
    ctx::BoundaryContext
end

struct _GradientProbeFailure <: Exception
    original_error::Any
end

struct _NonfiniteGradientProbe <: Exception
end

_gradient_probe_is_finite(gradient::AbstractArray) = all(isfinite, gradient)
_gradient_probe_is_finite(gradient::Number) = isfinite(gradient)
_gradient_probe_value(::Nothing, fallback) = fallback
_gradient_probe_value(gradient, fallback) = gradient

function _support_boundary_probe_is_valid(model::PDMPModel, ctx::BoundaryContext, t::Float64)
    return _support_boundary_probe_is_valid(model.grad, model, ctx, t)
end

function _support_boundary_probe_is_valid(grad::FullGradient, model::PDMPModel, ctx::BoundaryContext, t::Float64)
    x = Vector{Float64}(undef, model.d)
    grad_buf = Vector{Float64}(undef, model.d)
    @. x = ctx.x0 + t * ctx.v
    return try
        gradient = grad.f(grad_buf, x)
        _gradient_probe_is_finite(_gradient_probe_value(gradient, grad_buf))
    catch
        false
    end
end

function _support_boundary_probe_is_valid(grad::SubsampledGradient, model::PDMPModel, ctx::BoundaryContext, t::Float64)
    return _support_boundary_probe_is_valid(grad.full, model, ctx, t)
end

function _support_boundary_probe_is_valid(grad::CoordinateWiseGradient, model::PDMPModel, ctx::BoundaryContext, t::Float64)
    x = Vector{Float64}(undef, model.d)
    @. x = ctx.x0 + t * ctx.v
    return try
        for i in 1:model.d
            _gradient_probe_is_finite(grad.f(x, i)) || return false
        end
        true
    catch
        false
    end
end

# ── Bisection line search ────────────────────────────────────────────────────

"""
    localize_support_boundary!(model, ctx, opts) -> SupportBoundaryLocalization

Localize the support boundary on the ray `x(t) = x0 + t * v` using bisection
between `ctx.t_valid` (known valid) and `ctx.t_invalid` (known invalid).

The returned localization can also be destructured as `(estimated_boundary_time, safe_time)`
for compatibility with the original diagnostic API.
"""
function localize_support_boundary!(model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    return _localize_support_boundary!(model.grad, model, ctx, opts)
end

function _support_boundary_safe_time(t_lo::Float64, t_hi::Float64, opts::SupportBoundaryOptions)
    safe_time = opts.clip_fraction * t_lo
    if opts.min_safe_time > 0.0 && t_hi > opts.min_safe_time
        safe_time = min(safe_time, t_hi - opts.min_safe_time)
    end
    return max(0.0, safe_time)
end

function _localization_from_bracket(t_lo::Float64, t_hi::Float64, opts::SupportBoundaryOptions)
    safe_time = _support_boundary_safe_time(t_lo, t_hi, opts)
    return SupportBoundaryLocalization(t_lo, t_hi, t_hi, safe_time)
end

function _localize_support_boundary!(grad::FullGradient, model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    x0 = ctx.x0
    v = ctx.v
    t_lo = ctx.t_valid
    t_hi = ctx.t_invalid
    d = model.d
    grad_f = grad.f

    x_mid = Vector{Float64}(undef, d)
    grad_buf = Vector{Float64}(undef, d)

    for _ in 1:opts.max_bisection_steps
        if t_hi - t_lo <= opts.time_atol + opts.time_rtol * max(abs(t_lo), abs(t_hi))
            break
        end

        t_mid = (t_lo + t_hi) / 2
        @. x_mid = x0 + t_mid * v

        valid = try
            gradient = grad_f(grad_buf, x_mid)
            _gradient_probe_is_finite(_gradient_probe_value(gradient, grad_buf))
        catch
            false
        end

        if valid
            t_lo = t_mid
        else
            t_hi = t_mid
        end
    end

    return _localization_from_bracket(t_lo, t_hi, opts)
end

function _localize_support_boundary!(grad::SubsampledGradient, model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    return _localize_support_boundary!(grad.full, model, ctx, opts)
end

function _localize_support_boundary!(grad::CoordinateWiseGradient, model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    x0 = ctx.x0
    v = ctx.v
    t_lo = ctx.t_valid
    t_hi = ctx.t_invalid
    d = model.d
    grad_f = grad.f

    x_mid = Vector{Float64}(undef, d)

    for _ in 1:opts.max_bisection_steps
        if t_hi - t_lo <= opts.time_atol + opts.time_rtol * max(abs(t_lo), abs(t_hi))
            break
        end

        t_mid = (t_lo + t_hi) / 2
        @. x_mid = x0 + t_mid * v

        valid = try
            finite = true
            for i in 1:d
                if !_gradient_probe_is_finite(grad_f(x_mid, i))
                    finite = false
                    break
                end
            end
            finite
        catch
            false
        end

        if valid
            t_lo = t_mid
        else
            t_hi = t_mid
        end
    end

    return _localization_from_bracket(t_lo, t_hi, opts)
end

# ── Error construction helpers ───────────────────────────────────────────────

function _build_boundary_error(ctx::BoundaryContext, opts::SupportBoundaryOptions;
    estimated_boundary_time::Union{Nothing,Float64}=nothing,
    localized::Bool=false,
    localization::Union{Nothing,SupportBoundaryLocalization}=nothing,
    message::Union{Nothing,String}=nothing)

    msg = if message !== nothing
        message
    elseif localized
        "The trajectory appears to have left the valid support of the target. " *
        "Boundary localized via bisection."
    else
        "The trajectory appears to have left the valid support of the target. " *
        "The gradient or target density became undefined during forward probing."
    end

    last_valid_time = localization === nothing ? ctx.t_valid : localization.last_valid_time
    first_invalid_time = localization === nothing ? ctx.t_invalid : localization.first_invalid_time
    boundary_time = localization === nothing ? estimated_boundary_time : localization.estimated_boundary_time

    return SupportBoundaryError(
        msg,
        ctx.original_error,
        ctx.flow_type,
        ctx.algorithm_type,
        last_valid_time,
        first_invalid_time,
        boundary_time,
        localized,
    )
end

function _handle_boundary!(model::PDMPModel, ctx::BoundaryContext, opts::SupportBoundaryOptions)
    mode = opts.mode
    if mode === :error
        throw(_build_boundary_error(ctx, opts))
    elseif mode === :line_search || mode === :line_search_truncated_refresh
        localization = localize_support_boundary!(model, ctx, opts)
        throw(_build_boundary_error(ctx, opts; localized=true, localization))
    else
        throw(ArgumentError("Unknown support-boundary mode: $mode"))
    end
end


