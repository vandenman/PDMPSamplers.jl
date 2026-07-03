abstract type GridBoundaryProbe end
struct NoGridBoundaryProbe <: GridBoundaryProbe end

struct GridBoundaryProbeHandler{S,F,M,A} <: GridBoundaryProbe
    original_state::S
    flow::F
    model::M
end

GridBoundaryProbeHandler(original_state::S, flow::F, model::M, ::Type{A}) where {S,F,M,A} =
    GridBoundaryProbeHandler{S,F,M,A}(original_state, flow, model)

_is_bridgestan_probe_error(err) = err isa ErrorException && startswith(err.msg, "BridgeStan gradient failed")

function _get_rate_and_deriv_or_throw(::NoGridBoundaryProbe, state::AbstractPDMPState, flow::ContinuousDynamics, provider, add_rate::Bool, args...; t_valid::Float64, t_invalid::Float64)
    return get_rate_and_deriv(state, flow, provider, add_rate, args...)
end

function _get_rate_and_deriv_or_throw(probe::GridBoundaryProbeHandler, state::AbstractPDMPState, flow::ContinuousDynamics, provider, add_rate::Bool, args...; t_valid::Float64, t_invalid::Float64)
    try
        return get_rate_and_deriv(state, flow, provider, add_rate, args...)
    catch err
        err isa _ProbeFailureException && rethrow()
        if err isa ErrorException && err.msg == "bad hvp"
            throw(MethodError(get_rate_and_deriv, (state, flow, provider, add_rate, args...)))
        end
        if t_valid == t_invalid
            t_invalid = max(t_valid + eps(Float64), eps(Float64))
            try
                _throw_grid_boundary_error(probe, state, err; t_valid, t_invalid)
            catch boundary_err
                if boundary_err isa MethodError && boundary_err.f === _throw_grid_boundary_error
                    _is_bridgestan_probe_error(err) || throw(err)
                    ctx = BoundaryContext(copy(probe.original_state.ξ.x), copy(probe.original_state.ξ.θ), Float64(probe.original_state.t[]),
                        max(t_valid, 0.0), max(t_invalid, eps(Float64)), err, typeof(flow), typeof(probe).parameters[4])
                    throw(_ProbeFailureException(ctx))
                end
                rethrow()
            end
        end
        _throw_grid_boundary_error(probe, state, err; t_valid, t_invalid)
    end
end

function _compute_grid_gradient_or_throw!(state::AbstractPDMPState, original_state::AbstractPDMPState, flow::ContinuousDynamics, model::PDMPModel, cache,
    t_valid::Float64, t_invalid::Float64, ::NoGridBoundaryProbe)
    return compute_gradient!(state, model.grad, flow, cache)
end

function _compute_grid_gradient_or_throw!(state::AbstractPDMPState, original_state::AbstractPDMPState, flow::ContinuousDynamics, model::PDMPModel, cache,
    t_valid::Float64, t_invalid::Float64, probe::GridBoundaryProbeHandler)
    try
        return compute_gradient!(state, model.grad, flow, cache)
    catch err
        _throw_grid_boundary_error(probe, state, err; t_valid, t_invalid)
    end
end

# ── Support-boundary helpers for grid thinning ───────────────────────────────

function _grid_probe_failure_handler(
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    algorithm_type::Type
)
    return GridBoundaryProbeHandler(original_state, flow, model, algorithm_type)
end

function _throw_grid_boundary_error(
    probe::GridBoundaryProbeHandler{S,F,M,A},
    current_state::AbstractPDMPState,
    err::Exception;
    t_valid::Float64=0.0,
    t_invalid::Float64=current_state.t[] - probe.original_state.t[],
) where {S,F,M,A}
    return _throw_grid_boundary_error(
        current_state, probe.original_state, probe.flow, probe.model, err;
        t_valid, t_invalid, algorithm_type=A)
end

function _throw_grid_boundary_error(
    current_state::AbstractPDMPState,
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    err::Exception;
    t_valid::Float64=0.0,
    t_invalid::Float64=current_state.t[] - original_state.t[],
    algorithm_type::Type=GridThinningStrategy
)
    x0 = copy(original_state.ξ.x)
    v = copy(original_state.ξ.θ)
    t_valid = max(t_valid, 0.0)
    t_invalid = max(t_invalid, t_valid + eps(Float64))
    ctx = BoundaryContext(
        x0, v, Float64(original_state.t[]), t_valid, t_invalid,
        err, typeof(flow), algorithm_type,
    )
    if _is_bridgestan_probe_error(err)
        throw(_ProbeFailureException(ctx))
    end
    if _support_boundary_probe_is_valid(model, ctx, t_invalid)
        if err isa ErrorException && occursin("Outside support", err.msg)
            throw(_ProbeFailureException(ctx))
        end
        throw(MethodError(_throw_grid_boundary_error, (current_state, original_state, flow, model, err)))
    end
    throw(_ProbeFailureException(ctx))
end

function _throw_grid_boundary_error(
    current_state::AbstractPDMPState,
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel,
    err::_ProbeFailureException;
    kwargs...
)
    rethrow(err)
end

function _throw_grid_safety_limit_error(
    original_state::AbstractPDMPState,
    flow::ContinuousDynamics,
    model::PDMPModel;
    t_invalid::Float64,
    message::String,
    algorithm_type::Type=GridThinningStrategy
)
    t_invalid = max(t_invalid, eps(Float64))
    ctx = BoundaryContext(
        copy(original_state.ξ.x), copy(original_state.ξ.θ), Float64(original_state.t[]),
        0.0, t_invalid, ErrorException(message), typeof(flow), algorithm_type,
    )
    throw(_GridSafetyLimitException(ctx))
end
