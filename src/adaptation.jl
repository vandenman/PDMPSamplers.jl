# --- 1. Infrastructure ---
struct NoAdaptation <: AbstractAdapter end
adapt!(::NoAdaptation, args...) = nothing

struct SequenceAdapter{T} <: AbstractAdapter
    adapters::T
end

function adapt!(seq::SequenceAdapter, state, flow, grad, trace_mgr)
    for a in seq.adapters
        adapt!(a, state, flow, grad, trace_mgr)
    end
end


# --- 2. The Atomic Adapters ---

# A. Preconditioning
mutable struct PreconditionerAdapter <: AbstractAdapter
    const dt::Float64
    last_update::Float64
    no_updates_done::Int
    scheme::Symbol
end

function adapt!(ad::PreconditionerAdapter, state, flow, grad, trace_mgr)
    # Note: We use the raw 'flow' here. Dispatch on update_preconditioner! handles the check.
    if state.t[] < trace_mgr.t_warmup && (state.t[] - ad.last_update >= ad.dt)
        update_preconditioner!(flow, get_warmup_trace(trace_mgr), state, iszero(ad.no_updates_done))
        ad.last_update = state.t[]
        ad.no_updates_done += 1
    end
end

# B. Gradient Resampling (Subsampling)
struct GradientResampler <: AbstractAdapter end

# Dispatch specifically on SubsampledGradient for safety, or generic if 'resample_indices!' is standard
adapt!(::GradientResampler, state, flow, grad::SubsampledGradient, trace_mgr) = grad.resample_indices!(grad.nsub)

# C. Anchor Updating (Control Variates)
mutable struct AnchorUpdater <: AbstractAdapter
    dt::Float64
    last_update::Float64
end

function adapt!(ad::AnchorUpdater, state, flow, grad, trace_mgr)
    if state.t[] < trace_mgr.t_warmup && (state.t[] - ad.last_update >= ad.dt)
        grad.update_anchor!(get_warmup_trace(trace_mgr))
        ad.last_update = state.t[]
    end
end


# --- 3. The Factory Functions (Positional Args) ---

# --- Dynamics Factory ---
# Fallback: Swallow extra args (precond_dt, t0)
default_dynamics_adapter(::ContinuousDynamics, args...) = NoAdaptation()

# Specific:
function default_dynamics_adapter(::PreconditionedDynamics, precond_dt, t0)
    return PreconditionerAdapter(precond_dt, t0, 0, :default)
end


# --- Gradient Factory ---
# Fallback: Swallow extra args (anchor_dt, t0)
default_gradient_adapter(::Any, args...) = NoAdaptation()

# Specific:
function default_gradient_adapter(::SubsampledGradient, anchor_dt, t0)
    return SequenceAdapter((
        GradientResampler(),
        AnchorUpdater(anchor_dt, t0)
    ))
end


# --- 4. The Top-Level Interface ---

function default_adapter(flow::ContinuousDynamics, grad::GradientStrategy, precond_dt=10.0, anchor_dt=10.0, t0=0.0)
    # Explicitly pass positional args to the sub-factories
    adpt_flow = default_dynamics_adapter(flow, precond_dt, t0)
    adpt_grad = default_gradient_adapter(grad, anchor_dt, t0)

    # Clean return logic
    if adpt_flow isa NoAdaptation && adpt_grad isa NoAdaptation
        return NoAdaptation()
    end

    return SequenceAdapter((adpt_flow, adpt_grad))
end