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


# --- 5. Boomerang Adaptation (Phase 1: diagonal) ---

"""
    BoomerangWarmupStats

Online sufficient statistics for Boomerang adaptation, accumulated
incrementally from warmup trace segments. Tracks weighted mean and
second moment per coordinate using the Boomerang's analytic segment
integrals.
"""
mutable struct BoomerangWarmupStats
    total_time::Float64       # total integrated time
    sum_x::Vector{Float64}    # ∑ ∫ x_i(t) dt per coordinate
    sum_x2::Vector{Float64}   # ∑ ∫ x_i²(t) dt per coordinate
    cursor::Int               # index of last processed event in warmup trace
end

function BoomerangWarmupStats(d::Integer)
    BoomerangWarmupStats(0.0, zeros(d), zeros(d), 0)
end

"""
    stats_mean(stats::BoomerangWarmupStats)

Return the time-averaged mean from online stats.
"""
function stats_mean(stats::BoomerangWarmupStats)
    stats.total_time > 0 ? stats.sum_x / stats.total_time : zeros(length(stats.sum_x))
end

"""
    stats_var(stats::BoomerangWarmupStats)

Return the time-averaged variance from online stats (using E[X²] - E[X]²).
"""
function stats_var(stats::BoomerangWarmupStats)
    if stats.total_time <= 0
        return ones(length(stats.sum_x))
    end
    μ = stats.sum_x / stats.total_time
    ex2 = stats.sum_x2 / stats.total_time
    return max.(ex2 - μ .^ 2, 0.0)  # clamp to avoid negative from numerical noise
end

"""
    stats_std(stats::BoomerangWarmupStats)

Return the time-averaged standard deviation from online stats.
"""
stats_std(stats::BoomerangWarmupStats) = sqrt.(stats_var(stats))

"""
    update_stats!(stats::BoomerangWarmupStats, trace_mgr)

Incrementally accumulate mean and second-moment estimates from new
warmup trace segments (since last cursor position).

Uses piecewise-constant approximation (event positions weighted by
segment duration) rather than analytic Boomerang integrals, because
the flow's μ is mutable and may change between when segments were
recorded and when stats are computed.
"""
function update_stats!(stats::BoomerangWarmupStats, trace_mgr)
    trace = get_warmup_trace(trace_mgr)

    # PDMPTrace — segments indexed by consecutive events
    n = length(trace.times)
    n < 2 && return stats

    # Initialize cursor on first call
    if stats.cursor == 0
        stats.cursor = 1
    end

    # Process new segments: segment k uses position at event k, held for dt_k
    @inbounds for k in stats.cursor:(n - 1)
        t0 = trace.times[k]
        t1 = trace.times[k + 1]
        dt = t1 - t0
        dt <= 0 && continue

        stats.total_time += dt

        for i in axes(trace.positions, 1)
            x = trace.positions[i, k]
            stats.sum_x[i] += x * dt
            stats.sum_x2[i] += x^2 * dt
        end
    end

    stats.cursor = n
    return stats
end

"""
    adapt_interval(no_updates_done::Int, base_dt::Float64)

Geometrically growing adaptation interval: starts at `base_dt`,
doubles every 3 updates (caps at 32× base).
"""
function adapt_interval(no_updates_done::Int, base_dt::Float64)
    factor = min(2.0^(no_updates_done ÷ 3), 32.0)
    return base_dt * factor
end

"""
    BoomerangAdapter <: AbstractAdapter

Adapter for `MutableBoomerang` that learns μ and Γ during warmup.
Phase 1 supports `:diagonal` scheme only.
"""
mutable struct BoomerangAdapter <: AbstractAdapter
    const base_dt::Float64
    last_update::Float64
    no_updates_done::Int
    const scheme::Symbol
    stats::BoomerangWarmupStats
end

function BoomerangAdapter(base_dt::Float64, t0::Float64, d::Integer; scheme::Symbol=:diagonal)
    BoomerangAdapter(base_dt, t0, 0, scheme, BoomerangWarmupStats(d))
end

function adapt!(ad::BoomerangAdapter, state, flow::MutableBoomerang, grad, trace_mgr)
    # Always accumulate online stats from newly completed segments
    update_stats!(ad.stats, trace_mgr)

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    if state.t[] < trace_mgr.t_warmup && (state.t[] - ad.last_update >= dt_now)
        update_boomerang!(flow, ad.stats, Val(ad.scheme))
        refresh_velocity!(state, flow)
        ad.last_update = state.t[]
        ad.no_updates_done += 1
    end
end

# Fallback: if the flow is not MutableBoomerang, do nothing
adapt!(::BoomerangAdapter, state, flow, grad, trace_mgr) = nothing


# --- 6. update_boomerang! implementations ---

const BOOM_DIAG_FLOOR = 1e-8

"""
    update_boomerang!(flow::MutableBoomerang, stats::BoomerangWarmupStats, ::Val{:diagonal})

Update `flow.μ` and diagonal `flow.Γ` from online sufficient statistics.
Recomputes Cholesky factors `L` and `ΣL` without explicit matrix inverse.
"""
function update_boomerang!(flow::MutableBoomerang, stats::BoomerangWarmupStats, ::Val{:diagonal})
    stats.total_time <= 0 && return flow

    μ_est = stats_mean(stats)
    σ2_est = stats_var(stats)

    d = length(μ_est)

    # Update μ
    copyto!(flow.μ, μ_est)

    # Update diagonal Γ = diag(1/σ²), with floor on σ
    for i in 1:d
        σ2 = max(σ2_est[i], BOOM_DIAG_FLOOR^2)
        flow.Γ[i, i] = 1.0 / σ2
    end

    # Recompute Cholesky factors
    flow.L = cholesky(Symmetric(flow.Γ)).L
    # ΣL satisfies: ΣL * ΣL' = Γ⁻¹  (for sampling N(0, Γ⁻¹))
    # Use factorization-based solve: Γ⁻¹ = Symmetric(Γ) \ I
    flow.ΣL = cholesky(Symmetric(flow.Γ) \ I).L
    flow.eigen_cache = nothing

    return flow
end


# --- 7. Factory dispatch for MutableBoomerang ---

function default_dynamics_adapter(flow::MutableBoomerang, precond_dt, t0)
    d = length(flow.μ)
    return BoomerangAdapter(Float64(precond_dt), Float64(t0), d; scheme=:diagonal)
end