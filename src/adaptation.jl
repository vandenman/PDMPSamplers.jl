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
incrementally from warmup trace segments.

Uses a hybrid integration strategy:
- Linear integrals (`sum_x_lin`) for mean estimation (μ-independent, unbiased)
- Sinusoidal integrals (`sum_x`, `sum_x2`, `sum_xy`) for variance/covariance
  estimation (exact for bounded oscillatory Boomerang trajectory)
"""
mutable struct BoomerangWarmupStats
    coord_time::Vector{Float64}   # per-coordinate integrated free time
    sum_x_lin::Vector{Float64}    # ∑ ∫ x_i(t) dt LINEAR (for mean, μ-independent)
    sum_x::Vector{Float64}        # ∑ ∫ x_i(t) dt SINUSOIDAL (for var/cov consistency)
    sum_x2::Vector{Float64}       # ∑ ∫ x_i²(t) dt SINUSOIDAL
    sum_xy::Union{Nothing, Matrix{Float64}}  # ∑ ∫ x_i x_j dt SINUSOIDAL (fullrank only)
    cursor::Int                   # index of last processed event in warmup trace
end

function BoomerangWarmupStats(d::Integer; fullrank::Bool=false)
    BoomerangWarmupStats(zeros(d), zeros(d), zeros(d), zeros(d), fullrank ? zeros(d, d) : nothing, 0)
end

"""
    stats_mean(stats::BoomerangWarmupStats)

Return the time-averaged mean from online stats.
"""
function stats_mean(stats::BoomerangWarmupStats)
    d = length(stats.sum_x_lin)
    μ = zeros(d)
    for i in 1:d
        μ[i] = stats.coord_time[i] > 0 ? stats.sum_x_lin[i] / stats.coord_time[i] : 0.0
    end
    return μ
end

"""
    stats_var(stats::BoomerangWarmupStats)

Return the time-averaged variance from online stats (using E[X²] - E[X]²).
"""
function stats_var(stats::BoomerangWarmupStats)
    d = length(stats.sum_x)
    v = ones(d)
    for i in 1:d
        if stats.coord_time[i] > 0
            μi = stats.sum_x[i] / stats.coord_time[i]
            v[i] = max(stats.sum_x2[i] / stats.coord_time[i] - μi^2, 0.0)
        end
    end
    return v
end

"""
    stats_std(stats::BoomerangWarmupStats)

Return the time-averaged standard deviation from online stats.
"""
stats_std(stats::BoomerangWarmupStats) = sqrt.(stats_var(stats))

"""
    stats_cov(stats::BoomerangWarmupStats)

Return the time-averaged covariance matrix from online stats.
Requires `sum_xy` (fullrank mode). Uses `sum_x` (not `sum_x_lin`)
to stay internally consistent with the piecewise-constant `sum_xy`.
"""
function stats_cov(stats::BoomerangWarmupStats)
    d = length(stats.sum_x)
    stats.sum_xy === nothing && error("BoomerangWarmupStats was not initialized for fullrank (no sum_xy)")

    # Use piecewise-constant mean (from sum_x) to match sum_xy, not sum_x_lin
    μ = zeros(d)
    for i in 1:d
        μ[i] = stats.coord_time[i] > 0 ? stats.sum_x[i] / stats.coord_time[i] : 0.0
    end
    C = zeros(d, d)

    for j in 1:d, i in j:d
        t_pair = min(stats.coord_time[i], stats.coord_time[j])
        if t_pair > 0
            C[i, j] = stats.sum_xy[i, j] / t_pair - μ[i] * μ[j]
        else
            C[i, j] = i == j ? 1.0 : 0.0
        end
        C[j, i] = C[i, j]
    end

    return C
end

"""
    update_stats!(stats::BoomerangWarmupStats, trace_mgr)

Incrementally accumulate mean and second-moment estimates from new
warmup trace segments (since last cursor position).

Uses a hybrid integration strategy:
- TRAPEZOIDAL rule for `sum_x_lin`: (x₀ + x₁)/2 · dt gives an accurate,
  μ-independent mean estimate with O(dt³) per-segment error → used for
  updating flow.μ.
- PIECEWISE-CONSTANT for `sum_x`, `sum_x2`, `sum_xy`: weights event
  positions by segment duration (left Riemann sum). These are μ-independent
  and internally consistent, so Var = E[x²] - E[x]² is robust even when
  flow.μ is far from the true mean → used for updating flow.Γ.
"""
function update_stats!(stats::BoomerangWarmupStats, trace_mgr)
    trace = get_warmup_trace(trace_mgr)

    n = length(trace.times)
    n < 2 && return stats

    if stats.cursor == 0
        stats.cursor = 1
    end

    has_xy = stats.sum_xy !== nothing

    d = size(trace.positions, 1)

    @inbounds for k in stats.cursor:(n - 1)
        dt = trace.times[k + 1] - trace.times[k]
        dt <= 0 && continue

        for i in 1:d
            xi = trace.positions[i, k]
            xi_next = trace.positions[i, k + 1]

            # Always accumulate observation time (denominator must include all segments)
            stats.coord_time[i] += dt

            # TRAPEZOIDAL mean integral (μ-independent, O(dt³) per segment)
            stats.sum_x_lin[i] += (xi + xi_next) / 2 * dt

            # PIECEWISE-CONSTANT (left Riemann sum, μ-independent, for variance)
            stats.sum_x[i] += xi * dt
            stats.sum_x2[i] += xi^2 * dt
        end

        if has_xy
            for j in 1:d
                xj = trace.positions[j, k]
                for i in j:d
                    xi = trace.positions[i, k]
                    val = xi * xj * dt
                    stats.sum_xy[i, j] += val
                    i != j && (stats.sum_xy[j, i] += val)
                end
            end
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
Supports `:diagonal` (Phase 1) and `:fullrank` (Phase 2) schemes.
"""
mutable struct BoomerangAdapter <: AbstractAdapter
    const base_dt::Float64
    last_update::Float64
    no_updates_done::Int
    const scheme::Symbol
    stats::BoomerangWarmupStats
    did_update::Bool
end

function BoomerangAdapter(base_dt::Float64, t0::Float64, d::Integer; scheme::Symbol=:diagonal)
    stats = BoomerangWarmupStats(d; fullrank=(scheme == :fullrank || scheme == :lowrank))
    BoomerangAdapter(base_dt, t0, 0, scheme, stats, false)
end

function adapt!(ad::BoomerangAdapter, state, flow::MutableBoomerang, grad, trace_mgr)
    # Always accumulate online stats from newly completed segments
    update_stats!(ad.stats, trace_mgr)
    ad.did_update = false

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    if state.t[] < trace_mgr.t_warmup && (state.t[] - ad.last_update >= dt_now)
        update_boomerang!(flow, ad.stats, Val(ad.scheme))
        refresh_velocity!(state, flow)
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        ad.did_update = true
    end
end

# Fallback: if the flow is not MutableBoomerang, do nothing
adapt!(::BoomerangAdapter, state, flow, grad, trace_mgr) = nothing

# --- Query whether dynamics adaptation occurred (for sticky time invalidation) ---
did_dynamics_adapt(::AbstractAdapter) = false
did_dynamics_adapt(ad::BoomerangAdapter) = ad.did_update
did_dynamics_adapt(seq::SequenceAdapter) = any(did_dynamics_adapt, seq.adapters)


# --- 6. update_boomerang! implementations ---

const BOOM_DIAG_FLOOR = 1e-8

"""
    update_boomerang!(flow::MutableBoomerang, stats::BoomerangWarmupStats, ::Val{:diagonal})

Update `flow.μ` and diagonal `flow.Γ` from online sufficient statistics.
Recomputes Cholesky factors `L` and `ΣL` without explicit matrix inverse.
"""
function update_boomerang!(flow::MutableBoomerang, stats::BoomerangWarmupStats, ::Val{:diagonal})
    all(iszero, stats.coord_time) && return flow

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

    # Recompute factors directly from diagonal entries (O(d), no dense allocation)
    γ = flow.Γ.diag
    flow.L = Diagonal(sqrt.(γ))
    flow.ΣL = Diagonal(1.0 ./ sqrt.(γ))
    flow.eigen_cache = nothing

    return flow
end


# --- Fullrank adaptation (Phase 2) ---

const BOOM_COND_MAX = 1e6
const BOOM_SHRINKAGE_BASE = 0.5
const BOOM_SHRINKAGE_DROP_TIME = 200.0

"""
    update_boomerang!(flow::MutableBoomerang, stats::BoomerangWarmupStats, ::Val{:fullrank})

Update `flow.μ` and dense `flow.Γ` from online sufficient statistics.
Estimates full covariance, applies shrinkage regularization, and recomputes
Cholesky factors `L` and `ΣL`. Falls back to diagonal update on factorization failure.
"""
function update_boomerang!(flow::MutableBoomerang, stats::BoomerangWarmupStats, ::Val{:fullrank})
    all(iszero, stats.coord_time) && return flow

    d = length(flow.μ)
    μ_est = stats_mean(stats)
    Σ_est = stats_cov(stats)

    # Diagonal fallback target (also used as shrinkage anchor)
    σ2_diag = max.(diag(Σ_est), BOOM_DIAG_FLOOR^2)
    Σ0 = Diagonal(σ2_diag)

    # Shrinkage toward diagonal: strong initially, decays with observation time.
    # After sufficient warmup (T > BOOM_SHRINKAGE_DROP_TIME), allow α to reach zero
    # for unbiased convergence, provided condition number is healthy.
    total_time = maximum(stats.coord_time)
    α = BOOM_SHRINKAGE_BASE / (1.0 + total_time / 50.0)
    if total_time > BOOM_SHRINKAGE_DROP_TIME
        α = 0.0
    end

    Σ_reg = (1.0 - α) .* Σ_est .+ α .* Matrix(Σ0)

    # Enforce symmetry and diagonal floor
    Σ_sym = Symmetric((Σ_reg + Σ_reg') / 2)
    for i in 1:d
        Σ_sym.data[i, i] = max(Σ_sym[i, i], BOOM_DIAG_FLOOR^2)
    end

    # Attempt Cholesky factorization of Σ
    Σ_chol = try
        cholesky(Σ_sym)
    catch e
        e isa PosDefException || rethrow()
        _fullrank_diagonal_fallback!(flow, μ_est, σ2_diag)
        return flow
    end

    # Condition check via Cholesky diagonal
    L_diag = diag(Σ_chol.L)
    κ_approx = (maximum(L_diag) / max(minimum(L_diag), BOOM_DIAG_FLOOR))^2
    if κ_approx > BOOM_COND_MAX
        # Increase shrinkage and retry
        Σ_sym = Symmetric(0.5 .* Σ_sym .+ 0.5 .* Matrix(Σ0))
        Σ_chol = try
            cholesky(Σ_sym)
        catch e
            e isa PosDefException || rethrow()
            _fullrank_diagonal_fallback!(flow, μ_est, σ2_diag)
            return flow
        end
    end

    # Compute Γ = Σ⁻¹ and its Cholesky
    Γ_new = inv(Σ_chol)
    Γ_sym = Symmetric(Γ_new)
    Γ_chol = cholesky(Γ_sym)

    # Update flow in-place
    copyto!(flow.μ, μ_est)
    flow.Γ.data .= Γ_sym
    flow.L.data .= Γ_chol.L
    flow.ΣL.data .= Σ_chol.L
    return flow
end

function _fullrank_diagonal_fallback!(flow::MutableBoomerang, μ_est, σ2_diag)
    copyto!(flow.μ, μ_est)
    d = length(μ_est)

    fill!(flow.Γ.data, 0.0)
    fill!(flow.L.data, 0.0)
    fill!(flow.ΣL.data, 0.0)

    for i in 1:d
        γ_i = 1.0 / max(σ2_diag[i], BOOM_DIAG_FLOOR^2)
        flow.Γ.data[i, i] = γ_i
        flow.L.data[i, i] = sqrt(γ_i)
        flow.ΣL.data[i, i] = 1.0 / sqrt(γ_i)
    end

    return nothing
end


# --- Low-rank adaptation ---

function update_boomerang!(flow::MutableBoomerang, stats::BoomerangWarmupStats, ::Val{:lowrank})
    all(iszero, stats.coord_time) && return flow

    lrp = flow.Γ::LowRankPrecision
    d = length(flow.μ)
    r = length(lrp.Λ)

    μ_est = stats_mean(stats)
    Σ_est = stats_cov(stats)

    σ2_diag = max.(diag(Σ_est), BOOM_DIAG_FLOOR^2)
    Σ0 = Diagonal(σ2_diag)

    # Shrinkage schedule (same as fullrank)
    total_time = maximum(stats.coord_time)
    α = BOOM_SHRINKAGE_BASE / (1.0 + total_time / 50.0)
    if total_time > BOOM_SHRINKAGE_DROP_TIME
        α = 0.0
    end

    Σ_reg = (1.0 - α) .* Σ_est .+ α .* Matrix(Σ0)
    Σ_sym = Symmetric((Σ_reg + Σ_reg') / 2)
    for i in 1:d
        Σ_sym.data[i, i] = max(Σ_sym[i, i], BOOM_DIAG_FLOOR^2)
    end

    # Eigendecomposition (O(d³)); eigenvalues ascending for Symmetric in Julia
    F = eigen(Σ_sym)

    # Top-r eigenpairs are the last r entries
    for k in 1:r
        j = d - r + k
        lrp.Λ[k] = max(F.values[j], BOOM_DIAG_FLOOR^2)
        for i in 1:d
            lrp.V[i, k] = F.vectors[i, j]
        end
    end

    # D = diag(Σ) - diag(V Λ V')  (residual diagonal, always ≥ 0)
    for i in 1:d
        diag_vlv = 0.0
        for k in 1:r
            diag_vlv += lrp.V[i, k]^2 * lrp.Λ[k]
        end
        lrp.D[i] = max(Σ_sym[i, i] - diag_vlv, BOOM_DIAG_FLOOR^2)
    end

    lowrank_precompute!(lrp)
    copyto!(flow.μ, μ_est)
    return flow
end


# --- 7. Factory dispatch for MutableBoomerang ---

function default_dynamics_adapter(flow::MutableBoomerang, precond_dt, t0)
    d = length(flow.μ)
    if flow.Γ isa Diagonal
        scheme = :diagonal
    elseif flow.Γ isa LowRankPrecision
        scheme = :lowrank
    else
        scheme = :fullrank
    end
    return BoomerangAdapter(Float64(precond_dt), Float64(t0), d; scheme=scheme)
end