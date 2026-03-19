# --- 1. Infrastructure ---
struct NoAdaptation <: AbstractAdapter end
adapt!(::NoAdaptation, args...; kwargs...) = nothing

struct SequenceAdapter{T} <: AbstractAdapter
    adapters::T
end

function adapt!(seq::SequenceAdapter, state, flow, grad, trace_mgr; kwargs...)
    for a in seq.adapters
        adapt!(a, state, flow, grad, trace_mgr; kwargs...)
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

function adapt!(ad::PreconditionerAdapter, state, flow, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    if phase === :warmup && (state.t[] - ad.last_update >= ad.dt)
        update_preconditioner!(flow, get_warmup_trace(trace_mgr), state, iszero(ad.no_updates_done))
        ad.last_update = state.t[]
        ad.no_updates_done += 1
    end
end

# B. Gradient Resampling (Subsampling)
mutable struct GradientResampler <: AbstractAdapter
    const dt::Float64
    last_update::Float64
end
GradientResampler() = GradientResampler(0.0, -Inf)

function adapt!(ad::GradientResampler, state, flow, grad::SubsampledGradient, trace_mgr; phase::Symbol=:warmup, kwargs...)
    if ad.dt <= 0.0 || (state.t[] - ad.last_update >= ad.dt)
        grad.resample_indices!(grad.nsub)
        ad.last_update = state.t[]
    end
end

# C. Anchor Updating (Control Variates)
mutable struct AnchorUpdater <: AbstractAdapter
    dt::Float64
    last_update::Float64
end

function adapt!(ad::AnchorUpdater, state, flow, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    if phase === :warmup && (state.t[] - ad.last_update >= ad.dt)
        grad.update_anchor!(get_warmup_trace(trace_mgr))
        ad.last_update = state.t[]
    end
end


# --- 3. The Factory Functions (Positional Args) ---

# --- Dynamics Factory ---
# Fallback: Swallow extra args (precond_dt, t0)
default_dynamics_adapter(::ContinuousDynamics, args...) = NoAdaptation()

# Specific:
function default_dynamics_adapter(::PreconditionedDynamics, precond_dt, t0, t_warmup=0.0)
    return PreconditionerAdapter(precond_dt, t0, 0, :default)
end


# --- Gradient Factory ---
# Fallback: Swallow extra args (t_warmup, t0)
default_gradient_adapter(::Any, args...) = NoAdaptation()

# Specific: anchor_dt derived from grad.no_anchor_updates so each chain respects its own setting
function default_gradient_adapter(grad::SubsampledGradient, t_warmup, t0)
    resampler = GradientResampler(grad.resample_dt, t0)
    grad.no_anchor_updates == 0 && return resampler
    return SequenceAdapter((
        resampler,
        AnchorUpdater(t_warmup / grad.no_anchor_updates, t0)
    ))
end


# --- 4. The Top-Level Interface ---

function default_adapter(flow::ContinuousDynamics, grad::GradientStrategy, precond_dt=10.0, t_warmup=100.0, t0=0.0)
    # Explicitly pass positional args to the sub-factories
    adpt_flow = default_dynamics_adapter(flow, precond_dt, t0, t_warmup)
    adpt_grad = default_gradient_adapter(grad, t_warmup, t0)

    # Clean return logic
    if adpt_flow isa NoAdaptation && adpt_grad isa NoAdaptation
        return NoAdaptation()
    end

    return SequenceAdapter((adpt_flow, adpt_grad))
end

function default_adapter(flow::MutableBoomerang, grad::SubsampledGradient, precond_dt=10.0, t_warmup=100.0, t0=0.0)
    adpt_flow = default_dynamics_adapter(flow, precond_dt, t0, 0.0)
    adpt_grad = default_gradient_adapter(grad, t_warmup, t0)
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
    sum_xy = stats.sum_xy
    sum_xy === nothing && error("BoomerangWarmupStats was not initialized for fullrank (no sum_xy)")

    # Use piecewise-constant mean (from sum_x) to match sum_xy, not sum_x_lin
    μ = zeros(d)
    for i in 1:d
        μ[i] = stats.coord_time[i] > 0 ? stats.sum_x[i] / stats.coord_time[i] : 0.0
    end
    C = zeros(d, d)

    for j in 1:d, i in j:d
        t_pair = min(stats.coord_time[i], stats.coord_time[j])
        if t_pair > 0
            C[i, j] = sum_xy[i, j] / t_pair - μ[i] * μ[j]
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


# --- 5b. Streaming time-weighted adaptation ---

"""
    WelfordBoomerangStats

Streaming time-weighted statistics accumulator for Boomerang adaptation.
Computes the same integrals as `BoomerangWarmupStats` (trapezoidal mean,
left-Riemann variance) but operates per-event without needing the trace
buffer. Stores previous-event state so each `welford_update!` integrates
exactly one trajectory segment.
"""
mutable struct WelfordBoomerangStats
    total_time::Float64
    sum_x_lin::Vector{Float64}                # ∑ (xᵢ + xᵢ₊₁)/2 · dt  (trapezoidal mean)
    sum_x::Vector{Float64}                    # ∑ xᵢ · dt              (left Riemann, for var)
    sum_x2::Vector{Float64}                   # ∑ xᵢ² · dt             (left Riemann, for var)
    sum_xy::Union{Nothing, Matrix{Float64}}   # ∑ xᵢ xⱼ · dt           (fullrank only)
    prev_x::Vector{Float64}
    prev_t::Float64
    initialized::Bool
end

function WelfordBoomerangStats(d::Integer; fullrank::Bool=false)
    WelfordBoomerangStats(
        0.0, zeros(d), zeros(d), zeros(d),
        fullrank ? zeros(d, d) : nothing,
        zeros(d), 0.0, false,
    )
end

function welford_update!(ws::WelfordBoomerangStats, x::AbstractVector, t::Float64)
    if !ws.initialized
        ws.prev_x .= x
        ws.prev_t = t
        ws.initialized = true
        return ws
    end

    dt = t - ws.prev_t
    dt <= 0 && return ws

    d = length(ws.sum_x)
    has_xy = ws.sum_xy !== nothing
    ws.total_time += dt

    @inbounds for i in 1:d
        xi = ws.prev_x[i]
        ws.sum_x_lin[i] += (xi + x[i]) / 2 * dt
        ws.sum_x[i] += xi * dt
        ws.sum_x2[i] += xi * xi * dt
    end

    if has_xy
        sum_xy = ws.sum_xy::Matrix{Float64}
        @inbounds for j in 1:d
            xj = ws.prev_x[j]
            for i in j:d
                xi = ws.prev_x[i]
                val = xi * xj * dt
                sum_xy[i, j] += val
                i != j && (sum_xy[j, i] += val)
            end
        end
    end

    ws.prev_x .= x
    ws.prev_t = t
    return ws
end

function stats_mean(ws::WelfordBoomerangStats)
    d = length(ws.sum_x_lin)
    μ = zeros(d)
    ws.total_time > 0 || return μ
    @inbounds for i in 1:d
        μ[i] = ws.sum_x_lin[i] / ws.total_time
    end
    return μ
end

function stats_var(ws::WelfordBoomerangStats)
    d = length(ws.sum_x)
    v = ones(d)
    ws.total_time > 0 || return v
    @inbounds for i in 1:d
        μi = ws.sum_x[i] / ws.total_time
        v[i] = max(ws.sum_x2[i] / ws.total_time - μi * μi, 0.0)
    end
    return v
end

stats_std(ws::WelfordBoomerangStats) = sqrt.(stats_var(ws))

function stats_cov(ws::WelfordBoomerangStats)
    d = length(ws.sum_x)
    sum_xy = ws.sum_xy
    sum_xy === nothing && error("WelfordBoomerangStats not initialized for fullrank (no sum_xy)")
    ws.total_time <= 0 && return Matrix{Float64}(I, d, d)
    T = ws.total_time
    C = zeros(d, d)
    @inbounds for j in 1:d
        μj = ws.sum_x[j] / T
        for i in j:d
            μi = ws.sum_x[i] / T
            C[i, j] = sum_xy[i, j] / T - μi * μj
            C[j, i] = C[i, j]
        end
    end
    return C
end


# --- In-place mean/var/cov (allocation-free) ---

_coord_time(stats::WelfordBoomerangStats, ::Int) = stats.total_time
_coord_time(stats::BoomerangWarmupStats, i::Int) = stats.coord_time[i]

function stats_mean!(μ::AbstractVector, stats::WelfordBoomerangStats)
    T = stats.total_time
    if T > 0
        @inbounds for i in eachindex(μ)
            μ[i] = stats.sum_x_lin[i] / T
        end
    else
        fill!(μ, 0.0)
    end
    return μ
end

function stats_mean!(μ::AbstractVector, stats::BoomerangWarmupStats)
    @inbounds for i in eachindex(μ)
        T = stats.coord_time[i]
        μ[i] = T > 0 ? stats.sum_x_lin[i] / T : 0.0
    end
    return μ
end

function stats_cov!(C::AbstractMatrix, stats::WelfordBoomerangStats)
    d = size(C, 1)
    sum_xy = stats.sum_xy::Matrix{Float64}
    T = stats.total_time
    if T <= 0
        fill!(C, 0.0)
        @inbounds for i in 1:d; C[i, i] = 1.0; end
        return C
    end
    @inbounds for j in 1:d
        μj = stats.sum_x[j] / T
        for i in j:d
            μi = stats.sum_x[i] / T
            cij = sum_xy[i, j] / T - μi * μj
            C[i, j] = cij
            C[j, i] = cij
        end
    end
    return C
end

function stats_cov!(C::AbstractMatrix, stats::BoomerangWarmupStats)
    d = size(C, 1)
    sum_xy = stats.sum_xy
    sum_xy === nothing && error("BoomerangWarmupStats was not initialized for fullrank (no sum_xy)")
    @inbounds for j in 1:d
        μj = stats.coord_time[j] > 0 ? stats.sum_x[j] / stats.coord_time[j] : 0.0
        for i in j:d
            t_pair = min(stats.coord_time[i], stats.coord_time[j])
            μi = stats.coord_time[i] > 0 ? stats.sum_x[i] / stats.coord_time[i] : 0.0
            if t_pair > 0
                cij = sum_xy[i, j] / t_pair - μi * μj
            else
                cij = i == j ? 1.0 : 0.0
            end
            C[i, j] = cij
            C[j, i] = cij
        end
    end
    return C
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

struct FullrankWorkspace
    vec_d::Vector{Float64}
    mat_dd::Matrix{Float64}
end
FullrankWorkspace(d::Int) = FullrankWorkspace(Vector{Float64}(undef, d), Matrix{Float64}(undef, d, d))

"""
    BoomerangAdapter{S, W} <: AbstractAdapter

Adapter for `MutableBoomerang` that learns μ and Γ during warmup.
Supports `:diagonal` (Phase 1) and `:fullrank` (Phase 2) schemes.

Parameterized on the stats accumulator type `S`, which defaults to
`WelfordBoomerangStats` for numerically stable per-event updates.
`W` is the workspace type (`Nothing` for diagonal, `FullrankWorkspace`
for fullrank/lowrank).
"""
mutable struct BoomerangAdapter{S, W} <: AbstractAdapter
    const base_dt::Float64
    last_update::Float64
    no_updates_done::Int
    const scheme::Symbol
    stats::S
    did_update::Bool
    const workspace::W
end

function BoomerangAdapter(base_dt::Float64, t0::Float64, d::Integer; scheme::Symbol=:diagonal)
    stats = WelfordBoomerangStats(d; fullrank=(scheme == :fullrank || scheme == :lowrank))
    needs_ws = scheme == :fullrank || scheme == :lowrank
    ws = needs_ws ? FullrankWorkspace(d) : nothing
    BoomerangAdapter(base_dt, t0, 0, scheme, stats, false, ws)
end

function adapt!(ad::BoomerangAdapter{<:WelfordBoomerangStats}, state, flow::MutableBoomerang, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    ad.did_update = false

    if phase === :warmup
        welford_update!(ad.stats, state.ξ.x, state.t[])
    end

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    if phase === :warmup && (state.t[] - ad.last_update >= dt_now)
        update_boomerang!(flow, ad.stats, Val(ad.scheme), ad.workspace)
        refresh_velocity!(state, flow)
        # After flow.μ changes, reset prev_x so next segment starts fresh
        ad.stats.prev_x .= state.ξ.x
        ad.stats.prev_t = state.t[]
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        ad.did_update = true
    end
end

function adapt!(ad::BoomerangAdapter{<:BoomerangWarmupStats}, state, flow::MutableBoomerang, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    update_stats!(ad.stats, trace_mgr)
    ad.did_update = false

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    if phase === :warmup && (state.t[] - ad.last_update >= dt_now)
        update_boomerang!(flow, ad.stats, Val(ad.scheme), ad.workspace)
        refresh_velocity!(state, flow)
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        ad.did_update = true
    end
end

# Fallback: if the flow is not MutableBoomerang, do nothing
adapt!(::BoomerangAdapter, state, flow, grad, trace_mgr; kwargs...) = nothing

# --- Query whether dynamics adaptation occurred (for sticky time invalidation) ---
did_dynamics_adapt(::AbstractAdapter) = false
did_dynamics_adapt(ad::BoomerangAdapter) = ad.did_update
did_dynamics_adapt(seq::SequenceAdapter) = any(did_dynamics_adapt, seq.adapters)


# --- 6. update_boomerang! implementations ---

const BOOM_DIAG_FLOOR = 1e-8

const AnyBoomerangStats = Union{BoomerangWarmupStats, WelfordBoomerangStats}
_has_data(stats::BoomerangWarmupStats) = !all(iszero, stats.coord_time)
_has_data(stats::WelfordBoomerangStats) = stats.total_time > 0
_shrinkage_time(stats::BoomerangWarmupStats) = maximum(stats.coord_time)
_shrinkage_time(stats::WelfordBoomerangStats) = stats.total_time

"""
    update_boomerang!(flow::MutableBoomerang, stats, ::Val{:diagonal}, ::Nothing)

Update `flow.μ` and diagonal `flow.Γ` from online sufficient statistics.
Zero-allocation: computes mean, variance, and Cholesky factors in a single pass.
"""
function update_boomerang!(flow::MutableBoomerang, stats::AnyBoomerangStats, ::Val{:diagonal}, ::Nothing)
    _has_data(stats) || return flow
    d = length(flow.μ)
    stats_mean!(flow.μ, stats)
    @inbounds for i in 1:d
        T = _coord_time(stats, i)
        if T > 0
            μi = stats.sum_x[i] / T
            σ2 = max(stats.sum_x2[i] / T - μi * μi, BOOM_DIAG_FLOOR^2)
        else
            σ2 = 1.0
        end
        γ = 1.0 / σ2
        flow.Γ[i, i] = γ
        s = sqrt(γ)
        flow.L[i, i] = s
        flow.ΣL[i, i] = 1.0 / s
    end
    flow.eigen_cache = nothing
    return flow
end


# --- Fullrank adaptation (Phase 2) ---

const BOOM_COND_MAX = 1e6
const BOOM_SHRINKAGE_BASE = 0.5
const BOOM_SHRINKAGE_DROP_TIME = 200.0

"""
    update_boomerang!(flow::MutableBoomerang, stats, ::Val{:fullrank}, ws)

Update `flow.μ` and dense `flow.Γ` from online sufficient statistics.
Estimates full covariance, applies shrinkage regularization, and recomputes
Cholesky factors `L` and `ΣL`. Falls back to diagonal update on factorization failure.

Uses preallocated workspace `ws` for all intermediate computations.
Explicit symmetrization `(M + M')/2` is unnecessary since `stats_cov!` produces
symmetric output and shrinkage preserves it; `Symmetric()` wrappers suffice.
"""
function update_boomerang!(flow::MutableBoomerang, stats::AnyBoomerangStats, ::Val{:fullrank}, ws::FullrankWorkspace)
    _has_data(stats) || return flow
    d = length(flow.μ)
    vec_d = ws.vec_d
    work = ws.mat_dd

    stats_mean!(flow.μ, stats)
    stats_cov!(work, stats)

    # Store diagonal for shrinkage anchor and fallback
    @inbounds for i in 1:d
        vec_d[i] = max(work[i, i], BOOM_DIAG_FLOOR^2)
    end

    # Shrinkage toward diagonal: strong initially, decays with observation count/time.
    total_time = _shrinkage_time(stats)
    α = BOOM_SHRINKAGE_BASE / (1.0 + total_time / 50.0)
    if total_time > BOOM_SHRINKAGE_DROP_TIME
        α = 0.0
    end
    if α > 0
        onemα = 1.0 - α
        @inbounds for j in 1:d, i in 1:d
            if i == j
                work[i, j] = onemα * work[i, j] + α * vec_d[i]
            else
                work[i, j] *= onemα
            end
        end
    end

    # Enforce diagonal floor
    @inbounds for i in 1:d
        work[i, i] = max(work[i, i], BOOM_DIAG_FLOOR^2)
    end

    # Cholesky of Σ → store L_Σ in flow.ΣL (in-place)
    copyto!(flow.ΣL.data, work)
    Σ_chol = try
        cholesky!(Symmetric(flow.ΣL.data, :L))
    catch e
        e isa PosDefException || rethrow()
        _fullrank_diagonal_fallback!(flow, vec_d)
        return flow
    end

    # Condition check (allocation-free)
    L_max = -Inf
    L_min = Inf
    @inbounds for i in 1:d
        l = flow.ΣL.data[i, i]
        L_max = max(L_max, l)
        L_min = min(L_min, l)
    end
    κ_approx = (L_max / max(L_min, BOOM_DIAG_FLOOR))^2
    if κ_approx > BOOM_COND_MAX
        # Extra shrinkage on work (still intact) and retry
        @inbounds for j in 1:d, i in 1:d
            if i == j
                work[i, j] = 0.5 * work[i, j] + 0.5 * vec_d[i]
            else
                work[i, j] *= 0.5
            end
        end
        copyto!(flow.ΣL.data, work)
        Σ_chol = try
            cholesky!(Symmetric(flow.ΣL.data, :L))
        catch e
            e isa PosDefException || rethrow()
            _fullrank_diagonal_fallback!(flow, vec_d)
            return flow
        end
    end

    # Compute Γ = Σ⁻¹ in-place via Cholesky solve: Σ Γ = I
    fill!(flow.Γ.data, 0.0)
    @inbounds for i in 1:d; flow.Γ.data[i, i] = 1.0; end
    ldiv!(Σ_chol, flow.Γ.data)

    # Cholesky of Γ → L_Γ in flow.L (in-place)
    copyto!(flow.L.data, flow.Γ.data)
    cholesky!(Symmetric(flow.L.data, :L))

    return flow
end

function _fullrank_diagonal_fallback!(flow::MutableBoomerang, σ2_diag::AbstractVector)
    d = length(flow.μ)

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

function update_boomerang!(flow::MutableBoomerang, stats::AnyBoomerangStats, ::Val{:lowrank}, ws::FullrankWorkspace)
    _has_data(stats) || return flow

    lrp = flow.Γ::LowRankPrecision
    d = length(flow.μ)
    r = length(lrp.Λ)
    work = ws.mat_dd
    vec_d = ws.vec_d

    stats_mean!(flow.μ, stats)
    stats_cov!(work, stats)

    # Shrinkage schedule (same as fullrank)
    total_time = _shrinkage_time(stats)
    α = BOOM_SHRINKAGE_BASE / (1.0 + total_time / 50.0)
    if total_time > BOOM_SHRINKAGE_DROP_TIME
        α = 0.0
    end
    if α > 0
        onemα = 1.0 - α
        @inbounds for j in 1:d, i in 1:d
            if i == j
                σ2_anchor = max(work[i, i], BOOM_DIAG_FLOOR^2)
                work[i, j] = onemα * work[i, j] + α * σ2_anchor
            else
                work[i, j] *= onemα
            end
        end
    end
    @inbounds for i in 1:d
        work[i, i] = max(work[i, i], BOOM_DIAG_FLOOR^2)
    end

    # Save Σ diagonal before eigen! destroys work
    @inbounds for i in 1:d
        vec_d[i] = work[i, i]
    end

    # In-place eigendecomposition (O(d³)); eigenvalues ascending for Symmetric in Julia
    F = eigen!(Symmetric(work, :L))

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
        lrp.D[i] = max(vec_d[i] - diag_vlv, BOOM_DIAG_FLOOR^2)
    end

    lowrank_precompute!(lrp)
    return flow
end


# --- 7. Factory dispatch for MutableBoomerang ---

# --- 7a. RefreshRateAdapter (Phase 2C: warmup-only λ_ref adaptation) ---

"""
    RefreshRateAdapter

Warmup-only adapter that tunes the refreshment rate `λ_ref` of a `MutableBoomerang`.
Runs in stage 2 (after reference measure adaptation stabilizes).
Uses cost-aware stochastic line search: minimizes total model evaluations
per unit of PDMP time by multiplicative perturbation of λ_ref.
"""
mutable struct RefreshRateAdapter <: AbstractAdapter
    const base_dt::Float64
    const min_λref::Float64
    const max_λref::Float64
    const min_start_time::Float64
    last_update::Float64
    no_updates_done::Int
    prev_total_evals::Int
    prev_pdmp_time::Float64
    prev_evals_per_time::Float64
    search_direction::Int
    did_update::Bool
end

function RefreshRateAdapter(base_dt::Float64, min_start_time::Float64;
    min_λref::Float64=0.01, max_λref::Float64=10.0)
    RefreshRateAdapter(base_dt, min_λref, max_λref, min_start_time,
        min_start_time, 0, 0, 0.0, Inf, 1, false)
end

function adapt!(ad::RefreshRateAdapter, state, flow::MutableBoomerang, grad, trace_mgr;
    phase::Symbol=:warmup, stats::Union{StatisticCounter,Nothing}=nothing, kwargs...)
    ad.did_update = false
    phase === :warmup || return
    stats === nothing && return
    state.t[] >= ad.min_start_time || return

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    (state.t[] - ad.last_update >= dt_now) || return

    total_evals = stats.∇f_calls + stats.∇²f_calls
    window_evals = total_evals - ad.prev_total_evals
    window_time = state.t[] - ad.prev_pdmp_time

    if window_time <= 0 || window_evals <= 0
        ad.prev_total_evals = total_evals
        ad.prev_pdmp_time = state.t[]
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        return
    end

    evals_per_time = window_evals / window_time

    if isinf(ad.prev_evals_per_time)
        ad.prev_evals_per_time = evals_per_time
    else
        if evals_per_time <= ad.prev_evals_per_time
            ad.prev_evals_per_time = evals_per_time
        else
            ad.search_direction *= -1
            ad.prev_evals_per_time = evals_per_time
        end
    end

    step = 1.5 ^ (1.0 / (1.0 + ad.no_updates_done * 0.5))
    if ad.search_direction > 0
        flow.λref = min(flow.λref * step, ad.max_λref)
    else
        flow.λref = max(flow.λref / step, ad.min_λref)
    end

    ad.prev_total_evals = total_evals
    ad.prev_pdmp_time = state.t[]
    ad.last_update = state.t[]
    ad.no_updates_done += 1
    ad.did_update = true
end

adapt!(::RefreshRateAdapter, state, flow, grad, trace_mgr; kwargs...) = nothing
did_dynamics_adapt(::RefreshRateAdapter) = false

function default_dynamics_adapter(flow::MutableBoomerang, precond_dt, t0, t_warmup=0.0)
    d = length(flow.μ)
    if flow.Γ isa Diagonal
        scheme = :diagonal
    elseif flow.Γ isa LowRankPrecision
        scheme = :lowrank
    else
        scheme = :fullrank
    end
    boom_adapter = BoomerangAdapter(Float64(precond_dt), Float64(t0), d; scheme=scheme)
    if t_warmup > 0
        λref_start = Float64(t0 + t_warmup * 0.5)
        λref_adapter = RefreshRateAdapter(Float64(precond_dt * 2), λref_start)
        return SequenceAdapter((boom_adapter, λref_adapter))
    end
    return boom_adapter
end