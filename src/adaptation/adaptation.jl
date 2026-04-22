# --- 1. Infrastructure ---
struct NoAdaptation <: AbstractAdapter end
adapt!(::Random.AbstractRNG, ::NoAdaptation, args...; kwargs...) = nothing
adapt!(ad::AbstractAdapter, args...; kwargs...) = adapt!(Random.default_rng(), ad, args...;  kwargs...)

struct SequenceAdapter{T} <: AbstractAdapter
    adapters::T
end

function adapt!(rng::Random.AbstractRNG, seq::SequenceAdapter, state, flow, grad, trace_mgr; kwargs...)
    for a in seq.adapters
        adapt!(rng, a, state, flow, grad, trace_mgr; kwargs...)
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

function adapt!(rng::Random.AbstractRNG, ad::PreconditionerAdapter, state, flow, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    if phase === :warmup && (state.t[] - ad.last_update >= ad.dt)
        update_preconditioner!(rng, flow, get_warmup_trace(trace_mgr), state, iszero(ad.no_updates_done))
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

function adapt!(::Random.AbstractRNG, ad::GradientResampler, state, flow, grad::SubsampledGradient, trace_mgr; phase::Symbol=:warmup, kwargs...)
    if ad.dt <= 0.0 || (state.t[] - ad.last_update >= ad.dt)
        grad.resample_indices!(grad.nsub)
        ad.last_update = state.t[]
    end
end

# C. Anchor Updating (Control Variates)
mutable struct AnchorUpdater <: AbstractAdapter
    dt::Float64
    last_update::Float64
    warmup_only::Bool
end
AnchorUpdater(dt::Float64, last_update::Float64) = AnchorUpdater(dt, last_update, true)

_has_integrable_segment(::Nothing) = false

function _has_integrable_segment(trace)
    first_event = iterate(trace)
    first_event === nothing && return false
    second_event = iterate(trace, first_event[2])
    return second_event !== nothing
end

function adapt!(::Random.AbstractRNG, ad::AnchorUpdater, state, flow, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    if (state.t[] - ad.last_update >= ad.dt)
        if phase === :warmup
            trace = get_warmup_trace(trace_mgr)
            if _has_integrable_segment(trace)
                grad.update_anchor!(trace)
                ad.last_update = state.t[]
            end
        elseif !ad.warmup_only
            trace = get_main_trace(trace_mgr)
            if _has_integrable_segment(trace)
                grad.update_anchor!(trace)
                ad.last_update = state.t[]
            end
        end
    end
end

# D. Anchor Bank (spatial cache of anchors with nearest-neighbor selection)
mutable struct AnchorBankAdapter{F1, F2} <: AbstractAdapter
    select_fn!::F1
    update_fn!::F2
    update_dt::Float64
    last_update::Float64
    warmup_only::Bool
end

function adapt!(::Random.AbstractRNG, ad::AnchorBankAdapter, state, flow, grad, trace_mgr;
                phase::Symbol=:warmup, kwargs...)
    ad.select_fn!(state.ξ.x)

    if state.t[] - ad.last_update >= ad.update_dt
        if phase === :warmup
            trace = get_warmup_trace(trace_mgr)
            if _has_integrable_segment(trace)
                ad.update_fn!(trace)
                ad.last_update = state.t[]
            end
        elseif !ad.warmup_only
            trace = get_main_trace(trace_mgr)
            if _has_integrable_segment(trace)
                ad.update_fn!(trace)
                ad.last_update = state.t[]
            end
        end
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

function default_adapter(flow::ContinuousDynamics, grad::SubsampledGradient,
                         bank_adapter::AnchorBankAdapter,
                         precond_dt=10.0, t_warmup=100.0, t0=0.0)
    adpt_flow = default_dynamics_adapter(flow, precond_dt, t0, t_warmup)
    resampler = GradientResampler(grad.resample_dt, t0)
    bank_adapter.update_dt = grad.no_anchor_updates > 0 ? t_warmup / grad.no_anchor_updates : 0.0
    bank_adapter.last_update = t0
    return SequenceAdapter((adpt_flow, resampler, bank_adapter))
end

function default_adapter(flow::MutableBoomerang, grad::SubsampledGradient, precond_dt=10.0, t_warmup=100.0, t0=0.0)
    adpt_flow = default_dynamics_adapter(flow, precond_dt, t0, 0.0)
    adpt_grad = default_gradient_adapter(grad, t_warmup, t0)
    return SequenceAdapter((adpt_flow, adpt_grad))
end

function default_adapter(flow::MutableBoomerang, grad::SubsampledGradient,
                         bank_adapter::AnchorBankAdapter,
                         precond_dt=10.0, t_warmup=100.0, t0=0.0)
    adpt_flow = default_dynamics_adapter(flow, precond_dt, t0, 0.0)
    resampler = GradientResampler(grad.resample_dt, t0)
    bank_adapter.update_dt = grad.no_anchor_updates > 0 ? t_warmup / grad.no_anchor_updates : 0.0
    bank_adapter.last_update = t0
    return SequenceAdapter((adpt_flow, resampler, bank_adapter))
end


# --- 5. Boomerang Adaptation ---

"""
    WelfordBoomerangStats

Exact continuous-time raw-moment accumulator for Boomerang adaptation.
Each segment `[prev_t, t]` is integrated analytically using the exact
sinusoidal Boomerang trajectory with the reference `flow.μ` active at
that time. Stores raw sufficient statistics:

- `total_time T = Σ dt`
- `sum_x_dt S1 = Σ ∫ x(t) dt`
- `sum_x2_dt S2 = Σ ∫ x(t).² dt`
- `sum_xy_dt S11 = Σ ∫ x(t) x(t)' dt` (fullrank/lowrank only)

Derived moments: m̂ = S1/T, Var̂ = S2/T - m̂², Ĉov = S11/T - m̂ m̂'.
"""
mutable struct WelfordBoomerangStats
    total_time::Float64
    sum_x_dt::Vector{Float64}
    sum_x2_dt::Vector{Float64}
    sum_xy_dt::Union{Nothing, Matrix{Float64}}
    prev_x::Vector{Float64}
    prev_theta::Vector{Float64}
    prev_t::Float64
    initialized::Bool
end

function WelfordBoomerangStats(d::Integer; fullrank::Bool=false)
    WelfordBoomerangStats(
        0.0, zeros(d), zeros(d),
        fullrank ? zeros(d, d) : nothing,
        zeros(d), zeros(d), 0.0, false,
    )
end

# --- Private exact segment integral helpers ---

# Accumulate exact ∫₀^dt xᵢ(t) dt into S1.
# x(t) = (x0 - μ) cos(t) + θ0 sin(t) + μ
# ∫₀^dt xᵢ(t) dt = aᵢ sin(dt) + bᵢ (1 − cos(dt)) + μᵢ dt
function _boom_raw_S1!(S1::AbstractVector, x0::AbstractVector, theta0::AbstractVector,
                       mu::AbstractVector, dt::Float64)
    sd, cd = sincos(dt)
    omc = 1 - cd
    @inbounds for i in eachindex(S1)
        ai = x0[i] - mu[i]
        S1[i] += ai * sd + theta0[i] * omc + mu[i] * dt
    end
    return S1
end

# Accumulate exact ∫₀^dt xᵢ(t)² dt into S2 (diagonal of the raw second moment).
# ∫₀^dt xᵢ² dt = aᵢ²(dt/2 + s2d/4) + bᵢ²(dt/2 − s2d/4) + μᵢ² dt
#                + aᵢ bᵢ sd² + 2 aᵢ μᵢ sd + 2 bᵢ μᵢ (1−cd)
function _boom_raw_S2!(S2::AbstractVector, x0::AbstractVector, theta0::AbstractVector,
                       mu::AbstractVector, dt::Float64)
    sd, cd = sincos(dt)
    s2d = sin(2 * dt)
    omc = 1 - cd
    sd2 = sd * sd
    half_dt = dt / 2
    @inbounds for i in eachindex(S2)
        ai = x0[i] - mu[i]
        bi = theta0[i]
        mui = mu[i]
        S2[i] += (ai^2 * (half_dt + s2d / 4) +
                  bi^2 * (half_dt - s2d / 4) +
                  mui^2 * dt +
                  ai * bi * sd2 +
                  2 * ai * mui * sd +
                  2 * bi * mui * omc)
    end
    return S2
end

# Accumulate exact ∫₀^dt xᵢ(t) xⱼ(t) dt into S11 (full raw cross-moment matrix).
# ∫₀^dt xᵢ xⱼ dt = aᵢaⱼ(dt/2 + s2d/4) + bᵢbⱼ(dt/2 − s2d/4)
#                   + (aᵢbⱼ + aⱼbᵢ)/2 · sd² + μᵢμⱼ dt
#                   + (aᵢμⱼ + aⱼμᵢ) sd + (bᵢμⱼ + bⱼμᵢ)(1−cd)
function _boom_raw_S11!(S11::AbstractMatrix, x0::AbstractVector, theta0::AbstractVector,
                        mu::AbstractVector, dt::Float64)
    sd, cd = sincos(dt)
    s2d = sin(2 * dt)
    omc = 1 - cd
    sd2 = sd * sd
    half_dt = dt / 2
    d = length(x0)
    @inbounds for j in 1:d
        aj = x0[j] - mu[j]
        bj = theta0[j]
        muj = mu[j]
        for i in j:d
            ai = x0[i] - mu[i]
            bi = theta0[i]
            mui = mu[i]
            val = (ai * aj * (half_dt + s2d / 4) +
                   bi * bj * (half_dt - s2d / 4) +
                   (ai * bj + aj * bi) / 2 * sd2 +
                   mui * muj * dt +
                   (ai * muj + aj * mui) * sd +
                   (bi * muj + bj * mui) * omc)
            S11[i, j] += val
            i != j && (S11[j, i] += val)
        end
    end
    return S11
end

function welford_update!(ws::WelfordBoomerangStats, x::AbstractVector, theta::AbstractVector,
                         t::Float64, flow::MutableBoomerang)
    if !ws.initialized
        ws.prev_x .= x
        ws.prev_theta .= theta
        ws.prev_t = t
        ws.initialized = true
        return ws
    end

    dt = t - ws.prev_t
    if dt <= 0
        ws.prev_x .= x
        ws.prev_theta .= theta
        ws.prev_t = t
        return ws
    end

    mu = flow.μ
    ws.total_time += dt
    _boom_raw_S1!(ws.sum_x_dt, ws.prev_x, ws.prev_theta, mu, dt)
    _boom_raw_S2!(ws.sum_x2_dt, ws.prev_x, ws.prev_theta, mu, dt)
    ws.sum_xy_dt !== nothing && _boom_raw_S11!(ws.sum_xy_dt, ws.prev_x, ws.prev_theta, mu, dt)

    ws.prev_x .= x
    ws.prev_theta .= theta
    ws.prev_t = t
    return ws
end

function reset_segment_start!(ws::WelfordBoomerangStats, x::AbstractVector,
                               theta::AbstractVector, t::Float64)
    ws.prev_x .= x
    ws.prev_theta .= theta
    ws.prev_t = t
    return ws
end

function stats_mean(ws::WelfordBoomerangStats)
    d = length(ws.sum_x_dt)
    μ = zeros(d)
    ws.total_time > 0 || return μ
    T = ws.total_time
    @inbounds for i in 1:d
        μ[i] = ws.sum_x_dt[i] / T
    end
    return μ
end

function stats_var(ws::WelfordBoomerangStats)
    d = length(ws.sum_x_dt)
    v = ones(d)
    ws.total_time > 0 || return v
    T = ws.total_time
    @inbounds for i in 1:d
        m = ws.sum_x_dt[i] / T
        v[i] = max(ws.sum_x2_dt[i] / T - m * m, 0.0)
    end
    return v
end

stats_std(ws::WelfordBoomerangStats) = sqrt.(stats_var(ws))

function stats_cov(ws::WelfordBoomerangStats)
    d = length(ws.sum_x_dt)
    sum_xy = ws.sum_xy_dt
    sum_xy === nothing && error("WelfordBoomerangStats not initialized for fullrank (no sum_xy_dt)")
    ws.total_time <= 0 && return Matrix{Float64}(I, d, d)
    T = ws.total_time
    C = zeros(d, d)
    @inbounds for j in 1:d
        μj = ws.sum_x_dt[j] / T
        for i in j:d
            μi = ws.sum_x_dt[i] / T
            C[i, j] = sum_xy[i, j] / T - μi * μj
            C[j, i] = C[i, j]
        end
    end
    return C
end


# --- In-place mean/var/cov (allocation-free) ---

_coord_time(stats::WelfordBoomerangStats, ::Int) = stats.total_time

function stats_mean!(μ::AbstractVector, stats::WelfordBoomerangStats)
    T = stats.total_time
    if T > 0
        @inbounds for i in eachindex(μ)
            μ[i] = stats.sum_x_dt[i] / T
        end
    else
        fill!(μ, 0.0)
    end
    return μ
end

function stats_cov!(C::AbstractMatrix, stats::WelfordBoomerangStats)
    sum_xy = stats.sum_xy_dt
    sum_xy === nothing && error("WelfordBoomerangStats not initialized for fullrank (no sum_xy_dt)")
    _stats_cov_inner!(C, stats.sum_x_dt, sum_xy, stats.total_time)
end

function _stats_cov_inner!(C::AbstractMatrix, sum_x::Vector{Float64}, sum_xy::Matrix{Float64}, T::Float64)
    d = size(C, 1)
    if T <= 0
        fill!(C, 0.0)
        @inbounds for i in 1:d; C[i, i] = 1.0; end
        return C
    end
    @inbounds for j in 1:d
        μj = sum_x[j] / T
        for i in j:d
            μi = sum_x[i] / T
            cij = sum_xy[i, j] / T - μi * μj
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

function adapt!(rng::Random.AbstractRNG, ad::BoomerangAdapter{<:WelfordBoomerangStats}, state, flow::MutableBoomerang, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    ad.did_update = false

    if phase === :warmup
        welford_update!(ad.stats, state.ξ.x, state.ξ.θ, state.t[], flow)
    end

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    if phase === :warmup && (state.t[] - ad.last_update >= dt_now)
        update_boomerang!(flow, ad.stats, Val(ad.scheme), ad.workspace)
        refresh_velocity!(rng, state, flow)
        reset_segment_start!(ad.stats, state.ξ.x, state.ξ.θ, state.t[])
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        ad.did_update = true
    end
end

# Fallback: if the flow is not MutableBoomerang, do nothing
adapt!(::Random.AbstractRNG, ::BoomerangAdapter, state, flow, grad, trace_mgr; kwargs...) = nothing

# --- Query whether dynamics adaptation occurred (for sticky time invalidation) ---
did_dynamics_adapt(::AbstractAdapter) = false
did_dynamics_adapt(ad::BoomerangAdapter) = ad.did_update
did_dynamics_adapt(seq::SequenceAdapter) = any(did_dynamics_adapt, seq.adapters)


# --- 6. update_boomerang! implementations ---

const BOOM_DIAG_FLOOR = 1e-8

_has_data(stats::WelfordBoomerangStats) = stats.total_time > 0
_shrinkage_time(stats::WelfordBoomerangStats) = stats.total_time

"""
    update_boomerang!(flow::MutableBoomerang, stats, ::Val{:diagonal}, ::Nothing)

Update `flow.μ` and diagonal `flow.Γ` from online sufficient statistics.
Zero-allocation: computes mean, variance, and Cholesky factors in a single pass.
"""
function update_boomerang!(flow::MutableBoomerang, stats::WelfordBoomerangStats, ::Val{:diagonal}, ::Nothing)
    _has_data(stats) || return flow
    d = length(flow.μ)
    T = stats.total_time
    stats_mean!(flow.μ, stats)
    @inbounds for i in 1:d
        if T > 0
            μi = stats.sum_x_dt[i] / T
            σ2 = max(stats.sum_x2_dt[i] / T - μi * μi, BOOM_DIAG_FLOOR^2)
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
function update_boomerang!(flow::MutableBoomerang, stats::WelfordBoomerangStats, ::Val{:fullrank}, ws::FullrankWorkspace)
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

function update_boomerang!(flow::MutableBoomerang, stats::WelfordBoomerangStats, ::Val{:lowrank}, ws::FullrankWorkspace)
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

function adapt!(::Random.AbstractRNG, ad::RefreshRateAdapter, state, flow::MutableBoomerang, grad, trace_mgr;
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

adapt!(::Random.AbstractRNG, ::RefreshRateAdapter, state, flow, grad, trace_mgr; kwargs...) = nothing
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