# --- 1. Infrastructure ---
struct NoAdaptation <: AbstractAdapter end
adapt!(::Random.AbstractRNG, ::NoAdaptation, args...; kwargs...) = nothing
adapt!(ad::AbstractAdapter, args...; kwargs...) = adapt!(Random.default_rng(), ad, args...;  kwargs...)

"""
    BoomerangAdaptationOptions(; sticky_aware=true, sticky_min_free_time=10.0,
        sticky_free_shrink_time=50.0, adapt_refresh=true,
        refresh_objective=:evals_per_time, target_refresh_rate=NaN,
        min_λref=0.01, max_λref=10.0)

Explicit configuration for `AdaptiveBoomerang` warmup adaptation.
"""
struct BoomerangAdaptationOptions
    sticky_aware::Bool
    sticky_min_free_time::Float64
    sticky_free_shrink_time::Float64
    adapt_refresh::Bool
    refresh_objective::Symbol
    target_refresh_rate::Float64
    min_λref::Float64
    max_λref::Float64
end

function BoomerangAdaptationOptions(; sticky_aware::Bool=true,
    sticky_min_free_time::Real=10.0,
    sticky_free_shrink_time::Real=50.0,
    adapt_refresh::Bool=true, refresh_objective::Symbol=:evals_per_time,
    target_refresh_rate::Real=NaN, min_λref::Real=0.01, max_λref::Real=10.0)
    min_free = float(sticky_min_free_time)
    shrink_time = float(sticky_free_shrink_time)
    min_ref = float(min_λref)
    max_ref = float(max_λref)
    min_free >= 0 || throw(ArgumentError("sticky_min_free_time must be nonnegative"))
    shrink_time >= 0 || throw(ArgumentError("sticky_free_shrink_time must be nonnegative"))
    ispositive(min_ref) || throw(ArgumentError("min_λref must be positive"))
    max_ref >= min_ref || throw(ArgumentError("max_λref must be at least min_λref"))
    refresh_objective in (:evals_per_time, :refresh_rate) ||
        throw(ArgumentError("refresh_objective must be :evals_per_time or :refresh_rate"))
    target = float(target_refresh_rate)
    if refresh_objective === :refresh_rate && !(isfinite(target) && ispositive(target))
        throw(ArgumentError("target_refresh_rate must be positive and finite when refresh_objective=:refresh_rate"))
    end
    return BoomerangAdaptationOptions(sticky_aware, min_free, shrink_time,
        adapt_refresh, refresh_objective, target, min_ref, max_ref)
end

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
    did_update::Bool
end

function adapt!(rng::Random.AbstractRNG, ad::PreconditionerAdapter, state, flow, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    ad.did_update = false
    if phase === :warmup && (state.t[] - ad.last_update >= ad.dt)
        trace = get_warmup_trace(trace_mgr)
        _has_integrable_segment(trace) || return
        update_preconditioner!(rng, flow, trace, state, iszero(ad.no_updates_done))
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        ad.did_update = true
    end
end

_has_integrable_segment(::Nothing) = false

function _has_integrable_segment(trace)
    first_event = iterate(trace)
    first_event === nothing && return false
    second_event = iterate(trace, first_event[2])
    return second_event !== nothing
end

# Subsampling anchor banks select at every event boundary, including the main
# phase, but populate new entries only from warmup traces. The callbacks take
# the chain-local SubsampledControlVariate explicitly so copied/statistics-wrapped
# models cannot accidentally refresh another chain's provider.
mutable struct SubsamplingAnchorBankAdapter{F1,F2} <: AbstractAdapter
    select_fn!::F1
    update_fn!::F2
    update_dt::Float64
    last_update::Float64
end

function adapt!(::Random.AbstractRNG, ad::SubsamplingAnchorBankAdapter, state, flow,
        grad::SubsampledControlVariate, trace_mgr; phase::Symbol=:warmup, kwargs...)
    ad.select_fn!(grad, state.ξ.x, phase)
    if phase === :warmup && state.t[] - ad.last_update >= ad.update_dt
        trace = get_warmup_trace(trace_mgr)
        if _has_integrable_segment(trace)
            ad.update_fn!(grad, trace)
            ad.last_update = state.t[]
        end
    end
    return nothing
end


# --- 3. The Factory Functions (Positional Args) ---

# --- Dynamics Factory ---
# Fallback: Swallow extra args (precond_dt, t0)
default_dynamics_adapter(::ContinuousDynamics, args...) = NoAdaptation()

# Specific:
function default_dynamics_adapter(::PreconditionedDynamics, precond_dt, t0, t_warmup=0.0)
    return PreconditionerAdapter(precond_dt, t0, 0, :default, false)
end


# --- Gradient Factory ---
# Fallback: Swallow extra args (t_warmup, t0)
default_gradient_adapter(::Any, args...) = NoAdaptation()

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
    sum_xy_dt::Matrix{Float64}
    free_time::Vector{Float64}
    free_sum_x_dt::Vector{Float64}
    free_sum_x2_dt::Vector{Float64}
    prev_x::Vector{Float64}
    prev_theta::Vector{Float64}
    prev_free::BitVector
    prev_t::Float64
    initialized::Bool
end

function WelfordBoomerangStats(d::Integer; fullrank::Bool=false)
    WelfordBoomerangStats(
        0.0, zeros(d), zeros(d),
        fullrank ? zeros(d, d) : zeros(0, 0),
        zeros(d), Float64[], Float64[],
        zeros(d), zeros(d), trues(d), 0.0, false,
    )
end

function _ensure_free_moments!(ws::WelfordBoomerangStats)
    d = length(ws.sum_x_dt)
    if length(ws.free_sum_x_dt) != d
        resize!(ws.free_sum_x_dt, d)
        resize!(ws.free_sum_x2_dt, d)
        copyto!(ws.free_sum_x_dt, ws.sum_x_dt)
        copyto!(ws.free_sum_x2_dt, ws.sum_x2_dt)
    end
    return ws
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

function _boom_raw_S1_masked!(S1::AbstractVector, time::AbstractVector,
                              x0::AbstractVector, theta0::AbstractVector,
                              mu::AbstractVector, free::AbstractVector{Bool},
                              dt::Float64)
    sd, cd = sincos(dt)
    omc = 1 - cd
    @inbounds for i in eachindex(S1)
        free[i] || continue
        ai = x0[i] - mu[i]
        S1[i] += ai * sd + theta0[i] * omc + mu[i] * dt
        time[i] += dt
    end
    return S1
end

function _boom_raw_S2_masked!(S2::AbstractVector, x0::AbstractVector,
                              theta0::AbstractVector, mu::AbstractVector,
                              free::AbstractVector{Bool}, dt::Float64)
    sd, cd = sincos(dt)
    s2d = sin(2 * dt)
    omc = 1 - cd
    sd2 = sd * sd
    half_dt = dt / 2
    @inbounds for i in eachindex(S2)
        free[i] || continue
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

function welford_update!(ws::WelfordBoomerangStats, x::AbstractVector, theta::AbstractVector,
                         t::Float64, flow::MutableBoomerang)
    if !ws.initialized
        ws.prev_x .= x
        ws.prev_theta .= theta
        fill!(ws.prev_free, true)
        ws.prev_t = t
        ws.initialized = true
        return ws
    end

    dt = t - ws.prev_t
    mu = flow.μ
    ws.total_time += dt
    _boom_raw_S1!(ws.sum_x_dt, ws.prev_x, ws.prev_theta, mu, dt)
    _boom_raw_S2!(ws.sum_x2_dt, ws.prev_x, ws.prev_theta, mu, dt)
    !isempty(ws.sum_xy_dt) && _boom_raw_S11!(ws.sum_xy_dt, ws.prev_x, ws.prev_theta, mu, dt)
    @inbounds for i in eachindex(ws.free_time)
        ws.free_time[i] += dt
    end

    ws.prev_x .= x
    ws.prev_theta .= theta
    fill!(ws.prev_free, true)
    ws.prev_t = t
    return ws
end

function welford_update!(ws::WelfordBoomerangStats, x::AbstractVector, theta::AbstractVector,
                         free::AbstractVector{Bool}, t::Float64, flow::MutableBoomerang)
    if !ws.initialized
        ws.prev_x .= x
        ws.prev_theta .= theta
        ws.prev_free .= free
        ws.prev_t = t
        ws.initialized = true
        return ws
    end

    dt = t - ws.prev_t
    mu = flow.μ
    all_free = all(ws.prev_free)
    all_free || _ensure_free_moments!(ws)
    ws.total_time += dt
    _boom_raw_S1!(ws.sum_x_dt, ws.prev_x, ws.prev_theta, mu, dt)
    _boom_raw_S2!(ws.sum_x2_dt, ws.prev_x, ws.prev_theta, mu, dt)
    !isempty(ws.sum_xy_dt) && _boom_raw_S11!(ws.sum_xy_dt, ws.prev_x, ws.prev_theta, mu, dt)
    if all_free
        @inbounds for i in eachindex(ws.free_time)
            ws.free_time[i] += dt
        end
        if !isempty(ws.free_sum_x_dt)
            _boom_raw_S1!(ws.free_sum_x_dt, ws.prev_x, ws.prev_theta, mu, dt)
            _boom_raw_S2!(ws.free_sum_x2_dt, ws.prev_x, ws.prev_theta, mu, dt)
        end
    else
        _boom_raw_S1_masked!(ws.free_sum_x_dt, ws.free_time, ws.prev_x, ws.prev_theta, mu, ws.prev_free, dt)
        _boom_raw_S2_masked!(ws.free_sum_x2_dt, ws.prev_x, ws.prev_theta, mu, ws.prev_free, dt)
    end

    ws.prev_x .= x
    ws.prev_theta .= theta
    ws.prev_free .= free
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

function reset_segment_start!(ws::WelfordBoomerangStats, x::AbstractVector,
                              theta::AbstractVector, free::AbstractVector{Bool}, t::Float64)
    ws.prev_x .= x
    ws.prev_theta .= theta
    ws.prev_free .= free
    ws.prev_t = t
    return ws
end

function stats_mean(ws::WelfordBoomerangStats)
    d = length(ws.sum_x_dt)
    μ = zeros(d)
    ispositive(ws.total_time) || return μ
    T = ws.total_time
    @inbounds for i in 1:d
        μ[i] = ws.sum_x_dt[i] / T
    end
    return μ
end

function stats_var(ws::WelfordBoomerangStats)
    d = length(ws.sum_x_dt)
    v = ones(d)
    ispositive(ws.total_time) || return v
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
    isempty(sum_xy) && error("WelfordBoomerangStats not initialized for fullrank (no sum_xy_dt)")
    !ispositive(ws.total_time) && return Matrix{Float64}(I, d, d)
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
    if ispositive(T)
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
    isempty(sum_xy) && error("WelfordBoomerangStats not initialized for fullrank (no sum_xy_dt)")
    _stats_cov_inner!(C, stats.sum_x_dt, sum_xy, stats.total_time)
end

function _stats_cov_inner!(C::AbstractMatrix, sum_x::Vector{Float64}, sum_xy::Matrix{Float64}, T::Float64)
    d = size(C, 1)
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
    BoomerangAdapter{S, W, O} <: AbstractAdapter

Adapter for `MutableBoomerang` that learns μ and Γ during warmup.
Supports `:diagonal` (Phase 1) and `:fullrank` (Phase 2) schemes.

Parameterized on the stats accumulator type `S`, which defaults to
`WelfordBoomerangStats` for numerically stable per-event updates.
`W` is the workspace type (`Nothing` for diagonal, `FullrankWorkspace`
for fullrank/lowrank), and `O` is the explicit adaptation options type.
"""
mutable struct BoomerangAdapter{S, W, O} <: AbstractAdapter
    const base_dt::Float64
    last_update::Float64
    no_updates_done::Int
    const scheme::Symbol
    stats::S
    did_update::Bool
    const workspace::W
    const options::O
end

function BoomerangAdapter(base_dt::Float64, t0::Float64, d::Integer; scheme::Symbol=:diagonal,
    options::BoomerangAdaptationOptions=BoomerangAdaptationOptions())
    stats = WelfordBoomerangStats(d; fullrank=(scheme == :fullrank || scheme == :lowrank))
    needs_ws = scheme == :fullrank || scheme == :lowrank
    ws = needs_ws ? FullrankWorkspace(d) : nothing
    BoomerangAdapter(base_dt, t0, 0, scheme, stats, false, ws, options)
end

function adapt!(rng::Random.AbstractRNG, ad::BoomerangAdapter{<:WelfordBoomerangStats}, state, flow::MutableBoomerang, grad, trace_mgr; phase::Symbol=:warmup, kwargs...)
    ad.did_update = false

    if phase === :warmup
        _boomerang_stats_update!(ad.stats, state, flow)
    end

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    if phase === :warmup && (state.t[] - ad.last_update >= dt_now)
        update_boomerang!(flow, ad.stats, Val(ad.scheme), ad.workspace, ad.options)
        _invalidate_boundary_velocity_cache!(state)
        refresh_velocity!(rng, state, flow)
        _boomerang_stats_reset_start!(ad.stats, state)
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        ad.did_update = true
    end
end

_boomerang_stats_update!(stats::WelfordBoomerangStats, state, flow::MutableBoomerang) =
    welford_update!(stats, state.ξ.x, state.ξ.θ, state.t[], flow)
_boomerang_stats_update!(stats::WelfordBoomerangStats, state::StickyPDMPState, flow::MutableBoomerang) =
    welford_update!(stats, state.ξ.x, state.ξ.θ, state.free, state.t[], flow)

_boomerang_stats_reset_start!(stats::WelfordBoomerangStats, state) =
    reset_segment_start!(stats, state.ξ.x, state.ξ.θ, state.t[])
_boomerang_stats_reset_start!(stats::WelfordBoomerangStats, state::StickyPDMPState) =
    reset_segment_start!(stats, state.ξ.x, state.ξ.θ, state.free, state.t[])

# Fallback: if the flow is not MutableBoomerang, do nothing
adapt!(::Random.AbstractRNG, ::BoomerangAdapter, state, flow, grad, trace_mgr; kwargs...) = nothing

# --- Query whether dynamics adaptation occurred (for sticky time invalidation) ---
did_dynamics_adapt(::AbstractAdapter) = false
did_dynamics_adapt(ad::PreconditionerAdapter) = ad.did_update
did_dynamics_adapt(ad::BoomerangAdapter) = ad.did_update
did_dynamics_adapt(seq::SequenceAdapter) = any(did_dynamics_adapt, seq.adapters)


# --- 6. update_boomerang! implementations ---

const BOOM_DIAG_FLOOR = 1e-8
const BOOM_STICKY_MIN_FREE_TIME_DEFAULT = 10.0
const BOOM_STICKY_FREE_SHRINK_TIME_DEFAULT = 50.0

_has_data(stats::WelfordBoomerangStats) = ispositive(stats.total_time)
_shrinkage_time(stats::WelfordBoomerangStats) = stats.total_time

"""
    update_boomerang!(flow::MutableBoomerang, stats, ::Val{:diagonal}, ::Nothing)

Update `flow.μ` and diagonal `flow.Γ` from online sufficient statistics.
Zero-allocation: computes mean, variance, and Cholesky factors in a single pass.
"""
function update_boomerang!(flow::MutableBoomerang, stats::WelfordBoomerangStats, ::Val{:diagonal}, ::Nothing)
    return update_boomerang!(flow, stats, Val(:diagonal), nothing, BoomerangAdaptationOptions())
end

function update_boomerang!(flow::MutableBoomerang, stats::WelfordBoomerangStats,
    ::Val{:diagonal}, ::Nothing, options::BoomerangAdaptationOptions)
    _has_data(stats) || return flow
    d = length(flow.μ)
    T = stats.total_time
    @inbounds for i in 1:d
        μ_total = stats.sum_x_dt[i] / T
        σ2_total = max(stats.sum_x2_dt[i] / T - μ_total * μ_total, BOOM_DIAG_FLOOR^2)
        Ti_free = stats.free_time[i]
        μi = μ_total
        σ2 = σ2_total
        if options.sticky_aware && Ti_free >= options.sticky_min_free_time
            use_total_free_moments = isempty(stats.free_sum_x_dt) || Ti_free == T
            sx_free = use_total_free_moments ? stats.sum_x_dt[i] : stats.free_sum_x_dt[i]
            sx2_free = use_total_free_moments ? stats.sum_x2_dt[i] : stats.free_sum_x2_dt[i]
            μ_free = sx_free / Ti_free
            σ2_free = max(sx2_free / Ti_free - μ_free * μ_free, BOOM_DIAG_FLOOR^2)
            w = ispositive(options.sticky_free_shrink_time) ?
                Ti_free / (Ti_free + options.sticky_free_shrink_time) : 1.0
            μi = w * μ_free + (1 - w) * μ_total
            σ2 = w * σ2_free + (1 - w) * σ2_total
        end
        flow.μ[i] = μi
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
    return update_boomerang!(flow, stats, Val(:fullrank), ws, BoomerangAdaptationOptions())
end

function update_boomerang!(flow::MutableBoomerang, stats::WelfordBoomerangStats,
    ::Val{:fullrank}, ws::FullrankWorkspace, ::BoomerangAdaptationOptions)
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
    return update_boomerang!(flow, stats, Val(:lowrank), ws, BoomerangAdaptationOptions())
end

function update_boomerang!(flow::MutableBoomerang, stats::WelfordBoomerangStats,
    ::Val{:lowrank}, ws::FullrankWorkspace, ::BoomerangAdaptationOptions)
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
    const objective::Symbol
    const target_refresh_rate::Float64
    last_update::Float64
    no_updates_done::Int
    prev_total_evals::Int
    prev_refresh_events::Int
    prev_pdmp_time::Float64
    prev_evals_per_time::Float64
    search_direction::Int
    did_update::Bool
end

function RefreshRateAdapter(base_dt::Float64, min_start_time::Float64;
    min_λref::Float64=0.01, max_λref::Float64=10.0,
    objective::Symbol=:evals_per_time, target_refresh_rate::Float64=NaN)
    objective in (:evals_per_time, :refresh_rate) ||
        throw(ArgumentError("objective must be :evals_per_time or :refresh_rate"))
    if objective === :refresh_rate && !(isfinite(target_refresh_rate) && ispositive(target_refresh_rate))
        throw(ArgumentError("target_refresh_rate must be positive and finite when objective=:refresh_rate"))
    end
    RefreshRateAdapter(base_dt, min_λref, max_λref, min_start_time,
        objective, target_refresh_rate, min_start_time, 0, 0, 0, 0.0, Inf, 1, false)
end

function adapt!(::Random.AbstractRNG, ad::RefreshRateAdapter, state, flow::MutableBoomerang, grad, trace_mgr;
    phase::Symbol=:warmup, stats::Union{AbstractStatisticCounter,Nothing}=nothing, kwargs...)
    ad.did_update = false
    phase === :warmup || return
    stats === nothing && return
    state.t[] >= ad.min_start_time || return

    dt_now = adapt_interval(ad.no_updates_done, ad.base_dt)
    (state.t[] - ad.last_update >= dt_now) || return

    total_evals = stats.∇f_calls + stats.∇²f_calls
    window_evals = total_evals - ad.prev_total_evals
    refresh_events = _get_counter_refreshment_events(stats)
    window_refreshes = refresh_events - ad.prev_refresh_events
    window_time = state.t[] - ad.prev_pdmp_time

    if !ispositive(window_time) || (ad.objective === :evals_per_time && window_evals <= 0)
        ad.prev_total_evals = total_evals
        ad.prev_refresh_events = refresh_events
        ad.prev_pdmp_time = state.t[]
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        return
    end

    if ad.objective === :refresh_rate
        observed_rate = window_refreshes / window_time
        step = 1.5 ^ (1.0 / (1.0 + ad.no_updates_done * 0.5))
        flow.λref = observed_rate <= ad.target_refresh_rate ?
            min(flow.λref * step, ad.max_λref) :
            max(flow.λref / step, ad.min_λref)
        ad.prev_total_evals = total_evals
        ad.prev_refresh_events = refresh_events
        ad.prev_pdmp_time = state.t[]
        ad.last_update = state.t[]
        ad.no_updates_done += 1
        ad.did_update = true
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
    ad.prev_refresh_events = refresh_events
    ad.prev_pdmp_time = state.t[]
    ad.last_update = state.t[]
    ad.no_updates_done += 1
    ad.did_update = true
end

adapt!(::Random.AbstractRNG, ::RefreshRateAdapter, state, flow, grad, trace_mgr; kwargs...) = nothing
did_dynamics_adapt(::RefreshRateAdapter) = false

function default_dynamics_adapter(flow::MutableBoomerang, precond_dt, t0, t_warmup=0.0;
    options::BoomerangAdaptationOptions=BoomerangAdaptationOptions())
    d = length(flow.μ)
    if flow.Γ isa Diagonal
        scheme = :diagonal
    elseif flow.Γ isa LowRankPrecision
        scheme = :lowrank
    else
        scheme = :fullrank
    end
    boom_adapter = BoomerangAdapter(Float64(precond_dt), Float64(t0), d; scheme, options)
    if t_warmup > 0 && options.adapt_refresh
        λref_start = Float64(t0 + t_warmup * 0.5)
        λref_adapter = RefreshRateAdapter(Float64(precond_dt * 2), λref_start;
            min_λref=options.min_λref,
            max_λref=options.max_λref,
            objective=options.refresh_objective,
            target_refresh_rate=options.target_refresh_rate)
        return SequenceAdapter((boom_adapter, λref_adapter))
    end
    return boom_adapter
end
