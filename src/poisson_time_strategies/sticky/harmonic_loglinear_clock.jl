# Certified aggregate release clock for log-linear Gaussian scale slabs on a
# Boomerang orbit (HarmonicLogLinearAggregateClock).
#
# Along a Boomerang orbit every coordinate, and therefore every log scale of a
# log-linear slab, is a sinusoid c0 + cc cos t + cs sin t. The release rate is
# bounded on short cells from the exact minimum of each sinusoid, and the
# release time is sampled exactly by thinning against that piecewise-constant
# roof. Infinite horizons use the exact SummedRateClock fallback.

# Both flows below have the same physical harmonic path.  For the wrapped
# flow, harmonic coefficients come from `dynamics`, while proposal-clock
# constants must dispatch on the complete preconditioned flow.
const _CertifiedHarmonicBoomerang = Union{
    AnyBoomerang,
    PreconditionedDynamics{<:DiagonalPreconditioner,<:AnyBoomerang},
}

@inline _harmonic_boomerang(flow::AnyBoomerang) = flow
@inline _harmonic_boomerang(flow::PreconditionedDynamics{
    <:DiagonalPreconditioner,<:AnyBoomerang}) = flow.dynamics

@inline function _loglinear_boomerang_logscale_coeffs(
        provider::AbstractLogLinearIndependentGaussianSlab,
        flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState, j::Integer)
    c0 = provider.log_base_scales[j]
    cc = 0.0
    cs = 0.0
    if provider isa IndependentZeroMeanLogscaleGaussianSlab
        p0, pc, ps = _boomerang_position_coeffs(
            flow, state, provider.logscale_indices[j])
        c0 += p0
        cc += pc
        cs += ps
    else
        @inbounds for ptr in provider.rowptr[j]:(provider.rowptr[j + 1] - 1)
            p0, pc, ps = _boomerang_position_coeffs(
                flow, state,
                provider.logscale_indices[provider.colidx[ptr]])
            weight = provider.nzval[ptr]
            c0 += weight * p0
            cc += weight * pc
            cs += weight * ps
        end
    end
    (isfinite(c0) && isfinite(cc) && isfinite(cs)) || throw(DomainError(
        (c0, cc, cs),
        "non-finite harmonic log-scale coefficients for factor $j"))
    return c0, cc, cs
end

@inline function _checked_boomerang_boundary_constant(value::Real, i::Integer)
    isfinite(value) || throw(DomainError(
        value, "non-finite Boomerang boundary covariance constant for coordinate $i"))
    value >= 0 || throw(DomainError(
        value, "negative Boomerang boundary covariance constant for coordinate $i"))
    return Float64(value)
end

@inline function _harmonic_loglinear_rate(
        clock::HarmonicLogLinearAggregateClock,
        flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState,
        can_stick::BitVector, t::Real)
    clock.diagnostics.enabled && (clock.diagnostics.point_rate_evaluations += 1)
    provider = clock.slab_provider
    workspace = clock.workspace
    indices = beta_indices(provider)
    active_beta = workspace.active_beta
    stickable_beta = workspace.stickable_beta
    _prepare_harmonic_boundary_constants!(clock, flow, state)
    @inbounds for j in eachindex(indices)
        active_beta[j] = state.free[indices[j]]
        stickable_beta[j] = can_stick[indices[j]]
    end
    active_count = count(active_beta)
    s, c = sincos(Float64(t))
    max_lograte = -Inf
    scaled_total = 0.0
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = _log_model_add_odds_with_count(
            clock.model_prior, active_beta, j, active_count)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        c0, cc, cs = _loglinear_boomerang_logscale_coeffs(
            provider, flow, state, j)
        C = workspace.boundary_constants[j]
        iszero(C) && continue
        lograte = logodds + workspace.log_boundary_constants[j] -
            (c0 + cc * c + cs * s)
        if lograte <= max_lograte
            scaled_total += exp(lograte - max_lograte)
        else
            scaled_total = isfinite(max_lograte) ?
                scaled_total * exp(max_lograte - lograte) + 1.0 : 1.0
            max_lograte = lograte
        end
    end
    max_lograte == -Inf && return 0.0
    return exp(max_lograte) * scaled_total
end

@inline function _prepare_harmonic_boundary_constants!(
        clock::HarmonicLogLinearAggregateClock,
        flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:AnyBoomerang},
        state::StickyPDMPState)
    workspace = clock.workspace
    # Under the velocity-preserving sticky kernel the release constant is the
    # *individual stored speed*.  It changes at freeze and when a frozen
    # coordinate is refreshed, neither of which changes the flow object or
    # metric generation.  Rebuild these O(number-of-selectable-coordinates)
    # constants whenever the aggregate schedule is evaluated; caching them by
    # flow identity would leave stale release hazards.
    indices = beta_indices(clock.slab_provider)
    @inbounds for j in eachindex(indices)
        workspace.boundary_constants[j] =
            _checked_boomerang_boundary_constant(
                _boundary_proposal_clock_constant(
                    flow, state, indices[j]), indices[j])
        workspace.log_boundary_constants[j] =
            iszero(workspace.boundary_constants[j]) ? -Inf :
            log(workspace.boundary_constants[j]) - _LOG_SQRT_2PI
    end
    return workspace.boundary_constants
end

@inline function _prepare_harmonic_boundary_constants!(
        clock::HarmonicLogLinearAggregateClock, flow::AnyBoomerang,
        state::StickyPDMPState)
    workspace = clock.workspace
    indices = beta_indices(clock.slab_provider)
    @inbounds for j in eachindex(indices)
        workspace.boundary_constants[j] =
            _checked_boomerang_boundary_constant(
                _boundary_proposal_clock_constant(
                    flow, state, indices[j]), indices[j])
        workspace.log_boundary_constants[j] =
            iszero(workspace.boundary_constants[j]) ? -Inf :
            log(workspace.boundary_constants[j]) - _LOG_SQRT_2PI
    end
    return workspace.boundary_constants
end

function _harmonic_loglinear_cell_upper!(
        clock::HarmonicLogLinearAggregateClock,
        flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState,
        can_stick::BitVector, lo::Float64, hi::Float64)
    provider = clock.slab_provider
    workspace = clock.workspace
    indices = beta_indices(provider)
    active_beta = workspace.active_beta
    stickable_beta = workspace.stickable_beta
    _prepare_harmonic_boundary_constants!(clock, flow, state)
    @inbounds for j in eachindex(indices)
        active_beta[j] = state.free[indices[j]]
        stickable_beta[j] = can_stick[indices[j]]
    end
    active_count = count(active_beta)
    max_lograte = -Inf
    scaled_total = 0.0
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = _log_model_add_odds_with_count(
            clock.model_prior, active_beta, j, active_count)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        c0, cc, cs = _loglinear_boomerang_logscale_coeffs(
            provider, flow, state, j)
        eta_min, _ = _sinusoid_range_on_cell(c0, cc, cs, lo, hi)
        C = workspace.boundary_constants[j]
        iszero(C) && continue
        lograte = logodds + workspace.log_boundary_constants[j] - eta_min
        if lograte <= max_lograte
            scaled_total += exp(lograte - max_lograte)
        else
            scaled_total = isfinite(max_lograte) ?
                scaled_total * exp(max_lograte - lograte) + 1.0 : 1.0
            max_lograte = lograte
        end
    end
    max_lograte == -Inf && return 0.0
    upper = exp(max_lograte) * scaled_total
    return upper + _fp_pad(upper)
end

function _prepare_harmonic_loglinear_cells!(
        clock::HarmonicLogLinearAggregateClock,
        flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState,
        horizon::Float64, can_stick::BitVector)
    workspace = clock.workspace
    cells = min(clock.max_cells,
        max(1, ceil(Int, horizon / clock.max_cell_width)))
    workspace.cells_used = cells
    workspace.prefix[1] = 0.0
    @inbounds for cell in 1:cells
        lo = horizon * (cell - 1) / cells
        hi = horizon * cell / cells
        roof = _harmonic_loglinear_cell_upper!(
            clock, flow, state, can_stick, lo, hi)
        workspace.roofs[cell] = roof
        workspace.prefix[cell + 1] = workspace.prefix[cell] +
            roof * (hi - lo)
    end
    return workspace.prefix[cells + 1]
end

function rate(clock::HarmonicLogLinearAggregateClock,
        flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState,
        t::Real, can_stick::BitVector)
    t < 0 && throw(ArgumentError("t must be non-negative"))
    return _harmonic_loglinear_rate(clock, flow, state, can_stick, t)
end

cumulative_hazard(clock::HarmonicLogLinearAggregateClock,
    flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState,
    t0::Real, t1::Real, can_stick::BitVector) =
    cumulative_hazard(clock.fallback, flow, state, t0, t1, can_stick)

function sample_time(rng::Random.AbstractRNG,
        clock::HarmonicLogLinearAggregateClock,
        flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState,
        horizon::Real, can_stick::BitVector)
    clock.diagnostics.enabled && (clock.diagnostics.aggregate_calls += 1)
    _has_inactive_stickable_beta(clock.fallback, state, can_stick) || return Inf
    isfinite(horizon) || return sample_time(
        rng, clock.fallback, flow, state, horizon, can_stick)
    H = Float64(horizon)
    H <= 0 && return Inf
    total_hazard = _prepare_harmonic_loglinear_cells!(
        clock, flow, state, H, can_stick)
    iszero(total_hazard) && return Inf
    workspace = clock.workspace
    envelope_hazard = 0.0
    while true
        envelope_hazard += rand(rng, Exponential())
        envelope_hazard > total_hazard && return Inf
        cell = 1
        @inbounds while workspace.prefix[cell + 1] < envelope_hazard
            cell += 1
        end
        roof = workspace.roofs[cell]
        lo = H * (cell - 1) / workspace.cells_used
        proposal = lo +
            (envelope_hazard - workspace.prefix[cell]) / roof
        exact = _harmonic_loglinear_rate(
            clock, flow, state, can_stick, proposal)
        ratio = exact / roof
        diagnostics = clock.diagnostics
        diagnostics.proposals += 1
        diagnostics.max_envelope_ratio = max(
            diagnostics.max_envelope_ratio, ratio)
        ratio <= 1 + 1e-10 || throw(ArgumentError(
            "certified harmonic clock violation: exact/roof ratio $ratio exceeds 1"))
        if rand(rng) <= min(1.0, ratio)
            diagnostics.accepted += 1
            return proposal
        end
        diagnostics.rejected += 1
    end
end

sample_label(rng::Random.AbstractRNG,
    clock::HarmonicLogLinearAggregateClock,
    flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState,
    can_stick::BitVector) =
    sample_label(rng, clock.fallback, flow, state, can_stick)

stickable_coordinates(clock::HarmonicLogLinearAggregateClock) =
    beta_indices(clock.slab_provider)

reset_thinning_diagnostics!(clock::HarmonicLogLinearAggregateClock) =
    reset_thinning_diagnostics!(clock.diagnostics)
thinning_diagnostics(clock::HarmonicLogLinearAggregateClock) =
    clock.diagnostics

_fp_pad(x::Real; rel::Float64=1024eps(Float64), abs_pad::Float64=1024eps(Float64)) =
    Float64(rel * max(1.0, abs(Float64(x))) + abs_pad)

const _LOG_SQRT_2PI = 0.5 * log(2π)

function _sinusoid_range_on_cell(c0::Real, cc::Real, cs::Real, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    c0_f = Float64(c0)
    cc_f = Float64(cc)
    cs_f = Float64(cs)
    slo, clo = sincos(lo_f)
    shi, chi = sincos(hi_f)
    ylo = c0_f + cc_f * clo + cs_f * slo
    yhi = c0_f + cc_f * chi + cs_f * shi
    minv = min(ylo, yhi)
    maxv = max(ylo, yhi)
    ρ = hypot(cc_f, cs_f)
    if ρ > 0
        ϕ = atan(cs_f, cc_f)
        nlo = ceil(Int, (lo_f - ϕ) / π)
        nhi = floor(Int, (hi_f - ϕ) / π)
        for n in nlo:nhi
            t = ϕ + n * π
            lo_f <= t <= hi_f || continue
            y = iseven(n) ? c0_f + ρ : c0_f - ρ
            minv = min(minv, y)
            maxv = max(maxv, y)
        end
    end
    pad = _fp_pad(max(abs(minv), abs(maxv)))
    return minv - pad, maxv + pad
end

function _boomerang_position_coeffs(flow::_CertifiedHarmonicBoomerang, state::StickyPDMPState, i::Integer)
    boomerang = _harmonic_boomerang(flow)
    state.free[i] || return Float64(state.ξ.x[i]), 0.0, 0.0
    Δ = Float64(state.ξ.x[i] - boomerang.μ[i])
    return Float64(boomerang.μ[i]), Δ, Float64(state.ξ.θ[i])
end
