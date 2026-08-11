struct _ChebyshevResidualCell
    lo::Float64
    hi::Float64
    R::Float64
    residual_area::Float64
end

struct _ChebyshevResidualEnvelope
    segment::ScalarLogscaleGaussianLineSegment
    coeffs::Vector{Float64}
    power_coeffs::Vector{Float64}
    cells::Vector{_ChebyshevResidualCell}
    edges::Vector{Float64}
    residual_prefix::Vector{Float64}
    Hbar_horizon::Float64
end

struct _FourierResidualCell
    lo::Float64
    hi::Float64
    R::Float64
    residual_area::Float64
end

struct _FourierResidualEnvelope
    horizon::Float64
    a0::Float64
    a::Vector{Float64}
    b::Vector{Float64}
    cells::Vector{_FourierResidualCell}
    edges::Vector{Float64}
    residual_prefix::Vector{Float64}
    Hbar_horizon::Float64
end

struct _BoomerangNarrowPeakSegment
    c0::Float64
    cc::Float64
    cs::Float64
    log_scale::Float64
    log_scale_cos::Float64
    log_scale_sin::Float64
    log_total_weight::Float64
end

_BoomerangNarrowPeakSegment(c0::Real, cc::Real, cs::Real,
    log_scale::Real, log_total_weight::Real) =
    _BoomerangNarrowPeakSegment(Float64(c0), Float64(cc), Float64(cs),
        Float64(log_scale), 0.0, 0.0, Float64(log_total_weight))

struct _BoomerangNarrowPeakEnvelope
    segment::_BoomerangNarrowPeakSegment
    horizon::Float64
    Hbar_horizon::Float64
end

const _BOOMERANG_KRONROD_RULE = QuadGK.cachedrule(Float64, 15)

mutable struct _GaussianBoundaryWorkspace
    active_beta::BitVector
    stickable_beta::BitVector
    active_positions::Vector{Int}
    mean::Vector{Float64}
    cov::Matrix{Float64}
    cov_AA::Matrix{Float64}
    solve::Vector{Float64}
    alpha::Matrix{Float64}
    conditional_variance::Vector{Float64}
    cache_key::Union{Nothing,BitVector}
    nactive::Int
end

function _GaussianBoundaryWorkspace(m::Integer)
    return _GaussianBoundaryWorkspace(falses(m), falses(m), Vector{Int}(undef, m),
        zeros(m), zeros(m, m), zeros(m, m), zeros(m), zeros(m, m), zeros(m),
        nothing, 0)
end

mutable struct _ResidualEnvelopeWorkspace{C,B}
    coeffs::Vector{Float64}
    aux_coeffs::Vector{Float64}
    cells::Vector{C}
    edges::Vector{Float64}
    prefix::Vector{Float64}
    temp1::Vector{Float64}
    temp2::Vector{Float64}
    temp3::Vector{Float64}
    boundary::B
end

_chebyshev_residual_workspace(order::Integer, max_cells::Integer, ::Integer) =
    _ResidualEnvelopeWorkspace(zeros(Float64, Int(order) + 1),
        zeros(Float64, Int(order) + 1), _ChebyshevResidualCell[],
        Vector{Float64}(undef, Int(max_cells) + 1),
        Vector{Float64}(undef, Int(max_cells) + 1),
        zeros(Float64, Int(order) + 1), zeros(Float64, Int(order) + 1),
        zeros(Float64, Int(order) + 1), nothing)

_fourier_residual_workspace(order::Integer, cells::Integer, m::Integer) =
    _ResidualEnvelopeWorkspace(zeros(Float64, Int(order)),
        zeros(Float64, Int(order)), Vector{_FourierResidualCell}(undef, Int(cells)),
        Vector{Float64}(undef, Int(cells) + 1),
        Vector{Float64}(undef, Int(cells) + 1), Float64[], Float64[], Float64[],
        _GaussianBoundaryWorkspace(m))

abstract type _ResidualEnvelopeCapability end
struct _CertifiedResidualEnvelopeCapability <: _ResidualEnvelopeCapability end
struct _AnalyticResidualCapability <: _ResidualEnvelopeCapability end
struct _ExactResidualFallbackCapability <: _ResidualEnvelopeCapability end

"""
    residual_envelope_capability(clock, flow, horizon)

Select certified accelerated envelope construction, an exact analytic sampler,
or the exact summed-rate fallback before construction begins. Extension methods
must only return the certified-envelope capability when
`build_residual_envelope` and all interval bounds needed by it are implemented
for the complete provider/flow combination.
"""
residual_envelope_capability(::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock},
    ::ContinuousDynamics, ::Real) = _ExactResidualFallbackCapability()

residual_envelope_capability(
    ::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
    ::Union{ZigZag,BouncyParticle}, horizon::Real) =
    isfinite(horizon) ? _CertifiedResidualEnvelopeCapability() :
        _AnalyticResidualCapability()

function residual_envelope_capability(clock::FourierResidualAggregateClock,
                                      ::AnyBoomerang, horizon::Real)
    isfinite(horizon) || return _ExactResidualFallbackCapability()
    provider = clock.slab_provider
    certified = provider isa GlobalLogscaleExchangeableGaussianSlab ||
        (provider isa AbstractGaussianSlabProvider &&
         slab_cache_style(provider) isa FixedCovarianceCache)
    return certified ? _CertifiedResidualEnvelopeCapability() : _ExactResidualFallbackCapability()
end

_fp_pad(x::Real; rel::Float64=1024eps(Float64), abs_pad::Float64=1024eps(Float64)) =
    Float64(rel * max(1.0, abs(Float64(x))) + abs_pad)

const _LOG_SQRT_2PI = 0.5 * log(2π)

function _log_gaussian_zero_density(log_scale::Real, conditional_mean::Real)
    logs = Float64(log_scale)
    mean = Float64(conditional_mean)
    iszero(mean) && return -_LOG_SQRT_2PI - logs
    logs == -Inf && return -Inf
    logs == Inf && return -Inf
    log_z2 = 2 * (log(abs(mean)) - logs)
    z2 = log_z2 >= log(floatmax(Float64)) ? Inf : exp(log_z2)
    return -_LOG_SQRT_2PI - logs - 0.5 * z2
end

function _log_gaussian_zero_density_upper(log_scale_lo::Real,
        log_scale_hi::Real, mean_lo::Real, mean_hi::Real)
    logs_lo = Float64(log_scale_lo)
    logs_hi = Float64(log_scale_hi)
    mlo = Float64(mean_lo)
    mhi = Float64(mean_hi)
    min_abs_mean = mlo <= 0 <= mhi ? 0.0 : min(abs(mlo), abs(mhi))
    if iszero(min_abs_mean)
        return _log_gaussian_zero_density(logs_lo, 0.0)
    end
    # For fixed |mean|, -log(σ)-mean²/(2σ²) is maximized at
    # log(σ)=log(|mean|), clamped to the available scale interval.
    maximizing_log_scale = clamp(log(min_abs_mean), logs_lo, logs_hi)
    return _log_gaussian_zero_density(maximizing_log_scale, min_abs_mean)
end

function _standardized_mean_from_logscale(conditional_mean::Real, log_scale::Real)
    mean = Float64(conditional_mean)
    iszero(mean) && return 0.0
    logs = Float64(log_scale)
    logs == -Inf && return copysign(Inf, mean)
    logs == Inf && return copysign(0.0, mean)
    logabsz = log(abs(mean)) - logs
    logabsz >= log(floatmax(Float64)) && return copysign(Inf, mean)
    return copysign(exp(logabsz), mean)
end

function _normal_interval_probability(zlo::Real, zhi::Real)
    lo = Float64(zlo)
    hi = Float64(zhi)
    hi <= lo && return 0.0
    probability = if lo >= 0
        Distributions.normccdf(lo) - Distributions.normccdf(hi)
    elseif hi <= 0
        Distributions.normcdf(hi) - Distributions.normcdf(lo)
    else
        Distributions.normcdf(hi) - Distributions.normcdf(lo)
    end
    return clamp(probability, 0.0, 1.0)
end

function _constant_logscale_linear_mean_hazard(
        seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    hi <= lo && return 0.0
    iszero(seg.r) || throw(ArgumentError(
        "analytic linear-mean hazard requires a constant log scale"))
    iszero(seg.b) && return _scalar_logscale_gaussian_line_rate(seg, lo) *
        (Float64(hi) - Float64(lo))
    logs = log(seg.s0) + seg.ell0
    zlo = isinf(lo) ? copysign(Inf, seg.b * Float64(lo)) :
        _standardized_mean_from_logscale(seg.a + seg.b * Float64(lo), logs)
    zhi = isinf(hi) ? copysign(Inf, seg.b * Float64(hi)) :
        _standardized_mean_from_logscale(seg.a + seg.b * Float64(hi), logs)
    probability = seg.b > 0 ? _normal_interval_probability(zlo, zhi) :
        _normal_interval_probability(zhi, zlo)
    iszero(probability) && return 0.0
    log_hazard = seg.log_total_weight - log(abs(seg.b)) + log(probability)
    return exp(log_hazard)
end

function _shifted_standard_normal_quantile(z0::Float64,
        probability_increment::Float64, increasing::Bool)
    normal = Distributions.Normal()
    if increasing
        if z0 >= 0
            tail_probability = clamp(Distributions.normccdf(z0) -
                probability_increment, 0.0, 1.0)
            return Distributions.cquantile(normal, tail_probability)
        end
        probability = clamp(Distributions.normcdf(z0) +
            probability_increment, 0.0, 1.0)
        return Distributions.quantile(normal, probability)
    end
    if z0 <= 0
        probability = clamp(Distributions.normcdf(z0) -
            probability_increment, 0.0, 1.0)
        return Distributions.quantile(normal, probability)
    end
    tail_probability = clamp(Distributions.normccdf(z0) +
        probability_increment, 0.0, 1.0)
    return Distributions.cquantile(normal, tail_probability)
end

function _sample_constant_logscale_linear_mean(rng::Random.AbstractRNG,
        seg::ScalarLogscaleGaussianLineSegment, horizon::Real)
    threshold = rand(rng, Exponential())
    available = _constant_logscale_linear_mean_hazard(seg, 0.0, horizon)
    (!isfinite(available) || threshold < available) || return Inf
    log_probability_increment = log(threshold) + log(abs(seg.b)) -
        seg.log_total_weight
    probability_increment = exp(log_probability_increment)
    logs = log(seg.s0) + seg.ell0
    z0 = _standardized_mean_from_logscale(seg.a, logs)
    target_z = _shifted_standard_normal_quantile(z0,
        probability_increment, seg.b > 0)
    displacement = if iszero(target_z)
        0.0
    else
        logabs = logs + log(abs(target_z))
        logabs >= log(floatmax(Float64)) ? copysign(Inf, target_z) :
        logabs <= log(floatmin(Float64)) ? copysign(0.0, target_z) :
        copysign(exp(logabs), target_z)
    end
    t = -seg.a / seg.b + displacement / seg.b
    return clamp(t, 0.0, Float64(horizon))
end

function _scalar_logscale_gaussian_line_rate(seg::ScalarLogscaleGaussianLineSegment, t::Real)
    seg.log_total_weight == -Inf && return 0.0
    seg.log_total_weight == Inf && return Inf
    ell = seg.ell0 + seg.r * Float64(t)
    log_density = _log_gaussian_zero_density(
        log(seg.s0) + ell, seg.a + seg.b * Float64(t))
    return exp(seg.log_total_weight + log_density)
end

function _scalar_logscale_gaussian_line_upper(seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    seg.log_total_weight == -Inf && return 0.0
    seg.log_total_weight == Inf && return Inf
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    m0 = seg.a + seg.b * lo_f
    m1 = seg.a + seg.b * hi_f
    m_lo = min(m0, m1)
    m_hi = max(m0, m1)
    e0 = seg.ell0 + seg.r * lo_f
    e1 = seg.ell0 + seg.r * hi_f
    ell_lo = min(e0, e1)
    ell_hi = max(e0, e1)
    logs_lo = log(seg.s0) + ell_lo
    logs_hi = log(seg.s0) + ell_hi
    logλ_hi = seg.log_total_weight +
        _log_gaussian_zero_density_upper(logs_lo, logs_hi, m_lo, m_hi)
    upper = exp(logλ_hi)
    return upper + _fp_pad(upper)
end

"""
Return a root-centred integration window for a resolvable Gaussian rate peak.

The local time width is `scale(t₀)/abs(b)` for either sign of the log-scale
velocity. Peaks narrower than one representable time step are handled by
`_scalar_unresolved_peak`; all other narrow peaks are integrated in normalized
root coordinates so adaptive quadrature never has to discover them in time.
"""
function _scalar_peak_window(seg::ScalarLogscaleGaussianLineSegment;
        width_multiplier::Float64=14.0)
    iszero(seg.b) && return nothing
    t0 = -seg.a / seg.b
    t0 >= 0 && isfinite(t0) || return nothing
    log_width = log(seg.s0) + seg.ell0 + seg.r * t0 - log(abs(seg.b))
    log_width >= log(floatmin(Float64)) || return nothing
    w = exp(log_width)
    isfinite(w) && w > 0 || return nothing
    lo = max(0.0, t0 - width_multiplier * w)
    hi = t0 + width_multiplier * w
    hi > lo || return nothing
    return (t0=Float64(t0), width=Float64(w), lo=Float64(lo), hi=Float64(hi))
end

function _scalar_unresolved_peak(seg::ScalarLogscaleGaussianLineSegment)
    iszero(seg.b) && return nothing
    t0 = -seg.a / seg.b
    t0 >= 0 && isfinite(t0) || return nothing
    log_width = log(seg.s0) + seg.ell0 + seg.r * t0 - log(abs(seg.b))
    resolution = max(eps(max(1.0, abs(t0))), floatmin(Float64))
    log_width < log(resolution) || return nothing
    root = Float64(t0)
    cell_lo = max(0.0, prevfloat(root))
    cell_hi = nextfloat(root)
    peak = (t0=root, log_width=Float64(log_width),
        cell_lo=cell_lo, cell_hi=cell_hi)
    # No Float64 time lies inside this peak. Integrate its complete continuous
    # limiting profile before collapsing the sampled time to `root`; clipping
    # the profile at the adjacent floating-point values would lose real hazard
    # mass whenever the local width is comparable to (but below) one ulp.
    mass = _scalar_root_profile_integral(seg, log_width, -Inf, Inf)
    cell_mass = _scalar_unresolved_cell_hazard(
        seg, peak, cell_lo, cell_hi)
    return merge(peak,
        (mass=Float64(mass), cell_mass=Float64(cell_mass)))
end

function _scalar_normalized_root_coordinate(delta::Float64, log_width::Float64)
    iszero(delta) && return 0.0
    logabs = log(abs(delta)) - log_width
    logabs >= log(floatmax(Float64)) && return copysign(Inf, delta)
    logabs <= log(floatmin(Float64)) && return copysign(0.0, delta)
    return copysign(exp(logabs), delta)
end

function _scalar_root_profile_scale_velocity(
        seg::ScalarLogscaleGaussianLineSegment, log_width::Float64)
    iszero(seg.r) && return 0.0
    logabsq = log(abs(seg.r)) + log_width
    logabsq <= log(floatmin(Float64)) && return copysign(0.0, seg.r)
    logabsq >= log(floatmax(Float64)) && return copysign(Inf, seg.r)
    return copysign(exp(logabsq), seg.r)
end

function _scalar_root_profile_limit(
        seg::ScalarLogscaleGaussianLineSegment, log_width::Float64)
    q = _scalar_root_profile_scale_velocity(seg, log_width)
    return iszero(q) ? 40.0 : max(40.0, 40.0 / abs(q))
end

function _scalar_root_profile_integral(seg::ScalarLogscaleGaussianLineSegment,
        log_width::Float64, ylo::Float64, yhi::Float64)
    yhi <= ylo && return 0.0
    q = _scalar_root_profile_scale_velocity(seg, log_width)
    # On the widening-scale side the normalized mean tends to zero and the
    # remaining profile decays like exp(-|q*y|). Retain 40 e-foldings rather
    # than applying the constant-scale |y| cutoff there.
    profile_limit = _scalar_root_profile_limit(seg, log_width)
    lo = max(-profile_limit, ylo)
    hi = min(profile_limit, yhi)
    hi <= lo && return 0.0
    log_prefactor = seg.log_total_weight - log(abs(seg.b)) - _LOG_SQRT_2PI
    # Constant scale is the only closed form used here; otherwise the exact
    # scale-varying limiting profile is integrated in root coordinates.
    if iszero(q)
        probability = _normal_interval_probability(lo, hi)
        return exp(seg.log_total_weight - log(abs(seg.b))) * probability
    end
    integrand = y -> begin
        yf = Float64(y)
        qy = q * yf
        logabsz = iszero(yf) ? -Inf : log(abs(yf)) - qy
        z2 = logabsz >= 0.5log(floatmax(Float64)) ? Inf :
            logabsz == -Inf ? 0.0 : exp(2logabsz)
        exp(log_prefactor - qy - 0.5z2)
    end
    value, _ = QuadGK.quadgk(integrand, lo, hi;
        rtol=1e-11, atol=1e-13)
    return max(0.0, value)
end

function _scalar_unresolved_cell_hazard(seg::ScalarLogscaleGaussianLineSegment,
        peak, lo::Real, hi::Real)
    lo_f = max(Float64(lo), peak.cell_lo)
    hi_f = min(Float64(hi), peak.cell_hi)
    hi_f <= lo_f && return 0.0
    ylo = _scalar_normalized_root_coordinate(lo_f - peak.t0,
        peak.log_width)
    yhi = _scalar_normalized_root_coordinate(hi_f - peak.t0,
        peak.log_width)
    return _scalar_root_profile_integral(seg, peak.log_width, ylo, yhi)
end

function _scalar_regular_hazard(seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    hi_f <= lo_f && return 0.0
    value, _ = QuadGK.quadgk(t -> _scalar_logscale_gaussian_line_rate(seg, t), lo_f, hi_f; rtol=1e-8, atol=1e-12)
    return max(0.0, value)
end

function _scalar_logscale_transformed_regular_hazard(
        seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    iszero(seg.r) && return _scalar_regular_hazard(seg, lo, hi)
    hi <= lo && return 0.0
    base_log_scale = log(seg.s0) + seg.ell0
    y0 = base_log_scale + seg.r * Float64(lo)
    y1 = isinf(hi) ? copysign(Inf, seg.r) :
        base_log_scale + seg.r * Float64(hi)
    ylo, yhi = minmax(y0, y1)
    integrand = y -> begin
        t = (Float64(y) - base_log_scale) / seg.r
        _scalar_logscale_gaussian_line_rate(seg, t) / abs(seg.r)
    end
    value, _ = if ylo < 0 < yhi
        QuadGK.quadgk(integrand, ylo, 0.0, yhi; rtol=1e-8, atol=1e-12)
    else
        QuadGK.quadgk(integrand, ylo, yhi; rtol=1e-8, atol=1e-12)
    end
    return max(0.0, value)
end

function _scalar_peak_hazard(seg::ScalarLogscaleGaussianLineSegment, peak, lo::Real, hi::Real)
    lo_f = max(Float64(lo), peak.lo)
    hi_f = min(Float64(hi), peak.hi)
    hi_f <= lo_f && return 0.0
    ylo = (lo_f - peak.t0) / peak.width
    yhi = (hi_f - peak.t0) / peak.width
    q = seg.r * peak.width
    log_prefactor = seg.log_total_weight - log(abs(seg.b)) - _LOG_SQRT_2PI
    integrand = y -> begin
        yf = Float64(y)
        if iszero(yf)
            return exp(log_prefactor)
        end
        logabsz = log(abs(yf)) - q * yf
        z2 = logabsz >= 0.5log(floatmax(Float64)) ? Inf : exp(2logabsz)
        exp(log_prefactor - q * yf - 0.5z2)
    end
    value, _ = QuadGK.quadgk(integrand, ylo, yhi;
        rtol=1e-10, atol=1e-13)
    return max(0.0, value)
end

function _scalar_split_hazard(seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    hi_f <= lo_f && return 0.0
    unresolved = _scalar_unresolved_peak(seg)
    if unresolved !== nothing
        ylo = _scalar_normalized_root_coordinate(
            lo_f - unresolved.t0, unresolved.log_width)
        yhi = isinf(hi_f) ? Inf : _scalar_normalized_root_coordinate(
            hi_f - unresolved.t0, unresolved.log_width)
        return _scalar_root_profile_integral(
            seg, unresolved.log_width, ylo, yhi)
    end
    peak = _scalar_peak_window(seg)
    peak === nothing && return _scalar_regular_hazard(seg, lo_f, hi_f)
    total = 0.0
    total += _scalar_logscale_transformed_regular_hazard(
        seg, lo_f, min(hi_f, peak.lo))
    total += _scalar_peak_hazard(seg, peak, lo_f, hi_f)
    total += _scalar_logscale_transformed_regular_hazard(
        seg, max(lo_f, peak.hi), hi_f)
    return total
end

function _scalar_split_available_hazard(seg::ScalarLogscaleGaussianLineSegment)
    unresolved = _scalar_unresolved_peak(seg)
    if unresolved !== nothing
        ylo = _scalar_normalized_root_coordinate(
            -unresolved.t0, unresolved.log_width)
        return _scalar_root_profile_integral(
            seg, unresolved.log_width, ylo, Inf)
    end
    peak = _scalar_peak_window(seg)
    peak === nothing && return _scalar_logscale_transformed_regular_hazard(
        seg, 0.0, Inf)
    total = _scalar_logscale_transformed_regular_hazard(seg, 0.0, peak.lo)
    total += _scalar_peak_hazard(seg, peak, peak.lo, peak.hi)
    tail = _scalar_logscale_transformed_regular_hazard(seg, peak.hi, Inf)
    return max(0.0, total + tail)
end

function _sample_scalar_resolved_peak(rng::Random.AbstractRNG,
        seg::ScalarLogscaleGaussianLineSegment, horizon::Real;
        rtol::Real=1e-8, atol::Real=1e-10)
    peak = _scalar_peak_window(seg)
    peak === nothing && throw(ArgumentError(
        "resolved-peak sampling requires a root-centred peak window"))
    T = Float64(horizon)
    threshold = rand(rng, Exponential())
    peak_lo = max(0.0, peak.lo)
    peak_hi = min(T, peak.hi)
    before = _scalar_logscale_transformed_regular_hazard(
        seg, 0.0, min(T, peak_lo))
    if threshold < before
        objective = t -> _scalar_logscale_transformed_regular_hazard(
            seg, 0.0, t) - threshold
        return Roots.find_zero(objective, (0.0, peak_lo), Roots.Bisection();
            atol=Float64(atol), rtol=Float64(rtol))
    end
    threshold -= before
    peak_hi <= peak_lo && return Inf
    peak_mass = _scalar_peak_hazard(seg, peak, peak_lo, peak_hi)
    if threshold < peak_mass
        ylo = (peak_lo - peak.t0) / peak.width
        yhi = (peak_hi - peak.t0) / peak.width
        objective = y -> _scalar_peak_hazard(seg, peak, peak_lo,
            peak.t0 + peak.width * y) - threshold
        y = Roots.find_zero(objective, (ylo, yhi), Roots.Bisection();
            atol=1e-10, rtol=1e-10)
        return clamp(peak.t0 + peak.width * y, 0.0, T)
    end
    threshold -= peak_mass
    peak_hi >= T && return Inf
    after = _scalar_logscale_transformed_regular_hazard(seg, peak_hi, T)
    threshold < after || return Inf
    if isfinite(T)
        objective = t -> _scalar_logscale_transformed_regular_hazard(
            seg, peak_hi, t) - threshold
        return Roots.find_zero(objective, (peak_hi, T), Roots.Bisection();
            atol=Float64(atol), rtol=Float64(rtol))
    end
    hi = max(1.0, 2peak_hi)
    while _scalar_logscale_transformed_regular_hazard(seg, peak_hi, hi) < threshold
        hi *= 2
        isfinite(hi) || return Inf
    end
    objective = t -> _scalar_logscale_transformed_regular_hazard(
        seg, peak_hi, t) - threshold
    return Roots.find_zero(objective, (peak_hi, hi), Roots.Bisection();
        atol=Float64(atol), rtol=Float64(rtol))
end

function _sample_scalar_unresolved_peak(rng::Random.AbstractRNG,
        seg::ScalarLogscaleGaussianLineSegment, horizon::Real;
        rtol::Real=1e-8, atol::Real=1e-10)
    peak = _scalar_unresolved_peak(seg)
    peak === nothing && throw(ArgumentError(
        "unresolved-peak sampling requires a sub-resolution peak"))
    T = Float64(horizon)
    threshold = rand(rng, Exponential())
    ylo = _scalar_normalized_root_coordinate(-peak.t0, peak.log_width)
    yhi = isfinite(T) ? _scalar_normalized_root_coordinate(
        T - peak.t0, peak.log_width) : Inf
    available = _scalar_root_profile_integral(
        seg, peak.log_width, ylo, yhi)
    threshold < available || return Inf
    profile_limit = _scalar_root_profile_limit(seg, peak.log_width)
    search_lo = max(-profile_limit, ylo)
    search_hi = min(profile_limit, yhi)
    objective = y -> _scalar_root_profile_integral(
        seg, peak.log_width, ylo, Float64(y)) - threshold
    y = Roots.find_zero(objective, (search_lo, search_hi), Roots.Bisection();
        atol=1e-11, rtol=1e-11)
    displacement = if iszero(y)
        0.0
    else
        logabs = peak.log_width + log(abs(y))
        logabs <= log(floatmin(Float64)) ? copysign(0.0, y) :
        logabs >= log(floatmax(Float64)) ? copysign(Inf, y) :
        copysign(exp(logabs), y)
    end
    event_time = peak.t0 + displacement
    return isfinite(event_time) && event_time <= T ?
        max(0.0, event_time) : Inf
end

function _scalar_logscale_gaussian_line_cumulative_hazard(seg::ScalarLogscaleGaussianLineSegment, T::Real)
    T <= 0 && return 0.0
    seg.log_total_weight == -Inf && return 0.0
    seg.log_total_weight == Inf && return Inf
    if iszero(seg.r) && iszero(seg.b)
        return _scalar_logscale_gaussian_line_rate(seg, 0.0) * Float64(T)
    end
    iszero(seg.r) && return _constant_logscale_linear_mean_hazard(seg, 0.0, T)
    return _scalar_split_hazard(seg, 0.0, Float64(T))
end

function _scalar_logscale_gaussian_line_available_hazard(seg::ScalarLogscaleGaussianLineSegment)
    seg.log_total_weight == -Inf && return 0.0
    seg.log_total_weight == Inf && return Inf
    if iszero(seg.r)
        if iszero(seg.b)
            return iszero(_scalar_logscale_gaussian_line_rate(seg, 0.0)) ? 0.0 : Inf
        end
        return _constant_logscale_linear_mean_hazard(seg, 0.0, Inf)
    end
    if seg.r < 0 && iszero(seg.a) && iszero(seg.b)
        return Inf
    end
    return _scalar_split_available_hazard(seg)
end

function _sample_scalar_logscale_gaussian_line_infinite(rng::Random.AbstractRNG, seg::ScalarLogscaleGaussianLineSegment; initial_bracket::Real=1.0, bracket_multiplier::Real=2.0, rtol::Real=1e-8, atol::Real=1e-10)
    seg.log_total_weight == -Inf && return Inf
    seg.log_total_weight == Inf && return 0.0
    iszero(seg.r) && !iszero(seg.b) &&
        return _sample_constant_logscale_linear_mean(rng, seg, Inf)
    _scalar_unresolved_peak(seg) === nothing ||
        return _sample_scalar_unresolved_peak(rng, seg, Inf;
            rtol=rtol, atol=atol)
    _scalar_peak_window(seg) === nothing ||
        return _sample_scalar_resolved_peak(rng, seg, Inf;
            rtol=rtol, atol=atol)
    threshold = rand(rng, Exponential())
    if iszero(seg.r) && iszero(seg.b)
        λ = _scalar_logscale_gaussian_line_rate(seg, 0.0)
        return iszero(λ) ? Inf : threshold / λ
    end
    available = _scalar_logscale_gaussian_line_available_hazard(seg)
    isfinite(available) && available < threshold && return Inf
    lo = 0.0
    hi = Float64(initial_bracket)
    hi > 0 || (hi = 1.0)
    multiplier = Float64(bracket_multiplier)
    multiplier > 1 || (multiplier = 2.0)
    H_hi = _scalar_logscale_gaussian_line_cumulative_hazard(seg, hi)
    while H_hi < threshold
        lo = hi
        hi *= multiplier
        isfinite(hi) || throw(ArgumentError("scalar logscale infinite-horizon bracket overflowed before reaching the sampled hazard threshold"))
        H_hi = _scalar_logscale_gaussian_line_cumulative_hazard(seg, hi)
    end
    f = τ -> _scalar_logscale_gaussian_line_cumulative_hazard(seg, τ) - threshold
    return Roots.find_zero(f, (lo, hi), Roots.Bisection(); atol=Float64(atol), rtol=Float64(rtol))
end

function _chebyshev_fit_coeffs(f, T::Real, degree::Integer)
    n = max(4 * Int(degree) + 1, 65)
    coeffs = zeros(Float64, Int(degree) + 1)
    for k in 0:n-1
        θ = π * (k + 0.5) / n
        x = cos(θ)
        t = 0.5 * Float64(T) * (x + 1)
        y = Float64(f(t))
        coeffs[1] += y
        for j in 1:Int(degree)
            coeffs[j + 1] += y * cos(j * θ)
        end
    end
    coeffs[1] /= n
    for j in 2:length(coeffs)
        coeffs[j] *= 2 / n
    end
    return coeffs
end

function _chebyshev_fit_coeffs!(coeffs::Vector{Float64}, f, T::Real)
    degree = length(coeffs) - 1
    n = max(4degree + 1, 65)
    fill!(coeffs, 0.0)
    for k in 0:n-1
        θ = π * (k + 0.5) / n
        x = cos(θ)
        y = Float64(f(0.5 * Float64(T) * (x + 1)))
        coeffs[1] += y
        for j in 1:degree
            coeffs[j + 1] += y * cos(j * θ)
        end
    end
    coeffs[1] /= n
    @inbounds for j in 2:length(coeffs)
        coeffs[j] *= 2 / n
    end
    return coeffs
end

function _chebyshev_eval(coeffs::AbstractVector{Float64}, t::Real, T::Real)
    x = 2 * Float64(t) / Float64(T) - 1
    b1 = 0.0
    b2 = 0.0
    for j in length(coeffs):-1:2
        b0 = 2x * b1 - b2 + coeffs[j]
        b2 = b1
        b1 = b0
    end
    return x * b1 - b2 + coeffs[1]
end

function _chebyshev_to_power(coeffs::AbstractVector{Float64})
    K = length(coeffs) - 1
    out = zeros(Float64, K + 1)
    Tprev = zeros(Float64, K + 1)
    Tcurr = zeros(Float64, K + 1)
    Tprev[1] = 1.0
    out .+= coeffs[1] .* Tprev
    K == 0 && return out
    Tcurr[2] = 1.0
    out .+= coeffs[2] .* Tcurr
    for n in 2:K
        Tnext = zeros(Float64, K + 1)
        for i in 1:K
            Tnext[i + 1] += 2 * Tcurr[i]
        end
        Tnext .-= Tprev
        out .+= coeffs[n + 1] .* Tnext
        Tprev, Tcurr = Tcurr, Tnext
    end
    return out
end

function _chebyshev_to_power!(out::Vector{Float64}, coeffs::Vector{Float64},
                              Tprev::Vector{Float64}, Tcurr::Vector{Float64},
                              Tnext::Vector{Float64})
    K = length(coeffs) - 1
    fill!(out, 0.0)
    fill!(Tprev, 0.0)
    fill!(Tcurr, 0.0)
    fill!(Tnext, 0.0)
    Tprev[1] = 1.0
    out[1] = coeffs[1]
    K == 0 && return out
    Tcurr[2] = 1.0
    out[2] += coeffs[2]
    for n in 2:K
        fill!(Tnext, 0.0)
        @inbounds for i in 1:K
            Tnext[i + 1] += 2 * Tcurr[i]
        end
        @inbounds for i in eachindex(Tnext)
            Tnext[i] -= Tprev[i]
            out[i] += coeffs[n + 1] * Tnext[i]
        end
        Tprev, Tcurr, Tnext = Tcurr, Tnext, Tprev
    end
    return out
end

function _poly_integral_x(power_coeffs::AbstractVector{Float64}, x::Real)
    xf = Float64(x)
    total = 0.0
    xpow = xf
    for i in eachindex(power_coeffs)
        total += power_coeffs[i] * xpow / i
        xpow *= xf
    end
    return total
end

function _chebyshev_primitive(power_coeffs::AbstractVector{Float64}, t::Real, T::Real)
    x = 2 * Float64(t) / Float64(T) - 1
    return 0.5 * Float64(T) * (_poly_integral_x(power_coeffs, x) - _poly_integral_x(power_coeffs, -1.0))
end

_iadd(a, b) = (a[1] + b[1], a[2] + b[2])
_isub(a, b) = (a[1] - b[2], a[2] - b[1])
function _imul(a, b)
    vals = (a[1] * b[1], a[1] * b[2], a[2] * b[1], a[2] * b[2])
    return (minimum(vals), maximum(vals))
end
_iscale(c::Real, a) = c >= 0 ? (Float64(c) * a[1], Float64(c) * a[2]) : (Float64(c) * a[2], Float64(c) * a[1])

function _chebyshev_interval(coeffs::AbstractVector{Float64}, lo::Real, hi::Real, T::Real)
    x = (2 * Float64(lo) / Float64(T) - 1, 2 * Float64(hi) / Float64(T) - 1)
    total = (coeffs[1], coeffs[1])
    length(coeffs) == 1 && return total
    Tprev = (1.0, 1.0)
    Tcurr = x
    total = _iadd(total, _iscale(coeffs[2], Tcurr))
    for n in 2:length(coeffs)-1
        Tnext = _isub(_iscale(2.0, _imul(x, Tcurr)), Tprev)
        total = _iadd(total, _iscale(coeffs[n + 1], Tnext))
        Tprev, Tcurr = Tcurr, Tnext
    end
    pad = _fp_pad(max(abs(total[1]), abs(total[2])))
    return (total[1] - pad, total[2] + pad)
end

function _certify_scalar_cell(seg::ScalarLogscaleGaussianLineSegment, coeffs::Vector{Float64}, lo::Float64, hi::Float64, T::Float64)
    L_hi = _scalar_logscale_gaussian_line_upper(seg, lo, hi)
    P_lo, _ = _chebyshev_interval(coeffs, lo, hi, T)
    R = max(0.0, L_hi - P_lo, -P_lo)
    R += _fp_pad(max(L_hi, abs(P_lo), R))
    return _ChebyshevResidualCell(lo, hi, R, R * (hi - lo))
end

function _build_scalar_residual_envelope(clock::ChebyshevResidualAggregateClock, seg::ScalarLogscaleGaussianLineSegment)
    T = seg.horizon
    workspace = clock.workspace
    coeffs = _chebyshev_fit_coeffs!(workspace.coeffs,
        t -> _scalar_logscale_gaussian_line_rate(seg, t), T)
    power = _chebyshev_to_power!(workspace.aux_coeffs, coeffs,
        workspace.temp1, workspace.temp2, workspace.temp3)
    initial_cells = min(max(4, ceil(Int, sqrt(clock.max_cells))), clock.max_cells)
    heap = workspace.cells
    empty!(heap)
    for i in 0:initial_cells-1
        lo = T * i / initial_cells
        hi = T * (i + 1) / initial_cells
        push!(heap, _certify_scalar_cell(seg, coeffs, lo, hi, T))
    end
    grid_n = max(129, 8 * clock.order + 1)
    target_area = 0.0
    @inbounds for i in 0:grid_n-1
        target_area += _scalar_logscale_gaussian_line_rate(
            seg, T * (i + 0.5) / grid_n)
    end
    target_area *= T / grid_n
    target_residual = clock.residual_budget * max(target_area, eps(Float64))
    total_residual = 0.0
    @inbounds for cell in heap
        total_residual += cell.residual_area
    end
    while total_residual > target_residual && length(heap) < clock.max_cells
        idx = 1
        best_area = heap[1].residual_area
        for i in 2:length(heap)
            if heap[i].residual_area > best_area
                idx = i
                best_area = heap[i].residual_area
            end
        end
        parent = heap[idx]
        mid = 0.5 * (parent.lo + parent.hi)
        left = _certify_scalar_cell(seg, coeffs, parent.lo, mid, T)
        right = _certify_scalar_cell(seg, coeffs, mid, parent.hi, T)
        total_residual += left.residual_area + right.residual_area - parent.residual_area
        heap[idx] = left
        push!(heap, right)
    end
    sort!(heap, by = _residual_cell_lower_edge)
    edges = workspace.edges
    prefix = workspace.prefix
    resize!(edges, length(heap) + 1)
    resize!(prefix, length(heap) + 1)
    @inbounds for i in eachindex(heap)
        edges[i] = heap[i].lo
    end
    edges[end] = heap[end].hi
    prefix[1] = 0.0
    for i in eachindex(heap)
        prefix[i + 1] = prefix[i] + heap[i].residual_area
    end
    Hbar_T = _chebyshev_primitive(power, T, T) + prefix[end]
    return _ChebyshevResidualEnvelope(seg, coeffs, power, heap, edges, prefix, max(0.0, Hbar_T))
end

_residual_cell_lower_edge(cell) = cell.lo

_residual_horizon(env::_ChebyshevResidualEnvelope) = env.segment.horizon
_residual_horizon(env::_FourierResidualEnvelope) = env.horizon

function _residual_cell_index(env, t::Real)
    tf = Float64(t)
    tf <= 0 && return 1
    tf >= _residual_horizon(env) && return length(env.cells)
    return clamp(searchsortedlast(env.edges, tf), 1, length(env.cells))
end

function _residual_primitive(env, t::Real)
    tf = clamp(Float64(t), 0.0, _residual_horizon(env))
    idx = _residual_cell_index(env, tf)
    return env.residual_prefix[idx] + env.cells[idx].R * (tf - env.cells[idx].lo)
end

function _envelope_hazard(env::_ChebyshevResidualEnvelope, t::Real)
    return _chebyshev_primitive(env.power_coeffs, t, env.segment.horizon) + _residual_primitive(env, t)
end

function _envelope_rate(env::_ChebyshevResidualEnvelope, t::Real)
    idx = _residual_cell_index(env, t)
    return _chebyshev_eval(env.coeffs, t, env.segment.horizon) + env.cells[idx].R
end

function _fourier_fit_coeffs(f, T::Real, order::Integer)
    K = Int(order)
    n = max(8K + 1, 129)
    a0 = 0.0
    a = zeros(Float64, K)
    b = zeros(Float64, K)
    for j in 0:n-1
        y = Float64(f(Float64(T) * j / n))
        a0 += y
        for k in 1:K
            angle = 2π * k * j / n
            s, c = sincos(angle)
            a[k] += y * c
            b[k] += y * s
        end
    end
    a0 /= n
    a .*= 2 / n
    b .*= 2 / n
    return a0, a, b, n
end

function _fourier_fit_coeffs!(a::Vector{Float64}, b::Vector{Float64}, f, T::Real)
    K = length(a)
    n = max(8K + 1, 129)
    fill!(a, 0.0)
    fill!(b, 0.0)
    a0 = 0.0
    for j in 0:n-1
        y = Float64(f(Float64(T) * j / n))
        a0 += y
        for k in 1:K
            angle = 2π * k * j / n
            s, c = sincos(angle)
            a[k] += y * c
            b[k] += y * s
        end
    end
    a0 /= n
    rmul!(a, 2 / n)
    rmul!(b, 2 / n)
    return a0, n
end

function _fourier_eval(a0::Real, a::AbstractVector{Float64}, b::AbstractVector{Float64}, t::Real, T::Real)
    total = Float64(a0)
    tf = Float64(t)
    Tf = Float64(T)
    for k in eachindex(a)
        angle = 2π * k * tf / Tf
        s, c = sincos(angle)
        total += a[k] * c + b[k] * s
    end
    return total
end

function _fourier_derivative_eval(a::AbstractVector{Float64}, b::AbstractVector{Float64}, t::Real, T::Real)
    total = 0.0
    tf = Float64(t)
    Tf = Float64(T)
    for k in eachindex(a)
        ω = 2π * k / Tf
        angle = ω * tf
        s, c = sincos(angle)
        total += ω * (b[k] * c - a[k] * s)
    end
    return total
end

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

_square_lower_from_range(lo::Float64, hi::Float64) = lo <= 0 <= hi ? 0.0 : min(abs2(lo), abs2(hi))

function _boomerang_position_coeffs(flow::AnyBoomerang, state::StickyPDMPState, i::Integer)
    state.free[i] || return Float64(state.ξ.x[i]), 0.0, 0.0
    Δ = Float64(state.ξ.x[i] - flow.μ[i])
    return Float64(flow.μ[i]), Δ, Float64(state.ξ.θ[i])
end

function _boomerang_velocity_coeffs(flow::AnyBoomerang, state::StickyPDMPState, i::Integer)
    state.free[i] || return Float64(state.ξ.θ[i]), 0.0, 0.0
    Δ = Float64(state.ξ.x[i] - flow.μ[i])
    return 0.0, Float64(state.ξ.θ[i]), -Δ
end

function _boomerang_boundary_velocity_upper(flow::AnyBoomerang, state::StickyPDMPState, i::Integer, lo::Float64, hi::Float64)
    upper = _boundary_proposal_clock_constant(flow, state, i)
    return upper + _fp_pad(upper)
end

function _fourier_lower_on_cell(a0::Float64, a::Vector{Float64}, b::Vector{Float64}, lo::Float64, hi::Float64, T::Float64)
    T > 0 || throw(ArgumentError("Fourier period must be positive"))
    lower = a0
    magnitude = abs(a0)
    ω0 = 2π / T
    @inbounds for k in eachindex(a, b)
        ak, bk = a[k], b[k]
        r = hypot(ak, bk)
        magnitude += r
        iszero(r) && continue
        ω = k * ω0
        phase = atan(bk, ak)
        term_min = min(
            r * cos(ω * lo - phase),
            r * cos(ω * hi - phase),
        )
        # Each harmonic reaches its minimum where ωt-phase = π+2πn.
        n_lo = ceil((ω * lo - phase - π) / (2π))
        n_hi = floor((ω * hi - phase - π) / (2π))
        n_lo <= n_hi && (term_min = -r)
        lower += term_min
    end
    # Summing independently bounded harmonics is conservative. The explicit
    # roundoff allowance keeps the result an enclosure under floating-point
    # evaluation as well.
    return lower - _fp_pad(magnitude)
end

function _prepare_fixed_gaussian_boundary_workspace!(
    provider::AbstractGaussianSlabProvider,
    state::StickyPDMPState,
    can_stick::BitVector,
    workspace::_GaussianBoundaryWorkspace,
)
    slab_cache_style(provider) isa FixedCovarianceCache ||
        throw(ArgumentError("FourierResidualAggregateClock only supports fixed-covariance Gaussian slabs or structured global-logscale slabs"))
    indices = beta_indices(provider)
    m = length(indices)
    active_beta = workspace.active_beta
    stickable_beta = workspace.stickable_beta
    @inbounds for j in 1:m
        active_beta[j] = state.free[indices[j]]
        stickable_beta[j] = can_stick[indices[j]]
    end
    if workspace.cache_key === nothing || workspace.cache_key != active_beta
        gaussian_slab!(provider, workspace.mean, workspace.cov, state.ξ.x)
        nactive = 0
        @inbounds for j in 1:m
            if active_beta[j]
                nactive += 1
                workspace.active_positions[nactive] = j
            end
        end
        workspace.nactive = nactive
        if nactive > 0
            @inbounds for aidx in 1:nactive, bidx in 1:nactive
                workspace.cov_AA[aidx, bidx] = workspace.cov[
                    workspace.active_positions[aidx], workspace.active_positions[bidx]]
            end
            _cholesky_factor_prefix!(workspace.cov_AA, nactive)
        end
        @inbounds for j in 1:m
            if nactive == 0
                workspace.conditional_variance[j] = workspace.cov[j, j]
                continue
            end
            for aidx in 1:nactive
                workspace.solve[aidx] = workspace.cov[j, workspace.active_positions[aidx]]
            end
            _cholesky_solve_with_factor_prefix!(workspace.cov_AA, workspace.solve, nactive)
            cross = 0.0
            for aidx in 1:nactive
                α = workspace.solve[aidx]
                workspace.alpha[aidx, j] = α
                cross += workspace.cov[j, workspace.active_positions[aidx]] * α
            end
            workspace.conditional_variance[j] = workspace.cov[j, j] - cross
        end
        workspace.cache_key = slab_cache_key(provider, state.ξ.x, active_beta)
    end
    return workspace
end

function _fixed_gaussian_boundary_rate_upper(
    provider::AbstractGaussianSlabProvider,
    model_prior::AbstractModelPrior,
    flow::AnyBoomerang,
    state::StickyPDMPState,
    can_stick::BitVector,
    workspace::_GaussianBoundaryWorkspace,
    lo::Float64,
    hi::Float64,
)
    _prepare_fixed_gaussian_boundary_workspace!(provider, state, can_stick,
        workspace)
    indices = beta_indices(provider)
    m = length(indices)
    active_beta = workspace.active_beta
    stickable_beta = workspace.stickable_beta
    mean = workspace.mean
    total = 0.0
    @inbounds for j in 1:m
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(model_prior, active_beta, j)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        μj = mean[j]
        c0 = μj
        cc = 0.0
        cs = 0.0
        cond_var = workspace.conditional_variance[j]
        if workspace.nactive > 0
            for aidx in 1:workspace.nactive
                ibeta = workspace.active_positions[aidx]
                ifull = indices[ibeta]
                p0, pc, ps = _boomerang_position_coeffs(flow, state, ifull)
                α = workspace.alpha[aidx, j]
                c0 += α * (p0 - mean[ibeta])
                cc += α * pc
                cs += α * ps
            end
        end
        cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
        mlo, mhi = _sinusoid_range_on_cell(c0, cc, cs, lo, hi)
        z2_lo = _square_lower_from_range(mlo / sqrt(cond_var), mhi / sqrt(cond_var))
        q_upper = exp(logodds - 0.5 * (log2π + log(cond_var) + z2_lo))
        v_upper = _boomerang_boundary_velocity_upper(flow, state, indices[j], lo, hi)
        total += q_upper * v_upper
    end
    return total + _fp_pad(total)
end

function _global_logscale_boundary_rate_upper(
    provider::GlobalLogscaleExchangeableGaussianSlab,
    model_prior::AbstractModelPrior,
    flow::AnyBoomerang,
    state::StickyPDMPState,
    can_stick::BitVector,
    workspace::_GaussianBoundaryWorkspace,
    lo::Float64,
    hi::Float64,
)
    indices = beta_indices(provider)
    active_beta = workspace.active_beta
    stickable_beta = workspace.stickable_beta
    @inbounds for j in eachindex(indices)
        active_beta[j] = state.free[indices[j]]
        stickable_beta[j] = can_stick[indices[j]]
    end
    k = count(active_beta)
    denom = provider.u + k * provider.v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    c = provider.v / denom
    m0 = provider.mean
    mc = 0.0
    ms = 0.0
    @inbounds for j in eachindex(indices)
        active_beta[j] || continue
        p0, pc, ps = _boomerang_position_coeffs(flow, state, indices[j])
        m0 += c * (p0 - provider.mean)
        mc += c * pc
        ms += c * ps
    end
    e0, ec, es = _boomerang_position_coeffs(flow, state, provider.logscale_index)
    e0 += provider.logscale_offset
    mlo, mhi = _sinusoid_range_on_cell(m0, mc, ms, lo, hi)
    elo, ehi = _sinusoid_range_on_cell(e0, ec, es, lo, hi)
    s0 = sqrt(provider.u * (provider.u + (k + 1) * provider.v) / denom)
    log_q_upper = _log_gaussian_zero_density_upper(
        log(s0) + elo, log(s0) + ehi, mlo, mhi)
    max_log_weight = -Inf
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(model_prior, active_beta, j)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        velocity_upper = _boomerang_boundary_velocity_upper(
            flow, state, indices[j], lo, hi)
        iszero(velocity_upper) && continue
        max_log_weight = max(max_log_weight, logodds + log(velocity_upper))
    end
    max_log_weight == -Inf && return 0.0
    scaled_weight = 0.0
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(model_prior, active_beta, j)
        isfinite(logodds) || continue
        velocity_upper = _boomerang_boundary_velocity_upper(
            flow, state, indices[j], lo, hi)
        iszero(velocity_upper) && continue
        scaled_weight += exp(logodds + log(velocity_upper) - max_log_weight)
    end
    upper = exp(log_q_upper + max_log_weight + log(scaled_weight))
    return upper + _fp_pad(upper)
end

function _fixed_gaussian_boundary_rate(clock::FourierResidualAggregateClock,
    flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, t::Real)
    provider = clock.slab_provider
    workspace = clock.workspace.boundary
    _prepare_fixed_gaussian_boundary_workspace!(provider, state, can_stick,
        workspace)
    total = 0.0
    indices = beta_indices(provider)
    sτ, cτ = sincos(Float64(t))
    @inbounds for j in eachindex(indices)
        (workspace.stickable_beta[j] && !workspace.active_beta[j]) || continue
        logodds = log_model_add_odds(clock.model_prior, workspace.active_beta, j)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        conditional_mean = workspace.mean[j]
        for aidx in 1:workspace.nactive
            ibeta = workspace.active_positions[aidx]
            p0, pc, ps = _boomerang_position_coeffs(flow, state, indices[ibeta])
            conditional_mean += workspace.alpha[aidx, j] *
                (p0 + pc * cτ + ps * sτ - workspace.mean[ibeta])
        end
        variance = workspace.conditional_variance[j]
        variance > 0 || throw(ArgumentError(
            "conditional variance must be positive, got $variance"))
        total += exp(logodds - 0.5 * (log2π + log(variance) +
            abs2(conditional_mean) / variance)) *
            _boundary_proposal_clock_constant(flow, state, indices[j])
    end
    return total
end

function _global_logscale_boundary_rate(clock::FourierResidualAggregateClock,
    flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, t::Real)
    provider = clock.slab_provider
    workspace = clock.workspace.boundary
    indices = beta_indices(provider)
    active_beta = workspace.active_beta
    stickable_beta = workspace.stickable_beta
    @inbounds for j in eachindex(indices)
        active_beta[j] = state.free[indices[j]]
        stickable_beta[j] = can_stick[indices[j]]
    end
    k = count(active_beta)
    denom = provider.u + k * provider.v
    c = provider.v / denom
    sτ, cτ = sincos(Float64(t))
    conditional_mean = provider.mean
    @inbounds for j in eachindex(indices)
        active_beta[j] || continue
        p0, pc, ps = _boomerang_position_coeffs(flow, state, indices[j])
        conditional_mean += c * (p0 + pc * cτ + ps * sτ - provider.mean)
    end
    e0, ec, es = _boomerang_position_coeffs(flow, state, provider.logscale_index)
    ell = provider.logscale_offset + e0 + ec * cτ + es * sτ
    s0 = sqrt(provider.u * (provider.u + (k + 1) * provider.v) / denom)
    log_density = _log_gaussian_zero_density(log(s0) + ell,
        conditional_mean)
    max_log_weight = -Inf
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(clock.model_prior, active_beta, j)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        velocity = _boundary_proposal_clock_constant(flow, state, indices[j])
        iszero(velocity) && continue
        max_log_weight = max(max_log_weight, logodds + log(velocity))
    end
    max_log_weight == -Inf && return 0.0
    scaled_weight = 0.0
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(clock.model_prior, active_beta, j)
        isfinite(logodds) || continue
        velocity = _boundary_proposal_clock_constant(flow, state, indices[j])
        iszero(velocity) && continue
        scaled_weight += exp(logodds + log(velocity) - max_log_weight)
    end
    return exp(log_density + max_log_weight + log(scaled_weight))
end

"""
Detect a Boomerang boundary rate requiring root-centred integration, including
sinusoidally moving log scales.

Selection uses the smaller derivative-controlled transverse width and the
curvature-controlled quadratic width, rather than the sinusoid amplitude. A
root is collapsed to limiting mass only when its width is below one floating
point time step, neighbouring peaks are separated by at least 64 widths, and
the leading omitted curvature term is below `1e-12`. Nearly tangent or
overlapping roots instead use the stable transformed cell integral.
"""
function _boomerang_narrow_peak_segment(
        clock::FourierResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
        flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    workspace = clock.workspace.boundary
    indices = beta_indices(provider)
    active_beta = workspace.active_beta
    stickable_beta = workspace.stickable_beta
    @inbounds for j in eachindex(indices)
        active_beta[j] = state.free[indices[j]]
        stickable_beta[j] = can_stick[indices[j]]
    end
    k = count(active_beta)
    denom = provider.u + k * provider.v
    c = provider.v / denom
    c0 = provider.mean
    cc = 0.0
    cs = 0.0
    @inbounds for j in eachindex(indices)
        active_beta[j] || continue
        p0, pc, ps = _boomerang_position_coeffs(flow, state, indices[j])
        c0 += c * (p0 - provider.mean)
        cc += c * pc
        cs += c * ps
    end
    amplitude = hypot(cc, cs)
    iszero(amplitude) && return nothing

    e0, ec, es = _boomerang_position_coeffs(flow, state,
        provider.logscale_index)
    s0 = sqrt(provider.u * (provider.u + (k + 1) * provider.v) / denom)
    log_scale = log(s0) + provider.logscale_offset + e0

    max_log_weight = -Inf
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(clock.model_prior, active_beta, j)
        logodds == -Inf && continue
        logodds == Inf && return _BoomerangNarrowPeakSegment(
            c0, cc, cs, log_scale, ec, es, Inf)
        velocity = _boundary_proposal_clock_constant(flow, state, indices[j])
        iszero(velocity) && continue
        max_log_weight = max(max_log_weight, logodds + log(velocity))
    end
    max_log_weight == -Inf && return nothing
    scaled_weight = 0.0
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(clock.model_prior, active_beta, j)
        isfinite(logodds) || continue
        velocity = _boundary_proposal_clock_constant(flow, state, indices[j])
        iszero(velocity) && continue
        scaled_weight += exp(logodds + log(velocity) - max_log_weight)
    end
    segment = _BoomerangNarrowPeakSegment(c0, cc, cs, log_scale, ec, es,
        max_log_weight + log(scaled_weight))
    roots = _boomerang_peak_roots(segment)
    isempty(roots) && return nothing
    # Fourier point sampling cannot resolve a peak much narrower than a cell.
    # Selection is based on the derivative/curvature-controlled local width;
    # the subsequent integrator decides separately whether a root is truly
    # below floating-point time resolution and eligible for a delta limit.
    minimum_log_width = minimum(root ->
        _boomerang_root_log_width(segment, root), roots)
    minimum_log_width < log(1e-3) || return nothing
    return segment
end

function _boomerang_peak_roots(seg::_BoomerangNarrowPeakSegment)
    amplitude = hypot(seg.cc, seg.cs)
    q = -seg.c0 / amplitude
    abs(q) <= 1 || return Float64[]
    phase = atan(seg.cs, seg.cc)
    alpha = acos(clamp(q, -1.0, 1.0))
    roots = Float64[]
    for raw_root in (phase - alpha, phase + alpha)
        root = mod(raw_root, 2π)
        root == 2π && (root = 0.0)
        any(r -> abs(r - root) <= 8eps(Float64), roots) || push!(roots, root)
    end
    sort!(roots)
    return roots
end

function _boomerang_peak_mass(seg::_BoomerangNarrowPeakSegment, root::Float64)
    derivative = _boomerang_root_derivative(seg, root)
    if !iszero(derivative)
        return exp(seg.log_total_weight - log(abs(derivative)))
    end
    curvature = abs(seg.c0)
    iszero(curvature) && return Inf
    # For m(t₀+Δ)=curvature*Δ²/2+O(Δ⁴), integrating the normalized
    # Gaussian gives C*weight/sqrt(scale*curvature), C=1.2162802142575204.
    return exp(seg.log_total_weight -
        0.5 * (_boomerang_log_scale(seg, root) + log(curvature)) +
        log(1.2162802142575204))
end

_boomerang_log_scale(seg::_BoomerangNarrowPeakSegment, t::Float64) =
    seg.log_scale + seg.log_scale_cos * cos(t) + seg.log_scale_sin * sin(t)

_boomerang_log_scale_derivative(seg::_BoomerangNarrowPeakSegment, t::Float64) =
    -seg.log_scale_cos * sin(t) + seg.log_scale_sin * cos(t)

_boomerang_log_scale_curvature(seg::_BoomerangNarrowPeakSegment, t::Float64) =
    -seg.log_scale_cos * cos(t) - seg.log_scale_sin * sin(t)

_boomerang_root_derivative(seg::_BoomerangNarrowPeakSegment, root::Float64) =
    -seg.cc * sin(root) + seg.cs * cos(root)

function _boomerang_root_log_width(
        seg::_BoomerangNarrowPeakSegment, root::Float64)
    derivative = abs(_boomerang_root_derivative(seg, root))
    curvature = abs(seg.c0)
    root_log_scale = _boomerang_log_scale(seg, root)
    iszero(derivative) && return 0.5 *
        (log(2.0) + root_log_scale - log(curvature))
    linear_log_width = root_log_scale - log(derivative)
    iszero(curvature) && return linear_log_width
    log_curvature_ratio = root_log_scale + log(curvature) - 2log(derivative)
    return log_curvature_ratio < log(1 / 16) ? linear_log_width :
        0.5 * (log(2.0) + root_log_scale - log(curvature))
end

function _boomerang_root_uses_delta(seg::_BoomerangNarrowPeakSegment,
        roots::Vector{Float64}, index::Int)
    root = roots[index]
    derivative = abs(_boomerang_root_derivative(seg, root))
    curvature = abs(seg.c0)
    log_width = _boomerang_root_log_width(seg, root)
    resolution = eps(max(1.0, abs(root)))
    log_width < log(resolution) || return false
    previous = index == 1 ? roots[end] - 2π : roots[index - 1]
    following = index == length(roots) ? roots[1] + 2π : roots[index + 1]
    separation = min(root - previous, following - root)
    log_width + log(64.0) < log(separation) || return false
    width = exp(log_width)
    scale_slope_error = abs(_boomerang_log_scale_derivative(seg, root)) * width
    scale_curvature_error = abs(_boomerang_log_scale_curvature(seg, root)) * width^2
    max(scale_slope_error, scale_curvature_error) < 1e-12 || return false
    if iszero(derivative)
        # The first omitted sinusoidal term changes the quadratic-root mass by
        # O(width²), which is negligible only under this explicit tolerance.
        return 2log_width < log(1e-12)
    end
    iszero(curvature) && return true
    # The transverse delta mass has relative curvature error
    # O(scale*curvature/derivative²).
    return _boomerang_log_scale(seg, root) + log(curvature) -
        2log(derivative) < log(1e-12)
end

function _boomerang_all_roots_use_delta(seg::_BoomerangNarrowPeakSegment,
        roots::Vector{Float64})
    return all(i -> _boomerang_root_uses_delta(seg, roots, i),
        eachindex(roots))
end

function _boomerang_transformed_cell_hazard(
        seg::_BoomerangNarrowPeakSegment, root::Float64,
        lo::Float64, hi::Float64)
    hi <= lo && return 0.0
    derivative = _boomerang_root_derivative(seg, root)
    log_width = _boomerang_root_log_width(seg, root)
    width = exp(log_width)
    # The ordinary scaled-coordinate rule is substantially cheaper for peaks
    # whose finite prefactor is safely representable. Its range is derived
    # from the weighted Gaussian-tail budget (and local log-scale variation),
    # rather than being a fixed normalized-coordinate cutoff.
    root_log_scale = _boomerang_log_scale(seg, root)
    root_log_integrand = seg.log_total_weight + log_width -
        root_log_scale - _LOG_SQRT_2PI
    logscale_amplitude = hypot(seg.log_scale_cos, seg.log_scale_sin)
    if isfinite(width) && width > 0 && root_log_integrand < 600.0 &&
            logscale_amplitude <= 32.0
        return _boomerang_direct_scaled_cell_hazard(
            seg, root, derivative, log_width, width, lo, hi)
    end
    # Integrate the complete root cell with y=sinh(u). Unlike a fixed cutoff
    # in normalized root coordinates, this retains leading-tail hazard after
    # multiplication by an arbitrarily large finite model-add weight. The
    # asinh map also keeps a sub-resolution peak visible when the ordinary-time
    # cell is many orders of magnitude wider than the local peak.
    ulo = _boomerang_asinh_normalized_delta(lo - root, log_width)
    uhi = _boomerang_asinh_normalized_delta(hi - root, log_width)
    uhi <= ulo && return 0.0
    log_prefactor = seg.log_total_weight + log_width - _LOG_SQRT_2PI
    log_integrand = u -> begin
        uf = Float64(u)
        delta = _boomerang_delta_from_asinh_coordinate(uf, log_width)
        half_sine = sin(0.5delta)
        mean_value = 2seg.c0 * half_sine^2 + derivative * sin(delta)
        log_scale = _boomerang_log_scale(seg, root + delta)
        log_jacobian = _logcosh(uf)
        if iszero(mean_value)
            return log_prefactor - log_scale + log_jacobian
        end
        logabsz = log(abs(mean_value)) - log_scale
        z2 = logabsz >= 0.5log(floatmax(Float64)) ? Inf : exp(2logabsz)
        return log_prefactor - log_scale - 0.5z2 + log_jacobian
    end
    # Strong sinusoidal scale motion can create material shoulders away from
    # the root. Subdivide only that uncommon path so adaptive quadrature sees
    # every shoulder; mild-motion and large-weight transverse peaks retain the
    # low-allocation single-panel path.
    log_value = if logscale_amplitude > 32.0
        initial_panels = max(16, ceil(Int, 8sqrt(logscale_amplitude)))
        _boomerang_composite_log_kronrod(
            log_integrand, ulo, uhi, initial_panels)
    else
        _boomerang_log_quadgk(log_integrand, ulo, uhi)
    end
    log_value == -Inf && return 0.0
    log_value >= log(floatmax(Float64)) && return Inf
    return exp(log_value)
end

function _boomerang_direct_scaled_cell_hazard(
        seg::_BoomerangNarrowPeakSegment, root::Float64,
        derivative::Float64, log_width::Float64, width::Float64,
        lo::Float64, hi::Float64)
    # exp(-tail_radius^2/2) times the finite model weight is below exp(-800).
    # Account conservatively for scale changes across the retained root
    # neighbourhood and update once after measuring that variation.
    tail_radius = sqrt(2(max(seg.log_total_weight, 0.0) + 800.0))
    for _ in 1:2
        left = root - width * tail_radius
        right = root + width * tail_radius
        variation = max(abs(_boomerang_log_scale(seg, left) -
                _boomerang_log_scale(seg, root)),
            abs(_boomerang_log_scale(seg, right) -
                _boomerang_log_scale(seg, root)))
        tail_radius = sqrt(2(max(seg.log_total_weight, 0.0) +
            variation + 800.0))
    end
    ylo = max((lo - root) / width, -tail_radius)
    yhi = min((hi - root) / width, tail_radius)
    yhi <= ylo && return 0.0
    log_prefactor = seg.log_total_weight + log_width - _LOG_SQRT_2PI
    integrand = y -> begin
        delta = width * Float64(y)
        half_sine = sin(0.5delta)
        mean_value = 2seg.c0 * half_sine^2 + derivative * sin(delta)
        log_scale = _boomerang_log_scale(seg, root + delta)
        z = mean_value / exp(log_scale)
        return exp(log_prefactor - log_scale - 0.5z^2)
    end
    value, _ = QuadGK.quadgk(integrand, ylo, yhi;
        rtol=1e-11, atol=0.0)
    return value
end

function _boomerang_asinh_normalized_delta(delta::Float64, log_width::Float64)
    iszero(delta) && return 0.0
    logabs = log(abs(delta)) - log_width
    value = if logabs < log(floatmax(Float64))
        asinh(exp(logabs))
    else
        log(2.0) + logabs
    end
    return copysign(value, delta)
end

function _boomerang_delta_from_asinh_coordinate(
        u::Float64, log_width::Float64)
    iszero(u) && return 0.0
    absu = abs(u)
    logsinh = absu < 20.0 ? log(sinh(absu)) :
        absu - log(2.0) + log1p(-exp(-2absu))
    logabs = log_width + logsinh
    logabs >= log(floatmax(Float64)) && return copysign(Inf, u)
    # Preserve representable subnormal displacements. Collapsing everything
    # below `floatmin` (the smallest *normal*) can turn an overlapping
    # transformed root into an atom at zero displacement.
    logabs < log(nextfloat(0.0)) && return copysign(0.0, u)
    return copysign(exp(logabs), u)
end

function _logcosh(x::Float64)
    ax = abs(x)
    return ax + log1p(exp(-2ax)) - log(2.0)
end

function _boomerang_composite_log_kronrod(
        logf, lo::Float64, hi::Float64, initial_panels::Int)
    hi <= lo && return -Inf
    previous = -Inf
    panels = initial_panels
    for _ in 1:8
        panel_width = (hi - lo) / panels
        total = -Inf
        for panel in 0:(panels - 1)
            left = muladd(panel, panel_width, lo)
            right = panel == panels - 1 ? hi : left + panel_width
            center = 0.5left + 0.5right
            half_width = 0.5(right - left)
            nodes = _BOOMERANG_KRONROD_RULE[1]
            weights = _BOOMERANG_KRONROD_RULE[2]
            reference = logf(center)
            @inbounds for j in 1:(length(nodes) - 1)
                offset = half_width * abs(nodes[j])
                reference = max(reference,
                    logf(center - offset), logf(center + offset))
            end
            if reference != -Inf
                scaled = weights[end] * exp(logf(center) - reference)
                @inbounds for j in 1:(length(nodes) - 1)
                    offset = half_width * abs(nodes[j])
                    scaled += weights[j] * (
                        exp(logf(center - offset) - reference) +
                        exp(logf(center + offset) - reference))
                end
                panel_log = log(half_width) + reference + log(scaled)
                total = LogExpFunctions.logaddexp(total, panel_log)
            end
        end
        if isfinite(previous) && abs(total - previous) <=
                5e-12 * max(1.0, abs(total))
            return total
        end
        previous = total
        panels *= 2
    end
    return previous
end

function _boomerang_log_quadgk(logf, lo::Float64, hi::Float64)
    hi <= lo && return -Inf
    # Splitting at the root prevents a very long transformed cell from hiding
    # its narrow maximum from the adaptive rule.
    if lo < 0.0 < hi
        left = _boomerang_log_quadgk(logf, lo, 0.0)
        right = _boomerang_log_quadgk(logf, 0.0, hi)
        return LogExpFunctions.logaddexp(left, right)
    end
    midpoint = 0.5lo + 0.5hi
    reference = max(logf(lo), logf(midpoint), logf(hi)) + 32.0
    reference == -Inf && return -Inf
    scaled = u -> begin
        value = logf(Float64(u)) - reference
        value == -Inf ? 0.0 : exp(value)
    end
    value, _ = QuadGK.quadgk(scaled, lo, hi; rtol=1e-11, atol=0.0)
    value <= 0.0 && return -Inf
    return reference + log(value)
end

function _boomerang_partial_transformed_hazard(
        seg::_BoomerangNarrowPeakSegment, roots::Vector{Float64},
        horizon::Float64)
    horizon <= 0 && return 0.0
    total = 0.0
    nroots = length(roots)
    for index in eachindex(roots)
        root = roots[index]
        if _boomerang_root_uses_delta(seg, roots, index)
            mass = _boomerang_peak_mass(seg, root)
            if iszero(root)
                # The periodic atom at zero is shared by the two endpoints.
                # A complete period contains its full mass, while a positive
                # proper prefix contains the leading half only.
                if horizon >= 2π
                    total += mass
                elseif horizon > 0.0
                    total += 0.5mass
                end
            elseif horizon > root
                total += mass
            elseif horizon == root
                total += 0.5mass
            end
            continue
        end
        previous = index == 1 ? roots[end] - 2π : roots[index - 1]
        following = index == nroots ? roots[1] + 2π : roots[index + 1]
        cell_lo = 0.5 * (previous + root)
        cell_hi = 0.5 * (root + following)
        for shift in (-2π, 0.0, 2π)
            shifted_root = root + shift
            lo = max(0.0, cell_lo + shift)
            hi = min(horizon, cell_hi + shift)
            hi <= lo && continue
            shifted_cell_hi = cell_hi + shift
            strong_scale_motion = hypot(
                seg.log_scale_cos, seg.log_scale_sin) > 32.0
            if strong_scale_motion && hi <= shifted_root
                left_total = _boomerang_transformed_cell_hazard(
                    seg, shifted_root, lo, shifted_root)
                left_tail = hi < shifted_root ?
                    _boomerang_transformed_cell_hazard(
                        seg, shifted_root, hi, shifted_root) : 0.0
                total += _boomerang_stable_hazard_difference(
                    left_total, left_tail)
            elseif strong_scale_motion && hi > shifted_root
                # Evaluate post-root prefixes as a full right half minus its
                # shrinking tail. This makes plateau monotonicity structural
                # instead of relying on cancellation between independently
                # integrated, nearly equal prefixes.
                if lo < shifted_root
                    total += _boomerang_transformed_cell_hazard(
                        seg, shifted_root, lo, shifted_root)
                    right_total = _boomerang_transformed_cell_hazard(
                        seg, shifted_root, shifted_root, shifted_cell_hi)
                else
                    right_total = _boomerang_transformed_cell_hazard(
                        seg, shifted_root, lo, shifted_cell_hi)
                end
                right_tail = hi < shifted_cell_hi ?
                    _boomerang_transformed_cell_hazard(
                        seg, shifted_root, hi, shifted_cell_hi) : 0.0
                total += _boomerang_stable_hazard_difference(
                    right_total, right_tail)
            else
                total += _boomerang_transformed_cell_hazard(
                    seg, shifted_root, lo, hi)
            end
        end
    end
    return total
end

function _boomerang_stable_hazard_difference(total::Float64, tail::Float64)
    iszero(total) && return 0.0
    tail <= 1e-12 * total && return total
    difference = max(0.0, total - tail)
    difference <= 1e-12 * total && return 0.0
    return difference
end

function _boomerang_delta_cumulative_hazard(
        seg::_BoomerangNarrowPeakSegment, roots::Vector{Float64},
        horizon::Float64)
    horizon < 0 && return 0.0
    period = 2π
    period_mass = sum(root -> _boomerang_peak_mass(seg, root), roots)
    cycles, phase = fldmod(horizon, period)
    total = iszero(cycles) ? 0.0 : cycles * period_mass
    root_zero = first(roots) == 0.0
    if iszero(horizon)
        return root_zero ? 0.5 * _boomerang_peak_mass(seg, 0.0) : 0.0
    end
    # At an exact period boundary, the two half-atoms at the endpoints make
    # one complete root-zero mass. Just after the boundary, the new period's
    # leading half-atom has also been crossed.
    if root_zero && !iszero(phase)
        total += 0.5 * _boomerang_peak_mass(seg, 0.0)
    end
    for root in roots
        iszero(root) && continue
        root > phase && continue
        mass = _boomerang_peak_mass(seg, root)
        total += root == phase ? 0.5mass : mass
    end
    return total
end

function _boomerang_narrow_peak_cumulative_hazard(
        seg::_BoomerangNarrowPeakSegment, horizon::Real)
    horizon <= 0 && return 0.0
    roots = _boomerang_peak_roots(seg)
    isempty(roots) && return 0.0
    T = Float64(horizon)
    isinf(T) && return Inf
    seg.log_total_weight == Inf && return Inf
    _boomerang_all_roots_use_delta(seg, roots) &&
        return _boomerang_delta_cumulative_hazard(seg, roots, T)
    period_hazard = _boomerang_partial_transformed_hazard(seg, roots, 2π)
    cycles, remainder = fldmod(T, 2π)
    total = iszero(cycles) ? 0.0 : cycles * period_hazard
    if remainder > 0
        total += _boomerang_partial_transformed_hazard(seg, roots, remainder)
    end
    return total
end

function _boomerang_adjust_sampled_time(seg::_BoomerangNarrowPeakSegment,
        event_time::Float64, threshold::Float64, horizon::Float64)
    isfinite(event_time) || return Inf
    event_time <= horizon || return Inf
    # At enormous cycle counts the within-period phase is smaller than one
    # time ULP. Place the collapsed event on the adjacent Float64 whose hazard
    # interval contains the exponential target. This requires only a bounded
    # number of O(number of roots) evaluations.
    for _ in 1:4
        lower = _boomerang_narrow_peak_cumulative_hazard(
            seg, prevfloat(event_time))
        upper = _boomerang_narrow_peak_cumulative_hazard(
            seg, nextfloat(event_time))
        lower <= threshold <= upper && return event_time
        if threshold < lower
            event_time = prevfloat(event_time)
        else
            event_time = nextfloat(event_time)
            event_time <= horizon || return Inf
        end
    end
    return event_time
end

function _boomerang_invert_continuous_phase(
        seg::_BoomerangNarrowPeakSegment, roots::Vector{Float64},
        threshold::Float64, lo::Float64, hi::Float64,
        strict_inverse::Bool)
    hi < lo && return Inf
    lo == hi && return lo
    objective = t -> _boomerang_partial_transformed_hazard(
        seg, roots, t) - threshold
    return Roots.find_zero(objective, (lo, hi), Roots.Bisection();
        atol=strict_inverse ? 0.0 : 1e-12,
        rtol=strict_inverse ? 0.0 : 1e-10)
end

function _boomerang_mixed_phase_event(
        seg::_BoomerangNarrowPeakSegment, roots::Vector{Float64},
        threshold::Float64, horizon::Float64)
    # Mixed continuous/atomic periods must be inverted in temporal order.
    # A continuous root finder must never be asked to cross an atomic jump.
    strict_inverse = seg.log_total_weight >= 600.0 ||
        any(i -> _boomerang_root_uses_delta(seg, roots, i), eachindex(roots))
    root_zero_delta = iszero(first(roots)) &&
        _boomerang_root_uses_delta(seg, roots, firstindex(roots))
    initial_half_mass = root_zero_delta ?
        0.5 * _boomerang_peak_mass(seg, 0.0) : 0.0
    threshold <= initial_half_mass && return 0.0
    interval_lo = root_zero_delta ? nextfloat(0.0) : 0.0
    for index in eachindex(roots)
        _boomerang_root_uses_delta(seg, roots, index) || continue
        root = roots[index]
        iszero(root) && continue
        root > horizon && break
        before = prevfloat(root)
        if before >= interval_lo
            hazard_before = _boomerang_partial_transformed_hazard(
                seg, roots, before)
            if threshold <= hazard_before
                return _boomerang_invert_continuous_phase(
                    seg, roots, threshold, interval_lo, before,
                    strict_inverse)
            end
        else
            hazard_before = _boomerang_partial_transformed_hazard(
                seg, roots, before)
        end
        mass = _boomerang_peak_mass(seg, root)
        if threshold < hazard_before + mass
            return root
        end
        interval_lo = nextfloat(root)
    end
    if root_zero_delta && horizon == 2π
        before = prevfloat(2π)
        hazard_before = _boomerang_partial_transformed_hazard(
            seg, roots, before)
        if threshold <= hazard_before
            return _boomerang_invert_continuous_phase(
                seg, roots, threshold, interval_lo, before,
                strict_inverse)
        end
        return 2π
    end
    return _boomerang_invert_continuous_phase(
        seg, roots, threshold, interval_lo, horizon, strict_inverse)
end

function _sample_boomerang_narrow_peak(rng::Random.AbstractRNG,
        seg::_BoomerangNarrowPeakSegment, horizon::Real)
    roots = _boomerang_peak_roots(seg)
    isempty(roots) && return Inf
    T = Float64(horizon)
    # Infinite model-add odds multiply the positive Gaussian boundary density
    # everywhere, so the extended-real rate convention requires an immediate
    # event. This is distinct from a finite-weight local peak whose integrated
    # mass merely overflows.
    seg.log_total_weight == Inf && return 0.0
    threshold = rand(rng, Exponential())
    if isfinite(T)
        available = _boomerang_narrow_peak_cumulative_hazard(seg, T)
        threshold <= available || return Inf
    end
    if _boomerang_all_roots_use_delta(seg, roots)
        root_zero = first(roots) == 0.0
        initial_mass = 0.0
        if root_zero
            initial_mass = 0.5 * _boomerang_peak_mass(seg, 0.0)
            threshold < initial_mass && return 0.0
        end
        period_mass = sum(root -> _boomerang_peak_mass(seg, root), roots)
        iszero(period_mass) && return Inf
        if isinf(period_mass)
            residual = threshold - initial_mass
            for root in roots
                iszero(root) && continue
                mass = _boomerang_peak_mass(seg, root)
                residual < mass && return _boomerang_adjust_sampled_time(
                    seg, root, threshold, T)
                residual -= mass
            end
            return Inf
        end
        block, residual = fldmod(threshold, period_mass)
        if initial_mass > 0
            if residual >= initial_mass
                residual -= initial_mass
            else
                block -= 1.0
                residual = period_mass - (initial_mass - residual)
            end
        end
        for root in roots
            root == 0.0 && continue
            mass = _boomerang_peak_mass(seg, root)
            if residual < mass
                event_time = muladd(block, 2π, root)
                return _boomerang_adjust_sampled_time(
                    seg, event_time, threshold, T)
            end
            residual -= mass
        end
        if root_zero
            mass = _boomerang_peak_mass(seg, 0.0)
            if residual < mass
                event_time = (block + 1.0) * (2π)
                return _boomerang_adjust_sampled_time(
                    seg, event_time, threshold, T)
            end
            residual -= mass
        end
        # Rounding at a period boundary: advance one block without converting
        # the (possibly enormous) cycle count to Int.
        block += 1.0
        if root_zero
            event_time = block * (2π)
            return _boomerang_adjust_sampled_time(
                seg, event_time, threshold, T)
        end
        first_positive = findfirst(x -> !iszero(x), roots)
        if first_positive !== nothing
            event_time = muladd(block, 2π, roots[first_positive])
            return _boomerang_adjust_sampled_time(
                seg, event_time, threshold, T)
        end
        return Inf
    end

    period_hazard = _boomerang_partial_transformed_hazard(seg, roots, 2π)
    iszero(period_hazard) && return Inf
    block, residual = isinf(period_hazard) ? (0.0, threshold) :
        fldmod(threshold, period_hazard)
    root_zero_delta = iszero(first(roots)) &&
        _boomerang_root_uses_delta(seg, roots, firstindex(roots))
    if root_zero_delta
        initial_half_mass = 0.5 * _boomerang_peak_mass(seg, 0.0)
        threshold < initial_half_mass && return 0.0
        if residual < initial_half_mass
            block -= 1.0
            residual = period_hazard - (initial_half_mass - residual)
        end
    end
    base_time = block * (2π)
    isfinite(base_time) || return Inf
    base_time > T && return Inf
    remaining_horizon = min(2π, T - base_time)
    available = _boomerang_partial_transformed_hazard(
        seg, roots, remaining_horizon)
    if residual > available
        # At very large cycle counts the phase can be smaller than one ULP of
        # absolute time: `base_time` rounds to the finite horizon even though
        # the cumulative hazard at that horizon already crosses the target.
        # Recover the adjacent representable event instead of rejecting it as
        # beyond the horizon.
        return isfinite(T) ? _boomerang_adjust_sampled_time(
            seg, T, threshold, T) : Inf
    end
    phase = _boomerang_mixed_phase_event(
        seg, roots, residual, remaining_horizon)
    event_time = base_time + phase
    return _boomerang_adjust_sampled_time(seg, event_time, threshold, T)
end

_residual_target_rate(clock::FourierResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
    flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, t::Real) =
    _global_logscale_boundary_rate(clock, flow, state, can_stick, t)

_residual_horizon(env::_BoomerangNarrowPeakEnvelope) = env.horizon
_envelope_hazard(env::_BoomerangNarrowPeakEnvelope, t::Real) =
    _boomerang_narrow_peak_cumulative_hazard(env.segment,
        clamp(Float64(t), 0.0, env.horizon))
function _envelope_rate(env::_BoomerangNarrowPeakEnvelope, t::Real)
    tf = Float64(t)
    seg = env.segment
    mean_value = seg.c0 + seg.cc * cos(tf) + seg.cs * sin(tf)
    log_density = _log_gaussian_zero_density(
        _boomerang_log_scale(seg, tf), mean_value)
    return exp(seg.log_total_weight + log_density)
end

_residual_target_rate(clock::FourierResidualAggregateClock{<:AbstractGaussianSlabProvider},
    flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, t::Real) =
    _fixed_gaussian_boundary_rate(clock, flow, state, can_stick, t)

_boomerang_boundary_rate_upper(clock::FourierResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab}, flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, lo::Float64, hi::Float64) =
    _global_logscale_boundary_rate_upper(clock.slab_provider, clock.model_prior, flow, state, can_stick, clock.workspace.boundary, lo, hi)
_boomerang_boundary_rate_upper(clock::FourierResidualAggregateClock{<:AbstractGaussianSlabProvider}, flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, lo::Float64, hi::Float64) =
    _fixed_gaussian_boundary_rate_upper(clock.slab_provider, clock.model_prior, flow, state, can_stick, clock.workspace.boundary, lo, hi)

"""
    boundary_rate_upper(clock, flow, state, can_stick, lo, hi)

Return a rigorous upper bound on the aggregate boundary rate throughout the
closed interval `[lo, hi]`. Certified envelope builders may rely on this bound;
custom clocks opting into certified construction must provide an equally
conservative method for their provider and dynamics.
"""
boundary_rate_upper(clock::FourierResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
    flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, lo::Real, hi::Real) =
    _global_logscale_boundary_rate_upper(clock.slab_provider, clock.model_prior,
        flow, state, can_stick, clock.workspace.boundary, Float64(lo), Float64(hi))

boundary_rate_upper(clock::FourierResidualAggregateClock{<:AbstractGaussianSlabProvider},
    flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, lo::Real, hi::Real) =
    _fixed_gaussian_boundary_rate_upper(clock.slab_provider, clock.model_prior,
        flow, state, can_stick, clock.workspace.boundary, Float64(lo), Float64(hi))

function _fourier_primitive(env::_FourierResidualEnvelope, t::Real)
    tf = clamp(Float64(t), 0.0, env.horizon)
    total = env.a0 * tf
    for k in eachindex(env.a)
        ω = 2π * k / env.horizon
        s, c = sincos(ω * tf)
        total += env.a[k] * s / ω + env.b[k] * (1 - c) / ω
    end
    return total
end

function _fourier_primitive(a0::Real, a::Vector{Float64}, b::Vector{Float64},
                            horizon::Real, t::Real)
    tf = clamp(Float64(t), 0.0, Float64(horizon))
    total = Float64(a0) * tf
    @inbounds for k in eachindex(a, b)
        ω = 2π * k / horizon
        s, c = sincos(ω * tf)
        total += a[k] * s / ω + b[k] * (1 - c) / ω
    end
    return total
end

function _fourier_envelope_hazard(env::_FourierResidualEnvelope, t::Real)
    return _fourier_primitive(env, t) + _residual_primitive(env, t)
end

function _fourier_envelope_rate(env::_FourierResidualEnvelope, t::Real)
    idx = _residual_cell_index(env, t)
    return _fourier_eval(env.a0, env.a, env.b, t, env.horizon) + env.cells[idx].R
end

function _certify_fourier_cell(clock::FourierResidualAggregateClock, flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, a0::Float64, a::Vector{Float64}, b::Vector{Float64}, lo::Float64, hi::Float64, T::Float64)
    λ_hi = boundary_rate_upper(clock, flow, state, can_stick, lo, hi)
    q_lo = _fourier_lower_on_cell(a0, a, b, lo, hi, T)
    R = max(0.0, λ_hi - q_lo, -q_lo)
    R += _fp_pad(max(λ_hi, abs(q_lo), R))
    return _FourierResidualCell(lo, hi, R, R * (hi - lo))
end

"""
    build_residual_envelope(clock, flow, state, horizon, can_stick)

Construct a certified residual envelope after
`residual_envelope_capability` has selected the accelerated path. The returned
envelope must have nondecreasing cumulative hazard, dominate the true rate on
the full horizon, and provide a finite terminal hazard whenever its rate is
integrable.
"""
build_residual_envelope(clock::FourierResidualAggregateClock, flow::AnyBoomerang,
    state::StickyPDMPState, horizon::Real, can_stick::BitVector) =
    _build_fourier_residual_envelope(clock, flow, state, horizon, can_stick)

function build_residual_envelope(clock::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
    flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real,
    can_stick::BitVector)
    seg = scalar_logscale_gaussian_line_segment(
        clock.slab_provider, clock.model_prior, flow, state, can_stick, Float64(horizon))
    return _build_scalar_residual_envelope(clock, seg), max(4 * clock.order + 1, 65)
end

_envelope_hazard(env::_FourierResidualEnvelope, t::Real) = _fourier_envelope_hazard(env, t)
_envelope_rate(env::_FourierResidualEnvelope, t::Real) = _fourier_envelope_rate(env, t)

"""
    _sample_certified_envelope(rng, clock, envelope, true_rate; rate_evaluations=0)

Shared thinning loop for certified residual envelopes. Envelope implementations
provide cumulative hazard, instantaneous rate, horizon, terminal hazard, cells,
and residual prefix data. This helper owns inversion, acceptance validation,
horizon handling, and diagnostics semantics.
"""
function _sample_certified_envelope(rng::Random.AbstractRNG, clock, env, true_rate;
                                    rate_evaluations::Integer=0)
    d = clock.diagnostics
    d.last_cells = length(env.cells)
    d.residual_area += env.residual_prefix[end]
    d.envelope_hazard += env.Hbar_horizon
    max_residual = 0.0
    min_envelope = Inf
    @inbounds for cell in env.cells
        max_residual = max(max_residual, cell.R)
        min_envelope = min(min_envelope,
            _envelope_rate(env, 0.5 * (cell.lo + cell.hi)))
    end
    d.max_residual = max(d.max_residual, max_residual)
    d.min_envelope = min(d.min_envelope, min_envelope)
    d.rate_evaluations += rate_evaluations
    horizon = _residual_horizon(env)
    t = 0.0
    while t < horizon
        target = _envelope_hazard(env, t) + rand(rng, Exponential())
        target > env.Hbar_horizon && return Inf
        objective = τ -> _envelope_hazard(env, τ) - target
        proposal = Roots.find_zero(objective, (t, horizon), Roots.Bisection();
            atol=clock.fallback.atol, rtol=clock.fallback.rtol)
        d.proposals += 1
        λ = true_rate(proposal)
        λbar = _envelope_rate(env, proposal)
        λbar > 0 || throw(ArgumentError(
            "certified residual envelope produced a non-positive proposal rate"))
        ratio = λ / λbar
        d.max_envelope_ratio = max(d.max_envelope_ratio, ratio)
        ratio <= 1 + 1e-10 || throw(ArgumentError(
            "certified residual envelope violation: true/proposal rate ratio $ratio exceeds 1"))
        if rand(rng) <= min(1.0, ratio)
            d.accepted += 1
            return proposal
        end
        d.rejected += 1
        t = proposal
    end
    return Inf
end

function _build_fourier_residual_envelope(clock::FourierResidualAggregateClock, flow::AnyBoomerang, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    T = Float64(horizon)
    if clock.slab_provider isa GlobalLogscaleExchangeableGaussianSlab
        peak = _boomerang_narrow_peak_segment(clock, flow, state, can_stick)
        if peak !== nothing
            H = _boomerang_narrow_peak_cumulative_hazard(peak, T)
            return _BoomerangNarrowPeakEnvelope(peak, T, H), 0
        end
    end
    f = t -> _residual_target_rate(clock, flow, state, can_stick, t)
    workspace = clock.workspace
    a = workspace.coeffs
    b = workspace.aux_coeffs
    a0, fit_evaluations = _fourier_fit_coeffs!(a, b, f, T)
    cells = workspace.cells
    edges = workspace.edges
    rate_evaluations = fit_evaluations
    for i in 1:clock.cells
        lo = T * (i - 1) / clock.cells
        hi = T * i / clock.cells
        cells[i] = _certify_fourier_cell(clock, flow, state, can_stick, a0, a, b, lo, hi, T)
        edges[i] = lo
    end
    edges[end] = T
    prefix = workspace.prefix
    prefix[1] = 0.0
    for i in eachindex(cells)
        prefix[i + 1] = prefix[i] + cells[i].residual_area
    end
    Hbar_T = _fourier_primitive(a0, a, b, T, T) + prefix[end]
    return _FourierResidualEnvelope(T, a0, a, b, cells, edges, prefix, max(0.0, Hbar_T)), rate_evaluations
end

function rate(clock::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    τ < 0 && throw(ArgumentError("τ must be non-negative"))
    seg = scalar_logscale_gaussian_line_segment(clock.slab_provider, clock.model_prior, flow, state, can_stick, max(Float64(τ), eps(Float64)))
    return _scalar_logscale_gaussian_line_rate(seg, τ)
end

function sample_time(rng::Random.AbstractRNG, clock::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
                     flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    capability = residual_envelope_capability(clock, flow, horizon)
    if capability isa _AnalyticResidualCapability
        seg = scalar_logscale_gaussian_line_segment(clock.slab_provider, clock.model_prior, flow, state, can_stick, Inf)
        return _sample_scalar_logscale_gaussian_line_infinite(rng, seg;
            initial_bracket=clock.fallback.initial_bracket,
            bracket_multiplier=clock.fallback.bracket_multiplier,
            rtol=clock.fallback.rtol,
            atol=clock.fallback.atol)
    end
    capability isa _ExactResidualFallbackCapability &&
        return _sample_time_exact_fallback(rng, clock, flow, state, horizon,
            can_stick)
    horizon <= 0 && return Inf
    seg = scalar_logscale_gaussian_line_segment(clock.slab_provider,
        clock.model_prior, flow, state, can_stick, Float64(horizon))
    seg.log_total_weight == Inf && return 0.0
    iszero(seg.r) && !iszero(seg.b) &&
        return _sample_constant_logscale_linear_mean(rng, seg, horizon)
    _scalar_unresolved_peak(seg) === nothing ||
        return _sample_scalar_unresolved_peak(rng, seg, horizon;
            rtol=clock.fallback.rtol, atol=clock.fallback.atol)
    _scalar_peak_window(seg) === nothing ||
        return _sample_scalar_resolved_peak(rng, seg, horizon;
            rtol=clock.fallback.rtol, atol=clock.fallback.atol)
    _scalar_logscale_gaussian_line_rate(seg, 0.0) == Inf && return 0.0
    env = _build_scalar_residual_envelope(clock, seg)
    rate_evaluations = max(4 * clock.order + 1, 65)
    return _sample_certified_envelope(rng, clock, env,
        t -> _scalar_logscale_gaussian_line_rate(seg, t);
        rate_evaluations=rate_evaluations)
end

function sample_time(rng::Random.AbstractRNG, clock::FourierResidualAggregateClock, flow::AnyBoomerang, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    _has_inactive_stickable_beta(clock.fallback, state, can_stick) || return Inf
    horizon <= 0 && return Inf
    if clock.slab_provider isa GlobalLogscaleExchangeableGaussianSlab
        peak = _boomerang_narrow_peak_segment(clock, flow, state, can_stick)
        peak === nothing || return _sample_boomerang_narrow_peak(rng, peak, horizon)
    end
    capability = residual_envelope_capability(clock, flow, horizon)
    if capability isa _ExactResidualFallbackCapability
        rate(clock.fallback, flow, state, 0.0, can_stick) == Inf && return 0.0
        return _sample_time_exact_fallback(rng, clock, flow, state, horizon, can_stick)
    end
    _residual_target_rate(clock, flow, state, can_stick, 0.0) == Inf &&
        return 0.0
    env, rate_evaluations = build_residual_envelope(clock, flow, state, horizon, can_stick)
    return _sample_certified_envelope(rng, clock, env,
        t -> _residual_target_rate(clock, flow, state, can_stick, t);
        rate_evaluations=rate_evaluations)
end

function sample_label(rng::Random.AbstractRNG, clock::AbstractAggregateUnstickClock, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    state_at = _clock_state_at(state, flow, τ)
    return sample_label(rng, clock, flow, state_at, can_stick)
end
