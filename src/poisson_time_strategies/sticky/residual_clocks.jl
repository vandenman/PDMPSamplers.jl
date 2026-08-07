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

_fp_pad(x::Real; rel::Float64=1024eps(Float64), abs_pad::Float64=1024eps(Float64)) =
    Float64(rel * max(1.0, abs(Float64(x))) + abs_pad)

function _scalar_logscale_gaussian_line_rate(seg::ScalarLogscaleGaussianLineSegment, t::Real)
    seg.log_total_weight == -Inf && return 0.0
    seg.log_total_weight == Inf && return Inf
    ell = seg.ell0 + seg.r * Float64(t)
    s = seg.s0 * exp(ell)
    z = (seg.a + seg.b * Float64(t)) / s
    return exp(seg.log_total_weight - log(seg.s0) - ell - 0.5 * abs2(z) - 0.5 * log(2π))
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
    s_lo = exp(logs_lo)
    s_hi = exp(logs_hi)
    z1 = m_lo / s_lo
    z2 = m_lo / s_hi
    z3 = m_hi / s_lo
    z4 = m_hi / s_hi
    z_lo = min(z1, z2, z3, z4)
    z_hi = max(z1, z2, z3, z4)
    z2_lo = z_lo <= 0 <= z_hi ? 0.0 : min(abs2(z_lo), abs2(z_hi))
    logλ_hi = seg.log_total_weight - logs_lo - 0.5 * z2_lo - 0.5 * log(2π)
    upper = exp(logλ_hi)
    return upper + _fp_pad(upper)
end

function _scalar_peak_window(seg::ScalarLogscaleGaussianLineSegment; width_multiplier::Float64=14.0)
    (seg.r < 0 && !iszero(seg.b)) || return nothing
    t0 = -seg.a / seg.b
    t0 > 0 || return nothing
    s0 = seg.s0 * exp(seg.ell0 + seg.r * t0)
    w = s0 / abs(seg.b)
    isfinite(w) && w > 0 || return nothing
    lo = max(0.0, t0 - width_multiplier * w)
    hi = t0 + width_multiplier * w
    hi > lo || return nothing
    return (t0=Float64(t0), width=Float64(w), lo=Float64(lo), hi=Float64(hi))
end

function _scalar_regular_hazard(seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    hi_f <= lo_f && return 0.0
    value, _ = QuadGK.quadgk(t -> _scalar_logscale_gaussian_line_rate(seg, t), lo_f, hi_f; rtol=1e-8, atol=1e-12)
    return max(0.0, value)
end

function _scalar_peak_hazard(seg::ScalarLogscaleGaussianLineSegment, peak, lo::Real, hi::Real)
    lo_f = max(Float64(lo), peak.lo)
    hi_f = min(Float64(hi), peak.hi)
    hi_f <= lo_f && return 0.0
    ylo = (lo_f - peak.t0) / peak.width
    yhi = (hi_f - peak.t0) / peak.width
    value, _ = QuadGK.quadgk(y -> _scalar_logscale_gaussian_line_rate(seg, peak.t0 + peak.width * y) * peak.width,
        ylo, yhi; rtol=1e-8, atol=1e-12)
    return max(0.0, value)
end

function _scalar_split_hazard(seg::ScalarLogscaleGaussianLineSegment, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    hi_f <= lo_f && return 0.0
    peak = _scalar_peak_window(seg)
    peak === nothing && return _scalar_regular_hazard(seg, lo_f, hi_f)
    total = 0.0
    total += _scalar_regular_hazard(seg, lo_f, min(hi_f, peak.lo))
    total += _scalar_peak_hazard(seg, peak, lo_f, hi_f)
    total += _scalar_regular_hazard(seg, max(lo_f, peak.hi), hi_f)
    return total
end

function _scalar_split_available_hazard(seg::ScalarLogscaleGaussianLineSegment)
    peak = _scalar_peak_window(seg)
    peak === nothing && return _scalar_regular_hazard(seg, 0.0, Inf)
    total = _scalar_regular_hazard(seg, 0.0, peak.lo)
    total += _scalar_peak_hazard(seg, peak, peak.lo, peak.hi)
    tail, _ = QuadGK.quadgk(t -> _scalar_logscale_gaussian_line_rate(seg, t), peak.hi, Inf; rtol=1e-8, atol=1e-12)
    return max(0.0, total + tail)
end

function _scalar_logscale_gaussian_line_cumulative_hazard(seg::ScalarLogscaleGaussianLineSegment, T::Real)
    T <= 0 && return 0.0
    seg.log_total_weight == -Inf && return 0.0
    seg.log_total_weight == Inf && return Inf
    if iszero(seg.r) && iszero(seg.b)
        return _scalar_logscale_gaussian_line_rate(seg, 0.0) * Float64(T)
    end
    return _scalar_split_hazard(seg, 0.0, Float64(T))
end

function _scalar_logscale_gaussian_line_available_hazard(seg::ScalarLogscaleGaussianLineSegment)
    seg.log_total_weight == -Inf && return 0.0
    seg.log_total_weight == Inf && return Inf
    scaled_s = seg.s0 * exp(seg.ell0)
    if iszero(seg.r)
        return _linear_gaussian_component_total_hazard(seg.a, seg.b, scaled_s, seg.log_total_weight)
    end
    if seg.r < 0 && iszero(seg.a) && iszero(seg.b)
        return Inf
    end
    return _scalar_split_available_hazard(seg)
end

function _sample_scalar_logscale_gaussian_line_infinite(rng::Random.AbstractRNG, seg::ScalarLogscaleGaussianLineSegment; initial_bracket::Real=1.0, bracket_multiplier::Real=2.0, rtol::Real=1e-8, atol::Real=1e-10)
    seg.log_total_weight == -Inf && return Inf
    seg.log_total_weight == Inf && return 0.0
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
    coeffs = _chebyshev_fit_coeffs(t -> _scalar_logscale_gaussian_line_rate(seg, t), T, clock.order)
    power = _chebyshev_to_power(coeffs)
    initial_cells = min(max(4, ceil(Int, sqrt(clock.max_cells))), clock.max_cells)
    heap = _ChebyshevResidualCell[]
    for i in 0:initial_cells-1
        lo = T * i / initial_cells
        hi = T * (i + 1) / initial_cells
        push!(heap, _certify_scalar_cell(seg, coeffs, lo, hi, T))
    end
    grid_n = max(129, 8 * clock.order + 1)
    target_area = T / grid_n * sum(_scalar_logscale_gaussian_line_rate(seg, T * (i + 0.5) / grid_n) for i in 0:grid_n-1)
    target_residual = clock.residual_budget * max(target_area, eps(Float64))
    total_residual = sum(c.residual_area for c in heap)
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
    sort!(heap, by = c -> c.lo)
    edges = [c.lo for c in heap]
    push!(edges, heap[end].hi)
    prefix = zeros(Float64, length(heap) + 1)
    for i in eachindex(heap)
        prefix[i + 1] = prefix[i] + heap[i].residual_area
    end
    Hbar_T = _chebyshev_primitive(power, T, T) + prefix[end]
    return _ChebyshevResidualEnvelope(seg, coeffs, power, heap, edges, prefix, max(0.0, Hbar_T))
end

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

function _push_cell_sinusoid_critical_points!(points::Vector{Float64}, lo::Float64, hi::Float64, cc::Float64, cs::Float64)
    ρ = hypot(cc, cs)
    ρ <= 0 && return points
    ϕ = atan(cs, cc)
    nlo = ceil(Int, (lo - ϕ) / π)
    nhi = floor(Int, (hi - ϕ) / π)
    for n in nlo:nhi
        t = ϕ + n * π
        lo <= t <= hi && push!(points, t)
    end
    return points
end

function _sinusoid_range_on_cell(c0::Real, cc::Real, cs::Real, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    points = Float64[lo_f, hi_f]
    _push_cell_sinusoid_critical_points!(points, lo_f, hi_f, Float64(cc), Float64(cs))
    minv = Inf
    maxv = -Inf
    @inbounds for t in points
        s, c = sincos(t)
        y = Float64(c0) + Float64(cc) * c + Float64(cs) * s
        minv = min(minv, y)
        maxv = max(maxv, y)
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

function _fixed_gaussian_boundary_rate_upper(
    provider::AbstractGaussianSlabProvider,
    model_prior::AbstractModelPrior,
    flow::AnyBoomerang,
    state::StickyPDMPState,
    can_stick::BitVector,
    lo::Float64,
    hi::Float64,
)
    slab_cache_style(provider) isa FixedCovarianceCache ||
        throw(ArgumentError("FourierResidualAggregateClock only supports fixed-covariance Gaussian slabs or structured global-logscale slabs"))
    indices = beta_indices(provider)
    m = length(indices)
    active_beta = _active_beta_from_free(provider, state.free)
    stickable_beta = _stickable_beta_from_can_stick(provider, can_stick)
    mean, cov = _current_mean_cov(provider, state.ξ.x)
    A = _active_positions(active_beta, m)
    total = 0.0
    @inbounds for j in 1:m
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(model_prior, active_beta, j)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        μj = mean[j]
        σjj = cov[j, j]
        c0 = μj
        cc = 0.0
        cs = 0.0
        cond_var = σjj
        if !isempty(A)
            cov_AA = Matrix{Float64}(cov[A, A])
            cov_jA = Vector{Float64}(cov[j, A])
            F = cholesky(Symmetric(cov_AA); check=true)
            α = F \ cov_jA
            cond_var -= dot(cov_jA, α)
            for aidx in eachindex(A)
                ibeta = A[aidx]
                ifull = indices[ibeta]
                p0, pc, ps = _boomerang_position_coeffs(flow, state, ifull)
                c0 += α[aidx] * (p0 - mean[ibeta])
                cc += α[aidx] * pc
                cs += α[aidx] * ps
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
    lo::Float64,
    hi::Float64,
)
    indices = beta_indices(provider)
    active_beta = _active_beta_from_free(provider, state.free)
    stickable_beta = _stickable_beta_from_can_stick(provider, can_stick)
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
    s_lo = s0 * exp(elo)
    s_hi = s0 * exp(ehi)
    zlo = min(mlo / s_lo, mlo / s_hi, mhi / s_lo, mhi / s_hi)
    zhi = max(mlo / s_lo, mlo / s_hi, mhi / s_lo, mhi / s_hi)
    z2_lo = _square_lower_from_range(zlo, zhi)
    q_upper = exp(-log(s0) - elo - 0.5 * z2_lo - 0.5 * log2π)
    total_weight = 0.0
    @inbounds for j in eachindex(indices)
        (stickable_beta[j] && !active_beta[j]) || continue
        logodds = log_model_add_odds(model_prior, active_beta, j)
        logodds == -Inf && continue
        logodds == Inf && return Inf
        total_weight += exp(logodds) * _boomerang_boundary_velocity_upper(flow, state, indices[j], lo, hi)
    end
    upper = q_upper * total_weight
    return upper + _fp_pad(upper)
end

_boomerang_boundary_rate_upper(clock::FourierResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab}, flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, lo::Float64, hi::Float64) =
    _global_logscale_boundary_rate_upper(clock.slab_provider, clock.model_prior, flow, state, can_stick, lo, hi)
_boomerang_boundary_rate_upper(clock::FourierResidualAggregateClock{<:AbstractGaussianSlabProvider}, flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, lo::Float64, hi::Float64) =
    _fixed_gaussian_boundary_rate_upper(clock.slab_provider, clock.model_prior, flow, state, can_stick, lo, hi)

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

function _fourier_envelope_hazard(env::_FourierResidualEnvelope, t::Real)
    return _fourier_primitive(env, t) + _residual_primitive(env, t)
end

function _fourier_envelope_rate(env::_FourierResidualEnvelope, t::Real)
    idx = _residual_cell_index(env, t)
    return _fourier_eval(env.a0, env.a, env.b, t, env.horizon) + env.cells[idx].R
end

function _certify_fourier_cell(clock::FourierResidualAggregateClock, flow::AnyBoomerang, state::StickyPDMPState, can_stick::BitVector, a0::Float64, a::Vector{Float64}, b::Vector{Float64}, lo::Float64, hi::Float64, T::Float64)
    λ_hi = _boomerang_boundary_rate_upper(clock, flow, state, can_stick, lo, hi)
    q_lo = _fourier_lower_on_cell(a0, a, b, lo, hi, T)
    R = max(0.0, λ_hi - q_lo, -q_lo)
    R += _fp_pad(max(λ_hi, abs(q_lo), R))
    return _FourierResidualCell(lo, hi, R, R * (hi - lo))
end

function _build_fourier_residual_envelope(clock::FourierResidualAggregateClock, flow::AnyBoomerang, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    T = Float64(horizon)
    f = t -> rate(clock.fallback, flow, state, t, can_stick)
    a0, a, b, fit_evaluations = _fourier_fit_coeffs(f, T, clock.order)
    cells = Vector{_FourierResidualCell}(undef, clock.cells)
    edges = Vector{Float64}(undef, clock.cells + 1)
    rate_evaluations = fit_evaluations
    for i in 1:clock.cells
        lo = T * (i - 1) / clock.cells
        hi = T * i / clock.cells
        cells[i] = _certify_fourier_cell(clock, flow, state, can_stick, a0, a, b, lo, hi, T)
        edges[i] = lo
    end
    edges[end] = T
    prefix = zeros(Float64, length(cells) + 1)
    for i in eachindex(cells)
        prefix[i + 1] = prefix[i] + cells[i].residual_area
    end
    Hbar_T = _fourier_primitive(_FourierResidualEnvelope(T, a0, a, b, cells, edges, prefix, 0.0), T) + prefix[end]
    return _FourierResidualEnvelope(T, a0, a, b, cells, edges, prefix, max(0.0, Hbar_T)), rate_evaluations
end

function rate(clock::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    τ < 0 && throw(ArgumentError("τ must be non-negative"))
    seg = scalar_logscale_gaussian_line_segment(clock.slab_provider, clock.model_prior, flow, state, can_stick, max(Float64(τ), eps(Float64)))
    return _scalar_logscale_gaussian_line_rate(seg, τ)
end

function sample_time(rng::Random.AbstractRNG, clock::ChebyshevResidualAggregateClock{<:GlobalLogscaleExchangeableGaussianSlab},
                     flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    if !isfinite(horizon)
        seg = scalar_logscale_gaussian_line_segment(clock.slab_provider, clock.model_prior, flow, state, can_stick, Inf)
        return _sample_scalar_logscale_gaussian_line_infinite(rng, seg;
            initial_bracket=clock.fallback.initial_bracket,
            bracket_multiplier=clock.fallback.bracket_multiplier,
            rtol=clock.fallback.rtol,
            atol=clock.fallback.atol)
    end
    horizon <= 0 && return Inf
    seg = scalar_logscale_gaussian_line_segment(clock.slab_provider, clock.model_prior, flow, state, can_stick, Float64(horizon))
    seg.log_total_weight == Inf && return 0.0
    env = _build_scalar_residual_envelope(clock, seg)
    d = clock.diagnostics
    d.last_cells = length(env.cells)
    d.residual_area += env.residual_prefix[end]
    d.envelope_hazard += env.Hbar_horizon
    d.max_residual = max(d.max_residual, maximum(c.R for c in env.cells))
    d.min_envelope = min(d.min_envelope, minimum(_envelope_rate(env, 0.5 * (c.lo + c.hi)) for c in env.cells))
    d.rate_evaluations += max(4 * clock.order + 1, 65)

    t = 0.0
    while t < horizon
        target = _envelope_hazard(env, t) + rand(rng, Exponential())
        if target > env.Hbar_horizon
            return Inf
        end
        f = τ -> _envelope_hazard(env, τ) - target
        T = Roots.find_zero(f, (t, Float64(horizon)), Roots.Bisection(); atol=clock.fallback.atol, rtol=clock.fallback.rtol)
        d.proposals += 1
        λ = _scalar_logscale_gaussian_line_rate(seg, T)
        λbar = _envelope_rate(env, T)
        λbar <= 0 && throw(ArgumentError("certified residual envelope produced a non-positive proposal rate"))
        ratio = λ / λbar
        d.max_envelope_ratio = max(d.max_envelope_ratio, ratio)
        tolerance = 1 + 1e-10
        ratio <= tolerance || throw(ArgumentError("certified residual envelope violation: true/proposal rate ratio $ratio exceeds 1"))
        if rand(rng) <= min(1.0, ratio)
            d.accepted += 1
            return T
        end
        d.rejected += 1
        t = T
    end
    return Inf
end

function sample_time(rng::Random.AbstractRNG, clock::FourierResidualAggregateClock, flow::AnyBoomerang, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    _has_inactive_stickable_beta(clock.fallback, state, can_stick) || return Inf
    rate(clock.fallback, flow, state, 0.0, can_stick) == Inf && return 0.0
    if !isfinite(horizon)
        return _sample_time_exact_fallback(rng, clock, flow, state, horizon, can_stick)
    end
    horizon <= 0 && return Inf

    env, rate_evaluations = _build_fourier_residual_envelope(clock, flow, state, horizon, can_stick)
    d = clock.diagnostics
    d.last_cells = length(env.cells)
    d.residual_area += env.residual_prefix[end]
    d.envelope_hazard += env.Hbar_horizon
    d.max_residual = max(d.max_residual, maximum(c.R for c in env.cells))
    d.min_envelope = min(d.min_envelope, minimum(_fourier_envelope_rate(env, 0.5 * (c.lo + c.hi)) for c in env.cells))
    d.rate_evaluations += rate_evaluations

    t = 0.0
    while t < horizon
        target = _fourier_envelope_hazard(env, t) + rand(rng, Exponential())
        target > env.Hbar_horizon && return Inf
        f = τ -> _fourier_envelope_hazard(env, τ) - target
        T = Roots.find_zero(f, (t, Float64(horizon)), Roots.Bisection(); atol=clock.fallback.atol, rtol=clock.fallback.rtol)
        d.proposals += 1
        λ = rate(clock.fallback, flow, state, T, can_stick)
        λbar = _fourier_envelope_rate(env, T)
        λbar <= 0 && throw(ArgumentError("Fourier residual envelope produced a non-positive proposal rate"))
        ratio = λ / λbar
        d.max_envelope_ratio = max(d.max_envelope_ratio, ratio)
        ratio <= 1 + 1e-10 || throw(ArgumentError("Fourier residual envelope violation: true/proposal rate ratio $ratio exceeds 1"))
        if rand(rng) <= min(1.0, ratio)
            d.accepted += 1
            return T
        end
        d.rejected += 1
        t = T
    end
    return Inf
end

function sample_label(rng::Random.AbstractRNG, clock::AbstractAggregateUnstickClock, flow::ContinuousDynamics, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    state_at = _clock_state_at(state, flow, τ)
    return sample_label(rng, clock, flow, state_at, can_stick)
end
