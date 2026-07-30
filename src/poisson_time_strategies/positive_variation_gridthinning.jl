"""
    PositiveVariationGridThinningStrategy(; kwargs...)

EXPERIMENTAL, intentionally unexported and still in development.

Automatic positive-variation event search for Boomerang dynamics.

This is an experimental GridThinning variant.  The default mode uses the
positive-variation grid schedule to build local tangent envelopes, then runs
ordinary thinning accept/reject against the exact rate.  It is therefore exact
under the same envelope assumptions as `GridThinningStrategy`.  The old
rejection-free positive-variation clock is available only with
`approximate=true`.
"""
struct PositiveVariationGridThinningStrategy <: PoissonTimeStrategy
    N::Int
    N_min::Int
    t_max::Float64
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    max_refinement_depth::Int
    validation_rtol::Float64
    validation_atol::Float64
    min_cell_width::Float64
    max_skip_width::Float64
    dense_cell_width::Float64
    skip_slope_safety::Float64
    use_derivative_hermite::Bool
    derivative_hermite_on_demand::Bool
    derivative_hermite_trigger_scale::Float64
    approximate::Bool
    fallback::GridThinningStrategy
end

function PositiveVariationGridThinningStrategy(; N::Int=20, N_min::Int=5, t_max::Real=2.0,
    α⁺::Real=1.5, α⁻::Real=0.5, safety_limit::Int=500, max_refinement_depth::Int=8,
    validation_rtol::Real=0.05, validation_atol::Real=1e-8, min_cell_width::Real=1e-8,
    max_skip_width::Real=0.25, dense_cell_width::Real=0.0, skip_slope_safety::Real=0.0,
    use_derivative_hermite::Bool=false, derivative_hermite_on_demand::Bool=false,
    derivative_hermite_trigger_scale::Real=10.0, approximate::Bool=false,
    fallback::GridThinningStrategy=GridThinningStrategy(; N, N_min, t_max, α⁺, α⁻, safety_limit,
        use_fd_hvp=true, bound=:constant))

    N > 0 || throw(ArgumentError("N must be positive"))
    N_min > 0 || throw(ArgumentError("N_min must be positive"))
    t_max > 0 || throw(ArgumentError("t_max must be positive"))
    max_refinement_depth >= 0 || throw(ArgumentError("max_refinement_depth must be nonnegative"))
    validation_rtol >= 0 || throw(ArgumentError("validation_rtol must be nonnegative"))
    validation_atol >= 0 || throw(ArgumentError("validation_atol must be nonnegative"))
    min_cell_width > 0 || throw(ArgumentError("min_cell_width must be positive"))
    max_skip_width > 0 || throw(ArgumentError("max_skip_width must be positive"))
    dense_cell_width >= 0 || throw(ArgumentError("dense_cell_width must be nonnegative"))
    skip_slope_safety >= 0 || throw(ArgumentError("skip_slope_safety must be nonnegative"))
    derivative_hermite_trigger_scale >= 0 ||
        throw(ArgumentError("derivative_hermite_trigger_scale must be nonnegative"))
    return PositiveVariationGridThinningStrategy(
        N, N_min, Float64(t_max), Float64(α⁺), Float64(α⁻), safety_limit,
        max_refinement_depth, Float64(validation_rtol), Float64(validation_atol),
        Float64(min_cell_width), Float64(max_skip_width), Float64(dense_cell_width),
        Float64(skip_slope_safety), use_derivative_hermite, derivative_hermite_on_demand,
        Float64(derivative_hermite_trigger_scale), approximate, fallback)
end

function Base.show(io::IO, strat::PositiveVariationGridThinningStrategy)
    print(io, "PositiveVariationGridThinningStrategy(")
    print(io, "N=", strat.N, ", t_max=", strat.t_max)
    print(io, ", max_refinement_depth=", strat.max_refinement_depth)
    print(io, ")")
end

struct PositiveVariationGridAdaptiveState{S<:AbstractPDMPState,V<:AbstractVector,P,D,F<:GridAdaptiveState} <: PoissonTimeStrategy
    N::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64
    α⁻::Float64
    safety_limit::Int
    N_min::Int
    max_refinement_depth::Int
    validation_rtol::Float64
    validation_atol::Float64
    min_cell_width::Float64
    max_skip_width::Float64
    dense_cell_width::Float64
    skip_slope_safety::Float64
    use_derivative_hermite::Bool
    derivative_hermite_on_demand::Bool
    derivative_hermite_trigger_scale::Float64
    approximate::Bool
    state_cache::S
    state_cache2::S
    empty_∇ϕx::V
    cached_gradient::V
    has_cached_gradient::Base.RefValue{Bool}
    cached_f::Base.RefValue{Float64}
    cached_ψ::Base.RefValue{Float64}
    cached_df::Base.RefValue{Float64}
    grad_provider::P
    derivative_provider::D
    fallback::F
end

function _to_internal(strat::PositiveVariationGridThinningStrategy, rng::Random.AbstractRNG,
    flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter)

    fallback = _to_internal(strat.fallback, rng, flow, model, state, cache, stats)
    state_cache = copy(state)
    state_cache2 = copy(state)
    grad_provider = GradientProvider(state_cache.ξ.θ, flow, model.grad, cache)
    fd_grad_provider = grad_provider
    derivative_provider = FiniteDiffVHV(
        fd_grad_provider, similar(state.ξ.x), similar(state.ξ.x), similar(state.ξ.x), stats)
    return PositiveVariationGridAdaptiveState(
        Ref(strat.N), Ref(strat.t_max), strat.α⁺, strat.α⁻, strat.safety_limit,
        min_grid_cells(flow, strat.N_min, strat.N), strat.max_refinement_depth,
        strat.validation_rtol, strat.validation_atol, strat.min_cell_width, strat.max_skip_width,
        strat.dense_cell_width, strat.skip_slope_safety, strat.use_derivative_hermite,
        strat.derivative_hermite_on_demand, strat.derivative_hermite_trigger_scale,
        strat.approximate,
        state_cache, state_cache2, similar(state.ξ.x, 0), similar(state.ξ.x),
        Ref(false), Ref(NaN), Ref(NaN), Ref(NaN), grad_provider, derivative_provider, fallback)
end

accept_reflection_event(::Random.AbstractRNG, ::PositiveVariationGridAdaptiveState, args...) = true
accept_reflection_event(::PositiveVariationGridAdaptiveState, args...) = true

function reset_grid_scale!(alg::PositiveVariationGridAdaptiveState, t_max::Float64=2.0)
    alg.t_max[] = t_max
    alg.N[] = max(alg.N[], alg.N_min)
    alg.has_cached_gradient[] = false
    alg.cached_f[] = NaN
    alg.cached_ψ[] = NaN
    alg.cached_df[] = NaN
    reset_grid_scale!(alg.fallback, t_max)
    return nothing
end

function _pv_cache_accept_node!(alg::PositiveVariationGridAdaptiveState,
    grad::AbstractVector, f::Float64, ψ::Float64, df::Float64)

    copyto!(alg.cached_gradient, grad)
    alg.cached_f[] = f
    alg.cached_ψ[] = ψ
    alg.cached_df[] = df
    alg.has_cached_gradient[] = true
    return nothing
end

function _pv_shrink_grid_N!(alg::PositiveVariationGridAdaptiveState)
    if alg.N[] > alg.N_min
        alg.N[] = max(alg.N_min, alg.N[] - 1)
    end
    return nothing
end

function _linear_positive_area(fa::Real, fb::Real, h::Real)
    T = promote_type(typeof(fa), typeof(fb), typeof(h))
    z = zero(T)
    half = one(T) / 2
    !ispositive(h) && return z
    if !ispositive(fa) && !ispositive(fb)
        return z
    elseif ispositive(fa) && ispositive(fb)
        return half * h * (fa + fb)
    elseif !ispositive(fa)
        u0 = -fa / (fb - fa)
        return half * h * fb * (one(T) - u0)
    else
        u0 = fa / (fa - fb)
        return half * h * fa * u0
    end
end

function _linear_positive_area_time(fa::Real, fb::Real, h::Real, area::Real)
    T = promote_type(typeof(fa), typeof(fb), typeof(h), typeof(area))
    z = zero(T)
    o = one(T)
    epsT = eps(float(o))
    !ispositive(area) && return z
    total = _linear_positive_area(fa, fb, h)
    area >= total && return h
    s = fb - fa
    if ispositive(fa) && ispositive(fb)
        if abs(s) <= epsT * max(abs(fa), abs(fb), o)
            return area / max(fa, epsT)
        end
        disc = max(fa * fa + 2o * s * area / h, z)
        u = (-fa + sqrt(disc)) / s
        return h * clamp(u, z, o)
    elseif !ispositive(fa) && ispositive(fb)
        u0 = -fa / s
        u = u0 + sqrt(max(2o * area / (h * s), z))
        return h * clamp(u, u0, o)
    elseif ispositive(fa) && !ispositive(fb)
        disc = max(fa * fa + 2o * s * area / h, z)
        u = (-fa + sqrt(disc)) / s
        return h * clamp(u, z, fa / (fa - fb))
    end
    return h
end

@inline function _pv_tol(alg::PositiveVariationGridAdaptiveState, scale::Float64)
    return alg.validation_atol + alg.validation_rtol * max(scale, 1.0)
end

@inline function _pv_accept_rate_tol(alg::PositiveVariationGridAdaptiveState, scale::Float64=1.0)
    return max(alg.validation_atol * max(scale, 1.0), 64eps(Float64))
end

function _pv_reference_potential_from_delta(Γ, Δ::AbstractVector, cache)
    z = cache.z
    mul!(z, Γ, Δ)
    return 0.5 * dot(Δ, z)
end

_pv_reference_potential_from_delta(Γ::LowRankPrecision, Δ::AbstractVector, cache) =
    0.5 * lowrank_quadform(Γ, Δ)

function _pv_reference_potential(flow::AnyBoomerang, x::AbstractVector, cache)
    μ = flow.μ
    Δ = cache.tmp
    @inbounds for i in eachindex(x)
        Δ[i] = x[i] - μ[i]
    end
    return _pv_reference_potential_from_delta(flow.Γ, Δ, cache)
end

function _pv_residual_potential(model::PDMPModel, flow::AnyBoomerang,
    state::AbstractPDMPState, cache)

    U = _last_gradient_potential(model)
    U === nothing && return NaN
    return Float64(U) - _pv_reference_potential(flow, state.ξ.x, cache)
end

function _pv_positive_area_floor(area::Float64, ψa::Float64, ψb::Float64)
    if isfinite(ψa) && isfinite(ψb)
        return max(area, ψb - ψa, 0.0)
    end
    return area
end

@inline function _pv_quadratic_integral(A::Float64, B::Float64, C::Float64, u::Float64)
    return u * evalpoly(u, (C, 0.5 * B, A / 3.0))
end

@inline function _pv_sort2(a::Float64, b::Float64)
    return a <= b ? (a, b) : (b, a)
end

@inline function _pv_sort3(a::Float64, b::Float64, c::Float64)
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
    end
    if a > b
        a, b = b, a
    end
    return a, b, c
end

@inline function _pv_sort4(a::Float64, b::Float64, c::Float64, d::Float64)
    if a > b
        a, b = b, a
    end
    if c > d
        c, d = d, c
    end
    if a > c
        a, c = c, a
    end
    if b > d
        b, d = d, b
    end
    if b > c
        b, c = c, b
    end
    return a, b, c, d
end

@inline function _pv_quadratic_value(A::Float64, B::Float64, C::Float64, u::Float64)
    return evalpoly(u, (C, B, A))
end

function _pv_quadratic_positive_area(A::Float64, B::Float64, C::Float64, h::Float64)
    h <= 0.0 && return 0.0
    b1 = 0.0
    b2 = 1.0
    b3 = 1.0
    b4 = 1.0
    n = 2
    if abs(A) <= eps(Float64) * max(abs(B), abs(C), 1.0)
        if abs(B) > eps(Float64) * max(abs(C), 1.0)
            r = -C / B
            if 0.0 < r < 1.0
                b2 = r
                b3 = 1.0
                n = 3
            end
        end
    else
        disc = B * B - 4.0 * A * C
        if ispositive(disc)
            sdisc = sqrt(disc)
            r1 = (-B - sdisc) / (2.0 * A)
            r2 = (-B + sdisc) / (2.0 * A)
            if r1 > r2
                r1, r2 = r2, r1
            end
            r1_inside = 0.0 < r1 < 1.0
            r2_inside = 0.0 < r2 < 1.0
            if r1_inside && r2_inside
                b1, b2, b3, b4 = _pv_sort4(0.0, r1, r2, 1.0)
                n = 4
            elseif r1_inside
                b1, b2, b3 = _pv_sort3(0.0, r1, 1.0)
                b4 = b3
                n = 3
            elseif r2_inside
                b1, b2, b3 = _pv_sort3(0.0, r2, 1.0)
                b4 = b3
                n = 3
            else
                b1 = 0.0
                b2 = 1.0
                b3 = 1.0
                b4 = 1.0
                n = 2
            end
        end
    end

    total = 0.0
    prev = b1
    @inbounds for j in 1:(n - 1)
        hi = ifelse(j == 1, b2, ifelse(j == 2, b3, b4))
        lo = prev
        mid = 0.5 * (lo + hi)
        qmid = _pv_quadratic_value(A, B, C, mid)
        if ispositive(qmid)
            total += _pv_quadratic_integral(A, B, C, hi) -
                _pv_quadratic_integral(A, B, C, lo)
        end
        prev = hi
    end
    return h * max(total, 0.0)
end

function _pv_hermite_positive_area(ψa::Float64, fa::Float64, ψb::Float64, fb::Float64, h::Float64)
    if !(isfinite(ψa) && isfinite(ψb)) || h <= 0.0
        return NaN
    end
    invh = inv(h)
    A = 6.0 * (ψa - ψb) * invh + 3.0 * (fa + fb)
    B = 6.0 * (ψb - ψa) * invh - 4.0 * fa - 2.0 * fb
    C = fa
    return _pv_quadratic_positive_area(A, B, C, h)
end

@inline function _pv_quadratic_positive_area_prefix(A::Float64, B::Float64, C::Float64,
    h::Float64, u::Float64)

    u <= 0.0 && return 0.0
    u >= 1.0 && return _pv_quadratic_positive_area(A, B, C, h)
    b1 = 0.0
    b2 = u
    b3 = u
    b4 = u
    n = 2
    if abs(A) <= eps(Float64) * max(abs(B), abs(C), 1.0)
        if abs(B) > eps(Float64) * max(abs(C), 1.0)
            r = -C / B
            if 0.0 < r < u
                b2 = r
                b3 = u
                n = 3
            end
        end
    else
        disc = B * B - 4.0 * A * C
        if ispositive(disc)
            sdisc = sqrt(disc)
            r1 = (-B - sdisc) / (2.0 * A)
            r2 = (-B + sdisc) / (2.0 * A)
            if r1 > r2
                r1, r2 = r2, r1
            end
            r1_inside = 0.0 < r1 < u
            r2_inside = 0.0 < r2 < u
            if r1_inside && r2_inside
                b1, b2, b3, b4 = _pv_sort4(0.0, r1, r2, u)
                n = 4
            elseif r1_inside
                b1, b2, b3 = _pv_sort3(0.0, r1, u)
                b4 = b3
                n = 3
            elseif r2_inside
                b1, b2, b3 = _pv_sort3(0.0, r2, u)
                b4 = b3
                n = 3
            end
        end
    end
    total = 0.0
    prev = b1
    @inbounds for j in 1:(n - 1)
        hi = ifelse(j == 1, b2, ifelse(j == 2, b3, b4))
        lo = prev
        mid = 0.5 * (lo + hi)
        if ispositive(_pv_quadratic_value(A, B, C, mid))
            total += _pv_quadratic_integral(A, B, C, hi) -
                _pv_quadratic_integral(A, B, C, lo)
        end
        prev = hi
    end
    return h * max(total, 0.0)
end

function _pv_hermite_positive_area_time(ψa::Float64, fa::Float64, ψb::Float64,
    fb::Float64, h::Float64, area::Float64)

    area <= 0.0 && return 0.0
    if !(isfinite(ψa) && isfinite(ψb)) || h <= 0.0
        return _linear_positive_area_time(fa, fb, h, area)
    end
    invh = inv(h)
    A = 6.0 * (ψa - ψb) * invh + 3.0 * (fa + fb)
    B = 6.0 * (ψb - ψa) * invh - 4.0 * fa - 2.0 * fb
    C = fa
    total = _pv_quadratic_positive_area(A, B, C, h)
    area >= total && return h
    lo = 0.0
    hi = 1.0
    for _ in 1:40
        mid = 0.5 * (lo + hi)
        amid = _pv_quadratic_positive_area_prefix(A, B, C, h, mid)
        if amid < area
            lo = mid
        else
            hi = mid
        end
    end
    return h * 0.5 * (lo + hi)
end

@inline function _pv_cubic_hermite_coeffs(fa::Float64, da::Float64, fb::Float64, db::Float64,
    h::Float64)

    hda = h * da
    hdb = h * db
    c0 = fa
    c1 = hda
    c2 = -3.0 * fa - 2.0 * hda + 3.0 * fb - hdb
    c3 = 2.0 * fa + hda - 2.0 * fb + hdb
    return c0, c1, c2, c3
end

@inline function _pv_cubic_hermite_value(fa::Float64, da::Float64, fb::Float64, db::Float64,
    h::Float64, u::Float64)

    c0, c1, c2, c3 = _pv_cubic_hermite_coeffs(fa, da, fb, db, h)
    return evalpoly(u, (c0, c1, c2, c3))
end

@inline function _pv_cubic_hermite_integral(fa::Float64, da::Float64, fb::Float64, db::Float64,
    h::Float64, u::Float64)

    c0, c1, c2, c3 = _pv_cubic_hermite_coeffs(fa, da, fb, db, h)
    return h * u * evalpoly(u, (c0, 0.5 * c1, c2 / 3.0, 0.25 * c3))
end

@inline function _pv_cubic_hermite_derivative(fa::Float64, da::Float64, fb::Float64,
    db::Float64, h::Float64, u::Float64)

    _, c1, c2, c3 = _pv_cubic_hermite_coeffs(fa, da, fb, db, h)
    return evalpoly(u, (c1, 2.0 * c2, 3.0 * c3)) / h
end

function _pv_bisect_cubic_hermite_root(fa::Float64, da::Float64, fb::Float64, db::Float64,
    h::Float64, lo::Float64, hi::Float64)

    flo = _pv_cubic_hermite_value(fa, da, fb, db, h, lo)
    for _ in 1:40
        mid = 0.5 * (lo + hi)
        fmid = _pv_cubic_hermite_value(fa, da, fb, db, h, mid)
        if signbit(fmid) == signbit(flo)
            lo = mid
            flo = fmid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function _pv_cubic_hermite_positive_area(fa::Float64, da::Float64, fb::Float64, db::Float64,
    h::Float64)

    if !(isfinite(da) && isfinite(db)) || h <= 0.0
        return NaN
    end

    # Break monotone intervals at extrema of the cubic Hermite rate.
    A = 6.0 * fa + 3.0 * h * da - 6.0 * fb + 3.0 * h * db
    B = -6.0 * fa - 4.0 * h * da + 6.0 * fb - 2.0 * h * db
    C = h * da
    p1 = 0.0
    p2 = 1.0
    p3 = 1.0
    p4 = 1.0
    npts = 2
    if abs(A) <= eps(Float64) * max(abs(B), abs(C), 1.0)
        if abs(B) > eps(Float64) * max(abs(C), 1.0)
            r = -C / B
            if 0.0 < r < 1.0
                p1, p2, p3 = _pv_sort3(0.0, r, 1.0)
                p4 = p3
                npts = 3
            end
        end
    else
        disc = B * B - 4.0 * A * C
        if ispositive(disc)
            sdisc = sqrt(disc)
            r1 = (-B - sdisc) / (2.0 * A)
            r2 = (-B + sdisc) / (2.0 * A)
            r1_inside = 0.0 < r1 < 1.0
            r2_inside = 0.0 < r2 < 1.0
            if r1_inside && r2_inside
                p1, p2, p3, p4 = _pv_sort4(0.0, r1, r2, 1.0)
                npts = 4
            elseif r1_inside
                p1, p2, p3 = _pv_sort3(0.0, r1, 1.0)
                p4 = p3
                npts = 3
            elseif r2_inside
                p1, p2, p3 = _pv_sort3(0.0, r2, 1.0)
                p4 = p3
                npts = 3
            end
        end
    end

    r1 = 1.0
    r2 = 1.0
    r3 = 1.0
    nroots = 0
    prev = p1
    for j in 1:(npts - 1)
        hi = ifelse(j == 1, p2, ifelse(j == 2, p3, p4))
        lo = prev
        flo = _pv_cubic_hermite_value(fa, da, fb, db, h, lo)
        fhi = _pv_cubic_hermite_value(fa, da, fb, db, h, hi)
        if iszero(flo) && 0.0 < lo < 1.0
            nroots += 1
            if nroots == 1
                r1 = lo
            elseif nroots == 2
                r2 = lo
            else
                r3 = lo
            end
        end
        if signbit(flo) != signbit(fhi)
            root = _pv_bisect_cubic_hermite_root(fa, da, fb, db, h, lo, hi)
            nroots += 1
            if nroots == 1
                r1 = root
            elseif nroots == 2
                r2 = root
            else
                r3 = root
            end
        end
        prev = hi
    end

    if nroots == 1
        p1, p2, p3 = _pv_sort3(0.0, r1, 1.0)
        p4 = p3
        npts = 3
    elseif nroots == 2
        p1, p2 = _pv_sort2(r1, r2)
        p1, p2, p3, p4 = _pv_sort4(0.0, p1, p2, 1.0)
        npts = 4
    elseif nroots >= 3
        r1, r2, r3 = _pv_sort3(r1, r2, r3)
        # A cubic has at most three roots; process the first three discovered
        # roots together with endpoints without allocating breakpoint storage.
        p1 = 0.0
        p2 = r1
        p3 = r2
        p4 = r3
        npts = 5
    else
        p1 = 0.0
        p2 = 1.0
        p3 = 1.0
        p4 = 1.0
        npts = 2
    end

    total = 0.0
    prev = p1
    for j in 1:(npts - 1)
        hi = ifelse(j == 1, p2, ifelse(j == 2, p3, ifelse(j == 3, p4, 1.0)))
        lo = prev
        hi <= lo && continue
        mid = 0.5 * (lo + hi)
        if ispositive(_pv_cubic_hermite_value(fa, da, fb, db, h, mid))
            total += _pv_cubic_hermite_integral(fa, da, fb, db, h, hi) -
                _pv_cubic_hermite_integral(fa, da, fb, db, h, lo)
        end
        prev = hi
    end
    return max(total, 0.0)
end

function _pv_cubic_hermite_positive_area_time(fa::Float64, da::Float64, fb::Float64,
    db::Float64, h::Float64, area::Float64)

    area <= 0.0 && return 0.0
    total = _pv_cubic_hermite_positive_area(fa, da, fb, db, h)
    if !isfinite(total)
        return _linear_positive_area_time(fa, fb, h, area)
    end
    area >= total && return h
    lo = 0.0
    hi = h
    for _ in 1:40
        mid = 0.5 * (lo + hi)
        u = mid / h
        fmid = _pv_cubic_hermite_value(fa, da, fb, db, h, u)
        dfmid = _pv_cubic_hermite_derivative(fa, da, fb, db, h, u)
        # Reuse the exact cubic-area routine on the restricted interval by
        # constructing the Hermite cell [0, mid] from the parent polynomial.
        amid = _pv_cubic_hermite_positive_area(fa, da, fmid, dfmid, mid)
        if amid < area
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

struct _PVAreaModel
    kind::Symbol
    area::Float64
end

function _pv_area_model(fa::Float64, fb::Float64, h::Float64,
    ψa::Float64, ψb::Float64, da::Float64=NaN, db::Float64=NaN)

    area = _pv_positive_area_floor(_linear_positive_area(fa, fb, h), ψa, ψb)
    kind = :linear
    hermite_area = _pv_hermite_positive_area(ψa, fa, ψb, fb, h)
    if isfinite(hermite_area) && hermite_area > area
        area = hermite_area
        kind = :potential_hermite
    end
    derivative_hermite_area = _pv_cubic_hermite_positive_area(fa, da, fb, db, h)
    if isfinite(derivative_hermite_area) && derivative_hermite_area > area
        area = derivative_hermite_area
        kind = :derivative_hermite
    end
    return _PVAreaModel(kind, area)
end

function _pv_positive_area_estimate(fa::Float64, fb::Float64, h::Float64,
    ψa::Float64, ψb::Float64, da::Float64=NaN, db::Float64=NaN)

    return _pv_area_model(fa, fb, h, ψa, ψb, da, db).area
end

function _pv_positive_area_time(model::_PVAreaModel, fa::Float64, fb::Float64, h::Float64,
    threshold::Float64, ψa::Float64, ψb::Float64, da::Float64, db::Float64)

    if model.kind === :potential_hermite
        return _pv_hermite_positive_area_time(ψa, fa, ψb, fb, h, threshold)
    elseif model.kind === :derivative_hermite
        return _pv_cubic_hermite_positive_area_time(fa, da, fb, db, h, threshold)
    end
    return _linear_positive_area_time(fa, fb, h, threshold)
end

function _pv_lipschitz_positive_area_upper(fa::Float64, fb::Float64, h::Float64, L::Float64)
    h <= 0.0 && return 0.0
    L <= 0.0 && return h * max(fa, fb, 0.0)
    n = 8
    total = 0.0
    prev = max(fa, 0.0)
    @inbounds for i in 1:n
        t = h * i / n
        upper = min(fa + L * t, fb + L * (h - t))
        cur = max(upper, 0.0)
        total += 0.5 * (prev + cur) * h / n
        prev = cur
    end
    return total
end

function _pv_signed_rate_from_gradient(state::AbstractPDMPState, grad::AbstractVector)
    return Float64(dot(grad, state.ξ.θ))
end

function _pv_signed_rate_derivative_value(alg::PositiveVariationGridAdaptiveState,
    state::AbstractPDMPState, flow::AnyBoomerang, grad::AbstractVector)

    _, derivative = rate_and_derivative(state, flow, alg.derivative_provider, grad)
    return Float64(derivative)
end

function _pv_signed_rate_derivative(alg::PositiveVariationGridAdaptiveState,
    state::AbstractPDMPState, flow::AnyBoomerang, grad::AbstractVector)

    alg.use_derivative_hermite || return NaN
    return _pv_signed_rate_derivative_value(alg, state, flow, grad)
end

@inline function _pv_derivative_hermite_on_demand(alg::PositiveVariationGridAdaptiveState)
    return alg.derivative_hermite_on_demand && !alg.use_derivative_hermite
end

@inline function _pv_should_try_derivative_on_demand(
    alg::PositiveVariationGridAdaptiveState, threshold::Float64, area::Float64,
    child_area::Float64, tol::Float64)

    _pv_derivative_hermite_on_demand(alg) || return false
    margin = alg.derivative_hermite_trigger_scale * max(tol, alg.validation_atol)
    envelope = max(area, child_area)
    return threshold <= envelope + margin || abs(child_area - area) > margin
end

function _pv_observe_derivative!(alg::PositiveVariationGridAdaptiveState, model::PDMPModel,
    flow::AnyBoomerang, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    t::Float64, probe_failure_handler::GridBoundaryProbe)

    state_t = alg.state_cache2
    copyto!(state_t, state)
    t != 0.0 && move_forward_time!(state_t, t, flow)
    _inc_counter_grid_endpoint_evaluations(stats)
    _inc_counter_grid_endpoint_gradient_calls(stats)
    grad = _compute_grid_gradient_or_throw!(
        state_t, state, flow, model, cache, max(0.0, prevfloat(t)), t, probe_failure_handler)
    _inc_counter_grid_points_evaluated(stats, 1)
    return _pv_signed_rate_derivative_value(alg, state_t, flow, grad)
end

function _pv_ensure_derivative!(alg::PositiveVariationGridAdaptiveState, model::PDMPModel,
    flow::AnyBoomerang, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    t::Float64, df::Float64, probe_failure_handler::GridBoundaryProbe)

    isfinite(df) && return df
    return _pv_observe_derivative!(alg, model, flow, state, cache, stats, t, probe_failure_handler)
end

function _pv_observe_endpoint!(alg::PositiveVariationGridAdaptiveState, model::PDMPModel,
    flow::AnyBoomerang, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    t::Float64, probe_failure_handler::GridBoundaryProbe)

    state_t = alg.state_cache
    copyto!(state_t, state)
    t != 0.0 && move_forward_time!(state_t, t, flow)
    _inc_counter_grid_endpoint_evaluations(stats)
    _inc_counter_grid_endpoint_gradient_calls(stats)
    grad = _compute_grid_gradient_or_throw!(
        state_t, state, flow, model, cache, max(0.0, prevfloat(t)), t, probe_failure_handler)
    _inc_counter_grid_points_evaluated(stats, 1)
    ψ = _pv_residual_potential(model, flow, state_t, cache)
    f = _pv_signed_rate_from_gradient(state_t, grad)
    df = _pv_signed_rate_derivative(alg, state_t, flow, grad)
    return Float64(t), f, ψ, df
end

function _pv_observe_candidate!(alg::PositiveVariationGridAdaptiveState, model::PDMPModel,
    flow::AnyBoomerang, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    t::Float64, probe_failure_handler::GridBoundaryProbe)

    state_t = alg.state_cache2
    copyto!(state_t, state)
    t != 0.0 && move_forward_time!(state_t, t, flow)
    _inc_counter_grid_acceptance_gradient_calls(stats)
    grad = _compute_grid_gradient_or_throw!(
        state_t, state, flow, model, cache, max(0.0, prevfloat(t)), t, probe_failure_handler)
    _inc_counter_grid_acceptance_tests(stats)
    ψ = _pv_residual_potential(model, flow, state_t, cache)
    f = _pv_signed_rate_from_gradient(state_t, grad)
    df = _pv_signed_rate_derivative(alg, state_t, flow, grad)
    return Float64(t), f, ψ, df, grad
end

function _pv_observe_probe!(alg::PositiveVariationGridAdaptiveState, model::PDMPModel,
    flow::AnyBoomerang, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
    t::Float64, probe_failure_handler::GridBoundaryProbe)

    state_t = alg.state_cache2
    copyto!(state_t, state)
    t != 0.0 && move_forward_time!(state_t, t, flow)
    _inc_counter_grid_endpoint_evaluations(stats)
    _inc_counter_grid_endpoint_gradient_calls(stats)
    grad = _compute_grid_gradient_or_throw!(
        state_t, state, flow, model, cache, max(0.0, prevfloat(t)), t, probe_failure_handler)
    _inc_counter_grid_points_evaluated(stats, 1)
    ψ = _pv_residual_potential(model, flow, state_t, cache)
    f = _pv_signed_rate_from_gradient(state_t, grad)
    df = _pv_signed_rate_derivative(alg, state_t, flow, grad)
    return Float64(t), f, ψ, df
end

struct _PVAccept{G}
    τ::Float64
    gradient::G
end
struct _PVSkip
    area::Float64
end
struct _PVFallback end

function _pv_dense_resolve_cell!(alg::PositiveVariationGridAdaptiveState,
    model::PDMPModel, flow::AnyBoomerang, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, a::Float64, fa::Float64, b::Float64,
    threshold::Float64, ψa::Float64, dfa::Float64, probe_failure_handler::GridBoundaryProbe)

    n_sub = 64
    prev_t = a
    prev_f = fa
    prev_ψ = ψa
    prev_df = dfa
    total = 0.0
    for i in 1:n_sub
        t = a + (b - a) * i / n_sub
        _, f, ψ, df = _pv_observe_probe!(alg, model, flow, state, cache, stats, t, probe_failure_handler)
        cell_model = _pv_area_model(prev_f, f, t - prev_t, prev_ψ, ψ, prev_df, df)
        area = cell_model.area
        if total + area >= threshold
            τ = prev_t + _pv_positive_area_time(
                cell_model, prev_f, f, t - prev_t, threshold - total, prev_ψ, ψ, prev_df, df)
            _, fτ, ψτ, dfτ, gradτ = _pv_observe_candidate!(
                alg, model, flow, state, cache, stats, τ, probe_failure_handler)
            if fτ <= _pv_accept_rate_tol(alg, max(abs(prev_f), abs(f), abs(fτ), 1.0))
                _inc_counter_positive_variation_fallbacks(stats)
                return _PVFallback()
            end
            _pv_cache_accept_node!(alg, gradτ, fτ, ψτ, dfτ)
            _inc_counter_positive_variation_accepts(stats)
            return _PVAccept(τ, alg.cached_gradient)
        end
        total += area
        prev_t = t
        prev_f = f
        prev_ψ = ψ
        prev_df = df
    end
    _inc_counter_positive_variation_skipped_cells(stats, 1)
    return _PVSkip(total)
end

function _pv_resolve_cell!(rng::Random.AbstractRNG, alg::PositiveVariationGridAdaptiveState,
    model::PDMPModel, flow::AnyBoomerang, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, a::Float64, fa::Float64, b::Float64, fb::Float64,
    threshold::Float64, depth::Int, ψa::Float64, ψb::Float64, dfa::Float64, dfb::Float64,
    probe_failure_handler::GridBoundaryProbe)

    _inc_counter_positive_variation_cells(stats)
    h = b - a
    if h <= alg.min_cell_width
        _inc_counter_positive_variation_fallbacks(stats)
        return _pv_dense_resolve_cell!(
            alg, model, flow, state, cache, stats, a, fa, b, threshold, ψa, dfa, probe_failure_handler)
    end
    if ispositive(alg.dense_cell_width) && h <= alg.dense_cell_width
        return _pv_dense_resolve_cell!(
            alg, model, flow, state, cache, stats, a, fa, b, threshold, ψa, dfa, probe_failure_handler)
    end
    area = _pv_positive_area_estimate(fa, fb, h, ψa, ψb, dfa, dfb)
    skip_upper_area = area
    tol = _pv_tol(alg, max(abs(threshold), area))

    if depth < alg.max_refinement_depth && h > 2alg.min_cell_width
        mid = 0.5 * (a + b)
        _, fm, ψm, dfm = _pv_observe_probe!(alg, model, flow, state, cache, stats, mid, probe_failure_handler)
        left_area = _pv_positive_area_estimate(fa, fm, mid - a, ψa, ψm, dfa, dfm)
        right_area = _pv_positive_area_estimate(fm, fb, b - mid, ψm, ψb, dfm, dfb)
        child_area = left_area + right_area
        child_tol = _pv_tol(alg, max(abs(threshold), area, child_area))
        if _pv_should_try_derivative_on_demand(alg, threshold, area, child_area, child_tol)
            dfa = _pv_ensure_derivative!(
                alg, model, flow, state, cache, stats, a, dfa, probe_failure_handler)
            dfm = _pv_ensure_derivative!(
                alg, model, flow, state, cache, stats, mid, dfm, probe_failure_handler)
            dfb = _pv_ensure_derivative!(
                alg, model, flow, state, cache, stats, b, dfb, probe_failure_handler)
            area = _pv_positive_area_estimate(fa, fb, h, ψa, ψb, dfa, dfb)
            left_area = _pv_positive_area_estimate(fa, fm, mid - a, ψa, ψm, dfa, dfm)
            right_area = _pv_positive_area_estimate(fm, fb, b - mid, ψm, ψb, dfm, dfb)
            child_area = left_area + right_area
            child_tol = _pv_tol(alg, max(abs(threshold), area, child_area))
        end
        if ispositive(alg.skip_slope_safety)
            half_h = 0.5 * h
            L = alg.skip_slope_safety * max(
                abs(fm - fa) / half_h,
                abs(fb - fm) / half_h,
                abs(fb - fa) / h,
                eps(Float64),
            )
            left_upper = _pv_lipschitz_positive_area_upper(fa, fm, half_h, L)
            right_upper = _pv_lipschitz_positive_area_upper(fm, fb, half_h, L)
            skip_upper_area = max(child_area, left_upper + right_upper)
        else
            skip_upper_area = child_area
        end
        if h > alg.max_skip_width || abs(child_area - area) > child_tol
            _inc_counter_positive_variation_refinements(stats)
            left_result = _pv_resolve_cell!(rng, alg, model, flow, state, cache, stats,
                a, fa, mid, fm, threshold, depth + 1, ψa, ψm, dfa, dfm, probe_failure_handler)
            if left_result isa _PVAccept || left_result isa _PVFallback
                return left_result
            end
            remaining = threshold - left_result.area
            if remaining <= _pv_tol(alg, threshold)
                _, f_mid, ψ_mid, df_mid, grad_mid = _pv_observe_candidate!(
                    alg, model, flow, state, cache, stats, mid, probe_failure_handler)
                if f_mid <= _pv_accept_rate_tol(alg, max(abs(fa), abs(fm), abs(fb), 1.0))
                    return _pv_dense_resolve_cell!(
                        alg, model, flow, state, cache, stats, a, fa, b, threshold, ψa, dfa,
                        probe_failure_handler)
                end
                _pv_cache_accept_node!(alg, grad_mid, f_mid, ψ_mid, df_mid)
                _inc_counter_positive_variation_accepts(stats)
                return _PVAccept(mid, alg.cached_gradient)
            end
            right_result = _pv_resolve_cell!(rng, alg, model, flow, state, cache, stats,
                mid, fm, b, fb, remaining, depth + 1, ψm, ψb, dfm, dfb, probe_failure_handler)
            if right_result isa _PVAccept || right_result isa _PVFallback
                return right_result
            else
                return _PVSkip(left_result.area + right_result.area)
            end
        end
        area = child_area
        tol = child_tol
    end

    if area + tol < threshold
        if skip_upper_area + tol >= threshold
            return _pv_dense_resolve_cell!(
                alg, model, flow, state, cache, stats, a, fa, b, threshold, ψa, dfa, probe_failure_handler)
        end
        _inc_counter_positive_variation_skipped_cells(stats, 1)
        return _PVSkip(area)
    end
    if area <= tol
        _inc_counter_positive_variation_skipped_cells(stats, 1)
        return _PVSkip(0.0)
    end

    proposal_model = _pv_area_model(fa, fb, h, ψa, ψb, dfa, dfb)
    τ = a + _pv_positive_area_time(proposal_model, fa, fb, h, threshold, ψa, ψb, dfa, dfb)
    τ = clamp(τ, nextfloat(a), prevfloat(b))
    _, fτ, ψτ, dfτ, gradτ = _pv_observe_candidate!(alg, model, flow, state, cache, stats, τ, probe_failure_handler)
    left_area = _pv_positive_area_estimate(fa, fτ, τ - a, ψa, ψτ, dfa, dfτ)
    right_area = _pv_positive_area_estimate(fτ, fb, b - τ, ψτ, ψb, dfτ, dfb)
    child_area = left_area + right_area
    defect = abs(child_area - area)
    child_tol = _pv_tol(alg, max(abs(threshold), child_area, area))
    if _pv_should_try_derivative_on_demand(alg, threshold, area, child_area, child_tol)
        dfτ = isfinite(dfτ) ? dfτ : _pv_signed_rate_derivative_value(alg, alg.state_cache2, flow, gradτ)
        dfa = _pv_ensure_derivative!(
            alg, model, flow, state, cache, stats, a, dfa, probe_failure_handler)
        dfb = _pv_ensure_derivative!(
            alg, model, flow, state, cache, stats, b, dfb, probe_failure_handler)
        area = _pv_positive_area_estimate(fa, fb, h, ψa, ψb, dfa, dfb)
        left_area = _pv_positive_area_estimate(fa, fτ, τ - a, ψa, ψτ, dfa, dfτ)
        right_area = _pv_positive_area_estimate(fτ, fb, b - τ, ψτ, ψb, dfτ, dfb)
        child_area = left_area + right_area
        defect = abs(child_area - area)
        child_tol = _pv_tol(alg, max(abs(threshold), child_area, area))
    end

    if abs(left_area - threshold) <= child_tol &&
            fτ > _pv_accept_rate_tol(alg, max(abs(fa), abs(fτ), abs(fb), 1.0))
        _pv_cache_accept_node!(alg, gradτ, fτ, ψτ, dfτ)
        _inc_counter_positive_variation_accepts(stats)
        return _PVAccept(τ, alg.cached_gradient)
    end

    if depth >= alg.max_refinement_depth || defect > max(10.0 * child_tol, 0.5 * max(area, child_area, 1.0))
        _inc_counter_positive_variation_fallbacks(stats)
        return _pv_dense_resolve_cell!(
            alg, model, flow, state, cache, stats, a, fa, b, threshold, ψa, dfa, probe_failure_handler)
    end

    _inc_counter_positive_variation_refinements(stats)
    if left_area + child_tol >= threshold
        return _pv_resolve_cell!(rng, alg, model, flow, state, cache, stats,
            a, fa, τ, fτ, threshold, depth + 1, ψa, ψτ, dfa, dfτ, probe_failure_handler)
    end
    right_threshold = threshold - left_area
    return _pv_resolve_cell!(rng, alg, model, flow, state, cache, stats,
        τ, fτ, b, fb, right_threshold, depth + 1, ψτ, ψb, dfτ, dfb, probe_failure_handler)
end

function _pv_fallback!(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy},
    flow::ContinuousDynamics, alg::PositiveVariationGridAdaptiveState, state::AbstractPDMPState,
    cache, stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, probe_failure_handler::GridBoundaryProbe)

    alg.has_cached_gradient[] = false
    return _next_event_time_with_probe(rng, model, flow, alg.fallback, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _next_positive_variation_envelope_event_time!(rng::Random.AbstractRNG,
    model::PDMPModel{<:GlobalGradientStrategy}, flow::AnyBoomerang,
    alg::PositiveVariationGridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, probe_failure_handler::GridBoundaryProbe)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
    hard_horizon = min(max_horizon, max_grid_horizon(flow))
    search_horizon = min(hard_horizon, τ_refresh)
    horizon_event = if τ_refresh <= hard_horizon
        :refresh
    else
        max_horizon <= max_grid_horizon(flow) ? max_horizon_event : :horizon_hit
    end
    default_return = GradientMeta(alg.empty_∇ϕx)
    if search_horizon <= 0.0
        return 0.0, horizon_event, default_return
    end

    provider = _grid_event_provider(model, flow, alg.fallback, stats)
    state_left = alg.state_cache
    state_prop = alg.state_cache2
    copyto!(state_left, state)

    _inc_counter_grid_builds(stats)
    _record_grid_schedule!(stats, alg)
    _set_counter_grid_N_current(stats, alg.N[])

    if alg.has_cached_gradient[]
        _inc_counter_grid_cached_endpoint_reuses(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_left, d_left = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_left, flow, provider, false, alg.cached_gradient;
            t_valid=0.0, t_invalid=0.0)
        alg.has_cached_gradient[] = false
    else
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_left, d_left = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_left, flow, provider, false;
            t_valid=0.0, t_invalid=0.0)
        _inc_counter_grid_points_evaluated(stats, 1)
    end

    h = alg.t_max[] / alg.N[]
    t_left = 0.0
    exp_target = Random.randexp(rng)
    cumulative_area = 0.0
    proposal_attempts = 0
    proposal_rejections = 0
    safety = alg.safety_limit

    while safety > 0
        safety -= 1
        t_right = min(t_left + h, search_horizon)
        Δt_cell = t_right - t_left
        if !ispositive(Δt_cell)
            alg.has_cached_gradient[] = false
            _pv_shrink_grid_N!(alg)
            _set_counter_grid_N_current(stats, alg.N[])
            return search_horizon, horizon_event, default_return
        end
        _inc_counter_positive_variation_cells(stats)

        move_forward_time!(state_left, Δt_cell, flow)
        _inc_counter_grid_endpoint_evaluations(stats)
        _inc_counter_grid_endpoint_gradient_calls(stats)
        _inc_counter_grid_endpoint_hessian_calls(stats)
        y_right, d_right = _get_rate_and_deriv_or_throw(
            probe_failure_handler, state_left, flow, provider, false;
            t_valid=t_left, t_invalid=t_right)
        _inc_counter_grid_points_evaluated(stats, 1)

        Λ_cell = _tangent_intersection_bound(t_left, t_right, y_left, y_right, d_left, d_right)
        lb_cell = pos(Λ_cell)
        area_cell = lb_cell * Δt_cell

        if cumulative_area + area_cell < exp_target
            _inc_counter_positive_variation_skipped_cells(stats, 1)
            cumulative_area += area_cell
            t_left = t_right
            y_left = y_right
            d_left = d_right
            if t_left >= search_horizon
                alg.has_cached_gradient[] = false
                _pv_shrink_grid_N!(alg)
                _set_counter_grid_N_current(stats, alg.N[])
                return search_horizon, horizon_event, default_return
            end
            if area_cell <= alg.validation_atol
                h = min(h * alg.α⁺, alg.t_max[])
            end
            continue
        end

        while true
            if lb_cell <= 0.0
                cumulative_area += area_cell
                break
            end
            τ_proposal = t_left + (exp_target - cumulative_area) / lb_cell
            if τ_proposal >= t_right || !isfinite(τ_proposal)
                cumulative_area += area_cell
                break
            end
            if τ_refresh < τ_proposal
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return τ_refresh, :refresh, default_return
            end

            copyto!(state_prop, state)
            move_forward_time!(state_prop, τ_proposal, flow)
            _inc_counter_grid_acceptance_gradient_calls(stats)
            gradτ = _compute_grid_gradient_or_throw!(
                state_prop, state, flow, model, cache, t_left, τ_proposal, probe_failure_handler)
            l_actual = λ(state_prop.ξ, gradτ, flow)
            _inc_counter_grid_acceptance_tests(stats)
            proposal_attempts += 1

            if l_actual > lb_cell * (1 + 1e-10) + 1e-12
                _inc_counter_grid_bound_violations(stats)
                _inc_counter_positive_variation_fallbacks(stats)
                alg.has_cached_gradient[] = false
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                return _pv_fallback!(rng, model, flow, alg, state, cache, stats,
                    max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end

            if rand(rng) * lb_cell <= l_actual
                copyto!(alg.cached_gradient, gradτ)
                alg.cached_f[] = Float64(dot(gradτ, state_prop.ξ.θ))
                alg.cached_ψ[] = _pv_residual_potential(model, flow, state_prop, cache)
                alg.cached_df[] = NaN
                alg.has_cached_gradient[] = true
                tightness = lb_cell <= 0.0 ? 0.0 : l_actual / lb_cell
                _adapt_grid_N!(alg.fallback, tightness)
                alg.N[] = max(alg.N_min, alg.fallback.N[])
                alg.t_max[] = min(max_grid_horizon(flow), max(alg.t_max[] * alg.α⁻, τ_proposal * alg.α⁺))
                _set_counter_grid_N_current(stats, alg.N[])
                _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
                _inc_counter_positive_variation_accepts(stats)
                return τ_proposal, :reflect, GradientMeta(alg.cached_gradient)
            end

            proposal_rejections += 1
            exp_target += Random.randexp(rng)
            if cumulative_area + area_cell < exp_target
                _inc_counter_positive_variation_skipped_cells(stats, 1)
                cumulative_area += area_cell
                break
            end
        end

        t_left = t_right
        y_left = y_right
        d_left = d_right
        if t_left >= search_horizon
            alg.has_cached_gradient[] = false
            _pv_shrink_grid_N!(alg)
            _set_counter_grid_N_current(stats, alg.N[])
            _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
            return search_horizon, horizon_event, default_return
        end
    end

    _inc_counter_positive_variation_fallbacks(stats)
    alg.has_cached_gradient[] = false
    _record_lazy_search_stats!(stats, proposal_attempts, proposal_rejections)
    return _pv_fallback!(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy},
    flow::FL, alg::PositiveVariationGridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64=Inf, include_refresh::Bool=true,
    max_horizon_event::Symbol=:horizon_hit) where {FL<:ContinuousDynamics}

    return _next_positive_variation_event_time!(
        rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh,
        max_horizon_event, NoGridBoundaryProbe())
end

function next_event_time(rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy},
    flow::FL, alg::PositiveVariationGridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, detect_boundaries::Bool) where {FL<:ContinuousDynamics}

    detect_boundaries || return next_event_time(
        rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh,
        max_horizon_event)
    probe_failure_handler = _grid_probe_failure_handler(state, flow, model, PositiveVariationGridThinningStrategy)
    return _next_positive_variation_event_time!(
        rng, model, flow, alg, state, cache, stats, max_horizon, include_refresh,
        max_horizon_event, probe_failure_handler)
end

function _next_positive_variation_event_time!(rng::Random.AbstractRNG,
    model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
    alg::PositiveVariationGridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, probe_failure_handler::GridBoundaryProbe)

    return _pv_fallback!(rng, model, flow, alg, state, cache, stats, max_horizon,
        include_refresh, max_horizon_event, probe_failure_handler)
end

function _next_positive_variation_event_time!(rng::Random.AbstractRNG,
    model::PDMPModel{<:GlobalGradientStrategy}, flow::AnyBoomerang,
    alg::PositiveVariationGridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol, probe_failure_handler::GridBoundaryProbe)

    if !alg.approximate
        return _next_positive_variation_envelope_event_time!(
            rng, model, flow, alg, state, cache, stats, max_horizon,
            include_refresh, max_horizon_event, probe_failure_handler)
    end

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
    hard_horizon = min(max_horizon, max_grid_horizon(flow))
    search_horizon = min(hard_horizon, τ_refresh)
    horizon_event = if τ_refresh <= hard_horizon
        :refresh
    else
        max_horizon <= max_grid_horizon(flow) ? max_horizon_event : :horizon_hit
    end
    default_return = GradientMeta(alg.empty_∇ϕx)
    if search_horizon <= 0.0
        return 0.0, horizon_event, default_return
    end

    _inc_counter_grid_builds(stats)
    _record_grid_schedule!(stats, alg)
    _set_counter_grid_N_current(stats, alg.N[])

    threshold = Random.randexp(rng)
    h = alg.t_max[] / alg.N[]
    t_left, f_left, ψ_left, df_left = if alg.has_cached_gradient[]
        _inc_counter_grid_cached_endpoint_reuses(stats)
        alg.has_cached_gradient[] = false
        # Reusing the accepted-event potential before coherent Hermite
        # inversion can increase refinements because the area model and inverse
        # still disagree. Keep the cached rate/derivative, but leave ψ disabled
        # until the selected-model inversion work lands.
        0.0, alg.cached_f[], NaN, alg.cached_df[]
    else
        _pv_observe_endpoint!(alg, model, flow, state, cache, stats, 0.0, probe_failure_handler)
    end

    safety = alg.safety_limit
    while safety > 0
        safety -= 1
        t_right = min(t_left + h, search_horizon)
        if t_right <= t_left
            alg.has_cached_gradient[] = false
            return search_horizon, horizon_event, default_return
        end
        _, f_right, ψ_right, df_right = _pv_observe_endpoint!(alg, model, flow, state, cache, stats, t_right, probe_failure_handler)
        result = _pv_resolve_cell!(rng, alg, model, flow, state, cache, stats,
            t_left, f_left, t_right, f_right, threshold, 0, ψ_left, ψ_right, df_left, df_right,
            probe_failure_handler)
        if result isa _PVAccept
            _pv_shrink_grid_N!(alg)
            alg.t_max[] = min(max_grid_horizon(flow), max(alg.t_max[] * alg.α⁻, result.τ * alg.α⁺))
            _set_counter_grid_N_current(stats, alg.N[])
            return result.τ, :reflect, GradientMeta(result.gradient)
        elseif result isa _PVSkip
            threshold -= result.area
            t_left = t_right
            f_left = f_right
            ψ_left = ψ_right
            df_left = df_right
            if t_left >= search_horizon
                alg.has_cached_gradient[] = false
                _pv_shrink_grid_N!(alg)
                _set_counter_grid_N_current(stats, alg.N[])
                return search_horizon, horizon_event, default_return
            end
            if result.area <= alg.validation_atol
                h = min(h * alg.α⁺, alg.t_max[])
            end
        else
            return _pv_fallback!(rng, model, flow, alg, state, cache, stats,
                max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
        end
    end

    _inc_counter_positive_variation_fallbacks(stats)
    return _pv_fallback!(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end
