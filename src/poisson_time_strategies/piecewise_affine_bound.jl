"""
    PiecewiseAffineBound{T}

Reusable internal representation of a nonnegative piecewise-affine envelope.

Each active segment `j` is represented on `[t_breaks[j], t_breaks[j+1]]`
as

```julia
q_j(u) = y_left[j] + slopes[j] * u,  u = t - t_breaks[j].
```

`cum_area` stores the exact integrated envelope at the break points. This
type is intentionally independent of PDMP state movement; GridThinning can
later use it as a proposal envelope.
"""
mutable struct PiecewiseAffineBound{T<:Real}
    t_breaks::Vector{T}
    y_left::Vector{T}
    slopes::Vector{T}
    cum_area::Vector{T}
    n_segments::Int
end

function PiecewiseAffineBound{T}(capacity::Integer=0) where {T<:Real}
    capacity < 0 && throw(ArgumentError("capacity must be nonnegative"))
    t_breaks = Vector{T}(undef, capacity + 1)
    y_left = Vector{T}(undef, capacity)
    slopes = Vector{T}(undef, capacity)
    cum_area = Vector{T}(undef, capacity + 1)
    if !isempty(cum_area)
        cum_area[1] = zero(T)
    end
    return PiecewiseAffineBound{T}(t_breaks, y_left, slopes, cum_area, 0)
end

PiecewiseAffineBound(capacity::Integer=0) = PiecewiseAffineBound{Float64}(capacity)

function PiecewiseAffineBound(
    t_breaks::AbstractVector{T},
    y_left::AbstractVector{T},
    slopes::AbstractVector{T},
) where {T<:Real}
    n = length(y_left)
    length(slopes) == n || throw(ArgumentError("slopes must have length $(n)"))
    length(t_breaks) == n + 1 || throw(ArgumentError("t_breaks must have length $(n + 1)"))
    bound = PiecewiseAffineBound{T}(n)
    reset_affine_bound!(bound)
    for j in 1:n
        append_affine_segment!(bound, t_breaks[j], t_breaks[j + 1], y_left[j], slopes[j])
    end
    return bound
end

function reset_affine_bound!(bound::PiecewiseAffineBound{T}) where {T}
    bound.n_segments = 0
    if isempty(bound.cum_area)
        resize!(bound.cum_area, 1)
    end
    bound.cum_area[1] = zero(T)
    return bound
end

function _ensure_affine_capacity!(bound::PiecewiseAffineBound{T}, n_needed::Integer) where {T}
    current = length(bound.y_left)
    n_needed <= current && return nothing
    new_capacity = max(n_needed, max(1, 2current))
    resize!(bound.y_left, new_capacity)
    resize!(bound.slopes, new_capacity)
    resize!(bound.t_breaks, new_capacity + 1)
    resize!(bound.cum_area, new_capacity + 1)
    return nothing
end

_affine_segment_value(alpha, beta, u) = alpha + beta * u

function _affine_segment_area(alpha::Real, beta::Real, h::Real)
    return alpha * h + oftype(alpha * h, 0.5) * beta * h^2
end

function _affine_nonnegative_tolerance(vals...)
    scale = mapreduce(x -> abs(float(x)), max, vals; init=1.0)
    return 128 * eps(Float64) * max(1.0, scale)
end

function _check_affine_segment(t_left::Real, t_right::Real, alpha::Real, beta::Real)
    all(isfinite, (t_left, t_right, alpha, beta)) ||
        throw(ArgumentError("affine segment inputs must be finite"))

    h = t_right - t_left
    h > 0 || throw(ArgumentError("affine segment must have positive width"))

    y_right = _affine_segment_value(alpha, beta, h)
    tol = _affine_nonnegative_tolerance(alpha, y_right)
    if alpha < -tol || y_right < -tol
        throw(ArgumentError("affine segment endpoint values must be nonnegative"))
    end

    area = _affine_segment_area(alpha, beta, h)
    area_tol = _affine_nonnegative_tolerance(area, alpha * h, y_right * h)
    if area < -area_tol
        throw(ArgumentError("affine segment area must be nonnegative"))
    end

    return h, max(area, zero(area))
end

function append_affine_segment!(
    bound::PiecewiseAffineBound{T},
    t_left::Real,
    t_right::Real,
    alpha::Real,
    beta::Real,
) where {T}
    h, area = _check_affine_segment(t_left, t_right, alpha, beta)

    n_old = bound.n_segments
    if n_old > 0
        previous_right = bound.t_breaks[n_old + 1]
        tol = _affine_nonnegative_tolerance(previous_right, t_left)
        if abs(previous_right - t_left) > tol
            throw(ArgumentError("affine segments must be appended in contiguous order"))
        end
        t_left = previous_right
    end

    n_new = n_old + 1
    _ensure_affine_capacity!(bound, n_new)
    if n_old == 0
        bound.t_breaks[1] = T(t_left)
        bound.cum_area[1] = zero(T)
    end
    bound.t_breaks[n_new + 1] = T(t_right)
    bound.y_left[n_new] = T(alpha)
    bound.slopes[n_new] = T(beta)
    bound.cum_area[n_new + 1] = bound.cum_area[n_new] + T(area)
    bound.n_segments = n_new
    return bound
end

function _active_cum_area(bound::PiecewiseAffineBound)
    return @view bound.cum_area[1:(bound.n_segments + 1)]
end

function total_area(bound::PiecewiseAffineBound)
    bound.n_segments == 0 && return zero(eltype(bound.cum_area))
    return bound.cum_area[bound.n_segments + 1]
end

function (bound::PiecewiseAffineBound)(t::Real)
    n = bound.n_segments
    n == 0 && return 0.0
    t < bound.t_breaks[1] && return 0.0
    t > bound.t_breaks[n + 1] && return 0.0

    if t == bound.t_breaks[n + 1]
        j = n
    else
        j = searchsortedlast(@view(bound.t_breaks[1:(n + 1)]), t)
        j = clamp(j, 1, n)
    end

    u = t - bound.t_breaks[j]
    return bound.y_left[j] + bound.slopes[j] * u
end

function _invert_affine_segment(alpha::Real, beta::Real, h::Real, area_budget::Real)
    area_budget < 0 && throw(ArgumentError("area budget must be nonnegative"))
    iszero(area_budget) && return zero(h)
    full_area = _affine_segment_area(alpha, beta, h)
    if full_area <= 0
        area_budget == 0 && return zero(h)
        throw(ArgumentError("cannot invert a zero-area affine segment for positive area"))
    end

    budget_tol = _affine_nonnegative_tolerance(area_budget, full_area)
    if area_budget > full_area + budget_tol
        throw(ArgumentError("area budget exceeds segment area"))
    end
    r = min(area_budget, full_area)

    if iszero(beta)
        alpha <= 0 && throw(ArgumentError("cannot use constant inverse with nonpositive rate"))
        u = r / alpha
    else
        discriminant = alpha^2 + 2 * beta * r
        disc_tol = _affine_nonnegative_tolerance(discriminant, alpha^2, beta * r)
        if discriminant < -disc_tol
            throw(ArgumentError("negative affine inverse discriminant"))
        end
        sqrt_disc = sqrt(max(discriminant, zero(discriminant)))
        denom = alpha + sqrt_disc
        denom > 0 || throw(ArgumentError("unstable affine inverse denominator"))
        u = 2 * r / denom
    end

    u_tol = _affine_nonnegative_tolerance(u, h)
    if u < -u_tol || u > h + u_tol
        throw(ArgumentError("affine inverse left the segment"))
    end
    return clamp(u, zero(h), h)
end

function propose_event_time(
    rng::Random.AbstractRNG,
    bound::PiecewiseAffineBound,
    u::Real=rand(rng, Exponential()),
)
    u < 0 && throw(ArgumentError("integrated hazard budget must be nonnegative"))
    n = bound.n_segments
    n == 0 && return (Inf, 0.0)

    total = total_area(bound)
    if u >= total
        return (Inf, 0.0)
    end
    if iszero(u)
        τ = bound.t_breaks[1]
        return τ, bound(τ)
    end

    cum = _active_cum_area(bound)
    j = searchsortedlast(cum, u)
    j = clamp(j, 1, n)

    # Skip duplicate cumulative entries from zero-area segments.
    while j <= n && bound.cum_area[j + 1] <= u
        j += 1
    end
    j > n && return (Inf, 0.0)

    r = u - bound.cum_area[j]
    h = bound.t_breaks[j + 1] - bound.t_breaks[j]
    local_u = _invert_affine_segment(bound.y_left[j], bound.slopes[j], h, r)
    τ = bound.t_breaks[j] + local_u
    return τ, bound.y_left[j] + bound.slopes[j] * local_u
end

propose_event_time(bound::PiecewiseAffineBound, u::Real) =
    propose_event_time(Random.default_rng(), bound, u)
