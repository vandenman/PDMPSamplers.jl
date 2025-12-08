struct ThinningStrategy{T<:BoundStrategy} <: PoissonTimeStrategy
    c::T
end
_to_internal(x::PoissonTimeStrategy, args...) = x

function poisson_time(a::Number, u::Number)
    a <= 0 && return Inf
    -log(u) / a
end

function poisson_time(a, b, u)
    if ispositive(b)
        if isnegative(a)
            return sqrt(-log(u) * 2.0 / b) - a / b
        else
            return sqrt((a / b)^2 - log(u) * 2.0 / b) - a / b
        end
    elseif iszero(b)
        if ispositive(a)
            return -log(u) / a
        else
            return Inf
        end
    else
        return Inf
    end
end

function next_time(t, abc, z = rand())
    a, b, refresh_time = abc
    Δt = poisson_time(a, b, z)
    if Δt > refresh_time
        return t + refresh_time, true
    else
        return t + Δt, false
    end
end



# Concrete bound strategies
struct GlobalBounds <: BoundStrategy
    c::Float64 # could be FillArrays.Fill
    d::Int
end

struct LocalBounds <: BoundStrategy
    c::Vector{Float64}
end

get_bounds(b::ThinningStrategy) = get_bounds(b.c)
get_bounds(b::BoundStrategy) = b.c
get_bounds(b::GlobalBounds) = FillArrays.Fill(b.c, b.d)

ab(          ξ::SkeletonPoint, c::PoissonTimeStrategy, flow::ContinuousDynamics, cache) = ab(     ξ, get_bounds(c), flow, cache)
ab_i(i::Int, ξ::SkeletonPoint, c::PoissonTimeStrategy, flow::ContinuousDynamics, cache) = ab_i(i, ξ, get_bounds(c), flow, cache)

function next_event_time(grad::GlobalGradientStrategy, flow::ContinuousDynamics, alg::ThinningStrategy{<:BoundStrategy}, state::AbstractPDMPState, cache, stats::StatisticCounter)

    # t = state.t[]
    ξ = state.ξ
    # cache = add_gradient_to_cache((;), ξ)
    # abc = ab(ξ, alg, flow, cache)

    # t′, renew = next_time(t, abc, rand())
    # dt = t′ - t
    # event_type = renew ? :refresh : :reflect

    # return dt, event_type, abc

    # this step only should be done through dispatch!
    abc = ab(ξ, alg, flow, cache)
    reflect_time = poisson_time(abc[1], abc[2], rand())

    refresh_time = rand_refresh_time(flow)

    # at this point maybe sample/ compute the sticky times?


    # determine the first arrival
    if reflect_time < refresh_time
        dt = reflect_time
        event_type =  :reflect
    else
        dt = refresh_time
        event_type = :refresh
    end

    # probably we always want to return abc for type stability?
    return dt, event_type, abc

end

function next_event_time(::CoordinateWiseGradient, ::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, ::StatisticCounter)
    pq = cache.pq # rename for clarity
    i₀, t_event = dequeue_pair!(pq)
    τ = t_event - state.t[]
    @assert ispositive(τ) "$τ > $(zero(τ)) at t = $(state.t[]) with i₀ = $i₀ and t_event = $t_event"
    return τ, nothing, i₀ # meta = winning coordinate
end
