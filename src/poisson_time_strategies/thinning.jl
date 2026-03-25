struct ThinningStrategy{T<:BoundStrategy} <: PoissonTimeStrategy
    c::T
end
_to_internal(x::ThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, args...) = x

function poisson_time(a::Number, u::Number)
    !ispositive(a) && return Inf
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

next_time(t, abc, z) = next_time(Random.default_rng(), t, abc, z)

function next_time(rng::Random.AbstractRNG, t, abc, z=rand(rng))
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

ab(ξ::SkeletonPoint, c::PoissonTimeStrategy, flow::ContinuousDynamics, cache) = ab(ξ, get_bounds(c), flow, cache)
ab_i(i::Int, ξ::SkeletonPoint, c::PoissonTimeStrategy, flow::ContinuousDynamics, cache) = ab_i(i, ξ, get_bounds(c), flow, cache)

function next_event_time(rng::Random.AbstractRNG, ::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::ThinningStrategy{<:BoundStrategy}, state::AbstractPDMPState, cache, stats::StatisticCounter,
    # TODO: these only exist temporarily due to issues/ testing in gridthinning
    ignored1::Any=nothing, ignored2::Any=nothing
)

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
    reflect_time = poisson_time(abc[1], abc[2], rand(rng))

    refresh_time = rand_refresh_time(rng, flow)

    # at this point maybe sample/ compute the sticky times?


    # determine the first arrival
    if reflect_time < refresh_time
        dt = reflect_time
        event_type = :reflect
    else
        dt = refresh_time
        event_type = :refresh
    end

    return dt, event_type, BoundsMeta(abc[1], abc[2])

end

function next_event_time(rng::Random.AbstractRNG, ::PDMPModel{<:CoordinateWiseGradient}, ::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, ::StatisticCounter)
    pq = cache.pq # rename for clarity
    # i₀, t_event = dequeue_pair!(pq)
    i₀, t_event = Base.popfirst!(pq)
    τ = t_event - state.t[]
    @assert ispositive(τ) "$τ > $(zero(τ)) at t = $(state.t[]) with i₀ = $i₀ and t_event = $t_event"
    return τ, nothing, CoordinateMeta(i₀)
end

function accept_reflection_event(rng::Random.AbstractRNG, ::ThinningStrategy, ξ::SkeletonPoint, ∇ϕx::AbstractVector, flow::ContinuousDynamics, dt::Real, cache, meta::BoundsMeta)

    l = λ(ξ, ∇ϕx, flow)
    l_bound = pos(meta.a + meta.b * dt)

    # TODO: don't throw when adapting!
    #l > l_bound && !(l <= 1e-6) && error("Tuning parameter `c` too small: l=$l, lb=$l_bound")
    # for now, the bound should be way tighter
    # l / l_bound < 0.6 && error("Tuning parameter `c` too large? dt = $dt, l=$l, lb=$l_bound, l / l_bound = $(l / l_bound)")

    u = rand(rng)
    accept = u * l_bound <= l

    if accept
        l > l_bound && !(l <= 1e-6) && error("Tuning parameter `c` too small: l=$l, lb=$l_bound")
    else
        # @info "rejecting u * l_bound = $(u) * $(l_bound) = $(u * l_bound) <= l = $l where meta=$meta and dt=$dt"
    end

    return accept
end
