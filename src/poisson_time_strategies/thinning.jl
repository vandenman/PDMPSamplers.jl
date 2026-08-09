struct ThinningStrategy{T<:BoundStrategy} <: PoissonTimeStrategy
    c::T
end
_to_internal(x::ThinningStrategy, ::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, args...) = x

struct SubsamplingThinningState{A<:ThinningStrategy,S<:AbstractPDMPState,V<:AbstractVector} <: PoissonTimeStrategy
    strategy::A
    candidate::S
    empty_gradient::V
end

function _to_internal(strategy::ThinningStrategy, ::Random.AbstractRNG,
        flow::ContinuousDynamics, model::PDMPModel{<:SubsampledControlVariate},
        state::AbstractPDMPState, cache, stats::AbstractStatisticCounter)
    _validate_subsampling_thinning_envelope(flow, model.grad.envelope)
    return SubsamplingThinningState(strategy, copy(state), similar(state.ξ.x, 0))
end

_validate_subsampling_thinning_envelope(flow::PreconditionedDynamics, envelope) =
    _validate_subsampling_thinning_envelope(flow.dynamics, envelope)
function _validate_subsampling_thinning_envelope(::AnyBoomerang, envelope)
    envelope.component_cell_scales! === nothing && throw(ArgumentError(
        "SubsampledControlVariate ThinningStrategy requires an explicit certified " *
        "component_cell_scales! callback for Boomerang trajectories"))
    return nothing
end
function _validate_subsampling_thinning_envelope(
        ::Union{BouncyParticle,ZigZag}, envelope)
    _validate_unbounded_linear_scales(envelope.component_scales!)
    return nothing
end
_validate_unbounded_linear_scales(::Any) = throw(ArgumentError(
    "SubsampledControlVariate ThinningStrategy requires certified affine " *
    "component scales on an unbounded linear trajectory"))
_validate_unbounded_linear_scales(::CertifiedAffineComponentScales) = nothing
function _validate_unbounded_linear_scales(provider::TrajectoryComponentScales)
    any(ispositive, provider.growth_rates) && throw(ArgumentError(
        "SubsampledControlVariate ThinningStrategy cannot dominate positive " *
        "trajectory growth on an unbounded linear path with its affine clock; " *
        "use GridThinningStrategy"))
    return nothing
end
_validate_subsampling_thinning_envelope(flow::ContinuousDynamics, envelope) = throw(ArgumentError(
    "SubsampledControlVariate ThinningStrategy has no trajectory geometry for $(typeof(flow))"))

function initialize_cache(rng::Random.AbstractRNG, flow::ZigZag, ::CoordinateWiseGradient, thinningstrategy::ThinningStrategy, t::Real, ξ::SkeletonPoint)
    pq = PriorityQueue{Int,Float64}()
    for i in eachindex(ξ.x)
        abc_i = ab_i(i, ξ, thinningstrategy, flow, nothing)
        t_event = t + poisson_time(abc_i[1], abc_i[2], rand(rng))
        push!(pq, i => t_event)
    end
    return (; pq)
end

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
ab(state::AbstractPDMPState, c::PoissonTimeStrategy,
    flow::ContinuousDynamics, cache) = ab(state, get_bounds(c), flow, cache)
ab(state::AbstractPDMPState, c::AbstractVector,
    flow::ContinuousDynamics, cache) = ab(state.ξ, c, flow, cache)
ab_i(i::Int, ξ::SkeletonPoint, c::PoissonTimeStrategy, flow::ContinuousDynamics, cache) = ab_i(i, ξ, get_bounds(c), flow, cache)

function next_event_time(rng::Random.AbstractRNG, ::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::ThinningStrategy{<:BoundStrategy}, state::AbstractPDMPState, cache, stats::AbstractStatisticCounter,
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

function _subsampling_thinning_residual_coefficients(
        envelope::SeparableResidualEnvelope, state::AbstractPDMPState,
        flow::ContinuousDynamics)
    return _subsampling_thinning_residual_coefficients(
        envelope, state, flow, flow)
end

function _subsampling_thinning_residual_coefficients(envelope, state, flow, ::AnyBoomerang)
    scales = component_cell_scales!(envelope.cell_scales, envelope, state,
        flow, zero(state.t[]), 2π)
    return dot(scales, envelope.totals), 0.0
end

_subsampling_thinning_residual_coefficients(envelope, state, flow,
        wrapped::PreconditionedDynamics) =
    _subsampling_thinning_residual_coefficients(
        envelope, state, flow, wrapped.dynamics)

_subsampling_thinning_residual_coefficients(envelope, state, flow,
        ::Union{BouncyParticle,ZigZag}) =
    _subsampling_linear_residual_coefficients(
        envelope, state, flow, envelope.component_scales!)

function _subsampling_linear_residual_coefficients(envelope, state, flow,
        provider::TrajectoryComponentScales)
    displacement0, velocity = trajectory_geometry_bounds(
        flow, state, provider.anchor, 0.0)
    total_weight = sum(envelope.totals)
    return velocity * displacement0 * total_weight,
        velocity * norm(state.ξ.θ) * total_weight
end

function _subsampling_linear_residual_coefficients(envelope, state, flow,
        provider::DampedHCVComponentScales)
    displacement0, velocity = trajectory_geometry_bounds(
        flow, state, provider.anchor, 0.0)
    totals = envelope.totals
    return velocity * (displacement0 * totals[1] +
            provider.damping * totals[2]),
        velocity * norm(state.ξ.θ) * totals[1]
end

_subsampling_linear_residual_coefficients(envelope, state, flow, provider) =
    throw(ArgumentError("SubsampledControlVariate ThinningStrategy requires " *
        "certified affine component scales on an unbounded linear trajectory"))

_subsampling_linear_residual_coefficients(envelope, state, flow,
        provider::CertifiedAffineComponentScales) =
    _generic_subsampling_linear_residual_coefficients(envelope, state, flow)

function _generic_subsampling_linear_residual_coefficients(envelope, state, flow)
    B0 = total_residual_bound(envelope, state, flow, 0.0)
    B1 = total_residual_bound(envelope, state, flow, 1.0)
    return B0, pos(B1 - B0)
end

function _subsampling_bound_violation(::SubsamplingThinningState, stats,
        actual, bound, kind)
    _bound_violated(actual, bound) || return false
    _inc_counter_grid_bound_violations(stats)
    throw(ErrorException("subsampling ThinningStrategy $(kind) bound violated: " *
        "actual=$(actual), bound=$(bound); the invalid proposal was discarded"))
end

function next_event_time(rng::Random.AbstractRNG,
        model::PDMPModel{<:SubsampledControlVariate}, flow::ContinuousDynamics,
        alg::SubsamplingThinningState, state::AbstractPDMPState, cache,
        stats::AbstractStatisticCounter, max_horizon::Real=Inf,
        include_refresh::Bool=true, max_horizon_event::Symbol=:horizon_hit)
    cv = model.grad
    a, b = ab(state, alg.strategy, flow, cache)
    B_a, B_b = _subsampling_thinning_residual_coefficients(cv.envelope, state, flow)
    roof_a = pos(a) + B_a
    roof_b = pos(b) + B_b
    all(isfinite, (roof_a, roof_b)) || throw(ArgumentError(
        "subsampling ThinningStrategy requires finite affine envelope coefficients"))

    refresh = if include_refresh && ispositive(refresh_rate(flow))
        Random.randexp(rng) / refresh_rate(flow)
    else
        Inf
    end
    limit = min(refresh, max_horizon)
    limit_event = refresh <= max_horizon ? :refresh : max_horizon_event
    τ = 0.0
    default_meta = GradientMeta(alg.empty_gradient)

    while true
        Δ = poisson_time(roof_a + roof_b * τ, roof_b, rand(rng))
        τ += Δ
        if τ >= limit || !isfinite(τ)
            return limit, limit_event, default_meta
        end

        _inc_counter_subsampling_cell_roof_proposals(stats)
        D = pos(a + b * τ)
        B = total_residual_bound(cv.envelope, state, flow, τ)
        roof = roof_a + roof_b * τ
        _subsampling_bound_violation(alg, stats, D + B, roof, :aggregate)
        aggregate = D + B
        if !ispositive(aggregate) || rand(rng) * roof > aggregate
            continue
        end
        _inc_counter_subsampling_aggregate_accepts(stats)

        result = _evaluate_subsampling_candidate!(
            rng, cv, flow, state, alg.candidate, cache, stats, alg, τ, D, B)
        if result.accepted
            _inc_counter_subsampling_final_reflections(stats)
            return τ, :reflect, GradientMeta(result.G)
        end
    end
end

function next_event_time(rng::Random.AbstractRNG,
        model::PDMPModel{<:SubsampledControlVariate}, flow::ContinuousDynamics,
        alg::SubsamplingThinningState, state::AbstractPDMPState, cache,
        stats::AbstractStatisticCounter, max_horizon::Real,
        include_refresh::Bool, max_horizon_event::Symbol,
        detect_boundaries::Bool)
    detect_boundaries && throw(ArgumentError(
        "support-boundary detection is not yet supported by SubsampledControlVariate ThinningStrategy"))
    return next_event_time(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event)
end

accept_reflection_event(::Random.AbstractRNG, ::SubsamplingThinningState, args...) = true
accept_reflection_event(::SubsamplingThinningState, args...) = true

function next_event_time(rng::Random.AbstractRNG, ::PDMPModel{<:CoordinateWiseGradient}, ::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, ::AbstractStatisticCounter)
    pq = cache.pq # rename for clarity
    i₀, t_event = first(pq)
    τ = t_event - state.t[]
    ispositive(τ) || error("$τ > $(zero(τ)) at t = $(state.t[]) with i₀ = $i₀ and t_event = $t_event")
    return τ, :reflect, CoordinateMeta(i₀)
end

function accept_reflection_event(rng::Random.AbstractRNG, ::ThinningStrategy,
        state::AbstractPDMPState, ∇ϕx::AbstractVector,
        flow::ContinuousDynamics, dt::Real, cache, meta::BoundsMeta)

    l = λ(state, ∇ϕx, flow)
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
