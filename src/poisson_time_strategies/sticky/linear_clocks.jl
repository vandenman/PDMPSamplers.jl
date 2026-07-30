
_log_model_add_odds_with_count(prior::AbstractModelPrior, active::BitVector, j::Integer, ::Integer) =
    log_model_add_odds(prior, active, j)
function _log_model_add_odds_with_count(prior::BetaBernoulliModelPrior, active::BitVector, j::Integer, k::Integer)
    active[j] && throw(ArgumentError("log_model_add_odds is defined for adding an inactive coordinate; coordinate $j is already active"))
    denom = prior.b + prior.n - k - 1
    denom <= 0 && return Inf
    return log(prior.a + k) - log(denom)
end
function _log_model_add_odds_with_count(prior::ExchangeableModelSizePrior, active::BitVector, j::Integer, k::Integer)
    p = length(prior)
    active[j] && throw(ArgumentError("log_model_add_odds is defined for adding an inactive coordinate; coordinate $j is already active"))
    k < p || return -Inf
    return prior.log_omega[k + 2] - prior.log_omega[k + 1] + log(k + 1) - log(p - k)
end

"""
    _prepare_linear_gaussian_cache!(clock, state, can_stick, τ=0)

Populate the reusable linear-clock cache with beta values at elapsed time `τ`,
active/stickable beta masks, and active beta positions. Returns `(cache,
nactive)`.
"""
function _prepare_linear_gaussian_cache!(clock::LinearGaussianAggregateClock, state::StickyPDMPState, can_stick::BitVector, τ::Real=0.0)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    cache = clock.cache
    gaussian_slab!(provider, cache.mean, cache.cov, state.ξ.x)
    nactive = 0
    @inbounds for j in eachindex(indices)
        i = indices[j]
        active = state.free[i]
        cache.active_beta[j] = active
        cache.stickable_beta[j] = can_stick[i]
        cache.beta_values[j] = state.ξ.x[i] + τ * state.ξ.θ[i]
        cache.beta_velocity[j] = state.ξ.θ[i]
        if active
            nactive += 1
            cache.active_positions[nactive] = j
        end
    end
    return cache, nactive
end

"""
    _factor_linear_gaussian_active_block!(cache, nactive)

Factor the active covariance block in-place and overwrite `delta_A` and
`velocity_A` by the corresponding covariance solves.
"""
function _factor_linear_gaussian_active_block!(cache::LinearGaussianAggregateCache, nactive::Integer)
    iszero(nactive) && return cache
    @inbounds for aidx in 1:nactive
        ia = cache.active_positions[aidx]
        cache.delta_A[aidx] = cache.beta_values[ia] - cache.mean[ia]
        cache.velocity_A[aidx] = cache.beta_velocity[ia]
        for bidx in 1:nactive
            ib = cache.active_positions[bidx]
            cache.cov_AA[aidx, bidx] = cache.cov[ia, ib]
        end
    end
    _cholesky_factor_prefix!(cache.cov_AA, nactive)
    _cholesky_solve_with_factor_prefix!(cache.cov_AA, cache.delta_A, nactive)
    _cholesky_solve_with_factor_prefix!(cache.cov_AA, cache.velocity_A, nactive)
    return cache
end

"""
    _linear_gaussian_component_params!(cache, j, nactive)

Return `(a, b, s)` for inactive coordinate `j`, where the conditional boundary
mean along the segment is `a + b*t` and `s` is the conditional standard
deviation.
"""
function _linear_gaussian_component_params!(cache::LinearGaussianAggregateCache, j::Integer, nactive::Integer)
    if iszero(nactive)
        s2 = cache.cov[j, j]
        @assert s2 > 0
        return cache.mean[j], 0.0, sqrt(s2)
    end

    @inbounds for aidx in 1:nactive
        ia = cache.active_positions[aidx]
        value = cache.cov[j, ia]
        cache.cov_jA[aidx] = value
        cache.solved_cross[aidx] = value
    end
    _cholesky_solve_with_factor_prefix!(cache.cov_AA, cache.solved_cross, nactive)

    cross_delta = 0.0
    cross_velocity = 0.0
    cross_solve = 0.0
    @inbounds for aidx in 1:nactive
        c = cache.cov_jA[aidx]
        cross_delta += c * cache.delta_A[aidx]
        cross_velocity += c * cache.velocity_A[aidx]
        cross_solve += c * cache.solved_cross[aidx]
    end
    s2 = cache.cov[j, j] - cross_solve
    @assert s2 > 0
    return cache.mean[j] + cross_delta, cross_velocity, sqrt(s2)
end

function _linear_gaussian_has_component(clock::LinearGaussianAggregateClock, state::StickyPDMPState, can_stick::BitVector)
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    @inbounds for j in eachindex(cache.active_beta)
        cache.stickable_beta[j] && !cache.active_beta[j] &&
            _positive_logweight(_log_model_add_odds_with_count(clock.model_prior, cache.active_beta, j, nactive)) && return true
    end
    return false
end

function _linear_gaussian_available_hazard(clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    _factor_linear_gaussian_active_block!(cache, nactive)
    Cv = unstick_rate_constant(flow, 1)
    total = 0.0
    @inbounds for j in eachindex(cache.active_beta)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior, cache.active_beta, j, nactive)
            _positive_logweight(logρ) || continue
            logρ == Inf && return Inf
            a, b, s = _linear_gaussian_component_params!(cache, j, nactive)
            total += Cv * _linear_gaussian_component_total_hazard(a, b, s, logρ)
        end
    end
    return total
end

"""
    _linear_gaussian_component_hazard(a, b, s, log_weight, T)

Closed-form integral from `0` to `T` of a weighted Gaussian boundary density
with linear conditional mean `a + b*t`.
"""
function _linear_gaussian_component_hazard(a::Real, b::Real, s::Real, log_weight::Real, T::Real)
    T <= 0 && return 0.0
    log_weight == -Inf && return 0.0
    log_weight == Inf && return Inf
    w = exp(log_weight)
    if iszero(b)
        z = a / s
        return w * exp(-0.5 * abs2(z)) / sqrt(2π) / s * T
    end
    z0 = a / s
    zT = (a + b * T) / s
    return w / abs(b) * abs(Distributions.normcdf(zT) - Distributions.normcdf(z0))
end

"""
    _linear_gaussian_component_total_hazard(a, b, s, log_weight)

Closed-form total available future hazard for a single linear Gaussian boundary
component.
"""
function _linear_gaussian_component_total_hazard(a::Real, b::Real, s::Real, log_weight::Real)
    log_weight == -Inf && return 0.0
    log_weight == Inf && return Inf
    w = exp(log_weight)
    if iszero(b)
        return iszero(w) ? 0.0 : Inf
    end
    z0 = a / s
    if b > 0
        return w / b * (1 - Distributions.normcdf(z0))
    else
        return w / abs(b) * Distributions.normcdf(z0)
    end
end

function rate(clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    _factor_linear_gaussian_active_block!(cache, nactive)
    Cv = unstick_rate_constant(flow, 1)
    total = 0.0
    @inbounds for j in eachindex(cache.active_beta)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior, cache.active_beta, j, nactive)
            _positive_logweight(logρ) || continue
            logρ == Inf && return Inf
            a, b, s = _linear_gaussian_component_params!(cache, j, nactive)
            z = (a + b * τ) / s
            total += Cv * exp(logρ) * exp(-0.5 * abs2(z)) / sqrt(2π) / s
        end
    end
    return total
end

function cumulative_hazard(clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    t0 == t1 && return 0.0
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick)
    _factor_linear_gaussian_active_block!(cache, nactive)
    Cv = unstick_rate_constant(flow, 1)
    total = 0.0
    @inbounds for j in eachindex(cache.active_beta)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior, cache.active_beta, j, nactive)
            _positive_logweight(logρ) || continue
            logρ == Inf && return Inf
            a, b, s = _linear_gaussian_component_params!(cache, j, nactive)
            total += Cv * (
                _linear_gaussian_component_hazard(a, b, s, logρ, Float64(t1)) -
                _linear_gaussian_component_hazard(a, b, s, logρ, Float64(t0))
            )
        end
    end
    return max(0.0, total)
end

function sample_time(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    _linear_gaussian_has_component(clock, state, can_stick) || return Inf
    rate(clock, flow, state, 0.0, can_stick) == Inf && return 0.0
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = cumulative_hazard(clock, flow, state, 0.0, Float64(horizon), can_stick)
        H < threshold && return Inf
        f = τ -> cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
        return Roots.find_zero(f, (0.0, Float64(horizon)), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
    end

    total_available = _linear_gaussian_available_hazard(clock, flow, state, can_stick)
    total_available < threshold && return Inf

    lo = 0.0
    hi = 1.0
    H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
    while H_hi < threshold
        lo = hi
        next_hi = 2.0 * hi
        (!isfinite(next_hi) || next_hi <= hi) && return Inf
        hi = next_hi
        H_hi = cumulative_hazard(clock, flow, state, 0.0, hi, can_stick)
    end
    f = τ -> cumulative_hazard(clock, flow, state, 0.0, τ, can_stick) - threshold
    return Roots.find_zero(f, (lo, hi), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
end

function _linear_gaussian_label_weights!(clock::LinearGaussianAggregateClock, state::StickyPDMPState, can_stick::BitVector, τ::Real)
    cache, nactive = _prepare_linear_gaussian_cache!(clock, state, can_stick, τ)
    _factor_linear_gaussian_active_block!(cache, nactive)
    indices = beta_indices(clock.slab_provider)
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = _log_model_add_odds_with_count(clock.model_prior, cache.active_beta, j, nactive)
            if _positive_logweight(logρ)
                if logρ == Inf
                    logw = Inf
                else
                    a, _, s = _linear_gaussian_component_params!(cache, j, nactive)
                    z = a / s
                    logw = logρ - 0.5 * (log(2π) + 2log(s) + abs2(z))
                end
            else
                logw = -Inf
            end
            cache.log_weights[j] = logw
            max_logw = max(max_logw, logw)
        else
            cache.log_weights[j] = -Inf
        end
    end
    return cache, max_logw
end

_exchangeable_linear_mean(provider::ExchangeableGaussianSlab) = provider.mean
_exchangeable_linear_mean(::ZeroMeanExchangeableGaussianSlab) = 0.0

function _exchangeable_linear_params(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    active = clock.cache.active_beta
    stickable = clock.cache.stickable_beta
    μ = _exchangeable_linear_mean(provider)
    k = 0
    nU = 0
    sum_centered = 0.0
    sum_velocity = 0.0
    @inbounds for j in eachindex(indices)
        i = indices[j]
        is_active = state.free[i]
        active[j] = is_active
        stickable[j] = can_stick[i]
        if is_active
            k += 1
            sum_centered += state.ξ.x[i] - μ
            sum_velocity += state.ξ.θ[i]
        elseif can_stick[i]
            nU += 1
        end
    end
    denom = provider.u + k * provider.v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    c = provider.v / denom
    a = μ + c * sum_centered
    b = c * sum_velocity
    s2 = provider.u * (provider.u + (k + 1) * provider.v) / denom
    s2 > 0 || throw(ArgumentError("conditional variance must be positive, got $s2"))
    return active, stickable, k, nU, a, b, sqrt(s2)
end

function _exchangeable_log_weight_sum(prior::AbstractModelPrior, active::BitVector, stickable::BitVector, k::Integer)
    max_logw = -Inf
    @inbounds for j in eachindex(active)
        if stickable[j] && !active[j]
            max_logw = max(max_logw, _log_model_add_odds_with_count(prior, active, j, k))
        end
    end
    max_logw == -Inf && return -Inf
    max_logw == Inf && return Inf
    total = 0.0
    @inbounds for j in eachindex(active)
        if stickable[j] && !active[j]
            logw = _log_model_add_odds_with_count(prior, active, j, k)
            total += isfinite(logw) ? exp(logw - max_logw) : 0.0
        end
    end
    return max_logw + log(total)
end

function _exchangeable_log_weight_sum(prior::Union{ExchangeableModelSizePrior,BetaBernoulliModelPrior}, active::BitVector, stickable::BitVector, k::Integer)
    nU = count(j -> stickable[j] && !active[j], eachindex(active))
    iszero(nU) && return -Inf
    logρ = _log_model_add_odds_with_count(prior, active, findfirst(j -> stickable[j] && !active[j], eachindex(active)), k)
    logρ == -Inf && return -Inf
    return log(nU) + logρ
end

function _exchangeable_log_total_weight(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, active::BitVector, stickable::BitVector, k::Integer)
    log_sum = _exchangeable_log_weight_sum(clock.model_prior, active, stickable, k)
    log_sum == -Inf && return -Inf
    return log(unstick_rate_constant(flow, 1)) + log_sum
end

function rate(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    active, stickable, k, nU, a, b, s = _exchangeable_linear_params(clock, state, can_stick)
    iszero(nU) && return 0.0
    logW = _exchangeable_log_total_weight(clock, flow, active, stickable, k)
    logW == -Inf && return 0.0
    logW == Inf && return Inf
    z = (a + b * τ) / s
    return exp(logW) * exp(-0.5 * abs2(z)) / sqrt(2π) / s
end

function cumulative_hazard(clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    t0 == t1 && return 0.0
    active, stickable, k, nU, a, b, s = _exchangeable_linear_params(clock, state, can_stick)
    iszero(nU) && return 0.0
    logW = _exchangeable_log_total_weight(clock, flow, active, stickable, k)
    logW == -Inf && return 0.0
    logW == Inf && return Inf
    return max(0.0, _linear_gaussian_component_hazard(a, b, s, logW, Float64(t1)) -
                    _linear_gaussian_component_hazard(a, b, s, logW, Float64(t0)))
end

function _exchangeable_gaussian_line_total_hazard(a::Real, b::Real, s::Real, logW::Real)
    return _linear_gaussian_component_total_hazard(a, b, s, logW)
end

function _exchangeable_invert_gaussian_line(a::Real, b::Real, s::Real, W::Real, threshold::Real)
    if iszero(b)
        λ = W * exp(-0.5 * abs2(a / s)) / sqrt(2π) / s
        return iszero(λ) ? Inf : Float64(threshold) / λ
    end
    z0 = a / s
    Φ0 = Distributions.normcdf(z0)
    if b > 0
        target = Φ0 + b * Float64(threshold) / W
    else
        target = Φ0 - abs(b) * Float64(threshold) / W
    end
    0.0 < target < 1.0 || return Inf
    z = quantile(Normal(), target)
    τ = (s * z - a) / b
    return τ >= 0 ? τ : 0.0
end

function sample_time(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    active, stickable, k, nU, a, b, s = _exchangeable_linear_params(clock, state, can_stick)
    iszero(nU) && return Inf
    logW = _exchangeable_log_total_weight(clock, flow, active, stickable, k)
    logW == -Inf && return Inf
    logW == Inf && return 0.0
    W = exp(logW)
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = _linear_gaussian_component_hazard(a, b, s, logW, Float64(horizon))
        H < threshold && return Inf
        return _exchangeable_invert_gaussian_line(a, b, s, W, threshold)
    end
    total_available = _exchangeable_gaussian_line_total_hazard(a, b, s, logW)
    total_available < threshold && return Inf
    return _exchangeable_invert_gaussian_line(a, b, s, W, threshold)
end

function _sample_uniform_inactive_stickable(rng::Random.AbstractRNG, indices::AbstractVector{Int}, active::BitVector, stickable::BitVector, nU::Integer)
    nU > 0 || throw(ArgumentError("cannot sample an unstick label because there are no inactive stickable coordinates"))
    draw = rand(rng, 1:nU)
    seen = 0
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            seen += 1
            seen == draw && return indices[j]
        end
    end
    return indices[lastindex(indices)]
end

function _sample_static_model_prior_label(rng::Random.AbstractRNG, indices::AbstractVector{Int}, prior::AbstractModelPrior, active::BitVector, stickable::BitVector, k::Integer)
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            max_logw = max(max_logw, _log_model_add_odds_with_count(prior, active, j, k))
        end
    end
    max_logw == -Inf && throw(ArgumentError("cannot sample an unstick label because all inactive stickable rates are zero"))
    if max_logw == Inf
        logweights = fill(-Inf, length(indices))
        @inbounds for j in eachindex(indices)
            if stickable[j] && !active[j]
                logweights[j] = _log_model_add_odds_with_count(prior, active, j, k)
            end
        end
        return _sample_from_logweights(rng, indices, logweights)
    end
    total = 0.0
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            logw = _log_model_add_odds_with_count(prior, active, j, k)
            total += isfinite(logw) ? exp(logw - max_logw) : 0.0
        end
    end
    draw = rand(rng) * total
    last_candidate = 0
    @inbounds for j in eachindex(indices)
        if stickable[j] && !active[j]
            logw = _log_model_add_odds_with_count(prior, active, j, k)
            if isfinite(logw)
                last_candidate = j
                draw -= exp(logw - max_logw)
                draw <= 0 && return indices[j]
            end
        end
    end
    return indices[last_candidate]
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    active, stickable, k, nU, _, _, _ = _exchangeable_linear_params(clock, state, can_stick)
    indices = beta_indices(clock.slab_provider)
    return _sample_static_model_prior_label(rng, indices, clock.model_prior, active, stickable, k)
end

function sample_label(rng::Random.AbstractRNG,
        clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab,<:Union{ExchangeableModelSizePrior,BetaBernoulliModelPrior}},
        ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    active, stickable, _, nU, _, _, _ = _exchangeable_linear_params(clock, state, can_stick)
    return _sample_uniform_inactive_stickable(rng, beta_indices(clock.slab_provider), active, stickable, nU)
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:AbstractExchangeableGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    return sample_label(rng, clock, flow, state, can_stick)
end

function scalar_logscale_gaussian_line_segment(
    provider::GlobalLogscaleExchangeableGaussianSlab,
    model_prior::AbstractModelPrior,
    flow::Union{ZigZag,BouncyParticle},
    state::StickyPDMPState,
    can_stick::BitVector,
    horizon::Real,
)
    indices = beta_indices(provider)
    active = BitVector(undef, length(indices))
    stickable = BitVector(undef, length(indices))
    k = 0
    nU = 0
    sum_centered = 0.0
    sum_velocity = 0.0
    μ = provider.mean
    @inbounds for j in eachindex(indices)
        i = indices[j]
        is_active = state.free[i]
        active[j] = is_active
        stickable[j] = can_stick[i]
        if is_active
            k += 1
            sum_centered += state.ξ.x[i] - μ
            sum_velocity += state.ξ.θ[i]
        elseif can_stick[i]
            nU += 1
        end
    end
    iszero(nU) && throw(ArgumentError("cannot build a scalar logscale segment because there are no inactive stickable coordinates"))
    log_sum = _exchangeable_log_weight_sum(model_prior, active, stickable, k)
    log_sum == -Inf && throw(ArgumentError("cannot build a scalar logscale segment because all model-prior odds are zero"))
    logW = log(unstick_rate_constant(flow, 1)) + log_sum
    denom = provider.u + k * provider.v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    c = provider.v / denom
    a0 = μ + c * sum_centered
    b = c * sum_velocity
    s2 = provider.u * (provider.u + (k + 1) * provider.v) / denom
    s2 > 0 || throw(ArgumentError("conditional variance must be positive, got $s2"))
    s0 = sqrt(s2)
    ell0 = provider.logscale_offset + state.ξ.x[provider.logscale_index]
    r = state.ξ.θ[provider.logscale_index]
    return ScalarLogscaleGaussianLineSegment(Float64(a0), Float64(b), Float64(s0), Float64(ell0), Float64(r), Float64(logW), Float64(horizon))
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock, ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, _ = _linear_gaussian_label_weights!(clock, state, can_stick, 0.0)
    return _sample_from_logweights(rng, beta_indices(clock.slab_provider), cache.log_weights)
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    cache, _ = _linear_gaussian_label_weights!(clock, state, can_stick, τ)
    return _sample_from_logweights(rng, beta_indices(clock.slab_provider), cache.log_weights)
end

function _independent_fixed_logweights!(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    cache = clock.cache
    @inbounds for j in eachindex(indices)
        i = indices[j]
        active = state.free[i]
        cache.active_beta[j] = active
        cache.stickable_beta[j] = can_stick[i]
    end
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        if cache.stickable_beta[j] && !cache.active_beta[j]
            cache.log_weights[j] = log_model_add_odds(clock.model_prior, cache.active_beta, j) + provider.log_q_zero[j]
            max_logw = max(max_logw, cache.log_weights[j])
        else
            cache.log_weights[j] = -Inf
        end
    end
    return cache, max_logw
end

function _independent_fixed_rate(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, max_logw = _independent_fixed_logweights!(clock, state, can_stick)
    max_logw == -Inf && return 0.0
    max_logw == Inf && return Inf
    return unstick_rate_constant(flow, 1) * exp(LogExpFunctions.logsumexp(cache.log_weights))
end

function rate(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    return _independent_fixed_rate(clock, flow, state, can_stick)
end

function cumulative_hazard(clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    return (Float64(t1) - Float64(t0)) * _independent_fixed_rate(clock, flow, state, can_stick)
end

function sample_time(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    λ = _independent_fixed_rate(clock, flow, state, can_stick)
    iszero(λ) && return Inf
    λ == Inf && return 0.0
    τ = rand(rng, Exponential()) / λ
    return τ <= horizon ? τ : Inf
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, ::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    cache, _ = _independent_fixed_logweights!(clock, state, can_stick)
    return _sample_from_logweights(rng, beta_indices(clock.slab_provider), cache.log_weights)
end

function sample_label(rng::Random.AbstractRNG, clock::LinearGaussianAggregateClock{<:IndependentZeroMeanGaussianSlab}, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    return sample_label(rng, clock, flow, state, can_stick)
end

function _prepare_exponential_sum_cache!(clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    provider = clock.slab_provider
    indices = beta_indices(provider)
    cache = clock.cache
    Cv = unstick_rate_constant(flow, 1)
    logCv_phi0 = log(Cv) - 0.5 * log(2π)
    @inbounds for j in eachindex(indices)
        i = indices[j]
        cache.active_beta[j] = state.free[i]
        cache.stickable_beta[j] = can_stick[i]
        cache.slopes[j] = state.ξ.θ[provider.logscale_indices[j]]
        cache.logc[j] = -Inf
    end
    @inbounds for j in eachindex(indices)
        i = indices[j]
        if cache.stickable_beta[j] && !cache.active_beta[j]
            logρ = log_model_add_odds(clock.model_prior, cache.active_beta, j)
            log_s0 = provider.log_base_scales[j] + state.ξ.x[provider.logscale_indices[j]]
            cache.logc[j] = logCv_phi0 + logρ - log_s0
        end
    end
    return cache
end

function _exponential_component_hazard_from_logc(logc::Real, r::Real, T::Real)
    T <= 0 && return 0.0
    logc == -Inf && return 0.0
    logc == Inf && return Inf
    c = exp(logc)
    iszero(r) && return c * T
    return -c * expm1(-r * T) / r
end

function _exponential_sum_hazard_from_cache(cache::ExponentialSumAggregateCache, T::Real)
    total = 0.0
    @inbounds for j in eachindex(cache.logc)
        total += _exponential_component_hazard_from_logc(cache.logc[j], cache.slopes[j], T)
    end
    return max(0.0, total)
end

function rate(clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    @assert τ >= 0
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    max_logλ = -Inf
    @inbounds for j in eachindex(cache.logc)
        _positive_logweight(cache.logc[j]) && (max_logλ = max(max_logλ, cache.logc[j] - cache.slopes[j] * τ))
    end
    max_logλ == -Inf && return 0.0
    max_logλ == Inf && return Inf
    total = 0.0
    @inbounds for j in eachindex(cache.logc)
        if isfinite(cache.logc[j])
            total += exp(cache.logc[j] - cache.slopes[j] * τ - max_logλ)
        end
    end
    return exp(max_logλ) * total
end

function cumulative_hazard(clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, t0::Real, t1::Real, can_stick::BitVector)
    @assert 0 <= t0 <= t1
    t0 == t1 && return 0.0
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    any(==(Inf), cache.logc) && return Inf
    return max(0.0, _exponential_sum_hazard_from_cache(cache, Float64(t1)) -
                    _exponential_sum_hazard_from_cache(cache, Float64(t0)))
end

function _exponential_sum_available_hazard(cache::ExponentialSumAggregateCache)
    total = 0.0
    @inbounds for j in eachindex(cache.logc)
        if _positive_logweight(cache.logc[j])
            c = exp(cache.logc[j])
            r = cache.slopes[j]
            if r <= 0
                return Inf
            else
                total += c / r
            end
        end
    end
    return total
end

function sample_time(rng::Random.AbstractRNG, clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, horizon::Real, can_stick::BitVector)
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    any(_positive_logweight, cache.logc) || return Inf
    any(==(Inf), cache.logc) && return 0.0
    threshold = rand(rng, Exponential())
    if isfinite(horizon)
        H = _exponential_sum_hazard_from_cache(cache, Float64(horizon))
        H < threshold && return Inf
        f = τ -> _exponential_sum_hazard_from_cache(cache, τ) - threshold
        return Roots.find_zero(f, (0.0, Float64(horizon)), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
    end

    available = _exponential_sum_available_hazard(cache)
    available < threshold && return Inf
    lo = 0.0
    hi = 1.0
    H_hi = _exponential_sum_hazard_from_cache(cache, hi)
    iterations = 0
    while H_hi < threshold && iterations < 80
        lo = hi
        hi *= 2.0
        H_hi = _exponential_sum_hazard_from_cache(cache, hi)
        iterations += 1
    end
    H_hi < threshold && return Inf
    f = τ -> _exponential_sum_hazard_from_cache(cache, τ) - threshold
    return Roots.find_zero(f, (lo, hi), Roots.Bisection(); atol=clock.atol, rtol=clock.rtol)
end

function sample_label(rng::Random.AbstractRNG, clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, can_stick::BitVector)
    return sample_label(rng, clock, flow, state, 0.0, can_stick)
end

function sample_label(rng::Random.AbstractRNG, clock::ExponentialSumAggregateClock, flow::Union{ZigZag,BouncyParticle}, state::StickyPDMPState, τ::Real, can_stick::BitVector)
    cache = _prepare_exponential_sum_cache!(clock, flow, state, can_stick)
    indices = beta_indices(clock.slab_provider)
    max_logw = -Inf
    @inbounds for j in eachindex(indices)
        cache.log_weights[j] = _positive_logweight(cache.logc[j]) ? cache.logc[j] - cache.slopes[j] * τ : -Inf
        max_logw = max(max_logw, cache.log_weights[j])
    end
    return _sample_from_logweights(rng, indices, cache.log_weights)
end


"""
    unfreeze_time(alg::StickyLoopState, state::StickyPDMPState, i::Integer)

Simulate the time a stuck/ frozen particle takes to unfreeze/ unstick
"""
function unfreeze_time(rng::Random.AbstractRNG, alg::StickyLoopState, state::StickyPDMPState, i::Integer)
    validate_state(state, nothing, "in unfreeze_time")
    κ = get_κ(alg, i, state.ξ.x, state.free, state.ξ.θ)

    if κ isa Distribution

        retval = rand(rng, κ)
        if isnegative(retval)# || isinf(retval)
            @show κ, i, state.ξ.x, state.free, state.ξ.θ
            throw(ArgumentError("κ must be non-negative and finite!"))
        end
        # @show κ, i, state.ξ.x, state.free
        return retval
    else
        θf = state.old_velocity[i]
        if isnegative(κ)
            @show κ, i, state.ξ.x, state.free
            throw(ArgumentError("κ must be non-negative!"))
        end

        # return -log(rand()) / (κ * abs(θf)) # old approach
        return rand(rng, Exponential(inv(κ * abs(θf))))
    end
end
