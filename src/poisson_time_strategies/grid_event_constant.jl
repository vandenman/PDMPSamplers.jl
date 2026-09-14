const GridEvent = Tuple{Float64,Symbol,GradientMeta}

function _constant_bound_event_time(
    rng::Random.AbstractRNG, model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
    alg::GridAdaptiveState, state::AbstractPDMPState, cache,
    stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol=:horizon_hit,
    probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe(),
)::GridEvent
    λ_bound = alg.constant_bound_rate[]
    alg.has_cached_rate_derivative[] = false
    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))
    τ_refresh = ispositive(λ_refresh) ? Random.randexp(rng) / λ_refresh : Inf
    default_return = GradientMeta(alg.empty_∇ϕx)

    state_ = alg.state_cache
    copyto!(state_, state)
    t_max, horizon_event = _effective_grid_horizon(model.grad, alg.t_max[], τ_refresh, max_horizon, max_horizon_event)

    cumulative_exp = 0.0
    for _ in 1:alg.safety_limit
        cumulative_exp += Random.randexp(rng)
        τ_proposal = cumulative_exp / λ_bound

        if τ_proposal >= t_max
            _inc_counter_grid_horizon_hits(stats)
            alg.has_cached_gradient[] = false
            alg.has_cached_rate_derivative[] = false
            return t_max, horizon_event, default_return
        end

        if τ_refresh < τ_proposal
            alg.has_cached_gradient[] = false
            alg.has_cached_rate_derivative[] = false
            return τ_refresh, :refresh, default_return
        end

        _inc_counter_constant_bound_attempts(stats)
        copyto!(state_, state)
        move_forward_time!(state_, τ_proposal, flow)
        _inc_counter_grid_acceptance_gradient_calls(stats)
        ∇ϕx = _compute_grid_gradient_or_throw!(
            state_, state, flow, model, cache, 0.0, τ_proposal, probe_failure_handler)
        l_actual = λ(state_, ∇ϕx, flow)
        _inc_counter_grid_acceptance_tests(stats)

        if l_actual > λ_bound
            _inc_counter_constant_bound_violations(stats)
            _inc_counter_grid_bound_violations(stats)
            if alg.bound_violation === :throw
                signed_actual = flow isa BouncyParticle ? dot(∇ϕx, state_.ξ.θ) : NaN
                throw(ErrorException(_grid_bound_violation_message(
                    alg, stats, state_, flow, τ_proposal, signed_actual, l_actual,
                    λ_bound, cumulative_exp, λ_refresh, false)))
            elseif alg.bound_violation === :shrink
                alg.constant_bound_rate[] = NaN
                _shrink_grid_after_bound_violation!(alg, stats)
                return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
                    max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
            end
            alg.constant_bound_rate[] = NaN
            return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
                max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
        end

        if rand(rng) * λ_bound <= l_actual
            alg.max_observed_rate[] = max(alg.max_observed_rate[], l_actual)
            _inc_counter_constant_bound_accepts(stats)
            return τ_proposal, :reflect, GradientMeta(∇ϕx)
        end
        _inc_counter_constant_bound_rejections(stats)
    end

    _inc_counter_constant_bound_safety_fallbacks(stats)
    alg.constant_bound_rate[] = NaN
    return _next_event_time_with_probe(rng, model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler)
end

function _constant_bound_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::GridAdaptiveState,
    state::AbstractPDMPState, cache, stats::AbstractStatisticCounter, max_horizon::Float64, include_refresh::Bool,
    max_horizon_event::Symbol=:horizon_hit, probe_failure_handler::GridBoundaryProbe=NoGridBoundaryProbe())::GridEvent
    return _constant_bound_event_time(
        Random.default_rng(), model, flow, alg, state, cache, stats,
        max_horizon, include_refresh, max_horizon_event, probe_failure_handler,
    )
end
