pdmp_sample(d::Integer, args...; seed::Union{Integer,Nothing}=nothing, kwargs...) = pdmp_sample(randn(_make_rng(seed), d), args...; seed, kwargs...)
pdmp_sample(x₀::AbstractVector, θ₀::AbstractVector, flow::ContinuousDynamics, args...; kwargs...) = pdmp_sample(SkeletonPoint(x₀, θ₀), flow, args...; kwargs...)
pdmp_sample(x₀::AbstractVector, flow::ContinuousDynamics, args...; seed::Union{Integer,Nothing}=nothing, kwargs...) = pdmp_sample(SkeletonPoint(x₀, initialize_velocity(_make_rng(seed), flow, length(x₀))), flow, args...; seed, kwargs...)

_make_rng(::Nothing) = Random.default_rng()
_make_rng(seed::Integer) = Random.Xoshiro(seed)

"""
    pdmp_sample(ξ₀, flow, model, alg, t₀=0.0, T=10_000, t_warmup=0.0;
                stop=nothing, warmup_stop=nothing, n_chains=1, threaded=false,
                progress=true, adapter=NoAdaptation())

Run PDMP sampling with optional stopping criteria for warmup and main sampling phases.

If `warmup_stop === nothing`, warmup uses `FixedTimeCriterion(t₀ + t_warmup)`.
If `stop === nothing`, sampling uses `FixedTimeCriterion(T)`.
Provided criteria take precedence over time arguments.

Warmup runs first with adaptation enabled and writes to the warmup trace.
Main sampling runs second with adaptation disabled and writes to the main trace.
Criteria are initialized per phase, so mutable criteria (e.g. `WallTimeCriterion`, ESS counters)
are phase-local. An exception is `TotalWallTimeCriterion`, whose timer starts once globally
and is not re-initialized per phase.
"""

function pdmp_sample(
    ξ₀::SkeletonPoint, flow::ContinuousDynamics, model::PDMPModel,
    alg::PoissonTimeStrategy,
    t₀::Real=0.0, T::Real=10_000, t_warmup::Real=0.0;
    stop::Union{StoppingCriterion,Nothing}=nothing,
    warmup_stop::Union{StoppingCriterion,Nothing}=nothing,
    n_chains::Int=1, threaded::Bool=false,
    progress::Bool=true,
    adapter::AbstractAdapter=NoAdaptation(),
    seed::Union{Integer,Nothing}=nothing
)
    n_chains >= 1 || throw(ArgumentError("n_chains must be >= 1, got $n_chains"))

    if n_chains == 1
        rng = _make_rng(seed)
        trace, stats = _pdmp_sample_single(rng, ξ₀, flow, model, alg, t₀, T, t_warmup; progress, adapter, stop, warmup_stop)
        return PDMPChains([trace], [stats])
    end

    if threaded
        tasks = map(1:n_chains) do chain_i
            Threads.@spawn begin
                rng_i = _make_chain_rng(seed, chain_i)
                flow_i = _copy_flow(flow)
                model_i = _copy_model(model)
                stop_i = _maybe_copy_criterion(stop)
                warmup_stop_i = _maybe_copy_criterion(warmup_stop)
                _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, model_i, alg, t₀, T, t_warmup; progress=false, adapter, stop=stop_i, warmup_stop=warmup_stop_i)
            end
        end
        results = fetch.(tasks)
    else
        results = map(1:n_chains) do chain_i
            rng_i = _make_chain_rng(seed, chain_i)
            flow_i = _copy_flow(flow)
            model_i = _copy_model(model)
            stop_i = _maybe_copy_criterion(stop)
            warmup_stop_i = _maybe_copy_criterion(warmup_stop)
            _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, model_i, alg, t₀, T, t_warmup; progress=false, adapter, stop=stop_i, warmup_stop=warmup_stop_i)
        end
    end

    traces = [r[1] for r in results]
    all_stats = [r[2] for r in results]
    return PDMPChains(traces, all_stats)
end

_make_chain_rng(::Nothing, chain_i::Int) = Random.Xoshiro()
_make_chain_rng(seed::Integer, chain_i::Int) = Random.Xoshiro(seed + chain_i - 1)

function pdmp_sample(
    ξ₀::SkeletonPoint, flow::ContinuousDynamics, models::AbstractVector{<:PDMPModel},
    alg::PoissonTimeStrategy,
    t₀::Real=0.0, T::Real=10_000, t_warmup::Real=0.0;
    stop::Union{StoppingCriterion,Nothing}=nothing,
    warmup_stop::Union{StoppingCriterion,Nothing}=nothing,
    threaded::Bool=false,
    progress::Bool=true,
    adapter::AbstractAdapter=NoAdaptation(),
    seed::Union{Integer,Nothing}=nothing
)
    n_chains = length(models)
    n_chains >= 1 || throw(ArgumentError("models must be non-empty"))

    if n_chains == 1
        rng = _make_rng(seed)
        trace, stats = _pdmp_sample_single(rng, ξ₀, flow, models[1], alg, t₀, T, t_warmup; progress, adapter, stop, warmup_stop)
        return PDMPChains([trace], [stats])
    end

    if threaded
        tasks = map(1:n_chains) do i
            Threads.@spawn begin
                rng_i        = _make_chain_rng(seed, i)
                flow_i        = _copy_flow(flow)
                stop_i        = _maybe_copy_criterion(stop)
                warmup_stop_i = _maybe_copy_criterion(warmup_stop)
                _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, models[i], alg, t₀, T, t_warmup;
                    progress=false, adapter, stop=stop_i, warmup_stop=warmup_stop_i)
            end
        end
        results = fetch.(tasks)
    else
        results = map(1:n_chains) do i
            rng_i        = _make_chain_rng(seed, i)
            flow_i        = _copy_flow(flow)
            stop_i        = _maybe_copy_criterion(stop)
            warmup_stop_i = _maybe_copy_criterion(warmup_stop)
            _pdmp_sample_single(rng_i, copy(ξ₀), flow_i, models[i], alg, t₀, T, t_warmup;
                progress=false, adapter, stop=stop_i, warmup_stop=warmup_stop_i)
        end
    end

    traces    = [r[1] for r in results]
    all_stats = [r[2] for r in results]
    return PDMPChains(traces, all_stats)
end

_maybe_copy_criterion(::Nothing) = nothing
_maybe_copy_criterion(c::StoppingCriterion) = copy(c)

_copy_flow(flow::ContinuousDynamics) = flow
_copy_flow(flow::MutableBoomerang) = copy(flow)
_copy_flow(pd::PreconditionedDynamics) = PreconditionedDynamics(deepcopy(pd.metric), _copy_flow(pd.dynamics))

function _copy_model(model::PDMPModel)
    grad_new = copy(model.grad)
    hvp_new = model.hvp === nothing ? nothing : _copy_callable(model.hvp)
    vhv_new = model.vhv === nothing ? nothing : _copy_callable(model.vhv)
    joint_new = model.joint === nothing ? nothing : _copy_callable(model.joint)
    return PDMPModel(model.d, grad_new, hvp_new, vhv_new, false, false, joint_new)
end

# the inner workhorse that runs a single chain
function _step!(
    rng::Random.AbstractRNG,
    state::AbstractPDMPState,
    model_::PDMPModel,
    flow::FL,
    alg_::PoissonTimeStrategy,
    cache::NamedTuple,
    stats::StatisticCounter,
    trace_manager::TraceManager;
    phase::Symbol
) where {FL<:ContinuousDynamics}
    τ, event_type, meta = next_event_time(rng, model_, flow, alg_, state, cache, stats)

    @assert ispositive(τ) "Proposed event time τ ($τ) is non-positive. Sampler is stuck!"

    needs_saving, saving_args = handle_event!(rng, τ, model_.grad, flow, alg_, state, cache, event_type, meta, stats)
    needs_saving && record_event!(trace_manager, state, flow, saving_args; phase)

    return event_type
end

function _update_progress!(progress::Bool, prg, tstop::Base.RefValue{Float64}, T::Float64, progress_stops::Int, state::AbstractPDMPState)
    if progress && state.t[] > tstop[]
        tstop[] += T / progress_stops
        ProgressMeter.next!(prg)
    end
    return nothing
end

function _run_phase!(
    rng::Random.AbstractRNG,
    criterion::StoppingCriterion,
    state::AbstractPDMPState,
    model_::PDMPModel,
    flow::FL,
    alg_::PoissonTimeStrategy,
    cache::NamedTuple,
    trace_manager::TraceManager,
    stats::StatisticCounter,
    health::HealthMonitor;
    phase::Symbol,
    adapter::AbstractAdapter=NoAdaptation(),
    progress::Bool=false,
    prg=nothing,
    tstop::Base.RefValue{Float64}=Ref(Inf),
    T::Float64=0.0,
    progress_stops::Int=300
) where {FL<:ContinuousDynamics}
    initialize!(criterion, state, trace_manager, stats)

    _maybe_simplify_counter = 0

    while true
        if is_satisfied(criterion, state, trace_manager, stats)
            stats.stop_reason = stop_reason(criterion, state, trace_manager, stats)
            return nothing
        end

        event_type = _step!(rng, state, model_, flow, alg_, cache, stats, trace_manager; phase)
        update!(criterion, state, trace_manager, stats, event_type)

        adapt!(rng, adapter, state, flow, model_.grad, trace_manager; phase, stats)
        if model_.grad isa SubsampledGradient
            _invalidate_cached_gradient!(alg_)
        end
        if did_dynamics_adapt(adapter)
            _reset_inner_grid!(alg_)
            if alg_ isa StickyLoopState
                update_all_stick_times!(rng, alg_, state, flow)
            end
        end

        if phase === :main
            _maybe_simplify_counter += 1
            if _maybe_simplify_counter >= 100
                _maybe_activate_constant_bound!(alg_, stats)
                _maybe_simplify_counter = 0
            end
        end

        check_health!(health, stats)
        _update_progress!(progress, prg, tstop, T, progress_stops, state)
    end
end

function _pdmp_sample_single(
    rng::Random.AbstractRNG,
    ξ₀::SkeletonPoint, flow::FL, model::PDMPModel,
    alg::PoissonTimeStrategy,
    t₀::Real=0.0, T::Real=10_000, t_warmup::Real=0.0;
    progress::Bool=true,
    adapter::AbstractAdapter=NoAdaptation(),
    stop::Union{StoppingCriterion,Nothing}=nothing,
    warmup_stop::Union{StoppingCriterion,Nothing}=nothing
) where {FL<:ContinuousDynamics}

    # TODO: it's possible to sample to have t_warmup < T...
    # we should always sample T + t_warmup!

    # TODO: this is a better design!
    # stats = StatisticCounter()
    # p = PDMPSampler(flow, with_stats(grad), alg, t₀, ξ₀)
    # (state, grad_, alg_, cache) = iterate(p)

    # this part should be a new function, so that the top part only does setup/ resuming is easy
    # s = with_progress(with_stopping_criterion(p, <conditions>))
    # trace = PDMPTrace()

    # burnin
    # Iterators.advance(trace, 0)

    # aa = 1:2:11
    # it = Iterators.drop(aa, 3)
    # first(it)

    if isnothing(warmup_stop)
        t_warmup < (T - t₀) || throw(ArgumentError("t_warmup ($t_warmup) is larger than T - t₀ ($T - $t₀ = $(T - t₀)), which implies storing nothing at all. This is probably an error."))
    end

    t_start = time_ns()

    state, model_, alg_, cache, stats = initialize_state(rng, flow, model, alg, t₀, ξ₀)

    validate_state(state, flow, "at initialization")

    t_warmup_abs = t₀ + t_warmup
    trace_manager = TraceManager(state, flow, alg, t_warmup_abs)
    health = HealthMonitor()
    adapter = adapter isa NoAdaptation ? default_adapter(flow, model_.grad, t_warmup ÷ 10, t_warmup, t₀) : adapter

    warmup_criterion = isnothing(warmup_stop) ? FixedTimeCriterion(t_warmup_abs) : warmup_stop
    stop_criterion = isnothing(stop) ? FixedTimeCriterion(T) : stop

    # progressmanager = ProgressManager(progress, T, t₀, t_warmup, progress_stops)
    progress_stops = 300
    if progress
        prg = ProgressMeter.Progress(progress_stops, dt=1)
        tstop = Ref(T / progress_stops)
    else
        prg = nothing
        tstop = Ref(Inf)
    end

    _run_phase!(
        rng,
        warmup_criterion,
        state,
        model_,
        flow,
        alg_,
        cache,
        trace_manager,
        stats,
        health;
        phase=:warmup,
        adapter,
        progress,
        prg,
        tstop,
        T=Float64(T),
        progress_stops
    )

    _maybe_activate_constant_bound!(alg_, stats)

    _run_phase!(
        rng,
        stop_criterion,
        state,
        model_,
        flow,
        alg_,
        cache,
        trace_manager,
        stats,
        health;
        phase=:main,
        adapter,
        progress,
        prg,
        tstop,
        T=Float64(T),
        progress_stops
    )

    stats.elapsed_time = (time_ns() - t_start) / 1e9

    return compact(get_main_trace(trace_manager)), stats

end

function initialize_state(rng::Random.AbstractRNG, flow::ContinuousDynamics, model::PDMPModel, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint)
    ξ = copy(ξ₀)
    t = t₀
    stats = StatisticCounter()
    state = alg isa Sticky ? StickyPDMPState(t, ξ) : PDMPState(t, ξ)
    cache = add_gradient_to_cache(initialize_cache(rng, flow, model.grad, alg, t, ξ), ξ)
    alg_ = _to_internal(alg, rng, flow, model, state, cache, stats)
    model_ = with_stats(model, stats)
    model_.grad isa SubsampledGradient && model_.grad.resample_indices!(model_.grad.nsub)
    return state, model_, alg_, cache, stats
end
initialize_state(flow::ContinuousDynamics, model::PDMPModel, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint) = initialize_state(Random.default_rng(), flow, model, alg, t₀, ξ₀)

function add_gradient_to_cache(cache::NamedTuple, ξ::SkeletonPoint)
    if haskey(cache, :∇ϕx)
        if !(cache.∇ϕx isa typeof(ξ.x) && length(cache.∇ϕx) == length(ξ.x))
            throw(ArgumentError("cache.∇ϕx was given manually, but must be of the same type as ξ.x"))
        end
    else
        ∇ϕx = similar(ξ.x)
        cache = merge(cache, (; ∇ϕx))
    end
    return cache
end

# TODO: these need to move to the respective dynamics files?
function initialize_cache(::Random.AbstractRNG, ::ContinuousDynamics, ::GradientStrategy, ::PoissonTimeStrategy, ::Real, ::SkeletonPoint)
    (;)
end
function initialize_cache(rng::Random.AbstractRNG, flow::PreconditionedDynamics, grad::GlobalGradientStrategy, alg::PoissonTimeStrategy, t::Real, ξ::SkeletonPoint)
    return initialize_cache(rng, flow.dynamics, grad, alg, t, ξ)
end
function initialize_cache(::Random.AbstractRNG, flow::PreconditionedDynamics{DensePreconditioner}, grad::GlobalGradientStrategy, alg::PoissonTimeStrategy, t::Real, ξ::SkeletonPoint)
    return (; z=similar(ξ.x))
end
function initialize_cache(::Random.AbstractRNG, ::BouncyParticle, ::GlobalGradientStrategy, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z=similar(ξ.x))
end
function initialize_cache(::Random.AbstractRNG, ::AnyBoomerang, ::GlobalGradientStrategy, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z=similar(ξ.x))
end

initialize_cache(flow::ContinuousDynamics, grad::GradientStrategy, alg::PoissonTimeStrategy, t::Real, ξ::SkeletonPoint) = initialize_cache(Random.default_rng(), flow, grad, alg, t, ξ)

function initialize_cache(rng::Random.AbstractRNG, flow::ZigZag, ::CoordinateWiseGradient, thinningstrategy::ThinningStrategy, t::Real, ξ::SkeletonPoint)

    # PriorityQueue stores: Coordinate Index => Absolute Event Time
    # could also be a MutableBinaryMinHeap{Float64, DataStructures.FasterForward}, which might be more efficient.
    # see https://juliacollections.github.io/DataStructures.jl/stable/heaps/
    # not entirely sure to what extent this matters
    pq = PriorityQueue{Int,Float64}()

    # Initialize all d clocks with their proposed event times
    for i in eachindex(ξ.x)
        abc_i = ab_i(i, ξ, thinningstrategy, flow, nothing)
        t_event = t + poisson_time(abc_i[1], abc_i[2], rand(rng))
        # enqueue!(pq, i => t_event)
        push!(pq, i => t_event)
    end

    return (; pq)
end

function handle_event!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::GlobalGradientStrategy, flow::ContinuousDynamics, alg::PoissonTimeStrategy, state::AbstractPDMPState, cache, event_type::Symbol, meta, stats::StatisticCounter)

    # Always move forward in time
    move_forward_time!(state, τ, flow)
    validate_state(state, flow, "after moving forward in time")

    # assume a ghost event
    needs_saving = false
    saving_args = nothing
    stats.last_rejected = false
    # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)


    if event_type == :reflect

        stats.reflections_events += 1

        if alg isa ExactStrategy

            stats.reflections_accepted += 1

            if flow isa ZigZag
                i = meta.i
                reflect!(state.ξ, zero(eltype(cache.∇ϕx)), i, flow)
            else
                ∇ϕx = compute_gradient!(state, gradient_strategy, flow, cache)
                saving_args = reflect!(rng, state, ∇ϕx, flow, cache)
            end
            needs_saving = true
            (alg isa StickyLoopState && state isa StickyPDMPState) && update_all_stick_times!(rng, alg, state, flow)
            # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
        else

            # Reuse gradient from meta when available (e.g., GridThinningStrategy already computed it)
            if meta isa GradientMeta && length(meta.∇ϕx) == length(state.ξ.x)
                ∇ϕx = meta.∇ϕx
            else
                ∇ϕx = compute_gradient_for_reflection!(state, gradient_strategy, flow, cache)
            end

            # meta == abc for ThinningStrategy
            if accept_reflection_event(rng, alg, state.ξ, ∇ϕx, flow, τ, cache, meta)
                stats.reflections_accepted += 1
                # @show "before reflect", state, ∇ϕx, flow, cache
                saving_args = reflect!(rng, state, ∇ϕx, flow, cache)
                # @show "after reflect", state
                needs_saving = true

                (alg isa StickyLoopState && state isa StickyPDMPState) && update_all_stick_times!(rng, alg, state, flow)
                # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
            else
                # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
                stats.last_rejected = true
            end
        end
        # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
        # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)

    elseif event_type == :refresh

        # meta == nothing for ThinningStrategy
        # @show "before refresh", state
        refresh_velocity!(rng, state, flow)
        # @show "after refresh", state
        # saving_args = nothing # could be informed by the refresh function?
        needs_saving = true
        stats.refreshment_events += 1

        # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
        # should be only the unfreeze times?
        # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
        (alg isa StickyLoopState && state isa StickyPDMPState) && update_all_stick_times!(rng, alg, state, flow)

    elseif event_type == :sticky

        stats.sticky_events += 1
        i = meta.i
        stick_or_unstick!(rng, state::StickyPDMPState, flow, alg, i)
        validate_state(state, flow, "after stick_or_unstick!")
        needs_saving = true
        if isfactorized(flow)
            saving_args = i
        end

    elseif event_type == :horizon_hit

        # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
        # should be only the unfreeze times?
        # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
        (alg isa StickyLoopState && state isa StickyPDMPState) && update_all_stick_times!(rng, alg, state, flow)
        # for now only when horizon is reached.
        stats.last_rejected = true
    end

    # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)

    if alg isa StickyLoopState
        @assert all(>=(state.t[]), alg.sticky_times) "some sticky_times are negative!"
    end

    return needs_saving, saving_args
end

function handle_event!(rng::Random.AbstractRNG, τ::Real, gradient_strategy::CoordinateWiseGradient, flow::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, event_type, meta, stats::StatisticCounter)

    stats.last_rejected = false
    # rename for clarity
    pq = cache.pq
    i₀ = meta.i  # winning coordinate
    ξ = state.ξ
    t = state.t[]

    # TODO: not sure if this is correct, compare to src/sfact.jl#L119
    # there they use b[i]
    abc_i₀_old = ab_i(i₀, ξ, alg, flow, cache)
    move_forward_time!(state, τ, flow)

    ∇ϕ_i₀ = compute_gradient!(gradient_strategy, ξ.x, i₀, cache) # partial derivative
    l_i₀ = λ_i(i₀, ξ, ∇ϕ_i₀, flow)

    # Compute the bound rate for ONLY the winning coordinate
    # abc_i₀_old_2 = ab_i(i₀, SkeletonPoint(x_old, ξ.θ), alg, flow)
    # @assert abc_i₀_old_2 == abc_i₀_old

    l_bound_i₀ = pos(abc_i₀_old[1] + abc_i₀_old[2] * τ)

    stats.reflections_events += 1

    needs_saving = false
    saving_args = nothing

    if rand(rng) * l_bound_i₀ <= l_i₀

        if l_i₀ >= l_bound_i₀
            # TODO: somehow τ is negative!?!?
            # @show t, τ#, abc_i₀_old, l_bound_i₀, l_i₀
            error("Tuning parameter `c` too small: l_i₀=$l_i₀, l_bound_i₀=$l_bound_i₀")
        end

        saving_args = reflect!(state, ∇ϕ_i₀, i₀, flow)
        needs_saving = true
        stats.reflections_accepted += 1

        # TODO: this should not be necessary and defeats the purpose!
        # CASCADE UPDATE: This is the cost of a non-factorized model.
        # A change in θ[i₀] can affect the bounds of all other coordinates.
        # We must re-calculate and re-enqueue all d clocks.
        empty!(pq)
        for i in eachindex(ξ.x)
            # The `ab_i` calculation depends on the NEW θ for coordinate `i`,
            # but on the OLD θ for the dot product term involving `i₀` if `i!=i₀`.
            # For simplicity and correctness in the fully non-factorized case,
            # we recalculate all bounds with the fully updated state (t, x, θ).
            abc_i_new = ab_i(i, ξ, alg, flow, nothing)
            t_event = state.t[] + poisson_time(abc_i_new[1], abc_i_new[2], rand(rng))
            # enqueue!(pq, i => t_event)
            push!(pq, i => t_event)
        end
    else
        abc_i₀_new = ab_i(i₀, ξ, alg, flow, nothing)
        t_event = state.t[] + poisson_time(abc_i₀_new[1], abc_i₀_new[2], rand(rng))
        # enqueue!(pq, i₀ => t_event)
        push!(pq, i₀ => t_event)
        stats.last_rejected = true
    end

    return needs_saving, saving_args
end

