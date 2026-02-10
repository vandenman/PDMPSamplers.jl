pdmp_sample(d::Integer, args...; kwargs...) = pdmp_sample(randn(d), args...; kwargs...)
pdmp_sample(x₀::AbstractVector, θ₀::AbstractVector, flow::ContinuousDynamics, args...; kwargs...) = pdmp_sample(SkeletonPoint(x₀, θ₀), flow, args...; kwargs...)
pdmp_sample(x₀::AbstractVector, flow::ContinuousDynamics, args...; kwargs...) = pdmp_sample(SkeletonPoint(x₀, initialize_velocity(flow, length(x₀))), flow, args...; kwargs...)

# the inner workhorse that all convenience constructors end up at
function pdmp_sample(
    ξ₀::SkeletonPoint, flow::ContinuousDynamics, model::PDMPModel,
    alg::PoissonTimeStrategy,
    # needs to be a separate type to determine when it's finished!
    # options: Time and/ or no.events + ess (advanced)
    t₀::Real=0.0, T::Real=10_000, t_warmup::Real=0.0; # TODO: better default!
    progress::Bool=true
)

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

    t_warmup < (T - t₀) || throw(ArgumentError("t_warmup ($t_warmup) is larger than T - t₀ ($T - $t₀ = $(T - t₀)), which implies storing nothing at all. This is probably an error."))

    # state, grad_, alg_, cache, stats = initialize_state(flow, model.grad, alg, t₀, ξ₀)
    state, model_, alg_, cache, stats = initialize_state(flow, model, alg, t₀, ξ₀)

    validate_state(state, flow, "at initialization")

    trace_manager = TraceManager(state, flow, alg, t_warmup)
    health = HealthMonitor()
    adapter = default_adapter(flow, model_.grad, t_warmup ÷ 10, t₀, t_warmup ÷ 10)

    # progressmanager = ProgressManager(progress, T, t₀, t_warmup, progress_stops)
    if progress
        progress_stops = 300
        prg = ProgressMeter.Progress(progress_stops, dt=1)
        tstop = T / progress_stops
    end


    # if grad_ isa SubsampledGradient
    #     last_anchor_update_t = t₀
    #     anchor_update_Δt = (t_warmup - t₀) / grad_.no_anchor_updates
    #     update_anchor_during_warmup = ispositive(anchor_update_Δt) && isfinite(anchor_update_Δt)
    #     saved_warmup_trace_first_time = false
    #     warmup_trace = update_anchor_during_warmup ? TT(state, flow) : nothing
    # else
    #     last_anchor_update_t = t₀
    #     anchor_update_Δt = -1
    #     update_anchor_during_warmup = false
    # end

    # consecutive_reject_count = 0
    # consecutive_reject_limit = 1_000
    while state.t[] < T # !isdone(state, trace, stopping_criterion)

        τ, event_type, meta = next_event_time(model_, flow, alg_, state, cache, stats)

        @assert ispositive(τ) "Proposed event time τ ($τ) is non-positive. Sampler is stuck!"

        needs_saving, saving_args = handle_event!(τ, model_.grad, flow, alg_, state, cache, event_type, meta, stats)

        # @show τ, event_type, state.t[], stats.last_rejected, needs_saving, length(trace_manager.main_trace.events)
        needs_saving && record_event!(trace_manager, state, flow, saving_args)

        adapt!(adapter, state, flow, model_.grad, trace_manager)

        check_health!(health, stats)
        # if progress
        #     update_progress!(prog_meter, state.t[], t_stop)
        # end

        if progress && state.t[] > tstop
            tstop += T / progress_stops
            ProgressMeter.next!(prg)
        end

        # if needs_saving

        #     if state.t[] >= t_warmup

        #         if !saved_first_time
        #             trace = TT(state, flow)
        #             saved_first_time = true
        #         end

        #         if !isnothing(saving_args) && trace isa FactorizedTrace
        #             @assert flow isa FactorizedDynamics "saving_args provided, but trace is not FactorizedTrace, flow = $(typeof(flow)), trace = $(typeof(trace))!"
        #             push!(trace, state, saving_args)
        #         else
        #             push!(trace, state)
        #         end
        #     elseif update_anchor_during_warmup
        #         if !saved_warmup_trace_first_time
        #             warmup_trace = TT(state, flow)
        #             saved_warmup_trace_first_time = true
        #         end
        #         if !isnothing(saving_args) && warmup_trace isa FactorizedTrace
        #             @assert flow isa FactorizedDynamics "saving_args provided, but trace is not FactorizedTrace, flow = $(typeof(flow)), trace = $(typeof(trace))!"
        #             push!(warmup_trace, state, saving_args)
        #         else
        #             push!(warmup_trace, state)
        #         end
        #     end
        # end

        # let's do this only once per event
        # if grad_ isa SubsampledGradient
        #     grad_.resample_indices!(grad_.nsub)

        #     if update_anchor_during_warmup && state.t[] <= t_warmup && state.t[] - last_anchor_update_t >= anchor_update_Δt
        #         last_anchor_update_t = state.t[]
        #         grad_.update_anchor!(warmup_trace)
        #     end
        # end

        # if stats.last_rejected
        #     consecutive_reject_count += 1
        # else
        #     consecutive_reject_count = 0
        # end
        # stats.last_rejected = false

        # if consecutive_reject_count > consecutive_reject_limit
        #     throw(error("The algorithm rejected $(consecutive_reject_limit) consecutive proposals. Check the algorithm and model settings."))
        # end

        # --- PHASE 4: DIAGNOSTICS ---
        # check_health!(health, stats.last_rejected)
        # # if progress
        # #     update_progress!(prog_meter, state.t[], t_stop)
        # # end

        # if progress && state.t[] > tstop
        #     tstop += T / progress_stops
        #     ProgressMeter.next!(prg)
        # end
    end

    # this is misleading because the reflections events are more expensive... just count these manually?
    # if grad_ isa SubsampledGradient && grad_.use_full_gradient_for_reflections
    #     stats.∇f_calls += stats.reflections_events
    # end

    return get_main_trace(trace_manager), stats
    # return trace, stats

end


# """
# initialize_state(::ContinuousDynamics, ::GradientStrategy, t₀::Real, ξ₀::SkeletonPoint)

# Initialize all required information for the specific dynamics. Should return the following:

# """
# function initialize_state(flow::ContinuousDynamics, grad::GradientStrategy, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint)
#     ξ = copy(ξ₀)
#     t = t₀
#     stats = StatisticCounter()
#     state = alg isa Sticky ? StickyPDMPState(t, ξ) : PDMPState(t, ξ)
#     cache = add_gradient_to_cache(initialize_cache(flow, grad, alg, t, ξ), ξ)
#     alg_ = _to_internal(alg, flow, grad, state, cache, stats)
#     grad_ = with_stats(grad, stats)
#     grad_ isa SubsampledGradient && grad_.resample_indices!(grad_.nsub)
#     return state, grad_, alg_, cache, stats
# end

function initialize_state(flow::ContinuousDynamics, model::PDMPModel, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint)
    ξ = copy(ξ₀)
    t = t₀
    stats = StatisticCounter()
    state = alg isa Sticky ? StickyPDMPState(t, ξ) : PDMPState(t, ξ)
    cache = add_gradient_to_cache(initialize_cache(flow, model.grad, alg, t, ξ), ξ)
    alg_ = _to_internal(alg, flow, model, state, cache, stats)
    model_ = with_stats(model, stats)
    model_.grad isa SubsampledGradient && model_.grad.resample_indices!(model_.grad.nsub)
    return state, model_, alg_, cache, stats
end

function add_gradient_to_cache(cache::NamedTuple, ξ::SkeletonPoint)
    ∇ϕx = similar(ξ.x)
    if haskey(cache, :∇ϕx)
        if !(cache.∇ϕx isa typeof(ξ.x) && length(cache.∇ϕx) == length(ξ.x))
            throw(ArgumentError("cache.∇ϕx was given manually, but must be of the same type as ξ.x"))
        end
    else
        cache = merge(cache, (; ∇ϕx))
    end
    return cache
end

# TODO: these need to move to the respective dynamics files?
function initialize_cache(::ContinuousDynamics, ::GradientStrategy, ::PoissonTimeStrategy, ::Real, ::SkeletonPoint)
    (;)
end
function initialize_cache(flow::PreconditionedDynamics, grad::GlobalGradientStrategy, alg::PoissonTimeStrategy, t::Real, ξ::SkeletonPoint)
    return initialize_cache(flow.dynamics, grad, alg, t, ξ)
end
function initialize_cache(::BouncyParticle, ::GlobalGradientStrategy, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z=similar(ξ.x))
end
function initialize_cache(::Boomerang, ::GlobalGradientStrategy, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z=similar(ξ.x))
end

function initialize_cache(flow::ZigZag, ::CoordinateWiseGradient, thinningstrategy::ThinningStrategy, t::Real, ξ::SkeletonPoint)

    # PriorityQueue stores: Coordinate Index => Absolute Event Time
    # could also be a MutableBinaryMinHeap{Float64, DataStructures.FasterForward}, which might be more efficient.
    # see https://juliacollections.github.io/DataStructures.jl/stable/heaps/
    # not entirely sure to what extent this matters
    pq = PriorityQueue{Int,Float64}()

    # Initialize all d clocks with their proposed event times
    for i in eachindex(ξ.x)
        abc_i = ab_i(i, ξ, thinningstrategy, flow, nothing)
        t_event = t + poisson_time(abc_i[1], abc_i[2], rand())
        # enqueue!(pq, i => t_event)
        push!(pq, i => t_event)
    end

    return (; pq)
end

function handle_event!(τ::Real, gradient_strategy::GlobalGradientStrategy, flow::ContinuousDynamics, alg::PoissonTimeStrategy, state::AbstractPDMPState, cache, event_type::Symbol, meta, stats::StatisticCounter)

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
                i = meta
                reflect!(state.ξ, zero(eltype(cache.∇ϕx)), i, flow)
            else
                ∇ϕx = compute_gradient!(state, gradient_strategy, flow, cache)
                saving_args = reflect!(state, ∇ϕx, flow, cache)
            end
            needs_saving = true
            alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
            # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
        else

            # Reuse gradient from meta when available (e.g., GridThinningStrategy already computed it)
            if meta isa NamedTuple && haskey(meta, :∇ϕx) && length(meta.∇ϕx) == length(state.ξ.x)
                ∇ϕx = meta.∇ϕx
            else
                ∇ϕx = compute_gradient_for_reflection!(state, gradient_strategy, flow, cache)
            end

            # meta == abc for ThinningStrategy
            if accept_reflection_event(alg, state.ξ, ∇ϕx, flow, τ, cache, meta)
                stats.reflections_accepted += 1
                # @show "before reflect", state, ∇ϕx, flow, cache
                saving_args = reflect!(state, ∇ϕx, flow, cache)
                # @show "after reflect", state
                needs_saving = true

                alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
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
        refresh_velocity!(state, flow)
        # @show "after refresh", state
        # saving_args = nothing # could be informed by the refresh function?
        needs_saving = true
        stats.refreshment_events += 1

        # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
        # should be only the unfreeze times?
        # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
        alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)

    elseif event_type == :sticky

        stats.sticky_events += 1
        i = meta
        stick_or_unstick!(state, flow, alg, i)
        validate_state(state, flow, "after stick_or_unstick!")
        needs_saving = true
        if isfactorized(flow)
            saving_args = i
        end

    elseif event_type == :horizon_hit

        # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
        # should be only the unfreeze times?
        # alg isa StickyLoopState && update_all_freeze_times!(alg, state, flow)
        alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)
        # for now only when horizon is reached.
        stats.last_rejected = true
    end

    # alg isa StickyLoopState && update_all_stick_times!(alg, state, flow)

    if alg isa StickyLoopState
        @assert all(>=(state.t[]), alg.sticky_times) "some sticky_times are negative!"
    end

    return needs_saving, saving_args
end

function handle_event!(τ::Real, gradient_strategy::CoordinateWiseGradient, flow::ZigZag, alg::ThinningStrategy, state::PDMPState, cache, event_type, meta, stats::StatisticCounter)

    stats.last_rejected = false
    # rename for clarity
    pq = cache.pq
    i₀ = meta  # winning coordinate
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

    if rand() * l_bound_i₀ <= l_i₀

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
            t_event = state.t[] + poisson_time(abc_i_new[1], abc_i_new[2], rand())
            # enqueue!(pq, i => t_event)
            push!(pq, i => t_event)
        end
    else
        abc_i₀_new = ab_i(i₀, ξ, alg, flow, nothing)
        t_event = state.t[] + poisson_time(abc_i₀_new[1], abc_i₀_new[2], rand())
        # enqueue!(pq, i₀ => t_event)
        push!(pq, i₀ => t_event)
        stats.last_rejected = true
    end

    return needs_saving, saving_args
end


# TODO: needs to move to the respective strategies!
# ideally this function would not exist though... that is possible if thinning returns
# "ghost events" as a type.
# acceptance happens elsewhere
accept_reflection_event(::GridAdaptiveState, args...) = true
accept_reflection_event(::RootsPoissonTimeStrategy, args...) = true
accept_reflection_event(alg::StickyLoopState, args...) = accept_reflection_event(alg.inner_alg_state, args...)
# the only type for which we need this function!
function accept_reflection_event(::ThinningStrategy, ξ::SkeletonPoint, ∇ϕx::AbstractVector, flow::ContinuousDynamics, dt::Real, cache, meta)

    # rename for clarity
    abc = meta
    l = λ(ξ, ∇ϕx, flow)
    l_bound = pos(abc[1] + abc[2] * dt)

    # TODO: don't throw when adapting!
    #l > l_bound && !(l <= 1e-6) && error("Tuning parameter `c` too small: l=$l, lb=$l_bound")
    # for now, the bound should be way tighter
    # l / l_bound < 0.6 && error("Tuning parameter `c` too large? dt = $dt, l=$l, lb=$l_bound, l / l_bound = $(l / l_bound)")

    u = rand()
    accept = u * l_bound <= l

    if accept
        l > l_bound && !(l <= 1e-6) && error("Tuning parameter `c` too small: l=$l, lb=$l_bound")
    else
        # @info "rejecting u * l_bound = $(u) * $(l_bound) = $(u * l_bound) <= l = $l where abc=$abc and dt=$dt"
    end

    return accept
end
