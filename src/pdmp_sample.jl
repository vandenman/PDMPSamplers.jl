pdmp_sample(d::Integer, args...; kwargs...) = pdmp_sample(randn(d), args...; kwargs...)
pdmp_sample(x₀::AbstractVector, θ₀::AbstractVector, flow::ContinuousDynamics, args...; kwargs...) = pdmp_sample(SkeletonPoint(x₀, θ₀), flow, args...; kwargs...)
pdmp_sample(x₀::AbstractVector, flow::ContinuousDynamics, args...; kwargs...) = pdmp_sample(SkeletonPoint(x₀, initialize_velocity(flow, length(x₀))), flow, args...; kwargs...)
pdmp_sample(ξ₀::SkeletonPoint, flow::ContinuousDynamics, f::Function, args...; kwargs...) = pdmp_sample(ξ₀::SkeletonPoint, flow::ContinuousDynamics, FullGradient(f), args...; kwargs...)

# the inner workhorse that all convenience constructors end up at
function pdmp_sample(
        ξ₀::SkeletonPoint, flow::ContinuousDynamics, grad::GradientStrategy,
        alg::PoissonTimeStrategy,
        # needs to be a separate type to determine when it's finished!
        # options: Time and/ or no.events + ess (advanced)
        t₀::Real = 0.0, T::Real = 10_000;
        progress::Bool=true
    )

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


    state, grad_, alg_, cache, stats = initialize_state(flow, grad, alg, t₀, ξ₀)

    TT = _trace_type(flow, alg)
    trace = TT(state, flow)

    if progress
        progress_stops = 300
        prg = ProgressMeter.Progress(progress_stops, dt = 1)
        tstop = T / progress_stops
    end

    consecutive_reject_count = 0
    consecutive_reject_limit = 1_000
    while state.t[] < T # !isdone(state, trace, stopping_criterion)

        τ, event_type, meta = next_event_time(grad_, flow, alg_, state, cache, stats)

        @assert ispositive(τ) "Proposed event time τ is non-positive. Sampler is stuck!"

        needs_saving, saving_args = handle_event!(τ, grad_, flow, alg_, state, trace, cache, event_type, meta, stats)

        if needs_saving
            if !isnothing(saving_args) && trace isa FactorizedTrace
                @assert flow isa FactorizedDynamics "saving_args provided, but trace is not FactorizedTrace, flow = $(typeof(flow)), trace = $(typeof(trace))!"
                push!(trace, state, saving_args)
            else
                push!(trace, state)
            end
        end

        # let's do this only once per iteration
        grad_ isa SubsampledGradient && grad_.resample_indices!(grad_.nsub)

        if stats.last_rejected
            consecutive_reject_count += 1
        else
            consecutive_reject_count = 0
        end
        stats.last_rejected = false

        if consecutive_reject_count > consecutive_reject_limit
            throw(error("The algorithm rejected $(consecutive_reject_limit) consecutive proposals. Check the algorithm and model settings."))
        end

        if progress && state.t[] > tstop
            tstop += T / progress_stops
            ProgressMeter.next!(prg)
        end
    end

    return trace, stats

end


"""
initialize_state(::ContinuousDynamics, ::GradientStrategy, t₀::Real, ξ₀::SkeletonPoint)

Initialize all required information for the specific dynamics. Should return the following:

"""
function initialize_state(flow::ContinuousDynamics, grad::GradientStrategy, alg::PoissonTimeStrategy, t₀::Real, ξ₀::SkeletonPoint)
    ξ = copy(ξ₀)
    t = t₀
    stats = StatisticCounter()
    state = alg isa Sticky ? StickyPDMPState(t, ξ) : PDMPState(t, ξ)
    cache = add_gradient_to_cache(initialize_cache(flow, grad, alg, t, ξ), ξ)
    alg_ = _to_internal(alg, flow, grad, state, cache, stats)
    grad = with_stats(grad, stats)
    grad isa SubsampledGradient && grad.resample_indices!(grad.nsub)
    return state, grad, alg_, cache, stats
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
function initialize_cache(::BouncyParticle, ::FullGradient, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z = similar(ξ.x))
end

function initialize_cache(::Boomerang, ::FullGradient, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z = similar(ξ.x), Γx = similar(ξ.x), Γθ = similar(ξ.x))
end

function initialize_cache(flow::ZigZag, ::CoordinateWiseGradient, thinningstrategy::ThinningStrategy, t::Real, ξ::SkeletonPoint)

    # PriorityQueue stores: Coordinate Index => Absolute Event Time
    # could also be a MutableBinaryMinHeap{Float64, DataStructures.FasterForward}, which might be more efficient.
    # see https://juliacollections.github.io/DataStructures.jl/stable/heaps/
    # not entirely sure to what extent this matters
    pq = PriorityQueue{Int, Float64}()

    # Initialize all d clocks with their proposed event times
    for i in eachindex(ξ.x)
        abc_i = ab_i(i, ξ, thinningstrategy, flow, nothing)
        t_event = t + poisson_time(abc_i[1], abc_i[2], rand())
        enqueue!(pq, i => t_event)
    end

    return (; pq)
end

function handle_event!(τ::Real, gradient_strategy::GlobalGradientStrategy, flow::ContinuousDynamics, alg::PoissonTimeStrategy, state::AbstractPDMPState, trace::AbstractPDMPTrace, cache, event_type::Symbol, meta, stats::StatisticCounter)

    # Always move forward in time
    move_forward_time!(state, τ, flow)
    validate_state(state, flow, "after moving forward in time")

    # assume a ghost event
    needs_saving = false
    saving_args = nothing
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

            ∇ϕx = compute_gradient!(state, gradient_strategy, flow, cache)

            # meta == abc for ThinningStrategy
            if accept_reflection_event(alg, state.ξ, ∇ϕx, flow, τ, cache, meta)
                stats.reflections_accepted += 1
                saving_args = reflect!(state, ∇ϕx, flow, cache)
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
        refresh_velocity!(state, flow)
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
        saving_args = i

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

function handle_event!(τ::Real, gradient_strategy::CoordinateWiseGradient, flow::ZigZag, alg::ThinningStrategy, state::PDMPState, trace, cache, event_type, meta, stats::StatisticCounter)

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
            enqueue!(pq, i => t_event)
        end
    else
        abc_i₀_new = ab_i(i₀, ξ, alg, flow, nothing)
        t_event = state.t[] + poisson_time(abc_i₀_new[1], abc_i₀_new[2], rand())
        enqueue!(pq, i₀ => t_event)
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
    l_bound = pos(abc[1] + abc[2]*dt)

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
