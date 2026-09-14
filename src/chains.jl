struct PDMPTerminalState{F}
    t::Float64
    position::Vector{Float64}
    physical_velocity::Vector{Float64}
    stored_frozen_velocity::Vector{Float64}
    free::BitVector
    flow::F
end

function PDMPTerminalState(state::AbstractPDMPState, flow)
    stored = state isa StickyPDMPState ? copy(state.stored_velocity) :
        zeros(Float64, length(state.ξ))
    free = state isa StickyPDMPState ? copy(state.free) : trues(length(state.ξ))
    # Endpoint provenance must also work for legacy covariance wrappers whose
    # internal factor object has no `copy` method.  This does not alter the
    # sampler's working flow; it only snapshots it for export.
    flow_snapshot = try
        _copy_flow(flow)
    catch err
        err isa MethodError || rethrow()
        deepcopy(flow)
    end
    return PDMPTerminalState(Float64(state.t[]), Float64.(state.ξ.x),
        Float64.(state.ξ.θ), stored, free, flow_snapshot)
end

struct PDMPChains{T<:AbstractPDMPTrace,S,E}
    traces::Vector{T}
    stats::Vector{S}
    initial_states::Vector{E}
    retained_initial_states::Vector{E}
    terminal_states::Vector{E}
end

PDMPChains(traces::Vector{T}, stats::Vector{S}) where {T<:AbstractPDMPTrace,S} =
    PDMPChains(traces, stats, Any[], Any[], Any[])
PDMPChains(traces::Vector{T}, stats::Vector{S}, terminal_states::Vector{E}) where
        {T<:AbstractPDMPTrace,S,E} =
    PDMPChains(traces, stats, Any[], Any[], terminal_states)

n_chains(chains::PDMPChains) = length(chains.traces)

Base.getindex(chains::PDMPChains, i::Integer) = (chains.traces[i], chains.stats[i])
terminal_state(chains::PDMPChains, i::Integer=1) = chains.terminal_states[i]
initial_state(chains::PDMPChains, i::Integer=1) = chains.initial_states[i]
retained_initial_state(chains::PDMPChains, i::Integer=1) =
    chains.retained_initial_states[i]
Base.firstindex(chains::PDMPChains) = 1
Base.lastindex(chains::PDMPChains) = n_chains(chains)
Base.length(chains::PDMPChains) = n_chains(chains)
Base.eachindex(chains::PDMPChains) = eachindex(chains.traces)

function Base.iterate(chains::PDMPChains)
    nchains = n_chains(chains)
    iszero(nchains) && return nothing
    isone(nchains) && return chains.traces[1], Val(:stats)
    return (chains.traces[1], chains.stats[1]), 2
end
function Base.iterate(chains::PDMPChains, i::Int)
    i > n_chains(chains) && return nothing
    return (chains.traces[i], chains.stats[i]), i + 1
end
function Base.iterate(chains::PDMPChains, ::Val{:stats})
    return chains.stats[1], Val(:done)
end
Base.iterate(::PDMPChains, ::Val{:done}) = nothing

function Base.only(chains::PDMPChains)
    nchains = n_chains(chains)
    isone(nchains) || throw(ArgumentError("only(chains) requires exactly one chain, got $nchains"))
    return chains[1]
end

Statistics.mean(chains::PDMPChains; chain::Integer=1)    = Statistics.mean(chains.traces[chain])
Statistics.var(chains::PDMPChains; chain::Integer=1)     = Statistics.var(chains.traces[chain])
Statistics.var(chains::PDMPChains, means::AbstractVector; chain::Integer=1) = Statistics.var(chains.traces[chain], means)
Statistics.std(chains::PDMPChains; chain::Integer=1)     = Statistics.std(chains.traces[chain])
Statistics.std(chains::PDMPChains, means::AbstractVector; chain::Integer=1) = Statistics.std(chains.traces[chain], means)
Statistics.cov(chains::PDMPChains; chain::Integer=1)     = Statistics.cov(chains.traces[chain])
Statistics.cov(chains::PDMPChains, means::AbstractVector; chain::Integer=1) = Statistics.cov(chains.traces[chain], means)
Statistics.cor(chains::PDMPChains; chain::Integer=1)     = Statistics.cor(chains.traces[chain])
Statistics.median(chains::PDMPChains; kwargs...)          = Statistics.median(chains.traces[1]; kwargs...)

function Statistics.quantile(chains::PDMPChains, p; chain::Integer=1, kwargs...)
    Statistics.quantile(chains.traces[chain], p; kwargs...)
end

cdf(chains::PDMPChains, q::Real; chain::Integer=1, kwargs...) = cdf(chains.traces[chain], q; kwargs...)
ess(chains::PDMPChains; chain::Integer=1, kwargs...)           = ess(chains.traces[chain]; kwargs...)
ess(chains::PDMPChains, means::AbstractVector, variances::AbstractVector; chain::Integer=1, kwargs...) = ess(chains.traces[chain], means, variances; kwargs...)
inclusion_probs(chains::PDMPChains; chain::Integer=1)          = inclusion_probs(chains.traces[chain])

PDMPDiscretize(chains::PDMPChains, dt; chain::Integer=1) = PDMPDiscretize(chains.traces[chain], dt)

"""
    adaptive_dt(trace::AbstractPDMPTrace; n_batches=0) -> (dt, n_disc, ct_ess_min)

Compute the adaptive discretization step size for a PDMP trace.

The continuous-time ESS (ct-ESS) measures the efficiency of the time-average
estimator `(1/T)∫f(X_t)dt`, which benefits from cancellation of oscillations in
the autocorrelation function, especially for Boomerang dynamics.  Discrete
snapshots at spacing `Δt` do not benefit from this cancellation: consecutive
samples remain positively correlated (discrete IACT `τ_disc > 1`), giving
`disc-ESS = n_disc / τ_disc < ct-ESS` when `n_disc = ct-ESS`.

This function corrects for that bias via a two-pass approach:
1. Trial: discretize with `n_disc_trial = ceil(ct_ess_min)` and estimate
   `τ_disc = n_disc_trial / disc_ess_trial` from the resulting matrix.
2. Correction: set `n_disc = ceil(ct_ess_min × τ_disc)`, capped at
   `10 × n_disc_trial` to bound memory use.

For typical PDMPs `τ_disc ≈ 2–4`, so `n_disc ≈ 2–4 × ct-ESS` and
`disc-ESS ≈ ct-ESS` after correction.
"""
function adaptive_dt(trace::AbstractPDMPTrace; n_batches::Integer=0)
    ct_ess_vals = n_batches > 0 ? ess(trace; n_batches) : ess(trace)
    ct_ess_min = minimum(ct_ess_vals)
    t_start = first_event_time(trace)
    t_end = last_event_time(trace)
    total_time = t_end - t_start

    n_disc_trial = max(ceil(Int, ct_ess_min), 10)
    mat_trial = Matrix(PDMPDiscretize(trace, total_time / n_disc_trial))
    disc_ess_trial = _disc_ess_min(mat_trial)

    tau_disc = clamp(n_disc_trial / disc_ess_trial, 1.0, 10.0)
    n_disc = max(ceil(Int, ct_ess_min * tau_disc), n_disc_trial)
    dt = total_time / n_disc
    return dt, n_disc, ct_ess_min
end

function adaptive_dt(chains::PDMPChains; chain::Integer=1, kwargs...)
    adaptive_dt(chains.traces[chain]; kwargs...)
end

"""
    adaptive_discretize(trace::AbstractPDMPTrace; kwargs...)
    adaptive_discretize(chains::PDMPChains; chain=1, kwargs...)

Discretize a PDMP trace using an adaptive number of points determined by
the continuous-time ESS. Sets `n_disc = ceil(min(ct_ESS))` so that the
discretized samples preserve the information content of the continuous trace.

Returns `(matrix, n_disc, ct_ess_min)` where `matrix` is `n_disc × d`.
"""
function adaptive_discretize(trace::AbstractPDMPTrace; kwargs...)
    dt, n_disc, ct_ess_min = adaptive_dt(trace; kwargs...)
    mat = Matrix(PDMPDiscretize(trace, dt))
    return mat, n_disc, ct_ess_min
end

function adaptive_discretize(chains::PDMPChains; chain::Integer=1, kwargs...)
    adaptive_discretize(chains.traces[chain]; kwargs...)
end

function Base.show(io::IO, chains::PDMPChains)
    nc = n_chains(chains)
    n_events = [length(chains.traces[i]) for i in 1:nc]
    print(io, "PDMPChains with $nc chain$(isone(nc) ? "" : "s") ($(join(n_events, ", ")) events)")

    total_lazy_low_tightness = sum(_get_counter_lazy_fallback_low_tightness, chains.stats)
    total_lazy_bound_violation = sum(_get_counter_lazy_fallback_bound_violation, chains.stats)
    total_lazy_attempts = sum(_get_counter_lazy_proposal_attempts, chains.stats)
    total_lazy_rejections = sum(_get_counter_lazy_proposal_rejections, chains.stats)
    total_grid_resets = sum(_get_counter_grid_resets_from_dynamics_adaptation, chains.stats)

    if total_lazy_low_tightness > 0 || total_lazy_bound_violation > 0 || total_lazy_attempts > 0 || total_grid_resets > 0
        print(io,
            " | lazy_fallbacks(low_tightness=", total_lazy_low_tightness,
            ", bound_violation=", total_lazy_bound_violation,
            ") | lazy_proposals=", total_lazy_attempts,
            "/", total_lazy_rejections,
            " rejected | grid_resets=", total_grid_resets)
    end
end
