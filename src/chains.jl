struct PDMPChains{T<:AbstractPDMPTrace,S}
    traces::Vector{T}
    stats::Vector{S}
end

n_chains(chains::PDMPChains) = length(chains.traces)

Base.getindex(chains::PDMPChains, i::Integer) = (chains.traces[i], chains.stats[i])
Base.firstindex(chains::PDMPChains) = 1
Base.lastindex(chains::PDMPChains) = n_chains(chains)
Base.length(chains::PDMPChains) = n_chains(chains)
Base.eachindex(chains::PDMPChains) = eachindex(chains.traces)

function Base.iterate(chains::PDMPChains)
    n_chains(chains) == 0 && return nothing
    return chains.traces[1], Val(:stats)
end
function Base.iterate(chains::PDMPChains, ::Val{:stats})
    return chains.stats[1], Val(:done)
end
Base.iterate(::PDMPChains, ::Val{:done}) = nothing

Statistics.mean(chains::PDMPChains; chain::Integer=1)    = Statistics.mean(chains.traces[chain])
Statistics.var(chains::PDMPChains; chain::Integer=1)     = Statistics.var(chains.traces[chain])
Statistics.std(chains::PDMPChains; chain::Integer=1)     = Statistics.std(chains.traces[chain])
Statistics.cov(chains::PDMPChains; chain::Integer=1)     = Statistics.cov(chains.traces[chain])
Statistics.cor(chains::PDMPChains; chain::Integer=1)     = Statistics.cor(chains.traces[chain])
Statistics.median(chains::PDMPChains; kwargs...)          = Statistics.median(chains.traces[1]; kwargs...)

function Statistics.quantile(chains::PDMPChains, p; chain::Integer=1, kwargs...)
    Statistics.quantile(chains.traces[chain], p; kwargs...)
end

cdf(chains::PDMPChains, q::Real; chain::Integer=1, kwargs...) = cdf(chains.traces[chain], q; kwargs...)
ess(chains::PDMPChains; chain::Integer=1, kwargs...)           = ess(chains.traces[chain]; kwargs...)
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
    print(io, "PDMPChains with $nc chain$(nc == 1 ? "" : "s") ($(join(n_events, ", ")) events)")
end
