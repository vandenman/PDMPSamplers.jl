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
    return PDMPTerminalState(Float64(state.t[]), Float64.(state.ξ.x),
        Float64.(state.ξ.θ), stored, free, deepcopy(flow))
end

"""
    PDMPChains

The result of [`pdmp_sample`](@ref): one trace and its run statistics per
chain. `chains[i]` returns `(trace, stats)` of chain `i`.

The summaries `mean`, `var`, `std`, `cov`, `cor`, `quantile`, `median`, `cdf`,
`inclusion_probs` and `ess` pool all chains by default, weighting each chain by
the length of its run (`ess` sums the per-chain estimates). Pass `chain=i` to
summarize one chain. The same holds for the transformed estimators
(`mean(chains, transforms)` and so on). `PDMPDiscretize` reads off a single path, so it requires
`chain=i` when there is more than one chain.
"""
struct PDMPChains{T<:AbstractPDMPTrace,S,E}
    traces::Vector{T}
    stats::Vector{S}
    initial_states::Vector{E}
    retained_initial_states::Vector{E}
    terminal_states::Vector{E}
end

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

function Base.iterate(chains::PDMPChains, i::Int=1)
    i > n_chains(chains) && return nothing
    return chains[i], i + 1
end

function Base.only(chains::PDMPChains)
    nchains = n_chains(chains)
    isone(nchains) || throw(ArgumentError("only(chains) requires exactly one chain, got $nchains"))
    return chains[1]
end

# Summaries of `PDMPChains` pool all chains by default; pass `chain=i` for one
# chain. Pooling treats the chains as one process observed for their combined
# time, so each chain is weighted by the length of its run.

_chain_duration(trace::AbstractPDMPTrace) = last_event_time(trace) - first_event_time(trace)

function _chain_weights(chains::PDMPChains)
    durations = map(_chain_duration, chains.traces)
    return durations ./ sum(durations)
end

_pool(chains::PDMPChains, chain::Integer) = false
_pool(chains::PDMPChains, ::Nothing) = n_chains(chains) > 1
_single(chains::PDMPChains, chain::Integer) = chains.traces[chain]
_single(chains::PDMPChains, ::Nothing) = chains.traces[1]

# Time-weighted average of a per-chain summary.
_pooled_average(f, chains::PDMPChains) =
    sum(w * f(trace) for (w, trace) in zip(_chain_weights(chains), chains.traces))

function Statistics.mean(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.mean(_single(chains, chain))
    return _pooled_average(Statistics.mean, chains)
end

# Pooled variance: within-chain variance plus the spread of the chain means.
function Statistics.var(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.var(_single(chains, chain))
    return Statistics.var(chains, Statistics.mean(chains))
end
function Statistics.var(chains::PDMPChains, means::AbstractVector; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.var(_single(chains, chain), means)
    return _pooled_average(trace -> Statistics.var(trace, means), chains)
end

Statistics.std(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing) =
    sqrt.(Statistics.var(chains; chain))
Statistics.std(chains::PDMPChains, means::AbstractVector; chain::Union{Nothing,Integer}=nothing) =
    sqrt.(Statistics.var(chains, means; chain))

function Statistics.cov(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.cov(_single(chains, chain))
    return Statistics.cov(chains, Statistics.mean(chains))
end
function Statistics.cov(chains::PDMPChains, means::AbstractVector; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.cov(_single(chains, chain), means)
    return _pooled_average(trace -> Statistics.cov(trace, means), chains)
end

function Statistics.cor(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.cor(_single(chains, chain))
    Σ = Statistics.cov(chains)
    s = sqrt.(LinearAlgebra.diag(Σ))
    return Σ ./ (s * s')
end

function cdf(chains::PDMPChains, q::Real; coordinate::Integer, chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return cdf(_single(chains, chain), q; coordinate)
    return _pooled_average(trace -> cdf(trace, q; coordinate), chains)
end

function _pooled_quantile(chains::PDMPChains, p::Real, coordinate::Integer)
    bounds = [_trace_coordinate_bounds(trace, coordinate) for trace in chains.traces]
    lo, hi = minimum(first, bounds), maximum(last, bounds)
    return _invert_monotone_cdf(q -> cdf(chains, q; coordinate), lo, hi, p)
end

function Statistics.quantile(chains::PDMPChains, p::Real; coordinate::Integer=-1,
        chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.quantile(_single(chains, chain), p; coordinate)
    (0 < p < 1) || throw(DomainError(p, "Quantile probability must be in (0, 1)"))
    if coordinate == -1
        d = length(first(chains.traces[1]).position)
        return [_pooled_quantile(chains, p, j) for j in 1:d]
    end
    return _pooled_quantile(chains, p, coordinate)
end
function Statistics.quantile(chains::PDMPChains, p::AbstractVector{<:Real}; coordinate::Integer,
        chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.quantile(_single(chains, chain), p; coordinate)
    all(x -> 0 < x < 1, p) || throw(DomainError(p, "All quantile probabilities must be in (0, 1)"))
    return [_pooled_quantile(chains, pᵢ, coordinate) for pᵢ in p]
end

Statistics.median(chains::PDMPChains; coordinate::Integer=-1, chain::Union{Nothing,Integer}=nothing) =
    Statistics.quantile(chains, 0.5; coordinate, chain)

# The pooled ESS is the sum of the per-chain estimates.
function ess(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing, kwargs...)
    _pool(chains, chain) || return ess(_single(chains, chain); kwargs...)
    return sum(ess(trace; kwargs...) for trace in chains.traces)
end
function ess(chains::PDMPChains, means::AbstractVector, variances::AbstractVector;
        chain::Union{Nothing,Integer}=nothing, kwargs...)
    _pool(chains, chain) || return ess(_single(chains, chain), means, variances; kwargs...)
    return sum(ess(trace, means, variances; kwargs...) for trace in chains.traces)
end

function inclusion_probs(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return inclusion_probs(_single(chains, chain))
    return _pooled_average(inclusion_probs, chains)
end

# Transformed estimators (see transformed_estimators.jl), pooled the same way.
const _Transforms = AbstractVector{<:ParameterTransform}

function Statistics.mean(chains::PDMPChains, transforms::_Transforms; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.mean(_single(chains, chain), transforms)
    return _pooled_average(trace -> Statistics.mean(trace, transforms), chains)
end

function Statistics.var(chains::PDMPChains, transforms::_Transforms; chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.var(_single(chains, chain), transforms)
    m = Statistics.mean(chains, transforms)
    return _pooled_average(chains) do trace
        Statistics.var(trace, transforms) .+ (Statistics.mean(trace, transforms) .- m) .^ 2
    end
end

Statistics.std(chains::PDMPChains, transforms::_Transforms; chain::Union{Nothing,Integer}=nothing) =
    sqrt.(Statistics.var(chains, transforms; chain))

function cdf(chains::PDMPChains, q::Real, transforms::_Transforms; coordinate::Integer,
        chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return cdf(_single(chains, chain), q, transforms; coordinate)
    return _pooled_average(trace -> cdf(trace, q, transforms; coordinate), chains)
end

# The transforms are monotone, so a quantile of the transformed coordinate is
# the transform of a quantile of the pooled untransformed coordinate.
function Statistics.quantile(chains::PDMPChains, p::Real, transforms::_Transforms;
        coordinate::Integer=-1, chain::Union{Nothing,Integer}=nothing)
    _pool(chains, chain) || return Statistics.quantile(_single(chains, chain), p, transforms; coordinate)
    (0 < p < 1) || throw(DomainError(p, "Quantile probability must be in (0, 1)"))
    function transformed(j)
        transform = transforms[j]
        return transform(_pooled_quantile(chains, is_increasing(transform) ? p : 1 - p, j))
    end
    coordinate == -1 && return [transformed(j) for j in eachindex(transforms)]
    return transformed(coordinate)
end

Statistics.median(chains::PDMPChains, transforms::_Transforms; coordinate::Integer=-1,
        chain::Union{Nothing,Integer}=nothing) =
    Statistics.quantile(chains, 0.5, transforms; coordinate, chain)

# A discretization is a single path, so with several chains `chain` must be given.
function _discretize_chain(chains::PDMPChains, chain::Union{Nothing,Integer})
    chain === nothing && n_chains(chains) > 1 && throw(ArgumentError(
        "chains has $(n_chains(chains)) chains; pass `chain=i` to discretize one of them"))
    return _single(chains, chain)
end

PDMPDiscretize(chains::PDMPChains, dt; chain::Union{Nothing,Integer}=nothing) =
    PDMPDiscretize(_discretize_chain(chains, chain), dt)

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

function adaptive_dt(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing, kwargs...)
    adaptive_dt(_discretize_chain(chains, chain); kwargs...)
end

"""
    adaptive_discretize(trace::AbstractPDMPTrace; kwargs...)
    adaptive_discretize(chains::PDMPChains; chain=nothing, kwargs...)

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

function adaptive_discretize(chains::PDMPChains; chain::Union{Nothing,Integer}=nothing, kwargs...)
    adaptive_discretize(_discretize_chain(chains, chain); kwargs...)
end

function Base.show(io::IO, chains::PDMPChains)
    nc = n_chains(chains)
    n_events = [length(chains.traces[i]) for i in 1:nc]
    print(io, "PDMPChains with $nc chain$(isone(nc) ? "" : "s") ($(join(n_events, ", ")) events)")

    total_grid_resets = sum(_get_counter_grid_resets_from_dynamics_adaptation, chains.stats)
    total_grid_resets > 0 && print(io, " | grid_resets=", total_grid_resets)
end
