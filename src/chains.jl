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

function Base.show(io::IO, chains::PDMPChains)
    nc = n_chains(chains)
    n_events = [length(chains.traces[i]) for i in 1:nc]
    print(io, "PDMPChains with $nc chain$(nc == 1 ? "" : "s") ($(join(n_events, ", ")) events)")
end
