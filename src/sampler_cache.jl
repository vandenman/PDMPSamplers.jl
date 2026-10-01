function add_gradient_to_cache(cache::NamedTuple, ξ::SkeletonPoint)
    if haskey(cache, :∇ϕx)
        if !(cache.∇ϕx isa typeof(ξ.x) && length(cache.∇ϕx) == length(ξ.x))
            throw(ArgumentError("cache.∇ϕx was given manually, but must be of the same type as ξ.x"))
        end
    else
        ∇ϕx = similar(ξ.x)
        cache = merge(cache, (; ∇ϕx))
    end
    # Keep the event metadata alongside its reusable gradient buffer.  The
    # sampler consumes it before another gradient can overwrite that buffer.
    if cache.∇ϕx isa Vector{Float64} && !haskey(cache, :gradient_meta)
        cache = merge(cache, (; gradient_meta=GradientMeta(cache.∇ϕx)))
    end
    return cache
end

@inline _gradient_meta(cache::NamedTuple) = haskey(cache, :gradient_meta) ?
    cache.gradient_meta : GradientMeta(cache.∇ϕx)

function initialize_cache(::Random.AbstractRNG, ::ContinuousDynamics, ::GradientStrategy, ::PoissonTimeStrategy, ::Real, ::SkeletonPoint)
    (;)
end

initialize_cache(flow::ContinuousDynamics, grad::GradientStrategy, alg::PoissonTimeStrategy, t::Real, ξ::SkeletonPoint) =
    initialize_cache(Random.default_rng(), flow, grad, alg, t, ξ)
