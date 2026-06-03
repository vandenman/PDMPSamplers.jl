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

function initialize_cache(::Random.AbstractRNG, ::ContinuousDynamics, ::GradientStrategy, ::PoissonTimeStrategy, ::Real, ::SkeletonPoint)
    (;)
end

initialize_cache(flow::ContinuousDynamics, grad::GradientStrategy, alg::PoissonTimeStrategy, t::Real, ξ::SkeletonPoint) =
    initialize_cache(Random.default_rng(), flow, grad, alg, t, ξ)
