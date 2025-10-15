"""
    RootsPoissonTimeStrategy <: PoissonTimeStrategy

A Poisson time strategy that computes the next event time using root finding.
Samples R ~ Exp(1) and solves ∫₀^τ λ(x(s), v) ds = R for τ using numerical integration and root finding.

# Fields
- `τ_initial::Float64`: Initial bracket size for root finding (default: 1e-3)
- `bracket_multiplier::Float64`: Geometric growth factor for bracketing (default: 2.0)
- `rtol::Float64`: Relative tolerance for quadrature (default: 1e-8)
- `atol::Float64`: Absolute tolerance for quadrature (default: 1e-12)
"""
Base.@kwdef struct RootsPoissonTimeStrategy <: PoissonTimeStrategy
    τ_initial::Float64 = 1e-3
    bracket_multiplier::Float64 = 5.0
    rtol::Float64 = sqrt(eps(Float64))#1e-8
    atol::Float64 = sqrt(eps(Float64))#1e-12
end

mutable struct CachedIntegral
    τ_last::Float64
    I_last::Float64
end

"""
    integral_minus_R_factory(R, state, grad, flow, cache, λ; rtol, atol)

Returns a closure `f(τ)` that computes ∫₀^τ λ(x(s), v) ds - R,
reusing cached integrals between evaluations.
"""
function integral_minus_R_factory(R, state, grad, flow, cache, λ; rtol=1e-6, atol=1e-9)
    # Initialize cache
    cache_int = CachedIntegral(0.0, 0.0)

    function f(τ::Real)
        τ <= 0 && return -R

        # If τ < τ_last, restart cache (Brent may shrink interval)
        if τ < cache_int.τ_last
            cache_int.τ_last = 0.0
            cache_int.I_last = 0.0
        end

        # Compute incremental integral only
        inc, _ = QuadGK.quadgk(s -> begin
            state_s = move_forward_time(state, s, flow)
            ∇ϕ = compute_gradient!(state_s, grad, flow, cache)
            λ(state_s.ξ, ∇ϕ, flow)
        end, cache_int.τ_last, τ; rtol=rtol, atol=atol)

        # Update cache
        cache_int.I_last += inc
        cache_int.τ_last = τ

        return cache_int.I_last - R
    end

    return f
end


# using DataStructures: SortedDict

# mutable struct MultiCacheIntegral
#     cache::SortedDict{Float64,Float64}  # maps τ -> I(τ)
#     maxsize::Int
# end

mutable struct IntegralCache
    τL::Float64; IL::Float64   # left bracket
    τR::Float64; IR::Float64   # right bracket
    τC::Float64; IC::Float64   # last evaluated
end
function integral_minus_R_factory2(R, state, grad, flow, cache, λ; rtol=1e-6, atol=1e-9)
    ic = IntegralCache(0.0, 0.0, Inf, Inf, Inf, Inf)  # only left at (0,0) initially

    function f(τ::Real)

        τ <= 0 && return -R
        # Shortcuts if τ already cached (should never happen though?)
        τ == ic.τC && return ic.IC - R
        τ == ic.τL && return ic.IL - R
        τ == ic.τR && return ic.IR - R

        # Decide direction relative to last evaluation
        Iτ = if isfinite(ic.τC)
            if τ > ic.τC
                inc, _ = QuadGK.quadgk(ratefun, ic.τC, τ; rtol=rtol, atol=atol)
                ic.IC + inc
            else
                dec, _ = QuadGK.quadgk(ratefun, τ, ic.τC; rtol=rtol, atol=atol)
                ic.IC - dec
            end
        else
            inc, _ = QuadGK.quadgk(ratefun, 0.0, τ; rtol=rtol, atol=atol)
            inc
        end

        # Update cache
        if Iτ < R
            ic.τL, ic.IL = τ, Iτ
        else
            ic.τR, ic.IR = τ, Iτ
        end
        ic.τC, ic.IC = τ, Iτ

        return Iτ - R
    end

    state_s = copy(state)
    # inline rate function with gradient call
    function ratefun(s)
        # reset temporary
        state_s.t[] = state.t[]
        copyto!(state_s.ξ, state.ξ)
        move_forward_time!(state_s, s, flow)
        ∇ϕ = compute_gradient!(state_s, grad, flow, cache)
        # if any(isnan, ∇ϕ)
        #     @show state_s.t[], s, state.ξ.x, state.ξ.θ
        #     @assert any(isnan, ∇ϕ) "problem at state [$(state_s.t[])] "
        # end
        # @assert !any(isnan, state_s.ξ.x)
        # @assert !any(isnan, state_s.ξ.θ)
        result = λ(state_s.ξ, ∇ϕ, flow)
        # @assert !isnan(result)
        return result
    end

    memoize_dict = Dict{Float64,Float64}()
    function f_memoized(τ::Real)
        # haskey(memoize_dict, τ) && @show "Cache hit for τ = $τ"
        get!(memoize_dict, τ) do
            f(τ)
        end
    end

    return f_memoized # f
end

using DataStructures # Make sure you have this package added

# A stateful, callable struct using SortedDict for optimal performance
mutable struct ResumableIntegral
    # A SortedDict maps time points (keys) to accumulated integral values (values)
    # It guarantees O(log n) for lookups and insertions.
    cache::SortedDict{Float64, Float64}

    # The actual rate function to be integrated
    rate_function::Function

    # Tolerances for the quadrature
    rtol::Float64
    atol::Float64

    function ResumableIntegral(rate_function; rtol, atol)
        # Initialize with the integral at t=0, which is 0.
        new(SortedDict(0.0 => 0.0), rate_function, rtol, atol)
    end
end

# The callable method for our struct
function (res_integral::ResumableIntegral)(τ::Real)

    τ <= 0 && return -first(values(res_integral.cache))

    # searchsortedlast equivalent for SortedDict to find the predecessor.
    # We get a "token" (an iterator/pointer) to the first element >= τ.
    token = searchsortedlast(res_integral.cache, τ)

    @show token, res_integral.cache, τ

    # If the token is at the very beginning, our predecessor is the first element.
    # However, we must handle the edge case where τ is exactly the first key.
    if token == startof(res_integral.cache)
        τ_prev, I_prev = deref((res_integral.cache, token))
        # @show τ_prev, I_prev
        # If it's an exact hit on the first element, we are done.
        # Otherwise, our predecessor is the element at T=0.
        if τ != τ_prev
             τ_prev, I_prev = 0.0, res_integral.cache[0.0]
        end
    else
        # Otherwise, the predecessor is the element just before the token.
        # We also check for an exact hit, which `searchsortedfirst` would have found.
        τ_at_token, _ = deref((res_integral.cache, token))
        if τ == τ_at_token
             τ_prev, I_prev = τ_at_token, res_integral.cache[τ_at_token]
        else
             τ_prev, I_prev = deref((res_integral.cache, regress((res_integral.cache, token))))
        end
        # @show τ_prev, I_prev
    end

    # I_prev > 0.1 && error("stop for now")
    # If we have an exact cache hit, return immediately
    if τ_prev == τ
        return I_prev
    end

    # Otherwise, integrate the missing segment from the closest known point
    integral_increment, _ = QuadGK.quadgk(
        res_integral.rate_function,
        τ_prev,
        τ;
        rtol=res_integral.rtol,
        atol=res_integral.atol
    )

    # The new total integral
    I_new = I_prev + integral_increment

    # Update the cache. This is an efficient O(log n) operation.
    res_integral.cache[τ] = I_new

    return I_new
end

# The factory function remains identical, it just constructs the new struct
function integral_minus_R_factory3(R, state, grad, flow, cache, λ; rtol, atol)
    state_s = copy(state)
    ratefun = function(s)
        # ... (same as before)
        state_s.t[] = state.t[]
        copyto!(state_s.ξ, state.ξ)
        move_forward_time!(state_s, s, flow)
        ∇ϕ = compute_gradient!(state_s, grad, flow, cache)
        return λ(state_s.ξ, ∇ϕ, flow)
    end

    resumable_integral = ResumableIntegral(ratefun; rtol=rtol, atol=atol)
    return τ -> resumable_integral(τ) - R
end

# The ResumableIntegral struct remains the same, it's already correct.
mutable struct ResumableIntegral2{F<:Function}
    cache::SortedDict{Float64, Float64}
    rate_function::F
    rtol::Float64
    atol::Float64

    function ResumableIntegral2(rate_function; rtol, atol)
        new{typeof(rate_function)}(SortedDict(0.0 => 0.0), rate_function, rtol, atol)
    end
end

function (res_integral::ResumableIntegral2)(τ::Real)
    # ... This function is identical to the previous SortedDict version ...
    τ <= 0 && return -first(values(res_integral.cache))

    token = searchsortedlast(res_integral.cache, τ)

    if token == startof(res_integral.cache)
        τ_prev, I_prev = deref((res_integral.cache, token))
        if τ != τ_prev
             τ_prev, I_prev = 0.0, res_integral.cache[0.0]
        end
    else
        τ_at_token, _ = deref((res_integral.cache, token))
        if τ == τ_at_token
             τ_prev, I_prev = τ_at_token, res_integral.cache[τ_at_token]
        else
            #  τ_prev, I_prev = deref((res_integral.cache, prev(token)))
             τ_prev, I_prev = deref((res_integral.cache, regress((res_integral.cache, token))))
        end
    end
    if τ_prev == τ
        return I_prev
    end

    integral_increment, _ = QuadGK.quadgk(
        res_integral.rate_function,
        τ_prev, τ;
        rtol=res_integral.rtol,
        atol=res_integral.atol
    )
    I_new = I_prev + integral_increment
    res_integral.cache[τ] = I_new
    return I_new
end


# The factory is where we now add the second, low-level cache.
function integral_minus_R_factory4(R, state, grad, flow, cache, λ; rtol, atol)

    # --- LOW-LEVEL CACHE ---
    # This dictionary will store the results of ratefun(s)
    memoization_cache = Dict{Float64, Float64}()

    # The original, expensive rate function
    state_s = copy(state)
    _ratefun = function(s::Float64)
        state_s.t[] = state.t[]
        copyto!(state_s.ξ, state.ξ)
        move_forward_time!(state_s, s, flow)
        ∇ϕ = compute_gradient!(state_s, grad, flow, cache)
        return λ(state_s.ξ, ∇ϕ, flow)
    end

    # A memoized wrapper around the expensive rate function
    memoized_ratefun = function(s::Float64)
        # get! is a concise way to check, retrieve, or compute and store.
        get!(memoization_cache, s) do
            _ratefun(s)
        end
    end

    # --- HIGH-LEVEL CACHE ---
    # Instantiate our resumable integral calculator, but pass it the
    # CHEAP, MEMOIZED rate function.
    resumable_integral = ResumableIntegral2(memoized_ratefun; rtol=rtol, atol=atol)

    # Return the final function for the root-finder
    return τ -> resumable_integral(τ) - R
end

"""
    next_event_time(grad, flow, alg::RootsPoissonTimeStrategy, state, cache, stats)

Compute next event time using root finding. Samples R ~ Exp(1) and solves
∫₀^τ λ(x(s), v) ds = R for τ using numerical integration (QuadGK) and
root finding (Roots.jl).
"""
function next_event_time(grad::GlobalGradientStrategy, flow::ContinuousDynamics,
                        alg::RootsPoissonTimeStrategy, state::AbstractPDMPState,
                        cache, stats::StatisticCounter)

    # Compare with refresh time --  TODO: we can always use this as an upper bound for the root finding?
    τ_refresh = rand_refresh_time(flow)

    mustwork = isinf(τ_refresh)
    # if refresh time is Inf, this MUST work

    outer_iterations = 0
    max_outer_iterations = 10
    while outer_iterations < max_outer_iterations

        # Sample the exponential random variable
        R = rand(Exponential())

        # integral_minus_R = integral_minus_R_factory(R, state, grad, flow, cache, λ; rtol=alg.rtol, atol=alg.atol)
        integral_minus_R = integral_minus_R_factory2(R, state, grad, flow, cache, λ; rtol=alg.rtol, atol=alg.atol)
        # integral_minus_R = integral_minus_R_factory3(R, state, grad, flow, cache, λ; rtol=alg.rtol, atol=alg.atol)
        # integral_minus_R = integral_minus_R_factory4(R, state, grad, flow, cache, λ; rtol=alg.rtol, atol=alg.atol)

        τ_lower = zero(alg.τ_initial)
        τ_upper = min(τ_refresh, alg.τ_initial)

        # Start with function at lower & upper bounds
        f_lower = integral_minus_R(τ_lower)
        f_upper = integral_minus_R(τ_upper)

        # Expand until sign flips
        iterations = 0
        max_iterations = 50
        iterations_constant = 0
        max_iterations_constant = 10
        is_constant = isapprox(f_lower, f_upper, atol = alg.atol, rtol = alg.rtol)
        while f_upper < 0 && iterations < max_iterations
            τ_lower, f_lower = τ_upper, f_upper    # shift bracket
            τ_upper *= alg.bracket_multiplier
            τ_upper = min(τ_upper, τ_refresh)      # do not exceed refresh time
            f_upper = integral_minus_R(τ_upper)
            iterations += 1

            # @show τ_lower, τ_upper, f_lower, f_upper, iterations_constant
            if is_constant
                iterations_constant += 1
                is_constant = isapprox(f_lower, f_upper, atol = alg.atol, rtol = alg.rtol)
                iterations_constant > max_iterations_constant && break
            end
            if τ_upper == τ_refresh
                # no use to continue?
                break
            end
        end

        # TODO: not entirely sure how to handle this situation...
        # Inf is not always good idea
        # for the ZigZag, refresh time is also Inf, and then things break!
        # so basically, for the ZigZag this MUST work!
        if iterations_constant > max_iterations_constant

            # Infiltrator.@infiltrate flow isa ZigZag

            # for the ZigZag, this must work, so we simply try again
            # for other processes, we sample a refreshment
            mustwork && continue
            τ_event = τ_refresh + 1

        elseif iterations >= max_iterations

            # Infiltrator.@infiltrate flow isa ZigZag

            @warn "RootsPoissonTimeStrategy: Maximum bracketing iterations reached"
            mustwork && continue
            τ_event = τ_refresh + 1

        # elseif τ_upper == τ_refresh && isfinite(τ_refresh) && f_upper < 0
        #     # No root possible found before refresh time
        #     τ_event =  τ_refresh + 1
        else

            if sign(f_lower) == sign(f_upper)
                @show τ_lower, τ_upper, f_lower, f_upper, mustwork, iterations_constant, max_iterations_constant, iterations, max_iterations
                error("RootsPoissonTimeStrategy: Root not bracketed in [$(τ_lower), $(τ_upper)] with f_lower = $f_lower, f_upper = $f_upper")
            end

            # Now [τ_lower, τ_upper] brackets the root
            τ_event = try
                # Roots.find_zero((τ->integral_minus_R(τ)), (τ_lower, τ_upper), Roots.Brent())
                # Roots.find_zero((τ->integral_minus_R(τ)), (τ_lower, τ_upper), Roots.Order0(), atol = alg.atol, rtol = alg.rtol)
                # Order0 doesn't work well when we pass an upper and lower bound....
                # another way would be to transform the domain?
                # could also avoid all the stuff with finding the bounds above, but we sort of need this for
                # to figure out if/ when a function is constant...
                Roots.find_zero((τ->integral_minus_R(τ)), (τ_lower, τ_upper), atol = alg.atol, rtol = alg.rtol)
                # Roots.find_zero(τ->begin
                #     t = exp(τ)
                #     r = integral_minus_R(t)
                #     # @show τ, t, r
                #     return r
                # end, 2.0, Roots.Order0())
            catch e
                # @show τ_lower, τ_upper, f_lower, f_upper
                @warn "RootsPoissonTimeStrategy: Root finding failed, using fallback" exception=e
                τ_upper

                # Infiltrator.@infiltrate flow isa ZigZag

            end
            # τ_event = exp(τ_event)
        end

        if τ_event < τ_refresh
            return τ_event, :reflect, nothing
        else
            return τ_refresh, :refresh, nothing
        end
    end

    error("RootsPoissonTimeStrategy: Failed to find event time after $max_outer_iterations attempts.")

end
