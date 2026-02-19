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

mutable struct IntegralCache
    τL::Float64; IL::Float64   # left bracket
    τR::Float64; IR::Float64   # right bracket
    τC::Float64; IC::Float64   # last evaluated
end
function integral_minus_R_factory2(R, state, grad, flow, cache, λ; rtol=1e-6, atol=1e-9)
    ic = IntegralCache(0.0, 0.0, Inf, Inf, Inf, Inf)  # only left at (0,0) initially
    state_s = copy(state)

    # Define rate function outside to avoid boxing
    ratefun = let state = state, state_s = state_s, grad = grad, flow = flow, cache = cache, λ = λ
        function (s)
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
    end

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

    memoize_dict = Dict{Float64,Float64}()
    function f_memoized(τ::Real)
        # haskey(memoize_dict, τ) && @show "Cache hit for τ = $τ"
        get!(memoize_dict, τ) do
            f(τ)
        end
    end

    return f_memoized
end

"""
    next_event_time(grad, flow, alg::RootsPoissonTimeStrategy, state, cache, stats)

Compute next event time using root finding. Samples R ~ Exp(1) and solves
∫₀^τ λ(x(s), v) ds = R for τ using numerical integration (QuadGK) and
root finding (Roots.jl).
"""
function next_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics,
                        alg::RootsPoissonTimeStrategy, state::AbstractPDMPState,
                        cache, stats::StatisticCounter)

    grad = model.grad
    # Compare with refresh time --  TODO: we can always use this as an upper bound for the root finding?
    τ_refresh = rand_refresh_time(flow)

    mustwork = isinf(τ_refresh)
    # if refresh time is Inf, this MUST work

    outer_iterations = 0
    max_outer_iterations = 10
    while outer_iterations < max_outer_iterations

        # Sample the exponential random variable
        R = rand(Exponential())

        integral_minus_R = integral_minus_R_factory2(R, state, grad, flow, cache, λ; rtol=alg.rtol, atol=alg.atol)

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

            mustwork && continue
            τ_event = τ_refresh + 1

        elseif iterations >= max_iterations

            @warn "RootsPoissonTimeStrategy: Maximum bracketing iterations reached"
            mustwork && continue
            τ_event = τ_refresh + 1

        else

            if sign(f_lower) == sign(f_upper)
                @show τ_lower, τ_upper, f_lower, f_upper, mustwork, iterations_constant, max_iterations_constant, iterations, max_iterations
                error("RootsPoissonTimeStrategy: Root not bracketed in [$(τ_lower), $(τ_upper)] with f_lower = $f_lower, f_upper = $f_upper")
            end

            # Now [τ_lower, τ_upper] brackets the root
            τ_event = try
                Roots.find_zero((τ->integral_minus_R(τ)), (τ_lower, τ_upper), atol = alg.atol, rtol = alg.rtol)
            catch e
                @warn "RootsPoissonTimeStrategy: Root finding failed, using fallback" exception=e
                τ_upper
            end
        end

        if τ_event < τ_refresh
            return τ_event, :reflect, EmptyMeta()
        else
            return τ_refresh, :refresh, EmptyMeta()
        end
    end

    error("RootsPoissonTimeStrategy: Failed to find event time after $max_outer_iterations attempts.")

end
