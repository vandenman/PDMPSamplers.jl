
"""
A structure to hold the piecewise constant upper bound.
Contains the grid points and the constant rate values on each segment.
"""
struct PiecewiseConstantBound{T<:Real}
    t_grid::Vector{T} # Grid points [t_0, t_1, ..., t_N]
    Λ_vals::Vector{T} # Bound values [Λ_0, Λ_1, ..., Λ_{N-1}] on each interval
    y_vals::Vector{T} # y values [y_0, y_1, ..., y_{N-1}] on each interval
    d_vals::Vector{T} # d values [d_0, d_1, ..., d_{N-1}] on each interval
end
function PiecewiseConstantBound(t_grid::AbstractVector, Λ_vals::AbstractVector)
    return PiecewiseConstantBound(collect(t_grid), collect(Λ_vals), similar(Λ_vals, length(Λ_vals) + 1), similar(Λ_vals, length(Λ_vals) + 1))
end

"""
    (bound::PiecewiseConstantBound)(t::Real)

Functor to evaluate the piecewise constant bound Λ(t) at a given time t.
"""
function (bound::PiecewiseConstantBound)(t::Real)
    # Check if t is outside the horizon
    if t < bound.t_grid[1] || t >= bound.t_grid[end]
        return 0.0 # Or handle as an error
    end

    # Find which segment t falls into. `searchsortedlast` is efficient for this.
    i = searchsortedlast(bound.t_grid, t)
    return bound.Λ_vals[i]
end

"""
    construct_upper_bound(x₀, v₀, flow, U, t_max, N)

Constructs a piecewise-constant upper bound for the event rate λ(t)
using the grid-based method from Andral & Kamatani (2024).

# Arguments
- `x₀`, `v₀`: Initial position and velocity.
- `flow`: The deterministic dynamics (e.g., BouncyParticle, Boomerang).
- `U`: The potential energy function `U(x)`.
- `t_max`: The time horizon for the bound.
- `N`: The number of grid segments.

# Returns
- A `PiecewiseConstantBound` object.
"""
function construct_upper_bound(ξ::SkeletonPoint, flow, ∇U!::Function, t_max::Real, N::Int)

    pcb = PiecewiseConstantBound{eltype(ξ.x)}(Vector{eltype(ξ.x)}(undef, N + 1), Vector{eltype(ξ.x)}(undef, N))
    recompute_time_grid!(pcb, t_max, N)
    construct_upper_bound!(pcb, ξ, flow, ∇U!)
    return pcb
end

recompute_time_grid!(pcb::PiecewiseConstantBound, t_max::Real, N::Integer) = pcb.t_grid .= range(0.0, t_max, N + 1)

# Extract the gradient function for convenience
# TODO: these are different for Turing models!
# function make_grad_U_func(θ::AbstractVector, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
#     return function(out, x)
#         compute_gradient!(x, θ, gradient_strategy, flow, cache)
#     end
# end

# function make_grad_and_hess_func(θ::AbstractVecOrMat, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache, prep, g, jac, backend)
#     f = make_grad_U_func(θ, flow, gradient_strategy, cache)
#     grad_and_hess(x) = DI.value_and_jacobian!(f, g, jac, prep, backend, x)
#     return grad_and_hess
# end
# function make_grad_and_hess_func(θ::AbstractVecOrMat, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, alg::GridAdaptiveState, cache)
#     make_grad_and_hess_func(θ, flow, gradient_strategy, cache, alg.prep, alg.g, alg.jac, alg.backend)
# end

# function make_grad_and_hvp_func(θ::AbstractVecOrMat, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache, prep, g::AbstractVector, backend)
#     f  = make_grad_U_func(θ, flow, gradient_strategy, cache)
#     f1 = make_hvp_func(flow, gradient_strategy, cache)
#     hvp(x, v) = DI.gradient!(f1, g, prep, backend, x, DI.Constant(v))
#     return f, hvp
# end

# function make_grad_and_hvp_func(θ::AbstractVecOrMat, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, alg::GridAdaptiveState, cache)
#     make_grad_and_hvp_func(θ, flow, gradient_strategy, cache, alg.prep, alg.g, alg.backend)
# end

function make_grad_U_func(θ::AbstractVector, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return function (x)
        compute_gradient!(x, θ, gradient_strategy, flow, cache)
        # compute_gradient_uncorrected!(x, θ, gradient_strategy, flow, cache)
    end
end
function make_grad_U_func(state::AbstractPDMPState, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return make_grad_U_func(state.ξ.θ, flow, gradient_strategy, cache)
end

function make_hvp_func(flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return function (x, θ)
        dot(compute_gradient!(x, θ, gradient_strategy, flow, cache), θ)
        # dot(compute_gradient_uncorrected!(x, θ, gradient_strategy, flow, cache), θ)
    end
end
function make_hvp_func(::AbstractPDMPState, flow::ContinuousDynamics, gradient_strategy::GradientStrategy, cache)
    return make_hvp_func(flow, gradient_strategy, cache)
end



function construct_upper_bound_grad_and_hess!(pcb::PiecewiseConstantBound, state::AbstractPDMPState, flow::ContinuousDynamics,
    grad_and_hess_or_grad_and_hvp::Union{Function,NTuple{2,Function}}, add_rate::Bool=true)

    t_grid = pcb.t_grid
    Λ_vals = pcb.Λ_vals
    N = length(Λ_vals)
    y_vals = pcb.y_vals # similar(Λ_vals, N + 1)
    d_vals = pcb.d_vals # similar(y_vals, N + 1)

    #NOTE: asumes t_grid is sorted!
    state_t = copy(state)
    @assert iszero(t_grid[1])
    y_vals[1], d_vals[1] = get_rate_and_deriv(state_t, flow, grad_and_hess_or_grad_and_hvp, add_rate)
    for i in 2:N+1
        Δt = t_grid[i] - t_grid[i-1]
        move_forward_time!(state_t, Δt, flow) # Move the state forward by Δt
        validate_state(state_t, flow, "after moving forward in time in Grid algorithm")
        y_vals[i], d_vals[i] = get_rate_and_deriv(state_t, flow, grad_and_hess_or_grad_and_hvp, add_rate)
    end

    for i in 1:N
        tᵢ, tᵢ₊₁ = t_grid[i], t_grid[i+1]
        yᵢ, yᵢ₊₁ = y_vals[i], y_vals[i+1]
        dᵢ, dᵢ₊₁ = d_vals[i], d_vals[i+1]

        # Intersection of the tangents
        if abs(dᵢ - dᵢ₊₁) < 1e-9 # Handle parallel lines # TODO: tolerance should be sqrt(eps(T))

            mᵢ = yᵢ
        else
            # See paper for formula (transcribed from their Algorithm 2)
            x_intersect = (yᵢ₊₁ - yᵢ + dᵢ * tᵢ - dᵢ₊₁ * tᵢ₊₁) / (dᵢ - dᵢ₊₁)

            # Clip the intersection point to the segment
            x_clipped = clamp(x_intersect, tᵢ, tᵢ₊₁)

            # Height of the tangent line at the (clipped) intersection point
            mᵢ = dᵢ * x_clipped + yᵢ - dᵢ * tᵢ
        end

        Λ_vals[i] = max(yᵢ, yᵢ₊₁, mᵢ)
    end
end

function construct_upper_bound!(pcb::PiecewiseConstantBound, ξ::SkeletonPoint, flow::ContinuousDynamics, ∇U!::Function, use_hvp::Bool=true)
    construct_upper_bound!(pcb, PDMPState(0.0, ξ), flow, ∇U!, use_hvp)
end
function construct_upper_bound!(pcb::PiecewiseConstantBound, state::PDMPState, flow::ContinuousDynamics, ∇U!::Function,
    use_hvp::Bool=true)

    @assert use_hvp "Only HVP mode is supported"

    g = similar(state.ξ.x)
    out = similar(g)
    ∇U = Base.Fix1(∇U!, out)
    f = (x, v) -> dot(∇U(x), v)
    prep = DI.prepare_gradient(f, DI.AutoMooncake(), g, DI.Constant(copy(g)))
    
    # Avoid boxing prep by wrapping it in a struct or passing it explicitly
    hvp = let prep = prep, g = g, f = f
        (x, v) -> DI.gradient!(f, g, prep, DI.AutoMooncake(), x, DI.Constant(v))
    end

    return construct_upper_bound_grad_and_hess!(pcb, state, flow, (∇U, hvp))
end

# helper functions for λ(t) and λ'(t) ---
function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, (grad, hvp)::NTuple{2,Function}, add_rate::Bool=true)

    xt, vt = state.ξ.x, state.ξ.θ  # state already moved to time t

    ∇U_xt = grad(xt)
    Hxt_vt = hvp(xt, vt)  # Hessian-vector product

    # base rate (before positive-part)
    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)

    # @show ∇U_xt,  Hxt_vt, f_t

    # derivative of base rate
    # TODO: perhap define ∂λ∂t(state, ∇U_xt, ???, flow)
    if flow isa ZigZag
        f_prime_t = zero(eltype(vt))
        for i in eachindex(∇U_xt)
            if ispositive(vt[i] * ∇U_xt[i])
                f_prime_t += vt[i] * Hxt_vt[i]
            end
        end
    else
        f_prime_t = dot(vt, Hxt_vt)
    end

    if flow isa Boomerang

        # 1. Subtract the reference Hessian part from f_prime_t
        # H_Φ*v = Γ*v for the reference potential
        # @show f_prime_t, dot(vt, flow.Γ, vt)
        f_prime_t -= dot(vt, flow.Γ, vt)

        # 2. Add the velocity drift part
        # dv/dt = -∇Φ = Γ(μ - xt)
        vdot = flow.Γ * (flow.μ - xt)

        # We use `∇U_xt` directly because it IS the corrected gradient.
        # @show f_prime_t, vdot, dot(vdot, ∇U_xt)
        f_prime_t += dot(vdot, ∇U_xt)

    elseif !(flow isa ZigZag || flow isa BouncyParticle || flow isa PreconditionedDynamics{<:Any,<:Union{BouncyParticle,ZigZag}})
        throw(ArgumentError("Unsupported flow type!"))
    end

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv

end

function propose_event_time(pcb::PiecewiseConstantBound, u::Real=rand(Exponential()), refresh_rate::Real=0.0)

    area_before = zero(eltype(pcb.Λ_vals))
    segment_idx = 0
    integral = zero(eltype(pcb.Λ_vals))
    for i in eachindex(pcb.Λ_vals)
        integral += (pcb.Λ_vals[i] + refresh_rate) * (pcb.t_grid[i+1] - pcb.t_grid[i])
        if integral >= u
            segment_idx = i
            break
        end
        area_before = integral
    end

    if iszero(segment_idx)
        return (Inf, 0.0)
    end

    # Get the properties of this segment
    t_start = pcb.t_grid[segment_idx]
    Λ_val = pcb.Λ_vals[segment_idx]

    # This is how much of the random draw `u` we need to "spend" inside this segment
    u_remaining = u - area_before

    # Calculate the time into the segment: time = distance / speed
    time_in_segment = u_remaining / Λ_val

    τ_proposed = t_start + time_in_segment

    return τ_proposed, Λ_val

end


# Equidistant for now.
# TODO: there two possible algorithmic optimizations
# 1. non-equidistant grid points/ adaptive grid ala https://github.com/sschuldenzucker/ParametricAdaptiveSampling.jl
# 2. use Rational for α⁺ and α⁻. E.g,. when α⁻ = 1 // 2, we can reuse 50% of the previous grid evaluations. Assumes equidistant points though
Base.@kwdef struct GridThinningStrategy <: PoissonTimeStrategy
    N::Int = 30  # Number of grid points
    t_max::Float64 = 2.0 # Initial horizon
    α⁺::Float64 = 1.5 # Factor to increase t_max
    α⁻::Float64 = 0.5 # Factor to decrease t_max
    safety_limit::Int = 500
    # adtype::T = DI.AutoMooncake() # Automatic differentiation type
    # hvp::U = nothing # HVP function, if available
end

function _to_internal(strat::GridThinningStrategy, flow::ContinuousDynamics, model::PDMPModel, state::AbstractPDMPState, cache, stats::StatisticCounter)

    # t = state.t[]
    # ξ = state.ξ
    # g = similar(ξ.x)

    # if isnothing(strat.hvp)
    #     f! = make_hvp_func(state, flow, gradient_strategy, cache)
    #     prep = DI.prepare_gradient(f!, strat.adtype, ξ.x, DI.Constant(ξ.θ))
    #     hvp = (x, θ) -> begin
    #         stats.∇²f_calls += 1
    #         DI.gradient!(f!, g, prep, strat.adtype, x, DI.Constant(θ))
    #     end
    # else
    #     # if state isa StickyPDMPState
    #     #     θc = similar(ξ.θ)
    #     #     hvp = (x, θ, free) -> begin
    #     #         stats.∇²f_calls += 1
    #     #         copyto!(θc, θ)
    #     #         for i in eachindex(free)
    #     #             if !free[i]
    #     #                 θc[i] = zero(eltype(θc)) # Set frozen coordinates to zero
    #     #             end
    #     #         end
    #     #         strat.hvp(g, x, θc)
    #     #         for i in eachindex(free)
    #     #             if !free[i]
    #     #                 g[i] = zero(eltype(g))
    #     #             end
    #     #         end
    #     #         return g
    #     #     end
    #     # else
    #     hvp = (x, θ) -> begin
    #         stats.∇²f_calls += 1
    #         strat.hvp(g, x, θ)
    #     end
    #     # end
    # end

    T = typeof(strat.t_max)

    GridAdaptiveState(
        PiecewiseConstantBound(collect(range(0.0, strat.t_max, strat.N + 1)), zeros(T, strat.N)),
        Base.RefValue{Int}(strat.N),
        Base.RefValue{Float64}(strat.t_max),
        strat.α⁺,
        strat.α⁻,
        strat.safety_limit
        # hvp
    )

end

# internal state
struct GridAdaptiveState <: PoissonTimeStrategy # should technically substype something else but oh well
    pcb::PiecewiseConstantBound{Float64}
    N::Base.RefValue{Int}
    t_max::Base.RefValue{Float64}
    α⁺::Float64             # Factor to increase t_max
    α⁻::Float64             # Factor to decrease t_max
    # g::Vector{Float64}      # Gradient vector
    # jac::Matrix{Float64}    # Jacobian matrix
    # prep::T                 # AD prep
    # backend::U              # AD backend
    safety_limit::Int
    # hvp::T

end

recompute_time_grid!(alg::GridAdaptiveState) = recompute_time_grid!(alg.pcb, alg.t_max[], alg.N[])


function next_event_time(model::PDMPModel{<:GlobalGradientStrategy}, flow::ContinuousDynamics, alg::GridAdaptiveState, state::AbstractPDMPState, cache, stats::StatisticCounter,
    max_horizon::Float64=Inf, include_refresh::Bool=true)

    # TODO: need to rethink the logic once more
    # ghost events also allow for separating the logic w.r.t. other events
    # what if we keep this function as is, but add one layer around it that handles the different types of events?
    # then we remove the refresh time from this function (and the ab function as well)
    # and then we check afterward which event occurs first.
    # however, whenever an event is rejected, we would also need to exit the function (right?) TODO: double check this once more!

    # then I can individually sample a time for each event and then move on
    # the version below does everything within the while loop, so no ghost events
    # however, for sticky events I think this means that we'd need to make those
    # a part of this as well, which I don't think is right?

    pcb = alg.pcb
    state_ = copy(state)
    state2_ = copy(state)
    total_time = 0.0

    # TODO: can't this be done easier?
    grad_func = make_grad_U_func(state_, flow, model.grad, cache)
    hvp_func = model.hvp
    grad_and_hvp = (grad_func, hvp_func)

    λ_refresh = include_refresh ? refresh_rate(flow) : zero(refresh_rate(flow))

    # Sample refresh time independently once -- does not work!
    # time_refresh = ispositive(λ_refresh) ? rand(Exponential(inv(λ_refresh))) : Inf

    # ensure the return type is type-stable
    default_return = (; ∇ϕx=similar(state.ξ.x, 0))

    # use the minimum of algorithm's horizon and provided max_horizon (e.g., sticky time)
    # effective_t_max = min(alg.t_max[], max_horizon)
    # if max_horizon < alg.t_max[]
    #     alg.t_max[] = max_horizon
    #     recompute_time_grid!(alg)
    #     effective_t_max = max_horizon
    # else
    #     effective_t_max = alg.t_max[]
    # end

    safety_limit = alg.safety_limit
    while safety_limit > 0

        # Sample refresh time independently every iteration -- does work!
        τ_refresh = ispositive(λ_refresh) ? rand(Exponential(inv(λ_refresh))) : Inf

        # Construct bounds for REFLECTION ONLY (no refresh rate)
        construct_upper_bound_grad_and_hess!(pcb, state_, flow, grad_and_hvp, false)  # ← false = no refresh

        # Propose reflection event time
        τ_reflection, lb_reflection = propose_event_time(pcb)

        # Infiltrator.@infiltrate isinf(τ_reflection)

        # @show τ_refresh, τ_reflection, lb_reflection
        # @show τ_reflection, τ_refresh, alg.t_max[]
        # Check if the proposal is beyond the horizon

        if τ_reflection >= alg.t_max[]
            # if τ_reflection >= effective_t_max

            if τ_refresh < alg.t_max[]
                # if τ_refresh < effective_t_max
                return τ_refresh + total_time, :refresh, default_return
            end


            # if isinf(max_horizon)
            t_max = alg.t_max[]
            # Cap t_max to prevent overflow to Inf
            new_t_max = alg.t_max[] * alg.α⁺
            alg.t_max[] = min(new_t_max, 1e10)  # Cap at 10 billion time units
            recompute_time_grid!(alg)
            # effective_t_max = alg.t_max[]
            # end

            return t_max, :horizon_hit, default_return
            # return effective_t_max, :horizon_hit, default_return

        end
        # @show time_refresh, τ_reflection
        # Return whichever event happens first
        if τ_refresh < τ_reflection
            return τ_refresh + total_time, :refresh, default_return
        end

        # Test reflection acceptance
        move_forward_time!(state_, τ_reflection, flow)
        ∇ϕx = compute_gradient!(state_, model.grad, flow, cache)

        l_reflection = λ(state_.ξ, ∇ϕx, flow)  # Only reflection rate, no refresh


        # if l_reflection <= 1e-4 && lb_reflection > .1
        #     @show l_reflection, lb_reflection
        #     # @info "Very small true rate but large bound detected. This may indicate numerical instability., Try increasing N.", l_reflection, lb_reflection, state_.ξ.x, state_.ξ.θ
        # end

        # TODO: this does happen for the Boomerang, and the plot shows something is indeed wrong... the left tail is way higher than it should be!

        # Infiltrator.@infiltrate l_reflection <= 1e-4 && lb_reflection > .1
        # this shows a large mismatch between the true rates and the grid approximation
        # alg2 = deepcopy(alg)
        # plot_rate(copy(state.ξ.x), copy(state.ξ.θ), flow, gradient_strategy.f, 3.0, alg2; dt=0.01, c = .1)

        # fig = Figure()
        # ax = Axis(fig[1, 1])




        # Accept/reject reflection
        # @show τ_reflection, lb_reflection, l_reflection
        if rand() * lb_reflection <= l_reflection
            # @show :reflect
            # @assert l_reflection <= lb_reflection "incorrect bounds, $(l_reflection), $(lb_reflection)"
            return τ_reflection + total_time, :reflect, (; ∇ϕx)
        else
            # Reflection rejected - shrink horizon and retry
            alg.t_max[] *= alg.α⁻
            # effective_t_max = min(alg.t_max[], max_horizon)
            recompute_time_grid!(alg)
            # reset time
            state_.t[] = state2_.t[]
            # reset position and velocity
            copyto!(state_.ξ, state2_.ξ)
            safety_limit -= 1
        end
    end

    error("Safety limit reached")
end