abstract type AbstractPoissonProcess end

struct HomogeneousPoissonProcess <: AbstractPoissonProcess
    rate::Float64
end

struct InhomogeneousPoissonProcess <: AbstractPoissonProcess
    rate_function::Function
end

struct RateFunction <: AbstractPoissonProcess
     # independent priors
     fixed_rate::Float64
     # model contribution/ dependent spike priors
     # could also be a vector of callables?
     # but we could also just have 2 here for now?
     rate_function::Vector{Function}
end

# There are three components to the overall rate function:
# 1. deterministic components
# 2. homogeneous components
# 3. inhomogeneous components

# to sample the next event time, we should
# 1. compute the deterministic event times
# 2. compute the next event time for the homogeneous components
# 3. compute the next event time for the inhomogeneous components
# 4. take the minimum of these

# now, since the deterministic event times are deterministic they only need to be computed once,
# and can also inform the inhomogeneous components (e.g., t_max).
# The homogeneous components can be sampled using exponential distributions.
# however, these may need to be resampled (e.g., when gridthinning fails/ rejects).

# so a tricky bit is how to handle these arguments.
# perhaps this should only receive functions of time t.
# but they also evolve the state in exactly the same way, and should not do that multiple times.
# ideally we do something like

# move_forward!(state, Δt)
# total_rate = zero(eltype(individual_rates))
# for (i, rate_function) in enumerate(inhomogeneous_rate_functions)
#     individual_rates[i] = rate_function(state)
#     total_rate += individual_rates[i]
# end

# and then determine if an event took place using total_rate.
# we figure out which event took place by sampling from the a categorical distribution over the individual rates.

# rand(Categorical(individual_rates))

# most notably, we also need to evaluate the derivative of the rate function w.r.t. time.
# for the usual rate function, this is best implemented using the hessian vector product.
# but for other rate functions, like those for the sticky strategy, I'm just not sure.
# we can AD a lot, but it would be nice to also allow for analytic input.