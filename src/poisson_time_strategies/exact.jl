# should accept a function with some signature that returns a time t?
struct ExactStrategy{T} <: PoissonTimeStrategy
    get_next_event_time::T # Function to compute the next event time
end
next_event_time(model, flow, alg_::ExactStrategy, state::PDMPState, cache) = alg_.get_next_event_time(flow, state, cache)
