# should accept a function with some signature that returns a time t?
struct ExactStrategy{T} <: PoissonTimeStrategy
    get_next_event_time::T # Function to compute the next event time
end
function next_event_time(model, flow, alg_::ExactStrategy, state::PDMPState, cache)
    τ, event_type, raw_meta = alg_.get_next_event_time(flow, state, cache)
    return τ, event_type, wrap_meta(raw_meta)
end
