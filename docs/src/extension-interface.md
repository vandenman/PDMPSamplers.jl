# Extension interface

These functions are not exported, but they are a supported interface for
packages that build on PDMPSamplers.jl, such as PDMPSamplersR. They let such
packages replay streaming traces, reproduce the sampler's random number
generator, and supply their own subsampling envelopes. Call them qualified,
for example `PDMPSamplers.stream_cursor(trace)`, and extend them with
`PDMPSamplers.draw_component(rng, envelope::MyEnvelope, B) = ...`. Their
docstrings are on the [API](api.md) page.

Statistics counters are updated through the generated `_inc_counter_<name>`
functions described in `src/statistic_counters.jl`.

## Flows and random numbers

- [`PDMPSamplers.underlying_flow`](@ref)
- [`PDMPSamplers.make_initial_rng`](@ref)
- [`PDMPSamplers.initialize_velocity`](@ref)
- [`PDMPSamplers.initialize_flow_state!`](@ref)

## Streaming traces

- [`PDMPSamplers.stream_cursor`](@ref)
- [`PDMPSamplers.load_streaming_chunk`](@ref)
- [`PDMPSamplers.stream_move!`](@ref)
- [`PDMPSamplers.stream_apply_event!`](@ref)
- [`PDMPSamplers.STREAM_EVENT_REFLECT_FULL`](@ref)
- [`PDMPSamplers.flush_streaming_trace!`](@ref)
- [`PDMPSamplers.restore_streaming_trace`](@ref)
- [`PDMPSamplers.has_integrable_segment`](@ref)
- [`PDMPSamplers.compact`](@ref)
- [`PDMPSamplers.get_warmup_trace`](@ref)

## Subsampling envelopes

- [`PDMPSamplers.deterministic_gradient!`](@ref)
- [`PDMPSamplers.refresh_anchor_owned!`](@ref)
- [`PDMPSamplers.deterministic_rate_cell_bound`](@ref)
- [`PDMPSamplers.has_direct_deterministic_rate_cell_bound`](@ref)
- [`PDMPSamplers.validate_subsampling_grid_envelope`](@ref)
- [`PDMPSamplers.trajectory_scale_anchor`](@ref)
- [`PDMPSamplers.component_scales!`](@ref)
- [`PDMPSamplers.total_residual_bound`](@ref)
- [`PDMPSamplers.screening_residual_bound`](@ref)
- [`PDMPSamplers.n_observations`](@ref)
- [`PDMPSamplers.draw_subset!`](@ref)
- [`PDMPSamplers.draw_component`](@ref)
- [`PDMPSamplers.draw_distinguished`](@ref)
- [`PDMPSamplers.draw_base_subset!`](@ref)
- [`PDMPSamplers.signed_subset_bound`](@ref)
- [`PDMPSamplers.joint_signed_group_enabled`](@ref)
- [`PDMPSamplers.subsampling_subset_bound`](@ref)
- [`PDMPSamplers.subsampling_residual_subset_bound`](@ref)
- [`PDMPSamplers.subsampling_deterministic_signed_rate`](@ref)
- [`PDMPSamplers.subsampling_deferred_candidate_rate`](@ref)
- [`PDMPSamplers.subsampling_candidate_rate!`](@ref)
- [`PDMPSamplers.subsampling_cached_residual!`](@ref)
- [`PDMPSamplers.begin_subsampling_mark!`](@ref)
- [`PDMPSamplers.record_subsampling_proposal!`](@ref)
