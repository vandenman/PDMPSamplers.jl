@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _instrumentation_fixture()
    q = [0.5, 0.7, 1.0, 1.2]
    q ./= sum(q)
    envelope = TrajectoryResidualEnvelope(reshape(q, 1, :), [0.0])
    oracle = (out, x, subset, anchor) -> (out[1] = sum(q[i] for i in subset) * x[1])
    cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0), oracle,
        envelope, [0.0], 2)
    return PDMPModel(1, cv), BouncyParticle(1, 0.5)
end

struct _DirectInstrumentedGradient end
function (::_DirectInstrumentedGradient)(out, x)
    out[1] = 0.0
    return out
end
PDMPSamplers.has_direct_deterministic_rate_cell_bound(::_DirectInstrumentedGradient) = true
PDMPSamplers.deterministic_rate_cell_bound(::_DirectInstrumentedGradient,
    state::PDMPSamplers.AbstractPDMPState,
    flow::PDMPSamplers.ContinuousDynamics, left::Real, right::Real) = 0.0

function _direct_instrumentation_fixture()
    q = [0.5, 0.7, 1.0, 1.2]
    q ./= sum(q)
    envelope = TrajectoryResidualEnvelope(reshape(q, 1, :), [0.0])
    oracle = (out, x, subset, anchor) ->
        (out[1] = sum(q[i] for i in subset) * x[1])
    refresh = anchor -> TrajectoryResidualEnvelope(reshape(q, 1, :), anchor)
    cv = SubsampledControlVariate(_DirectInstrumentedGradient(), oracle,
        envelope, [0.0], 2; refresh_anchor! = refresh)
    return PDMPModel(1, cv), BouncyParticle(1, 0.5)
end

mutable struct CandidateRecorder
    rows::Vector{NTuple{7,Any}}
end

@testset "statistics preserve direct-certificate event dispatch" begin
    model_a, flow_a = _direct_instrumentation_fixture()
    model_b, flow_b = _direct_instrumentation_fixture()
    wrapped = PDMPSamplers.with_stats(model_b.grad,
        PDMPSamplers.DevelStatisticCounter())
    @test PDMPSamplers.has_direct_deterministic_rate_cell_bound(
        model_a.grad.deterministic_gradient!)
    @test PDMPSamplers.has_direct_deterministic_rate_cell_bound(
        wrapped.deterministic_gradient!)
    PDMPSamplers.set_active_set!(wrapped, trues(1))
    PDMPSamplers.refresh_anchor!(wrapped, [0.1])
    @test wrapped.anchor == [0.1]
    @test PDMPSamplers.has_direct_deterministic_rate_cell_bound(
        wrapped.deterministic_gradient!)

    trace_a, _ = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow_a, model_a,
        GridThinningStrategy(N=4, t_max=1.0, lazy=true), 0.0, 4.0;
        seed=7732, progress=false,
        statistic_counter=PDMPSamplers.StatisticCounter)
    trace_b, stats_b = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow_b, model_b,
        GridThinningStrategy(N=4, t_max=1.0, lazy=true), 0.0, 4.0;
        seed=7732, progress=false,
        statistic_counter=PDMPSamplers.DevelStatisticCounter)
    @test trace_a.times == trace_b.times
    @test trace_a.positions == trace_b.positions
    @test trace_a.velocities == trace_b.velocities
    @test trace_a.free_masks == trace_b.free_masks
    @test stats_b.grid_bound_evaluations > 0
    @test stats_b.grid_endpoint_gradient_calls == 0
end
function PDMPSamplers.record_subsampling_candidate!(
        recorder::CandidateRecorder, args...)
    push!(recorder.rows, args)
    return nothing
end

@testset "subsampling pipeline counters are observational" begin
    model_a, flow_a = _instrumentation_fixture()
    model_b, flow_b = _instrumentation_fixture()
    trace_a, _ = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow_a, model_a,
        GridThinningStrategy(N=4, t_max=1.0, lazy=false), 0.0, 4.0;
        seed=7731, progress=false, statistic_counter=PDMPSamplers.StatisticCounter)
    trace_b, stats_b = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow_b, model_b,
        GridThinningStrategy(N=4, t_max=1.0, lazy=false), 0.0, 4.0;
        seed=7731, progress=false, statistic_counter=PDMPSamplers.DevelStatisticCounter)
    @test trace_a.times == trace_b.times
    @test trace_a.positions == trace_b.positions
    @test trace_a.velocities == trace_b.velocities
    @test trace_a.free_masks == trace_b.free_masks
    @test stats_b.clock_candidates >= 0
    @test stats_b.subset_draws >= 0
    @test stats_b.residual_oracle_evaluations >= 0

    pipeline_ints = (
        :clock_candidates, :grid_bound_evaluations,
        :pointwise_screen_evaluations, :pointwise_screen_passes,
        :subset_draws, :selected_subset_bound_evaluations,
        :selected_subset_bound_passes, :residual_oracle_evaluations,
        :final_thinning_acceptances, :reflection_events, :sticky_events,
        :grid_schedule_builds, :grid_schedule_searches,
        :poisson_time_generations, :candidate_state_copies,
        :candidate_move_forwards, :candidate_loop_iterations,
        :statistic_recordings,
    )
    pipeline_times = (
        :grid_bound_seconds, :pointwise_screen_seconds,
        :subset_draw_seconds, :selected_subset_refinement_seconds,
        :residual_oracle_seconds, :final_thinning_seconds,
        :reflection_update_seconds,
        :grid_schedule_build_seconds, :grid_schedule_search_seconds,
        :poisson_time_generation_seconds, :candidate_state_copy_seconds,
        :candidate_move_forward_seconds, :candidate_loop_overhead_seconds,
        :statistic_recording_seconds,
    )
    phase_fields = vcat(
        [Symbol(prefix, name) for prefix in ("warmup_", "main_") for name in pipeline_ints],
        [Symbol(prefix, name) for prefix in ("warmup_", "main_") for name in pipeline_times],
    )
    for name in phase_fields
        @test hasproperty(stats_b, name)
        @test isfinite(getproperty(stats_b, name))
        @test getproperty(stats_b, name) >= 0
    end
    # The phase snapshots are deltas of the same cumulative counters.  This
    # guards against accidentally reporting warmup work as retained work.
    for name in pipeline_ints
        @test getproperty(stats_b, Symbol("warmup_", name)) +
            getproperty(stats_b, Symbol("main_", name)) == getproperty(stats_b, name)
    end
    for name in pipeline_times
        @test getproperty(stats_b, Symbol("warmup_", name)) +
            getproperty(stats_b, Symbol("main_", name)) ≈ getproperty(stats_b, name)
    end
    # These are control-flow inequalities, not assumptions about an
    # alternative proposal law: every pass is downstream of its evaluation.
    @test stats_b.pointwise_screen_passes <= stats_b.pointwise_screen_evaluations
    @test stats_b.selected_subset_bound_passes <= stats_b.selected_subset_bound_evaluations
    @test stats_b.final_thinning_acceptances <= stats_b.residual_oracle_evaluations
    @test stats_b.reflection_events <= stats_b.final_thinning_acceptances
end

@testset "subsampling candidate instrumentation is observational" begin
    q = fill(0.25, 4)
    envelope = TrajectoryResidualEnvelope(reshape(q, 1, :), [0.0])
    recorder = CandidateRecorder(NTuple{7,Any}[])
    cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0), recorder,
        envelope, [0.0], 2)
    @test PDMPSamplers._base_subset_probability(
        cv, cv.subset_design) == 1 / 6
    @test PDMPSamplers._selected_subset_probability(cv, 1.0, 2.0, 3.0) == 1 / 6
end
