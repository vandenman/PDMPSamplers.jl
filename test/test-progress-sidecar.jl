@testset "progress sidecar counter interface" begin
    progress_getters = (
        :_get_counter_reflections_events,
        :_get_counter_∇f_calls,
        :_get_counter_full_gradient_calls,
        :_get_counter_grid_horizon_hits,
        :_get_counter_adaptation_updates,
        :_get_counter_grid_resets_from_dynamics_adaptation,
        :_get_counter_refreshment_events,
        :_get_counter_sticky_events,
        :_get_counter_sticky_freezes,
        :_get_counter_sticky_unfreezes,
    )
    @test all(isdefined(PDMPSamplers, name) for name in progress_getters)

    # The R bridge uses DevelStatisticCounter, whose composite contains every
    # concrete counter read by the progress writer.
    stats = PDMPSamplers.DevelStatisticCounter()
    @test any(c isa PDMPSamplers.GridThinningCounter for c in stats.counters)
    @test any(c isa PDMPSamplers.RunSummaryCounter for c in stats.counters)
    @test any(c isa PDMPSamplers.GradientCallCounter for c in stats.counters)
    @test any(c isa PDMPSamplers.BasicEventCounter for c in stats.counters)

    path = tempname() * ".progress"
    old_path = get(ENV, "PDMPSAMPLERS_PROGRESS_PATH", nothing)
    old_interval = get(ENV, "PDMPSAMPLERS_PROGRESS_INTERVAL", nothing)
    try
        ENV["PDMPSAMPLERS_PROGRESS_PATH"] = path
        ENV["PDMPSAMPLERS_PROGRESS_INTERVAL"] = "0"
        PDMPSamplers._reset_progress_sidecar_counters!()
        state = PDMPSamplers.PDMPState(0.0,
            PDMPSamplers.SkeletonPoint([0.0, 0.0], [1.0, -1.0]))

        # Forced phase entry uses the actual writer and the actual bridge
        # composite, rather than a manually created key/value fixture.
        PDMPSamplers._write_progress_sidecar!(:warmup, state, stats;
            force=true, flow=nothing, alg=nothing)
        PDMPSamplers._inc_counter_grid_horizon_hits(stats)
        PDMPSamplers._inc_counter_grid_horizon_hits(stats)
        PDMPSamplers._inc_counter_reflections_events(stats)
        PDMPSamplers._inc_counter_∇f_calls(stats)
        PDMPSamplers._inc_counter_full_gradient_calls(stats)
        PDMPSamplers._set_counter_adaptation_updates(stats, 3)
        PDMPSamplers._inc_counter_grid_resets_from_dynamics_adaptation(stats)
        state.t[] = 0.25
        PDMPSamplers._write_progress_sidecar!(:warmup, state, stats;
            force=true, flow=nothing, alg=nothing)

        lines = readlines(path)
        lastvalue(name) = split(last(filter(x -> startswith(x, name * "="),
            lines)), "="; limit=2)[2]
        @test lastvalue("physical_pdmp_time") == "0.25"
        @test lastvalue("grid_horizon_hits") == "2"
        @test lastvalue("reflections") == "1"
        @test lastvalue("gradient_calls") == "1"
        @test lastvalue("full_gradient_calls") == "1"
        @test lastvalue("adaptation_updates") == "3"
        @test lastvalue("dynamics_adaptation_resets") == "1"

        # The lighter public bundle deliberately omits GridThinningCounter.
        # Missing grid measurements are explicit, not fabricated as zeros;
        # its Basic/Gradient/RunSummary values remain genuinely recorded.
        light = PDMPSamplers.StatisticCounter()
        PDMPSamplers._write_progress_sidecar!(:main, state, light;
            force=true, flow=nothing, alg=nothing)
        lines = readlines(path)
        @test lastvalue("grid_horizon_hits") == "unavailable"
        @test lastvalue("dynamics_adaptation_resets") == "unavailable"
        @test lastvalue("adaptation_updates") == "0"
    finally
        old_path === nothing ? delete!(ENV, "PDMPSAMPLERS_PROGRESS_PATH") :
            (ENV["PDMPSAMPLERS_PROGRESS_PATH"] = old_path)
        old_interval === nothing ? delete!(ENV,
            "PDMPSAMPLERS_PROGRESS_INTERVAL") :
            (ENV["PDMPSAMPLERS_PROGRESS_INTERVAL"] = old_interval)
        rm(path; force=true)
    end
end
