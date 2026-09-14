@testset "bounded typed streaming trace" begin
    function gaussian_model(d)
        gradient!(out, x) = (copyto!(out, x); out)
        hvp!(out, x, direction) = (copyto!(out, direction); out)
        return PDMPModel(d, FullGradient(gradient!), hvp!)
    end

    function matched_run(flow, algorithm, directory; streaming)
        d = 3
        model = gaussian_model(d)
        initial = SkeletonPoint([0.7, -0.4, 1.1], [0.3, -0.8, 0.5])
        storage = streaming ? StreamingTraceStorage(directory;
            buffer_events=3) : nothing
        return pdmp_sample(initial, flow, model, algorithm, 0.0, 8.0;
            seed=319, progress=false, trace_storage=storage)
    end

    for (name, flow) in (
            (:bps, BouncyParticle(3, 0.4)),
            (:boomerang, Boomerang(3, 0.4)),
            (:zigzag, ZigZag(3)))
        mktempdir() do directory
            algorithm = GridThinningStrategy()
            dense = matched_run(deepcopy(flow), deepcopy(algorithm),
                directory; streaming=false)
            streamed = matched_run(deepcopy(flow), deepcopy(algorithm),
                directory; streaming=true)
            dense_endpoint = terminal_state(dense)
            streamed_endpoint = terminal_state(streamed)
            @test dense_endpoint.t == streamed_endpoint.t
            @test dense_endpoint.position == streamed_endpoint.position
            @test dense_endpoint.physical_velocity ==
                streamed_endpoint.physical_velocity
            @test dense_endpoint.free == streamed_endpoint.free
            @test dense.stats[1].reflections_events ==
                streamed.stats[1].reflections_events
            @test dense.stats[1].refreshment_events ==
                streamed.stats[1].refreshment_events
            @test mean(dense.traces[1]) ≈ mean(streamed.traces[1])
            @test var(dense.traces[1]) ≈ var(streamed.traces[1])
            @test inclusion_probs(dense.traces[1]) ≈
                inclusion_probs(streamed.traces[1])
            manifest = streaming_trace_manifest(streamed.traces[1])
            @test manifest.format == "pdmpsamplers_typed_stream_v1"
            @test all(isfile(chunk.path) for chunk in
                streamed.traces[1].chunks)
            @test streamed.traces[1].buffer.n_events == 0
            @test streamed.traces[1].buffer_maximum <= 3
        end
    end

    # Flush is an I/O operation. Force it directly around a sticky state and
    # establish that neither the physical state nor an unrelated RNG moves.
    mktempdir() do directory
        flow = BouncyParticle(4, 0.0)
        state = StickyPDMPState(0.0,
            SkeletonPoint([0.0, 1.0, -0.2, 0.4], [0.0, 0.3, -0.7, 0.2]),
            BitVector([false, true, true, true]), [1.25, 0.0, 0.0, 0.0])
        trace = PDMPSamplers.StreamingPDMPTrace(state, flow,
            StreamingTraceStorage(directory; buffer_events=1), "manual")
        PDMPSamplers.begin_trace_phase!(trace, state, flow)
        rng = Xoshiro(88)
        rng_reference = copy(rng)
        before = (state.t[], copy(state.ξ.x), copy(state.ξ.θ),
            copy(state.free), copy(state.stored_velocity))
        PDMPSamplers._stream_record_full!(trace, state,
            PDMPSamplers._STREAM_EVENT_REFRESH)
        PDMPSamplers._flush_streaming_trace!(trace)
        after = (state.t[], copy(state.ξ.x), copy(state.ξ.θ),
            copy(state.free), copy(state.stored_velocity))
        @test before == after
        @test rand(rng, UInt64, 8) == rand(rng_reference, UInt64, 8)
    end

    # Computational boundaries advance the observed deterministic interval
    # but add no physical snapshot. Freeze/release records remain scalar.
    mktempdir() do directory
        flow = BouncyParticle(2, 0.0)
        state = StickyPDMPState(0.0,
            SkeletonPoint([1.0, 0.0], [0.5, 0.0]), trues(2), zeros(2))
        trace = PDMPSamplers.StreamingPDMPTrace(state, flow,
            StreamingTraceStorage(directory; buffer_events=8), "typed")
        PDMPSamplers.begin_trace_phase!(trace, state, flow)
        state.t[] = 0.25
        state.ξ.x[1] += 0.125
        PDMPSamplers._record_streaming_event!(trace, state, flow, nothing,
            :horizon_hit)
        @test trace.physical_events == 0
        @test trace.computational_boundaries == 1
        state.t[] = 0.5
        state.ξ.x[1] += 0.125
        state.ξ.x[2] = 0.0
        state.ξ.θ[2] = 0.0
        state.stored_velocity[2] = -0.9
        state.free[2] = false
        PDMPSamplers._record_streaming_event!(trace, state, flow, 2, :sticky)
        state.t[] = 0.75
        state.ξ.x[1] += 0.125
        state.ξ.θ[2] = -0.9
        state.stored_velocity[2] = 0.0
        state.free[2] = true
        PDMPSamplers._record_streaming_event!(trace, state, flow, 2, :sticky)
        PDMPSamplers.finish_trace_phase!(trace, state, flow)
        chunk = PDMPSamplers._load_streaming_chunk(only(trace.chunks))
        @test chunk.kinds == [PDMPSamplers._STREAM_EVENT_FREEZE,
            PDMPSamplers._STREAM_EVENT_RELEASE]
        @test chunk.coordinates == Int32[2, 2]
        @test size(chunk.full_effective_velocities, 2) == 0
    end

    # Buffer storage is fixed by dimension and capacity, not event count.
    mktempdir() do directory
        d = 556
        flow = BouncyParticle(d, 0.0)
        state = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        trace = PDMPSamplers.StreamingPDMPTrace(state, flow,
            StreamingTraceStorage(directory; buffer_events=7), "large")
        PDMPSamplers.begin_trace_phase!(trace, state, flow)
        fixed_bytes = Base.summarysize(trace.buffer)
        for event in 1:50
            state.t[] = event / 100
            state.ξ.x .+= 0.01 .* state.ξ.θ
            state.ξ.θ .*= -1
            PDMPSamplers._record_streaming_event!(trace, state, flow,
                nothing, :reflect)
        end
        @test Base.summarysize(trace.buffer) == fixed_bytes
        @test trace.buffer_maximum <= 7
        @test length(trace.chunks) >= 7
    end

    # Published chunks remain valid if a later in-memory buffer is lost. A
    # completed logical trace can be rebuilt from chunk headers and endpoint
    # provenance without constructing a dense coordinate-by-event matrix.
    mktempdir() do directory
        d = 4
        flow = BouncyParticle(d, 0.0)
        state = StickyPDMPState(0.0,
            SkeletonPoint(fill(0.2, d), collect(0.1:0.1:0.4)), trues(d),
            zeros(d))
        trace = PDMPSamplers.StreamingPDMPTrace(state, flow,
            StreamingTraceStorage(directory; buffer_events=2), "recover")
        PDMPSamplers.begin_trace_phase!(trace, state, flow)
        initial = PDMPSamplers.PDMPTerminalState(state, flow)
        for event in 1:3
            state.t[] += 0.1
            state.ξ.x .+= 0.1 .* state.ξ.θ
            state.ξ.θ .*= -1
            PDMPSamplers._record_streaming_event!(trace, state, flow,
                nothing, :reflect)
        end
        @test length(trace.chunks) == 1
        @test PDMPSamplers._load_streaming_chunk(only(trace.chunks)).times ==
            [0.1, 0.2]
        @test isempty(filter(path -> occursin(".tmp.", path),
            readdir(dirname(only(trace.chunks).path); join=true)))
        PDMPSamplers.finish_trace_phase!(trace, state, flow)
        terminal = PDMPSamplers.PDMPTerminalState(state, flow)
        restored = PDMPSamplers._restore_streaming_trace(
            [chunk.path for chunk in trace.chunks], flow, initial, terminal;
            physical_events=trace.physical_events,
            computational_boundaries=trace.computational_boundaries,
            buffer_maximum=trace.buffer_maximum)
        @test mean(restored) ≈ mean(trace)
        @test var(restored) ≈ var(trace)
        @test inclusion_probs(restored) ≈ inclusion_probs(trace)
        @test last(restored).position == terminal.position

        # A missing or truncated chunk cannot be reopened as a complete trace.
        copied = joinpath(directory, "copied")
        mkpath(copied)
        copied_paths = String[]
        for (i, chunk) in enumerate(trace.chunks)
            path = joinpath(copied,
                "chunk_" * lpad(string(i), 8, '0') * ".bin")
            cp(chunk.path, path)
            push!(copied_paths, path)
        end
        rm(last(copied_paths))
        @test_throws ArgumentError PDMPSamplers._restore_streaming_trace(
            copied, length(copied_paths), flow, initial, terminal;
            physical_events=trace.physical_events,
            computational_boundaries=trace.computational_boundaries,
            buffer_maximum=trace.buffer_maximum)
        cp(last(trace.chunks).path, last(copied_paths))
        open(last(copied_paths), "r+") do io
            truncate(io, max(1, filesize(last(copied_paths)) - 7))
        end
        @test_throws EOFError PDMPSamplers._restore_streaming_trace(
            copied, length(copied_paths), flow, initial, terminal;
            physical_events=trace.physical_events,
            computational_boundaries=trace.computational_boundaries,
            buffer_maximum=trace.buffer_maximum)
    end

    # Harmonic replay spans multiple chunks, a freeze/release, a refresh that
    # changes a frozen stored velocity, an omitted computational boundary and
    # a terminal interval without a physical event. Restoration independently
    # reproduces the endpoint and integrated summaries before the endpoint is
    # treated as authoritative.
    mktempdir() do directory
        flow = Boomerang(2, 0.0)
        initial_x2 = -tan(0.2) * 0.7
        state = StickyPDMPState(0.0,
            SkeletonPoint([0.4, initial_x2], [0.2, 0.7]), trues(2), zeros(2))
        trace = PDMPSamplers.StreamingPDMPTrace(state, flow,
            StreamingTraceStorage(directory; buffer_events=2), "main")
        PDMPSamplers.begin_trace_phase!(trace, state, flow)
        initial = PDMPSamplers.PDMPTerminalState(state, flow)
        move_forward_time!(state, 0.2, flow)
        state.stored_velocity[2] = state.ξ.θ[2]
        state.ξ.θ[2] = 0.0; state.free[2] = false
        PDMPSamplers._record_streaming_event!(trace, state, flow, 2, :sticky)
        move_forward_time!(state, 0.2, flow)
        state.ξ.θ[1] = -0.35; state.stored_velocity[2] = 1.1
        PDMPSamplers._record_streaming_event!(trace, state, flow, nothing,
            :refresh)
        move_forward_time!(state, 0.2, flow)
        state.ξ.θ[2] = state.stored_velocity[2]
        state.stored_velocity[2] = 0.0; state.free[2] = true
        PDMPSamplers._record_streaming_event!(trace, state, flow, 2, :sticky)
        move_forward_time!(state, 0.2, flow)
        state.ξ.θ .= [0.45, -0.6]
        PDMPSamplers._record_streaming_event!(trace, state, flow, nothing,
            :reflect)
        move_forward_time!(state, 0.1, flow)
        PDMPSamplers._record_streaming_event!(trace, state, flow, nothing,
            :anchor_selection_boundary)
        move_forward_time!(state, 0.1, flow)
        PDMPSamplers.finish_trace_phase!(trace, state, flow)
        terminal = PDMPSamplers.PDMPTerminalState(state, flow)
        @test length(trace.chunks) >= 2
        restored = PDMPSamplers._restore_streaming_trace(trace.directory,
            length(trace.chunks), flow, initial, terminal;
            physical_events=trace.physical_events,
            computational_boundaries=trace.computational_boundaries,
            buffer_maximum=trace.buffer_maximum)
        @test last(restored).position ≈ terminal.position atol=1e-12
        @test last(restored).physical_velocity ≈
            terminal.physical_velocity atol=1e-12
        @test last(restored).stored_frozen_velocity ≈
            terminal.stored_frozen_velocity atol=1e-12
        @test last(restored).free == terminal.free
        @test mean(restored) ≈ mean(trace) atol=1e-12
        @test var(restored) ≈ var(trace) atol=1e-12
        @test inclusion_probs(restored) ≈ inclusion_probs(trace) atol=1e-12
        # Chunk artifacts outlive the in-memory trace object.
        paths = [chunk.path for chunk in trace.chunks]
        trace = nothing; GC.gc()
        @test all(isfile, paths)
    end
end
