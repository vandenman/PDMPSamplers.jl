@testset "online streaming summaries" begin
    function gaussian_model(d)
        gradient!(out, x) = (copyto!(out, x); out)
        hvp!(out, x, direction) = (copyto!(out, direction); out)
        return PDMPModel(d, FullGradient(gradient!), hvp!)
    end

    # Position at `t` inside the dense segment starting at event k.
    function dense_position(trace, k, t, harmonic)
        x0 = trace.positions[:, k]
        free = trace.free_masks === nothing ? trues(length(x0)) :
            trace.free_masks[:, k]
        v = trace.velocities[:, k] .* free
        h = t - trace.times[k]
        return harmonic ? ifelse.(free, x0 .* cos(h) .+ v .* sin(h), x0) :
            x0 .+ v .* h
    end

    # Batch integrals of x, x², x⁴ and the free indicator by fine midpoint
    # quadrature of the dense trace (exact up to quadrature error).
    function dense_batches(trace, edges, harmonic; n=4000)
        d = size(trace.positions, 1)
        out = [zeros(d, length(edges) - 1) for _ in 1:4]
        for b in 1:length(edges)-1
            h = (edges[b+1] - edges[b]) / n
            for s in 1:n
                t = edges[b] + (s - 0.5) * h
                k = searchsortedlast(trace.times, t)
                x = dense_position(trace, k, t, harmonic)
                free = trace.free_masks === nothing ? trues(d) :
                    trace.free_masks[:, k]
                out[1][:, b] .+= x .* h
                out[2][:, b] .+= x .^ 2 .* h
                out[3][:, b] .+= x .^ 4 .* h
            end
        end
        # The free indicator is a step function: integrate it exactly.
        for b in 1:length(edges)-1, k in 1:length(trace.times)-1
            overlap = min(edges[b+1], trace.times[k+1]) - max(edges[b], trace.times[k])
            overlap > 0 || continue
            free = trace.free_masks === nothing ? trues(d) : trace.free_masks[:, k]
            out[4][:, b] .+= free .* overlap
        end
        return out
    end

    function run(flow, algorithm, directory; online)
        storage = online ? StreamingTraceStorage(directory; buffer_events=4,
            write_events=false, online_batches=16, online_end_time=8.0,
            online_grid_spacing=0.5) : nothing
        initial = SkeletonPoint([0.7, -0.4, 1.1], [0.3, -0.8, 0.5])
        return pdmp_sample(initial, flow, gaussian_model(3), algorithm,
            0.0, 8.0; seed=911, progress=false, trace_storage=storage)
    end

    for (name, flow, algorithm) in (
            (:bps, BouncyParticle(3, 0.4), GridThinningStrategy()),
            (:zigzag, ZigZag(3), GridThinningStrategy()),
            (:boomerang, Boomerang(3, 0.4), GridThinningStrategy()),
            (:sticky_bps, BouncyParticle(3, 0.4),
                Sticky(GridThinningStrategy(), [0.8, 0.8, 0.8])))
        @testset "$name" begin
            mktempdir() do directory
                harmonic = name === :boomerang
                dense = run(deepcopy(flow), deepcopy(algorithm), directory;
                    online=false).traces[1]
                dense isa PDMPSamplers.FactorizedTrace && (dense = PDMPTrace(dense))
                streamed = run(deepcopy(flow), deepcopy(algorithm), directory;
                    online=true).traces[1]
                online = online_summaries(streamed)
                @test online isa StreamingOnlineSummaries
                d = 3
                nb = length(online.batch_time)
                @test nb == 16
                @test sum(online.batch_time) ≈ 8.0
                edges = collect(range(0.0, 8.0; length=17))
                ref = dense_batches(dense, edges, harmonic)
                got = [reshape(v, d, nb) for v in (online.batch_x,
                    online.batch_x2, online.batch_x4, online.batch_free)]
                @test got[1] ≈ ref[1] atol = 1e-5
                @test got[2] ≈ ref[2] atol = 1e-5
                if harmonic
                    @test all(isnan, got[3])
                else
                    @test got[3] ≈ ref[3] atol = 1e-4
                end
                @test got[4] ≈ ref[4] atol = 1e-5
                # Totals agree with the existing exact streaming moments.
                @test vec(sum(got[1]; dims=2)) ≈ streamed.moments.sum_x
                @test vec(sum(got[2]; dims=2)) ≈ streamed.moments.sum_x2
                # Transitions and grid snapshots.
                if dense.free_masks !== nothing
                    masks = Matrix(dense.free_masks)
                    flips = vec(sum(masks[:, 2:end] .!= masks[:, 1:end-1]; dims=2))
                    @test online.transitions == flips
                    @test sum(flips) > 0
                end
                @test vec(sum(reshape(online.batch_transitions, d, nb); dims=2)) ==
                    online.transitions
                # Model-size integrals: S = number of free coordinates.
                @test online.batch_size ≈ vec(sum(ref[4]; dims=1)) atol = 1e-5
                if dense.free_masks === nothing
                    @test online.batch_size2 ≈ 9 .* online.batch_time
                else
                    # Jensen within each batch, with equality iff S is constant.
                    @test all(online.batch_size2 .>=
                        online.batch_size .^ 2 ./ online.batch_time .- 1e-9)
                end
                @test online.grid_times ≈ collect(0.0:0.5:8.0)
                grid_x = reshape(online.grid_x, d, :)
                for (j, t) in enumerate(online.grid_times[1:end-1])
                    k = searchsortedlast(dense.times, t)
                    @test grid_x[:, j] ≈ dense_position(dense, k, t, harmonic) atol = 1e-10
                end
                # Nothing but the summary file was written for the main phase.
                main = joinpath(directory, "main")
                @test isfile(joinpath(main, "online_summaries.bin"))
                @test !any(startswith("chunk_"), readdir(main))
                @test isempty(streamed.chunks)
            end
        end
    end
end
