@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Support-boundary handling" begin

    # ── Synthetic model with a boundary at x[1] >= 1.0 ──

    d_boundary = 2
    function _boundary_grad!(out, x)
        if x[1] >= 1.0
            error("Outside support: x[1] = $(x[1]) >= 1.0")
        end
        out .= x
        return out
    end

    function _make_boundary_model()
        return PDMPModel(d_boundary, FullGradient(_boundary_grad!))
    end

    function _make_boundary_flow()
        return BouncyParticle(Matrix{Float64}(I, d_boundary, d_boundary), zeros(d_boundary))
    end

    function _make_boundary_alg(; N=20, t_max=2.0)
        return GridThinningStrategy(; N, t_max)
    end

    @testset "SupportBoundaryOptions — mode field" begin
        opts = SupportBoundaryOptions(; mode=:line_search_truncated_refresh, max_refresh_attempts=7, refresh_probe_time=1e-5)
        @test opts.mode === :line_search_truncated_refresh
        @test opts.max_refresh_attempts == 7
        @test opts.refresh_probe_time == 1e-5
    end

    @testset "localize_support_boundary! — basic bisection" begin
        model = _make_boundary_model()
        x0 = [0.0, 0.0]   # valid starting point
        v  = [1.0, 0.0]   # moves toward x[1] = 1.0
        t_valid = 0.0
        t_invalid = 2.0   # x[1] = 2.0, definitely invalid

        ctx = PDMPSamplers.BoundaryContext(
            x0, v, t_valid, t_invalid,
            ErrorException("test"), BouncyParticle, FullGradient,
        )
        opts = SupportBoundaryOptions()

        loc = @inferred PDMPSamplers.localize_support_boundary!(model, ctx, opts)
        tau_star, tau_safe = loc

        # Boundary should be near x[1] = 1.0, i.e., t ≈ 1.0
        @test loc isa PDMPSamplers.SupportBoundaryLocalization
        @test tau_star ≈ 1.0 atol=1e-3
        @test tau_safe < tau_star
        @test tau_safe > 0.0
    end

    @testset "localize_support_boundary! — tolerances" begin
        model = _make_boundary_model()
        x0 = [0.0, 0.0]
        v  = [1.0, 0.0]
        ctx = PDMPSamplers.BoundaryContext(
            x0, v, 0.0, 2.0, ErrorException("test"), BouncyParticle, FullGradient,
        )

        # Tight tolerance
        opts_tight = SupportBoundaryOptions(; time_rtol=1e-12, time_atol=1e-12, max_bisection_steps=60)
        tau_tight, _ = PDMPSamplers.localize_support_boundary!(model, ctx, opts_tight)
        @test tau_tight ≈ 1.0 atol=1e-6

        # Loose tolerance, few steps
        opts_loose = SupportBoundaryOptions(; time_rtol=0.1, time_atol=0.1, max_bisection_steps=5)
        tau_loose, _ = PDMPSamplers.localize_support_boundary!(model, ctx, opts_loose)
        @test tau_loose ≈ 1.0 atol=0.2
    end

    @testset "localize_support_boundary! — max_bisection_steps respected" begin
        model = _make_boundary_model()
        x0 = [0.0, 0.0]
        v  = [1.0, 0.0]
        ctx = PDMPSamplers.BoundaryContext(
            x0, v, 0.0, 2.0, ErrorException("test"), BouncyParticle, FullGradient,
        )

        opts = SupportBoundaryOptions(; max_bisection_steps=3, time_rtol=0.0, time_atol=0.0)
        tau, _ = PDMPSamplers.localize_support_boundary!(model, ctx, opts)
        # After only 3 bisection steps from [0, 2], range should be roughly 2.0/8 = 0.25
        @test 0.5 <= tau <= 2.0
    end

    @testset "localize_support_boundary! — valid at start" begin
        model = _make_boundary_model()
        x0 = [0.5, 0.0]   # valid starting point
        v  = [1.0, 0.0]   # moves toward boundary
        ctx = PDMPSamplers.BoundaryContext(
            x0, v, 0.0, 1.0, ErrorException("test"), BouncyParticle, FullGradient,
        )

        tau_star, _ = PDMPSamplers.localize_support_boundary!(model, ctx, SupportBoundaryOptions())
        # Boundary should be near x[1] = 1.0 => t ≈ 0.5
        @test tau_star ≈ 0.5 atol=1e-3
    end

    @testset "pdmp_sample — :error mode throws SupportBoundaryError" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        @test_throws SupportBoundaryError begin
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:error))
        end

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:error))
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == false
            @test err.last_valid_time >= 0.0
            @test err.first_invalid_time > err.last_valid_time
        end
    end

    @testset "pdmp_sample — :line_search mode localizes boundary" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        @test_throws SupportBoundaryError begin
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:line_search))
        end

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:line_search))
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == true
            @test err.estimated_boundary_time !== nothing
            @test err.estimated_boundary_time > err.last_valid_time
            @test err.estimated_boundary_time <= err.first_invalid_time
            @test err.algorithm_type === GridThinningStrategy
        end
    end

    @testset "pdmp_sample — SupportBoundaryOptions mode API" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        opts = SupportBoundaryOptions(; mode=:line_search)

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_options=opts)
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == true
        end
    end

    @testset "pdmp_sample — :line_search with custom options" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        opts = SupportBoundaryOptions(; mode=:line_search, max_bisection_steps=10, time_rtol=1e-6, time_atol=1e-8)

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
            progress=false, seed=42, support_boundary_options=opts)
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == true
        end
    end

    @testset "pdmp_sample — ThinningStrategy integration" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = ThinningStrategy(GlobalBounds(1e-2 / d_boundary, d_boundary))
        x0 = zeros(d_boundary)

        @test_throws SupportBoundaryError begin
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:error))
        end

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:line_search))
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == true
            @test err.algorithm_type <: ThinningStrategy
        end
    end

    @testset "pdmp_sample — line_search_truncated_refresh recovers for BPS family" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg(; N=20, t_max=2.0)
        x0 = zeros(d_boundary)
        opts = SupportBoundaryOptions(; mode=:line_search_truncated_refresh, max_refresh_attempts=200, refresh_probe_time=1e-4)

        result = pdmp_sample(x0, flow, model, alg, 0.0, 20.0, 0.0;
            progress=false, seed=42, support_boundary_options=opts)

        @test result isa PDMPChains
        @test result.stats[1].support_boundary_events >= 1
        @test result.stats[1].support_boundary_refresh_attempts >= result.stats[1].support_boundary_events
    end

    @testset "pdmp_sample — line_search_truncated_refresh recovers for PreconditionedBPS" begin
        model = _make_boundary_model()
        flow = PreconditionedBPS(Matrix{Float64}(I, d_boundary, d_boundary), zeros(d_boundary))
        alg = _make_boundary_alg(; N=20, t_max=2.0)
        x0 = zeros(d_boundary)
        opts = SupportBoundaryOptions(; mode=:line_search_truncated_refresh, max_refresh_attempts=200, refresh_probe_time=1e-4)

        result = pdmp_sample(x0, flow, model, alg, 0.0, 20.0, 0.0;
            progress=false, seed=42, support_boundary_options=opts)

        @test result isa PDMPChains
    end

    @testset "line_search_truncated_refresh falls back from current-invalid zero bracket" begin
        rng = PDMPSamplers.Random.Xoshiro(42)
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg(; N=20, t_max=2.0)

        previous_state = PDMPState(0.0, SkeletonPoint([0.0, 0.0], [1.0, 0.0]))
        state = PDMPState(1.0, SkeletonPoint([1.0, 0.0], [1.0, 0.0]))
        trace_manager = PDMPSamplers.TraceManager(state, flow, alg, 0.0)
        PDMPSamplers.record_event!(trace_manager, previous_state, flow, nothing; phase=:main)

        cache = PDMPSamplers.add_gradient_to_cache(
            PDMPSamplers.initialize_cache(rng, flow, model.grad, alg, state.t[], state.ξ), state.ξ,
        )
        stats = PDMPSamplers.StatisticCounter()
        ctx = PDMPSamplers.BoundaryContext(
            copy(state.ξ.x), copy(state.ξ.θ), state.t[],
            0.0, eps(Float64), ErrorException("test"), BouncyParticle, GridThinningStrategy,
        )
        opts = SupportBoundaryOptions(; mode=:line_search_truncated_refresh, max_refresh_attempts=1, refresh_probe_time=0.0)

        result = PDMPSamplers._line_search_truncated_refresh_boundary!(
            rng, state, model, flow, alg, cache, stats, trace_manager, ctx, opts; phase=:main,
        )

        @test result === :refresh
        @test state.t[] < 1.0
        @test state.ξ.x[1] < 1.0
        @test stats.support_boundary_events == 1
        @test stats.support_boundary_refresh_attempts == 1
    end

    @testset "line_search_truncated_refresh backtracks from invalid clipped safe point" begin
        rng = PDMPSamplers.Random.Xoshiro(321)
        function _gap_grad!(out, x)
            if (0.5 < x[1] < 0.9) || x[1] >= 1.5
                error("Outside support gap: x[1] = $(x[1])")
            end
            out .= x
            return out
        end
        model = PDMPModel(d_boundary, FullGradient(_gap_grad!))
        flow = _make_boundary_flow()
        alg = _make_boundary_alg(; N=20, t_max=2.0)
        state = PDMPState(0.0, SkeletonPoint([0.0, 0.0], [1.0, 0.0]))
        trace_manager = PDMPSamplers.TraceManager(state, flow, alg, 0.0)
        cache = PDMPSamplers.add_gradient_to_cache(
            PDMPSamplers.initialize_cache(rng, flow, model.grad, alg, state.t[], state.ξ), state.ξ,
        )
        stats = PDMPSamplers.StatisticCounter()
        ctx = PDMPSamplers.BoundaryContext(
            copy(state.ξ.x), copy(state.ξ.θ), state.t[],
            1.0, 2.0, ErrorException("test"), BouncyParticle, GridThinningStrategy,
        )
        opts = SupportBoundaryOptions(;
            mode=:line_search_truncated_refresh,
            max_bisection_steps=0,
            clip_fraction=0.75,
            max_refresh_attempts=1,
            refresh_probe_time=0.0,
            min_safe_time=0.0,
        )

        result = PDMPSamplers._line_search_truncated_refresh_boundary!(
            rng, state, model, flow, alg, cache, stats, trace_manager, ctx, opts; phase=:main,
        )

        @test result === :refresh
        @test state.t[] ≈ 1.0
        @test state.ξ.x[1] ≈ 1.0
        @test stats.support_boundary_events == 1
        @test stats.support_boundary_refresh_attempts == 1
    end

    @testset "line_search_truncated_refresh backtracks after failed refresh probe" begin
        rng = PDMPSamplers.Random.Xoshiro(1234)
        function _narrow_grad!(out, x)
            x[1] >= 0.5 && error("Outside support: x[1] = $(x[1])")
            out .= x
            return out
        end
        model = PDMPModel(d_boundary, FullGradient(_narrow_grad!))
        flow = _make_boundary_flow()
        alg = _make_boundary_alg(; N=20, t_max=2.0)
        state = PDMPState(0.0, SkeletonPoint([0.0, 0.0], [1.0, 0.0]))
        cache = PDMPSamplers.add_gradient_to_cache(
            PDMPSamplers.initialize_cache(rng, flow, model.grad, alg, state.t[], state.ξ), state.ξ,
        )
        ctx = PDMPSamplers.BoundaryContext(
            copy(state.ξ.x), copy(state.ξ.θ), state.t[],
            0.0, 1.0, ErrorException("test"), BouncyParticle, GridThinningStrategy,
        )
        opts = SupportBoundaryOptions(; max_bisection_steps=10)

        found_valid_time, safe_time = PDMPSamplers._try_backtrack_valid_boundary_time!(
            state, model, flow, cache, ctx, 0.75, opts,
        )

        @test found_valid_time
        @test safe_time ≈ 0.375
        @test state.ξ.x[1] ≈ 0.375
    end

    @testset "line_search_truncated_refresh shrinks refresh probe time" begin
        function _probe_grad!(out, x)
            x[1] >= 0.5 && error("Outside support: x[1] = $(x[1])")
            out .= x
            return out
        end
        model = PDMPModel(d_boundary, FullGradient(_probe_grad!))
        flow = _make_boundary_flow()
        alg = _make_boundary_alg(; N=20, t_max=2.0)
        state = PDMPState(0.0, SkeletonPoint([0.49, 0.0], [1.0, 0.0]))
        rng = PDMPSamplers.Random.Xoshiro(4321)
        cache = PDMPSamplers.add_gradient_to_cache(
            PDMPSamplers.initialize_cache(rng, flow, model.grad, alg, state.t[], state.ξ), state.ξ,
        )
        opts = SupportBoundaryOptions(; refresh_probe_time=0.02, max_bisection_steps=10)

        @test !PDMPSamplers._short_forward_probe_is_valid(state, model, flow, cache, opts.refresh_probe_time)
        @test PDMPSamplers._short_forward_probe_is_valid(state, model, flow, cache, opts)
    end

    @testset "grid safety limit routes through support-boundary handling" begin
        rng = PDMPSamplers.Random.Xoshiro(42)
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg(; N=20, t_max=2.0)
        state = PDMPState(0.0, SkeletonPoint([0.0, 0.0], [1.0, 0.0]))
        trace_manager = PDMPSamplers.TraceManager(state, flow, alg, 0.0)
        cache = PDMPSamplers.add_gradient_to_cache(
            PDMPSamplers.initialize_cache(rng, flow, model.grad, alg, state.t[], state.ξ), state.ξ,
        )
        stats = PDMPSamplers.StatisticCounter()
        ctx = PDMPSamplers.BoundaryContext(
            copy(state.ξ.x), copy(state.ξ.θ), state.t[],
            0.0, 2.0, ErrorException("Safety limit reached"), BouncyParticle, GridThinningStrategy,
        )

        try
            PDMPSamplers._handle_grid_safety_limit!(
                rng, state, model, flow, alg, cache, stats, trace_manager, ctx,
                SupportBoundaryOptions(; mode=:line_search); phase=:main,
            )
            @test false
        catch err
            @test err isa SupportBoundaryError
            @test err.localized
            @test occursin("Grid thinning", err.message)
        end

        ctx_valid_horizon = PDMPSamplers.BoundaryContext(
            copy(state.ξ.x), copy(state.ξ.θ), state.t[],
            0.0, 0.5, ErrorException("Safety limit reached"), BouncyParticle, GridThinningStrategy,
        )
        try
            PDMPSamplers._handle_grid_safety_limit!(
                rng, state, model, flow, alg, cache, stats, trace_manager, ctx_valid_horizon,
                SupportBoundaryOptions(; mode=:line_search); phase=:main,
            )
            @test false
        catch err
            @test err isa SupportBoundaryError
            @test !err.localized
            io = IOBuffer()
            showerror(io, err)
            @test !occursin("Hint: use support_boundary_options", String(take!(io)))
        end

        opts = SupportBoundaryOptions(; mode=:line_search_truncated_refresh, max_refresh_attempts=1, refresh_probe_time=0.0)
        result = PDMPSamplers._handle_grid_safety_limit!(
            rng, state, model, flow, alg, cache, stats, trace_manager, ctx, opts; phase=:main,
        )

        @test result === :refresh
        @test stats.support_boundary_events == 1
        @test stats.support_boundary_refresh_attempts == 1
    end

    @testset "pdmp_sample — line_search_truncated_refresh unsupported flow" begin
        model = _make_boundary_model()
        flow = ZigZag(Matrix{Float64}(I, d_boundary, d_boundary), zeros(d_boundary))
        alg = _make_boundary_alg(; N=20, t_max=2.0)
        x0 = zeros(d_boundary)
        opts = SupportBoundaryOptions(; mode=:line_search_truncated_refresh)

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 0.0;
                progress=false, seed=42, support_boundary_options=opts)
            @test false
        catch err
            @test err isa SupportBoundaryError
            @test occursin("BPS-family", err.message)
        end
    end

    @testset "pdmp_sample — non-boundary HVP errors are not converted" begin
        function normal_grad!(out, x)
            out .= x
            return out
        end
        function bad_hvp!(out, x, v)
            error("bad hvp")
        end

        model = PDMPModel(d_boundary, FullGradient(normal_grad!), bad_hvp!)
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 10.0, 0.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:line_search))
            @test false
        catch err
            @test !(err isa SupportBoundaryError)
            @test occursin("bad hvp", sprint(showerror, err))
        end
    end

    @testset "pdmp_sample — PreconditionedBPS integration" begin
        model = _make_boundary_model()
        flow = PreconditionedBPS(Matrix{Float64}(I, d_boundary, d_boundary), zeros(d_boundary))
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 0.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:error))
        catch err
            @test err isa SupportBoundaryError
        end

        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 0.0;
                progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:line_search))
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == true
        end
    end

    @testset "pdmp_sample — default behavior unchanged (no keyword)" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        # Default behavior should be :error
        @test_throws SupportBoundaryError begin
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42)
        end
    end

    @testset "SupportBoundaryError — showerror" begin
        err = SupportBoundaryError(
            "test message",
            ErrorException("inner"),
            ZigZag,
            GridThinningStrategy,
            0.0, 1.0, 0.5, true,
        )
        buf = IOBuffer()
        showerror(buf, err)
        msg = String(take!(buf))
        @test occursin("SupportBoundaryError", msg)
        @test occursin("test message", msg)
        @test occursin("Estimated boundary time: 0.5", msg)
        @test occursin("ZigZag", msg)
        @test occursin("GridThinningStrategy", msg)

        # Custom unlocalized errors should not include the generic line-search hint.
        err2 = SupportBoundaryError(
            "test", ErrorException("inner"), ZigZag, GridThinningStrategy,
            0.0, 1.0, nothing, false,
        )
        buf2 = IOBuffer()
        showerror(buf2, err2)
        msg2 = String(take!(buf2))
        @test !occursin("line_search", msg2)

        # The default undefined-gradient diagnostic should still point users to localization.
        err3 = SupportBoundaryError(
            "The gradient or target density became undefined during forward probing.",
            ErrorException("inner"), ZigZag, GridThinningStrategy,
            0.0, 1.0, nothing, false,
        )
        buf3 = IOBuffer()
        showerror(buf3, err3)
        msg3 = String(take!(buf3))
        @test occursin("line_search", msg3)
    end

    @testset "pdmp_sample — valid model unaffected by boundary plumbing" begin
        # A model with full support should still work normally
        d = 2
        function normal_grad!(out, x)
            out .= x
            return out
        end
        model = PDMPModel(d, FullGradient(normal_grad!))
        flow = BouncyParticle(Matrix{Float64}(I, d, d), zeros(d))
        alg = _make_boundary_alg()
        x0 = randn(d)

        result = pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 5.0;
            progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:error))
        @test result isa PDMPChains
        @test length(result.traces) == 1

        result2 = pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 5.0;
            progress=false, seed=42, support_boundary_options=SupportBoundaryOptions(; mode=:line_search))
        @test result2 isa PDMPChains
    end

    @testset "pdmp_sample — multi-chain support" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)

        # Multi-chain with :error mode
        @test_throws SupportBoundaryError begin
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, n_chains=2, support_boundary_options=SupportBoundaryOptions(; mode=:error))
        end
    end

end
