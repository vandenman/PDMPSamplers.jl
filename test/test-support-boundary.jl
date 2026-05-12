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
        
        tau_star, tau_safe = PDMPSamplers.localize_support_boundary!(model, ctx, opts)
        
        # Boundary should be near x[1] = 1.0, i.e., t ≈ 1.0
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
                progress=false, seed=42, support_boundary_mode=:error)
        end
        
        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_mode=:error)
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
                progress=false, seed=42, support_boundary_mode=:line_search)
        end
        
        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_mode=:line_search)
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == true
            @test err.estimated_boundary_time !== nothing
            @test err.estimated_boundary_time > err.last_valid_time
            @test err.estimated_boundary_time <= err.first_invalid_time
        end
    end
    
    @testset "pdmp_sample — :line_search with custom options" begin
        model = _make_boundary_model()
        flow = _make_boundary_flow()
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)
        
        opts = SupportBoundaryOptions(; max_bisection_steps=10, time_rtol=1e-6, time_atol=1e-8)
        
        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_mode=:line_search,
                support_boundary_options=opts)
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
                progress=false, seed=42, support_boundary_mode=:error)
        end
        
        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 10.0;
                progress=false, seed=42, support_boundary_mode=:line_search)
        catch err
            @test err isa SupportBoundaryError
            @test err.localized == true
        end
    end
    
    @testset "pdmp_sample — PreconditionedBPS integration" begin
        model = _make_boundary_model()
        flow = PreconditionedBPS(Matrix{Float64}(I, d_boundary, d_boundary), zeros(d_boundary))
        alg = _make_boundary_alg()
        x0 = zeros(d_boundary)
        
        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 0.0;
                progress=false, seed=42, support_boundary_mode=:error)
        catch err
            @test err isa SupportBoundaryError
        end
        
        try
            pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 0.0;
                progress=false, seed=42, support_boundary_mode=:line_search)
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
        
        # Default (no support_boundary_mode) should behave like :error
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
        
        # Unlocalized error should include hint
        err2 = SupportBoundaryError(
            "test", ErrorException("inner"), ZigZag, GridThinningStrategy,
            0.0, 1.0, nothing, false,
        )
        buf2 = IOBuffer()
        showerror(buf2, err2)
        msg2 = String(take!(buf2))
        @test occursin("line_search", msg2)
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
            progress=false, seed=42, support_boundary_mode=:error)
        @test result isa PDMPChains
        @test length(result.traces) == 1
        
        result2 = pdmp_sample(x0, flow, model, alg, 0.0, 100.0, 5.0;
            progress=false, seed=42, support_boundary_mode=:line_search)
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
                progress=false, seed=42, n_chains=2, support_boundary_mode=:error)
        end
    end
    
end
