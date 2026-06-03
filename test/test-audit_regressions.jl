@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _audit_target_setup(d::Int)
    target = gen_data(Distributions.ZeroMeanIsoNormal, d)
    grad = FullGradient(Base.Fix1(neg_gradient!, target))
    model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
    alg = GridThinningStrategy()
    return target, model, alg
end

function _simple_trace(x0::Real)
    flow = BouncyParticle(1, 1.0)
    events = [
        PDMPEvent(0.0, [Float64(x0)], [1.0]),
        PDMPEvent(1.0, [Float64(x0) + 1.0], [1.0]),
    ]
    return PDMPTrace(events, flow)
end

@testset "Audit regressions" begin

    @testset "DensePreconditionedZigZag refresh is a no-op" begin
        Random.seed!(11)
        flow = DensePreconditionedZigZag(3)
        flow.metric.L .= [2.0 0.0 0.0; 0.5 3.0 0.0; -0.25 0.75 4.0]
        flow.metric.Linv .= inv(LowerTriangular(flow.metric.L))

        ξ = SkeletonPoint(randn(3), PDMPSamplers.initialize_velocity(flow, 3))
        θ_before = copy(ξ.θ)
        v_before = copy(flow.metric.v_canonical)

        PDMPSamplers.refresh_velocity!(ξ, flow)

        @test ξ.θ ≈ θ_before
        @test flow.metric.v_canonical ≈ v_before
        @test ξ.θ ≈ flow.metric.L * flow.metric.v_canonical
    end

    @testset "Factorized trace bounds replay scalar coordinates" begin
        flow = ZigZag(2)
        state = PDMPState(0.0, SkeletonPoint([1.0, -2.0], [0.5, -1.0]))
        trace = PDMPSamplers.FactorizedTrace(state, flow)
        push!(trace.events, PDMPSamplers.FactorizedEvent(2, 1.0, -3.0, 2.0))

        lo, hi = PDMPSamplers._trace_coordinate_bounds(trace, 2)

        @test lo == -3.0
        @test hi == -2.0
    end

    @testset "Immediate main-phase stop keeps a one-point trace" begin
        Random.seed!(22)
        d = 2
        _, model, alg = _audit_target_setup(d)
        flow = ZigZag(d)
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))

        trace, stats = pdmp_sample(
            ξ0,
            flow,
            model,
            alg,
            0.0,
            10.0,
            0.0;
            warmup_stop=FixedTimeCriterion(0.0),
            stop=FixedTimeCriterion(0.0),
            progress=false,
        )

        @test isone(length(trace))
        @test iszero(first(trace).time)
        @test iszero(last(trace).time)
        @test iszero(PDMPSamplers.first_event_time(trace))
        @test iszero(PDMPSamplers.last_event_time(trace))
        @test stats.stop_reason == :reached_time
    end

    @testset "PDMPChains iteration yields one item per chain" begin
        trace1 = _simple_trace(0.0)
        trace2 = _simple_trace(10.0)
        chains = PDMPChains([trace1, trace2], [:stats1, :stats2])

        @test collect(chains) == [(trace1, :stats1), (trace2, :stats2)]
    end

    @testset "TotalWallTimeCriterion copies share the same timer" begin
        criterion = TotalWallTimeCriterion(1.0)
        criterion_copy = copy(criterion)

        @test criterion.start_ns === criterion_copy.start_ns

        PDMPSamplers.initialize!(criterion_copy, nothing, nothing, nothing)

        @test criterion.start_ns[] == criterion_copy.start_ns[]
        @test criterion.start_ns[] != zero(UInt64)
    end

    @testset "Finite-difference curvature handles zero velocity" begin
        Random.seed!(33)
        d = 3
        target = gen_data(Distributions.MvNormal, d, 2.0)
        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), zeros(d)))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end

        fd_hvp = PDMPSamplers.FiniteDiffHVP(grad_func, zeros(d))
        rate_hvp, deriv_hvp = PDMPSamplers.get_rate_and_deriv(state, flow, fd_hvp)
        @test iszero(rate_hvp)
        @test iszero(deriv_hvp)
        @test all(isfinite, fd_hvp.hvp_buf)

        fd_vhv = PDMPSamplers.FiniteDiffVHV(grad_func, zeros(d))
        rate_vhv, deriv_vhv = PDMPSamplers.get_rate_and_deriv(state, flow, fd_vhv)
        @test iszero(rate_vhv)
        @test iszero(deriv_vhv)
    end
end
