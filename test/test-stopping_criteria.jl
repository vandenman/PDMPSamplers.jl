@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _stopping_setup(; d::Int=3)
    target = gen_data(Distributions.ZeroMeanIsoNormal, d)
    flow = ZigZag(inv(Symmetric(cov(target.D))), mean(target.D))
    grad = FullGradient(Base.Fix1(neg_gradient!, target))
    model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
    alg = GridThinningStrategy()
    ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
    return ξ0, flow, model, alg
end

mutable struct MockCriterion <: PDMPSamplers.StoppingCriterion
    fired::Bool
    reason::Symbol
end

PDMPSamplers.is_satisfied(c::MockCriterion, state, trace_manager, stats) = c.fired
PDMPSamplers.stop_reason(c::MockCriterion) = c.reason

@testset "Stopping criteria" begin

    @testset "Constructors and helper" begin
        @test_throws ArgumentError EventCountCriterion(0)
        @test_throws ArgumentError WallTimeCriterion(0.0)
        @test_throws ArgumentError WallTimeCriterion(Inf)
        @test_throws ArgumentError TotalWallTimeCriterion(0.0)
        @test_throws ArgumentError TotalWallTimeCriterion(Inf)
        @test_throws ArgumentError ESSCriterion(0.0)
        @test_throws ArgumentError ESSCriterion(1.0; check_every=0)
        @test_throws ArgumentError ESSCriterion(1.0; min_trace_length=1)
        @test_throws ArgumentError ESSCriterion(1.0; trace_selector=:invalid)
        @test_throws ArgumentError OnlineESSCriterion(0.0)
        @test_throws ArgumentError OnlineESSCriterion(1.0; check_every=0)
        @test_throws ArgumentError OnlineESSCriterion(1.0; min_samples=1)
        @test_throws ArgumentError OnlineESSCriterion(1.0; batch_size=0)
        @test_throws ArgumentError OnlineESSCriterion(1.0; trace_selector=:invalid)
        @test_throws ArgumentError AnyCriterion()
        @test_throws ArgumentError AllCriteria()

        @test stop_after(; T=10.0) isa FixedTimeCriterion
        @test stop_after(; events=100) isa EventCountCriterion
        @test stop_after(; ess=50.0, events=100) isa AnyCriterion
        @test stop_after(; wall_seconds=0.1) isa WallTimeCriterion
        @test stop_after(; ess=50.0, ess_mode=:online) isa OnlineESSCriterion
        @test stop_after(; ess=50.0, ess_mode=:online, events=100) isa AnyCriterion
        @test_throws ArgumentError stop_after(; ess=50.0, ess_mode=:invalid)
        @test_throws ArgumentError stop_after()
    end

    @testset "Backward compatible default" begin
        Random.seed!(1201)
        ξ0a, flowa, modela, alga = _stopping_setup()
        trace_a, stats_a = pdmp_sample(ξ0a, flowa, modela, alga, 0.0, 200.0, 50.0; progress=false)

        Random.seed!(1201)
        ξ0b, flowb, modelb, algb = _stopping_setup()
        trace_b, stats_b = pdmp_sample(
            ξ0b,
            flowb,
            modelb,
            algb,
            0.0,
            200.0,
            50.0;
            stop=FixedTimeCriterion(200.0),
            warmup_stop=FixedTimeCriterion(50.0),
            progress=false
        )

        @test length(trace_a) == length(trace_b)
        @test stats_a.reflections_events == stats_b.reflections_events
        @test stats_a.reflections_accepted == stats_b.reflections_accepted
        @test stats_a.refreshment_events == stats_b.refreshment_events
        @test stats_a.sticky_events == stats_b.sticky_events
        @test stats_a.stop_reason == :reached_time
        @test stats_b.stop_reason == :reached_time
    end

    @testset "Unit criteria in sampler" begin
        Random.seed!(991)
        ξ0, flow, model, alg = _stopping_setup()
        trace_event, stats_event = pdmp_sample(
            ξ0,
            flow,
            model,
            alg,
            0.0,
            10_000.0,
            0.0;
            warmup_stop=FixedTimeCriterion(0.0),
            stop=EventCountCriterion(20),
            progress=false
        )
        total_events = stats_event.reflections_events + stats_event.refreshment_events + stats_event.sticky_events
        @test total_events >= 20
        @test stats_event.stop_reason == :reached_event_budget
        @test length(trace_event) > 1

        Random.seed!(992)
        ξ0, flow, model, alg = _stopping_setup()
        _, stats_wall = pdmp_sample(
            ξ0,
            flow,
            model,
            alg,
            0.0,
            10_000.0,
            0.0;
            warmup_stop=FixedTimeCriterion(0.0),
            stop=WallTimeCriterion(1e-6),
            progress=false
        )
        @test stats_wall.stop_reason == :reached_wall_time

        Random.seed!(993)
        ξ0, flow, model, alg = _stopping_setup()
        trace_ess, stats_ess = pdmp_sample(
            ξ0,
            flow,
            model,
            alg,
            0.0,
            10_000.0,
            0.0;
            warmup_stop=FixedTimeCriterion(0.0),
            stop=ESSCriterion(0.1; check_every=1, min_trace_length=5),
            progress=false
        )
        @test stats_ess.stop_reason == :reached_ess
        @test length(trace_ess) >= 5

        Random.seed!(994)
        ξ0, flow, model, alg = _stopping_setup()
        _, stats_total_wall = pdmp_sample(
            ξ0,
            flow,
            model,
            alg,
            0.0,
            10_000.0,
            0.0;
            warmup_stop=FixedTimeCriterion(0.0),
            stop=TotalWallTimeCriterion(1e-6),
            progress=false
        )
        @test stats_total_wall.stop_reason == :reached_wall_time

        Random.seed!(995)
        ξ0, flow, model, alg = _stopping_setup()
        trace_online, stats_online = pdmp_sample(
            ξ0,
            flow,
            model,
            alg,
            0.0,
            10_000.0,
            0.0;
            warmup_stop=FixedTimeCriterion(0.0),
            stop=OnlineESSCriterion(0.1; check_every=1, min_samples=5, batch_size=3),
            progress=false
        )
        @test stats_online.stop_reason == :reached_ess
        @test length(trace_online) >= 5
    end

    @testset "TotalWallTimeCriterion unit" begin
        c = TotalWallTimeCriterion(1.0)
        @test c.start_ns[] == zero(UInt64)
        @test !PDMPSamplers.is_satisfied(c, nothing, nothing, nothing)
        PDMPSamplers.initialize!(c, nothing, nothing, nothing)
        @test c.start_ns[] != zero(UInt64)
        t1 = c.start_ns[]
        PDMPSamplers.initialize!(c, nothing, nothing, nothing)
        @test c.start_ns[] == t1

        c_fast = TotalWallTimeCriterion(1e-9)
        PDMPSamplers.initialize!(c_fast, nothing, nothing, nothing)
        sleep(0.001)
        @test PDMPSamplers.is_satisfied(c_fast, nothing, nothing, nothing)
        @test PDMPSamplers.stop_reason(c_fast) == :reached_wall_time

        c_copy = copy(c)
        @test c !== c_copy
        @test c.start_ns !== c_copy.start_ns
        @test c.start_ns[] == c_copy.start_ns[]
    end

    @testset "Combinators and copies" begin
        c1 = MockCriterion(false, :one)
        c2 = MockCriterion(true, :two)

        any_c = AnyCriterion(c1, c2)
        all_c = AllCriteria(c1, c2)

        @test PDMPSamplers.is_satisfied(any_c, nothing, nothing, nothing)
        @test !PDMPSamplers.is_satisfied(all_c, nothing, nothing, nothing)
        @test PDMPSamplers.stop_reason(any_c, nothing, nothing, nothing) == :two
        @test PDMPSamplers.stop_reason(all_c) == :all_criteria_satisfied

        crit = AnyCriterion(WallTimeCriterion(1.0), ESSCriterion(10.0))
        crit_copy = copy(crit)
        @test crit !== crit_copy
        @test crit.criteria[1] !== crit_copy.criteria[1]
        @test crit.criteria[2] !== crit_copy.criteria[2]

        crit_all = AllCriteria(WallTimeCriterion(1.0), ESSCriterion(10.0))
        crit_all_copy = copy(crit_all)
        @test crit_all !== crit_all_copy
        @test crit_all.criteria[1] !== crit_all_copy.criteria[1]
        @test crit_all.criteria[2] !== crit_all_copy.criteria[2]

        crit_online = OnlineESSCriterion(10.0; check_every=5, min_samples=4, batch_size=2)
        crit_online_copy = copy(crit_online)
        @test crit_online !== crit_online_copy
        @test crit_online.batch_sum !== crit_online_copy.batch_sum
        @test crit_online.target_ess == crit_online_copy.target_ess
    end

    @testset "Warmup phase routing and multi-chain criterion isolation" begin
        Random.seed!(77)
        target = gen_data(Distributions.ZeroMeanIsoNormal, 3)
        flow_nf = BouncyParticle(inv(Symmetric(cov(target.D))), mean(target.D))
        grad_nf = FullGradient(Base.Fix1(neg_gradient!, target))
        model_nf = PDMPModel(3, grad_nf, Base.Fix1(neg_hvp!, target))
        alg_nf = GridThinningStrategy()
        ξ0_nf = SkeletonPoint(randn(3), PDMPSamplers.initialize_velocity(flow_nf, 3))
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(flow_nf, model_nf, alg_nf, 0.0, ξ0_nf)
        trace_manager = PDMPSamplers.TraceManager(state, flow_nf, alg_nf, 0.0)

        PDMPSamplers.record_event!(trace_manager, state, flow_nf, nothing; phase=:warmup)
        PDMPSamplers.record_event!(trace_manager, state, flow_nf, nothing; phase=:main)

        @test length(PDMPSamplers.get_warmup_trace(trace_manager)) == 1
        @test length(PDMPSamplers.get_main_trace(trace_manager)) == 1

        base_stop = AnyCriterion(EventCountCriterion(30), WallTimeCriterion(5.0))
        chains = pdmp_sample(
            ξ0_nf,
            flow_nf,
            model_nf,
            alg_nf,
            0.0,
            1_000.0,
            0.0;
            n_chains=2,
            threaded=false,
            warmup_stop=EventCountCriterion(10),
            stop=base_stop,
            progress=false
        )
        @test n_chains(chains) == 2
        @test base_stop.criteria[2].start_ns == zero(UInt64)
    end
end
