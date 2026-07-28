@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

import ADTypes
import DifferentiationInterface as DI
import ForwardDiff

struct TestNoopCounter <: PDMPSamplers.AbstractStatisticCounter end

mutable struct ActiveSetRecorder
    free::BitVector
    calls::Int
end
ActiveSetRecorder(n::Integer) = ActiveSetRecorder(falses(n), 0)
function (r::ActiveSetRecorder)(out, x)
    copyto!(out, x)
    return out
end
PDMPSamplers.set_active_set!(r::ActiveSetRecorder, free::BitVector) = (r.free .= free; r.calls += 1; nothing)

@testset "Miscellaneous" begin

    @testset "HVP sign for FullGradient path" begin
        d = 3
        A = [2.0 0.5 0.1; 0.5 3.0 0.2; 0.1 0.2 1.5]
        f_grad!(out, x) = (mul!(out, A, x); out)
        grad = FullGradient(f_grad!)
        backend = DI.AutoForwardDiff()
        model = PDMPModel(d, grad, backend, true)

        x_test = [1.0, -0.5, 2.0]
        v_test = [0.3, 0.7, -0.2]
        result = model.hvp(x_test, v_test)
        expected = A * v_test
        @test result ≈ expected
    end

    @testset "_integrate on small traces" begin
        flow = ZigZag(3)
        d = 3
        trace_empty = PDMPTrace(Float64[], PDMPSamplers.ElasticMatrix{Float64}(undef, d, 0), PDMPSamplers.ElasticMatrix{Float64}(undef, d, 0), flow)
        @test_throws ErrorException("Cannot compute statistics on an empty trace") mean(trace_empty)

        x0 = [1.0, 2.0, 3.0]
        θ0 = [1.0, -1.0, 1.0]
        trace_single = PDMPTrace([PDMPEvent(0.0, x0, θ0)], flow)
        @test_throws ErrorException("Cannot compute statistics on a trace with fewer than 2 events") mean(trace_single)
    end

    @testset "Boomerang freezing_time" begin
        flow_zero_mu = Boomerang(1)

        @testset "μ=0, θ=0, x>0 → π/2" begin
            ξ = SkeletonPoint([1.0], [0.0])
            @test PDMPSamplers.freezing_time(ξ, flow_zero_mu, 1) ≈ π / 2
        end

        @testset "μ=0, θ=0, x<0 → π/2" begin
            ξ = SkeletonPoint([-1.0], [0.0])
            @test PDMPSamplers.freezing_time(ξ, flow_zero_mu, 1) ≈ π / 2
        end

        @testset "μ=0, θ=0, x=0 → Inf" begin
            ξ = SkeletonPoint([0.0], [0.0])
            @test PDMPSamplers.freezing_time(ξ, flow_zero_mu, 1) == Inf
        end

        @testset "x=2μ singularity → finite positive" begin
            μ_val = 1.5
            flow_nonzero = Boomerang(Diagonal([1.0]), [μ_val])
            ξ = SkeletonPoint([2μ_val], [1.0])
            t = PDMPSamplers.freezing_time(ξ, flow_nonzero, 1)
            @test isfinite(t)
            @test t > 0
        end

        @testset "General cases: compare to brute-force root finding" begin
            for (x, θ, μ) in [(2.0, 1.0, 0.0), (0.5, -1.5, 0.0),
                               (1.0, 0.5, 0.3), (3.0, -1.0, 1.0)]
                flow_i = Boomerang(Diagonal([1.0]), [μ])
                ξ = SkeletonPoint([x], [θ])
                t_computed = PDMPSamplers.freezing_time(ξ, flow_i, 1)
                trajectory(t) = (x - μ) * cos(t) + θ * sin(t) + μ
                if isfinite(t_computed)
                    @test abs(trajectory(t_computed)) < 1e-10
                    @test t_computed > 0
                end
            end
        end
    end

    @testset "ZigZag reflect! coordinate selection distribution" begin
        d = 4
        flow = ZigZag(d)
        x = [1.0, 2.0, 3.0, 4.0]
        θ = ones(d)
        ∇ϕ = [1.0, 2.0, 4.0, 8.0]

        grad = FullGradient((out, x) -> out .= x)
        ξ = SkeletonPoint(copy(x), copy(θ))
        cache = PDMPSamplers.add_gradient_to_cache(
            PDMPSamplers.initialize_cache(flow, grad, ThinningStrategy(GlobalBounds(1.0, d)), 0.0, ξ),
            ξ
        )

        n_samples = 50_000
        counts = zeros(Int, d)
        for _ in 1:n_samples
            ξ_test = SkeletonPoint(copy(x), ones(d))
            PDMPSamplers.reflect!(ξ_test, copy(∇ϕ), flow, cache)
            for i in 1:d
                if ξ_test.θ[i] != 1.0
                    counts[i] += 1
                    break
                end
            end
        end

        rates = [max(0.0, θ[i] * ∇ϕ[i]) for i in 1:d]
        expected_probs = rates ./ sum(rates)
        empirical_probs = counts ./ n_samples
        for i in 1:d
            @test abs(empirical_probs[i] - expected_probs[i]) < 0.02
        end
    end

    @testset "Sticky constructor error for bare Function κ" begin
        @test_throws ArgumentError Sticky(GridThinningStrategy(), (i, x, γ, θ) -> 1.0)
    end

    @testset "Statistic counter composition and defaults" begin
        c1 = PDMPSamplers.GridThinningCounter()
        c2 = PDMPSamplers.GridThinningCounter()
        auto1 = PDMPSamplers.CertifiedAutoCounter()
        auto2 = PDMPSamplers.CertifiedAutoCounter()
        basic = PDMPSamplers.BasicEventCounter()
        multi = PDMPSamplers.MultiCounter((c1, c2))
        auto_multi = PDMPSamplers.MultiCounter((auto1, auto2))
        noop = TestNoopCounter()

        @test basic.last_rejected == false
        phase = PDMPSamplers.PhaseSummaryCounter()
        @test phase.stop_reason === :none
        @test PDMPSamplers._get_counter_grid_acceptance_tests(noop) == 0
        @test PDMPSamplers._get_counter_auto_area_saved(noop) == 0.0
        @test !PDMPSamplers._get_counter_last_rejected(noop)

        PDMPSamplers._inc_counter_grid_acceptance_tests(c1)
        PDMPSamplers._inc_counter_grid_acceptance_tests(c2)
        PDMPSamplers._inc_counter_grid_acceptance_tests(c2)
        @test PDMPSamplers._get_counter_grid_acceptance_tests(multi) == 3

        PDMPSamplers._inc_counter_auto_area_saved(auto1, 0.25)
        PDMPSamplers._inc_counter_auto_area_saved(auto2, 0.5)
        @test PDMPSamplers._get_counter_auto_area_saved(auto_multi) == 0.75

        PDMPSamplers._set_counter_last_rejected(basic, true)
        @test PDMPSamplers._get_counter_last_rejected(PDMPSamplers.MultiCounter((c1, basic)))
        @test multi.grid_acceptance_tests == 1
        multi.grid_acceptance_tests = 7
        @test c1.grid_acceptance_tests == 7

        @test_throws ErrorException multi.not_a_counter_field
        @test_throws ErrorException setproperty!(multi, :not_a_counter_field, 1)

        monitor = PDMPSamplers.HealthMonitor(; consecutive_reject_limit=1)
        PDMPSamplers.check_health!(monitor, basic)
        @test_throws ErrorException PDMPSamplers.check_health!(monitor, basic)

        @test PDMPSamplers._counter_struct_name(:MyCounter) === :MyCounter
        @test PDMPSamplers._counter_struct_name(:(MyCounter <: PDMPSamplers.AbstractStatisticCounter)) === :MyCounter
        @test PDMPSamplers._counter_struct_name(:(MyCounter{T})) === :MyCounter
        @test_throws ErrorException PDMPSamplers._counter_struct_name(1)

        component = PDMPSamplers.ComponentwiseAffineCounter()
        PDMPSamplers._record_counter_componentwise_cell_diagnostics!(
            component, 3, 2, 1, 0.75, 0.25)
        @test component.componentwise_proposed_breakpoints_per_cell == [3.0]
        @test component.componentwise_segments_per_cell == [2.0]
        @test component.componentwise_zero_crossings_per_cell == [1.0]
        @test component.componentwise_area_saved_per_cell == [0.75]
        @test component.componentwise_area_saved_fraction_per_cell == [0.25]
        @test isnothing(PDMPSamplers._record_counter_componentwise_cell_diagnostics!(nothing, 1, 2, 3))
    end

    @testset "Gradient purpose counters" begin
        stats = PDMPSamplers.DevelStatisticCounter()
        x = [1.0, 2.0]
        out = zeros(2)

        full = PDMPSamplers.with_stats(FullGradient((out, x) -> copyto!(out, x)), stats)
        PDMPSamplers.compute_gradient!(full, x, out)
        @test stats.∇f_calls == 1
        @test stats.full_gradient_calls == 1
        @test stats.stochastic_gradient_calls == 0

        sub = PDMPSamplers.with_stats(SubsampledGradient(
            (out, x) -> (out .= 2 .* x),
            n -> nothing,
            trace -> nothing,
            (out, x) -> (out .= 3 .* x),
            1,
            0,
            true,
        ), stats)
        PDMPSamplers.compute_gradient!(sub, x, out)
        PDMPSamplers.compute_gradient_for_reflection!(sub, x, out)
        @test stats.∇f_calls == 3
        @test stats.stochastic_gradient_calls == 1
        @test stats.full_reflection_gradient_calls == 1

        prior = PDMPSamplers.with_stats((out, x) -> copyto!(out, -x), stats, Val(:prior_gradient))
        prior(out, x)
        @test stats.∇f_calls == 4
        @test stats.prior_gradient_calls == 1

        fd_probe = PDMPSamplers.WithFDCurvatureStats(
            PDMPSamplers.with_stats((out, x) -> copyto!(out, x), stats, Val(:stochastic_gradient)),
            stats,
        )
        fd_probe(out, x)
        @test stats.∇f_calls == 5
        @test stats.stochastic_gradient_calls == 2
        @test stats.fd_curvature_gradient_calls == 1

        hvp = PDMPSamplers.WithStatsHVP((x, v) -> v, stats)
        @test hvp(x, x) == x
        @test stats.∇²f_calls == 1

        PDMPSamplers._record_phase_stats!(
            stats, :main, 0, 0, 0, 0, 0, 0, 0, 0, 0, time_ns())
        @test stats.main_gradient_calls == stats.∇f_calls
        @test stats.main_hessian_calls == stats.∇²f_calls
        @test stats.main_stochastic_gradient_calls == stats.stochastic_gradient_calls
        @test stats.main_full_gradient_calls == stats.full_gradient_calls
        @test stats.main_full_reflection_gradient_calls == stats.full_reflection_gradient_calls
        @test stats.main_prior_gradient_calls == stats.prior_gradient_calls
        @test stats.main_fd_curvature_gradient_calls == stats.fd_curvature_gradient_calls
        @test stats.main_exact_curvature_calls == stats.∇²f_calls
    end

    @testset "Active-set propagation through wrappers" begin
        free = BitVector([true, false, true])

        fixed_rec = ActiveSetRecorder(3)
        PDMPSamplers.set_active_set!(Base.Fix2((x, rec) -> rec, fixed_rec), free)
        @test fixed_rec.free == free
        @test fixed_rec.calls == 1

        full_rec = ActiveSetRecorder(3)
        coord_rec = ActiveSetRecorder(3)
        PDMPSamplers.set_active_set!(FullGradient(full_rec), free)
        PDMPSamplers.set_active_set!(CoordinateWiseGradient(coord_rec), free)
        @test full_rec.free == free
        @test coord_rec.free == free

        stoch_rec = ActiveSetRecorder(3)
        full_sub_rec = ActiveSetRecorder(3)
        sub = SubsampledGradient(stoch_rec, n -> nothing, tr -> nothing, FullGradient(full_sub_rec), 1, 0, false, 0.25)
        PDMPSamplers.set_active_set!(sub, free)
        @test stoch_rec.free == free
        @test full_sub_rec.free == free

        stats = PDMPSamplers.StatisticCounter()
        wrapped_rec = ActiveSetRecorder(3)
        PDMPSamplers.set_active_set!(PDMPSamplers.with_stats(wrapped_rec, stats), free)
        PDMPSamplers.set_active_set!(PDMPSamplers.WithFDCurvatureStats(wrapped_rec, stats), free)
        @test wrapped_rec.calls == 2

        vhv_rec = ActiveSetRecorder(3)
        joint_rec = ActiveSetRecorder(3)
        PDMPSamplers.set_active_set!(PDMPSamplers.WithStatsVHV(vhv_rec, stats), free)
        PDMPSamplers.set_active_set!(PDMPSamplers.WithStatsJoint(joint_rec, stats), free)
        @test vhv_rec.free == free
        @test joint_rec.free == free

        unavailable = PDMPModel(3, FullGradient((out, x) -> copyto!(out, x)))
        @test !PDMPSamplers._potential_available(unavailable)
    end

    @testset "FiniteDiffVHV counts only shifted curvature gradients" begin
        stats = PDMPSamplers.DevelStatisticCounter()
        grad = PDMPSamplers.with_stats(x -> copy(x), stats, Val(:ordinary_full_gradient))
        fd = PDMPSamplers.FiniteDiffVHV(grad, zeros(1), zeros(1), zeros(1), stats)
        state = PDMPState(0.0, SkeletonPoint([1.0], [1.0]))
        flow = BouncyParticle(1, 0.0)

        rate, deriv = PDMPSamplers.get_rate_and_deriv(state, flow, fd, false)

        @test rate ≈ 1.0
        @test deriv ≈ 1.0
        @test stats.∇f_calls == 2
        @test stats.full_gradient_calls == 2
        @test stats.fd_curvature_gradient_calls == 1
    end

    # @testset "PreconditionedDynamics with warmup adaptation" begin
    #     d = 3
    #     target = gen_data(Distributions.MvNormal, d, 1.0)

    #     flow = PreconditionedZigZag(d)
    #     grad = FullGradient(Base.Fix1(neg_gradient!, target))
    #     model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
    #     alg = GridThinningStrategy()

    #     Random.seed!(123)
    #     ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
    #     t_warmup = 2_000.0
    #     T_run = 10_000.0
    #     trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run, t_warmup; progress=false)
    #     @test length(trace) > 100

    #     # scales should have been updated from the initial ones(d)
    #     @test flow.metric.scale != ones(d)
    # end

    @testset "Sticky with partial can_stick" begin
        d = 4
        Random.seed!(789)

        D_ss, κ_full, slab_target = gen_data(SpikeAndSlabDist{Bernoulli,ZeroMeanIsoNormal}, d)
        # only allow variables 1 and 3 to stick
        κ_partial = copy(κ_full)
        κ_partial[2] = Inf
        κ_partial[4] = Inf

        flow = ZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, slab_target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, slab_target))
        alg = Sticky(GridThinningStrategy(), κ_partial)

        @test alg.can_stick == BitVector([true, false, true, false])

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=false)
        @test stats.sticky_events > 0
        @test length(trace) > 100

        # variables 2 and 4 should never be exactly zero (they can't stick)
        ip = inclusion_probs(trace)
        @test ip[2] ≈ 1.0 atol=1e-10
        @test ip[4] ≈ 1.0 atol=1e-10
    end

    @testset "Sticky with BPS (all-frozen branch)" begin
        d = 2
        Random.seed!(321)
        D_ss, κ, slab_target = gen_data(SpikeAndSlabDist{Bernoulli,ZeroMeanIsoNormal}, d)

        # use high κ to make freezing very likely → all-frozen branch
        κ_high = κ .* 100.0
        flow = BouncyParticle(d, 0.0) # no refresh to not interfere
        grad = FullGradient(Base.Fix1(neg_gradient!, slab_target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, slab_target))
        alg = Sticky(GridThinningStrategy(), κ_high)

        ξ0 = SkeletonPoint(0.01 .* randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 10_000.0; progress=false)
        @test stats.sticky_events > 0
    end

    @testset "pdmp_sample with pre-built model vector" begin
        d = 2
        μ_true = [1.0, -1.0]
        A = [2.0 0.5; 0.5 1.5]
        neg_grad!(out, x) = (mul!(out, A, x .- μ_true); out)
        make_model() = PDMPModel(d, FullGradient(neg_grad!), nothing, nothing, false, false)

        flow = ZigZag(d)
        alg  = GridThinningStrategy()
        models = [make_model(), make_model()]

        chains = pdmp_sample(d, flow, models, alg, 0.0, 2_000.0; progress=false, threaded=false)
        @test length(chains.traces) == 2
        m = mean(chains)
        @test m ≈ μ_true atol = 1.25
    end
end
