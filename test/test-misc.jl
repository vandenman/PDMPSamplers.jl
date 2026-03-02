@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

import ADTypes
import DifferentiationInterface as DI
import ForwardDiff

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
end
