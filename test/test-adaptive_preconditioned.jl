@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Adaptive Preconditioned Dynamics" begin

    @testset "PreconditionedZigZag adaptation quality" begin
        d = 5
        Random.seed!(12345)
        σ_true = rand(d) .* 2 .+ 0.3
        target = gen_data(Distributions.MvNormal, d, 100.0, zeros(d), σ_true, Matrix{Float64}(I(d)))

        flow = PreconditionedZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 100_000.0, 10_000.0; progress=show_progress)
        @test length(trace) > 100

        # scales should have been updated from ones(d)
        @test flow.metric.scale != ones(d)
        # adapted scales should approximate true σ
        for i in 1:d
            @test isapprox(flow.metric.scale[i], σ_true[i]; rtol=0.8)
        end

        # sampling quality: means close to zero
        trace_mean = mean(trace)
        for i in 1:d
            @test abs(trace_mean[i]) < 2.0
        end
    end

    @testset "PreconditionedBPS adaptation quality" begin
        d = 5
        Random.seed!(54321)
        σ_true = rand(d) .* 2 .+ 0.3
        target = gen_data(Distributions.MvNormal, d, 100.0, zeros(d), σ_true, Matrix{Float64}(I(d)))

        flow = PreconditionedBPS(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 100_000.0, 10_000.0; progress=show_progress)
        @test length(trace) > 100

        # scales should have been updated from ones(d)
        @test flow.metric.scale != ones(d)
        # adapted scales should approximate true σ
        for i in 1:d
            @test isapprox(flow.metric.scale[i], σ_true[i]; rtol=0.8)
        end

        # sampling quality: means close to zero
        trace_mean = mean(trace)
        for i in 1:d
            @test abs(trace_mean[i]) < 2.0
        end
    end

    @testset "PreconditionedZigZag N(μ, Σ)" begin
        d = 3
        Random.seed!(11111)
        target = gen_data(Distributions.MvNormal, d, 1.0)
        D = target.D

        flow = PreconditionedZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0, 5_000.0; progress=show_progress)
        @test length(trace) > 100

        trace_mean = mean(trace)
        true_mean = mean(D)
        for i in 1:d
            @test abs(trace_mean[i] - true_mean[i]) < 2.0
        end
    end

    @testset "PreconditionedBPS N(μ, Σ)" begin
        d = 3
        Random.seed!(22222)
        target = gen_data(Distributions.MvNormal, d, 1.0)
        D = target.D

        flow = PreconditionedBPS(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0, 5_000.0; progress=show_progress)
        @test length(trace) > 100

        trace_mean = mean(trace)
        true_mean = mean(D)
        for i in 1:d
            @test abs(trace_mean[i] - true_mean[i]) < 2.0
        end
    end

end
