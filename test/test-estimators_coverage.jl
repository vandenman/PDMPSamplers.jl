@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Estimators coverage" begin

    @testset "MutableBoomerang trapezoidal mean integration" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = AdaptiveBoomerang(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        m = mean(trace)
        @test length(m) == d
        @test all(isfinite, m)
        @test m ≈ μ_true atol = 0.5
    end

    @testset "Boomerang var/cov integration" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Σ_true = cov(target.D)
        Γ_true = inv(Symmetric(Σ_true))

        flow = Boomerang(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        v = var(trace)
        @test length(v) == d
        @test all(v .> 0)
        @test v ≈ diag(Σ_true) rtol = 0.3

        C = cov(trace)
        @test size(C) == (d, d)
        @test C ≈ Σ_true rtol = 0.5
    end

    @testset "Boomerang inclusion_probs" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = Boomerang(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 10_000.0; progress=show_progress)

        ip = inclusion_probs(trace)
        @test length(ip) == d
        @test all(0 .≤ ip .≤ 1)
        @test all(ip .> 0.9)  # non-sparse target → always included
    end

    @testset "_time_below_segment for Boomerang" begin
        boom = Boomerang(3)
        μj = 0.0

        # Below the full circle → time = τ
        @test PDMPSamplers._time_below_segment(boom, 0.0, 1.0, 1.0, 100.0, μj) ≈ 1.0
        # Above the full circle → time = 0
        @test PDMPSamplers._time_below_segment(boom, 0.0, 1.0, 1.0, -100.0, μj) ≈ 0.0
        # At the center, half circle → about half time
        result = PDMPSamplers._time_below_segment(boom, 1.0, 0.0, 2π, 0.0, μj)
        @test 0.0 < result < 2π
    end

    @testset "_time_below_segment for PreconditionedDynamics forwarding" begin
        zz = ZigZag(3)
        precond_zz = PDMPSamplers.PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(), zz)
        result = PDMPSamplers._time_below_segment(precond_zz, 0.0, 1.0, 2.0, 1.0)
        expected = PDMPSamplers._time_below_segment(zz, 0.0, 1.0, 2.0, 1.0)
        @test result ≈ expected
    end

    @testset "Boomerang CDF" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = Boomerang(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        for j in 1:d
            @test PDMPSamplers.cdf(trace, μ_true[j]; coordinate=j) ≈ 0.5 atol = 0.15
        end
    end

    @testset "Boomerang quantile (bisection)" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        σ_true = sqrt.(diag(cov(target.D)))
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = Boomerang(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        # scalar quantile
        q50 = quantile(trace, 0.5)
        @test length(q50) == d
        @test q50 ≈ μ_true atol = 0.5

        # per-coordinate
        for j in 1:d
            q = quantile(trace, 0.5; coordinate=j)
            @test q ≈ μ_true[j] atol = 0.5
        end

        # vector quantile (exercises _quantile_boomerang_vector for PDMPTrace)
        ps = [0.1, 0.5, 0.9]
        for j in 1:d
            q_vec = quantile(trace, ps; coordinate=j)
            @test length(q_vec) == 3
            @test q_vec[1] < q_vec[2] < q_vec[3]
        end
    end

    @testset "_trace_coordinate_bounds for Boomerang" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = Boomerang(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 5_000.0; progress=show_progress)

        for j in 1:d
            lo, hi = PDMPSamplers._trace_coordinate_bounds(trace, j)
            @test lo < hi
            @test lo < μ_true[j]
            @test hi > μ_true[j]
        end
    end

    @testset "PreconditionedDynamics _integrate_segment forwarding" begin
        zz = ZigZag(3)
        precond_zz = PDMPSamplers.PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(), zz)

        x0 = [1.0, 2.0, 3.0]
        x1 = [1.5, 2.5, 3.5]
        θ0 = [1.0, 1.0, 1.0]
        θ1 = [-1.0, 1.0, -1.0]
        t0 = 0.0
        t1 = 0.5

        result = PDMPSamplers._integrate_segment(Statistics.mean, precond_zz, x0, x1, θ0, θ1, t0, t1)
        expected = PDMPSamplers._integrate_segment(Statistics.mean, zz, x0, x1, θ0, θ1, t0, t1)
        @test result ≈ expected
    end

    @testset "FactorizedTrace mean fast path" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = ZigZag(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        @test trace isa PDMPSamplers.FactorizedTrace

        # Compare FactorizedTrace fast-path with PDMPTrace conversion
        pdmp_trace = PDMPTrace(trace)
        m_fact = mean(trace)
        m_pdmp = mean(pdmp_trace)
        @test m_fact ≈ m_pdmp rtol = 0.01

        v_fact = var(trace)
        v_pdmp = var(pdmp_trace)
        @test v_fact ≈ v_pdmp rtol = 0.05

        ip_fact = inclusion_probs(trace)
        ip_pdmp = inclusion_probs(pdmp_trace)
        @test ip_fact ≈ ip_pdmp rtol = 0.01
    end

    @testset "FactorizedTrace CDF and quantile" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = ZigZag(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        @test trace isa PDMPSamplers.FactorizedTrace

        for j in 1:d
            @test PDMPSamplers.cdf(trace, μ_true[j]; coordinate=j) ≈ 0.5 atol = 0.15
        end

        # vector quantile for FactorizedTrace
        ps = [0.25, 0.5, 0.75]
        for j in 1:d
            q_vec = quantile(trace, ps; coordinate=j)
            @test length(q_vec) == 3
            @test q_vec[1] < q_vec[2] < q_vec[3]
        end
    end

    @testset "_collect_sweep_events for FactorizedTrace" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = ZigZag(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 10_000.0; progress=show_progress)

        @test trace isa PDMPSamplers.FactorizedTrace

        for j in 1:d
            total_time, dc, pm = PDMPSamplers._collect_sweep_events(trace, j)
            @test total_time > 0
            @test length(dc) > 0
        end
    end

    @testset "_underlying_flow dispatch" begin
        zz = ZigZag(3)
        @test PDMPSamplers._underlying_flow(zz) === zz

        precond_zz = PDMPSamplers.PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(), zz)
        @test PDMPSamplers._underlying_flow(precond_zz) === zz
    end

    @testset "Quantile domain errors" begin
        d = 3
        flow = ZigZag(d)
        events = [
            PDMPEvent(0.0, randn(d), ones(d)),
            PDMPEvent(1.0, randn(d), -ones(d)),
            PDMPEvent(2.0, randn(d), ones(d)),
        ]
        trace = PDMPTrace(events, flow)

        @test_throws DomainError quantile(trace, 0.0; coordinate=1)
        @test_throws DomainError quantile(trace, 1.0; coordinate=1)
        @test_throws DomainError quantile(trace, [-0.1, 0.5]; coordinate=1)
    end

    @testset "ESS edge cases" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        μ_true = mean(target.D)
        Γ_true = inv(Symmetric(cov(target.D)))

        flow = ZigZag(Γ_true, μ_true)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(μ_true .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 50_000.0; progress=show_progress)

        e = ess(trace; n_batches=10)
        @test length(e) == d
        @test all(e .> 0)
    end
end
