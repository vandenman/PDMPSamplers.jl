@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Adaptive Boomerang (Phase 1)" begin

    # ──────────────────────────────────────────────────────────────────────
    # Test 0: Basic type and constructor tests
    # ──────────────────────────────────────────────────────────────────────
    @testset "Types and constructors" begin
        d = 5
        flow = AdaptiveBoomerang(d)
        @test flow isa MutableBoomerang
        @test flow isa PDMPSamplers.AnyBoomerang
        @test flow.μ == zeros(d)
        @test flow.Γ == Diagonal(ones(d))
        @test flow.λref == 0.1
        @test flow.ρ == 0.0
        @test flow.eigen_cache === nothing

        # Custom initial guess
        Γ0 = Diagonal([2.0, 3.0])
        μ0 = [1.0, -1.0]
        flow2 = AdaptiveBoomerang(Γ0, μ0; λref=0.5)
        @test flow2 isa MutableBoomerang
        @test flow2.μ == μ0
        @test flow2.λref == 0.5

        # AnyBoomerang includes both types
        flow_immut = Boomerang(d)
        @test flow_immut isa PDMPSamplers.AnyBoomerang
        @test flow isa PDMPSamplers.AnyBoomerang
    end

    # ──────────────────────────────────────────────────────────────────────
    # Test 0b: Dynamics methods work for MutableBoomerang
    # ──────────────────────────────────────────────────────────────────────
    @testset "Dynamics methods" begin
        d = 3
        flow = AdaptiveBoomerang(d)

        # initialize_velocity
        θ = PDMPSamplers.initialize_velocity(flow, d)
        @test length(θ) == d
        @test all(isfinite, θ)

        # refresh_velocity!
        ξ = SkeletonPoint(randn(d), randn(d))
        PDMPSamplers.refresh_velocity!(ξ, flow)
        @test all(isfinite, ξ.θ)

        # move_forward_time!
        state = PDMPSamplers.PDMPState(0.0, SkeletonPoint(randn(d), randn(d)))
        x_before = copy(state.ξ.x)
        PDMPSamplers.move_forward_time!(state, 0.5, flow)
        @test state.t[] ≈ 0.5
        @test all(isfinite, state.ξ.x)

        # reflect!
        ξ2 = SkeletonPoint(randn(d), randn(d))
        ∇ϕ = randn(d)
        cache = (; z=similar(ξ2.x))
        θ_before = copy(ξ2.θ)
        PDMPSamplers.reflect!(ξ2, ∇ϕ, flow, cache)
        @test all(isfinite, ξ2.θ)

        # λ
        ξ3 = SkeletonPoint(randn(d), randn(d))
        rate = PDMPSamplers.λ(ξ3, randn(d), flow)
        @test rate >= 0

        # refresh_rate
        @test PDMPSamplers.refresh_rate(flow) == flow.λref
    end

    # ──────────────────────────────────────────────────────────────────────
    # Test 1a: MvNormal with diagonal covariance
    # ──────────────────────────────────────────────────────────────────────
    @testset "MvNormal diagonal" begin
        Random.seed!(12345)
        d = 5
        μ_true = randn(d) .* 3
        σ_true = rand(d) .* 2 .+ 0.3  # avoid very small σ
        Σ = Diagonal(σ_true .^ 2)
        Σ_inv = inv(Σ)
        D = MvNormal(μ_true, Σ)

        target = gen_data(Distributions.MvNormal, d, 100.0, μ_true, σ_true, Matrix{Float64}(I(d)))

        flow = AdaptiveBoomerang(d)
        x0 = μ_true + randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        T = 100_000.0
        t_warmup = 10_000.0
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T, t_warmup; progress=show_progress)

        # Adaptation quality: μ should be close to true (element-wise)
        for i in 1:d
            @test abs(flow.μ[i] - μ_true[i]) < 1.0
        end

        # Adaptation quality: diagonal Γ should approximate Σ_inv
        adapted_diag = diag(flow.Γ)
        true_diag = diag(Σ_inv)
        for i in 1:d
            @test adapted_diag[i] > 0
            @test isapprox(adapted_diag[i], true_diag[i]; rtol=0.5)
        end

        # Trace mean should be in the right ballpark
        # Note: with adapted (close but not exact) μ, the Boomerang mean
        # converges slowly because bouncing rate is very low. We use a
        # loose element-wise tolerance rather than a strict T² test.
        trace_mean = mean(trace)
        for i in 1:d
            @test abs(trace_mean[i] - μ_true[i]) < 1.5
        end

        # Basic sanity
        @test length(trace) > 100
        @test stats.refreshment_events > 10
    end

    # ──────────────────────────────────────────────────────────────────────
    # Test 1b: MvNormal with correlated covariance (diagonal adapter is suboptimal)
    # ──────────────────────────────────────────────────────────────────────
    @testset "MvNormal correlated" begin
        Random.seed!(54321)
        d = 3
        η = 2.0  # moderate correlation via LKJ

        target = gen_data(Distributions.MvNormal, d, η)
        D = target.D

        flow = AdaptiveBoomerang(d)
        x0 = mean(D) + randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        T = 100_000.0
        t_warmup = 10_000.0
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T, t_warmup; progress=show_progress)

        # Adapted μ should be roughly correct (element-wise)
        for i in 1:d
            @test abs(flow.μ[i] - mean(D)[i]) < 1.5
        end

        # Trace mean should be in the right ballpark
        trace_mean = mean(trace)
        for i in 1:d
            @test abs(trace_mean[i] - mean(D)[i]) < 2.0
        end

        @test length(trace) > 100
    end

    # ──────────────────────────────────────────────────────────────────────
    # Test 1c: Logistic regression
    # ──────────────────────────────────────────────────────────────────────
    @testset "Logistic regression" begin
        Random.seed!(99999)
        d = 3
        n = 200
        target = gen_data(LogisticRegressionModel, d, n)

        flow = AdaptiveBoomerang(d)
        x0 = randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        T = 100_000.0
        t_warmup = 10_000.0
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T, t_warmup; progress=show_progress)

        trace_mean = mean(trace)

        @test length(trace) > 100
        @test all(isfinite, trace_mean)
        @test all(isfinite, flow.μ)
        @test all(x -> x > 0, diag(flow.Γ))
    end

    # ──────────────────────────────────────────────────────────────────────
    # Test 1d: Sticky + adaptive Boomerang (BetaBernoulli)
    # ──────────────────────────────────────────────────────────────────────
    @testset "Sticky adaptive" begin
        Random.seed!(77777)
        d = 5

        D, κ, slab_target = gen_data(SpikeAndSlabDist{BetaBernoulli, ZeroMeanIsoNormal}, d)

        flow = AdaptiveBoomerang(d; λref=1.0)
        x0 = randn(d) .* 0.1
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        grad = FullGradient(Base.Fix1(neg_gradient!, slab_target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, slab_target))
        alg = Sticky(GridThinningStrategy(), κ, trues(d))

        T = 100_000.0
        t_warmup = 10_000.0
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T, t_warmup; progress=show_progress)

        @test length(trace) > 100
        @test all(isfinite, mean(trace))
        @test all(isfinite, flow.μ)
        @test all(x -> x > 0, diag(flow.Γ))
    end

    # ──────────────────────────────────────────────────────────────────────
    # Test: Velocity refresh covariance check
    # ──────────────────────────────────────────────────────────────────────
    @testset "Velocity refresh distribution" begin
        Random.seed!(42)
        d = 4
        σ = [0.5, 1.0, 2.0, 0.3]
        Γ = Diagonal(1.0 ./ σ .^ 2)
        flow = AdaptiveBoomerang(Γ, zeros(d))

        # Sample many velocities from refresh_velocity!
        n_samples = 10_000
        samples = zeros(d, n_samples)
        for j in 1:n_samples
            θ = randn(d)
            PDMPSamplers.refresh_velocity!(θ, flow)
            samples[:, j] = θ
        end

        # Velocity distribution should be N(0, Γ⁻¹) = N(0, diag(σ²))
        for i in 1:d
            empirical_var = var(view(samples, i, :))
            @test isapprox(empirical_var, σ[i]^2; rtol=0.15)
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # Test: Online stats correctness
    # ──────────────────────────────────────────────────────────────────────
    @testset "BoomerangWarmupStats" begin
        d = 2
        stats = PDMPSamplers.BoomerangWarmupStats(d)
        @test stats.coord_time == zeros(d)
        @test stats.sum_x == zeros(d)
        @test stats.sum_x2 == zeros(d)
        @test stats.cursor == 0

        # After no data, mean and var have safe defaults
        μ = PDMPSamplers.stats_mean(stats)
        σ² = PDMPSamplers.stats_var(stats)
        @test μ == zeros(d)
        @test σ² == ones(d)
    end

end
