@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

import QuadGK

@testset "Transformed estimators" begin

    # ──────────────────────────────────────────────────────────────────────────
    # Level 1: segment integral correctness (unit tests vs QuadGK)
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Segment integrals" begin
        y0, v, dt = 1.5, -0.8, 0.3

        all_transforms = [
            LowerBoundTransform(0.0),
            LowerBoundTransform(2.0),
            UpperBoundTransform(5.0),
            DoubleBoundTransform(0.0, 1.0),
            DoubleBoundTransform(-3.0, 7.0),
        ]

        @testset "Mean: $T" for T in all_transforms
            # Closed-form / GL-5 vs QuadGK reference
            cf = PDMPSamplers._transformed_mean_segment(T, ZigZag(1), y0, v, dt, 0.0)
            ref, _ = QuadGK.quadgk(s -> T(y0 + v * s), 0.0, dt)
            @test cf ≈ ref atol = 1e-10

            # v = 0 edge case
            cf0 = PDMPSamplers._transformed_mean_segment(T, ZigZag(1), y0, 0.0, dt, 0.0)
            ref0, _ = QuadGK.quadgk(s -> T(y0), 0.0, dt)
            @test cf0 ≈ ref0 atol = 1e-12
        end

        @testset "Mean: Identity" begin
            for base in (ZigZag(1), BouncyParticle(1))
                cf = PDMPSamplers._transformed_mean_segment(IdentityTransform(), base, y0, v, dt, 0.0)
                ref, _ = QuadGK.quadgk(s -> y0 + v * s, 0.0, dt)
                @test cf ≈ ref atol = 1e-12
            end

            μ_b = 0.5
            cf = PDMPSamplers._transformed_mean_segment(IdentityTransform(), Boomerang(1), y0, v, dt, μ_b)
            ref, _ = QuadGK.quadgk(s -> μ_b + (y0 - μ_b) * cos(s) + v * sin(s), 0.0, dt)
            @test cf ≈ ref atol = 1e-12
        end

        @testset "Mean: Boomerang GL-5" begin
            μ_b, Δx, θ_b, dt_b = 0.5, 1.7, -0.7, 0.5
            for T in all_transforms
                f_boom(s) = T(μ_b + (Δx - μ_b) * cos(s) + θ_b * sin(s))
                ref, _ = QuadGK.quadgk(f_boom, 0.0, dt_b)
                gl = PDMPSamplers._transformed_mean_segment(T, Boomerang(1), Δx, θ_b, dt_b, μ_b)
                @test gl ≈ ref atol = 1e-8
            end
        end

        @testset "Variance: $T" for T in all_transforms
            μ_est = T(y0 + v * dt / 2)  # rough midpoint transform as mock mean
            ref, _ = QuadGK.quadgk(s -> (T(y0 + v * s) - μ_est)^2, 0.0, dt)
            result = PDMPSamplers._transformed_var_segment(T, ZigZag(1), y0, v, dt, 0.0, μ_est)
            @test result ≈ ref atol = 1e-9

            # v = 0 edge case
            ref0, _ = QuadGK.quadgk(s -> (T(y0) - μ_est)^2, 0.0, dt)
            result0 = PDMPSamplers._transformed_var_segment(T, ZigZag(1), y0, 0.0, dt, 0.0, μ_est)
            @test result0 ≈ ref0 atol = 1e-12
        end

        @testset "Variance: Identity" begin
            μ_est = y0 + v * dt / 2
            for base in (ZigZag(1), BouncyParticle(1))
                ref, _ = QuadGK.quadgk(s -> (y0 + v * s - μ_est)^2, 0.0, dt)
                result = PDMPSamplers._transformed_var_segment(IdentityTransform(), base, y0, v, dt, 0.0, μ_est)
                @test result ≈ ref atol = 1e-12
            end

            μ_b = 0.5
            ref, _ = QuadGK.quadgk(0.0, dt) do s
                (μ_b + (y0 - μ_b) * cos(s) + v * sin(s) - μ_est)^2
            end
            result = PDMPSamplers._transformed_var_segment(IdentityTransform(), Boomerang(1), y0, v, dt, μ_b, μ_est)
            @test result ≈ ref atol = 1e-12
        end
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Level 2: transform type properties
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Transform properties" begin
        transforms = [
            (LowerBoundTransform(0.0),   0.01, 100.0),
            (LowerBoundTransform(-5.0), -4.99, 100.0),
            (UpperBoundTransform(10.0), -100.0, 9.99),
            (DoubleBoundTransform(0.0, 1.0), 0.01, 0.99),
            (DoubleBoundTransform(-2.0, 3.0), -1.99, 2.99),
        ]
        for (T, x_lo, x_hi) in transforms
            @testset "$T" begin
                # Round-trip: transform ∘ inv_transform ≈ identity
                for x_test in range(x_lo, x_hi; length=10)
                    @test T(inv_transform(T, x_test)) ≈ x_test atol = 1e-10
                end

                # Monotonicity
                ys = range(-5.0, 5.0; length=100)
                xs = [T(y) for y in ys]
                if is_increasing(T)
                    @test issorted(xs)
                else
                    @test issorted(xs; rev=true)
                end
            end
        end

        # Identity round-trip
        @test IdentityTransform()(3.14) ≈ 3.14
        @test inv_transform(IdentityTransform(), 3.14) ≈ 3.14
        @test is_increasing(IdentityTransform())

        # DoubleBoundTransform validation
        @test_throws ArgumentError DoubleBoundTransform(5.0, 2.0)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Level 3: full-trace constrained estimators vs discretized samples
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Full-trace estimators" begin

        d = 3
        μ_unc = [1.0, -0.5, 0.3]
        σ_unc = [0.5, 1.0, 0.7]
        Σ = Diagonal(σ_unc .^ 2)
        D = MvNormal(μ_unc, Σ)
        Σ_inv = inv(Σ)
        potential = Σ_inv * μ_unc

        grad_fn!(out, x) = (mul!(out, Σ_inv, x); out .-= potential; nothing)
        hvp_fn!(out, _, v) = mul!(out, Σ_inv, v)
        grad = FullGradient(grad_fn!)
        model = PDMPModel(d, grad, hvp_fn!)
        alg = GridThinningStrategy()
        T_sim = 50_000.0

        transforms_lb = [LowerBoundTransform(0.0) for _ in 1:d]

        # Log-normal true statistics: x = exp(y), y ~ N(μ, σ²)
        true_mean = exp.(μ_unc .+ σ_unc .^ 2 ./ 2)
        true_var = (exp.(σ_unc .^ 2) .- 1) .* exp.(2μ_unc .+ σ_unc .^ 2)

        @testset "$pdmp_type" for pdmp_type in (ZigZag, BouncyParticle, Boomerang)

            flow = pdmp_type(Σ_inv, μ_unc)
            ξ0 = SkeletonPoint(μ_unc .+ 0.1 * randn(d), PDMPSamplers.initialize_velocity(flow, d))
            trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, T_sim; progress=false)

            dt_disc = mean(diff(PDMPSamplers.event_times(trace)))
            samples = Matrix(PDMPDiscretize(trace, dt_disc))

            min_ess = minimum(MCMCDiagnosticTools.ess.(eachcol(samples)))
            se_factor = 2.0 / sqrt(min_ess)
            rtol = max(0.15, se_factor * 0.5)
            atol = max(0.15, se_factor * 0.1)

            # --- Constrained mean ---
            c_mean = mean(trace, transforms_lb)
            disc_mean = vec(mean(exp, samples; dims=1))
            @test c_mean ≈ disc_mean rtol = rtol atol = atol
            @test c_mean ≈ true_mean rtol = max(0.2, se_factor)

            # --- Constrained variance ---
            c_var = var(trace, transforms_lb)
            disc_var = vec(var(exp.(samples); dims=1))
            @test c_var ≈ disc_var rtol = rtol atol = atol

            # --- Constrained std ---
            c_std = std(trace, transforms_lb)
            @test c_std ≈ sqrt.(c_var) atol = 1e-12

            # --- Constrained quantile ---
            for j in 1:d, p in [0.1, 0.5, 0.9]
                q_c = quantile(trace, p, transforms_lb; coordinate=j)
                q_disc = quantile(exp.(@view(samples[:, j])), p)
                q_true = quantile(LogNormal(μ_unc[j], σ_unc[j]), p)
                @test q_c ≈ q_disc atol = atol * true_mean[j]
                @test q_c ≈ q_true rtol = max(0.2, se_factor)
            end

            # --- Constrained median (all coordinates at once) ---
            c_med = median(trace, transforms_lb)
            @test length(c_med) == d
            for j in 1:d
                @test c_med[j] ≈ quantile(trace, 0.5, transforms_lb; coordinate=j)
            end

            # --- Constrained CDF ---
            for j in 1:d
                q_val = true_mean[j]
                c_cdf = PDMPSamplers.cdf(trace, q_val, transforms_lb; coordinate=j)
                true_cdf = Distributions.cdf(LogNormal(μ_unc[j], σ_unc[j]), q_val)
                @test c_cdf ≈ true_cdf atol = 0.15
            end
        end

        # --- Double-bounded: logistic transform ---
        @testset "DoubleBound trace" begin
            transforms_db = [DoubleBoundTransform(0.0, 1.0) for _ in 1:d]

            flow = BouncyParticle(Σ_inv, μ_unc)
            ξ0 = SkeletonPoint(μ_unc .+ 0.1 * randn(d), PDMPSamplers.initialize_velocity(flow, d))
            trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, T_sim; progress=false)

            dt_disc = mean(diff(PDMPSamplers.event_times(trace)))
            samples = Matrix(PDMPDiscretize(trace, dt_disc))

            c_mean = mean(trace, transforms_db)
            disc_mean = vec(mean(LogExpFunctions.logistic, samples; dims=1))

            min_ess = minimum(MCMCDiagnosticTools.ess.(eachcol(samples)))
            se_factor = 2.0 / sqrt(min_ess)
            rtol = max(0.15, se_factor * 0.5)
            atol = max(0.15, se_factor * 0.1)

            @test c_mean ≈ disc_mean rtol = rtol atol = atol
            @test all(0.0 .< c_mean .< 1.0)
        end
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Level 4: identity transform regression
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Identity transform regression" begin

        d = 3
        target = gen_data(MvNormal, d, 2.0)
        D = target.D
        Γ = inv(Symmetric(cov(D)))
        μ = mean(D)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()
        T_sim = 50_000.0

        id_transforms = fill(IdentityTransform(), d)

        @testset "$pdmp_type" for pdmp_type in (ZigZag, BouncyParticle, Boomerang)
            flow = pdmp_type(Γ, μ)
            ξ0 = SkeletonPoint(μ .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
            trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, T_sim; progress=false)

            @test mean(trace, id_transforms) ≈ mean(trace) atol = 1e-10
            @test var(trace, id_transforms) ≈ var(trace) atol = 1e-10
            @test std(trace, id_transforms) ≈ std(trace) atol = 1e-10

            for j in 1:d
                for p in [0.1, 0.5, 0.9]
                    @test quantile(trace, p, id_transforms; coordinate=j) ≈
                          quantile(trace, p; coordinate=j) atol = 1e-10
                end
                @test median(trace, id_transforms; coordinate=j) ≈
                      median(trace; coordinate=j) atol = 1e-10
                @test PDMPSamplers.cdf(trace, μ[j], id_transforms; coordinate=j) ≈
                      PDMPSamplers.cdf(trace, μ[j]; coordinate=j) atol = 1e-10
            end
        end
    end
end
