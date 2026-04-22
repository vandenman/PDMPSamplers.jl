@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

import QuadGK

@testset "Welford exact continuous-time accumulator" begin

    # ──────────────────────────────────────────────────────────────────────
    # 1. Exact kernel unit tests
    # ──────────────────────────────────────────────────────────────────────
    @testset "Exact segment helpers – deterministic cases" begin
        d = 3
        S1  = zeros(d)
        S2  = zeros(d)
        S11 = zeros(d, d)

        x0  = [1.0, -2.0, 0.5]
        θ0  = [0.5,  1.0, -1.0]
        mu  = [0.3,  0.1,  0.2]

        for (label, dt, x0_in, θ0_in) in [
            ("dt=0",    0.0,        x0,        θ0),
            ("dt=pi/2", Float64(π)/2, x0,      θ0),
            ("dt=pi",   Float64(π),  x0,        θ0),
            ("dt=2pi",  2*Float64(π), x0,       θ0),
            ("theta0=0", 1.0,        x0,        zeros(d)),
            ("x0=mu",    1.0,        mu,        θ0),
        ]
            S1_test  = zeros(d)
            S2_test  = zeros(d)
            S11_test = zeros(d, d)

            PDMPSamplers._boom_raw_S1!(S1_test, x0_in, θ0_in, mu, dt)
            PDMPSamplers._boom_raw_S2!(S2_test, x0_in, θ0_in, mu, dt)
            PDMPSamplers._boom_raw_S11!(S11_test, x0_in, θ0_in, mu, dt)

            if dt == 0.0
                @test S1_test ≈ zeros(d)  rtol=1e-12
                @test S2_test ≈ zeros(d)  rtol=1e-12
                @test S11_test ≈ zeros(d, d) rtol=1e-12
            end

            # S11 must be symmetric
            @test issymmetric(S11_test)

            # Diagonal of S11 must agree with S2
            for i in 1:d
                S11_diag_test = zeros(d, d)
                S2_diag_test  = zeros(d)
                PDMPSamplers._boom_raw_S11!(S11_diag_test, x0_in, θ0_in, mu, dt)
                PDMPSamplers._boom_raw_S2!(S2_diag_test, x0_in, θ0_in, mu, dt)
                @test S11_diag_test[i, i] ≈ S2_diag_test[i]  rtol=1e-10 atol=1e-14
            end

            # x0 = mu, theta0 = 0 → constant trajectory at mu
            if x0_in ≈ mu && all(θ0_in .== 0)
                @test S1_test ≈ mu .* dt  rtol=1e-12
                @test S2_test ≈ mu .^ 2 .* dt  rtol=1e-12
            end
        end
    end

    @testset "Exact segment helpers – quadrature comparison" begin
        Random.seed!(999)
        d = 4

        for _ in 1:20
            x0 = randn(d) .* 3
            θ0 = randn(d) .* 2
            mu = randn(d)
            dt = rand() * 2π + 0.1

            S1_exact  = zeros(d)
            S2_exact  = zeros(d)
            S11_exact = zeros(d, d)
            PDMPSamplers._boom_raw_S1!(S1_exact, x0, θ0, mu, dt)
            PDMPSamplers._boom_raw_S2!(S2_exact, x0, θ0, mu, dt)
            PDMPSamplers._boom_raw_S11!(S11_exact, x0, θ0, mu, dt)

            # Numerical quadrature reference
            for i in 1:d
                xi(t) = (x0[i] - mu[i]) * cos(t) + θ0[i] * sin(t) + mu[i]
                S1_quad, _ = QuadGK.quadgk(xi, 0.0, dt; rtol=1e-12)
                @test S1_exact[i] ≈ S1_quad  rtol=1e-8

                S2_quad, _ = QuadGK.quadgk(t -> xi(t)^2, 0.0, dt; rtol=1e-12)
                @test S2_exact[i] ≈ S2_quad  rtol=1e-8

                for j in 1:d
                    xj(t) = (x0[j] - mu[j]) * cos(t) + θ0[j] * sin(t) + mu[j]
                    S11_quad, _ = QuadGK.quadgk(t -> xi(t) * xj(t), 0.0, dt; rtol=1e-12)
                    @test S11_exact[i, j] ≈ S11_quad  rtol=1e-8
                end
            end
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 2. Online accumulator correctness
    # ──────────────────────────────────────────────────────────────────────
    @testset "Online accumulator – synthetic event stream" begin
        Random.seed!(77)
        d = 3
        flow = AdaptiveBoomerang(d; scheme=:fullrank)

        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        n_events = 30
        ts = cumsum([0.0; rand(n_events) .* 2 .+ 0.1])
        xs = [randn(d) for _ in 1:(n_events + 1)]
        θs = [randn(d) for _ in 1:(n_events + 1)]

        # Accumulate online
        for k in 1:(n_events + 1)
            PDMPSamplers.welford_update!(ws, xs[k], θs[k], ts[k], flow)
        end

        # Reference: direct sum of exact segment formulas
        S1_ref  = zeros(d)
        S2_ref  = zeros(d)
        S11_ref = zeros(d, d)
        T_ref   = 0.0

        for k in 1:n_events
            dt = ts[k + 1] - ts[k]
            T_ref += dt
            PDMPSamplers._boom_raw_S1!(S1_ref,  xs[k], θs[k], flow.μ, dt)
            PDMPSamplers._boom_raw_S2!(S2_ref,  xs[k], θs[k], flow.μ, dt)
            PDMPSamplers._boom_raw_S11!(S11_ref, xs[k], θs[k], flow.μ, dt)
        end

        @test ws.total_time ≈ T_ref  rtol=1e-12
        @test ws.sum_x_dt   ≈ S1_ref  rtol=1e-12
        @test ws.sum_x2_dt  ≈ S2_ref  rtol=1e-12
        @test ws.sum_xy_dt  ≈ S11_ref rtol=1e-12
    end

    @testset "Online accumulator – compare to immutable Boomerang trace statistics" begin
        Random.seed!(42)
        d = 3
        flow_immutable = Boomerang(d)
        μ_ref = flow_immutable.μ

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow_immutable, d))
        target = gen_data(Distributions.MvNormal, d, 1.0)
        grad   = FullGradient(Base.Fix1(neg_gradient!, target))
        model  = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg    = GridThinningStrategy()
        trace, _ = pdmp_sample(ξ0, flow_immutable, model, alg, 0.0, 2000.0; progress=false)

        # Build accumulator from trace skeleton events
        ws_trace = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        flow_mut = AdaptiveBoomerang(Diagonal(ones(d)), collect(μ_ref))
        for k in 1:length(trace.times)
            t = trace.times[k]
            x = trace.positions[:, k]
            θ = trace.velocities[:, k]
            PDMPSamplers.welford_update!(ws_trace, x, θ, t, flow_mut)
        end

        # Compare derived moments to trace statistics
        m_acc  = PDMPSamplers.stats_mean(ws_trace)
        m_trc  = mean(trace)
        @test m_acc ≈ m_trc  rtol=1e-6

        v_acc  = PDMPSamplers.stats_var(ws_trace)
        v_trc  = var(trace)
        @test v_acc ≈ v_trc  rtol=1e-6

        C_acc  = PDMPSamplers.stats_cov(ws_trace)
        C_trc  = cov(trace)
        @test C_acc ≈ C_trc  rtol=1e-6
    end

    # ──────────────────────────────────────────────────────────────────────
    # 3. Update-boundary and ordering tests
    # ──────────────────────────────────────────────────────────────────────
    @testset "Segment ending at adaptation boundary uses pre-update reference" begin
        Random.seed!(123)
        d = 2

        flow_a = AdaptiveBoomerang(d; scheme=:diagonal)
        flow_b = AdaptiveBoomerang(d; scheme=:diagonal)

        x0 = [1.0, 2.0]; θ0 = [0.5, -0.3]; t0 = 0.0
        x1 = [1.5, 1.8]; θ1 = [0.4, -0.5]; t1 = 1.0

        # Correct order: accumulate segment with pre-update reference
        ws_correct = PDMPSamplers.WelfordBoomerangStats(d)
        PDMPSamplers.welford_update!(ws_correct, x0, θ0, t0, flow_a)
        PDMPSamplers.welford_update!(ws_correct, x1, θ1, t1, flow_a)
        # Now update reference
        flow_a.μ .= [5.0, 5.0]

        # Wrong order: if reference was updated first, accumulation would differ
        ws_wrong = PDMPSamplers.WelfordBoomerangStats(d)
        PDMPSamplers.welford_update!(ws_wrong, x0, θ0, t0, flow_b)
        flow_b.μ .= [5.0, 5.0]
        PDMPSamplers.welford_update!(ws_wrong, x1, θ1, t1, flow_b)

        # The segment integrals must differ when references are different
        @test !(ws_correct.sum_x_dt ≈ ws_wrong.sum_x_dt)
        @test !(ws_correct.sum_x2_dt ≈ ws_wrong.sum_x2_dt)
    end

    @testset "Reset segment start after velocity refresh" begin
        d = 2
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        flow = AdaptiveBoomerang(d; scheme=:diagonal)

        x = [1.0, 2.0]; θ_old = [0.3, -0.1]; t = 5.0
        PDMPSamplers.welford_update!(ws, x, θ_old, 0.0, flow)
        PDMPSamplers.welford_update!(ws, x, θ_old, t, flow)

        # Simulate velocity refresh: new theta
        θ_new = [0.7, 0.5]
        PDMPSamplers.reset_segment_start!(ws, x, θ_new, t)

        @test ws.prev_theta ≈ θ_new
        @test ws.prev_x ≈ x
        @test ws.prev_t == t
    end

    @testset "Same-time call after reset does not accumulate stale theta" begin
        d = 2
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        flow = AdaptiveBoomerang(d; scheme=:diagonal)

        x = [1.0, 2.0]; θ_old = [1.0, 1.0]; t = 3.0
        PDMPSamplers.welford_update!(ws, x, θ_old, 0.0, flow)
        PDMPSamplers.welford_update!(ws, x, θ_old, t, flow)

        T_before = ws.total_time
        θ_new = [0.0, 0.0]
        PDMPSamplers.reset_segment_start!(ws, x, θ_new, t)

        # dt = 0 → no new accumulation
        PDMPSamplers.welford_update!(ws, x, θ_new, t, flow)
        @test ws.total_time == T_before
        # prev_theta should now be the refreshed theta
        @test ws.prev_theta ≈ θ_new
    end

    # ──────────────────────────────────────────────────────────────────────
    # 4. Edge-case tests
    # ──────────────────────────────────────────────────────────────────────
    @testset "Edge cases" begin
        d = 3
        flow = AdaptiveBoomerang(d; scheme=:fullrank)
        θ0 = zeros(d)

        # dt = 0: no accumulation, but prev state updated
        ws = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        PDMPSamplers.welford_update!(ws, ones(d), θ0, 0.0, flow)
        PDMPSamplers.welford_update!(ws, ones(d), θ0, 0.0, flow)
        @test ws.total_time == 0.0
        @test ws.sum_x_dt  ≈ zeros(d)
        @test ws.sum_x2_dt ≈ zeros(d)
        @test ws.sum_xy_dt ≈ zeros(d, d)

        # Repeated same-time events: total_time stays at first non-zero dt
        PDMPSamplers.welford_update!(ws, ones(d) .* 2, θ0, 1.0, flow)
        T1 = ws.total_time
        PDMPSamplers.welford_update!(ws, ones(d) .* 2, θ0, 1.0, flow)
        @test ws.total_time == T1

        # Very small positive dt
        ws2 = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        PDMPSamplers.welford_update!(ws2, ones(d), θ0, 0.0, flow)
        PDMPSamplers.welford_update!(ws2, ones(d) .* 2, θ0, 1e-14, flow)
        @test ws2.total_time == 1e-14
        @test all(isfinite, ws2.sum_x_dt)
        @test all(isfinite, ws2.sum_x2_dt)

        # No divide-by-zero before any data
        ws3 = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        @test all(PDMPSamplers.stats_mean(ws3) .== 0.0)
        @test all(PDMPSamplers.stats_var(ws3) .== 1.0)
        C = PDMPSamplers.stats_cov(ws3)
        @test C == Matrix{Float64}(I, d, d)

        # Symmetric covariance output after long near-constant run
        ws4 = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        t = 0.0
        for _ in 1:1000
            t += 0.01
            PDMPSamplers.welford_update!(ws4, fill(5.0, d), θ0, t, flow)
        end
        C4 = PDMPSamplers.stats_cov(ws4)
        @test issymmetric(C4)
        @test all(diag(C4) .>= 0.0)
    end

    # ──────────────────────────────────────────────────────────────────────
    # 5. Additivity across blocks
    # ──────────────────────────────────────────────────────────────────────
    @testset "Additivity across warmup blocks" begin
        Random.seed!(31415)
        d = 3

        # Three blocks with different fixed references
        μs = [zeros(d), [1.0, -1.0, 0.5], [2.0, 0.0, -1.0]]
        block_events = [8, 6, 10]  # number of skeleton events in each block

        ts_all  = Float64[]
        xs_all  = Vector{Float64}[]
        θs_all  = Vector{Float64}[]

        t = 0.0
        for (blk, n) in enumerate(block_events)
            for _ in 1:n
                t += rand() + 0.1
                push!(ts_all, t)
                push!(xs_all, randn(d))
                push!(θs_all, randn(d))
            end
        end
        # One extra event past the end of the last block
        t += rand() + 0.1
        push!(ts_all, t); push!(xs_all, randn(d)); push!(θs_all, randn(d))

        # Block boundaries (last event index belonging to each block)
        # blk_boundaries[k] = last event fed in block k (= init for block k+1)
        blk_boundaries = cumsum(block_events)               # [8, 14, 24]
        blk_starts     = [1; blk_boundaries[1:end-1]]       # [1, 8, 14]
        blk_ends       = [blk_boundaries[1:end-1]; length(ts_all)]  # [8, 14, 25]

        # Cumulative online accumulator: each segment integrated with the reference active at that time
        ws_cum = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
        flow_cum = AdaptiveBoomerang(d; scheme=:fullrank)

        cur_blk = 1
        for k in eachindex(ts_all)
            PDMPSamplers.welford_update!(ws_cum, xs_all[k], θs_all[k], ts_all[k], flow_cum)
            if cur_blk < 3 && k == blk_boundaries[cur_blk]
                cur_blk += 1
                flow_cum.μ .= μs[cur_blk]
                PDMPSamplers.reset_segment_start!(ws_cum, xs_all[k], θs_all[k], ts_all[k])
            end
        end

        # Per-block accumulators: each feeds [init_event, block_events...]
        S1_sum  = zeros(d)
        S2_sum  = zeros(d)
        S11_sum = zeros(d, d)
        T_sum   = 0.0

        for blk in 1:3
            flow_blk = AdaptiveBoomerang(d; scheme=:fullrank)
            flow_blk.μ .= μs[blk]
            ws_blk = PDMPSamplers.WelfordBoomerangStats(d; fullrank=true)
            for k in blk_starts[blk]:blk_ends[blk]
                PDMPSamplers.welford_update!(ws_blk, xs_all[k], θs_all[k], ts_all[k], flow_blk)
            end
            S1_sum  .+= ws_blk.sum_x_dt
            S2_sum  .+= ws_blk.sum_x2_dt
            S11_sum .+= ws_blk.sum_xy_dt
            T_sum   += ws_blk.total_time
        end

        @test ws_cum.total_time ≈ T_sum   rtol=1e-12
        @test ws_cum.sum_x_dt   ≈ S1_sum  rtol=1e-12
        @test ws_cum.sum_x2_dt  ≈ S2_sum  rtol=1e-12
        @test ws_cum.sum_xy_dt  ≈ S11_sum rtol=1e-12
    end

    # ──────────────────────────────────────────────────────────────────────
    # 6. Public estimator non-regression
    # ──────────────────────────────────────────────────────────────────────
    @testset "MutableBoomerang mean(trace) unchanged" begin
        Random.seed!(42)
        d = 2
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        target = gen_data(Distributions.MvNormal, d, 1.0)
        grad   = FullGradient(Base.Fix1(neg_gradient!, target))
        model  = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg    = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        # Run WITHOUT adaptation (NoAdaptation) to keep reference fixed
        trace, _ = pdmp_sample(ξ0, flow, model, alg, 0.0, 2000.0;
                               progress=false, adapter=PDMPSamplers.NoAdaptation())

        m = mean(trace)
        @test all(isfinite, m)
        @test m ≈ mean(target.D)  atol=0.5
    end

    # ──────────────────────────────────────────────────────────────────────
    # 7. Allocation regression
    # ──────────────────────────────────────────────────────────────────────
    @testset "Allocation-free steady-state welford_update!" begin
        d = 4
        flow = AdaptiveBoomerang(d; scheme=:diagonal)
        ws = PDMPSamplers.WelfordBoomerangStats(d)
        x = randn(d); θ = randn(d)
        # Initialize
        PDMPSamplers.welford_update!(ws, x, θ, 0.0, flow)
        PDMPSamplers.welford_update!(ws, x, θ, 1.0, flow)

        # Steady-state: measure allocations
        allocs = minimum(
            @allocated(PDMPSamplers.welford_update!(ws, x, θ, 2.0 + Float64(i)*0.1, flow))
            for i in 1:20
        )
        @test allocs == 0

        # Allocation-free in-place stats
        μ_buf = zeros(d)
        allocs_mean = minimum(@allocated(PDMPSamplers.stats_mean!(μ_buf, ws)) for _ in 1:10)
        @test allocs_mean == 0
    end

end
