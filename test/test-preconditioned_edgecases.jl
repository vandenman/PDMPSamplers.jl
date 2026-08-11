@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Preconditioned dynamics edge cases" begin

    @testset "DensePreconditioner constructors and fields" begin
        dp = DensePreconditioner(4)
        @test dp.L ≈ Matrix{Float64}(I, 4, 4)

        @test_throws ArgumentError DensePreconditioner([1.0 1.0; 0.0 1.0])
        @test_throws DimensionMismatch DensePreconditioner(ones(2, 3))
        @test_throws MethodError DensePreconditioner(
            Matrix{Float64}(I, 2, 2), zeros(2), 99.0)

        old_L = copy(dp.L)
        old_norm = dp.L_operator_norm
        @test_throws SingularException set_dense_preconditioner!(dp, zeros(4, 4))
        @test_throws ArgumentError set_dense_preconditioner!(dp,
            [1.0 1.0 0.0 0.0; 0.0 1.0 0.0 0.0;
             0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0])
        @test dp.L == old_L
        @test dp.L_operator_norm == old_norm

        updated = Diagonal([100.0, 1.0, 1.0, 1.0])
        set_dense_preconditioner!(dp, updated)
        @test dp.L == updated
        @test dp.L_operator_norm == 100.0
        canonical = [1.0, -1.0, 1.0, -1.0]
        ξ = SkeletonPoint(zeros(4), dp.L * canonical)
        flow = PreconditionedDynamics(dp, ZigZag(4))
        state = PDMPState(0.0, ξ)
        PDMPSamplers._synchronize_dense_zigzag_velocity!(state, flow)
        @test state.boundary_scratch.canonical_signs ≈ canonical
    end

    @testset "subpreconditioner" begin
        free = BitVector([true, false, true, true])

        ip = PDMPSamplers.IdentityPreconditioner()
        @test PDMPSamplers.subpreconditioner(ip, free) === ip

        dp_diag = DiagonalPreconditioner([1.0, 2.0, 3.0, 4.0])
        sub_diag = PDMPSamplers.subpreconditioner(dp_diag, free)
        @test sub_diag isa DiagonalPreconditioner
        @test length(sub_diag.scale) == 3

        dp_dense = DensePreconditioner(4)
        set_dense_preconditioner!(dp_dense, LowerTriangular(randn(4, 4)) + 4I)
        sub_dense = PDMPSamplers.subpreconditioner(dp_dense, free)
        @test sub_dense isa DensePreconditioner
        @test size(sub_dense.L) == (3, 3)
        @test sub_dense.L_operator_norm ≈ opnorm(sub_dense.L)
    end

    @testset "isfactorized dispatch" begin
        zz = DensePreconditionedZigZag(3)
        @test !PDMPSamplers.isfactorized(zz)

        pzz = PreconditionedZigZag(3)
        @test PDMPSamplers.isfactorized(pzz)

        pbps = PreconditionedBPS(3)
        @test !PDMPSamplers.isfactorized(pbps)

        dbps = DensePreconditionedBPS(3)
        @test !PDMPSamplers.isfactorized(dbps)
    end

    @testset "transform_velocity!" begin
        v = [1.0, 2.0, 3.0]

        PDMPSamplers.transform_velocity!(v, PDMPSamplers.IdentityPreconditioner())
        @test v == [1.0, 2.0, 3.0]

        v2 = [1.0, 2.0, 3.0]
        PDMPSamplers.transform_velocity!(v2, DiagonalPreconditioner([2.0, 3.0, 4.0]))
        @test v2 == [2.0, 6.0, 12.0]

        dp = DensePreconditioner(3)
        set_dense_preconditioner!(dp,
            [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
        v3 = [1.0, 2.0, 3.0]
        PDMPSamplers.transform_velocity!(v3, dp)
        @test v3 ≈ [2.0, 6.0, 12.0]
    end

    @testset "DensePreconditionedZigZag constructors" begin
        d = 4
        zz = DensePreconditionedZigZag(d)
        @test zz isa PreconditionedDynamics{DensePreconditioner, <:ZigZag}
        @test zz.metric isa DensePreconditioner
        @test size(zz.metric.L) == (d, d)

        Γ = Matrix{Float64}(I, d, d)
        μ = zeros(d)
        zz2 = DensePreconditionedZigZag(Γ, μ)
        @test zz2 isa PreconditionedDynamics{DensePreconditioner, <:ZigZag}
    end

    @testset "DensePreconditionedBPS constructors" begin
        d = 3
        bps = DensePreconditionedBPS(d)
        @test bps isa PreconditionedDynamics{DensePreconditioner, <:BouncyParticle}

        bps2 = DensePreconditionedBPS(d; refresh_rate=2.0)
        @test PDMPSamplers.refresh_rate(bps2) == 2.0

        Γ = Matrix{Float64}(I, d, d)
        μ = zeros(d)
        bps3 = DensePreconditionedBPS(Γ, μ; refresh_rate=0.5)
        @test PDMPSamplers.refresh_rate(bps3) == 0.5
    end

    @testset "PreconditionedZigZag/BPS Γ,μ constructors" begin
        d = 3
        Γ = Matrix{Float64}(2I, d, d)
        μ = ones(d)
        pzz = PreconditionedZigZag(Γ, μ)
        @test pzz isa PreconditionedDynamics{<:DiagonalPreconditioner, <:ZigZag}

        pbps = PreconditionedBPS(Γ, μ; refresh_rate=0.5)
        @test pbps isa PreconditionedDynamics{<:DiagonalPreconditioner, <:BouncyParticle}
        @test PDMPSamplers.refresh_rate(pbps) == 0.5
    end

    @testset "DensePreconditionedZigZag λ correctness" begin
        d = 3
        Random.seed!(42)
        zz = DensePreconditionedZigZag(d)
        M = zz.metric

        # Set a non-trivial L
        L_new = [1.0 0.0 0.0; 0.5 1.0 0.0; 0.2 0.3 1.0]
        set_dense_preconditioner!(M, L_new)
        v = [1.0, -1.0, 1.0]
        ξ = SkeletonPoint(randn(d), M.L * v)
        state = PDMPState(0.0, ξ)
        ∇ϕ = randn(d)

        rate = PDMPSamplers.λ(state, ∇ϕ, zz)
        @test rate >= 0

        # Manual computation of the rate
        L = M.L
        expected_rate = 0.0
        for i in 1:d
            grad_z_i = sum(L[j, i] * ∇ϕ[j] for j in 1:d)
            expected_rate += max(0.0, v[i] * grad_z_i)
        end
        @test rate ≈ expected_rate
    end

    @testset "DensePreconditionedZigZag reflect!" begin
        d = 4
        Random.seed!(123)
        zz = DensePreconditionedZigZag(d)
        M = zz.metric

        Σ_true = [1.0 0.5 0.2 0.1; 0.5 1.0 0.3 0.2; 0.2 0.3 1.0 0.4; 0.1 0.2 0.4 1.0]
        L_new = cholesky(Symmetric(Σ_true)).L
        set_dense_preconditioner!(M, L_new)
        signs = [1.0, -1.0, 1.0, -1.0]
        ξ = SkeletonPoint(randn(d), M.L * signs)
        state = PDMPState(0.0, ξ)
        ∇ϕ = randn(d)
        cache = (; z=similar(ξ.x))

        θ_before = copy(ξ.θ)
        PDMPSamplers.reflect!(Random.default_rng(), state, ∇ϕ, zz, cache)
        @test all(isfinite, ξ.θ)
        # Exactly one canonical coordinate should have flipped
        recovered = M.L \ ξ.θ
        @test count(i -> !isapprox(recovered[i], signs[i]; atol=1e-12), 1:d) == 1
    end

    @testset "DensePreconditionedZigZag reflect! zero-rate branch" begin
        d = 3
        zz = DensePreconditionedZigZag(d)
        M = zz.metric
        signs = [1.0, -1.0, 1.0]
        ξ = SkeletonPoint(randn(d), M.L * signs)
        # Set gradient so that all v_i * grad_z_i ≤ 0 → total_rate = 0
        ∇ϕ = -inv(M.L') * signs
        cache = (; z=similar(ξ.x))

        PDMPSamplers.reflect!(ξ, ∇ϕ, zz, cache)
        @test all(isfinite, ξ.θ)
    end

    @testset "reflect! via AbstractPDMPState dispatch" begin
        d = 3
        zz = DensePreconditionedZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(zz, d)))
        ∇ϕ = randn(d)
        cache = (; z=similar(state.ξ.x))
        PDMPSamplers.reflect!(state, ∇ϕ, zz, cache)
        @test all(isfinite, state.ξ.θ)
    end

    @testset "Forwarding methods" begin
        d = 3
        zz = DensePreconditionedZigZag(d)
        ξ = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(zz, d))
        state = PDMPState(0.0, ξ)

        # move_forward_time!
        PDMPSamplers.move_forward_time!(state, 0.5, zz)
        @test state.t[] ≈ 0.5

        # λ forwarding (non-dense preconditioned λ uses inner dynamics)
        pzz = PreconditionedZigZag(d)
        ξ2 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(pzz, d))
        rate = PDMPSamplers.λ(ξ2, randn(d), pzz)
        @test rate >= 0

        # ab forwarding
        c = ones(d)
        cache2 = (; z=similar(ξ2.x), ∇ϕx=similar(ξ2.x))
        a, b = PDMPSamplers.ab(ξ2, c, pzz, cache2)
        @test isfinite(a) && isfinite(b)

        # refresh_rate forwarding
        bps = DensePreconditionedBPS(d; refresh_rate=0.7)
        @test PDMPSamplers.refresh_rate(bps) == 0.7
    end

    @testset "initialize_velocity applies preconditioner" begin
        d = 3
        dp = DensePreconditioner(d)
        set_dense_preconditioner!(dp,
            [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])

        zz = PreconditionedDynamics(dp, ZigZag(d))
        Random.seed!(42)
        v = PDMPSamplers.initialize_velocity(zz, d)
        @test length(v) == d
        # Velocities should be ±scale_i (not ±1)
        @test all(abs.(v) .∈ Ref([2.0, 3.0, 4.0]))
    end

    @testset "refresh_velocity! applies preconditioner" begin
        d = 3
        dp = DensePreconditioner(d)
        set_dense_preconditioner!(dp,
            [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])

        bps = PreconditionedDynamics(dp, BouncyParticle(d, 1.0))
        ξ = SkeletonPoint(randn(d), randn(d))
        PDMPSamplers.refresh_velocity!(ξ, bps)
        @test all(isfinite, ξ.θ)
    end

    @testset "subflow for PreconditionedDynamics" begin
        d = 4
        pzz = PreconditionedZigZag(d; scale=[2.0, 3.0, 4.0, 5.0])
        free = BitVector([true, false, true, true])
        sub = PDMPSamplers.subflow(pzz, free)
        @test sub isa PreconditionedDynamics
        @test length(sub.metric.scale) == 3
    end

    @testset "update_preconditioner! fallback" begin
        d = 3
        flow = ZigZag(d)
        trace = PDMPSamplers.PDMPTrace(
            [PDMPEvent(0.0, randn(d), randn(d)), PDMPEvent(1.0, randn(d), randn(d))], flow)
        state = PDMPState(0.0, SkeletonPoint(randn(d), randn(d)))
        result = PDMPSamplers.update_preconditioner!(flow, trace, state)
        @test result === flow
    end

    @testset "update_preconditioner! DiagonalPreconditioner" begin
        d = 3
        pzz = PreconditionedZigZag(d)
        flow_inner = ZigZag(d)

        events = [
            PDMPEvent(Float64(i), randn(d), randn(d)) for i in 0:100
        ]
        trace = PDMPSamplers.PDMPTrace(events, pzz)
        state = PDMPState(50.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(pzz, d)))

        old_scale = copy(pzz.metric.scale)
        PDMPSamplers.update_preconditioner!(pzz, trace, state, false)
        @test pzz.metric.scale != old_scale

        # Test first_update=true branch
        pzz2 = PreconditionedZigZag(d)
        PDMPSamplers.update_preconditioner!(pzz2, trace, state, true)
        @test pzz2.metric.scale != ones(d)
    end

    @testset "update_preconditioner! DiagonalPreconditioner zero sigma" begin
        d = 2
        pzz = PreconditionedZigZag(d)
        # Trace with identical positions → zero std
        events = [
            PDMPEvent(Float64(i), [1.0, 2.0], randn(d)) for i in 0:10
        ]
        trace = PDMPSamplers.PDMPTrace(events, pzz)
        state = PDMPState(5.0, SkeletonPoint([1.0, 2.0], PDMPSamplers.initialize_velocity(pzz, d)))
        PDMPSamplers.update_preconditioner!(pzz, trace, state)
        # Should not error; zero sigma should use old_scale / 2
        @test all(isfinite, pzz.metric.scale)
        @test all(pzz.metric.scale .> 0)
    end

    @testset "update_preconditioner! DiagonalPreconditioner with StickyPDMPState" begin
        d = 3
        pzz = PreconditionedZigZag(d)
        events = [
            PDMPEvent(Float64(i), randn(d), randn(d)) for i in 0:50
        ]
        trace = PDMPSamplers.PDMPTrace(events, pzz)
        ξ = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(pzz, d))
        state = StickyPDMPState(25.0, ξ, BitVector([true, false, true]))
        state.ξ.θ[2] = 0.0
        PDMPSamplers.update_preconditioner!(pzz, trace, state)
        @test all(isfinite, pzz.metric.scale)
        @test iszero(state.ξ.θ[2])
        @test all(abs.(state.ξ.θ[state.free]) .== pzz.metric.scale[state.free])
    end

    @testset "update_preconditioner! DensePreconditioner" begin
        d = 3
        dzz = DensePreconditionedZigZag(d)
        events = [
            PDMPEvent(Float64(i), randn(d), randn(d)) for i in 0:100
        ]
        trace = PDMPSamplers.PDMPTrace(events, dzz)
        state = PDMPState(50.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(dzz, d)))

        old_L = copy(dzz.metric.L)
        PDMPSamplers.update_preconditioner!(dzz, trace, state)
        @test dzz.metric.L != old_L
        @test all(isfinite, dzz.metric.L)
        @test dzz.metric.L_operator_norm ≈ opnorm(dzz.metric.L)
    end

    @testset "update_preconditioner! DensePreconditioner with BPS" begin
        d = 3
        dbps = DensePreconditionedBPS(d)
        events = [
            PDMPEvent(Float64(i), randn(d), randn(d)) for i in 0:100
        ]
        trace = PDMPSamplers.PDMPTrace(events, dbps)
        state = PDMPState(50.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(dbps, d)))

        PDMPSamplers.update_preconditioner!(dbps, trace, state)
        @test all(isfinite, dbps.metric.L)
        # BPS should get randn canonical velocity
        @test all(isfinite, state.ξ.θ)
    end

    @testset "update_preconditioner! DensePreconditioner non-posdef fallback" begin
        d = 2
        dzz = DensePreconditionedZigZag(d)

        # Wrap a trace so that Statistics.cov returns a non-positive-definite matrix.
        # Real trace covariances are always PSD; this tests the defensive catch block.
        struct _NonPDCovTrace <: PDMPSamplers.AbstractPDMPTrace
            flow::Any
        end
        Statistics.cov(::_NonPDCovTrace) = [1.0 2.0; 2.0 1.0]

        fake_trace = _NonPDCovTrace(dzz)
        state = PDMPState(5.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(dzz, d)))
        old_L = copy(dzz.metric.L)
        PDMPSamplers.update_preconditioner!(dzz, fake_trace, state)
        @test dzz.metric.L == old_L
    end

    @testset "sticking_time forwarding" begin
        d = 3
        pzz = PreconditionedZigZag(d)
        ξ = SkeletonPoint([1.0, -1.0, 2.0], [1.0, 1.0, -1.0])
        for i in 1:d
            t = PDMPSamplers.sticking_time(ξ, pzz, i)
            @test t >= 0
        end
    end

    @testset "ab_i forwarding" begin
        d = 3
        pzz = PreconditionedZigZag(d)
        ξ = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(pzz, d))
        c = ones(d)
        cache = (; z=similar(ξ.x), ∇ϕx=similar(ξ.x))
        for i in 1:d
            a_i, b_i = PDMPSamplers.ab_i(i, ξ, c, pzz, cache)
            @test isfinite(a_i) && isfinite(b_i)
        end
    end

    @testset "DensePreconditionedZigZag end-to-end sampling" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        D = target.D

        flow = DensePreconditionedZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 20_000.0, 5_000.0; progress=show_progress)
        @test length(trace) > 50
        @test all(isfinite, mean(trace))
    end

    @testset "DensePreconditionedBPS end-to-end sampling" begin
        d = 3
        Random.seed!(43)
        target = gen_data(Distributions.MvNormal, d, 2.0)
        D = target.D

        flow = DensePreconditionedBPS(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = GridThinningStrategy()

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, 20_000.0, 5_000.0; progress=show_progress)
        @test length(trace) > 50
        @test all(isfinite, mean(trace))
    end

    @testset "max_grid_horizon and min_grid_cells forwarding" begin
        d = 3
        pzz = PreconditionedZigZag(d)
        dzz = DensePreconditionedZigZag(d)
        pbps = PreconditionedBPS(d)

        @test PDMPSamplers.max_grid_horizon(pzz) == PDMPSamplers.max_grid_horizon(ZigZag(d))
        @test PDMPSamplers.max_grid_horizon(dzz) == PDMPSamplers.max_grid_horizon(ZigZag(d))
        @test PDMPSamplers.min_grid_cells(pzz, 5, 20) == PDMPSamplers.min_grid_cells(ZigZag(d), 5, 20)
    end

    @testset "∂λ∂t forwarding through PreconditionedDynamics" begin
        d = 3
        pzz = PreconditionedZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(pzz, d)))
        ∇U = randn(d)
        Hv = randn(d)
        result = PDMPSamplers.∂λ∂t(state, ∇U, Hv, pzz)
        @test isfinite(result)
    end
end
