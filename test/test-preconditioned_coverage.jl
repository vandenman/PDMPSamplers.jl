@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Preconditioned dynamics coverage" begin

    @testset "DensePreconditioner constructors and fields" begin
        dp = DensePreconditioner(4)
        @test dp.L ≈ Matrix{Float64}(I, 4, 4)
        @test dp.Linv ≈ Matrix{Float64}(I, 4, 4)
        @test dp.v_canonical == zeros(4)
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
        dp_dense.L .= randn(4, 4)
        dp_dense.Linv .= randn(4, 4)
        sub_dense = PDMPSamplers.subpreconditioner(dp_dense, free)
        @test sub_dense isa DensePreconditioner
        @test size(sub_dense.L) == (3, 3)
        @test size(sub_dense.Linv) == (3, 3)
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
        dp.L .= [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]
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
        M.L .= L_new
        M.Linv .= inv(LowerTriangular(L_new))
        M.v_canonical .= [1.0, -1.0, 1.0]

        ξ = SkeletonPoint(randn(d), M.L * M.v_canonical)
        ∇ϕ = randn(d)

        rate = PDMPSamplers.λ(ξ, ∇ϕ, zz)
        @test rate >= 0

        # Manual computation of the rate
        v = M.v_canonical
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
        M.L .= L_new
        M.Linv .= inv(LowerTriangular(L_new))
        M.v_canonical .= [1.0, -1.0, 1.0, -1.0]

        ξ = SkeletonPoint(randn(d), M.L * M.v_canonical)
        ∇ϕ = randn(d)
        cache = (; z=similar(ξ.x))

        θ_before = copy(ξ.θ)
        PDMPSamplers.reflect!(ξ, ∇ϕ, zz, cache)
        @test all(isfinite, ξ.θ)
        # Exactly one canonical coordinate should have flipped
        n_flipped = sum(M.v_canonical[i] != [1.0, -1.0, 1.0, -1.0][i] for i in 1:d)
        @test 0 ≤ n_flipped ≤ 1
    end

    @testset "DensePreconditionedZigZag reflect! zero-rate branch" begin
        d = 3
        zz = DensePreconditionedZigZag(d)
        M = zz.metric
        M.v_canonical .= [1.0, -1.0, 1.0]

        ξ = SkeletonPoint(randn(d), M.L * M.v_canonical)
        # Set gradient so that all v_i * grad_z_i ≤ 0 → total_rate = 0
        ∇ϕ = -(M.L') * M.v_canonical  # ensures v_i * (L'∇ϕ)_i = -v_i^2 ≤ 0
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
        dp.L .= [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]

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
        dp.L .= [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]

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
        state = StickyPDMPState(25.0, ξ, BitVector([true, false, true]), randn(d))
        old_old_vel = copy(state.old_velocity)
        PDMPSamplers.update_preconditioner!(pzz, trace, state)
        @test all(isfinite, pzz.metric.scale)
        # old_velocity for frozen coordinate should be updated
        @test state.old_velocity[2] != old_old_vel[2]
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
        @test all(isfinite, dzz.metric.Linv)
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
        # Trace with identical positions → singular covariance
        events = [
            PDMPEvent(Float64(i), [1.0, 2.0], randn(d)) for i in 0:10
        ]
        trace = PDMPSamplers.PDMPTrace(events, dzz)
        state = PDMPState(5.0, SkeletonPoint([1.0, 2.0], PDMPSamplers.initialize_velocity(dzz, d)))
        old_L = copy(dzz.metric.L)
        PDMPSamplers.update_preconditioner!(dzz, trace, state)
        # Should return without error (PosDefException caught)
        @test dzz.metric.L == old_L
    end

    @testset "freezing_time forwarding" begin
        d = 3
        pzz = PreconditionedZigZag(d)
        ξ = SkeletonPoint([1.0, -1.0, 2.0], [1.0, 1.0, -1.0])
        for i in 1:d
            t = PDMPSamplers.freezing_time(ξ, pzz, i)
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
