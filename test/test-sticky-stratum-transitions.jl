@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _dense_stratum_factor(state, flow)
    scratch = PDMPSamplers._dense_zigzag_stratum!(state, flow)
    k = scratch.active_count
    return scratch, Matrix(LowerTriangular(view(scratch.ΣAA, 1:k, 1:k)))
end

@testset "Exact sticky stratum transitions" begin
    @testset "dense ZigZag stratum algebra" begin
        Σ = [1.0 0.55 -0.2; 0.55 1.7 0.35; -0.2 0.35 1.3]
        flow = DensePreconditionedZigZag(3)
        set_dense_preconditioner!(flow.metric, cholesky(Symmetric(Σ)).L)
        state = StickyPDMPState(0.0,
            SkeletonPoint([0.0, 0.3, -0.4], zeros(3)),
            BitVector([false, true, true]))
        rng = Random.Xoshiro(0xd35e)

        PDMPSamplers.draw_stratum_velocity!(rng, state, flow)
        scratch, L_F = _dense_stratum_factor(state, flow)
        F = scratch.active[1:scratch.active_count]
        signs = copy(scratch.canonical_signs[1:scratch.active_count])
        @test L_F * L_F' ≈ Σ[F, F]
        @test state.ξ.θ[F] ≈ L_F * signs
        @test all(iszero, state.ξ.θ[.!state.free])

        gradient = [4.0, -0.7, 1.2]
        transformed = L_F' * gradient[F]
        @test PDMPSamplers.λ(state, gradient, flow) ≈
              sum(max.(0.0, signs .* transformed))

        cache = (; z=zeros(3), tmp=zeros(3))
        before = copy(signs)
        PDMPSamplers.reflect!(rng, state, gradient, flow, cache)
        scratch, L_F = _dense_stratum_factor(state, flow)
        after = copy(scratch.canonical_signs[1:scratch.active_count])
        @test sum(before .!= after) == 1
        @test state.ξ.θ[F] ≈ L_F * after
        @test iszero(state.ξ.θ[1])

        # Freeze another coordinate: the complete lower-stratum velocity is
        # redrawn from its own active covariance marginal.
        state.free[2] = false
        PDMPSamplers._invalidate_boundary_velocity_cache!(state)
        PDMPSamplers.draw_stratum_velocity!(rng, state, flow)
        scratch, L_J = _dense_stratum_factor(state, flow)
        J = scratch.active[1:scratch.active_count]
        @test J == [3]
        @test L_J * L_J' ≈ Σ[J, J]
        @test all(iszero, state.ξ.θ[[1, 2]])
    end

    @testset "dense ZigZag flux proposal is exact" begin
        Σ = [1.2 0.75 -0.15; 0.75 1.4 0.45; -0.15 0.45 1.1]
        flow = DensePreconditionedZigZag(3)
        set_dense_preconditioner!(flow.metric, cholesky(Symmetric(Σ)).L)
        state = StickyPDMPState(0.0,
            SkeletonPoint([0.2, -0.1, 0.0], [0.0, 0.0, 0.0]),
            BitVector([true, true, false]))
        rng = Random.Xoshiro(0xb0a7)
        PDMPSamplers.draw_stratum_velocity!(rng, state, flow)

        scratch = PDMPSamplers._prepare_stratum!(flow, state, 3)
        p = findfirst(==(3), scratch.active[1:scratch.active_count])
        row = collect(view(scratch.ΣAA, p, 1:p))
        A = sum(abs, row)
        sign_vectors = vec(collect(Iterators.product(ntuple(_ -> (-1.0, 1.0), p)...)))
        weights = [abs(dot(row, collect(s))) for s in sign_vectors]
        expected = weights ./ sum(weights)

        counts = zeros(Int, length(sign_vectors))
        accepted = 0
        proposals = 6_000
        saw_rejection = false
        rejection_preserved_state = true
        for _ in 1:proposals
            x_before = copy(state.ξ.x)
            θ_before = copy(state.ξ.θ)
            free_before = copy(state.free)
            ok = PDMPSamplers.propose_boundary_velocity!(rng, state, flow, 3)
            if ok
                accepted += 1
                scratch = state.boundary_scratch
                s = Tuple(scratch.canonical_signs[1:scratch.active_count])
                counts[findfirst(==(s), sign_vectors)] += 1
                state.ξ.θ[3] = 0.0
            else
                saw_rejection = true
                rejection_preserved_state &= state.ξ.x == x_before &&
                    state.ξ.θ == θ_before && state.free == free_before
            end
        end
        @test saw_rejection
        @test rejection_preserved_state
        @test accepted / proposals ≈ mean(weights) / A atol=0.03
        @test counts ./ accepted ≈ expected atol=0.045
    end

    @testset "product ZigZag flux proposals" begin
        for (seed, flow, scales) in (
                (1, ZigZag(2), [1.0, 1.0]),
                (2, PreconditionedZigZag(2; scale=[1.7, 0.6]), [1.7, 0.6]))
            state = StickyPDMPState(0.0,
                SkeletonPoint([0.0, 0.3], zeros(2)),
                BitVector([false, true]))
            rng = Random.Xoshiro(0x91a0 + seed)
            positive = zeros(Int, 2)
            n = 5_000
            all_accepted = true
            all_scales = true
            for _ in 1:n
                all_accepted &= PDMPSamplers.propose_boundary_velocity!(
                    rng, state, flow, 1)
                all_scales &= abs.(state.ξ.θ) ≈ scales
                positive .+= state.ξ.θ .> 0
                state.ξ.θ[1] = 0.0
            end
            @test all_accepted
            @test all_scales
            @test PDMPSamplers._boundary_proposal_clock_constant(flow, state, 1) == scales[1]
            @test positive ./ n ≈ fill(0.5, 2) atol=0.025
        end
    end

    @testset "callable sticky rates are validated when frozen" begin
        model = PDMPModel(1, FullGradient((out, x) -> fill!(out, 0.0)))
        initial = SkeletonPoint([0.5], [-1.0])
        function frozen_callable_fixture(value, seed)
            strategy = Sticky(GridThinningStrategy(),
                (i, x, free, velocity) -> value, trues(1))
            rng = Random.Xoshiro(seed)
            state, _, alg, _, _ = PDMPSamplers.initialize_state(
                rng, ZigZag(1), model, strategy, 0.0, initial)
            state.ξ.x[1] = 0.0
            state.ξ.θ[1] = 0.0
            state.free[1] = false
            PDMPSamplers._invalidate_active_stratum_cache!(state)
            return rng, state, alg
        end

        invalid_values = (NaN, Inf, -Inf, Exponential(1.0), :invalid, -0.1)
        for (seed, value) in enumerate(invalid_values)
            rng, state, alg = frozen_callable_fixture(value, 20 + seed)
            @test_throws ArgumentError PDMPSamplers.unfreeze_time(
                rng, alg, state, ZigZag(1), 1)
        end

        zero_rng, zero_state, zero_alg = frozen_callable_fixture(0.0, 30)
        next_draw = rand(copy(zero_rng))
        @test isinf(PDMPSamplers.unfreeze_time(
            zero_rng, zero_alg, zero_state, ZigZag(1), 1))
        @test rand(zero_rng) == next_draw

        valid_rng, valid_state, valid_alg = frozen_callable_fixture(2.0, 31)
        τ = PDMPSamplers.unfreeze_time(
            valid_rng, valid_alg, valid_state, ZigZag(1), 1)
        @test isfinite(τ)
        @test τ > 0
    end

    @testset "zero sticky rates remain frozen through a finite horizon" begin
        model = PDMPModel(1, FullGradient((out, x) -> fill!(out, 0.0)))
        initial = SkeletonPoint([0.0], [0.0])
        horizon = 1.25
        strategies = (
            Sticky(GridThinningStrategy(), [0.0], trues(1)),
            Sticky(GridThinningStrategy(),
                (i, x, free, velocity) -> 0.0, trues(1)),
        )

        for (seed, strategy) in enumerate(strategies)
            trace, stats = pdmp_sample(initial, ZigZag(1), model, strategy,
                0.0, horizon; seed=seed, progress=false)
            @test last_event_time(trace) == horizon
            @test stats.stop_reason == :reached_time
            @test stats.sticky_unfreezes == 0
            @test all(.!trace.free_masks)
            @test all(iszero, trace.positions)
            @test all(iszero, trace.velocities)
        end
    end

    @testset "vector sticky rates are validated at initialization" begin
        model = PDMPModel(2, FullGradient((out, x) -> fill!(out, 0.0)))
        initial = SkeletonPoint([0.2, -0.3], [1.0, -1.0])
        @test_throws DimensionMismatch PDMPSamplers.initialize_state(
            Random.Xoshiro(18), ZigZag(2), model,
            Sticky(GridThinningStrategy(), [1.0]), 0.0, initial)
        for κ in ([1.0, -0.1], [1.0, NaN], [1.0, -Inf])
            @test_throws ArgumentError PDMPSamplers.initialize_state(
                Random.Xoshiro(18), ZigZag(2), model,
                Sticky(GridThinningStrategy(), κ), 0.0, initial)
        end
        @test_throws ArgumentError PDMPSamplers.initialize_state(
            Random.Xoshiro(18), ZigZag(2), model,
            Sticky(GridThinningStrategy(), [1.0, Inf], trues(2)),
            0.0, initial)
        state, _, alg, _, _ = PDMPSamplers.initialize_state(
            Random.Xoshiro(19), ZigZag(2), model,
            Sticky(GridThinningStrategy(), [1.0, Inf]), 0.0, initial)
        @test alg.can_stick == BitVector([true, false])
        @test state.free == trues(2)
    end

    @testset "aggregate rejected unfreeze only resamples aggregate clock" begin
        flow = DensePreconditionedZigZag(2)
        set_dense_preconditioner!(flow.metric,
            cholesky(Symmetric([1.0 0.92; 0.92 1.0])).L)
        model = PDMPModel(2, FullGradient((out, x) -> copyto!(out, x)))
        make_strategy() = AggregateSticky(GridThinningStrategy(),
            PDMPSamplers.LinearGaussianAggregateClock(
                PDMPSamplers.DenseGaussianSlab(
                    zeros(2), Matrix{Float64}(I, 2, 2), 1:2),
                BernoulliModelPrior(fill(0.5, 2))), trues(2))
        initial = SkeletonPoint([0.4, 0.0], [1.0, 0.0])
        rejected, _, rejected_alg, _, _ = PDMPSamplers.initialize_state(
            Random.Xoshiro(42), flow, model, make_strategy(), 0.0, initial)
        control, _, control_alg, _, _ = PDMPSamplers.initialize_state(
            Random.Xoshiro(42), flow, model, make_strategy(), 0.0, initial)
        rejection_seed = findfirst(1:10_000) do seed
            probe = copy(rejected)
            !PDMPSamplers.propose_boundary_velocity!(
                Random.Xoshiro(seed), probe, flow, 2)
        end
        @test rejection_seed !== nothing

        sticky_times_before = copy(rejected_alg.sticky_times)
        priority_before = collect(rejected_alg.sticky_pq)
        aggregate_before = rejected_alg.aggregate_unstick_time
        rejected_rng = Random.Xoshiro(something(rejection_seed))
        control_rng = Random.Xoshiro(something(rejection_seed))
        @test !PDMPSamplers.stick_or_unstick!(
            rejected_rng, rejected, flow, rejected_alg, 2)

        disposable = copy(control)
        @test !PDMPSamplers.propose_boundary_velocity!(
            control_rng, disposable, flow, 2)
        PDMPSamplers.update_all_unfreeze_times!(
            control_rng, control_alg, control, flow)

        @test rejected_alg.sticky_times == sticky_times_before
        @test collect(rejected_alg.sticky_pq) == priority_before
        @test !isequal(rejected_alg.aggregate_unstick_time, aggregate_before)
        @test rejected_alg.aggregate_unstick_time ==
              control_alg.aggregate_unstick_time
        @test rejected.ξ.x == control.ξ.x
        @test rejected.ξ.θ == control.ξ.θ
        @test rejected.free == control.free
        @test rand(copy(rejected_rng)) == rand(copy(control_rng))
    end

    @testset "plain sticky BPS reflection is allocation-free" begin
        state = StickyPDMPState(0.0,
            SkeletonPoint([0.2, 0.0, -0.4], [1.0, 0.0, -0.5]),
            BitVector([true, false, true]))
        gradient = [0.3, 7.0, -1.2]
        cache = (; z=zeros(3))
        flow = BouncyParticle(3, 0.0)
        PDMPSamplers.reflect!(Random.Xoshiro(1), state, gradient, flow, cache)
        before = copy(state.ξ.θ)
        rng = Random.Xoshiro(2)
        bytes = @allocated PDMPSamplers.reflect!(
            rng, state, gradient, flow, cache)
        @test bytes == 0
        @test iszero(state.ξ.θ[2])
        @test dot(state.ξ.θ[state.free], gradient[state.free]) ≈
              -dot(before[state.free], gradient[state.free])
    end

    @testset "Gaussian flux moments and low-rank equivalence" begin
        Σ = [1.3 0.45; 0.45 0.9]
        diagonal_boomerang_covariance = Diagonal([1.4, 0.8])
        diagonal_boomerang = Boomerang(
            inv(diagonal_boomerang_covariance), zeros(2), 0.0)
        dense_bps = DensePreconditionedBPS(2; refresh_rate=0.0)
        set_dense_preconditioner!(dense_bps.metric, cholesky(Symmetric(Σ)).L)
        dense_boomerang = Boomerang(inv(Symmetric(Σ)), zeros(2), 0.0)

        adaptive_diagonal = AdaptiveBoomerang(2; λref=0.0, scheme=:diagonal)
        adaptive_fullrank = AdaptiveBoomerang(2; λref=0.0, scheme=:fullrank)

        lowrank = AdaptiveBoomerang(2; λref=0.0, scheme=:lowrank, rank=1)
        lowrank.Γ.D .= [1.2, 0.8]
        lowrank.Γ.V .= [0.5; 0.3]
        lowrank.Γ.Λ .= [0.4]
        PDMPSamplers.lowrank_precompute!(lowrank.Γ)
        Σ_lr = Diagonal(lowrank.Γ.D) +
               lowrank.Γ.V * Diagonal(lowrank.Γ.Λ) * lowrank.Γ.V'
        equivalent_dense = Boomerang(inv(Symmetric(Σ_lr)), zeros(2), 0.0)

        for (seed, flow, covariance) in (
                (1, BouncyParticle(2, 0.0), Matrix{Float64}(I, 2, 2)),
                (2, PreconditionedBPS(2; refresh_rate=0.0, scale=[1.4, 0.7]),
                    Diagonal([1.4^2, 0.7^2])),
                (3, dense_bps, Σ),
                (4, diagonal_boomerang, diagonal_boomerang_covariance),
                (5, dense_boomerang, Σ),
                (6, adaptive_diagonal, Matrix{Float64}(I, 2, 2)),
                (7, adaptive_fullrank, Matrix{Float64}(I, 2, 2)),
                (8, lowrank, Σ_lr),
                (9, equivalent_dense, Σ_lr))
            state = StickyPDMPState(0.0,
                SkeletonPoint([0.0, 0.2], [0.0, 0.0]),
                BitVector([false, true]))
            rng = Random.Xoshiro(seed)
            n = 4_000
            second_i = 0.0
            cross = 0.0
            second_j = 0.0
            for _ in 1:n
                PDMPSamplers.propose_boundary_velocity!(rng, state, flow, 1)
                vi, vj = state.ξ.θ
                second_i += vi^2
                cross += vi * vj
                second_j += vj^2
                state.ξ.θ[1] = 0.0
            end
            second_i /= n
            cross /= n
            second_j /= n
            @test PDMPSamplers._boundary_proposal_clock_constant(flow, state, 1) ≈
                  sqrt(2 / π) * sqrt(covariance[1, 1])
            @test second_i ≈ 2covariance[1, 1] rtol=0.07
            @test cross ≈ 2covariance[1, 2] atol=0.10
            expected_j = covariance[2, 2] +
                         covariance[2, 1]^2 / covariance[1, 1]
            @test second_j ≈ expected_j rtol=0.08
        end

        state_lr = StickyPDMPState(0.0,
            SkeletonPoint([0.0, 0.1], zeros(2)), BitVector([false, true]))
        state_dense = copy(state_lr)
        @test PDMPSamplers._boundary_proposal_clock_constant(lowrank, state_lr, 1) ≈
              PDMPSamplers._boundary_proposal_clock_constant(equivalent_dense, state_dense, 1)
        column_lr = zeros(2)
        column_dense = zeros(2)
        PDMPSamplers._gaussian_covariance_column!(column_lr, lowrank, 1,
            PDMPSamplers._prepare_stratum!(lowrank, state_lr, 1))
        PDMPSamplers._gaussian_covariance_column!(column_dense,
            equivalent_dense, 1,
            PDMPSamplers._prepare_stratum!(equivalent_dense, state_dense, 1))
        @test column_lr ≈ column_dense
    end

    @testset "dense metric changes preserve stratum invariants atomically" begin
        flow = DensePreconditionedZigZag(2)
        initial_factor = [1.0 0.0; 0.35 1.2]
        set_dense_preconditioner!(flow.metric, initial_factor)
        state = StickyPDMPState(0.0,
            SkeletonPoint([0.0, 0.4], zeros(2)),
            BitVector([false, true]))
        rng = Random.Xoshiro(0xa7a7)
        PDMPSamplers.draw_stratum_velocity!(rng, state, flow)

        factor_before = copy(flow.metric.L)
        velocity_before = copy(state.ξ.θ)
        generation_before = flow.metric.generation
        @test_throws SingularException set_dense_preconditioner!(
            flow.metric, zeros(2, 2))
        @test flow.metric.L == factor_before
        @test flow.metric.generation == generation_before
        @test state.ξ.θ == velocity_before
        @test state.free == BitVector([false, true])

        updated_factor = [1.3 0.0; -0.2 0.9]
        set_dense_preconditioner!(flow.metric, updated_factor)
        PDMPSamplers._invalidate_boundary_velocity_cache!(state)
        PDMPSamplers.draw_stratum_velocity!(rng, state, flow)
        scratch, L_F = _dense_stratum_factor(state, flow)
        signs = view(scratch.canonical_signs, 1:scratch.active_count)
        @test iszero(state.ξ.θ[1])
        @test state.ξ.θ[2:2] ≈ L_F * signs
    end

    @testset "Gaussian sticky transitions reuse covariance representations" begin
        dense = DensePreconditionedBPS(6; refresh_rate=0.0)
        set_dense_preconditioner!(dense.metric,
            cholesky(Symmetric([1.0 + (i == j ? 1.0 : 0.2 / (1 + abs(i-j)))
                for i in 1:6, j in 1:6])).L)
        lowrank = AdaptiveBoomerang(6; λref=0.0, scheme=:lowrank, rank=2)
        lowrank.Γ.V .= reshape(range(-0.3, 0.3; length=12), 6, 2)
        lowrank.Γ.Λ .= [0.2, 0.4]
        PDMPSamplers.lowrank_precompute!(lowrank.Γ)

        for (seed, flow) in ((61, BouncyParticle(6, 0.0)),
                             (62, PreconditionedBPS(6; scale=1:6)),
                             (63, dense), (64, Boomerang(6)), (65, lowrank))
            state = StickyPDMPState(0.0,
                SkeletonPoint([0.0, 0.0, 0.2, -0.3, 0.4, -0.5], zeros(6)),
                BitVector([false, false, true, true, true, true]))
            rng = Random.Xoshiro(seed)
            for transition in 1:100
                i = isodd(transition) ? 1 : 2
                @test PDMPSamplers.propose_boundary_velocity!(rng, state, flow, i)
                state.free[i] = true
                state.free[3 - i] = false
                state.ξ.x[3 - i] = 0.0
                state.ξ.θ[3 - i] = 0.0
                PDMPSamplers._invalidate_active_stratum_cache!(state)
            end
            @test size(state.boundary_scratch.ΣAA) == (0, 0)
            @test !state.boundary_scratch.active_factor_valid
        end
    end

    @testset "short production transition matrix" begin
        flows = (
            ZigZag(2), BouncyParticle(2),
            PreconditionedZigZag(2; scale=[1.0, 1.6]),
            PreconditionedBPS(2; refresh_rate=0.2, scale=[1.0, 1.6]),
            DensePreconditionedZigZag(2), DensePreconditionedBPS(2),
            Boomerang(2), AdaptiveBoomerang(2; scheme=:fullrank),
            AdaptiveBoomerang(2; scheme=:lowrank, rank=1),
        )
        for (seed, flow) in enumerate(flows)
            model = PDMPModel(2, FullGradient((g, x) -> copyto!(g, x)))
            strategy = Sticky(GridThinningStrategy(), ones(2))
            rng = Random.Xoshiro(seed)
            initial = SkeletonPoint([1.0, -1.0],
                PDMPSamplers.initialize_velocity(rng, flow, 2))
            trace, stats = pdmp_sample(initial, flow, model, strategy,
                0.0, 30.0; seed, progress=false)
            @test stats.sticky_events >= 10
            @test stats.sticky_freezes >= 2
            @test stats.sticky_unfreezes >= 2
            @test length(trace) >= 10
            @test all(isfinite, last(trace).velocity)
            ntrace = size(trace.free_masks, 2)
            @test count(view(trace.free_masks, :, 1:(ntrace - 1)) .&
                        .!view(trace.free_masks, :, 2:ntrace)) >= 2
            @test count(.!view(trace.free_masks, :, 1:(ntrace - 1)) .&
                        view(trace.free_masks, :, 2:ntrace)) >= 2
        end
    end

    @testset "AggregateSticky transition matrix" begin
        flows = (
            ZigZag(2), BouncyParticle(2),
            PreconditionedZigZag(2; scale=[1.0, 1.6]),
            PreconditionedBPS(2; refresh_rate=0.2, scale=[1.0, 1.6]),
            DensePreconditionedZigZag(2), DensePreconditionedBPS(2),
            Boomerang(2), AdaptiveBoomerang(2; scheme=:fullrank),
            AdaptiveBoomerang(2; scheme=:lowrank, rank=1),
        )
        for (seed, flow) in enumerate(flows)
            model = PDMPModel(2, FullGradient((g, x) -> copyto!(g, x)))
            clock = PDMPSamplers.LinearGaussianAggregateClock(
                PDMPSamplers.DenseGaussianSlab(
                    zeros(2), Matrix{Float64}(I, 2, 2), 1:2),
                BernoulliModelPrior(fill(0.5, 2)))
            strategy = AggregateSticky(GridThinningStrategy(), clock, trues(2))
            rng = Random.Xoshiro(0xa660 + seed)
            initial = SkeletonPoint([1.0, -1.0],
                PDMPSamplers.initialize_velocity(rng, flow, 2))
            trace, stats = pdmp_sample(initial, flow, model, strategy,
                0.0, 30.0; seed=0xa660 + seed, progress=false)
            @test stats.sticky_events >= 5
            @test stats.sticky_freezes >= 1
            @test stats.sticky_unfreezes >= 1
            @test length(trace) >= 5
            @test all(isfinite, last(trace).velocity)
            ntrace = size(trace.free_masks, 2)
            @test count(view(trace.free_masks, :, 1:(ntrace - 1)) .&
                        .!view(trace.free_masks, :, 2:ntrace)) >= 1
            @test count(.!view(trace.free_masks, :, 1:(ntrace - 1)) .&
                        view(trace.free_masks, :, 2:ntrace)) >= 1
        end
    end

    @testset "non-diagonal dense ZigZag long sticky stationarity" begin
        function dense_flow()
            flow = DensePreconditionedZigZag(2)
            set_dense_preconditioner!(flow.metric,
                cholesky(Symmetric([1.0 0.8; 0.8 1.0])).L)
            return flow
        end
        model = PDMPModel(2, FullGradient((out, x) -> copyto!(out, x)),
                          (out, x, v) -> copyto!(out, v))
        κ = fill(pdf(Normal(), 0.0), 2)
        clock = PDMPSamplers.LinearGaussianAggregateClock(
            PDMPSamplers.DenseGaussianSlab(
                zeros(2), Matrix{Float64}(I, 2, 2), 1:2),
            BernoulliModelPrior(fill(0.5, 2)))
        strategies = (
            Sticky(GridThinningStrategy(), κ),
            AggregateSticky(GridThinningStrategy(), clock, trues(2)),
        )
        for (seed, strategy) in enumerate(strategies)
            flow = dense_flow()
            trace, stats = pdmp_sample(
                SkeletonPoint([0.3, -0.4], flow.metric.L * [1.0, -1.0]),
                flow, model, strategy, 0.0, 20_000.0;
                seed=0x5a70 + seed, progress=false)
            @test mean(trace) ≈ zeros(2) atol=0.14
            @test diag(cov(trace)) ≈ fill(0.5, 2) atol=0.16
            @test stats.sticky_freezes > 30
            @test stats.sticky_unfreezes > 30
            @test stats.sticky_unfreeze_rejections > 0
        end
    end
end
