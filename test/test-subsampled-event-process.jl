@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _subsampling_fixture(weights, m; scales=ones(size(weights, 1)))
    envelope = SeparableResidualEnvelope(weights,
        (out, state, flow, t) -> copyto!(out, scales);
        certified_affine=true)
    oracle = (out, x, subset, anchor) -> fill!(out, 0.0)
    return SubsampledControlVariate((out, x) -> fill!(out, 0.0), oracle,
        envelope, [0.0], m)
end

function _subset_key(values)
    return Tuple(sort!(collect(values)))
end

function _enumerated_subsets(N, m)
    result = Tuple[]
    for mask in 0:(UInt(1) << N) - 1
        count_ones(mask) == m || continue
        push!(result, Tuple(i for i in 1:N if !iszero(mask & (UInt(1) << (i - 1)))))
    end
    return result
end

mutable struct _CopyTrackedHVP
    calls::Int
end
function (h::_CopyTrackedHVP)(out, x, v)
    h.calls += 1
    copyto!(out, v)
end
Base.copy(::_CopyTrackedHVP) = _CopyTrackedHVP(0)

mutable struct _SwitchingGradient
    calls::Int
    switch_after::Int
    low::Float64
    high::Float64
end
function (f::_SwitchingGradient)(out, x)
    f.calls += 1
    out[1] = f.calls <= f.switch_after ? f.low : f.high
end

mutable struct _SwitchingScale
    calls::Int
    switch_after::Int
    low::Float64
    high::Float64
end
function (f::_SwitchingScale)(out, state, flow, t)
    f.calls += 1
    out[1] = f.calls <= f.switch_after ? f.low : f.high
end

@testset "Subsampling subsampled event process" begin
    @testset "public API has no legacy terminology" begin
        legacy_term = "mark" * "ed"
        @test all(name -> !occursin(legacy_term, lowercase(String(name))),
            names(PDMPSamplers; all=false, imported=false))
        for type in (SubsampledControlVariate,
                PDMPSamplers.SubsamplingThinningState,
                PDMPSamplers.SubsamplingCounter,
                PDMPSamplers.SubsamplingAnchorBankAdapter)
            @test all(name -> !occursin(legacy_term, lowercase(String(name))),
                fieldnames(type))
        end
    end

    @testset "residual-envelope derived caches are constructed coherently" begin
        envelope = SeparableResidualEnvelope(
            [1.0 2.0; 3.0 4.0],
            (out, state, flow, t) -> fill!(out, 1.0);
            certified_affine=true)
        @test envelope.weights == [1.0 2.0; 3.0 4.0]
        @test envelope.totals == [3.0, 7.0]
        @test all(!isnothing, envelope.alias_tables)
        @test length(envelope.scales) == 2
        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        @test PDMPSamplers.total_residual_bound(
            envelope, state, BouncyParticle(1, 0.0), 0.0) == 10.0
        @test envelope.cumulative_masses == [3.0, 10.0]

        affine = SeparableResidualEnvelope(ones(1, 1),
            (out, state, flow, t) -> (out[1] = t);
            certified_affine=true)
        PDMPSamplers.component_cell_scales!(affine.cell_scales, affine,
            state, BouncyParticle(1, 0.0), 1.0, 2.0)
        @test affine.cell_scales == [2.0]

        nonaffine = (out, state, flow, t) -> (out[1] = 1 - (2t - 1)^2)
        @test_throws ArgumentError SeparableResidualEnvelope(
            ones(1, 1), nonaffine)

        grouped_scales = [2.0, 3.0, 5.0, 7.0]
        grouped_callback = (out, state, flow, args...) ->
            copyto!(out, grouped_scales)
        grouped = PDMPSamplers.GroupedResidualEnvelope(
            [1 1 2; 3 4 4], 4, grouped_callback;
            component_cell_scales! = grouped_callback)
        @test grouped.totals == [2.0, 1.0, 1.0, 2.0]
        @test PDMPSamplers.n_observations(grouped) == 3
        PDMPSamplers.component_scales!(grouped.scales, grouped,
            state, BouncyParticle(1, 0.0), 0.0)
        @test [PDMPSamplers.observation_residual_bound(grouped, i)
            for i in 1:3] == [7.0, 9.0, 10.0]
        @test PDMPSamplers.total_residual_bound(
            grouped, state, BouncyParticle(1, 0.0), 0.0) == 26.0
        @test grouped.cumulative_masses == [4.0, 7.0, 12.0, 26.0]
        grouped_cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            grouped, [0.0], 1)
        grouped_subset_bound = PDMPSamplers.draw_subset!(
            MersenneTwister(81), grouped_cv, 0.0, 26.0)
        @test grouped_subset_bound == 3 *
            PDMPSamplers.observation_residual_bound(
                grouped, only(grouped_cv.subset))

        block_weights = [1.0 2.0 3.0; 0.0 1.0 0.0;
                         4.0 0.0 1.0; 1.0 1.0 1.0]
        block_scales = [2.0, 3.0, 5.0, 7.0]
        block_callback = (out, state, flow, args...) ->
            copyto!(out, block_scales)
        block = BlockSeparableResidualEnvelope(
            block_weights, [1, 1, 2, 2], 2, block_callback;
            component_cell_scales! = block_callback)
        @test block.totals == [6.0, 1.0, 5.0, 3.0]
        @test PDMPSamplers.n_observations(block) == 6
        PDMPSamplers.component_scales!(block.scales, block,
            state, BouncyParticle(1, 0.0), 0.0)
        @test [PDMPSamplers.observation_residual_bound(block, i)
            for i in 1:6] == [2.0, 7.0, 6.0, 27.0, 7.0, 12.0]
        @test PDMPSamplers.total_residual_bound(
            block, state, BouncyParticle(1, 0.0), 0.0) == 61.0
        @test_throws DimensionMismatch BlockSeparableResidualEnvelope(
            block_weights, [1, 2], 2, block_callback;
            component_cell_scales! = block_callback)
        @test_throws ArgumentError BlockSeparableResidualEnvelope(
            block_weights, [1, 1, 3, 3], 2, block_callback;
            component_cell_scales! = block_callback)
        block_cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            block, [0.0], 1)
        block_subset_bound = PDMPSamplers.draw_subset!(
            MersenneTwister(82), block_cv, 0.0, 61.0)
        @test block_subset_bound == 6 *
            PDMPSamplers.observation_residual_bound(
                block, only(block_cv.subset))
    end

    @testset "trajectory residual geometry dominates every supported flow" begin
        anchor = [-0.4, 0.25]
        x = [0.7, -0.3]
        velocity = [0.6, -1.1]
        flows = ContinuousDynamics[
            BouncyParticle(2, 0.0),
            ZigZag(2),
            PreconditionedBPS(2; refresh_rate=0.0, scale=[0.6, 1.1]),
            PreconditionedZigZag(2; scale=[0.6, 1.1]),
            DensePreconditionedBPS(2; refresh_rate=0.0),
            DensePreconditionedZigZag(2),
            Boomerang(Matrix{Float64}(I, 2, 2), [0.2, -0.1], 0.0),
            AdaptiveBoomerang(2; λref=0.0),
        ]
        for flow in flows
            state = PDMPState(0.0, SkeletonPoint(copy(x), copy(velocity)))
            PDMPSamplers.initialize_flow_state!(state, flow)
            envelope = PDMPSamplers.TrajectoryResidualEnvelope(
                ones(2, 3), anchor; growth_rates=[0.0, 0.8])
            left, right = 0.17, 0.91
            cell = zeros(2)
            PDMPSamplers.component_cell_scales!(cell, envelope, state, flow,
                left, right)
            for t in range(left, right; length=101)
                point = zeros(2)
                PDMPSamplers.component_scales!(point, envelope, state, flow, t)
                @test all(point .<= cell .* (1 + 1e-12) .+ 1e-12)
            end
        end

        sticky = StickyPDMPState(Ref(0.0),
            SkeletonPoint(copy(x), [0.0, velocity[2]]),
            BitVector([false, true]))
        flow = Boomerang(Matrix{Float64}(I, 2, 2), [0.2, -0.1], 0.0)
        envelope = PDMPSamplers.TrajectoryResidualEnvelope(
            ones(1, 2), anchor; growth_rates=[0.4])
        cell = zeros(1)
        PDMPSamplers.component_cell_scales!(cell, envelope, sticky, flow,
            0.0, 1.2)
        for t in range(0.0, 1.2; length=101)
            point = zeros(1)
            PDMPSamplers.component_scales!(point, envelope, sticky, flow, t)
            @test point[1] <= cell[1] * (1 + 1e-12) + 1e-12
        end
    end

    @testset "damped HCV rank-two trajectory envelope" begin
        anchor = [-0.4, 0.25]
        x = [0.7, -0.3]
        velocity = [0.6, -1.1]
        damping = 2.7
        flows = ContinuousDynamics[
            BouncyParticle(2, 0.0), ZigZag(2),
            PreconditionedBPS(2; refresh_rate=0.0, scale=[0.6, 1.1]),
            PreconditionedZigZag(2; scale=[0.6, 1.1]),
            DensePreconditionedBPS(2; refresh_rate=0.0),
            DensePreconditionedZigZag(2),
            Boomerang(Matrix{Float64}(I, 2, 2), [0.2, -0.1], 0.0),
            AdaptiveBoomerang(2; λref=0.0),
        ]
        for flow in flows
            state = PDMPState(0.0, SkeletonPoint(copy(x), copy(velocity)))
            PDMPSamplers.initialize_flow_state!(state, flow)
            envelope = DampedHCVResidualEnvelope(
                [0.5, 1.0, 1.5], [0.2, 0.4, 0.6], anchor;
                damping)
            left, right = 0.17, 0.91
            cell = zeros(2)
            PDMPSamplers.component_cell_scales!(cell, envelope, state, flow,
                left, right)
            for t in range(left, right; length=101)
                point = zeros(2)
                PDMPSamplers.component_scales!(point, envelope, state, flow, t)
                @test all(point .<= cell .* (1 + 1e-12) .+ 1e-12)
                δ = PDMPSamplers.trajectory_displacement_bound(
                    flow, state, anchor, t)
                V = PDMPSamplers.residual_rate_dual_bound(flow, state, t)
                α = damping / (damping + δ^2)
                @test point ≈ [V * δ * (1 - α), V * δ^2 * α]
            end
        end

        @test_throws DimensionMismatch DampedHCVResidualEnvelope(
            [1.0], [1.0, 2.0], anchor)
        @test_throws ArgumentError DampedHCVResidualEnvelope(
            [1.0], [1.0], anchor; damping=0.0)
        integer_anchor_envelope = DampedHCVResidualEnvelope(
            [1.0], [1.0], [0, 1])
        @test eltype(integer_anchor_envelope.component_scales!.anchor) === Float64
        integer_anchor_envelope.component_scales!.anchor .= [0.25, 1.5]
        @test integer_anchor_envelope.component_scales!.anchor == [0.25, 1.5]
    end

    @testset "chain-local deterministic HVP" begin
        provider = _CopyTrackedHVP(0)
        envelope = SeparableResidualEnvelope(zeros(1, 1),
            (out, state, flow, t) -> (out[1] = 0.0);
            certified_affine=true)
        cv = SubsampledControlVariate((out, x) -> copyto!(out, x),
            (out, x, subset, anchor) -> fill!(out, 0.0), envelope, [0.0], 1;
            deterministic_hvp! = provider)
        @test_throws ArgumentError PDMPModel(1, cv,
            (out, x, v) -> copyto!(out, v))
        copied = copy(PDMPModel(1, cv))
        copied.hvp([0.0], [1.0])
        @test provider.calls == 0
        @test copied.grad.deterministic_hvp!.calls == 1
    end

    @testset "support-boundary detection is explicit" begin
        envelope = SeparableResidualEnvelope(zeros(1, 1),
            (out, state, flow, t) -> (out[1] = 0.0);
            certified_affine=true)
        cv = SubsampledControlVariate((out, x) -> copyto!(out, x),
            (out, x, subset, anchor) -> fill!(out, 0.0), envelope, [0.0], 1;
            deterministic_hvp! = ((out, x, v) -> copyto!(out, v)))
        model = PDMPModel(1, cv)
        flow = BouncyParticle(1, 0.0)
        rng = Random.Xoshiro(81)
        state, counted_model, alg, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, GridThinningStrategy(lazy=false), 0.0,
            SkeletonPoint([0.2], [1.0]))
        @test_throws ArgumentError PDMPSamplers.next_event_time(rng,
            counted_model, flow, alg, state, cache, stats, Inf, true,
            :horizon_hit, true)
    end

    @testset "held-mark event clock" begin
        rng = Random.Xoshiro(0x51a7)
        n = 40_000
        correct = Vector{Float64}(undef, n)
        held = similar(correct)
        wrong_fresh = similar(correct)
        for k in 1:n
            correct[k] = randexp(rng) / 2
            held[k] = randexp(rng) / (rand(rng, Bool) ? 1 : 3)
            # The known clamped wrong-envelope construction has rate 4/3.
            wrong_fresh[k] = randexp(rng) / (4 / 3)
        end
        @test mean(correct) ≈ 1 / 2 atol=0.008
        @test mean(held) ≈ 2 / 3 atol=0.012
        @test mean(wrong_fresh) ≈ 3 / 4 atol=0.015
        # Integrated hazards for the correct clock are Exp(1).
        transformed = 2 .* correct
        @test mean(transformed) ≈ 1 atol=0.015
        @test mean(transformed .<= log(2)) ≈ 0.5 atol=0.012

        # Exercise the production candidate loop, not only the scalar laws.
        weights = reshape([0.5, 1.5], 1, :)
        envelope = SeparableResidualEnvelope(weights,
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        oracle = (out, x, subset, anchor) ->
            (out[1] = only(subset) == 1 ? 0.5 : 1.5)
        cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0), oracle,
            envelope, [0.0], 1;
            deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
        flow = BouncyParticle(1, 0.0)
        event_rng = Random.Xoshiro(0xc10c)
        state, counted_model, alg, cache, stats = PDMPSamplers.initialize_state(
            event_rng, flow, PDMPModel(1, cv),
            GridThinningStrategy(N=1, t_max=20.0, lazy=false,
                bound_violation=:throw), 0.0, SkeletonPoint([0.0], [1.0]))
        alg.schedule_frozen[] = true
        production_waits = [first(PDMPSamplers.next_event_time(event_rng,
            counted_model, flow, alg, state, cache, stats)) for _ in 1:5_000]
        @test mean(production_waits) ≈ 1 / 2 atol=0.018
        @test mean(2 .* production_waits .<= log(2)) ≈ 0.5 atol=0.018
    end

    @testset "size-biased subset law" begin
        rng = Random.Xoshiro(0x5ab5e7)
        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        cases = (
            (reshape([0.2, 0.7, 1.1, 2.0], 1, :), [1.0], 2, 0.8),
            ([0.0 0.4 1.2 0.2; 0.5 0.0 0.3 1.4], [0.7, 1.3], 2, 0.6),
            (reshape([0.2, 0.7, 1.1, 2.0], 1, :), [1.0], 1, 0.0),
            (reshape([0.2, 0.7, 1.1, 2.0], 1, :), [1.0], 4, 0.8),
            (zeros(2, 4), [1.0, 2.0], 2, 0.8),
        )
        for (weights, scales, m, D) in cases
            cv = _subsampling_fixture(weights, m; scales)
            B = PDMPSamplers.total_residual_bound(cv.envelope, state, 0.0)
            subsets = _enumerated_subsets(size(weights, 2), m)
            counts = Dict(S => 0 for S in subsets)
            draws = 20_000
            max_bound_error = 0.0
            for _ in 1:draws
                M = PDMPSamplers.draw_subset!(rng, cv, D, B)
                S = _subset_key(cv.subset)
                counts[S] += 1
                expected_M = D + size(weights, 2) / m * sum(
                    scales[r] * sum(weights[r, i] for i in S)
                    for r in axes(weights, 1))
                max_bound_error = max(max_bound_error, abs(M - expected_M))
            end
            @test max_bound_error < 1e-12
            uniform_probability = inv(binomial(size(weights, 2), m))
            for S in subsets
                subset_bound = D + size(weights, 2) / m * sum(
                    scales[r] * sum(weights[r, i] for i in S)
                    for r in axes(weights, 1))
                exact = iszero(D + B) ? uniform_probability :
                    uniform_probability * subset_bound / (D + B)
                @test counts[S] / draws ≈ exact atol=0.012
            end
        end
    end

    @testset "balanced stratified size-biased subset law" begin
        rng = Random.Xoshiro(0x57a71f1ed)
        weights = reshape([0.2, 0.7, 1.1, 0.4, 1.7, 0.9], 1, :)
        scales = [1.3]
        envelope = SeparableResidualEnvelope(weights,
            (out, state, flow, t) -> copyto!(out, scales);
            certified_affine=true)
        design = PDMPSamplers.balanced_stratified_subsampling_design(6, 2, 4)
        oracle = (out, x, subset, anchor) -> fill!(out, 0.0)
        cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0), oracle, envelope, [0.0], 4;
            subset_design=design)
        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        B = PDMPSamplers.total_residual_bound(envelope, state, 0.0)
        D = 0.8
        subsets = Tuple[]
        for left in _enumerated_subsets(3, 2), right0 in _enumerated_subsets(3, 2)
            right = Tuple(i + 3 for i in right0)
            push!(subsets, Tuple(sort(vcat(collect(left), collect(right)))))
        end
        counts = Dict(S => 0 for S in subsets)
        draws = 30_000
        balanced = true
        max_bound_error = 0.0
        for _ in 1:draws
            M = PDMPSamplers.draw_subset!(rng, cv, D, B)
            S = _subset_key(cv.subset)
            balanced &= count(<=(3), S) == 2 && count(>(3), S) == 2
            counts[S] += 1
            expected_M = D + 6 / 4 * sum(scales[1] * weights[1, i] for i in S)
            max_bound_error = max(max_bound_error, abs(M - expected_M))
        end
        @test balanced
        @test max_bound_error < 1e-12
        uniform_probability = 1 / length(subsets)
        for S in subsets
            subset_bound = D + 6 / 4 * sum(scales[1] * weights[1, i] for i in S)
            exact = uniform_probability * subset_bound / (D + B)
            @test counts[S] / draws ≈ exact atol=0.01
        end
    end

    @testset "candidate lifecycle and stored reflection gradient" begin
        subsets_seen = Int[]
        weights = reshape([5.0, 15.0], 1, :)
        envelope = SeparableResidualEnvelope(weights,
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        oracle = function (out, x, subset, anchor)
            push!(subsets_seen, only(subset))
            out[1] = only(subset) == 1 ? 0.5 : 1.5
        end
        cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0), oracle,
            envelope, [7.0], 1;
            deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
        model = PDMPModel(1, cv)
        flow = BouncyParticle(1, 0.0)
        ξ = SkeletonPoint([0.0], [1.0])
        rng = Random.Xoshiro(0xcad1da7e)
        state, counted_model, alg, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, GridThinningStrategy(N=2, t_max=20.0,
                lazy=false, bound_violation=:throw), 0.0, ξ)
        x_before, v_before, t_before = copy(state.ξ.x), copy(state.ξ.θ), state.t[]
        τ, event, meta = PDMPSamplers.next_event_time(rng, counted_model,
            flow, alg, state, cache, stats)
        @test event === :reflect
        @test length(subsets_seen) > 1
        @test all(i -> i in (1, 2), subsets_seen)
        @test state.ξ.x == x_before
        @test state.ξ.θ == v_before
        @test state.t[] == t_before
        @test counted_model.grad.anchor == [7.0]
        @test stats.residual_oracle_calls == length(subsets_seen)
        oracle_calls = stats.residual_oracle_calls
        PDMPSamplers.handle_event!(rng, τ, counted_model, flow, alg, state,
            cache, event, meta, stats)
        @test stats.residual_oracle_calls == oracle_calls
        @test state.ξ.θ ≈ [-1.0]
    end

    @testset "subsampling ThinningStrategy event law and lifecycle" begin
        weights = reshape([0.5, 1.5], 1, :)
        envelope = SeparableResidualEnvelope(weights,
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        oracle = (out, x, subset, anchor) ->
            (out[1] = only(subset) == 1 ? 0.5 : 1.5)
        cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0), oracle,
            envelope, [0.0], 1;
            deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
        flow = BouncyParticle(1, 0.0)
        rng = Random.Xoshiro(0x7a11)
        state, model, alg, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, PDMPModel(1, cv),
            ThinningStrategy(GlobalBounds(0.0, 1)), 0.0,
            SkeletonPoint([0.0], [1.0]);
            statistic_counter=PDMPSamplers.DevelStatisticCounter)

        waits = [first(PDMPSamplers.next_event_time(rng, model, flow, alg,
            state, cache, stats)) for _ in 1:8_000]
        @test mean(waits) ≈ 0.5 atol=0.015
        @test mean(2 .* waits .<= log(2)) ≈ 0.5 atol=0.015

        x_before, θ_before, t_before = copy(state.ξ.x), copy(state.ξ.θ), state.t[]
        τ, event, meta = PDMPSamplers.next_event_time(rng, model, flow, alg,
            state, cache, stats)
        @test event === :reflect
        @test state.ξ.x == x_before
        @test state.ξ.θ == θ_before
        @test state.t[] == t_before
        oracle_calls = stats.residual_oracle_calls
        PDMPSamplers.handle_event!(rng, τ, model, flow, alg, state, cache,
            event, meta, stats)
        @test stats.residual_oracle_calls == oracle_calls
        @test state.ξ.θ ≈ [-1.0]

        subsets_seen = Int[]
        rejection_envelope = SeparableResidualEnvelope(
            reshape([5.0, 15.0], 1, :),
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        rejection_oracle = function (out, x, subset, anchor)
            push!(subsets_seen, only(subset))
            out[1] = only(subset) == 1 ? 0.5 : 1.5
        end
        rejection_cv = SubsampledControlVariate(
            (out, x) -> (out[1] = 0.0), rejection_oracle,
            rejection_envelope, [0.0], 1)
        rejection_rng = Random.Xoshiro(0x7a12)
        rejection_state, rejection_model, rejection_alg, rejection_cache,
            rejection_stats = PDMPSamplers.initialize_state(
                rejection_rng, flow, PDMPModel(1, rejection_cv),
                ThinningStrategy(GlobalBounds(0.0, 1)), 0.0,
                SkeletonPoint([0.0], [1.0]))
        _, rejection_event, _ = PDMPSamplers.next_event_time(rejection_rng,
            rejection_model, flow, rejection_alg, rejection_state,
            rejection_cache, rejection_stats)
        @test rejection_event === :reflect
        @test length(subsets_seen) > 1
        @test rejection_stats.residual_oracle_calls == length(subsets_seen)

        horizon_envelope = SeparableResidualEnvelope(ones(1, 1),
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        horizon_cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            horizon_envelope, [0.0], 1)
        horizon_trace, _ = pdmp_sample(SkeletonPoint([0.0], [1.0]), flow,
            PDMPModel(1, horizon_cv), ThinningStrategy(GlobalBounds(0.0, 1)),
            0.0, 0.01; seed=0x7a13, progress=false)
        @test last(horizon_trace).time == 0.01
    end

    @testset "subsampling ThinningStrategy certified roofs" begin
        linear_flow = BouncyParticle(1, 0.0)
        growing = TrajectoryResidualEnvelope(ones(1, 2), [0.0];
            growth_rates=[0.2])
        growing_cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            growing, [0.0], 1)
        @test_throws ArgumentError PDMPSamplers.initialize_state(
            Random.Xoshiro(41), linear_flow, PDMPModel(1, growing_cv),
            ThinningStrategy(GlobalBounds(1.0, 1)), 0.0,
            SkeletonPoint([0.0], [1.0]))

        # Periodicity makes the same positive-growth envelope globally
        # representable by a constant Boomerang roof.
        boom_flow = Boomerang(reshape([1.0], 1, 1), [0.0], 0.2)
        boom_cv = SubsampledControlVariate(
            (out, x) -> (out[1] = 5.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            TrajectoryResidualEnvelope(ones(1, 2), [0.0]; growth_rates=[0.2]),
            [0.0], 1)
        boom_state = PDMPSamplers.initialize_state(Random.Xoshiro(42),
            boom_flow, PDMPModel(1, boom_cv),
            ThinningStrategy(GlobalBounds(10.0, 1)), 0.0,
            SkeletonPoint([0.4], [1.0]))
        @test boom_state[3] isa PDMPSamplers.SubsamplingThinningState

        generic_periodic = SeparableResidualEnvelope(ones(1, 2),
            (out, state, flow, t) -> (out[1] = 1 + sin(t));
            component_cell_scales! =
                (out, state, flow, left, right) -> (out[1] = 2.0))
        generic_cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            generic_periodic, [0.0], 1)
        periodic_state = PDMPSamplers.initialize_state(
            Random.Xoshiro(43), boom_flow, PDMPModel(1, generic_cv),
            ThinningStrategy(GlobalBounds(1.0, 1)), 0.0,
            SkeletonPoint([0.4], [1.0]))
        @test periodic_state[3] isa PDMPSamplers.SubsamplingThinningState
        grid_state = PDMPSamplers.initialize_state(
            Random.Xoshiro(44), boom_flow, PDMPModel(1, generic_cv),
            GridThinningStrategy(N=4, t_max=π, lazy=false), 0.0,
            SkeletonPoint([0.4], [1.0]))
        @test grid_state[3] isa PDMPSamplers.GridAdaptiveState
    end

    @testset "dense-preconditioned ZigZag deterministic roof" begin
        ρ = 0.9
        flow = DensePreconditionedZigZag(2)
        set_dense_preconditioner!(flow.metric,
            [1.0 0.0; ρ sqrt(1 - ρ^2)])
        @test flow.metric.L_operator_norm ≈ opnorm(flow.metric.L)
        canonical_velocity = [1.0, -1.0]
        physical_velocity = flow.metric.L * canonical_velocity
        ξ = SkeletonPoint([0.0, -1.0], physical_velocity)
        state = PDMPState(0.0, ξ)
        PDMPSamplers.initialize_flow_state!(state, flow)
        cache = PDMPSamplers.initialize_cache(Random.Xoshiro(44), flow,
            FullGradient((out, x) -> copyto!(out, x)),
            ThinningStrategy(GlobalBounds(0.0, 2)), 0.0, ξ)

        a, b, _ = PDMPSamplers.ab(ξ, zeros(2), flow, cache)
        actual0 = PDMPSamplers.λ(ξ, copy(ξ.x), flow)
        @test a ≈ sqrt(1 - ρ^2)
        @test actual0 ≈ sqrt(1 - ρ^2)
        @test actual0 <= a
        for t in range(0.0, 3.0; length=31)
            xt = ξ.x .+ t .* ξ.θ
            actual = PDMPSamplers.λ(SkeletonPoint(xt, ξ.θ), xt, flow)
            @test actual <= a + b * t + 1e-12
        end

        # A large post-construction factor must update the cached dual norm;
        # this is the configuration that previously produced rate 100 under
        # a stale identity-factor roof.
        set_dense_preconditioner!(flow.metric, [100.0 0.0; 0.0 1.0])
        high_velocity = flow.metric.L * canonical_velocity
        high_state = PDMPState(0.0, SkeletonPoint(zeros(2), high_velocity))
        PDMPSamplers.initialize_flow_state!(high_state, flow)
        high_actual = PDMPSamplers.λ(high_state.ξ, [1.0, 0.0], flow)
        high_bound = PDMPSamplers.residual_rate_dual_bound(flow, high_state, 0.0)
        @test high_actual == 100.0
        @test high_bound >= high_actual
    end

    @testset "bound violations discard invalid proposals" begin
        flow = BouncyParticle(1, 0.0)
        zero_hvp = (out, x, v) -> (out[1] = 0.0)
        zero_oracle = (out, x, subset, anchor) -> fill!(out, 0.0)

        # The first deterministic grid is stale by construction. Shrinking
        # must restart before drawing/evaluating a residual mark.
        switching_gradient = _SwitchingGradient(0, 2, 0.0, 10.0)
        envelope = SeparableResidualEnvelope(ones(1, 1),
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        cv = SubsampledControlVariate(switching_gradient, zero_oracle,
            envelope, [0.0], 1; deterministic_hvp! = zero_hvp)
        rng = Random.Xoshiro(191)
        state, model, alg, cache, stats = PDMPSamplers.initialize_state(rng,
            flow, PDMPModel(1, cv), GridThinningStrategy(N=1, t_max=20.0,
                lazy=false, bound_violation=:shrink), 0.0,
            SkeletonPoint([0.0], [1.0]);
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event, _ = PDMPSamplers.next_event_time(rng, model, flow, alg,
            state, cache, stats)
        @test event === :reflect
        @test τ > 0
        @test stats.grid_shrinks == 1
        @test stats.residual_oracle_calls == stats.grid_acceptance_tests

        # An aggregate violation likewise restarts before invoking the oracle.
        switching_scale = _SwitchingScale(0, 2, 0.0, 10.0)
        envelope2 = SeparableResidualEnvelope(ones(1, 1), switching_scale;
            certified_affine=true)
        cv2 = SubsampledControlVariate((out, x) -> (out[1] = 1.0), zero_oracle,
            envelope2, [0.0], 1; deterministic_hvp! = zero_hvp)
        rng2 = Random.Xoshiro(192)
        state2, model2, alg2, cache2, stats2 = PDMPSamplers.initialize_state(rng2,
            flow, PDMPModel(1, cv2), GridThinningStrategy(N=1, t_max=20.0,
                lazy=false, bound_violation=:shrink), 0.0,
            SkeletonPoint([0.0], [1.0]);
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        _, event2, _ = PDMPSamplers.next_event_time(rng2, model2, flow,
            alg2, state2, cache2, stats2)
        @test event2 === :reflect
        @test stats2.grid_shrinks == 1
        @test stats2.residual_oracle_calls == stats2.grid_acceptance_tests
        @test stats2.residual_oracle_calls >= 1

        # A residual envelope cannot be repaired by changing the D grid.
        bad_envelope = SeparableResidualEnvelope(fill(0.1, 1, 1),
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        bad_oracle = (out, x, subset, anchor) -> (out[1] = 10.0)
        bad_cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0),
            bad_oracle, bad_envelope, [0.0], 1; deterministic_hvp! = zero_hvp)
        bad_rng = Random.Xoshiro(193)
        bad_state, bad_model, bad_alg, bad_cache, bad_stats =
            PDMPSamplers.initialize_state(bad_rng, flow, PDMPModel(1, bad_cv),
                GridThinningStrategy(N=1, t_max=20.0, lazy=false,
                    bound_violation=:shrink), 0.0, SkeletonPoint([0.0], [1.0]);
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test_throws ErrorException PDMPSamplers.next_event_time(bad_rng,
            bad_model, flow, bad_alg, bad_state, bad_cache, bad_stats)
        @test bad_stats.grid_shrinks == 0

        # A loose but valid subsampling envelope is not a safety-limit failure.
        loose_envelope = SeparableResidualEnvelope(fill(1000.0, 1, 1),
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        loose_cv = SubsampledControlVariate((out, x) -> (out[1] = 1.0),
            zero_oracle, loose_envelope, [0.0], 1;
            deterministic_hvp! = zero_hvp)
        loose_rng = Random.Xoshiro(194)
        loose_state, loose_model, loose_alg, loose_cache, loose_stats =
            PDMPSamplers.initialize_state(loose_rng, flow,
                PDMPModel(1, loose_cv), GridThinningStrategy(N=1, t_max=20.0,
                    lazy=false, safety_limit=1, bound_violation=:throw), 0.0,
                SkeletonPoint([0.0], [1.0]);
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
        _, loose_event, _ = PDMPSamplers.next_event_time(loose_rng,
            loose_model, flow, loose_alg, loose_state, loose_cache, loose_stats)
        @test loose_event === :reflect
        @test loose_stats.grid_acceptance_tests > 100
    end

    @testset "budget-first subsampling grid and deterministic adaptation" begin
        calls_by_N = Int[]
        for N_grid in (20, 100, 500)
            envelope = SeparableResidualEnvelope(zeros(1, 1),
                (out, state, flow, t) -> (out[1] = 0.0);
                certified_affine=true)
            cv = SubsampledControlVariate((out, x) -> (out[1] = 1000.0),
                (out, x, subset, anchor) -> fill!(out, 0.0),
                envelope, [0.0], 1;
                deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
            rng = Random.Xoshiro(991)
            state, model, alg, cache, stats = PDMPSamplers.initialize_state(
                rng, BouncyParticle(1, 0.0), PDMPModel(1, cv),
                GridThinningStrategy(N=N_grid, N_min=min(5, N_grid),
                    t_max=1.0, lazy=false, bound_violation=:throw), 0.0,
                SkeletonPoint([0.0], [1.0]))
            alg.schedule_frozen[] = true
            subsampling_bound = alg.subsampling_bound
            _, event, _ = PDMPSamplers.next_event_time(rng, model,
                BouncyParticle(1, 0.0), alg, state, cache, stats)
            @test event === :reflect
            @test alg.subsampling_bound === subsampling_bound
            push!(calls_by_N, stats.deterministic_gradient_calls)
            @test stats.residual_oracle_calls == 1
        end
        @test maximum(calls_by_N) <= 4

        # A loose residual envelope must not request a finer deterministic grid.
        loose_envelope = SeparableResidualEnvelope(fill(100.0, 1, 1),
            (out, state, flow, t) -> (out[1] = 1.0);
            certified_affine=true)
        loose_cv = SubsampledControlVariate((out, x) -> (out[1] = 1.0),
            (out, x, subset, anchor) -> fill!(out, 0.0), loose_envelope,
            [0.0], 1; deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
        loose_rng = Random.Xoshiro(994)
        loose_state, loose_model, loose_alg, loose_cache, loose_stats =
            PDMPSamplers.initialize_state(loose_rng, BouncyParticle(1, 0.0),
                PDMPModel(1, loose_cv), GridThinningStrategy(N=20, N_min=5,
                    t_max=20.0, lazy=false, bound_violation=:throw), 0.0,
                SkeletonPoint([0.0], [1.0]))
        initial_N = loose_alg.N[]
        _, loose_event, _ = PDMPSamplers.next_event_time(loose_rng,
            loose_model, BouncyParticle(1, 0.0), loose_alg, loose_state,
            loose_cache, loose_stats)
        @test loose_event === :reflect
        @test loose_alg.N[] < initial_N
    end

    @testset "cell-roof slack is aggregate-thinned before subsampling evaluation" begin
        point_scale! = (out, state, flow, t) -> (out[1] = t)
        cell_scale! = (out, state, flow, left, right) -> (out[1] = right)
        envelope = SeparableResidualEnvelope(ones(1, 1), point_scale!;
            component_cell_scales! = cell_scale!)
        oracle = (out, x, subset, anchor) -> (out[1] = x[1])
        cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0), oracle,
            envelope, [0.0], 1;
            deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
        rng = Random.Xoshiro(0x5ce11)
        flow = BouncyParticle(1, 0.0)
        state, model, alg, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, PDMPModel(1, cv),
            GridThinningStrategy(N=1, N_min=1, t_max=1.0, lazy=false,
                bound_violation=:throw), 0.0, SkeletonPoint([0.0], [1.0]);
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        alg.schedule_frozen[] = true

        n_trials = 25_000
        events = 0
        for _ in 1:n_trials
            _, event, _ = PDMPSamplers.next_event_time(rng, model, flow, alg,
                state, cache, stats, 1.0, false, :horizon_hit)
            events += event === :reflect
        end
        observed = events / n_trials
        expected = 1 - exp(-0.5)
        @test observed ≈ expected atol=0.01
        @test abs(observed - (1 - exp(-1))) > 0.15
        @test stats.subsampling_cell_roof_proposals > stats.subsampling_aggregate_accepts
        @test stats.subsampling_aggregate_accepts == stats.subsampling_subset_evaluations
        @test stats.subsampling_subset_evaluations == stats.subsampling_final_reflections
        @test stats.residual_oracle_calls == stats.subsampling_subset_evaluations
    end

    @testset "aggregate rejection precedes candidate propagation" begin
        envelope = SeparableResidualEnvelope(ones(1, 1),
            (out, state, flow, t) -> (out[1] = 0.0);
            component_cell_scales! =
                (out, state, flow, left, right) -> (out[1] = 100.0))
        cv = SubsampledControlVariate((out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0), envelope, [0.0], 1;
            deterministic_hvp! = ((out, x, v) -> fill!(out, 0.0)))
        rng = Random.Xoshiro(0xa99e9a7e)
        flow = BouncyParticle(1, 0.0)
        state, model, alg, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, PDMPModel(1, cv),
            GridThinningStrategy(N=1, N_min=1, t_max=1.0, lazy=false,
                bound_violation=:throw), 0.0, SkeletonPoint([0.0], [1.0]);
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        alg.schedule_frozen[] = true
        alg.state_cache2.ξ.x[1] = 123.0
        _, event, _ = PDMPSamplers.next_event_time(rng, model, flow, alg,
            state, cache, stats, 1.0, false, :horizon_hit)
        @test event === :horizon_hit
        @test stats.subsampling_cell_roof_proposals > 0
        @test stats.subsampling_aggregate_accepts == 0
        @test alg.state_cache2.ξ.x[1] == 123.0
    end

    @testset "subsampling anchor refresh is coherent" begin
        provider_anchor = Ref([0.0])
        envelope = TrajectoryResidualEnvelope(ones(1, 2), provider_anchor[])
        refresh! = function(anchor)
            refreshed = TrajectoryResidualEnvelope(
                fill(1 + abs(anchor[1]), 1, 2), anchor)
            provider_anchor[] = anchor
            return refreshed
        end
        cv = SubsampledControlVariate(
            (out, x) -> (out[1] = x[1] - provider_anchor[][1]),
            (out, x, subset, anchor) -> (out[1] = x[1] - anchor[1]),
            envelope, provider_anchor[], 1;
            deterministic_hvp! = ((out, x, v) -> (out[1] = v[1])),
            refresh_anchor! = refresh!)

        PDMPSamplers.refresh_anchor!(cv, [2.0])
        @test cv.anchor == [2.0]
        @test cv.envelope.component_scales!.anchor == [2.0]
        @test cv.envelope.weights == fill(3.0, 1, 2)
        out = zeros(1)
        PDMPSamplers.deterministic_gradient!(out, cv, [5.0])
        @test out == [3.0]
        @test_throws ArgumentError copy(cv)

        no_refresh = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            TrajectoryResidualEnvelope(ones(1, 1), [0.0]), [0.0], 1)
        @test_throws ArgumentError PDMPSamplers.refresh_anchor!(no_refresh, [1.0])
    end

    @testset "subsampling anchor lifecycle validation" begin
        envelope = TrajectoryResidualEnvelope(ones(1, 1), [0.0])
        @test_throws ArgumentError SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            envelope, [1.0], 1)
        cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            envelope, [0.0], 1;
            refresh_anchor! = anchor -> TrajectoryResidualEnvelope(
                ones(1, 1), zeros(length(anchor))))
        @test_throws ArgumentError PDMPSamplers.refresh_anchor!(cv, [1.0])
        @test cv.anchor == [0.0]
        @test cv.envelope === envelope

        point_scale_a = (out, state, flow, t) -> fill!(out, 1.0)
        point_scale_b = (out, state, flow, t) -> fill!(out, 2.0)
        typed_envelope = SeparableResidualEnvelope(ones(1, 1), point_scale_a;
            certified_affine=true)
        typed_cv = SubsampledControlVariate(
            (out, x) -> fill!(out, 0.0),
            (out, x, subset, anchor) -> fill!(out, 0.0),
            typed_envelope, [0.0], 1;
            refresh_anchor! = anchor ->
                SeparableResidualEnvelope(ones(1, 1), point_scale_b;
                    certified_affine=true))
        @test_throws ArgumentError PDMPSamplers.refresh_anchor!(typed_cv, [1.0])
        @test typed_cv.anchor == [0.0]
        @test typed_cv.envelope === typed_envelope
    end

    function gaussian_subsampling_model(flow, N=24, m=3)
        q = collect(range(0.5, 1.5; length=N))
        q ./= sum(q)
        anchor = [0.0]
        envelope = TrajectoryResidualEnvelope(reshape(q, 1, :), anchor)
        oracle = (out, x, subset, a) ->
            (out[1] = sum(q[i] for i in subset) * (x[1] - a[1]))
        cv = SubsampledControlVariate((out, x) -> (out[1] = 0.0), oracle,
            envelope, anchor, m;
            deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
        return PDMPModel(1, cv)
    end

    @testset "Gaussian stationarity for BPS and ZigZag" begin
        for (flow, seed) in ((BouncyParticle(1, 0.5), 9101), (ZigZag(1), 9102))
            model = gaussian_subsampling_model(flow)
            trace, stats = pdmp_sample(SkeletonPoint([0.7], [1.0]), flow,
                model, GridThinningStrategy(N=12, t_max=2.0, lazy=false,
                    bound_violation=:throw), 0.0, 12_000.0;
                seed, progress=false)
            moments = mean(trace)
            @test moments[1] ≈ 0 atol=0.09
            @test cov(trace)[1, 1] ≈ 1 atol=0.12
            @test stats.residual_oracle_calls > 100
            @test stats.full_gradient_calls == 0
        end
        for flow in (BouncyParticle(1, 0.5), ZigZag(1))
            trace, _ = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow,
                gaussian_subsampling_model(flow),
                GridThinningStrategy(N=8, t_max=1.5, lazy=false,
                    bound=:linear, bound_violation=:throw), 0.0, 50.0;
                seed=774, progress=false)
            @test length(trace) > 2
        end
    end

    @testset "subsampling ThinningStrategy stationarity and dynamics" begin
        for (flow, seed) in ((BouncyParticle(1, 0.5), 9201), (ZigZag(1), 9202))
            trace, stats = pdmp_sample(SkeletonPoint([0.7], [1.0]), flow,
                gaussian_subsampling_model(flow),
                ThinningStrategy(GlobalBounds(0.0, 1)), 0.0, 8_000.0;
                seed, progress=false)
            @test mean(trace)[1] ≈ 0 atol=0.11
            @test cov(trace)[1, 1] ≈ 1 atol=0.14
            @test stats.residual_oracle_calls > 100
        end

        flows = Any[
            PreconditionedBPS(1; refresh_rate=0.5, scale=[0.7]),
            PreconditionedZigZag(1; scale=[0.7]),
            DensePreconditionedBPS(1; refresh_rate=0.5),
            DensePreconditionedZigZag(1),
            Boomerang(reshape([1.0], 1, 1), [0.0], 0.2),
            AdaptiveBoomerang(1; λref=0.2),
        ]
        for (seed, flow) in enumerate(flows)
            trace, stats = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow,
                gaussian_subsampling_model(flow),
                ThinningStrategy(GlobalBounds(100.0, 1)), 0.0, 12.0, 2.0;
                seed=9300 + seed, progress=false)
            @test length(trace) > 1
            @test stats.residual_oracle_calls > 0
        end

        sticky_trace, sticky_stats = pdmp_sample(
            SkeletonPoint([0.4], [1.0]), BouncyParticle(1, 0.5),
            gaussian_subsampling_model(BouncyParticle(1, 0.5)),
            Sticky(ThinningStrategy(GlobalBounds(100.0, 1)), [1.0]),
            0.0, 8.0; seed=9399, progress=false)
        @test length(sticky_trace) > 1
        @test sticky_stats.residual_oracle_calls > 0
    end

    @testset "subsampling sticky lifecycle across supported flows and strategies" begin
        flow_factories = (
            () -> BouncyParticle(2, 0.0),
            () -> ZigZag(2),
            () -> PreconditionedBPS(2; refresh_rate=0.0, scale=ones(2)),
            () -> PreconditionedZigZag(2; scale=ones(2)),
            () -> DensePreconditionedBPS(2; refresh_rate=0.0),
            () -> DensePreconditionedZigZag(2),
            () -> Boomerang(Matrix{Float64}(I, 2, 2), zeros(2), 0.0),
            () -> AdaptiveBoomerang(2; λref=0.0),
        )
        strategy_factories = (
            () -> GridThinningStrategy(N=4, t_max=1.0, lazy=false,
                bound_violation=:throw),
            () -> ThinningStrategy(GlobalBounds(0.0, 2)),
        )

        for make_flow in flow_factories, make_strategy in strategy_factories
            flow = make_flow()
            residual_calls_while_frozen = Ref(0)
            residual_oracle = function(out, x, subset, anchor)
                iszero(x[1]) && (residual_calls_while_frozen[] += 1)
                out[1] = 0.0
                out[2] = 0.1
                return out
            end
            point_scales = (out, state, flow, t) -> (out[1] = 10.0)
            cell_scales = (out, state, flow, left, right) -> (out[1] = 10.0)
            envelope = SeparableResidualEnvelope(ones(1, 1), point_scales;
                component_cell_scales! = cell_scales,
                certified_affine=true)
            deterministic_gradient = (out, x) -> copyto!(out, x)
            deterministic_hvp = (out, x, v) -> copyto!(out, v)
            cv = SubsampledControlVariate(deterministic_gradient, residual_oracle,
                envelope, zeros(2), 1; deterministic_hvp! = deterministic_hvp)
            trace, stats = pdmp_sample(
                SkeletonPoint([0.1, 1.0], [-1.0, 1.0]), flow,
                PDMPModel(2, cv), Sticky(make_strategy(), [1.0, Inf]),
                0.0, 30.0; seed=4, progress=false,
                statistic_counter=PDMPSamplers.DevelStatisticCounter)

            @test stats.sticky_events >= 4
            @test residual_calls_while_frozen[] > 0
            if hasproperty(trace, :free_masks)
                frozen = findall(.!trace.free_masks[1, :])
                @test !isempty(frozen)
                @test all(iszero, trace.positions[1, frozen])
                @test all(iszero, trace.velocities[1, frozen])
            else
                frozen_events = filter(trace.events) do event
                    event.index == 1 && iszero(event.position) &&
                        iszero(event.velocity)
                end
                @test !isempty(frozen_events)
            end
        end

    end

    @testset "dense ZigZag rejected unfreeze is state-neutral in production" begin
        make_flow() = begin
            flow = DensePreconditionedZigZag(2)
            set_dense_preconditioner!(flow.metric,
                cholesky(Symmetric([1.0 0.92; 0.92 1.0])).L)
            flow
        end
        make_model() = begin
            envelope = TrajectoryResidualEnvelope(ones(1, 8), zeros(2))
            cv = SubsampledControlVariate(
                (out, x) -> copyto!(out, x),
                (out, x, subset, anchor) -> fill!(out, 0.0),
                envelope, zeros(2), 2;
                deterministic_hvp! = (out, x, v) -> copyto!(out, v))
            PDMPModel(2, cv)
        end

        same_physical_state(a, b) = a.t[] == b.t[] && a.ξ.x == b.ξ.x &&
            a.ξ.θ == b.ξ.θ && a.free == b.free
        function same_stratum_cache(a, b)
            sa, sb = a.boundary_scratch, b.boundary_scratch
            k = sa.active_count
            return k == sb.active_count &&
                sa.active_factor_valid == sb.active_factor_valid &&
                sa.factor_source === sb.factor_source &&
                sa.factor_generation == sb.factor_generation &&
                sa.active[1:k] == sb.active[1:k] &&
                all(a.free[sa.active[j]] ?
                    sa.canonical_signs[j] == sb.canonical_signs[j] : true
                    for j in 1:k) &&
                LowerTriangular(view(sa.ΣAA, 1:k, 1:k)) ==
                    LowerTriangular(view(sb.ΣAA, 1:k, 1:k))
        end
        candidate_state(inner::PDMPSamplers.GridAdaptiveState) = inner.state_cache
        candidate_state(inner::PDMPSamplers.SubsamplingThinningState) = inner.candidate

        for (seed, strategy_a, strategy_b) in (
                (0xd301,
                 GridThinningStrategy(N=12, t_max=1.0, lazy=false,
                    bound_violation=:throw),
                 GridThinningStrategy(N=12, t_max=1.0, lazy=false,
                    bound_violation=:throw)),
                (0xd302, ThinningStrategy(GlobalBounds(0.0, 2)),
                           ThinningStrategy(GlobalBounds(0.0, 2))))
            flow = make_flow()
            initial = SkeletonPoint([0.4, 0.0], [1.0, 0.0])
            init_a = Random.Xoshiro(seed)
            init_b = Random.Xoshiro(seed)
            rejected, model_a, alg_a, cache_a, stats_a =
                PDMPSamplers.initialize_state(init_a, flow, make_model(),
                    Sticky(strategy_a, [3.0, 4.0]), 1.25, initial;
                    statistic_counter=PDMPSamplers.DevelStatisticCounter)
            control, model_b, alg_b, cache_b, stats_b =
                PDMPSamplers.initialize_state(init_b, flow, make_model(),
                    Sticky(strategy_b, [3.0, 4.0]), 1.25, initial;
                    statistic_counter=PDMPSamplers.DevelStatisticCounter)
            @test same_physical_state(rejected, control)
            @test same_stratum_cache(rejected, control)
            @test !rejected.free[2]

            # Give the other stickable coordinate a finite deterministic
            # hitting time so preservation is tested on a nontrivial clock.
            rejected.ξ.x[1] = -sign(rejected.ξ.θ[1]) * 0.4
            control.ξ.x[1] = rejected.ξ.x[1]
            PDMPSamplers._update_sticky_time_at_index!(
                init_a, alg_a, rejected, flow, 1)
            PDMPSamplers._update_sticky_time_at_index!(
                init_b, alg_b, control, flow, 1)
            @test isfinite(alg_a.sticky_times[1])
            sticky_times_before = copy(alg_a.sticky_times)

            rejection_seed = findfirst(1:10_000) do proposal_seed
                probe = copy(rejected)
                !PDMPSamplers.propose_boundary_velocity!(
                    Random.Xoshiro(proposal_seed), probe, flow, 2)
            end
            @test rejection_seed !== nothing
            transition_rng = Random.Xoshiro(something(rejection_seed))
            control_rng = Random.Xoshiro(something(rejection_seed))
            physical_snapshot = (rejected.t[], copy(rejected.ξ.x),
                copy(rejected.ξ.θ), copy(rejected.free))
            @test !PDMPSamplers.stick_or_unstick!(
                transition_rng, rejected, flow, alg_a, 2)

            # Consume the identical proposal draw on a disposable state, then
            # resample only the rejected coordinate on the untouched control.
            disposable = copy(control)
            @test !PDMPSamplers.propose_boundary_velocity!(
                control_rng, disposable, flow, 2)
            PDMPSamplers._update_sticky_time_at_index!(
                control_rng, alg_b, control, flow, 2)

            @test (rejected.t[], rejected.ξ.x, rejected.ξ.θ,
                   rejected.free) == physical_snapshot
            @test same_physical_state(rejected, control)
            @test same_stratum_cache(rejected, control)
            @test isequal(alg_a.sticky_times[1], sticky_times_before[1])
            @test !isequal(alg_a.sticky_times[2], sticky_times_before[2])
            @test alg_a.sticky_times == alg_b.sticky_times
            @test collect(alg_a.sticky_pq) == collect(alg_b.sticky_pq)
            @test rand(copy(transition_rng)) == rand(copy(control_rng))

            result_a = PDMPSamplers._bounded_inner_event_time(
                transition_rng, model_a, flow, alg_a.inner_alg_state,
                rejected, cache_a, stats_a, 0.35)
            result_b = PDMPSamplers._bounded_inner_event_time(
                control_rng, model_b, flow, alg_b.inner_alg_state,
                control, cache_b, stats_b, 0.35)
            @test result_a[1] == result_b[1]
            @test result_a[2] == result_b[2]
            @test typeof(result_a[3]) == typeof(result_b[3])
            @test result_a[3].∇ϕx == result_b[3].∇ϕx
            @test same_physical_state(rejected, control)
            @test cache_a.∇ϕx == cache_b.∇ϕx
            @test same_physical_state(candidate_state(alg_a.inner_alg_state),
                                      candidate_state(alg_b.inner_alg_state))
            @test same_stratum_cache(candidate_state(alg_a.inner_alg_state),
                                     candidate_state(alg_b.inner_alg_state))
        end
    end

    @testset "subsampling production loop covers full dynamics and sticky wrappers" begin
        flows = Any[
            PreconditionedBPS(1; refresh_rate=0.5, scale=[0.7]),
            PreconditionedZigZag(1; scale=[0.7]),
            DensePreconditionedBPS(1; refresh_rate=0.5),
            DensePreconditionedZigZag(1),
            Boomerang(reshape([1.0], 1, 1), [0.0], 0.2),
            AdaptiveBoomerang(1; λref=0.2),
        ]
        for (seed, flow) in enumerate(flows)
            trace, stats = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow,
                gaussian_subsampling_model(flow),
                GridThinningStrategy(N=10, t_max=1.0, lazy=false,
                    bound_violation=:throw), 0.0, 20.0, 2.0;
                seed=8300 + seed, progress=false,
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
            @test length(trace) > 2
            @test stats.residual_oracle_calls > 0
            @test stats.warmup_subsampling_cell_roof_proposals +
                stats.main_subsampling_cell_roof_proposals ==
                stats.subsampling_cell_roof_proposals
            @test stats.warmup_subsampling_aggregate_accepts +
                stats.main_subsampling_aggregate_accepts ==
                stats.subsampling_aggregate_accepts
            @test stats.warmup_subsampling_subset_evaluations +
                stats.main_subsampling_subset_evaluations ==
                stats.subsampling_subset_evaluations
            @test stats.warmup_subsampling_final_reflections +
                stats.main_subsampling_final_reflections ==
                stats.subsampling_final_reflections
        end

        for (seed, flow) in enumerate((
                BouncyParticle(1, 0.5), ZigZag(1),
                PreconditionedBPS(1; refresh_rate=0.5, scale=[0.8]),
                PreconditionedZigZag(1; scale=[0.8]),
                Boomerang(reshape([1.0], 1, 1), [0.0], 0.2),
                AdaptiveBoomerang(1; λref=0.2)))
            alg = Sticky(GridThinningStrategy(N=8, t_max=0.8, lazy=false,
                bound_violation=:throw), [1.0])
            trace, stats = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow,
                gaussian_subsampling_model(flow), alg, 0.0, 12.0;
                seed=8400 + seed, progress=false)
            @test length(trace) > 1
            @test stats.residual_oracle_calls > 0
        end
    end

    @testset "candidate observation work is minibatch-sized" begin
        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        for N in (100, 10_000)
            work = Ref(0)
            envelope = SeparableResidualEnvelope(ones(1, N),
                (out, state, flow, t) -> (out[1] = 1.0);
                certified_affine=true)
            oracle = function (out, x, subset, anchor)
                work[] += length(subset)
                fill!(out, 0.0)
            end
            cv = SubsampledControlVariate((out, x) -> fill!(out, 0.0), oracle,
                envelope, [0.0], 7)
            B = PDMPSamplers.total_residual_bound(envelope, state, 0.0)
            for _ in 1:100
                PDMPSamplers.draw_subset!(Random.default_rng(), cv, 1.0, B)
                cv.residual_oracle(cv.residual_buffer, state.ξ.x,
                    cv.subset, cv.anchor)
            end
            @test work[] == 700
        end
    end

end
