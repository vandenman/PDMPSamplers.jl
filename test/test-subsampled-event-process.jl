@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _marked_fixture(weights, m; scales=ones(size(weights, 1)))
    envelope = SeparableResidualEnvelope(weights,
        (out, state, t) -> copyto!(out, scales))
    oracle = (out, x, subset, anchor) -> fill!(out, 0.0)
    return MarkedControlVariate((out, x) -> fill!(out, 0.0), oracle,
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
function (f::_SwitchingScale)(out, state, t)
    f.calls += 1
    out[1] = f.calls <= f.switch_after ? f.low : f.high
end

@testset "Marked subsampled event process" begin
    @testset "legacy held-batch contract is rejected" begin
        legacy = SubsampledGradient((out, x) -> copyto!(out, x), n -> nothing, 1)
        @test_throws ArgumentError PDMPSamplers.initialize_state(
            Random.Xoshiro(11), BouncyParticle(1), PDMPModel(1, legacy),
            GridThinningStrategy(), 0.0, SkeletonPoint([0.0], [1.0]))
    end


    @testset "chain-local deterministic HVP" begin
        provider = _CopyTrackedHVP(0)
        envelope = SeparableResidualEnvelope(zeros(1, 1),
            (out, state, t) -> (out[1] = 0.0))
        cv = MarkedControlVariate((out, x) -> copyto!(out, x),
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
            (out, state, t) -> (out[1] = 0.0))
        cv = MarkedControlVariate((out, x) -> copyto!(out, x),
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
            (out, state, t) -> (out[1] = 1.0))
        oracle = (out, x, subset, anchor) ->
            (out[1] = only(subset) == 1 ? 0.5 : 1.5)
        cv = MarkedControlVariate((out, x) -> (out[1] = 0.0), oracle,
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
            cv = _marked_fixture(weights, m; scales)
            B = PDMPSamplers.total_residual_bound(cv.envelope, state, 0.0)
            subsets = _enumerated_subsets(size(weights, 2), m)
            counts = Dict(S => 0 for S in subsets)
            draws = 20_000
            max_bound_error = 0.0
            for _ in 1:draws
                M = PDMPSamplers.draw_subset!(rng, cv, state, 0.0, D, B)
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

    @testset "candidate lifecycle and stored reflection gradient" begin
        subsets_seen = Int[]
        searches = Ref(0)
        weights = reshape([5.0, 15.0], 1, :)
        envelope = SeparableResidualEnvelope(weights, (out, state, t) -> (out[1] = 1.0))
        oracle = function (out, x, subset, anchor)
            push!(subsets_seen, only(subset))
            out[1] = only(subset) == 1 ? 0.5 : 1.5
        end
        cv = MarkedControlVariate((out, x) -> (out[1] = 0.0), oracle,
            envelope, [7.0], 1;
            begin_search! = (cv, state) -> (searches[] += 1),
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
        @test counted_model.grad.active_anchor == [7.0]
        @test searches[] == 1
        @test stats.residual_oracle_calls == length(subsets_seen)
        oracle_calls = stats.residual_oracle_calls
        full_calls = stats.full_reflection_gradient_calls
        PDMPSamplers.handle_event!(rng, τ, counted_model, flow, alg, state,
            cache, event, meta, stats)
        @test stats.residual_oracle_calls == oracle_calls
        @test stats.full_reflection_gradient_calls == full_calls == 0
        @test state.ξ.θ ≈ [-1.0]
    end

    @testset "bound violations discard invalid proposals" begin
        flow = BouncyParticle(1, 0.0)
        zero_hvp = (out, x, v) -> (out[1] = 0.0)
        zero_oracle = (out, x, subset, anchor) -> fill!(out, 0.0)

        # The first deterministic grid is stale by construction. Shrinking
        # must restart before drawing/evaluating a residual mark.
        switching_gradient = _SwitchingGradient(0, 2, 0.0, 10.0)
        shrink_searches = Ref(0)
        envelope = SeparableResidualEnvelope(ones(1, 1),
            (out, state, t) -> (out[1] = 1.0))
        cv = MarkedControlVariate(switching_gradient, zero_oracle,
            envelope, [0.0], 1; deterministic_hvp! = zero_hvp,
            begin_search! = (cv, state) -> (shrink_searches[] += 1))
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
        @test shrink_searches[] == 1

        # An aggregate violation likewise restarts before invoking the oracle.
        switching_scale = _SwitchingScale(0, 2, 0.0, 10.0)
        envelope2 = SeparableResidualEnvelope(ones(1, 1), switching_scale)
        cv2 = MarkedControlVariate((out, x) -> (out[1] = 1.0), zero_oracle,
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
            (out, state, t) -> (out[1] = 1.0))
        bad_oracle = (out, x, subset, anchor) -> (out[1] = 10.0)
        bad_cv = MarkedControlVariate((out, x) -> (out[1] = 0.0),
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
    end

    @testset "budget-first marked grid and deterministic adaptation" begin
        calls_by_N = Int[]
        for N_grid in (20, 100, 500)
            envelope = SeparableResidualEnvelope(zeros(1, 1),
                (out, state, t) -> (out[1] = 0.0))
            cv = MarkedControlVariate((out, x) -> (out[1] = 1000.0),
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
            marked_bound = alg.marked_bound
            _, event, _ = PDMPSamplers.next_event_time(rng, model,
                BouncyParticle(1, 0.0), alg, state, cache, stats)
            @test event === :reflect
            @test alg.marked_bound === marked_bound
            push!(calls_by_N, stats.deterministic_gradient_calls)
            @test stats.residual_oracle_calls == 1
        end
        @test maximum(calls_by_N) <= 4

        # A loose residual envelope must not request a finer deterministic grid.
        loose_envelope = SeparableResidualEnvelope(fill(100.0, 1, 1),
            (out, state, t) -> (out[1] = 1.0))
        loose_cv = MarkedControlVariate((out, x) -> (out[1] = 1.0),
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

    @testset "internal callback hooks are not exported" begin
        public_names = names(PDMPSamplers)
        @test :draw_subset! ∉ public_names
        @test :begin_search! ∉ public_names
        @test :residual_gradient! ∉ public_names
    end

    function gaussian_marked_model(flow, N=24, m=3)
        q = collect(range(0.5, 1.5; length=N))
        q ./= sum(q)
        anchor = [0.0]
        envelope = SeparableResidualEnvelope(reshape(q, 1, :),
            (out, state, t) -> begin
                v = only(state.ξ.θ)
                out[1] = abs(v) * (abs(only(state.ξ.x)) + abs(v) * t)
            end)
        oracle = (out, x, subset, a) ->
            (out[1] = sum(q[i] for i in subset) * (x[1] - a[1]))
        cv = MarkedControlVariate((out, x) -> (out[1] = 0.0), oracle,
            envelope, anchor, m;
            deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
        return PDMPModel(1, cv)
    end

    @testset "Gaussian stationarity for BPS and ZigZag" begin
        for (flow, seed) in ((BouncyParticle(1, 0.5), 9101), (ZigZag(1), 9102))
            model = gaussian_marked_model(flow)
            trace, stats = pdmp_sample(SkeletonPoint([0.7], [1.0]), flow,
                model, GridThinningStrategy(N=12, t_max=2.0, lazy=false,
                    bound_violation=:throw), 0.0, 12_000.0;
                seed, progress=false)
            moments = mean(trace)
            @test moments[1] ≈ 0 atol=0.09
            @test cov(trace)[1, 1] ≈ 1 atol=0.12
            @test stats.residual_oracle_calls > 100
            @test stats.full_gradient_calls == 0
            @test stats.full_reflection_gradient_calls == 0
        end
        for flow in (BouncyParticle(1, 0.5), ZigZag(1))
            trace, _ = pdmp_sample(SkeletonPoint([0.4], [1.0]), flow,
                gaussian_marked_model(flow),
                GridThinningStrategy(N=8, t_max=1.5, lazy=false,
                    bound=:linear, bound_violation=:throw), 0.0, 50.0;
                seed=774, progress=false)
            @test length(trace) > 2
        end
    end

    @testset "candidate observation work is minibatch-sized" begin
        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        for N in (100, 10_000)
            work = Ref(0)
            envelope = SeparableResidualEnvelope(ones(1, N),
                (out, state, t) -> (out[1] = 1.0))
            oracle = function (out, x, subset, anchor)
                work[] += length(subset)
                fill!(out, 0.0)
            end
            cv = MarkedControlVariate((out, x) -> fill!(out, 0.0), oracle,
                envelope, [0.0], 7)
            B = PDMPSamplers.total_residual_bound(envelope, state, 0.0)
            for _ in 1:100
                PDMPSamplers.draw_subset!(Random.default_rng(), cv, state, 0.0, 1.0, B)
                PDMPSamplers.residual_gradient!(cv.residual_buffer, cv.residual_oracle,
                    state.ξ.x, cv.subset, cv.active_anchor)
            end
            @test work[] == 700
        end
    end

    @testset "affine-logistic BPS integration" begin
        rng = Random.Xoshiro(0x1091571c)
        N, d, m = 40, 2, 5
        X = randn(rng, N, d)
        βtrue = [0.4, -0.7]
        y = Float64.(rand(rng, N) .< LogExpFunctions.logistic.(X * βtrue))
        anchor = zeros(d)
        prior_precision = 0.25
        logistic_grad = function (out, x, rows)
            fill!(out, 0.0)
            for i in rows
                η = dot(view(X, i, :), x)
                out .+= (LogExpFunctions.logistic(η) - y[i]) .* view(X, i, :)
            end
            out
        end
        anchor_likelihood = zeros(d)
        logistic_grad(anchor_likelihood, anchor, axes(X, 1))
        deterministic = (out, x) -> (out .= anchor_likelihood .+ prior_precision .* x)
        oracle = function (out, x, subset, a)
            logistic_grad(out, x, subset)
            anchor_part = zeros(d)
            logistic_grad(anchor_part, a, subset)
            out .-= anchor_part
        end
        weights = reshape([0.25 * sum(abs2, view(X, i, :)) for i in axes(X, 1)], 1, :)
        envelope = SeparableResidualEnvelope(weights,
            (out, state, t) -> begin
                vnorm = norm(state.ξ.θ)
                out[1] = vnorm * (norm(state.ξ.x - anchor) + vnorm * t)
            end)
        cv = MarkedControlVariate(deterministic, oracle, envelope, anchor, m;
            deterministic_hvp! = ((out, x, v) -> (out .= prior_precision .* v)))
        marked_model = PDMPModel(d, cv)
        full_gradient = function (out, x)
            logistic_grad(out, x, axes(X, 1))
            out .+= prior_precision .* x
        end
        full_hvp = function (out, x, v)
            out .= prior_precision .* v
            for i in axes(X, 1)
                xi = view(X, i, :)
                p = LogExpFunctions.logistic(dot(xi, x))
                out .+= (p * (1 - p) * dot(xi, v)) .* xi
            end
        end
        full_model = PDMPModel(d, FullGradient(full_gradient), full_hvp)
        flow = BouncyParticle(d, 0.7)
        ξ = SkeletonPoint([0.2, -0.2], [1.0, 0.0])
        alg = GridThinningStrategy(N=16, t_max=1.5, lazy=false,
            bound_violation=:throw)
        marked_trace, marked_stats = pdmp_sample(ξ, flow, marked_model, alg,
            0.0, 8_000.0; seed=808, progress=false)
        full_trace, _ = pdmp_sample(ξ, flow, full_model, alg,
            0.0, 8_000.0; seed=809, progress=false)
        @test mean(marked_trace) ≈ mean(full_trace) atol=0.18
        @test diag(cov(marked_trace)) ≈ diag(cov(full_trace)) atol=0.2
        @test marked_stats.residual_oracle_calls > 0
        @test marked_stats.full_gradient_calls == 0
        @test marked_stats.full_reflection_gradient_calls == 0
    end
end
