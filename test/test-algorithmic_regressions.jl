@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "Algorithmic regressions" begin

    @testset "StickyLoopState priority queue stays synchronized" begin
        d = 6
        target = gen_data(Distributions.ZeroMeanIsoNormal, d)
        flow = ZigZag(inv(Symmetric(cov(target.D))), mean(target.D))
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = Sticky(GridThinningStrategy(), fill(0.5, d))
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))

        rng = Random.Xoshiro(1234)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0)

        @test alg_ isa PDMPSamplers.StickyLoopState

        state.free .= false
        state.ξ.x .= 0.0
        state.ξ.θ .= 0.0
        state.old_velocity .= 1.0
        PDMPSamplers.update_all_stick_times!(rng, alg_, state, flow)

        pq_i, pq_t = first(alg_.sticky_pq)
        @test pq_t ≈ minimum(alg_.sticky_times)

        dt, event_type, meta = PDMPSamplers.next_event_time(rng, model_, flow, alg_, state, cache, stats)
        @test event_type == :sticky
        @test meta.i == pq_i
        @test dt ≈ pq_t - state.t[]

        PDMPSamplers.stick_or_unstick!(rng, state, flow, alg_, meta.i)
        _, pq_t2 = first(alg_.sticky_pq)
        @test pq_t2 ≈ minimum(alg_.sticky_times)
    end

    @testset "Sticky ZigZag fast-path schedule updates preserve unaffected coordinates" begin
        d = 5
        target = gen_data(Distributions.ZeroMeanIsoNormal, d)
        flow = ZigZag(inv(Symmetric(cov(target.D))), mean(target.D))
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        alg = Sticky(GridThinningStrategy(), fill(0.5, d))
        ξ0 = SkeletonPoint(fill(2.0, d), ones(d))

        rng = Random.Xoshiro(4321)
        state, _, alg_, _, _ = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0)

        state.free .= true
        state.old_velocity .= one(eltype(state.old_velocity))
        state.ξ.x .= [2.0, -1.0, 3.0, 4.0, 5.0]
        state.ξ.θ .= [1.0, 1.0, 1.0, 1.0, 1.0]
        PDMPSamplers.update_all_stick_times!(rng, alg_, state, flow)

        sticky_before = copy(alg_.sticky_times)
        PDMPSamplers.move_forward_time!(state, 0.25, flow)
        PDMPSamplers.reflect!(state.ξ, 0.0, 2, flow)
        PDMPSamplers._update_sticky_schedule_after_reflect!(rng, alg_, state, flow, CoordinateMeta(2))

        @test alg_.sticky_times[1] == sticky_before[1]
        @test alg_.sticky_times[3] == sticky_before[3]
        @test alg_.sticky_times[4] == sticky_before[4]
        @test alg_.sticky_times[5] == sticky_before[5]
        @test alg_.sticky_times[2] == state.t[] + PDMPSamplers.freezing_time(state.ξ, flow, 2)
        @test last(first(alg_.sticky_pq)) == minimum(alg_.sticky_times)

        sticky_before_refresh = copy(alg_.sticky_times)
        PDMPSamplers.move_forward_time!(state, 0.1, flow)
        PDMPSamplers._update_sticky_schedule_after_refresh!(rng, alg_, state, flow)
        @test alg_.sticky_times == sticky_before_refresh

        sticky_before_horizon = copy(alg_.sticky_times)
        PDMPSamplers.move_forward_time!(state, 0.1, flow)
        PDMPSamplers._update_sticky_schedule_after_horizon_hit!(rng, alg_, state, flow)
        @test alg_.sticky_times == sticky_before_horizon
    end

    @testset "Sticky ZigZag fast-path does not enqueue non-stickable coordinates" begin
        d = 4
        target = gen_data(Distributions.ZeroMeanIsoNormal, d)
        flow = ZigZag(inv(Symmetric(cov(target.D))), mean(target.D))
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
        κ = [0.5, Inf, 0.75, Inf]
        alg = Sticky(GridThinningStrategy(), κ)
        ξ0 = SkeletonPoint(fill(2.0, d), ones(d))

        rng = Random.Xoshiro(8765)
        state, _, alg_, _, _ = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0)

        state.free .= true
        state.ξ.x .= [1.0, -1.0, 2.0, 3.0]
        state.ξ.θ .= [1.0, 1.0, 1.0, 1.0]
        PDMPSamplers.update_all_stick_times!(rng, alg_, state, flow)

        @test !haskey(alg_.sticky_pq, 2)
        @test isinf(alg_.sticky_times[2])

        PDMPSamplers.move_forward_time!(state, 0.25, flow)
        PDMPSamplers.reflect!(state.ξ, 0.0, 2, flow)
        PDMPSamplers._update_sticky_schedule_after_reflect!(rng, alg_, state, flow, CoordinateMeta(2))

        @test !haskey(alg_.sticky_pq, 2)
        @test isinf(alg_.sticky_times[2])
    end

    @testset "FactorizedTrace caches stay correct after repeated queries and mutation" begin
        flow = ZigZag(2)
        state0 = PDMPState(0.0, SkeletonPoint([0.0, 1.0], [1.0, -1.0]))
        trace = PDMPSamplers.FactorizedTrace(state0, flow)

        push!(trace, PDMPSamplers.FactorizedEvent(2, 1.0, 0.0, 1.0))
        push!(trace, PDMPSamplers.FactorizedEvent(1, 2.0, 2.0, -1.0))

        probs = [0.25, 0.5, 0.75]
        q1 = quantile(trace, probs; coordinate=1)
        q2 = quantile(trace, probs; coordinate=1)
        q_dense = quantile(PDMPTrace(trace), probs; coordinate=1)
        @test q1 == q2
        @test q1 ≈ q_dense

        last1 = last(trace)
        last2 = last(trace)
        @test last1.time == 2.0
        @test last1.time == last2.time
        @test last1.position == last2.position
        @test last1.velocity == last2.velocity

        push!(trace, PDMPSamplers.FactorizedEvent(1, 3.0, 1.0, 1.0))

        q3 = quantile(trace, probs; coordinate=1)
        q_dense2 = quantile(PDMPTrace(trace), probs; coordinate=1)
        @test q3 ≈ q_dense2

        last3 = last(trace)
        @test last3.time == 3.0
        @test last3.position == [1.0, 2.0]
        @test last3.velocity == [1.0, 1.0]
    end
end
