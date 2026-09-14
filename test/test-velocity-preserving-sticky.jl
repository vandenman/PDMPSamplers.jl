@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _vp_fixture(flow; velocity=[-2.0, 3.0])
    model = PDMPModel(2, FullGradient((out, x) -> fill!(out, 0.0)))
    strategy = Sticky(GridThinningStrategy(), [2.0, Inf])
    state, _, alg, _, _ = PDMPSamplers.initialize_state(
        Random.Xoshiro(91), flow, model, strategy, 0.0,
        SkeletonPoint([0.0, 0.4], copy(velocity));
        initial_free=BitVector([true, true]),
        initial_stored_velocity=zeros(2))
    return state, alg
end

_vp_freeze_alloc(state) = @allocated PDMPSamplers._freeze_preserved_velocity!(state, 1)
_vp_release_alloc(state) = @allocated PDMPSamplers._release_preserved_velocity!(state, 1)
_vp_clock_alloc(flow, state) = @allocated PDMPSamplers._boundary_proposal_clock_constant(flow, state, 1)

@testset "velocity-preserving diagonal sticky boundary kernel" begin
    flows = (
        PreconditionedDynamics(DiagonalPreconditioner([1.7, 0.8]), ZigZag(2)),
        PreconditionedDynamics(DiagonalPreconditioner([1.7, 0.8]),
            BouncyParticle(2, 0.0)),
        PreconditionedDynamics(DiagonalPreconditioner([1.7, 0.8]),
            Boomerang(2)),
    )

    for flow in flows
        state, alg = _vp_fixture(flow)
        before_other = state.ξ.θ[2]
        incoming = state.ξ.θ[1]
        @test PDMPSamplers.stick_or_unstick!(
            Random.Xoshiro(11), state, flow, alg, 1)
        @test !state.free[1]
        @test iszero(state.ξ.x[1]) && iszero(state.ξ.θ[1])
        @test state.stored_velocity[1] == incoming
        @test state.ξ.θ[2] == before_other

        if PDMPSamplers._underlying_flow(flow) isa PDMPSamplers.AnyBoomerang
            PDMPSamplers.move_forward_time!(state, 0.73, flow)
            @test iszero(state.ξ.x[1]) && iszero(state.ξ.θ[1])
            @test state.stored_velocity[1] == incoming
        end

        before_other = state.ξ.θ[2]
        @test PDMPSamplers.stick_or_unstick!(
            Random.Xoshiro(12), state, flow, alg, 1)
        @test state.free[1]
        @test state.ξ.θ[1] == incoming
        @test iszero(state.stored_velocity[1])
        @test state.ξ.θ[2] == before_other
    end
end

@testset "stored speed controls release rate and holding time" begin
    flow = PreconditionedDynamics(DiagonalPreconditioner([2.0]), ZigZag(1))
    model = PDMPModel(1, FullGradient((out, x) -> fill!(out, 0.0)))
    strategy = Sticky(GridThinningStrategy(), [3.0])
    state, _, alg, _, _ = PDMPSamplers.initialize_state(
        Random.Xoshiro(2), flow, model, strategy, 0.0,
        SkeletonPoint([0.0], [0.0]);
        initial_free=falses(1), initial_stored_velocity=[-2.0])
    @test PDMPSamplers._boundary_proposal_clock_constant(flow, state, 1) == 2.0

    rng = Random.Xoshiro(731)
    draws = [PDMPSamplers.unsticking_time(rng, alg, state, flow, 1)
             for _ in 1:20_000]
    @test mean(draws) ≈ 1 / 6 atol=0.006
end

@testset "optimized linear clocks use stored speeds" begin
    flows = (
        BouncyParticle(1, 0.0),
        PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(),
            BouncyParticle(1, 0.0)),
        PreconditionedDynamics(DiagonalPreconditioner([1.0]),
            BouncyParticle(1, 0.0)),
    )
    provider = IndependentZeroMeanGaussianSlab([pdf(Normal(), 0.0)], [1])
    prior = BernoulliModelPrior([0.3])
    can_stick = trues(1)
    expected_unit = pdf(Normal(), 0.0) * 0.3 / 0.7

    for flow in flows
        clock = default_aggregate_unstick_clock(provider, prior, flow)
        @test clock isa PDMPSamplers.LinearGaussianAggregateClock
        for speed in (1.0, 3.0)
            state = StickyPDMPState(0.0, SkeletonPoint([0.0], [0.0]),
                falses(1), [speed])
            λ = PDMPSamplers.rate(clock, flow, state, 0.0, can_stick)
            @test λ ≈ speed * expected_unit rtol=1e-14
            @test PDMPSamplers.cumulative_hazard(clock, flow, state, 0.0, 2.5,
                can_stick) ≈ 2.5λ rtol=1e-14
            draws = [PDMPSamplers.sample_time(Random.Xoshiro(10_000 + i), clock, flow,
                state, Inf, can_stick) for i in 1:10_000]
            @test mean(draws) ≈ inv(λ) rtol=0.035
        end
    end

    # Unequal stored speeds must also determine the aggregate-event label.
    provider2 = IndependentZeroMeanGaussianSlab(
        fill(pdf(Normal(), 0.0), 2), [1, 2])
    prior2 = BernoulliModelPrior(fill(0.3, 2))
    clock2 = default_aggregate_unstick_clock(provider2, prior2,
        BouncyParticle(2, 0.0))
    state2 = StickyPDMPState(0.0, SkeletonPoint(zeros(2), zeros(2)),
        falses(2), [1.0, 3.0])
    rng = Random.Xoshiro(781)
    labels = [PDMPSamplers.sample_label(rng, clock2, BouncyParticle(2, 0.0), state2,
        trues(2)) for _ in 1:20_000]
    @test count(==(2), labels) / length(labels) ≈ 0.75 atol=0.012
end

@testset "exchangeable and logscale clocks use stored speeds" begin
    flows = (
        BouncyParticle(4, 0.0),
        PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(),
            BouncyParticle(4, 0.0)),
        PreconditionedDynamics(DiagonalPreconditioner(ones(4)),
            BouncyParticle(4, 0.0)),
    )
    prior = BernoulliModelPrior(fill(0.4, 2))
    fixed = ZeroMeanExchangeableGaussianSlab([1, 2], 1.0, 0.2)
    varying = IndependentZeroMeanLogscaleGaussianSlab(
        [1, 2], [3, 4], zeros(2))
    state = StickyPDMPState(0.0,
        SkeletonPoint(zeros(4), zeros(4)),
        BitVector([false, false, true, true]), [1.0, 3.0, 0.0, 0.0])
    can_stick = BitVector([true, true, false, false])

    for provider in (fixed, varying), flow in flows
        clock = default_aggregate_unstick_clock(provider, prior, flow)
        λ1 = PDMPSamplers.rate(clock, flow, state, 0.37, can_stick)
        state.stored_velocity[2] = 6.0
        λ2 = PDMPSamplers.rate(clock, flow, state, 0.37, can_stick)
        @test λ2 > λ1
        @test λ2 / λ1 ≈ 7 / 4 rtol=1e-12
        state.stored_velocity[2] = 3.0
        rng = Random.Xoshiro(1781)
        labels = [PDMPSamplers.sample_label(rng, clock, flow, state, 0.37, can_stick)
                  for _ in 1:20_000]
        @test count(==(2), labels) / length(labels) ≈ 0.75 atol=0.012
    end

    shared = GlobalLogscaleExchangeableGaussianSlab(
        [1, 2], 3, 1.0, 0.2)
    shared_state = StickyPDMPState(0.0,
        SkeletonPoint(zeros(3), zeros(3)),
        BitVector([false, false, true]), [1.0, 3.0, 0.0])
    shared_mask = BitVector([true, true, false])
    shared_flow = BouncyParticle(3, 0.0)
    shared_clock = default_aggregate_unstick_clock(shared, prior, shared_flow)
    @test shared_clock isa PDMPSamplers.ChebyshevResidualAggregateClock
    λ1 = PDMPSamplers.rate(shared_clock, shared_flow, shared_state, 0.2,
        shared_mask)
    shared_state.stored_velocity[2] = 6.0
    λ2 = PDMPSamplers.rate(shared_clock, shared_flow, shared_state, 0.2,
        shared_mask)
    @test λ2 / λ1 ≈ 7 / 4 rtol=1e-11
end

@testset "complete boundary ratio under both model priors" begin
    q0 = pdf(Normal(0.0, 1.7), 0.0)
    provider = IndependentZeroMeanGaussianSlab([q0], [1])
    flow = PreconditionedDynamics(DiagonalPreconditioner([2.3]),
        BouncyParticle(1, 0.0))
    state = StickyPDMPState(0.0, SkeletonPoint([0.0], [0.0]),
        falses(1), [-1.4])
    can_stick = trues(1)
    for prior in (BernoulliModelPrior([0.3]),
                  BetaBernoulliModelPrior(1, 0.7, 1.3))
        clock = PDMPSamplers.SummedRateClock(provider, prior)
        expected = abs(state.stored_velocity[1]) * q0 *
            exp(PDMPSamplers.log_model_add_odds(prior, falses(1), 1))
        @test PDMPSamplers.rate(clock, flow, state, 0.0, can_stick) ≈
            expected rtol=1e-14
    end
end

@testset "prewarmed boundary transitions allocate zero bytes" begin
    flow = PreconditionedDynamics(DiagonalPreconditioner([1.5, 0.75]),
        ZigZag(2))
    state, alg = _vp_fixture(flow)
    PDMPSamplers._freeze_preserved_velocity!(state, 1)
    PDMPSamplers._release_preserved_velocity!(state, 1)
    @test _vp_freeze_alloc(state) == 0
    @test _vp_release_alloc(state) == 0
    @test _vp_clock_alloc(flow, state) == 0
end

@testset "Gaussian refresh preserves frozen sign and updates its clock" begin
    flows = (
        PreconditionedDynamics(DiagonalPreconditioner([2.0, 0.5]),
            BouncyParticle(2, 1.0)),
        PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(),
            Boomerang(2)),
        PreconditionedDynamics(DiagonalPreconditioner([2.0, 0.5]),
            Boomerang(2)),
    )
    for flow in flows
        state = StickyPDMPState(0.0,
            SkeletonPoint([0.0, 0.2], [0.0, 0.3]),
            BitVector([false, true]), [-1.25, 0.0])
        PDMPSamplers.refresh_velocity!(Random.Xoshiro(18), state, flow)
        @test state.stored_velocity[1] < 0
        @test iszero(state.ξ.θ[1])
        @test !iszero(state.ξ.θ[2])
        @test PDMPSamplers._boundary_proposal_clock_constant(flow, state, 1) ==
              abs(state.stored_velocity[1])
    end
end

@testset "harmonic aggregate clock refreshes stored-speed constants" begin
    provider = IndependentZeroMeanLogscaleGaussianSlab(
        [1, 2], [3, 4], zeros(2))
    prior = BernoulliModelPrior([0.35, 0.65])
    clock = PDMPSamplers.HarmonicLogLinearAggregateClock(provider, prior)
    flow = PreconditionedDynamics(DiagonalPreconditioner(ones(4)),
        Boomerang(4))
    state = StickyPDMPState(0.0,
        SkeletonPoint([0.0, 0.0, 0.2, -0.3], [0.0, 0.0, 0.1, -0.2]),
        BitVector([false, false, true, true]), [1.0, -2.0, 0.0, 0.0])
    can_stick = BitVector([true, true, false, false])

    first_rate = PDMPSamplers.rate(clock, flow, state, 0.17, can_stick)
    @test first_rate ≈ PDMPSamplers.rate(
        clock.fallback, flow, state, 0.17, can_stick) rtol=1e-12

    # A frozen-coordinate refresh changes the individual release speed while
    # leaving the flow and metric generation unchanged.
    state.stored_velocity[1] = -4.5
    second_rate = PDMPSamplers.rate(clock, flow, state, 0.17, can_stick)
    @test second_rate ≈ PDMPSamplers.rate(
        clock.fallback, flow, state, 0.17, can_stick) rtol=1e-12
    @test second_rate != first_rate
end

@testset "authoritative terminal sticky state supports restart" begin
    flow = PreconditionedDynamics(DiagonalPreconditioner([1.3]), ZigZag(1))
    model = PDMPModel(1, FullGradient((out, x) -> fill!(out, 0.0)))
    strategy = Sticky(GridThinningStrategy(), [0.0])
    first_run = pdmp_sample(SkeletonPoint([0.0], [0.0]), flow, model,
        strategy, 0.0, 0.2; seed=3, progress=false)
    endpoint = terminal_state(first_run)
    @test !endpoint.free[1]
    @test !iszero(endpoint.stored_frozen_velocity[1])

    second_run = pdmp_sample(
        SkeletonPoint(copy(endpoint.position), copy(endpoint.physical_velocity)),
        endpoint.flow, model, strategy, endpoint.t, endpoint.t + 0.2;
        seed=4, progress=false, initial_free=endpoint.free,
        initial_stored_velocity=endpoint.stored_frozen_velocity)
    restarted = terminal_state(second_run)
    @test restarted.free == endpoint.free
    @test restarted.stored_frozen_velocity == endpoint.stored_frozen_velocity
    @test restarted.t == endpoint.t + 0.2
end

@testset "factorized Zig–Zag trace reconstructs the live endpoint" begin
    flow = PreconditionedDynamics(DiagonalPreconditioner([1.2, 0.7]), ZigZag(2))
    model = PDMPModel(2, FullGradient((out, x) -> copyto!(out, x)),
        (out, x, v) -> copyto!(out, v))
    strategy = Sticky(GridThinningStrategy(), fill(pdf(Normal(), 0.0), 2))
    chains = pdmp_sample(SkeletonPoint([0.35, -0.42], [-1.2, 0.7]),
        flow, model, strategy, 0.0, 20.0; seed=912, progress=false)
    trace = chains.traces[1]
    @test trace isa PDMPSamplers.FactorizedTrace
    dense = PDMPTrace(trace)
    endpoint = terminal_state(chains)
    @test last(dense).position ≈ endpoint.position atol=1e-12 rtol=1e-12
    @test last(dense).velocity ≈ endpoint.physical_velocity atol=1e-12 rtol=1e-12
    reconstructed_free = .!(iszero.(last(dense).position) .&
        iszero.(last(dense).velocity))
    @test reconstructed_free == endpoint.free
    @test chains.stats[1].sticky_freezes > 0
    @test chains.stats[1].sticky_unfreezes > 0
end

@testset "analytic spike-and-slab occupancy" begin
    model = PDMPModel(1, FullGradient((out, x) -> copyto!(out, x)),
        (out, x, v) -> copyto!(out, v))
    strategy = Sticky(GridThinningStrategy(), [pdf(Normal(), 0.0)])
    chains = pdmp_sample(SkeletonPoint([0.4], [-1.0]), ZigZag(1), model,
        strategy, 0.0, 20_000.0; seed=1907, progress=false)
    @test inclusion_probs(chains)[1] ≈ 0.5 atol=0.035
    @test chains.stats[1].sticky_freezes > 100
    @test chains.stats[1].sticky_unfreezes > 100
end

@testset "aggregate-clock BPS and Boomerang occupancy" begin
    provider = IndependentZeroMeanGaussianSlab([pdf(Normal(), 0.0)], [1])
    prior = BernoulliModelPrior([0.5])
    model = PDMPModel(1, FullGradient((out, x) -> copyto!(out, x)),
        (out, x, v) -> copyto!(out, v))
    for flow in (BouncyParticle(1, 0.1), Boomerang(1))
        clock = default_aggregate_unstick_clock(provider, prior, flow)
        strategy = AggregateSticky(GridThinningStrategy(), clock, trues(1))
        chains = pdmp_sample(SkeletonPoint([0.4], [-1.0]), flow, model,
            strategy, 0.0, 5_000.0; seed=2907, progress=false)
        @test inclusion_probs(chains)[1] ≈ 0.5 atol=0.055
        @test chains.stats[1].sticky_freezes > 50
        @test chains.stats[1].sticky_unfreezes > 50
        if flow isa BouncyParticle
            @test chains.stats[1].refreshment_events > 50
        end
    end
end
