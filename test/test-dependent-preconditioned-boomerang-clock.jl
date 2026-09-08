@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _preconditioned_loglinear_clock_fixture(; active=BitVector([false, false]))
    beta = [1, 2]
    logs = [3, 4, 5]
    provider = PDMPSamplers.LogLinearGaussianScaleSlab(beta, logs,
        [0.0, 0.1], [1.0 0.5 0.5; 1.0 0.5 0.0])
    prior = PDMPSamplers.BernoulliModelPrior(fill(0.2, 2))
    scales = [2.0, 0.7, 1.2, 0.8, 1.5]
    flow = PDMPSamplers.PreconditionedDynamics(
        PDMPSamplers.DiagonalPreconditioner(scales),
        PDMPSamplers.Boomerang(Diagonal(ones(5)), zeros(5), 0.0))
    free = BitVector([active[1], active[2], true, true, true])
    x = [active[1] ? 0.2 : 0.0, active[2] ? -0.1 : 0.0,
         0.2, -0.3, 0.1]
    theta = [active[1] ? 0.25 : 0.0, active[2] ? -0.2 : 0.0,
             0.4, -0.2, 0.3]
    state = PDMPSamplers.StickyPDMPState(Ref(0.0),
        PDMPSamplers.SkeletonPoint(x, theta), free)
    return provider, prior, flow, state, trues(5)
end

function _clock_sample_allocated(rng, clock, flow, state, can_stick)
    return @allocated PDMPSamplers.sample_time(
        rng, clock, flow, state, 1.0, can_stick)
end

@testset "Certified dependent clock for diagonal preconditioned Boomerang" begin
    for active in (BitVector([false, false]), BitVector([true, false]),
                   BitVector([true, true]))
        provider, prior, flow, state, can_stick =
            _preconditioned_loglinear_clock_fixture(; active)
        clock = PDMPSamplers.default_aggregate_unstick_clock(
            provider, prior, flow)
        exact = PDMPSamplers.SummedRateClock(provider, prior)
        @test clock isa PDMPSamplers.HarmonicLogLinearAggregateClock

        for t in range(0.0, 1.0; length=257)
            @test PDMPSamplers.rate(clock, flow, state, t, can_stick) ≈
                  PDMPSamplers.rate(exact, flow, state, t, can_stick)
        end
        if !all(active)
            PDMPSamplers._prepare_harmonic_loglinear_cells!(
                clock, flow, state, 1.0, can_stick)
            for t in range(0.0, 1.0; length=20001)
                cell = min(clock.workspace.cells_used,
                    floor(Int, t * clock.workspace.cells_used) + 1)
                @test clock.workspace.roofs[cell] + 1e-10 >=
                    PDMPSamplers.rate(exact, flow, state, t, can_stick)
            end
            # Boundary constants must come from the complete preconditioned
            # flow, not the unwrapped unit-covariance Boomerang.
            for i in 1:2
                @test PDMPSamplers.unstick_rate_constant(flow, i) ≈
                      sqrt(2 / π) * flow.metric.scale[i]
            end
            rng = Xoshiro(44)
            PDMPSamplers.sample_time(rng, clock, flow, state, 1.0, can_stick)
            @test _clock_sample_allocated(
                Xoshiro(44), clock, flow, state, can_stick) == 0

            if !any(active)
                # The clock is rejection based, so fixed-stream event times
                # need not match the quadrature/root clock.  Instead compare
                # the survival law against its numerically integrated hazard.
                test_horizon = 0.25
                survival = exp(-PDMPSamplers.cumulative_hazard(
                    exact, flow, state, 0.0, test_horizon, can_stick))
                n_survival = 3000
                rng_survival = Xoshiro(8102)
                empirical_survival = count(_ -> !isfinite(
                    PDMPSamplers.sample_time(rng_survival, clock, flow,
                        state, test_horizon, can_stick)), 1:n_survival) /
                    n_survival
                survival_se = sqrt(survival * (1 - survival) / n_survival)
                @test abs(empirical_survival - survival) <=
                      5survival_se + 0.005

                # Label sampling must use the exact boundary weights, not the
                # cell-envelope component weights.
                individual_rates = zeros(2)
                for j in 1:2
                    state_j = deepcopy(state)
                    state_j.free[1:2] .= true
                    state_j.free[j] = false
                    individual_rates[j] = PDMPSamplers.rate(
                        exact, flow, state_j, 0.0, can_stick)
                end
                expected_label_one = individual_rates[1] /
                    sum(individual_rates)
                rng_label = Xoshiro(8103)
                n_labels = 20000
                observed_label_one = count(_ -> PDMPSamplers.sample_label(
                    rng_label, clock, flow, state, can_stick) == 1,
                    1:n_labels) / n_labels
                label_se = sqrt(expected_label_one *
                    (1 - expected_label_one) / n_labels)
                @test abs(observed_label_one - expected_label_one) <=
                      5label_se + 0.002
            end
        else
            @test PDMPSamplers.sample_time(
                Xoshiro(1), clock, flow, state, 1.0, can_stick) == Inf
        end
    end
end
