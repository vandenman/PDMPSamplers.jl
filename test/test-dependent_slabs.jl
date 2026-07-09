@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _active_prior_grad_alloc(provider, out, x, active)
    return @allocated active_prior_grad!(provider, out, x, active)
end

function _linear_gaussian_rate_oracle(provider, odds, flow, state, τ, can_stick)
    indices = beta_indices(provider)
    mean, cov = gaussian_slab(provider, state.ξ.x)
    active = BitVector(undef, length(indices))
    @inbounds for j in eachindex(indices)
        active[j] = state.free[indices[j]]
    end
    A = findall(active)
    beta_x = state.ξ.x[indices]
    beta_v = state.ξ.θ[indices]
    Cv = unstick_rate_constant(flow, 1)
    total = 0.0

    for j in eachindex(indices)
        if can_stick[indices[j]] && !active[j]
            logρ = log_model_add_odds(odds, active, j)
            isfinite(logρ) || continue
            if isempty(A)
                cond_mean = mean[j]
                cond_var = cov[j, j]
            else
                cov_AA = cov[A, A]
                cov_jA = cov[j, A]
                delta_A = beta_x[A] .+ τ .* beta_v[A] .- mean[A]
                cond_mean = mean[j] + dot(cov_jA, cov_AA \ delta_A)
                cond_var = cov[j, j] - dot(cov_jA, cov_AA \ cov[A, j])
            end
            total += Cv * exp(logρ) * pdf(Normal(cond_mean, sqrt(cond_var)), 0.0)
        end
    end
    return total
end

@testset "Dependent slab primitives" begin
    @testset "Model-prior odds" begin
        bern = BernoulliModelPriorOdds([0.2, 0.5, 1.0, 0.0])
        active = BitVector([false, true, false, false])
        @test log_model_add_odds(bern, active, 1) ≈ log(0.2) - log1p(-0.2)
        @test log_model_add_odds(bern, active, 2) ≈ 0.0
        @test log_model_add_odds(bern, active, 3) == Inf
        @test log_model_add_odds(bern, active, 4) == -Inf

        bb = BetaBernoulliModelPriorOdds(4, 2.0, 3.0)
        active_bb = BitVector([true, false, true, false])
        @test length(bb) == 4
        @test log_model_add_odds(bb, active_bb, 2) ≈ log(2.0 + 2) - log(3.0 + 4 - 2 - 1)
        @test_throws ArgumentError log_model_add_odds(bb, active_bb, 1)

        log_omega = log.([0.1, 0.2, 0.3, 0.4])
        size_prior = ExchangeableModelSizePrior(log_omega; normalize=true)
        @test exp(LogExpFunctions.logsumexp(size_prior.log_omega)) ≈ 1.0
        @test length(size_prior) == 3
        size_prior_copy = copy(size_prior)
        size_prior.log_omega[1] = -99.0
        @test size_prior_copy.log_omega[1] != size_prior.log_omega[1]

        active_size = BitVector([true, false, false])
        @test log_model_add_odds(size_prior_copy, active_size, 2) ≈
              size_prior_copy.log_omega[3] - size_prior_copy.log_omega[2] + log(2) - log(2)
        @test_throws BoundsError log_model_add_odds(size_prior_copy, active_size, 0)
        @test_throws DimensionMismatch log_model_add_odds(size_prior_copy, BitVector([false, true]), 1)
        @test_throws ArgumentError log_model_add_odds(size_prior_copy, active_size, 1)
    end

    @testset "Dense Gaussian slab conditioning" begin
        mean = [1.0, -0.5, 0.25]
        cov = [2.0 0.4 -0.2;
               0.4 1.5 0.3;
              -0.2 0.3 1.2]
        indices = [2, 4, 5]
        provider = DenseGaussianSlab(mean, cov, indices)
        x = [10.0, 1.3, 20.0, -0.1, 0.8]
        slab_mean, slab_cov = gaussian_slab(provider, x)
        @test slab_mean == mean
        @test slab_cov == cov
        @test log_boundary_density_zero(provider, x, falses(3), 2) ≈
              conditional_logdensity_zero(provider, x, falses(3), 2)
        @test_throws ArgumentError DenseGaussianSlab(zeros(2), [1.0 2.0; 2.0 1.0], 1:2)

        active = BitVector([true, false, true])
        @test active_logdensity(provider, x, active) ≈
              logpdf(MvNormal(mean[[1, 3]], cov[[1, 3], [1, 3]]), x[indices[[1, 3]]])

        empty_active = falses(3)
        @test conditional_logdensity_zero(provider, x, empty_active, 2) ≈
              logpdf(Normal(mean[2], sqrt(cov[2, 2])), 0.0)

        cov_AA = cov[[1, 3], [1, 3]]
        cov_jA = cov[2, [1, 3]]
        delta_A = x[indices[[1, 3]]] - mean[[1, 3]]
        cond_mean = mean[2] + dot(cov_jA, cov_AA \ delta_A)
        cond_var = cov[2, 2] - dot(cov_jA, cov_AA \ cov[[1, 3], 2])
        @test conditional_logdensity_zero(provider, x, active, 2) ≈
              logpdf(Normal(cond_mean, sqrt(cond_var)), 0.0)
        @test conditional_density_zero(provider, x, active, 2) ≈
              pdf(Normal(cond_mean, sqrt(cond_var)), 0.0)
        @test_throws ArgumentError conditional_logdensity_zero(provider, x, active, 1)

        diag_provider = DenseGaussianSlab(zeros(3), Matrix(Diagonal([4.0, 9.0, 16.0])), indices)
        @test conditional_density_zero(diag_provider, x, active, 2) ≈ pdf(Normal(0.0, 3.0), 0.0)
    end

    @testset "Active prior gradient" begin
        mean = [1.0, -0.5, 0.25]
        cov = [2.0 0.4 -0.2;
               0.4 1.5 0.3;
              -0.2 0.3 1.2]
        indices = [2, 4, 5]
        provider = DenseGaussianSlab(mean, cov, indices)
        x = [10.0, 1.3, 20.0, -0.1, 0.8]
        active = BitVector([true, false, true])
        out = fill(NaN, length(x))
        active_prior_grad!(provider, out, x, active)

        A = [1, 3]
        expected_active = cov[A, A] \ (x[indices[A]] - mean[A])
        expected = zeros(length(x))
        expected[indices[A]] .= expected_active
        @test out ≈ expected

        out_alias = fill(NaN, length(x))
        active_prior_neggrad!(provider, out_alias, x, active)
        @test out_alias ≈ expected

        active_prior_grad!(provider, out, x, active)
        _active_prior_grad_alloc(provider, out, x, active)
        @test _active_prior_grad_alloc(provider, out, x, active) == 0
        @test_throws DimensionMismatch active_prior_grad!(provider, out, x, BitVector([true, false]))

        out_block = fill(NaN, length(indices))
        active_prior_grad!(provider, out_block, x, active)
        expected_block = zeros(length(indices))
        expected_block[A] .= expected_active
        @test out_block ≈ expected_block

        @test_throws DimensionMismatch active_prior_grad!(provider, fill(NaN, 4), x, active)

        all_active = trues(3)
        active_prior_grad!(provider, out_block, x, all_active)
        @test out_block ≈ cov \ (x[indices] - mean)

        empty_out = fill(NaN, length(x))
        active_prior_grad!(provider, empty_out, x, falses(3))
        @test empty_out == zeros(length(x))

        exch = ExchangeableGaussianSlab(indices, 0.2, 1.3, 0.1)
        generic_empty_out = fill(NaN, length(x))
        active_prior_grad!(exch, generic_empty_out, x, falses(3))
        @test generic_empty_out == zeros(length(x))
        active_prior_grad!(exch, generic_empty_out, x, BitVector([true, false, true]))
        exch_mean, exch_cov = gaussian_slab(exch, x)
        expected_exch = zeros(length(x))
        expected_exch[indices[[1, 3]]] .= exch_cov[[1, 3], [1, 3]] \ (x[indices[[1, 3]]] - exch_mean[[1, 3]])
        @test generic_empty_out ≈ expected_exch
    end

    @testset "DependentSlabTarget and active-set synchronization" begin
        d = 3
        posterior_grad!(out, x) = (fill!(out, 0.0); out)
        prior_grad!(out, x) = (fill!(out, 0.0); out)
        slab = DenseGaussianSlab(zeros(2), Matrix(I, 2, 2), [1, 2])
        odds = BernoulliModelPriorOdds([0.5, 0.5])
        target = DependentSlabTarget(d, posterior_grad!, prior_grad!, slab, odds)
        model = PDMPModel(target)
        flow = ZigZag(d)
        x = [3.0, -2.0, 1.0]
        θ = [1.0, 0.0, -1.0]
        state = StickyPDMPState(Ref(0.0), SkeletonPoint(copy(x), copy(θ)), BitVector([true, false, true]), zeros(d))
        cache = (; ∇ϕx=zeros(d))

        grad = compute_gradient!(state, model.grad, flow, cache)
        @test grad ≈ [3.0, 0.0, 0.0]
        @test target.free == state.free

        state.free .= BitVector([false, true, true])
        grad2 = PDMPSamplers.compute_gradient_for_reflection!(state, model.grad, flow, cache)
        @test grad2 ≈ [0.0, -2.0, 0.0]
        @test target.free == state.free

        target_copy = copy(target)
        @test target_copy !== target
        @test target_copy.free == target.free
        @test target_copy.free !== target.free
        @test target_copy.slab_provider !== target.slab_provider
        target.free[1] = !target.free[1]
        @test target_copy.free != target.free
    end

    @testset "Arbitrary slab target baseline" begin
        d = 3
        posterior_grad!(out, x) = (copyto!(out, x); out)
        prior_grad!(out, x) = (fill!(out, 0.5); out)
        provider = ArbitrarySlabBoundary([1, 3];
            active_prior_neggrad! = (out, x, active) -> (fill!(out, 0.0); active[1] && (out[1] = 2x[1]); active[2] && (out[3] = 3x[3]); out),
            log_q_zero! = (x, active, j) -> logpdf(Normal(), 0.0),
        )
        target = DependentSlabTarget(d, posterior_grad!, prior_grad!, provider, BernoulliModelPriorOdds(fill(0.5, 2));
            initial_free=BitVector([true, false, true]))
        out = fill(NaN, d)
        x = [1.0, 2.0, -1.0]
        target(out, x)
        @test out ≈ x .- 0.5 .+ [2.0, 0.0, -3.0]
        @test PDMPModel(target) isa PDMPModel
    end

    @testset "Arbitrary slab boundary and summed aggregate clock" begin
        d = 3
        beta_idx = [1, 2, 3]
        σ = [2.0, 3.0, 5.0]
        provider = ArbitrarySlabBoundary(beta_idx;
            active_prior_neggrad! = (out, x, active) -> (fill!(out, 0.0); out),
            log_q_zero! = (x, active, j) -> logpdf(Normal(0.0, σ[j]), 0.0),
        )
        odds = BernoulliModelPriorOdds(fill(0.5, d))
        clock = SummedRateClock(provider, odds)
        flow = ZigZag(d)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint(zeros(d), zeros(d)),
            falses(d),
            zeros(d),
        )
        can_stick = BitVector([true, false, true])

        expected = pdf(Normal(0.0, σ[1]), 0.0) + pdf(Normal(0.0, σ[3]), 0.0)
        @test PDMPSamplers.rate(clock, flow, state, 0.0, can_stick) ≈ expected

        rng = MersenneTwister(42)
        @test PDMPSamplers.sample_label(rng, clock, flow, state, can_stick) in (1, 3)

        can_stick_only3 = BitVector([false, false, true])
        for _ in 1:20
            @test PDMPSamplers.sample_label(rng, clock, flow, state, can_stick_only3) == 3
        end
    end

    @testset "Gaussian aggregate weights and cache traits" begin
        mean = [0.4, -0.2, 0.1, 0.7]
        cov = [2.0 0.3 -0.1 0.2;
               0.3 1.7 0.4 -0.2;
              -0.1 0.4 1.4 0.25;
               0.2 -0.2 0.25 1.9]
        indices = [1, 2, 4, 5]
        provider = DenseGaussianSlab(mean, cov, indices)
        x = [0.8, -0.4, 10.0, 0.3, -0.9]
        active = BitVector([true, false, true, false])
        stickable = BitVector([true, true, false, true])
        odds = BernoulliModelPriorOdds([0.25, 0.5, 0.75, 0.8])

        weights = fill(NaN, 4)
        boundary_logweights!(weights, provider, odds, x, active, stickable)
        expected = fill(-Inf, 4)
        for j in eachindex(expected)
            if stickable[j] && !active[j]
                expected[j] = log_model_add_odds(odds, active, j) +
                              conditional_logdensity_zero(provider, x, active, j)
            end
        end
        @test weights ≈ expected

        lograte = aggregate_lograte(provider, odds, log(2.0), x, active, stickable)
        @test lograte ≈ log(2.0 * sum(exp, expected[isfinite.(expected)]))

        rng = MersenneTwister(7)
        @test sample_unstick_label(rng, provider, odds, x, active, stickable) in (2, 5)
        stickable_only4 = BitVector([false, false, false, true])
        for _ in 1:20
            @test sample_unstick_label(rng, provider, odds, x, active, stickable_only4) == 5
        end

        @test slab_cache_style(provider) isa FixedCovarianceCache
        @test slab_cache_key(provider, x, active) == active

        cb = CallbackGaussianSlab(indices;
            mean_cov! = (mean_out, cov_out, x) -> (copyto!(mean_out, mean); copyto!(cov_out, cov); nothing),
            active_prior_grad! = (out, x, active) -> (fill!(out, 0.0); out),
        )
        @test slab_cache_style(cb) isa NoSlabCache
        @test slab_cache_key(cb, x, active) === nothing
        cb_copy = copy(cb)
        @test beta_indices(cb_copy) == indices
        @test beta_indices(cb_copy) !== beta_indices(cb)

        cb_grad = CallbackGaussianSlab(indices;
            mean_cov! = (mean_out, cov_out, x) -> (copyto!(mean_out, mean); copyto!(cov_out, cov); nothing),
            active_prior_grad! = (out, x, active) -> (out .= active .* x[indices]; out),
        )
        cb_out = fill(NaN, 4)
        @test active_prior_grad!(cb_grad, cb_out, x, active) === cb_out
        @test cb_out ≈ active .* x[indices]
        cb_no_grad = CallbackGaussianSlab(indices;
            mean_cov! = (mean_out, cov_out, x) -> (copyto!(mean_out, mean); copyto!(cov_out, cov); nothing),
        )
        @test_throws ArgumentError active_prior_grad!(cb_no_grad, cb_out, x, active)

        cb_weights = fill(NaN, 4)
        boundary_logweights!(cb_weights, cb, odds, x, active, stickable)
        @test cb_weights ≈ expected

        empty_active = falses(4)
        empty_weights = fill(NaN, 4)
        boundary_logweights!(empty_weights, cb, odds, x, empty_active, stickable)
        expected_empty = fill(-Inf, 4)
        for j in eachindex(expected_empty)
            if stickable[j]
                expected_empty[j] = log_model_add_odds(odds, empty_active, j) +
                                    logpdf(Normal(mean[j], sqrt(cov[j, j])), 0.0)
            end
        end
        @test empty_weights ≈ expected_empty
    end

    @testset "Exchangeable slabs and model-size prior" begin
        indices = [1, 2, 3, 4]
        μ = 0.3
        u = 1.4
        v = 0.25
        dense_cov = Matrix{Float64}(I, 4, 4) .* u .+ fill(v, 4, 4)
        dense = DenseGaussianSlab(fill(μ, 4), dense_cov, indices)
        exch = ExchangeableGaussianSlab(indices, μ, u, v)
        zero_exch = ZeroMeanExchangeableGaussianSlab(indices, u, v)
        x = [0.8, 0.0, -0.2, 0.0]
        active = BitVector([true, false, true, false])
        stickable = trues(4)
        odds = BernoulliModelPriorOdds(fill(0.4, 4))

        dense_weights = fill(NaN, 4)
        exch_weights = fill(NaN, 4)
        boundary_logweights!(dense_weights, dense, odds, x, active, stickable)
        boundary_logweights!(exch_weights, exch, odds, x, active, stickable)
        @test exch_weights ≈ dense_weights
        @test slab_cache_style(exch) isa FixedCovarianceCache
        @test slab_cache_style(zero_exch) isa FixedCovarianceCache
        @test slab_cache_key(exch, x, active) == active
        @test slab_cache_key(exch, x, active) !== active
        @test slab_cache_key(zero_exch, x, active) == active

        exch_mean = fill(NaN, 4)
        exch_cov = fill(NaN, 4, 4)
        @test gaussian_slab!(exch, exch_mean, exch_cov, x) === nothing
        @test exch_mean == fill(μ, 4)
        @test exch_cov ≈ dense_cov
        @test_throws DimensionMismatch gaussian_slab!(exch, fill(NaN, 3), exch_cov, x)
        @test_throws DimensionMismatch gaussian_slab!(exch, exch_mean, fill(NaN, 3, 3), x)

        zero_mean = fill(NaN, 4)
        zero_cov = fill(NaN, 4, 4)
        @test gaussian_slab!(zero_exch, zero_mean, zero_cov, x) === nothing
        @test zero_mean == zeros(4)
        @test zero_cov ≈ dense_cov

        exch_copy = copy(exch)
        zero_copy = copy(zero_exch)
        @test beta_indices(exch_copy) == indices
        @test beta_indices(exch_copy) !== beta_indices(exch)
        @test beta_indices(zero_copy) == indices
        @test beta_indices(zero_copy) !== beta_indices(zero_exch)

        x_same_sum = [0.5, 0.0, 0.1, 0.0]
        x_other_same_sum = [0.2, 0.0, 0.4, 0.0]
        log_q1 = PDMPSamplers._exchangeable_log_q_zero(zero_exch, x_same_sum, active)
        log_q2 = PDMPSamplers._exchangeable_log_q_zero(zero_exch, x_other_same_sum, active)
        @test log_q1 ≈ log_q2

        log_omega = log.([0.2, 0.3, 0.25, 0.15, 0.1])
        size_prior = ExchangeableModelSizePrior(log_omega)
        lograte = aggregate_lograte(exch, size_prior, log(1.0), x, active, stickable)
        k = count(active)
        q = exp(PDMPSamplers._exchangeable_log_q_zero(exch, x, active))
        expected_rate = q * (4 - k) * exp(log_omega[k + 2] - log_omega[k + 1] + log(k + 1) - log(4 - k))
        @test exp(lograte) ≈ expected_rate
        short_prior = ExchangeableModelSizePrior(log.([0.3, 0.4, 0.3]))
        @test_throws DimensionMismatch aggregate_lograte(exch, short_prior, log(1.0), x, active, stickable)
        @test_throws DimensionMismatch SummedRateClock(exch, short_prior)
        @test_throws DimensionMismatch LinearGaussianAggregateClock(exch, short_prior)

        rng = MersenneTwister(11)
        labels = [sample_unstick_label(rng, exch, size_prior, x, active, stickable) for _ in 1:50]
        @test all(in((2, 4)), labels)
    end

    @testset "Linear Gaussian aggregate clock" begin
        mean = [0.2, -0.1, 0.4, -0.25]
        cov = [1.5 0.35 -0.15 0.2;
               0.35 1.2 0.25 -0.1;
              -0.15 0.25 1.8 0.3;
               0.2 -0.1 0.3 1.4]
        provider = DenseGaussianSlab(mean, cov, 1:4)
        odds = BernoulliModelPriorOdds([0.5, 0.35, 0.6, 0.8])
        clock = LinearGaussianAggregateClock(provider, odds)
        callback_provider = CallbackGaussianSlab(1:4;
            mean_cov! = (mean_out, cov_out, x) -> (copyto!(mean_out, mean); copyto!(cov_out, cov); nothing),
            active_prior_grad! = (out, x, active) -> (fill!(out, 0.0); out),
        )
        @test_throws ArgumentError LinearGaussianAggregateClock(callback_provider, odds)
        flow = ZigZag(4)

        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.7, 0.0, -0.3, 0.0], [1.0, 0.0, -1.0, 0.0]),
            BitVector([true, false, true, false]),
            zeros(4),
        )
        can_stick = BitVector([true, true, true, false])
        for τ in (0.0, 0.2, 0.75, 1.4)
            @test PDMPSamplers.rate(clock, flow, state, τ, can_stick) ≈
                  _linear_gaussian_rate_oracle(provider, odds, flow, state, τ, can_stick)
            bps_flow = BouncyParticle(4)
            @test PDMPSamplers.rate(clock, bps_flow, state, τ, can_stick) ≈
                  _linear_gaussian_rate_oracle(provider, odds, bps_flow, state, τ, can_stick)
        end
        τ_label = 0.4
        state_at = copy(state)
        move_forward_time!(state_at, τ_label, flow)
        for seed in 1:10
            @test PDMPSamplers.sample_label(MersenneTwister(seed), clock, flow, state, τ_label, can_stick) ==
                  PDMPSamplers.sample_label(MersenneTwister(seed), clock, flow, state_at, can_stick)
        end
        rng_alloc = MersenneTwister(21)
        PDMPSamplers.sample_label(rng_alloc, clock, flow, state, τ_label, can_stick)
        @test (@allocated PDMPSamplers.sample_label(rng_alloc, clock, flow, state, τ_label, can_stick)) == 0

        T = 1.25
        H = PDMPSamplers.cumulative_hazard(clock, flow, state, 0.0, T, can_stick)
        H_quad, _ = PDMPSamplers.QuadGK.quadgk(t -> PDMPSamplers.rate(clock, flow, state, t, can_stick), 0.0, T; rtol=1e-10, atol=1e-12)
        @test H ≈ H_quad rtol=1e-8 atol=1e-10

        const_provider = DenseGaussianSlab(zeros(2), Matrix(Diagonal([4.0, 9.0])), 1:2)
        const_clock = LinearGaussianAggregateClock(const_provider, BernoulliModelPriorOdds(fill(0.5, 2)))
        const_flow = ZigZag(2)
        const_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint(zeros(2), zeros(2)),
            falses(2),
            zeros(2),
        )
        const_can_stick = trues(2)
        λ0 = PDMPSamplers.rate(const_clock, const_flow, const_state, 0.0, const_can_stick)
        rng_time = MersenneTwister(99)
        rng_ref = MersenneTwister(99)
        τ = PDMPSamplers.sample_time(rng_time, const_clock, const_flow, const_state, Inf, const_can_stick)
        @test τ ≈ rand(rng_ref, Exponential()) / λ0 rtol=1e-7 atol=1e-8
    end

    @testset "AggregateSticky requests sticky state" begin
        d = 2
        provider = ArbitrarySlabBoundary(1:d;
            active_prior_neggrad! = (out, x, active) -> (fill!(out, 0.0); out),
            log_q_zero! = (x, active, j) -> logpdf(Normal(), 0.0),
        )
        clock = SummedRateClock(provider, BernoulliModelPriorOdds(fill(0.5, d)))
        alg = AggregateSticky(GridThinningStrategy(), clock, trues(d))
        @test PDMPSamplers.requires_sticky_state(alg)

        posterior_grad!(out, x) = (fill!(out, 0.0); out)
        model = PDMPModel(d, FullGradient(posterior_grad!), nothing)
        ξ = SkeletonPoint([1.0, 2.0], [1.0, -1.0])
        state, _, alg_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(123), ZigZag(d), model, alg, 0.0, ξ)
        @test state isa StickyPDMPState
        @test alg_internal isa PDMPSamplers.AggregateStickyLoopState

        linear_clock = LinearGaussianAggregateClock(DenseGaussianSlab(zeros(d), Matrix(I, d, d), 1:d), BernoulliModelPriorOdds(fill(0.5, d)))
        linear_alg = AggregateSticky(GridThinningStrategy(), linear_clock, trues(d))
        _, _, linear_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(124), ZigZag(d), model, linear_alg, 0.0, ξ)
        @test linear_internal.clock !== linear_clock
        @test linear_internal.clock.cache !== linear_clock.cache

        non_grid_alg = AggregateSticky(ThinningStrategy(GlobalBounds(1.0, d)), clock, trues(d))
        @test_throws ArgumentError PDMPSamplers.initialize_state(MersenneTwister(125), ZigZag(d), model, non_grid_alg, 0.0, ξ)
        @test_throws ArgumentError PDMPSamplers.initialize_state(MersenneTwister(126), Boomerang(d), model, alg, 0.0, ξ)
        @test_throws ArgumentError PDMPSamplers.initialize_state(MersenneTwister(127), PreconditionedZigZag(d), model, alg, 0.0, ξ)
        @test_throws ArgumentError PDMPSamplers.unstick_rate_constant(Boomerang(d), 1)
        @test_throws ArgumentError PDMPSamplers.draw_boundary_velocity!(MersenneTwister(128), state, Boomerang(d), 1)
        @test_throws ArgumentError PDMPSamplers.unstick_rate_constant(PreconditionedZigZag(d), 1)
        @test_throws ArgumentError PDMPSamplers.draw_boundary_velocity!(MersenneTwister(129), state, PreconditionedZigZag(d), 1)
    end
end
