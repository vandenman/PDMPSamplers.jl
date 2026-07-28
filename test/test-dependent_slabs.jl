@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function _active_prior_grad_alloc(provider, out, x, active)
    return @allocated active_prior_grad!(provider, out, x, active)
end

function _conditional_logdensity_zero_alloc(provider, x, active, j)
    return @allocated conditional_logdensity_zero(provider, x, active, j)
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

        endpoint_provider = DenseGaussianSlab([0.0], reshape([1.0], 1, 1), [1])
        endpoint_prior_one = BernoulliModelPriorOdds([1.0])
        endpoint_prior_zero = BernoulliModelPriorOdds([0.0])
        endpoint_active = falses(1)
        endpoint_stickable = trues(1)
        endpoint_weights = fill(NaN, 1)
        boundary_logweights!(endpoint_weights, endpoint_provider, endpoint_prior_one, zeros(1), endpoint_active, endpoint_stickable)
        @test endpoint_weights == [Inf]
        @test aggregate_lograte(endpoint_provider, endpoint_prior_one, 0.0, zeros(1), endpoint_active, endpoint_stickable) == Inf
        @test aggregate_lograte(endpoint_provider, endpoint_prior_zero, 0.0, zeros(1), endpoint_active, endpoint_stickable) == -Inf
        @test sample_unstick_label(MersenneTwister(101), endpoint_provider, endpoint_prior_one, zeros(1), endpoint_active, endpoint_stickable) == 1
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
        _conditional_logdensity_zero_alloc(provider, x, active, 2)
        @test _conditional_logdensity_zero_alloc(provider, x, active, 2) == 0
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

        κ = [0.2, 0.5, 0.8]
        indep = IndependentZeroMeanGaussianSlab(κ, indices)
        indep_out = fill(NaN, length(x))
        active_prior_grad!(indep, indep_out, x, active)
        expected_indep = zeros(length(x))
        expected_indep[indices[[1, 3]]] .= @. 2π * κ[[1, 3]]^2 * x[indices[[1, 3]]]
        @test indep_out ≈ expected_indep
        @test conditional_logdensity_zero(indep, x, active, 2) ≈ log(κ[2])
        indep_block = fill(NaN, length(indices))
        active_prior_grad!(indep, indep_block, x, active)
        expected_indep_block = zeros(length(indices))
        expected_indep_block[[1, 3]] .= @. 2π * κ[[1, 3]]^2 * x[indices[[1, 3]]]
        @test indep_block ≈ expected_indep_block
        @test_throws DimensionMismatch active_prior_grad!(indep, fill(NaN, 4), x, active)

        logscale = IndependentZeroMeanLogscaleGaussianSlab(indices, [1, 1, 3], log.([2.0, 3.0, 4.0]))
        logscale_x = [0.1, 1.3, -0.2, -0.1, 0.8]
        logscale_active = BitVector([true, true, false])
        logscale_out = zeros(length(logscale_x))
        active_prior_grad!(logscale, logscale_out, logscale_x, logscale_active)
        expected_logscale = zeros(length(logscale_x))
        for j in (1, 2)
            β = logscale_x[indices[j]]
            log_s = logscale.log_base_scales[j] + logscale_x[logscale.logscale_indices[j]]
            inv_s2 = exp(-2log_s)
            expected_logscale[indices[j]] += β * inv_s2
            expected_logscale[logscale.logscale_indices[j]] += 1 - β^2 * inv_s2
        end
        @test logscale_out ≈ expected_logscale
        logscale_block = fill(NaN, length(indices))
        active_prior_grad!(logscale, logscale_block, logscale_x, logscale_active)
        expected_logscale_block = zeros(length(indices))
        for j in (1, 2)
            log_s = logscale.log_base_scales[j] + logscale_x[logscale.logscale_indices[j]]
            expected_logscale_block[j] = logscale_x[indices[j]] / exp(2log_s)
        end
        @test logscale_block ≈ expected_logscale_block
        @test_throws DimensionMismatch active_prior_grad!(logscale, fill(NaN, 4), logscale_x, logscale_active)
        @test conditional_logdensity_zero(logscale, logscale_x, logscale_active, 3) ≈
              -0.5 * log(2π) - (logscale.log_base_scales[3] + logscale_x[logscale.logscale_indices[3]])
        @test_throws ArgumentError IndependentZeroMeanLogscaleGaussianSlab([1, 2], [2, 3], zeros(2))
        @test_throws ArgumentError GlobalLogscaleExchangeableGaussianSlab([1, 2], 2, 1.0, 0.1)
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

        nuisance_posterior_grad!(out, x) = (out .= [x[1], 2x[2]]; out)
        slab_only_prior_grad!(out, x) = (out .= [x[1], 0.0]; out)
        one_beta_slab = DenseGaussianSlab([0.0], reshape([1.0], 1, 1), [1])
        nuisance_target = DependentSlabTarget(2, nuisance_posterior_grad!, slab_only_prior_grad!, one_beta_slab, BernoulliModelPriorOdds([0.5]);
            initial_free=BitVector([false, true]))
        nuisance_out = fill(NaN, 2)
        nuisance_target(nuisance_out, [3.0, 4.0])
        @test nuisance_out ≈ [0.0, 8.0]

        strategy_target = DependentSlabTarget(d, FullGradient((out, x) -> copyto!(out, 2 .* x)), prior_grad!, slab, odds)
        strategy_out = zeros(d)
        strategy_x = [1.0, -2.0, 3.0]
        strategy_target(strategy_out, strategy_x)
        @test strategy_out ≈ [3.0, -6.0, 6.0]
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
        indep = IndependentZeroMeanGaussianSlab([0.4, 0.5, 0.6, 0.7], indices)
        x_with_scales = vcat(x, [0.1, -0.2])
        logscale = IndependentZeroMeanLogscaleGaussianSlab(indices, [5, 5, 6, 6], log.([1.0, 1.5, 2.0, 2.5]))
        global_logscale = GlobalLogscaleExchangeableGaussianSlab(indices, 5, u, v; mean=μ, logscale_offset=0.2)
        @test slab_cache_style(indep) isa FixedCovarianceCache
        @test slab_cache_style(logscale) isa NoSlabCache
        @test slab_cache_style(global_logscale) isa NoSlabCache
        @test slab_cache_key(indep, x, active) == active
        @test slab_cache_key(indep, x, active) !== active
        @test slab_cache_key(logscale, x, active) === nothing
        @test slab_cache_key(global_logscale, x, active) === nothing

        exch_mean = fill(NaN, 4)
        exch_cov = fill(NaN, 4, 4)
        @test gaussian_slab!(exch, exch_mean, exch_cov, x) === nothing
        @test exch_mean == fill(μ, 4)
        @test exch_cov ≈ dense_cov
        indep_mean = fill(NaN, 4)
        indep_cov = fill(NaN, 4, 4)
        @test gaussian_slab!(indep, indep_mean, indep_cov, x) === nothing
        @test indep_mean == zeros(4)
        @test diag(indep_cov) ≈ inv.(indep.precision)
        logscale_mean = fill(NaN, 4)
        logscale_cov = fill(NaN, 4, 4)
        @test gaussian_slab!(logscale, logscale_mean, logscale_cov, x_with_scales) === nothing
        @test logscale_mean == zeros(4)
        @test diag(logscale_cov) ≈ exp.(2 .* (logscale.log_base_scales .+ x_with_scales[logscale.logscale_indices]))

        zero_mean = fill(NaN, 4)
        zero_cov = fill(NaN, 4, 4)
        @test gaussian_slab!(zero_exch, zero_mean, zero_cov, x) === nothing
        @test zero_mean == zeros(4)
        @test zero_cov ≈ dense_cov

        exch_copy = copy(exch)
        zero_copy = copy(zero_exch)
        indep_copy = copy(indep)
        logscale_copy = copy(logscale)
        @test beta_indices(exch_copy) == indices
        @test beta_indices(exch_copy) !== beta_indices(exch)
        @test beta_indices(zero_copy) == indices
        @test beta_indices(zero_copy) !== beta_indices(zero_exch)
        @test beta_indices(indep_copy) == indices
        @test beta_indices(indep_copy) !== beta_indices(indep)
        @test indep_copy.kappa == indep.kappa
        @test indep_copy.kappa !== indep.kappa
        @test beta_indices(logscale_copy) == indices
        @test logscale_copy.logscale_indices == logscale.logscale_indices
        @test logscale_copy.logscale_indices !== logscale.logscale_indices

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

        κ = [0.25, 0.5, 0.75]
        indep_provider = IndependentZeroMeanGaussianSlab(κ, 1:3)
        indep_dense = DenseGaussianSlab(zeros(3), Matrix(Diagonal(@. inv(2π * κ^2))), 1:3)
        indep_odds = BernoulliModelPriorOdds([0.3, 0.5, 0.8])
        indep_clock = LinearGaussianAggregateClock(indep_provider, indep_odds)
        indep_dense_clock = LinearGaussianAggregateClock(indep_dense, indep_odds)
        indep_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([1.0, 0.0, -0.5], [0.2, 0.0, -0.3]),
            BitVector([true, false, false]),
            zeros(3),
        )
        indep_can_stick = BitVector([true, true, false])
        @test PDMPSamplers.rate(indep_clock, flow, indep_state, 0.0, indep_can_stick) ≈
              PDMPSamplers.rate(indep_dense_clock, flow, indep_state, 0.0, indep_can_stick)
        @test PDMPSamplers.rate(indep_clock, flow, indep_state, 1.5, indep_can_stick) ≈
              PDMPSamplers.rate(indep_clock, flow, indep_state, 0.0, indep_can_stick)
        λ_indep = PDMPSamplers.rate(indep_clock, flow, indep_state, 0.0, indep_can_stick)
        @test PDMPSamplers.sample_time(MersenneTwister(200), indep_clock, flow, indep_state, Inf, indep_can_stick) ≈
              rand(MersenneTwister(200), Exponential()) / λ_indep rtol=1e-7 atol=1e-8

        endpoint_linear = LinearGaussianAggregateClock(DenseGaussianSlab([0.0], reshape([1.0], 1, 1), [1]), BernoulliModelPriorOdds([1.0]))
        endpoint_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0], [0.0]), falses(1), zeros(1))
        @test PDMPSamplers.rate(endpoint_linear, flow, endpoint_state, 0.0, trues(1)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(203), endpoint_linear, flow, endpoint_state, 1.0, trues(1)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(204), endpoint_linear, flow, endpoint_state, trues(1)) == 1
        endpoint_underflow = LinearGaussianAggregateClock(DenseGaussianSlab([1000.0], reshape([1.0], 1, 1), [1]), BernoulliModelPriorOdds([1.0]))
        @test PDMPSamplers.rate(endpoint_underflow, flow, endpoint_state, 0.0, trues(1)) == Inf
        @test PDMPSamplers.cumulative_hazard(endpoint_underflow, flow, endpoint_state, 0.0, 1.0, trues(1)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(209), endpoint_underflow, flow, endpoint_state, 1.0, trues(1)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(210), endpoint_underflow, flow, endpoint_state, trues(1)) == 1

        μ = 0.3
        u = 1.4
        v = 0.25
        exch_provider = ExchangeableGaussianSlab(1:4, μ, u, v)
        dense_exch_provider = DenseGaussianSlab(fill(μ, 4), Matrix{Float64}(I, 4, 4) .* u .+ fill(v, 4, 4), 1:4)
        nonexchangeable_odds = BernoulliModelPriorOdds([0.2, 0.5, 0.7, 0.8])
        exch_clock = LinearGaussianAggregateClock(exch_provider, nonexchangeable_odds)
        dense_exch_clock = LinearGaussianAggregateClock(dense_exch_provider, nonexchangeable_odds)
        exch_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.8, 0.0, -0.2, 0.0], [1.1, 0.0, -0.7, 0.0]),
            BitVector([true, false, true, false]),
            zeros(4),
        )
        exch_can_stick = BitVector([true, true, true, false])
        for flow_i in (ZigZag(4), BouncyParticle(4))
            for τi in (0.0, 0.2, 0.9)
                @test PDMPSamplers.rate(exch_clock, flow_i, exch_state, τi, exch_can_stick) ≈
                      PDMPSamplers.rate(dense_exch_clock, flow_i, exch_state, τi, exch_can_stick)
            end
            for T_i in (0.3, 1.5)
                @test PDMPSamplers.cumulative_hazard(exch_clock, flow_i, exch_state, 0.0, T_i, exch_can_stick) ≈
                      PDMPSamplers.cumulative_hazard(dense_exch_clock, flow_i, exch_state, 0.0, T_i, exch_can_stick) rtol=1e-9 atol=1e-10
            end
            @test PDMPSamplers.sample_time(MersenneTwister(201), exch_clock, flow_i, exch_state, 2.0, exch_can_stick) ≈
                  PDMPSamplers.sample_time(MersenneTwister(201), dense_exch_clock, flow_i, exch_state, 2.0, exch_can_stick) rtol=1e-7 atol=1e-8
            @test PDMPSamplers.sample_label(MersenneTwister(202), exch_clock, flow_i, exch_state, 0.7, exch_can_stick) ==
                  PDMPSamplers.sample_label(MersenneTwister(202), exch_clock, flow_i, exch_state, exch_can_stick)
        end

        zero_slope_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.8, 0.0, -0.2, 0.0], [1.0, 0.0, -1.0, 0.0]),
            BitVector([true, false, true, false]),
            zeros(4),
        )
        zero_slope_clock = LinearGaussianAggregateClock(exch_provider, BernoulliModelPriorOdds(fill(0.5, 4)))
        λ_const = PDMPSamplers.rate(zero_slope_clock, flow, zero_slope_state, 0.0, exch_can_stick)
        @test PDMPSamplers.rate(zero_slope_clock, flow, zero_slope_state, 1.0, exch_can_stick) ≈ λ_const
        @test PDMPSamplers.cumulative_hazard(zero_slope_clock, flow, zero_slope_state, 0.0, 1.3, exch_can_stick) ≈ 1.3 * λ_const

        bb_clock = LinearGaussianAggregateClock(exch_provider, BetaBernoulliModelPriorOdds(4, 2.0, 3.0))
        bb_labels = [PDMPSamplers.sample_label(MersenneTwister(seed), bb_clock, flow, exch_state, exch_can_stick) for seed in 1:200]
        @test all(==(2), bb_labels)

        subset_all_inactive = BitVector([false, true, false, true])
        subset_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint(zeros(4), zeros(4)),
            falses(4),
            zeros(4),
        )
        size_prior_subset = ExchangeableModelSizePrior(log.([0.2, 0.3, 0.25, 0.15, 0.1]))
        subset_clock = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:4, u, v), size_prior_subset)
        full_rate = PDMPSamplers.rate(subset_clock, flow, subset_state, 0.0, trues(4))
        subset_rate = PDMPSamplers.rate(subset_clock, flow, subset_state, 0.0, subset_all_inactive)
        @test subset_rate ≈ full_rate * count(subset_all_inactive) / 4

        endpoint_exch = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:2, 1.0, 0.1), BernoulliModelPriorOdds([1.0, 0.0]))
        endpoint_exch_state = StickyPDMPState(Ref(0.0), SkeletonPoint(zeros(2), zeros(2)), falses(2), zeros(2))
        @test PDMPSamplers.rate(endpoint_exch, flow, endpoint_exch_state, 0.0, trues(2)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(205), endpoint_exch, flow, endpoint_exch_state, 1.0, trues(2)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(206), endpoint_exch, flow, endpoint_exch_state, trues(2)) == 1
        endpoint_exch_second = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:2, 1.0, 0.1), BernoulliModelPriorOdds([0.0, 1.0]))
        @test PDMPSamplers.rate(endpoint_exch_second, flow, endpoint_exch_state, 0.0, trues(2)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(207), endpoint_exch_second, flow, endpoint_exch_state, 1.0, trues(2)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(208), endpoint_exch_second, flow, endpoint_exch_state, trues(2)) == 2

        independent_exch = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:4, u, 0.0), BernoulliModelPriorOdds(fill(0.5, 4)))
        independent_dense = LinearGaussianAggregateClock(DenseGaussianSlab(zeros(4), Matrix{Float64}(I, 4, 4) .* u, 1:4), BernoulliModelPriorOdds(fill(0.5, 4)))
        @test PDMPSamplers.rate(independent_exch, flow, exch_state, 0.4, trues(4)) ≈
              PDMPSamplers.rate(independent_dense, flow, exch_state, 0.4, trues(4))
    end

    @testset "Independent logscale exponential-sum clock" begin
        provider = IndependentZeroMeanLogscaleGaussianSlab([1, 2, 3], [4, 4, 5], log.([2.0, 3.0, 4.0]))
        odds = BernoulliModelPriorOdds([0.25, 0.5, 0.75])
        clock = ExponentialSumAggregateClock(provider, odds)
        flow = ZigZag(5)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.2, -0.1], [0.0, 0.0, 0.0, 0.3, -0.2]),
            falses(5),
            zeros(5),
        )
        can_stick = BitVector([true, false, true, false, false])

        function direct_rate(t)
            total = 0.0
            active = falses(3)
            for j in (1, 3)
                logρ = log_model_add_odds(odds, active, j)
                log_s0 = provider.log_base_scales[j] + state.ξ.x[provider.logscale_indices[j]]
                r = state.ξ.θ[provider.logscale_indices[j]]
                total += exp(logρ - 0.5 * log(2π) - log_s0 - r * t)
            end
            return total
        end

        for τ in (0.0, 0.4, 1.2)
            @test PDMPSamplers.rate(clock, flow, state, τ, can_stick) ≈ direct_rate(τ)
        end
        T = 1.7
        H = PDMPSamplers.cumulative_hazard(clock, flow, state, 0.0, T, can_stick)
        H_quad, _ = PDMPSamplers.QuadGK.quadgk(direct_rate, 0.0, T; rtol=1e-10, atol=1e-12)
        @test H ≈ H_quad rtol=1e-9 atol=1e-10

        τ_sample = PDMPSamplers.sample_time(MersenneTwister(301), clock, flow, state, 2.0, can_stick)
        if isfinite(τ_sample)
            threshold = rand(MersenneTwister(301), Exponential())
            @test PDMPSamplers.cumulative_hazard(clock, flow, state, 0.0, τ_sample, can_stick) ≈ threshold rtol=1e-7 atol=1e-8
        end

        state_at = copy(state)
        move_forward_time!(state_at, 0.6, flow)
        @test PDMPSamplers.sample_label(MersenneTwister(302), clock, flow, state, 0.6, can_stick) ==
              PDMPSamplers.sample_label(MersenneTwister(302), clock, flow, state_at, can_stick)
        @test default_aggregate_unstick_clock(provider, odds) isa ExponentialSumAggregateClock
        clock_copy = copy(clock)
        @test clock_copy !== clock
        @test clock_copy.slab_provider !== clock.slab_provider
        @test clock_copy.model_prior_odds !== clock.model_prior_odds
        @test clock_copy.rtol == clock.rtol
        @test clock_copy.atol == clock.atol
    end

    @testset "Global logscale exchangeable segment capability" begin
        provider = GlobalLogscaleExchangeableGaussianSlab(1:3, 4, 1.2, 0.3; mean=0.0, logscale_offset=0.1)
        odds = BernoulliModelPriorOdds(fill(0.5, 3))
        @test default_aggregate_unstick_clock(provider, odds) isa ChebyshevResidualAggregateClock
        flow = ZigZag(4)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.7, 0.0, -0.2, 0.4], [1.0, 0.0, -0.5, 0.25]),
            BitVector([true, false, true, true]),
            zeros(4),
        )
        can_stick = BitVector([true, true, true, false])
        seg = scalar_logscale_gaussian_line_segment(provider, odds, flow, state, can_stick, 2.0)
        k = 2
        c = provider.v / (provider.u + k * provider.v)
        base_s2 = provider.u * (provider.u + (k + 1) * provider.v) / (provider.u + k * provider.v)
        @test seg.a ≈ c * (state.ξ.x[1] + state.ξ.x[3])
        @test seg.b ≈ c * (state.ξ.θ[1] + state.ξ.θ[3])
        @test seg.s0 ≈ sqrt(base_s2)
        @test seg.ell0 ≈ provider.logscale_offset + state.ξ.x[provider.logscale_index]
        @test seg.r ≈ state.ξ.θ[provider.logscale_index]

        scaled_mean = fill(NaN, 3)
        scaled_cov = fill(NaN, 3, 3)
        gaussian_slab!(provider, scaled_mean, scaled_cov, state.ξ.x)
        dense_scaled = DenseGaussianSlab(scaled_mean, scaled_cov, 1:3)
        active_beta = BitVector([true, false, true])
        @test conditional_logdensity_zero(provider, state.ξ.x, active_beta, 2) ≈
              conditional_logdensity_zero(dense_scaled, state.ξ.x, active_beta, 2)

        grad_out = zeros(4)
        active_prior_grad!(provider, grad_out, state.ξ.x, active_beta)
        function active_energy(xv)
            mean_tmp, cov_tmp = gaussian_slab(provider, xv)
            A = [1, 3]
            return -logpdf(MvNormal(mean_tmp[A], cov_tmp[A, A]), xv[A])
        end
        eps_fd = 1e-6
        for i in (1, 3, 4)
            xp = copy(state.ξ.x); xm = copy(state.ξ.x)
            xp[i] += eps_fd; xm[i] -= eps_fd
            @test grad_out[i] ≈ (active_energy(xp) - active_energy(xm)) / (2eps_fd) rtol=1e-5 atol=1e-6
        end
    end

    @testset "Residual aggregate clocks delegate to exact fallback" begin
        provider = ArbitrarySlabBoundary(1:2;
            active_prior_neggrad! = (out, x, active) -> (fill!(out, 0.0); out),
            log_q_zero! = (x, active, j) -> logpdf(Normal(0.0, 1.0 + 0.25 * abs(x[j])), 0.0),
        )
        odds = BernoulliModelPriorOdds(fill(0.5, 2))
        summed = SummedRateClock(provider, odds; rtol=1e-8, atol=1e-10)
        cheb = ChebyshevResidualAggregateClock(provider, odds; order=8, max_cells=4, residual_budget=0.0)
        fourier = FourierResidualAggregateClock(provider, odds; order=8, cells=4, residual_budget=0.0)
        fourier_copy = copy(fourier)
        @test fourier_copy !== fourier
        @test fourier_copy.slab_provider !== fourier.slab_provider
        @test fourier_copy.model_prior_odds !== fourier.model_prior_odds
        @test fourier_copy.order == fourier.order
        @test fourier_copy.cells == fourier.cells
        @test default_aggregate_unstick_clock(provider, odds) isa SummedRateClock
        flow = ZigZag(2)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.5, -0.25], [1.0, -1.0]),
            falses(2),
            zeros(2),
        )
        can_stick = trues(2)

        for residual_clock in (cheb, fourier)
            @test PDMPSamplers.rate(residual_clock, flow, state, 0.3, can_stick) ≈
                  PDMPSamplers.rate(summed, flow, state, 0.3, can_stick)
            @test PDMPSamplers.cumulative_hazard(residual_clock, flow, state, 0.0, 0.7, can_stick) ≈
                  PDMPSamplers.cumulative_hazard(summed, flow, state, 0.0, 0.7, can_stick)
            @test PDMPSamplers.sample_label(MersenneTwister(11), residual_clock, flow, state, can_stick) ==
                  PDMPSamplers.sample_label(MersenneTwister(11), summed, flow, state, can_stick)
            @test PDMPSamplers.sample_time(MersenneTwister(12), residual_clock, flow, state, 0.5, can_stick) ==
                  PDMPSamplers.sample_time(MersenneTwister(12), summed, flow, state, 0.5, can_stick)
        end

        @test_throws ArgumentError ChebyshevResidualAggregateClock(provider, odds; order=0)
        @test_throws ArgumentError ChebyshevResidualAggregateClock(provider, odds; max_cells=0)
        @test_throws ArgumentError FourierResidualAggregateClock(provider, odds; order=0)
        @test_throws ArgumentError FourierResidualAggregateClock(provider, odds; cells=0)
        @test_throws ArgumentError FourierResidualAggregateClock(provider, odds; residual_budget=-1.0)

        copied = copy(cheb)
        @test copied.order == cheb.order
        @test copied.max_cells == cheb.max_cells
        @test copied.residual_budget == cheb.residual_budget
        @test copied.allow_slow_fallback == cheb.allow_slow_fallback
        @test copied.slab_provider !== cheb.slab_provider

        strict_cheb = ChebyshevResidualAggregateClock(provider, odds; order=8, max_cells=4, allow_slow_fallback=false)
        strict_fourier = FourierResidualAggregateClock(provider, odds; order=8, cells=4, allow_slow_fallback=false)
        @test PDMPSamplers.certified_residual_capability(provider, flow) isa PDMPSamplers.NoCertifiedResidualCapability
        @test_throws ArgumentError PDMPSamplers.sample_time(MersenneTwister(16), strict_cheb, flow, state, 1.0, can_stick)
        @test_throws ArgumentError PDMPSamplers.sample_time(MersenneTwister(17), strict_fourier, flow, state, 1.0, can_stick)
        @test PDMPSamplers.thinning_diagnostics(strict_cheb).fallbacks == 0

        scalar_provider = GlobalLogscaleExchangeableGaussianSlab(1:3, 4, 1.1, 0.25; logscale_offset=0.1)
        scalar_odds = BernoulliModelPriorOdds(fill(0.5, 3))
        @test default_aggregate_unstick_clock(scalar_provider, scalar_odds) isa ChebyshevResidualAggregateClock
        @test !default_aggregate_unstick_clock(scalar_provider, scalar_odds).allow_slow_fallback
        scalar_clock = ChebyshevResidualAggregateClock(scalar_provider, scalar_odds;
            order=10, max_cells=64, residual_budget=1e-3, allow_slow_fallback=false)
        scalar_flow = ZigZag(4)
        scalar_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.6, 0.0, -0.3, 0.2], [0.8, 0.0, -0.4, 0.15]),
            BitVector([true, false, true, true]),
            zeros(4),
        )
        scalar_can_stick = BitVector([true, true, true, false])
        seg = scalar_logscale_gaussian_line_segment(scalar_provider, scalar_odds, scalar_flow, scalar_state, scalar_can_stick, 1.5)
        env = PDMPSamplers._build_scalar_residual_envelope(scalar_clock, seg)
        for t in range(0.0, 1.5; length=501)
            @test PDMPSamplers._envelope_rate(env, t) + 1e-10 >=
                  PDMPSamplers._scalar_logscale_gaussian_line_rate(seg, t)
        end
        τ_scalar = PDMPSamplers.sample_time(MersenneTwister(18), scalar_clock, scalar_flow, scalar_state, 1.5, scalar_can_stick)
        @test τ_scalar == Inf || 0.0 <= τ_scalar <= 1.5
        scalar_diag = thinning_diagnostics(scalar_clock)
        @test scalar_diag.fallbacks == 0
        @test scalar_diag.last_cells <= scalar_clock.max_cells
        @test ismissing(scalar_diag.hazard_acceptance)

        endpoint_scalar_odds = BernoulliModelPriorOdds([1.0, 0.0, 0.0])
        endpoint_scalar_clock = ChebyshevResidualAggregateClock(scalar_provider, endpoint_scalar_odds;
            order=10, max_cells=64, residual_budget=1e-3, allow_slow_fallback=false)
        endpoint_scalar_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.0], zeros(4)),
            falses(4),
            zeros(4),
        )
        @test PDMPSamplers.rate(endpoint_scalar_clock, scalar_flow, endpoint_scalar_state, 0.1, scalar_can_stick) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(19), endpoint_scalar_clock, scalar_flow, endpoint_scalar_state, 1.0, scalar_can_stick) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(20), endpoint_scalar_clock, scalar_flow, endpoint_scalar_state, scalar_can_stick) == 1

        default_scalar_clock = default_aggregate_unstick_clock(scalar_provider, endpoint_scalar_odds)
        all_frozen_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.15]),
            BitVector([false, false, false, true]),
            zeros(4),
        )
        @test PDMPSamplers.sample_time(MersenneTwister(21), default_scalar_clock, scalar_flow, all_frozen_state, Inf, scalar_can_stick) == 0.0

        tiny_provider = GlobalLogscaleExchangeableGaussianSlab(1:1, 2, 1.0, 0.0; logscale_offset=100.0)
        tiny_odds = BernoulliModelPriorOdds([0.5])
        tiny_clock = default_aggregate_unstick_clock(tiny_provider, tiny_odds)
        tiny_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0], [0.0, 0.0]),
            falses(2),
            zeros(2),
        )
        tiny_can_stick = BitVector([true, false])
        tiny_rate = PDMPSamplers.rate(tiny_clock, ZigZag(2), tiny_state, 0.0, tiny_can_stick)
        tiny_threshold = rand(MersenneTwister(22), Exponential())
        @test tiny_rate > 0
        @test PDMPSamplers.sample_time(MersenneTwister(22), tiny_clock, ZigZag(2), tiny_state, Inf, tiny_can_stick) ≈
              tiny_threshold / tiny_rate rtol=1e-12

        defective_provider = GlobalLogscaleExchangeableGaussianSlab(1:1, 2, 1.0, 0.0; logscale_offset=10.0)
        defective_clock = default_aggregate_unstick_clock(defective_provider, tiny_odds)
        defective_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0], [0.0, 1.0]),
            falses(2),
            zeros(2),
        )
        @test PDMPSamplers.sample_time(MersenneTwister(23), defective_clock, ZigZag(2), defective_state, Inf, tiny_can_stick) == Inf

        peak_provider = GlobalLogscaleExchangeableGaussianSlab(1:2, 3, 1.0, 1.0)
        peak_clock = default_aggregate_unstick_clock(peak_provider, BernoulliModelPriorOdds(fill(0.5, 2)))
        peak_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([-20.0, 0.0, 0.0], [2.0, 0.0, -1.0]),
            BitVector([true, false, true]),
            zeros(3),
        )
        peak_can_stick = BitVector([true, true, false])
        peak_seg = scalar_logscale_gaussian_line_segment(peak_provider, BernoulliModelPriorOdds(fill(0.5, 2)), ZigZag(3), peak_state, peak_can_stick, Inf)
        @test peak_seg.a ≈ -10.0
        @test peak_seg.b ≈ 1.0
        @test peak_seg.r ≈ -1.0
        @test PDMPSamplers._scalar_logscale_gaussian_line_available_hazard(peak_seg) ≈ 1.0 rtol=1e-6
        peak_time = PDMPSamplers.sample_time(MersenneTwister(1), peak_clock, ZigZag(3), peak_state, Inf, peak_can_stick)
        @test isfinite(peak_time)
        @test abs(peak_time - 10.0) < 1e-2

        residual_clock = ChebyshevResidualAggregateClock(provider, odds; order=8, max_cells=4, residual_budget=0.5)
        PDMPSamplers.reset_thinning_diagnostics!(residual_clock)
        τ = PDMPSamplers.sample_time(MersenneTwister(13), residual_clock, flow, state, 3.0, can_stick)
        @test τ == PDMPSamplers.sample_time(MersenneTwister(13), summed, flow, state, 3.0, can_stick)
        diagnostics = PDMPSamplers.thinning_diagnostics(residual_clock)
        @test diagnostics.fallbacks == 1
        @test diagnostics.proposals == 0
        @test diagnostics.residual_area == 0.0
        @test diagnostics.envelope_hazard == 0.0
        @test diagnostics.rate_evaluations == 0

        PDMPSamplers.reset_thinning_diagnostics!(residual_clock)
        @test PDMPSamplers.thinning_diagnostics(residual_clock).proposals == 0
        @test PDMPSamplers.thinning_diagnostics(residual_clock).fallbacks == 0

        infinite_clock = FourierResidualAggregateClock(provider, odds; order=8, cells=4, residual_budget=0.5)
        PDMPSamplers.sample_time(MersenneTwister(14), infinite_clock, flow, state, Inf, can_stick)
        @test PDMPSamplers.thinning_diagnostics(infinite_clock).fallbacks == 1

        fourier_residual = FourierResidualAggregateClock(provider, odds; order=4, cells=6, residual_budget=0.25)
        @test PDMPSamplers.sample_time(MersenneTwister(15), fourier_residual, flow, state, 2.0, can_stick) ==
              PDMPSamplers.sample_time(MersenneTwister(15), summed, flow, state, 2.0, can_stick)
        fourier_diagnostics = PDMPSamplers.thinning_diagnostics(fourier_residual)
        @test fourier_diagnostics.fallbacks == 1
        @test fourier_diagnostics.rate_evaluations == 0
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

        scalar_provider = GlobalLogscaleExchangeableGaussianSlab(1:2, 3, 1.0, 0.1)
        scalar_clock = default_aggregate_unstick_clock(scalar_provider, BernoulliModelPriorOdds([0.0, 1.0]))
        scalar_alg = AggregateSticky(GridThinningStrategy(), scalar_clock, BitVector([true, true, false]))
        scalar_model = PDMPModel(3, FullGradient((out, x) -> (fill!(out, 0.0); out)), nothing)
        all_frozen_ξ = SkeletonPoint([0.0, 0.0, 0.2], [0.0, 0.0, 0.15])
        _, _, all_frozen_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(130), ZigZag(3), scalar_model, scalar_alg, 0.0, all_frozen_ξ)
        @test all_frozen_internal.aggregate_unstick_time == 0.0
        moving_away_ξ = SkeletonPoint([1.0, 0.0, 0.2], [1.0, 0.0, 0.15])
        _, _, moving_away_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(131), ZigZag(3), scalar_model, scalar_alg, 0.0, moving_away_ξ)
        @test moving_away_internal.aggregate_unstick_time == 0.0

        defective_provider2 = GlobalLogscaleExchangeableGaussianSlab(1:1, 2, 1.0, 0.0; logscale_offset=10.0)
        defective_clock2 = default_aggregate_unstick_clock(defective_provider2, BernoulliModelPriorOdds([0.5]))
        defective_alg = AggregateSticky(GridThinningStrategy(), defective_clock2, BitVector([true, false]))
        defective_model = PDMPModel(2, FullGradient((out, x) -> (fill!(out, 0.0); out)), nothing)
        defective_ξ = SkeletonPoint([0.0, 0.0], [0.0, 1.0])
        defective_state2, _, defective_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(132), ZigZag(2), defective_model, defective_alg, 0.0, defective_ξ)
        @test defective_internal.aggregate_unstick_time == Inf
        PDMPSamplers.update_all_unfreeze_times!(MersenneTwister(133), defective_internal, defective_state2, ZigZag(2))
        @test defective_internal.aggregate_unstick_time == Inf

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
