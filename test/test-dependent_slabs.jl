@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

const ChebyshevResidualAggregateClock = PDMPSamplers.ChebyshevResidualAggregateClock
const ExponentialSumAggregateClock = PDMPSamplers.ExponentialSumAggregateClock
const FixedCovarianceCache = PDMPSamplers.FixedCovarianceCache
const FourierResidualAggregateClock = PDMPSamplers.FourierResidualAggregateClock
const LinearGaussianAggregateClock = PDMPSamplers.LinearGaussianAggregateClock
const NoSlabCache = PDMPSamplers.NoSlabCache
const SummedRateClock = PDMPSamplers.SummedRateClock
const active_logdensity = PDMPSamplers.active_logdensity
const active_prior_grad! = PDMPSamplers.active_prior_grad!
const active_prior_neggrad! = PDMPSamplers.active_prior_neggrad!
const aggregate_lograte = PDMPSamplers.aggregate_lograte
const beta_indices = PDMPSamplers.beta_indices
const boundary_logweights! = PDMPSamplers.boundary_logweights!
const conditional_density_zero = PDMPSamplers.conditional_density_zero
const conditional_logdensity_zero = PDMPSamplers.conditional_logdensity_zero
const gaussian_slab = PDMPSamplers.gaussian_slab
const gaussian_slab! = PDMPSamplers.gaussian_slab!
const log_boundary_density_zero = PDMPSamplers.log_boundary_density_zero
const log_model_add_odds = PDMPSamplers.log_model_add_odds
const sample_unstick_label = PDMPSamplers.sample_unstick_label
const scalar_logscale_gaussian_line_segment = PDMPSamplers.scalar_logscale_gaussian_line_segment
const slab_cache_key = PDMPSamplers.slab_cache_key
const slab_cache_style = PDMPSamplers.slab_cache_style
const thinning_diagnostics = PDMPSamplers.thinning_diagnostics
const unstick_rate_constant = PDMPSamplers.unstick_rate_constant

struct _AlwaysAdaptedAdapter <: PDMPSamplers.AbstractAdapter end
PDMPSamplers.did_dynamics_adapt(::_AlwaysAdaptedAdapter) = true

struct _FutureContinuousDynamics <: ContinuousDynamics end
PDMPSamplers.unstick_rate_constant(::_FutureContinuousDynamics, ::Integer) = 1.0
function PDMPSamplers.move_forward_time!(state::PDMPSamplers.AbstractPDMPState,
        τ::Real, ::_FutureContinuousDynamics)
    state.t[] += τ
    state.ξ.x .+= τ .* state.ξ.θ
    return state
end

struct _DeclaredStickableClock{V} <: PDMPSamplers.AbstractAggregateUnstickClock
    coordinates::V
end
struct _MissingStickableClock <: PDMPSamplers.AbstractAggregateUnstickClock end

Base.copy(clock::_DeclaredStickableClock) = clock
PDMPSamplers.stickable_coordinates(clock::_DeclaredStickableClock) =
    clock.coordinates
PDMPSamplers.sample_time(::Random.AbstractRNG, ::_DeclaredStickableClock,
    ::ContinuousDynamics, ::StickyPDMPState, ::Real, ::BitVector) = Inf
PDMPSamplers.sample_label(::Random.AbstractRNG, clock::_DeclaredStickableClock,
    ::ContinuousDynamics, ::StickyPDMPState, ::BitVector) =
    first(clock.coordinates)

function _active_prior_grad_alloc(provider, out, x, active)
    return @allocated active_prior_grad!(provider, out, x, active)
end

function _conditional_logdensity_zero_alloc(provider, x, active, j)
    return @allocated conditional_logdensity_zero(provider, x, active, j)
end

function _seeded_fourier_sample_allocation(clock, flow, state, horizon,
        can_stick, seed)
    rng = MersenneTwister(seed)
    return @allocated PDMPSamplers.sample_time(
        rng, clock, flow, state, horizon, can_stick)
end

function _scalar_root_transformed_oracle(segment, ylo, yhi)
    return setprecision(BigFloat, 256) do
        log_width = BigFloat(log(segment.s0) + segment.ell0 +
            segment.r * (-segment.a / segment.b) - log(abs(segment.b)))
        q = BigFloat(segment.r) * exp(log_width)
        prefactor = exp(BigFloat(segment.log_total_weight -
            log(abs(segment.b)))) / sqrt(2BigFloat(pi))
        integrand = y -> begin
            z = y * exp(-q * y)
            prefactor * exp(-q * y - z^2 / 2)
        end
        value, _ = PDMPSamplers.QuadGK.quadgk(
            integrand, BigFloat(ylo), BigFloat(yhi);
            rtol=big"1e-40", atol=big"1e-50")
        Float64(value)
    end
end

function _scalar_limiting_profile_oracle(q, ylo=-Inf, yhi=Inf)
    return setprecision(BigFloat, 256) do
        qbig = BigFloat(q)
        integrand = y -> begin
            z = y * exp(-qbig * y)
            exp(-qbig * y - z^2 / 2) / sqrt(2BigFloat(pi))
        end
        value, _ = PDMPSamplers.QuadGK.quadgk(integrand,
            BigFloat(ylo), BigFloat(yhi); rtol=big"1e-40", atol=big"1e-50")
        Float64(value)
    end
end

function _boomerang_bigfloat_cell_oracle(segment, root, lo, hi)
    return setprecision(BigFloat, 256) do
        root_b = BigFloat(root)
        lo_b = BigFloat(lo)
        hi_b = BigFloat(hi)
        log_width = BigFloat(PDMPSamplers._boomerang_root_log_width(
            segment, root))
        width = exp(log_width)
        ulo = asinh((lo_b - root_b) / width)
        uhi = asinh((hi_b - root_b) / width)
        derivative = BigFloat(PDMPSamplers._boomerang_root_derivative(
            segment, root))
        c0 = BigFloat(segment.c0)
        log_prefactor = BigFloat(segment.log_total_weight) + log_width -
            log(sqrt(2BigFloat(pi)))
        integrand = u -> begin
            delta = width * sinh(u)
            mean_value = 2c0 * sin(delta / 2)^2 + derivative * sin(delta)
            t = root_b + delta
            log_scale = BigFloat(segment.log_scale) +
                BigFloat(segment.log_scale_cos) * cos(t) +
                BigFloat(segment.log_scale_sin) * sin(t)
            z = mean_value * exp(-log_scale)
            exp(log_prefactor - log_scale - z^2 / 2) * cosh(u)
        end
        value, _ = PDMPSamplers.QuadGK.quadgk(integrand, ulo, uhi;
            rtol=big"1e-35", atol=big"1e-45")
        return value
    end
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
        bern = BernoulliModelPrior([0.2, 0.5, 1.0, 0.0])
        active = BitVector([false, true, false, false])
        @test log_model_add_odds(bern, active, 1) ≈ log(0.2) - log1p(-0.2)
        @test log_model_add_odds(bern, active, 2) ≈ 0.0
        @test log_model_add_odds(bern, active, 3) == Inf
        @test log_model_add_odds(bern, active, 4) == -Inf

        bb = BetaBernoulliModelPrior(4, 2.0, 3.0)
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
        endpoint_prior_one = BernoulliModelPrior([1.0])
        endpoint_prior_zero = BernoulliModelPrior([0.0])
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
        @test_throws DimensionMismatch active_prior_grad!(provider, out_block, x, active)

        @test_throws DimensionMismatch active_prior_grad!(provider, fill(NaN, 4), x, active)

        all_active = trues(3)
        active_prior_grad!(provider, out, x, all_active)
        expected_all = zeros(length(x))
        expected_all[indices] .= cov \ (x[indices] - mean)
        @test out ≈ expected_all

        # When the state and beta block have equal lengths, the output still
        # follows full-state layout rather than the provider's beta ordering.
        permuted = DenseGaussianSlab(zeros(3), Matrix(I, 3, 3), [3, 1, 2])
        permuted_x = [10.0, 20.0, 30.0]
        permuted_out = zeros(3)
        active_prior_grad!(permuted, permuted_out, permuted_x, trues(3))
        @test permuted_out == permuted_x

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
        @test_throws DimensionMismatch active_prior_grad!(indep, indep_block, x, active)
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
        @test_throws DimensionMismatch active_prior_grad!(logscale, logscale_block, logscale_x, logscale_active)
        @test_throws DimensionMismatch active_prior_grad!(logscale, fill(NaN, 4), logscale_x, logscale_active)
        @test conditional_logdensity_zero(logscale, logscale_x, logscale_active, 3) ≈
              -0.5 * log(2π) - (logscale.log_base_scales[3] + logscale_x[logscale.logscale_indices[3]])

        structured_design = [1.0 0.0; 0.5 0.5; 0.0 1.0]
        structured = LogLinearGaussianScaleSlab(
            indices, [1, 3], log.([2.0, 3.0, 4.0]), structured_design)
        structured_out = zeros(length(logscale_x))
        active_prior_grad!(structured, structured_out, logscale_x, logscale_active)
        expected_structured = zeros(length(logscale_x))
        for j in (1, 2)
            beta = logscale_x[indices[j]]
            ell = structured.log_base_scales[j] +
                dot(structured_design[j, :], logscale_x[[1, 3]])
            z = beta^2 * exp(-2ell)
            expected_structured[indices[j]] += beta * exp(-2ell)
            expected_structured[[1, 3]] .+= structured_design[j, :] .* (1 - z)
        end
        @test structured_out ≈ expected_structured
        @test conditional_logdensity_zero(structured, logscale_x,
            logscale_active, 3) ≈ -0.5 * log(2π) -
                (structured.log_base_scales[3] + logscale_x[3])
        structured_clock = ExponentialSumAggregateClock(
            structured, BernoulliModelPrior(fill(0.5, 3)))
        @test structured_clock isa ExponentialSumAggregateClock
        @test_throws ArgumentError IndependentZeroMeanLogscaleGaussianSlab([1, 2], [2, 3], zeros(2))
        @test_throws ArgumentError GlobalLogscaleExchangeableGaussianSlab([1, 2], 2, 1.0, 0.1)
    end

    @testset "DependentSlabTarget active-set synchronization at state transitions" begin
        d = 3
        posterior_grad!(out, x) = (out .= [x[1], x[2], 0.0]; out)
        slab = DenseGaussianSlab(zeros(2), Matrix(I, 2, 2), [1, 2])
        odds = BernoulliModelPrior([0.5, 0.5])
        target = DependentSlabTarget(d, posterior_grad!, slab, odds)
        model = PDMPModel(target)
        flow = ZigZag(d)
        x = [3.0, -2.0, 1.0]
        θ = [1.0, 0.0, -1.0]
        state = StickyPDMPState(Ref(0.0), SkeletonPoint(copy(x), copy(θ)), BitVector([true, false, true]))
        cache = (; ∇ϕx=zeros(d))

        set_active_set!(model, state.free)
        grad = compute_gradient!(state, model.grad, flow, cache)
        @test grad ≈ [3.0, 0.0, 0.0]
        @test target.free == state.free

        state.free .= BitVector([false, true, true])
        set_active_set!(model, state.free)
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
        one_beta_slab = DenseGaussianSlab([0.0], reshape([1.0], 1, 1), [1])
        nuisance_target = DependentSlabTarget(2, nuisance_posterior_grad!, one_beta_slab, BernoulliModelPrior([0.5]);
            initial_free=BitVector([false, true]))
        nuisance_out = fill(NaN, 2)
        nuisance_target(nuisance_out, [3.0, 4.0])
        @test nuisance_out ≈ [0.0, 8.0]

        strategy_target = DependentSlabTarget(d, FullGradient((out, x) -> copyto!(out, 2 .* x)), slab, odds)
        strategy_out = zeros(d)
        strategy_x = [1.0, -2.0, 3.0]
        strategy_target(strategy_out, strategy_x)
        @test strategy_out ≈ [2.0, -4.0, 6.0]

        hvp_target = DependentSlabTarget(d, posterior_grad!, slab, odds; initial_free=BitVector([true, false, true]))
        hvp_model = PDMPModel(hvp_target; hvp=true)
        hvp_x = [3.0, -2.0, 1.0]
        hvp_v = [0.7, -0.4, 0.2]
        hvp_w = [1.5, 2.0, -0.3]
        @test hvp_model.hvp(hvp_x, hvp_v) ≈ [0.7, 0.0, 0.0] atol=1e-6
        @test hvp_model.vhv(hvp_x, hvp_v, hvp_w) ≈ 1.05 atol=1e-6

        set_active_set!(hvp_model, BitVector([false, true, true]))
        @test hvp_model.hvp(hvp_x, hvp_v) ≈ [0.0, -0.4, 0.0] atol=1e-6

        # Sticky event handling must synchronize every independently copied
        # target in PDMPModel, not only the gradient target.
        transition_slab = DenseGaussianSlab(zeros(2), Matrix(I, 2, 2), 1:2)
        transition_target = DependentSlabTarget(
            2, (out, x) -> copyto!(out, x),
            transition_slab, BernoulliModelPrior(fill(0.5, 2));
            initial_free=trues(2),
        )
        transition_model = PDMPModel(transition_target; hvp=true)
        transition_flow = ZigZag(2)
        transition_alg = Sticky(GridThinningStrategy(), fill(0.5, 2))
        transition_state, transition_model_, transition_alg_, transition_cache, transition_stats =
            PDMPSamplers.initialize_state(
                MersenneTwister(812),
                transition_flow,
                transition_model,
                transition_alg,
                0.0,
                SkeletonPoint(zeros(2), ones(2)),
            )
        direction = ones(2)
        @test transition_model_.hvp(zeros(2), direction) ≈ ones(2) atol=1e-6
        PDMPSamplers._handle_event_no_boundary!(
            MersenneTwister(813),
            0.0,
            transition_model_,
            transition_flow,
            transition_alg_,
            transition_state,
            transition_cache,
            :sticky,
            CoordinateMeta(1),
            transition_stats,
        )
        @test transition_state.free == BitVector([false, true])
        @test transition_model_.hvp(zeros(2), direction) ≈ [0.0, 1.0] atol=1e-6
        @test transition_model_.vhv(zeros(2), direction, direction) ≈ 1.0 atol=1e-6

        hvp_trace, _ = pdmp_sample(SkeletonPoint([0.1, -0.2, 0.3], [0.4, 0.2, -0.1]), Boomerang(d, 0.0), hvp_model, GridThinningStrategy(), 0.0, 0.05)
        @test !isempty(hvp_trace.times)
    end

    @testset "Arbitrary slab target baseline" begin
        d = 3
        posterior_grad!(out, x) = (copyto!(out, x); out)
        provider = ArbitrarySlabBoundary([1, 3];
            active_prior_neggrad! = (out, x, active) -> (fill!(out, 0.0); active[1] && (out[1] = 2x[1]); active[2] && (out[3] = 3x[3]); out),
            log_q_zero! = (x, active, j) -> logpdf(Normal(), 0.0),
        )
        target = DependentSlabTarget(d, posterior_grad!, provider, BernoulliModelPrior(fill(0.5, 2));
            initial_free=BitVector([true, false, false]))
        out = fill(NaN, d)
        x = [1.0, 2.0, -1.0]
        target(out, x)
        @test out ≈ [1.0, 2.0, 2.0]
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
        odds = BernoulliModelPrior(fill(0.5, d))
        clock = SummedRateClock(provider, odds)
        flow = ZigZag(d)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint(zeros(d), zeros(d)),
            falses(d),
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
        odds = BernoulliModelPrior([0.25, 0.5, 0.75, 0.8])

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
            active_prior_grad! = (out, x, active) -> begin
                fill!(out, 0.0)
                out[indices] .= active .* x[indices]
                out
            end,
        )
        cb_out = fill(NaN, length(x))
        @test active_prior_grad!(cb_grad, cb_out, x, active) === cb_out
        expected_cb = zeros(length(x))
        expected_cb[indices] .= active .* x[indices]
        @test cb_out ≈ expected_cb
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
        odds = BernoulliModelPrior(fill(0.4, 4))

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
        odds = BernoulliModelPrior([0.5, 0.35, 0.6, 0.8])
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
        const_clock = LinearGaussianAggregateClock(const_provider, BernoulliModelPrior(fill(0.5, 2)))
        const_flow = ZigZag(2)
        const_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint(zeros(2), zeros(2)),
            falses(2),
        )
        const_can_stick = trues(2)
        λ0 = PDMPSamplers.rate(const_clock, const_flow, const_state, 0.0, const_can_stick)
        rng_time = MersenneTwister(99)
        rng_ref = MersenneTwister(99)
        τ = PDMPSamplers.sample_time(rng_time, const_clock, const_flow, const_state, Inf, const_can_stick)
        @test τ ≈ rand(rng_ref, Exponential()) / λ0 rtol=1e-7 atol=1e-8

        tiny_prior = BernoulliModelPrior(fill(1e-20, 2))
        tiny_linear = LinearGaussianAggregateClock(const_provider, tiny_prior)
        tiny_summed = SummedRateClock(const_provider, tiny_prior)
        @test PDMPSamplers.sample_time(
            MersenneTwister(199), tiny_linear, const_flow, const_state, Inf,
            const_can_stick) ≈ PDMPSamplers.sample_time(
                MersenneTwister(199), tiny_summed, const_flow, const_state, Inf,
                const_can_stick) rtol=1e-7

        κ = [0.25, 0.5, 0.75]
        indep_provider = IndependentZeroMeanGaussianSlab(κ, 1:3)
        indep_dense = DenseGaussianSlab(zeros(3), Matrix(Diagonal(@. inv(2π * κ^2))), 1:3)
        indep_odds = BernoulliModelPrior([0.3, 0.5, 0.8])
        indep_clock = LinearGaussianAggregateClock(indep_provider, indep_odds)
        indep_dense_clock = LinearGaussianAggregateClock(indep_dense, indep_odds)
        indep_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([1.0, 0.0, -0.5], [0.2, 0.0, -0.3]),
            BitVector([true, false, false]),
        )
        indep_can_stick = BitVector([true, true, false])
        @test PDMPSamplers.rate(indep_clock, flow, indep_state, 0.0, indep_can_stick) ≈
              PDMPSamplers.rate(indep_dense_clock, flow, indep_state, 0.0, indep_can_stick)
        @test PDMPSamplers.rate(indep_clock, flow, indep_state, 1.5, indep_can_stick) ≈
              PDMPSamplers.rate(indep_clock, flow, indep_state, 0.0, indep_can_stick)
        λ_indep = PDMPSamplers.rate(indep_clock, flow, indep_state, 0.0, indep_can_stick)
        @test PDMPSamplers.sample_time(MersenneTwister(200), indep_clock, flow, indep_state, Inf, indep_can_stick) ≈
              rand(MersenneTwister(200), Exponential()) / λ_indep rtol=1e-7 atol=1e-8

        endpoint_linear = LinearGaussianAggregateClock(DenseGaussianSlab([0.0], reshape([1.0], 1, 1), [1]), BernoulliModelPrior([1.0]))
        endpoint_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0], [0.0]), falses(1))
        @test PDMPSamplers.rate(endpoint_linear, flow, endpoint_state, 0.0, trues(1)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(203), endpoint_linear, flow, endpoint_state, 1.0, trues(1)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(204), endpoint_linear, flow, endpoint_state, trues(1)) == 1
        endpoint_underflow = LinearGaussianAggregateClock(DenseGaussianSlab([1000.0], reshape([1.0], 1, 1), [1]), BernoulliModelPrior([1.0]))
        @test PDMPSamplers.rate(endpoint_underflow, flow, endpoint_state, 0.0, trues(1)) == Inf
        @test PDMPSamplers.cumulative_hazard(endpoint_underflow, flow, endpoint_state, 0.0, 1.0, trues(1)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(209), endpoint_underflow, flow, endpoint_state, 1.0, trues(1)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(210), endpoint_underflow, flow, endpoint_state, trues(1)) == 1

        μ = 0.3
        u = 1.4
        v = 0.25
        exch_provider = ExchangeableGaussianSlab(1:4, μ, u, v)
        dense_exch_provider = DenseGaussianSlab(fill(μ, 4), Matrix{Float64}(I, 4, 4) .* u .+ fill(v, 4, 4), 1:4)
        nonexchangeable_odds = BernoulliModelPrior([0.2, 0.5, 0.7, 0.8])
        exch_clock = LinearGaussianAggregateClock(exch_provider, nonexchangeable_odds)
        dense_exch_clock = LinearGaussianAggregateClock(dense_exch_provider, nonexchangeable_odds)
        exch_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.8, 0.0, -0.2, 0.0], [1.1, 0.0, -0.7, 0.0]),
            BitVector([true, false, true, false]),
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
        )
        zero_slope_clock = LinearGaussianAggregateClock(exch_provider, BernoulliModelPrior(fill(0.5, 4)))
        λ_const = PDMPSamplers.rate(zero_slope_clock, flow, zero_slope_state, 0.0, exch_can_stick)
        @test PDMPSamplers.rate(zero_slope_clock, flow, zero_slope_state, 1.0, exch_can_stick) ≈ λ_const
        @test PDMPSamplers.cumulative_hazard(zero_slope_clock, flow, zero_slope_state, 0.0, 1.3, exch_can_stick) ≈ 1.3 * λ_const

        bb_clock = LinearGaussianAggregateClock(exch_provider, BetaBernoulliModelPrior(4, 2.0, 3.0))
        bb_labels = [PDMPSamplers.sample_label(MersenneTwister(seed), bb_clock, flow, exch_state, exch_can_stick) for seed in 1:200]
        @test all(==(2), bb_labels)

        subset_all_inactive = BitVector([false, true, false, true])
        subset_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint(zeros(4), zeros(4)),
            falses(4),
        )
        size_prior_subset = ExchangeableModelSizePrior(log.([0.2, 0.3, 0.25, 0.15, 0.1]))
        subset_clock = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:4, u, v), size_prior_subset)
        full_rate = PDMPSamplers.rate(subset_clock, flow, subset_state, 0.0, trues(4))
        subset_rate = PDMPSamplers.rate(subset_clock, flow, subset_state, 0.0, subset_all_inactive)
        @test subset_rate ≈ full_rate * count(subset_all_inactive) / 4

        endpoint_exch = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:2, 1.0, 0.1), BernoulliModelPrior([1.0, 0.0]))
        endpoint_exch_state = StickyPDMPState(Ref(0.0), SkeletonPoint(zeros(2), zeros(2)), falses(2))
        @test PDMPSamplers.rate(endpoint_exch, flow, endpoint_exch_state, 0.0, trues(2)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(205), endpoint_exch, flow, endpoint_exch_state, 1.0, trues(2)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(206), endpoint_exch, flow, endpoint_exch_state, trues(2)) == 1
        endpoint_exch_second = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:2, 1.0, 0.1), BernoulliModelPrior([0.0, 1.0]))
        @test PDMPSamplers.rate(endpoint_exch_second, flow, endpoint_exch_state, 0.0, trues(2)) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(207), endpoint_exch_second, flow, endpoint_exch_state, 1.0, trues(2)) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(208), endpoint_exch_second, flow, endpoint_exch_state, trues(2)) == 2

        independent_exch = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:4, u, 0.0), BernoulliModelPrior(fill(0.5, 4)))
        independent_dense = LinearGaussianAggregateClock(DenseGaussianSlab(zeros(4), Matrix{Float64}(I, 4, 4) .* u, 1:4), BernoulliModelPrior(fill(0.5, 4)))
        @test PDMPSamplers.rate(independent_exch, flow, exch_state, 0.4, trues(4)) ≈
              PDMPSamplers.rate(independent_dense, flow, exch_state, 0.4, trues(4))
    end

    @testset "Independent logscale exponential-sum clock" begin
        provider = IndependentZeroMeanLogscaleGaussianSlab([1, 2, 3], [4, 4, 5], log.([2.0, 3.0, 4.0]))
        odds = BernoulliModelPrior([0.25, 0.5, 0.75])
        clock = ExponentialSumAggregateClock(provider, odds)
        flow = ZigZag(5)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.2, -0.1], [0.0, 0.0, 0.0, 0.3, -0.2]),
            falses(5),
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
        @test clock_copy.model_prior !== clock.model_prior
        @test clock_copy.rtol == clock.rtol
        @test clock_copy.atol == clock.atol
    end

    @testset "Global logscale exchangeable scalar segment" begin
        provider = GlobalLogscaleExchangeableGaussianSlab(1:3, 4, 1.2, 0.3; mean=0.0, logscale_offset=0.1)
        odds = BernoulliModelPrior(fill(0.5, 3))
        @test default_aggregate_unstick_clock(provider, odds) isa ChebyshevResidualAggregateClock
        flow = ZigZag(4)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.7, 0.0, -0.2, 0.4], [1.0, 0.0, -0.5, 0.25]),
            BitVector([true, false, true, true]),
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

    @testset "Extreme global-logscale boundary rates stay in the extended reals" begin
        prior = BernoulliModelPrior([0.5])
        can_stick = BitVector([true, false])
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0], [0.0, 0.0]),
            falses(2),
        )
        active_beta = BitVector([false])

        for offset in (-1000.0, 1000.0), conditional_mean in (0.0, 1.0)
            provider = GlobalLogscaleExchangeableGaussianSlab(
                1:1, 2, 1.0, 0.0;
                mean=conditional_mean, logscale_offset=offset)
            logdensity = conditional_logdensity_zero(
                provider, state.ξ.x, active_beta, 1)
            @test !isnan(logdensity)
            for inclusion_probability in (0.0, 1.0)
                endpoint_weights = fill(NaN, 1)
                boundary_logweights!(endpoint_weights, provider,
                    BernoulliModelPrior([inclusion_probability]), state.ξ.x,
                    active_beta, BitVector([true]))
                @test !isnan(only(endpoint_weights))
            end

            for flow in (ZigZag(2), BouncyParticle(2))
                clock = PDMPSamplers.ChebyshevResidualAggregateClock(
                    provider, prior; order=8, max_cells=8,
                    residual_budget=0.0, allow_slow_fallback=false)
                segment = scalar_logscale_gaussian_line_segment(
                    provider, prior, flow, state, can_stick, 1.0)
                direct_rate = PDMPSamplers._scalar_logscale_gaussian_line_rate(
                    segment, 0.0)
                direct_upper = PDMPSamplers._scalar_logscale_gaussian_line_upper(
                    segment, 0.0, 1.0)
                @test !isnan(direct_rate)
                @test !isnan(direct_upper)
                @test direct_upper >= direct_rate
                @test !isnan(PDMPSamplers.rate(
                    clock, flow, state, 0.0, can_stick))
                for horizon in (1.0, Inf)
                    sampled = PDMPSamplers.sample_time(MersenneTwister(701),
                        clock, flow, state, horizon, can_stick)
                    @test !isnan(sampled)
                    if offset < 0 && iszero(conditional_mean)
                        @test sampled == 0.0
                    else
                        @test sampled == Inf
                    end
                end
            end

            flow = Boomerang(2)
            clock = PDMPSamplers.FourierResidualAggregateClock(
                provider, prior; order=8, cells=8,
                residual_budget=0.0, allow_slow_fallback=true)
            direct_rate = PDMPSamplers._residual_target_rate(
                clock, flow, state, can_stick, 0.0)
            direct_upper = PDMPSamplers.boundary_rate_upper(
                clock, flow, state, can_stick, 0.0, 1.0)
            @test !isnan(direct_rate)
            @test !isnan(direct_upper)
            @test direct_upper >= direct_rate
            for horizon in (1.0, Inf)
                sampled = PDMPSamplers.sample_time(MersenneTwister(702),
                    clock, flow, state, horizon, can_stick)
                @test !isnan(sampled)
                if offset < 0 && iszero(conditional_mean)
                    @test sampled == 0.0
                else
                    @test sampled == Inf
                end
            end
        end
    end

    @testset "Extreme moving-mean global-logscale hazards" begin
        prior = BernoulliModelPrior([0.5, 0.5])
        can_stick = BitVector([true, true, false])
        crossing_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            BitVector([true, false, true]),
        )
        for offset in (-1000.0, 1000.0), flow in (ZigZag(3), BouncyParticle(3))
            provider = GlobalLogscaleExchangeableGaussianSlab(
                1:2, 3, 1.0, 1.0; mean=0.0, logscale_offset=offset)
            clock = ChebyshevResidualAggregateClock(provider, prior;
                order=8, max_cells=8, residual_budget=0.0,
                allow_slow_fallback=false)
            segment = scalar_logscale_gaussian_line_segment(
                provider, prior, flow, crossing_state, can_stick, Inf)
            hazards = [PDMPSamplers._scalar_logscale_gaussian_line_cumulative_hazard(
                segment, T) for T in (0.5, 1.0, 2.0)]
            @test all(isfinite, hazards)
            @test issorted(hazards)
            @test hazards[1] == 0.0
            if offset < 0
                oracle_mass = exp(segment.log_total_weight) / abs(segment.b)
                @test hazards[2] ≈ 0.5oracle_mass rtol=1e-14
                @test hazards[3] ≈ oracle_mass rtol=1e-14
                @test PDMPSamplers.sample_time(MersenneTwister(1), clock,
                    flow, crossing_state, 2.0, can_stick) == 1.0
                @test PDMPSamplers.sample_time(MersenneTwister(1), clock,
                    flow, crossing_state, Inf, can_stick) == 1.0
            else
                @test hazards == zeros(3)
                @test PDMPSamplers.sample_time(MersenneTwister(1), clock,
                    flow, crossing_state, 2.0, can_stick) == Inf
                @test PDMPSamplers.sample_time(MersenneTwister(1), clock,
                    flow, crossing_state, Inf, can_stick) == Inf
            end
            @test PDMPSamplers.sample_time(MersenneTwister(1), clock,
                flow, crossing_state, 0.5, can_stick) == Inf
        end


        moving_scale_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([-2.0, 0.0, 0.0], [2.0, 0.0, 0.5]),
            BitVector([true, false, true]),
        )
        moving_provider = GlobalLogscaleExchangeableGaussianSlab(
            1:2, 3, 1.0, 1.0; mean=0.0, logscale_offset=-1000.0)
        for flow in (ZigZag(3), BouncyParticle(3))
            moving_clock = ChebyshevResidualAggregateClock(
                moving_provider, prior; order=8, max_cells=8,
                residual_budget=0.0, allow_slow_fallback=false)
            moving_segment = scalar_logscale_gaussian_line_segment(
                moving_provider, prior, flow, moving_scale_state,
                can_stick, 2.0)
            @test PDMPSamplers._scalar_unresolved_peak(moving_segment) !== nothing
            moving_hazards = [PDMPSamplers._scalar_logscale_gaussian_line_cumulative_hazard(
                moving_segment, T) for T in (0.5, 1.0, 2.0)]
            @test all(isfinite, moving_hazards)
            @test issorted(moving_hazards)
            @test moving_hazards[1] == 0.0
            @test moving_hazards[2] ≈ 0.5moving_hazards[3] rtol=1e-14
            @test PDMPSamplers.sample_time(MersenneTwister(1), moving_clock,
                flow, moving_scale_state, 2.0, can_stick) == 1.0
            available = PDMPSamplers._scalar_logscale_gaussian_line_available_hazard(
                moving_segment)
            @test isfinite(available)
            @test available >= moving_hazards[end]
            @test PDMPSamplers.sample_time(MersenneTwister(1), moving_clock,
                flow, moving_scale_state, Inf, can_stick) == 1.0
        end

        for offset in (-20.0, -30.0, -1000.0),
                logscale_velocity in (-1.0, 1.0),
                flow in (ZigZag(3), BouncyParticle(3))
            provider = GlobalLogscaleExchangeableGaussianSlab(
                1:2, 3, 1.0, 1.0; mean=0.0,
                logscale_offset=offset)
            state = StickyPDMPState(
                Ref(0.0),
                SkeletonPoint([-2.0, 0.0, 0.0],
                    [2.0, 0.0, logscale_velocity]),
                BitVector([true, false, true]),
            )
            segment = scalar_logscale_gaussian_line_segment(
                provider, prior, flow, state, can_stick, 2.0)
            clock = ChebyshevResidualAggregateClock(provider, prior;
                order=8, max_cells=8, residual_budget=0.0,
                allow_slow_fallback=false)
            resolved = PDMPSamplers._scalar_peak_window(segment)
            unresolved = PDMPSamplers._scalar_unresolved_peak(segment)
            @test (resolved === nothing) != (unresolved === nothing)
            hazards = [PDMPSamplers._scalar_logscale_gaussian_line_cumulative_hazard(
                segment, T) for T in (0.5, 1.0, 2.0)]
            oracle_mass = exp(segment.log_total_weight) / abs(segment.b)
            oracle_half = _scalar_root_transformed_oracle(
                segment, -20.0, 0.0)
            oracle_full = _scalar_root_transformed_oracle(
                segment, -20.0, 20.0)
            @test hazards[1] == 0.0
            @test hazards[2] ≈ oracle_half rtol=1e-10
            @test hazards[3] ≈ oracle_full rtol=1e-10
            @test oracle_full ≈ oracle_mass rtol=1e-8
            @test issorted(hazards)
            @test all(isfinite, hazards)
            @test PDMPSamplers.sample_time(MersenneTwister(1), clock,
                flow, state, 0.5, can_stick) == Inf
            finite_event = PDMPSamplers.sample_time(MersenneTwister(1),
                clock, flow, state, 2.0, can_stick)
            infinite_event = PDMPSamplers.sample_time(MersenneTwister(1),
                clock, flow, state, Inf, can_stick)
            tolerance = resolved === nothing ? eps(1.0) :
                20resolved.width
            @test abs(finite_event - 1.0) <= tolerance
            @test abs(infinite_event - 1.0) <= tolerance
        end

        provider = GlobalLogscaleExchangeableGaussianSlab(
            1:2, 3, 1.0, 1.0; mean=0.0, logscale_offset=-1000.0)
        flow = Boomerang(3)
        clock = FourierResidualAggregateClock(provider, prior;
            order=8, cells=8, residual_budget=0.0,
            allow_slow_fallback=false)
        peak = PDMPSamplers._boomerang_narrow_peak_segment(
            clock, flow, crossing_state, can_stick)
        @test peak !== nothing
        roots = PDMPSamplers._boomerang_peak_roots(peak)
        @test first(roots) ≈ π / 4
        hazards = [PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            peak, T) for T in (0.5, π / 4, 1.0, 2.0)]
        @test all(isfinite, hazards)
        @test issorted(hazards)
        @test hazards[1] == 0.0
        @test hazards[2] ≈ 0.5hazards[3] rtol=1e-14
        @test hazards[3] == hazards[4]
        env, evaluations = PDMPSamplers.build_residual_envelope(
            clock, flow, crossing_state, 2.0, can_stick)
        @test evaluations == 0
        @test env.Hbar_horizon == hazards[end]
        @test issorted([PDMPSamplers._envelope_hazard(env, T)
            for T in range(0.0, 2.0; length=33)])
        @test !isnan(PDMPSamplers._residual_target_rate(
            clock, flow, crossing_state, can_stick, π / 4))
        @test PDMPSamplers.boundary_rate_upper(
            clock, flow, crossing_state, can_stick, 0.5, 1.0) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(8), clock, flow,
            crossing_state, 0.5, can_stick) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(8), clock, flow,
            crossing_state, 2.0, can_stick) ≈ π / 4
        @test PDMPSamplers.sample_time(MersenneTwister(8), clock, flow,
            crossing_state, Inf, can_stick) ≈ π / 4
    end

    @testset "Unresolved scalar peaks retain scale variation" begin
        for q in (0.0, -0.1, 0.1, -1.0, 1.0)
            segment = PDMPSamplers.ScalarLogscaleGaussianLineSegment(
                -1e16, 1.0, 1.0, -q * 1e16, q, 0.0, Inf)
            peak = PDMPSamplers._scalar_unresolved_peak(segment)
            @test peak !== nothing
            oracle = _scalar_limiting_profile_oracle(q)
            @test peak.mass ≈ oracle rtol=2e-11
            @test PDMPSamplers._scalar_logscale_gaussian_line_available_hazard(
                segment) ≈ oracle rtol=2e-11
            if q == 0.1
                profile_limit = PDMPSamplers._scalar_root_profile_limit(
                    segment, peak.log_width)
                old_bracket_hazard = PDMPSamplers._scalar_root_profile_integral(
                    segment, peak.log_width, -profile_limit, 40.0)
                @test old_bracket_hazard < peak.mass
                tail_seed = findfirst(1:10_000) do seed
                    threshold = rand(MersenneTwister(seed), Exponential())
                    old_bracket_hazard < threshold < peak.mass
                end
                @test tail_seed !== nothing
                @test isfinite(PDMPSamplers._sample_scalar_unresolved_peak(
                    MersenneTwister(tail_seed), segment, Inf))
            end
            for horizon in (2e16, Inf)
                rng = MersenneTwister(1)
                threshold = rand(MersenneTwister(1), Exponential())
                event_time = PDMPSamplers._sample_scalar_unresolved_peak(
                    rng, segment, horizon)
                @test isfinite(event_time)
                @test event_time <= horizon
                lower = PDMPSamplers._scalar_logscale_gaussian_line_cumulative_hazard(
                    segment, prevfloat(event_time))
                upper = PDMPSamplers._scalar_logscale_gaussian_line_cumulative_hazard(
                    segment, nextfloat(event_time))
                @test lower <= threshold <= upper
            end
        end

        for r in (-1.0, 1.0)
            segment = PDMPSamplers.ScalarLogscaleGaussianLineSegment(
                -1.0, 1.0, 1.0, -1000.0, r, 0.0, Inf)
            peak = PDMPSamplers._scalar_unresolved_peak(segment)
            @test peak !== nothing
            @test peak.mass ≈ 1.0 rtol=1e-14
            @test PDMPSamplers._sample_scalar_unresolved_peak(
                MersenneTwister(1), segment, 2.0) == 1.0
            @test PDMPSamplers._sample_scalar_unresolved_peak(
                MersenneTwister(1), segment, Inf) == 1.0
        end
    end

    @testset "Boomerang transverse, tangent, and periodic narrow peaks" begin
        for logscale_sine in (-1.0, 1.0), offset in (-20.0, -30.0, -1000.0)
            moving = PDMPSamplers._BoomerangNarrowPeakSegment(
                0.0, 1.0, 0.0, offset, 0.0, logscale_sine, 0.0)
            moving_roots = PDMPSamplers._boomerang_peak_roots(moving)
            @test length(moving_roots) == 2
            moving_hazards = [
                PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                    moving, T) for T in range(0.0, 2π; length=33)]
            @test issorted(moving_hazards)
            # Each transverse root has limiting mass weight/|m'(root)| = 1.
            @test moving_hazards[end] ≈ 2.0 rtol=2e-10
            for horizon in (2π, Inf)
                threshold = rand(MersenneTwister(1), Exponential())
                event_time = PDMPSamplers._sample_boomerang_narrow_peak(
                    MersenneTwister(1), moving, horizon)
                @test isfinite(event_time)
                lower = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                    moving, prevfloat(event_time))
                upper = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                    moving, nextfloat(event_time))
                @test lower <= threshold <= upper
            end
        end

        transverse = PDMPSamplers._BoomerangNarrowPeakSegment(
            0.0, 1.0, 0.0, log(1e-8), 0.0)
        H_transverse = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            transverse, 2π)
        @test H_transverse ≈ 2.0 rtol=1e-12

        δ = eps(Float64)
        near_tangent = PDMPSamplers._BoomerangNarrowPeakSegment(
            -1.0 + δ, 1.0, 0.0, log(1e-16), log(sqrt(2 / π)))
        near_roots = PDMPSamplers._boomerang_peak_roots(near_tangent)
        @test length(near_roots) == 2
        @test near_roots[2] - near_roots[1] > 2π - 1e-6
        @test !PDMPSamplers._boomerang_all_roots_use_delta(
            near_tangent, near_roots)
        H_near = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            near_tangent, 2π)
        near_oracle = 8.4201005e7
        @test H_near ≈ near_oracle rtol=1e-8

        tangent = PDMPSamplers._BoomerangNarrowPeakSegment(
            -1.0, 1.0, 0.0, log(1e-16), 0.0)
        tangent_roots = PDMPSamplers._boomerang_peak_roots(tangent)
        @test tangent_roots == [0.0]
        @test isfinite(PDMPSamplers._boomerang_peak_mass(tangent, 0.0))
        H_tangent = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            tangent, 2π)
        tangent_oracle = 1.2162802142575204e8
        @test H_tangent ≈ tangent_oracle rtol=1e-8

        moving_tangent = PDMPSamplers._BoomerangNarrowPeakSegment(
            -1.0, 1.0, 0.0, log(1e-16) - 0.5, 0.5, 0.0, 0.0)
        @test PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            moving_tangent, 2π) ≈ tangent_oracle rtol=1e-8

        for segment in (transverse, near_tangent, tangent)
            threshold = rand(MersenneTwister(1), Exponential())
            event_time = PDMPSamplers._sample_boomerang_narrow_peak(
                MersenneTwister(1), segment, 2π)
            @test 0.0 <= event_time <= 2π
            event_hazard = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                segment, event_time)
            @test event_hazard ≈ threshold rtol=2e-8 atol=1e-9
        end

        for mass in (1e-20, 1e-100, 1e-300)
            segment = PDMPSamplers._BoomerangNarrowPeakSegment(
                0.0, 1.0, 0.0, -1000.0, log(mass))
            event_time = PDMPSamplers._sample_boomerang_narrow_peak(
                MersenneTwister(1), segment, Inf)
            @test isfinite(event_time)
            threshold = rand(MersenneTwister(1), Exponential())
            @test PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                segment, event_time) ≈ threshold rtol=1e-14
        end

        tiny = PDMPSamplers._BoomerangNarrowPeakSegment(
            0.0, 1.0, 0.0, -1000.0, log(1e-300))
        huge_horizon = (Float64(typemax(Int)) + 1.0) * (2π)
        huge_hazard = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            tiny, huge_horizon)
        @test huge_hazard ≈ 2e-300 * floor(huge_horizon / (2π))
        @test isfinite(huge_hazard)

        public_provider = GlobalLogscaleExchangeableGaussianSlab(
            1:2, 3, 1.0, 1.0; mean=-2.0 + 2δ,
            logscale_offset=log(1e-16 / sqrt(1.5)))
        public_prior = BernoulliModelPrior([0.5, 0.5])
        public_flow = Boomerang(3)
        public_state = StickyPDMPState(
            Ref(0.0), SkeletonPoint([2.0, 0.0, 0.0], zeros(3)),
            BitVector([true, false, true]))
        public_mask = BitVector([true, true, false])
        public_clock = FourierResidualAggregateClock(public_provider,
            public_prior; allow_slow_fallback=false)
        public_peak = PDMPSamplers._boomerang_narrow_peak_segment(
            public_clock, public_flow, public_state, public_mask)
        @test public_peak !== nothing
        @test PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            public_peak, 2π) ≈ near_oracle rtol=1e-8

        for velocity in (-1.0, 1.0), offset in (-20.0, -1000.0)
            moving_provider = GlobalLogscaleExchangeableGaussianSlab(
                1:2, 3, 1.0, 1.0; mean=0.0,
                logscale_offset=offset)
            moving_clock = FourierResidualAggregateClock(moving_provider,
                public_prior; allow_slow_fallback=false)
            moving_state = StickyPDMPState(Ref(0.0),
                SkeletonPoint([-2.0, 0.0, 0.0], [2.0, 0.0, velocity]),
                BitVector([true, false, true]))
            peak = PDMPSamplers._boomerang_narrow_peak_segment(
                moving_clock, public_flow, moving_state, public_mask)
            @test peak !== nothing
            @test peak.log_scale_sin == velocity
            envelope, _ = PDMPSamplers.build_residual_envelope(
                moving_clock, public_flow, moving_state, 2π, public_mask)
            @test isfinite(envelope.Hbar_horizon)
            @test envelope.Hbar_horizon ≈ 2 / sqrt(π) rtol=2e-10
            PDMPSamplers.sample_time(MersenneTwister(1), moving_clock,
                public_flow, moving_state, 2π, public_mask)
            allocation = @allocated PDMPSamplers.sample_time(
                MersenneTwister(1), moving_clock, public_flow,
                moving_state, 2π, public_mask)
            @test allocation < 200_000
        end
    end

    @testset "Large finite transformed Boomerang peaks use log quadrature" begin
        tiny = nextfloat(0.0)
        provider = GlobalLogscaleExchangeableGaussianSlab(
            1:2, 3, 1.0, 1.0; mean=0.0, logscale_offset=-10.0)
        flow = Boomerang(3)
        state = StickyPDMPState(Ref(0.0),
            SkeletonPoint([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            BitVector([true, false, true]))
        can_stick = BitVector([true, true, false])

        for log_weight_scale in (680.0, 709.0, -log(tiny))
            prior = ExchangeableModelSizePrior(
                [-log_weight_scale, -log_weight_scale, 0.0]; normalize=true)
            clock = FourierResidualAggregateClock(provider, prior;
                allow_slow_fallback=false)
            peak = PDMPSamplers._boomerang_narrow_peak_segment(
                clock, flow, state, can_stick)
            roots = PDMPSamplers._boomerang_peak_roots(peak)
            @test isfinite(peak.log_total_weight)
            @test !PDMPSamplers._boomerang_all_roots_use_delta(peak, roots)
            @test PDMPSamplers.rate(clock, flow, state, 0.0,
                can_stick) == 0.0
            threshold = rand(Xoshiro(1), Exponential())
            event_time = PDMPSamplers.sample_time(Xoshiro(1), clock, flow,
                state, Inf, can_stick)
            @test 0.0 < event_time < first(roots)
            lower = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                peak, prevfloat(event_time))
            upper = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                peak, nextfloat(event_time))
            @test lower <= threshold <= upper
            @test PDMPSamplers.sample_time(Xoshiro(1), clock, flow, state,
                prevfloat(event_time), can_stick) == Inf
            finite_event = PDMPSamplers.sample_time(Xoshiro(1), clock, flow,
                state, nextfloat(event_time), can_stick)
            @test isfinite(finite_event)
            @test finite_event <= nextfloat(event_time)
            hazards = [PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                peak, t) for t in range(0.0, first(roots); length=17)]
            @test issorted(hazards)

            if log_weight_scale == -log(tiny)
                oracle = _boomerang_bigfloat_cell_oracle(
                    peak, first(roots), 0.0, event_time)
                @test BigFloat(lower) <= BigFloat(threshold) <= BigFloat(upper)
                @test BigFloat(PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                    peak, event_time)) ≈ oracle rtol=big"2e-9"
            end
        end

        # The public near-tangent configuration remains on the transformed
        # path and is sampled by the same complete-cell quadrature.
        delta = eps(Float64)
        near_provider = GlobalLogscaleExchangeableGaussianSlab(
            1:2, 3, 1.0, 1.0; mean=-2.0 + 2delta,
            logscale_offset=log(1e-16 / sqrt(1.5)))
        near_clock = FourierResidualAggregateClock(near_provider,
            BernoulliModelPrior([0.5, 0.5]); allow_slow_fallback=false)
        near_state = StickyPDMPState(Ref(0.0),
            SkeletonPoint([2.0, 0.0, 0.0], zeros(3)),
            BitVector([true, false, true]))
        near_peak = PDMPSamplers._boomerang_narrow_peak_segment(
            near_clock, flow, near_state, can_stick)
        @test !PDMPSamplers._boomerang_all_roots_use_delta(
            near_peak, PDMPSamplers._boomerang_peak_roots(near_peak))
        @test isfinite(PDMPSamplers.sample_time(Xoshiro(1), near_clock,
            flow, near_state, Inf, can_stick))
    end

    @testset "Mixed transformed and delta Boomerang roots" begin
        flow = Boomerang(3)
        can_stick = BitVector([true, true, false])
        prior = BernoulliModelPrior([0.5, 0.5])
        first_root = Float64(π / 4)

        function mixed_fixture(logscale_cosine)
            offset = -10.0 - abs(logscale_cosine) * cos(first_root)
            provider = GlobalLogscaleExchangeableGaussianSlab(
                1:2, 3, 1.0, 1.0; mean=0.0,
                logscale_offset=offset)
            state = StickyPDMPState(Ref(0.0),
                SkeletonPoint([-2.0, 0.0, logscale_cosine],
                    [2.0, 0.0, 0.0]),
                BitVector([true, false, true]))
            clock = FourierResidualAggregateClock(provider, prior;
                allow_slow_fallback=false)
            peak = PDMPSamplers._boomerang_narrow_peak_segment(
                clock, flow, state, can_stick)
            roots = PDMPSamplers._boomerang_peak_roots(peak)
            modes = [PDMPSamplers._boomerang_root_uses_delta(
                peak, roots, i) for i in eachindex(roots)]
            return clock, state, peak, roots, modes
        end

        for coefficient in (-1000.0, -500.0, -100.0,
                100.0, 500.0, 1000.0)
            clock, state, peak, roots, modes = mixed_fixture(coefficient)
            @test roots ≈ [π / 4, 5π / 4]
            @test modes == (coefficient < 0 ? [true, false] :
                [false, true])

            transformed_index = findfirst(!, modes)
            delta_index = findfirst(identity, modes)
            transformed_root = roots[transformed_index]
            delta_root = roots[delta_index]
            previous = transformed_index == 1 ? roots[end] - 2π :
                roots[transformed_index - 1]
            following = transformed_index == length(roots) ?
                roots[1] + 2π : roots[transformed_index + 1]
            cell_lo = 0.5(previous + transformed_root)
            cell_hi = 0.5(transformed_root + following)
            transformed_mass = PDMPSamplers._boomerang_transformed_cell_hazard(
                peak, transformed_root, cell_lo, cell_hi)
            oracle = _boomerang_bigfloat_cell_oracle(
                peak, transformed_root, cell_lo, cell_hi)
            @test BigFloat(transformed_mass) ≈ oracle rtol=big"2e-9"

            # Pick an explicit target on the leading continuous flank.
            flank_lo = max(0.0, cell_lo)
            H_flank_lo = PDMPSamplers._boomerang_partial_transformed_hazard(
                peak, roots, flank_lo)
            H_flank_hi = PDMPSamplers._boomerang_partial_transformed_hazard(
                peak, roots, prevfloat(transformed_root))
            flank_threshold = 0.5(H_flank_lo + H_flank_hi)
            flank_event = PDMPSamplers._boomerang_mixed_phase_event(
                peak, roots, flank_threshold, 2π)
            @test flank_lo < flank_event < transformed_root

            # The atomic root jumps at the root itself, never at its cell edge.
            H_before = PDMPSamplers._boomerang_partial_transformed_hazard(
                peak, roots, prevfloat(delta_root))
            H_at = PDMPSamplers._boomerang_partial_transformed_hazard(
                peak, roots, delta_root)
            H_after = PDMPSamplers._boomerang_partial_transformed_hazard(
                peak, roots, nextfloat(delta_root))
            delta_mass = PDMPSamplers._boomerang_peak_mass(peak, delta_root)
            @test H_at - H_before ≈ 0.5delta_mass rtol=1e-12
            @test H_after - H_at ≈ 0.5delta_mass rtol=1e-12
            atomic_threshold = H_before + 0.25delta_mass
            @test PDMPSamplers._boomerang_mixed_phase_event(
                peak, roots, atomic_threshold, 2π) == delta_root

            seed_lower = findfirst(1:10_000) do seed
                H_before < rand(Xoshiro(seed), Exponential()) < H_at
            end
            seed_upper = findfirst(1:10_000) do seed
                H_at < rand(Xoshiro(seed), Exponential()) < H_after
            end
            @test seed_lower !== nothing
            @test seed_upper !== nothing
            @test PDMPSamplers.sample_time(Xoshiro(seed_lower), clock,
                flow, state, prevfloat(delta_root), can_stick) == Inf
            @test PDMPSamplers.sample_time(Xoshiro(seed_lower), clock,
                flow, state, delta_root, can_stick) == delta_root
            @test PDMPSamplers.sample_time(Xoshiro(seed_upper), clock,
                flow, state, delta_root, can_stick) == Inf
            @test PDMPSamplers.sample_time(Xoshiro(seed_upper), clock,
                flow, state, nextfloat(delta_root), can_stick) == delta_root

            period_hazard = PDMPSamplers._boomerang_partial_transformed_hazard(
                peak, roots, 2π)
            later_threshold = period_hazard + flank_threshold
            later_event = 2π + flank_event
            lower = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                peak, prevfloat(later_event))
            upper = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                peak, nextfloat(later_event))
            @test lower <= later_threshold <= upper

            times = sort!(unique!(vcat(collect(range(0.0, 2π; length=65)),
                [prevfloat(root) for root in roots], roots,
                [nextfloat(root) for root in roots])))
            hazards = [PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                peak, time) for time in times]
            @test all(diff(hazards) .>=
                -1e-12 * max(1.0, maximum(hazards)))

            for seed in (1, 30, 80)
                threshold = rand(Xoshiro(seed), Exponential())
                event = PDMPSamplers.sample_time(Xoshiro(seed), clock,
                    flow, state, Inf, can_stick)
                @test event != 0.5(roots[1] + roots[2])
                @test PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                    peak, prevfloat(event)) <= threshold <=
                    PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                        peak, nextfloat(event))
            end
        end

        # Exact public reproducer: seed 30 belongs to the second, atomic peak.
        clock, state, peak, roots, modes = mixed_fixture(1000.0)
        @test modes == [false, true]
        @test PDMPSamplers.sample_time(Xoshiro(30), clock, flow, state,
            Inf, can_stick) == roots[2]

        # The inverse asinh map must retain subnormal, nonzero displacements.
        smallest = nextfloat(0.0)
        @test PDMPSamplers._boomerang_delta_from_asinh_coordinate(
            asinh(1.0), log(smallest)) == smallest
    end

    @testset "Tiny Boomerang period masses use stable inverse arithmetic" begin
        function tiny_period_fixture(probability; transformed::Bool)
            if transformed
                delta = eps(Float64)
                provider = GlobalLogscaleExchangeableGaussianSlab(
                    1:2, 3, 1.0, 1.0;
                    mean=-2.0 + 2delta,
                    logscale_offset=log(1e-16 / sqrt(1.5)))
                state = StickyPDMPState(Ref(0.0),
                    SkeletonPoint([2.0, 0.0, 0.0], zeros(3)),
                    BitVector([true, false, true]))
            else
                provider = GlobalLogscaleExchangeableGaussianSlab(
                    1:2, 3, 1.0, 1.0;
                    mean=0.0, logscale_offset=-1000.0)
                state = StickyPDMPState(Ref(0.0),
                    SkeletonPoint([-2.0, 0.0, 0.0], [2.0, 0.0, 1.0]),
                    BitVector([true, false, true]))
            end
            clock = FourierResidualAggregateClock(provider,
                BernoulliModelPrior([0.5, probability]);
                allow_slow_fallback=false)
            flow = Boomerang(3)
            can_stick = BitVector([true, true, false])
            segment = PDMPSamplers._boomerang_narrow_peak_segment(
                clock, flow, state, can_stick)
            roots = PDMPSamplers._boomerang_peak_roots(segment)
            return clock, flow, state, can_stick, segment, roots
        end

        cases = ((false, 1e-20), (false, 1e-100), (false, 1e-300),
            (true, 3e-28), (true, 3e-108), (true, 3e-308))
        for (transformed, probability) in cases
            clock, flow, state, can_stick, segment, roots =
                tiny_period_fixture(probability; transformed)
            uses_delta = PDMPSamplers._boomerang_all_roots_use_delta(
                segment, roots)
            @test uses_delta == !transformed
            period_hazard = uses_delta ?
                sum(root -> PDMPSamplers._boomerang_peak_mass(
                    segment, root), roots) :
                PDMPSamplers._boomerang_partial_transformed_hazard(
                    segment, roots, 2π)
            # Complete-cell integration retains the near-tangent tails that
            # the former fixed normalized-coordinate cutoff discarded.
            @test 1e-301 < period_hazard < 3e-20
            for seed in (1, 43, 80)
                threshold = rand(Xoshiro(seed), Exponential())
                event_time = PDMPSamplers.sample_time(Xoshiro(seed), clock,
                    flow, state, Inf, can_stick)
                @test isfinite(event_time)
                lower = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                    segment, prevfloat(event_time))
                upper = PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
                    segment, nextfloat(event_time))
                @test lower <= threshold <= upper
                @test PDMPSamplers.sample_time(Xoshiro(seed), clock, flow,
                    state, prevfloat(event_time), can_stick) == Inf
                finite_event = PDMPSamplers.sample_time(Xoshiro(seed), clock,
                    flow, state, nextfloat(event_time), can_stick)
                @test isfinite(finite_event)
                @test finite_event <= nextfloat(event_time)
            end
        end
    end

    @testset "Infinite Boomerang model-add odds are immediate" begin
        provider = GlobalLogscaleExchangeableGaussianSlab(
            1:2, 3, 1.0, 1.0; mean=0.0, logscale_offset=-1000.0)
        flow = Boomerang(3)
        can_stick = BitVector([true, true, false])
        for logscale_velocity in (0.0, 1.0)
            state = StickyPDMPState(Ref(0.0),
                SkeletonPoint([-2.0, 0.0, 0.0],
                    [2.0, 0.0, logscale_velocity]),
                BitVector([true, false, true]))
            for horizon in (2.0, Inf)
                zero_clock = FourierResidualAggregateClock(provider,
                    BernoulliModelPrior([0.5, 0.0]);
                    allow_slow_fallback=false)
                @test PDMPSamplers.rate(zero_clock, flow, state, 0.0,
                    can_stick) == 0.0
                @test PDMPSamplers.sample_time(Xoshiro(1), zero_clock,
                    flow, state, horizon, can_stick) == Inf

                infinite_clock = FourierResidualAggregateClock(provider,
                    BernoulliModelPrior([0.5, 1.0]);
                    allow_slow_fallback=false)
                @test PDMPSamplers.rate(infinite_clock, flow, state, 0.0,
                    can_stick) == Inf
                @test PDMPSamplers.sample_time(Xoshiro(1), infinite_clock,
                    flow, state, horizon, can_stick) == 0.0
            end
        end

        # A finite model weight whose local delta mass overflows is not a
        # globally infinite rate: sampling waits for the first peak.
        local_overflow = PDMPSamplers._BoomerangNarrowPeakSegment(
            0.0, 1.0, 0.0, -1000.0, 710.0)
        local_roots = PDMPSamplers._boomerang_peak_roots(local_overflow)
        @test isfinite(local_overflow.log_total_weight)
        @test isinf(PDMPSamplers._boomerang_peak_mass(
            local_overflow, first(local_roots)))
        @test PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            local_overflow, 1.0) == 0.0
        @test PDMPSamplers._boomerang_narrow_peak_cumulative_hazard(
            local_overflow, 2.0) == Inf
        @test PDMPSamplers._sample_boomerang_narrow_peak(
            Xoshiro(1), local_overflow, Inf) ≈ π / 2
    end

    @testset "Global-logscale provider validates and stabilizes scalar inputs" begin
        for (name, value) in ((:mean, NaN), (:mean, Inf),
                (:logscale_offset, NaN), (:logscale_offset, Inf))
            kwargs = name === :mean ? (; mean=value) : (; logscale_offset=value)
            @test_throws ArgumentError GlobalLogscaleExchangeableGaussianSlab(
                1:1, 2, 1.0, 0.0; kwargs...)
        end
        provider = GlobalLogscaleExchangeableGaussianSlab(
            1:1, 2, 1.0, 0.0; mean=1.0,
            logscale_offset=-floatmax(Float64))
        logdensity = conditional_logdensity_zero(provider,
            [0.0, -floatmax(Float64)], BitVector([false]), 1)
        @test logdensity == -Inf
        @test !isnan(logdensity)
    end

    @testset "Default global-logscale clocks preserve exact fallback" begin
        provider = GlobalLogscaleExchangeableGaussianSlab(1:1, 2, 1.0, 0.0)
        prior = BernoulliModelPrior([0.5])
        can_stick = BitVector([true, false])
        model = PDMPModel(2,
            FullGradient((out, x) -> (fill!(out, 0.0); out)), nothing)
        initial = SkeletonPoint([0.0, 0.0], [0.0, 0.0])

        for flow in (Boomerang(2), AdaptiveBoomerang(2))
            clock = default_aggregate_unstick_clock(provider, prior, flow)
            @test clock isa FourierResidualAggregateClock
            @test clock.allow_slow_fallback
            alg = AggregateSticky(GridThinningStrategy(), clock, can_stick)
            _, _, internal, _, _ = PDMPSamplers.initialize_state(
                MersenneTwister(1), flow, model, alg, 0.0, initial)
            @test isfinite(internal.aggregate_unstick_time)
            @test thinning_diagnostics(internal.clock).fallbacks == 1
        end

        no_flow_clock = default_aggregate_unstick_clock(provider, prior)
        @test no_flow_clock isa ChebyshevResidualAggregateClock
        @test no_flow_clock.allow_slow_fallback
        no_flow_alg = AggregateSticky(
            GridThinningStrategy(), no_flow_clock, can_stick)
        _, _, no_flow_internal, _, _ = PDMPSamplers.initialize_state(
            MersenneTwister(1), Boomerang(2), model, no_flow_alg, 0.0, initial)
        @test isfinite(no_flow_internal.aggregate_unstick_time)
        @test thinning_diagnostics(no_flow_internal.clock).fallbacks == 1

        finite_clock = default_aggregate_unstick_clock(
            provider, prior, Boomerang(2))
        finite_state = StickyPDMPState(
            Ref(0.0), initial, BitVector([false, true]))
        PDMPSamplers.sample_time(MersenneTwister(2), finite_clock,
            Boomerang(2), finite_state, 1.0, can_stick)
        @test thinning_diagnostics(finite_clock).fallbacks == 0
        @test thinning_diagnostics(finite_clock).rate_evaluations > 0

        strict_clock = FourierResidualAggregateClock(provider, prior;
            allow_slow_fallback=false)
        @test_throws ArgumentError PDMPSamplers.sample_time(MersenneTwister(3),
            strict_clock, Boomerang(2), finite_state, Inf, can_stick)
    end

    @testset "Residual aggregate clocks delegate to exact fallback" begin
        provider = ArbitrarySlabBoundary(1:2;
            active_prior_neggrad! = (out, x, active) -> (fill!(out, 0.0); out),
            log_q_zero! = (x, active, j) -> logpdf(Normal(0.0, 1.0 + 0.25 * abs(x[j])), 0.0),
        )
        odds = BernoulliModelPrior(fill(0.5, 2))
        summed = SummedRateClock(provider, odds; rtol=1e-8, atol=1e-10)
        cheb = ChebyshevResidualAggregateClock(provider, odds; order=8, max_cells=4, residual_budget=0.0)
        fourier = FourierResidualAggregateClock(provider, odds; order=8, cells=4, residual_budget=0.0)
        fourier_copy = copy(fourier)
        @test fourier_copy !== fourier
        @test fourier_copy.slab_provider !== fourier.slab_provider
        @test fourier_copy.model_prior !== fourier.model_prior
        @test fourier_copy.order == fourier.order
        @test fourier_copy.cells == fourier.cells
        @test default_aggregate_unstick_clock(provider, odds) isa SummedRateClock
        flow = ZigZag(2)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.5, -0.25], [1.0, -1.0]),
            falses(2),
        )
        can_stick = trues(2)

        future_provider = DenseGaussianSlab(zeros(2), Matrix{Float64}(I, 2, 2), 1:2)
        future_summed = SummedRateClock(future_provider, odds)
        future_accelerated_clock = LinearGaussianAggregateClock(future_provider, odds)
        future_flow = _FutureContinuousDynamics()
        @test PDMPSamplers.rate(future_accelerated_clock, future_flow, state,
            0.3, can_stick) ≈ PDMPSamplers.rate(future_summed, future_flow,
            state, 0.3, can_stick)
        @test PDMPSamplers.sample_time(MersenneTwister(120),
            future_accelerated_clock, future_flow, state, 0.5, can_stick) ==
            PDMPSamplers.sample_time(MersenneTwister(120), future_summed,
                future_flow, state, 0.5, can_stick)

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
        @test_throws ArgumentError PDMPSamplers.sample_time(MersenneTwister(16), strict_cheb, flow, state, 1.0, can_stick)
        @test_throws ArgumentError PDMPSamplers.sample_time(MersenneTwister(17), strict_fourier, flow, state, 1.0, can_stick)
        @test PDMPSamplers.thinning_diagnostics(strict_cheb).fallbacks == 0

        scalar_provider = GlobalLogscaleExchangeableGaussianSlab(1:3, 4, 1.1, 0.25; logscale_offset=0.1)
        scalar_odds = BernoulliModelPrior(fill(0.5, 3))
        @test default_aggregate_unstick_clock(scalar_provider, scalar_odds) isa ChebyshevResidualAggregateClock
        @test default_aggregate_unstick_clock(scalar_provider, scalar_odds).allow_slow_fallback
        scalar_clock = ChebyshevResidualAggregateClock(scalar_provider, scalar_odds;
            order=10, max_cells=64, residual_budget=1e-3, allow_slow_fallback=false)
        scalar_flow = ZigZag(4)
        scalar_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.6, 0.0, -0.3, 0.2], [0.8, 0.0, -0.4, 0.15]),
            BitVector([true, false, true, true]),
        )
        scalar_can_stick = BitVector([true, true, true, false])
        scalar_active = BitVector([true, false, true])
        scalar_mean, scalar_cov = gaussian_slab(scalar_provider,
            scalar_state.ξ.x)
        scalar_dense = DenseGaussianSlab(scalar_mean, scalar_cov, 1:3)
        @test conditional_logdensity_zero(scalar_provider,
            scalar_state.ξ.x, scalar_active, 2) ≈
            conditional_logdensity_zero(scalar_dense, scalar_state.ξ.x,
                scalar_active, 2)
        extreme_scalar_x = copy(scalar_state.ξ.x)
        extreme_scalar_x[scalar_provider.logscale_index] = 1e155
        @test !isnan(conditional_logdensity_zero(scalar_provider,
            extreme_scalar_x, scalar_active, 2))
        seg = scalar_logscale_gaussian_line_segment(scalar_provider, scalar_odds, scalar_flow, scalar_state, scalar_can_stick, 1.5)
        env = PDMPSamplers._build_scalar_residual_envelope(scalar_clock, seg)
        for cell in env.cells
            λ_hi = PDMPSamplers._scalar_logscale_gaussian_line_upper(seg, cell.lo, cell.hi)
            q_lo, _ = PDMPSamplers._chebyshev_interval(env.coeffs, cell.lo, cell.hi, seg.horizon)
            @test cell.R + 1e-10 >= max(0.0, λ_hi - q_lo, -q_lo)
        end
        # Use twice the maximum cell count, plus both endpoints, to exercise
        # this invariant without hundreds of duplicate evaluations under
        # coverage instrumentation.
        for t in range(0.0, 1.5; length=129)
            @test PDMPSamplers._envelope_rate(env, t) + 1e-10 >=
                  PDMPSamplers._scalar_logscale_gaussian_line_rate(seg, t)
        end
        τ_scalar = PDMPSamplers.sample_time(MersenneTwister(18), scalar_clock, scalar_flow, scalar_state, 1.5, scalar_can_stick)
        @test τ_scalar == Inf || 0.0 <= τ_scalar <= 1.5
        scalar_diag = thinning_diagnostics(scalar_clock)
        @test scalar_diag.fallbacks == 0
        @test scalar_diag.last_cells <= scalar_clock.max_cells
        @test ismissing(scalar_diag.hazard_acceptance)

        endpoint_scalar_odds = BernoulliModelPrior([1.0, 0.0, 0.0])
        endpoint_scalar_clock = ChebyshevResidualAggregateClock(scalar_provider, endpoint_scalar_odds;
            order=10, max_cells=64, residual_budget=1e-3, allow_slow_fallback=false)
        endpoint_scalar_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.0], zeros(4)),
            falses(4),
        )
        @test PDMPSamplers.rate(endpoint_scalar_clock, scalar_flow, endpoint_scalar_state, 0.1, scalar_can_stick) == Inf
        @test PDMPSamplers.sample_time(MersenneTwister(19), endpoint_scalar_clock, scalar_flow, endpoint_scalar_state, 1.0, scalar_can_stick) == 0.0
        @test PDMPSamplers.sample_label(MersenneTwister(20), endpoint_scalar_clock, scalar_flow, endpoint_scalar_state, scalar_can_stick) == 1

        default_scalar_clock = default_aggregate_unstick_clock(scalar_provider, endpoint_scalar_odds)
        all_frozen_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.15]),
            BitVector([false, false, false, true]),
        )
        @test PDMPSamplers.sample_time(MersenneTwister(21), default_scalar_clock, scalar_flow, all_frozen_state, Inf, scalar_can_stick) == 0.0

        tiny_provider = GlobalLogscaleExchangeableGaussianSlab(1:1, 2, 1.0, 0.0; logscale_offset=100.0)
        tiny_odds = BernoulliModelPrior([0.5])
        tiny_clock = default_aggregate_unstick_clock(tiny_provider, tiny_odds)
        tiny_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0], [0.0, 0.0]),
            falses(2),
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
        )
        @test PDMPSamplers.sample_time(MersenneTwister(23), defective_clock, ZigZag(2), defective_state, Inf, tiny_can_stick) == Inf

        peak_provider = GlobalLogscaleExchangeableGaussianSlab(1:2, 3, 1.0, 1.0)
        peak_clock = default_aggregate_unstick_clock(peak_provider, BernoulliModelPrior(fill(0.5, 2)))
        peak_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([-20.0, 0.0, 0.0], [2.0, 0.0, -1.0]),
            BitVector([true, false, true]),
        )
        peak_can_stick = BitVector([true, true, false])
        peak_seg = scalar_logscale_gaussian_line_segment(peak_provider, BernoulliModelPrior(fill(0.5, 2)), ZigZag(3), peak_state, peak_can_stick, Inf)
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
        @test fourier_diagnostics.last_fallback_provider == nameof(typeof(provider))
        @test fourier_diagnostics.last_fallback_dynamics == nameof(typeof(flow))

        boomerang_provider = DenseGaussianSlab([0.0, 0.1], [1.0 0.25; 0.25 1.4], 1:2)
        boomerang_odds = BernoulliModelPrior(fill(0.5, 2))
        boomerang_summed = SummedRateClock(boomerang_provider, boomerang_odds; rtol=1e-8, atol=1e-10)
        boomerang_fourier = FourierResidualAggregateClock(boomerang_provider, boomerang_odds; order=6, cells=8, residual_budget=1e-3, allow_slow_fallback=false)
        boomerang_flow = Boomerang(inv([1.0 0.35; 0.35 1.6]), zeros(2), 0.0)
        boomerang_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.45], [0.0, 0.7]), BitVector([false, true]))
        boomerang_can_stick = trues(2)
        @test PDMPSamplers.rate(boomerang_fourier, boomerang_flow, boomerang_state, 0.35, boomerang_can_stick) ≈
              PDMPSamplers.rate(boomerang_summed, boomerang_flow, boomerang_state, 0.35, boomerang_can_stick)

        fourier_env, _ = PDMPSamplers._build_fourier_residual_envelope(boomerang_fourier, boomerang_flow, boomerang_state, 1.2, boomerang_can_stick)
        for cell in fourier_env.cells
            λ_hi = PDMPSamplers._boomerang_boundary_rate_upper(boomerang_fourier, boomerang_flow, boomerang_state, boomerang_can_stick, cell.lo, cell.hi)
            q_lo = PDMPSamplers._fourier_lower_on_cell(fourier_env.a0, fourier_env.a, fourier_env.b, cell.lo, cell.hi, fourier_env.horizon)
            @test cell.R + 1e-10 >= max(0.0, λ_hi - q_lo, -q_lo)
        end
        for t in range(0.0, 1.2; length=129)
            @test PDMPSamplers._fourier_envelope_rate(fourier_env, t) + 1e-10 >=
                  PDMPSamplers.rate(boomerang_summed, boomerang_flow, boomerang_state, t, boomerang_can_stick)
        end
        root_location = 0.007
        root_cosine = cos(2π * root_location)
        square_a0 = 0.5 + root_cosine^2
        square_a = [-2root_cosine, 0.5]
        square_lower = PDMPSamplers._fourier_lower_on_cell(
            square_a0, square_a, zeros(2), 0.0, 1.0, 1.0)
        @test square_lower <= 0.0
        τ_boomerang = PDMPSamplers.sample_time(MersenneTwister(24), boomerang_fourier, boomerang_flow, boomerang_state, 1.2, boomerang_can_stick)
        @test τ_boomerang == Inf || 0.0 <= τ_boomerang <= 1.2
        boomerang_fourier_diagnostics = PDMPSamplers.thinning_diagnostics(boomerang_fourier)
        @test boomerang_fourier_diagnostics.fallbacks == 0
        @test boomerang_fourier_diagnostics.rate_evaluations > 0
        @test boomerang_fourier_diagnostics.last_cells == boomerang_fourier.cells
        @test boomerang_fourier_diagnostics.envelope_hazard > 0
        @test boomerang_fourier_diagnostics.min_envelope >= 0
        @test_throws ArgumentError PDMPSamplers.sample_time(MersenneTwister(25), boomerang_fourier, boomerang_flow, boomerang_state, Inf, boomerang_can_stick)

        # Repeated rescheduling reuses all dimension/order/cell-dependent
        # Fourier storage and the active-stratum covariance factorization.
        fourier_workspace = boomerang_fourier.workspace
        workspace_arrays = (fourier_workspace.coeffs, fourier_workspace.aux_coeffs,
            fourier_workspace.cells, fourier_workspace.edges, fourier_workspace.prefix,
            fourier_workspace.boundary.cov_AA, fourier_workspace.boundary.alpha)
        @test all(a === b for (a, b) in zip(workspace_arrays,
            (fourier_workspace.coeffs, fourier_workspace.aux_coeffs,
             fourier_workspace.cells, fourier_workspace.edges,
             fourier_workspace.prefix, fourier_workspace.boundary.cov_AA,
             fourier_workspace.boundary.alpha)))
        @test fourier_workspace.boundary.cache_key == boomerang_state.free

        # Deterministic d=8 rescheduling contract: every measurement recreates
        # the RNG, so all trials follow the identical proposal path.
        allocation_d = 8
        allocation_cov = Matrix{Float64}(I, allocation_d, allocation_d)
        for i in 1:allocation_d-1
            allocation_cov[i, i + 1] = 0.12
            allocation_cov[i + 1, i] = 0.12
        end
        allocation_provider = DenseGaussianSlab(
            collect(range(-0.2, 0.2; length=allocation_d)), allocation_cov,
            1:allocation_d)
        allocation_clock = FourierResidualAggregateClock(allocation_provider,
            BernoulliModelPrior(fill(0.5, allocation_d)); order=8, cells=8,
            residual_budget=1e-3, allow_slow_fallback=false)
        allocation_flow = Boomerang(Matrix{Float64}(I, allocation_d,
            allocation_d), zeros(allocation_d), 0.0)
        allocation_state = StickyPDMPState(Ref(0.0), SkeletonPoint(
            [0.25, -0.4, 0.0, 0.3, 0.0, -0.2, 0.0, 0.0],
            [0.4, -0.3, 0.0, 0.25, 0.0, -0.35, 0.0, 0.0]),
            BitVector([true, true, false, true, false, true, false, false]))
        allocation_mask = trues(allocation_d)
        expected_time = PDMPSamplers.sample_time(MersenneTwister(240),
            allocation_clock, allocation_flow, allocation_state, 1.2,
            allocation_mask)
        @test PDMPSamplers.sample_time(MersenneTwister(240), allocation_clock,
            allocation_flow, allocation_state, 1.2,
            allocation_mask) == expected_time
        allocation_trials = [_seeded_fourier_sample_allocation(
            allocation_clock, allocation_flow, allocation_state, 1.2,
            allocation_mask, 240) for _ in 1:3]
        @test all(==(first(allocation_trials)), allocation_trials)
        @test first(allocation_trials) <= 5_000

        changed_stratum_state = StickyPDMPState(Ref(0.0),
            SkeletonPoint([0.2, 0.0], [0.3, 0.0]), BitVector([true, false]))
        PDMPSamplers._build_fourier_residual_envelope(boomerang_fourier,
            boomerang_flow, changed_stratum_state, 0.8, boomerang_can_stick)
        @test fourier_workspace.boundary.cache_key == changed_stratum_state.free
        @test PDMPSamplers._residual_target_rate(boomerang_fourier,
            boomerang_flow, changed_stratum_state, boomerang_can_stick, 0.3) ≈
            PDMPSamplers.rate(boomerang_summed, boomerang_flow,
                changed_stratum_state, 0.3, boomerang_can_stick)

        no_inactive_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.2, 0.45], [0.3, 0.7]), trues(2))
        PDMPSamplers.reset_thinning_diagnostics!(boomerang_fourier)
        @test PDMPSamplers.sample_time(MersenneTwister(26), boomerang_fourier, boomerang_flow, no_inactive_state, 1.2, boomerang_can_stick) == Inf
        @test PDMPSamplers.thinning_diagnostics(boomerang_fourier).rate_evaluations == 0

        state_at_label = copy(boomerang_state)
        move_forward_time!(state_at_label, 0.4, boomerang_flow)
        @test PDMPSamplers.sample_label(MersenneTwister(27), boomerang_fourier, boomerang_flow, boomerang_state, 0.4, boomerang_can_stick) ==
              PDMPSamplers.sample_label(MersenneTwister(27), boomerang_summed, boomerang_flow, state_at_label, boomerang_can_stick)

        lowrank_fourier_flow = AdaptiveBoomerang(2; λref=0.0, scheme=:lowrank, rank=1)
        lowrank_fourier_flow.Γ.D .= [0.9, 1.7]
        lowrank_fourier_flow.Γ.V .= [0.45; -0.2]
        lowrank_fourier_flow.Γ.Λ .= [0.6]
        PDMPSamplers.lowrank_precompute!(lowrank_fourier_flow.Γ)
        lowrank_fourier = FourierResidualAggregateClock(boomerang_provider, boomerang_odds; order=6, cells=8, residual_budget=1e-3, allow_slow_fallback=false)
        lowrank_fourier_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, -0.35], [0.0, -0.5]), BitVector([false, true]))
        lowrank_env, _ = PDMPSamplers._build_fourier_residual_envelope(lowrank_fourier, lowrank_fourier_flow, lowrank_fourier_state, 0.9, boomerang_can_stick)
        for t in range(0.0, 0.9; length=129)
            @test PDMPSamplers._fourier_envelope_rate(lowrank_env, t) + 1e-10 >=
                  PDMPSamplers.rate(boomerang_summed, lowrank_fourier_flow, lowrank_fourier_state, t, boomerang_can_stick)
        end
        τ_lowrank = PDMPSamplers.sample_time(MersenneTwister(28), lowrank_fourier, lowrank_fourier_flow, lowrank_fourier_state, 0.9, boomerang_can_stick)
        @test τ_lowrank == Inf || 0.0 <= τ_lowrank <= 0.9
        @test PDMPSamplers.thinning_diagnostics(lowrank_fourier).fallbacks == 0

        logscale_boomerang_provider = GlobalLogscaleExchangeableGaussianSlab(1:2, 3, 1.0, 0.2; logscale_offset=0.1)
        logscale_boomerang_odds = BernoulliModelPrior(fill(0.5, 2))
        logscale_boomerang_clock = FourierResidualAggregateClock(logscale_boomerang_provider, logscale_boomerang_odds;
            order=6, cells=8, residual_budget=1e-3, allow_slow_fallback=false)
        logscale_boomerang_summed = SummedRateClock(logscale_boomerang_provider, logscale_boomerang_odds)
        logscale_boomerang_flow = Boomerang(Diagonal([1.0, 1.4, 0.8]), zeros(3), 0.0)
        logscale_boomerang_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.35, 0.2], [0.0, 0.4, 0.3]), BitVector([false, true, true]))
        logscale_boomerang_can_stick = BitVector([true, true, false])
        logscale_env, _ = PDMPSamplers._build_fourier_residual_envelope(
            logscale_boomerang_clock, logscale_boomerang_flow, logscale_boomerang_state, 1.1, logscale_boomerang_can_stick)
        for cell in logscale_env.cells
            λ_hi = PDMPSamplers._boomerang_boundary_rate_upper(
                logscale_boomerang_clock, logscale_boomerang_flow, logscale_boomerang_state, logscale_boomerang_can_stick, cell.lo, cell.hi)
            q_lo = PDMPSamplers._fourier_lower_on_cell(logscale_env.a0, logscale_env.a, logscale_env.b, cell.lo, cell.hi, logscale_env.horizon)
            @test cell.R + 1e-10 >= max(0.0, λ_hi - q_lo, -q_lo)
        end
        for t in range(0.0, 1.1; length=129)
            @test PDMPSamplers._fourier_envelope_rate(logscale_env, t) + 1e-10 >=
                  PDMPSamplers.rate(logscale_boomerang_summed, logscale_boomerang_flow, logscale_boomerang_state, t, logscale_boomerang_can_stick)
        end
        τ_logscale = PDMPSamplers.sample_time(MersenneTwister(29), logscale_boomerang_clock, logscale_boomerang_flow, logscale_boomerang_state, 1.1, logscale_boomerang_can_stick)
        @test τ_logscale == Inf || 0.0 <= τ_logscale <= 1.1
        @test PDMPSamplers.thinning_diagnostics(logscale_boomerang_clock).fallbacks == 0

        @testset "residual capability selection never leaks internal dispatch" begin
            callback_provider = CallbackGaussianSlab(1:2;
                mean_cov! = (mean, cov, x) -> begin
                    fill!(mean, 0.0)
                    fill!(cov, 0.0)
                    cov[1, 1] = exp(0.1x[2])
                    cov[2, 2] = 1.5
                end)
            arbitrary_provider = ArbitrarySlabBoundary(1:2;
                active_prior_neggrad! = (out, x, active) -> fill!(out, 0.0),
                log_q_zero! = (x, active, j) -> logpdf(Normal(0, j), 0.0))
            fallback_state = StickyPDMPState(Ref(0.0),
                SkeletonPoint([0.0, 0.4], zeros(2)), BitVector([false, true]))
            fallback_flows = (
                ZigZag(2),
                BouncyParticle(2, 0.0),
                boomerang_flow,
                AdaptiveBoomerang(2; λref=0.0, scheme=:fullrank),
                PreconditionedZigZag(2; scale=[0.8, 1.2]),
                PreconditionedBPS(2; refresh_rate=0.0, scale=[0.8, 1.2]),
                DensePreconditionedZigZag(2),
                DensePreconditionedBPS(2; refresh_rate=0.0),
                PreconditionedDynamics(DiagonalPreconditioner([0.8, 1.2]),
                    boomerang_flow),
            )

            for unsupported_provider in (callback_provider, arbitrary_provider)
                unsupported_odds = BernoulliModelPrior(fill(0.5, 2))
                summed_clock = SummedRateClock(unsupported_provider,
                    unsupported_odds)
                for fallback_flow in fallback_flows, horizon in (0.4, Inf)
                    fallback_clock = FourierResidualAggregateClock(
                        unsupported_provider, unsupported_odds; allow_slow_fallback=true)
                    τ = PDMPSamplers.sample_time(MersenneTwister(401),
                        fallback_clock, fallback_flow, fallback_state, horizon,
                        trues(2))
                    @test τ == PDMPSamplers.sample_time(MersenneTwister(401),
                        summed_clock, fallback_flow, fallback_state, horizon,
                        trues(2))
                    diag = thinning_diagnostics(fallback_clock)
                    @test diag.fallbacks == 1
                    @test diag.last_fallback_provider == nameof(typeof(unsupported_provider))
                    @test diag.last_fallback_dynamics == nameof(typeof(fallback_flow))

                    strict_clock = FourierResidualAggregateClock(
                        unsupported_provider, unsupported_odds; allow_slow_fallback=false)
                    err = try
                        PDMPSamplers.sample_time(MersenneTwister(402), strict_clock,
                            fallback_flow, fallback_state, horizon, trues(2))
                        nothing
                    catch exception
                        exception
                    end
                    @test err isa ArgumentError
                    @test !(err isa MethodError)
                    @test occursin("no certified residual sampler", sprint(showerror, err))
                end
            end

            for fallback_provider in (boomerang_provider,)
                fallback_odds = BernoulliModelPrior(fill(0.5, 2))
                summed_clock = SummedRateClock(fallback_provider, fallback_odds)
                for fallback_flow in (fallback_flows[1], fallback_flows[2],
                        fallback_flows[5], fallback_flows[6],
                        fallback_flows[7], fallback_flows[8],
                        fallback_flows[9]), horizon in (0.4, Inf)
                    fallback_clock = FourierResidualAggregateClock(
                        fallback_provider, fallback_odds;
                        allow_slow_fallback=true)
                    @test PDMPSamplers.sample_time(MersenneTwister(403),
                        fallback_clock, fallback_flow, fallback_state, horizon,
                        trues(2)) == PDMPSamplers.sample_time(
                            MersenneTwister(403), summed_clock, fallback_flow,
                            fallback_state, horizon, trues(2))
                    @test thinning_diagnostics(fallback_clock).fallbacks == 1

                    strict_clock = FourierResidualAggregateClock(
                        fallback_provider, fallback_odds;
                        allow_slow_fallback=false)
                    @test_throws ArgumentError PDMPSamplers.sample_time(
                        MersenneTwister(404), strict_clock, fallback_flow,
                        fallback_state, horizon, trues(2))
                end
            end


            global_fallback_flows = (
                PreconditionedZigZag(3; scale=[0.8, 1.2, 0.9]),
                PreconditionedBPS(3; refresh_rate=0.0,
                    scale=[0.8, 1.2, 0.9]),
                DensePreconditionedZigZag(3),
                DensePreconditionedBPS(3; refresh_rate=0.0),
                PreconditionedDynamics(DiagonalPreconditioner([0.8, 1.2, 0.9]),
                    logscale_boomerang_flow),
            )
            global_fallback_state = StickyPDMPState(Ref(0.0),
                SkeletonPoint(copy(logscale_boomerang_state.ξ.x), zeros(3)),
                copy(logscale_boomerang_state.free))
            for fallback_flow in global_fallback_flows, horizon in (0.4, Inf)
                fallback_clock = FourierResidualAggregateClock(
                    logscale_boomerang_provider, logscale_boomerang_odds;
                    allow_slow_fallback=true)
                @test PDMPSamplers.sample_time(MersenneTwister(405),
                    fallback_clock, fallback_flow, global_fallback_state,
                    horizon, logscale_boomerang_can_stick) ==
                    PDMPSamplers.sample_time(MersenneTwister(405),
                        logscale_boomerang_summed, fallback_flow,
                        global_fallback_state, horizon,
                        logscale_boomerang_can_stick)
                @test thinning_diagnostics(fallback_clock).fallbacks == 1
            end
        end
    end

    @testset "AggregateSticky requests sticky state" begin
        d = 2
        provider = ArbitrarySlabBoundary(1:d;
            active_prior_neggrad! = (out, x, active) -> (fill!(out, 0.0); out),
            log_q_zero! = (x, active, j) -> logpdf(Normal(), 0.0),
        )
        clock = SummedRateClock(provider, BernoulliModelPrior(fill(0.5, d)))
        alg = AggregateSticky(GridThinningStrategy(), clock, trues(d))
        @test PDMPSamplers.requires_sticky_state(alg)

        posterior_grad!(out, x) = (fill!(out, 0.0); out)
        model = PDMPModel(d, FullGradient(posterior_grad!), nothing)
        ξ = SkeletonPoint([1.0, 2.0], [1.0, -1.0])
        state, _, alg_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(123), ZigZag(d), model, alg, 0.0, ξ)
        @test state isa StickyPDMPState
        @test alg_internal isa PDMPSamplers.AggregateStickyLoopState

        one_beta_provider = DenseGaussianSlab([0.0], reshape([1.0], 1, 1), [1])
        one_beta_clock = SummedRateClock(one_beta_provider, BernoulliModelPrior([0.5]))
        @test stickable_coordinates(one_beta_clock) == [1]
        accepted_subset = AggregateSticky(GridThinningStrategy(), one_beta_clock,
            BitVector([true, false]))
        _, _, accepted_subset_internal, _, _ = PDMPSamplers.initialize_state(
            MersenneTwister(122), ZigZag(2), model, accepted_subset, 0.0, ξ)
        @test accepted_subset_internal.stickable_indices == [1]
        nuisance_trace, _ = pdmp_sample(ξ, ZigZag(2), model, accepted_subset,
            0.0, 0.2; seed=122, progress=false)
        @test all(view(nuisance_trace.free_masks, 2, :))
        invalid_mask = AggregateSticky(GridThinningStrategy(), one_beta_clock,
            BitVector([true, true]))
        invalid_error = try
            PDMPSamplers.initialize_state(
                MersenneTwister(121), ZigZag(2), model, invalid_mask, 0.0, ξ)
            nothing
        catch exception
            exception
        end
        @test invalid_error isa ArgumentError
        @test occursin("unsupported coordinates [2]", sprint(showerror, invalid_error))

        for bad_coordinates in (Any[1.5], [0], [-1], [3], [1, 1],
                [BigInt(typemax(Int)) + 1], [UInt(typemax(Int)) + UInt(1)])
            bad_clock = _DeclaredStickableClock(bad_coordinates)
            bad_alg = AggregateSticky(GridThinningStrategy(), bad_clock,
                BitVector([false, false]))
            error = try
                PDMPSamplers.initialize_state(MersenneTwister(119), ZigZag(2),
                    model, bad_alg, 0.0, ξ)
                nothing
            catch exception
                exception
            end
            @test error isa ArgumentError
            @test occursin("stickable_coordinates", sprint(showerror, error))
        end

        missing_alg = AggregateSticky(GridThinningStrategy(),
            _MissingStickableClock(), BitVector([false, false]))
        missing_error = try
            PDMPSamplers.initialize_state(MersenneTwister(118), ZigZag(2),
                model, missing_alg, 0.0, ξ)
            nothing
        catch exception
            exception
        end
        @test missing_error isa ArgumentError
        @test occursin("must implement PDMPSamplers.stickable_coordinates",
            sprint(showerror, missing_error))

        custom_subset = AggregateSticky(GridThinningStrategy(),
            _DeclaredStickableClock([1, 2]), BitVector([true, false]))
        _, _, custom_subset_internal, _, _ = PDMPSamplers.initialize_state(
            MersenneTwister(117), ZigZag(2), model, custom_subset, 0.0, ξ)
        @test custom_subset_internal.stickable_indices == [1]

        two_beta_subset_clock = SummedRateClock(
            DenseGaussianSlab(zeros(2), Matrix(I, 2, 2), [1, 2]),
            BernoulliModelPrior(fill(0.5, 2)))
        beta_subset = AggregateSticky(GridThinningStrategy(), two_beta_subset_clock,
            BitVector([true, false]))
        _, _, beta_subset_internal, _, _ = PDMPSamplers.initialize_state(
            MersenneTwister(120), ZigZag(2), model, beta_subset, 0.0, ξ)
        @test beta_subset_internal.stickable_indices == [1]

        linear_clock = LinearGaussianAggregateClock(DenseGaussianSlab(zeros(d), Matrix(I, d, d), 1:d), BernoulliModelPrior(fill(0.5, d)))
        linear_alg = AggregateSticky(GridThinningStrategy(), linear_clock, trues(d))
        _, _, linear_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(124), ZigZag(d), model, linear_alg, 0.0, ξ)
        @test linear_internal.clock !== linear_clock
        @test linear_internal.clock.cache !== linear_clock.cache

        scalar_provider = GlobalLogscaleExchangeableGaussianSlab(1:2, 3, 1.0, 0.1)
        scalar_clock = default_aggregate_unstick_clock(scalar_provider, BernoulliModelPrior([0.0, 1.0]))
        scalar_prior = BernoulliModelPrior(fill(0.5, 2))
        @test default_aggregate_unstick_clock(
            scalar_provider, scalar_prior, PreconditionedZigZag(3)) isa SummedRateClock
        @test default_aggregate_unstick_clock(
            scalar_provider, scalar_prior, PreconditionedBPS(3)) isa SummedRateClock
        @test default_aggregate_unstick_clock(
            scalar_provider, scalar_prior,
            PreconditionedDynamics(DiagonalPreconditioner(ones(3)), Boomerang(3))) isa SummedRateClock

        tiny_clock = SummedRateClock(
            DenseGaussianSlab([0.0], reshape([1.0], 1, 1), 1:1),
            BernoulliModelPrior([1e-20]))
        tiny_state = StickyPDMPState(
            Ref(0.0), SkeletonPoint([0.0], [0.0]), falses(1))
        @test isfinite(PDMPSamplers.sample_time(
            MersenneTwister(202), tiny_clock, ZigZag(1), tiny_state, Inf, trues(1)))
        scalar_alg = AggregateSticky(GridThinningStrategy(), scalar_clock, BitVector([true, true, false]))
        scalar_model = PDMPModel(3, FullGradient((out, x) -> (fill!(out, 0.0); out)), nothing)
        all_frozen_ξ = SkeletonPoint([0.0, 0.0, 0.2], [0.0, 0.0, 0.15])
        _, _, all_frozen_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(130), ZigZag(3), scalar_model, scalar_alg, 0.0, all_frozen_ξ)
        @test all_frozen_internal.aggregate_unstick_time == 0.0
        moving_away_ξ = SkeletonPoint([1.0, 0.0, 0.2], [1.0, 0.0, 0.15])
        _, _, moving_away_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(131), ZigZag(3), scalar_model, scalar_alg, 0.0, moving_away_ξ)
        @test moving_away_internal.aggregate_unstick_time == 0.0
        endpoint_alg = AggregateSticky(
            GridThinningStrategy(),
            default_aggregate_unstick_clock(
                scalar_provider, BernoulliModelPrior([0.5, 1.0])),
            BitVector([true, true, false]))
        endpoint_trace, _ = pdmp_sample(
            all_frozen_ξ, ZigZag(3), scalar_model, endpoint_alg, 0.0, 0.1;
            progress=false)
        @test last_event_time(endpoint_trace) == 0.1

        defective_provider2 = GlobalLogscaleExchangeableGaussianSlab(1:1, 2, 1.0, 0.0; logscale_offset=10.0)
        defective_clock2 = default_aggregate_unstick_clock(defective_provider2, BernoulliModelPrior([0.5]))
        defective_alg = AggregateSticky(GridThinningStrategy(), defective_clock2, BitVector([true, false]))
        defective_model = PDMPModel(2, FullGradient((out, x) -> (fill!(out, 0.0); out)), nothing)
        defective_ξ = SkeletonPoint([0.0, 0.0], [0.0, 1.0])
        defective_state2, _, defective_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(132), ZigZag(2), defective_model, defective_alg, 0.0, defective_ξ)
        @test defective_internal.aggregate_unstick_time == Inf
        PDMPSamplers.reschedule_aggregate_unstick_time!(MersenneTwister(133), defective_internal, defective_state2, ZigZag(2))
        @test defective_internal.aggregate_unstick_time == Inf
        defective_trace, defective_stats = pdmp_sample(defective_ξ, ZigZag(2), defective_model, defective_alg, 0.0, 1.0; progress=false)
        @test isfinite(last_event_time(defective_trace))
        @test length(defective_trace) == 2
        @test last_event_time(defective_trace) == 1.0
        @test defective_stats.stop_reason == :reached_time

        boomerang = Boomerang(Diagonal([4.0, 1.0]), zeros(d), 0.0)
        _, _, boomerang_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(126), boomerang, model, alg, 0.0, ξ)
        @test boomerang_internal isa PDMPSamplers.AggregateStickyLoopState
        adaptive_boomerang = AdaptiveBoomerang(d; λref=0.0, scheme=:diagonal)
        _, _, adaptive_boomerang_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(126), adaptive_boomerang, model, alg, 0.0, ξ)
        @test adaptive_boomerang_internal isa PDMPSamplers.AggregateStickyLoopState
        @test PDMPSamplers.unstick_rate_constant(boomerang, 1) ≈ sqrt(2 / π) / 2
        inactive_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.1]), BitVector([false, true]))
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(128), inactive_state, boomerang, 1)
        θ_boundary = inactive_state.ξ.θ[1]
        @test isfinite(θ_boundary)
        @test isfinite(inactive_state.ξ.θ[2])
        boomerang_rate = PDMPSamplers.rate(linear_clock, boomerang, inactive_state, 0.0, trues(d))
        @test boomerang_rate ≈ (sqrt(2 / π) / 2) * pdf(Normal(), 0.0)

        Σ_dense = [1.0 0.4; 0.4 2.0]
        dense_boomerang = Boomerang(inv(Σ_dense), zeros(d), 0.0)
        dense_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.75]), BitVector([false, true]))
        dense_rate = PDMPSamplers.rate(clock, dense_boomerang, dense_state, 0.0, trues(d))
        dense_constant = sqrt(2 / π) * sqrt(Σ_dense[1, 1])
        @test dense_rate ≈ dense_constant * pdf(Normal(), 0.0)
        @test PDMPSamplers.rate(linear_clock, dense_boomerang, dense_state, 0.0, trues(d)) ≈ dense_rate
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(129), dense_state, dense_boomerang, 1)
        dense_θ = dense_state.ξ.θ[1]
        @test isfinite(dense_θ)
        @test isfinite(dense_state.ξ.θ[2])
        _, _, dense_boomerang_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(127), dense_boomerang, model, alg, 0.0, ξ)
        @test dense_boomerang_internal isa PDMPSamplers.AggregateStickyLoopState

        lowrank_boomerang = AdaptiveBoomerang(d; λref=0.0, scheme=:lowrank, rank=1)
        lrp = lowrank_boomerang.Γ
        lrp.D .= [1.0, 2.0]
        lrp.V .= [0.5; -0.25]
        lrp.Λ .= [0.8]
        PDMPSamplers.lowrank_precompute!(lrp)
        Σ_lowrank = Diagonal(lrp.D) + lrp.V * Diagonal(lrp.Λ) * lrp.V'
        lowrank_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, -0.2], [0.0, -0.4]), BitVector([false, true]))
        @test PDMPSamplers.rate(clock, lowrank_boomerang, lowrank_state, 0.0, trues(d)) ≈ sqrt(2 / π) * sqrt(Σ_lowrank[1, 1]) * pdf(Normal(), 0.0)
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(130), lowrank_state, lowrank_boomerang, 1)
        @test isfinite(lowrank_state.ξ.θ[1])
        _, _, lowrank_boomerang_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(128), lowrank_boomerang, model, alg, 0.0, ξ)
        @test lowrank_boomerang_internal isa PDMPSamplers.AggregateStickyLoopState

        adapted_flow = AdaptiveBoomerang(d; λref=0.0, scheme=:fullrank)
        adapted_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.75]), BitVector([false, true]))
        adapted_alg = PDMPSamplers._to_internal(alg, MersenneTwister(140), adapted_flow, model, adapted_state,
            PDMPSamplers.add_gradient_to_cache(PDMPSamplers.initialize_cache(MersenneTwister(141), adapted_flow, model.grad, alg, 0.0, adapted_state.ξ), adapted_state.ξ),
            PDMPSamplers.StatisticCounter())
        adapted_alg.aggregate_unstick_time = -123.0
        adapted_flow.Γ.data .= inv([1.0 0.4; 0.4 2.0])
        copyto!(adapted_flow.ΣL.data, cholesky(Symmetric([1.0 0.4; 0.4 2.0])).L)
        copyto!(adapted_flow.L.data, cholesky(Symmetric(adapted_flow.Γ.data)).L)
        PDMPSamplers._handle_dynamics_adaptation!(MersenneTwister(142), _AlwaysAdaptedAdapter(), adapted_alg, adapted_state, adapted_flow, PDMPSamplers.StatisticCounter())
        @test adapted_alg.aggregate_unstick_time != -123.0
        @test adapted_alg.aggregate_unstick_time >= adapted_state.t[]

        fourier_clock = FourierResidualAggregateClock(DenseGaussianSlab(zeros(d), Matrix(I, d, d), 1:d), BernoulliModelPrior(fill(0.5, d));
            order=6, cells=8, residual_budget=1e-3, allow_slow_fallback=false)
        fourier_alg = AggregateSticky(GridThinningStrategy(), fourier_clock, trues(d))
        fourier_ξ = SkeletonPoint([0.0, 0.5], [0.0, -1.0])
        fourier_state, _, fourier_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(143), boomerang, model, fourier_alg, 0.0, fourier_ξ)
        @test fourier_internal isa PDMPSamplers.AggregateStickyLoopState
        @test fourier_internal.clock isa FourierResidualAggregateClock
        @test PDMPSamplers.thinning_diagnostics(fourier_internal.clock).fallbacks == 0
        @test PDMPSamplers.thinning_diagnostics(fourier_internal.clock).rate_evaluations > 0
        @test fourier_internal.aggregate_unstick_time >= fourier_state.t[]

        precond_zz = PreconditionedZigZag(d; scale=[2.0, 0.5])
        @test PDMPSamplers.unstick_rate_constant(precond_zz, 1) == 2.0
        all_inactive_state = StickyPDMPState(Ref(0.0), SkeletonPoint(zeros(d), zeros(d)), falses(d))
        precond_zz_weights = zeros(d)
        PDMPSamplers._boundary_logweights_with_velocity!(precond_zz_weights, clock, precond_zz, all_inactive_state, trues(d))
        @test precond_zz_weights ≈ logpdf(Normal(), 0.0) .+ log.([2.0, 0.5])
        @test PDMPSamplers.rate(clock, precond_zz, all_inactive_state, 0.0, trues(d)) ≈ pdf(Normal(), 0.0) * 2.5
        @test PDMPSamplers.rate(linear_clock, precond_zz, all_inactive_state, 0.0, trues(d)) ≈ PDMPSamplers.rate(clock, precond_zz, all_inactive_state, 0.0, trues(d))
        @test PDMPSamplers.sample_time(MersenneTwister(138), linear_clock, precond_zz, all_inactive_state, 1.0, trues(d)) isa Real
        precond_zz_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.1]), BitVector([false, true]))
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(131), precond_zz_state, precond_zz, 1)
        @test abs(precond_zz_state.ξ.θ[1]) == 2.0
        _, _, precond_zz_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(127), precond_zz, model, alg, 0.0, ξ)
        @test precond_zz_internal isa PDMPSamplers.AggregateStickyLoopState

        precond_bps = PreconditionedBPS(d; scale=[2.0, 0.5])
        @test PDMPSamplers.unstick_rate_constant(precond_bps, 1) ≈ 2sqrt(2 / π)
        @test PDMPSamplers.rate(clock, precond_bps, all_inactive_state, 0.0, trues(d)) ≈ sqrt(2 / π) * pdf(Normal(), 0.0) * 2.5
        @test PDMPSamplers.rate(linear_clock, precond_bps, all_inactive_state, 0.0, trues(d)) ≈ PDMPSamplers.rate(clock, precond_bps, all_inactive_state, 0.0, trues(d))
        precond_bps_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.1]), BitVector([false, true]))
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(132), precond_bps_state, precond_bps, 1)
        precond_bps_draw = precond_bps_state.ξ.θ[1]
        @test isfinite(precond_bps_draw)
        _, _, precond_bps_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(133), precond_bps, model, alg, 0.0, ξ)
        @test precond_bps_internal isa PDMPSamplers.AggregateStickyLoopState

        identity_bps = PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(), BouncyParticle(d, 0.0))
        identity_bps_state = StickyPDMPState(
            Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.1]),
            BitVector([false, true]))
        @test PDMPSamplers.rate(clock, identity_bps, identity_bps_state, 0.0, trues(d)) > 0
        @test PDMPSamplers._boundary_proposal_clock_constant(identity_bps, identity_bps_state, 1) ≈ sqrt(2 / π)

        copied_state = copy(precond_bps_state)
        @test copied_state.boundary_scratch !== precond_bps_state.boundary_scratch
        copied_state.boundary_scratch.active_count = 99
        @test precond_bps_state.boundary_scratch.active_count != 99
        roots_probe_state = PDMPSamplers._roots_probe_state(precond_bps_state)
        @test roots_probe_state.boundary_scratch === precond_bps_state.boundary_scratch
        @test roots_probe_state.ξ !== precond_bps_state.ξ

        dense_bps = DensePreconditionedBPS(d; refresh_rate=0.0)
        Σ_dense_precond = [1.0 0.3; 0.3 1.6]
        set_dense_preconditioner!(dense_bps.metric,
            cholesky(Symmetric(Σ_dense_precond)).L)
        @test PDMPSamplers.unstick_rate_constant(dense_bps, 1) ≈ sqrt(2 / π) * sqrt(Σ_dense_precond[1, 1])
        dense_bps_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.75]), BitVector([false, true]))
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(134), dense_bps_state, dense_bps, 1)
        @test isfinite(dense_bps_state.ξ.θ[1])

        precond_boomerang = PreconditionedDynamics(DiagonalPreconditioner([2.0, 0.5]), boomerang)
        precond_boomerang_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.1]), BitVector([false, true]))
        @test PDMPSamplers.unstick_rate_constant(precond_boomerang, 1) ≈ sqrt(2 / π)
        identity_boomerang = PreconditionedDynamics(PDMPSamplers.IdentityPreconditioner(), boomerang)
        @test PDMPSamplers._boundary_proposal_clock_constant(identity_boomerang, inactive_state, 1) ≈
              PDMPSamplers._boundary_proposal_clock_constant(boomerang, inactive_state, 1)
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(135), precond_boomerang_state, precond_boomerang, 1)
        @test isfinite(precond_boomerang_state.ξ.θ[1])

        dense_preconditioner = PDMPSamplers.DensePreconditioner(d)
        set_dense_preconditioner!(dense_preconditioner, [1.1 0.0; 0.4 0.7])
        dense_preconditioned_boomerang = PreconditionedDynamics(dense_preconditioner, dense_boomerang)
        Σ_preconditioned_boomerang = dense_preconditioner.L * Σ_dense * dense_preconditioner.L'
        dense_preconditioned_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.75]), BitVector([false, true]))
        PDMPSamplers._boundary_proposal_clock_constant(dense_preconditioned_boomerang, dense_preconditioned_state, 1)
        @test PDMPSamplers.unstick_rate_constant(dense_preconditioned_boomerang, 1) ≈ sqrt(2 / π) * sqrt(Σ_preconditioned_boomerang[1, 1])
        @test PDMPSamplers._boundary_proposal_clock_constant(
            dense_preconditioned_boomerang, dense_preconditioned_state, 1) ≈
            sqrt(2 / π) * sqrt(Σ_preconditioned_boomerang[1, 1])

        old_l11 = dense_preconditioner.L[1, 1]
        updated_factor = Matrix(dense_preconditioner.L)
        updated_factor[1, 1] = old_l11 + 0.2
        set_dense_preconditioner!(dense_preconditioner, updated_factor)
        Σ_updated = dense_preconditioner.L * Σ_dense * dense_preconditioner.L'
        @test PDMPSamplers._boundary_proposal_clock_constant(
            dense_preconditioned_boomerang, dense_preconditioned_state, 1) ≈
            sqrt(2 / π) * sqrt(Σ_updated[1, 1])
        updated_factor[1, 1] = old_l11
        set_dense_preconditioner!(dense_preconditioner, updated_factor)

        dense_zz = DensePreconditionedZigZag(d)
        non_grid_alg = AggregateSticky(ThinningStrategy(GlobalBounds(1.0, d)), clock, trues(d))
        @test_throws ArgumentError PDMPSamplers.initialize_state(MersenneTwister(125), ZigZag(d), model, non_grid_alg, 0.0, ξ)
        _, _, dense_zz_internal, _, _ = PDMPSamplers.initialize_state(
            MersenneTwister(136), dense_zz, model, alg, 0.0, ξ)
        @test dense_zz_internal isa PDMPSamplers.AggregateStickyLoopState
        dense_zz_state = StickyPDMPState(Ref(0.0),
            SkeletonPoint([0.0, 0.5], [0.0, 0.0]), BitVector([false, true]))
        @test PDMPSamplers._boundary_proposal_clock_constant(
            dense_zz, dense_zz_state, 1) > 0
    end
end
