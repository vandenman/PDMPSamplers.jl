@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

const ExponentialSumAggregateClock = PDMPSamplers.ExponentialSumAggregateClock
const FixedCovarianceCache = PDMPSamplers.FixedCovarianceCache
const HarmonicLogLinearAggregateClock = PDMPSamplers.HarmonicLogLinearAggregateClock
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
const slab_cache_key = PDMPSamplers.slab_cache_key
const slab_cache_style = PDMPSamplers.slab_cache_style
const thinning_diagnostics = PDMPSamplers.thinning_diagnostics
const unstick_rate_constant = PDMPSamplers.unstick_rate_constant

# Under the velocity-preserving sticky kernel a frozen coordinate is released
# at the speed it stored when it froze.  Fixtures that build a frozen state
# directly store distinctive speeds ±(0.5 + 0.3i), which differ from every
# release constant of the old law, so the expected values below hold only
# under the stored-velocity law.
_frozen_speed(i::Integer) = (isodd(i) ? 1.0 : -1.0) * (0.5 + 0.3i)

function _store_frozen_speeds!(state::StickyPDMPState)
    for i in eachindex(state.free)
        state.stored_velocity[i] = state.free[i] ? 0.0 : _frozen_speed(i)
    end
    return state
end

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

@testset "Harmonic Boomerang log-linear aggregate clock" begin
    beta = [1, 2, 3, 4]
    logscale = [5, 6, 7, 8]
    x = [0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.3, -0.2]
    θ = [0.4, -0.3, 0.2, -0.1, 0.7, -0.5, 0.6, -0.4]
    free = BitVector([false, false, true, false, true, true, true, true])
    can_stick = BitVector([true, true, false, true, false, false, false, false])
    prior = BetaBernoulliModelPrior(4, 1.0, 1.0)
    flow = Boomerang(8)

    designs = (
        [1.0 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0;
         0.0 0.0 1.0 0.0;
         0.0 0.0 0.0 1.0],
        [1.0 0.0 0.0 0.0;
         1.0 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0;
         0.0 0.0 1.0 1.0],
    )

    for design in designs
        provider = LogLinearGaussianScaleSlab(
            beta, logscale, log.([1.2, 0.9, 1.1, 0.8]), design)
        state = StickyPDMPState(
            Ref(0.0), SkeletonPoint(copy(x), copy(θ)), copy(free))
        _store_frozen_speeds!(state)
        clock = PDMPSamplers.default_aggregate_unstick_clock(
            provider, prior, flow)
        @test clock isa HarmonicLogLinearAggregateClock

        direct_rate = function (t)
            active = BitVector(state.free[beta])
            total = 0.0
            for j in eachindex(beta)
                (can_stick[beta[j]] && !active[j]) || continue
                ell = PDMPSamplers._loglinear_boomerang_logscale_coeffs(
                    provider, flow, state, j)
                c0, cc, cs = ell
                logscale_t = c0 + cc * cos(t) + cs * sin(t)
                C = PDMPSamplers._boundary_proposal_clock_constant(
                    flow, state, beta[j])
                total += exp(log_model_add_odds(prior, active, j) -
                    0.5 * log(2π) - logscale_t + log(C))
            end
            total
        end

        for t in range(0.0, 8π; length=33)
            @test PDMPSamplers.rate(clock, flow, state, t, can_stick) ≈
                  direct_rate(t) rtol=1e-11
        end

        # The certified cell roofs bound the exact rate on every cell.
        horizon = 8π
        PDMPSamplers._prepare_harmonic_loglinear_cells!(
            clock, flow, state, horizon, can_stick)
        cells = clock.workspace.cells_used
        for t in range(0.0, horizon; length=1025)
            cell = min(cells, floor(Int, t / horizon * cells) + 1)
            @test clock.workspace.roofs[cell] + 1e-10 >= direct_rate(t)
        end

        rng = MersenneTwister(445)
        event_time = PDMPSamplers.sample_time(
            rng, clock, flow, state, 8π, can_stick)
        @test isfinite(event_time)
        @test event_time >= 0.0
        @test PDMPSamplers.sample_label(
            rng, clock, flow, state, event_time, can_stick) in beta
        @test thinning_diagnostics(clock).fallbacks == 0

        preconditioned = PreconditionedDynamics(
            DiagonalPreconditioner(ones(8)), flow)
        pre_clock = PDMPSamplers.default_aggregate_unstick_clock(
            provider, prior, preconditioned)
        @test pre_clock isa HarmonicLogLinearAggregateClock
    end
end

@testset "Adaptive Boomerang harmonic clock survives diagonal adaptation" begin
    d = 8
    beta = [1, 2, 3, 4]
    logscale = [5, 6, 7, 8]
    design = Matrix{Float64}(I, 4, 4)
    provider = LogLinearGaussianScaleSlab(
        beta, logscale, zeros(4), design)
    prior = BetaBernoulliModelPrior(4, 1.0, 1.0)
    flow = AdaptiveBoomerang(d; λref=0.0, scheme=:diagonal)
    state = StickyPDMPState(
        Ref(0.0), SkeletonPoint(randn(d), randn(d)),
        BitVector([false, false, true, false, true, true, true, true]))
    _store_frozen_speeds!(state)
    can_stick = BitVector([true, true, false, true, false, false, false, false])
    clock = PDMPSamplers.default_aggregate_unstick_clock(provider, prior, flow)
    reference = SummedRateClock(provider, prior)

    @test flow isa MutableBoomerang
    @test clock isa HarmonicLogLinearAggregateClock
    rate_before = PDMPSamplers.rate(clock, flow, state, 0.7, can_stick)
    @test rate_before ≈ PDMPSamplers.rate(
        reference, flow, state, 0.7, can_stick) rtol=1e-10
    horizon = 8π
    PDMPSamplers._prepare_harmonic_loglinear_cells!(
        clock, flow, state, horizon, can_stick)
    cells = clock.workspace.cells_used
    for t in rand(MersenneTwister(121), 20) .* horizon
        cell = min(cells, floor(Int, t / horizon * cells) + 1)
        @test clock.workspace.roofs[cell] + 1e-10 >=
            PDMPSamplers.rate(reference, flow, state, t, can_stick)
    end

    adapter = PDMPSamplers.BoomerangAdapter(
        1.0, 0.0, d; scheme=:diagonal, can_stick=falses(d))
    stats = adapter.stats
    stats.total_time = 2.0
    stats.sum_x_dt .= collect(1.0:d)
    stats.sum_x2_dt .= collect(1.0:d).^2 .+ collect(1.0:d)
    old_cov = PDMPSamplers._boomerang_covariance_entry(flow, 1, 1)
    state.t[] = 2.0
    PDMPSamplers.adapt!(
        MersenneTwister(122), adapter, state, flow, nothing, nothing;
        phase=:warmup)
    new_cov = PDMPSamplers._boomerang_covariance_entry(flow, 1, 1)
    @test new_cov != old_cov
    @test adapter.no_updates_done == 1
    rate_after = PDMPSamplers.rate(clock, flow, state, 0.7, can_stick)
    @test rate_after ≈ PDMPSamplers.rate(
        reference, flow, state, 0.7, can_stick) rtol=1e-10
    @test rate_after != rate_before
    @test thinning_diagnostics(clock).fallbacks == 0

    dense_covariance = dot(view(flow.ΣL, 1, :), view(flow.ΣL, 1, :))
    @test PDMPSamplers._boomerang_covariance_entry(flow, 1, 1) ≈ dense_covariance

    bad_provider = LogLinearGaussianScaleSlab(
        beta, logscale, zeros(4), design)
    bad_provider.log_base_scales[1] = NaN
    bad_clock = HarmonicLogLinearAggregateClock(bad_provider, prior)
    @test_throws DomainError PDMPSamplers.rate(
        bad_clock, flow, state, 0.2, can_stick)
    @test_throws DomainError PDMPSamplers._prepare_harmonic_loglinear_cells!(
        bad_clock, flow, state, 1.0, can_stick)

    # TODO: a non-finite ΣL is not caught where it is used.  Sticky refreshment
    # (`_refresh_preserved_sticky_velocity!` for a diagonal Boomerang) silently
    # writes NaN into `ξ.θ` or `stored_velocity`; it surfaces only later, via
    # `validate_state` or a NaN sticky time during sampling.
    #
    # Under the stored-velocity law a frozen coordinate is released at its
    # stored speed, so the release rate does not read the reference covariance
    # factor ΣL; ΣL only enters velocity draws.  Coordinate 1 is frozen and
    # stickable, so under the old law (constant sqrt(2/π) Σ₁₁^{1/2}) changing
    # Σ₁₁ would change the rate.
    reference_flow = AdaptiveBoomerang(d; λref=0.0, scheme=:diagonal)
    reference_clock = PDMPSamplers.default_aggregate_unstick_clock(
        provider, prior, reference_flow)
    release_times = (0.0, 0.2, 1.3)
    reference_rates = [PDMPSamplers.rate(
        reference_clock, reference_flow, state, t, can_stick)
        for t in release_times]
    @test all(r -> isfinite(r) && r > 0, reference_rates)
    for changed_scale in (3.0, 1e-3, NaN)
        changed_flow = AdaptiveBoomerang(d; λref=0.0, scheme=:diagonal)
        changed_flow.ΣL.diag[1] = changed_scale
        changed_clock = PDMPSamplers.default_aggregate_unstick_clock(
            provider, prior, changed_flow)
        @test [PDMPSamplers.rate(changed_clock, changed_flow, state, t,
            can_stick) for t in release_times] == reference_rates
    end
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
            Cv = PDMPSamplers._boundary_proposal_clock_constant(
                flow, state, indices[j])
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
        _active_prior_grad_alloc(
            exch, generic_empty_out, x, BitVector([true, false, true]))
        @test _active_prior_grad_alloc(
            exch, generic_empty_out, x, BitVector([true, false, true])) == 0

        zero_exch = ZeroMeanExchangeableGaussianSlab(indices, 1.3, 0.1)
        zero_exch_out = fill(NaN, length(x))
        active_prior_grad!(zero_exch, zero_exch_out, x, active)
        zero_exch_cov = Matrix(1.3I, 3, 3) .+ 0.1
        expected_zero_exch = zeros(length(x))
        expected_zero_exch[indices[A]] .=
            zero_exch_cov[A, A] \ x[indices[A]]
        @test zero_exch_out ≈ expected_zero_exch
        _active_prior_grad_alloc(zero_exch, zero_exch_out, x, active)
        @test _active_prior_grad_alloc(zero_exch, zero_exch_out, x, active) == 0

        logscale_x = [0.1, 1.3, -0.2, -0.1, 0.8]
        logscale_active = BitVector([true, true, false])

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

    @testset "Dependent slab target baseline" begin
        d = 3
        posterior_grad!(out, x) = (copyto!(out, x); out)
        # Precision diag(2, 3): the active slab gradient is (2 x₁, 3 x₃).
        provider = DenseGaussianSlab(zeros(2), Matrix(Diagonal([1 / 2, 1 / 3])), [1, 3])
        target = DependentSlabTarget(d, posterior_grad!, provider, BernoulliModelPrior(fill(0.5, 2));
            initial_free=BitVector([true, false, false]))
        out = fill(NaN, d)
        x = [1.0, 2.0, -1.0]
        target(out, x)
        @test out ≈ [1.0, 2.0, 2.0]
        @test PDMPModel(target) isa PDMPModel
    end

    @testset "Diagonal Gaussian slab and summed aggregate clock" begin
        d = 3
        beta_idx = [1, 2, 3]
        σ = [2.0, 3.0, 5.0]
        provider = DenseGaussianSlab(zeros(d), Matrix(Diagonal(σ .^ 2)), beta_idx)
        odds = BernoulliModelPrior(fill(0.5, d))
        clock = SummedRateClock(provider, odds)
        flow = ZigZag(d)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint(zeros(d), zeros(d)),
            falses(d),
        )
        _store_frozen_speeds!(state)
        can_stick = BitVector([true, false, true])

        expected = abs(_frozen_speed(1)) * pdf(Normal(0.0, σ[1]), 0.0) +
            abs(_frozen_speed(3)) * pdf(Normal(0.0, σ[3]), 0.0)
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

        empty_active = falses(4)
        empty_weights = fill(NaN, 4)
        boundary_logweights!(empty_weights, provider, odds, x, empty_active, stickable)
        expected_empty = fill(-Inf, 4)
        for j in eachindex(expected_empty)
            if stickable[j]
                expected_empty[j] = log_model_add_odds(odds, empty_active, j) +
                                    logpdf(Normal(mean[j], sqrt(cov[j, j])), 0.0)
            end
        end
        @test empty_weights ≈ expected_empty
    end

    @testset "Exchangeable slabs" begin
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

        exch_mean = fill(NaN, 4)
        exch_cov = fill(NaN, 4, 4)
        @test gaussian_slab!(exch, exch_mean, exch_cov, x) === nothing
        @test exch_mean == fill(μ, 4)
        @test exch_cov ≈ dense_cov

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

        short_prior = BernoulliModelPrior(fill(0.4, 2))
        @test_throws DimensionMismatch aggregate_lograte(exch, short_prior, log(1.0), x, active, stickable)
        @test_throws DimensionMismatch SummedRateClock(exch, short_prior)
        @test_throws DimensionMismatch LinearGaussianAggregateClock(exch, short_prior)

        rng = MersenneTwister(11)
        labels = [sample_unstick_label(rng, exch, odds, x, active, stickable) for _ in 1:50]
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
        # The linear clock needs a fixed slab covariance.
        moving_scale_provider = LogLinearGaussianScaleSlab(
            1:4, 5:8, zeros(4), Matrix{Float64}(I, 4, 4))
        @test_throws ArgumentError LinearGaussianAggregateClock(moving_scale_provider, odds)
        flow = ZigZag(4)

        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.7, 0.0, -0.3, 0.0], [1.0, 0.0, -1.0, 0.0]),
            BitVector([true, false, true, false]),
        )
        _store_frozen_speeds!(state)
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
        _store_frozen_speeds!(const_state)
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

        endpoint_linear = LinearGaussianAggregateClock(DenseGaussianSlab([0.0], reshape([1.0], 1, 1), [1]), BernoulliModelPrior([1.0]))
        endpoint_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0], [0.0]), falses(1))
        _store_frozen_speeds!(endpoint_state)
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
        _store_frozen_speeds!(exch_state)
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

        preconditioned_bps = PreconditionedBPS(
            4; refresh_rate=0.0, scale=[0.03, 0.4, 1.2, 2.0])
        for τi in (0.0, 0.2, 0.9)
            @test PDMPSamplers.rate(
                exch_clock, preconditioned_bps, exch_state, τi,
                exch_can_stick) ≈ PDMPSamplers.rate(
                exch_clock.fallback, preconditioned_bps, exch_state, τi,
                exch_can_stick)
        end
        for T_i in (0.3, 1.5)
            @test PDMPSamplers.cumulative_hazard(
                exch_clock, preconditioned_bps, exch_state, 0.0, T_i,
                exch_can_stick) ≈ PDMPSamplers.cumulative_hazard(
                exch_clock.fallback, preconditioned_bps, exch_state, 0.0, T_i,
                exch_can_stick) rtol=1e-8 atol=1e-9
        end
        @test PDMPSamplers.sample_time(
            MersenneTwister(211), exch_clock, preconditioned_bps,
            exch_state, 2.0, exch_can_stick) ≈
            PDMPSamplers.sample_time(
                MersenneTwister(211), exch_clock.fallback,
                preconditioned_bps, exch_state, 2.0,
                exch_can_stick) rtol=1e-7 atol=1e-8
        @test PDMPSamplers.sample_label(
            MersenneTwister(212), exch_clock, preconditioned_bps,
            exch_state, 0.7, exch_can_stick) ==
            PDMPSamplers.sample_label(
                MersenneTwister(212), exch_clock.fallback,
                preconditioned_bps, exch_state, 0.7,
                exch_can_stick)

        zero_slope_state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.8, 0.0, -0.2, 0.0], [1.0, 0.0, -1.0, 0.0]),
            BitVector([true, false, true, false]),
        )
        _store_frozen_speeds!(zero_slope_state)
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
        _store_frozen_speeds!(subset_state)
        size_prior_subset = BetaBernoulliModelPrior(4, 2.0, 3.0)
        subset_clock = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:4, u, v), size_prior_subset)
        full_rate = PDMPSamplers.rate(subset_clock, flow, subset_state, 0.0, trues(4))
        subset_rate = PDMPSamplers.rate(subset_clock, flow, subset_state, 0.0, subset_all_inactive)
        # With no active coordinate every term shares the odds and density at
        # zero, so each coordinate contributes in proportion to its speed.
        all_speeds = abs.(_frozen_speed.(1:4))
        @test subset_rate ≈ full_rate *
            sum(all_speeds[subset_all_inactive]) / sum(all_speeds)

        endpoint_exch = LinearGaussianAggregateClock(ZeroMeanExchangeableGaussianSlab(1:2, 1.0, 0.1), BernoulliModelPrior([1.0, 0.0]))
        endpoint_exch_state = StickyPDMPState(Ref(0.0), SkeletonPoint(zeros(2), zeros(2)), falses(2))
        _store_frozen_speeds!(endpoint_exch_state)
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

    @testset "Log-linear scale exponential-sum clock" begin
        # Coefficients 1 and 2 share the log scale x[4], coefficient 3 uses x[5].
        scale_coord = [4, 4, 5]
        provider = LogLinearGaussianScaleSlab([1, 2, 3], [4, 5],
            log.([2.0, 3.0, 4.0]), [1.0 0.0; 1.0 0.0; 0.0 1.0])
        odds = BernoulliModelPrior([0.25, 0.5, 0.75])
        clock = ExponentialSumAggregateClock(provider, odds)
        flow = ZigZag(5)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.2, -0.1], [0.0, 0.0, 0.0, 0.3, -0.2]),
            falses(5),
        )
        _store_frozen_speeds!(state)
        can_stick = BitVector([true, false, true, false, false])

        function direct_rate(t)
            total = 0.0
            active = falses(3)
            for j in (1, 3)
                logρ = log_model_add_odds(odds, active, j)
                log_s0 = provider.log_base_scales[j] + state.ξ.x[scale_coord[j]]
                r = state.ξ.θ[scale_coord[j]]
                total += abs(_frozen_speed(j)) *
                    exp(logρ - 0.5 * log(2π) - log_s0 - r * t)
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

    @testset "Preconditioned exponential-sum clock and stable interval hazards" begin
        provider = LogLinearGaussianScaleSlab(
            [1, 2, 3], [4, 5], log.([1.5, 2.0, 2.5]),
            [1.0 0.0; 0.5 -0.25; 0.0 0.75])
        odds = BernoulliModelPrior(fill(0.5, 3))
        clock = ExponentialSumAggregateClock(provider, odds)
        state = StickyPDMPState(
            Ref(0.0),
            SkeletonPoint([0.0, 0.0, 0.0, 0.2, -0.1],
                          [0.0, 0.0, 0.0, 0.3, -0.2]),
            falses(5))
        can_stick = BitVector([true, true, true, false, false])
        flows = (
            ZigZag(5),
            BouncyParticle(5, 0.0),
            PreconditionedZigZag(5; scale=[2.0, 0.5, 1.25, 1.0, 1.0]),
            PreconditionedBPS(5; refresh_rate=0.0,
                              scale=[2.0, 0.5, 1.25, 1.0, 1.0]),
        )
        preconditioned_boomerang = PreconditionedDynamics(
            DiagonalPreconditioner(ones(5)), Boomerang(5))
        boomerang_clock = default_aggregate_unstick_clock(
            provider, odds, preconditioned_boomerang)
        boomerang_method = which(PDMPSamplers.sample_time,
            (MersenneTwister, typeof(boomerang_clock),
             typeof(preconditioned_boomerang), typeof(state), Float64, BitVector))
        @test !occursin("linear_clocks.jl", String(boomerang_method.file))
        for flow in flows
            _store_frozen_speeds!(state)
            indices = beta_indices(provider)
            direct_rate(t) = sum(begin
                j = findfirst(==(i), indices)
                logρ = log_model_add_odds(odds, falses(3), j)
                log_s = PDMPSamplers._independent_log_scale(provider,
                    state.ξ.x, j) + PDMPSamplers._independent_log_scale_slope(
                    provider, state.ξ.θ, j) * t
                C = PDMPSamplers._boundary_proposal_clock_constant(flow, state, i)
                C * exp(logρ - 0.5 * log(2π) - log_s)
            end for i in indices)
            for t in (0.0, 0.4, 1.2)
                @test PDMPSamplers.rate(clock, flow, state, t, can_stick) ≈ direct_rate(t)
            end
            H = PDMPSamplers.cumulative_hazard(clock, flow, state, 0.0, 1.7, can_stick)
            H_quad, _ = PDMPSamplers.QuadGK.quadgk(direct_rate, 0.0, 1.7;
                rtol=1e-10, atol=1e-12)
            @test H ≈ H_quad rtol=1e-9 atol=1e-10
            τ = PDMPSamplers.sample_time(MersenneTwister(991), clock, flow,
                state, 2.0, can_stick)
            if isfinite(τ)
                threshold = rand(MersenneTwister(991), Exponential())
                @test PDMPSamplers.cumulative_hazard(clock, flow, state,
                    0.0, τ, can_stick) ≈ threshold rtol=1e-7 atol=1e-8
            end
            labels = [PDMPSamplers.sample_label(MersenneTwister(1000 + k),
                clock, flow, state, 0.4, can_stick) for k in 1:1000]
            @test all(in(indices), labels)
            @test all(isfinite, labels)
        end

        for (slope, t0, t1) in ((0.0, 0.0, 2.0), (3.0, 0.2, 1.7),
                                (-3.0, 0.2, 1.7))
            h = PDMPSamplers._exponential_component_interval_hazard(
                log(0.7), slope, t0, t1)
            @test isfinite(h) && h >= 0.0
            @test PDMPSamplers._exponential_component_interval_hazard(
                log(0.7), slope, t0, t1 + 0.5) >= h
        end
        @test PDMPSamplers._exponential_component_interval_hazard(
            -700.0, 1.0e308, 1.0, 2.0) == 0.0
        @test isinf(PDMPSamplers._exponential_component_interval_hazard(
            0.0, -1.0e308, 0.0, 2.0))
        @test !isnan(PDMPSamplers._exponential_component_interval_hazard(
            0.0, -100.0, 1.0e3, 1.0e3 + 1.0))
        @test_throws DomainError PDMPSamplers._safe_exp_logvalue(NaN;
            logc=1.0, slope=-2.0, t0=0.0, t1=1.0)
    end

    @testset "Accelerated clocks fall back to the summed-rate clock for other flows" begin
        provider = DenseGaussianSlab(zeros(2), Matrix{Float64}(I, 2, 2), 1:2)
        odds = BernoulliModelPrior(fill(0.5, 2))
        summed = SummedRateClock(provider, odds)
        accelerated = LinearGaussianAggregateClock(provider, odds)
        state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.5, -0.25], [1.0, -1.0]), falses(2))
        _store_frozen_speeds!(state)
        flow = _FutureContinuousDynamics()
        @test PDMPSamplers.rate(accelerated, flow, state, 0.3, trues(2)) ≈
              PDMPSamplers.rate(summed, flow, state, 0.3, trues(2))
        @test PDMPSamplers.sample_time(MersenneTwister(120), accelerated, flow,
                  state, 0.5, trues(2)) ==
              PDMPSamplers.sample_time(MersenneTwister(120), summed, flow,
                  state, 0.5, trues(2))
    end

    @testset "Stored-velocity release law for preconditioned Zig-Zag and BPS" begin
        # Two frozen coordinates with stored speeds w = (0.7, -1.3).  Under the
        # stored-velocity law coordinate j is released at rate
        #     oddsⱼ |wⱼ| fⱼ(0),
        # and the release label has probability proportional to that rate.
        # The old law replaced |wⱼ| by the flow constant (the metric scale for
        # Zig-Zag, sqrt(2/π) × scale for BPS), which these values exclude.
        speeds = [0.7, -1.3]
        scales = [2.0, 0.5]
        inclusion = [0.3, 0.6]
        odds = inclusion ./ (1 .- inclusion)
        sds = [1.0, 1.5]
        f0 = [pdf(Normal(0.0, sds[j]), 0.0) for j in 1:2]
        expected_terms = odds .* abs.(speeds) .* f0
        expected_rate = sum(expected_terms)
        expected_probabilities = expected_terms ./ expected_rate

        prior = BernoulliModelPrior(inclusion)
        dense_provider = DenseGaussianSlab(zeros(2), Matrix(Diagonal(sds .^ 2)), 1:2)
        summed_clock = SummedRateClock(dense_provider, prior)
        linear_clock = LinearGaussianAggregateClock(dense_provider, prior)
        state = StickyPDMPState(Ref(0.0),
            SkeletonPoint(zeros(2), zeros(2)), falses(2))
        state.stored_velocity .= speeds
        can_stick = trues(2)
        normalized(logw) = (w = exp.(logw .- maximum(logw)); w ./ sum(w))

        flows = (
            PreconditionedZigZag(2; scale=scales),
            PreconditionedBPS(2; refresh_rate=0.0, scale=scales),
        )
        for flow in flows
            old_law_rate = sum(odds .* f0 .*
                [unstick_rate_constant(flow, j) for j in 1:2])
            @test !isapprox(old_law_rate, expected_rate; rtol=1e-3)

            @test PDMPSamplers.rate(summed_clock, flow, state, 0.0,
                can_stick) ≈ expected_rate rtol=1e-12
            @test PDMPSamplers.rate(linear_clock, flow, state, 0.0,
                can_stick) ≈ expected_rate rtol=1e-12
            # Frozen coordinates do not move, so the rate is constant in time.
            @test PDMPSamplers.rate(linear_clock, flow, state, 0.8,
                can_stick) ≈ expected_rate rtol=1e-12
            @test PDMPSamplers.cumulative_hazard(linear_clock, flow, state,
                0.0, 1.7, can_stick) ≈ 1.7 * expected_rate rtol=1e-10

            # Exact label probabilities from the log weights sample_label uses.
            summed_logw = zeros(2)
            PDMPSamplers._boundary_logweights_with_velocity!(summed_logw,
                summed_clock, flow, state, can_stick)
            @test summed_logw ≈ log.(expected_terms) rtol=1e-12
            @test normalized(summed_logw) ≈ expected_probabilities rtol=1e-12
            linear_cache, _ = PDMPSamplers._linear_gaussian_label_weights!(
                linear_clock, flow, state, can_stick, 0.0)
            @test normalized(linear_cache.log_weights) ≈
                expected_probabilities rtol=1e-12
        end

        # Exchangeable slab: equal marginal density at zero for both
        # coordinates, so labels are proportional to oddsⱼ |wⱼ|.
        μ, u, v = 0.3, 1.4, 0.25
        exch_f0 = pdf(Normal(μ, sqrt(u + v)), 0.0)
        exch_terms = odds .* abs.(speeds) .* exch_f0
        exch_clock = LinearGaussianAggregateClock(
            ExchangeableGaussianSlab(1:2, μ, u, v), prior)
        for flow in flows
            @test PDMPSamplers.rate(exch_clock, flow, state, 0.0,
                can_stick) ≈ sum(exch_terms) rtol=1e-12
            # Preconditioned BPS delegates labels to the exact fallback; both
            # paths leave the log weights they sampled from in a cache.
            PDMPSamplers.sample_label(MersenneTwister(5), exch_clock, flow,
                state, can_stick)
            label_cache = flow.dynamics isa BouncyParticle ?
                exch_clock.fallback.cache : exch_clock.cache
            @test normalized(label_cache.log_weights) ≈
                exch_terms ./ sum(exch_terms) rtol=1e-12
        end
    end

    @testset "AggregateSticky requests sticky state" begin
        d = 2
        # Independent standard-normal slab: density pdf(Normal(), 0) at zero.
        provider = DenseGaussianSlab(zeros(d), Matrix{Float64}(I, d, d), 1:d)
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
        nuisance_dense = PDMPTrace(nuisance_trace)
        nuisance_free = .!(iszero.(nuisance_dense.positions) .&
            iszero.(nuisance_dense.velocities))
        @test all(view(nuisance_free, 2, :))
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

        # Coordinate 2 has infinite model-prior odds, so it is released
        # immediately.
        scalar_provider = ZeroMeanExchangeableGaussianSlab(1:2, 1.0, 0.1)
        scalar_clock = default_aggregate_unstick_clock(scalar_provider, BernoulliModelPrior([0.0, 1.0]))

        tiny_clock = SummedRateClock(
            DenseGaussianSlab([0.0], reshape([1.0], 1, 1), 1:1),
            BernoulliModelPrior([1e-20]))
        tiny_state = StickyPDMPState(
            Ref(0.0), SkeletonPoint([0.0], [0.0]), falses(1))
        _store_frozen_speeds!(tiny_state)
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

        # The log scale of coordinate 1 grows along the path from exp(10), so
        # the total release hazard is finite and tiny: the clock is defective.
        defective_provider2 = LogLinearGaussianScaleSlab([1], [2], [10.0], reshape([1.0], 1, 1))
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
        # Release at the stored speed |w₁| = 1.3, not at the old constant
        # sqrt(2/π) Σ₁₁^{1/2} = sqrt(2/π) / 2; the model-prior odds are 1.
        release_state = StickyPDMPState(Ref(0.0),
            SkeletonPoint([0.0, 0.5], [0.0, 0.1]), BitVector([false, true]))
        release_state.stored_velocity[1] = -1.3
        boomerang_rate = PDMPSamplers.rate(linear_clock, boomerang, release_state, 0.0, trues(d))
        @test boomerang_rate ≈ 1.3 * pdf(Normal(), 0.0)

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
        adapted_trace = PDMPSamplers.TraceManager(
            adapted_state, adapted_flow, alg, 1.0)
        PDMPSamplers._handle_dynamics_adaptation!(MersenneTwister(142),
            _AlwaysAdaptedAdapter(), adapted_alg, adapted_state, adapted_flow,
            PDMPSamplers.StatisticCounter(), adapted_trace, :warmup)
        @test adapted_alg.aggregate_unstick_time != -123.0
        @test adapted_alg.aggregate_unstick_time >= adapted_state.t[]

        precond_zz = PreconditionedZigZag(d; scale=[2.0, 0.5])
        @test PDMPSamplers.unstick_rate_constant(precond_zz, 1) == 2.0
        # Stored speeds differ from the metric scales [2.0, 0.5], which were
        # the old release constants; odds are 1 and f(0) = pdf(Normal(), 0).
        all_inactive_speeds = [0.7, -1.3]
        all_inactive_state = StickyPDMPState(Ref(0.0), SkeletonPoint(zeros(d), zeros(d)), falses(d))
        all_inactive_state.stored_velocity .= all_inactive_speeds
        precond_zz_weights = zeros(d)
        PDMPSamplers._boundary_logweights_with_velocity!(precond_zz_weights, clock, precond_zz, all_inactive_state, trues(d))
        @test precond_zz_weights ≈ logpdf(Normal(), 0.0) .+ log.(abs.(all_inactive_speeds))
        @test PDMPSamplers.rate(clock, precond_zz, all_inactive_state, 0.0, trues(d)) ≈ pdf(Normal(), 0.0) * 2.0
        @test PDMPSamplers.rate(linear_clock, precond_zz, all_inactive_state, 0.0, trues(d)) ≈ PDMPSamplers.rate(clock, precond_zz, all_inactive_state, 0.0, trues(d))
        @test PDMPSamplers.sample_time(MersenneTwister(138), linear_clock, precond_zz, all_inactive_state, 1.0, trues(d)) isa Real
        precond_zz_state = StickyPDMPState(Ref(0.0), SkeletonPoint([0.0, 0.5], [0.0, 0.1]), BitVector([false, true]))
        @test PDMPSamplers.propose_boundary_velocity!(MersenneTwister(131), precond_zz_state, precond_zz, 1)
        @test abs(precond_zz_state.ξ.θ[1]) == 2.0
        _, _, precond_zz_internal, _, _ = PDMPSamplers.initialize_state(MersenneTwister(127), precond_zz, model, alg, 0.0, ξ)
        @test precond_zz_internal isa PDMPSamplers.AggregateStickyLoopState

        precond_bps = PreconditionedBPS(d; scale=[2.0, 0.5])
        @test PDMPSamplers.unstick_rate_constant(precond_bps, 1) ≈ 2sqrt(2 / π)
        # The stored speeds, not the BPS constant sqrt(2/π) × scale, set the rate.
        @test PDMPSamplers.rate(clock, precond_bps, all_inactive_state, 0.0, trues(d)) ≈ pdf(Normal(), 0.0) * 2.0
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
        identity_bps_state.stored_velocity[1] = 0.7
        @test PDMPSamplers.rate(clock, identity_bps, identity_bps_state, 0.0, trues(d)) ≈ 0.7 * pdf(Normal(), 0.0)
        @test PDMPSamplers._boundary_proposal_clock_constant(identity_bps, identity_bps_state, 1) == 0.7

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
