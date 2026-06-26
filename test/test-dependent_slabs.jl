@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

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
        @test log_model_add_odds(bb, active_bb, 2) ≈ log(2.0 + 2) - log(3.0 + 4 - 2 - 1)
        @test_throws ArgumentError log_model_add_odds(bb, active_bb, 1)
    end

    @testset "Dense Gaussian slab conditioning" begin
        mean = [1.0, -0.5, 0.25]
        cov = [2.0 0.4 -0.2;
               0.4 1.5 0.3;
              -0.2 0.3 1.2]
        indices = [2, 4, 5]
        provider = DenseGaussianSlab(mean, cov, indices)
        x = [10.0, 1.3, 20.0, -0.1, 0.8]

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
    end
end
