using Test
using LinearAlgebra
using Random
using PDMPSamplers

_allocated_dependent_hvp(h, x, v) = @allocated h(x, v)

@testset "dependent log-linear slab analytic HVP" begin
    rng = MersenneTwister(90210)
    d = 7
    beta = [1, 2, 3]
    logs = [4, 5, 6, 7]
    design = [1.0 0.5 0.5 0.0;
              1.0 0.5 0.0 0.5;
              1.0 0.0 0.5 0.5]
    slab = LogLinearGaussianScaleSlab(beta, logs, zeros(3), design)
    odds = BernoulliModelPrior(fill(0.5, 3))
    A = Diagonal(collect(1.0:d))
    posterior_grad!(out, x) = mul!(out, A, x)
    posterior_hvp!(out, x, v) = mul!(out, A, v)
    ph = PDMPSamplers.InplaceHVP(posterior_hvp!, zeros(d))

    for free_beta in (falses(3), BitVector([true, false, true]), trues(3))
        free = trues(d); free[beta] .= free_beta
        target = DependentSlabTarget(d, posterior_grad!, slab, odds;
            initial_free=free, posterior_hvp=ph)
        model = PDMPModel(target; hvp=true)
        @test occursin("Analytic", string(typeof(model.hvp.f)))
        for _ in 1:20
            x = 0.3 .* randn(rng, d)
            v = randn(rng, d)
            analytic = copy(model.hvp(x, v))
            eps_fd = 2e-6 / max(norm(v), 1.0)
            gp = zeros(d); gm = zeros(d)
            target(gp, x .+ eps_fd .* v)
            target(gm, x .- eps_fd .* v)
            fd = (gp .- gm) ./ (2eps_fd)
            @test analytic ≈ fd atol=2e-7 rtol=2e-7
        end
        x = 0.1 .* randn(rng, d); v = randn(rng, d)
        model.hvp(x, v)
        @test _allocated_dependent_hvp(model.hvp, x, v) == 0
    end

    # Providers without an analytic slab HVP retain the existing fallback.
    dense = DenseGaussianSlab(zeros(3), Matrix(I, 3, 3), beta)
    fallback = PDMPModel(DependentSlabTarget(d, posterior_grad!, dense, odds;
        posterior_hvp=ph); hvp=true)
    @test occursin("FiniteDiff", string(typeof(fallback.hvp.f)))
end
