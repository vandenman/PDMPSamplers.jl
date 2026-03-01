@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

import ADTypes
import DifferentiationInterface as DI
import ForwardDiff

@testset "PDMPModel Tests" begin

    # Dummy functions
    d = 2
    f_logdensity(x) = -0.5 * sum(x .^ 2)
    f_grad!(out, x) = (out .= -x)
    f_hvp!(out, x, v) = (out .= -v)

    @testset "Constructors" begin
        # 1. From Gradient Strategy
        grad_strat = FullGradient(f_grad!)
        model = PDMPModel(d, grad_strat)
        @test model.grad === grad_strat
        @test isnothing(model.hvp)

        # 2. From Gradient Strategy + HVP
        model_hvp = PDMPModel(d, grad_strat, f_hvp!)
        @test model_hvp.grad === grad_strat
        @test model_hvp.hvp isa PDMPSamplers.InplaceHVP
        @test model_hvp.hvp.f === f_hvp!

        # 3. From LogDensity with AD backend
        ld = LogDensity(f_logdensity)
        backend = DI.AutoForwardDiff()

        model_ld = PDMPModel(d, ld, backend)
        @test model_ld.d == d
        @test model_ld.grad isa FullGradient
        @test isnothing(model_ld.hvp)

        # 3b. Test that the gradient is correct (should be -∇log p = x for standard normal)
        x_test = [1.0, 2.0]
        out_test = zeros(d)
        compute_gradient!(model_ld.grad, x_test, out_test)
        @test out_test ≈ x_test

        # 4. From FullGradient + AD backend + HVP
        backend_hvp = DI.AutoForwardDiff()
        model_fg_hvp = PDMPModel(d, grad_strat, backend_hvp, true)
        @test model_fg_hvp.hvp isa Function
        hvp_result2 = model_fg_hvp.hvp(x_test, [0.0, 1.0])
        @test hvp_result2 isa AbstractVector
    end

    @testset "LogDensity + HVP" begin
        ld = LogDensity(f_logdensity)
        backend = DI.AutoForwardDiff()
        x_test = [1.0, 2.0]
        model_ld_hvp = PDMPModel(d, ld, backend, true)
        @test model_ld_hvp.hvp isa Function
    end

    @testset "SkeletonPoint copyto!" begin
        x1 = [1.0, 2.0]
        θ1 = [3.0, 4.0]
        x2 = [5.0, 6.0]
        θ2 = [7.0, 8.0]
        src = SkeletonPoint(x1, θ1)
        dest = SkeletonPoint(x2, θ2)
        result = copyto!(dest, src)
        @test result === dest
        @test dest.x == [1.0, 2.0]
        @test dest.θ == [3.0, 4.0]
    end
end
