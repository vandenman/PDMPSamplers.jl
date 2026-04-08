@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "RNG reproducibility" begin

    d = 3
    target = ZeroMeanIsoNormalTarget(Distributions.MvNormal(zeros(d), I(d)))
    grad = FullGradient(Base.Fix1(neg_gradient!, target))
    model = PDMPModel(d, grad)
    flow = BouncyParticle(I(d), zeros(d))
    alg = GridThinningStrategy()

    x0 = ones(d)
    θ0 = PDMPSamplers.initialize_velocity(flow, d)
    ξ0 = SkeletonPoint(x0, θ0)
    T_run = 5_000.0

    @testset "Same seed produces identical traces" begin
        result1 = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; seed=42, progress=false)
        result2 = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; seed=42, progress=false)

        trace1 = result1.traces[1]
        trace2 = result2.traces[1]

        @test trace1.times == trace2.times
        @test Matrix(trace1.positions) == Matrix(trace2.positions)
        @test Matrix(trace1.velocities) == Matrix(trace2.velocities)
    end

    @testset "Different seeds produce different traces" begin
        result1 = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; seed=42, progress=false)
        result2 = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; seed=99, progress=false)

        trace1 = result1.traces[1]
        trace2 = result2.traces[1]

        @test trace1.times != trace2.times
    end

    @testset "Multi-chain produces distinct traces" begin
        n_chains = 4
        result = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; seed=123, n_chains, progress=false)

        @test length(result.traces) == n_chains

        for i in 1:n_chains, j in (i+1):n_chains
            @test result.traces[i].times != result.traces[j].times
        end
    end

    @testset "Multi-chain with seed is deterministic" begin
        n_chains = 3
        result1 = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; seed=456, n_chains, progress=false)
        result2 = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; seed=456, n_chains, progress=false)

        for i in 1:n_chains
            @test result1.traces[i].times == result2.traces[i].times
            @test Matrix(result1.traces[i].positions) == Matrix(result2.traces[i].positions)
            @test Matrix(result1.traces[i].velocities) == Matrix(result2.traces[i].velocities)
        end
    end
end
