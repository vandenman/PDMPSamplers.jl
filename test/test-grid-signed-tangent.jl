@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

mutable struct TangentScriptedRNG <: Random.AbstractRNG
    exps::Vector{Float64}
    uniforms::Vector{Float64}
end
Random.randexp(rng::TangentScriptedRNG) = popfirst!(rng.exps)
Random.rand(rng::TangentScriptedRNG) = popfirst!(rng.uniforms)

@testset "Signed tangent bounds retain interior positive rates" begin
    grad!(out, x) = (fill!(out, -100.0); out[1] = (x[1]-1)*(x[1]-2)*(x[1]-10); out)
    hvp!(out, x, v) = (fill!(out, 0.0); out[1] = (3x[1]^2-26x[1]+32)*v[1]; out)
    for backend in (:exact, :finite_difference),
            flow in (BouncyParticle(1, 0.0), ZigZag(1), ZigZag(2))
        d = length(flow.μ)
        model = PDMPModel(d, FullGradient(grad!), hvp!)
        strategy = GridThinningStrategy(N=2, N_min=2, t_max=5.0,
            bound=:constant, curvature_backend=backend, bound_violation=:throw)
        state, model_, alg, cache, stats = PDMPSamplers.initialize_state(
            Random.Xoshiro(1), flow, model, strategy, 0.0,
            SkeletonPoint(zeros(d), ones(d)); statistic_counter=PDMPSamplers.DevelStatisticCounter)
        provider = PDMPSamplers._grid_event_provider(model_, flow, alg, stats)
        PDMPSamplers.construct_rate_bound_grid!(alg.pcb, state, flow, provider, nothing)
        maximum_x = (13-sqrt(73))/3
        maximum_rate = (maximum_x-1)*(maximum_x-2)*(maximum_x-10)
        @test alg.pcb.Λ_vals[1] >= maximum_rate > 0
        budget = 1.5 * alg.pcb.Λ_vals[1]
        time, event, _ = PDMPSamplers.next_event_time(
            TangentScriptedRNG([budget], [0.0]), model_, flow, alg, state,
            cache, stats, 5.0, false)
        @test event === :reflect
        @test time ≈ 1.5
        @test stats.grid_bound_violations == 0
    end
end
