@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function allocation_contract_grad!(out, x)
    copyto!(out, x)
    return out
end

function allocation_contract_hvp!(out, x, v)
    copyto!(out, v)
    return out
end

allocation_contract_rate(state, flow, provider) =
    PDMPSamplers.rate_and_derivative(state, flow, provider)

allocation_contract_grid_rate(state, flow, provider) =
    PDMPSamplers.get_rate_and_deriv(state, flow, provider, false)

allocation_contract_next_event!(rng, model, flow, alg, state, cache, stats) =
    PDMPSamplers.next_event_time(rng, model, flow, alg, state, cache, stats, 1.0, true, :horizon_hit)

function allocation_contract_setup(flow)
    d = 4
    model = PDMPModel(d, FullGradient(allocation_contract_grad!), allocation_contract_hvp!)
    ξ = SkeletonPoint(randn(d), randn(d))
    alg = GridThinningStrategy(; N=8, lazy=false)
    return PDMPSamplers.initialize_state(Xoshiro(1), flow, model, alg, 0.0, ξ)
end

@testset "Hot-loop allocation contract" begin
    @testset "Boomerang rate derivatives" begin
        state, model, alg, cache, stats = allocation_contract_setup(Boomerang(4))
        provider = alg.grad_hvp_provider
        flow = provider.grad.flow
        for _ in 1:5
            allocation_contract_rate(state, flow, provider)
            allocation_contract_grid_rate(state, flow, provider)
        end
        @test @allocated(allocation_contract_rate(state, flow, provider)) == 0
        @test @allocated(allocation_contract_grid_rate(state, flow, provider)) == 0
    end

    @testset "GridThinning event proposal" begin
        flow = BouncyParticle(4)
        state, model, alg, cache, stats = allocation_contract_setup(flow)
        rng = Xoshiro(2)
        for _ in 1:5
            allocation_contract_next_event!(rng, model, flow, alg, state, cache, stats)
        end
        @test @allocated(allocation_contract_next_event!(rng, model, flow, alg, state, cache, stats)) == 0
    end
end
