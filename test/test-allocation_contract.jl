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

function allocation_contract_next_event!(rng, model, flow, alg, state, cache, stats)
    PDMPSamplers.next_event_time(rng, model, flow, alg, state, cache, stats, 1.0, true, :horizon_hit)
    return nothing
end

function allocation_contract_setup(flow)
    d = 4
    model = PDMPModel(d, FullGradient(allocation_contract_grad!), allocation_contract_hvp!)
    ξ = SkeletonPoint(randn(d), randn(d))
    alg = GridThinningStrategy(; N=8, lazy=false)
    return PDMPSamplers.initialize_state(Xoshiro(1), flow, model, alg, 0.0, ξ)
end

struct AllocationContractResidualOracle
    residual_bound::Float64
    subset_bound::Float64
    residual::Float64
end

function (oracle::AllocationContractResidualOracle)(out, x, subset, anchor)
    out[1] = oracle.residual
    return out
end

PDMPSamplers.subsampling_residual_subset_bound(
    oracle::AllocationContractResidualOracle, state, gradient, flow,
    D, M, subset, scale) = oracle.residual_bound
PDMPSamplers.subsampling_subset_bound(
    oracle::AllocationContractResidualOracle, state, gradient, flow,
    D, M, subset, scale) = oracle.subset_bound

function allocation_contract_subsampled_setup(oracle;
        point_scale=1.0, cell_scale=1.0, bound=nothing,
        curvature_bound=nothing)
    point_scale! = let point_scale=point_scale
        (out, state, flow, t) -> (out[1] = point_scale)
    end
    cell_scale! = let cell_scale=cell_scale
        (out, state, flow, left, right) -> (out[1] = cell_scale)
    end
    envelope = SeparableResidualEnvelope(ones(1, 4), point_scale!;
        component_cell_scales! = cell_scale!)
    cv = SubsampledControlVariate(
        (out, x) -> (out[1] = 0.0), oracle, envelope, [0.0], 2;
        deterministic_hvp! = ((out, x, v) -> (out[1] = 0.0)))
    rng = Xoshiro(0x0a110c)
    flow = BouncyParticle(1, 0.0)
    state, model, alg, cache, stats = PDMPSamplers.initialize_state(
        rng, flow, PDMPModel(1, cv),
        GridThinningStrategy(N=1, N_min=1, t_max=1.0, lazy=false,
            bound=bound, curvature_bound=curvature_bound,
            bound_violation=:throw), 0.0, SkeletonPoint([0.0], [1.0]))
    alg.schedule_frozen[] = true
    return (; rng, flow, state, model, alg, cache, stats)
end

function allocation_contract_draw_subset!(fixture)
    cv = fixture.model.grad
    PDMPSamplers.prepare_residual_sampling!(
        cv.envelope, fixture.state, fixture.flow, 0.0)
    B = PDMPSamplers.screening_residual_bound(
        cv.envelope, fixture.state, fixture.flow, 0.0)
    PDMPSamplers.draw_subset!(fixture.rng, cv, 0.0, B)
    return nothing
end

function allocation_contract_candidate!(fixture, D, B, roof)
    PDMPSamplers._evaluate_subsampling_candidate!(fixture.rng,
        fixture.model.grad, fixture.flow, fixture.state,
        fixture.alg.state_cache2, fixture.cache, fixture.stats,
        fixture.alg, 0.25, D, B, roof)
    return nothing
end

function allocation_contract_subsampled_event!(fixture, horizon=1.0)
    PDMPSamplers.next_event_time(fixture.rng, fixture.model, fixture.flow,
        fixture.alg, fixture.state, fixture.cache, fixture.stats,
        horizon, false, :horizon_hit)
    return nothing
end

function allocation_contract_clear_grid!(fixture)
    @inbounds for i in eachindex(fixture.alg.pcb.Λ_vals)
        fixture.alg.pcb.Λ_vals[i] = 0.0
    end
    return nothing
end

@testset "Hot-loop allocation contract" begin
    @testset "Boomerang rate derivatives" begin
        flow = Boomerang(4)
        state, model, alg, cache, stats = allocation_contract_setup(flow)
        provider = PDMPSamplers._grid_event_provider(model, flow, alg, stats)
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


    @testset "SubsampledControlVariate production branches" begin
        quadratic = allocation_contract_subsampled_setup(
            AllocationContractResidualOracle(0.0, 0.0, 0.0);
            point_scale=0.0, cell_scale=0.0,
            bound=:value_quadratic, curvature_bound=8.0)
        provider = PDMPSamplers._grid_event_provider(
            quadratic.model, quadratic.flow, quadratic.alg, quadratic.stats)
        modes = PDMPSamplers._grid_bound_modes(
            quadratic.alg, quadratic.state, quadratic.flow, provider)
        @test modes.use_single_pass_signed
        n, _ = PDMPSamplers._build_grid_bound_prefix!(quadratic.alg.pcb,
            quadratic.state, quadratic.flow, provider, quadratic.alg,
            quadratic.stats, quadratic.alg.state_cache, 1.0, Inf,
            PDMPSamplers.NoGridBoundaryProbe(), modes)
        @test n == 1
        # The signed-rate line construction uses two inflated endpoint
        # tangents and is conservatively L*h^2/4 for this zero-rate fixture.
        @test quadratic.alg.pcb.Λ_vals[1] == 2.0
        fixtures = (
            selected_reject = allocation_contract_subsampled_setup(
                AllocationContractResidualOracle(0.0, 0.0, 0.0)),
            final_reject = allocation_contract_subsampled_setup(
                AllocationContractResidualOracle(4.0, 4.0, 0.0)),
            accepted = allocation_contract_subsampled_setup(
                AllocationContractResidualOracle(4.0, 4.0, 2.0)),
            aggregate_reject = allocation_contract_subsampled_setup(
                AllocationContractResidualOracle(0.0, 0.0, 0.0);
                point_scale=0.0, cell_scale=100.0),
            horizon = allocation_contract_subsampled_setup(
                AllocationContractResidualOracle(0.0, 0.0, 0.0);
                point_scale=0.0, cell_scale=0.0),
        )
        for fixture in values(fixtures)
            allocation_contract_draw_subset!(fixture)
        end
        for _ in 1:5
            allocation_contract_candidate!(fixtures.selected_reject, 0.0, 4.0, 4.0)
            allocation_contract_candidate!(fixtures.final_reject, 0.0, 4.0, 4.0)
            allocation_contract_candidate!(fixtures.accepted, 0.0, 4.0, 4.0)
            allocation_contract_subsampled_event!(fixtures.aggregate_reject)
            allocation_contract_subsampled_event!(fixtures.horizon)
        end
        GC.gc()
        @test @allocated(allocation_contract_draw_subset!(fixtures.accepted)) == 0
        @test @allocated(allocation_contract_clear_grid!(fixtures.horizon)) == 0
        @test @allocated(allocation_contract_candidate!(
            fixtures.selected_reject, 0.0, 4.0, 4.0)) == 0
        @test @allocated(allocation_contract_candidate!(
            fixtures.final_reject, 0.0, 4.0, 4.0)) == 0
        @test @allocated(allocation_contract_candidate!(
            fixtures.accepted, 0.0, 4.0, 4.0)) == 0
        @test @allocated(allocation_contract_subsampled_event!(
            fixtures.aggregate_reject)) == 0
        @test @allocated(allocation_contract_subsampled_event!(
            fixtures.horizon)) == 0
    end
end
