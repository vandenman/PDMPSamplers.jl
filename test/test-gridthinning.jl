@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

const RUN_EXTENDED_GRID_SMOKE_TESTS =
    get(ENV, "PDMPSAMPLERS_EXTENDED_GRID_SMOKE_TESTS", "false") == "true"

struct GridTuningNoopCounter <: PDMPSamplers.AbstractStatisticCounter end

@testset "experimental shared-node bound" begin
    bound_linear = PDMPSamplers._shared_node_cell_bound(NaN, NaN, 0.0, 1.0, 1.0, 2.0, 0.5)
    @test bound_linear == 2.5

    bound_concave = PDMPSamplers._shared_node_cell_bound(0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0)
    @test bound_concave ≈ 1.0

    # This control is deliberately numerical: two zero endpoint rates cannot
    # reveal a positive oscillatory episode hidden inside the first cell.
    hidden_peak_bound = PDMPSamplers._shared_node_cell_bound(
        NaN, NaN, 0.0, 0.0, 1.0, 0.0, 100.0)
    @test hidden_peak_bound == 0.0
    @test sinpi(0.5)^2 > hidden_peak_bound

    @test_throws ArgumentError GridThinningStrategy(; bound=:shared_node, curvature_bound=2.0)
end

import DifferentiationInterface as DI

struct TestCellBound
    calls::Base.RefValue{Int}
end

function (cert::TestCellBound)(state, flow, a, b)
    cert.calls[] += 1
    return 0.0
end

struct TestGridRateDerivatives
    calls::Base.RefValue{Int}
end

function PDMPSamplers.rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    provider::TestGridRateDerivatives,
    state::PDMPSamplers.AbstractPDMPState,
    flow::PDMPSamplers.ContinuousDynamics,
    t_grid::AbstractVector,
    n_points::Integer,
)
    provider.calls[] += 1
    for i in 1:n_points
        values[1, i] = -0.5 + t_grid[i]
        derivatives[1, i] = 1.0
    end
    return values, derivatives
end

struct TestBoomerangLogisticBound
    X::Matrix{Float64}
end

function (cert::TestBoomerangLogisticBound)(state, flow, a, b)
    y = state.ξ.x .- flow.μ
    cubic_sum = 0.0
    quadratic_sum = 0.0
    linear_sum = 0.0
    @inbounds for i in axes(cert.X, 1)
        α = 0.0
        β = 0.0
        for j in axes(cert.X, 2)
            xij = cert.X[i, j]
            α += xij * y[j]
            β += xij * state.ξ.θ[j]
        end
        r = hypot(α, β)
        cubic_sum += r^3
        quadratic_sum += r^2
        linear_sum += r
    end
    L = cubic_sum / (6sqrt(3.0)) + (3 / 8) * quadratic_sum + linear_sum
    return L
end

@testset "Grid thinning" begin

    @testset "PiecewiseConstantBound functor" begin
        t_grid = [0.0, 1.0, 2.0, 3.0]
        Λ_vals = [1.5, 2.0, 0.5]
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, Λ_vals)

        @test pcb(0.5) == 1.5
        @test pcb(1.5) == 2.0
        @test pcb(2.5) == 0.5
        # Outside bounds
        @test pcb(-0.5) == 0.0
        @test pcb(3.5) == 0.0
    end

    @testset "recompute_time_grid!" begin
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [1.0])
        PDMPSamplers.recompute_time_grid!(pcb, 5.0, 10)
        @test length(pcb.t_grid) == 11
        @test pcb.t_grid[1] ≈ 0.0
        @test pcb.t_grid[end] ≈ 5.0
        @test length(pcb.Λ_vals) == 10
    end

    @testset "legacy grid upper bound error path and safety errors" begin
        flow = BouncyParticle(1, 0.0)
        ξ = SkeletonPoint([0.25], [1.0])
        grad!(out, x) = (out[1] = x[1]; out)

        @test_throws ErrorException PDMPSamplers.construct_upper_bound!(
            PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], zeros(1)),
            ξ, flow, grad!, false)

        model = PDMPModel(1, FullGradient(grad!))
        err = try
            PDMPSamplers._throw_grid_safety_limit_error(
                PDMPState(0.0, ξ), flow, model;
                t_invalid=0.0,
                message="Safety limit reached in test")
        catch err
            err
        end
        @test err isa PDMPSamplers._GridSafetyLimitException
        @test err.ctx.t_invalid == eps(Float64)
        @test err.ctx.original_error.msg == "Safety limit reached in test"
        @test err.ctx.flow_type === typeof(flow)
        @test err.ctx.algorithm_type === GridThinningStrategy
    end

    @testset "propose_event_time" begin
        t_grid = [0.0, 1.0, 2.0, 3.0]
        Λ_vals = [2.0, 3.0, 1.0]
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, Λ_vals)

        # Small u → event in first segment
        τ, lb = PDMPSamplers.propose_event_time(pcb, 0.5, 0.0)
        @test 0.0 ≤ τ ≤ 1.0
        @test lb == 2.0

        # Large u → event in late segments or beyond
        τ2, lb2 = PDMPSamplers.propose_event_time(pcb, 100.0, 0.0)
        @test isinf(τ2)

        # With refresh_rate: cell 0 has Λ=2.0, total rate = 3.0
        # u=0.5 should give τ = 0.5 / 3.0 ≈ 0.1667
        τ3, lb3 = PDMPSamplers.propose_event_time(pcb, 0.5, 1.0)
        @test τ3 ≈ 0.5 / 3.0 atol=1e-14
        @test lb3 ≈ 3.0

        # refresh_rate with zero Λ_val: avoid divide-by-zero
        pcb_zero = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [0.0])
        τ4, lb4 = PDMPSamplers.propose_event_time(pcb_zero, 0.25, 1.0)
        @test τ4 ≈ 0.25 / 1.0 atol=1e-14
        @test lb4 ≈ 1.0
    end

    @testset "PiecewiseAffineBound mechanics" begin
        pab = PDMPSamplers.PiecewiseAffineBound(2)
        PDMPSamplers.append_affine_segment!(pab, 0.0, 1.0, 1.0, 2.0)
        PDMPSamplers.append_affine_segment!(pab, 1.0, 3.0, 3.0, -1.0)

        @test pab.n_segments == 2
        @test PDMPSamplers.total_area(pab) ≈ (1.0 * 1.0 + 0.5 * 2.0 * 1.0^2) +
                                               (3.0 * 2.0 + 0.5 * -1.0 * 2.0^2)
        @test pab(0.0) ≈ 1.0
        @test pab(0.5) ≈ 2.0
        @test pab(1.0) ≈ 3.0
        @test pab(2.0) ≈ 2.0
        @test pab(3.0) ≈ 1.0
        @test pab(-0.1) == 0.0
        @test pab(3.1) == 0.0

        PDMPSamplers.reset_affine_bound!(pab)
        @test pab.n_segments == 0
        @test PDMPSamplers.total_area(pab) == 0.0
    end

    @testset "PiecewiseAffineBound inverse integrated hazard" begin
        cases = (
            (2.0, 0.0, 3.0),
            (0.0, 2.0, 3.0),
            (3.0, -1.0, 3.0),
            (1.0, 1e-12, 2.0),
            (1.0, -1e-12, 2.0),
        )

        for (α, β, h) in cases
            full_area = PDMPSamplers._affine_segment_area(α, β, h)
            @test full_area >= 0
            for p in (0.0, 1e-12, 0.1, 0.5, 0.9, 1.0 - 1e-12, 1.0)
                r = p * full_area
                u = PDMPSamplers._invert_affine_segment(α, β, h, r)
                @test 0.0 <= u <= h
                @test PDMPSamplers._affine_segment_area(α, β, u) ≈ r rtol=1e-10 atol=1e-10
            end
        end
    end

    @testset "PiecewiseAffineBound proposal and zero-area segments" begin
        pab = PDMPSamplers.PiecewiseAffineBound(3)
        PDMPSamplers.append_affine_segment!(pab, 0.0, 1.0, 0.0, 0.0)
        PDMPSamplers.append_affine_segment!(pab, 1.0, 2.0, 2.0, 0.0)
        PDMPSamplers.append_affine_segment!(pab, 2.0, 3.0, 2.0, -1.0)

        τ, lb = PDMPSamplers.propose_event_time(pab, 1.0)
        @test τ ≈ 1.5
        @test lb ≈ 2.0

        τ2, lb2 = PDMPSamplers.propose_event_time(pab, 2.5)
        @test 2.0 < τ2 < 3.0
        @test lb2 ≈ pab(τ2)

        τ3, lb3 = PDMPSamplers.propose_event_time(pab, PDMPSamplers.total_area(pab))
        @test isinf(τ3)
        @test lb3 == 0.0

        empty = PDMPSamplers.PiecewiseAffineBound()
        τ4, lb4 = PDMPSamplers.propose_event_time(empty, 0.1)
        @test isinf(τ4)
        @test lb4 == 0.0
    end

    @testset "PiecewiseAffineBound constructors and inverse errors" begin
        pab = PDMPSamplers.PiecewiseAffineBound([0.0, 1.0, 3.0], [1.0, 2.0], [0.0, -0.5])
        @test pab.n_segments == 2
        @test pab(0.5) ≈ 1.0
        @test pab(2.0) ≈ 1.5
        @test PDMPSamplers.total_area(pab) ≈ 1.0 + 3.0

        @test_throws ArgumentError PDMPSamplers.PiecewiseAffineBound([0.0, 1.0], [1.0], [0.0, 1.0])
        @test_throws ArgumentError PDMPSamplers.PiecewiseAffineBound([0.0, 1.0, 2.0], [1.0], [0.0])

        empty = PDMPSamplers.PiecewiseAffineBound(0)
        empty.cum_area = Float64[]
        PDMPSamplers.reset_affine_bound!(empty)
        @test empty.n_segments == 0
        @test empty.cum_area == [0.0]

        @test_throws ArgumentError PDMPSamplers._invert_affine_segment(0.0, 0.0, 1.0, 0.1)
        @test_throws ArgumentError PDMPSamplers._invert_affine_segment(1.0, 0.0, 1.0, 1.1)
        @test_throws ArgumentError PDMPSamplers._invert_affine_segment(0.1, -10.0, 1.0, 0.001)

        τ0, lb0 = PDMPSamplers.propose_event_time(pab, 0.0)
        @test τ0 == 0.0
        @test lb0 == pab(0.0)
        @test_throws ArgumentError PDMPSamplers.propose_event_time(pab, -0.1)
    end

    @testset "PiecewiseAffineBound rejects invalid segments" begin
        pab = PDMPSamplers.PiecewiseAffineBound(1)
        @test_throws ArgumentError PDMPSamplers.append_affine_segment!(pab, 0.0, 0.0, 1.0, 0.0)
        @test_throws ArgumentError PDMPSamplers.append_affine_segment!(pab, 0.0, 1.0, -1.0, 0.0)
        @test_throws ArgumentError PDMPSamplers.append_affine_segment!(pab, 0.0, 1.0, 1.0, -2.0)

        PDMPSamplers.append_affine_segment!(pab, 0.0, 1.0, 1.0, 0.0)
        @test_throws ArgumentError PDMPSamplers.append_affine_segment!(pab, 2.0, 3.0, 1.0, 0.0)
    end

    @testset "hybrid affine roof builder on concave maximum cell" begin
        # λ(t) = 2 - 0.5(t - 1)^2 on [0, 2].
        # Endpoint tangents intersect at the interior maximum.
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 2.0], [2.5])
        pcb.y_vals[1] = 1.5
        pcb.y_vals[2] = 1.5
        pcb.d_vals[1] = 1.0
        pcb.d_vals[2] = -1.0

        pab = PDMPSamplers.PiecewiseAffineBound(2)
        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.build_hybrid_affine_bound!(pab, pcb, 1, stats)

        @test pab.n_segments == 2
        @test stats.affine_roof_cells == 1
        @test stats.affine_constant_cells == 0
        @test PDMPSamplers.total_area(pab) < 5.0
        @test stats.affine_area_hybrid ≈ PDMPSamplers.total_area(pab)
        @test stats.affine_area_constant_equiv ≈ 5.0

        for t in range(0.0, 2.0; length=101)
            λ_true = 2.0 - 0.5 * (t - 1.0)^2
            @test λ_true <= pab(t) + 1e-12
            @test pab(t) <= 2.5 + 1e-12
        end
    end

    @testset "hybrid affine builder falls back for monotone cells" begin
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [2.0])
        pcb.y_vals[1] = 1.0
        pcb.y_vals[2] = 2.0
        pcb.d_vals[1] = 1.0
        pcb.d_vals[2] = 1.0

        pab = PDMPSamplers.PiecewiseAffineBound(2)
        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.build_hybrid_affine_bound!(pab, pcb, 1, stats)

        @test pab.n_segments == 1
        @test stats.affine_roof_cells == 0
        @test stats.affine_constant_cells == 1
        @test pab(0.5) == 2.0
        @test PDMPSamplers.total_area(pab) == 2.0
    end

    @testset "Grid bound normalization and curvature matrix helpers" begin
        @test PDMPSamplers._normalize_grid_bound(nothing) === :constant
        @test PDMPSamplers._normalize_grid_bound(:flat) === :flat
        @test PDMPSamplers._normalize_grid_bound(:linear) === :linear
        @test PDMPSamplers._normalize_grid_bound(:auto) === :auto
        @test PDMPSamplers._normalize_grid_bound(:sticky_auto) === :sticky_auto
        @test_throws ArgumentError PDMPSamplers._normalize_grid_bound(:bogus)

        strat = GridThinningStrategy(; N=3, N_min=1, t_max=2.0, bound=:linear,
            linear_area_threshold=0.8, linear_min_area_gain=0.1)
        shown = sprint(show, strat)
        @test occursin("GridThinningStrategy", shown)
        @test occursin("bound=linear", shown)
        @test occursin("linear_area_threshold=0.8", shown)

        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        flow = BouncyParticle(1, 0.0)
        rng = MersenneTwister(31)
        model = PDMPModel(1, FullGradient((out, x) -> copyto!(out, x)))
        ξ0 = SkeletonPoint([0.0], [1.0])
        _, _, alg_, cache, stats_msg = PDMPSamplers.initialize_state(rng, flow, model, strat, 0.0, ξ0)
        alg_empty = deepcopy(alg_)
        empty!(alg_empty.pcb.t_grid)
        msg_empty = PDMPSamplers._grid_bound_violation_message(
            alg_empty, stats_msg, state, flow, 0.25, 0.5, 1.0, 0.0, 0.2, 0.0, false)
        @test occursin("cell_index=0", msg_empty)
        @test occursin("ratio=Inf", msg_empty)

        alg_.pcb.t_grid .= [0.0, 0.5, 1.0, 1.5]
        alg_.pcb.Λ_vals .= [2.0, 3.0, 4.0]
        alg_.pcb.y_vals .= [1.0, 1.5, 2.0, 2.5]
        alg_.pcb.d_vals .= [0.0, 0.25, 0.5, 0.75]
        alg_.affine_bound.t_breaks[1:3] .= [0.0, 0.4, 1.0]
        alg_.affine_bound.y_left[1:2] .= [1.0, 1.2]
        alg_.affine_bound.slopes[1:2] .= [0.5, 0.75]
        alg_.affine_bound.n_segments = 2
        msg_linear = PDMPSamplers._grid_bound_violation_message(
            alg_, stats_msg, state, flow, 1.0, 0.5, 1.0, 2.0, 0.2, 0.0, true)
        @test occursin("cell_index=3", msg_linear)
        @test occursin("segment_index=2", msg_linear)
        @test occursin("segment_slope=0.75", msg_linear)

        @test all(isnan, PDMPSamplers._metric_scale_extrema(flow))
        @test PDMPSamplers._metric_scale_extrema(PreconditionedZigZag(3; scale=[0.5, 2.0, 1.0])) == (0.5, 2.0)
        dense_bps = DensePreconditionedBPS(2)
        dense_bps.metric.L .= [2.0 0.0; 0.1 3.0]
        @test PDMPSamplers._metric_scale_extrema(dense_bps) == (2.0, 3.0)
        dense_zz = DensePreconditionedZigZag(2)
        dense_zz.metric.L .= [4.0 0.0; 0.2 1.5]
        @test PDMPSamplers._metric_scale_extrema(dense_zz) == (1.5, 4.0)

        t_grid = [0.0, 0.5, 1.0]
        stats = PDMPSamplers.DevelStatisticCounter()
        L_none = PDMPSamplers._channel_curvature_matrix(nothing, state, flow, t_grid, 2, 2, stats)
        @test L_none == zeros(2, 2)
        @test stats.grid_certificate_fallbacks == 4

        L_real = PDMPSamplers._channel_curvature_matrix(1.25, state, flow, t_grid, 2, 2, nothing)
        @test L_real == fill(1.25, 2, 2)

        calls = Ref(0)
        cert = (state, flow, a, b) -> begin
            calls[] += 1
            calls[] == 1 ? 2.0 : 3.0
        end
        stats2 = PDMPSamplers.DevelStatisticCounter()
        L_callable = PDMPSamplers._channel_curvature_matrix(cert, state, flow, t_grid, 2, 2, stats2)
        @test L_callable == [2.0 3.0; 2.0 3.0]
        @test calls[] == 2
        @test stats2.grid_certificate_calls == 2

        prepared = (global_value=nothing, first_value=nothing, has_first=false, cell_values=[4.0, 5.0])
        @test PDMPSamplers._prepared_or_cell_curvature_value(
            prepared, 2, cert, state, flow, 0.5, 1.0, nothing) == 5.0
    end

    @testset "inflated affine builder uses bounded curvature bound" begin
        # λ(t) = 1 + t^2 on [0, 1] has λ'' = 2.  The bounded inflated
        # bound from both endpoints is 1 + t, which dominates λ and is
        # tighter than the constant cap 2.
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [2.0])
        pcb.y_vals[1] = 1.0
        pcb.y_vals[2] = 2.0
        pcb.d_vals[1] = 0.0
        pcb.d_vals[2] = 2.0

        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        flow = BouncyParticle(1, 0.0)
        pab = PDMPSamplers.PiecewiseAffineBound(2)
        stats = PDMPSamplers.DevelStatisticCounter()
        cert = (state, flow, a, b) -> (2.0)
        PDMPSamplers.build_linear_bound!(pab, pcb, 1, state, flow, cert, stats)

        @test pab.n_segments == 1
        @test stats.affine_inflated_cells == 1
        @test stats.affine_constant_cells == 0
        @test PDMPSamplers.total_area(pab) ≈ 1.5
        for t in range(0.0, 1.0; length=21)
            @test 1 + t^2 <= pab(t) + 1e-12
            @test pab(t) ≈ 1 + t
        end
    end

    @testset "inflated affine builder falls back at positive-part kinks" begin
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [1.0])
        pcb.y_vals[1] = 0.0
        pcb.y_vals[2] = 0.0
        pcb.d_vals[1] = 0.0
        pcb.d_vals[2] = 0.0

        state = PDMPState(0.0, SkeletonPoint([0.0], [1.0]))
        flow = BouncyParticle(1, 0.0)
        pab = PDMPSamplers.PiecewiseAffineBound(2)
        stats = PDMPSamplers.DevelStatisticCounter()
        cert = (state, flow, a, b) -> (2.0)
        PDMPSamplers.build_linear_bound!(pab, pcb, 1, state, flow, cert, stats)

        @test pab.n_segments == 1
        @test stats.affine_inflated_cells == 0
        @test stats.affine_constant_cells == 1
        @test pab(0.5) == 1.0
        @test PDMPSamplers.total_area(pab) == 1.0
    end

    @testset "signed inflated builder clips Gaussian zero crossing exactly" begin
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [1.0])
        # Signed Gaussian BPS rate g(t) = x(t)v = -0.5 + t.
        pcb.y_vals[1] = -0.5
        pcb.y_vals[2] = 0.5
        pcb.d_vals[1] = 1.0
        pcb.d_vals[2] = 1.0

        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        flow = BouncyParticle(1, 0.0)
        pab = PDMPSamplers.PiecewiseAffineBound(4)
        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.build_rate_linear_bound!(
            pab, pcb, 1, state, flow,
            (state, flow, a, b) -> (0.0), stats)

        @test stats.affine_inflated_cells == 1
        @test stats.affine_area_saved ≈ 0.875
        @test stats.affine_segments_added == 2
        @test pab.n_segments == 2
        @test pab(0.25) ≈ 0.0 atol=1e-12
        @test pab(0.5) ≈ 0.0 atol=1e-12
        @test pab(0.75) ≈ 0.25 atol=1e-12
        @test PDMPSamplers.total_area(pab) ≈ 0.125
        for t in (0.0, 0.5, 1.0)
            @test pab(t) ≈ max(-0.5 + t, 0.0) atol=1e-12
        end
    end

    @testset "signed inflated builder skips affine cells below absolute area gain" begin
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [0.5])
        pcb.y_vals[1] = -0.5
        pcb.y_vals[2] = 0.5
        pcb.d_vals[1] = 1.0
        pcb.d_vals[2] = 1.0

        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        flow = BouncyParticle(1, 0.0)
        pab = PDMPSamplers.PiecewiseAffineBound(4)
        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.build_rate_linear_bound!(
            pab, pcb, 1, state, flow,
            (0.0), stats;
            linear_area_threshold=1.0,
            linear_min_area_gain=0.4)

        @test stats.affine_inflated_cells == 0
        @test stats.affine_cells_skipped_by_min_gain == 1
        @test stats.affine_constant_cells == 1
        @test stats.affine_segments_added == 1
        @test pab.n_segments == 1
        @test PDMPSamplers.total_area(pab) ≈ 0.5
    end

    @testset "signed inflated Gaussian one-cell flat and affine bounds dominate dense grid" begin
        flow = BouncyParticle(1, 0.0)
        cert = (0.0)

        for (x0, v0, tmax) in ((-0.5, 1.0, 1.0), (0.5, 1.0, 1.0))
            state = PDMPState(0.0, SkeletonPoint([x0], [v0]))
            provider = PDMPSamplers.GradHVPProvider(x -> [x[1]], (x, v) -> [v[1]])

            pcb_flat = PDMPSamplers.PiecewiseConstantBound([0.0, tmax], zeros(1))
            pab_flat = PDMPSamplers.PiecewiseAffineBound(2)
            n_flat = PDMPSamplers.construct_rate_bound_grid!(
                pab_flat, pcb_flat, state, flow, provider, cert;
                build_affine=false,)

            pcb_affine = PDMPSamplers.PiecewiseConstantBound([0.0, tmax], zeros(1))
            pab_affine = PDMPSamplers.PiecewiseAffineBound(4)
            n_affine = PDMPSamplers.construct_rate_bound_grid!(
                pab_affine, pcb_affine, state, flow, provider, cert;
                build_affine=true,)

            @test n_flat == 1
            @test n_affine == 1
            @test pab_affine.n_segments >= 1
            for t in range(0.0, tmax; length=101)
                actual = max((x0 + t * v0) * v0, 0.0)
                @test pcb_flat.Λ_vals[1] + 1e-12 >= actual
                @test pab_affine(t) + 1e-12 >= actual
            end
        end
    end

    @testset "auto chooses flat or affine cells by area gain" begin
        flow = BouncyParticle(1, 0.0)
        cert = (0.0)
        provider = PDMPSamplers.GradHVPProvider(x -> [x[1]], (x, v) -> [v[1]])

        flat_state = PDMPState(0.0, SkeletonPoint([1.0], [0.0]))
        flat_pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], zeros(1))
        flat_pab = PDMPSamplers.PiecewiseAffineBound(2)
        flat_stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.construct_rate_bound_grid!(
            flat_pab, flat_pcb, flat_state, flow, provider, cert;
            stats=flat_stats,
            build_affine=true,
            auto=true,
            linear_area_threshold=0.9)

        @test flat_stats.auto_flat_cells == 1
        @test flat_stats.auto_affine_cells == 0
        @test flat_stats.auto_area_saved == 0.0

        affine_state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        affine_pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], zeros(1))
        affine_pab = PDMPSamplers.PiecewiseAffineBound(4)
        affine_stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.construct_rate_bound_grid!(
            affine_pab, affine_pcb, affine_state, flow, provider, cert;
            stats=affine_stats,
            build_affine=true,
            auto=true,
            linear_area_threshold=0.9)

        @test affine_stats.auto_flat_cells == 0
        @test affine_stats.auto_affine_cells == 1
        @test affine_stats.auto_area_saved > 0
        for t in range(0.0, 1.0; length=101)
            actual = max(-0.5 + t, 0.0)
            @test affine_pab(t) + 1e-12 >= actual
        end
    end

    @testset "signed inflated builder uses flat cell when affine gain is tiny" begin
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [1.0])
        pcb.y_vals[1] = 1.0
        pcb.y_vals[2] = 1.0
        pcb.d_vals[1] = 0.0
        pcb.d_vals[2] = 0.0

        state = PDMPState(0.0, SkeletonPoint([1.0], [1.0]))
        flow = BouncyParticle(1, 0.0)
        pab = PDMPSamplers.PiecewiseAffineBound(4)
        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.build_rate_linear_bound!(
            pab, pcb, 1, state, flow, (0.0), stats;
            linear_area_threshold=0.95)

        @test stats.grid_certificate_calls == 0
        @test stats.affine_inflated_cells == 0
        @test stats.affine_constant_cells == 1
        @test pab.n_segments == 1
        @test PDMPSamplers.total_area(pab) ≈ 1.0
    end

    @testset "single-pass signed inflated grid avoids duplicate endpoint pass" begin
        grad = x -> [x[1]]
        hvp = (x, v) -> [v[1]]
        provider = PDMPSamplers.GradHVPProvider(grad, hvp)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        flow = BouncyParticle(1, 0.0)
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 1.0], [0.0])
        pab = PDMPSamplers.PiecewiseAffineBound(4)
        stats = PDMPSamplers.DevelStatisticCounter()
        cert_calls = Ref(0)
        cert = (state, flow, a, b) -> begin
            cert_calls[] += 1
            (0.0)
        end

        n = PDMPSamplers.construct_rate_bound_grid!(
            pab, pcb, state, flow, provider, cert;
            stats,
            build_affine=true)

        @test n == 1
        @test stats.grid_endpoint_evaluations == 0
        @test stats.grid_endpoint_derivative_calls == 1
        @test stats.grid_certificate_calls == 1
        @test cert_calls[] == 1
        @test stats.grid_certificate_fallbacks == 0
        @test stats.affine_inflated_cells == 1
        @test pab(0.25) ≈ 0.0 atol=1e-12
        @test pab(0.75) ≈ 0.25 atol=1e-12
    end

    @testset "single-pass signed inflated grid accepts callable cell certificates" begin
        grad = x -> [x[1]]
        hvp = (x, v) -> [v[1]]
        provider = PDMPSamplers.GradHVPProvider(grad, hvp)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        flow = BouncyParticle(1, 0.0)
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 0.5, 1.0], zeros(2))
        pab = PDMPSamplers.PiecewiseAffineBound(6)
        stats = PDMPSamplers.DevelStatisticCounter()
        cert = TestCellBound(Ref(0))

        n = PDMPSamplers.construct_rate_bound_grid!(
            pab, pcb, state, flow, provider, cert;
            stats,
            build_affine=true)

        @test n == 2
        @test cert.calls[] == 2
        @test stats.grid_certificate_calls == 2
        @test stats.grid_certificate_fallbacks == 0
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "end-to-end signed inflated Gaussian is exact across zero crossing" begin
        function gaussian_grad!(out, x)
            out[1] = x[1]
            return out
        end
        function gaussian_hvp!(out, x, v)
            out[1] = v[1]
            return out
        end

        model = PDMPModel(1, FullGradient(gaussian_grad!), gaussian_hvp!)
        flow = BouncyParticle(1, 0.0)
        alg = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
            bound=:linear,
            curvature_bound=(state, flow, a, b) -> (0.0),
            bound_violation=:throw)
        ξ0 = SkeletonPoint([-0.5], [1.0])
        rng = Xoshiro(20260630)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        for _ in 1:20
            τ, event_type, meta = PDMPSamplers.next_event_time(rng, model_, flow, alg_, state, cache, stats, Inf, false)
            @test isfinite(τ) || τ === Inf
            event_type === :horizon_hit && break
            PDMPSamplers._handle_event_no_boundary!(rng, τ, model_.grad, flow, alg_, state, cache, event_type, meta, stats)
        end

        @test stats.grid_bound_violations == 0
        @test stats.affine_bound_violations == 0
    end

    @testset "end-to-end inflated constant Gaussian has no proposal-time violations" begin
        function gaussian_grad!(out, x)
            out[1] = x[1]
            return out
        end
        function gaussian_hvp!(out, x, v)
            out[1] = v[1]
            return out
        end

        model = PDMPModel(1, FullGradient(gaussian_grad!), gaussian_hvp!)
        flow = BouncyParticle(1, 0.05)
        alg = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
            bound=:flat,
            curvature_bound=(0.0),
            bound_violation=:throw)
        ξ0 = SkeletonPoint([0.25], [1.0])

        for seed in 20260701:20260702
            rng = Xoshiro(seed)
            state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
                rng, flow, model, alg, 0.0, ξ0;
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
            τ, event_type, meta = PDMPSamplers.next_event_time(
                rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
            @test 0.0 <= τ <= alg_.t_max[]
            @test event_type in (:reflect, :horizon_hit)
            @test meta isa PDMPSamplers.GradientMeta
            @test stats.grid_bound_violations == 0
            @test stats.affine_bound_violations == 0
        end
    end

    @testset "bounded scalar-BPS smoke has no proposal-time violations" begin
        function quartic_grad!(out, x)
            out[1] = x[1]^3 - x[1]
            return out
        end
        function quartic_hvp!(out, x, v)
            out[1] = (3x[1]^2 - 1) * v[1]
            return out
        end
        quartic_cert = (state, flow, a, b) -> begin
            x0 = state.ξ.x[1]
            v0 = state.ξ.θ[1]
            c = 6 * v0^3
            return max(c * (x0 + a * v0), c * (x0 + b * v0))
        end

        function trig_grad!(out, x)
            out[1] = 2sin(x[1]) + 0.1x[1]
            return out
        end
        function trig_hvp!(out, x, v)
            out[1] = (2cos(x[1]) + 0.1) * v[1]
            return out
        end
        trig_cert = (state, flow, a, b) -> 2abs(state.ξ.θ[1])^3

        for (grad!, hvp!, cert, ξ0, T, seed) in (
            (quartic_grad!, quartic_hvp!, quartic_cert, SkeletonPoint([0.25], [1.0]), 1.0, 701),
            (trig_grad!, trig_hvp!, trig_cert, SkeletonPoint([0.75], [1.0]), 1.0, 702),
        )
            model = PDMPModel(1, FullGradient(grad!), hvp!)
            flow = BouncyParticle(1, 0.05)
            alg = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
                bound=:linear,
                curvature_bound=cert,
                bound_violation=:throw,
                linear_area_threshold=0.9)

            rng = Xoshiro(seed)
            state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
                rng, flow, model, alg, 0.0, ξ0;
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
            τ, event_type, meta = PDMPSamplers.next_event_time(
                rng, model_, flow, alg_, state, cache, stats, T, false)
            @test 0.0 <= τ <= max(T, alg_.t_max[])
            @test event_type in (:reflect, :horizon_hit)
            @test meta isa PDMPSamplers.GradientMeta
            @test stats.grid_bound_violations == 0
            @test stats.affine_bound_violations == 0
        end
    end

    @testset "auto scalar-BPS smoke has no proposal-time violations" begin
        function gaussian_grad!(out, x)
            out[1] = x[1]
            return out
        end
        function gaussian_hvp!(out, x, v)
            out[1] = v[1]
            return out
        end
        gaussian_cert = (0.0)

        function quartic_grad!(out, x)
            out[1] = x[1]^3 - x[1]
            return out
        end
        function quartic_hvp!(out, x, v)
            out[1] = (3x[1]^2 - 1) * v[1]
            return out
        end
        quartic_cert = (state, flow, a, b) -> begin
            x0 = state.ξ.x[1]
            v0 = state.ξ.θ[1]
            c = 6 * v0^3
            return max(c * (x0 + a * v0), c * (x0 + b * v0))
        end

        function trig_grad!(out, x)
            out[1] = 2sin(x[1]) + 0.1x[1]
            return out
        end
        function trig_hvp!(out, x, v)
            out[1] = (2cos(x[1]) + 0.1) * v[1]
            return out
        end
        trig_cert = (state, flow, a, b) -> 2abs(state.ξ.θ[1])^3

        for (grad!, hvp!, cert, ξ0, seed) in (
            (gaussian_grad!, gaussian_hvp!, gaussian_cert, SkeletonPoint([0.25], [1.0]), 711),
            (quartic_grad!, quartic_hvp!, quartic_cert, SkeletonPoint([0.25], [1.0]), 712),
            (trig_grad!, trig_hvp!, trig_cert, SkeletonPoint([0.75], [1.0]), 713),
        )
            model = PDMPModel(1, FullGradient(grad!), hvp!)
            flow = BouncyParticle(1, 0.05)
            alg = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
                bound=:auto,
                curvature_bound=cert,
                bound_violation=:throw,
                linear_area_threshold=0.9)

            rng = Xoshiro(seed)
            state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
                rng, flow, model, alg, 0.0, ξ0;
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
            τ, event_type, meta = PDMPSamplers.next_event_time(
                rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
            @test 0.0 <= τ <= alg_.t_max[]
            @test event_type in (:reflect, :horizon_hit)
            @test meta isa PDMPSamplers.GradientMeta
            @test stats.grid_bound_violations == 0
            @test stats.affine_bound_violations == 0
            @test stats.auto_flat_cells + stats.auto_affine_cells > 0
        end
    end
    end

    @testset "budget-first tail restart offsets returned event time" begin
        function neg_gaussian_grad!(out, x)
            out[1] = -x[1]
            return out
        end
        function neg_gaussian_hvp!(out, x, v)
            out[1] = -v[1]
            return out
        end

        model = PDMPModel(1, FullGradient(neg_gaussian_grad!), neg_gaussian_hvp!)
        flow = BouncyParticle(1, 0.0)
        alg = GridThinningStrategy(; N=1, N_min=1, t_max=2.0, lazy=false,
            bound=:flat,
            curvature_bound=(0.0),
            bound_violation=:throw,
            max_rejections_before_tail_restart=1,
            safety_limit=20)
        ξ0 = SkeletonPoint([-1.0], [1.0])
        rng = Xoshiro(3)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(rng, model_, flow, alg_, state, cache, stats, Inf, false)

        @test event_type === :horizon_hit
        @test τ > alg.t_max
        @test stats.grid_budget_tail_restarts > 0
        @test alg_.pcb.t_grid[1] == 0.0
        @test stats.grid_bound_violations == 0
        @test stats.affine_bound_violations == 0
    end

    @testset "componentwise bounded ZigZag Gaussian channels" begin
        flow = ZigZag(2)
        state = PDMPState(0.0, SkeletonPoint([-0.5, 0.25], [1.0, -1.0]))
        t_grid = collect(range(0.0, 1.0, 5))
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(length(t_grid) - 1))
        pab = PDMPSamplers.PiecewiseAffineBound(8)
        grad = x -> copy(x)
        hvp = (x, v) -> copy(v)
        cert = (0.0)

        @test PDMPSamplers._rate_aggregation(flow) === :componentwise
        @test PDMPSamplers._can_use_signed_grid(state, flow, PDMPSamplers.GradHVPProvider(grad, hvp))
        @test !PDMPSamplers._can_use_signed_grid(state, flow, PDMPSamplers.GradientOnlyProvider(grad))
        G, dG = PDMPSamplers.rate_derivatives_for_grid(
            PDMPSamplers.GradHVPProvider(grad, hvp), state, flow, t_grid, length(t_grid))
        @test size(G) == (2, length(t_grid))
        @test all(dG .≈ 1.0)

        stats = PDMPSamplers.DevelStatisticCounter()
        n = PDMPSamplers.construct_rate_bound_grid!(
            pab, pcb, state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), cert;
            build_affine=true,
            stats)
        @test n == length(pcb.Λ_vals)

        flat_area = sum(pcb.Λ_vals[i] * (t_grid[i + 1] - t_grid[i])
            for i in eachindex(pcb.Λ_vals))
        @test PDMPSamplers.total_area(pab) <= flat_area + 1e-12
        @test stats.componentwise_affine_cells > 0
        @test stats.componentwise_affine_segments_added > 0

        for cell in eachindex(pcb.Λ_vals)
            a, b = t_grid[cell], t_grid[cell + 1]
            for t in range(a, b; length=11)
                channels = state.ξ.θ .* (state.ξ.x .+ t .* state.ξ.θ)
                positive_channels = max.(channels, 0.0)
                @test all(positive_channels[j] <= pcb.Λ_vals[cell] + 1e-12 for j in 1:2)
                @test sum(positive_channels) <= pcb.Λ_vals[cell] + 1e-12
                @test sum(positive_channels) <= pab(t) + 1e-12
            end
        end
    end

    @testset "componentwise ZigZag affine aggregate handles zero crossings" begin
        flow = ZigZag(2)
        state = PDMPState(0.0, SkeletonPoint([-0.5, 0.25], [1.0, -1.0]))
        t_grid = [0.0, 1.0]
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, [0.0])
        pab = PDMPSamplers.PiecewiseAffineBound(8)
        grad = x -> copy(x)
        hvp = (x, v) -> copy(v)
        cert = (0.0)
        stats = PDMPSamplers.DevelStatisticCounter()

        PDMPSamplers.construct_rate_bound_grid!(
            pab, pcb, state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), cert;
            build_affine=true,
            stats)

        @test pab.n_segments == 3
        @test stats.componentwise_flat_fallback_cells == 0
        @test stats.componentwise_affine_cells == 1
        @test PDMPSamplers.total_area(pab) <= pcb.Λ_vals[1] + 1e-12
        for t in range(0.0, 1.0; length=41)
            channels = state.ξ.θ .* (state.ξ.x .+ t .* state.ξ.θ)
            @test sum(max.(channels, 0.0)) <= pab(t) + 1e-12
        end
    end

    @testset "componentwise ZigZag high-dimensional crossing diagnostic" begin
        d = 100
        flow = ZigZag(d)
        θ = ones(d)
        x = [-i / (d + 1) for i in 1:d]
        state = PDMPState(0.0, SkeletonPoint(x, θ))
        t_grid = [0.0, 1.0]
        grad = x -> copy(x)
        hvp = (x, v) -> copy(v)
        cert = (0.0)
        crossings = count(i -> 0.0 < -θ[i] * x[i] < 1.0, 1:d)

        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, [0.0])
        pab = PDMPSamplers.PiecewiseAffineBound(2d + 2)
        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.construct_rate_bound_grid!(
            pab, pcb, state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), cert;
            build_affine=true,
            stats,
            max_componentwise_affine_segments_per_cell=256)

        @test crossings == d
        @test pab.n_segments == crossings + 1
        @test stats.componentwise_flat_fallback_cells == 0
        @test PDMPSamplers.total_area(pab) <= pcb.Λ_vals[1] + 1e-12
        for t in range(0.0, 1.0; length=51)
            exact = sum(max(x[i] + t, 0.0) for i in 1:d)
            @test exact <= pab(t) + 1e-10
        end

        pcb_cap = PDMPSamplers.PiecewiseConstantBound(t_grid, [0.0])
        pab_cap = PDMPSamplers.PiecewiseAffineBound(8)
        stats_cap = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.construct_rate_bound_grid!(
            pab_cap, pcb_cap, state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), cert;
            build_affine=true,
            stats=stats_cap,
            max_componentwise_affine_segments_per_cell=32)

        @test stats_cap.componentwise_flat_fallback_cells == 1
        @test pab_cap.n_segments == 1
        @test PDMPSamplers.total_area(pab_cap) ≈ pcb_cap.Λ_vals[1]
    end

    @testset "Boomerang rate derivatives use corrected-gradient convention" begin
        Γ = Diagonal([1.5, 2.0, 2.5])
        μ = [0.2, -0.3, 0.4]
        flow = Boomerang(Γ, μ, 0.0)
        x = [0.7, -0.1, -0.2]
        θ = [0.4, -0.5, 0.3]
        state = PDMPState(0.0, SkeletonPoint(copy(x), copy(θ)))
        corrected_grad = x -> zeros(length(x))
        raw_hvp = (x, v) -> Γ * v

        @test PDMPSamplers._rate_aggregation(flow) === :scalar
        provider = PDMPSamplers.GradHVPProvider(corrected_grad, raw_hvp)
        @test PDMPSamplers._can_use_signed_grid(state, flow, provider)
        g, dg = PDMPSamplers.rate_and_derivative(
            state, flow, provider)
        @test g ≈ 0.0 atol=1e-12
        @test dg ≈ 0.0 atol=1e-12
    end

    @testset "Boomerang quadratic residual rate derivatives and certificate" begin
        Γ = Diagonal([1.2, 1.6, 2.1])
        μ = [0.1, -0.2, 0.3]
        A = Symmetric([0.7 0.1 -0.05; 0.1 0.5 0.02; -0.05 0.02 0.4])
        b = [0.3, -0.2, 0.15]
        flow = Boomerang(Γ, μ, 0.0)
        x = [0.8, -0.4, 0.1]
        θ = [0.25, -0.7, 0.35]
        state = PDMPState(0.0, SkeletonPoint(copy(x), copy(θ)))
        corrected_grad = x -> A * (x .- μ) .+ b
        raw_hvp = (x, v) -> (Γ + A) * v

        y = x .- μ
        a = A * y .+ b
        g_expected = dot(a, θ)
        dg_expected = dot(θ, A * θ) - dot(a, y)
        provider = PDMPSamplers.GradHVPProvider(corrected_grad, raw_hvp)
        g, dg = PDMPSamplers.rate_and_derivative(state, flow, provider)
        @test g ≈ g_expected atol=1e-12
        @test dg ≈ dg_expected atol=1e-12

        R = sqrt(dot(y, y) + dot(θ, θ))
        Lcert = 2opnorm(Matrix(A)) * R^2 + norm(b) * R
        for t in range(0.0, 2π; length=31)
            st = copy(state)
            move_forward_time!(st, t, flow)
            yt = st.ξ.x .- μ
            gdd = -4dot(yt, A * st.ξ.θ) - dot(b, st.ξ.θ)
            @test abs(gdd) <= Lcert + 1e-10
        end
    end

    @testset "Boomerang signed derivative matches sampler convention" begin
        Γ = Diagonal([1.2, 1.6, 2.0])
        μ = [0.1, -0.25, 0.3]
        A = Symmetric([0.6 0.12 -0.04; 0.12 0.4 0.08; -0.04 0.08 0.5])
        b = [0.2, -0.3, 0.1]
        flow = Boomerang(Γ, μ, 0.0)
        x = [0.55, -0.45, 0.05]
        θ = [0.35, -0.2, 0.45]
        state = PDMPState(0.0, SkeletonPoint(copy(x), copy(θ)))

        function residual_grad!(out, x)
            y = x .- μ
            out .= Γ * y .+ A * y .+ b
            return out
        end
        function residual_hvp!(out, x, v)
            out .= (Γ + A) * v
            return out
        end
        model = PDMPModel(3, FullGradient(residual_grad!), residual_hvp!)
        alg = GridThinningStrategy()

        function actual_rate(t)
            st = copy(state)
            move_forward_time!(st, t, flow)
            cache = PDMPSamplers.initialize_cache(Xoshiro(31), flow, model.grad, alg, st.t[], st.ξ)
            cache = PDMPSamplers.add_gradient_to_cache(cache, st.ξ)
            ∇corr = compute_gradient!(st, model.grad, flow, cache)
            return dot(∇corr, st.ξ.θ)
        end

        corrected_grad = x -> A * (x .- μ) .+ b
        raw_hvp = (x, v) -> (Γ + A) * v
        provider = PDMPSamplers.GradHVPProvider(corrected_grad, raw_hvp)
        g, dg = PDMPSamplers.rate_and_derivative(state, flow, provider)
        h = 1e-6
        fd = (actual_rate(h) - actual_rate(-h)) / (2h)

        @test g ≈ actual_rate(0.0) atol=1e-12
        @test dg ≈ fd rtol=1e-7 atol=1e-8
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "Boomerang signed grid accepts optional generic curvature bound" begin
        Γ = Diagonal([1.0, 1.4])
        μ = [0.0, 0.1]
        A = Symmetric([0.3 0.05; 0.05 0.2])
        b = [0.15, -0.1]
        flow = Boomerang(Γ, μ, 0.0)
        function residual_grad!(out, x)
            y = x .- μ
            out .= Γ * y .+ A * y .+ b
            return out
        end
        function residual_hvp!(out, x, v)
            out .= (Γ + A) * v
            return out
        end
        model = PDMPModel(2, FullGradient(residual_grad!), residual_hvp!)
        ξ0 = SkeletonPoint([0.25, -0.2], [0.4, -0.3])
        generic = GridThinningStrategy(;
            N=2,
            bound=:flat,
            curvature_bound=(1.0),
            lazy=false)
        adaptive = GridThinningStrategy(;
            N=2,
            bound=:flat,
            lazy=false)
        bounded = GridThinningStrategy(;
            N=2,
            bound=:flat,
            curvature_bound=(10.0),
            bound_violation=:throw,
            lazy=false)

        _, generic_stats = pdmp_sample(
            ξ0, flow, model, generic, 0.0, 0.5;
            seed=32,
            progress=false,
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        _, adaptive_stats = pdmp_sample(
            ξ0, flow, model, adaptive, 0.0, 0.5;
            seed=34,
            progress=false,
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        _, stats = pdmp_sample(
            ξ0, flow, model, bounded, 0.0, 0.5;
            seed=33,
            progress=false,
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test generic_stats.grid_bound_violations == 0
        @test adaptive_stats.grid_bound_violations >= 0
        @test stats.grid_bound_violations == 0
        @test stats.grid_certificate_fallbacks == 0
    end

    @testset "bounded Boomerang quadratic residual has no violations" begin
        Γ = Diagonal([1.1, 1.7])
        μ = [0.05, -0.15]
        A = Symmetric([0.4 0.08; 0.08 0.3])
        b = [0.12, -0.2]
        flow = Boomerang(Γ, μ, 0.0)
        function residual_grad!(out, x)
            y = x .- μ
            out .= Γ * y .+ A * y .+ b
            return out
        end
        function residual_hvp!(out, x, v)
            out .= (Γ + A) * v
            return out
        end
        model = PDMPModel(2, FullGradient(residual_grad!), residual_hvp!)
        ξ0 = SkeletonPoint([0.3, -0.45], [0.5, -0.25])
        cert = (25.0)

        for bound in (:flat, :linear, :auto)
            alg = GridThinningStrategy(;
                N=2,
                N_min=1,
                t_max=0.5,
                bound,
                curvature_bound=cert,
                linear_area_threshold=1.0,
                linear_min_area_gain=0.0,
                bound_violation=:throw,
                lazy=false)
            _, stats = pdmp_sample(
                ξ0, flow, model, alg, 0.0, 1.0;
                seed=34,
                progress=false,
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
            @test stats.grid_bound_violations == 0
            @test stats.affine_bound_violations == 0
            @test stats.grid_certificate_fallbacks == 0
        end
    end

    @testset "exact-reference Boomerang has no reflection events" begin
        Γ = Diagonal([1.0, 1.6])
        μ = [0.2, -0.1]
        flow = Boomerang(Γ, μ, 0.25)
        function reference_grad!(out, x)
            out .= Γ * (x .- μ)
            return out
        end
        function reference_hvp!(out, x, v)
            out .= Γ * v
            return out
        end
        model = PDMPModel(2, FullGradient(reference_grad!), reference_hvp!)
        alg = GridThinningStrategy(;
            N=2,
            bound=:flat,
            curvature_bound=(0.0),
            bound_violation=:throw,
            lazy=false)
        ξ0 = SkeletonPoint([0.45, -0.35], [0.25, 0.4])
        _, stats = pdmp_sample(
            ξ0, flow, model, alg, 0.0, 2.0;
            seed=35,
            progress=false,
            statistic_counter=PDMPSamplers.DevelStatisticCounter)

        @test stats.reflections_events == 0
        @test stats.reflections_accepted == 0
        @test stats.grid_bound_violations == 0
        @test stats.grid_certificate_fallbacks == 0
    end
    end

    @testset "Boomerang logistic certificate matches sampler convention" begin
        X = [0.4 -0.2 0.1; -0.3 0.5 0.2; 0.2 0.1 -0.4; -0.1 -0.3 0.35]
        yobs = [1.0, 0.0, 1.0, 0.0]
        Γ = Diagonal([1.3, 1.6, 2.0])
        μ = [0.1, -0.2, 0.05]
        flow = Boomerang(Γ, μ, 0.0)
        x0 = [0.35, -0.45, 0.2]
        θ0 = [0.25, -0.15, 0.4]
        state = PDMPState(0.0, SkeletonPoint(copy(x0), copy(θ0)))
        sigmoid(z) = inv(1 + exp(-z))

        function logistic_grad!(out, x)
            η = X * x
            w = sigmoid.(η) .- yobs
            mul!(out, transpose(X), w)
            out .+= Γ * (x .- μ)
            return out
        end
        function logistic_hvp!(out, x, v)
            η = X * x
            Xv = X * v
            w = sigmoid.(η) .* (1 .- sigmoid.(η)) .* Xv
            mul!(out, transpose(X), w)
            out .+= Γ * v
            return out
        end
        model = PDMPModel(3, FullGradient(logistic_grad!), logistic_hvp!)
        alg = GridThinningStrategy()

        function actual_rate(t)
            st = copy(state)
            move_forward_time!(st, t, flow)
            cache = PDMPSamplers.initialize_cache(Xoshiro(41), flow, model.grad, alg, st.t[], st.ξ)
            cache = PDMPSamplers.add_gradient_to_cache(cache, st.ξ)
            ∇corr = compute_gradient!(st, model.grad, flow, cache)
            return dot(∇corr, st.ξ.θ)
        end

        corrected_grad = x -> begin
            η = X * x
            vec(transpose(X) * (sigmoid.(η) .- yobs))
        end
        raw_hvp = (x, v) -> begin
            η = X * x
            Xv = X * v
            w = sigmoid.(η) .* (1 .- sigmoid.(η)) .* Xv
            vec(transpose(X) * w .+ Γ * v)
        end
        provider = PDMPSamplers.GradHVPProvider(corrected_grad, raw_hvp)
        g, dg = PDMPSamplers.rate_and_derivative(state, flow, provider)
        h = 1e-6
        fd = (actual_rate(h) - actual_rate(-h)) / (2h)
        @test g ≈ actual_rate(0.0) atol=1e-12
        @test dg ≈ fd rtol=1e-7 atol=1e-8

        cert = TestBoomerangLogisticBound(X)
        L = cert(state, flow, 0.0, 1.0)
        for t in range(0.0, 2π; length=41)
            st = copy(state)
            move_forward_time!(st, t, flow)
            gdd = 0.0
            for i in axes(X, 1)
                z = dot(view(X, i, :), st.ξ.x)
                q = dot(view(X, i, :), st.ξ.θ)
                u = dot(view(X, i, :), st.ξ.x .- μ)
                s = sigmoid(z)
                gdd += s * (1 - s) * (1 - 2s) * q^3
                gdd -= 3s * (1 - s) * q * u
                gdd -= (s - yobs[i]) * q
            end
            @test abs(gdd) <= L + 1e-10
        end

        t_grid = collect(range(0.0, 1.0, 5))
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(length(t_grid) - 1))
        pab = PDMPSamplers.PiecewiseAffineBound(16)
        PDMPSamplers.construct_rate_bound_grid!(
            pab, pcb, state, flow, provider, cert;
            build_affine=true,
            stats=PDMPSamplers.DevelStatisticCounter())
        for cell in eachindex(pcb.Λ_vals), t in range(t_grid[cell], t_grid[cell + 1]; length=11)
            @test max(actual_rate(t), 0.0) <= pcb.Λ_vals[cell] + 1e-10
            @test max(actual_rate(t), 0.0) <= pab(t) + 1e-10
        end
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "bounded Boomerang logistic smoke has no violations" begin
        rng = Xoshiro(42)
        n, p = 100, 5
        X = randn(rng, n, p) ./ sqrt(p)
        β_true = range(-0.4, 0.4; length=p)
        yobs = Float64.(rand.(Ref(rng), Bernoulli.(inv.(1 .+ exp.(-(X * β_true))))))
        Γ = Diagonal(fill(1.25, p))
        μ = zeros(p)
        flow = Boomerang(Γ, μ, 0.05)
        sigmoid(z) = inv(1 + exp(-z))

        function logistic_grad!(out, x)
            η = X * x
            w = sigmoid.(η) .- yobs
            mul!(out, transpose(X), w)
            out .+= Γ * (x .- μ)
            return out
        end
        function logistic_hvp!(out, x, v)
            η = X * x
            Xv = X * v
            s = sigmoid.(η)
            mul!(out, transpose(X), s .* (1 .- s) .* Xv)
            out .+= Γ * v
            return out
        end

        model = PDMPModel(p, FullGradient(logistic_grad!), logistic_hvp!)
        cert = TestBoomerangLogisticBound(X)
        ξ0 = SkeletonPoint(fill(0.05, p), collect(range(-0.4, 0.4; length=p)))
        for bound in (:flat, :linear, :auto)
            alg = GridThinningStrategy(;
                N=2,
                N_min=1,
                t_max=1.0,
                bound,
                curvature_bound=cert,
                linear_area_threshold=1.0,
                linear_min_area_gain=0.0,
                bound_violation=:throw,
                lazy=false)
            rng = Xoshiro(44)
            state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
                rng, flow, model, alg, 0.0, ξ0;
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
            τ, event_type, meta = PDMPSamplers.next_event_time(
                rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
            @test 0.0 <= τ <= alg_.t_max[]
            @test event_type in (:reflect, :horizon_hit)
            @test meta isa PDMPSamplers.GradientMeta
            @test stats.grid_bound_violations == 0
            @test stats.affine_bound_violations == 0
            @test stats.grid_certificate_fallbacks == 0
        end
    end
    end

    @testset "preconditioned signed-rate geometry conventions" begin
        d = 3
        grad = x -> copy(x)
        hvp = (x, v) -> copy(v)
        state = PDMPState(0.0, SkeletonPoint([0.4, -0.3, 0.2], [1.2, -0.8, 0.5]))

        dbps = PreconditionedBPS(d; refresh_rate=0.0, scale=[0.5, 1.5, 2.0])
        @test PDMPSamplers._rate_aggregation(dbps) === :scalar
        @test PDMPSamplers._can_use_signed_grid(state, dbps, PDMPSamplers.GradHVPProvider(grad, hvp))
        g, dg = PDMPSamplers.rate_and_derivative(state, dbps, PDMPSamplers.GradHVPProvider(grad, hvp))
        @test g ≈ dot(state.ξ.x, state.ξ.θ)
        @test dg ≈ dot(state.ξ.θ, state.ξ.θ)

        dense_bps = DensePreconditionedBPS(d; refresh_rate=0.0)
        @test PDMPSamplers._rate_aggregation(dense_bps) === :scalar
        @test PDMPSamplers._can_use_signed_grid(state, dense_bps, PDMPSamplers.GradHVPProvider(grad, hvp))
        g_dense, dg_dense = PDMPSamplers.rate_and_derivative(
            state, dense_bps, PDMPSamplers.GradHVPProvider(grad, hvp))
        @test g_dense ≈ dot(state.ξ.x, state.ξ.θ)
        @test dg_dense ≈ dot(state.ξ.θ, state.ξ.θ)

        dzz = PreconditionedZigZag(d; scale=[0.5, 1.5, 2.0])
        @test PDMPSamplers._rate_aggregation(dzz) === :componentwise
        @test PDMPSamplers._can_use_signed_grid(state, dzz, PDMPSamplers.GradHVPProvider(grad, hvp))
        G_diag, dG_diag = PDMPSamplers.rate_derivatives_for_grid(
            PDMPSamplers.GradHVPProvider(grad, hvp), state, dzz, [0.0], 1)
        @test vec(G_diag[:, 1]) ≈ state.ξ.θ .* state.ξ.x
        @test vec(dG_diag[:, 1]) ≈ state.ξ.θ .* state.ξ.θ

        L = [1.0 0.0 0.0; 0.25 1.1 0.0; -0.2 0.15 0.9]
        dense = PDMPSamplers.DensePreconditioner(d)
        dense.L .= L
        dense.Linv .= inv(LowerTriangular(L))
        flow = PDMPSamplers.PreconditionedDynamics(dense, ZigZag(d))
        v = [1.0, -1.0, 1.0]
        copyto!(flow.metric.v_canonical, v)
        θ = L * v
        zz_state = PDMPState(0.0, SkeletonPoint([0.3, -0.4, 0.2], θ))
        @test PDMPSamplers._rate_aggregation(flow) === :componentwise
        @test PDMPSamplers._can_use_signed_grid(zz_state, flow, PDMPSamplers.GradHVPProvider(grad, hvp))
        G, dG = PDMPSamplers.rate_derivatives_for_grid(
            PDMPSamplers.GradHVPProvider(grad, hvp), zz_state, flow, [0.0], 1)
        η = L' * zz_state.ξ.x
        Hθ_z = L' * θ
        @test vec(G[:, 1]) ≈ v .* η
        @test vec(dG[:, 1]) ≈ v .* Hθ_z
        @test sum(max.(G[:, 1], 0.0)) ≈ PDMPSamplers.λ(zz_state.ξ, zz_state.ξ.x, flow)
    end

    @testset "dense preconditioned ZigZag bounded aggregate dominates rate" begin
        d = 3
        L = [1.0 0.0 0.0; 0.2 1.0 0.0; -0.1 0.3 0.8]
        dense = PDMPSamplers.DensePreconditioner(d)
        dense.L .= L
        dense.Linv .= inv(LowerTriangular(L))
        flow = PDMPSamplers.PreconditionedDynamics(dense, ZigZag(d))
        v = [1.0, -1.0, 1.0]
        copyto!(flow.metric.v_canonical, v)
        θ = L * v
        state = PDMPState(0.0, SkeletonPoint([-0.4, 0.25, -0.1], θ))
        grad = x -> copy(x)
        hvp = (x, v) -> copy(v)
        cert = (0.0)
        t_grid = collect(range(0.0, 1.0, 4))
        pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(length(t_grid) - 1))
        pab = PDMPSamplers.PiecewiseAffineBound(16)

        PDMPSamplers.construct_rate_bound_grid!(
            pab, pcb, state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), cert;
            build_affine=true,
            stats=PDMPSamplers.DevelStatisticCounter())
        for t in range(0.0, 1.0; length=31)
            st = copy(state)
            move_forward_time!(st, t, flow)
            @test PDMPSamplers.λ(st.ξ, st.ξ.x, flow) <= pab(t) + 1e-12
        end
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "componentwise bounded ZigZag smoke has no violations" begin
        function zz_gaussian_grad!(out, x)
            copyto!(out, x)
            return out
        end
        function zz_gaussian_hvp!(out, x, v)
            copyto!(out, v)
            return out
        end
        model = PDMPModel(2, FullGradient(zz_gaussian_grad!), zz_gaussian_hvp!)
        flow = ZigZag(2)
        alg = GridThinningStrategy(;
            N=4,
            bound=:linear,
            curvature_bound=(0.0),
            bound_violation=:throw,
            lazy=false)
        ξ0 = SkeletonPoint([-0.5, 0.25], [1.0, -1.0])
        rng = Xoshiro(23)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
        @test 0.0 <= τ <= alg_.t_max[]
        @test event_type in (:reflect, :horizon_hit)
        @test meta isa PDMPSamplers.GradientMeta
        @test stats.grid_bound_violations == 0
        @test stats.affine_bound_violations == 0
    end
    end

    @testset "_compute_cell_bound!" begin
        t_grid = [0.0, 1.0]
        y_vals = [2.0, 3.0]
        d_vals = [1.0, -1.0]
        Λ_vals = [0.0]

        PDMPSamplers._compute_cell_bound!(Λ_vals, t_grid, y_vals, d_vals, 1)
        @test Λ_vals[1] >= max(y_vals[1], y_vals[2])

        # Equal derivatives → use yᵢ directly
        d_vals2 = [1.0, 1.0]
        Λ_vals2 = [0.0]
        PDMPSamplers._compute_cell_bound!(Λ_vals2, t_grid, y_vals, d_vals2, 1)
        @test Λ_vals2[1] >= max(y_vals[1], y_vals[2])
    end

    @testset "get_rate_and_deriv with GradientOnlyProvider" begin
        d = 3
        target = gen_data(Distributions.MvNormal, d, 2.0)
        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end

        rate, deriv = PDMPSamplers.get_rate_and_deriv(state, flow, PDMPSamplers.GradientOnlyProvider(grad_func))
        @test rate >= 0
        @test deriv == 0.0  # no HVP → zero derivative
    end

    @testset "exact curvature is skipped at zero rate" begin
        state = PDMPState(0.0, SkeletonPoint([1.0, 1.0], [1.0, 1.0]))
        grad_calls = Ref(0)
        hvp_calls = Ref(0)
        vhv_calls = Ref(0)
        grad = x -> begin
            grad_calls[] += 1
            [-1.0, -1.0]
        end
        hvp = (x, v) -> begin
            hvp_calls[] += 1
            copy(v)
        end
        vhv = (x, v, w) -> begin
            vhv_calls[] += 1
            dot(v, w)
        end

        for flow in (BouncyParticle(2, 0.0), ZigZag(2))
            rate, deriv = PDMPSamplers.get_rate_and_deriv(state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), false)
            @test rate == 0.0
            @test deriv == 0.0
            @test hvp_calls[] == 0

            rate_cached, deriv_cached = PDMPSamplers.get_rate_and_deriv(
                state, flow, PDMPSamplers.GradHVPProvider(grad, hvp), false, [-1.0, -1.0])
            @test rate_cached == rate
            @test deriv_cached == deriv
            @test hvp_calls[] == 0

            provider = PDMPSamplers.VHVProvider(grad, vhv, zeros(2))
            rate_vhv, deriv_vhv = PDMPSamplers.get_rate_and_deriv(
                state, flow, provider, false)
            @test rate_vhv == rate
            @test deriv_vhv == deriv
            @test vhv_calls[] == 0
        end

        refresh_flow = BouncyParticle(2, 1.0)
        rate_refresh, _ = PDMPSamplers.get_rate_and_deriv(
            state, refresh_flow, PDMPSamplers.GradHVPProvider(grad, hvp), true)
        @test rate_refresh == 1.0
        @test hvp_calls[] == 1
    end

    @testset "get_rate_and_deriv with FiniteDiffHVP" begin
        d = 3
        target = gen_data(Distributions.MvNormal, d, 2.0)
        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end

        fd_hvp = PDMPSamplers.FiniteDiffHVP(grad_func, zeros(d))
        rate_fd, deriv_fd = PDMPSamplers.get_rate_and_deriv(state, flow, fd_hvp)
        @test rate_fd >= 0
        @test isfinite(deriv_fd)

        # Compare with exact HVP
        hvp_func = (x, v) -> begin
            out = similar(x)
            neg_hvp!(target, out, x, v)
            out
        end
        rate_exact, deriv_exact = PDMPSamplers.get_rate_and_deriv(state, flow, PDMPSamplers.GradHVPProvider(grad_func, hvp_func))
        @test rate_fd ≈ rate_exact
        @test isapprox(deriv_fd, deriv_exact; rtol=0.01)
    end

    @testset "FiniteDiffVHV with gridthinning fallback" begin
        d = 3
        target = gen_data(Distributions.MvNormal, d, 2.0)
        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end

        fd_vhv = PDMPSamplers.FiniteDiffVHV(grad_func, zeros(d))
        rate_fd, deriv_fd = PDMPSamplers.get_rate_and_deriv(state, flow, fd_vhv)
        @test rate_fd >= 0
        @test isfinite(deriv_fd)

        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP/VHV so gridthinning uses finite-diff fallback
        alg = GridThinningStrategy(; use_fd_hvp=true, N=16, t_max=1.5)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        rng = Xoshiro(1493)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
        @test 0.0 <= τ <= alg_.t_max[]
        @test event_type in (:reflect, :horizon_hit)
        @test meta isa PDMPSamplers.GradientMeta
    end

    @testset "∂λ∂t for different flow types" begin
        d = 3
        Random.seed!(42)

        ξ = SkeletonPoint(randn(d), randn(d))
        state = PDMPState(0.0, ξ)
        ∇U = randn(d)
        Hv = randn(d)

        # Generic ContinuousDynamics (BouncyParticle)
        bps = BouncyParticle(d, 1.0)
        result_bps = PDMPSamplers.∂λ∂t(state, ∇U, Hv, bps)
        @test result_bps ≈ dot(ξ.θ, Hv)

        # ZigZag
        zz = ZigZag(d)
        result_zz = PDMPSamplers.∂λ∂t(state, ∇U, Hv, zz)
        @test isfinite(result_zz)

        # Boomerang
        boom = Boomerang(d)
        result_boom = PDMPSamplers.∂λ∂t(state, ∇U, Hv, boom)
        @test isfinite(result_boom)

        # MutableBoomerang
        aboom = AdaptiveBoomerang(d)
        result_aboom = PDMPSamplers.∂λ∂t(state, ∇U, Hv, aboom)
        @test isfinite(result_aboom)

        # LowRankMutableBoomerang
        lr_boom = AdaptiveBoomerang(d; scheme=:lowrank, rank=2)
        result_lr = PDMPSamplers.∂λ∂t(state, ∇U, Hv, lr_boom)
        @test isfinite(result_lr)
    end

    @testset "∂λ∂t Boomerang with StickyPDMPState" begin
        d = 3
        boom = Boomerang(d)
        ξ = SkeletonPoint(randn(d), randn(d))
        free = BitVector([true, false, true])
        state = StickyPDMPState(0.0, ξ, free, randn(d))
        ∇U = randn(d)
        Hv = randn(d)
        result = PDMPSamplers.∂λ∂t(state, ∇U, Hv, boom)
        @test isfinite(result)
    end

    @testset "∂λ∂t LowRankMutableBoomerang with StickyPDMPState" begin
        d = 3
        lr_boom = AdaptiveBoomerang(d; scheme=:lowrank, rank=2)
        ξ = SkeletonPoint(randn(d), randn(d))
        free = BitVector([true, false, true])
        state = StickyPDMPState(0.0, ξ, free, randn(d))
        ∇U = randn(d)
        Hv = randn(d)
        result = PDMPSamplers.∂λ∂t(state, ∇U, Hv, lr_boom)
        @test isfinite(result)
    end

    @testset "min_grid_cells and max_grid_horizon" begin
        zz = ZigZag(3)
        bps = BouncyParticle(3, 1.0)
        boom = Boomerang(3)

        @test PDMPSamplers.min_grid_cells(zz, 5, 20) == 5
        @test PDMPSamplers.min_grid_cells(boom, 5, 20) == 5
        @test PDMPSamplers.min_grid_cells(boom, 15, 20) == 15
        constant_strategy = GridThinningStrategy(; N=8, N_min=2)
        value_strategy = GridThinningStrategy(; N=8, N_min=2,
            bound=:value_quadratic, curvature_bound=3000.0)
        @test PDMPSamplers._grid_min_cells(constant_strategy, boom, 8) == 5
        @test PDMPSamplers._grid_min_cells(value_strategy, boom, 8) == 2
        small_boomerang_strategy = GridThinningStrategy(;
            N=8, N_min=2, allow_small_boomerang=true)
        @test PDMPSamplers._grid_min_cells(small_boomerang_strategy, boom, 8) == 2

        @test PDMPSamplers.max_grid_horizon(zz) == 1e10
        @test PDMPSamplers.max_grid_horizon(boom) ≈ 8π
    end

    @testset "_adapt_grid_N! and _increase_grid_N!" begin
        d = 3
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, N_min=5, t_max=2.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state_cache, 20, 5, Inf)

        # Tight bound → shrink N
        PDMPSamplers._adapt_grid_N!(alg, 0.8)
        @test alg.N[] < 20

        # Reset
        alg.N[] = 20
        PDMPSamplers.recompute_time_grid!(alg)

        # Loose bound → increase N (when below max)
        alg.N[] = 10
        PDMPSamplers._adapt_grid_N!(alg, 0.05)
        @test alg.N[] > 10

        # Already at max → no change
        alg.N[] = 20
        PDMPSamplers._adapt_grid_N!(alg, 0.05)
        @test alg.N[] == 20

        # Already at min → no change
        alg.N[] = 5
        PDMPSamplers._adapt_grid_N!(alg, 0.8)
        @test alg.N[] == 5
    end

    @testset "_increase_grid_N!" begin
        d = 3
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, N_min=5, t_max=2.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state_cache, 16, 5, Inf)

        PDMPSamplers._increase_grid_N!(alg)
        @test alg.N[] == 16

        # Already at max
        PDMPSamplers._increase_grid_N!(alg)
        @test alg.N[] == 16
    end

    @testset "lazy fallback disables pathological lazy searches until reset" begin
        function x7_grad!(out, x)
            out[1] = 7 * x[1]^6
            return out
        end

        model = PDMPModel(1, FullGradient(x7_grad!))
        flow = BouncyParticle(1, 0.0)
        strat = GridThinningStrategy(; N=20, t_max=5.0, safety_limit=10, lazy=true)
        ξ0 = SkeletonPoint([0.0], [2.0])
        rng = Xoshiro(1004)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model, strat, 0.0, ξ0)

        @test alg_.lazy_enabled[]

        τ, event_type, _ = PDMPSamplers.next_event_time(rng, model_, flow, alg_, state, cache, stats, Inf, true)

        @test isfinite(τ)
        @test event_type == :reflect
        @test !alg_.lazy_enabled[]

        PDMPSamplers.reset_grid_scale!(alg_, strat.t_max)
        @test alg_.lazy_enabled[]
    end

    @testset "reset_grid_scale!" begin
        d = 3
        state_cache = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, N_min=5, t_max=5.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state_cache, 10, 5, Inf)

        PDMPSamplers.reset_grid_scale!(alg, 3.0)
        @test alg.t_max[] ≈ 3.0
        @test alg.N[] == 10
        @test length(alg.pcb.t_grid) == 11
    end

    @testset "GridThinningStrategy construction with use_fd_hvp" begin
        strat = GridThinningStrategy(; use_fd_hvp=true, N=30)
        @test strat.use_fd_hvp
        @test strat.curvature_backend === :finite_difference
        @test strat.N == 30

        exact_strat = GridThinningStrategy(; curvature_backend=:exact)
        @test exact_strat.curvature_backend === :exact
        @test !exact_strat.use_fd_hvp
        @test_throws ArgumentError GridThinningStrategy(;
            use_fd_hvp=true, curvature_backend=:exact)
        @test_throws ArgumentError GridThinningStrategy(;
            curvature_backend=:unknown)

        strat_linear = GridThinningStrategy(; bound=:linear, lazy=false,
            curvature_bound=(args...) -> (0.0))
        @test strat_linear.bound === :linear
        @test strat_linear.curvature_bound !== nothing
        @test strat_linear.linear_area_threshold == 0.95
        @test strat_linear.linear_min_area_gain == 0.0

        auto_strategy = GridThinningStrategy(;
            bound=:auto,
            N=1,
            linear_area_threshold=0.9,
            curvature_bound=(0.0))
        @test auto_strategy.N == 1
        @test auto_strategy.bound === :auto
        @test auto_strategy.linear_area_threshold == 0.9
        @test auto_strategy.linear_min_area_gain == 0.0

        scalar_strategy = GridThinningStrategy(;
            bound=:linear,
            N=1,
            linear_area_threshold=0.9,
            curvature_bound=(0.0))
        @test scalar_strategy.N == 1
        @test scalar_strategy.bound === :linear
        @test scalar_strategy.linear_area_threshold == 0.9
        @test scalar_strategy.linear_min_area_gain == 0.0

        flat_strategy = GridThinningStrategy(;
            bound=:flat,
            N=1,
            curvature_bound=(0.0))
        @test flat_strategy.N == 1
        @test flat_strategy.bound === :flat

        value_strategy = GridThinningStrategy(;
            bound=:value_quadratic,
            N=1,
            curvature_bound=(state, flow, a, b) -> 3.0)
        @test value_strategy.bound === :value_quadratic
        @test value_strategy.curvature_bound !== nothing

        warmup_bound = WarmupCurvatureBound(;
            initial_bound=10.0,
            min_bound=1.0,
            safety_factor=2.0,
            probe_fraction=0.5,
            probe_stride=1)
        @test warmup_bound.current_bound == 10.0
        @test warmup_bound.active
    end

    @testset "value-quadratic callable and warmup curvature providers" begin
        function gaussian_grad!(out, x)
            out[1] = x[1]
            return out
        end

        model = PDMPModel(1, FullGradient(gaussian_grad!))
        flow = BouncyParticle(1, 0.0)

        calls = Ref(0)
        callable_bound = (state, flow, a, b) -> begin
            calls[] += 1
            0.0
        end
        alg_callable = GridThinningStrategy(; N=1, N_min=1, t_max=0.5,
            bound=:value_quadratic, curvature_bound=callable_bound,
            bound_violation=:throw)
        trace_callable, stats_callable = pdmp_sample(
            SkeletonPoint([0.1], [1.0]), flow, model, alg_callable, 0.0, 1.0;
            seed=20260717, progress=false,
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test length(trace_callable) > 0
        @test calls[] > 0
        @test stats_callable.grid_bound_violations == 0

        warmup_bound = WarmupCurvatureBound(; initial_bound=10.0, min_bound=1.0,
            safety_factor=2.0, probe_fraction=0.5, probe_stride=1, apply=true)
        alg_warmup = GridThinningStrategy(; N=1, N_min=1, t_max=0.5,
            bound=:value_quadratic, curvature_bound=warmup_bound,
            bound_violation=:throw)
        trace_warmup, stats_warmup = pdmp_sample(
            SkeletonPoint([0.1], [1.0]), flow, model, alg_warmup, 0.0, 1.5, 0.5;
            seed=20260718, progress=false,
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        @test length(trace_warmup) > 0
        @test !warmup_bound.active
        @test warmup_bound.observations > 0
        @test warmup_bound.current_bound >= warmup_bound.min_bound
        @test stats_warmup.grid_bound_violations == 0
    end

    @testset "R bridge GridThinning compatibility surface" begin
        d = 2
        Γ = Diagonal([1.2, 1.8])
        μ = [0.1, -0.2]
        function compat_grad!(out, x)
            out .= Γ * (x .- μ)
            return out
        end
        function compat_hvp!(out, x, v)
            out .= Γ * v
            return out
        end
        model = PDMPModel(d, FullGradient(compat_grad!), compat_hvp!)
        alg = GridThinningStrategy(;
            N=30,
            t_max=2.0,
            use_fd_hvp=false,
            post_warmup_simplify=true)
        @test alg.bound === :constant

        fields = (
            :reflections_events,
            :reflections_accepted,
            :refreshment_events,
            :sticky_events,
            :support_boundary_events,
            :support_boundary_refresh_attempts,
            :support_boundary_refresh_failures,
            :∇f_calls,
            :∇²f_calls,
            :elapsed_time,
            :grid_builds,
            :grid_shrinks,
            :grid_grows,
            :grid_early_stops,
            :grid_points_evaluated,
            :grid_points_skipped,
            :grid_N_current,
            :lazy_fallback_low_tightness,
            :lazy_fallback_bound_violation,
            :lazy_proposal_attempts,
            :lazy_proposal_rejections,
            :grid_resets_from_dynamics_adaptation,
        )

        flows = (
            ZigZag(Γ, μ),
            BouncyParticle(Γ, μ),
            Boomerang(Γ, μ),
            AdaptiveBoomerang(Γ, μ; λref=0.1),
            PreconditionedZigZag(Matrix(Γ), μ),
            PreconditionedBPS(Matrix(Γ), μ),
        )
        for (i, flow) in enumerate(flows)
            rng = Xoshiro(20_260_702 + i)
            θ0 = PDMPSamplers.initialize_velocity(rng, flow, d)
            ξ0 = SkeletonPoint(copy(μ), θ0)
            _, _, _, _, stats = PDMPSamplers.initialize_state(
                rng, flow, model, alg, 0.0, ξ0;
                statistic_counter=PDMPSamplers.DevelStatisticCounter)
            for field in fields
                @test field in propertynames(stats)
                @test getproperty(stats, field) !== nothing
            end
        end
    end

    @testset "warmup grid tuning requires counters and freezes the selected schedule" begin
        function tuning_grad!(out, x)
            out .= x
            return out
        end

        flow = Boomerang(Diagonal(ones(2)), zeros(2), 0.0)
        model = PDMPModel(2, FullGradient(tuning_grad!))
        tuning = GridWarmupTuning()
        strategy = GridThinningStrategy(;
            N=5, N_min=2, t_max=1.0, use_fd_hvp=true, warmup_tuning=tuning)
        state0 = SkeletonPoint([0.2, -0.1], [0.4, 0.3])
        rng = Xoshiro(20260718)
        _, _, alg_noop, _, stats_noop = PDMPSamplers.initialize_state(
            rng, flow, model, strategy, 0.0, state0;
            statistic_counter=GridTuningNoopCounter)
        @test_throws ArgumentError PDMPSamplers.finish_warmup!(alg_noop, stats_noop, flow)

        _, _, alg, _, stats = PDMPSamplers.initialize_state(
            rng, flow, model, strategy, 0.0, state0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        stats.warmup_events = 10
        stats.warmup_grid_endpoint_gradient_calls = 80
        stats.warmup_grid_acceptance_gradient_calls = 10
        PDMPSamplers.finish_warmup!(alg, stats, flow)
        selected_N = alg.N[]
        selected_tmax = alg.t_max[]
        @test alg.schedule_frozen[]
        @test stats.grid_schedule_frozen
        @test stats.grid_final_N == selected_N
        @test stats.grid_final_tmax == selected_tmax
        @test stats.grid_warmup_objective_gradients_per_event == 9.0
        PDMPSamplers._adapt_grid_N!(alg, 0.9)
        PDMPSamplers._adapt_grid_t_max!(alg, 0.01, model.grad)
        @test alg.N[] == selected_N
        @test alg.t_max[] == selected_tmax
    end

    @testset "Early stopping in construct_upper_bound_grad_and_hess!" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))

        pcb = PDMPSamplers.PiecewiseConstantBound(collect(range(0.0, 2.0, 21)), zeros(20))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end
        hvp_func = (x, v) -> begin
            out = similar(x)
            neg_hvp!(target, out, x, v)
            out
        end

        stats = PDMPSamplers.DevelStatisticCounter()
        PDMPSamplers.construct_upper_bound_grad_and_hess!(pcb, state, flow, PDMPSamplers.GradHVPProvider(grad_func, hvp_func);
            early_stop_threshold=0.001, stats=stats)
        # With a very low threshold, early stopping should trigger
        @test stats.grid_builds == 1
    end

    @testset "constant BPS grid uses batched rate derivatives when available" begin
        flow = BouncyParticle(1, 0.0)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        pcb = PDMPSamplers.PiecewiseConstantBound([0.0, 0.5, 1.0], zeros(2))
        stats = PDMPSamplers.DevelStatisticCounter()
        provider = TestGridRateDerivatives(Ref(0))

        n = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            pcb, state, flow, provider, false;
            stats,
        )

        @test n == 2
        @test provider.calls[] == 1
        @test stats.grid_endpoint_derivative_calls == 1
        @test stats.grid_endpoint_evaluations == 0
        @test stats.grid_endpoint_gradient_calls == 0
        @test stats.grid_endpoint_hessian_calls == 0
        @test pcb.y_vals[1:3] ≈ [0.0, 0.0, 0.5]
        @test pcb.d_vals[1:3] ≈ [0.0, 0.0, 1.0]
        @test pcb.Λ_vals ≈ [0.0, 0.5]
    end

    @testset "append-only constant budget extension matches full construction" begin
        flow = BouncyParticle(1, 0.0)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        provider_full = TestGridRateDerivatives(Ref(0))
        provider_append = TestGridRateDerivatives(Ref(0))
        t_grid = collect(range(0.0, 1.0, 11))

        full = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        n_full = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            full, state, flow, provider_full, false; early_stop_threshold=0.08)

        appended = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        n_first = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            appended, state, flow, provider_append, false; early_stop_threshold=0.01)
        prefix_Λ = copy(appended.Λ_vals[1:n_first])
        prefix_y = copy(appended.y_vals[1:(n_first + 1)])
        prefix_d = copy(appended.d_vals[1:(n_first + 1)])
        built_area = PDMPSamplers._piecewise_constant_area(appended)

        n_appended = PDMPSamplers.construct_upper_bound_grad_and_hess!(
            appended, state, flow, provider_append, false;
            early_stop_threshold=0.08,
            start_cell=n_first + 1,
            initial_integral=built_area)

        @test n_appended == n_full
        @test appended.Λ_vals[1:n_full] ≈ full.Λ_vals[1:n_full]
        @test appended.y_vals[1:(n_full + 1)] ≈ full.y_vals[1:(n_full + 1)]
        @test appended.d_vals[1:(n_full + 1)] ≈ full.d_vals[1:(n_full + 1)]
        @test appended.Λ_vals[1:n_first] == prefix_Λ
        @test appended.y_vals[1:(n_first + 1)] == prefix_y
        @test appended.d_vals[1:(n_first + 1)] == prefix_d

        for budget in (0.01, 0.04, 0.08)
            τ_full, lb_full = PDMPSamplers.propose_event_time(full, budget)
            τ_app, lb_app = PDMPSamplers.propose_event_time(appended, budget)
            @test τ_app ≈ τ_full
            @test lb_app ≈ lb_full
        end
    end

    @testset "append-only inflated affine budget extension preserves prefix" begin
        flow = BouncyParticle(1, 0.0)
        state = PDMPState(0.0, SkeletonPoint([-0.5], [1.0]))
        provider_full = TestGridRateDerivatives(Ref(0))
        provider_append = TestGridRateDerivatives(Ref(0))
        cert = (0.0)
        t_grid = collect(range(0.0, 1.0, 11))

        full_pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        full_pab = PDMPSamplers.PiecewiseAffineBound(32)
        n_full = PDMPSamplers.construct_rate_bound_grid!(
            full_pab, full_pcb, state, flow, provider_full, cert;
            early_stop_threshold=0.08,
            build_affine=true)

        app_pcb = PDMPSamplers.PiecewiseConstantBound(t_grid, zeros(10))
        app_pab = PDMPSamplers.PiecewiseAffineBound(32)
        n_first = PDMPSamplers.construct_rate_bound_grid!(
            app_pab, app_pcb, state, flow, provider_append, cert;
            early_stop_threshold=0.01,
            build_affine=true)
        prefix_segments = app_pab.n_segments
        prefix_breaks = copy(app_pab.t_breaks[1:(prefix_segments + 1)])
        prefix_y_left = copy(app_pab.y_left[1:prefix_segments])
        prefix_slopes = copy(app_pab.slopes[1:prefix_segments])
        prefix_cum_area = copy(app_pab.cum_area[1:(prefix_segments + 1)])
        built_area = PDMPSamplers.total_area(app_pab)

        n_appended = PDMPSamplers.construct_rate_bound_grid!(
            app_pab, app_pcb, state, flow, provider_append, cert;
            early_stop_threshold=0.08,
            build_affine=true,
            start_cell=n_first + 1,
            initial_integral=built_area,
            append=true)

        @test n_appended == n_full
        @test app_pcb.Λ_vals[1:n_full] ≈ full_pcb.Λ_vals[1:n_full]
        @test app_pcb.y_vals[1:(n_full + 1)] ≈ full_pcb.y_vals[1:(n_full + 1)]
        @test app_pcb.d_vals[1:(n_full + 1)] ≈ full_pcb.d_vals[1:(n_full + 1)]
        @test app_pab.t_breaks[1:(prefix_segments + 1)] == prefix_breaks
        @test app_pab.y_left[1:prefix_segments] == prefix_y_left
        @test app_pab.slopes[1:prefix_segments] == prefix_slopes
        @test app_pab.cum_area[1:(prefix_segments + 1)] == prefix_cum_area
        @test PDMPSamplers.total_area(app_pab) ≈ PDMPSamplers.total_area(full_pab)

        for budget in (0.01, 0.04, 0.08)
            τ_full, lb_full = PDMPSamplers.propose_event_time(full_pab, budget)
            τ_app, lb_app = PDMPSamplers.propose_event_time(app_pab, budget)
            @test τ_app ≈ τ_full
            @test lb_app ≈ lb_full
        end
    end

    @testset "construct_upper_bound_grad_and_hess! with cached values" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        state = PDMPState(0.0, SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d)))
        pcb = PDMPSamplers.PiecewiseConstantBound(collect(range(0.0, 2.0, 11)), zeros(10))

        grad_func = x -> begin
            out = similar(x)
            neg_gradient!(target, out, x)
            out
        end
        hvp_func = (x, v) -> begin
            out = similar(x)
            neg_hvp!(target, out, x, v)
            out
        end

        stats = PDMPSamplers.DevelStatisticCounter()
        # With cached_y0 and cached_d0
        PDMPSamplers.construct_upper_bound_grad_and_hess!(pcb, state, flow, PDMPSamplers.GradHVPProvider(grad_func, hvp_func);
            cached_y0=1.0, cached_d0=0.5, stats=stats)
        @test pcb.y_vals[1] == 1.0
        @test pcb.d_vals[1] == 0.5
        @test stats.grid_builds == 1
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "End-to-end with FD-HVP" begin
        d = 3
        Random.seed!(42)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP
        alg = GridThinningStrategy(; use_fd_hvp=true)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        rng = Xoshiro(2058)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
        @test 0.0 <= τ <= alg_.t_max[]
        @test event_type in (:reflect, :horizon_hit)
        @test meta isa PDMPSamplers.GradientMeta
        @test stats.∇²f_calls >= 0
    end

    @testset "End-to-end without HVP (grad-only)" begin
        d = 3
        Random.seed!(43)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = ZigZag(d)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP
        alg = GridThinningStrategy()  # no FD-HVP either

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        rng = Xoshiro(2074)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
        @test 0.0 <= τ <= alg_.t_max[]
        @test event_type in (:reflect, :horizon_hit)
        @test meta isa PDMPSamplers.GradientMeta
        @test stats.∇²f_calls == 0
    end
    end

    @testset "Strategy interface defaults and roots acceptance" begin
        roots = PDMPSamplers.RootsPoissonTimeStrategy()
        stats = PDMPSamplers.DevelStatisticCounter()

        @test PDMPSamplers._reset_inner_grid!(roots) === nothing
        @test PDMPSamplers._maybe_activate_constant_bound!(roots, stats) === nothing
        @test PDMPSamplers.accept_reflection_event(roots, :dummy) === true
    end

    @testset "Post-warmup constant-bound activation" begin
        d = 3
        state = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, t_max=2.0, post_warmup_simplify=true)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state, 20, 5, 5.0)

        stats = PDMPSamplers.DevelStatisticCounter()
        stats.reflections_accepted = 2
        stats.refreshment_events = 18
        alg.max_observed_rate[] = 3.5
        PDMPSamplers._maybe_activate_constant_bound!(alg, stats)
        @test alg.constant_bound_rate[] ≈ 7.0

        alg.constant_bound_rate[] = NaN
        stats.reflections_accepted = 8
        stats.refreshment_events = 2
        PDMPSamplers._maybe_activate_constant_bound!(alg, stats)
        @test isnan(alg.constant_bound_rate[])
    end

    @testset "Subsampled gradient grid adaptation branches" begin
        d = 3
        state = PDMPState(0.0, SkeletonPoint(zeros(d), ones(d)))
        strat = GridThinningStrategy(; N=20, t_max=2.0)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state, 20, 5, 5.0)

        grad_sub = SubsampledGradient(
            (out, x) -> (out .= x),
            n -> nothing,
            tr -> nothing,
            (out, x) -> (out .= x),
            5,
            0,
            false;
            resample_dt=0.1,
        )

        alg.t_max[] = 10.0
        PDMPSamplers._adapt_grid_t_max!(alg, 0.2, grad_sub)
        @test alg.t_max[] ≈ 4.0

        old_tmax = alg.t_max[]
        PDMPSamplers._shrink_t_max_on_rejection!(alg, alg.pcb, 0.01, grad_sub)
        @test alg.t_max[] == old_tmax
    end

    @testset "Subsampled early stopping is fixed-batch capability gated" begin
        grad_default = SubsampledGradient(
            (out, x) -> (out .= x),
            n -> nothing,
            tr -> nothing,
            (out, x) -> (out .= x),
            5,
            0,
            false;
            resample_dt=0.1,
        )
        grad_fixed = SubsampledGradient(
            (out, x) -> (out .= x),
            n -> nothing,
            tr -> nothing,
            (out, x) -> (out .= x),
            5,
            0,
            false;
            resample_dt=0.1,
            fixed_batch_within_event=true,
        )

        @test grad_default.fixed_batch_within_event === false
        @test grad_fixed.fixed_batch_within_event === true
        @test isinf(PDMPSamplers._adjust_early_stop(grad_default, 2.5))
        @test PDMPSamplers._adjust_early_stop(grad_fixed, 2.5) == 2.5
        @test copy(grad_fixed).fixed_batch_within_event === true
        @test PDMPSamplers.with_stats(grad_fixed, PDMPSamplers.StatisticCounter()).fixed_batch_within_event === true
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "End-to-end with joint directional curvature" begin
        d = 2
        f_logdensity(x) = -0.5 * sum(abs2, x)

        flow = BouncyParticle(d, 1.0)
        model = PDMPModel(d, LogDensity(f_logdensity), DI.AutoForwardDiff(), true)
        alg = GridThinningStrategy(; N=16, t_max=1.5)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        rng = Xoshiro(2174)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
        @test 0.0 <= τ <= alg_.t_max[]
        @test event_type in (:reflect, :horizon_hit)
        @test meta isa PDMPSamplers.GradientMeta
        @test stats.∇²f_calls >= 0
    end

    @testset "get_rate_and_deriv FiniteDiffVHV with BouncyParticle (ContinuousDynamics dispatch)" begin
        d = 3
        Random.seed!(44)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = BouncyParticle(d, 1.0)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)  # no HVP
        alg = GridThinningStrategy(; use_fd_hvp=true, N=16, t_max=1.5)

        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        rng = Xoshiro(2191)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, alg_.t_max[], false)
        @test 0.0 <= τ <= alg_.t_max[]
        @test event_type in (:reflect, :horizon_hit)
        @test meta isa PDMPSamplers.GradientMeta
        @test stats.∇²f_calls >= 0
    end
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "End-to-end eager inflated affine GridThinning uses certificate" begin
        function convex_rate_grad!(out, x)
            out[1] = x[1]^2 + 1.0
            return out
        end
        function convex_rate_hvp!(out, x, v)
            out[1] = 2.0 * x[1] * v[1]
            return out
        end

        model = PDMPModel(1, FullGradient(convex_rate_grad!), convex_rate_hvp!)
        flow = BouncyParticle(1, 0.0)
        cert = (state, flow, a, b) -> begin
            v = state.ξ.θ[1]
            (2.0 * v^3)
        end
        alg = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
            bound=:linear, curvature_bound=cert)
        ξ0 = SkeletonPoint([0.0], [1.0])
        rng = Xoshiro(20260629)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(rng, model_, flow, alg_, state, cache, stats, Inf, false)

        @test isfinite(τ)
        @test event_type === :reflect
        @test stats.affine_inflated_cells > 0
        @test stats.affine_area_hybrid < stats.affine_area_constant_equiv
        @test stats.grid_bound_violations == 0
    end
    end

    @testset "inflated affine GridThinning accepts finite-diff derivatives" begin
        function convex_rate_grad_only!(out, x)
            out[1] = x[1]^2 + 1.0
            return out
        end

        model = PDMPModel(1, FullGradient(convex_rate_grad_only!))
        flow = BouncyParticle(1, 0.0)
        alg = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
            bound=:linear,
            curvature_bound=(state, flow, a, b) -> (2.0),)
        ξ0 = SkeletonPoint([0.0], [1.0])
        rng = Xoshiro(20260630)

        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0;
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, meta = PDMPSamplers.next_event_time(rng, model_, flow, alg_, state, cache, stats, Inf, false)

        @test isfinite(τ)
        @test stats.affine_inflated_cells > 0
        @test stats.affine_area_hybrid > 0.0
        @test stats.affine_area_constant_equiv > 0.0
        @test stats.grid_bound_violations == 0
    end

    @testset "Boomerang finite-diff provider supports signed grid derivatives" begin
        function boomerang_grad_only!(out, x)
            out[1] = x[1]^2 + 1.0
            return out
        end

        model = PDMPModel(1, FullGradient(boomerang_grad_only!))
        flow = Boomerang(Diagonal([1.0]), [0.0], 0.0)
        alg = GridThinningStrategy(; N=4, N_min=1, t_max=1.0, lazy=false,
            bound=:auto)
        ξ0 = SkeletonPoint([0.25], [1.0])
        rng = Xoshiro(20260713)

        state, model_, alg_, _, stats = PDMPSamplers.initialize_state(rng, flow, model, alg, 0.0, ξ0)
        values = Matrix{Float64}(undef, 1, 3)
        derivatives = similar(values)
        t_grid = [0.0, 0.25, 0.5]
        provider = PDMPSamplers._grid_event_provider(model_, flow, alg_, stats)

        PDMPSamplers.rate_derivatives_for_grid!(
            values, derivatives, provider, state, flow, t_grid, length(t_grid))

        @test all(isfinite, values)
        @test all(isfinite, derivatives)
    end

    if RUN_EXTENDED_GRID_SMOKE_TESTS
    @testset "inflated affine bound works without mandatory curvature certificates" begin
        function simple_grad!(out, x)
            out[1] = x[1]
            return out
        end

        model_grad_only = PDMPModel(1, FullGradient(simple_grad!))
        flow = BouncyParticle(1, 0.0)
        alg_required_fd = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
            bound=:linear,
            curvature_bound=(state, flow, a, b) -> (0.0),)
        ξ0 = SkeletonPoint([0.0], [1.0])
        rng = Xoshiro(20260630)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model_grad_only, alg_required_fd, 0.0, ξ0)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, Inf, false)
        @test isfinite(τ)

        function simple_hvp!(out, x, v)
            out[1] = v[1]
            return out
        end
        model_hvp = PDMPModel(1, FullGradient(simple_grad!), simple_hvp!)
        alg_raw_cert = GridThinningStrategy(; N=1, N_min=1, t_max=1.0, lazy=false,
            bound=:linear,
            curvature_bound=(state, flow, a, b) -> 0.0,)
        state, model_, alg_, cache, stats = PDMPSamplers.initialize_state(rng, flow, model_hvp, alg_raw_cert, 0.0, ξ0)
        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model_, flow, alg_, state, cache, stats, Inf, false)
        @test isfinite(τ)
    end
    end

    @testset "_constant_bound_event_time direct call" begin
        d = 3
        Random.seed!(45)
        target = gen_data(Distributions.MvNormal, d, 2.0)

        flow = BouncyParticle(d, 1.0)
        grad = FullGradient(Base.Fix1(neg_gradient!, target))
        model = PDMPModel(d, grad)
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        state = PDMPState(0.0, ξ0)

        strat = GridThinningStrategy(; N=16, t_max=2.0, post_warmup_simplify=true)
        alg = PDMPSamplers._build_grid_adaptive_state(strat, state, 16, 5, 5.0)
        cache = (; z=similar(ξ0.x), ∇ϕx=similar(ξ0.x))
        stats = PDMPSamplers.DevelStatisticCounter()

        # Arm the constant bound to a deliberately high value so thinning always accepts
        alg.constant_bound_rate[] = 1e6

        τ, event_type, meta = PDMPSamplers._constant_bound_event_time(
            model, flow, alg, state, cache, stats, Inf, false)
        @test isfinite(τ)
        @test τ > 0.0
        @test event_type === :reflect

        # With include_refresh=true and high refresh rate, should sometimes return :refresh
        flow_refresh = BouncyParticle(d, 1e6)
        alg2 = PDMPSamplers._build_grid_adaptive_state(strat, state, 16, 5, 5.0)
        alg2.constant_bound_rate[] = 1.0
        τ2, event_type2, _ = PDMPSamplers._constant_bound_event_time(
            model, flow_refresh, alg2, state, cache, stats, Inf, true)
        @test isfinite(τ2)
        @test event_type2 === :refresh

        # Scenario 3: actual rate exceeds bound → fallback, constant_bound_rate → NaN
        # ZeroMeanIsoNormal: gradient(x) = x, so rate = max(0, dot(x, θ))
        # With x=[10,0,...] and θ=[1,0,...], rate = 10 + τ at proposal time τ.
        # Setting λ_bound=10 guarantees l_actual = 10+τ > 10 for any τ > 0.
        target3 = gen_data(Distributions.ZeroMeanIsoNormal, d)
        grad3 = FullGradient(Base.Fix1(neg_gradient!, target3))
        model3 = PDMPModel(d, grad3)
        x3 = zeros(d); x3[1] = 10.0
        θ3 = zeros(d); θ3[1] = 1.0
        ξ3 = SkeletonPoint(x3, θ3)
        state3 = PDMPState(0.0, ξ3)
        alg3 = PDMPSamplers._build_grid_adaptive_state(strat, state3, 16, 5, 5.0)
        alg3.constant_bound_rate[] = 10.0
        τ3, _, _ = PDMPSamplers._constant_bound_event_time(
            model3, flow, alg3, state3, cache, stats, Inf, false)
        @test isnan(alg3.constant_bound_rate[])
        @test isfinite(τ3)
    end
end
