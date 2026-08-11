@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

struct _PVTestQuadraticGradient
    scale::Float64
    last_potential::Base.RefValue{Float64}
end

function (gradient::_PVTestQuadraticGradient)(out, x)
    gradient.last_potential[] = 0.5 * gradient.scale * dot(x, x)
    out .= gradient.scale .* x
    return out
end

PDMPSamplers._last_gradient_potential(gradient::_PVTestQuadraticGradient) =
    gradient.last_potential[]
PDMPSamplers._potential_available(::_PVTestQuadraticGradient) = true
PDMPSamplers._potential(gradient::_PVTestQuadraticGradient, x::Vector{Float64}) =
    0.5 * gradient.scale * dot(x, x)

function _pv_test_hvp!(out, x, v)
    out .= 20.0 .* v
    return out
end

function _pv_test_model()
    gradient = _PVTestQuadraticGradient(20.0, Ref(NaN))
    return PDMPModel(1, FullGradient(gradient), _pv_test_hvp!)
end

function _pv_test_internal(strategy; seed=1, flow=Boomerang(Diagonal([1.0]), zeros(1), 0.0))
    rng = Xoshiro(seed)
    state, model, alg, cache, stats = PDMPSamplers.initialize_state(
        rng,
        flow,
        _pv_test_model(),
        strategy,
        0.0,
        SkeletonPoint([1.0], [1.0]);
        statistic_counter=PDMPSamplers.DevelStatisticCounter,
    )
    return rng, flow, state, model, alg, cache, stats
end

@testset "PositiveVariationGridThinningStrategy" begin
    @testset "positive-area kernels agree with numerical integration" begin
        linear_cases = (
            (-2.0, -1.0),
            (1.0, 2.0),
            (-1.0, 2.0),
            (2.0, -1.0),
            (1.0, 1.0),
        )
        h = 1.7
        for (fa, fb) in linear_cases
            f(t) = max(fa + (fb - fa) * t / h, 0.0)
            expected, _ = PDMPSamplers.QuadGK.quadgk(f, 0.0, h; rtol=1e-12)
            area = PDMPSamplers._linear_positive_area(fa, fb, h)
            @test area ≈ expected atol=1e-12
            if area > 0
                target = 0.37area
                τ = PDMPSamplers._linear_positive_area_time(fa, fb, h, target)
                partial, _ = PDMPSamplers.QuadGK.quadgk(f, 0.0, τ; rtol=1e-12)
                @test partial ≈ target atol=1e-10
            end
        end
        @test PDMPSamplers._linear_positive_area(1.0, 2.0, 0.0) == 0.0
        @test PDMPSamplers._linear_positive_area_time(1.0, 2.0, h, 0.0) == 0.0
        @test PDMPSamplers._linear_positive_area_time(-2.0, -1.0, h, 0.1) == h
        @test PDMPSamplers._pv_sort2(2.0, 1.0) == (1.0, 2.0)
        @test PDMPSamplers._pv_sort3(3.0, 2.0, 1.0) == (1.0, 2.0, 3.0)
        @test PDMPSamplers._pv_sort3(2.0, 3.0, 1.0) == (1.0, 2.0, 3.0)
        @test PDMPSamplers._pv_sort4(4.0, 3.0, 2.0, 1.0) == (1.0, 2.0, 3.0, 4.0)
        @test PDMPSamplers._pv_sort4(2.0, 4.0, 1.0, 3.0) == (1.0, 2.0, 3.0, 4.0)

        quadratic_cases = (
            (0.0, 2.0, -1.0),       # linear, one interior root
            (1.0, -1.0, 0.1875),    # two interior roots
            (1.0, 0.5, -0.5),       # only the larger root is interior
            (1.0, -3.0, 2.0),       # one interior root
            (1.0, 0.0, 1.0),        # no real roots
            (-1.0, 0.0, -1.0),      # everywhere negative
        )
        hq = 2.0
        for (A, B, C) in quadratic_cases
            q(u) = max(evalpoly(u, (C, B, A)), 0.0)
            expected, _ = PDMPSamplers.QuadGK.quadgk(q, 0.0, 1.0; rtol=1e-12)
            @test PDMPSamplers._pv_quadratic_positive_area(A, B, C, hq) ≈
                  hq * expected atol=1e-10
            for u in (0.2, 0.6, 0.9, 1.0)
                prefix, _ = PDMPSamplers.QuadGK.quadgk(q, 0.0, u; rtol=1e-12)
                @test PDMPSamplers._pv_quadratic_positive_area_prefix(A, B, C, hq, u) ≈
                      hq * prefix atol=1e-10
            end
        end
        @test PDMPSamplers._pv_quadratic_positive_area(1.0, 0.0, 1.0, 0.0) == 0.0
        @test PDMPSamplers._pv_hermite_positive_area_time(
            NaN, -1.0, 0.0, 2.0, 1.0, 0.1) ≈
              PDMPSamplers._linear_positive_area_time(-1.0, 2.0, 1.0, 0.1)

        cubic_cases = (
            (u -> 1.0, 1.0, 0.0, 1.0, 0.0),
            (u -> 2u - 1, -1.0, 2.0, 1.0, 2.0),
            (u -> (u - 0.25) * (u - 0.75), 0.1875, -1.0, 0.1875, 1.0),
            (u -> (u - 0.2) * (u - 0.5) * (u - 0.8), -0.08, 0.66, 0.08, 0.66),
        )
        for (polynomial, fa, da, fb, db) in cubic_cases
            expected, _ =
                PDMPSamplers.QuadGK.quadgk(u -> max(polynomial(u), 0.0), 0.0, 1.0; rtol=1e-12)
            area = PDMPSamplers._pv_cubic_hermite_positive_area(fa, da, fb, db, 1.0)
            @test area ≈ expected atol=1e-10
            @test PDMPSamplers._pv_cubic_hermite_value(fa, da, fb, db, 1.0, 0.0) ≈ fa
            @test PDMPSamplers._pv_cubic_hermite_value(fa, da, fb, db, 1.0, 1.0) ≈ fb
            if area > 0
                target = 0.43area
                τ = PDMPSamplers._pv_cubic_hermite_positive_area_time(
                    fa, da, fb, db, 1.0, target)
                partial, _ = PDMPSamplers.QuadGK.quadgk(
                    u -> max(polynomial(u), 0.0), 0.0, τ; rtol=1e-12)
                @test partial ≈ target atol=1e-9
            end
        end
        @test isnan(PDMPSamplers._pv_cubic_hermite_positive_area(
            1.0, NaN, 1.0, 0.0, 1.0))
        @test PDMPSamplers._pv_cubic_hermite_positive_area_time(
            -1.0, NaN, 2.0, 0.0, 1.0, 0.1) ≈
              PDMPSamplers._linear_positive_area_time(-1.0, 2.0, 1.0, 0.1)
    end

    @testset "area models select coherent inverses" begin
        potential_args = (-1.0, -1.0, 1.0, 0.0, 1.0, NaN, NaN)
        potential_model = PDMPSamplers._pv_area_model(potential_args...)
        @test potential_model.kind === :potential_hermite
        @test potential_model.area > 0
        potential_target = 0.25potential_model.area
        potential_τ = PDMPSamplers._pv_positive_area_time(
            potential_model, potential_args[1], potential_args[2], potential_args[3],
            potential_target, potential_args[4], potential_args[5],
            potential_args[6], potential_args[7])
        potential_partial, _ = PDMPSamplers.QuadGK.quadgk(
            u -> max(-12u^2 + 12u - 1, 0.0), 0.0, potential_τ; rtol=1e-12)
        @test potential_partial ≈ potential_target atol=1e-9

        derivative_args = (
            -0.4479970766765322, 0.7798812494512521, 1.0, NaN, NaN,
            0.29871024143647007, -0.28080584331379616,
        )
        derivative_model = PDMPSamplers._pv_area_model(derivative_args...)
        @test derivative_model.kind === :derivative_hermite
        @test derivative_model.area > 0
        derivative_target = 0.25derivative_model.area
        derivative_τ = PDMPSamplers._pv_positive_area_time(
            derivative_model, derivative_args[1], derivative_args[2], derivative_args[3],
            derivative_target, derivative_args[4], derivative_args[5],
            derivative_args[6], derivative_args[7])
        derivative_partial, _ = PDMPSamplers.QuadGK.quadgk(
            u -> max(PDMPSamplers._pv_cubic_hermite_value(
                derivative_args[1], derivative_args[6], derivative_args[2],
                derivative_args[7], 1.0, u), 0.0),
            0.0, derivative_τ; rtol=1e-12)
        @test derivative_partial ≈ derivative_target atol=1e-9

        linear_model = PDMPSamplers._pv_area_model(1.0, 2.0, 1.0, NaN, NaN)
        @test linear_model.kind === :linear
        linear_target = 0.25linear_model.area
        linear_τ = PDMPSamplers._pv_positive_area_time(
            linear_model, 1.0, 2.0, 1.0, linear_target, NaN, NaN, NaN, NaN)
        @test PDMPSamplers._linear_positive_area(
            1.0, 1.0 + linear_τ, linear_τ) ≈ linear_target atol=1e-12

        @test PDMPSamplers._pv_positive_area_floor(0.1, 0.0, 0.4) == 0.4
        @test PDMPSamplers._pv_positive_area_floor(0.1, NaN, 0.4) == 0.1
        @test PDMPSamplers._pv_lipschitz_positive_area_upper(-1.0, 1.0, 1.0, 2.0) ≥
              PDMPSamplers._linear_positive_area(-1.0, 1.0, 1.0)
        @test PDMPSamplers._pv_lipschitz_positive_area_upper(1.0, 2.0, 0.0, 2.0) == 0.0
    end

    @testset "short approximate searches refine and reuse state" begin
        fallback = GridThinningStrategy(;
            N=4, N_min=1, t_max=1.0, lazy=false, bound=:flat, use_fd_hvp=true)
        strategies = (
            PDMPSamplers.PositiveVariationGridThinningStrategy(;
                N=2, N_min=1, t_max=1.0, approximate=true,
                use_derivative_hermite=true, max_skip_width=0.2,
                fallback),
            PDMPSamplers.PositiveVariationGridThinningStrategy(;
                N=2, N_min=1, t_max=1.0, approximate=true,
                derivative_hermite_on_demand=true,
                derivative_hermite_trigger_scale=1e6, max_skip_width=0.2,
                fallback),
            PDMPSamplers.PositiveVariationGridThinningStrategy(;
                N=1, N_min=1, t_max=0.5, approximate=true,
                dense_cell_width=1.0, fallback),
        )

        for (seed, strategy) in enumerate(strategies)
            rng, flow, state, model, alg, cache, stats =
                _pv_test_internal(strategy; seed=40_000 + seed)
            τ, event_type, meta = PDMPSamplers.next_event_time(
                rng, model, flow, alg, state, cache, stats, 0.5, false)
            @test 0.0 <= τ <= 0.5
            @test event_type in (:reflect, :horizon_hit)
            @test meta isa PDMPSamplers.GradientMeta
            @test stats.positive_variation_cells > 0
            @test stats.grid_endpoint_gradient_calls > 0
            if event_type === :reflect
                @test alg.has_cached_gradient[]
                @test all(isfinite, meta.∇ϕx)
            end
        end

        skip_strategy = PDMPSamplers.PositiveVariationGridThinningStrategy(;
            N=1, N_min=1, t_max=0.1, approximate=true, fallback)
        skip_rng, skip_flow, skip_state, skip_model, skip_alg, skip_cache, skip_stats =
            _pv_test_internal(skip_strategy; seed=40_100)
        skip_state.ξ.x[1] = -1.0
        skip_τ, skip_event, _ = PDMPSamplers.next_event_time(
            skip_rng, skip_model, skip_flow, skip_alg, skip_state, skip_cache,
            skip_stats, 0.01, false)
        @test skip_τ == 0.01
        @test skip_event === :horizon_hit
        @test skip_stats.positive_variation_skipped_cells > 0
        @test skip_stats.positive_variation_accepts == 0
    end

    @testset "exact envelope accepts, reuses its endpoint, and yields to refreshment" begin
        fallback = GridThinningStrategy(;
            N=4, N_min=1, t_max=1.0, lazy=false, bound=:flat, use_fd_hvp=true)
        strategy = PDMPSamplers.PositiveVariationGridThinningStrategy(;
            N=2, N_min=1, t_max=1.0, fallback)
        rng, flow, state, model, alg, cache, stats =
            _pv_test_internal(strategy; seed=1)

        τ, event_type, meta = PDMPSamplers.next_event_time(
            rng, model, flow, alg, state, cache, stats, 0.5, false)
        @test event_type === :reflect
        @test 0.0 < τ < 0.5
        @test stats.positive_variation_accepts == 1
        @test stats.positive_variation_fallbacks == 0
        @test alg.has_cached_gradient[]

        move_forward_time!(state, τ, flow)
        PDMPSamplers.reflect!(rng, state.ξ, meta.∇ϕx, flow, cache)
        cached_reuses = stats.grid_cached_endpoint_reuses
        τ2, event_type2, _ = PDMPSamplers.next_event_time(
            rng, model, flow, alg, state, cache, stats, 0.05, false)
        @test 0.0 <= τ2 <= 0.05
        @test event_type2 in (:reflect, :horizon_hit)
        @test stats.grid_cached_endpoint_reuses == cached_reuses + 1

        zero_τ, zero_event, _ = PDMPSamplers.next_event_time(
            rng, model, flow, alg, state, cache, stats, 0.0, false, :support_hit)
        @test zero_τ == 0.0
        @test zero_event === :support_hit

        refresh_flow = Boomerang(Diagonal([1.0]), zeros(1), 1e6)
        refresh_rng, _, refresh_state, refresh_model, refresh_alg, refresh_cache, refresh_stats =
            _pv_test_internal(strategy; seed=2, flow=refresh_flow)
        refresh_τ, refresh_event, _ = PDMPSamplers.next_event_time(
            refresh_rng, refresh_model, refresh_flow, refresh_alg, refresh_state,
            refresh_cache, refresh_stats, 0.5, true)
        @test 0.0 <= refresh_τ < 0.5
        @test refresh_event === :refresh
    end

    @testset "constructor validation and unsupported-flow fallback" begin
        constructor = PDMPSamplers.PositiveVariationGridThinningStrategy
        @test_throws ArgumentError constructor(; N=0)
        @test_throws ArgumentError constructor(; N_min=0)
        @test_throws ArgumentError constructor(; t_max=0.0)
        @test_throws ArgumentError constructor(; max_refinement_depth=-1)
        @test_throws ArgumentError constructor(; validation_rtol=-1.0)
        @test_throws ArgumentError constructor(; validation_atol=-1.0)
        @test_throws ArgumentError constructor(; min_cell_width=0.0)
        @test_throws ArgumentError constructor(; max_skip_width=0.0)
        @test_throws ArgumentError constructor(; dense_cell_width=-1.0)
        @test_throws ArgumentError constructor(; skip_slope_safety=-1.0)
        @test_throws ArgumentError constructor(; derivative_hermite_trigger_scale=-1.0)
        @test occursin("PositiveVariationGridThinningStrategy", sprint(show, constructor()))

        strategy = constructor(; N=2, N_min=1, t_max=0.25)
        rng = Xoshiro(41_000)
        flow = ZigZag(1)
        state, model, alg, cache, stats = PDMPSamplers.initialize_state(
            rng, flow, _pv_test_model(), strategy, 0.0,
            SkeletonPoint([1.0], [1.0]);
            statistic_counter=PDMPSamplers.DevelStatisticCounter)
        τ, event_type, _ = PDMPSamplers.next_event_time(
            rng, model, flow, alg, state, cache, stats, 0.1, false)
        @test 0.0 <= τ <= 0.1
        @test event_type in (:reflect, :horizon_hit)
        @test stats.positive_variation_cells == 0

        alg.has_cached_gradient[] = true
        alg.cached_f[] = 1.0
        alg.cached_ψ[] = 2.0
        alg.cached_df[] = 3.0
        PDMPSamplers.reset_grid_scale!(alg, 0.4)
        @test alg.t_max[] == 0.4
        @test !alg.has_cached_gradient[]
        @test all(isnan, (alg.cached_f[], alg.cached_ψ[], alg.cached_df[]))
        alg.N[] = alg.N_min + 1
        PDMPSamplers._pv_shrink_grid_N!(alg)
        @test alg.N[] == alg.N_min
    end
end
