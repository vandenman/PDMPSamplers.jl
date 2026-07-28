@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

struct ThinningCase{F,M,R}
    name::String
    flow::F
    model::M
    ξ0::SkeletonPoint
    horizon::Float64
    rate::R
end

function _quadratic_grad!(out, x)
    out[1] = x[1]
    return out
end

function _quadratic_hvp!(out, x, v)
    out[1] = v[1]
    return out
end

function _quadratic_case(flow; x0=0.35, θ0=1.0, horizon=0.35, refresh_rate=0.0)
    model = PDMPModel(1, FullGradient(_quadratic_grad!), _quadratic_hvp!)
    ξ0 = SkeletonPoint([x0], [θ0])
    rate = (state, t) -> begin
        st = copy(state)
        move_forward_time!(st, t, flow)
        PDMPSamplers.λ(st.ξ, st.ξ.x, flow) + refresh_rate
    end
    return ThinningCase(string(typeof(flow)), flow, model, ξ0, horizon, rate)
end

function _initial_thinning_state(case::ThinningCase, strategy, seed::Integer)
    rng = Xoshiro(seed)
    state, model, alg, cache, stats = PDMPSamplers.initialize_state(
        rng, case.flow, case.model, strategy, 0.0, case.ξ0;
        statistic_counter=PDMPSamplers.DevelStatisticCounter)
    return rng, state, model, alg, cache, stats
end

function _check_event_sane(case::ThinningCase, rng, state, model, alg, cache, stats)
    τ, event_type, meta = PDMPSamplers.next_event_time(
        rng, model, case.flow, alg, state, cache, stats, case.horizon, false)

    @test 0.0 <= τ <= case.horizon
    @test event_type in (:reflect, :refresh, :horizon_hit)
    @test meta isa PDMPSamplers.GradientMeta
    @test all(isfinite, meta.∇ϕx)
    @test stats.grid_bound_violations == 0
    return τ, event_type, meta
end

function _dense_rate_bound_check(case::ThinningCase, model, state, alg, stats)
    hasproperty(alg, :pcb) || return nothing
    alg.bound in (:flat, :linear, :auto) || return nothing
    provider = PDMPSamplers._grid_event_provider(model, case.flow, alg, stats)
    modes = PDMPSamplers._grid_bound_modes(alg, state, case.flow, provider)
    PDMPSamplers._build_grid_bound_prefix!(alg.pcb, state, case.flow, provider, alg, stats, alg.state_cache, case.horizon, Inf, PDMPSamplers.NoGridBoundaryProbe(), modes)
    isempty(alg.pcb.Λ_vals) && return nothing
    t_stop = min(case.horizon, alg.pcb.t_grid[end])
    t_stop > 0 || return nothing
    for t in range(0.0, t_stop; length=11)
        t == t_stop && continue
        bound = max(alg.pcb(t), 0.0)
        if hasproperty(alg, :affine_bound) && alg.affine_bound.n_segments > 0
            bound = min(bound, max(alg.affine_bound(t), 0.0))
        end
        @test case.rate(state, t) <= bound + 1e-8
    end
    return nothing
end

function _validate_strategy(case::ThinningCase, strategy; seed::Integer)
    rng, state, model, alg, cache, stats = _initial_thinning_state(case, strategy, seed)
    _check_event_sane(case, rng, state, model, alg, cache, stats)
    _, bound_state, bound_model, bound_alg, _, _ = _initial_thinning_state(case, strategy, seed + 10_000)
    _dense_rate_bound_check(case, bound_model, bound_state, bound_alg, stats)
    return stats
end

@testset "Shared thinning strategy harness" begin
    bps_case = _quadratic_case(BouncyParticle(1, 0.0))
    zz_case = _quadratic_case(ZigZag(1))
    boomerang_case = _quadratic_case(Boomerang(Diagonal([1.0]), [0.0], 0.0);
        x0=0.2, θ0=0.4, horizon=0.25)

    @testset "GridThinningStrategy matrix" begin
        strategies = (
            "flat exact" => GridThinningStrategy(; N=3, N_min=1, t_max=0.35,
                lazy=false, bound=:flat, curvature_bound=0.0, bound_violation=:throw),
            "linear exact" => GridThinningStrategy(; N=3, N_min=1, t_max=0.35,
                lazy=false, bound=:linear, curvature_bound=0.0, bound_violation=:throw),
            "auto exact" => GridThinningStrategy(; N=3, N_min=1, t_max=0.35,
                lazy=false, bound=:auto, curvature_bound=0.0, bound_violation=:throw),
            "value quadratic" => GridThinningStrategy(; N=3, N_min=1, t_max=0.35,
                lazy=false, bound=:value_quadratic, curvature_bound=0.0, bound_violation=:throw),
            "finite diff" => GridThinningStrategy(; N=3, N_min=1, t_max=0.35,
                lazy=false, bound=:linear, curvature_bound=0.0, curvature_backend=:finite_difference,
                bound_violation=:throw),
        )
        for (i, (name, strategy)) in enumerate(strategies)
            @testset "$name BPS" begin
                stats = _validate_strategy(bps_case, strategy; seed=30_000 + i)
                @test stats.grid_builds >= 1
            end
        end

        @testset "componentwise ZigZag" begin
            stats = _validate_strategy(zz_case, GridThinningStrategy(;
                N=3, N_min=1, t_max=0.35, lazy=false, bound=:linear,
                curvature_bound=0.0, bound_violation=:throw); seed=31_001)
            @test stats.grid_builds >= 1
        end
    end

    @testset "VectorVariationThinningStrategy" begin
        strategy = PDMPSamplers.VectorVariationThinningStrategy(;
            N=3, N_min=1, t_max=0.35, validation_rtol=0.0, validation_atol=1e-10,
            fallback=GridThinningStrategy(; N=3, N_min=1, t_max=0.35,
                lazy=false, bound=:flat, bound_violation=:throw))
        stats = _validate_strategy(zz_case, strategy; seed=32_001)
        @test stats.positive_variation_fallbacks == 0
    end

    @testset "PositiveVariationGridThinningStrategy" begin
        strategy = PDMPSamplers.PositiveVariationGridThinningStrategy(;
            N=3, N_min=1, t_max=0.25, validation_rtol=0.0, validation_atol=1e-10,
            fallback=GridThinningStrategy(; N=3, N_min=1, t_max=0.25,
                lazy=false, bound=:flat, curvature_bound=0.0, bound_violation=:throw))
        stats = _validate_strategy(boomerang_case, strategy; seed=33_001)
        @test stats.grid_builds >= 1
    end
end
