import PrecompileTools

PrecompileTools.@compile_workload begin

    d = 2
    ∇f! = (out, x) -> (out .= x; out)
    ∇²f! = (out, x, v) -> (out .= v; out)

    # GridThinningStrategy requires GlobalGradientModel (FullGradient) for all flows,
    # including ZigZag, since the grid-based algorithm dispatches on GlobalGradientStrategy.
    model_global = PDMPModel(d, FullGradient(∇f!), ∇²f!)

    alg = GridThinningStrategy()
    T = 50.0   # upper bound fallback for event-budgeted runs
    stop_precompile = EventCountCriterion(10)
    x0 = ones(d)

    # ── 1. BouncyParticle ────────────────────────────────────────────────────
    flow_bps = BouncyParticle(I(d), zeros(d))
    chains_bps = pdmp_sample(x0, flow_bps, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)
    mean(chains_bps); var(chains_bps); std(chains_bps); cov(chains_bps); cor(chains_bps)
    quantile(chains_bps, 0.5; coordinate=1); median(chains_bps; coordinate=1)
    cdf(chains_bps, 0.0; coordinate=1); ess(chains_bps)

    # Compile adaptive discretization helpers on a slightly longer trace.
    chains_bps_long = pdmp_sample(x0, flow_bps, model_global, alg, 0.0, T; stop=EventCountCriterion(40), progress=false)
    dt_adapt, _, _ = adaptive_dt(chains_bps_long)
    PDMPDiscretize(chains_bps_long, dt_adapt)
    adaptive_discretize(chains_bps_long)

    # Compile stopping criterion constructors used in user-facing APIs.
    stop_after(events=10)
    stop_after(events=10, T=T)

    # ── 2. ZigZag ────────────────────────────────────────────────────────────
    flow_zz = ZigZag(I(d), zeros(d))
    chains_zz = pdmp_sample(x0, flow_zz, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)
    mean(chains_zz); inclusion_probs(chains_zz)

    # ── 3. Boomerang ─────────────────────────────────────────────────────────
    flow_boom = Boomerang(I(d), zeros(d))
    chains_boom = pdmp_sample(x0, flow_boom, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)
    mean(chains_boom)

    # ── 3b. Matrix{Float64} variants (matching R/JuliaCall interface types) ─
    prec_mat = Matrix{Float64}(I(d))
    flow_zz_mat = ZigZag(prec_mat, zeros(d))
    chains_zz_mat = pdmp_sample(x0, flow_zz_mat, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)
    mean(chains_zz_mat)

    flow_boom_mat = Boomerang(prec_mat, zeros(d))
    chains_boom_mat = pdmp_sample(x0, flow_boom_mat, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)
    mean(chains_boom_mat)

    flow_bps_mat = BouncyParticle(prec_mat, zeros(d))
    pdmp_sample(x0, flow_bps_mat, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)

    # ── 3c. Diagonal variants (R default identity path) ─────────────────────
    prec_diag = Diagonal(ones(d))
    flow_zz_diag = ZigZag(prec_diag, zeros(d))
    pdmp_sample(x0, flow_zz_diag, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)

    flow_boom_diag = Boomerang(prec_diag, zeros(d))
    pdmp_sample(x0, flow_boom_diag, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)

    flow_bps_diag = BouncyParticle(prec_diag, zeros(d))
    pdmp_sample(x0, flow_bps_diag, model_global, alg, 0.0, T; stop=copy(stop_precompile), progress=false)

    # ── 4. AdaptiveBoomerang (MutableBoomerang with BoomerangAdapter) ────────
    # Adaptive / preconditioned flows can error on trivial targets (safety limits,
    # empty trace during warmup). Compilation still happens before any error.
    try
        flow_aboom = AdaptiveBoomerang(d)
        chains_aboom = pdmp_sample(x0, flow_aboom, model_global, alg, 0.0, T, 10.0; stop=copy(stop_precompile), progress=false)
        mean(chains_aboom)
    catch
    end

    # ── 5. PreconditionedZigZag ──────────────────────────────────────────────
    try
        flow_pzz = PreconditionedZigZag(d)
        chains_pzz = pdmp_sample(x0, flow_pzz, model_global, alg, 0.0, T, 10.0; stop=copy(stop_precompile), progress=false)
        mean(chains_pzz)
    catch
    end

    # ── 6. PreconditionedBPS ─────────────────────────────────────────────────
    try
        flow_pbps = PreconditionedBPS(d)
        chains_pbps = pdmp_sample(x0, flow_pbps, model_global, alg, 0.0, T, 10.0; stop=copy(stop_precompile), progress=false)
        mean(chains_pbps)
    catch
    end
end
