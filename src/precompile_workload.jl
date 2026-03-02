import PrecompileTools

PrecompileTools.@compile_workload begin

    d = 2
    ∇f! = (out, x) -> (out .= x; out)
    ∇²f! = (out, x, v) -> (out .= v; out)

    # GridThinningStrategy requires GlobalGradientModel (FullGradient) for all flows,
    # including ZigZag, since the grid-based algorithm dispatches on GlobalGradientStrategy.
    model_global = PDMPModel(d, FullGradient(∇f!), ∇²f!)

    alg = GridThinningStrategy()
    T = 50.0   # just enough events to trigger compilation
    x0 = zeros(d)

    # ── 1. BouncyParticle ────────────────────────────────────────────────────
    flow_bps = BouncyParticle(I(d), zeros(d))
    chains_bps = pdmp_sample(x0, flow_bps, model_global, alg, 0.0, T)
    mean(chains_bps); cov(chains_bps); ess(chains_bps)

    # ── 2. ZigZag ────────────────────────────────────────────────────────────
    flow_zz = ZigZag(I(d), zeros(d))
    chains_zz = pdmp_sample(x0, flow_zz, model_global, alg, 0.0, T)
    mean(chains_zz)

    # ── 3. Boomerang ─────────────────────────────────────────────────────────
    flow_boom = Boomerang(I(d), zeros(d))
    chains_boom = pdmp_sample(x0, flow_boom, model_global, alg, 0.0, T)
    mean(chains_boom)

    # ── 4. AdaptiveBoomerang (MutableBoomerang with BoomerangAdapter) ────────
    # Adaptive / preconditioned flows can error on trivial targets (safety limits,
    # empty trace during warmup). Compilation still happens before any error.
    try
        flow_aboom = AdaptiveBoomerang(d)
        chains_aboom = pdmp_sample(x0, flow_aboom, model_global, alg, 0.0, T, 10.0)
        mean(chains_aboom)
    catch
    end

    # ── 5. PreconditionedZigZag ──────────────────────────────────────────────
    try
        flow_pzz = PreconditionedZigZag(d)
        chains_pzz = pdmp_sample(x0, flow_pzz, model_global, alg, 0.0, T, 10.0)
        mean(chains_pzz)
    catch
    end

    # ── 6. PreconditionedBPS ─────────────────────────────────────────────────
    try
        flow_pbps = PreconditionedBPS(d)
        chains_pbps = pdmp_sample(x0, flow_pbps, model_global, alg, 0.0, T, 10.0)
        mean(chains_pbps)
    catch
    end
end
