import PrecompileTools

PrecompileTools.@setup_workload begin
    d = 2
    ∇f! = (out, x) -> (out .= x; out)
    ∇²f! = (out, x, v) -> (out .= v; out)

    model = PDMPModel(d, FullGradient(∇f!), ∇²f!)
    alg = ThinningStrategy(GlobalBounds(2.0, d))
    x0 = ones(d)
    ξ0 = SkeletonPoint(x0, [1.0, -1.0])

    flow_bps = BouncyParticle(I(d), zeros(d))

    # Compile common flow constructors without paying to run every flow/metric pair.
    flow_zz = ZigZag(I(d), zeros(d))
    flow_boom = Boomerang(I(d), zeros(d))
    flow_zz_diag = ZigZag(Diagonal(ones(d)), zeros(d))
    flow_boom_diag = Boomerang(Diagonal(ones(d)), zeros(d))
    flow_bps_diag = BouncyParticle(Diagonal(ones(d)), zeros(d))
    flow_aboom = AdaptiveBoomerang(d)
    flow_pzz = PreconditionedZigZag(d)
    flow_pbps = PreconditionedBPS(d)

    trace = PDMPTrace([
        PDMPEvent(0.0, [0.0, 0.0], [1.0, -1.0]),
        PDMPEvent(0.2, [0.2, -0.2], [-1.0, -1.0]),
        PDMPEvent(0.4, [0.0, -0.4], [-1.0, 1.0]),
        PDMPEvent(0.6, [-0.2, -0.2], [1.0, 1.0]),
        PDMPEvent(0.8, [0.0, 0.0], [1.0, -1.0]),
        PDMPEvent(1.0, [0.2, -0.2], [-1.0, -1.0]),
        PDMPEvent(1.2, [0.0, -0.4], [-1.0, 1.0]),
        PDMPEvent(1.4, [-0.2, -0.2], [1.0, 1.0]),
        PDMPEvent(1.6, [0.0, 0.0], [1.0, -1.0]),
        PDMPEvent(1.8, [0.2, -0.2], [-1.0, -1.0]),
        PDMPEvent(2.0, [0.0, -0.4], [-1.0, 1.0]),
        PDMPEvent(2.2, [-0.2, -0.2], [1.0, 1.0]),
    ], flow_bps)
    chains = PDMPChains([trace], [StatisticCounter()])

    PrecompileTools.@compile_workload begin
        precompile(pdmp_sample, (
            typeof(ξ0), typeof(flow_bps), typeof(model), typeof(alg),
            Float64, Float64))
        precompile(pdmp_sample, (
            typeof(x0), typeof(flow_bps), typeof(model), typeof(alg),
            Float64, Float64))
        TraceManager(PDMPState(0.0, copy(ξ0)), flow_bps, alg, 0.0)

        initialize_velocity(flow_zz, d)
        initialize_velocity(flow_boom, d)
        initialize_velocity(flow_zz_diag, d)
        initialize_velocity(flow_boom_diag, d)
        initialize_velocity(flow_bps_diag, d)
        initialize_velocity(flow_aboom, d)
        initialize_velocity(flow_pzz, d)
        initialize_velocity(flow_pbps, d)

        mean(chains); var(chains); std(chains); cov(chains); cor(chains)
        quantile(chains, 0.5; coordinate=1); median(chains; coordinate=1)
        cdf(chains, 0.0; coordinate=1)
        moments = _trace_moments(trace)
        var(trace, moments.mean); cov(trace, moments.mean)
        Matrix(PDMPDiscretize(trace, 0.2))

        stop_after(events=10)
        stop_after(events=10, T=1.0)
    end
end
