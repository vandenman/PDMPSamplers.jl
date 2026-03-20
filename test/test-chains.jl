@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

@testset "PDMPChains & multi-chain" begin

    d = 2
    target = gen_data(Distributions.ZeroMeanIsoNormal, d)
    D = target.D
    grad = FullGradient(Base.Fix1(neg_gradient!, target))
    model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
    flow = ZigZag(d)
    alg = GridThinningStrategy()
    T_run = 10_000.0

    @testset "convenience constructors" begin
        Random.seed!(999)
        chains_from_d = pdmp_sample(d, flow, model, alg, 0.0, T_run; progress=false)
        @test chains_from_d isa PDMPChains
        @test n_chains(chains_from_d) == 1

        x0 = randn(d)
        chains_from_x = pdmp_sample(x0, flow, model, alg, 0.0, T_run; progress=false)
        @test n_chains(chains_from_x) == 1

        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        chains_from_xθ = pdmp_sample(x0, θ0, flow, model, alg, 0.0, T_run; progress=false)
        @test n_chains(chains_from_xθ) == 1
    end

    @testset "n_chains error" begin
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        @test_throws ArgumentError pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; n_chains=0, progress=false)
    end

    @testset "t_warmup error" begin
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        @test_throws ArgumentError pdmp_sample(ξ0, flow, model, alg, 0.0, 100.0, 200.0; progress=false)
    end

    @testset "multi-chain sequential" begin
        Random.seed!(42)
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        chains = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; n_chains=2, threaded=false, progress=false)
        @test chains isa PDMPChains
        @test n_chains(chains) == 2
        @test length(chains) == 2

        trace1, stats1 = chains[1]
        trace2, stats2 = chains[2]
        @test trace1 isa PDMPSamplers.AbstractPDMPTrace
        @test trace2 isa PDMPSamplers.AbstractPDMPTrace
        @test length(trace1) > 10
        @test length(trace2) > 10
        @test stats1.elapsed_time > 0
        @test stats2.elapsed_time > 0

        @testset "PDMPChains API" begin
            @test firstindex(chains) == 1
            @test lastindex(chains) == 2
            @test collect(eachindex(chains)) == [1, 2]

            # iterate protocol
            items = collect(Iterators.take(Iterators.flatten([chains]), 3))
            @test length(items) >= 1

            # statistics
            m = mean(chains)
            @test m isa AbstractVector
            @test length(m) == d
            m2 = mean(chains; chain=2)
            @test length(m2) == d

            v = var(chains)
            @test all(v .> 0)
            v2 = var(chains; chain=2)
            @test all(v2 .> 0)

            s = std(chains)
            @test s ≈ sqrt.(v)

            C = cov(chains)
            @test size(C) == (d, d)

            R = cor(chains)
            @test size(R) == (d, d)
            @test all(abs.(R) .<= 1.0 + 1e-10)

            med = Statistics.median(chains; coordinate=1)
            @test med isa Real

            q = Statistics.quantile(chains, 0.5; coordinate=1)
            @test q isa Real

            c = PDMPSamplers.cdf(chains, 0.0; coordinate=1)
            @test 0.0 <= c <= 1.0

            e = ess(chains)
            @test e isa AbstractVector
            @test all(e .> 0)

            disc = PDMPDiscretize(chains, 1.0)
            @test disc isa PDMPDiscretize

            # show
            buf = IOBuffer()
            show(buf, chains)
            str = String(take!(buf))
            @test occursin("PDMPChains with 2 chains", str)
        end
    end

    @testset "multi-chain threaded" begin
        Random.seed!(42)
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        chains = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; n_chains=2, threaded=true, progress=false)
        @test n_chains(chains) == 2
        @test length(chains[1][1]) > 10
        @test length(chains[2][1]) > 10
    end

    @testset "single chain show" begin
        Random.seed!(42)
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        chains = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; progress=false)
        buf = IOBuffer()
        show(buf, chains)
        @test occursin("1 chain", String(take!(buf)))
    end

    @testset "adaptive_dt and adaptive_discretize" begin
        Random.seed!(123)
        ξ0 = SkeletonPoint(randn(d), PDMPSamplers.initialize_velocity(flow, d))
        chains = pdmp_sample(ξ0, flow, model, alg, 0.0, T_run; progress=false)
        trace = chains.traces[1]

        # adaptive_dt returns (dt, n_disc, ct_ess_min)
        dt_val, n_disc, ct_ess_min = adaptive_dt(trace)
        @test dt_val > 0
        @test n_disc >= 10
        @test n_disc == max(ceil(Int, ct_ess_min), 10)
        t_start = PDMPSamplers.first_event_time(trace)
        t_end = PDMPSamplers.last_event_time(trace)
        @test dt_val ≈ (t_end - t_start) / n_disc

        # chains method delegates correctly
        dt_ch, n_ch, ess_ch = adaptive_dt(chains; chain=1)
        @test dt_ch == dt_val
        @test n_ch == n_disc
        @test ess_ch == ct_ess_min

        # adaptive_discretize returns (matrix, n_disc, ct_ess_min)
        mat, nd, ess_val = adaptive_discretize(trace)
        @test size(mat, 2) == d
        @test nd == n_disc
        @test ess_val == ct_ess_min

        # chains method
        mat2, nd2, ess2 = adaptive_discretize(chains; chain=1)
        @test mat2 == mat
    end

    @testset "inclusion_probs on chains" begin
        Random.seed!(42)
        d_sticky = 3
        D_ss, κ, slab_target = gen_data(SpikeAndSlabDist{Bernoulli,ZeroMeanIsoNormal}, d_sticky)
        grad_ss = FullGradient(Base.Fix1(neg_gradient!, slab_target))
        model_ss = PDMPModel(d_sticky, grad_ss, Base.Fix1(neg_hvp!, slab_target))
        flow_ss = ZigZag(d_sticky)
        alg_ss = Sticky(GridThinningStrategy(), κ)
        ξ0 = SkeletonPoint(randn(d_sticky), PDMPSamplers.initialize_velocity(flow_ss, d_sticky))
        chains = pdmp_sample(ξ0, flow_ss, model_ss, alg_ss, 0.0, 50_000.0; progress=false)
        ip = inclusion_probs(chains)
        @test ip isa AbstractVector
        @test all(0.0 .<= ip .<= 1.0)
    end
end
