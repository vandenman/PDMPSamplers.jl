@testset "Sampling" begin
    show_progress = true
@testset "Basic PDMP Sampler Tests" begin

    pdmp_types = (ZigZag, BouncyParticle, )
    # pdmp_types = (ZigZag, BouncyParticle, Boomerang)
    # pdmp_types = (BouncyParticle, Boomerang)

    # pdmp_types = (Boomerang, )
    factorized_gradient_types    = (CoordinateWiseGradient, FullGradient)
    # factorized_gradient_types    = (FullGradient, )
    nonfactorized_gradient_types = (FullGradient, )
    # algorithms = (ThinningStrategy, GridThinningStrategy)
    get_gradient_types(::Type{<:FactorizedDynamics})    = factorized_gradient_types
    get_gradient_types(::Type{<:NonFactorizedDynamics}) = nonfactorized_gradient_types
    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:FullGradient})           = (ThinningStrategy, GridThinningStrategy, )
    # get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:FullGradient})           = (ThinningStrategy, GridThinningStrategy, RootsPoissonTimeStrategy)
    # get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:FullGradient})           = (RootsPoissonTimeStrategy, )
    # get_algorithm_types(::Type{<:ZigZag}, ::Type{<:FullGradient})                       = (ThinningStrategy, GridThinningStrategy)#, ExactStrategy)
    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:CoordinateWiseGradient}) = (ThinningStrategy, )
    data_types = (
        Distributions.ZeroMeanIsoNormal,
        Distributions.MvNormal,
    )

    ds = (2, 5, )#10)
    ηs = (1., 2., 5.)
    data_args = Dict(
        Distributions.ZeroMeanIsoNormal => ds,
        Distributions.MvNormal => Iterators.product(ds, ηs)
    )

    # map(Base.Fix2(dist_name, 3), data_types)

    pdmp_type = ZigZag
    gradient_type = FullGradient
    # data_type = MvNormal
    # data_arg = (5, 1.0)
    data_type = ZeroMeanIsoNormal
    data_arg = (2, )
    # algorithm = GridThinningStrategy
    algorithm = ThinningStrategy # Works poorly with Roots, probably because no longer monotone?
    # I think the while loop to find the bounds got stuck for a long time.
    show_progress = true

    @testset "$pdmp_type"     for pdmp_type     in pdmp_types
    @testset "$gradient_type" for gradient_type in get_gradient_types(pdmp_type)
    @testset "$algorithm"     for algorithm     in get_algorithm_types(pdmp_type, gradient_type)

    @testset "$(data_name(data_type, data_arg))" for
        data_type in data_types, data_arg  in data_args[data_type]

        D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(data_type, data_arg...)

        d = first(data_arg) # could also be length(D)

        T = 50_000.0

        if algorithm === ThinningStrategy
            # TODO: this should still be an informed value!
            if gradient_type === CoordinateWiseGradient
                c0 = 1e-2
                alg = ThinningStrategy(LocalBounds(fill(c0, d)))
            else
                # c0 = 1e-6
                c0 = pdmp_type === ZigZag ? 1e-6 : 1e-2
                alg = ThinningStrategy(GlobalBounds(c0 / d, d))
            end
        elseif algorithm === GridThinningStrategy
            alg = GridThinningStrategy(; hvp = ∇²f!)
        elseif algorithm === ExactStrategy
            alg = ExactStrategy(analytic_next_event_time_Gaussian)
        elseif algorithm === RootsPoissonTimeStrategy
            alg = RootsPoissonTimeStrategy()
        end

        # Use same constructor as working test: ZigZag(sparse(I(d)), zeros(d), σ_value)
        # TODO: this constructor should not need a sparse matrix, we can also use an empty constructor?
        # or some default types for zero I
        # flow = pdmp_type(d)
        flow = pdmp_type(inv(Symmetric(cov(D))), mean(D))

        x0 = randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        grad = gradient_type == CoordinateWiseGradient ? CoordinateWiseGradient(∂fxᵢ) : FullGradient(∇f!)
        trace, stats = pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=show_progress)
        stats.∇f_calls

        # @profview pdmp_sample(ξ0, flow, grad, alg, 0.0, 1000, progress=false)

        # trace0, stats0 = pdmp_sample(ξ0, flow, grad, ThinningStrategy(GlobalBounds(1e-6, d)), 0.0, T, progress=show_progress)
        # stats0.∇f_calls
        # stats0.reflections_accepted / stats0.reflections_events # TODO: why isn't this near 1?
        # trace1, stats1 = pdmp_sample(ξ0, flow, grad, GridThinningStrategy(; hvp = ∇²f!), 0.0, T, progress=show_progress)
        # stats1.∇f_calls
        # stats1.∇²f_calls


        # @profview pdmp_sample(ξ0, flow, grad, alg, 0.0, 1000, progress=false)
        acceptance_prob = stats.reflections_accepted / stats.reflections_events
        stats.refreshment_events
        # stats.∇f_calls
        # stats.∇²f_calls
        # 1192349823, original
        # 368739457,  basic cache
        # 328056135,  new cache
        # 282237352, new cache, lower tolerance
        # 252426211, also lower bounds on root algorithm
        # 139971236, above + memoization
        # 9801165
        # 5790402, gridthinning
        # 673952,  Thinning with exact bounds
        # 139971236 / 9801165
        # 439567501

        # ts = [trace.events[i].time for i in eachindex(trace.events)]
        # diff_ts = diff(ts)
        # obs_qs = quantile(diff_ts, .01:.01:.99)
        # true_qs = quantile(Exponential(), .01:.01:.99)
        # f, ax, _ = scatter(true_qs, obs_qs)
        # ablines!(ax, 0, 1, color=:grey, linestyle=:dash)
        # f

        PDMPSamplers.ispositive(flow.λref) && @test stats.refreshment_events > 100

        # for the boomerang we use the exact distribution for the dynamics.
        # in this case, the inhomogenous process reduces to a constant 0 and no reflections occur
        # this doesn't matter (the samples are from the correct distribution) but it does mean the acceptance probability becomes undefined

        if !(flow isa Boomerang)
            @test acceptance_prob > 0.4  # Minimum acceptance rate
        end
        @test length(trace.events) > 100  # Sufficient events generated
        # @profview pdmp_sample(ξ0, flow, grad, alg, 0.0, 500)

        ts = [event.time for event in trace.events]
        dt = mean(diff(ts))

        # dtrace = PDMPDiscretize(trace, dt)
        # next = iterate(dtrace)
        # i, state = next
        # i1, state1 = iterate(dtrace, state)
        # dt
        # state1
        # @assert dt < i1.first
        # first(dtrace.trace)
        # dtrace.trace.events[1]
        # aa = Vector{typeof(i)}(undef, 0)
        # for i in dtrace
        #     push!(aa, i)
        # end
        # first.(aa) == PDMPSamplers._to_range(dtrace)

        # collect_manual = [i for i in dtrace]
        # length(PDMPSamplers._to_range(dtrace))
        # length(dtrace)
        # @test length(dtrace) == length(collect_manual)#length(collect(dtrace))

        samples = Matrix(PDMPDiscretize(trace, dt))
        test_approximation(samples, D)

        # mean_and_var(trace)



        # [mean(D) vec(mean(samples, dims = 1))]
        # [cov(D) cov(samples)]

        # need some test on efficiency here! but not sure what to expect...
        # MCMCDiagnosticTools.ess_rhat.(eachcol(samples))


    end

    end # algorithm
    end # gradient
    end # pdmp

end

@testset "Sticky PDMP Sampler Tests" begin

    # pdmp_types = (ZigZag, BouncyParticle, Boomerang)
    pdmp_types = (ZigZag, BouncyParticle, )
    factorized_gradient_types    = (FullGradient, )
    nonfactorized_gradient_types = (FullGradient, )
    # algorithms = (ThinningStrategy, GridAdaptiveState)
    get_gradient_types(::Type{<:FactorizedDynamics})    = factorized_gradient_types
    get_gradient_types(::Type{<:NonFactorizedDynamics}) = nonfactorized_gradient_types
    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:FullGradient})           = (ThinningStrategy, GridThinningStrategy, )
    # get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:FullGradient})           = (ThinningStrategy, GridThinningStrategy, RootsPoissonTimeStrategy)

    get_algorithm_types(::Type{<:ContinuousDynamics}, ::Type{<:CoordinateWiseGradient}) = (ThinningStrategy, )

    data_types = (
        SpikeAndSlabDist{Bernoulli, ZeroMeanIsoNormal},
        SpikeAndSlabDist{BetaBernoulli, ZeroMeanIsoNormal},
        # SpikeAndSlabDist{BetaBernoulliHierarchical, ZeroMeanIsoNormal}
    )

    ds = (2, 5, )#10)
    data_args = Dict(
        SpikeAndSlabDist{Bernoulli, ZeroMeanIsoNormal} => ds,
        SpikeAndSlabDist{BetaBernoulli, ZeroMeanIsoNormal} => ds,
        # SpikeAndSlabDist{BetaBernoulliHierarchical, ZeroMeanIsoNormal} => ds
    )

    pdmp_type = Boomerang#
    gradient_type = FullGradient
    data_type = SpikeAndSlabDist{BetaBernoulli, ZeroMeanIsoNormal}
    # data_type = SpikeAndSlabDist{Bernoulli, ZeroMeanIsoNormal}
    # data_type = SpikeAndSlabDist{BetaBernoulliHierarchical, ZeroMeanIsoNormal}
    data_arg = (5, )#1.0)
    # algorithm = RootsPoissonTimeStrategy
    algorithm = GridThinningStrategy
    show_progress = true

    @testset "$pdmp_type"     for pdmp_type     in pdmp_types
    @testset "$gradient_type" for gradient_type in get_gradient_types(pdmp_type)
    @testset "$algorithm"     for algorithm     in get_algorithm_types(pdmp_type, gradient_type)

    @show "algorithm = $algorithm, gradient = $gradient_type, pdmp = $pdmp_type"

    @testset "$(data_name(data_type, data_arg))" for
        data_type in data_types, data_arg  in data_args[data_type]

        Random.seed!(1234)
        D, κ, ∇f!, ∇²f!, ∂fxᵢ = gen_data(data_type, data_arg...)

        d = first(data_arg) # could also be length(D)?

        T = 300_000.0

        c0 = pdmp_type === ZigZag ? 1e-4 : 1e-2
        if data_type <: SpikeAndSlabDist{BetaBernoulliHierarchical}
            c0 = pdmp_type === ZigZag ? 3.5 : 5.0
            if cov(D.slab_dist) isa PDMats.PDiagMat
                flow_Γ = Matrix(I(d))
                flow_μ = zeros(d)
            else
                flow_Γ, flow_μ = inv(Symmetric(cov(D.slab_dist))), mean(D.slab_dist)
                flow_Γ = [
                    flow_Γ          zeros(d - 1, 1)
                    zeros(1, d - 1) 1
                ]
                flow_μ = [flow_μ; 0]
            end
            flow = pdmp_type(flow_Γ, flow_μ)
        else
            flow = pdmp_type(inv(Symmetric(cov(D.slab_dist))), mean(D.slab_dist))
        end

        if algorithm === ThinningStrategy

            if gradient_type === CoordinateWiseGradient
                alg0 = ThinningStrategy(LocalBounds(fill(c0, d)))
            else
                alg0 = ThinningStrategy(GlobalBounds(c0, d))
            end
        else
            alg0 = GridThinningStrategy(hvp = ∇²f!, N = 50)
        end
        if κ isa Function
            can_stick = trues(d)
            if D.spike_dist isa BetaBernoulliHierarchical
                can_stick[end] = false # last coordinate is θ
            end
            alg = Sticky(alg0, κ, can_stick)
        else
            alg = Sticky(alg0, κ)
        end

        x0 = randn(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        # Δϕ = similar(x0)

        # cache = add_gradient_to_cache(initialize_cache(flow, grad, alg, 0.0, ξ0), ξ0)
        # sticky = trues(d)
        # sticky[1:2] .= false
        # x0[.!sticky] .= 0
        # θ0[.!sticky] .= 0
        # state = StickyPDMPState(0.0, SkeletonPoint(copy(x0), copy(θ0)), sticky)
        # validate_state(state, flow)
        # compute_gradient!(state, grad, flow, cache)
        # compute_gradient_uncorrected!(state, grad, flow, cache)
        # ∇f!(Δϕ, x0)
        # reflect!(state, Δϕ, flow, cache)
        # validate_state(state, flow)
        # ξ0.x
        # ξ0.θ
        # reflect!(ξ0, Δϕ, flow, cache)

        grad = gradient_type == CoordinateWiseGradient ? CoordinateWiseGradient(∂fxᵢ) : FullGradient(∇f!)
        trace, stats = pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=show_progress)

        # inds = findall(sticky)
        # inds = sort!(sample(1:d, 3, replace = false))
        # Σ   = randn(d, d)
        # Σ   = Σ' * Σ
        # L   = cholesky(Symmetric(Σ)).L
        # Σs  = Σ[inds, inds]
        # Ls  = cholesky(Symmetric(Σs)).L
        # L[inds, :] * L[inds, :]' ≈ Σs
        # u = randn(d)
        # L[inds, :] * u
        # cov(L[inds, :] * randn(d, 1_000_000), dims = 2) - Σs



        # @profview pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=false)

        acceptance_prob = stats.reflections_accepted / stats.reflections_events
        sticky_events = stats.sticky_events
        refresh_events = stats.refreshment_events
        if (
            !(data_type <: SpikeAndSlabDist{BetaBernoulliHierarchical}) || algorithm !== ThinningStrategy
        ) && !(flow isa Boomerang)
            @test acceptance_prob > 0.55
        end

        # @profview pdmp_sample(ξ0, flow, grad, alg, 0.0, 10_000)

        ts = [event.time for event in trace.events]
        dt = mean(diff(ts)) # Discretization step
        samples = Matrix(PDMPDiscretize(trace, dt))

        test_approximation(samples, D)

        # test_approximation(samples[50_000:end, :], D)
        # mean(!iszero, samples, dims = 1)
        # mean(D.spike_dist)
        # inclusion_probs(trace)
        # cov(trace)
        # cov(samples)
        # sum(mean(D.spike_dist))
        # sum(mean(!iszero, samples, dims = 1))

    end

    end # algorithm
    end # gradient
    end # pdmp

end

if false
@testset "Subsampling" begin

    pdmp_type = ZigZag
    # data_type = GaussianMeanModel
    data_type = LogisticRegressionModel
    algorithm = GridThinningStrategy
    # algorithm = ThinningStrategy # TODO: ab is incorrect when thinning!
    # the main issue is (of course) that ab is now informed by the flow
    # however, this shoud to be informed by the specific subsample!
    # see section 4.1 of Bierkens
    gradient_type = SubsampledGradient#(∇f_sub!)
    d = 5
    n = 1000
    data_arg = (d, n)
    m_perc = 0.05#1 / 10^(floor(Int, log10(n))-2)#0.1
    # [(n, 1 / 10^(floor(Int, log10(n))-1), n / 10^(floor(Int, log10(n))-1)) for n in (10, 100, 1000, 10_000)]
    nsub = floor(Int, data_arg[2] * m_perc)
    show_progress = true

    μ = 3 .+ randn(d)
    Σ = rand(LKJ(d, 3.0))
    # D, ∇f!, ∇²f!, ∂fxᵢ, ∇f_sub!, ∇²f_sub!, resample_indices! = gen_data(data_type, d, n, μ, Symmetric(Σ))
    β_true = randn(d)

    # TODO: missing ∂fxᵢ!
    # also, the subsampling and control variate approaches are mixed.
    # the subsampling works quite poorly though, so perhaps we should abandon that approach.
    if data_type <: GaussianMeanModel
        D, ∇f!, ∇²f!, ∇f_sub!, ∇²f_sub!, resample_indices! = gen_data(data_type, d, n, β_true, μ, Symmetric(Σ))
    else
        D, ∇f!, ∇²f!, ∇f_sub!, ∇²f_sub!, resample_indices!, set_anchor!, anchor_info = gen_data(data_type, d, n, β_true)
    end

    # β_eval = randn(d)
    # out = similar(β_eval)
    # ∇f_true = copy(∇f!(out, β_eval))
    # set_anchor!(β_eval)   # set anchor at truth for testing
    # resample_indices!(n)
    # ∇f_rep = copy(∇f_sub!(out, β_eval))
    # resample_indices!(nsub)
    # ∇f_approx = copy(∇f_sub!(out, β_eval))
    # [∇f_true ∇f_rep ∇f_approx]
    # ∇f_sub_cv_samples = stack(_ -> begin
    #         resample_indices!(nsub)
    #         ∇f_sub!(out, β_true)
    #         copy(out)
    # end, 1:100_000, dims = 1
    # )
    # @btime ∇f!($out, $β_true)
    # @btime ∇f_sub!($out, $β_true)
    # @profview foreach(_->∇f!(out, β_true), 1:10000)
    # @profview foreach(_->∇f_sub!($out, $β_true), 1:10000)
    # @code_warntype ∇f!(out, β_true)
    # @code_warntype ∇f_sub!(out, β_true)
    # ∇f_sub_cv_exp = mean.(eachcol(∇f_sub_cv_samples))
    # ∇f_sub_cv_var = var.(eachcol(∇f_sub_cv_samples))
    # ∇f_sub_cv_exp .- ∇f!(out, β_true)
    # ∇f_sub_cv_qs = stack(x->quantile(x, [.25, .75]), eachcol(∇f_sub_cv_samples))
    # ∇f_sub_cv_qs .- ∇f!(out, β_true)'

    # TODO: test Hessian as well!

    df = DF.DataFrame(D.X, :auto)
    df.y = D.y
    f = StatsModels.@formula(y ~ 0 + x1 + x2 + x3 + x4 + x5)
    m = GLM.glm(f, df, GLM.Binomial(), GLM.LogitLink())
    mle = GLM.coef(m)
    se  = GLM.stderror(m)

    d = first(data_arg) # could also be length(D)

    T = 50_000.0

    if algorithm === ThinningStrategy
        # TODO: this should still be an informed value!
        if gradient_type === CoordinateWiseGradient
            c0 = 1e-2
            alg = ThinningStrategy(LocalBounds(fill(c0, d)))
        elseif gradient_type === FullGradient
            # c0 = 1e-6
            c0 = pdmp_type === ZigZag ? 1e-6 : 1e-2
            alg = ThinningStrategy(GlobalBounds(c0 / d, d))
        else
            c0 = 5
            alg = ThinningStrategy(GlobalBounds(c0 / d, d))
        end
    elseif algorithm === GridThinningStrategy
        alg = GridThinningStrategy(; N = 10, hvp = ∇²f_sub!)
    elseif algorithm === ExactStrategy
        alg = ExactStrategy(analytic_next_event_time_Gaussian)
    end

    # Use same constructor as working test: ZigZag(sparse(I(d)), zeros(d), σ_value)
    # TODO: this constructor should not need a sparse matrix, we can also use an empty constructor?
    # or some default types for zero I
    # flow = pdmp_type(d)

    if data_type <: LogisticRegressionModel
        flow = pdmp_type(d)
    else
        flow = pdmp_type(inv(Symmetric(cov(D))), mean(D))
    end

    x0 = randn(d)
    θ0 = PDMPSamplers.initialize_velocity(flow, d)
    ξ0 = SkeletonPoint(x0, θ0)

    temp_out = similar(x0)
    true_grad = copy(∇f!(temp_out, x0))
    data_type <: GaussianMeanModel && @test -gradlogpdf(D, x0) ≈ true_grad

    mc_est = mean(begin
        resample_indices!(nsub)
        ∇f_sub!(temp_out, x0)
    end
    for _ in 1:100_000)

    # could also use this to inform the test tolerance?
    @test mc_est ≈ true_grad rtol = .1

    # TODO: No speed benefit at the moment!
    # 1. verify correctness of current approach
    # 2. try to make nsub smaller!
    # 3. no allocations probably also helps
    grad_full = FullGradient(∇f!)
    alg_full = GridThinningStrategy(; N = 10, hvp = ∇²f!)
    trace_full, stats_full = pdmp_sample(ξ0, flow, grad_full, alg_full, 0.0, T, progress=show_progress)
    [GLM.coef(m)     mean(trace_full)]
    [GLM.stderror(m) std(trace_full)]

    anchor_info()
    set_anchor!(zeros(d))    # could set this at the mle!
    # set_anchor!(GLM.coef(m)) # for optimal performance and real life this should be set dynamically, based on the sampler or so...

    grad = SubsampledGradient(∇f_sub!, resample_indices!, nsub)
    trace, stats = pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=show_progress)
    [GLM.coef(m)     mean(trace_full)   mean(trace)]
    [GLM.stderror(m) std(trace_full)    std(trace)]

    # @profview pdmp_sample(ξ0, flow, grad, alg, 0.0, 5000, progress=false)

    acceptance_prob = stats.reflections_accepted / stats.reflections_events

    ts = [trace.events[i].time for i in eachindex(trace.events)]
    # diff_ts = diff(ts)
    # obs_qs = quantile(diff_ts, .01:.01:.99)
    # true_qs = quantile(Exponential(), .01:.01:.99)
    # f, ax, _ = scatter(true_qs, obs_qs)
    # ablines!(ax, 0, 1, color=:grey, linestyle=:dash)
    # f

    ispositive(flow.λref) && @test stats.refreshment_events > 100

    @test acceptance_prob > 0.4  # Minimum acceptance rate
    @test length(trace.events) > 100  # Sufficient events generated
    # @profview pdmp_sample(ξ0, flow, grad, alg, 0.0, 500)

    if data_type <: GaussianMeanModel
        ts = [event.time for event in trace.events]
        dt = 100 * mean(diff(ts))
        samples = Matrix(PDMPDiscretize(trace, dt))
        test_approximation(samples, D)
        # mean(D), vec(mean(samples, dims = 1))
        # cov(D), cov(samples)

    else

        ts_full = [event.time for event in trace_full.events]
        dt_full = 10 * mean(diff(ts_full))
        ts_sub  = [event.time for event in trace.events]
        dt_sub  = 10 * mean(diff(ts_sub))
        samples_full = Matrix(PDMPDiscretize(trace_full, dt_full))
        samples_sub  = Matrix(PDMPDiscretize(trace, dt_sub))

        stack(x->StatsBase.autocor(x, 0:20), eachcol(samples_full))
        stack(x->StatsBase.autocor(x, 0:20), eachcol(samples_sub))
        MCMCDiagnosticTools.ess_rhat.(eachcol(samples_full))
        MCMCDiagnosticTools.ess_rhat.(eachcol(samples_sub))

        quant_Δ = .01
        quant_probs = quant_Δ:quant_Δ:1 - quant_Δ
        quantile(samples_full, quant_probs)
        qs_full = stack(x->quantile(x, quant_probs), eachcol(samples_full))
        qs_sub  = stack(x->quantile(x, quant_probs), eachcol(samples_sub))
        for i in axes(qs_full, 2)

            @show i
            slope = cov(qs_full[:, i], qs_sub[:, i]) / var(qs_full[:, i])
            intercept = mean(qs_sub[:, i]) - slope * mean(qs_full[:, i])
            @test -0.25 <= intercept <= 0.25
            @test  0.7  <= slope     <= 1.3 # should be symmetric

            @test qs_full[:, i] ≈ qs_sub[:, i] rtol = .25 atol = .1
        end
        scatter(qs_full[:, i], qs_sub[:, i])

        # visual inspection
        # fig = Figure()
        # ax1 = Axis(fig[1, 1])
        # ax2 = Axis(fig[1, 2])
        # ii = 2
        # density!(ax1, samples_full[:, ii], color = (:plum2, 0.4))
        # density!(ax2, samples_sub[:, ii], color = (:palegreen2, 0.4))
        # fign
        # fig = Figure()
        # gl1 = fig[1, 1] = GridLayout()
        # gl2 = fig[1, 2] = GridLayout()
        # nc = isqrt(d)
        # for i in axes(qs_full, 2)
        #     ir, ic = fldmod1(i, nc)
        #     ax = Axis(gl1[ir, ic], autolimitaspect = 1, title = "dim $i")
        #     ablines!(ax, 0, 1, color=:grey, linestyle=:dash)
        #     scatter!(ax, qs_full[:, i], qs_sub[:, i])
        # end
        # for i in axes(qs_full, 2)
        #     ir, ic = fldmod1(i, nc)
        #     ax = Axis(gl2[ir, ic], title = "dim $i")
        #     density!(ax, samples_full[:, i], color = (:plum2, 0.4))
        #     density!(ax, samples_sub[:, i],  color = (:palegreen2, 0.4))
        # end
        # fig
    end

end
end

@testset "General properties of PDMP flows" begin

    d = 25
    x  = Vector{Float64}(undef, d)
    θ  = Vector{Float64}(undef, d)
    ∇ϕx = Vector{Float64}(undef, d)
    θ2  = Vector{Float64}(undef, d)
    θ3  = Vector{Float64}(undef, d)
    for pdmp_type in (Boomerang, )
    # This does not seem to hold for other flows?
    # for pdmp_type in (ZigZag, BouncyParticle, Boomerang)

        # Generate test problem
        D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(MvNormal, d, 2.0)

        # Create flow
        flow = pdmp_type(inv(Symmetric(cov(D))), mean(D))


        sqrt_inv_Σ = sqrt(flow.Γ)
        # Test multiple random initial conditions
        for _ in 1:5
            randn!(x)
            randn!(θ)
            randn!(∇ϕx)
            copyto!(θ2, θ)

            ξ = SkeletonPoint(x, θ2)
            # flow = Boomerang(sparse(I(d)), zeros(d), 0.1)

            grad = FullGradient((out, x) -> out .= x)
            cache = PDMPSamplers.add_gradient_to_cache(
                PDMPSamplers.initialize_cache(flow, grad, ThinningStrategy(GlobalBounds(1.0, d)), 0.0, ξ),
                ξ
            )
            # state = PDMPState(0.0, ξ)
            # ∇ϕx = PDMPSamplers.compute_gradient_uncorrected!(state, grad, flow, cache)

            # Equation 4 of Bierkens et al., (2020), http://proceedings.mlr.press/v119/bierkens20a.html
            rhs = -dot(ξ.θ, ∇ϕx)
            PDMPSamplers.reflect!(ξ, ∇ϕx, flow, cache)
            lhs = dot(ξ.θ, ∇ϕx)
            @test lhs ≈ rhs

            # Equation 5 of Bierkens et al., (2020), http://proceedings.mlr.press/v119/bierkens20a.html
            lhs = norm(sqrt_inv_Σ * ξ.θ)
            rhs = norm(sqrt_inv_Σ * θ)
            @test lhs ≈ rhs

        end
    end
end

@testset "Discretized and analytic estimators agree" begin

    d = 10
    D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(MvNormal, d, 2.0, 1:d, 1:d)
    grad = FullGradient(∇f!)

    Γ, μ = inv(cov(D)), mean(D)

    # TODO: need to implement these for Boomerang!
    pdmp_types = (ZigZag, BouncyParticle, )
    alg = GridThinningStrategy(hvp = ∇²f!)

    # pdmp_type = ZigZag
    # show_progress = true

    @testset "flow: $(pdmp_type)" for pdmp_type in pdmp_types

        flow = pdmp_type(Γ, μ)
        ξ0 = SkeletonPoint(μ .+ randn(d), PDMPSamplers.initialize_velocity(flow, d))
        T = 50_000.0
        trace, stats = pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=show_progress)

        dt = mean(diff([event.time for event in trace.events]))

        # some checks on the behavior of PDMPDiscretize
        samples0 = collect(PDMPDiscretize(trace, dt))
        @test all(isassigned(samples0, i) for i in eachindex(samples0))
        time_range = PDMPSamplers._to_range(PDMPDiscretize(trace, dt))
        @test first.(samples0) == time_range

        samples = Matrix(PDMPDiscretize(trace, dt))

        min_ess = minimum(MCMCDiagnosticTools.ess.(eachcol(samples)))
        se_factor = 2.0 / sqrt(min_ess)  # 2 standard errors for ~95% confidence

        # Base tolerances (these are reasonable for MCMC)
        base_rtol = 0.1   # 10% relative tolerance
        base_atol = 0.1   # absolute tolerance for near-zero values

        # ESS-adjusted tolerances
        rtol = max(base_rtol, se_factor * 0.5)  # At least 10%, but scale with ESS
        atol = max(base_atol, se_factor * 0.1)
        # MCMCDiagnosticTools.ess.(eachcol(samples))

        @test vec(mean(samples, dims=1)) ≈ mean(trace) rtol=rtol atol=atol
        @test vec(std(samples, dims=1))  ≈ std(trace)  rtol=rtol atol=atol
        @test vec(var(samples, dims=1))  ≈ var(trace)  rtol=rtol atol=atol
        @test cov(samples)               ≈ cov(trace)  rtol=rtol atol=atol
        @test cor(samples)               ≈ cor(trace)  rtol=rtol atol=atol

        # or:
        # for f in (mean, std, var)
        #     @test vec(f(samples, dims=1)) ≈ f(trace) rtol=rtol atol=atol
        # end
        # test_approximation(samples, D)

    end
end

end
