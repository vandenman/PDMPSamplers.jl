@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function find_map(target, d::Int; x0::Vector{Float64}=zeros(d), maxiter::Int=50, tol::Float64=1e-8)
    x = copy(x0)
    g = similar(x)
    Hv = similar(x)
    H = zeros(d, d)
    e_j = zeros(d)
    for _ in 1:maxiter
        neg_gradient!(target, g, x)
        norm(g) < tol && break
        for j in 1:d
            fill!(e_j, 0.0)
            e_j[j] = 1.0
            neg_hvp!(target, Hv, x, e_j)
            H[:, j] .= Hv
        end
        x .-= H \ g
    end
    return x
end

function test_logistic_approximation(trace, β_map::Vector{Float64}; name::String="LogReg", elapsed::Union{Real,Nothing}=nothing)
    d = length(β_map)
    min_ess = minimum(ess(trace))
    if min_ess < 100
        show_test_diagnostics && println("SKIP | $(rpad(_flow_name(trace), 22)) | $(rpad(name, 16)) | ESS=$(lpad(round(Int, min_ess), 7)) (too low)")
        return nothing
    end

    mc = 1.0 / sqrt(min_ess)
    mean_atol = 0.3 + 3.0 * mc

    trace_mean = mean(trace)
    @test isapprox(trace_mean, β_map; atol=mean_atol)

    c_mean = _isapprox_closeness(trace_mean, β_map; atol=mean_atol)
    failed = c_mean > 1.0
    if failed || show_test_diagnostics
        label = failed ? "FAIL" : "ok  "
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(name, 16)) | ESS=$(lpad(round(Int, min_ess), 7)) | $(_format_elapsed(elapsed)) | c_mean=$(_f3(c_mean))")
    end

    _record_test!(; flow=_flow_name(trace), target=name, d=length(β_map),
                   min_ess, elapsed, passed=!failed)
end

@testset "Logistic Regression PDMP Tests" begin

    d, n = 3, 200
    β_gen = [0.5, 1.0, -0.8]

    Random.seed!(2024)
    target = gen_data(LogisticRegressionModel, d, n, β_gen)

    β_map = find_map(target, d)

    pdmp_types = (ZigZag, BouncyParticle, Boomerang, MutableBoomerang, PreconditionedZigZag, PreconditionedBPS)

    @testset "Full gradient" begin
        @testset "$pdmp_type" for pdmp_type in pdmp_types

            Random.seed!(stable_test_seed(pdmp_type, :logistic_full))

            T = 15_000.0

            flow = if pdmp_type <: PreconditionedDynamics
                pdmp_type(d)
            else
                pdmp_type(Matrix(1.0I(d)), zeros(d))
            end

            x0 = zeros(d)
            θ0 = PDMPSamplers.initialize_velocity(flow, d)
            ξ0 = SkeletonPoint(x0, θ0)

            grad = FullGradient(Base.Fix1(neg_gradient!, target))
            model = PDMPModel(d, grad, Base.Fix1(neg_hvp!, target))
            alg = GridThinningStrategy()

            trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

            acceptance_prob = stats.reflections_accepted / stats.reflections_events
            if !(flow isa AnyBoomerang)
                @test acceptance_prob > 0.4
            end
            @test length(trace) > 100

            test_logistic_approximation(trace, β_map; name="LogReg($d,$n)", elapsed=stats.elapsed_time)
        end
    end

    @testset "Sticky ZigZag" begin
        d_s, n_s = 5, 200
        β_gen_s = [0.5, 0.0, 1.0, 0.0, -0.8]

        Random.seed!(2024)
        target_s = gen_data(LogisticRegressionModel, d_s, n_s, β_gen_s)

        β_map_s = find_map(target_s, d_s)

        # spike-and-slab sticking rates: κᵢ = p₀/(1-p₀) × N(0; μ₀ᵢ, σ₀ᵢ²)
        p0 = 0.5
        slab_pdf_zero = [pdf(Normal(target_s.obj.prior_μ[i], sqrt(target_s.obj.prior_Σ[i, i])), 0.0) for i in 1:d_s]
        κ = (p0 / (1 - p0)) .* slab_pdf_zero
        κ[1] = Inf  # intercept never sticks

        Random.seed!(stable_test_seed(:zigzag, :logistic_sticky))

        T = 20_000.0
        flow = ZigZag(Matrix(1.0I(d_s)), zeros(d_s))

        alg0 = GridThinningStrategy(; N=50)
        alg = Sticky(alg0, κ)

        x0 = randn(d_s)
        θ0 = PDMPSamplers.initialize_velocity(flow, d_s)
        ξ0 = SkeletonPoint(x0, θ0)

        grad = FullGradient(Base.Fix1(neg_gradient!, target_s))
        model = PDMPModel(d_s, grad, Base.Fix1(neg_hvp!, target_s))

        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

        @test length(trace) > 100

        # non-zero coefficients should have higher inclusion probability than zero ones
        incl = inclusion_probs(trace)
        @test incl[3] > incl[2]
        @test incl[5] > incl[4]

        if show_test_diagnostics
            incl_str = join([_f3(incl[i]) for i in 1:d_s], ", ")
            min_ess = minimum(ess(trace))
            println("ok   | $(rpad(_flow_name(trace), 22)) | $(rpad("LogReg($d_s,$n_s) sticky", 26)) | ESS=$(lpad(round(Int, min_ess), 7)) | $(_format_elapsed(stats.elapsed_time)) | incl=[$incl_str]")
        end
    end

    @testset "Affine-logistic subsampling integration" begin
        rng = Random.Xoshiro(0x1091571c)
        N_sub, d_sub, minibatch = 40, 2, 5
        X_sub = randn(rng, N_sub, d_sub)
        β_true_sub = [0.4, -0.7]
        y_sub = Float64.(rand(rng, N_sub) .<
            LogExpFunctions.logistic.(X_sub * β_true_sub))
        anchor = zeros(d_sub)
        prior_precision = 0.25
        logistic_grad! = function (out, x, rows)
            fill!(out, 0.0)
            for i in rows
                η = dot(view(X_sub, i, :), x)
                out .+= (LogExpFunctions.logistic(η) - y_sub[i]) .*
                    view(X_sub, i, :)
            end
            return out
        end
        anchor_likelihood = zeros(d_sub)
        logistic_grad!(anchor_likelihood, anchor, axes(X_sub, 1))
        deterministic! = (out, x) ->
            (out .= anchor_likelihood .+ prior_precision .* x)
        residual_oracle! = function (out, x, subset, a)
            logistic_grad!(out, x, subset)
            anchor_part = zeros(d_sub)
            logistic_grad!(anchor_part, a, subset)
            out .-= anchor_part
        end
        weights = reshape([0.25 * sum(abs2, view(X_sub, i, :))
            for i in axes(X_sub, 1)], 1, :)
        envelope = SeparableResidualEnvelope(weights,
            (out, state, flow, t) -> begin
                vnorm = norm(state.ξ.θ)
                out[1] = vnorm *
                    (norm(state.ξ.x - anchor) + vnorm * t)
            end; certified_affine=true)
        cv = SubsampledControlVariate(deterministic!, residual_oracle!,
            envelope, anchor, minibatch;
            deterministic_hvp! = ((out, x, v) ->
                (out .= prior_precision .* v)))
        subsampling_model = PDMPModel(d_sub, cv)
        chain_model_a = copy(subsampling_model)
        chain_model_b = copy(subsampling_model)
        @test chain_model_a.grad.anchor !== chain_model_b.grad.anchor
        @test chain_model_a.grad.residual_buffer !==
            chain_model_b.grad.residual_buffer
        @test chain_model_a.grad.envelope !== chain_model_b.grad.envelope

        full_gradient! = function (out, x)
            logistic_grad!(out, x, axes(X_sub, 1))
            out .+= prior_precision .* x
        end
        full_hvp! = function (out, x, v)
            out .= prior_precision .* v
            for i in axes(X_sub, 1)
                xi = view(X_sub, i, :)
                p = LogExpFunctions.logistic(dot(xi, x))
                out .+= (p * (1 - p) * dot(xi, v)) .* xi
            end
        end
        full_model = PDMPModel(d_sub, FullGradient(full_gradient!), full_hvp!)
        flow = BouncyParticle(d_sub, 0.7)
        ξ = SkeletonPoint([0.2, -0.2], [1.0, 0.0])
        alg = GridThinningStrategy(N=16, t_max=1.5, lazy=false,
            bound_violation=:throw)
        full_chains = pdmp_sample(ξ, flow, full_model, alg, 0.0, 8_000.0;
            seed=809, progress=false)
        full_trace = full_chains.traces[1]
        subsampling_chains = pdmp_sample(ξ, flow, subsampling_model, alg,
            0.0, 8_000.0; n_chains=2, threaded=false, seed=[808, 810],
            progress=false)

        @test n_chains(subsampling_chains) == 2
        for chain in 1:2
            subsampling_trace, subsampling_stats = subsampling_chains[chain]
            @test length(subsampling_trace) > 100
            @test mean(subsampling_trace) ≈ mean(full_trace) atol=0.18
            @test diag(cov(subsampling_trace)) ≈
                diag(cov(full_trace)) atol=0.2
            @test subsampling_stats.residual_oracle_calls > 0
            @test subsampling_stats.full_gradient_calls == 0
        end
    end

end
