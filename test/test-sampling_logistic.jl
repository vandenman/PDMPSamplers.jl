@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

function find_map(∇f!::Function, ∇²f!::Function, d::Int; x0::Vector{Float64}=zeros(d), maxiter::Int=50, tol::Float64=1e-8)
    x = copy(x0)
    g = similar(x)
    Hv = similar(x)
    H = zeros(d, d)
    e_j = zeros(d)
    for _ in 1:maxiter
        ∇f!(g, x)
        norm(g) < tol && break
        for j in 1:d
            fill!(e_j, 0.0)
            e_j[j] = 1.0
            ∇²f!(Hv, x, e_j)
            H[:, j] .= Hv
        end
        x .-= H \ g
    end
    return x
end

function test_logistic_approximation(trace, β_map::Vector{Float64}; name::String="LogReg")
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
        println("$label | $(rpad(_flow_name(trace), 22)) | $(rpad(name, 16)) | ESS=$(lpad(round(Int, min_ess), 7)) | c_mean=$(_f3(c_mean))")
    end
end

@testset "Logistic Regression PDMP Tests" begin

    d, n = 3, 200
    β_gen = [0.5, 1.0, -0.8]

    Random.seed!(2024)
    obj, ∇f!, ∇²f!, ∇f_sub_cv!, ∇²f_sub!, resample_indices!, set_anchor!, anchor_info, β_true =
        gen_data(LogisticRegressionModel, d, n, β_gen)

    β_map = find_map(∇f!, ∇²f!, d)

    pdmp_types = (ZigZag, BouncyParticle, Boomerang, PreconditionedZigZag, PreconditionedBPS)

    @testset "Full gradient" begin
        @testset "$pdmp_type" for pdmp_type in pdmp_types

            Random.seed!(hash((pdmp_type, :logistic_full)))

            T = 15_000.0

            flow = if pdmp_type <: PreconditionedDynamics
                pdmp_type(d)
            else
                pdmp_type(Matrix(1.0I(d)), zeros(d))
            end

            x0 = zeros(d)
            θ0 = PDMPSamplers.initialize_velocity(flow, d)
            ξ0 = SkeletonPoint(x0, θ0)

            grad = FullGradient(∇f!)
            model = PDMPModel(d, grad, ∇²f!)
            alg = GridThinningStrategy()

            trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

            acceptance_prob = stats.reflections_accepted / stats.reflections_events
            if !(flow isa Boomerang)
                @test acceptance_prob > 0.4
            end
            @test length(trace.events) > 100

            test_logistic_approximation(trace, β_map; name="LogReg($d,$n)")
        end
    end

    @testset "Subsampled gradient (ZigZag)" begin
        Random.seed!(hash((:zigzag, :logistic_sub)))

        nsub = n ÷ 10
        T = 20_000.0

        grad = SubsampledGradient(∇f_sub_cv!, resample_indices!, nsub)
        flow = ZigZag(Matrix(1.0I(d)), zeros(d))
        model = PDMPModel(d, grad, ∇²f_sub!)
        alg = GridThinningStrategy()

        x0 = zeros(d)
        θ0 = PDMPSamplers.initialize_velocity(flow, d)
        ξ0 = SkeletonPoint(x0, θ0)

        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

        @test length(trace.events) > 100

        test_logistic_approximation(trace, β_map; name="LogReg($d,$n) sub")
    end

    @testset "Sticky ZigZag" begin
        d_s, n_s = 5, 200
        β_gen_s = [0.5, 0.0, 1.0, 0.0, -0.8]

        Random.seed!(2024)
        obj_s, ∇f_s!, ∇²f_s!, _, _, _, _, _, β_true_s =
            gen_data(LogisticRegressionModel, d_s, n_s, β_gen_s)

        β_map_s = find_map(∇f_s!, ∇²f_s!, d_s)

        # spike-and-slab sticking rates: κᵢ = p₀/(1-p₀) × N(0; μ₀ᵢ, σ₀ᵢ²)
        p0 = 0.5
        slab_pdf_zero = [pdf(Normal(obj_s.prior_μ[i], sqrt(obj_s.prior_Σ[i, i])), 0.0) for i in 1:d_s]
        κ = (p0 / (1 - p0)) .* slab_pdf_zero
        κ[1] = Inf  # intercept never sticks

        Random.seed!(hash((:zigzag, :logistic_sticky)))

        T = 20_000.0
        flow = ZigZag(Matrix(1.0I(d_s)), zeros(d_s))

        alg0 = GridThinningStrategy(; N=50)
        alg = Sticky(alg0, κ)

        x0 = randn(d_s)
        θ0 = PDMPSamplers.initialize_velocity(flow, d_s)
        ξ0 = SkeletonPoint(x0, θ0)

        grad = FullGradient(∇f_s!)
        model = PDMPModel(d_s, grad, ∇²f_s!)

        trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress=show_progress)

        @test length(trace.events) > 100

        # non-zero coefficients should have higher inclusion probability than zero ones
        incl = inclusion_probs(trace)
        @test incl[3] > incl[2]
        @test incl[5] > incl[4]

        if show_test_diagnostics
            incl_str = join([_f3(incl[i]) for i in 1:d_s], ", ")
            min_ess = minimum(ess(trace))
            println("ok   | $(rpad(_flow_name(trace), 22)) | $(rpad("LogReg($d_s,$n_s) sticky", 26)) | ESS=$(lpad(round(Int, min_ess), 7)) | incl=[$incl_str]")
        end
    end

end
