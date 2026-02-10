#=
    Benchmark and correctness test for GridThinningStrategy.

    This file benchmarks the current implementation and provides correctness checks
    to verify that performance improvements do not break sampling.

    Usage:
        julia --startup-file=no --project=playground playground/benchmark_gridthinning.jl
=#

using PDMPSamplers
using Distributions
using LinearAlgebra
using Random
using Statistics

# ── Helper: generate MvNormal target ─────────────────────────────────────────

function gen_mvnormal(d::Int; η::Float64 = 2.0, seed::Int = 42)
    Random.seed!(seed)
    μ = rand(Normal(0, 5), d)
    σ = rand(LogNormal(0, 1), d)
    R = rand(LKJ(d, η))
    Σ = Symmetric(Diagonal(σ) * R * Diagonal(σ))
    Σ_inv = inv(Σ)
    potential = Σ_inv * μ
    buffer = similar(potential)
    D = MvNormal(μ, Σ)

    ∇f! = (out, x) -> begin
        mul!(buffer, Σ_inv, x)
        buffer .-= potential
        out .= buffer
    end

    ∇²f! = (out, _, v) -> mul!(out, Σ_inv, v)

    ∂fxᵢ = (x, i) -> dot(view(Σ_inv, :, i), x) - potential[i]

    return D, ∇f!, ∇²f!, ∂fxᵢ
end

# ── Helper: run one PDMP sample and return trace + stats ─────────────────────

function run_pdmp(; d::Int, T::Float64, sticky::Bool, seed::Int = 123)
    D, ∇f!, ∇²f!, _ = gen_mvnormal(d)

    Random.seed!(seed)
    flow = ZigZag(inv(Symmetric(cov(D))), mean(D))
    x0 = randn(d)
    θ0 = PDMPSamplers.initialize_velocity(flow, d)
    ξ0 = SkeletonPoint(x0, θ0)

    model = PDMPModel(d, FullGradient(∇f!), ∇²f!)

    if sticky
        κ = fill(0.5, d)
        alg = Sticky(GridThinningStrategy(), κ)
    else
        alg = GridThinningStrategy()
    end

    trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress = false)
    return trace, stats
end

# ── Correctness check ────────────────────────────────────────────────────────

function check_correctness(; d::Int, T::Float64, sticky::Bool, atol_mean::Float64, atol_var::Float64)
    D, _, _, _ = gen_mvnormal(d)
    trace, stats = run_pdmp(; d, T, sticky)

    ts = [event.time for event in trace.events]
    dt = mean(diff(ts))
    samples = Matrix(PDMPDiscretize(trace, dt))

    est_mean = vec(mean(samples; dims = 1))
    est_var  = vec(var(samples; dims = 1))
    true_mean = mean(D)
    true_var  = diag(cov(D))

    mean_ok = all(abs.(est_mean .- true_mean) .< atol_mean)
    # use relative tolerance for variance since scales vary across dimensions
    rel_var_err = abs.(est_var .- true_var) ./ max.(abs.(true_var), 1.0)
    var_ok  = all(rel_var_err .< atol_var)

    return (; mean_ok, var_ok, est_mean, est_var, true_mean, true_var,
              stats, n_events = length(trace.events))
end

# ── Performance metrics ──────────────────────────────────────────────────────

function collect_metrics(; d::Int, T::Float64, sticky::Bool)
    _, stats = run_pdmp(; d, T, sticky)
    acc_rate = stats.reflections_accepted / max(stats.reflections_events, 1)
    return (;
        d, T, sticky,
        grad_calls   = stats.∇f_calls,
        hvp_calls    = stats.∇²f_calls,
        reflections  = stats.reflections_events,
        accepted     = stats.reflections_accepted,
        refreshments = stats.refreshment_events,
        sticky_evts  = stats.sticky_events,
        acc_rate,
    )
end

# ── Benchmark timing ─────────────────────────────────────────────────────────

function benchmark_pdmp(; d::Int, T::Float64, sticky::Bool, n_runs::Int = 5)
    times = Vector{Float64}(undef, n_runs)
    bytes = Vector{Int}(undef, n_runs)
    for i in 1:n_runs
        stats = @timed run_pdmp(; d, T, sticky)
        times[i] = stats.time
        bytes[i] = stats.bytes
    end
    return (; times, bytes)
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    configs = [
        (d =  5, T = 50_000.0, sticky = false, label = "ZigZag d=5"),
        (d =  5, T = 50_000.0, sticky = true,  label = "ZigZag d=5, sticky"),
        (d = 50, T =  2_000.0, sticky = false, label = "ZigZag d=50"),
        (d = 50, T =  2_000.0, sticky = true,  label = "ZigZag d=50, sticky"),
    ]

    println("=" ^ 80)
    println("GridThinningStrategy benchmark — $(Threads.nthreads()) thread(s)")
    println("=" ^ 80)

    # ── correctness ──────────────────────────────────────────────────────
    println("\n── Correctness checks ─────────────────────────────────────")
    for cfg in configs

        # more tolerance for sticky since sparsity introduces bias in marginal variance
        atol_mean = cfg.d <= 10 ? 0.5 : 1.5
        atol_var  = cfg.d <= 10 ? 0.5 : 2.0  # relative tolerance
        if cfg.sticky
            atol_mean *= 2.0
            atol_var  *= 2.0
        end

        result = check_correctness(; d = cfg.d, T = cfg.T, sticky = cfg.sticky, atol_mean, atol_var)
        status_mean = result.mean_ok ? "PASS" : "FAIL"
        status_var  = result.var_ok  ? "PASS" : "FAIL"
        println("  $(cfg.label): mean=$(status_mean), var=$(status_var), events=$(result.n_events)")

        if !result.mean_ok
            max_err = maximum(abs.(result.est_mean .- result.true_mean))
            println("    mean error: max abs diff = $(round(max_err; digits = 4))")
        end
        if !result.var_ok
            max_err = maximum(abs.(result.est_var .- result.true_var))
            println("    var error:  max abs diff = $(round(max_err; digits = 4))")
        end
    end

    # ── metrics ──────────────────────────────────────────────────────────
    println("\n── Performance metrics ─────────────────────────────────────")
    println(rpad("Config", 30), rpad("∇f", 12), rpad("HVP", 12), rpad("Reflect", 12),
            rpad("Accept", 12), rpad("Acc%", 8), rpad("\u2207f/evt", 10), rpad("HVP/evt", 10))
    println("-" ^ 106)
    for cfg in configs
        m = collect_metrics(; d = cfg.d, T = cfg.T, sticky = cfg.sticky)
        n_events = m.reflections + m.refreshments + m.sticky_evts
        grad_per_evt = round(m.grad_calls / max(n_events, 1); digits = 1)
        hvp_per_evt  = round(m.hvp_calls / max(n_events, 1); digits = 1)
        println(rpad(cfg.label, 30),
                rpad(string(m.grad_calls), 12),
                rpad(string(m.hvp_calls), 12),
                rpad(string(m.reflections), 12),
                rpad(string(m.accepted), 12),
                rpad(string(round(m.acc_rate * 100; digits = 1)) * "%", 8),
                rpad(string(grad_per_evt), 10),
                rpad(string(hvp_per_evt), 10))
    end

    # ── timing ───────────────────────────────────────────────────────────
    println("\n── Timing (@timed, n=5) ────────────────────────────────────")
    for cfg in configs
        print("  $(cfg.label): ")
        # warmup
        run_pdmp(; d = cfg.d, T = min(cfg.T, 100.0), sticky = cfg.sticky)
        b = benchmark_pdmp(; d = cfg.d, T = cfg.T, sticky = cfg.sticky, n_runs = 5)
        t_median = sort(b.times)[3] * 1000  # ms
        t_min    = minimum(b.times) * 1000
        mem      = median(b.bytes) / 1024^2  # MiB
        println("median=$(round(t_median; digits = 1))ms, min=$(round(t_min; digits = 1))ms, mem=$(round(mem; digits = 1))MiB")
    end

    println("\n" * "=" ^ 80)
    println("Done.")
end

main()
