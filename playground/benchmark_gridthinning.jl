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

    return D, Σ_inv, ∇f!, ∇²f!
end

# ── Helper: construct flow for a given dynamics type ─────────────────────────

function make_flow(::Val{:zigzag}, Σ_inv, μ, d)
    ZigZag(Σ_inv, μ)
end

function make_flow(::Val{:bouncy}, Σ_inv, μ, d)
    BouncyParticle(Σ_inv, μ, 1.0)
end

function make_flow(::Val{:boomerang}, Σ_inv, μ, d)
    Boomerang(Σ_inv, μ, 1.0)
end

function make_flow(::Val{:precond_zigzag}, Σ_inv, μ, d)
    PreconditionedZigZag(d)
end

function make_flow(::Val{:precond_bouncy}, Σ_inv, μ, d)
    PreconditionedBPS(d; refresh_rate = 1.0)
end

# ── Helper: run one PDMP sample and return trace + stats ─────────────────────

function run_pdmp(; d::Int, T::Float64, dynamics::Symbol, sticky::Bool, seed::Int = 123)
    D, Σ_inv, ∇f!, ∇²f! = gen_mvnormal(d)

    Random.seed!(seed)
    flow = make_flow(Val(dynamics), Σ_inv, mean(D), d)
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

    # @profview pdmp_sample(ξ0, flow, model, alg, 0.0, 10000.; progress = false)
    # TODO: should really fix this allocation!
    # @profview_allocs pdmp_sample(ξ0, flow, model, alg, 0.0, 10000.; progress = false)

    trace, stats = pdmp_sample(ξ0, flow, model, alg, 0.0, T; progress = false)
    return trace, stats
end

# ── Correctness check ────────────────────────────────────────────────────────

function check_correctness(; d::Int, T::Float64, dynamics::Symbol, sticky::Bool,
                             atol_mean::Float64, atol_var::Float64)
    D, _ = gen_mvnormal(d)
    trace, stats = run_pdmp(; d, T, dynamics, sticky)

    ts = [event.time for event in trace.events]
    dt = mean(diff(ts))
    samples = Matrix(PDMPDiscretize(trace, dt))

    est_mean = vec(mean(samples; dims = 1))
    est_var  = vec(var(samples; dims = 1))
    true_mean = mean(D)
    true_var  = diag(cov(D))

    mean_ok = all(abs.(est_mean .- true_mean) .< atol_mean)
    rel_var_err = abs.(est_var .- true_var) ./ max.(abs.(true_var), 1.0)
    var_ok  = all(rel_var_err .< atol_var)

    return (; mean_ok, var_ok, est_mean, est_var, true_mean, true_var,
              stats, n_events = length(trace.events))
end

# ── Performance metrics ──────────────────────────────────────────────────────

function collect_metrics(; d::Int, T::Float64, dynamics::Symbol, sticky::Bool)
    _, stats = run_pdmp(; d, T, dynamics, sticky)
    acc_rate = stats.reflections_accepted / max(stats.reflections_events, 1)
    return (;
        d, T, dynamics, sticky,
        grad_calls   = stats.∇f_calls,
        hvp_calls    = stats.∇²f_calls,
        reflections  = stats.reflections_events,
        accepted     = stats.reflections_accepted,
        refreshments = stats.refreshment_events,
        sticky_evts  = stats.sticky_events,
        acc_rate,
        grid_builds       = stats.grid_builds,
        grid_shrinks      = stats.grid_shrinks,
        grid_grows        = stats.grid_grows,
        grid_early_stops  = stats.grid_early_stops,
        grid_pts_eval     = stats.grid_points_evaluated,
        grid_pts_skip     = stats.grid_points_skipped,
        grid_N_final      = stats.grid_N_current,
    )
end

# ── Benchmark timing ─────────────────────────────────────────────────────────

function benchmark_pdmp(; d::Int, T::Float64, dynamics::Symbol, sticky::Bool, n_runs::Int = 5)
    times = Vector{Float64}(undef, n_runs)
    bytes = Vector{Int}(undef, n_runs)
    for i in 1:n_runs
        stats = @timed run_pdmp(; d, T, dynamics, sticky)
        times[i] = stats.time
        bytes[i] = stats.bytes
    end
    return (; times, bytes)
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    d = 15
    T_long  = 10_000.0
    T_short =  5_000.0

    configs = [
        (d = d, T = T_long,  dynamics = :zigzag,         sticky = false, label = "ZigZag"),
        (d = d, T = T_long,  dynamics = :zigzag,         sticky = true,  label = "ZigZag sticky"),
        (d = d, T = T_long,  dynamics = :bouncy,         sticky = false, label = "BouncyParticle"),
        (d = d, T = T_long,  dynamics = :bouncy,         sticky = true,  label = "BouncyParticle sticky"),
        (d = d, T = T_short, dynamics = :boomerang,      sticky = false, label = "Boomerang"),
        (d = d, T = T_long,  dynamics = :precond_zigzag, sticky = false, label = "PrecondZigZag"),
        (d = d, T = T_long,  dynamics = :precond_bouncy, sticky = false, label = "PrecondBPS"),
    ]

    println("=" ^ 80)
    println("GridThinningStrategy benchmark — d=$d, $(Threads.nthreads()) thread(s)")
    println("=" ^ 80)

    # ── correctness ──────────────────────────────────────────────────────
    println("\n── Correctness checks ─────────────────────────────────────")
    for cfg in configs
        atol_mean = 1.0
        atol_var  = 1.0
        if cfg.sticky
            atol_mean *= 2.0
            atol_var  *= 2.0
        end
        if cfg.dynamics in (:precond_zigzag, :precond_bouncy)
            atol_mean *= 1.5
            atol_var  *= 1.5
        end

        result = check_correctness(; d = cfg.d, T = cfg.T, dynamics = cfg.dynamics,
                                     sticky = cfg.sticky, atol_mean, atol_var)
        status_mean = result.mean_ok ? "PASS" : "FAIL"
        status_var  = result.var_ok  ? "PASS" : "FAIL"
        println("  $(rpad(cfg.label, 25)) mean=$(status_mean), var=$(status_var), events=$(result.n_events)")

        if !result.mean_ok
            max_err = maximum(abs.(result.est_mean .- result.true_mean))
            println("    mean error: max abs diff = $(round(max_err; digits = 4))")
        end
        if !result.var_ok
            max_err = maximum(abs.(result.est_var .- result.true_var) ./ max.(abs.(result.true_var), 1.0))
            println("    var error:  max rel diff = $(round(max_err; digits = 4))")
        end
    end

    # ── metrics ──────────────────────────────────────────────────────────
    println("\n── Performance metrics ─────────────────────────────────────")
    println(rpad("Config", 28), rpad("∇f", 10), rpad("HVP", 10), rpad("Reflect", 10),
            rpad("Accept", 10), rpad("Refresh", 10), rpad("Acc%", 8),
            rpad("∇f/evt", 9), rpad("HVP/evt", 9))
    println("-" ^ 114)
    for cfg in configs
        m = collect_metrics(; d = cfg.d, T = cfg.T, dynamics = cfg.dynamics, sticky = cfg.sticky)
        n_events = m.reflections + m.refreshments + m.sticky_evts
        grad_per_evt = round(m.grad_calls / max(n_events, 1); digits = 1)
        hvp_per_evt  = round(m.hvp_calls / max(n_events, 1); digits = 1)
        println(rpad(cfg.label, 28),
                rpad(string(m.grad_calls), 10),
                rpad(string(m.hvp_calls), 10),
                rpad(string(m.reflections), 10),
                rpad(string(m.accepted), 10),
                rpad(string(m.refreshments), 10),
                rpad(string(round(m.acc_rate * 100; digits = 1)) * "%", 8),
                rpad(string(grad_per_evt), 9),
                rpad(string(hvp_per_evt), 9))
    end

    # ── grid statistics ──────────────────────────────────────────────────
    println("\n── Grid statistics ─────────────────────────────────────────")
    println(rpad("Config", 28), rpad("Builds", 10), rpad("Shrinks", 10), rpad("Grows", 10),
            rpad("EarlyStop", 10), rpad("PtsEval", 12), rpad("PtsSkip", 12),
            rpad("Avg/build", 10), rpad("N_final", 8))
    println("-" ^ 120)
    for cfg in configs
        m = collect_metrics(; d = cfg.d, T = cfg.T, dynamics = cfg.dynamics, sticky = cfg.sticky)
        avg_pts = round(m.grid_pts_eval / max(m.grid_builds, 1); digits = 1)
        println(rpad(cfg.label, 28),
                rpad(string(m.grid_builds), 10),
                rpad(string(m.grid_shrinks), 10),
                rpad(string(m.grid_grows), 10),
                rpad(string(m.grid_early_stops), 10),
                rpad(string(m.grid_pts_eval), 12),
                rpad(string(m.grid_pts_skip), 12),
                rpad(string(avg_pts), 10),
                rpad(string(m.grid_N_final), 8))
    end

    # ── timing ───────────────────────────────────────────────────────────
    println("\n── Timing (@timed, n=5) ────────────────────────────────────")
    for cfg in configs
        print("  $(rpad(cfg.label, 25)): ")
        run_pdmp(; d = cfg.d, T = min(cfg.T, 100.0), dynamics = cfg.dynamics, sticky = cfg.sticky)
        b = benchmark_pdmp(; d = cfg.d, T = cfg.T, dynamics = cfg.dynamics, sticky = cfg.sticky, n_runs = 5)
        t_median = sort(b.times)[3] * 1000
        t_min    = minimum(b.times) * 1000
        mem      = median(b.bytes) / 1024^2
        println("median=$(round(t_median; digits = 1))ms, min=$(round(t_min; digits = 1))ms, mem=$(round(mem; digits = 1))MiB")
    end

    println("\n" * "=" ^ 80)
    println("Done.")
end

main()
