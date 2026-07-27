# Internal helper callables used by generated precompile signatures.
# They keep src/precompile_statements.jl independent of test-local closures.
function _precompile_neg_gradient!(out::AbstractVector, x::AbstractVector)
    copyto!(out, x)
    return out
end

function _precompile_neg_hvp!(out::AbstractVector, x::AbstractVector, v::AbstractVector)
    copyto!(out, v)
    return out
end

function _precompile_model(d::Integer)
    return PDMPModel(Int(d), FullGradient(_precompile_neg_gradient!), _precompile_neg_hvp!)
end

function _precompile_phase_signature(flow::ContinuousDynamics, alg::PoissonTimeStrategy, d::Int)
    # Build representative argument types only; `precompile` does not run a sampler phase.
    rng = Random.Xoshiro(1234 + d)
    model = _precompile_model(d)
    θ = initialize_velocity(flow, d)
    ξ = SkeletonPoint(collect(range(0.1, 0.1d; length=d)), θ)
    state, model_, alg_, cache, stats = initialize_state(rng, flow, model, alg, 0.0, ξ)
    trace_manager = TraceManager(state, flow, alg, 0.0)
    criterion = _CommonPhaseCriterion(; max_events=2)
    args = (rng, criterion, state, model_, flow, alg_, cache, trace_manager, stats, HealthMonitor(), :main, NoAdaptation(), false, nothing, Ref(Inf), 10.0, 300, NoBoundaryHandling(), model, SupportBoundaryOptions())
    precompile(Tuple{typeof(_run_phase_for_policy!), map(typeof, args)...})
    return nothing
end

function _precompile_curated_deep_signatures()
    d = 2
    matrix_metric = Matrix{Float64}(I, d, d)
    origin = zeros(d)
    _precompile_phase_signature(ZigZag(matrix_metric, origin), GridThinningStrategy(), d)
    _precompile_phase_signature(BouncyParticle(matrix_metric, origin), GridThinningStrategy(), d)
    _precompile_phase_signature(ZigZag(matrix_metric, origin), Sticky(GridThinningStrategy(; N=10), fill(0.2, d)), d)
    _precompile_phase_signature(BouncyParticle(matrix_metric, origin), ThinningStrategy(GlobalBounds(2.0, d)), d)
    return nothing
end

include("precompile_statements.jl")
