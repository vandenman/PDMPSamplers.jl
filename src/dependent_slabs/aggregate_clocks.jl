"""
    active_prior_neggrad!(provider, out, x, active_beta)

Compatibility name for the active slab negative-gradient contract. New
providers should think of this as the canonical sign convention; historical
Gaussian providers also dispatch through `active_prior_grad!`.
"""
active_prior_neggrad!(provider::AbstractGaussianSlabProvider, out::AbstractVector, x::AbstractVector, active_beta::BitVector) =
    active_prior_grad!(provider, out, x, active_beta)

function active_prior_neggrad!(provider::ArbitrarySlabBoundary, out::AbstractVector, x::AbstractVector, active_beta::BitVector)
    provider.active_prior_neggrad!(out, x, active_beta)
    return out
end

active_prior_grad!(provider::ArbitrarySlabBoundary, out::AbstractVector, x::AbstractVector, active_beta::BitVector) =
    active_prior_neggrad!(provider, out, x, active_beta)

"""
    SummedRateClock(slab_provider, model_prior; rtol=1e-8, atol=1e-10,
                    initial_bracket=1.0, bracket_multiplier=2.0)

Generic aggregate unstick clock. It evaluates the summed rate over all inactive
stickable beta coordinates by calling the provider's boundary densities and the
model prior. This is the flexible arbitrary-prior baseline; it may allocate
and uses numerical quadrature/root finding for inhomogeneous clocks.
"""
mutable struct SummedRateClockCache
    active_beta::BitVector
    stickable_beta::BitVector
    log_weights::Vector{Float64}
end

function SummedRateClockCache(m::Integer)
    return SummedRateClockCache(BitVector(undef, m), BitVector(undef, m), Vector{Float64}(undef, m))
end

struct SummedRateClock{P<:AbstractSlabPrior,O<:AbstractModelPrior,T<:Real} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior::O
    rtol::T
    atol::T
    initial_bracket::T
    bracket_multiplier::T
    cache::SummedRateClockCache
end

"""
    AggregateClockDiagnostics

Mutable counters for residual-envelope aggregate clocks. `residual_area` is
reserved for certified proposal-envelope implementations; exact fallback
sampling leaves residual and envelope counters at zero and increments
`fallbacks`.
"""
mutable struct AggregateClockDiagnostics
    proposals::Int
    accepted::Int
    rejected::Int
    fallbacks::Int
    residual_area::Float64
    envelope_hazard::Float64
    max_envelope_ratio::Float64
    max_residual::Float64
    min_envelope::Float64
    rate_evaluations::Int
    last_cells::Int
end

AggregateClockDiagnostics() = AggregateClockDiagnostics(0, 0, 0, 0, 0.0, 0.0, 1.0, 0.0, Inf, 0, 0)

function reset_thinning_diagnostics!(diagnostics::AggregateClockDiagnostics)
    diagnostics.proposals = 0
    diagnostics.accepted = 0
    diagnostics.rejected = 0
    diagnostics.fallbacks = 0
    diagnostics.residual_area = 0.0
    diagnostics.envelope_hazard = 0.0
    diagnostics.max_envelope_ratio = 1.0
    diagnostics.max_residual = 0.0
    diagnostics.min_envelope = Inf
    diagnostics.rate_evaluations = 0
    diagnostics.last_cells = 0
    return diagnostics
end

"""
    ChebyshevResidualAggregateClock(slab_provider, model_prior; ...)

Phase-6 moving-scale aggregate clock entry point for linear-flow residual
envelopes. Structured scalar residual paths use cell-wise analytic bounds with
conservative floating-point safety inflation. Unsupported providers route
through the exact `SummedRateClock` fallback when `allow_slow_fallback=true`.
Set `allow_slow_fallback=false` to require a concrete residual sampler and
error if none is available.
"""
struct ChebyshevResidualAggregateClock{P<:AbstractSlabPrior,O<:AbstractModelPrior,T<:Real,S<:SummedRateClock} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior::O
    order::Int
    max_cells::Int
    residual_budget::T
    allow_slow_fallback::Bool
    diagnostics::AggregateClockDiagnostics
    fallback::S
end

"""
    FourierResidualAggregateClock(slab_provider, model_prior; ...)

Phase-6 moving-scale aggregate clock entry point for Boomerang/Fourier residual
proposal envelopes. Finite-horizon Boomerang sampling uses a local Fourier
proposal plus cell-wise analytic residual bounds. Unsupported providers or
infinite horizons route through the exact `SummedRateClock` fallback when
`allow_slow_fallback=true`. Set `allow_slow_fallback=false` to require a
concrete residual sampler and error if none is available.
"""
struct FourierResidualAggregateClock{P<:AbstractSlabPrior,O<:AbstractModelPrior,T<:Real,S<:SummedRateClock} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior::O
    order::Int
    cells::Int
    residual_budget::T
    allow_slow_fallback::Bool
    diagnostics::AggregateClockDiagnostics
    fallback::S
end

mutable struct LinearGaussianAggregateCache
    active_beta::BitVector
    stickable_beta::BitVector
    active_positions::Vector{Int}
    mean::Vector{Float64}
    cov::Matrix{Float64}
    beta_values::Vector{Float64}
    beta_velocity::Vector{Float64}
    cov_AA::Matrix{Float64}
    delta_A::Vector{Float64}
    velocity_A::Vector{Float64}
    cov_jA::Vector{Float64}
    solved_cross::Vector{Float64}
    log_weights::Vector{Float64}
end

function LinearGaussianAggregateCache(m::Integer)
    return LinearGaussianAggregateCache(
        BitVector(undef, m),
        BitVector(undef, m),
        Vector{Int}(undef, m),
        Vector{Float64}(undef, m),
        Matrix{Float64}(undef, m, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Matrix{Float64}(undef, m, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
        Vector{Float64}(undef, m),
    )
end

"""
    LinearGaussianAggregateClock(slab_provider, model_prior; rtol=1e-8, atol=1e-10)

Optimized aggregate unstick clock for fixed-covariance Gaussian slabs under
linear flows. The provider must have `FixedCovarianceCache`; state-dependent
Gaussian callbacks should use `SummedRateClock`.
"""
struct LinearGaussianAggregateClock{P<:AbstractGaussianSlabProvider,O<:AbstractModelPrior,T<:Real,S<:SummedRateClock} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior::O
    rtol::T
    atol::T
    cache::LinearGaussianAggregateCache
    fallback::S
end

mutable struct ExponentialSumAggregateCache
    active_beta::BitVector
    stickable_beta::BitVector
    logc::Vector{Float64}
    slopes::Vector{Float64}
    log_weights::Vector{Float64}
end

function ExponentialSumAggregateCache(m::Integer)
    return ExponentialSumAggregateCache(BitVector(undef, m), BitVector(undef, m),
        Vector{Float64}(undef, m), Vector{Float64}(undef, m), Vector{Float64}(undef, m))
end

"""
    ExponentialSumAggregateClock(provider, model_prior; rtol=1e-8, atol=1e-10)

Exact aggregate clock for independent zero-mean Gaussian slabs whose log
standard deviations are affine along linear-flow trajectories. The aggregate
rate has the form `sum_j c_j * exp(-r_j * t)`.
"""
struct ExponentialSumAggregateClock{P<:IndependentZeroMeanLogscaleGaussianSlab,O<:AbstractModelPrior,T<:Real,S<:SummedRateClock} <: AbstractAggregateUnstickClock
    slab_provider::P
    model_prior::O
    rtol::T
    atol::T
    cache::ExponentialSumAggregateCache
    fallback::S
end

function _check_model_prior_length(model_prior::AbstractModelPrior, slab_provider::AbstractSlabPrior)
    m = length(beta_indices(slab_provider))
    length(model_prior) == m ||
        throw(DimensionMismatch("model-prior length $(length(model_prior)) does not match beta dimension $m"))
    return nothing
end

function LinearGaussianAggregateClock(
    slab_provider::AbstractGaussianSlabProvider,
    model_prior::AbstractModelPrior;
    rtol::Real=1e-8,
    atol::Real=1e-10,
)
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    atol >= 0 || throw(ArgumentError("atol must be non-negative"))
    slab_cache_style(slab_provider) isa FixedCovarianceCache ||
        throw(ArgumentError("LinearGaussianAggregateClock requires a fixed-covariance slab provider; use SummedRateClock for state-dependent Gaussian callbacks"))
    _check_model_prior_length(model_prior, slab_provider)
    fallback = SummedRateClock(slab_provider, model_prior; rtol, atol)
    return LinearGaussianAggregateClock(slab_provider, model_prior, Float64(rtol), Float64(atol), LinearGaussianAggregateCache(length(beta_indices(slab_provider))), fallback)
end

function ExponentialSumAggregateClock(
    slab_provider::IndependentZeroMeanLogscaleGaussianSlab,
    model_prior::AbstractModelPrior;
    rtol::Real=1e-8,
    atol::Real=1e-10,
)
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    atol >= 0 || throw(ArgumentError("atol must be non-negative"))
    _check_model_prior_length(model_prior, slab_provider)
    fallback = SummedRateClock(slab_provider, model_prior; rtol, atol)
    return ExponentialSumAggregateClock(slab_provider, model_prior, Float64(rtol), Float64(atol), ExponentialSumAggregateCache(length(beta_indices(slab_provider))), fallback)
end

Base.copy(clock::LinearGaussianAggregateClock) = LinearGaussianAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior);
    rtol=clock.rtol,
    atol=clock.atol,
)

Base.copy(clock::ExponentialSumAggregateClock) = ExponentialSumAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior);
    rtol=clock.rtol,
    atol=clock.atol,
)

"""
    default_aggregate_unstick_clock(provider, model_prior[, flow])

Choose an aggregate unstick clock. Pass `flow` when it is available so the
default can select a compatible residual sampler; in particular,
Boomerang-family flows use a Fourier residual clock for global-logscale slabs.
"""
default_aggregate_unstick_clock(provider::IndependentZeroMeanLogscaleGaussianSlab, model_prior::AbstractModelPrior) =
    ExponentialSumAggregateClock(provider, model_prior)
default_aggregate_unstick_clock(provider::GlobalLogscaleExchangeableGaussianSlab, model_prior::AbstractModelPrior) =
    ChebyshevResidualAggregateClock(provider, model_prior; allow_slow_fallback=false)
default_aggregate_unstick_clock(provider::AbstractGaussianSlabProvider, model_prior::AbstractModelPrior) =
    slab_cache_style(provider) isa FixedCovarianceCache ? LinearGaussianAggregateClock(provider, model_prior) : SummedRateClock(provider, model_prior)
default_aggregate_unstick_clock(provider::AbstractSlabPrior, model_prior::AbstractModelPrior) =
    SummedRateClock(provider, model_prior)

default_aggregate_unstick_clock(provider::AbstractSlabPrior, model_prior::AbstractModelPrior, ::ContinuousDynamics) =
    default_aggregate_unstick_clock(provider, model_prior)

function SummedRateClock(
    slab_provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior;
    rtol::Real=1e-8,
    atol::Real=1e-10,
    initial_bracket::Real=1.0,
    bracket_multiplier::Real=2.0,
)
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    atol >= 0 || throw(ArgumentError("atol must be non-negative"))
    initial_bracket > 0 || throw(ArgumentError("initial_bracket must be positive"))
    bracket_multiplier > 1 || throw(ArgumentError("bracket_multiplier must be greater than 1"))
    _check_model_prior_length(model_prior, slab_provider)
    return SummedRateClock(slab_provider, model_prior, Float64(rtol), Float64(atol), Float64(initial_bracket), Float64(bracket_multiplier), SummedRateClockCache(length(beta_indices(slab_provider))))
end

Base.copy(clock::SummedRateClock) = SummedRateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior);
    rtol=clock.rtol,
    atol=clock.atol,
    initial_bracket=clock.initial_bracket,
    bracket_multiplier=clock.bracket_multiplier,
)

function ChebyshevResidualAggregateClock(
    slab_provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior;
    order::Integer=16,
    max_cells::Integer=64,
    residual_budget::Real=1e-8,
    allow_slow_fallback::Bool=true,
    rtol::Real=1e-8,
    atol::Real=1e-10,
    initial_bracket::Real=1.0,
    bracket_multiplier::Real=2.0,
)
    order > 0 || throw(ArgumentError("order must be positive"))
    max_cells > 0 || throw(ArgumentError("max_cells must be positive"))
    residual_budget >= 0 || throw(ArgumentError("residual_budget must be non-negative"))
    fallback = SummedRateClock(slab_provider, model_prior; rtol, atol, initial_bracket, bracket_multiplier)
    return ChebyshevResidualAggregateClock(slab_provider, model_prior, Int(order), Int(max_cells), Float64(residual_budget), Bool(allow_slow_fallback), AggregateClockDiagnostics(), fallback)
end

Base.copy(clock::ChebyshevResidualAggregateClock) = ChebyshevResidualAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior);
    order=clock.order,
    max_cells=clock.max_cells,
    residual_budget=clock.residual_budget,
    allow_slow_fallback=clock.allow_slow_fallback,
    rtol=clock.fallback.rtol,
    atol=clock.fallback.atol,
    initial_bracket=clock.fallback.initial_bracket,
    bracket_multiplier=clock.fallback.bracket_multiplier,
)

function FourierResidualAggregateClock(
    slab_provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior;
    order::Integer=16,
    cells::Integer=32,
    residual_budget::Real=1e-8,
    allow_slow_fallback::Bool=true,
    rtol::Real=1e-8,
    atol::Real=1e-10,
    initial_bracket::Real=1.0,
    bracket_multiplier::Real=2.0,
)
    order > 0 || throw(ArgumentError("order must be positive"))
    cells > 0 || throw(ArgumentError("cells must be positive"))
    residual_budget >= 0 || throw(ArgumentError("residual_budget must be non-negative"))
    fallback = SummedRateClock(slab_provider, model_prior; rtol, atol, initial_bracket, bracket_multiplier)
    return FourierResidualAggregateClock(slab_provider, model_prior, Int(order), Int(cells), Float64(residual_budget), Bool(allow_slow_fallback), AggregateClockDiagnostics(), fallback)
end

Base.copy(clock::FourierResidualAggregateClock) = FourierResidualAggregateClock(
    _copy_callable(clock.slab_provider),
    _copy_callable(clock.model_prior);
    order=clock.order,
    cells=clock.cells,
    residual_budget=clock.residual_budget,
    allow_slow_fallback=clock.allow_slow_fallback,
    rtol=clock.fallback.rtol,
    atol=clock.fallback.atol,
    initial_bracket=clock.fallback.initial_bracket,
    bracket_multiplier=clock.fallback.bracket_multiplier,
)

reset_thinning_diagnostics!(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock}) =
    reset_thinning_diagnostics!(clock.diagnostics)

function thinning_diagnostics(clock::Union{ChebyshevResidualAggregateClock,FourierResidualAggregateClock})
    d = clock.diagnostics
    proposal_count = max(d.proposals, 1)
    return (
        proposals=d.proposals,
        accepted=d.accepted,
        rejected=d.rejected,
        fallbacks=d.fallbacks,
        residual_area=d.residual_area,
        envelope_hazard=d.envelope_hazard,
        empirical_acceptance=d.accepted / proposal_count,
        hazard_acceptance=missing,
        max_envelope_ratio=d.max_envelope_ratio,
        max_residual=d.max_residual,
        min_envelope=d.min_envelope,
        rate_evaluations=d.rate_evaluations,
        last_cells=d.last_cells,
    )
end

function _active_beta_from_free(provider::AbstractSlabPrior, free::BitVector)
    active_beta = BitVector(undef, length(beta_indices(provider)))
    return _active_beta_from_free!(active_beta, provider, free)
end

function _active_beta_from_free!(active_beta::BitVector, provider::AbstractSlabPrior, free::BitVector)
    indices = beta_indices(provider)
    @inbounds for k in eachindex(indices)
        active_beta[k] = free[indices[k]]
    end
    return active_beta
end

function _stickable_beta_from_can_stick(provider::AbstractSlabPrior, can_stick::BitVector)
    stickable_beta = BitVector(undef, length(beta_indices(provider)))
    return _stickable_beta_from_can_stick!(stickable_beta, provider, can_stick)
end

function _stickable_beta_from_can_stick!(stickable_beta::BitVector, provider::AbstractSlabPrior, can_stick::BitVector)
    indices = beta_indices(provider)
    @inbounds for k in eachindex(indices)
        stickable_beta[k] = can_stick[indices[k]]
    end
    return stickable_beta
end

"""
    boundary_logweights!(out, provider, model_prior, x, active_beta, stickable_beta)

Write log unnormalized boundary weights for inactive stickable beta coordinates.
Entries that are active or not stickable are set to `-Inf`.
"""
function boundary_logweights!(
    out::AbstractVector,
    provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    m = length(beta_indices(provider))
    length(out) == m || throw(DimensionMismatch("out length $(length(out)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))
    fill!(out, -Inf)
    @inbounds for j in 1:m
        if stickable_beta[j] && !active_beta[j]
            out[j] = log_model_add_odds(model_prior, active_beta, j) +
                     log_boundary_density_zero(provider, x, active_beta, j)
        end
    end
    return out
end

_exchangeable_mean(provider::ExchangeableGaussianSlab) = provider.mean
_exchangeable_mean(::ZeroMeanExchangeableGaussianSlab) = 0.0

function _exchangeable_conditional_params(provider::AbstractExchangeableGaussianSlab, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    m = length(indices)
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    k = count(active_beta)
    μ = _exchangeable_mean(provider)
    u = provider.u
    v = provider.v
    denom = u + k * v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    sum_centered = 0.0
    @inbounds for j in 1:m
        active_beta[j] && (sum_centered += x[indices[j]] - μ)
    end
    c = v / denom
    cond_mean = μ + c * sum_centered
    cond_var = u * (u + (k + 1) * v) / denom
    cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
    return cond_mean, cond_var
end

function _exchangeable_conditional_params(provider::GlobalLogscaleExchangeableGaussianSlab, x::AbstractVector, active_beta::BitVector)
    indices = beta_indices(provider)
    m = length(indices)
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    k = count(active_beta)
    μ = provider.mean
    denom = provider.u + k * provider.v
    denom > 0 || throw(ArgumentError("u + k*v must be positive, got $denom"))
    sum_centered = 0.0
    @inbounds for j in 1:m
        active_beta[j] && (sum_centered += x[indices[j]] - μ)
    end
    c = provider.v / denom
    cond_mean = μ + c * sum_centered
    base_var = provider.u * (provider.u + (k + 1) * provider.v) / denom
    scale2 = exp(2 * (provider.logscale_offset + x[provider.logscale_index]))
    cond_var = scale2 * base_var
    cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
    return cond_mean, cond_var
end

function _exchangeable_log_q_zero(provider::AbstractExchangeableGaussianSlab, x::AbstractVector, active_beta::BitVector)
    cond_mean, cond_var = _exchangeable_conditional_params(provider, x, active_beta)
    return -0.5 * (log2π + log(cond_var) + cond_mean^2 / cond_var)
end

function boundary_logweights!(
    out::AbstractVector,
    provider::AbstractExchangeableGaussianSlab,
    model_prior::AbstractModelPrior,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    m = length(beta_indices(provider))
    length(out) == m || throw(DimensionMismatch("out length $(length(out)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))
    fill!(out, -Inf)
    log_q = _exchangeable_log_q_zero(provider, x, active_beta)
    @inbounds for j in 1:m
        if stickable_beta[j] && !active_beta[j]
            out[j] = log_model_add_odds(model_prior, active_beta, j) + log_q
        end
    end
    return out
end

"""
    aggregate_lograte(provider, model_prior, log_Cv, x, active_beta, stickable_beta)

Return the log aggregate unstick rate, equal to `log_Cv` plus the log-sum of
inactive stickable boundary weights.
"""
function aggregate_lograte(
    provider::AbstractExchangeableGaussianSlab,
    model_prior::ExchangeableModelSizePrior,
    log_Cv::Real,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    m = length(beta_indices(provider))
    length(model_prior) == m ||
        throw(DimensionMismatch("model-prior length $(length(model_prior)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))
    n_inactive_stickable = count(j -> stickable_beta[j] && !active_beta[j], 1:m)
    iszero(n_inactive_stickable) && return -Inf
    k = count(active_beta)
    k < length(model_prior) || return -Inf
    log_q = _exchangeable_log_q_zero(provider, x, active_beta)
    log_rho = model_prior.log_omega[k + 2] - model_prior.log_omega[k + 1] + log(k + 1) - log(m - k)
    return Float64(log_Cv) + log_q + log(n_inactive_stickable) + log_rho
end

function boundary_logweights!(
    out::AbstractVector,
    provider::AbstractGaussianSlabProvider,
    model_prior::AbstractModelPrior,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    indices = beta_indices(provider)
    m = length(indices)
    length(out) == m || throw(DimensionMismatch("out length $(length(out)) does not match beta dimension $m"))
    length(active_beta) == m || throw(DimensionMismatch("active_beta length $(length(active_beta)) does not match beta dimension $m"))
    length(stickable_beta) == m || throw(DimensionMismatch("stickable_beta length $(length(stickable_beta)) does not match beta dimension $m"))

    fill!(out, -Inf)
    mean, cov = _current_mean_cov(provider, x)
    beta_values = x[indices]
    A = _active_positions(active_beta, m)

    if isempty(A)
        @inbounds for j in 1:m
            if stickable_beta[j] && !active_beta[j]
                var = cov[j, j]
                var > 0 || throw(ArgumentError("conditional variance must be positive, got $var"))
                log_q = -0.5 * (log2π + log(var) + mean[j]^2 / var)
                out[j] = log_model_add_odds(model_prior, active_beta, j) + log_q
            end
        end
        return out
    end

    cov_AA = Matrix{Float64}(cov[A, A])
    F = cholesky(Symmetric(cov_AA); check=true)
    delta_A = Vector{Float64}(beta_values[A] .- mean[A])
    solved_delta = F \ delta_A

    @inbounds for j in 1:m
        if stickable_beta[j] && !active_beta[j]
            cov_jA = Vector{Float64}(cov[j, A])
            solved_cross = F \ cov_jA
            cond_mean = mean[j] + dot(cov_jA, solved_delta)
            cond_var = cov[j, j] - dot(cov_jA, solved_cross)
            cond_var > 0 || throw(ArgumentError("conditional variance must be positive, got $cond_var"))
            log_q = -0.5 * (log2π + log(cond_var) + cond_mean^2 / cond_var)
            out[j] = log_model_add_odds(model_prior, active_beta, j) + log_q
        end
    end
    return out
end

function aggregate_lograte(
    provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior,
    log_Cv::Real,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    weights = Vector{Float64}(undef, length(beta_indices(provider)))
    return aggregate_lograte!(weights, provider, model_prior, log_Cv, x, active_beta, stickable_beta)
end

function aggregate_lograte!(
    weights::AbstractVector{Float64},
    provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior,
    log_Cv::Real,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    boundary_logweights!(weights, provider, model_prior, x, active_beta, stickable_beta)
    lse = LogExpFunctions.logsumexp(weights)
    lse == -Inf && return -Inf
    return Float64(log_Cv) + lse
end

function _sample_from_logweights(rng::Random.AbstractRNG, indices::AbstractVector{Int}, logweights::AbstractVector{Float64})
    maxv = maximum(logweights)
    maxv == -Inf && throw(ArgumentError("cannot sample an unstick label because all inactive stickable rates are zero"))
    if maxv == Inf
        count = 0
        choice = 0
        @inbounds for j in eachindex(indices)
            if logweights[j] == Inf
                count += 1
                rand(rng, 1:count) == 1 && (choice = indices[j])
            end
        end
        return choice
    end
    total = 0.0
    @inbounds for j in eachindex(indices)
        total += isfinite(logweights[j]) ? exp(logweights[j] - maxv) : 0.0
    end
    draw = rand(rng) * total
    last_candidate = 0
    @inbounds for j in eachindex(indices)
        if isfinite(logweights[j])
            last_candidate = j
            draw -= exp(logweights[j] - maxv)
            draw <= 0 && return indices[j]
        end
    end
    return indices[last_candidate]
end

"""
    sample_unstick_label(rng, provider, model_prior, x, active_beta, stickable_beta)

Sample a full-state coordinate label from the inactive stickable beta
coordinates, with probabilities proportional to the boundary weights.
"""
function sample_unstick_label(
    rng::Random.AbstractRNG,
    provider::AbstractSlabPrior,
    model_prior::AbstractModelPrior,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    weights = Vector{Float64}(undef, length(beta_indices(provider)))
    boundary_logweights!(weights, provider, model_prior, x, active_beta, stickable_beta)
    return _sample_from_logweights(rng, beta_indices(provider), weights)
end

function sample_unstick_label(
    rng::Random.AbstractRNG,
    provider::AbstractExchangeableGaussianSlab,
    model_prior::ExchangeableModelSizePrior,
    x::AbstractVector,
    active_beta::BitVector,
    stickable_beta::BitVector,
)
    indices = beta_indices(provider)
    candidates = Int[]
    for j in eachindex(indices)
        stickable_beta[j] && !active_beta[j] && push!(candidates, indices[j])
    end
    isempty(candidates) && throw(ArgumentError("cannot sample an unstick label because there are no inactive stickable coordinates"))
    return rand(rng, candidates)
end
