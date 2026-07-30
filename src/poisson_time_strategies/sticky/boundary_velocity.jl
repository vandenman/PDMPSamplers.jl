function unstick_rate_constant end
function draw_boundary_velocity! end

function _boomerang_covariance_entry(flow::AnyBoomerang, i::Integer, j::Integer)
    ΣL = flow.ΣL
    return dot(view(ΣL, i, :), view(ΣL, j, :))
end

function _boomerang_covariance_entry(flow::LowRankMutableBoomerang, i::Integer, j::Integer)
    lrp = flow.Γ
    value = i == j ? lrp.D[i] : zero(eltype(lrp.D))
    @inbounds for k in eachindex(lrp.Λ)
        value += lrp.V[i, k] * lrp.Λ[k] * lrp.V[j, k]
    end
    return value
end

function _materialize_boomerang_covariance!(dest::AbstractMatrix{Float64}, flow::AnyBoomerang)
    mul!(dest, flow.ΣL, transpose(flow.ΣL))
    return dest
end

function _materialize_boomerang_covariance!(dest::AbstractMatrix{Float64}, flow::LowRankMutableBoomerang)
    lrp = flow.Γ
    @inbounds for j in axes(dest, 2), i in axes(dest, 1)
        value = i == j ? lrp.D[i] : 0.0
        for k in eachindex(lrp.Λ)
            value += lrp.V[i, k] * lrp.Λ[k] * lrp.V[j, k]
        end
        dest[i, j] = value
    end
    return dest
end

function _materialize_metric_covariance!(dest::AbstractMatrix{Float64}, ::IdentityPreconditioner)
    fill!(dest, 0.0)
    @inbounds for i in axes(dest, 1)
        dest[i, i] = 1.0
    end
    return dest
end

function _materialize_metric_covariance!(dest::AbstractMatrix{Float64}, metric::DiagonalPreconditioner)
    fill!(dest, 0.0)
    @inbounds for i in axes(dest, 1)
        dest[i, i] = abs2(metric.scale[i])
    end
    return dest
end

function _materialize_metric_covariance!(dest::AbstractMatrix{Float64}, metric::DensePreconditioner)
    mul!(dest, metric.L, transpose(metric.L))
    return dest
end

function _materialize_boundary_covariance!(scratch::BoundaryVelocityScratch, flow::AnyBoomerang)
    return _materialize_boomerang_covariance!(scratch.covariance, flow)
end

function _materialize_boundary_covariance!(
    scratch::BoundaryVelocityScratch,
    flow::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle},
)
    return _materialize_metric_covariance!(scratch.covariance, flow.metric)
end

function _materialize_boundary_covariance!(
    scratch::BoundaryVelocityScratch,
    flow::PreconditionedDynamics{IdentityPreconditioner,<:AnyBoomerang},
)
    return _materialize_boomerang_covariance!(scratch.covariance, flow.dynamics)
end

function _materialize_boundary_covariance!(
    scratch::BoundaryVelocityScratch,
    flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:AnyBoomerang},
)
    covariance = _materialize_boomerang_covariance!(scratch.covariance, flow.dynamics)
    scale = flow.metric.scale
    @inbounds for j in axes(covariance, 2), i in axes(covariance, 1)
        covariance[i, j] *= scale[i] * scale[j]
    end
    return covariance
end

function _materialize_boundary_covariance!(
    scratch::BoundaryVelocityScratch,
    flow::PreconditionedDynamics{DensePreconditioner,<:AnyBoomerang},
)
    d = length(scratch.active)
    inner_cov = scratch.covariance_work
    if inner_cov === nothing || size(inner_cov) != (d, d)
        inner_cov = Matrix{Float64}(undef, d, d)
        scratch.covariance_work = inner_cov
    end
    _materialize_boomerang_covariance!(inner_cov, flow.dynamics)
    mul!(scratch.ΣAA, flow.metric.L, inner_cov)
    mul!(scratch.covariance, scratch.ΣAA, transpose(flow.metric.L))
    return scratch.covariance
end

function _invalidate_boundary_velocity_cache!(state::StickyPDMPState)
    scratch = state.boundary_scratch
    scratch.covariance_valid = false
    scratch.active_factor_valid = false
    fill!(scratch.conditional_std_valid, false)
    return state
end

function _prepare_boundary_velocity_cache!(
    flow::Union{
        AnyBoomerang,
        PreconditionedDynamics{<:AbstractPreconditioner,<:Union{BouncyParticle,AnyBoomerang}},
    },
    state::StickyPDMPState,
)
    d = length(state.free)
    scratch = _ensure_boundary_scratch!(state.boundary_scratch, d)
    covariance_changed = !scratch.covariance_valid || scratch.covariance_source !== flow
    if covariance_changed
        _materialize_boundary_covariance!(scratch, flow)
        scratch.covariance_source = flow
        scratch.covariance_generation += one(UInt)
        scratch.covariance_valid = true
        scratch.active_factor_valid = false
        fill!(scratch.conditional_std_valid, false)
    end

    active_changed = !scratch.active_factor_valid || scratch.cached_free != state.free
    if active_changed
        copyto!(scratch.cached_free, state.free)
        k = 0
        @inbounds for i in eachindex(state.free)
            if state.free[i]
                k += 1
                scratch.active[k] = i
            end
        end
        scratch.active_count = k
        fill!(scratch.conditional_std_valid, false)
        if ispositive(k)
            @inbounds for b in 1:k, a in 1:k
                scratch.ΣAA[a, b] = scratch.covariance[scratch.active[a], scratch.active[b]]
            end
            cholesky!(Symmetric(view(scratch.ΣAA, 1:k, 1:k), :L); check=true)
        end
        scratch.active_factor_valid = true
    end

    k = scratch.active_count
    if ispositive(k)
        solved_θ = view(scratch.solved_θ, 1:k)
        @inbounds for a in 1:k
            solved_θ[a] = state.ξ.θ[scratch.active[a]]
        end
        _solve_cached_active!(solved_θ, scratch, k)
    end
    return scratch
end

function _solve_cached_active!(out::AbstractVector, scratch::BoundaryVelocityScratch, k::Int)
    L = LowerTriangular(view(scratch.ΣAA, 1:k, 1:k))
    ldiv!(L, out)
    ldiv!(adjoint(L), out)
    return out
end

function _conditional_boundary_velocity_params_prepared(
    state::StickyPDMPState,
    i::Integer,
    context::AbstractString,
)
    scratch = state.boundary_scratch
    covariance = scratch.covariance
    σ2 = covariance[i, i]
    ispositive(σ2) || throw(ArgumentError("$context boundary velocity variance must be positive, got $σ2"))

    # Aggregate unstick clocks and boundary draws call this while i is inactive,
    # so every inactive label shares the same cached active-block factorization.
    if state.free[i]
        scratch.active_factor_valid = false
        fill!(scratch.conditional_std_valid, false)
        return _conditional_boundary_velocity_params(
            (a, b) -> covariance[a, b], state, i, context)
    end

    k = scratch.active_count
    iszero(k) && return 0.0, sqrt(σ2)
    ΣiA = scratch.ΣiA
    solved_θ = view(scratch.solved_θ, 1:k)
    @inbounds for a in 1:k
        ia = scratch.active[a]
        ΣiA[a] = covariance[i, ia]
    end
    ΣiA_view = view(ΣiA, 1:k)
    μ = dot(ΣiA_view, solved_θ)
    if !scratch.conditional_std_valid[i]
        solved_cross = view(scratch.solved_cross, 1:k)
        copyto!(solved_cross, ΣiA_view)
        _solve_cached_active!(solved_cross, scratch, k)
        σ2_cond = σ2 - dot(ΣiA_view, solved_cross)
        ispositive(σ2_cond) ||
            throw(ArgumentError("$context conditional boundary velocity variance must be positive, got $σ2_cond"))
        scratch.conditional_std[i] = sqrt(σ2_cond)
        scratch.conditional_std_valid[i] = true
    end
    return μ, scratch.conditional_std[i]
end

_stdnormal_pdf(z::Real) = exp(-0.5 * abs2(z)) / sqrt(2π)

function _normal_first_moment_between(μ::Real, σ::Real, a::Real, b::Real)
    a < b || return 0.0
    za = (Float64(a) - Float64(μ)) / Float64(σ)
    zb = (Float64(b) - Float64(μ)) / Float64(σ)
    Φdiff = Distributions.normcdf(zb) - Distributions.normcdf(za)
    return Float64(μ) * Φdiff + Float64(σ) * (_stdnormal_pdf(za) - _stdnormal_pdf(zb))
end

function _abs_normal_moment_between(μ::Real, σ::Real, a::Real, b::Real)
    a < b || return 0.0
    b <= 0 && return -_normal_first_moment_between(μ, σ, a, b)
    a >= 0 && return _normal_first_moment_between(μ, σ, a, b)
    return -_normal_first_moment_between(μ, σ, a, 0.0) + _normal_first_moment_between(μ, σ, 0.0, b)
end

_abs_normal_mean(μ::Real, σ::Real) = _abs_normal_moment_between(μ, σ, -Inf, Inf)
_abs_tilted_normal_cdf(μ::Real, σ::Real, x::Real, normalizer::Real) = _abs_normal_moment_between(μ, σ, -Inf, x) / normalizer

function _rand_abs_tilted_normal(rng::Random.AbstractRNG, μ::Real, σ::Real)
    ispositive(σ) || throw(ArgumentError("σ must be positive"))
    if iszero(μ)
        magnitude = Float64(σ) * sqrt(rand(rng, Exponential(2.0)))
        return rand(rng, Bool) ? magnitude : -magnitude
    end

    normalizer = _abs_normal_mean(μ, σ)
    target = rand(rng)
    lo = Float64(μ) - Float64(σ)
    hi = Float64(μ) + Float64(σ)
    step = Float64(σ)
    while _abs_tilted_normal_cdf(μ, σ, lo, normalizer) > target
        step *= 2
        lo -= step
    end
    step = Float64(σ)
    while _abs_tilted_normal_cdf(μ, σ, hi, normalizer) < target
        step *= 2
        hi += step
    end
    for _ in 1:80
        mid = 0.5 * (lo + hi)
        if _abs_tilted_normal_cdf(μ, σ, mid, normalizer) < target
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function _conditional_boundary_velocity_params(cov_entry, state::StickyPDMPState, i::Integer, context::AbstractString)
    σ2 = cov_entry(i, i)
    ispositive(σ2) || throw(ArgumentError("$context boundary velocity variance must be positive, got $σ2"))
    scratch = _ensure_boundary_scratch!(state.boundary_scratch, length(state.free))
    active = scratch.active
    k = 0
    @inbounds for j in eachindex(state.free)
        if state.free[j] && j != i
            k += 1
            active[k] = j
        end
    end
    iszero(k) && return 0.0, sqrt(σ2)

    ΣAA = scratch.ΣAA
    ΣiA = scratch.ΣiA
    θA = scratch.θA
    @inbounds for a in 1:k
        ia = active[a]
        ΣiA[a] = cov_entry(i, ia)
        θA[a] = state.ξ.θ[ia]
        for b in 1:k
            ΣAA[a, b] = cov_entry(ia, active[b])
        end
    end
    F = cholesky!(Symmetric(view(ΣAA, 1:k, 1:k)); check=true)
    solved_θ = view(scratch.solved_θ, 1:k)
    solved_cross = view(scratch.solved_cross, 1:k)
    copyto!(solved_θ, view(θA, 1:k))
    copyto!(solved_cross, view(ΣiA, 1:k))
    ldiv!(F, solved_θ)
    ldiv!(F, solved_cross)
    ΣiA_view = view(ΣiA, 1:k)
    μ = dot(ΣiA_view, solved_θ)
    σ2_cond = σ2 - dot(ΣiA_view, solved_cross)
    ispositive(σ2_cond) || throw(ArgumentError("$context conditional boundary velocity variance must be positive, got $σ2_cond"))
    return μ, sqrt(σ2_cond)
end

function _boomerang_boundary_velocity_params(flow::AnyBoomerang, state::StickyPDMPState, i::Integer)
    _prepare_boundary_velocity_cache!(flow, state)
    return _conditional_boundary_velocity_params_prepared(state, i, "Boomerang")
end

"""
    unstick_rate_constant(flow, i)

Return the boundary velocity normalizing constant used by aggregate sticky
clocks for coordinate `i`. For Boomerang this is the unconditional normalizer
`E(abs(Vᵢ))` for the Gaussian reference velocity marginal.
"""
unstick_rate_constant(::ZigZag, ::Integer) = 1.0
unstick_rate_constant(::BouncyParticle, ::Integer) = sqrt(2 / π)
unstick_rate_constant(flow::AnyBoomerang, i::Integer) = sqrt(2 / π) * sqrt(_boomerang_covariance_entry(flow, i, i))

_preconditioner_covariance_entry(::IdentityPreconditioner, i::Integer, j::Integer) = i == j ? 1.0 : 0.0
_preconditioner_covariance_entry(metric::DiagonalPreconditioner, i::Integer, j::Integer) = i == j ? abs2(metric.scale[i]) : 0.0
_preconditioner_covariance_entry(metric::DensePreconditioner, i::Integer, j::Integer) =
    dot(view(metric.L, i, :), view(metric.L, j, :))

function _preconditioned_boomerang_covariance_entry(flow::PreconditionedDynamics{<:AbstractPreconditioner,<:AnyBoomerang}, i::Integer, j::Integer)
    metric = flow.metric
    inner = flow.dynamics
    total = 0.0
    @inbounds for a in axes(metric.L, 2), b in axes(metric.L, 2)
        total += metric.L[i, a] * _boomerang_covariance_entry(inner, a, b) * metric.L[j, b]
    end
    return total
end
_preconditioned_boomerang_covariance_entry(flow::PreconditionedDynamics{IdentityPreconditioner,<:AnyBoomerang}, i::Integer, j::Integer) =
    _boomerang_covariance_entry(flow.dynamics, i, j)
_preconditioned_boomerang_covariance_entry(flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:AnyBoomerang}, i::Integer, j::Integer) =
    flow.metric.scale[i] * flow.metric.scale[j] * _boomerang_covariance_entry(flow.dynamics, i, j)

function _preconditioned_gaussian_covariance_entry(flow::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle}, i::Integer, j::Integer)
    return _preconditioner_covariance_entry(flow.metric, i, j)
end
function _preconditioned_gaussian_covariance_entry(flow::PreconditionedDynamics{<:AbstractPreconditioner,<:AnyBoomerang}, i::Integer, j::Integer)
    return _preconditioned_boomerang_covariance_entry(flow, i, j)
end

function _preconditioned_gaussian_boundary_velocity_params(flow::PreconditionedDynamics, state::StickyPDMPState, i::Integer)
    _prepare_boundary_velocity_cache!(flow, state)
    return _conditional_boundary_velocity_params_prepared(state, i, "preconditioned")
end

_preconditioned_gaussian_boundary_velocity_params(
    ::PreconditionedDynamics{IdentityPreconditioner,<:BouncyParticle},
    ::StickyPDMPState,
    ::Integer,
) = (0.0, 1.0)

_preconditioned_gaussian_boundary_velocity_params(
    flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:BouncyParticle},
    ::StickyPDMPState,
    i::Integer,
) = (0.0, abs(Float64(flow.metric.scale[i])))

unstick_rate_constant(flow::PreconditionedDynamics{<:IdentityPreconditioner,<:ZigZag}, i::Integer) =
    unstick_rate_constant(flow.dynamics, i)
unstick_rate_constant(flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}, i::Integer) =
    abs(flow.metric.scale[i])
unstick_rate_constant(::DensePreconditionedZigZag, ::Integer) =
    throw(ArgumentError("AggregateSticky does not support dense-preconditioned ZigZag coordinate boundary laws"))
unstick_rate_constant(flow::PreconditionedDynamics{<:AbstractPreconditioner,<:Union{BouncyParticle,AnyBoomerang}}, i::Integer) =
    sqrt(2 / π) * sqrt(_preconditioned_gaussian_covariance_entry(flow, i, i))

_unstick_rate_constant(flow::ContinuousDynamics, ::StickyPDMPState, i::Integer) = unstick_rate_constant(flow, i)
_prepare_boundary_velocity_cache!(::ContinuousDynamics, ::StickyPDMPState) = nothing
_prepare_boundary_velocity_cache!(
    ::PreconditionedDynamics{IdentityPreconditioner,<:BouncyParticle},
    ::StickyPDMPState,
) = nothing
_prepare_boundary_velocity_cache!(
    ::PreconditionedDynamics{<:DiagonalPreconditioner,<:BouncyParticle},
    ::StickyPDMPState,
) = nothing
_unstick_rate_constant_prepared(flow::ContinuousDynamics, state::StickyPDMPState, i::Integer) =
    _unstick_rate_constant(flow, state, i)
function _unstick_rate_constant(flow::AnyBoomerang, state::StickyPDMPState, i::Integer)
    μ, σ = _boomerang_boundary_velocity_params(flow, state, i)
    return _abs_normal_mean(μ, σ)
end
function _unstick_rate_constant_prepared(flow::AnyBoomerang, state::StickyPDMPState, i::Integer)
    μ, σ = _conditional_boundary_velocity_params_prepared(state, i, "Boomerang")
    return _abs_normal_mean(μ, σ)
end
function _unstick_rate_constant(flow::PreconditionedDynamics{<:AbstractPreconditioner,<:Union{BouncyParticle,AnyBoomerang}}, state::StickyPDMPState, i::Integer)
    μ, σ = _preconditioned_gaussian_boundary_velocity_params(flow, state, i)
    return _abs_normal_mean(μ, σ)
end
function _unstick_rate_constant_prepared(
    flow::PreconditionedDynamics{<:AbstractPreconditioner,<:Union{BouncyParticle,AnyBoomerang}},
    state::StickyPDMPState,
    i::Integer,
)
    μ, σ = _conditional_boundary_velocity_params_prepared(state, i, "preconditioned")
    return _abs_normal_mean(μ, σ)
end
_unstick_rate_constant_prepared(
    ::PreconditionedDynamics{IdentityPreconditioner,<:BouncyParticle},
    ::StickyPDMPState,
    ::Integer,
) = sqrt(2 / π)
_unstick_rate_constant_prepared(
    flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:BouncyParticle},
    ::StickyPDMPState,
    i::Integer,
) = sqrt(2 / π) * abs(Float64(flow.metric.scale[i]))

function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, ::ZigZag, i::Integer)
    state.ξ.θ[i] = rand(rng, (-1.0, 1.0))
    state.old_velocity[i] = 0.0
    return state.ξ.θ[i]
end

function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, ::BouncyParticle, i::Integer)
    magnitude = sqrt(rand(rng, Exponential(2.0)))
    state.ξ.θ[i] = rand(rng, Bool) ? magnitude : -magnitude
    state.old_velocity[i] = 0.0
    return state.ξ.θ[i]
end

function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::AnyBoomerang, i::Integer)
    μ, σ = _boomerang_boundary_velocity_params(flow, state, i)
    state.ξ.θ[i] = _rand_abs_tilted_normal(rng, μ, σ)
    state.old_velocity[i] = 0.0
    return state.ξ.θ[i]
end
function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::PreconditionedDynamics{<:IdentityPreconditioner,<:ZigZag}, i::Integer)
    return draw_boundary_velocity!(rng, state, flow.dynamics, i)
end
function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}, i::Integer)
    state.ξ.θ[i] = rand(rng, (-abs(flow.metric.scale[i]), abs(flow.metric.scale[i])))
    state.old_velocity[i] = 0.0
    return state.ξ.θ[i]
end
draw_boundary_velocity!(::Random.AbstractRNG, ::StickyPDMPState, ::DensePreconditionedZigZag, ::Integer) =
    throw(ArgumentError("AggregateSticky does not support dense-preconditioned ZigZag coordinate boundary laws"))
function draw_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState, flow::PreconditionedDynamics{<:AbstractPreconditioner,<:Union{BouncyParticle,AnyBoomerang}}, i::Integer)
    μ, σ = _preconditioned_gaussian_boundary_velocity_params(flow, state, i)
    state.ξ.θ[i] = _rand_abs_tilted_normal(rng, μ, σ)
    state.old_velocity[i] = 0.0
    return state.ξ.θ[i]
end
