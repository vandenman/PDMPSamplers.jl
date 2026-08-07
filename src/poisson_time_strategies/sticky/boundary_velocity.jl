function unstick_rate_constant end
function draw_stratum_velocity! end
function propose_boundary_velocity! end

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

function _invalidate_active_stratum_cache!(state::AbstractPDMPState)
    state.boundary_scratch.active_factor_valid = false
    return state
end

function _invalidate_boundary_velocity_cache!(state::AbstractPDMPState)
    scratch = state.boundary_scratch
    scratch.active_factor_valid = false
    scratch.factor_source = nothing
    scratch.factor_generation = zero(UInt)
    return state
end

function _copy_compatible_boundary_cache!(dest::BoundaryVelocityScratch,
        src::BoundaryVelocityScratch)
    d = length(src.cached_free)
    _ensure_product_boundary_scratch!(dest, d)
    copyto!(view(dest.canonical_signs, 1:d), view(src.canonical_signs, 1:d))
    copyto!(view(dest.proposal_signs, 1:d), view(src.proposal_signs, 1:d))
    src.active_factor_valid || return (dest.active_factor_valid = false; dest)

    compatible = dest.active_factor_valid &&
        dest.factor_source === src.factor_source &&
        dest.factor_generation == src.factor_generation &&
        dest.active_count == src.active_count &&
        view(dest.cached_free, 1:d) == view(src.cached_free, 1:d)
    if !compatible
        _ensure_boundary_scratch!(dest, d)
        k = src.active_count
        copyto!(view(dest.active, 1:k), view(src.active, 1:k))
        copyto!(view(dest.cached_free, 1:d), view(src.cached_free, 1:d))
        copyto!(view(dest.ΣAA, 1:k, 1:k), view(src.ΣAA, 1:k, 1:k))
        dest.active_count = k
        dest.factor_source = src.factor_source
        dest.factor_generation = src.factor_generation
        dest.active_factor_valid = true
    end
    return dest
end

_boundary_generation(::ContinuousDynamics) = zero(UInt)
_boundary_generation(flow::PreconditionedDynamics{DensePreconditioner}) =
    flow.metric.generation

_coordinate_active(::PDMPState, ::Integer, ::Integer) = true
_coordinate_active(state::StickyPDMPState, j::Integer, activating::Integer) =
    state.free[j] || j == activating

function _fill_active_stratum!(scratch::BoundaryVelocityScratch,
        state::AbstractPDMPState, activating::Integer)
    d = length(state.ξ)
    k = 0
    changed = scratch.active_count > d
    @inbounds for j in 1:d
        active = _coordinate_active(state, j, activating)
        changed |= scratch.cached_free[j] != active
        scratch.cached_free[j] = active
        if active
            k += 1
            scratch.active[k] = j
        end
    end
    changed |= scratch.active_count != k
    scratch.active_count = k
    return changed
end

function _prepare_dense_zigzag_stratum!(flow::DensePreconditionedZigZag,
        state::AbstractPDMPState, activating::Integer=0)
    d = length(state.ξ)
    scratch = _ensure_boundary_scratch!(state.boundary_scratch, d)
    generation = _boundary_generation(flow)
    active_changed = _fill_active_stratum!(scratch, state, activating)
    factor_changed = scratch.factor_source !== flow ||
        scratch.factor_generation != generation
    if factor_changed || active_changed || !scratch.active_factor_valid
        k = scratch.active_count
        if ispositive(k)
            @inbounds for b in 1:k, a in 1:k
                ia = scratch.active[a]
                ib = scratch.active[b]
                scratch.ΣAA[a, b] = dot(view(flow.metric.L, ia, :),
                                               view(flow.metric.L, ib, :))
            end
            cholesky!(Symmetric(view(scratch.ΣAA, 1:k, 1:k), :L); check=true)
        end
        scratch.factor_source = flow
        scratch.factor_generation = generation
        scratch.active_factor_valid = true
    end
    return scratch
end

function _prepare_product_stratum!(state::AbstractPDMPState, activating::Integer=0)
    scratch = _ensure_product_boundary_scratch!(
        state.boundary_scratch, length(state.ξ))
    _fill_active_stratum!(scratch, state, activating)
    return scratch
end

_prepare_stratum!(flow::DensePreconditionedZigZag, state::AbstractPDMPState,
    activating::Integer=0) = _prepare_dense_zigzag_stratum!(flow, state, activating)
_prepare_stratum!(::ZigZag, state::AbstractPDMPState, activating::Integer=0) =
    _prepare_product_stratum!(state, activating)
_prepare_stratum!(::PreconditionedDynamics{<:Union{IdentityPreconditioner,
    DiagonalPreconditioner},<:ZigZag}, state::AbstractPDMPState,
    activating::Integer=0) = _prepare_product_stratum!(state, activating)

_prepare_stratum!(::Union{BouncyParticle,AnyBoomerang,
    PreconditionedDynamics{<:AbstractPreconditioner,
        <:Union{BouncyParticle,AnyBoomerang}}}, state::AbstractPDMPState,
    activating::Integer=0) = _prepare_product_stratum!(state, activating)

function _rand_abs_tilted_normal(rng::Random.AbstractRNG, σ::Real)
    ispositive(σ) || throw(ArgumentError("σ must be positive"))
    magnitude = Float64(σ) * sqrt(rand(rng, Exponential(2.0)))
    return rand(rng, Bool) ? magnitude : -magnitude
end

"""
    unstick_rate_constant(flow, i)

Return `E(abs(Vᵢ))` under the full-stratum invariant velocity law. This
state-free helper is only defined when that quantity is also the exact sticky
proposal-clock constant. State-dependent clocks use
`_boundary_proposal_clock_constant(flow, state, i)`.
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
    L = metric.L
    @inbounds for a in axes(L, 2), b in axes(L, 2)
        total += L[i, a] * _boomerang_covariance_entry(inner, a, b) * L[j, b]
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

unstick_rate_constant(flow::PreconditionedDynamics{<:IdentityPreconditioner,<:ZigZag}, i::Integer) =
    unstick_rate_constant(flow.dynamics, i)
unstick_rate_constant(flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}, i::Integer) =
    abs(flow.metric.scale[i])
unstick_rate_constant(flow::PreconditionedDynamics{<:AbstractPreconditioner,<:Union{BouncyParticle,AnyBoomerang}}, i::Integer) =
    sqrt(2 / π) * sqrt(_preconditioned_gaussian_covariance_entry(flow, i, i))

function _install_active_velocity!(state::AbstractPDMPState,
        scratch::BoundaryVelocityScratch)
    fill!(state.ξ.θ, 0.0)
    @inbounds for a in 1:scratch.active_count
        state.ξ.θ[scratch.active[a]] = scratch.θA[a]
    end
    return state
end

function _draw_product_zigzag!(rng::Random.AbstractRNG, state::AbstractPDMPState,
        flow, scale)
    scratch = _prepare_stratum!(flow, state)
    fill!(state.ξ.θ, 0.0)
    @inbounds for a in 1:scratch.active_count
        i = scratch.active[a]
        state.ξ.θ[i] = rand(rng, Bool) ? scale(i) : -scale(i)
    end
    return state
end

draw_stratum_velocity!(rng::Random.AbstractRNG, state::AbstractPDMPState,
    flow::ZigZag) = _draw_product_zigzag!(rng, state, flow, _ -> 1.0)
draw_stratum_velocity!(rng::Random.AbstractRNG, state::AbstractPDMPState,
    flow::PreconditionedDynamics{IdentityPreconditioner,<:ZigZag}) =
    _draw_product_zigzag!(rng, state, flow, _ -> 1.0)
draw_stratum_velocity!(rng::Random.AbstractRNG, state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}) =
    _draw_product_zigzag!(rng, state, flow, i -> abs(Float64(flow.metric.scale[i])))

function _draw_dense_zigzag!(rng::Random.AbstractRNG, state::AbstractPDMPState,
        flow::DensePreconditionedZigZag, activating::Integer=0;
        proposal::Bool=false)
    scratch = _prepare_stratum!(flow, state, activating)
    k = scratch.active_count
    signs = proposal ? scratch.proposal_signs : scratch.canonical_signs
    @inbounds for a in 1:k
        signs[a] = rand(rng, Bool) ? 1.0 : -1.0
    end
    mul!(view(scratch.θA, 1:k),
         LowerTriangular(view(scratch.ΣAA, 1:k, 1:k)), view(signs, 1:k))
    proposal || _install_active_velocity!(state, scratch)
    return scratch
end

draw_stratum_velocity!(rng::Random.AbstractRNG, state::AbstractPDMPState,
    flow::DensePreconditionedZigZag) = (_draw_dense_zigzag!(rng, state, flow); state)

_sample_full_gaussian!(rng, velocity, scratch, ::BouncyParticle) =
    randn!(rng, velocity)

function _sample_full_gaussian!(rng, velocity, scratch,
        flow::LowRankMutableBoomerang)
    lowrank_sample!(rng, velocity, flow.Γ)
end

function _sample_full_gaussian!(rng, velocity, scratch, flow::AnyBoomerang)
    randn!(rng, scratch.solved_θ)
    mul!(velocity, flow.ΣL, scratch.solved_θ)
    return velocity
end

function _sample_full_gaussian!(rng, velocity, scratch,
        flow::PreconditionedDynamics{IdentityPreconditioner,
            <:Union{BouncyParticle,AnyBoomerang}})
    return _sample_full_gaussian!(rng, velocity, scratch, flow.dynamics)
end

function _sample_full_gaussian!(rng, velocity, scratch,
        flow::PreconditionedDynamics{<:DiagonalPreconditioner,
            <:Union{BouncyParticle,AnyBoomerang}})
    _sample_full_gaussian!(rng, velocity, scratch, flow.dynamics)
    velocity .*= flow.metric.scale
    return velocity
end

function _sample_full_gaussian!(rng, velocity, scratch,
        flow::PreconditionedDynamics{DensePreconditioner,
            <:Union{BouncyParticle,AnyBoomerang}})
    _sample_full_gaussian!(rng, scratch.θA, scratch, flow.dynamics)
    mul!(velocity, flow.metric.L, scratch.θA)
    return velocity
end

function _draw_gaussian_stratum!(rng::Random.AbstractRNG,
        state::AbstractPDMPState, flow::ContinuousDynamics, activating::Integer=0)
    scratch = _prepare_stratum!(flow, state, activating)
    _sample_full_gaussian!(rng, state.ξ.θ, scratch, flow)
    @inbounds for i in eachindex(state.ξ.θ)
        _coordinate_active(state, i, activating) || (state.ξ.θ[i] = 0.0)
    end
    return scratch
end

draw_stratum_velocity!(rng::Random.AbstractRNG, state::AbstractPDMPState,
    flow::Union{BouncyParticle,AnyBoomerang}) =
    (_draw_gaussian_stratum!(rng, state, flow); state)
draw_stratum_velocity!(rng::Random.AbstractRNG, state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:AbstractPreconditioner,
        <:Union{BouncyParticle,AnyBoomerang}}) =
    (_draw_gaussian_stratum!(rng, state, flow); state)

_gaussian_covariance_entry(::BouncyParticle, i::Integer, j::Integer) =
    i == j ? 1.0 : 0.0
_gaussian_covariance_entry(flow::AnyBoomerang, i::Integer, j::Integer) =
    _boomerang_covariance_entry(flow, i, j)
_gaussian_covariance_entry(flow::PreconditionedDynamics{<:AbstractPreconditioner,
    <:Union{BouncyParticle,AnyBoomerang}}, i::Integer, j::Integer) =
    _preconditioned_gaussian_covariance_entry(flow, i, j)

_boundary_proposal_clock_constant(flow::Union{ZigZag,BouncyParticle,AnyBoomerang,
        PreconditionedDynamics{<:Union{IdentityPreconditioner,DiagonalPreconditioner},<:ZigZag},
        PreconditionedDynamics{<:AbstractPreconditioner,<:Union{BouncyParticle,AnyBoomerang}}},
    ::StickyPDMPState, i::Integer) = unstick_rate_constant(flow, i)

function _dense_boundary_row_bound(flow::DensePreconditionedZigZag,
        state::StickyPDMPState, i::Integer)
    scratch = _prepare_stratum!(flow, state, i)
    p = findfirst(==(i), view(scratch.active, 1:scratch.active_count))
    p === nothing && error("activated coordinate is absent from dense stratum")
    return sum(abs, view(scratch.ΣAA, p, 1:p)), p
end
_boundary_proposal_clock_constant(flow::DensePreconditionedZigZag,
    state::StickyPDMPState, i::Integer) = first(_dense_boundary_row_bound(flow, state, i))

function _propose_product_boundary!(rng::Random.AbstractRNG,
        state::StickyPDMPState, flow, i::Integer, scale)
    scratch = _prepare_product_stratum!(state, i)
    fill!(state.ξ.θ, 0.0)
    @inbounds for a in 1:scratch.active_count
        j = scratch.active[a]
        state.ξ.θ[j] = rand(rng, Bool) ? scale(j) : -scale(j)
    end
    return true
end

propose_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState,
    flow::ZigZag, i::Integer) =
    _propose_product_boundary!(rng, state, flow, i, _ -> 1.0)
propose_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState,
    flow::PreconditionedDynamics{IdentityPreconditioner,<:ZigZag}, i::Integer) =
    _propose_product_boundary!(rng, state, flow, i, _ -> 1.0)
propose_boundary_velocity!(rng::Random.AbstractRNG, state::StickyPDMPState,
    flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}, i::Integer) =
    _propose_product_boundary!(rng, state, flow, i,
        j -> abs(Float64(flow.metric.scale[j])))

function propose_boundary_velocity!(rng::Random.AbstractRNG,
        state::StickyPDMPState, flow::DensePreconditionedZigZag, i::Integer)
    scratch = _draw_dense_zigzag!(rng, state, flow, i; proposal=true)
    A, p = _dense_boundary_row_bound(flow, state, i)
    speed = abs(dot(view(scratch.ΣAA, p, 1:p),
                    view(scratch.proposal_signs, 1:p)))
    if rand(rng) * A > speed
        return false
    end
    copyto!(view(scratch.canonical_signs, 1:scratch.active_count),
            view(scratch.proposal_signs, 1:scratch.active_count))
    _install_active_velocity!(state, scratch)
    return true
end

function _gaussian_covariance_mul!(out, ::BouncyParticle, x, scratch)
    copyto!(out, x)
    return out
end

function _gaussian_covariance_mul!(out, flow::LowRankMutableBoomerang, x,
        scratch)
    lrp = flow.Γ
    coeff = scratch.proposal_signs
    r = length(lrp.Λ)
    mul!(view(coeff, 1:r), transpose(lrp.V), x)
    @inbounds for k in 1:r
        coeff[k] *= lrp.Λ[k]
    end
    @inbounds for j in eachindex(out)
        value = lrp.D[j] * x[j]
        for k in 1:r
            value += lrp.V[j, k] * coeff[k]
        end
        out[j] = value
    end
    return out
end

function _gaussian_covariance_mul!(out, flow::AnyBoomerang, x, scratch)
    mul!(scratch.solved_θ, transpose(flow.ΣL), x)
    mul!(out, flow.ΣL, scratch.solved_θ)
    return out
end

function _gaussian_covariance_mul!(out,
        flow::PreconditionedDynamics{IdentityPreconditioner,
            <:Union{BouncyParticle,AnyBoomerang}}, x, scratch)
    return _gaussian_covariance_mul!(out, flow.dynamics, x, scratch)
end

function _gaussian_covariance_mul!(out,
        flow::PreconditionedDynamics{<:DiagonalPreconditioner,
            <:Union{BouncyParticle,AnyBoomerang}}, x, scratch)
    scale = flow.metric.scale
    @inbounds for j in eachindex(x)
        scratch.θA[j] = scale[j] * x[j]
    end
    _gaussian_covariance_mul!(out, flow.dynamics, scratch.θA, scratch)
    @inbounds for j in eachindex(out)
        out[j] *= scale[j]
    end
    return out
end

function _gaussian_covariance_mul!(out,
        flow::PreconditionedDynamics{DensePreconditioner,
            <:Union{BouncyParticle,AnyBoomerang}}, x, scratch)
    mul!(scratch.θA, transpose(flow.metric.L), x)
    _gaussian_covariance_mul!(scratch.solved_cross, flow.dynamics,
                              scratch.θA, scratch)
    mul!(scratch.proposal_signs, flow.metric.L, scratch.solved_cross)
    copyto!(out, scratch.proposal_signs)
    return out
end

function _gaussian_covariance_column!(out, flow, i::Integer, scratch)
    fill!(scratch.canonical_signs, 0.0)
    scratch.canonical_signs[i] = 1.0
    _gaussian_covariance_mul!(out, flow, scratch.canonical_signs, scratch)
    return out
end

function propose_boundary_velocity!(rng::Random.AbstractRNG,
        state::StickyPDMPState,
        flow::Union{BouncyParticle,AnyBoomerang,
            PreconditionedDynamics{<:AbstractPreconditioner,
                <:Union{BouncyParticle,AnyBoomerang}}}, i::Integer)
    scratch = _prepare_stratum!(flow, state, i)
    k = scratch.active_count
    p = findfirst(==(i), view(scratch.active, 1:k))
    p === nothing && error("activated coordinate is absent from Gaussian stratum")

    # W is an ordinary draw from ψ_F. Replacing W_i by its |v_i|-tilted
    # marginal and applying the Gaussian conditional correction draws Q_Fi.
    _sample_full_gaussian!(rng, state.ξ.θ, scratch, flow)
    _gaussian_covariance_column!(scratch.solved_cross, flow, i, scratch)
    σ2 = scratch.solved_cross[i]
    old_i = state.ξ.θ[i]
    new_i = _rand_abs_tilted_normal(rng, sqrt(σ2))
    delta = (new_i - old_i) / σ2
    @inbounds for j in eachindex(state.ξ.θ)
        if _coordinate_active(state, j, i)
            state.ξ.θ[j] += scratch.solved_cross[j] * delta
        else
            state.ξ.θ[j] = 0.0
        end
    end
    return true
end

function _dense_zigzag_stratum!(state::StickyPDMPState,
        flow::DensePreconditionedZigZag)
    scratch = _prepare_stratum!(flow, state)
    k = scratch.active_count
    if ispositive(k)
        @inbounds for a in 1:k
            scratch.solved_θ[a] = state.ξ.θ[scratch.active[a]]
        end
        ldiv!(LowerTriangular(view(scratch.ΣAA, 1:k, 1:k)),
              view(scratch.solved_θ, 1:k))
        @inbounds for a in 1:k
            scratch.canonical_signs[a] =
                ifelse(scratch.solved_θ[a] < 0, -1.0, 1.0)
        end
    end
    return scratch
end

function reflect!(::Random.AbstractRNG, state::StickyPDMPState,
        gradient::AbstractVector,
        flow::PreconditionedDynamics{<:AbstractPreconditioner,
            <:Union{BouncyParticle,AnyBoomerang}}, cache)
    scratch = _prepare_stratum!(flow, state)
    k = scratch.active_count
    z = scratch.solved_cross
    fill!(scratch.canonical_signs, 0.0)
    @inbounds for a in 1:k
        ia = scratch.active[a]
        scratch.canonical_signs[ia] = gradient[ia]
    end
    _gaussian_covariance_mul!(z, flow, scratch.canonical_signs, scratch)
    numerator = zero(eltype(state.ξ.θ))
    denominator = zero(eltype(state.ξ.θ))
    @inbounds for a in 1:k
        ia = scratch.active[a]
        numerator += state.ξ.θ[ia] * gradient[ia]
        denominator += gradient[ia] * z[ia]
    end
    iszero(denominator) && return nothing
    coefficient = 2 * numerator / denominator
    @inbounds for a in 1:k
        ia = scratch.active[a]
        state.ξ.θ[ia] -= coefficient * z[ia]
    end
    return nothing
end
