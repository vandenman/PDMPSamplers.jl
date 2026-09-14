function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix, provider, state::AbstractPDMPState,
    flow::ContinuousDynamics, t_grid::AbstractVector, n_points::Integer)
    throw(ArgumentError(
        "bound=:linear requires rate derivatives for $(typeof(flow)); " *
        "use bound=:constant or implement rate_derivatives_for_grid!"))
end

function _scalar_rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix, provider, state::AbstractPDMPState,
    flow::ContinuousDynamics, t_grid::AbstractVector, n_points::Integer)
    size(values, 1) >= 1 && size(values, 2) >= n_points ||
        throw(ArgumentError("values matrix is too small"))
    size(derivatives, 1) >= 1 && size(derivatives, 2) >= n_points ||
        throw(ArgumentError("derivatives matrix is too small"))
    return _scalar_rate_derivatives_for_grid_unchecked!(
        values, derivatives, provider, state, flow, t_grid, n_points, copy(state))
end

function _scalar_rate_derivatives_for_grid_unchecked!(values::AbstractMatrix, derivatives::AbstractMatrix, provider, state::AbstractPDMPState,
    flow::ContinuousDynamics, t_grid::AbstractVector, n_points::Integer, state_t::AbstractPDMPState)

    for k in 1:n_points
        copyto!(state_t, state)
        move_forward_time!(state_t, t_grid[k], flow)
        values[1, k], derivatives[1, k] = rate_and_derivative(state_t, flow, provider)
    end
    return values, derivatives
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::AbstractRateDerivativeProvider, state::AbstractPDMPState,
    flow::BouncyParticle, t_grid::AbstractVector, n_points::Integer)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix, provider::GradHVPProvider,
    state::AbstractPDMPState, flow::AnyBoomerang, t_grid::AbstractVector, n_points::Integer)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::Union{VHVProvider,FiniteDiffVHV,WithStatsJoint}, state::AbstractPDMPState,
    flow::AnyBoomerang, t_grid::AbstractVector, n_points::Integer)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::AbstractRateDerivativeProvider, state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle}, t_grid::AbstractVector, n_points::Integer)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::GradHVPProvider, state::AbstractPDMPState,
    flow::Union{ZigZag,PreconditionedDynamics{<:AbstractPreconditioner,<:ZigZag}},
    t_grid::AbstractVector, n_points::Integer)
    _check_rate_derivative_storage!(values, derivatives, _rate_channel_count(state, flow), n_points)
    return _componentwise_rate_derivatives_for_grid_unchecked!(
        values, derivatives, provider, state, flow, t_grid, n_points,
        Vector{Float64}(undef, length(state.ξ.x)))
end

_rate_aggregation(::ContinuousDynamics) = :unsupported
_rate_aggregation(::BouncyParticle) = :scalar
_rate_aggregation(::AnyBoomerang) = :scalar
_rate_aggregation(::ZigZag) = :componentwise
_rate_aggregation(::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle}) = :scalar
_rate_aggregation(::PreconditionedDynamics{<:AbstractPreconditioner,<:ZigZag}) = :componentwise
_rate_aggregation(::PreconditionedDynamics{<:AbstractPreconditioner,<:AnyBoomerang}) = :unsupported

_rate_channel_count(state::AbstractPDMPState, flow::ContinuousDynamics) =
    _rate_aggregation(flow) === :scalar ? 1 : length(state.ξ.θ)
_rate_channel_count(state::AbstractPDMPState, ::DensePreconditionedZigZag) =
    length(state.ξ)

_provider_rate_derivative_capability(_) = :directional
_provider_rate_derivative_capability(::GradientOnlyProvider) = :gradient_only
_provider_rate_derivative_capability(::GradHVPProvider{G,Nothing}) where {G} = :gradient_only
_provider_rate_derivative_capability(::GradHVPProvider) = :hvp
_provider_rate_derivative_capability(::VHVProvider) = :directional
_provider_rate_derivative_capability(::FiniteDiffVHV) = :finite_difference
_provider_rate_derivative_capability(::WithStatsJoint) = :directional
_provider_has_directional_derivative(provider) =
    _provider_rate_derivative_capability(provider) !== :gradient_only

_flow_rate_derivative_capability(::ContinuousDynamics) = :unsupported
_flow_rate_derivative_capability(::BouncyParticle) = :scalar_gradient
_flow_rate_derivative_capability(::AnyBoomerang) = :scalar_directional
_flow_rate_derivative_capability(::ZigZag) = :componentwise_hvp_or_fd
_flow_rate_derivative_capability(::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle}) = :scalar_gradient
_flow_rate_derivative_capability(::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}) = :componentwise_hvp_or_fd
_flow_rate_derivative_capability(::PreconditionedDynamics{DensePreconditioner,<:ZigZag}) = :componentwise_hvp
_flow_rate_derivative_capability(::PreconditionedDynamics{<:AbstractPreconditioner,<:AnyBoomerang}) = :unsupported

function _supports_rate_derivatives(provider, flow::ContinuousDynamics)
    provider_capability = _provider_rate_derivative_capability(provider)
    flow_capability = _flow_rate_derivative_capability(flow)
    flow_capability === :scalar_gradient && return true
    flow_capability === :scalar_directional &&
        return provider_capability !== :gradient_only
    flow_capability === :componentwise_hvp_or_fd &&
        return provider_capability in (:hvp, :finite_difference)
    flow_capability === :componentwise_hvp &&
        return provider_capability === :hvp
    return false
end

function _can_use_signed_grid(state::AbstractPDMPState, flow::ContinuousDynamics, provider)
    _rate_aggregation(flow) === :unsupported && return false
    return _supports_rate_derivatives(provider, flow)
end

_uses_builtin_grid_provider(_) = false
_uses_builtin_grid_provider(::AbstractRateDerivativeProvider) = true

function _fill_rate_derivatives!(values::AbstractMatrix, derivatives::AbstractMatrix, provider, state, flow, t_grid, n_points)
    return rate_derivatives_for_grid!(values, derivatives, provider, state, flow, t_grid, n_points)
end

function _fill_rate_derivatives!(values::AbstractMatrix, derivatives::AbstractMatrix, provider, state, flow, t_grid, n_points,
    ::AbstractPDMPState)
    return _fill_rate_derivatives!(values, derivatives, provider, state, flow, t_grid, n_points)
end

function _fill_rate_derivatives!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::AbstractRateDerivativeProvider, state::AbstractPDMPState,
    flow::Union{BouncyParticle,AnyBoomerang,PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle}},
    t_grid, n_points, state_cache::AbstractPDMPState)

    return _scalar_rate_derivatives_for_grid_unchecked!(
        values, derivatives, provider, state, flow, t_grid, n_points, state_cache)
end

function _fill_rate_derivatives!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::GradHVPProvider, state::AbstractPDMPState,
    flow::Union{ZigZag,PreconditionedDynamics{<:AbstractPreconditioner,<:ZigZag}},
    t_grid, n_points, state_cache::AbstractPDMPState)
    return _componentwise_rate_derivatives_for_grid_unchecked!(
        values, derivatives, provider, state, flow, t_grid, n_points, state_cache.ξ.x)
end

function _check_rate_derivative_storage!(
    values::AbstractMatrix, derivatives::AbstractMatrix, n_channels::Integer, n_points::Integer)
    size(values, 1) >= n_channels && size(values, 2) >= n_points ||
        throw(ArgumentError("values matrix is too small"))
    size(derivatives, 1) >= n_channels && size(derivatives, 2) >= n_points ||
        throw(ArgumentError("derivatives matrix is too small"))
    return nothing
end

function _componentwise_rate_derivatives_for_grid_unchecked!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::GradHVPProvider, state::AbstractPDMPState, flow,
    t_grid::AbstractVector, n_points::Integer, x::AbstractVector)
    grad = _provider_grad(provider)
    hvp = _provider_hvp(provider)
    θ = state.ξ.θ
    for k in 1:n_points
        _rate_grid_position!(x, state, flow, t_grid[k])
        ∇U = grad(x)
        Hθ = hvp(x, θ)
        _project_rate_derivative_columns!(
            @view(values[:, k]), @view(derivatives[:, k]), flow, state, ∇U, Hθ)
    end
    return values, derivatives
end

function _rate_grid_position!(x::AbstractVector, state::AbstractPDMPState, flow, τ)
    copyto!(x, state.ξ.x)
    move_forward_time!(SkeletonPoint(x, state.ξ.θ), τ, flow)
    return x
end

function _project_rate_derivative_columns!(values, derivatives,
    ::Union{ZigZag,PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}},
    state::AbstractPDMPState, ∇U::AbstractVector, Hθ::AbstractVector)
    θ = state.ξ.θ
    @inbounds for j in eachindex(θ)
        values[j] = θ[j] * ∇U[j]
        derivatives[j] = θ[j] * Hθ[j]
    end
    return values, derivatives
end

function _project_rate_derivative_columns!(values, derivatives,
    flow::DensePreconditionedZigZag, state::AbstractPDMPState,
    ∇U::AbstractVector, Hθ::AbstractVector)
    scratch = _dense_zigzag_stratum!(state, flow)
    fill!(values, 0.0)
    fill!(derivatives, 0.0)
    k = scratch.active_count
    @inbounds for b in 1:k
        gv = zero(eltype(values))
        hv = zero(eltype(derivatives))
        for a in b:k
            L_ab = scratch.ΣAA[a, b]
            i = scratch.active[a]
            gv += L_ab * ∇U[i]
            hv += L_ab * Hθ[i]
        end
        values[b] = scratch.canonical_signs[b] * gv
        derivatives[b] = scratch.canonical_signs[b] * hv
    end
    return values, derivatives
end

function rate_derivatives_for_grid(provider, state, flow, t_grid, n_points::Integer)
    values = Matrix{Float64}(undef, _rate_channel_count(state, flow), n_points)
    derivatives = similar(values)
    _fill_rate_derivatives!(values, derivatives, provider, state, flow, t_grid, n_points)
    return values, derivatives
end

function _rate_derivative_scratch!(value_buf::Vector{Float64}, derivative_buf::Vector{Float64}, n_channels::Integer, n_points::Integer)
    len = n_channels * n_points
    length(value_buf) < len && resize!(value_buf, len)
    length(derivative_buf) < len && resize!(derivative_buf, len)
    values = reshape(@view(value_buf[1:len]), n_channels, n_points)
    derivatives = reshape(@view(derivative_buf[1:len]), n_channels, n_points)
    return values, derivatives
end

# --- Grid parameter caps, dispatched on flow type ---

_joint_compatible(::BouncyParticle) = true
_joint_compatible(pd::PreconditionedDynamics) = _joint_compatible(pd.dynamics)
_joint_compatible(::ContinuousDynamics) = false

min_grid_cells(::ContinuousDynamics, N_min::Int, ::Int) = N_min
min_grid_cells(::AnyBoomerang, N_min::Int, ::Int) = max(N_min, 5)
min_grid_cells(pd::PreconditionedDynamics, N_min::Int, N::Int) = min_grid_cells(pd.dynamics, N_min, N)

max_grid_horizon(::ContinuousDynamics) = 1e10
max_grid_horizon(::AnyBoomerang) = 8π
max_grid_horizon(pd::PreconditionedDynamics) = max_grid_horizon(pd.dynamics)

# --- helper functions for λ(t) and λ'(t) ---
# Generic fallback: ignore cached gradient for providers that don't support it
get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider, add_rate::Bool, ::AbstractVector) =
    get_rate_and_deriv(state, flow, provider, add_rate)

_rate_gradient(provider, state::AbstractPDMPState, ::Nothing) = _provider_grad(provider)(state.ξ.x)
_rate_gradient(provider, ::AbstractPDMPState, cached_gradient::AbstractVector) = cached_gradient

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::GradHVPProvider, add_rate::Bool=true)
    return _get_rate_and_deriv_hvp(state, flow, provider, add_rate, nothing)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::GradHVPProvider,
    add_rate::Bool, cached_gradient::AbstractVector)
    return _get_rate_and_deriv_hvp(state, flow, provider, add_rate, cached_gradient)
end

function _get_rate_and_deriv_hvp(state::AbstractPDMPState, flow::ContinuousDynamics,
    provider::GradHVPProvider, add_rate::Bool, cached_gradient)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = _rate_gradient(provider, state, cached_gradient)
    f_t = λ(state, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    ispositive(f_t) || return rate, zero(f_t)
    f_prime_t = ∂λ∂t(state, ∇U_xt, _provider_hvp(provider)(xt, vt), flow)
    return rate, f_prime_t
end

function _compute_vhv_scalar(provider::VHVProvider, state::AbstractPDMPState, ∇U_xt::AbstractVector, ::ContinuousDynamics)
    xt, vt = state.ξ.x, state.ξ.θ
    return provider.vhv(xt, vt, vt)
end

function _compute_vhv_scalar(provider::VHVProvider, state::AbstractPDMPState, ∇U_xt::AbstractVector, ::ZigZag)
    xt, vt = state.ξ.x, state.ξ.θ
    w = provider.w_buf === nothing ? similar(vt) : provider.w_buf
    for i in eachindex(vt)
        w[i] = ispositive(vt[i] * ∇U_xt[i]) ? vt[i] : zero(eltype(vt))
    end
    return provider.vhv(xt, vt, w)
end

function _compute_vhv_scalar(provider::VHVProvider, state::AbstractPDMPState, ∇U_xt::AbstractVector, pd::PreconditionedDynamics)
    return _compute_vhv_scalar(provider, state, ∇U_xt, pd.dynamics)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::VHVProvider, add_rate::Bool=true)
    return _get_rate_and_deriv_vhv(state, flow, provider, add_rate, nothing)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::VHVProvider,
    add_rate::Bool, cached_gradient::AbstractVector)
    return _get_rate_and_deriv_vhv(state, flow, provider, add_rate, cached_gradient)
end

_rate_gradient(provider::VHVProvider, state::AbstractPDMPState, ::Nothing) = provider.grad(state.ξ.x)

function _get_rate_and_deriv_vhv(state::AbstractPDMPState, flow::ContinuousDynamics,
    provider::VHVProvider, add_rate::Bool, cached_gradient)
    ∇U_xt = _rate_gradient(provider, state, cached_gradient)
    f_t = λ(state, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    ispositive(f_t) || return rate, zero(f_t)
    f_prime_t = ∂λ∂t(state, ∇U_xt, _compute_vhv_scalar(provider, state, ∇U_xt, flow), flow)
    return rate, f_prime_t
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::WithStatsJoint, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    dphi, d2phi = provider(xt, vt)

    f_t = pos(dphi) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = d2phi

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::WithStatsJoint, add_rate::Bool, ::AbstractVector)
    return get_rate_and_deriv(state, flow, provider, add_rate)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::GradientOnlyProvider, add_rate::Bool=true)
    return _get_rate_and_deriv_gradient_only(state, flow, provider, add_rate, nothing)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::GradientOnlyProvider,
    add_rate::Bool, cached_gradient::AbstractVector)
    return _get_rate_and_deriv_gradient_only(state, flow, provider, add_rate, cached_gradient)
end

function _get_rate_and_deriv_gradient_only(state::AbstractPDMPState, flow::ContinuousDynamics,
    provider::GradientOnlyProvider, add_rate::Bool, cached_gradient)
    ∇U_xt = cached_gradient === nothing ? provider.grad(state.ξ.x) : cached_gradient
    f_t = λ(state, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    return rate, zero(rate)
end

struct FiniteDiffHVP{G} <: AbstractRateDerivativeProvider
    grad::G
    buf::Vector{Float64}
    grad_buf::Vector{Float64}
    hvp_buf::Vector{Float64}
end
FiniteDiffHVP(grad, buf::Vector{Float64}) = FiniteDiffHVP(grad, buf, similar(buf), similar(buf))

function _fd_step_size(xt::AbstractVector, vt::AbstractVector)
    vnorm = norm(vt)
    iszero(vnorm) && return vnorm
    return oftype(vnorm, 1e-5) * max(one(vnorm), norm(xt) / vnorm)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffHVP, add_rate::Bool=true)
    return _get_rate_and_deriv_fd_hvp(state, flow, fd, add_rate, nothing)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffHVP,
    add_rate::Bool, cached_gradient::AbstractVector)
    return _get_rate_and_deriv_fd_hvp(state, flow, fd, add_rate, cached_gradient)
end

function _get_rate_and_deriv_fd_hvp(state::AbstractPDMPState, flow::ContinuousDynamics,
    fd::FiniteDiffHVP, add_rate::Bool, cached_gradient)
    if cached_gradient === nothing
        copyto!(fd.grad_buf, fd.grad(state.ξ.x))
    else
        copyto!(fd.grad_buf, cached_gradient)
    end
    f_t = λ(state, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    ispositive(f_t) || return rate, zero(f_t)

    xt, vt = state.ξ.x, state.ξ.θ
    h = _fd_step_size(xt, vt)
    if iszero(h)
        fill!(fd.hvp_buf, 0.0)
    else
        fd.buf .= xt .+ h .* vt
        ∇U_shifted = fd.grad(fd.buf)
        fd.hvp_buf .= (∇U_shifted .- fd.grad_buf) ./ h
    end
    return rate, ∂λ∂t(state, fd.grad_buf, fd.hvp_buf, flow)
end

function _fd_vhv_scalar(fd::FiniteDiffVHV, xt::AbstractVector, vt::AbstractVector, wt::AbstractVector)
    h = _fd_step_size(xt, vt)
    iszero(h) && return h
    fd.buf .= xt .+ h .* vt
    fd.stats !== nothing && _inc_counter_fd_curvature_gradient_calls(fd.stats)
    ∇U_shifted = fd.grad(fd.buf)
    return (dot(wt, ∇U_shifted) - dot(wt, fd.grad_buf)) / h
end

_restore_reference_vhv(vhv::Real, ::AbstractVector, ::ContinuousDynamics) = vhv
function _restore_reference_vhv(vhv::Real, vt::AbstractVector, flow::AnyBoomerang)
    # `fd.grad` calls compute_gradient!, so for Boomerang it differentiates
    # ∇U - Γ(x-μ). The Boomerang ∂λ∂t method expects curvature of the raw
    # target gradient ∇U and subtracts the reference contribution itself.
    return vhv + dot(vt, flow.Γ, vt)
end
function _restore_reference_vhv(vhv::Real, vt::AbstractVector, flow::LowRankMutableBoomerang)
    return vhv + lowrank_quadform(flow.Γ, vt)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffVHV, add_rate::Bool=true)
    return _get_rate_and_deriv_fd_vhv(state, flow, fd, add_rate, nothing)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffVHV,
    add_rate::Bool, cached_gradient::AbstractVector)
    return _get_rate_and_deriv_fd_vhv(state, flow, fd, add_rate, cached_gradient)
end

function _get_rate_and_deriv_fd_vhv(state::AbstractPDMPState, flow::ContinuousDynamics,
    fd::FiniteDiffVHV, add_rate::Bool, cached_gradient)
    xt, vt = state.ξ.x, state.ξ.θ
    cached_gradient === nothing ? copyto!(fd.grad_buf, fd.grad(xt)) : copyto!(fd.grad_buf, cached_gradient)
    f_t = λ(state, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    ispositive(f_t) || return rate, zero(f_t)

    vhv_scalar = _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
    return rate, ∂λ∂t(state, fd.grad_buf, vhv_scalar, flow)
end

function rate_and_derivative(state::AbstractPDMPState, flow::AnyBoomerang, fd::FiniteDiffVHV)
    xt = state.ξ.x
    vt = state.ξ.θ
    ∇U = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U)
    vhv = _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
    return dot(fd.grad_buf, vt), ∂λ∂t(state, fd.grad_buf, vhv, flow)
end

function rate_and_derivative(
    state::AbstractPDMPState,
    flow::AnyBoomerang,
    fd::FiniteDiffVHV,
    cached_gradient::AbstractVector,
)
    copyto!(fd.grad_buf, cached_gradient)
    xt = state.ξ.x
    vt = state.ξ.θ
    vhv = _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
    return dot(fd.grad_buf, vt), ∂λ∂t(state, fd.grad_buf, vhv, flow)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ZigZag, fd::FiniteDiffVHV, add_rate::Bool=true)
    return _get_rate_and_deriv_fd_vhv_zigzag(state, flow, fd, add_rate, nothing)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ZigZag, fd::FiniteDiffVHV,
    add_rate::Bool, cached_gradient::AbstractVector)
    return _get_rate_and_deriv_fd_vhv_zigzag(state, flow, fd, add_rate, cached_gradient)
end

function _get_rate_and_deriv_fd_vhv_zigzag(state::AbstractPDMPState, flow::ZigZag,
    fd::FiniteDiffVHV, add_rate::Bool, cached_gradient)
    xt, vt = state.ξ.x, state.ξ.θ
    cached_gradient === nothing ? copyto!(fd.grad_buf, fd.grad(xt)) : copyto!(fd.grad_buf, cached_gradient)
    f_t = λ(state, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    ispositive(f_t) || return rate, zero(f_t)

    w = fd.w_buf
    for i in eachindex(vt)
        w[i] = ispositive(vt[i] * fd.grad_buf[i]) ? vt[i] : zero(eltype(vt))
    end
    return rate, _fd_vhv_scalar(fd, xt, vt, w)
end

function rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    fd::FiniteDiffVHV,
    state::AbstractPDMPState,
    flow::ZigZag,
    t_grid::AbstractVector,
    n_points::Integer,
)
    θ = state.ξ.θ
    n_channels = length(θ)
    _check_rate_derivative_storage!(values, derivatives, n_channels, n_points)
    return _finite_diff_vhv_zigzag_rate_derivatives_for_grid_unchecked!(
        values, derivatives, fd, state, flow, t_grid, n_points)
end

function _fill_rate_derivatives!(values::AbstractMatrix, derivatives::AbstractMatrix,
    fd::FiniteDiffVHV, state::AbstractPDMPState,
    flow::Union{ZigZag,PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}},
    t_grid, n_points, ::AbstractPDMPState)
    return _finite_diff_vhv_zigzag_rate_derivatives_for_grid_unchecked!(
        values, derivatives, fd, state, flow, t_grid, n_points)
end

function _finite_diff_vhv_zigzag_rate_derivatives_for_grid_unchecked!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    fd::FiniteDiffVHV,
    state::AbstractPDMPState,
    flow::ZigZag,
    t_grid,
    n_points::Integer,
)
    x0 = state.ξ.x
    θ = state.ξ.θ
    n_channels = length(θ)
    for k in 1:n_points
        @inbounds for j in 1:n_channels
            fd.buf[j] = x0[j] + t_grid[k] * θ[j]
        end
        h = _fd_step_size(fd.buf, θ)
        ∇U = fd.grad(fd.buf)
        copyto!(fd.grad_buf, ∇U)
        any_active = false
        @inbounds for j in 1:n_channels
            values[j, k] = θ[j] * fd.grad_buf[j]
            any_active |= ispositive(values[j, k])
        end
        if iszero(h)
            @inbounds for j in 1:n_channels
                derivatives[j, k] = 0.0
            end
        elseif !any_active
            @inbounds for j in 1:n_channels
                derivatives[j, k] = 0.0
            end
        else
            @inbounds for j in 1:n_channels
                fd.buf[j] += h * θ[j]
            end
            ∇U_shifted = fd.grad(fd.buf)
            @inbounds for j in 1:n_channels
                derivatives[j, k] = θ[j] * (∇U_shifted[j] - fd.grad_buf[j]) / h
            end
        end
    end
    return values, derivatives
end

function _finite_diff_vhv_zigzag_rate_derivatives_for_grid_unchecked!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    fd::FiniteDiffVHV,
    state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag},
    t_grid,
    n_points::Integer,
)
    return _finite_diff_vhv_zigzag_rate_derivatives_for_grid_unchecked!(
        values, derivatives, fd, state, flow.dynamics, t_grid, n_points)
end

function rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    fd::FiniteDiffVHV,
    state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag},
    t_grid::AbstractVector,
    n_points::Integer,
)
    return rate_derivatives_for_grid!(
        values, derivatives, fd, state, flow.dynamics, t_grid, n_points)
end
