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
    state_t = copy(state)
    for k in 1:n_points
        copyto!(state_t, state)
        move_forward_time!(state_t, t_grid[k], flow)
        values[1, k], derivatives[1, k] = rate_and_derivative(state_t, flow, provider)
    end
    return values, derivatives
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::Union{Tuple,GradHVPProvider,VHVProvider,FiniteDiffVHV,WithStatsJoint}, state::AbstractPDMPState,
    flow::BouncyParticle, t_grid::AbstractVector, n_points::Integer)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix, provider::Union{Tuple,GradHVPProvider},
    state::AbstractPDMPState, flow::AnyBoomerang, t_grid::AbstractVector, n_points::Integer)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(values::AbstractMatrix, derivatives::AbstractMatrix,
    provider::Union{Tuple,GradHVPProvider,VHVProvider,FiniteDiffVHV,WithStatsJoint}, state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle}, t_grid::AbstractVector, n_points::Integer)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
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
_rate_channel_count(state::AbstractPDMPState, flow::DensePreconditionedZigZag) =
    length(flow.metric.v_canonical)

_provider_has_directional_derivative(_) = true
_provider_has_directional_derivative(::Tuple{G,Nothing}) where {G} = false
_provider_has_directional_derivative(::GradHVPProvider{G,Nothing}) where {G} = false

_supports_rate_derivatives(provider, ::ContinuousDynamics) = false
_supports_rate_derivatives(provider, ::BouncyParticle) = true
_supports_rate_derivatives(provider, ::AnyBoomerang) =
    _provider_has_directional_derivative(provider)
_supports_rate_derivatives(provider, ::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle}) = true
_supports_rate_derivatives((grad, hvp)::Tuple{G,H}, ::ZigZag) where {G,H} =
    !(H <: Nothing)
_supports_rate_derivatives(provider::GradHVPProvider{G,H}, ::ZigZag) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives((grad, hvp)::Tuple{G,H},
    ::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives(provider::GradHVPProvider{G,H},
    ::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag}) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives((grad, hvp)::Tuple{G,H},
    ::PreconditionedDynamics{DensePreconditioner,<:ZigZag}) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives(provider::GradHVPProvider{G,H},
    ::PreconditionedDynamics{DensePreconditioner,<:ZigZag}) where {G,H} = !(H <: Nothing)

function _can_use_signed_grid(state::AbstractPDMPState, flow::ContinuousDynamics, provider)
    _rate_aggregation(flow) === :unsupported && return false
    return _supports_rate_derivatives(provider, flow)
end

_uses_builtin_grid_provider(_) = false
_uses_builtin_grid_provider(::Tuple) = true
_uses_builtin_grid_provider(::GradHVPProvider) = true
_uses_builtin_grid_provider(::VHVProvider) = true
_uses_builtin_grid_provider(::FiniteDiffVHV) = true
_uses_builtin_grid_provider(::WithStatsJoint) = true

function _fill_rate_derivatives!(values::AbstractMatrix, derivatives::AbstractMatrix, provider, state, flow, t_grid, n_points)
    return rate_derivatives_for_grid!(values, derivatives, provider, state, flow, t_grid, n_points)
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

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::Union{Tuple,GradHVPProvider}, add_rate::Bool=true)

    xt, vt = state.ξ.x, state.ξ.θ  # state already moved to time t

    grad = _provider_grad(provider)
    hvp = _provider_hvp(provider)
    ∇U_xt = grad(xt)
    Hxt_vt = hvp(xt, vt)  # Hessian-vector product

    # base rate (before positive-part)
    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)

    f_prime_t = ∂λ∂t(state, ∇U_xt, Hxt_vt, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv

end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::Union{Tuple,GradHVPProvider},
    add_rate::Bool, cached_gradient::AbstractVector)
    xt, vt = state.ξ.x, state.ξ.θ
    hvp = _provider_hvp(provider)
    Hxt_vt = hvp(xt, vt)
    f_t = λ(state.ξ, cached_gradient, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, cached_gradient, Hxt_vt, flow)
    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
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
    xt, vt = state.ξ.x, state.ξ.θ

    ∇U_xt = provider.grad(xt)

    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)

    curvature_scalar = _compute_vhv_scalar(provider, state, ∇U_xt, flow)
    f_prime_t = ∂λ∂t(state, ∇U_xt, curvature_scalar, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)

    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, provider::VHVProvider,
    add_rate::Bool, cached_gradient::AbstractVector)
    f_t = λ(state.ξ, cached_gradient, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    curvature_scalar = _compute_vhv_scalar(provider, state, cached_gradient, flow)
    f_prime_t = ∂λ∂t(state, cached_gradient, curvature_scalar, flow)
    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
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

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, grad_and_nothing::Tuple{G,Nothing}, add_rate::Bool=true) where {G}
    grad = grad_and_nothing[1]
    xt = state.ξ.x
    ∇U_xt = grad(xt)
    f_t = λ(state.ξ, ∇U_xt, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    return rate, zero(rate)
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, grad_and_nothing::Tuple{G,Nothing},
    add_rate::Bool, cached_gradient::AbstractVector) where {G}
    f_t = λ(state.ξ, cached_gradient, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    rate = pos(f_t)
    return rate, zero(rate)
end

struct FiniteDiffHVP{G}
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
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    h = _fd_step_size(xt, vt)
    if iszero(h)
        fill!(fd.hvp_buf, 0.0)
    else
        fd.buf .= xt .+ h .* vt
        ∇U_shifted = fd.grad(fd.buf)
        fd.hvp_buf .= (∇U_shifted .- fd.grad_buf) ./ h
    end

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, fd.hvp_buf, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffHVP,
    add_rate::Bool, cached_gradient::AbstractVector)
    copyto!(fd.grad_buf, cached_gradient)

    xt, vt = state.ξ.x, state.ξ.θ
    h = _fd_step_size(xt, vt)
    if iszero(h)
        fill!(fd.hvp_buf, 0.0)
    else
        fd.buf .= xt .+ h .* vt
        ∇U_shifted = fd.grad(fd.buf)
        fd.hvp_buf .= (∇U_shifted .- fd.grad_buf) ./ h
    end

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, fd.hvp_buf, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function _fd_vhv_scalar(fd::FiniteDiffVHV, xt::AbstractVector, vt::AbstractVector, wt::AbstractVector)
    h = _fd_step_size(xt, vt)
    iszero(h) && return h
    fd.buf .= xt .+ h .* vt
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
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    vhv_scalar = _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, vhv_scalar, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ContinuousDynamics, fd::FiniteDiffVHV,
    add_rate::Bool, cached_gradient::AbstractVector)
    copyto!(fd.grad_buf, cached_gradient)

    xt, vt = state.ξ.x, state.ξ.θ
    vhv_scalar = _restore_reference_vhv(_fd_vhv_scalar(fd, xt, vt, vt), vt, flow)
    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = ∂λ∂t(state, fd.grad_buf, vhv_scalar, flow)

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ZigZag, fd::FiniteDiffVHV, add_rate::Bool=true)
    xt, vt = state.ξ.x, state.ξ.θ
    ∇U_xt = fd.grad(xt)
    copyto!(fd.grad_buf, ∇U_xt)

    w = fd.w_buf
    for i in eachindex(vt)
        w[i] = ispositive(vt[i] * fd.grad_buf[i]) ? vt[i] : zero(eltype(vt))
    end
    whv_scalar = _fd_vhv_scalar(fd, xt, vt, w)

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = whv_scalar

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end

function get_rate_and_deriv(state::AbstractPDMPState, flow::ZigZag, fd::FiniteDiffVHV,
    add_rate::Bool, cached_gradient::AbstractVector)
    copyto!(fd.grad_buf, cached_gradient)

    vt = state.ξ.θ
    w = fd.w_buf
    for i in eachindex(vt)
        w[i] = ispositive(vt[i] * fd.grad_buf[i]) ? vt[i] : zero(eltype(vt))
    end
    xt = state.ξ.x
    whv_scalar = _fd_vhv_scalar(fd, xt, vt, w)

    f_t = λ(state.ξ, fd.grad_buf, flow) + (add_rate ? refresh_rate(flow) : 0.0)
    f_prime_t = whv_scalar

    rate = pos(f_t)
    rate_deriv = ispositive(f_t) ? f_prime_t : zero(f_prime_t)
    return rate, rate_deriv
end
