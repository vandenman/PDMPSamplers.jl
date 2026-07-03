function rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    provider,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    t_grid::AbstractVector,
    n_points::Integer,
)
    throw(ArgumentError(
        "bound=:linear requires rate derivatives for $(typeof(flow)); " *
        "use bound=:constant or implement rate_derivatives_for_grid!"))
end

function _scalar_rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    provider,
    state::AbstractPDMPState,
    flow::ContinuousDynamics,
    t_grid::AbstractVector,
    n_points::Integer,
)
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

function rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    provider::Union{Tuple,GradHVPProvider,VHVProvider,FiniteDiffVHV,WithStatsJoint},
    state::AbstractPDMPState,
    flow::BouncyParticle,
    t_grid::AbstractVector,
    n_points::Integer,
)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    provider::Union{Tuple,GradHVPProvider},
    state::AbstractPDMPState,
    flow::AnyBoomerang,
    t_grid::AbstractVector,
    n_points::Integer,
)
    return _scalar_rate_derivatives_for_grid!(
        values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    provider::Union{Tuple,GradHVPProvider,VHVProvider,FiniteDiffVHV,WithStatsJoint},
    state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle},
    t_grid::AbstractVector,
    n_points::Integer,
)
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
_supports_rate_derivatives(
    provider,
    ::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle},
) = true
_supports_rate_derivatives((grad, hvp)::Tuple{G,H}, ::ZigZag) where {G,H} =
    !(H <: Nothing)
_supports_rate_derivatives(provider::GradHVPProvider{G,H}, ::ZigZag) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives(
    (grad, hvp)::Tuple{G,H},
    ::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag},
) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives(
    provider::GradHVPProvider{G,H},
    ::PreconditionedDynamics{<:DiagonalPreconditioner,<:ZigZag},
) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives(
    (grad, hvp)::Tuple{G,H},
    ::PreconditionedDynamics{DensePreconditioner,<:ZigZag},
) where {G,H} = !(H <: Nothing)
_supports_rate_derivatives(
    provider::GradHVPProvider{G,H},
    ::PreconditionedDynamics{DensePreconditioner,<:ZigZag},
) where {G,H} = !(H <: Nothing)

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

function _fill_rate_derivatives!(
    values::AbstractMatrix,
    derivatives::AbstractMatrix,
    provider,
    state,
    flow,
    t_grid,
    n_points,
)
    return rate_derivatives_for_grid!(values, derivatives, provider, state, flow, t_grid, n_points)
end

function rate_derivatives_for_grid(provider, state, flow, t_grid, n_points::Integer)
    values = Matrix{Float64}(undef, _rate_channel_count(state, flow), n_points)
    derivatives = similar(values)
    _fill_rate_derivatives!(values, derivatives, provider, state, flow, t_grid, n_points)
    return values, derivatives
end

function _rate_derivative_scratch!(
    value_buf::Vector{Float64},
    derivative_buf::Vector{Float64},
    n_channels::Integer,
    n_points::Integer,
)
    len = n_channels * n_points
    length(value_buf) < len && resize!(value_buf, len)
    length(derivative_buf) < len && resize!(derivative_buf, len)
    values = reshape(@view(value_buf[1:len]), n_channels, n_points)
    derivatives = reshape(@view(derivative_buf[1:len]), n_channels, n_points)
    return values, derivatives
end
