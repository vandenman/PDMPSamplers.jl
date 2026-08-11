struct IdentityPreconditioner <: AbstractPreconditioner end

struct DiagonalPreconditioner{T<:AbstractVector{<:Real}} <: AbstractPreconditioner
    scale::T
end

mutable struct DensePreconditioner <: AbstractPreconditioner
    L::Matrix{Float64}
    L_operator_norm::Float64
    generation::UInt
    DensePreconditioner(L::Matrix{Float64}, L_operator_norm::Float64,
        generation::UInt, ::Val{:prepared}) = new(L, L_operator_norm, generation)
end

function _prepare_dense_factor(L::AbstractMatrix; expected_size=nothing)
    stored_L = Matrix{Float64}(L)
    size(stored_L, 1) == size(stored_L, 2) || throw(DimensionMismatch(
        "dense preconditioner factor must be square"))
    expected_size === nothing || size(stored_L) == expected_size ||
        throw(DimensionMismatch("dense preconditioner factor has the wrong dimensions"))
    istril(stored_L) || throw(ArgumentError(
        "dense preconditioner factor must be lower triangular"))
    all(isfinite, stored_L) || throw(ArgumentError(
        "dense preconditioner factor must contain only finite values"))
    singular_index = findfirst(iszero, diag(stored_L))
    singular_index === nothing || throw(SingularException(singular_index))
    stored_norm = opnorm(stored_L)
    isfinite(stored_norm) || throw(ArgumentError(
        "dense preconditioner factor produced a non-finite operator norm"))
    return stored_L, stored_norm
end

"""
    DensePreconditioner(L)

Construct a dense preconditioner from a nonsingular lower-triangular factor.
The operator norm and lower-triangular nonsingularity are validated once during
construction; event-time hot paths assume this invariant. Changing `L`
requires `set_dense_preconditioner!` so the cached norm remains synchronized.
Directly mutating entries of `L` is unsupported and can invalidate sampler
correctness.
"""
function DensePreconditioner(L::AbstractMatrix)
    stored_L, stored_norm = _prepare_dense_factor(L)
    return DensePreconditioner(stored_L, stored_norm, zero(UInt), Val(:prepared))
end

function DensePreconditioner(d::Integer)
    L = Matrix{Float64}(I, d, d)
    DensePreconditioner(L)
end

subpreconditioner(pd::IdentityPreconditioner, ::BitVector) = pd
subpreconditioner(pd::DiagonalPreconditioner, free::BitVector) = DiagonalPreconditioner(view(pd.scale, free))
function subpreconditioner(pd::DensePreconditioner, free::BitVector)
    ΣF = (pd.L * transpose(pd.L))[free, free]
    DensePreconditioner(cholesky(Symmetric(ΣF)).L)
end

"""
    set_dense_preconditioner!(pd, L)

Atomically replace a dense preconditioner factor and its cached operator norm.
"""
function set_dense_preconditioner!(pd::DensePreconditioner, L::AbstractMatrix)
    new_L, new_norm = _prepare_dense_factor(L; expected_size=size(pd.L))
    copyto!(pd.L, new_L)
    pd.L_operator_norm = new_norm
    pd.generation += one(UInt)
    return pd
end


struct PreconditionedDynamics{P <: AbstractPreconditioner, D <: ContinuousDynamics} <: ContinuousDynamics
    metric::P
    dynamics::D
end

update_preconditioner!(::Random.AbstractRNG, flow::ContinuousDynamics, ::AbstractPDMPTrace, state::AbstractPDMPState) = flow
update_preconditioner!(flow::ContinuousDynamics, trace::AbstractPDMPTrace, state::AbstractPDMPState, args...) = update_preconditioner!(Random.default_rng(), flow, trace, state, args...)

function update_preconditioner!(rng::Random.AbstractRNG, flow::PreconditionedDynamics{<:DiagonalPreconditioner}, trace::AbstractPDMPTrace, state::AbstractPDMPState, first_update::Bool = false)
    sigmas = Statistics.std(trace)
    for i in eachindex(sigmas)

        old_scale = flow.metric.scale[i]
        if iszero(sigmas[i])
            new_scale = old_scale / 2.0
        else
            new_scale = sigmas[i]
        end

        flow.metric.scale[i] = new_scale
    end
    _invalidate_boundary_velocity_cache!(state)
    draw_stratum_velocity!(rng, state, flow)
    flow
end

function update_preconditioner!(rng::Random.AbstractRNG, flow::PreconditionedDynamics{DensePreconditioner}, trace::AbstractPDMPTrace, state::AbstractPDMPState, first_update::Bool = false)
    M = flow.metric
    Σ_est = Statistics.cov(trace)
    d = size(Σ_est, 1)

    for i in 1:d
        Σ_est[i, i] = max(Σ_est[i, i], 1e-8)
    end
    Σ_sym = Symmetric((Σ_est + Σ_est') / 2)

    L_new = try
        cholesky(Σ_sym).L
    catch e
        e isa PosDefException || rethrow()
        return flow
    end

    set_dense_preconditioner!(M, L_new)

    _invalidate_boundary_velocity_cache!(state)
    draw_stratum_velocity!(rng, state, flow)
    flow
end

_draw_canonical_velocity!(rng, velocity, ::ZigZag) =
    rand!(rng, velocity, (-1.0, 1.0))
_draw_canonical_velocity!(rng, velocity, ::ContinuousDynamics) =
    randn!(rng, velocity)

isfactorized(::PreconditionedDynamics{<:Any, T}) where {T<:ContinuousDynamics} = isfactorized(T)
isfactorized(::PreconditionedDynamics{DensePreconditioner, <:ZigZag}) = false

"""
Transform velocity vector `v` according to preconditioner `M`.
Usually equivalent to v := M * v
"""
transform_velocity!(v, ::IdentityPreconditioner) = nothing # No-op
transform_velocity!(v, M::DiagonalPreconditioner) = (v .*= M.scale)
function transform_velocity!(v, M::DensePreconditioner)
    canonical = copy(v)
    mul!(v, M.L, canonical)
    return v
end


# 1. Trajectory Movement (Kinematics are invariant)
move_forward_time!(ξ::SkeletonPoint, τ, pd::PreconditionedDynamics) = move_forward_time!(ξ, τ, pd.dynamics)
move_forward_time!(state::AbstractPDMPState, τ, pd::PreconditionedDynamics) = move_forward_time!(state, τ, pd.dynamics)

# 2. Event Rates (Dot products are invariant)
λ(ξ::SkeletonPoint, ∇ϕ::AbstractVector, pd::PreconditionedDynamics) = λ(ξ, ∇ϕ, pd.dynamics)

function rate_and_derivative(
    state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle},
    provider,
    args...,
)
    return rate_and_derivative(state, flow.dynamics, provider, args...)
end

function rate_and_derivative(
    state::AbstractPDMPState,
    flow::PreconditionedDynamics{<:AbstractPreconditioner,<:BouncyParticle},
    provider,
    cached_gradient::AbstractVector,
)
    return rate_and_derivative(state, flow.dynamics, provider, cached_gradient)
end

# 3. Reflection Logic (Mirroring is invariant)
reflect!(rng::Random.AbstractRNG, ξ::SkeletonPoint, ∇ϕ::AbstractVector, pd::PreconditionedDynamics, cache) = reflect!(rng, ξ, ∇ϕ, pd.dynamics, cache)
reflect!(rng::Random.AbstractRNG, state::AbstractPDMPState, ∇ϕ::AbstractVector, pd::PreconditionedDynamics, cache) = reflect!(rng, state, ∇ϕ, pd.dynamics, cache)


# 4. Hitting Times (Geometry is invariant)
sticking_time(ξ::SkeletonPoint, pd::PreconditionedDynamics, i::Integer) = sticking_time(ξ, pd.dynamics, i)

"""
Get the refreshment rate for the given continuous dynamics.

    Fallback implementation assumes `flow` has a field `λref` and returns `flow.λref`.
"""
refresh_rate(flow::PreconditionedDynamics) = refresh_rate(flow.dynamics)


# 5. Initialization
function initialize_velocity(rng::Random.AbstractRNG, pd::PreconditionedDynamics, d::Integer)
    # 1. Ask inner dynamics for a "canonical" velocity (e.g., {-1, 1})
    v = initialize_velocity(rng, pd.dynamics, d)
    # 2. Stretch it to physical space (e.g., {-a_i, a_i})
    transform_velocity!(v, pd.metric)
    return v
end

# 6. Refreshment
function refresh_velocity!(rng::Random.AbstractRNG, ξ::SkeletonPoint, pd::PreconditionedDynamics)
    # 1. Let the inner dynamics refresh to a canonical state
    # (Note: This assumes the inner dynamics resets v to something standard, like N(0,I))
    refresh_velocity!(rng, ξ, pd.dynamics)
    # 2. Apply the preconditioner again
    transform_velocity!(ξ.θ, pd.metric)
end

refresh_velocity!(::Random.AbstractRNG, ::SkeletonPoint, ::PreconditionedDynamics{<:AbstractPreconditioner, <:ZigZag}) = nothing
refresh_velocity!(::Random.AbstractRNG, ::SkeletonPoint,
    ::PreconditionedDynamics{DensePreconditioner,<:ZigZag}) = nothing

function initialize_flow_state!(state::AbstractPDMPState,
        pd::PreconditionedDynamics{DensePreconditioner,<:ZigZag})
    _synchronize_dense_zigzag_velocity!(state, pd)
    return nothing
end

subflow(pd::PreconditionedDynamics, free::BitVector) = PreconditionedDynamics(subpreconditioner(pd.metric, free), subflow(pd.dynamics, free))

function initialize_cache(rng::Random.AbstractRNG, flow::PreconditionedDynamics, grad::GlobalGradientStrategy, alg::PoissonTimeStrategy, t::Real, ξ::SkeletonPoint)
    return initialize_cache(rng, flow.dynamics, grad, alg, t, ξ)
end

function initialize_cache(::Random.AbstractRNG, ::PreconditionedDynamics{DensePreconditioner}, ::GlobalGradientStrategy, ::PoissonTimeStrategy, ::Real, ξ::SkeletonPoint)
    return (; z=similar(ξ.x), tmp=similar(ξ.x))
end

# --- Dense-preconditioned ZigZag overrides ---
# In canonical space z = L⁻¹x, the rate decomposes coordinate-wise.
# Canonical gradient: ∇z = L' ∇x. Rate: Σ pos(v_i * ∇z_i).

const DensePreconditionedZigZag = PreconditionedDynamics{DensePreconditioner, <:ZigZag}
const DensePreconditionedBPS = PreconditionedDynamics{DensePreconditioner, <:BouncyParticle}

function _full_dense_zigzag_signs!(state::PDMPState, pd::DensePreconditionedZigZag)
    scratch = _ensure_boundary_scratch!(state.boundary_scratch, length(state.ξ))
    d = length(state.ξ)
    if !scratch.active_factor_valid || scratch.factor_source !== pd.metric ||
            scratch.factor_generation != pd.metric.generation
        @inbounds for i in 1:d
            scratch.active[i] = i
            scratch.cached_free[i] = true
            for j in 1:d
                scratch.ΣAA[i, j] = pd.metric.L[i, j]
            end
        end
        scratch.active_count = d
        scratch.factor_source = pd.metric
        scratch.factor_generation = pd.metric.generation
        scratch.active_factor_valid = true
        copyto!(scratch.solved_θ, state.ξ.θ)
        ldiv!(LowerTriangular(pd.metric.L), view(scratch.solved_θ, 1:d))
        @inbounds for i in 1:d
            scratch.canonical_signs[i] = ifelse(scratch.solved_θ[i] < 0, -1.0, 1.0)
        end
    end
    return scratch
end

_dense_zigzag_stratum!(state::PDMPState, pd::DensePreconditionedZigZag) =
    _full_dense_zigzag_signs!(state, pd)

function _synchronize_dense_zigzag_velocity!(state::AbstractPDMPState,
        pd::DensePreconditionedZigZag)
    scratch = _dense_zigzag_stratum!(state, pd)
    k = scratch.active_count
    mul!(view(scratch.θA, 1:k),
         LowerTriangular(view(scratch.ΣAA, 1:k, 1:k)),
         view(scratch.canonical_signs, 1:k))
    fill!(state.ξ.θ, 0.0)
    @inbounds for a in 1:k
        state.ξ.θ[scratch.active[a]] = scratch.θA[a]
    end
    return state
end

function λ(state::AbstractPDMPState, ∇ϕ::AbstractVector,
        pd::DensePreconditionedZigZag)
    scratch = _dense_zigzag_stratum!(state, pd)
    k = scratch.active_count
    rate = zero(eltype(∇ϕ))
    @inbounds for b in 1:k
        grad_z = zero(eltype(∇ϕ))
        for a in b:k
            grad_z += scratch.ΣAA[a, b] * ∇ϕ[scratch.active[a]]
        end
        rate += pos(scratch.canonical_signs[b] * grad_z)
    end
    return rate
end

function λ(ξ::SkeletonPoint, ∇ϕ::AbstractVector, pd::DensePreconditionedZigZag)
    M = pd.metric
    L = M.L
    v = copy(ξ.θ)
    ldiv!(LowerTriangular(L), v)
    rate = zero(eltype(∇ϕ))
    d = length(v)
    @inbounds for i in 1:d
        grad_z_i = zero(eltype(∇ϕ))
        for j in 1:d
            grad_z_i += L[j, i] * ∇ϕ[j]
        end
        rate += pos(v[i] * grad_z_i)
    end
    return rate
end

function reflect!(rng::Random.AbstractRNG, ξ::SkeletonPoint, ∇ϕ::AbstractVector,
        pd::DensePreconditionedZigZag, cache)
    M = pd.metric
    L = M.L
    v = copy(ξ.θ)
    ldiv!(LowerTriangular(L), v)
    d = length(v)

    # Compute canonical gradient L'∇ϕ and weighted rates
    z = cache.z
    mul!(z, L', ∇ϕ)

    i₀ = _rand_posdot_index(rng, v, z)

    # Flip canonical velocity and update physical velocity
    old_vi = v[i₀]
    v[i₀] = -old_vi
    @inbounds for j in 1:d
        ξ.θ[j] -= 2.0 * old_vi * L[j, i₀]
    end
    return nothing
end

function reflect!(rng::Random.AbstractRNG, state::AbstractPDMPState,
        ∇ϕ::AbstractVector, pd::DensePreconditionedZigZag, cache)
    scratch = _dense_zigzag_stratum!(state, pd)
    k = scratch.active_count
    z = view(scratch.solved_θ, 1:k)
    @inbounds for b in 1:k
        z[b] = zero(eltype(z))
        for a in b:k
            z[b] += scratch.ΣAA[a, b] * ∇ϕ[scratch.active[a]]
        end
    end
    i₀ = _rand_posdot_index(rng, view(scratch.canonical_signs, 1:k), z)
    old_sign = scratch.canonical_signs[i₀]
    scratch.canonical_signs[i₀] = -old_sign
    @inbounds for a in i₀:k
        state.ξ.θ[scratch.active[a]] -= 2.0 * old_sign * scratch.ΣAA[a, i₀]
    end
    return nothing
end

# Type aliases for common combinations
const PreconditionedZigZag{T} = PreconditionedDynamics{DiagonalPreconditioner{T}, ZigZag}
const PreconditionedBPS{T} = PreconditionedDynamics{DiagonalPreconditioner{T}, BouncyParticle}

# Convenience constructors
"""
    PreconditionedZigZag(d::Integer; scale=ones(d))

Create a preconditioned Zig-Zag sampler with `d` dimensions.
"""
function PreconditionedZigZag(d::Integer; scale::AbstractVector{T}=ones(d)) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), ZigZag(d))
end

"""
    PreconditionedBPS(d::Integer; refresh_rate=1.0, scale=ones(d))

Create a preconditioned BPS sampler with `d` dimensions.
"""
function PreconditionedBPS(d::Integer; refresh_rate::Real=1.0, scale::AbstractVector{T}=ones(d)) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), BouncyParticle(d, refresh_rate))
end

function PreconditionedBPS(Γ::AbstractMatrix, μ::AbstractVector; refresh_rate::Real=1.0, scale::AbstractVector{T}=ones(length(μ))) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), BouncyParticle(Γ, μ, refresh_rate))
end

function PreconditionedZigZag(Γ::AbstractMatrix, μ::AbstractVector; scale::AbstractVector{T}=ones(length(μ))) where {T}
    PreconditionedDynamics(DiagonalPreconditioner(collect(scale)), ZigZag(Γ, μ))
end

# Dense ZigZag rates live in canonical coordinates. Its deterministic affine
# roof must therefore transform both the reference gradient and its trajectory
# derivative with L', just like λ does.
function ab(ξ::SkeletonPoint, c::AbstractVector,
        pd::DensePreconditionedZigZag, cache)
    M = pd.metric
    flow = pd.dynamics
    z = cache.z
    tmp = cache.tmp

    @inbounds for i in eachindex(tmp, ξ.x, flow.μ)
        tmp[i] = ξ.x[i] - flow.μ[i]
    end
    mul!(z, flow.Γ, tmp)
    mul!(tmp, transpose(M.L), z)
    signs = copy(ξ.θ)
    ldiv!(LowerTriangular(M.L), signs)
    a = sum(c) + posdot(signs, tmp)

    mul!(z, flow.Γ, ξ.θ)
    mul!(tmp, transpose(M.L), z)
    b = posdot(signs, tmp)
    return a, b, Inf
end

function ab(state::AbstractPDMPState, c::AbstractVector,
        pd::DensePreconditionedZigZag, cache)
    scratch = _dense_zigzag_stratum!(state, pd)
    k = scratch.active_count
    z = cache.z
    tmp = cache.tmp
    @inbounds for i in eachindex(tmp, state.ξ.x, pd.dynamics.μ)
        tmp[i] = state.ξ.x[i] - pd.dynamics.μ[i]
    end
    mul!(z, pd.dynamics.Γ, tmp)
    a = sum(c)
    @inbounds for b in 1:k
        transformed = zero(eltype(z))
        for aa in b:k
            transformed += scratch.ΣAA[aa, b] * z[scratch.active[aa]]
        end
        a += pos(scratch.canonical_signs[b] * transformed)
    end
    mul!(z, pd.dynamics.Γ, state.ξ.θ)
    b_rate = zero(eltype(z))
    @inbounds for b in 1:k
        transformed = zero(eltype(z))
        for aa in b:k
            transformed += scratch.ΣAA[aa, b] * z[scratch.active[aa]]
        end
        b_rate += pos(scratch.canonical_signs[b] * transformed)
    end
    return a, b_rate, Inf
end

ab(state::AbstractPDMPState, c::AbstractVector,
    pd::PreconditionedDynamics, cache) = ab(state.ξ, c, pd, cache)

# Forwarding ab methods
ab(ξ::SkeletonPoint, c::AbstractVector, pd::PreconditionedDynamics, cache) = ab(ξ, c, pd.dynamics, cache)
ab_i(i::Integer, ξ::SkeletonPoint, c::AbstractVector, pd::PreconditionedDynamics, cache) = ab_i(i, ξ, c, pd.dynamics, cache)


# Convenience constructors for dense-preconditioned dynamics
function DensePreconditionedZigZag(d::Integer)
    PreconditionedDynamics(DensePreconditioner(d), ZigZag(d))
end

function DensePreconditionedZigZag(Γ::AbstractMatrix, μ::AbstractVector)
    PreconditionedDynamics(DensePreconditioner(length(μ)), ZigZag(Γ, μ))
end

function DensePreconditionedBPS(d::Integer; refresh_rate::Real=1.0)
    PreconditionedDynamics(DensePreconditioner(d), BouncyParticle(d, refresh_rate))
end

function DensePreconditionedBPS(Γ::AbstractMatrix, μ::AbstractVector; refresh_rate::Real=1.0)
    PreconditionedDynamics(DensePreconditioner(length(μ)), BouncyParticle(Γ, μ, refresh_rate))
end

function ∂λ∂t(state::AbstractPDMPState, ∇U_xt::AbstractVector, curvature_input, pd::PreconditionedDynamics)
    return ∂λ∂t(state, ∇U_xt, curvature_input, pd.dynamics)
end
function default_aggregate_unstick_clock(provider::GlobalLogscaleExchangeableGaussianSlab,
                                         model_prior::AbstractModelPrior,
                                         flow::PreconditionedDynamics)
    return SummedRateClock(provider, model_prior)
end
