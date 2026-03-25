# Hessian control variate (HCV) for subsampled PDMP gradients.
#
# Provides a second-order correction that reduces gradient variance near the
# anchor by using cached Hessian information with Cauchy damping:
#   α(x) = v*² / (v*² + ||x - a||²)
#
# The correction term is:
#   Δ = α · (s · HVP_sub(a, v) + C(a) · v)
# where v = x - a, C(a) = (1-s)·H_prior(a) - H_full(a), and s = N/m.
#
# HVP_sub is computed per-evaluation via a callback (BridgeStan HVP call).
# C(a)·v is a cached matrix-vector product, O(d²).

using LinearAlgebra: dot, mul!

"""
    HCVState

Mutable state for the damped Hessian control variate.

Stores the cached correction matrix C(a) and working buffers.
Updated at each anchor update; read at each gradient evaluation.
"""
mutable struct HCVState
    C_matrix::Matrix{Float64}
    Cv_buf::Vector{Float64}
    v_buf::Vector{Float64}
    vstar2::Float64
    enabled::Bool
end

function HCVState(d::Integer; σ_star::Float64=1.0)
    HCVState(zeros(d, d), zeros(d), zeros(d), σ_star^2 * d, false)
end

"""
    apply_hcv_correction!(out, hcv, x, anchor, hvp_at_anchor!, s)

Apply the damped HCV correction to the gradient estimate `out` in-place.

- `hvp_at_anchor!(hvp_out, v)`: callback computing `s · H_sub(a) · v` into `hvp_out`
  (the caller is responsible for the scaling by `s`)
- `s`: the scaling factor N/m
"""
function apply_hcv_correction!(out::Vector{Float64}, hcv::HCVState, x::Vector{Float64},
                               anchor::Vector{Float64}, hvp_at_anchor!,
                               s::Float64, hvp_buf::Vector{Float64})
    hcv.enabled || return out

    @. hcv.v_buf = x - anchor
    vnorm2 = dot(hcv.v_buf, hcv.v_buf)
    α = hcv.vstar2 / (hcv.vstar2 + vnorm2)

    hvp_at_anchor!(hvp_buf, hcv.v_buf)
    mul!(hcv.Cv_buf, hcv.C_matrix, hcv.v_buf)

    @. out += α * (s * hvp_buf + hcv.Cv_buf)
    return out
end

"""
    update_hcv!(hcv, H_full_flat, H_prior_flat, s, d)

Recompute the HCV correction matrix and adaptive trust radius from
flat Hessian arrays (column-major, length d²).

Sets C(a) = (1-s)·H_prior - H_full (log-density Hessians) and
v*² = Tr(diag(-H_full)⁻¹) as a diagonal approximation.
"""
function update_hcv!(hcv::HCVState, H_full_flat::Vector{Float64},
                     H_prior_flat::Vector{Float64}, s::Float64, d::Integer)
    H_full = reshape(H_full_flat, d, d)
    H_prior = reshape(H_prior_flat, d, d)

    @. hcv.C_matrix = (1 - s) * H_prior - H_full

    hcv.vstar2 = sum(1.0 / max(abs(H_full[j, j]), 1e-8) for j in 1:d)
    hcv.enabled = true
    return hcv
end
