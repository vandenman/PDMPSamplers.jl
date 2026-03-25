# Anchor bank for subsampled PDMP gradients.
#
# Maintains a spatial cache of K anchor points in the parameter space. Each
# anchor stores its position and the full gradient at that position.
#
# At each trajectory segment start, the closest anchor is selected (Euclidean
# distance, O(K·d) brute-force). This bounds ||x - a|| and hence the
# control-variate variance.
#
# Anchors are populated during warmup via periodic full-data evaluations. The
# bank has a fixed capacity (default K=20); when full, the least-recently-used
# entry is evicted.
#
# Future work: post-warmup anchor creation when min distance exceeds a
# threshold; adaptive capacity; Mahalanobis distance using the adapted mass
# matrix.

using LinearAlgebra: dot

"""
    AnchorEntry

A single anchor in the bank. Stores the position and full gradient.
Optionally stores HCV state for second-order correction.
"""
mutable struct AnchorEntry
    position::Vector{Float64}
    full_gradient::Vector{Float64}
    hcv::Union{Nothing, HCVState}
    age::Int
end

function AnchorEntry(d::Integer)
    AnchorEntry(zeros(d), zeros(d), nothing, 0)
end

function AnchorEntry(d::Integer, hcv::HCVState)
    AnchorEntry(zeros(d), zeros(d), hcv, 0)
end

"""
    AnchorBank

A fixed-capacity bank of anchor points for subsampled PDMP control variates.

# Fields
- `entries`: pre-allocated anchor slots (length = capacity)
- `active_idx`: index of the currently selected anchor (0 if empty)
- `n_populated`: number of filled entries
- `strategy`: selection strategy (`:bruteforce` or `:triangle_prune`)
- `inter_dist`: inter-anchor Euclidean distance matrix (used by `:triangle_prune`)
"""
mutable struct AnchorBank
    entries::Vector{AnchorEntry}
    active_idx::Int
    n_populated::Int
    strategy::Symbol
    inter_dist::Matrix{Float64}
end

function AnchorBank(d::Integer; capacity::Int=20, use_hcv::Bool=false, strategy::Symbol=:bruteforce)
    entries = if use_hcv
        [AnchorEntry(d, HCVState(d)) for _ in 1:capacity]
    else
        [AnchorEntry(d) for _ in 1:capacity]
    end
    AnchorBank(entries, 0, 0, strategy, zeros(capacity, capacity))
end

"""
    has_active_anchor(bank)

Return true if the bank has at least one populated anchor.
"""
has_active_anchor(bank::AnchorBank) = bank.active_idx > 0

Base.isempty(bank::AnchorBank) = iszero(bank.n_populated)

function Base.empty!(bank::AnchorBank)
    bank.active_idx = 0
    bank.n_populated = 0
    return bank
end

"""
    active_entry(bank)

Return the currently selected anchor entry.
"""
active_entry(bank::AnchorBank) = bank.entries[bank.active_idx]

"""
    select_nearest!(bank, x)

Select the anchor closest to `x` (Euclidean distance). Updates `active_idx`
and age counters. Returns the selected entry, or `nothing` if the bank is empty.

Uses the bank's selection strategy: `:bruteforce` (default) performs exhaustive
search; `:triangle_prune` uses inter-anchor distances to skip candidates.
"""
function select_nearest!(bank::AnchorBank, x::AbstractVector{Float64})
    isempty(bank) && return nothing
    if bank.strategy === :triangle_prune && bank.n_populated >= 2
        return _select_nearest_prune!(bank, x)
    end
    return _select_nearest_bruteforce!(bank, x)
end

function _select_nearest_bruteforce!(bank::AnchorBank, x::AbstractVector{Float64})
    best_idx = 1
    best_d2 = Inf
    @inbounds for k in 1:bank.n_populated
        entry = bank.entries[k]
        d2 = _sq_dist(x, entry.position)
        if d2 < best_d2
            best_d2 = d2
            best_idx = k
        end
        entry.age += 1
    end
    bank.entries[best_idx].age = 0
    bank.active_idx = best_idx
    return bank.entries[best_idx]
end

function _select_nearest_prune!(bank::AnchorBank, x::AbstractVector{Float64})
    pivot = bank.active_idx > 0 ? bank.active_idx : 1
    pivot_d2 = _sq_dist(x, bank.entries[pivot].position)
    pivot_dist = sqrt(pivot_d2)
    best_idx = pivot
    best_d2 = pivot_d2
    best_dist = pivot_dist

    @inbounds for k in 1:bank.n_populated
        bank.entries[k].age += 1
        k == pivot && continue
        # Triangle inequality: |d(a_k, a_pivot) - d(x, a_pivot)| >= d_best => skip
        if abs(bank.inter_dist[k, pivot] - pivot_dist) >= best_dist
            continue
        end
        d2 = _sq_dist(x, bank.entries[k].position)
        if d2 < best_d2
            best_d2 = d2
            best_dist = sqrt(d2)
            best_idx = k
        end
    end

    bank.entries[best_idx].age = 0
    bank.active_idx = best_idx
    return bank.entries[best_idx]
end

function _sq_dist(a::AbstractVector, b::AbstractVector)
    s = zero(eltype(a))
    @inbounds for i in eachindex(a, b)
        d = a[i] - b[i]
        s += d * d
    end
    return s
end

function _update_inter_dist!(bank::AnchorBank, idx::Int)
    bank.inter_dist[idx, idx] = 0.0
    @inbounds for j in 1:bank.n_populated
        j == idx && continue
        d = sqrt(_sq_dist(bank.entries[idx].position, bank.entries[j].position))
        bank.inter_dist[idx, j] = d
        bank.inter_dist[j, idx] = d
    end
end

"""
    add_anchor!(bank; position, full_gradient)

Add a new anchor to the bank. If the bank is full, evicts the entry with the
highest age (least recently used). Returns the index of the new/updated entry.
"""
function add_anchor!(bank::AnchorBank;
                     position::Vector{Float64},
                     full_gradient::Vector{Float64})
    idx = if bank.n_populated < length(bank.entries)
        bank.n_populated += 1
        bank.n_populated
    else
        _evict_lru(bank)
    end

    entry = bank.entries[idx]
    copyto!(entry.position, position)
    copyto!(entry.full_gradient, full_gradient)
    entry.age = 0

    if bank.strategy === :triangle_prune
        _update_inter_dist!(bank, idx)
    end

    if bank.active_idx == 0
        bank.active_idx = idx
    end

    return idx
end

function _evict_lru(bank::AnchorBank)
    max_age = -1
    max_idx = 1
    @inbounds for k in 1:bank.n_populated
        if bank.entries[k].age > max_age
            max_age = bank.entries[k].age
            max_idx = k
        end
    end
    return max_idx
end
