"""
    AbstractModelPrior

Interface for model priors used by dependent slab clocks. Subtypes implement
`log_model_add_odds(prior, active, j)`, the log prior odds implied by the model
prior when adding inactive beta coordinate `j` to the active model encoded by
`active`.
"""
abstract type AbstractModelPrior end

"""
    AbstractSlabPrior

Interface for active-face slab boundary providers. A boundary provider supplies
the beta coordinate map, active-face negative gradients, and boundary densities
at zero for inactive coordinates.
"""
abstract type AbstractSlabPrior end

"""
    AbstractUnstickClock
    AbstractAggregateUnstickClock

Clock interfaces for sticky unfreezing. Aggregate clocks sample a single
summed unsticking time over all inactive stickable beta coordinates and then
sample the coordinate label conditionally.
"""
abstract type AbstractUnstickClock end

"""
    AbstractAggregateUnstickClock

Subtype of `AbstractUnstickClock` for clocks that sample a single aggregate
unstick process over all inactive stickable beta coordinates.
"""
abstract type AbstractAggregateUnstickClock <: AbstractUnstickClock end

"""
    AbstractSlabCacheStyle
    NoSlabCache
    FixedCovarianceCache

Traits describing whether a slab boundary provider can be cached by active set.
`FixedCovarianceCache` means the Gaussian slab mean/covariance are fixed over a
linear-flow segment; state-dependent callback providers should use `NoSlabCache`.
"""
abstract type AbstractSlabCacheStyle end

"""
    ScalarLogscaleGaussianLineSegment

Trajectory parameters used by the structured scalar residual clock for
globally scaled exchangeable Gaussian slabs under linear flows.
"""
struct ScalarLogscaleGaussianLineSegment
    a::Float64
    b::Float64
    s0::Float64
    ell0::Float64
    r::Float64
    log_total_weight::Float64
    horizon::Float64
end

"""
    NoSlabCache

Cache trait for slab boundary providers whose active-face quantities cannot be
cached by active set alone, for example state-dependent callbacks.
"""
struct NoSlabCache <: AbstractSlabCacheStyle end

"""
    FixedCovarianceCache

Cache trait for slab boundary providers whose Gaussian covariance is fixed over
the relevant trajectory segment.
"""
struct FixedCovarianceCache <: AbstractSlabCacheStyle end
