module PDMPSamplers

using LinearAlgebra, Random, Statistics
using Distributions
using StatsBase
using DataStructures: PriorityQueue, dequeue_pair!, enqueue!

import PDMats
import SpecialFunctions
import LogExpFunctions

import FillArrays

import ProgressMeter

import QuadGK
import Roots

import ADTypes
import DifferentiationInterface as DI


# for debugging
import Infiltrator

# for plotting
# using Makie
# import CairoMakie

# # for debugging
# import Infiltrator

include("abstract_types.jl")
include("betabernoulli.jl")
include("utils.jl")
include("PDMPState.jl")

# PDMP types
include("dynamics/continuousdynamics.jl")
include("dynamics/zigzag.jl")
include("dynamics/bouncyparticle.jl")
include("dynamics/boomerang.jl")

# Gradient strategies
include("gradient_strategies.jl")

# Algorithms
include("poisson_time_strategies/gridthinning.jl")
include("poisson_time_strategies/rootspoissontime.jl")
include("poisson_time_strategies/thinning.jl")
include("poisson_time_strategies/sticky.jl")
# these need to be implemented/ fixed
include("poisson_time_strategies/adaptivethinning.jl")
include("poisson_time_strategies/exact.jl")


include("trace.jl")

include("pdmp_sample.jl")

export
    # Core types
    SkeletonPoint,
    PDMPState,
    StickyPDMPState,
    PDMPSampler,
    PDMPEvent,
    PDMPTrace,
    PDMPDiscretize,

    BetaBernoulli,

    # Dynamics
    ContinuousDynamics,
    FactorizedDynamics,
    NonFactorizedDynamics,
    ZigZag,
    BouncyParticle,
    Boomerang,

    λ,
    move_forward_time,
    move_forward_time!,


    # Gradient strategies
    FullGradient,
    SubsampledGradient,
    CoordinateWiseGradient,

    compute_gradient!,
    compute_gradient_uncorrected!,

    # Poisson time strategies
    ThinningStrategy,
    GridThinningStrategy,
    RootsPoissonTimeStrategy,
    Sticky,
    ExactStrategy,
    # StickyLoopState,

    # Bound strategies
    GlobalBounds,
    LocalBounds,
    AdaptiveBounds,


    # Main interface
    pdmp_sample,
    # initialize_velocity,
    # refresh_velocity!,
    # move_forward_time!,
    # reflect!,

    # Trace/statistics
    # mean,
    # var,
    # std,
    # cov,
    # cor,
    inclusion_probs

    # Helper functions
    # ab,
    # ab_i,
    # poisson_time,
    # next_time,
    # analytic_next_event_time_Gaussian,

    # Testing utilities
    # test_approximation,
    # test_rate_bounds,
    # test_coordinate_bounds,
    # test_bound_consistency,
    # test_boomerang_target,
    # test_boomerang_dynamics


end # module PDMPSamplers
