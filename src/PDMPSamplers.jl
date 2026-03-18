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

import ElasticArrays: ElasticArray, ElasticMatrix




# for plotting
# using Makie
# import CairoMakie

# # for debugging
# import Infiltrator

include("abstract_types.jl")
include("betabernoulli.jl")
include("utils.jl")
include("PDMPState.jl")
include("gradient_strategies.jl")
include("model.jl")

# PDMP types
# could do
# dynamics/interface # as well?
# dynamics/linearinterface?
include("dynamics/continuousdynamics.jl")
include("dynamics/zigzag.jl")
include("dynamics/bouncyparticle.jl")
include("dynamics/boomerang.jl")
include("dynamics/preconditioned.jl")

# Gradient strategies
# include("gradient_strategies.jl") # This line was moved up

# Algorithms
include("poisson_time_strategies/interface.jl")
include("poisson_time_strategies/gridthinning.jl")
include("poisson_time_strategies/thinning.jl")
include("poisson_time_strategies/sticky.jl")
# these need to be implemented/ fixed
include("poisson_time_strategies/rootspoissontime.jl")
include("poisson_time_strategies/adaptivethinning.jl")
include("poisson_time_strategies/exact.jl")
# include("poisson_time_strategies/optimistic_failsafe.jl") # needs work!


include("trace.jl")
include("transforms.jl")
include("estimators.jl")
include("transformed_estimators.jl")
include("adaptation.jl")
include("stopping_criteria.jl")

include("pdmp_sample.jl")
include("chains.jl")

export
    # Core types
    SkeletonPoint,
    PDMPState,
    StickyPDMPState,
    # PDMPSampler,
    PDMPEvent,
    PDMPTrace,
    PDMPChains,
    PDMPDiscretize,
    PDMPModel,
    GlobalGradientModel,
    CoordinateWiseGradientModel,

    # Not really a part of this package, but useful
    BetaBernoulli,
    BetaBernoulliKappa,

    # Dynamics
    ContinuousDynamics,
    FactorizedDynamics,
    NonFactorizedDynamics,
    ZigZag,
    BouncyParticle,
    Boomerang,
    MutableBoomerang,
    AnyBoomerang,
    LowRankMutableBoomerang,
    LowRankPrecision,
    AdaptiveBoomerang,
    PreconditionedDynamics,
    AbstractPreconditioner,
    DiagonalPreconditioner,
    DensePreconditioner,
    # short hands
    PreconditionedZigZag,
    PreconditionedBPS,
    DensePreconditionedZigZag,
    DensePreconditionedBPS,
    λ, # should be renamed to rate or so to have a non-unicode name
    move_forward_time,
    move_forward_time!,


    # convenience, but mostly for interop with other packages?
    LogDensity,
    # Gradient strategies
    FullGradient,
    SubsampledGradient,
    CoordinateWiseGradient,
    compute_gradient!,

    # Event metadata
    PDMPEventMeta,
    BoundsMeta,
    GradientMeta,
    CoordinateMeta,
    EmptyMeta,

    # Poisson time strategies
    ThinningStrategy,
    GridThinningStrategy,
    OptimisticStrategy,
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
    StoppingCriterion,
    FixedTimeCriterion,
    EventCountCriterion,
    WallTimeCriterion,
    TotalWallTimeCriterion,
    ESSCriterion,
    OnlineESSCriterion,
    AnyCriterion,
    AllCriteria,
    stop_after,
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
    cdf,
    inclusion_probs,
    refresh_rate,
    event_times,
    first_event_time,
    last_event_time,
    ess,
    n_chains,

    # Parameter transforms
    ParameterTransform,
    IdentityTransform,
    LowerBoundTransform,
    UpperBoundTransform,
    DoubleBoundTransform,
    inv_transform,
    is_increasing

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

include("precompile_workload.jl")

end # module PDMPSamplers
