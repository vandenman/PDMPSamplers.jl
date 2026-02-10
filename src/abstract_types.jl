# Strategy abstract types
abstract type PoissonTimeStrategy end
abstract type BoundStrategy end

abstract type GradientStrategy end
abstract type GlobalGradientStrategy         <: GradientStrategy end
abstract type CoordinateWiseGradientStrategy <: GradientStrategy end # factorized

# Dynamics abstract types
abstract type ContinuousDynamics end
abstract type FactorizedDynamics    <: ContinuousDynamics end
abstract type NonFactorizedDynamics <: ContinuousDynamics end

isfactorized(flow::ContinuousDynamics) = flow isa FactorizedDynamics
isfactorized(::Type{T}) where {T<:ContinuousDynamics} = T <: FactorizedDynamics

# Preconditioner abstract type
abstract type AbstractPreconditioner end

# pdmp trace & events
abstract type AbstractPDMPTrace end
abstract type AbstractPDMPEvent end

# adaptation
abstract type AbstractAdapter end