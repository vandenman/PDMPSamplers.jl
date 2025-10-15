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
