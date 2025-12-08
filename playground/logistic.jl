using PDMPSamplers
using Random, Test, Distributions, LinearAlgebra
using Makie
import PDMats,
    GLM,
    StatsModels,
    DataFrames as DF,
    MCMCDiagnosticTools,
    Statistics,
    LogExpFunctions,
    CairoMakie

# import DifferentiationInterface as DI
# import Mooncake
include("../test/helper-gen_data.jl")

pdmp_type = ZigZag
# pdmp_type = BouncyParticle
# pdmp_type = Boomerang
gradient_type = FullGradient
# gradient_type = CoordinateWiseGradient
data_type = LogisticRegressionModel
data_arg = (5, 100)
algorithm = GridThinningStrategy
# algorithm = ThinningStrategy # Works poorly with Roots, probably because no longer monotone?
# algorithm = RootsPoissonTimeStrategy
show_progress = true

n = 100
X = randn(n, d-1)
X .-= mean(X, dims=1) # center predictors
X = hcat(ones(n), X)  # first column is the intercept
η = X * β_true
# p = LogExpFunctions.logistic.(η)
# y = rand.(Bernoulli.(p))
y = rand.(BernoulliLogit.(η))

obj, ∇f!, ∇²f!, ∇f_sub_cv!, ∇²f_sub!, resample_indices!, set_anchor!, anchor_info = gen_data(data_type, data_arg...)


d = first(data_arg) # could also be length(D)

T = 50_000.0

if algorithm === ThinningStrategy
    # TODO: this should still be an informed value!
    if gradient_type === CoordinateWiseGradient
        c0 = 1e-2
        alg = ThinningStrategy(LocalBounds(fill(c0, d)))
    else
        # c0 = 1e-6
        c0 = pdmp_type === ZigZag ? 1e-6 : 1e-2
        alg = ThinningStrategy(GlobalBounds(c0 / d, d))
    end
elseif algorithm === GridThinningStrategy
    alg = GridThinningStrategy(; hvp = ∇²f!)
elseif algorithm === ExactStrategy
    alg = ExactStrategy(analytic_next_event_time_Gaussian)
elseif algorithm === RootsPoissonTimeStrategy
    alg = RootsPoissonTimeStrategy()
end

# Use same constructor as working test: ZigZag(sparse(I(d)), zeros(d), σ_value)
# TODO: this constructor should not need a sparse matrix, we can also use an empty constructor?
# or some default types for zero I
# flow = pdmp_type(d)
flow = pdmp_type(d)

x0 = randn(d)
θ0 = PDMPSamplers.initialize_velocity(flow, d)
ξ0 = SkeletonPoint(x0, θ0)

grad = gradient_type == CoordinateWiseGradient ? CoordinateWiseGradient(∂fxᵢ) : FullGradient(∇f!)

trace0, stats0 = pdmp_sample(ξ0, flow, grad, GridThinningStrategy(; hvp = ∇²f!), 0.0, T, progress=show_progress)
stats0.∇f_calls

ts = [event.time for event in trace0.events]
dt = mean(diff(ts))
samples = Matrix(PDMPDiscretize(trace0, dt))

[vec(mean(samples, dims=1)) β_true]

