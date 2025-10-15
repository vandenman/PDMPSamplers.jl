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

# pdmp_type = ZigZag
# pdmp_type = BouncyParticle
pdmp_type = Boomerang
gradient_type = FullGradient
# gradient_type = CoordinateWiseGradient
data_type = MvNormal
data_arg = (5, 1.0)
algorithm = GridThinningStrategy
# algorithm = ThinningStrategy # Works poorly with Roots, probably because no longer monotone?
# algorithm = RootsPoissonTimeStrategy
show_progress = true





D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(data_type, data_arg...)

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
flow = pdmp_type(inv(Symmetric(cov(D))), mean(D))

x0 = randn(d)
θ0 = PDMPSamplers.initialize_velocity(flow, d)
ξ0 = SkeletonPoint(x0, θ0)

grad = gradient_type == CoordinateWiseGradient ? CoordinateWiseGradient(∂fxᵢ) : FullGradient(∇f!)

trace0, stats0 = pdmp_sample(ξ0, flow, grad, GridThinningStrategy(; hvp = ∇²f!), 0.0, T, progress=show_progress)
stats0.∇f_calls

trace1, stats1 = pdmp_sample(ξ0, flow, grad, RootsPoissonTimeStrategy(bracket_multiplier = 20., rtol = 1e-3, atol = 1e-3), 0.0, T, progress=show_progress)
stats1.∇f_calls
ts = [event.time for event in trace1.events]
dt = mean(diff(ts))
samples = Matrix(PDMPDiscretize(trace1, dt))
test_approximation(samples, D)

trace, stats = pdmp_sample(ξ0, flow, grad, alg, 0.0, T, progress=show_progress)
stats.∇f_calls
# 4_666_758
# 26_592_421
# 130_976_116
# 240_991_578

@profview pdmp_sample(ξ0, flow, grad, alg, 0.0, T/10, progress=false)

ts = [event.time for event in trace.events]
dt = mean(diff(ts))
samples = Matrix(PDMPDiscretize(trace, dt))
test_approximation(samples, D)

[vec(mean(samples, dims=1)) mean(D)]
[cov(samples, dims=1) cov(D)]

fulltrace = PDMPTrace(trace)
samples2 = Matrix(PDMPDiscretize(fulltrace, dt))
test_approximation(samples2, D)

samples ≈ samples2

dtrace1 = PDMPDiscretize(trace, dt)
dtrace2 = PDMPDiscretize(fulltrace, dt)

i0, s0 = iterate(dtrace1)
i1, s1 = iterate(dtrace1, s0)
i2, s2 = iterate(dtrace1, s1)
i3, s3 = iterate(dtrace1, s2)

j0, p0 = iterate(dtrace2)
j1, p1 = iterate(dtrace2, s0)
j2, p2 = iterate(dtrace2, s1)
j3, p3 = iterate(dtrace2, s2)

trace.initial_state == fulltrace.events[1]
state = SkeletonPoint(copy(trace.initial_state.position), copy(trace.initial_state.velocity))
fulltrace.events[2]
trace.initial_state == fulltrace.events[1]

i0.first ≈ j0.first && i0.second ≈ j0.second
i1.first ≈ j1.first && i1.second ≈ j1.second
i2.first ≈ j2.first && i2.second ≈ j2.second



trace
collect(trace)

val0, state0 = iterate(trace)

(t, x, θ, k) = state0
PDMPSamplers._to_next_event!(x, θ, trace.events[k])
val1, state1 = iterate(trace, state0)