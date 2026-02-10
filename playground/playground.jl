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
    alg = GridThinningStrategy()
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

# PDMPModel(d, grad, ∇²f!)

model = PDMPModel(d, grad, ∇²f!)

@edit PDMPModel(d, grad, ∇²f!)

out = similar(x0)
model.grad.f(out, x0)
model.hvp(x0, θ0)

trace0, stats0 = pdmp_sample(ξ0, flow, model, GridThinningStrategy(), 0.0, T, progress=show_progress)


trace0, stats0 = pdmp_sample(ξ0, flow, grad, GridThinningStrategy(; hvp=∇²f!), 0.0, T, progress=show_progress)
stats0.∇f_calls

trace1, stats1 = pdmp_sample(ξ0, flow, grad, RootsPoissonTimeStrategy(bracket_multiplier=20., rtol=1e-3, atol=1e-3), 0.0, T, progress=show_progress)
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

@profview pdmp_sample(ξ0, flow, grad, alg, 0.0, T / 10, progress=false)

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



d = 2
flow0 = ZigZag(d)
flow1 = PreconditionedDynamics(PDMPSamplers.DiagonalPreconditioner(ones(d)), ZigZag(d))
gradient_type = FullGradient

# gradient_type = CoordinateWiseGradient
data_type = MvNormal
data_arg = (5, 1.0)
algorithm = GridThinningStrategy
# algorithm = ThinningStrategy # Works poorly with Roots, probably because no longer monotone?
# algorithm = RootsPoissonTimeStrategy
show_progress = true

import MCMCDiagnosticTools

D, ∇f!, ∇²f!, ∂fxᵢ = gen_data(data_type, d, 1.0, zeros(d), [1., 1_000.], I(d))

flow0 = ZigZag(d)
flow1 = PreconditionedDynamics(PDMPSamplers.DiagonalPreconditioner(ones(d)), ZigZag(d))

x0 = randn(d)
θ00 = PDMPSamplers.initialize_velocity(flow0, d)
θ01 = PDMPSamplers.initialize_velocity(flow1, d)

ξ00 = SkeletonPoint(x0, θ00)
ξ01 = SkeletonPoint(x0, θ01)

grad = gradient_type == CoordinateWiseGradient ? CoordinateWiseGradient(∂fxᵢ) : FullGradient(∇f!)
alg = GridThinningStrategy(; hvp=∇²f!)

trace0, stats0 = pdmp_sample(ξ00, flow0, grad, alg, 0.0, 10000_000., 10_000., progress=show_progress)
stats0.∇f_calls

trace1, stats1 = pdmp_sample(ξ01, flow1, grad, alg, 0.0, 10000_000., 10_000., progress=show_progress)

trace0.initial_state
trace0.events[1].time
trace1.initial_state
trace1.events[1].time

length(trace0.events)
length(trace1.events)


trace1.flow

dt0 = mean(diff([event.time for event in trace0.events]))
dt1 = mean(diff([event.time for event in trace1.events]))
dtrace0 = Matrix(PDMPDiscretize(trace0, dt0))
dtrace1 = Matrix(PDMPDiscretize(trace1, dt1))


stats0.∇f_calls, stats1.∇f_calls, stats1.∇f_calls / stats0.∇f_calls
dt0, dt1

MCMCDiagnosticTools.ess.(eachcol(dtrace0))
MCMCDiagnosticTools.ess.(eachcol(dtrace1))
[mean(trace0) mean(trace1) vec(mean(dtrace0, dims=1)) vec(mean(dtrace1, dims=1))]
[std(trace0) std(trace1) vec(std(dtrace0, dims=1)) vec(std(dtrace1, dims=1))] # somehow allways underestimates?

mean(1_000 * randn() for _ in 1:ceil(Int, 4.094864395953101e6))
std(1_000 * randn() for _ in 1:ceil(Int, 4.094864395953101e6))

cov(trace0) ./ cov(D)
cov(trace1) ./ cov(D)
var(trace0) ./ var(D)
var(trace1) ./ var(D)
var(trace1) ./ flow1.metric.scale .^ 2
flow1.metric.scale
var(trace1) ./ det(Diagonal(flow1.metric.scale))

skps = collect(trace0)
skps = collect(trace1)

@edit iterate(PDMPDiscretize(trace1, dt1))
i0, s0 = iterate(trace1)
i1, s1 = iterate(trace1, s0)

s0
i0

e0 = trace1.initial_state
e1 =
    e0.position

@edit iterate(trace1, s)

trace0
trace0.initial_state

trace1
trace1.initial_state
trace1.initial_state
trace1.events[1]
trace1.events[2]

collect(trace0)
trace0.events[end].time