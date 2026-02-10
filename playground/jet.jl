import JET
using PDMPSamplers
using Distributions, LinearAlgebra
import Random
include("../test/helper-gen_data.jl")

pdmp_type = ZigZag
gradient_type = FullGradient
data_type = MvNormal
data_arg = (5, 1.0)
algorithm = GridThinningStrategy
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
d = data_arg[1]
flow = pdmp_type(inv(Symmetric(cov(D))), mean(D))

x0 = randn(d)
θ0 = PDMPSamplers.initialize_velocity(flow, d)
ξ0 = SkeletonPoint(x0, θ0)

grad = gradient_type == CoordinateWiseGradient ? CoordinateWiseGradient(∂fxᵢ) : FullGradient(∇f!)

model = PDMPModel(d, grad, ∇²f!)

out = similar(x0)
model.grad.f(out, x0)
model.hvp(x0, θ0)

trace0, stats0 = pdmp_sample(ξ0, flow, model, GridThinningStrategy(), 0.0, 5000., progress=show_progress)

JET.@report_opt pdmp_sample(ξ0, flow, model, GridThinningStrategy(), 0.0, 5000., progress=show_progress)

JET.@report_call pdmp_sample(ξ0, flow, model, GridThinningStrategy(), 0.0, 5000., progress=show_progress)

JET.report_package(PDMPSamplers)