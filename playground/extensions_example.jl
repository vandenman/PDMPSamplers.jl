# Example usage of PDMPSamplers extensions
# This file is not run automatically - it requires optional dependencies

# === DynamicPPL Extension Example ===
# Uncomment to test (requires DynamicPPL and Mooncake)
#=
using PDMPSamplers
using DynamicPPL, Distributions
import Mooncake
import ADTypes

@model function normal_model(y)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0, 1), 0, Inf)
    for i in eachindex(y)
        y[i] ~ Normal(μ, σ)
    end
end

data = randn(10) .+ 2.0
model = normal_model(data)

# Create PDMPModel from DynamicPPL model
pdmp_model = PDMPModel(model, ADTypes.AutoMooncake(); needs_hvp=true)

# Sample
flow = ZigZag(pdmp_model.d)
alg = GridThinningStrategy()
x0 = randn(pdmp_model.d)
trace, stats = pdmp_sample(x0, flow, pdmp_model, alg, 0.0, 5_000.0)

println("DynamicPPL example completed")
println("Mean: ", mean(trace))
println("Std: ", std(trace))
=#

# === BridgeStan Extension Example ===
# Uncomment to test (requires BridgeStan and compiled Stan model)
#=
using PDMPSamplers
using BridgeStan

# Assuming you have a compiled Stan model
# sm = StanModel("path/to/model.so", "path/to/data.json")
# pdmp_model = PDMPModel(sm; needs_hvp=true)

# Or load directly from paths
# pdmp_model = PDMPModel(
#     "path/to/model.so",
#     "path/to/data.json";
#     needs_hvp=true
# )

# Sample
# flow = BouncyParticle(pdmp_model.d)
# alg = GridThinningStrategy()
# x0 = randn(pdmp_model.d)
# trace, stats = pdmp_sample(x0, flow, pdmp_model, alg, 0.0, 5_000.0)

# println("BridgeStan example completed")
=#
