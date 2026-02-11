#!/usr/bin/env julia
#
# Run individual test files from the PDMPSamplers test suite.
#
# Usage (from project root):
#   julia --startup-file=no test/run.jl test-flow_properties.jl
#   julia --startup-file=no test/run.jl flow_properties
#   julia --startup-file=no test/run.jl                         # runs all tests

import Pkg
Pkg.activate(@__DIR__)
push!(LOAD_PATH, dirname(@__DIR__))

include(joinpath(@__DIR__, "testsetup.jl"))

import DifferentiationInterface as DI
import ForwardDiff

if isempty(ARGS)
    tests = sort!(filter!(x -> startswith(x, "test-") && endswith(x, ".jl"), readdir(@__DIR__)))
else
    tests = map(ARGS) do arg
        name = arg
        endswith(name, ".jl") || (name *= ".jl")
        startswith(name, "test-") || (name = "test-" * name)
        name
    end
end

@testset verbose = true "PDMPSamplers" begin
    for t in tests
        path = joinpath(@__DIR__, t)
        @testset "Test $t" begin
            Random.seed!(345679)
            include(path)
        end
    end
end
