using Test, PDMPSamplers
using Random
using LinearAlgebra

# following R conventions (I know) everything is sourced that starts with "test-" and ends with ".jl"
const test_dir = basename(pwd()) == "PDMPSamplers.jl" ? joinpath(pwd(), "test") : pwd()
const tests = joinpath.(test_dir,
    filter!(x->startswith(x, "test-") && endswith(x, ".jl"), readdir(test_dir))
)

const on_ci = haskey(ENV, "CI") ? ENV["CI"] == "true" : false

using Distributions
import PDMats # for MvTDist test
import DataFrames as DF
import GLM, StatsModels # testing separation in logistic regression
import MCMCDiagnosticTools
import Statistics
import LogExpFunctions

import DifferentiationInterface as DI
import Mooncake
import ForwardDiff

include("helper-gen_data.jl")

function skip_test(test_name::String)
    key = "skip_$test_name"
    if get(ENV, key, "") == "true"
        @info "Skipping $test_name"
        return true
    else
        return false
    end
end

@testset verbose = true "PDMPSamplers" begin

    for t in tests

        skip_test(basename(t)) && continue

        @testset "Test $t" begin
            Random.seed!(345679)
            include("$t")
        end
    end
end