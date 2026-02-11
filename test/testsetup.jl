if !@isdefined(_PDMP_TEST_SETUP_DONE)

using Test
using PDMPSamplers
using Random
using LinearAlgebra
using Distributions
using Statistics

import PDMats
import MCMCDiagnosticTools
import StatsBase
import LogExpFunctions

const TEST_DIR = @__DIR__
const show_progress = get(ENV, "CI", "") != "true"

include(joinpath(TEST_DIR, "helper-gen_data.jl"))

const _PDMP_TEST_SETUP_DONE = true
end
