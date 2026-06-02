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
import ForwardDiff
using Printf

const TEST_DIR = @__DIR__
const show_progress = isinteractive() && get(ENV, "CI", "") != "true"
const show_test_diagnostics = get(ENV, "CI", "") != "true"

function stable_test_seed(args...)
    h = UInt32(0x811c9dc5)
    for b in codeunits(repr(args))
        h = (h ⊻ UInt32(b)) * UInt32(0x01000193)
    end
    return Int(h % UInt32(typemax(Int32) - 1)) + 1
end

include(joinpath(TEST_DIR, "helper-gen_data.jl"))

const _PDMP_TEST_SETUP_DONE = true
end
