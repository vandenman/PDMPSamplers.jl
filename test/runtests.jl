include("testsetup.jl")

import DifferentiationInterface as DI
import ForwardDiff

function skip_test(test_name::String)
    key = "skip_$test_name"
    get(ENV, key, "") == "true"
end

const tests = joinpath.(TEST_DIR,
    sort!(filter!(x -> startswith(x, "test-") && endswith(x, ".jl"), readdir(TEST_DIR)))
)

@testset verbose = true "PDMPSamplers" begin
    for t in tests
        skip_test(basename(t)) && continue
        @testset "Test $(basename(t))" begin
            Random.seed!(345679)
            include(t)
        end
    end
end

print_test_summary()