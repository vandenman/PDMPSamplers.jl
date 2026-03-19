include("testsetup.jl")

import DifferentiationInterface as DI
import ForwardDiff

# TODO: formalize this idea a little bit
# could be a shell script?

function skip_test(test_name::String)
    key = "skip_$test_name"
    get(ENV, key, "") == "true"
end

function _normalize_test_arg(arg::String)
    name = basename(arg)
    startswith(name, "test-") || (name = "test-" * name)
    endswith(name, ".jl") || (name *= ".jl")
    return name
end

const tests = joinpath.(TEST_DIR,
    sort!(filter!(x -> startswith(x, "test-") && endswith(x, ".jl"), readdir(TEST_DIR)))
)

const requested_tests = isempty(ARGS) ? String[] : map(_normalize_test_arg, ARGS)
const tests_to_run = if isempty(requested_tests)
    tests
else
    selected = filter(t -> basename(t) in requested_tests, tests)
    isempty(selected) && error("No tests matched ARGS=$(ARGS). Expected names like test-gridthinning.jl")
    selected
end

@testset verbose = true "PDMPSamplers" begin
    for t in tests_to_run
        skip_test(basename(t)) && continue
        @testset "Test $(basename(t))" begin
            Random.seed!(345679)
            include(t)
        end
    end
end

print_test_summary()