# using Pkg
# Pkg.activate(dirname(@__DIR__))
# Pkg.instantiate()

# # Add Coverage.jl to a temporary environment to avoid polling Project.toml
# using Pkg
# Pkg.activate(; temp=true)
# Pkg.add("Coverage")
# using Coverage

# # Run tests with coverage
# # We run the tests in a separate process to ensure clean state and proper coverage tracking
# cmd = `julia --project=$(dirname(@__DIR__)) --code-coverage=user test/runtests.jl`
# run(cmd)

# # Process coverage
# repo_path = dirname(@__DIR__)
# coverage = process_folder(repo_path)

# # Calculate coverage percentage
# covered_lines, total_lines = get_summary(coverage)
# percentage = covered_lines / total_lines * 100
# println("Coverage: $(percentage)%")

# # Clean up .cov files
# clean_folder(repo_path)

# # You can also generate an LCOV file if needed
# # LCOV.writefile("lcov.info", coverage)
