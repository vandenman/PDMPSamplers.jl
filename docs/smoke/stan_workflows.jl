# Run the published Stan examples with short trajectories in either language.
# Usage: julia --project=docs docs/smoke/stan_workflows.jl --language julia
const DOCS = dirname(@__DIR__)
const EXAMPLES = (
    ("## A multivariate normal: Zig-Zag and two event clocks", "## Independent spike-and-slab prior", "normal"),
    ("## Independent spike-and-slab prior", "### Beta-binomial inclusion with independent slabs", "independent"),
    ("### Beta-binomial inclusion with independent slabs", "## Correlated slab and beta-binomial model prior", "beta_binomial"),
    ("## Logistic regression with variable selection", "## A Bayesian two-sample test", "logistic"),
    ("## A Bayesian two-sample test", "## Saving the result", "ttest"),
)

function example_code(page, heading, next_heading, language)
    section = split(split(page, heading; limit=2)[2], next_heading; limit=2)[1]
    label, fence = language == "r" ? ("R", "r") : ("Julia", "julia")
    block = match(Regex("== " * label * "\n\n```" * fence * "\n(.*?)\n```", "s"), section)
    isnothing(block) && error("Missing $label example: $heading")
    code = block.captures[1]
    if language == "r"
        code = replace(code, r"\bT = (?:10000|20000|100000)\b" => "T = 4")
    else
        code = replace(code, "10_000.0" => "4.0", "20_000.0" => "4.0", "100_000.0" => "4.0")
    end
    return code * "\n"
end

function main(args)
    length(args) == 2 && args[1] == "--language" && args[2] in ("julia", "r") ||
        error("Usage: julia --project=docs docs/smoke/stan_workflows.jl --language julia|r")
    language = args[2]
    page = read(joinpath(DOCS, "src", "examples", "stan-workflows.md"), String)
    mktempdir(; prefix="pdmp-stan-smoke-") do work
        for name in ("mvnormal.stan", "logistic_regression.stan",
                     "logistic_regression_data.json", "ttest.stan", "ttest_data.json")
            cp(joinpath(DOCS, "src", "public", "stan", name), joinpath(work, name))
        end
        blocks = String[]
        for (heading, next_heading, name) in EXAMPLES
            code = example_code(page, heading, next_heading, language)
            push!(blocks, language == "r" ?
                "cat(\"Running R $name\\n\")\nlocal({\n$code})\n" :
                "println(\"Running Julia $name\")\nmodule Smoke_$name\n$(code)end\n")
        end
        script = joinpath(work, language == "r" ? "stan_workflows.R" : "stan_workflows.jl")
        write(script, join(blocks, "\n"))
        cd(work) do
            rscript = get(ENV, "RSCRIPT", "Rscript")
            r_bootstrap = """
            args <- commandArgs(trailingOnly = TRUE)
            Sys.setenv(RENV_PROJECT = args[[1L]],
                       RENV_PATHS_LIBRARY = file.path(args[[1L]], "renv/library"))
            source(file.path(args[[1L]], "renv/activate.R"))
            if (!requireNamespace("PDMPSamplersR", quietly = TRUE)) {
                stop("Optional R smoke tests need PDMPSamplersR installed in docs/renv with renv::install()")
            }
            source(args[[2L]])
            """
            command = language == "r" ? `$rscript --vanilla -e $r_bootstrap $DOCS $script` :
                `$(Base.julia_cmd()) --project=$DOCS $script`
            run(command)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
