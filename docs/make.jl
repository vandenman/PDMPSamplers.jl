using Documenter, DocumenterVitepress
using PDMPSamplers

include("check_example_outputs.jl")
check_r_example_outputs()

isfile(joinpath(@__DIR__, "src", "public", "r-api", "reference", "index.html")) ||
    error("Build the R reference first: Rscript docs/build_r_reference.R /path/to/PDMPSamplersR")

makedocs(;
    sitename = "PDMPSamplers",
    authors = "Don van den Bergh",
    repo = "github.com/vandenman/PDMPSamplers.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "What are PDMPs?" => "what-are-pdmps.md",
        "Variable Selection" => "variable-selection.md",
        "Subsampling" => "subsampling.md",
        "Dependent Slabs" => "dependent-slabs.md",
        "Choosing a Sampler" => "choosing-a-sampler.md",
        "Using Julia" => [
            "Stan models" => "julia/stan.md",
            "Turing.jl models" => "julia/turing.md",
        ],
        "Worked Examples" => [
            "Stan targets and variable selection" => "examples/stan-workflows.md",
            "RI-CLPM with dependent slabs" => "examples/riclpm.md",
            "Ordinal MRF subsampling" => "examples/omrf.md",
        ],
        "Using R" => [
            "Overview" => "r.md",
            "Getting started" => "r/getting-started.md",
            "Stan models" => "r/stan.md",
            "brms models" => "r/brms.md",
            "Results" => "r/results.md",
            "Function reference" => "r/reference.md",
        ],
        "API" => "api.md",
        "Extension Interface" => "extension-interface.md",
    ],
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/vandenman/PDMPSamplers.jl",
        devbranch = "main",
        devurl = "dev",
    ),
)
