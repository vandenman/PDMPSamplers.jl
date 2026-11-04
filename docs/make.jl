using Documenter, DocumenterVitepress
using PDMPSamplers

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
            "Sampling functions" => "r/reference/sampling.md",
            "Stan and subsampling functions" => "r/reference/stan-and-subsampling.md",
            "Prior and selection functions" => "r/reference/priors-and-selection.md",
            "Result functions" => "r/reference/results.md",
            "Setup and transform functions" => "r/reference/setup-and-transforms.md",
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

DocumenterVitepress.deploydocs(;
    repo = "github.com/vandenman/PDMPSamplers.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
