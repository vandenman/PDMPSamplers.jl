using Documenter, DocumenterVitepress
using PDMPSamplers

makedocs(;
    sitename = "PDMPSamplers",
    authors = "Don van den Bergh",
    repo = "github.com/vandenman/PDMPSamplers.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "API" => "api.md",
        "R Package" => "r.md",
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
