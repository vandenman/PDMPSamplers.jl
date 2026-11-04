using SHA
using Markdown

const DOCS_R_OUTPUT_DIR = joinpath(@__DIR__, "examples", "output")

function r_example_output(slug)
    lines = readlines(joinpath(DOCS_R_OUTPUT_DIR, "getting-started-r-$slug.md"))
    # The first line records provenance for validation, not for the website.
    return Markdown.parse(join(lines[3:end], "\n"))
end

# Saved output may be reused locally, but it must belong to the displayed code.
function check_r_example_outputs()
    page = read(joinpath(@__DIR__, "src", "getting-started.md"), String)
    for (label, slug) in (("manual", "manual"), ("Stan", "stan"), ("brms", "brms"))
        block = match(Regex("== R - " * label * "\\n\\n```r\\n(.*?)\\n```", "s"), page)
        isnothing(block) && error("Missing R example: $label")
        fingerprint = bytes2hex(sha256(block.captures[1]))
        output = joinpath(DOCS_R_OUTPUT_DIR, "getting-started-r-$slug.md")
        isfile(output) && occursin("source-sha256: $fingerprint", read(output, String)) ||
            error("R example output is missing or stale ($label). Run docs/update-examples.sh /path/to/PDMPSamplersR")
    end
end
