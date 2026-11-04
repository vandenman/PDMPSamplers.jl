using DocumenterVitepress

# Validate in the deployment entry point too, so a manual deployment cannot
# bypass the same check that CI runs after building.
rscript = get(ENV, "RSCRIPT", "Rscript")
run(`$rscript --vanilla $(joinpath(@__DIR__, "smoke", "check_r_reference.R"))`)

DocumenterVitepress.deploydocs(;
    root = @__DIR__,
    repo = "github.com/vandenman/PDMPSamplers.jl.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
