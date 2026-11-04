#!/usr/bin/env Rscript

# Render the canonical R help with pkgdown, then let VitePress copy it as
# static content into every version of the shared site.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: Rscript docs/build_r_reference.R /path/to/PDMPSamplersR")
}
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
docs <- dirname(normalizePath(script))
Sys.setenv(RENV_PROJECT = docs,
           RENV_PATHS_LIBRARY = file.path(docs, "renv", "library"))
source(file.path(docs, "renv", "activate.R"))
if (!requireNamespace("pkgdown", quietly = TRUE)) {
  stop("Run docs/instantiate.sh to restore the R documentation environment")
}
r_repo <- normalizePath(args[[1L]], mustWork = TRUE)
output <- file.path(docs, "src", "public", "r-api")

# Inherit the R package's reference grouping, but keep navigation within this
# site. Relative URLs work for local previews, dev, releases and PR previews.
pkg <- pkgdown::as_pkgdown(r_repo, override = list(
  destination = output,
  url = NULL,
  development = list(mode = "release"),
  template = list(bootstrap = 5),
  navbar = list(
    structure = list(left = c("guides", "reference"), right = c("search", "github")),
    components = list(
      guides = list(text = "R guides", href = "../r.html"),
      github = list(
        icon = "fa-github",
        href = "https://github.com/vandenman/PDMPSamplersR",
        `aria-label` = "R package source"
      )
    )
  )
))

# Start afresh so removed topics cannot linger in the published reference.
if (dir.exists(output)) unlink(output, recursive = TRUE)
pkgdown::init_site(pkg)
pkgdown::build_reference(pkg, lazy = FALSE, examples = FALSE, preview = FALSE)
pkgdown::build_search(pkg)

# The search widget lives on reference/*.html. Keep its result URLs relative
# too, so searches stay inside the same version of the shared site.
search_file <- file.path(output, "search.json")
search <- jsonlite::read_json(search_file)
search <- Filter(function(entry) is.character(entry$path) && length(entry$path) == 1L, search)
search <- lapply(search, function(entry) {
  entry$path <- sub("^/", "../", entry$path)
  entry
})
jsonlite::write_json(search, search_file, auto_unbox = TRUE, null = "null")

# pkgdown's brand links to its root; the guides provide that landing page.
writeLines(c(
  '<!doctype html><html lang="en"><head><meta charset="utf-8">',
  '<meta http-equiv="refresh" content="0;url=../r/reference.html">',
  '<title>R function reference</title></head><body>',
  '<a href="../r/reference.html">R function reference</a></body></html>'
), file.path(output, "index.html"))
cat("Built pkgdown reference in ", output, "\n", sep = "")
