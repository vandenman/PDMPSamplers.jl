#!/usr/bin/env Rscript
# Check links from the built guides into pkgdown, and pkgdown's local links,
# assets and fragments. Run before publishing any build.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
docs <- dirname(dirname(normalizePath(script)))
Sys.setenv(RENV_PROJECT = docs,
           RENV_PATHS_LIBRARY = file.path(docs, "renv", "library"))
source(file.path(docs, "renv", "activate.R"))
if (!requireNamespace("xml2", quietly = TRUE)) stop("Run docs/instantiate.sh first")
build <- file.path(docs, "build")
roots <- list.dirs(build, recursive = FALSE, full.names = TRUE)
roots <- roots[grepl("^[0-9]+$", basename(roots))]
if (!length(roots)) stop("No rendered site found; run docs/make.jl first")

for (root in roots) {
  reference <- file.path(root, "r-api", "reference", "index.html")
  if (!file.exists(reference)) stop("Missing pkgdown reference: ", reference)
  if (!file.exists(file.path(root, "r-api", "search.json"))) {
    stop("Missing pkgdown search index in ", root)
  }
  portal <- file.path(root, "r", "reference.html")
  if (!file.exists(portal)) stop("Missing R reference guide: ", portal)
  parsed <- new.env(parent = emptyenv())
  html <- function(path) {
    if (!exists(path, parsed, inherits = FALSE)) {
      assign(path, xml2::read_html(path), parsed)
    }
    get(path, parsed, inherits = FALSE)
  }
  check_target <- function(source, link, pkgdown = FALSE) {
    if (!nzchar(link) || link == "#") return()
    if (grepl("^(https?:|mailto:|tel:|data:|javascript:|//)", link)) return()
    parts <- strsplit(link, "#", fixed = TRUE)[[1L]]
    path <- sub("\\?.*$", "", parts[[1L]])
    if (grepl("(^|/)r-api/", path)) {
      target <- file.path(root, "r-api", sub("^.*?r-api/", "", path))
    } else if (pkgdown && !startsWith(path, "/")) {
      target <- if (nzchar(path)) file.path(dirname(source), URLdecode(path)) else source
    } else {
      return()
    }
    if (dir.exists(target)) target <- file.path(target, "index.html")
    if (!file.exists(target)) stop("Broken local link in ", source, ": ", link)
    if (length(parts) > 1L && nzchar(parts[[2L]]) && grepl("\\.html$", target)) {
      ids <- xml2::xml_attr(xml2::xml_find_all(html(target), "//*[@id]"), "id")
      if (!URLdecode(parts[[2L]]) %in% ids) {
        stop("Missing fragment in ", source, ": ", link)
      }
    }
  }
  guides <- list.files(root, "\\.html$", recursive = TRUE, full.names = TRUE)
  guides <- guides[!startsWith(guides, file.path(root, "r-api", ""))]
  guide_links <- 0L
  for (source in guides) {
    nodes <- xml2::xml_find_all(html(source), "//a[@href]")
    nodes <- nodes[grepl("(^|/)r-api/", xml2::xml_attr(nodes, "href"))]
    # These static pages need a full navigation, rather than VitePress routing.
    if (any(is.na(xml2::xml_attr(nodes, "target")) |
            xml2::xml_attr(nodes, "target") != "_self")) {
      stop("Reference links must use target=\"_self\" in ", source)
    }
    links <- xml2::xml_attr(nodes, "href")
    guide_links <- guide_links + length(links)
    for (link in links) check_target(source, link)
  }
  if (!guide_links) stop("No guide links to the R reference found in ", root)
  pages <- list.files(file.path(root, "r-api"), "\\.html$", recursive = TRUE, full.names = TRUE)
  for (source in pages) {
    nodes <- xml2::xml_find_all(html(source), "//*[@href or @src]")
    links <- c(xml2::xml_attr(nodes, "href"), xml2::xml_attr(nodes, "src"))
    for (link in links[!is.na(links)]) check_target(source, link, pkgdown = TRUE)
  }
  search <- jsonlite::read_json(file.path(root, "r-api", "search.json"))
  for (entry in search) {
    if (is.null(entry) || is.null(entry$path)) next
    if (!is.character(entry$path) || length(entry$path) != 1L) {
      stop("Invalid search result path in ", root)
    }
    link <- entry$path
    if (!is.null(entry$id)) link <- paste0(link, "#", entry$id)
    check_target(reference, link, pkgdown = TRUE)
  }
  cat("Checked ", length(pages), " pkgdown pages and ", guide_links,
      " guide links in ", root, "\n", sep = "")
}
