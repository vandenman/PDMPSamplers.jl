#!/usr/bin/env Rscript

# Generate the R reference in this site from PDMPSamplersR's canonical Rd files.
# Usage: Rscript docs/generate_r_reference.R /path/to/PDMPSamplersR [--check]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L ||
    (length(args) == 2L && args[[2L]] != "--check")) {
  stop("Usage: Rscript docs/generate_r_reference.R /path/to/PDMPSamplersR [--check]")
}
check <- length(args) == 2L
r_repo <- normalizePath(args[[1L]], mustWork = TRUE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) != 1L) stop("Run this generator with Rscript")
docs <- dirname(normalizePath(sub("^--file=", "", script_arg)))
output <- file.path(docs, "src", "r")
man <- file.path(r_repo, "man")
namespace <- file.path(r_repo, "NAMESPACE")
if (!dir.exists(man) || !file.exists(namespace)) stop("Not a PDMPSamplersR checkout: ", r_repo)

# Each Rd page is included once. A page may document several exported aliases.
groups <- list(
  "Sampling" = c("pdmp_sample", "pdmp_sample_from_stanmodel", "brm_pdmp",
                 "sampler_control", "stan_compile_options",
                 "support_boundary_control", "online_summary_options"),
  "Stan and subsampling" = c("stan_subsampling", "stan_subsampling_diagnostics",
                             "position_anchor_bank", "write_stan_json"),
  "Priors and selection" = c("bernoulli", "betabernoulli",
                             "dependent_slab_priors", "inclusion_probs",
                             "inclusion_prob", "inclusion_bf",
                             "model_averaged_mean", "median_probability_model",
                             "check_inclusion_stability", "plot_inclusion"),
  "Results" = c("mean.pdmp_result", "median.pdmp_result", "var", "sd",
                "cov", "cor", "quantile.pdmp_result", "ess", "cdf",
                "discretize", "materialize", "read_online_summaries"),
  "Setup and transforms" = c("pdmpsamplers_setup", "pdmpsamplers_status",
                             "pdmpsamplers_update", "use_local_pdmpsamplers",
                             "check_for_julia_setup", "cleanup_worker_state",
                             "identity_transform", "lower_transform",
                             "upper_transform", "double_transform")
)
slugs <- c("sampling", "stan-and-subsampling", "priors-and-selection",
           "results", "setup-and-transforms")
stopifnot(length(groups) == length(slugs),
          !anyDuplicated(unlist(groups)))

rd_files <- list.files(man, "\\.Rd$", full.names = TRUE)
aliases <- lapply(rd_files, function(path) {
  lines <- readLines(path, warn = FALSE)
  sub("^\\\\alias\\{(.*)\\}$", "\\1", grep("^\\\\alias\\{", lines, value = TRUE))
})
names(aliases) <- sub("\\.Rd$", "", basename(rd_files))
exported <- sub("^export\\(([^)]+)\\)$", "\\1",
                grep("^export\\(", readLines(namespace), value = TRUE))
chosen <- unlist(groups)
if (!all(chosen %in% names(aliases))) {
  stop("Missing Rd pages: ", paste(setdiff(chosen, names(aliases)), collapse = ", "))
}
covered <- unique(unlist(aliases[chosen], use.names = FALSE))
if (!all(exported %in% covered)) {
  stop("Exported functions without reference pages: ",
       paste(setdiff(exported, covered), collapse = ", "))
}

rd_body <- function(path) {
  html_file <- tempfile(fileext = ".html")
  on.exit(unlink(html_file))
  tools::Rd2HTML(tools::parse_Rd(path), out = html_file)
  html <- paste(readLines(html_file, warn = FALSE), collapse = "\n")
  html <- sub("(?s)^.*?<main>", "", html, perl = TRUE)
  html <- sub("(?s)</main>.*$", "", html, perl = TRUE)
  html <- sub("(?s)^\\s*<table[^>]*>.*?</table>\\s*", "", html, perl = TRUE)
  html <- sub("(?s)^\\s*<h2>.*?</h2>\\s*", "", html, perl = TRUE)
  trimws(html)
}

outputs <- list()
redirects <- character()
index <- c("# R function reference", "",
           "This reference is generated from the help files in PDMPSamplersR.",
           "Run `?PDMPSamplersR::function_name` in R for the same help locally.", "")
for (i in seq_along(groups)) {
  group <- groups[[i]]
  slug <- slugs[[i]]
  title <- names(groups)[[i]]
  page <- c(paste0("# ", title), "",
            "[R function reference](../reference.md)", "",
            "Generated from the PDMPSamplersR help files. The R package is the source of truth.", "")
  links <- character()
  for (name in group) {
    path <- file.path(man, paste0(name, ".Rd"))
    anchor <- paste0("r-", name)
    page <- c(page, "```@raw html", sprintf('<span id="%s"></span>', anchor),
              "```", "", paste0("## `", name, "`"), "")
    public_aliases <- intersect(aliases[[name]], exported)
    if (!length(public_aliases)) public_aliases <- name
    if (length(public_aliases) > 1L) {
      page <- c(page, paste0("Also documents: ",
                             paste(sprintf("`%s`", setdiff(public_aliases, name)),
                                   collapse = ", "), "."), "")
    }
    page <- c(page, "```@raw html", rd_body(path), "```", "")
    links <- c(links, sprintf("- [`%s`](reference/%s.md#%s)",
                             public_aliases, slug, anchor))
    redirect_names <- unique(c(name, aliases[[name]]))
    redirects[redirect_names] <- paste0(slug, "#", anchor)
  }
  outputs[[file.path(output, "reference", paste0(slug, ".md"))]] <-
    paste(page, collapse = "\n")
  index <- c(index, paste0("## [", title, "](reference/", slug, ".md)"),
             "", links, "")
}
outputs[[file.path(output, "reference.md")]] <- paste(index, collapse = "\n")
redirects <- redirects[order(names(redirects))]
map <- paste0("{\n", paste(sprintf('  "%s": "%s"', names(redirects), redirects),
                          collapse = ",\n"), "\n}")
outputs[[file.path(docs, "src", "public", "r", "reference-map.json")]] <- map

for (path in names(outputs)) {
  content <- paste0(outputs[[path]], "\n")
  if (check) {
    if (!file.exists(path) || !identical(readChar(path, file.info(path)$size), content)) {
      stop("R reference is out of date: ", path)
    }
  } else {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    writeChar(content, path, eos = NULL)
  }
}
cat(if (check) "R reference is current\n" else "Generated R reference\n")
