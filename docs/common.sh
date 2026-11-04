#!/usr/bin/env bash
# Shared paths and executable overrides for the documentation scripts.
set -euo pipefail

docs_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$docs_dir")"
julia="${JULIA:-julia}"
rscript="${RSCRIPT:-Rscript}"
export RENV_PROJECT="$docs_dir"
export RENV_PATHS_LIBRARY="$docs_dir/renv/library"
export RENV_CONFIG_AUTO_SNAPSHOT=FALSE
