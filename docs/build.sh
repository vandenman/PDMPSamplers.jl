#!/usr/bin/env bash
# Build and validate the shared site, using a checkout of PDMPSamplersR.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -gt 1 ]]; then
    echo "Usage: docs/build.sh [path/to/PDMPSamplersR]" >&2
    exit 2
fi
if [[ "${1:-}" == --help ]]; then
    echo "Usage: docs/build.sh [path/to/PDMPSamplersR]"
    exit 0
fi
r_source="${1:-${PDMPSAMPLERS_R_SOURCE:-$repo_dir/r-package}}"
if [[ ! -f "$r_source/DESCRIPTION" ]]; then
    echo "R package source not found: $r_source" >&2
    echo "Pass its checkout path, or set PDMPSAMPLERS_R_SOURCE." >&2
    exit 1
fi
"$rscript" --vanilla "$docs_dir/build_r_reference.R" "$r_source"
"$julia" --project="$docs_dir" "$docs_dir/make.jl"
"$rscript" --vanilla "$docs_dir/smoke/check_r_reference.R"
