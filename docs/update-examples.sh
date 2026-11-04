#!/usr/bin/env bash
# Regenerate saved output by executing the R examples. CI always runs this.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -gt 1 ]]; then
    echo "Usage: docs/update-examples.sh [path/to/PDMPSamplersR]" >&2
    exit 2
fi
if [[ "${1:-}" == --help ]]; then
    echo "Usage: docs/update-examples.sh [path/to/PDMPSamplersR]"
    exit 0
fi
r_source="${1:-${PDMPSAMPLERS_R_SOURCE:-$repo_dir/r-package}}"
if [[ ! -f "$r_source/DESCRIPTION" ]]; then
    echo "Pass the PDMPSamplersR checkout path, or set PDMPSAMPLERS_R_SOURCE." >&2
    exit 1
fi
export RENV_PROFILE=examples
export RENV_CONFIG_AUTO_SNAPSHOT=FALSE
export PDMPSAMPLERS_DOCS_JULIA_BINDIR
PDMPSAMPLERS_DOCS_JULIA_BINDIR="$("$julia" --startup-file=no -e 'print(Sys.BINDIR)')"
# Read environment files (including GitHub credentials), but skip startup R code.
"$rscript" --no-init-file --no-site-file --no-restore --no-save "$docs_dir/render_r_examples.R" "$r_source"
