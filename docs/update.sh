#!/usr/bin/env bash
# Intentionally update documentation dependencies and the R lockfile.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -gt 0 ]]; then
    echo "Usage: docs/update.sh"
    [[ $# -eq 1 && "$1" == --help ]] && exit 0
    exit 2
fi

"$docs_dir/instantiate.sh"
(
    cd "$docs_dir"
    "$rscript" --vanilla -e '
        source("renv/activate.R")
        renv::update(prompt = FALSE)
        renv::snapshot(prompt = FALSE)
    '
)
"$julia" --project="$docs_dir" -e 'using Pkg; Pkg.update()'
