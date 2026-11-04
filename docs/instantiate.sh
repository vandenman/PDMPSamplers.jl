#!/usr/bin/env bash
# Restore the locked R environment and instantiate the Julia docs environment.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -gt 0 ]]; then
    echo "Usage: docs/instantiate.sh"
    [[ $# -eq 1 && "$1" == --help ]] && exit 0
    exit 2
fi

(
    cd "$docs_dir"
    "$rscript" --vanilla -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'
)
"$julia" --project="$docs_dir" -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
'
