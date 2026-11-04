#!/usr/bin/env bash
# Build the site and start VitePress; --no-build serves an existing build.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

if [[ "${1:-}" == --no-build ]]; then
    shift
    if [[ $# -ne 0 || ! -f "$docs_dir/build/.documenter/index.md" ]]; then
        echo "Build first with docs/build.sh [path/to/PDMPSamplersR]." >&2
        exit 1
    fi
else
    if [[ "${1:-}" == --help ]]; then
        echo "Usage: docs/preview.sh [path/to/PDMPSamplersR | --no-build]"
        exit 0
    fi
    "$docs_dir/build.sh" "$@"
fi
exec "$julia" --project="$docs_dir" -e '
    using DocumenterVitepress
    DocumenterVitepress.dev_docs(ARGS[1])
' "$docs_dir/build"
