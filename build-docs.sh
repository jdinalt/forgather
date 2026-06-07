#!/usr/bin/env bash
# Build the pre-rendered API-docs cache (docs/.built/) the webui Docs viewer
# reads from. This expands mkdocstrings-style ::: directives into markdown.
#
# This is independent of the webui SPA build (./build-webui.sh) — the two are
# separate artifacts. The Docs viewer falls back to raw markdown when the
# cache is absent, so this step is optional.
#
# Usage:
#   ./build-docs.sh            # incremental build of docs/.built/
#   ./build-docs.sh --clean    # remove docs/.built/ first, then rebuild
#   ./build-docs.sh --check    # exit non-zero if anything is stale (CI); no write
#   ./build-docs.sh -q         # quiet (errors + summary only)
#
# Any flags are forwarded to `forgather docs build`. Run from anywhere — the
# script resolves the repo root relative to its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v forgather >/dev/null 2>&1; then
    echo "error: forgather not found on PATH — install/activate the env first" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
echo "[build-docs] forgather docs build ${*:-(incremental)}"
exec forgather docs build "$@"
