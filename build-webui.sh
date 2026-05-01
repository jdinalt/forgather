#!/usr/bin/env bash
# Build the forgather-server web UI.
#
# Usage:
#   ./build-webui.sh             # incremental build (skips npm install if up-to-date)
#   ./build-webui.sh --clean     # wipe dist/ and Vite's cache, then rebuild
#   ./build-webui.sh --install   # force `npm install` even if node_modules exists
#   ./build-webui.sh --watch     # run `vite` dev server (live reload, no static dist)
#
# Run from anywhere — the script resolves paths relative to its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$SCRIPT_DIR/tools/forgather_server/webui"

clean=false
force_install=false
watch=false
for arg in "$@"; do
    case "$arg" in
        --clean) clean=true ;;
        --install) force_install=true ;;
        --watch|--dev) watch=true ;;
        -h|--help)
            sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

if ! command -v npm >/dev/null 2>&1; then
    echo "error: npm not found on PATH — install Node.js first" >&2
    exit 1
fi

cd "$WEBUI_DIR"

# `npm install` is the slow part; skip it when node_modules already exists
# and package-lock.json hasn't changed since the last install.
if $force_install || [[ ! -d node_modules ]] \
        || [[ package-lock.json -nt node_modules/.package-lock.json ]]; then
    echo "[build-webui] npm install"
    npm install
else
    echo "[build-webui] node_modules up-to-date — skipping npm install (use --install to force)"
fi

if $clean; then
    echo "[build-webui] cleaning dist/ and Vite cache"
    rm -rf dist node_modules/.vite
fi

if $watch; then
    echo "[build-webui] starting Vite dev server (Ctrl+C to stop)"
    exec npm run dev
fi

echo "[build-webui] npm run build"
npm run build
echo "[build-webui] done — bundle in $WEBUI_DIR/dist/"
