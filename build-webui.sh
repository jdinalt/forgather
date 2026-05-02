#!/usr/bin/env bash
# Build the forgather-server web UI.
#
# Usage:
#   ./build-webui.sh             # incremental build (skips npm install if up-to-date)
#   ./build-webui.sh --clean     # wipe dist/ and Vite's cache, then exit (no build)
#   ./build-webui.sh --install   # force `npm install` even if node_modules exists
#   ./build-webui.sh --watch     # run `vite` dev server (live reload, no static dist)
#
# --clean is "clean only" so it's a one-shot reset (e.g. before
# testing the Docker build's webui post-step from a known-empty
# state). To clean and then rebuild, run --clean and then re-run the
# script with no flags.
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

# --clean is a one-shot reset: wipe dist/ and Vite cache, then exit.
# Doesn't touch node_modules (use `rm -rf node_modules` if you want
# to reset that too — but `npm install` is the slow part, so skipping
# it on `--clean` keeps the next build fast).
if $clean; then
    echo "[build-webui] cleaning dist/ and Vite cache"
    rm -rf dist node_modules/.vite
    echo "[build-webui] done — re-run without --clean to rebuild"
    exit 0
fi

# `npm install` is the slow part; skip it when node_modules already exists
# and package-lock.json hasn't changed since the last install.
if $force_install || [[ ! -d node_modules ]] \
        || [[ package-lock.json -nt node_modules/.package-lock.json ]]; then
    echo "[build-webui] npm install"
    npm install
else
    echo "[build-webui] node_modules up-to-date — skipping npm install (use --install to force)"
fi

if $watch; then
    echo "[build-webui] starting Vite dev server (Ctrl+C to stop)"
    exec npm run dev
fi

echo "[build-webui] npm run build"
npm run build
echo "[build-webui] done — bundle in $WEBUI_DIR/dist/"
