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
# Per-arch node_modules: this script keeps webui/node_modules as a symlink
# to webui/.node_modules-$(uname -m) so x86_64 and aarch64 hosts sharing
# the same repo (e.g. over NFS) don't trample each other's native binaries.
# Both .node_modules-*/ directories are gitignored.
#
# Run from anywhere — the script resolves paths relative to its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$SCRIPT_DIR/tools/forgather_server/webui"
ARCH="$(uname -m)"
REAL_NM_NAME=".node_modules-$ARCH"

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

# Point node_modules at the per-arch real directory.
# npm only installs the platform-matching @rollup/rollup-linux-* optional
# native dep (the rest are filtered by os/cpu), so a node_modules populated
# on x86 is useless to an aarch64 host and vice versa. Keeping each arch's
# install in its own sibling dir avoids the conflict on shared repos.
mkdir -p "$REAL_NM_NAME"
if [[ -L node_modules ]]; then
    current_target="$(readlink node_modules)"
    if [[ "$current_target" != "$REAL_NM_NAME" ]]; then
        echo "[build-webui] repointing node_modules symlink: $current_target -> $REAL_NM_NAME"
        rm node_modules
        ln -s "$REAL_NM_NAME" node_modules
    fi
elif [[ -d node_modules ]]; then
    echo "[build-webui] migrating existing node_modules/ into $REAL_NM_NAME/ for per-arch isolation"
    rm -rf "$REAL_NM_NAME"
    mv node_modules "$REAL_NM_NAME"
    ln -s "$REAL_NM_NAME" node_modules
elif [[ ! -e node_modules ]]; then
    ln -s "$REAL_NM_NAME" node_modules
fi

# --clean is a one-shot reset: wipe dist/ and Vite cache, then exit.
# Doesn't touch node_modules (use `rm -rf .node_modules-*` if you want
# to reset that too — but `npm install` is the slow part, so skipping
# it on `--clean` keeps the next build fast).
if $clean; then
    echo "[build-webui] cleaning dist/ and Vite cache"
    rm -rf dist node_modules/.vite
    echo "[build-webui] done — re-run without --clean to rebuild"
    exit 0
fi

# Map (uname -s, uname -m) to the Rollup native package npm should have
# installed for this host. Empty string = unknown platform, skip the check.
expected_rollup_native_pkg() {
    local os
    os="$(uname -s)"
    case "$os" in
        Linux)
            case "$ARCH" in
                x86_64)         echo "@rollup/rollup-linux-x64-gnu" ;;
                aarch64|arm64)  echo "@rollup/rollup-linux-arm64-gnu" ;;
                *)              echo "" ;;
            esac
            ;;
        Darwin)
            case "$ARCH" in
                arm64)          echo "@rollup/rollup-darwin-arm64" ;;
                x86_64)         echo "@rollup/rollup-darwin-x64" ;;
                *)              echo "" ;;
            esac
            ;;
        *) echo "" ;;
    esac
}
ROLLUP_NATIVE_PKG="$(expected_rollup_native_pkg)"

# `npm install` is the slow part; skip it when node_modules already exists
# and package-lock.json hasn't changed since the last install.
need_install=false
if $force_install || [[ ! -d node_modules ]] \
        || [[ package-lock.json -nt node_modules/.package-lock.json ]]; then
    need_install=true
elif [[ -n "$ROLLUP_NATIVE_PKG" && ! -d "node_modules/$ROLLUP_NATIVE_PKG" ]]; then
    # node_modules looks fresh but the arch-matching Rollup binary is missing
    # (e.g. someone copied a node_modules dir across archs, or the per-arch
    # dir was created empty by a prior interrupted install). Force a repair.
    echo "[build-webui] expected $ROLLUP_NATIVE_PKG not installed for arch $ARCH — forcing reinstall"
    need_install=true
fi

if $need_install; then
    echo "[build-webui] npm install (into $REAL_NM_NAME/)"
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
