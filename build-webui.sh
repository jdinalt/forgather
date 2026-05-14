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
# Per-arch node_modules: npm only installs the platform-matching
# @rollup/rollup-<os>-<arch>-* optional native binary (the rest are filtered
# by os/cpu), so a node_modules populated on x86 is missing the arm64 native
# binary and vice versa. On a checkout shared between hosts of different
# arch (NFS, bind mounts, etc.) that breaks the second host's build.
#
# This script handles it transparently: before every run it inspects the
# current webui/node_modules/ and, if it was last installed for a different
# arch, renames it to webui/.node_modules-<that-arch>/ and (if available)
# rotates the matching arch's stashed install back into webui/node_modules/.
# npm install then operates on a real, arch-matching node_modules — no
# symlinks (npm's reify step replaces them with real dirs). Both
# .node_modules-*/ sibling directories are gitignored.
#
# Run from anywhere — the script resolves paths relative to its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$SCRIPT_DIR/tools/forgather_server/webui"
ARCH="$(uname -m)"
STASH_DIR=".node_modules-$ARCH"

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

# Map (uname -s, uname -m) to the Rollup native package npm should have
# installed for that host. Empty string = unknown platform.
rollup_native_pkg_for() {
    local os="$1" arch="$2"
    case "$os" in
        Linux)
            case "$arch" in
                x86_64)         echo "@rollup/rollup-linux-x64-gnu" ;;
                aarch64|arm64)  echo "@rollup/rollup-linux-arm64-gnu" ;;
                *)              echo "" ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                arm64)          echo "@rollup/rollup-darwin-arm64" ;;
                x86_64)         echo "@rollup/rollup-darwin-x64" ;;
                *)              echo "" ;;
            esac
            ;;
        *) echo "" ;;
    esac
}
EXPECTED_ROLLUP_PKG="$(rollup_native_pkg_for "$(uname -s)" "$ARCH")"

# Detect which arch a populated node_modules/ was installed for, by looking
# at which @rollup/rollup-<os>-<arch>-* native package is actually present
# (npm only installs the one matching the install host). Returns the
# `uname -m`-style arch, or empty string if we can't tell.
detect_installed_arch() {
    [[ -d node_modules/@rollup ]] || return 0
    if [[ -d node_modules/@rollup/rollup-linux-x64-gnu ]]; then echo "x86_64"; return; fi
    if [[ -d node_modules/@rollup/rollup-linux-arm64-gnu ]]; then echo "aarch64"; return; fi
    if [[ -d node_modules/@rollup/rollup-darwin-x64 ]]; then echo "x86_64-darwin"; return; fi
    if [[ -d node_modules/@rollup/rollup-darwin-arm64 ]]; then echo "arm64-darwin"; return; fi
}

# Migrate the previous symlink-based layout (if anyone is upgrading from
# an earlier version of this script) — collapse the symlink so the rest
# of this script can treat node_modules as a regular path.
if [[ -L node_modules ]]; then
    link_target="$(readlink node_modules)"
    rm node_modules
    if [[ -d "$link_target" ]]; then
        mv "$link_target" node_modules
    fi
fi

# Rotate per-arch stashes. If the currently-installed node_modules belongs
# to a different arch, swap it out and (if available) swap our arch's
# stashed copy in. After this block: node_modules/ is either absent (and
# `npm install` below will populate it) or belongs to $ARCH.
if [[ -d node_modules ]]; then
    installed_arch="$(detect_installed_arch || true)"
    if [[ -n "$installed_arch" && "$installed_arch" != "$ARCH" ]]; then
        stash_other=".node_modules-$installed_arch"
        echo "[build-webui] stashing $installed_arch install -> $stash_other/"
        rm -rf "$stash_other"
        mv node_modules "$stash_other"
    fi
fi
if [[ ! -e node_modules && -d "$STASH_DIR" ]]; then
    echo "[build-webui] restoring $ARCH install from $STASH_DIR/"
    mv "$STASH_DIR" node_modules
fi

# --clean is a one-shot reset: wipe dist/ and Vite cache, then exit.
# Doesn't touch node_modules (use `rm -rf node_modules .node_modules-*`
# if you want to reset that too — but `npm install` is the slow part,
# so skipping it on `--clean` keeps the next build fast).
if $clean; then
    echo "[build-webui] cleaning dist/ and Vite cache"
    rm -rf dist node_modules/.vite
    echo "[build-webui] done — re-run without --clean to rebuild"
    exit 0
fi

# `npm install` is the slow part; skip it when node_modules already exists
# and package-lock.json hasn't changed since the last install. Also force
# a repair if the arch-matching Rollup native binary is missing (catches
# manual node_modules copies between archs and interrupted prior installs).
need_install=false
if $force_install || [[ ! -d node_modules ]] \
        || [[ package-lock.json -nt node_modules/.package-lock.json ]]; then
    need_install=true
elif [[ -n "$EXPECTED_ROLLUP_PKG" && ! -d "node_modules/$EXPECTED_ROLLUP_PKG" ]]; then
    echo "[build-webui] $EXPECTED_ROLLUP_PKG missing — forcing reinstall"
    need_install=true
fi

if $need_install; then
    echo "[build-webui] npm install (arch=$ARCH)"
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
