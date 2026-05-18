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
# Per-platform node_modules: npm only installs the platform-matching
# @rollup/rollup-<os>-<arch>-{gnu,musl,...} optional native binary (the rest
# are filtered by os/cpu), so a node_modules populated on linux-x86_64 is
# missing the linux-aarch64 or darwin-arm64 native binary and vice versa.
# On a checkout shared between hosts of different platform (NFS, bind
# mounts, etc.) that breaks the second host's build.
#
# This script handles it transparently: before every run it inspects the
# current webui/node_modules/ and, if it was last installed for a different
# platform, renames it to webui/.node_modules-<that-platform>/ and (if
# available) rotates the matching platform's parked install back into
# webui/node_modules/. npm install then operates on a real, platform-
# matching node_modules — no symlinks (npm's reify step replaces them
# with real dirs). All .node_modules-*/ sibling directories are gitignored.
#
# Platform tag format: `<os>[-musl]-<arch>` (e.g. linux-x86_64,
# linux-aarch64, linux-musl-aarch64, darwin-aarch64). The libc tag is
# only emitted on Linux/musl; glibc Linux and Darwin use the bare form.
#
# Run from anywhere — the script resolves paths relative to its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$SCRIPT_DIR/tools/forgather_server/webui"

# Normalize uname -m output (x86_64/amd64 -> x86_64, aarch64/arm64 -> aarch64)
# so the platform tag is identical whether `uname -m` says `arm64` (Darwin)
# or `aarch64` (Linux).
canonicalize_arch() {
    case "$1" in
        x86_64|amd64)   echo "x86_64" ;;
        aarch64|arm64)  echo "aarch64" ;;
        *)              echo "$1" ;;
    esac
}

# Linux can be glibc or musl; the Rollup binary that npm installs differs
# (-gnu vs -musl) and they are not interchangeable. Emit a `-musl` libc tag
# only when we can confirm musl; otherwise empty (glibc is the default and
# doesn't need a tag).
detect_libc_tag() {
    case "$(uname -s)" in
        Linux)
            if [[ -f /lib/ld-musl-x86_64.so.1 || -f /lib/ld-musl-aarch64.so.1 ]] \
                    || ldd --version 2>&1 | grep -qi musl; then
                echo "-musl"
            fi
            ;;
    esac
}

OS_TAG="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH_TAG="$(canonicalize_arch "$(uname -m)")"
LIBC_TAG="$(detect_libc_tag)"
PLATFORM_TAG="${OS_TAG}${LIBC_TAG}-${ARCH_TAG}"
PARKED_DIR=".node_modules-$PLATFORM_TAG"

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

# Map a platform tag to the Rollup native package npm should have installed
# for it. Empty string = platform tag we don't know about (in practice,
# Windows or an unrecognised arch — npm install will still run, we just
# can't perform the binary-presence sanity check or rotation).
rollup_native_pkg_for() {
    case "$1" in
        linux-x86_64)        echo "@rollup/rollup-linux-x64-gnu" ;;
        linux-aarch64)       echo "@rollup/rollup-linux-arm64-gnu" ;;
        linux-musl-x86_64)   echo "@rollup/rollup-linux-x64-musl" ;;
        linux-musl-aarch64)  echo "@rollup/rollup-linux-arm64-musl" ;;
        darwin-x86_64)       echo "@rollup/rollup-darwin-x64" ;;
        darwin-aarch64)      echo "@rollup/rollup-darwin-arm64" ;;
        *)                   echo "" ;;
    esac
}
EXPECTED_ROLLUP_PKG="$(rollup_native_pkg_for "$PLATFORM_TAG")"

# Detect which platform a populated node_modules/ was installed for, by
# looking at which @rollup/rollup-* native package is actually present
# (npm only installs the one matching the install host). Returns a
# platform tag in the same format as $PLATFORM_TAG, or empty string if we
# can't tell.
detect_installed_platform() {
    [[ -d node_modules/@rollup ]] || return 0
    if [[ -d node_modules/@rollup/rollup-linux-x64-gnu ]];     then echo "linux-x86_64";       return 0; fi
    if [[ -d node_modules/@rollup/rollup-linux-arm64-gnu ]];   then echo "linux-aarch64";      return 0; fi
    if [[ -d node_modules/@rollup/rollup-linux-x64-musl ]];    then echo "linux-musl-x86_64";  return 0; fi
    if [[ -d node_modules/@rollup/rollup-linux-arm64-musl ]];  then echo "linux-musl-aarch64"; return 0; fi
    if [[ -d node_modules/@rollup/rollup-darwin-x64 ]];        then echo "darwin-x86_64";      return 0; fi
    if [[ -d node_modules/@rollup/rollup-darwin-arm64 ]];      then echo "darwin-aarch64";     return 0; fi
    return 0
}

# Migrate the symlink-based layout used by an early version of this script.
# `cd "$WEBUI_DIR"` above made CWD the webui dir, so a relative symlink
# target like `.node_modules-x86_64` resolves correctly here.
if [[ -L node_modules ]]; then
    link_target="$(readlink node_modules)"
    rm node_modules
    if [[ -d "$link_target" ]]; then
        mv "$link_target" node_modules
    fi
fi

# Migrate the arch-only naming (.node_modules-<arch>/) used by the first
# published version of this layout. We can only safely rename Linux ones —
# bare `arm64` was ambiguous between Linux and Darwin in the previous
# scheme, and a Darwin entry on a Linux host (or vice versa) isn't a
# meaningful rename, so leave those alone.
if [[ "$OS_TAG" == "linux" && -z "$LIBC_TAG" ]]; then
    for old_name in .node_modules-x86_64 .node_modules-aarch64; do
        new_name=".node_modules-linux${old_name#.node_modules}"
        if [[ -d "$old_name" && ! -e "$new_name" ]]; then
            mv "$old_name" "$new_name"
        fi
    done
fi

# Rotate per-platform installs by renaming. If the currently-active
# node_modules/ belongs to a different platform, rename it out to its
# sibling .node_modules-<that-platform>/ and (if a matching sibling for
# our platform exists) rename that one into node_modules/. After this
# block: node_modules/ is either absent (and `npm install` below will
# populate it) or belongs to $PLATFORM_TAG. No git stash, no symlinks —
# just plain mv.
if [[ -d node_modules ]]; then
    installed_platform="$(detect_installed_platform)"
    if [[ -n "$installed_platform" && "$installed_platform" != "$PLATFORM_TAG" ]]; then
        other_sibling=".node_modules-$installed_platform"
        echo "[build-webui] renaming node_modules -> $other_sibling/ ($installed_platform)"
        rm -rf "$other_sibling"
        mv node_modules "$other_sibling"
    fi
fi
if [[ ! -e node_modules && -d "$PARKED_DIR" ]]; then
    echo "[build-webui] renaming $PARKED_DIR/ -> node_modules ($PLATFORM_TAG)"
    mv "$PARKED_DIR" node_modules
fi

# --clean is a one-shot reset: wipe dist/ and the Vite cache of the active
# node_modules plus every parked sibling so a subsequent build is fully
# clean. Doesn't touch node_modules itself (use
# `rm -rf node_modules .node_modules-*` if you want to reset that too —
# but `npm install` is the slow part, so skipping it on `--clean` keeps
# the next build fast).
if $clean; then
    echo "[build-webui] cleaning dist/ and Vite cache(s)"
    rm -rf dist node_modules/.vite .node_modules-*/.vite
    echo "[build-webui] done — re-run without --clean to rebuild"
    exit 0
fi

# `npm install` is the slow part; skip it when node_modules already exists
# and package-lock.json hasn't changed since the last install. Also force
# a repair if the platform-matching Rollup native binary is missing —
# catches manual node_modules copies between platforms and interrupted
# prior installs. The Rollup native binary is only used by `vite build`,
# not by the dev server, so skip the check entirely in --watch mode.
need_install=false
if $force_install || [[ ! -d node_modules ]] \
        || [[ package-lock.json -nt node_modules/.package-lock.json ]]; then
    need_install=true
elif ! $watch && [[ -n "$EXPECTED_ROLLUP_PKG" && ! -d "node_modules/$EXPECTED_ROLLUP_PKG" ]]; then
    echo "[build-webui] $EXPECTED_ROLLUP_PKG missing — forcing reinstall"
    need_install=true
fi

if $need_install; then
    echo "[build-webui] npm install (platform=$PLATFORM_TAG)"
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
