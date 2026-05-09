#!/bin/bash
# Build the Forgather development image.
#
# The image is now distributable: the in-container user is fixed at
# uid/gid 1000 and the entrypoint remaps to PUID/PGID via gosu at
# container start. There are no host-user build args to pass.
#
# After the image build succeeds we run ./build-webui.sh inside a
# transient container against the host clone, so the SPA dist/ lands
# in the user's tree and the Forgather server's web UI works the
# moment they start it. Skip this post-step with SKIP_WEBUI_BUILD=1.
#
# Usage:
#   docker/build.sh                    # tag: forgather-dev:latest
#   docker/build.sh my-tag             # custom tag
#   docker/build.sh --claude           # also bake in Claude Code (npm global)
#   docker/build.sh -- --no-cache      # pass extra args to docker build
#
# Flags can be combined with the tag:
#   docker/build.sh forgather-dev:claude --claude
#   docker/build.sh --claude my-tag -- --no-cache
#
# Skip the webui post-step (e.g. you'll iterate on the SPA via
# `npm run dev` instead of the static dist/):
#
#   SKIP_WEBUI_BUILD=1 docker/build.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ``--help`` MUST short-circuit before the docker invocation. Without
# this, ``--help`` falls through into the positional-argument logic
# below: TAG="--help" gets reset to forgather-dev:latest but never
# shifted off, then ``--help`` ends up in the ``docker build`` argv.
# ``docker build --help`` succeeds with rc=0 (printing its own help),
# which lets the rest of the script — including the post-build
# ``./build-webui.sh`` step — keep running.
for tok in "$@"; do
    case "${tok}" in
        -h|--help)
            sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

# Pre-parse our own flags out of argv so they compose with the
# positional TAG and the ``--`` passthrough cleanly.
INSTALL_CLAUDE=0
ARGV=()
for tok in "$@"; do
    case "${tok}" in
        --claude) INSTALL_CLAUDE=1 ;;
        *)        ARGV+=("${tok}") ;;
    esac
done
set -- "${ARGV[@]}"

TAG="${1:-forgather-dev:latest}"
# Only consume $1 as TAG if it's not a docker passthrough flag.
if [[ "${TAG}" == --* ]] || [[ "${TAG}" == -* ]]; then
    TAG="forgather-dev:latest"
else
    shift || true
fi
# Drop a leading "--" separator so callers can pass extra docker args:
#   docker/build.sh forgather-dev:latest -- --progress=plain
if [[ "${1:-}" == "--" ]]; then shift; fi

echo "Building ${TAG}"
if [[ "${INSTALL_CLAUDE}" = "1" ]]; then
    echo "  --claude: also installing Claude Code (npm global)"
fi

docker build \
    -t "${TAG}" \
    --build-arg "INSTALL_CLAUDE=${INSTALL_CLAUDE}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "$@" \
    "${REPO_ROOT}"

if [[ -n "${SKIP_WEBUI_BUILD:-}" ]]; then
    echo "[build.sh] SKIP_WEBUI_BUILD set; not running ./build-webui.sh"
    exit 0
fi

echo
echo "[build.sh] running ./build-webui.sh in the just-built image"
echo "[build.sh] (one-time; skip with SKIP_WEBUI_BUILD=1)"

# The image's entrypoint runs as root and would gosu-drop to the
# in-container `dev` user, but we want the build artifacts owned by
# the host user so the dist/ files in the bind-mounted clone are
# writable host-side. Run the throw-away container with --user set
# to the host UID/GID and override the entrypoint to bash directly,
# bypassing the entrypoint's editable-install + gosu-drop logic.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

docker run --rm \
    --user "${HOST_UID}:${HOST_GID}" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${REPO_ROOT}" \
    --entrypoint bash \
    "${TAG}" \
    -lc "./build-webui.sh"

echo "[build.sh] webui build complete: ${REPO_ROOT}/tools/forgather_server/webui/dist"
