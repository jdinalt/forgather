#!/bin/bash
# Build the Forgather development image.
#
# The dev image is single-user and host-scoped: ``id -u``, ``id -g``,
# and ``id -un`` are baked into the image as the in-container user's
# UID/GID/name, so files created inside the container land owned by
# the same identity on the host. The default image tag is
# ``forgather-dev:<host-username>`` to avoid cross-user collisions on
# shared hosts. (For the user-agnostic, build-once-deploy-everywhere
# image, see ``docker/runtime/build.sh``.)
#
# After the image build succeeds we run ./build-webui.sh inside a
# transient container against the host clone, so the SPA dist/ lands
# in the user's tree and the Forgather server's web UI works the
# moment they start it. Skip this post-step with SKIP_WEBUI_BUILD=1.
#
# Usage:
#   docker/build.sh                    # tag: forgather-dev:<host-user>
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

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="$(id -un)"

DEFAULT_TAG="forgather-dev:${HOST_USER}"
TAG="${1:-${DEFAULT_TAG}}"
# Only consume $1 as TAG if it's not a docker passthrough flag.
if [[ "${TAG}" == --* ]] || [[ "${TAG}" == -* ]]; then
    TAG="${DEFAULT_TAG}"
else
    shift || true
fi
# Drop a leading "--" separator so callers can pass extra docker args:
#   docker/build.sh forgather-dev:dinalt -- --progress=plain
if [[ "${1:-}" == "--" ]]; then shift; fi

echo "Building ${TAG}"
echo "  in-container user: ${HOST_USER} (uid=${HOST_UID}, gid=${HOST_GID})"
if [[ "${INSTALL_CLAUDE}" = "1" ]]; then
    echo "  --claude: also installing Claude Code (npm global)"
fi

docker build \
    -t "${TAG}" \
    --build-arg "INSTALL_CLAUDE=${INSTALL_CLAUDE}" \
    --build-arg "USER_NAME=${HOST_USER}" \
    --build-arg "USER_UID=${HOST_UID}" \
    --build-arg "USER_GID=${HOST_GID}" \
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

# The image's in-container user is already the host operator (baked
# in via build args above), so files created inside the container
# land owned correctly on the host without --user gymnastics. We
# still override --entrypoint to bash here to skip the editable-
# install dance: this throw-away container doesn't need
# FORGATHER_REPO set up, just a writable cwd to invoke npm.
docker run --rm \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${REPO_ROOT}" \
    --entrypoint bash \
    "${TAG}" \
    -lc "./build-webui.sh"

echo "[build.sh] webui build complete: ${REPO_ROOT}/tools/forgather_server/webui/dist"
