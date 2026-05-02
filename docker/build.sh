#!/bin/bash
# Build the Forgather development image with build args matching the
# host user, so a bind-mounted home keeps correct file ownership.
#
# After the image build succeeds we run ./build-webui.sh inside a
# transient container against the host clone, so the SPA dist/ lands
# in the user's tree and the Forgather server's web UI works the
# moment they start it. Skip this post-step with SKIP_WEBUI_BUILD=1.
#
# Usage:
#   docker/build.sh                    # tag: forgather-dev:latest
#   docker/build.sh my-tag             # custom tag
#   docker/build.sh -- --no-cache      # pass extra args to docker build
#
# Override the user identity with env vars if you need something other
# than the current host user (e.g. building on CI):
#
#   USER_NAME=dev USER_UID=1000 USER_GID=1000 docker/build.sh
#
# Skip the webui post-step (e.g. you'll iterate on the SPA via
# `npm run dev` instead of the static dist/):
#
#   SKIP_WEBUI_BUILD=1 docker/build.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="${1:-forgather-dev:latest}"
shift || true
# Drop a leading "--" separator so callers can pass extra docker args:
#   docker/build.sh forgather-dev:latest -- --progress=plain
if [[ "${1:-}" == "--" ]]; then shift; fi

USER_NAME="${USER_NAME:-$(id -un)}"
USER_UID="${USER_UID:-$(id -u)}"
USER_GID="${USER_GID:-$(id -g)}"

echo "Building ${TAG}"
echo "  USER_NAME=${USER_NAME}"
echo "  USER_UID=${USER_UID}"
echo "  USER_GID=${USER_GID}"

docker build \
    --build-arg "USER_NAME=${USER_NAME}" \
    --build-arg "USER_UID=${USER_UID}" \
    --build-arg "USER_GID=${USER_GID}" \
    -t "${TAG}" \
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

# The image's entrypoint warns about a missing webui/dist and tries
# to install forgather editable; we want neither for this throw-away
# build. Override the entrypoint to bash and run the script directly.
docker run --rm \
    --user "${USER_UID}:${USER_GID}" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${REPO_ROOT}" \
    --entrypoint bash \
    "${TAG}" \
    -lc "./build-webui.sh"

echo "[build.sh] webui build complete: ${REPO_ROOT}/tools/forgather_server/webui/dist"
