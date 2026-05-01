#!/bin/bash
# Build the Forgather development image with build args matching the
# host user, so a bind-mounted home keeps correct file ownership.
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

exec docker build \
    --build-arg "USER_NAME=${USER_NAME}" \
    --build-arg "USER_UID=${USER_UID}" \
    --build-arg "USER_GID=${USER_GID}" \
    -t "${TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "$@" \
    "${REPO_ROOT}"
