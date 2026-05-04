#!/bin/bash
# Build the Forgather runtime (distributable) image.
#
# Unlike `docker/build.sh`, this image is generic — there are no
# host-user build args. The in-image `forgather` user is fixed at
# UID/GID 1000, and the runtime entrypoint remaps to PUID/PGID at
# container start so the same image works for any host user.
#
# The webui SPA is built INSIDE the image (no host post-step
# required), and the source tree is baked in at /opt/forgather/repo.
#
# Usage:
#   docker/runtime/build.sh                    # tag: forgather:latest
#   docker/runtime/build.sh my/forgather:1.1   # custom tag
#   docker/runtime/build.sh -- --no-cache      # pass extra args to docker build
#
# Tip: pair `--no-cache` with a clean checkout for release-test builds.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TAG="${1:-forgather:latest}"
shift || true
if [[ "${1:-}" == "--" ]]; then shift; fi

echo "Building runtime image ${TAG}"

docker build \
    -t "${TAG}" \
    -f "${REPO_ROOT}/Dockerfile.runtime" \
    "$@" \
    "${REPO_ROOT}"

cat <<EOF

[build.sh] runtime image built: ${TAG}

Try it:
    docker/runtime/run.sh                 # creates a container, starts forgather server
    docker/runtime/run.sh --status        # report state
    docker/runtime/run.sh --shell         # diagnostic shell as the forgather user
EOF
