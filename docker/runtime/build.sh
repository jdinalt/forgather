#!/bin/bash
# Build the Forgather runtime (distributable) image.
#
# Unlike `docker/build.sh`, this image is generic — there are no
# host-user build args, and the source tree is fetched fresh from
# git at build time rather than COPY'd from the host. The in-image
# `forgather` user is fixed at UID/GID 1000, and the runtime
# entrypoint remaps to PUID/PGID at container start so the same
# image works for any host user.
#
# The webui SPA is built INSIDE the image (no host post-step
# required); the cloned source tree lives at /opt/forgather/repo.
#
# Usage:
#   docker/runtime/build.sh                       # tag: forgather:latest, ref: main
#   docker/runtime/build.sh my/forgather:1.1.0    # custom tag
#   FORGATHER_GIT_REF=v1.1.0 docker/runtime/build.sh
#                                                 # pin to a release tag
#   FORGATHER_GIT_URL=https://my.fork.example/forgather.git \
#       FORGATHER_GIT_REF=feature/foo docker/runtime/build.sh
#                                                 # build from a fork/branch
#   docker/runtime/build.sh -- --no-cache         # pass extra args to docker build
#
# Tip: BuildKit is required (DOCKER_BUILDKIT=1, or any reasonably
# modern docker installation). Apt + uv caches are mounted as
# BuildKit cache volumes, so clean rebuilds skip the package
# downloads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ``--help`` MUST short-circuit before any docker invocation — otherwise
# it gets consumed as the positional TAG and the resulting ``docker
# build -t --help ...`` call attempts a real build with a bogus tag.
for tok in "$@"; do
    case "${tok}" in
        -h|--help)
            sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

TAG="${1:-forgather:latest}"
shift || true
if [[ "${1:-}" == "--" ]]; then shift; fi

GIT_URL="${FORGATHER_GIT_URL:-https://github.com/jdinalt/forgather.git}"
GIT_REF="${FORGATHER_GIT_REF:-dev}"

echo "Building runtime image ${TAG}"
echo "  git url: ${GIT_URL}"
echo "  git ref: ${GIT_REF}"

export DOCKER_BUILDKIT=1
docker build \
    -t "${TAG}" \
    -f "${REPO_ROOT}/Dockerfile.runtime" \
    --build-arg "FORGATHER_GIT_URL=${GIT_URL}" \
    --build-arg "FORGATHER_GIT_REF=${GIT_REF}" \
    "$@" \
    "${REPO_ROOT}"

cat <<EOF

[build.sh] runtime image built: ${TAG}

Try it:
    docker/runtime/run.sh                 # creates a container, starts forgather server
    docker/runtime/run.sh --status        # report state
    docker/runtime/run.sh --shell         # diagnostic shell as the forgather user
EOF
