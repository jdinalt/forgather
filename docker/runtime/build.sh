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
# TLS seed (optional, for build-once-distribute-everywhere clusters):
#
#   TLS_BAKE_FROM_HOST=1 docker/runtime/build.sh
#                                                 # bake the operator's
#                                                 # ~/.config/forgather/tls/ into
#                                                 # the image. Every container
#                                                 # from this image then trusts
#                                                 # the operator's CA *and*
#                                                 # serves the same cert; peers
#                                                 # can talk to each other and
#                                                 # to the operator's dev box
#                                                 # without `tls install` on
#                                                 # every node.
#   TLS_BAKE_FROM_DIR=/path/to/tls docker/runtime/build.sh
#                                                 # explicit path; must contain
#                                                 # at minimum server.crt,
#                                                 # server.key, ca/ca.crt.
#                                                 # See docs/operations/tls.md
#                                                 # "Docker runtime image".
#
# *** SECURITY ***: a TLS-baked image is itself a secret — it carries
# the CA private key (so any holder can mint certs in the cluster's
# trust domain) and the server private key (so any holder can
# impersonate any node). Never publish a TLS-baked image to a public
# registry. For shipping to teammates: use a private registry, or
# distribute via `docker save | ssh ... docker load` over a trusted
# channel.
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
            sed -n '2,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

TAG="${1:-forgather:latest}"
shift || true
if [[ "${1:-}" == "--" ]]; then shift; fi

GIT_URL="${FORGATHER_GIT_URL:-https://github.com/jdinalt/forgather.git}"
GIT_REF="${FORGATHER_GIT_REF:-dev}"

# --- TLS bake (optional) ------------------------------------------
# The runtime image's Dockerfile accepts FORGATHER_TLS_SEED_DIR as a
# *path inside the build context*. Translate the user-friendly
# TLS_BAKE_FROM_HOST / TLS_BAKE_FROM_DIR env vars into that arg by
# staging the seed into a hidden directory under REPO_ROOT. The
# directory is removed on exit via a trap so the build context stays
# clean even on failure.
TLS_BAKE_FROM_HOST="${TLS_BAKE_FROM_HOST:-}"
TLS_BAKE_FROM_DIR="${TLS_BAKE_FROM_DIR:-}"
TLS_SEED_ARG=""
TLS_STAGED_DIR=""

if [[ -n "${TLS_BAKE_FROM_HOST}" && -n "${TLS_BAKE_FROM_DIR}" ]]; then
    echo "[build.sh] error: set only one of TLS_BAKE_FROM_HOST and TLS_BAKE_FROM_DIR" >&2
    exit 2
fi

if [[ -n "${TLS_BAKE_FROM_HOST}" ]]; then
    # Resolve via the forgather CLI / Python so non-Linux hosts (macOS,
    # Windows-via-WSL with custom XDG, etc.) get the right path instead
    # of a hard-coded ~/.config/forgather/. Falls back to the Linux
    # default if Python isn't available — the operator can always use
    # TLS_BAKE_FROM_DIR=PATH if the auto-detect doesn't fit.
    if command -v python3 >/dev/null 2>&1; then
        TLS_SRC="$(python3 -c \
            'from forgather.tls.config import tls_dir; print(tls_dir())' \
            2>/dev/null || true)"
    fi
    TLS_SRC="${TLS_SRC:-${HOME}/.config/forgather/tls}"
elif [[ -n "${TLS_BAKE_FROM_DIR}" ]]; then
    TLS_SRC="${TLS_BAKE_FROM_DIR%/}"
else
    TLS_SRC=""
fi

if [[ -n "${TLS_SRC}" ]]; then
    if [[ ! -d "${TLS_SRC}" ]]; then
        echo "[build.sh] error: TLS source directory not found: ${TLS_SRC}" >&2
        echo "[build.sh]   run 'forgather tls init' first, or point TLS_BAKE_FROM_DIR at a provisioned dir" >&2
        exit 2
    fi
    if [[ ! -f "${TLS_SRC}/server.crt" || ! -f "${TLS_SRC}/server.key" ]]; then
        echo "[build.sh] error: ${TLS_SRC} missing server.crt or server.key" >&2
        exit 2
    fi
    if [[ ! -f "${TLS_SRC}/ca/ca.crt" ]]; then
        echo "[build.sh] error: ${TLS_SRC}/ca/ca.crt missing (need a CA-holding host)" >&2
        exit 2
    fi
    # Stage into the build context under a hidden directory so Docker
    # can see it via FORGATHER_TLS_SEED_DIR. Use a process-local name
    # so concurrent builds don't trample each other; clean up via trap.
    TLS_STAGED_DIR=".tls-seed-build-$$"
    STAGED_PATH="${REPO_ROOT}/${TLS_STAGED_DIR}"
    rm -rf "${STAGED_PATH}"
    mkdir -p "${STAGED_PATH}"
    # Trap covers all exit paths (success, failure, ^C).
    trap 'rm -rf "${STAGED_PATH}"' EXIT
    # `cp -a` preserves the 0600 mode on the private keys. The bind-
    # mount inside the Dockerfile is read-only, so the build can't
    # accidentally widen the perms.
    cp -a "${TLS_SRC}/." "${STAGED_PATH}/"
    TLS_SEED_ARG="${TLS_STAGED_DIR}"
    cat <<EOF >&2
[build.sh] *** TLS-BAKED BUILD ***
[build.sh]   source:  ${TLS_SRC}
[build.sh]   staging: ${STAGED_PATH} (auto-cleanup on exit)
[build.sh]   The resulting image WILL CONTAIN the CA private key and
[build.sh]   the server private key. Treat the image as a secret —
[build.sh]   never push to a public registry. Distribute via private
[build.sh]   registry or 'docker save | ssh ... docker load'.
EOF
fi

echo "Building runtime image ${TAG}"
echo "  git url: ${GIT_URL}"
echo "  git ref: ${GIT_REF}"
if [[ -n "${TLS_SEED_ARG}" ]]; then
    echo "  tls seed: ${TLS_SEED_ARG} (from ${TLS_SRC})"
fi

export DOCKER_BUILDKIT=1
EXTRA_BUILD_ARGS=()
if [[ -n "${TLS_SEED_ARG}" ]]; then
    EXTRA_BUILD_ARGS+=(--build-arg "FORGATHER_TLS_SEED_DIR=${TLS_SEED_ARG}")
fi

docker build \
    -t "${TAG}" \
    -f "${REPO_ROOT}/Dockerfile.runtime" \
    --build-arg "FORGATHER_GIT_URL=${GIT_URL}" \
    --build-arg "FORGATHER_GIT_REF=${GIT_REF}" \
    "${EXTRA_BUILD_ARGS[@]}" \
    "$@" \
    "${REPO_ROOT}"

cat <<EOF

[build.sh] runtime image built: ${TAG}

Try it:
    docker/runtime/run.sh                 # creates a container, starts forgather server
    docker/runtime/run.sh --status        # report state
    docker/runtime/run.sh --shell         # diagnostic shell as the forgather user
EOF
