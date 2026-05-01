#!/bin/bash
# Launch an interactive shell in the Forgather development container.
#
# Bind-mounts the host home directory at the same path used inside the
# container so absolute paths in your shell history, configs, and
# project files keep resolving correctly. The image carries an account
# matching your host UID/GID (set at build time), so file ownership is
# preserved across the boundary.
#
# Usage:
#   docker/run.sh                      # interactive bash, GPUs enabled
#   docker/run.sh forgather ls -r      # one-shot command
#   IMAGE=forgather-dev:my-tag docker/run.sh
#   GPUS=none docker/run.sh            # CPU only
#   GPUS='"device=0,1"' docker/run.sh  # specific GPUs
#   EXTRA_PORTS='-p 6006:6006' docker/run.sh
#
# By default we forward the canonical Forgather server / job ports
# (8765 / 8137 / 6006 / 8000) so the host browser can reach the
# services without per-invocation -p flags.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-forgather-dev:latest}"
GPUS="${GPUS:-all}"
EXTRA_PORTS="${EXTRA_PORTS:-}"

GPU_ARGS=()
if [[ "${GPUS}" != "none" ]]; then
    GPU_ARGS=(--gpus "${GPUS}")
fi

# Default port-forwards mirror the table in docs/getting-started.
PORT_ARGS=(
    -p 127.0.0.1:8765:8765
    -p 127.0.0.1:8137:8137
    -p 127.0.0.1:6006:6006
    -p 127.0.0.1:8000:8000
)

# Allow the caller to bind-mount additional paths via EXTRA_MOUNTS,
# e.g. EXTRA_MOUNTS="-v /scratch:/scratch -v /data:/data".
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

exec docker run --rm -it \
    --hostname "forgather-dev" \
    --name "forgather-dev-$$" \
    "${GPU_ARGS[@]}" \
    --shm-size=8g \
    --ipc=host \
    -v "${HOME}:${HOME}" \
    -w "${REPO_ROOT}" \
    -e "FORGATHER_REPO=${REPO_ROOT}" \
    -e "HOME=${HOME}" \
    -e "TERM=${TERM:-xterm-256color}" \
    "${PORT_ARGS[@]}" \
    ${EXTRA_PORTS} \
    ${EXTRA_MOUNTS} \
    "${IMAGE}" \
    "$@"
