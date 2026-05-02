#!/bin/bash
# Launch an interactive shell in the Forgather development container.
#
# Bind-mounts the host home directory at the same path used inside the
# container so absolute paths in your shell history, configs, and
# project files keep resolving correctly. The image carries an account
# matching your host UID/GID (set at build time), so file ownership is
# preserved across the boundary.
#
# Networking: defaults to host networking (--network host) so the
# container shares the host's network stack — every service inside
# the container is reachable on its bound port without -p mappings,
# and services that default to 127.0.0.1 (forgather server, mkdocs,
# tensorboard, inference) Just Work without per-tool --host /
# --bind_all flags. Set NETWORK=bridge to opt back into bridge
# networking with explicit port-forwards (the original behaviour).
#
# Usage:
#   docker/run.sh                      # interactive bash, GPUs enabled, --network host
#   docker/run.sh forgather ls -r      # one-shot command
#   IMAGE=forgather-dev:my-tag docker/run.sh
#   GPUS=none docker/run.sh            # CPU only
#   GPUS='"device=0,1"' docker/run.sh  # specific GPUs
#   NETWORK=bridge docker/run.sh       # bridge networking with -p forwards
#   HOST_BIND=0.0.0.0 NETWORK=bridge \  # bridge + LAN exposure on host side
#       docker/run.sh
#   EXTRA_PORTS='-p 6006:6006' docker/run.sh   # only with NETWORK=bridge

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-forgather-dev:latest}"
GPUS="${GPUS:-all}"
NETWORK="${NETWORK:-host}"
EXTRA_PORTS="${EXTRA_PORTS:-}"

GPU_ARGS=()
if [[ "${GPUS}" != "none" ]]; then
    GPU_ARGS=(--gpus "${GPUS}")
fi

NET_ARGS=()
PORT_ARGS=()
case "${NETWORK}" in
    host)
        NET_ARGS=(--network host)
        if [[ -n "${EXTRA_PORTS}" ]]; then
            echo "warning: EXTRA_PORTS is ignored under NETWORK=host (host networking" >&2
            echo "         exposes every bound port directly). Use NETWORK=bridge to" >&2
            echo "         opt back into explicit port forwarding." >&2
        fi
        ;;
    bridge)
        # Forward the canonical Forgather server / job ports. Host
        # side defaults to 127.0.0.1 (same exposure as `forgather
        # server` running on bare metal); set HOST_BIND=0.0.0.0 for
        # LAN access. Note: under bridge networking, services *inside*
        # the container must bind to 0.0.0.0 (not 127.0.0.1) — pass
        # `-H 0.0.0.0` / `--host 0.0.0.0` / `--bind_all` to whichever
        # tool you're starting.
        HOST_BIND="${HOST_BIND:-127.0.0.1}"
        PORT_ARGS=(
            -p ${HOST_BIND}:8765:8765
            -p ${HOST_BIND}:8137:8137
            -p ${HOST_BIND}:6006:6006
            -p ${HOST_BIND}:8000:8000
        )
        ;;
    *)
        echo "error: unknown NETWORK=${NETWORK} (expected 'host' or 'bridge')" >&2
        exit 2
        ;;
esac

# Allow the caller to bind-mount additional paths via EXTRA_MOUNTS,
# e.g. EXTRA_MOUNTS="-v /scratch:/scratch -v /data:/data".
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

# `--hostname` is rejected by docker when --network=host is used
# (the container inherits the host's hostname).
HOSTNAME_ARGS=()
if [[ "${NETWORK}" != "host" ]]; then
    HOSTNAME_ARGS=(--hostname "forgather-dev")
fi

exec docker run --rm -it \
    "${HOSTNAME_ARGS[@]}" \
    --name "forgather-dev-$$" \
    "${GPU_ARGS[@]}" \
    "${NET_ARGS[@]}" \
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
