#!/bin/bash
# Launch / attach to the Forgather development container.
#
# The container is long-lived: first invocation creates it detached
# (PID 1 is `sleep infinity`), subsequent invocations re-attach via
# `docker exec`. Logging out of an interactive shell does NOT stop
# the container, so a `forgather server` started in one session keeps
# running, and you can re-attach from a new terminal to inspect /
# control jobs without restarting anything.
#
# Bind-mounts the host home directory at the same path used inside
# the container so absolute paths in shell history, configs, and
# project files keep resolving correctly. The image carries an
# account matching your host UID/GID (set at build time), so file
# ownership is preserved across the boundary.
#
# Networking: defaults to host networking (--network host) so the
# container shares the host's network stack — every service inside
# the container is reachable on its bound port without -p mappings,
# and services that default to 127.0.0.1 (forgather server, mkdocs,
# tensorboard, inference) Just Work without per-tool --host /
# --bind_all flags. Set NETWORK=bridge to opt back into bridge
# networking with explicit port-forwards.
#
# Usage:
#   docker/run.sh                      # interactive bash (creates container if needed)
#   docker/run.sh forgather ls -r      # one-shot command in the same container
#   docker/run.sh --status             # report container state
#   docker/run.sh --stop               # stop (but keep) the container
#   docker/run.sh --rm                 # stop and remove the container
#   docker/run.sh --recreate           # remove and recreate from scratch
#
# Environment overrides (only applied when the container is being
# CREATED — ignored on re-attach):
#   IMAGE=forgather-dev:my-tag         # default: forgather-dev:latest
#   NAME=my-forgather                  # default: forgather-dev-$USER
#   GPUS=none                          # default: all
#   GPUS='"device=0,1"'                # specific GPUs
#   NETWORK=bridge                     # default: host
#   HOST_BIND=0.0.0.0                  # bridge mode only; default: 127.0.0.1
#   EXTRA_PORTS='-p 5173:5173'         # bridge mode only
#   EXTRA_MOUNTS='-v /scratch:/scratch'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-forgather-dev:latest}"
NAME="${NAME:-forgather-dev-${USER:-$(id -un)}}"
GPUS="${GPUS:-all}"
NETWORK="${NETWORK:-host}"
EXTRA_PORTS="${EXTRA_PORTS:-}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

container_state() {
    # Prints "running", "stopped", or "absent".
    local s
    s="$(docker inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null || true)"
    case "${s}" in
        running) echo running ;;
        "") echo absent ;;
        *) echo stopped ;;
    esac
}

create_container() {
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
                echo "warning: EXTRA_PORTS is ignored under NETWORK=host" >&2
            fi
            ;;
        bridge)
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

    HOSTNAME_ARGS=()
    if [[ "${NETWORK}" != "host" ]]; then
        HOSTNAME_ARGS=(--hostname "forgather-dev")
    fi

    echo "[run.sh] creating container ${NAME} from ${IMAGE}" >&2

    # Detached, no --rm. PID 1 is sleep infinity so the container
    # survives across `docker exec` sessions and shell logouts.
    docker run -d \
        --name "${NAME}" \
        "${HOSTNAME_ARGS[@]}" \
        "${GPU_ARGS[@]}" \
        "${NET_ARGS[@]}" \
        --shm-size=8g \
        --ipc=host \
        -v "${HOME}:${HOME}" \
        -w "${REPO_ROOT}" \
        -e "FORGATHER_REPO=${REPO_ROOT}" \
        -e "HOME=${HOME}" \
        "${PORT_ARGS[@]}" \
        ${EXTRA_PORTS} \
        ${EXTRA_MOUNTS} \
        "${IMAGE}" \
        sleep infinity > /dev/null
}

ensure_running() {
    local state
    state="$(container_state)"
    case "${state}" in
        running)
            ;;
        stopped)
            echo "[run.sh] starting existing container ${NAME}" >&2
            docker start "${NAME}" > /dev/null
            ;;
        absent)
            create_container
            ;;
    esac
}

attach_shell() {
    # Pass current TERM through; default to a sane value if absent
    # (e.g. invoked from a non-interactive parent).
    exec docker exec -it \
        -w "${REPO_ROOT}" \
        -e "TERM=${TERM:-xterm-256color}" \
        "${NAME}" \
        "$@"
}

# ---------- subcommand dispatch -----------------------------------

case "${1:-}" in
    --status)
        echo "container: ${NAME}"
        echo "state:     $(container_state)"
        if [[ "$(container_state)" != "absent" ]]; then
            docker inspect -f \
                'image:     {{.Config.Image}}{{"\n"}}network:   {{.HostConfig.NetworkMode}}{{"\n"}}started:   {{.State.StartedAt}}' \
                "${NAME}" 2>/dev/null || true
        fi
        exit 0
        ;;
    --stop)
        if [[ "$(container_state)" == "running" ]]; then
            echo "[run.sh] stopping ${NAME}" >&2
            docker stop "${NAME}" > /dev/null
        else
            echo "[run.sh] container ${NAME} is not running" >&2
        fi
        exit 0
        ;;
    --rm)
        if [[ "$(container_state)" != "absent" ]]; then
            echo "[run.sh] removing ${NAME}" >&2
            docker rm -f "${NAME}" > /dev/null
        else
            echo "[run.sh] container ${NAME} does not exist" >&2
        fi
        exit 0
        ;;
    --recreate)
        if [[ "$(container_state)" != "absent" ]]; then
            echo "[run.sh] removing existing ${NAME}" >&2
            docker rm -f "${NAME}" > /dev/null
        fi
        create_container
        shift
        attach_shell bash -l "$@"
        ;;
    -h|--help)
        sed -n '2,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    "")
        ensure_running
        attach_shell bash -l
        ;;
    *)
        ensure_running
        attach_shell "$@"
        ;;
esac
