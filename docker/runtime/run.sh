#!/bin/bash
# Launch / manage the Forgather runtime container.
#
# Defaults:
#   - bridge networking with -p ${HOST_BIND}:${PORT}:8765
#   - PUID/PGID forwarded from the calling host user (so files written
#     to bind-mounted volumes match host ownership)
#   - bind-mounts the host's ~/.cache/huggingface so HF datasets/models
#     are shared with the host install (the central pain point this
#     image is designed around)
#   - named volume `forgather-state` mounted at ~/.forgather so server
#     state (auth token, queue, GPU policy, ...) survives `docker rm`
#   - --gpus all (set GPUS=none for CPU-only)
#
# Usage:
#   docker/runtime/run.sh                 # create + start (or attach if exists)
#   docker/runtime/run.sh --status        # report container state
#   docker/runtime/run.sh --logs          # tail container logs
#   docker/runtime/run.sh --shell         # diagnostic shell (`docker exec -u forgather -ti ... bash`)
#   docker/runtime/run.sh --token         # print the server's auth token
#   docker/runtime/run.sh --stop          # stop (but keep) the container
#   docker/runtime/run.sh --rm            # stop + remove the container
#   docker/runtime/run.sh --recreate      # remove and recreate
#
# Environment overrides (applied at container CREATE time only):
#   IMAGE=forgather:my-tag                # default: forgather:latest
#   NAME=my-server                        # default: forgather-server
#   PORT=8765                             # default: 8765
#   HOST_BIND=0.0.0.0                     # default: 127.0.0.1 (loopback only)
#   GPUS=none                             # default: all
#   GPUS='"device=0,1"'                   # specific GPUs
#   HF_CACHE_HOST=/path/to/host/cache     # default: $HOME/.cache/huggingface
#   STATE_VOLUME=my-volume                # default: forgather-state (named volume)
#                                         # set to a host path to bind-mount instead
#   EXTRA_MOUNTS='-v /scratch:/scratch'   # additional volume args
#   EXTRA_PORTS='-p 6006:6006'            # forward additional ports (e.g. tensorboard)

set -euo pipefail

IMAGE="${IMAGE:-forgather:latest}"
NAME="${NAME:-forgather-server}"
PORT="${PORT:-8765}"
HOST_BIND="${HOST_BIND:-127.0.0.1}"
GPUS="${GPUS:-all}"
HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
STATE_VOLUME="${STATE_VOLUME:-forgather-state}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
EXTRA_PORTS="${EXTRA_PORTS:-}"

container_state() {
    local s
    s="$(docker inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null || true)"
    case "${s}" in
        running) echo running ;;
        "") echo absent ;;
        *) echo stopped ;;
    esac
}

create_container() {
    # Lazily create the host HF cache dir so the bind-mount lands somewhere.
    mkdir -p "${HF_CACHE_HOST}"

    GPU_ARGS=()
    if [[ "${GPUS}" != "none" ]]; then
        GPU_ARGS=(--gpus "${GPUS}")
    fi

    # State mount. Docker accepts both a named-volume identifier and
    # a host path on the LHS of `-v`, so we don't need to discriminate.
    STATE_MOUNT="-v ${STATE_VOLUME}:/home/forgather/.forgather"

    echo "[run.sh] creating container ${NAME} from ${IMAGE}" >&2
    echo "[run.sh]   PUID=$(id -u)  PGID=$(id -g)" >&2
    echo "[run.sh]   port:    ${HOST_BIND}:${PORT} -> 8765" >&2
    echo "[run.sh]   hf cache: ${HF_CACHE_HOST} -> /home/forgather/.cache/huggingface" >&2
    echo "[run.sh]   state:   ${STATE_VOLUME} -> /home/forgather/.forgather" >&2

    docker run -d \
        --name "${NAME}" \
        --hostname "forgather" \
        "${GPU_ARGS[@]}" \
        --shm-size=8g \
        --ipc=host \
        -p "${HOST_BIND}:${PORT}:8765" \
        -e "PUID=$(id -u)" \
        -e "PGID=$(id -g)" \
        -v "${HF_CACHE_HOST}:/home/forgather/.cache/huggingface" \
        ${STATE_MOUNT} \
        ${EXTRA_PORTS} \
        ${EXTRA_MOUNTS} \
        "${IMAGE}" > /dev/null

    cat >&2 <<EOF

[run.sh] forgather server is starting in the background.

  url:           http://${HOST_BIND}:${PORT}/
  auth token:    docker/runtime/run.sh --token
  diag shell:    docker/runtime/run.sh --shell
  logs:          docker/runtime/run.sh --logs
EOF
}

ensure_running() {
    case "$(container_state)" in
        running) ;;
        stopped)
            echo "[run.sh] starting existing container ${NAME}" >&2
            docker start "${NAME}" > /dev/null
            ;;
        absent)
            create_container
            ;;
    esac
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
        ;;
    --logs)
        ensure_running
        exec docker logs -f "${NAME}"
        ;;
    --shell)
        ensure_running
        exec docker exec -it \
            -u forgather \
            -e "TERM=${TERM:-xterm-256color}" \
            "${NAME}" \
            bash -l
        ;;
    --token)
        ensure_running
        # The server creates the auth token on first start. Wait briefly
        # if a freshly-started container hasn't written it yet.
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            if docker exec "${NAME}" test -f /home/forgather/.forgather/server/auth_token; then
                exec docker exec "${NAME}" cat /home/forgather/.forgather/server/auth_token
            fi
            sleep 1
        done
        echo "[run.sh] timed out waiting for auth token; check 'docker/runtime/run.sh --logs'" >&2
        exit 1
        ;;
    --stop)
        if [[ "$(container_state)" == "running" ]]; then
            echo "[run.sh] stopping ${NAME}" >&2
            docker stop "${NAME}" > /dev/null
        else
            echo "[run.sh] container ${NAME} is not running" >&2
        fi
        ;;
    --rm)
        if [[ "$(container_state)" != "absent" ]]; then
            echo "[run.sh] removing ${NAME}" >&2
            docker rm -f "${NAME}" > /dev/null
        else
            echo "[run.sh] container ${NAME} does not exist" >&2
        fi
        ;;
    --recreate)
        if [[ "$(container_state)" != "absent" ]]; then
            echo "[run.sh] removing existing ${NAME}" >&2
            docker rm -f "${NAME}" > /dev/null
        fi
        create_container
        ;;
    -h|--help)
        sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        ;;
    "")
        ensure_running
        ;;
    *)
        echo "unknown subcommand: $1 (try --help)" >&2
        exit 2
        ;;
esac
