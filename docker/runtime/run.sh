#!/bin/bash
# Launch / manage the Forgather runtime container.
#
# This script is opinionated about networking, GPU access, and the
# auth-token surface, but it is *not* opinionated about which host
# directories you expose into the container. NO host paths are
# bind-mounted by default. If you want to share a HuggingFace
# cache, a scratch dir, or a dataset volume, pass it explicitly via
# HF_CACHE_HOST or EXTRA_MOUNTS — see below.
#
# Defaults that ARE applied:
#   - bridge networking with -p ${HOST_BIND}:${PORT}:8765
#   - PUID/PGID forwarded from the calling host user (so files
#     written to any volumes you DO mount get host-correct ownership)
#   - --gpus all (set GPUS=none for CPU-only)
#   - named docker volume `forgather-state` at ~/.forgather inside
#     the container, for auth-token / queue / GPU-policy state
#     persistence across `docker rm`. This is a docker-managed
#     volume, not a host bind-mount — set STATE_VOLUME= (empty)
#     to disable, or to a host path to bind-mount instead.
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
#   HF_CACHE_HOST=$HOME/.cache/huggingface
#                                         # opt-in: bind-mount the host's HF
#                                         # cache into the container so
#                                         # downloads are shared with the host
#                                         # install. Default: unset (no mount).
#   STATE_VOLUME=my-volume                # default: forgather-state.
#                                         # Empty = no state mount (token does
#                                         # not persist across `docker rm`).
#                                         # Set to /host/path for a bind-mount.
#   EXTRA_MOUNTS='-v /scratch:/scratch'   # additional volume args
#   EXTRA_PORTS='-p 6006:6006'            # forward additional ports (e.g. tensorboard)

set -euo pipefail

IMAGE="${IMAGE:-forgather:latest}"
NAME="${NAME:-forgather-server}"
PORT="${PORT:-8765}"
HOST_BIND="${HOST_BIND:-127.0.0.1}"
GPUS="${GPUS:-all}"
# HF_CACHE_HOST is unset by default — the user opts in to this bind-mount.
HF_CACHE_HOST="${HF_CACHE_HOST:-}"
# STATE_VOLUME defaults to a docker-managed named volume (not a host mapping).
# Set to empty string to disable, or a host path for a bind-mount.
STATE_VOLUME="${STATE_VOLUME-forgather-state}"
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

# Poll the in-container state volume for the auth-token file the
# server writes on first start. Echoes the token on stdout if found,
# returns non-zero on timeout. Tries up to ~${1:-10} seconds.
read_auth_token() {
    local attempts="${1:-10}"
    local i
    for ((i = 0; i < attempts; i++)); do
        if docker exec "${NAME}" test -f /home/forgather/.forgather/server/auth_token 2>/dev/null; then
            docker exec "${NAME}" cat /home/forgather/.forgather/server/auth_token
            return 0
        fi
        sleep 1
    done
    return 1
}

create_container() {
    GPU_ARGS=()
    if [[ "${GPUS}" != "none" ]]; then
        GPU_ARGS=(--gpus "${GPUS}")
    fi

    # Build the volume args explicitly. Nothing is mounted unless the
    # user opted in via env var.
    VOLUME_ARGS=()

    if [[ -n "${HF_CACHE_HOST}" ]]; then
        # Lazily create the host dir so the bind-mount lands somewhere.
        mkdir -p "${HF_CACHE_HOST}"
        VOLUME_ARGS+=(-v "${HF_CACHE_HOST}:/home/forgather/.cache/huggingface")
    fi

    if [[ -n "${STATE_VOLUME}" ]]; then
        # Docker's -v LHS accepts both a named-volume identifier and a
        # host path; no need to discriminate.
        VOLUME_ARGS+=(-v "${STATE_VOLUME}:/home/forgather/.forgather")
    fi

    echo "[run.sh] creating container ${NAME} from ${IMAGE}" >&2
    echo "[run.sh]   PUID=$(id -u)  PGID=$(id -g)" >&2
    echo "[run.sh]   port:    ${HOST_BIND}:${PORT} -> 8765" >&2
    if [[ -n "${HF_CACHE_HOST}" ]]; then
        echo "[run.sh]   hf cache: ${HF_CACHE_HOST} -> /home/forgather/.cache/huggingface" >&2
    else
        echo "[run.sh]   hf cache: <not mounted; set HF_CACHE_HOST to share with host>" >&2
    fi
    if [[ -n "${STATE_VOLUME}" ]]; then
        echo "[run.sh]   state:   ${STATE_VOLUME} -> /home/forgather/.forgather" >&2
    else
        echo "[run.sh]   state:   <ephemeral; auth token will not persist across docker rm>" >&2
    fi
    if [[ -n "${EXTRA_MOUNTS}" ]]; then
        echo "[run.sh]   extra:   ${EXTRA_MOUNTS}" >&2
    fi

    docker run -d \
        --name "${NAME}" \
        --hostname "forgather" \
        "${GPU_ARGS[@]}" \
        --shm-size=8g \
        --ipc=host \
        -p "${HOST_BIND}:${PORT}:8765" \
        -e "PUID=$(id -u)" \
        -e "PGID=$(id -g)" \
        "${VOLUME_ARGS[@]}" \
        ${EXTRA_PORTS} \
        ${EXTRA_MOUNTS} \
        "${IMAGE}" > /dev/null

    echo "[run.sh] container started; waiting for auth token..." >&2

    # Mirror the dev-image experience: print a clickable URL with the
    # token embedded so the operator doesn't have to chase down
    # `--token` on first start. The token file is created by the
    # server on first request and persists in the state volume across
    # restarts, so subsequent re-attaches get the same value.
    local token=""
    if token="$(read_auth_token 30)"; then
        cat >&2 <<EOF

[run.sh] forgather server is up.

  url (clickable):  http://${HOST_BIND}:${PORT}/?token=${token}
  url (plain):      http://${HOST_BIND}:${PORT}/
  auth token:       ${token}

Re-fetch later:   docker/runtime/run.sh --token
Diagnostic shell: docker/runtime/run.sh --shell
Server logs:      docker/runtime/run.sh --logs
EOF
    else
        cat >&2 <<EOF

[run.sh] container is running but the auth token file hasn't appeared yet.
[run.sh] Re-check with:
  docker/runtime/run.sh --token
  docker/runtime/run.sh --logs

  url:    http://${HOST_BIND}:${PORT}/
EOF
    fi
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
        if ! read_auth_token 10; then
            echo "[run.sh] timed out waiting for auth token; check 'docker/runtime/run.sh --logs'" >&2
            exit 1
        fi
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
