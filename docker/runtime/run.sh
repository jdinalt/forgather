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
#   NETWORK=host                          # default: bridge (with -p
#                                         #   forwards). Use 'host' for
#                                         #   multi-node operation —
#                                         #   the cluster's mDNS
#                                         #   discovery uses multicast,
#                                         #   which doesn't traverse
#                                         #   docker bridge networks.
#                                         #   Under host networking,
#                                         #   PORT and HOST_BIND are
#                                         #   ignored (the server binds
#                                         #   directly on the host's
#                                         #   network namespace) and
#                                         #   EXTRA_PORTS is warned
#                                         #   about + ignored.
#   PORT=8765                             # default: 8765 (bridge only)
#   HOST_BIND=0.0.0.0                     # default: 127.0.0.1 (loopback only;
#                                         #   bridge only)
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
#
# Persistent overrides: if $XDG_CONFIG_HOME/forgather/docker.env (or
# ~/.config/forgather/docker.env) exists it is sourced before defaults
# are applied. Override the path with FORGATHER_DOCKER_CONFIG. Use the
# `: "${VAR:=default}"` pattern in the file so command-line
# `VAR=... run.sh` still wins.

set -euo pipefail

# Shared scaffold (config-file load, container_state, ensure_running,
# common subcommand dispatch). See docker/_lib.sh for the contract.
# shellcheck source=../_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
lib_load_config

IMAGE="${IMAGE:-forgather:latest}"
NAME="${NAME:-forgather-server}"
# Default to bridge networking (portable). Set NETWORK=host for
# multi-node operation where mDNS discovery needs unrestricted
# multicast access to the host network namespace.
NETWORK="${NETWORK:-bridge}"
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

do_create_container() {
    GPU_ARGS=()
    if [[ "${GPUS}" != "none" ]]; then
        GPU_ARGS=(--gpus "${GPUS}")
    fi

    # Networking mode: bridge (default, with -p forwards) or host
    # (for multi-node mDNS / multicast discovery).
    NET_ARGS=()
    PORT_ARGS=()
    case "${NETWORK}" in
        host)
            NET_ARGS=(--network host)
            if [[ -n "${EXTRA_PORTS}" ]]; then
                echo "[run.sh] warning: EXTRA_PORTS is ignored under NETWORK=host" >&2
            fi
            echo "[run.sh] networking: host (multi-node-friendly; PORT/HOST_BIND/EXTRA_PORTS ignored)" >&2
            ;;
        bridge)
            PORT_ARGS=(-p "${HOST_BIND}:${PORT}:8765")
            ;;
        *)
            echo "error: unknown NETWORK=${NETWORK} (expected 'host' or 'bridge')" >&2
            exit 2
            ;;
    esac

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
    if [[ "${NETWORK}" == "host" ]]; then
        echo "[run.sh]   port:    host networking (server binds 0.0.0.0:8765 on host)" >&2
    else
        echo "[run.sh]   port:    ${HOST_BIND}:${PORT} -> 8765" >&2
    fi
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

    # --hostname is incompatible with --network host (docker rejects
    # the combination), so only set it under bridge networking.
    HOSTNAME_ARGS=()
    if [[ "${NETWORK}" != "host" ]]; then
        HOSTNAME_ARGS=(--hostname "forgather")
    fi

    # Under host networking, EXTRA_PORTS was already warned about and
    # is intentionally dropped from the docker run line.
    EXTRA_PORTS_FINAL="${EXTRA_PORTS}"
    if [[ "${NETWORK}" == "host" ]]; then
        EXTRA_PORTS_FINAL=""
    fi

    docker run -d \
        --name "${NAME}" \
        "${HOSTNAME_ARGS[@]}" \
        "${GPU_ARGS[@]}" \
        "${NET_ARGS[@]}" \
        --shm-size=8g \
        --ipc=host \
        "${PORT_ARGS[@]}" \
        -e "PUID=$(id -u)" \
        -e "PGID=$(id -g)" \
        "${VOLUME_ARGS[@]}" \
        ${EXTRA_PORTS_FINAL} \
        ${EXTRA_MOUNTS} \
        "${IMAGE}" > /dev/null

    echo "[run.sh] container started; waiting for auth token..." >&2

    # Mirror the dev-image experience: print a clickable URL with the
    # token embedded so the operator doesn't have to chase down
    # `--token` on first start. The token file is created by the
    # server on first request and persists in the state volume across
    # restarts, so subsequent re-attaches get the same value.
    #
    # Under host networking the server binds 0.0.0.0:8765 directly on
    # the host, so the URL that will Just Work from this host is
    # http://127.0.0.1:8765 (and the LAN IP if reachable) — not
    # ${HOST_BIND}:${PORT}, which only apply under bridge networking.
    local url_host url_port
    if [[ "${NETWORK}" == "host" ]]; then
        url_host="127.0.0.1"
        url_port="8765"
    else
        url_host="${HOST_BIND}"
        url_port="${PORT}"
    fi

    local token=""
    if token="$(read_auth_token 30)"; then
        cat >&2 <<EOF

[run.sh] forgather server is up.

  url (clickable):  http://${url_host}:${url_port}/?token=${token}
  url (plain):      http://${url_host}:${url_port}/
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

  url:    http://${url_host}:${url_port}/
EOF
    fi
}

# ---------- subcommand dispatch -----------------------------------
# Common subcommands (--status, --stop, --rm) are handled by the
# shared lib; image-specific ones live below.

if lib_handle_common_subcommand "${1:-}"; then
    exit 0
fi

case "${1:-}" in
    --logs)
        lib_ensure_running
        exec docker logs -f "${NAME}"
        ;;
    --shell)
        lib_ensure_running
        exec docker exec -it \
            -u forgather \
            -e "TERM=${TERM:-xterm-256color}" \
            "${NAME}" \
            bash -l
        ;;
    --token)
        lib_ensure_running
        if ! read_auth_token 10; then
            echo "[run.sh] timed out waiting for auth token; check 'docker/runtime/run.sh --logs'" >&2
            exit 1
        fi
        ;;
    --recreate)
        if [[ "$(lib_container_state)" != "absent" ]]; then
            echo "[run.sh] removing existing ${NAME}" >&2
            docker rm -f "${NAME}" > /dev/null
        fi
        do_create_container
        ;;
    -h|--help)
        sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        ;;
    "")
        lib_ensure_running
        ;;
    *)
        echo "unknown subcommand: $1 (try --help)" >&2
        exit 2
        ;;
esac
