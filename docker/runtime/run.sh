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
#   - named docker volume `forgather-state` at ~/.config/forgather
#     inside the container, for auth-token / queue / GPU-policy state
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
#   docker/runtime/run.sh --dev [PATH] --recreate
#                                         # debug only: bind-mount a
#                                         # host clone over the image's
#                                         # baked-in /opt/forgather/repo
#                                         # so host-side edits go live
#                                         # without rebuilding the image.
#                                         # PATH defaults to this script's
#                                         # repo root. See README.
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
#   CLUSTER=my-cluster                    # default: unset. When set, the
#                                         #   container's CMD is
#                                         #   `forgather server -H 0.0.0.0
#                                         #   -p 8765 --cluster <name>`
#                                         #   (instead of just the server
#                                         #   invocation). For multi-node
#                                         #   operation NETWORK=host is
#                                         #   strongly recommended — mDNS
#                                         #   discovery doesn't traverse
#                                         #   docker bridge networks.
#                                         #   See docs/guides/multi-node-training.md.
#   CLUSTER_ADDRESS=10.0.0.5              # default: unset. When set,
#                                         #   appends `--cluster-address
#                                         #   <ip>` to the server args
#                                         #   (overrides the auto-detected
#                                         #   interface IP advertised over
#                                         #   mDNS). Only meaningful when
#                                         #   CLUSTER is also set.
#   NO_AUTH=1                             # default: unset. When set, the
#                                         #   server starts with --no-auth
#                                         #   (no bearer-token gate).
#                                         #   TRUSTED-LAN ONLY — any host on
#                                         #   the network can hit the API.
#                                         #   Intended for smoke tests + the
#                                         #   multi-node testing flow where
#                                         #   token wrangling across N
#                                         #   containers is friction.
#   DEV=1                                 # default: unset. DEBUG-ONLY.
#   DEV=/path/to/forgather                #   When set, bind-mounts a host-
#                                         #   side forgather clone over
#                                         #   /opt/forgather/repo so an
#                                         #   operator can hot-fix the
#                                         #   image without rebuilding.
#                                         #   DEV=1 uses the script's own
#                                         #   repo root; DEV=/path uses
#                                         #   that path. The runtime image
#                                         #   is intended to be IMMUTABLE
#                                         #   AND IDENTICAL across a
#                                         #   distribution — this option
#                                         #   exists to test fixes without
#                                         #   rebuilding, NOT to deploy
#                                         #   live edits to production.
#                                         #   The flag --dev [PATH] is
#                                         #   equivalent to DEV=...
#
# Persistent overrides: if $XDG_CONFIG_HOME/forgather/docker.env (or
# ~/.config/forgather/docker.env) exists it is sourced before defaults
# are applied. Override the path with FORGATHER_DOCKER_CONFIG. Use the
# `: "${VAR:=default}"` pattern in the file so command-line
# `VAR=... run.sh` still wins.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
#
# Tip: to share state with the dev image (forgather control list,
# trainer logs, queue, auth token, ...), set
# STATE_VOLUME=$HOME/.config/forgather so both images bind-mount the
# same host directory at /home/forgather/.config/forgather. The dev
# image lands ~/.config/forgather there transparently because it
# bind-mounts $HOME wholesale; the runtime image opts in via this
# env var.
STATE_VOLUME="${STATE_VOLUME-forgather-state}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
EXTRA_PORTS="${EXTRA_PORTS:-}"
# Multi-node opt-in. When CLUSTER is set, the container's CMD becomes
# `forgather server -H 0.0.0.0 -p 8765 --cluster <name>` instead of
# just the server invocation. CLUSTER_ADDRESS, when set, appends
# `--cluster-address <ip>`.
CLUSTER="${CLUSTER:-}"
CLUSTER_ADDRESS="${CLUSTER_ADDRESS:-}"
# NO_AUTH=1 starts the server with --no-auth, disabling the bearer-token
# gate. Use ONLY on a trusted LAN: any user on the network can hit the
# API and submit jobs / read state. Intended for smoke tests and
# multi-node testing where token-fetching across N containers is
# operational friction. Default off — production deployments leave the
# token gate in place.
NO_AUTH="${NO_AUTH:-}"
# DEV opts in to bind-mounting a host-side forgather clone over the
# image's baked-in /opt/forgather/repo. Empty = production mode (default).
# "1" or unset-but-flag-passed = use ${REPO_ROOT}. Any other value
# is treated as an explicit host path.
DEV="${DEV:-}"

# Poll the in-container state volume for the auth-token file the
# server writes on first start. Echoes the token on stdout if found,
# returns non-zero on timeout. Tries up to ~${1:-10} seconds.
read_auth_token() {
    local attempts="${1:-10}"
    local i
    for ((i = 0; i < attempts; i++)); do
        if docker exec "${NAME}" test -f /home/forgather/.config/forgather/server/auth_token 2>/dev/null; then
            docker exec "${NAME}" cat /home/forgather/.config/forgather/server/auth_token
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
        VOLUME_ARGS+=(-v "${STATE_VOLUME}:/home/forgather/.config/forgather")
    fi

    # ---- DEV bind-mount (debug only) ---------------------------------
    # Bind-mount a host-side forgather clone over the image's baked-in
    # /opt/forgather/repo. The image is installed editable from that
    # path, so mounting over it makes any host-side edit go live without
    # rebuilding the image.
    DEV_PATH=""
    if [[ -n "${DEV}" ]]; then
        if [[ "${DEV}" == "1" ]]; then
            DEV_PATH="${REPO_ROOT}"
        else
            DEV_PATH="${DEV}"
        fi
        if [[ ! -f "${DEV_PATH}/pyproject.toml" ]]; then
            echo "[run.sh] error: --dev path '${DEV_PATH}' does not look like a Forgather checkout (no pyproject.toml)" >&2
            exit 2
        fi
        VOLUME_ARGS+=(-v "${DEV_PATH}:/opt/forgather/repo")
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
        echo "[run.sh]   state:   ${STATE_VOLUME} -> /home/forgather/.config/forgather" >&2
    else
        echo "[run.sh]   state:   <ephemeral; auth token will not persist across docker rm>" >&2
    fi
    if [[ -n "${EXTRA_MOUNTS}" ]]; then
        echo "[run.sh]   extra:   ${EXTRA_MOUNTS}" >&2
    fi
    if [[ -n "${DEV_PATH}" ]]; then
        cat >&2 <<EOF

[run.sh] *** WARNING: DEV mode is enabled ***
[run.sh]   bind-mount: ${DEV_PATH} -> /opt/forgather/repo
[run.sh]
[run.sh]   The runtime image is intended to be IMMUTABLE and IDENTICAL
[run.sh]   across a distribution. The supported deployment model is
[run.sh]   "build once, distribute, run on N nodes" — host-side edits
[run.sh]   bypass that model entirely.
[run.sh]
[run.sh]   Use --dev / DEV= to test a fix WITHOUT rebuilding the image.
[run.sh]   For production, rebuild and redistribute the image instead.

EOF
    fi
    if [[ -n "${CLUSTER}" ]]; then
        if [[ -n "${CLUSTER_ADDRESS}" ]]; then
            echo "[run.sh]   cluster: ${CLUSTER} +address ${CLUSTER_ADDRESS}" >&2
        else
            echo "[run.sh]   cluster: ${CLUSTER}" >&2
        fi
        if [[ "${NETWORK}" != "host" ]]; then
            cat >&2 <<EOF
[run.sh] warning: CLUSTER is set but NETWORK=${NETWORK}. Forgather's
[run.sh]   cluster discovery uses mDNS / multicast, which doesn't
[run.sh]   traverse docker bridge networks. Peers on other nodes
[run.sh]   will not see this server. Use NETWORK=host for multi-node.
[run.sh]   See docs/guides/multi-node-training.md.
EOF
        fi
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

    # When CLUSTER or NO_AUTH is set, override the image's default
    # CMD so the server starts with the right flags. We replicate
    # the default arguments (-H 0.0.0.0 -p 8765) so operators don't
    # have to think about the base CMD; they only care about
    # --cluster / address / auth.
    CMD_ARGS=()
    if [[ -n "${CLUSTER}" || -n "${NO_AUTH}" ]]; then
        CMD_ARGS=(forgather server -H 0.0.0.0 -p 8765)
        if [[ -n "${CLUSTER}" ]]; then
            CMD_ARGS+=(--cluster "${CLUSTER}")
            if [[ -n "${CLUSTER_ADDRESS}" ]]; then
                CMD_ARGS+=(--cluster-address "${CLUSTER_ADDRESS}")
            fi
        fi
        if [[ -n "${NO_AUTH}" ]]; then
            CMD_ARGS+=(--no-auth)
            echo "[run.sh]   auth: DISABLED (NO_AUTH=${NO_AUTH}); trusted-LAN only" >&2
        fi
    fi

    # ``--init`` puts Docker's bundled tini in front of the entrypoint
    # so PID 1 properly reaps orphan grandchildren. The Forgather
    # server spawns training subprocesses (torchrun + workers) that can
    # outlive their immediate parent on a hung save-stop or crash; if
    # they get re-parented to PID 1 and nobody waitpid()s them, they
    # pile up as zombies. The dev image's run wrapper passes --init
    # for the same reason. See docs/guides/multi-node-training.md for
    # the full rationale (the multi-node hang debug session is where
    # this shape of bug surfaces).
    docker run -d \
        --init \
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
        "${IMAGE}" \
        "${CMD_ARGS[@]}" > /dev/null

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

# ---------- pre-parse: --dev [PATH] -------------------------------
# Recognized anywhere in the argv (typical positions: before
# `--recreate` or alone). Sets DEV; the rest of the dispatch logic
# operates on the unchanged subcommand position (--recreate, --shell,
# etc.). DEV may also have come from the env or the docker.env config
# file — the flag just provides a CLI shortcut.
PARSED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            shift
            if [[ $# -gt 0 && "$1" != --* ]]; then
                DEV="$1"
                shift
            else
                DEV=1
            fi
            ;;
        *)
            PARSED_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${PARSED_ARGS[@]}"

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
        sed -n '2,113p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        ;;
    "")
        lib_ensure_running
        ;;
    *)
        echo "unknown subcommand: $1 (try --help)" >&2
        exit 2
        ;;
esac
