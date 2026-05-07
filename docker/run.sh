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
#
# Persistent overrides: if $XDG_CONFIG_HOME/forgather/docker.env (or
# ~/.config/forgather/docker.env) exists it is sourced before defaults
# are applied. Use `:= ` so a command-line `VAR=... docker/run.sh`
# still wins:
#
#   # ~/.config/forgather/docker.env
#   : "${EXTRA_MOUNTS:=-v /mnt/rust:/mnt/rust}"
#   : "${GPUS:=all}"
#
# Override the path with FORGATHER_DOCKER_CONFIG.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_FILE="${FORGATHER_DOCKER_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/forgather/docker.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
fi

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

check_uncovered_symlinks() {
    # Two checks:
    #
    # 1. FATAL: if REPO_ROOT itself resolves through a symlink to a
    #    path outside the bind-mounted roots, Docker will fail at
    #    workdir setup with a confusing "mkdir: file exists" OCI
    #    error. Bail with a clear message instead.
    #
    # 2. WARNING: $HOME-rooted symlinks at depth <= 3 whose targets
    #    resolve outside the bind-mounted roots. Non-fatal — those
    #    links will dangle inside the container but only matter if
    #    the user actually dereferences them.
    local roots=("${HOME}") prev="" tok r
    for tok in ${EXTRA_MOUNTS}; do
        case "${prev}" in
            -v|--volume) roots+=("${tok%%:*}") ;;
        esac
        prev="${tok}"
    done

    is_covered() {
        local p="$1" root
        for root in "${roots[@]}"; do
            if [[ "${p}" == "${root}" || "${p}" == "${root}"/* ]]; then
                return 0
            fi
        done
        return 1
    }

    suggest_root() {
        # First two path components (or one if shallower).
        echo "$1" | awk -F/ 'NF>=3 {print "/" $2 "/" $3} NF<3 {print "/" $2}'
    }

    # ---- 1. fatal: REPO_ROOT resolves outside covered mounts ----
    local repo_real
    repo_real="$(readlink -f -- "${REPO_ROOT}" 2>/dev/null || true)"
    if [[ -n "${repo_real}" ]] && ! is_covered "${repo_real}"; then
        local repo_top
        repo_top="$(suggest_root "${repo_real}")"
        {
            echo "[run.sh] error: forgather repo path resolves outside the bind-mount:"
            echo "[run.sh]   ${REPO_ROOT}"
            echo "[run.sh]   -> ${repo_real}"
            echo "[run.sh] without a bind-mount covering the target, the container's"
            echo "[run.sh] workdir will fail to resolve. add the target root to EXTRA_MOUNTS:"
            echo "[run.sh]   EXTRA_MOUNTS=\"-v ${repo_top}:${repo_top}\" docker/run.sh --recreate"
        } >&2
        exit 2
    fi

    # ---- 2. warn-only: other dangling symlinks under $HOME ----
    local entries
    entries="$(
        find "${HOME}" -maxdepth 3 -type l -lname '/*' 2>/dev/null \
        | while IFS= read -r link; do
            target="$(readlink -f -- "${link}" 2>/dev/null)" || continue
            [[ -z "${target}" ]] && continue
            case "${target}" in
                /proc/*|/sys/*|/dev/*|/etc/*|/usr/*|/var/*|/run/*) continue ;;
                /bin/*|/sbin/*|/lib/*|/lib32/*|/lib64/*|/tmp/*|/boot/*) continue ;;
            esac
            covered=0
            for r in "${roots[@]}"; do
                if [[ "${target}" == "${r}" || "${target}" == "${r}"/* ]]; then
                    covered=1
                    break
                fi
            done
            [[ "${covered}" -eq 1 ]] && continue
            root="$(echo "${target}" | awk -F/ 'NF>=3 {print "/" $2 "/" $3} NF<3 {print "/" $2}')"
            printf '%s\t%s\n' "${root}" "${link}"
        done | sort -u
    )"

    [[ -z "${entries}" ]] && return 0

    local mounts
    mounts="$(printf '%s\n' "${entries}" \
        | awk -F'\t' '!seen[$1]++ { printf " -v %s:%s", $1, $1 } END { print "" }' \
        | sed 's/^ //')"

    {
        echo "[run.sh] warning: \$HOME-rooted symlinks resolve outside the bind-mount:"
        printf '%s\n' "${entries}" \
            | awk -F'\t' '!shown[$1]++ { printf "[run.sh]   %s   (e.g. %s)\n", $1, $2 }'
        echo "[run.sh] those links will dangle inside the container."
        echo "[run.sh] add the target roots to EXTRA_MOUNTS so paths keep resolving:"
        echo "[run.sh]   EXTRA_MOUNTS=\"${mounts}\" docker/run.sh --recreate"
    } >&2
}

create_container() {
    check_uncovered_symlinks

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
    #
    # ``--init`` puts Docker's bundled tini in front of the entrypoint
    # so PID 1 properly reaps orphan grandchildren. Without it, when
    # torchrun gets killed and its worker subprocesses get re-parented
    # to PID 1 (= sleep), nobody calls wait() on them and they pile up
    # as zombies. Operators ran into exactly this on the multi-node
    # cluster after a hung save-stop. tini intercepts SIGCHLD and
    # waitpid()s for any orphan, regardless of whether it's a child
    # of a tracked process — solving the problem at the layer that
    # can actually see those orphans (the Forgather server itself
    # cannot, because the orphans are no longer its children).
    docker run -d \
        --init \
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
        # Validate before destructive removal so a bad config doesn't
        # leave the user with no container at all.
        check_uncovered_symlinks
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
