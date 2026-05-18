#!/bin/bash
# Shared scaffolding for the Forgather run scripts (docker/run and
# docker/runtime/run.sh). NOT a standalone script — sourced by the
# wrappers via `. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"` (or with
# the appropriate relative path).
#
# What lives here:
#   - lib_load_config           load $XDG_CONFIG_HOME/forgather/docker.env
#                               (or ~/.config/forgather/docker.env) so
#                               operators can persist env-overrides
#   - lib_container_state       prints "running" / "stopped" / "absent"
#                               for ${NAME}
#   - lib_ensure_running        the create-or-start dispatch shared by
#                               both wrappers; calls do_create_container
#                               (image-specific, defined by the caller)
#                               on the absent path
#   - lib_handle_common_subcommand <subcommand>
#                               returns 0 (handled, caller should exit)
#                               or 1 (unhandled, caller's own logic
#                               should take over) for the subcommands
#                               the two wrappers share verbatim:
#                               --status, --stop, --rm
#
# Caller contract:
#   - Define ${NAME} (container name) before calling these.
#   - Define a `do_create_container` shell function for
#     lib_ensure_running to call. The function takes no args.
#
# Image-specific subcommands (--recreate / --logs / --shell / --token /
# --help and the wrapper-specific defaults) stay in the per-image
# script; the lib doesn't try to abstract everything.

# -------- env-override config file --------
# Source $FORGATHER_DOCKER_CONFIG (or the default
# ~/.config/forgather/docker.env) so operators can persist EXTRA_MOUNTS,
# GPUS, NETWORK, etc., across invocations. Any var the caller wants to
# pick up from the config file must use the `: "${VAR:=default}"`
# pattern *after* this returns, so a command-line `VAR=... ./run`
# (or `VAR=... runtime/run.sh`) still wins over the file.
lib_load_config() {
    local config_file="${FORGATHER_DOCKER_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/forgather/docker.env}"
    if [[ -f "${config_file}" ]]; then
        # shellcheck disable=SC1090
        source "${config_file}"
    fi
}

# -------- container state inspection --------
lib_container_state() {
    # Prints "running", "stopped", or "absent" for ${NAME}.
    local s
    s="$(docker inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null || true)"
    case "${s}" in
        running) echo running ;;
        "") echo absent ;;
        *) echo stopped ;;
    esac
}

# -------- start-or-create dispatch --------
# Caller must have defined `do_create_container`. We call it on the
# absent path; the existing-but-stopped path just `docker start`s.
#
# Callers that need to wait for the entrypoint's PUID remap before
# attaching (i.e. the runtime image, which usermods at container
# start) should call ``lib_wait_for_entrypoint_remap`` themselves
# after this returns. The dev image bakes the host user directly
# and does no runtime remap, so it doesn't need the wait.
lib_ensure_running() {
    local state
    state="$(lib_container_state)"
    case "${state}" in
        running)
            ;;
        stopped)
            echo "[${0##*/}] starting existing container ${NAME}" >&2
            docker start "${NAME}" > /dev/null
            ;;
        absent)
            do_create_container
            ;;
    esac
}

# -------- wait for the entrypoint's privilege-drop --------
# Only relevant for images whose entrypoint does a runtime UID remap
# (i.e. ``Dockerfile.runtime``). The entrypoint runs as root, does
# usermod to PUID, then execs gosu to drop privileges. ``docker run
# -d`` returns as soon as PID 1 starts, BEFORE the entrypoint has
# had a chance to run usermod — so a follow-up ``docker exec -u
# <name>`` can race against the entrypoint and resolve the username
# to its pre-remap UID. After the remap that pre-remap UID is no
# longer in /etc/passwd, leaving the attached shell at an orphaned
# UID it can't look up (whoami fails, sudo refuses, $HOME is
# inaccessible).
#
# Poll the in-image user's /etc/passwd entry via a one-shot
# ``docker exec`` until ``id -u <name>`` reports the target PUID
# — that's the unambiguous signal the entrypoint has finished
# usermod. Using docker exec (rather than reading PID 1's UID from
# /proc) keeps the check tini-agnostic: ``--init`` puts tini at PID
# 1 (always root), so PID-1-UID polling would never converge.
#
# Caller must set ``USER_NAME_IN_IMAGE`` to the build-time username
# (e.g. ``forgather`` for the runtime image) before invoking.
lib_wait_for_entrypoint_remap() {
    local user="${USER_NAME_IN_IMAGE:?USER_NAME_IN_IMAGE must be set}"
    local expected="${PUID:-$(id -u)}"
    local elapsed=0 actual
    while (( elapsed < 100 )); do  # 10s total at 100ms granularity
        actual="$(docker exec "${NAME}" id -u "${user}" 2>/dev/null || true)"
        if [[ "${actual}" == "${expected}" ]]; then
            return 0
        fi
        sleep 0.1
        elapsed=$((elapsed + 1))
    done
    echo "[${0##*/}] warning: container entrypoint did not complete UID remap within 10s — attached shell may land at the wrong UID" >&2
}

# -------- shared subcommand handler --------
# Returns 0 if the subcommand was handled (caller should exit 0).
# Returns 1 if the subcommand was not one of the shared set (caller
# falls through to its own dispatch).
lib_handle_common_subcommand() {
    case "${1:-__unset__}" in
        --status)
            echo "container: ${NAME}"
            echo "state:     $(lib_container_state)"
            if [[ "$(lib_container_state)" != "absent" ]]; then
                docker inspect -f \
                    'image:     {{.Config.Image}}{{"\n"}}network:   {{.HostConfig.NetworkMode}}{{"\n"}}started:   {{.State.StartedAt}}' \
                    "${NAME}" 2>/dev/null || true
            fi
            return 0
            ;;
        --stop)
            if [[ "$(lib_container_state)" == "running" ]]; then
                echo "[${0##*/}] stopping ${NAME}" >&2
                docker stop "${NAME}" > /dev/null
            else
                echo "[${0##*/}] container ${NAME} is not running" >&2
            fi
            return 0
            ;;
        --rm)
            if [[ "$(lib_container_state)" != "absent" ]]; then
                echo "[${0##*/}] removing ${NAME}" >&2
                docker rm -f "${NAME}" > /dev/null
            else
                echo "[${0##*/}] container ${NAME} does not exist" >&2
            fi
            return 0
            ;;
    esac
    return 1
}
