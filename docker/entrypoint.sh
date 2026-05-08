#!/bin/bash
# Unified entrypoint for the Forgather dev and runtime images.
#
# Both images share the same gosu-drop scaffolding (run as root,
# optionally remap an in-container user to PUID/PGID, drop privileges
# via gosu, exec the real command). They differ in two small ways:
#
#   - dev image: $FORGATHER_REPO points at a bind-mounted host clone.
#     We re-install the package in editable mode against that path
#     after the gosu drop, so host-side edits show up live.
#
#   - runtime image: source tree is baked in, $FORGATHER_REPO is
#     unset, no editable install needed.
#
# The mode is selected purely by whether FORGATHER_REPO is set —
# there's no `--mode runtime` flag. Both Dockerfiles install this one
# script as the entrypoint.
#
# If the container was launched with `docker run --user uid:gid` (so
# we're not root), the remap is skipped and we go straight to phase 2.

set -e

USER_NAME="${USER_NAME:-forgather}"
VENV_DIR="${VENV_DIR:-/opt/forgather/venv}"

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

# ----------------------------------------------------------------------
# Phase 1 (root): optionally remap the in-container user, chown the
# in-image state, drop privileges via gosu, re-exec ourselves under
# FORGATHER_ENTRYPOINT_PHASE=2 so phase 2 runs in the same script.
# ----------------------------------------------------------------------
if [[ "${FORGATHER_ENTRYPOINT_PHASE:-}" != "2" && "$(id -u)" == "0" ]]; then
    PUID="${PUID:-1000}"
    PGID="${PGID:-1000}"

    current_uid="$(id -u "${USER_NAME}")"
    current_gid="$(id -g "${USER_NAME}")"

    if [[ "${current_gid}" != "${PGID}" ]]; then
        groupmod -o -g "${PGID}" "${USER_NAME}"
    fi
    if [[ "${current_uid}" != "${PUID}" ]]; then
        usermod -o -u "${PUID}" -g "${PGID}" "${USER_NAME}"
    fi

    # Fix ownership of in-image state IFF we actually remapped. Skipped
    # on the common path (PUID/PGID == 1000) so cold start is fast.
    # Only the in-image directories get chowned — never bind-mounted
    # host paths (the remapped UID already matches the host owner, and
    # chowning a populated host home recursively is both pointless and
    # potentially huge).
    if [[ "${current_uid}" != "${PUID}" || "${current_gid}" != "${PGID}" ]]; then
        chown -R "${PUID}:${PGID}" "/home/${USER_NAME}" /opt/forgather
    fi

    # Pre-create the persistent state dirs so the server (runtime
    # image) doesn't fail trying to mkdir into an empty named volume.
    # Harmless in dev mode (touches the unused /home/dev tree); the
    # dev image's HOME is bind-mounted from the host elsewhere.
    install -d -o "${USER_NAME}" -g "${USER_NAME}" -m 0700 \
        "/home/${USER_NAME}/.forgather" \
        "/home/${USER_NAME}/.forgather/server" \
        "/home/${USER_NAME}/.cache" \
        "/home/${USER_NAME}/.cache/huggingface"

    # Make sure HOME is set for the dropped-privilege process — gosu
    # does NOT set it (unlike `su -`). Default to the in-image home
    # for the remapped user; the dev image's run wrapper passes a
    # bind-mounted host home via -e HOME=<host home>, which wins here.
    : "${HOME:=/home/${USER_NAME}}"

    export FORGATHER_ENTRYPOINT_PHASE=2
    exec gosu "${USER_NAME}" env \
        HOME="${HOME}" \
        VIRTUAL_ENV="${VENV_DIR}" \
        PATH="${VENV_DIR}/bin:${PATH}" \
        FORGATHER_ENTRYPOINT_PHASE=2 \
        "$0" "$@"
fi

# ----------------------------------------------------------------------
# Phase 2: running unprivileged (either the gosu re-exec from phase 1,
# or because the container was launched with --user uid:gid). Do the
# dev-mode editable install if FORGATHER_REPO is set, then exec.
# ----------------------------------------------------------------------

if [[ -z "${HOME:-}" || ! -w "${HOME}" ]]; then
    export HOME=/tmp
fi

if [[ -n "${FORGATHER_REPO:-}" ]]; then
    # Dev image flow: re-install editable against the bind-mounted host
    # clone so host-side edits go live without rebuilding the image.
    if [[ ! -f "${FORGATHER_REPO}/pyproject.toml" ]]; then
        echo "[forgather-entrypoint] WARNING: FORGATHER_REPO=${FORGATHER_REPO} doesn't" >&2
        echo "[forgather-entrypoint]          point at a Forgather checkout. The venv has" >&2
        echo "[forgather-entrypoint]          all dependencies but NOT forgather itself;" >&2
        echo "[forgather-entrypoint]          the \`forgather\` command will not be available." >&2
    else
        # Cheap idempotency check: the .pth/dist-info created by an
        # editable install records the install location. Skip the
        # reinstall when it already points at the right tree.
        current="$(python -c '
import importlib.util, pathlib
spec = importlib.util.find_spec("forgather")
if spec and spec.origin:
    print(pathlib.Path(spec.origin).resolve().parent.parent.parent)
' 2>/dev/null || true)"

        target="$(readlink -f "${FORGATHER_REPO}")"
        if [[ "${current}" != "${target}" ]]; then
            echo "[forgather-entrypoint] Installing forgather (editable): ${target}" >&2
            uv pip install --python "${VIRTUAL_ENV}/bin/python" \
                --no-deps --quiet -e "${FORGATHER_REPO}" || \
                echo "[forgather-entrypoint] WARNING: editable install failed" >&2
        fi

        # The webui dist/ is checkout-local — docker/build.sh runs
        # ./build-webui.sh as a post-step against the host clone. If
        # dist/ is still missing here (different checkout, manual
        # build, etc.), warn so the user knows to run it themselves.
        if [[ ! -d "${FORGATHER_REPO}/tools/forgather_server/webui/dist" ]]; then
            echo "[forgather-entrypoint] NOTE: ${FORGATHER_REPO}/tools/forgather_server/webui/dist is missing." >&2
            echo "[forgather-entrypoint]       Run './build-webui.sh' from \$FORGATHER_REPO before starting the web server." >&2
        fi
    fi
fi

exec "$@"
