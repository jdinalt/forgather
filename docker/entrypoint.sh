#!/bin/bash
# Entrypoint for the Forgather development image.
#
# Runs as root, optionally remaps the in-container user to PUID/PGID
# (defaulting to 1000:1000 if unset), then drops privileges via gosu
# and execs the real command. This mirrors the runtime image's
# entrypoint pattern, so a single prebuilt dev image works for any
# host user.
#
# The image carries every Forgather dependency in its venv but NOT
# the Forgather package itself — there is no in-image copy of the
# repo. On container start (or whenever FORGATHER_REPO points at a
# different tree than last time) the package is installed in editable
# mode against the bind-mounted host clone, so host-side edits show
# up live without rebuilding the image. The editable install runs
# *after* the gosu drop, as the unprivileged user.
#
# If the container was launched with `docker run --user uid:gid` (so
# we're not root to begin with), the remap is skipped and we go
# straight to the editable install + exec.

set -e

USER_NAME="${USER_NAME:-dev}"
VENV_DIR="${VENV_DIR:-/opt/forgather/venv}"

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

# ----------------------------------------------------------------------
# Phase 1 (runs as root): optionally remap, chown the venv, gosu drop.
# Phase 2 (runs as the unprivileged user): editable-install forgather
#         against $FORGATHER_REPO, then exec "$@".
#
# We re-exec ourselves under gosu with FORGATHER_ENTRYPOINT_PHASE=2
# so the same script runs both phases without splitting into two
# files. The phase-2 path also handles the "caller used --user
# uid:gid" case (we're not root, can't remap, just install + exec).
# ----------------------------------------------------------------------

if [[ "${FORGATHER_ENTRYPOINT_PHASE:-}" != "2" && "$(id -u)" == "0" ]]; then
    # ---- root branch: remap if needed, then re-exec under gosu ----
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

    # Fix ownership of the in-image venv IFF we actually remapped.
    # Skipped on the common path (PUID/PGID == 1000) so cold start
    # is fast — the venv already has correct ownership from build.
    # Never chown the bind-mounted host home: the host user already
    # owns it, the remapped UID matches that, and a recursive chown
    # over a developer's $HOME is potentially huge and slow.
    if [[ "${current_uid}" != "${PUID}" || "${current_gid}" != "${PGID}" ]]; then
        chown -R "${PUID}:${PGID}" /opt/forgather
    fi

    # Make sure HOME is set for the dropped-privilege process — gosu
    # does NOT set it (unlike `su -`). Default to the bind-mounted
    # host home if the run wrapper passed it through; otherwise fall
    # back to the in-image home for the remapped user.
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
# Phase 2: running as the unprivileged user (either from the gosu
# re-exec above, or because the container was launched with
# `--user uid:gid`). Do the editable install, then exec.
# ----------------------------------------------------------------------

if [[ -z "${HOME:-}" || ! -w "${HOME}" ]]; then
    export HOME=/tmp
fi

if [[ -z "${FORGATHER_REPO}" || ! -f "${FORGATHER_REPO}/pyproject.toml" ]]; then
    echo "[forgather-entrypoint] WARNING: FORGATHER_REPO is unset or doesn't point" >&2
    echo "[forgather-entrypoint]          at a Forgather checkout. The venv has all" >&2
    echo "[forgather-entrypoint]          dependencies but NOT forgather itself; the" >&2
    echo "[forgather-entrypoint]          \`forgather\` command will not be available." >&2
    echo "[forgather-entrypoint]          Run via docker/run.sh, or pass" >&2
    echo "[forgather-entrypoint]          -e FORGATHER_REPO=<path-to-clone>." >&2
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

exec "$@"
