#!/bin/bash
# Entrypoint for the Forgather runtime image.
#
# Runs as root, remaps the in-container `forgather` user to PUID/PGID
# (defaulting to 1000:1000 if unset), fixes ownership of the
# in-image state dirs, then drops privileges via gosu and execs the
# real command (CMD = `forgather server -H 0.0.0.0 -p 8765`).
#
# This pattern (linuxserver.io style) lets a single prebuilt image
# work for any host user. The remap is what makes a bind-mounted
# host `~/.cache/huggingface` writable by the in-container user
# without breaking host-side ownership.
#
# If the container is launched with `--user uid:gid` (so we're not
# running as root to begin with), we skip the remap entirely and
# just exec — the caller has already done the UID dance themselves.

set -e

REPO_DIR="${REPO_DIR:-/opt/forgather/repo}"
VENV_DIR="${VENV_DIR:-/opt/forgather/venv}"
USER_NAME="${USER_NAME:-forgather}"

# ---- non-root branch: caller used `docker run --user uid:gid` ----
if [[ "$(id -u)" != "0" ]]; then
    # No remap possible (we're not root), just make sure HOME points
    # somewhere writable so the server can create ~/.forgather/server
    # and ~/.cache/huggingface. /tmp is the safe fallback; the run
    # wrapper would normally have set HOME explicitly.
    if [[ -z "${HOME:-}" || ! -w "${HOME}" ]]; then
        export HOME=/tmp
    fi
    exec "$@"
fi

# ---- root branch: optionally remap, then drop privileges ----
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
if [[ "${current_uid}" != "${PUID}" || "${current_gid}" != "${PGID}" ]]; then
    # Only chown the in-image directories — never the bind-mounted
    # HF cache or state volume. The remapped UID already matches
    # the host user that owns those, so chowning a populated cache
    # is both pointless and slow.
    chown -R "${PUID}:${PGID}" "/home/${USER_NAME}" /opt/forgather
fi

# Pre-create the persistent state dirs so the server doesn't fail
# trying to mkdir into an empty named volume. Using `install` so
# ownership and mode are correct on first creation; harmless on
# subsequent runs (existing dirs are left alone, just chmod'd).
install -d -o "${USER_NAME}" -g "${USER_NAME}" -m 0700 \
    "/home/${USER_NAME}/.forgather" \
    "/home/${USER_NAME}/.forgather/server" \
    "/home/${USER_NAME}/.cache" \
    "/home/${USER_NAME}/.cache/huggingface"

# Make sure HOME is set for the dropped-privilege process — gosu
# does NOT set it (unlike `su -`). The server reads $HOME directly
# via Path.home() in places.
exec gosu "${USER_NAME}" env \
    HOME="/home/${USER_NAME}" \
    VIRTUAL_ENV="${VENV_DIR}" \
    PATH="${VENV_DIR}/bin:${PATH}" \
    "$@"
