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
# CUDA driver / wheel sanity probe.
#
# PyTorch wheels carry their own CUDA runtime, but if the host's NVIDIA
# driver is older than what the wheel needs, you get an opaque CUDA
# error tens of minutes into a training run. A 1-line nvidia-smi probe
# at container start makes this fast to spot. The probe is non-fatal:
# operators run CPU-only sometimes, the dev image's GPUS=none path is
# supported, and the runtime image has no business refusing to boot
# just because no GPU is visible.
#
# Runs in phase 1 (root) so PATH covers /usr/bin where nvidia-smi
# normally lives. Runs only on the first entry — the gosu re-exec
# sets FORGATHER_ENTRYPOINT_PHASE=2, which suppresses the second
# print.
# ----------------------------------------------------------------------
if [[ "${FORGATHER_ENTRYPOINT_PHASE:-}" != "2" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[forgather-entrypoint] nvidia-smi: not available; container cannot see a GPU (CPU-only mode)" >&2
    elif ! _nv_out="$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>&1)"; then
        echo "[forgather-entrypoint] nvidia-smi: failed to run; container cannot see a GPU (CPU-only mode)" >&2
        echo "[forgather-entrypoint]   ${_nv_out}" >&2
    else
        # Count non-empty lines = visible devices; first column of any
        # row is the driver version. nvidia-smi prints one row per GPU.
        _nv_count="$(printf '%s\n' "${_nv_out}" | grep -c .)"
        if [[ "${_nv_count}" -eq 0 ]]; then
            echo "[forgather-entrypoint] nvidia-smi: 0 CUDA devices visible (was --gpus passed?)" >&2
        else
            _nv_driver="$(printf '%s\n' "${_nv_out}" | head -1 | awk -F', ' '{print $1}')"
            echo "[forgather-entrypoint] nvidia-smi: driver=${_nv_driver}, ${_nv_count} device(s) visible" >&2
        fi
    fi
    unset _nv_out _nv_count _nv_driver
fi

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

    # ONLY change the uid — keep the primary gid at the build-time
    # value (gid 1000). The venv at /opt/forgather is owned by gid
    # 1000 from the build, with mode g+rwX baked in by the
    # Dockerfile, so leaving the user in gid 1000 keeps the venv
    # accessible without any runtime chown. A previous version of
    # this entrypoint changed the primary gid via ``groupmod`` (or
    # ``usermod -g``) and then recursive-chowned /opt/forgather to
    # the new gid; that runs over thousands of venv files (PyTorch
    # alone drops several thousand) and adds tens of seconds to
    # every container start when host UID != 1000. Files written by
    # the user to bind-mounted host paths land as uid=PUID gid=1000;
    # the host sees the uid (correct, accountable), and the gid is
    # cosmetic for most operators.
    if [[ "${current_uid}" != "${PUID}" ]]; then
        usermod -o -u "${PUID}" "${USER_NAME}"
    fi

    # Fix ownership of the in-image home only — small (just shell
    # init files and the welcome banner state) so this stays cheap
    # even when remapped. /opt/forgather is intentionally NOT
    # touched here; see the chmod g+rwX in the Dockerfiles.
    if [[ "${current_uid}" != "${PUID}" ]]; then
        chown -R "${PUID}:${current_gid}" "/home/${USER_NAME}"
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
    # Drop build-time env that points at root-owned paths so phase 2
    # (the post-gosu-drop user) doesn't try to write there.
    # ``UV_CACHE_DIR=/root/.cache/uv`` is set in the Dockerfile so the
    # build-time uv install hits the BuildKit cache mount; at runtime
    # the unprivileged user can't write to /root, so the editable
    # reinstall fails with "Failed to initialize cache". Unsetting
    # lets uv fall back to its XDG default (``~/.cache/uv``).
    exec env -u UV_CACHE_DIR gosu "${USER_NAME}" env \
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
