#!/bin/bash
# Entrypoint for the Forgather development image.
#
# The image carries every Forgather dependency in its venv but NOT
# the Forgather package itself — there is no in-image copy of the
# repo. On first container start (or whenever FORGATHER_REPO points
# at a different tree than last time) we install the package in
# editable mode against the bind-mounted host clone, so host-side
# edits show up live without rebuilding the image.

set -e

export VIRTUAL_ENV=/opt/forgather/venv
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

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
