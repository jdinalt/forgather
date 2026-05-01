#!/bin/bash
# Entrypoint for the Forgather development image.
#
# Activates the baked-in venv and, if FORGATHER_REPO points at a
# bind-mounted checkout that differs from the in-image copy, re-points
# the editable install at the live source so host-side edits show up
# without rebuilding the image.

set -e

export VIRTUAL_ENV=/opt/forgather/venv
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

if [[ -n "${FORGATHER_REPO}" && -f "${FORGATHER_REPO}/pyproject.toml" ]]; then
    # Cheap idempotency check: the egg-link / .pth created by an
    # editable install records the install location. Skip the
    # reinstall when it already points at the right tree.
    current="$(python -c '
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("forgather")
if spec and spec.origin:
    print(pathlib.Path(spec.origin).resolve().parent.parent.parent)
' 2>/dev/null || true)"

    target="$(readlink -f "${FORGATHER_REPO}")"
    if [[ "${current}" != "${target}" ]]; then
        echo "[forgather-entrypoint] Re-linking editable install: ${target}" >&2
        uv pip install --python "${VIRTUAL_ENV}/bin/python" \
            --no-deps --quiet -e "${FORGATHER_REPO}" || \
            echo "[forgather-entrypoint] WARNING: editable reinstall failed; using bundled copy" >&2
    fi
fi

exec "$@"
