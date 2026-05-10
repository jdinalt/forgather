"""Dataset-server convenience endpoints.

So far this is just the "ensure the default config file exists" hook
behind the Tools menu's right-click "Edit Configuration…" item. When
the user picks that item the frontend wants an absolute path it can
hand to the editor view; if the file doesn't exist yet we create a
commented stub matching the dataset_server's documented YAML shape so
the user has a starting point rather than an empty buffer.

Path: ``<forgather_config_dir>/dataset_server/config.yaml`` — same
directory the standalone dataset_server itself reads from when
``--config`` is omitted.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from forgather.preprocess import forgather_config_dir

log = logging.getLogger("forgather_server.routes.dataset_server")
router = APIRouter(tags=["dataset_server"])


# Stub content for a freshly-created config.yaml. Mirrors the YAML
# example in tools/dataset_server/README.md; every line is commented
# so the file is "valid YAML with no overrides" until the user
# actively edits in their settings.
_STUB_CONFIG = """\
# Forgather dataset_server configuration
#
# This file is loaded automatically when `forgather dataset-server start`
# is run without `--config`. CLI flags always override file values.
# See tools/dataset_server/README.md for the full reference.

# Bind address + port. Loopback by default; switch to 0.0.0.0 for LAN.
# host: 127.0.0.1
# port: 8766
# log_level: INFO

# Auth (mutually exclusive; default = auto-generated per-port token).
# Setting `no_auth: true` disables the bearer-token gate entirely —
# only do this on a trusted network.
# no_auth: false
# auth_token_file: ~/.fdss.token

# Loading policy — all default to the safe option.
# no_hf: false              # disable HF cache loading (local/* only)
# allow_paths: false        # allow loads by absolute filesystem path
# allow_downloads: false    # allow HF downloads on cache miss

# Named local datasets. Clients request these as `local/<name>`.
# Paths must exist at server startup.
# local:
#   stories: /data/tinystories
#   mycorpus: /data/saved_corpus
"""


def _default_config_path() -> Path:
    """Same path the dataset_server itself reads at startup.

    Kept in sync with ``tools/dataset_server/server.py::default_config_file``.
    Duplicated rather than imported so the forgather_server package
    doesn't take a hard dependency on the dataset_server entry-point
    module's sys.path shenanigans.
    """
    return Path(forgather_config_dir()) / "dataset_server" / "config.yaml"


class EnsureStubResponse(BaseModel):
    path: str
    created: bool


@router.post("/dataset-server/config/ensure-stub", response_model=EnsureStubResponse)
def ensure_stub() -> EnsureStubResponse:
    """Return the absolute path of the default config; create stub if absent.

    The stub is created with 0600 perms inside a 0700 directory — same
    tightening the rest of the dataset_server's persistent state uses,
    since the file may eventually contain an auth_token_file path or
    other operator-sensitive content.
    """
    path = _default_config_path()
    if path.is_file():
        return EnsureStubResponse(path=str(path), created=False)

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError as e:
        log.warning("could not chmod %s to 0700: %s", parent, e)

    # O_EXCL guards against a TOCTOU race: two concurrent webui clicks
    # would otherwise both hit the "not is_file()" branch and one would
    # clobber the other's just-written file. EEXIST means somebody else
    # won; treat that as "already there".
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError:
        return EnsureStubResponse(path=str(path), created=False)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_STUB_CONFIG)
    except Exception:
        # Best-effort cleanup: if the write fails partway through we
        # don't want to leave an empty / half-written file that the
        # editor opens with no content.
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    log.info("created stub dataset_server config at %s", path)
    return EnsureStubResponse(path=str(path), created=True)
