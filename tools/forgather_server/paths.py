"""Filesystem locations used by the Forgather server for persistent state.

All runtime state (search roots, queue, TB registry, captured TTY) lives under
``<forgather_config_dir>/server/`` (on Linux, ``~/.config/forgather/server/``)
so that the server can crash, restart, or be upgraded without losing user
data. Everything here is plain JSON or log files so the user can inspect or
edit state with ordinary tools.

Directories holding sensitive state (auth token, password hash, queue,
overrides, captured TTYs) are chmod'd to ``0o700`` on every access. Other
local users on the same host should not be able to read or modify the
server's persisted state — loopback ports are not isolated by uid so the
filesystem is the next defensive layer.
"""

import logging
import os
import stat
from pathlib import Path

from forgather.preprocess import forgather_config_dir

log = logging.getLogger("forgather_server.paths")


def _tighten_dir(path: Path, mode: int = 0o700) -> None:
    """Best-effort chmod; idempotent and safe to call every access."""
    try:
        os.chmod(path, mode)
    except OSError as e:
        log.warning("could not chmod %s to %o: %s", path, mode, e)


def server_state_dir() -> Path:
    home = Path(forgather_config_dir())
    d = home / "server"
    d.mkdir(parents=True, exist_ok=True)
    _tighten_dir(home, 0o700)
    _tighten_dir(d, 0o700)
    return d


def search_roots_file() -> Path:
    return server_state_dir() / "search_roots.json"


def queue_file() -> Path:
    return server_state_dir() / "queue.json"


def jobs_tty_dir() -> Path:
    d = server_state_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    _tighten_dir(d, 0o700)
    return d


def overrides_dir() -> Path:
    d = server_state_dir() / "overrides"
    d.mkdir(parents=True, exist_ok=True)
    _tighten_dir(d, 0o700)
    return d


def gpu_policy_file() -> Path:
    return server_state_dir() / "gpu_policy.json"


def auth_token_file() -> Path:
    """Persistent bearer token shared between the server and CLI clients.

    Mode 0600. Generated lazily on first server start. Plain ASCII so
    users can ``cat`` it.
    """
    return server_state_dir() / "auth_token"


def inference_tokens_dir() -> Path:
    """Per-job bearer tokens for spawned inference servers.

    One file per queue_id, mode 0600. Lets the spawn pass the token via
    ``--auth-token-file`` instead of argv (where it'd be visible to any
    local user via ``ps``).
    """
    d = server_state_dir() / "inference"
    d.mkdir(parents=True, exist_ok=True)
    _tighten_dir(d, 0o700)
    return d


def inference_token_file(queue_id: str) -> Path:
    """Path to the per-job inference auth token (0600)."""
    return inference_tokens_dir() / f"{queue_id}.token"


def dataset_server_tokens_dir() -> Path:
    """Per-job bearer tokens for spawned dataset_server instances.

    Mirrors ``inference_tokens_dir`` — one file per queue_id, mode 0600,
    passed to the spawn via ``--auth-token-file`` so the token never
    lands in argv.
    """
    d = server_state_dir() / "dataset_server"
    d.mkdir(parents=True, exist_ok=True)
    _tighten_dir(d, 0o700)
    return d


def dataset_server_token_file(queue_id: str) -> Path:
    """Path to the per-job dataset_server auth token (0600)."""
    return dataset_server_tokens_dir() / f"{queue_id}.token"


def dataset_server_registry_file() -> Path:
    """User-added dataset_server URLs + tokens.

    Lives at ``<config>/server/dataset_server_registry.json``, mode 0600.
    Entries are persistent across restarts; the per-job token files for
    server-spawned instances are a separate concern.
    """
    return server_state_dir() / "dataset_server_registry.json"


def cluster_state_dir() -> Path:
    """Persistent directory for multi-node cluster state.

    Lives at ``<forgather_config_dir>/cluster/`` (peer of ``server/``) so
    that the cluster identity outlives any individual server instance and
    multiple servers on the same host could in principle share one
    node identity. Mode 0700 like the rest of the user state.
    """
    home = Path(forgather_config_dir())
    d = home / "cluster"
    d.mkdir(parents=True, exist_ok=True)
    _tighten_dir(home, 0o700)
    _tighten_dir(d, 0o700)
    return d


def cluster_node_id_file() -> Path:
    """Persistent UUID identifying this node within any cluster.

    Mode 0600. Generated lazily at first cluster-enabled startup. The
    UUID is stable across restarts; rotating it effectively makes the
    node look like a brand-new peer to the rest of the cluster.
    """
    return cluster_state_dir() / "node_id"


def cluster_journal_dir() -> Path:
    """Append-only journal of global-state mutations (Phase 4 seam)."""
    d = cluster_state_dir() / "journal"
    d.mkdir(parents=True, exist_ok=True)
    _tighten_dir(d, 0o700)
    return d


def password_hash_file() -> Path:
    """Optional pbkdf2_sha256 password hash for browser logins.

    Mode 0600. Format: ``pbkdf2_sha256$<iters>$<salt-hex>$<hash-hex>``.
    """
    return server_state_dir() / "password_hash"


# Files known to live directly under ``<forgather_config_dir>/server/`` that
# may carry secrets or per-user job metadata. Anything looser than 0600 gets
# tightened on startup; legacy installs may have shipped 0644 here before
# the chmod-on-write fix landed.
_SENSITIVE_TOPLEVEL_FILES = (
    "auth_token",
    "password_hash",
    "queue.json",
    "job_records.json",
    "gpu_policy.json",
    "search_roots.json",
)


def tighten_existing_state_perms() -> None:
    """Best-effort startup migration for legacy installs.

    Earlier versions of the server inherited the umask for state files
    and directories. Once we shipped the chmod-on-write path, new files
    are 0600 — but a long-running install still has 0644 files on disk.
    Walk the known set and tighten anything looser than 0600. Errors are
    logged at WARNING and never raised; this is opportunistic cleanup.
    """
    home = Path(forgather_config_dir())
    server = home / "server"
    for d in (home, server):
        if d.is_dir():
            _tighten_dir(d, 0o700)

    if not server.is_dir():
        return

    candidates = [server / name for name in _SENSITIVE_TOPLEVEL_FILES]
    overrides = server / "overrides"
    if overrides.is_dir():
        _tighten_dir(overrides, 0o700)
        candidates.extend(overrides.glob("*.json"))

    for f in candidates:
        try:
            st = f.stat()
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        if stat.S_IMODE(st.st_mode) & 0o077:
            try:
                os.chmod(f, 0o600)
            except OSError as e:
                log.warning("could not chmod %s to 0600: %s", f, e)
