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


# ---------------------------------------------------------------------------
# Filesystem-root allowlist (jupyter-lab-style chroot for path-accepting APIs)
# ---------------------------------------------------------------------------
#
# Set by ``configure_fs_roots`` at startup (via ``--fs-root``). When the tuple
# is empty, every path is allowed and ``is_path_in_fs_root`` is a no-op —
# matching the historical "browse anywhere the server uid can read" behaviour.
# When populated, every path-accepting API handler that takes a client-supplied
# path is expected to call ``is_path_in_fs_root`` and 403 on miss.
#
# Roots are stored as fully-resolved absolute paths so the containment check
# is a straightforward ``Path.relative_to`` after resolving the candidate.
# Symlink-following is intentional here: callers that need to reject symlink
# *components* (vs. just verify the resolved target is under a root) layer
# the symlink-chain check on top — see fs.py:_reject_symlink_in_chain.
_fs_roots: tuple[Path, ...] = ()


def configure_fs_roots(roots) -> None:
    """Install the fs-root allowlist. Pass an empty list to disable."""
    global _fs_roots
    resolved: list[Path] = []
    for r in roots:
        try:
            p = Path(os.path.expanduser(str(r))).resolve()
        except (OSError, RuntimeError) as e:
            log.warning("ignoring unresolvable fs-root %r: %s", r, e)
            continue
        if not p.is_dir():
            log.warning("ignoring fs-root that isn't a directory: %s", p)
            continue
        resolved.append(p)
    # De-duplicate while preserving order; drop any root that's a descendant
    # of another (the ancestor already covers it).
    deduped: list[Path] = []
    for p in resolved:
        if any(_is_descendant(p, anc) for anc in deduped):
            continue
        deduped = [d for d in deduped if not _is_descendant(d, p)]
        deduped.append(p)
    _fs_roots = tuple(deduped)
    if _fs_roots:
        log.info(
            "fs-root allowlist active (%d root(s)): %s",
            len(_fs_roots),
            ", ".join(str(p) for p in _fs_roots),
        )


def fs_roots() -> tuple[Path, ...]:
    """Return the configured fs-roots tuple (empty = unrestricted)."""
    return _fs_roots


def fs_roots_active() -> bool:
    """True if a non-empty fs-root allowlist is configured."""
    return len(_fs_roots) > 0


def _is_descendant(candidate: Path, ancestor: Path) -> bool:
    try:
        candidate.relative_to(ancestor)
        return True
    except ValueError:
        return False


def is_path_in_fs_root(path) -> bool:
    """True if ``path`` resolves to a descendant of some configured root.

    Always True when no allowlist is configured. Returns False for paths
    that can't be resolved (broken symlink, permission error during
    realpath, etc.) — failing closed is the right default once an
    allowlist is in force.

    The candidate is realpath'd before the check so symlink targets are
    what's evaluated. Callers that also want to reject symlink *components*
    (e.g. the delete path) should layer ``_reject_symlink_in_chain`` on
    top; this function only answers "where does this path actually land."
    """
    if not _fs_roots:
        return True
    try:
        resolved = Path(os.path.expanduser(str(path))).resolve()
    except (OSError, RuntimeError):
        return False
    return any(_is_descendant(resolved, root) for root in _fs_roots)


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


def dataset_server_registry_file() -> Path:
    """User-added dataset_server URLs + tokens.

    Lives at ``<config>/server/dataset_server_registry.json``, mode 0600.
    Entries are persistent across restarts; the per-job token files for
    server-spawned instances are a separate concern.
    """
    return server_state_dir() / "dataset_server_registry.json"


def inference_server_registry_file() -> Path:
    """User-added inference-server URLs + tokens.

    Lives at ``<config>/server/inference_server_registry.json``, mode
    0600. Same shape as the dataset_server registry — see
    :mod:`forgather_server.inference_server_registry`.
    """
    return server_state_dir() / "inference_server_registry.json"


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
    "sessions.json",
    "dataset_server_registry.json",
    "inference_server_registry.json",
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
