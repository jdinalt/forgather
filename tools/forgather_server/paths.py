"""Filesystem locations used by the Forgather server for persistent state.

All runtime state (search roots, queue, TB registry, captured TTY) lives under
``~/.forgather/server/`` so that the server can crash, restart, or be upgraded
without losing user data. Everything here is plain JSON or log files so the
user can inspect or edit state with ordinary tools.
"""

from pathlib import Path

from forgather.preprocess import forgather_home_dir


def server_state_dir() -> Path:
    d = Path(forgather_home_dir()) / "server"
    d.mkdir(parents=True, exist_ok=True)
    return d


def search_roots_file() -> Path:
    return server_state_dir() / "search_roots.json"


def queue_file() -> Path:
    return server_state_dir() / "queue.json"


def jobs_tty_dir() -> Path:
    d = server_state_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def overrides_dir() -> Path:
    d = server_state_dir() / "overrides"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gpu_policy_file() -> Path:
    return server_state_dir() / "gpu_policy.json"
