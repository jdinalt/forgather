"""Unit tests for ``scheduler._resolve_dataset_server_token``.

The reuse-vs-mint behavior is the core fix in PR #25: webui-launched
dataset_servers should reuse the same per-port persisted token across
restarts so remote training peers don't have to refetch credentials.
``regen_token=True`` is the operator escape hatch to rotate.

These tests pin the contract by redirecting ``forgather_config_dir`` to
a ``tmp_path`` — the standalone_token_file path resolves underneath it,
so we never touch real user state.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _patch_config_dir(tmp_path: Path):
    """Point both the dataset_server auth module and the scheduler's
    re-import at a temp dir. Two patches are needed because both modules
    captured ``forgather_config_dir`` at import time.
    """
    return patch(
        "dataset_server.auth.forgather_config_dir",
        return_value=str(tmp_path),
    )


def test_first_spawn_mints_and_persists(tmp_path):
    """No existing per-port file -> mint + write a 0600 file."""
    from forgather_server import scheduler
    from dataset_server.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        token = scheduler._resolve_dataset_server_token(port=18766, regen=False)
        path = standalone_token_file(18766)

    assert len(token) == 64  # secrets.token_hex(32) -> 64 hex chars
    assert path.is_file()
    assert path.read_text().strip() == token
    # write_standalone_token enforces 0600
    assert (path.stat().st_mode & 0o777) == 0o600


def test_second_spawn_reuses_persisted_token(tmp_path):
    """An existing non-empty per-port file -> reuse its contents."""
    from forgather_server import scheduler
    from dataset_server.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        first = scheduler._resolve_dataset_server_token(port=18767, regen=False)
        second = scheduler._resolve_dataset_server_token(port=18767, regen=False)
        path = standalone_token_file(18767)

    assert first == second, "reuse should return the same token across calls"
    assert path.read_text().strip() == first, "file content shouldn't change on reuse"


def test_regen_rotates_existing_token(tmp_path):
    """``regen=True`` always mints + overwrites, even when a file exists."""
    from forgather_server import scheduler
    from dataset_server.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        original = scheduler._resolve_dataset_server_token(port=18768, regen=False)
        rotated = scheduler._resolve_dataset_server_token(port=18768, regen=True)
        path = standalone_token_file(18768)

    assert original != rotated, "regen should mint a fresh token"
    assert path.read_text().strip() == rotated, "regen should overwrite the file"


def test_empty_file_is_treated_as_missing(tmp_path):
    """A zero-byte token file shouldn't lock the server out — mint fresh."""
    from forgather_server import scheduler
    from dataset_server.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        path = standalone_token_file(18769)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")  # corrupt / empty
        token = scheduler._resolve_dataset_server_token(port=18769, regen=False)

    assert len(token) == 64
    assert path.read_text().strip() == token


def test_whitespace_only_file_is_treated_as_missing(tmp_path):
    """A token file with just whitespace also counts as missing."""
    from forgather_server import scheduler
    from dataset_server.auth import standalone_token_file

    with _patch_config_dir(tmp_path):
        path = standalone_token_file(18770)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("   \n")
        token = scheduler._resolve_dataset_server_token(port=18770, regen=False)

    assert token.strip()
    assert path.read_text().strip() == token


def test_persist_failure_still_returns_token(tmp_path, monkeypatch, caplog):
    """If write_standalone_token raises OSError (e.g. read-only home),
    the spawn must still get a valid token. The next start won't
    auto-discover it, but failing the spawn would be worse."""
    from forgather_server import scheduler

    def _raise(*args, **kwargs):
        raise OSError("simulated read-only filesystem")

    with _patch_config_dir(tmp_path):
        monkeypatch.setattr(
            scheduler, "dataset_server_write_standalone_token", _raise
        )
        with caplog.at_level("WARNING"):
            token = scheduler._resolve_dataset_server_token(port=18771, regen=False)

    assert len(token) == 64
    assert any(
        "could not persist dataset_server token" in rec.message
        for rec in caplog.records
    )
