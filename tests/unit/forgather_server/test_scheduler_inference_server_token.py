"""Unit tests for ``scheduler._resolve_inference_server_token``.

Mirrors test_scheduler_dataset_server_token.py: the inference-server
token must persist per-port across restarts so a remote operator
who copied the token doesn't have to refetch it every time the
inference server bounces. ``regen_token=True`` is the operator
escape hatch to rotate.

Both modules use ``inference_server.auth_paths.standalone_token_file``
which resolves under ``forgather_config_dir`` — we redirect that to
``tmp_path`` so the tests never touch real user state.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _patch_config_dir(tmp_path: Path):
    """Point auth_paths' captured ``forgather_config_dir`` at a temp dir."""
    return patch(
        "inference_server.auth_paths.forgather_config_dir",
        return_value=str(tmp_path),
    )


def test_first_spawn_mints_and_persists(tmp_path):
    """No existing per-port file -> mint + write a 0600 file."""
    from forgather_server import scheduler
    from inference_server.auth_paths import standalone_token_file

    with _patch_config_dir(tmp_path):
        token = scheduler._resolve_inference_server_token(port=18137, regen=False)
        path = standalone_token_file(18137)

    assert len(token) == 64  # secrets.token_hex(32) -> 64 hex chars
    assert path.is_file()
    assert path.read_text().strip() == token
    # write_standalone_token enforces 0600
    assert (path.stat().st_mode & 0o777) == 0o600


def test_second_spawn_reuses_persisted_token(tmp_path):
    """An existing non-empty per-port file -> reuse its contents."""
    from forgather_server import scheduler
    from inference_server.auth_paths import standalone_token_file

    with _patch_config_dir(tmp_path):
        first = scheduler._resolve_inference_server_token(port=18138, regen=False)
        second = scheduler._resolve_inference_server_token(port=18138, regen=False)
        path = standalone_token_file(18138)

    assert first == second, "reuse should return the same token across calls"
    assert path.read_text().strip() == first, "file content shouldn't change on reuse"


def test_regen_rotates_existing_token(tmp_path):
    """``regen=True`` always mints + overwrites, even when a file exists."""
    from forgather_server import scheduler
    from inference_server.auth_paths import standalone_token_file

    with _patch_config_dir(tmp_path):
        original = scheduler._resolve_inference_server_token(port=18139, regen=False)
        rotated = scheduler._resolve_inference_server_token(port=18139, regen=True)
        path = standalone_token_file(18139)

    assert original != rotated, "regen should mint a fresh token"
    assert path.read_text().strip() == rotated, "regen should overwrite the file"


def test_empty_file_is_treated_as_missing(tmp_path):
    """A zero-byte token file shouldn't lock the server out — mint fresh."""
    from forgather_server import scheduler
    from inference_server.auth_paths import standalone_token_file

    with _patch_config_dir(tmp_path):
        path = standalone_token_file(18140)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")  # corrupt / empty
        token = scheduler._resolve_inference_server_token(port=18140, regen=False)

    assert len(token) == 64
    assert path.read_text().strip() == token


def test_whitespace_only_file_is_treated_as_missing(tmp_path):
    """A token file with just whitespace also counts as missing."""
    from forgather_server import scheduler
    from inference_server.auth_paths import standalone_token_file

    with _patch_config_dir(tmp_path):
        path = standalone_token_file(18141)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("   \n")
        token = scheduler._resolve_inference_server_token(port=18141, regen=False)

    assert token.strip()
    assert path.read_text().strip() == token


def test_persist_failure_still_returns_token(tmp_path, monkeypatch, caplog):
    """If write_standalone_token raises OSError (e.g. read-only home),
    the spawn must still get a valid token. The next start won't
    auto-discover it, but failing the spawn would be worse.

    The import inside ``_resolve_inference_server_token`` is lazy, so
    we patch the source module's symbol — the function will pick up
    the patched version when it re-imports.
    """
    from forgather_server import scheduler
    from inference_server import auth_paths

    def _raise(*args, **kwargs):
        raise OSError("simulated read-only filesystem")

    with _patch_config_dir(tmp_path):
        monkeypatch.setattr(auth_paths, "write_standalone_token", _raise)
        with caplog.at_level("WARNING"):
            token = scheduler._resolve_inference_server_token(port=18142, regen=False)

    assert len(token) == 64
    assert any(
        "could not persist inference-server token" in rec.message
        for rec in caplog.records
    )
