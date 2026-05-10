"""End-to-end perm checks: state files written through the public APIs
land at 0600 and the surrounding directories at 0700.

The auth_token / password_hash / queue / gpu_policy / overrides / search-roots
files all sit in ``~/.config/forgather/server/``, which is reachable by any other
local user with default umasks. These tests pin the chmod-on-write contract
so a regression won't silently widen perms again.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect ``forgather_config_dir`` to a fresh tmp path for the test."""
    monkeypatch.setattr(
        "forgather_server.paths.forgather_config_dir", lambda: str(tmp_path)
    )
    from forgather_server import auth

    auth._reset_sessions_for_tests()
    auth._auth_disabled = False
    yield tmp_path


class TestDirPerms:
    def test_server_state_dir_0700(self, isolated_home):
        from forgather_server import paths

        d = paths.server_state_dir()
        assert _mode(d) == 0o700
        # Parent forgather_config_dir should also be tightened.
        assert _mode(Path(isolated_home)) == 0o700

    def test_overrides_dir_0700(self, isolated_home):
        from forgather_server import paths

        d = paths.overrides_dir()
        assert _mode(d) == 0o700

    def test_jobs_tty_dir_0700(self, isolated_home):
        from forgather_server import paths

        d = paths.jobs_tty_dir()
        assert _mode(d) == 0o700


class TestStateFilePerms:
    def test_auth_token_0600(self, isolated_home):
        from forgather_server import auth, paths

        auth.generate_and_save_token()
        assert _mode(paths.auth_token_file()) == 0o600

    def test_password_hash_0600(self, isolated_home):
        from forgather_server import auth, paths

        auth.set_password("hunter2")
        assert _mode(paths.password_hash_file()) == 0o600

    def test_queue_file_0600(self, isolated_home):
        from forgather_server import paths, queue_store

        item = queue_store.QueueItem.new(
            project_dir="/p",
            config="c.yaml",
            dynamic_args={},
            requested_gpus=0,
            priority=0,
        )
        queue_store.add_item(item)
        assert _mode(paths.queue_file()) == 0o600

    def test_gpu_policy_file_0600(self, isolated_home):
        from forgather_server import gpu_policy, paths

        gpu_policy.set_policy(0, disabled=True)
        assert _mode(paths.gpu_policy_file()) == 0o600

    def test_overrides_file_0600(self, isolated_home):
        from forgather_server import overrides_store, paths

        overrides_store.set_overrides("/p", "c.yaml", {"max_steps": 1})
        files = list(paths.overrides_dir().glob("*.json"))
        assert files, "expected exactly one override file"
        assert _mode(files[0]) == 0o600

    def test_search_roots_file_0600(self, isolated_home):
        from forgather_server import paths, search_roots

        search_roots.add_root(str(isolated_home))
        assert _mode(paths.search_roots_file()) == 0o600

    def test_job_records_file_0600(self, isolated_home):
        from forgather_server import job_records, paths

        rec = job_records.JobRecord(queue_id="q_test")
        job_records.add_record(rec)
        records_file = paths.server_state_dir() / "job_records.json"
        assert _mode(records_file) == 0o600


class TestLegacyMigration:
    def test_tighten_existing_state_perms_fixes_loose_files(self, isolated_home):
        from forgather_server import paths

        server = paths.server_state_dir()
        # Simulate a legacy install: 0644 token, 0755 server dir.
        token = server / "auth_token"
        token.write_text("legacy\n")
        os.chmod(token, 0o644)
        os.chmod(server, 0o755)
        os.chmod(Path(isolated_home), 0o755)

        paths.tighten_existing_state_perms()

        assert _mode(token) == 0o600
        assert _mode(server) == 0o700
        assert _mode(Path(isolated_home)) == 0o700

    def test_tighten_existing_state_perms_idempotent(self, isolated_home):
        from forgather_server import paths

        server = paths.server_state_dir()
        token = server / "auth_token"
        token.write_text("x\n")
        os.chmod(token, 0o600)

        paths.tighten_existing_state_perms()
        paths.tighten_existing_state_perms()

        assert _mode(token) == 0o600
        assert _mode(server) == 0o700

    def test_tighten_existing_state_perms_no_server_dir(self, tmp_path, monkeypatch):
        # Should not raise even if the server dir doesn't exist yet.
        fresh = tmp_path / "fresh"
        monkeypatch.setattr(
            "forgather_server.paths.forgather_config_dir", lambda: str(fresh)
        )
        from forgather_server import paths

        paths.tighten_existing_state_perms()
