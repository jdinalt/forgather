"""Tests for tools/forgather_server/_gc.py."""

import time
from pathlib import Path

import forgather_server._gc as gc
import forgather_server.job_records as job_records
import forgather_server.paths as paths
import forgather_server.queue_store as queue_store
import pytest
from forgather_server.job_records import JobRecord
from forgather_server.queue_store import QueueItem


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Redirect every persistent path under tmp_path."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(job_records, "server_state_dir", lambda: state_dir)
    monkeypatch.setattr(queue_store, "queue_file", lambda: state_dir / "queue.json")

    tty_dir = state_dir / "jobs"
    tty_dir.mkdir()
    monkeypatch.setattr(paths, "jobs_tty_dir", lambda: tty_dir)
    # The _gc module imports jobs_tty_dir directly into its namespace.
    monkeypatch.setattr(gc, "jobs_tty_dir", lambda: tty_dir)
    yield state_dir


def _make_record(
    queue_id="q-001",
    *,
    tty_path=None,
    logs_dir=None,
    status="running",
    job_type="training",
):
    return JobRecord(
        queue_id=queue_id,
        project_dir="/proj",
        config="train.yaml",
        status=status,
        job_type=job_type,
        tty_log_path=str(tty_path) if tty_path else None,
        logs_dir=str(logs_dir) if logs_dir else None,
    )


def _write_tty(tty_dir: Path, queue_id: str, content: bytes = b"hello\n") -> Path:
    p = tty_dir / f"{queue_id}.tty"
    p.write_bytes(content)
    return p


class TestRelocateTtyToLogs:
    def test_happy_path(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        src = _write_tty(tty_dir, "q-1", b"output\n")
        logs_dir = tmp_path / "run" / "logs"
        record = _make_record("q-1", tty_path=src, logs_dir=logs_dir)
        job_records.add_record(record)

        new = gc.relocate_tty_to_logs(record)

        assert new == logs_dir / "tty.log"
        assert new.read_bytes() == b"output\n"
        assert not src.exists()
        # Record's tty_log_path is updated to follow the move.
        reread = job_records.get_record("q-1")
        assert reread.tty_log_path == str(new)

    def test_skips_when_no_logs_dir(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        src = _write_tty(tty_dir, "q-2")
        record = _make_record("q-2", tty_path=src, logs_dir=None)
        job_records.add_record(record)

        new = gc.relocate_tty_to_logs(record)

        assert new is None
        # Source untouched — non-training jobs keep their TTY in central dir
        # until remove_record() unlinks it.
        assert src.exists()

    def test_skips_when_tty_outside_central(self, tmp_path):
        # tty_log_path lives somewhere else (e.g. already relocated). The
        # safety guard must not move files we don't own.
        external = tmp_path / "elsewhere" / "tty.log"
        external.parent.mkdir(parents=True)
        external.write_bytes(b"existing\n")
        logs_dir = tmp_path / "run" / "logs"
        record = _make_record("q-3", tty_path=external, logs_dir=logs_dir)
        job_records.add_record(record)

        result = gc.relocate_tty_to_logs(record)

        assert result is None
        assert external.exists()
        assert not (logs_dir / "tty.log").exists()

    def test_replaces_existing_symlink(self, tmp_path):
        # Mirror what scheduler._try_link_tty does: a symlink at the
        # destination pointing back at the source. Relocation must
        # replace it with the actual file.
        tty_dir = paths.jobs_tty_dir()
        src = _write_tty(tty_dir, "q-4", b"linked\n")
        logs_dir = tmp_path / "run" / "logs"
        logs_dir.mkdir(parents=True)
        link = logs_dir / "tty.log"
        link.symlink_to(src)

        record = _make_record("q-4", tty_path=src, logs_dir=logs_dir)
        job_records.add_record(record)

        new = gc.relocate_tty_to_logs(record)

        assert new == link
        assert not link.is_symlink()
        assert link.read_bytes() == b"linked\n"
        assert not src.exists()

    def test_handles_missing_source(self, tmp_path):
        # tty_log_path points into central dir but the file's already
        # gone (race — concurrent unlink). Skip cleanly.
        tty_dir = paths.jobs_tty_dir()
        ghost = tty_dir / "q-5.tty"
        logs_dir = tmp_path / "run" / "logs"
        record = _make_record("q-5", tty_path=ghost, logs_dir=logs_dir)
        job_records.add_record(record)

        result = gc.relocate_tty_to_logs(record)

        assert result is None
        assert not (logs_dir / "tty.log").exists()


class TestSweepOrphanTtys:
    def test_keeps_files_referenced_by_record(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        keep = _write_tty(tty_dir, "q-keep")
        # Backdate well past the TTL so the only protection is the
        # live-id check.
        old = time.time() - 99999
        import os

        os.utime(keep, (old, old))

        job_records.add_record(_make_record("q-keep", tty_path=keep))

        removed = gc.sweep_orphan_ttys(ttl_seconds=60)

        assert removed == 0
        assert keep.exists()

    def test_keeps_files_referenced_by_queue(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        keep = _write_tty(tty_dir, "q-queued")
        import os

        old = time.time() - 99999
        os.utime(keep, (old, old))

        item = QueueItem(
            queue_id="q-queued",
            project_dir="/proj",
            config="train.yaml",
        )
        queue_store.add_item(item)

        removed = gc.sweep_orphan_ttys(ttl_seconds=60)

        assert removed == 0
        assert keep.exists()

    def test_keeps_recent_orphans(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        # Just-touched file with no record — protected by the TTL.
        recent = _write_tty(tty_dir, "q-fresh")

        removed = gc.sweep_orphan_ttys(ttl_seconds=3600)

        assert removed == 0
        assert recent.exists()

    def test_removes_old_orphans(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        old_orphan = _write_tty(tty_dir, "q-old")
        import os

        old = time.time() - 99999
        os.utime(old_orphan, (old, old))

        removed = gc.sweep_orphan_ttys(ttl_seconds=60)

        assert removed == 1
        assert not old_orphan.exists()

    def test_skips_non_tty_files(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        notty = tty_dir / "q-001.log"
        notty.write_bytes(b"not us")
        import os

        old = time.time() - 99999
        os.utime(notty, (old, old))

        removed = gc.sweep_orphan_ttys(ttl_seconds=60)

        assert removed == 0
        assert notty.exists()


class TestDeleteCentralTtyFor:
    def test_deletes_file_in_central(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        f = _write_tty(tty_dir, "q-del")
        record = _make_record("q-del", tty_path=f)

        ok = gc.delete_central_tty_for(record)

        assert ok is True
        assert not f.exists()

    def test_skips_file_outside_central(self, tmp_path):
        # Already-relocated file in a run dir; delete must not touch it.
        elsewhere = tmp_path / "run" / "logs" / "tty.log"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_bytes(b"keep\n")
        record = _make_record("q-keep", tty_path=elsewhere)

        ok = gc.delete_central_tty_for(record)

        assert ok is False
        assert elsewhere.exists()

    def test_skips_record_without_tty_path(self):
        record = _make_record("q-none", tty_path=None)
        assert gc.delete_central_tty_for(record) is False

    def test_handles_missing_central_file(self, tmp_path):
        tty_dir = paths.jobs_tty_dir()
        ghost = tty_dir / "q-ghost.tty"
        record = _make_record("q-ghost", tty_path=ghost)
        # File was never created — should not raise.
        assert gc.delete_central_tty_for(record) is False
