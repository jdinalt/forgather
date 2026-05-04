"""Tests for tools/forgather_server/job_records.py."""

import forgather_server.job_records as job_records
import pytest
from forgather_server.job_records import RUNNING_STATUSES, TERMINAL_STATUSES, JobRecord


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # job_records imports server_state_dir from paths; patch the local binding.
    monkeypatch.setattr(job_records, "server_state_dir", lambda: state_dir)
    yield state_dir


def _make_record(queue_id="qid-001", status="starting", **kwargs):
    defaults = dict(
        queue_id=queue_id,
        project_dir="/proj",
        config="train.yaml",
        status=status,
    )
    defaults.update(kwargs)
    return JobRecord(**defaults)


class TestAddAndList:
    def test_empty_initially(self):
        assert job_records.list_records() == []

    def test_add_then_list(self):
        r = _make_record("q1")
        job_records.add_record(r)
        records = job_records.list_records()
        assert len(records) == 1
        assert records[0].queue_id == "q1"

    def test_add_replaces_same_queue_id(self):
        r1 = _make_record("q1", project_dir="/original")
        r2 = _make_record("q1", project_dir="/replaced")
        job_records.add_record(r1)
        job_records.add_record(r2)
        records = job_records.list_records()
        assert len(records) == 1
        assert records[0].project_dir == "/replaced"


class TestGetRecord:
    def test_get_existing(self):
        r = _make_record("qid-get")
        job_records.add_record(r)
        got = job_records.get_record("qid-get")
        assert got is not None
        assert got.queue_id == "qid-get"

    def test_get_missing_returns_none(self):
        assert job_records.get_record("no-such-id") is None


class TestUpdateRecord:
    def test_update_field(self):
        r = _make_record("q-upd", status="starting")
        job_records.add_record(r)
        updated = job_records.update_record("q-upd", status="running", pid=12345)
        assert updated is not None
        assert updated.status == "running"
        assert updated.pid == 12345

    def test_update_persisted(self):
        r = _make_record("q-upd-p", status="starting")
        job_records.add_record(r)
        job_records.update_record("q-upd-p", status="running")
        reread = job_records.get_record("q-upd-p")
        assert reread.status == "running"

    def test_update_missing_returns_none(self):
        result = job_records.update_record("no-such", status="done")
        assert result is None


class TestUpdateIfNotTerminal:
    """Core CAS behavior: terminal records must not be overwritten."""

    @pytest.mark.parametrize("terminal_status", list(TERMINAL_STATUSES))
    def test_terminal_record_not_updated(self, terminal_status):
        r = _make_record("q-term", status=terminal_status)
        job_records.add_record(r)
        result = job_records.update_if_not_terminal("q-term", status="running")
        assert result is None
        # Status on disk unchanged.
        reread = job_records.get_record("q-term")
        assert reread.status == terminal_status

    @pytest.mark.parametrize("non_terminal_status", list(RUNNING_STATUSES))
    def test_non_terminal_record_updated(self, non_terminal_status):
        r = _make_record("q-live", status=non_terminal_status)
        job_records.add_record(r)
        result = job_records.update_if_not_terminal(
            "q-live", status="done", exit_code=0
        )
        assert result is not None
        assert result.status == "done"
        assert result.exit_code == 0

    def test_missing_record_returns_none(self):
        result = job_records.update_if_not_terminal("no-such", status="done")
        assert result is None


class TestRemoveRecord:
    def test_remove_existing(self):
        r = _make_record("q-rm")
        job_records.add_record(r)
        ok = job_records.remove_record("q-rm")
        assert ok is True
        assert job_records.get_record("q-rm") is None

    def test_remove_nonexistent_returns_false(self):
        assert job_records.remove_record("no-id") is False

    def test_remove_leaves_others(self):
        a = _make_record("q-a")
        b = _make_record("q-b")
        job_records.add_record(a)
        job_records.add_record(b)
        job_records.remove_record("q-a")
        assert job_records.get_record("q-b") is not None
