"""Job-visibility agent tools (list_jobs / read_job_output) + the shared
TTY-tail reader they reuse from the jobs route."""

from __future__ import annotations

import pytest

from forgather_server import job_records, queue_store
from forgather_server.agent import tools_jobs
from forgather_server.agent.registry import READ, ToolRegistry
from forgather_server.routes import jobs as jobs_route


def _rec(**kw):
    base = dict(queue_id="q1", job_type="dataset", project_dir="/p", config="c.yaml")
    base.update(kw)
    return job_records.JobRecord(**base)


def _item(**kw):
    base = dict(queue_id="i1", project_dir="/p", config="c.yaml")
    base.update(kw)
    return queue_store.QueueItem(**base)


def test_jobs_tools_registered():
    reg = ToolRegistry()
    tools_jobs.register_all(reg)
    names = {s.name for s in reg.specs()}
    assert {"list_jobs", "read_job_output"} <= names
    for s in reg.specs():
        assert s.risk == READ  # phase 1 adds only read tools


def test_list_jobs_projects_safe_fields_and_merges_queue(monkeypatch):
    monkeypatch.setattr(
        job_records,
        "list_records",
        lambda: [
            _rec(queue_id="q1", status="running", auth_token="SECRET", submitted_at=10),
            _rec(queue_id="q2", status="done", exit_code=0, submitted_at=20, finished_at=25),
        ],
    )
    monkeypatch.setattr(
        queue_store, "list_items", lambda: [_item(queue_id="i1", submitted_at=30)]
    )
    out = tools_jobs._list_jobs({})
    rows = out["jobs"]
    assert out["total"] == 3
    # Newest-first: queued i1 (30) before q2 (25) before q1 (10).
    assert [r["queue_id"] for r in rows] == ["i1", "q2", "q1"]
    # Queue item surfaces as status "queued".
    assert rows[0]["status"] == "queued"
    # Secrets / heavy fields are never projected.
    for r in rows:
        assert "auth_token" not in r
        assert "job_params" not in r
        assert "dynamic_args" not in r
        assert "tty_log_path" not in r


def test_list_jobs_filters(monkeypatch):
    monkeypatch.setattr(
        job_records,
        "list_records",
        lambda: [
            _rec(queue_id="q1", job_type="dataset", status="running"),
            _rec(queue_id="q2", job_type="training", status="done"),
        ],
    )
    monkeypatch.setattr(queue_store, "list_items", lambda: [])
    only_ds = tools_jobs._list_jobs({"job_type": "dataset"})
    assert [r["queue_id"] for r in only_ds["jobs"]] == ["q1"]
    only_done = tools_jobs._list_jobs({"status": "done"})
    assert [r["queue_id"] for r in only_done["jobs"]] == ["q2"]
    with pytest.raises(ValueError, match="unknown status"):
        tools_jobs._list_jobs({"status": "bogus"})


def test_read_job_output_tails(monkeypatch, tmp_path):
    f = tmp_path / "tty.log"
    f.write_text("\n".join(f"line {i}" for i in range(1000)) + "\n")
    monkeypatch.setattr(
        job_records,
        "get_record",
        lambda qid: _rec(queue_id=qid, status="done", exit_code=0, tty_log_path=str(f)),
    )
    out = tools_jobs._read_job_output({"queue_id": "q1", "tail_lines": 5})
    lines = out["tail"].splitlines()
    assert lines == ["line 995", "line 996", "line 997", "line 998", "line 999"]
    assert out["status"] == "done"
    assert out["exit_code"] == 0


def test_read_job_output_byte_cap(monkeypatch, tmp_path):
    # A log far larger than the 16 KiB cap is tail-truncated, not fully read.
    f = tmp_path / "big.log"
    f.write_text("x" * (200 * 1024))
    monkeypatch.setattr(
        job_records, "get_record", lambda qid: _rec(tty_log_path=str(f))
    )
    out = tools_jobs._read_job_output({"queue_id": "q1", "tail_lines": 100000})
    assert len(out["tail"]) <= tools_jobs._OUTPUT_MAX_BYTES


def test_read_job_output_errors(monkeypatch):
    monkeypatch.setattr(job_records, "get_record", lambda qid: None)
    with pytest.raises(ValueError, match="no job"):
        tools_jobs._read_job_output({"queue_id": "nope"})
    monkeypatch.setattr(
        job_records, "get_record", lambda qid: _rec(tty_log_path=None)
    )
    with pytest.raises(ValueError, match="no console output"):
        tools_jobs._read_job_output({"queue_id": "q1"})


def test_read_tty_tail_helper(tmp_path):
    f = tmp_path / "t.log"
    f.write_text("aaaa\nbbbb\ncccc\n")
    # Missing file -> "".
    assert jobs_route.read_tty_tail(str(tmp_path / "missing"), max_bytes=1024) == ""
    assert jobs_route.read_tty_tail(None, max_bytes=1024) == ""
    # tail_lines trims to the last N.
    assert jobs_route.read_tty_tail(str(f), max_bytes=1024, tail_lines=2) == "bbbb\ncccc"
    # A small byte cap drops the leading partial line.
    out = jobs_route.read_tty_tail(str(f), max_bytes=6)
    assert "aaaa" not in out and "cccc" in out
