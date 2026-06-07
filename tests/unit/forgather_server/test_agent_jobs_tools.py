"""Job-visibility agent tools (list_jobs / read_job_output) + the shared
TTY-tail reader they reuse from the jobs route."""

from __future__ import annotations

import pytest

from forgather_server import dataset_ops, job_records, queue_ops, queue_store
from forgather_server.agent import tools_jobs
from forgather_server.agent.registry import CONFIRM, READ, Proposal, ToolRegistry
from forgather_server.routes import jobs as jobs_route


def _rec(**kw):
    base = dict(queue_id="q1", job_type="dataset", project_dir="/p", config="c.yaml")
    base.update(kw)
    return job_records.JobRecord(**base)


def _item(**kw):
    base = dict(queue_id="i1", project_dir="/p", config="c.yaml")
    base.update(kw)
    return queue_store.QueueItem(**base)


def test_system_prompt_mentions_dataset_workflow():
    from forgather_server.agent import runtime

    p = runtime.SYSTEM_PROMPT
    for tool in ("run_dataset", "dataset_info", "list_jobs", "read_job_output"):
        assert tool in p


def test_jobs_tools_registered():
    reg = ToolRegistry()
    tools_jobs.register_all(reg)
    by_name = {s.name: s for s in reg.specs()}
    assert {"list_jobs", "read_job_output", "run_dataset"} <= set(by_name)
    assert by_name["list_jobs"].risk == READ
    assert by_name["read_job_output"].risk == READ
    assert by_name["run_dataset"].risk == CONFIRM  # gated: runs code + downloads


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


def test_run_dataset_preview_builds_command_no_enqueue(monkeypatch):
    captured = {}

    def fake_build(**kw):
        captured.update(kw)
        return ["python", "-m", "forgather.cli", "dataset", "--target", kw["target"]]

    enqueued = []
    monkeypatch.setattr(dataset_ops, "build_dataset_command", fake_build)
    monkeypatch.setattr(
        queue_ops, "validate_and_enqueue", lambda **kw: enqueued.append(kw)
    )

    prop = tools_jobs._run_dataset(
        {"project_dir": "/p", "config_name": "c.yaml", "examples": 3, "truncate": 64}
    )
    assert isinstance(prop, Proposal)
    # Default target; examples/truncate threaded into the preview command.
    assert captured["target"] == "train_dataset_split"
    assert captured["examples"] == 3 and captured["truncate"] == 64
    assert "command" in prop.extra and "train_dataset_split" in prop.extra["command"]
    assert "slow" in prop.extra["warning"].lower() or "long time" in prop.extra["warning"].lower()
    assert prop.commit is not None
    # Preview must NOT enqueue anything.
    assert enqueued == []


def test_run_dataset_commit_enqueues(monkeypatch):
    monkeypatch.setattr(
        dataset_ops, "build_dataset_command", lambda **kw: ["python", "dataset"]
    )

    class _Item:
        queue_id = "q_new"

    seen = {}

    def fake_enqueue(**kw):
        seen.update(kw)
        return _Item()

    monkeypatch.setattr(queue_ops, "validate_and_enqueue", fake_enqueue)
    prop = tools_jobs._run_dataset(
        {"project_dir": "/p", "config_name": "c.yaml", "target": "validation_dataset_split", "truncate": 64}
    )
    msg = prop.commit()
    assert seen["job_type"] == "dataset"
    assert seen["requested_gpus"] == 0
    assert seen["enforce_fs_root"] is True
    assert seen["job_params"] == {"target": "validation_dataset_split", "truncate": 64}
    assert "q_new" in msg


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
