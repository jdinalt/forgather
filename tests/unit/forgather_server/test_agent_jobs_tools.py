"""Job-visibility agent tools (list_jobs / read_job_output) + the shared
TTY-tail reader they reuse from the jobs route."""

from __future__ import annotations

import asyncio

import pytest

from forgather_server import (
    construct_ops,
    dataset_ops,
    job_records,
    queue_ops,
    queue_store,
)
from forgather_server.agent import tools_jobs
from forgather_server.agent.registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry
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
    for tool in (
        "run_dataset",
        "run_construct",
        "run_train",
        "dataset_info",
        "list_jobs",
        "read_job_output",
    ):
        assert tool in p


def test_jobs_tools_registered():
    reg = ToolRegistry()
    tools_jobs.register_all(reg)
    by_name = {s.name: s for s in reg.specs()}
    assert {
        "list_jobs",
        "read_job_output",
        "wait_for_job",
        "run_dataset",
        "run_construct",
        "run_train",
    } <= set(by_name)
    assert by_name["list_jobs"].risk == READ
    assert by_name["read_job_output"].risk == READ
    assert by_name["wait_for_job"].risk == READ
    # All run-as-job tools are gated: they execute config code / reserve GPUs.
    assert by_name["run_dataset"].risk == CONFIRM
    assert by_name["run_construct"].risk == CONFIRM
    assert by_name["run_train"].risk == CONFIRM


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


def test_wait_for_job_returns_on_terminal(monkeypatch, tmp_path):
    # Job is "running" for two polls, then "done".
    f = tmp_path / "tty.log"
    f.write_text("built ok\n")
    seq = iter(["running", "running", "done"])
    state = {"status": "running"}

    def fake_get(qid):
        try:
            state["status"] = next(seq)
        except StopIteration:
            pass
        return _rec(queue_id=qid, status=state["status"], exit_code=0, tty_log_path=str(f))

    monkeypatch.setattr(job_records, "get_record", fake_get)
    monkeypatch.setattr(tools_jobs, "_WAIT_POLL_SECONDS", 0.01)  # tiny real delay

    out = asyncio.run(tools_jobs._wait_for_job({"queue_id": "q1", "timeout_seconds": 60}))
    assert out["status"] == "done"
    assert out["timed_out"] is False
    assert out["exit_code"] == 0
    assert "built ok" in out["tail"]


def test_wait_for_job_times_out(monkeypatch):
    # Never terminal -> returns timed_out once the deadline passes.
    monkeypatch.setattr(
        job_records, "get_record",
        lambda qid: _rec(queue_id=qid, status="running", tty_log_path=None),
    )
    monkeypatch.setattr(tools_jobs, "_WAIT_POLL_SECONDS", 0.01)
    out = asyncio.run(tools_jobs._wait_for_job({"queue_id": "q1", "timeout_seconds": 0.05}))
    assert out["status"] == "running"
    assert out["timed_out"] is True


def test_wait_for_job_unknown(monkeypatch):
    monkeypatch.setattr(job_records, "get_record", lambda qid: None)
    with pytest.raises(ValueError, match="no job"):
        asyncio.run(tools_jobs._wait_for_job({"queue_id": "nope"}))


def test_wait_for_job_until_running_returns_when_up(monkeypatch):
    # A service: starting -> running, and never terminal. until="running"
    # must return as soon as it's up (not wait out the timeout).
    seq = iter(["starting", "starting", "running"])
    state = {"status": "starting"}

    def fake_get(qid):
        try:
            state["status"] = next(seq)
        except StopIteration:
            pass
        return _rec(queue_id=qid, status=state["status"], tty_log_path=None)

    monkeypatch.setattr(job_records, "get_record", fake_get)
    monkeypatch.setattr(tools_jobs, "_WAIT_POLL_SECONDS", 0.01)
    out = asyncio.run(
        tools_jobs._wait_for_job({"queue_id": "q1", "until": "running", "timeout_seconds": 60})
    )
    assert out["status"] == "running" and out["timed_out"] is False


def test_wait_for_job_until_running_returns_on_early_failure(monkeypatch):
    # A service that dies before coming up still returns (terminal), so the
    # agent learns it failed instead of waiting out the timeout.
    seq = iter(["starting", "failed"])
    state = {"status": "starting"}

    def fake_get(qid):
        try:
            state["status"] = next(seq)
        except StopIteration:
            pass
        return _rec(queue_id=qid, status=state["status"], exit_code=1, tty_log_path=None)

    monkeypatch.setattr(job_records, "get_record", fake_get)
    monkeypatch.setattr(tools_jobs, "_WAIT_POLL_SECONDS", 0.01)
    out = asyncio.run(
        tools_jobs._wait_for_job({"queue_id": "q1", "until": "running", "timeout_seconds": 60})
    )
    assert out["status"] == "failed" and out["timed_out"] is False


def test_wait_for_job_invalid_until(monkeypatch):
    monkeypatch.setattr(
        job_records, "get_record",
        lambda qid: _rec(queue_id=qid, status="running", tty_log_path=None),
    )
    with pytest.raises(ValueError, match="until"):
        asyncio.run(tools_jobs._wait_for_job({"queue_id": "q1", "until": "bogus"}))


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
        job_type = "dataset"

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


def _fake_item(job_type):
    class _Item:
        queue_id = "q_new"

    _Item.job_type = job_type
    return _Item()


def test_run_construct_preview_and_commit(monkeypatch):
    captured = {}

    def fake_build(**kw):
        captured.update(kw)
        return ["python", "-m", "forgather.cli", "construct", "--target", kw["target"]]

    seen = {}
    monkeypatch.setattr(construct_ops, "build_construct_command", fake_build)
    monkeypatch.setattr(
        queue_ops,
        "validate_and_enqueue",
        lambda **kw: seen.update(kw) or _fake_item("construct"),
    )

    prop = tools_jobs._run_construct(
        {"project_dir": "/p", "config_name": "c.yaml", "target": "model", "call": True}
    )
    assert isinstance(prop, Proposal)
    assert prop.commit is not None
    # Preview builds the command but does not enqueue.
    assert captured["target"] == "model" and captured["call"] is True
    assert "model" in prop.extra["command"]
    assert seen == {}

    msg = prop.commit()
    assert seen["job_type"] == "construct"
    assert seen["requested_gpus"] == 0  # default
    assert seen["enforce_fs_root"] is True
    assert seen["job_params"] == {"target": "model", "call": True}
    assert "q_new" in msg


def test_run_construct_gpus_override(monkeypatch):
    monkeypatch.setattr(
        construct_ops, "build_construct_command", lambda **kw: ["python", "construct"]
    )
    seen = {}
    monkeypatch.setattr(
        queue_ops,
        "validate_and_enqueue",
        lambda **kw: seen.update(kw) or _fake_item("construct"),
    )
    prop = tools_jobs._run_construct(
        {"project_dir": "/p", "config_name": "c.yaml", "gpus": 2}
    )
    prop.commit()
    assert seen["requested_gpus"] == 2
    assert seen["job_params"] == {"target": "main", "call": False}  # defaults


def test_run_train_defaults_one_gpu(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        queue_ops,
        "validate_and_enqueue",
        lambda **kw: seen.update(kw) or _fake_item("training"),
    )
    prop = tools_jobs._run_train({"project_dir": "/p", "config_name": "c.yaml"})
    assert isinstance(prop, Proposal)
    # Preview shows the SCHEDULED invocation (submit), NOT a foreground `train`.
    assert "submit" in prop.extra["command"]
    assert " train" not in prop.extra["command"]
    assert seen == {}

    msg = prop.commit()
    assert seen["job_type"] == "training"
    # nproc_per_node can't be read for a fake project -> falls back to 1.
    assert seen["requested_gpus"] == 1
    assert seen["priority"] == 0
    assert seen["dataset_source"] is None  # in-process loader by default
    assert seen["enforce_fs_root"] is True
    assert seen["job_params"] == {}  # no nproc
    assert "q_new" in msg


def test_run_train_infers_gpus_and_passes_priority_and_dataset(monkeypatch):
    from forgather_server import config_ops

    monkeypatch.setattr(
        config_ops, "load_output_dir_info",
        lambda pd, c: type("I", (), {"nproc_per_node": 4})(),
    )
    seen = {}
    monkeypatch.setattr(
        queue_ops, "validate_and_enqueue",
        lambda **kw: seen.update(kw) or _fake_item("training"),
    )
    prop = tools_jobs._run_train(
        {"project_dir": "/p", "config_name": "c.yaml", "priority": 5,
         "dataset_server_id": "local:q1"}
    )
    prop.commit()
    assert seen["requested_gpus"] == 4  # inferred from nproc_per_node
    assert seen["priority"] == 5
    assert seen["dataset_source"] == {"kind": "server", "server_id": "local:q1"}


def test_run_train_gpus_and_nproc(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        queue_ops,
        "validate_and_enqueue",
        lambda **kw: seen.update(kw) or _fake_item("training"),
    )
    prop = tools_jobs._run_train(
        {"project_dir": "/p", "config_name": "c.yaml", "gpus": 0, "nproc": "auto"}
    )
    prop.commit()
    assert seen["requested_gpus"] == 0  # CPU smoke-test
    assert seen["job_params"] == {"nproc": "auto"}


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


# ---- cleanup_jobs ----------------------------------------------------------


def test_cleanup_jobs_registered():
    reg = ToolRegistry()
    tools_jobs.register_all(reg)
    spec = {s.name: s for s in reg.specs()}["cleanup_jobs"]
    assert spec.risk == CONFIRM and spec.tier != EXTENDED  # core: nudged for routine use


def test_cleanup_jobs_requires_an_argument():
    with pytest.raises(ValueError, match="queue_ids"):
        tools_jobs._cleanup_jobs({})


def test_cleanup_jobs_rejects_both_args():
    with pytest.raises(ValueError, match="either"):
        tools_jobs._cleanup_jobs({"queue_ids": ["q1"], "all_terminal": True})


def test_cleanup_jobs_targets_only_terminal(monkeypatch):
    recs = {
        "qd": _rec(queue_id="qd", status="done"),
        "qr": _rec(queue_id="qr", status="running"),
    }
    monkeypatch.setattr(job_records, "get_record", lambda q: recs.get(q))
    prop = tools_jobs._cleanup_jobs({"queue_ids": ["qd", "qr", "qmissing"]})
    assert isinstance(prop, Proposal)
    assert prop.extra["queue_ids"] == ["qd"]  # running + missing excluded


def test_cleanup_jobs_none_terminal_raises(monkeypatch):
    monkeypatch.setattr(
        job_records, "get_record",
        lambda q: _rec(queue_id=q, status="running"),
    )
    with pytest.raises(ValueError, match="no removable"):
        tools_jobs._cleanup_jobs({"queue_ids": ["qr"]})


def test_cleanup_jobs_commit_removes_specific(monkeypatch):
    monkeypatch.setattr(
        job_records, "get_record",
        lambda q: _rec(queue_id=q, status="done"),
    )
    removed = []
    monkeypatch.setattr(jobs_route, "remove_job", lambda qid: removed.append(qid))
    prop = tools_jobs._cleanup_jobs({"queue_ids": ["qa", "qb"]})
    assert removed == []  # preview did not remove
    msg = prop.commit()
    assert removed == ["qa", "qb"] and "removed 2" in msg


def test_cleanup_jobs_all_terminal(monkeypatch):
    monkeypatch.setattr(
        job_records, "list_records",
        lambda: [_rec(queue_id="qd", status="done"), _rec(queue_id="qr", status="running")],
    )
    monkeypatch.setattr(jobs_route, "cleanup_jobs", lambda: {"removed": ["qd"], "count": 1})
    prop = tools_jobs._cleanup_jobs({"all_terminal": True})
    assert prop.extra["all_terminal"] is True and prop.extra["queue_ids"] == ["qd"]
    msg = prop.commit()
    assert "removed 1 completed" in msg


def test_system_prompt_docs_first_and_scheduling():
    from forgather_server.agent import runtime

    sp = runtime.SYSTEM_PROMPT
    assert "READ THE DOCS FIRST" in sp  # docs-first bias
    # run_train is described as scheduling, not foreground.
    reg = ToolRegistry()
    tools_jobs.register_all(reg)
    desc = {s.name: s for s in reg.specs()}["run_train"].description
    assert "SCHEDULE" in desc and "submit" in desc and "foreground" in desc
