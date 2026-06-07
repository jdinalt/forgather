"""Tier-2 parity tools: gpu_status, resolve_output_dir, eval, control_job."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest

from forgather_server import config_ops, eval_ops, gpu_monitor, job_records, queue_ops
from forgather_server.agent import tools_jobs, tools_readonly
from forgather_server.agent.registry import (
    CONFIRM,
    EXTENDED,
    READ,
    Proposal,
    ToolRegistry,
)


def _jobs_reg():
    reg = ToolRegistry()
    tools_jobs.register_all(reg)
    return reg


# ---- registration ----------------------------------------------------------


def test_registration_risk_and_tier():
    by = {s.name: s for s in _jobs_reg().specs()}
    assert by["gpu_status"].risk == READ and by["gpu_status"].tier != EXTENDED
    assert by["control_job"].risk == CONFIRM and by["control_job"].tier != EXTENDED
    assert by["list_eval_configs"].risk == READ and by["list_eval_configs"].tier == EXTENDED
    assert by["run_eval"].risk == CONFIRM and by["run_eval"].tier == EXTENDED

    ro = ToolRegistry()
    tools_readonly.register_all(ro)
    spec = {s.name: s for s in ro.specs()}["resolve_output_dir"]
    assert spec.risk == READ and spec.tier == EXTENDED


# ---- gpu_status ------------------------------------------------------------


def test_gpu_status_compact_projection(monkeypatch):
    g = SimpleNamespace(
        index=0, name="H100", total_mem_bytes=100, used_mem_bytes=30,
        util_pct=12, mem_util_pct=30, temp_c=40, power_w=99.0,
        disabled=False, excluded=False, min_priority=0, unified_memory=False,
        processes=[object(), object()],
    )
    monkeypatch.setattr(gpu_monitor, "snapshot", lambda: [g])
    out = tools_jobs._gpu_status({})["gpus"][0]
    assert out["free_mem_bytes"] == 70 and out["process_count"] == 2
    assert "processes" not in out  # full process list omitted


# ---- resolve_output_dir ----------------------------------------------------


def test_resolve_output_dir_serializes(monkeypatch):
    info = config_ops.OutputDirInfo(
        output_dir="/o", models_dir="/m", output_dir_exists=True, models_dir_exists=False
    )
    monkeypatch.setattr(config_ops, "load_output_dir_info", lambda p, c: info)
    out = tools_readonly._resolve_output_dir({"project_dir": "/p", "config_name": "c.yaml"})
    assert out["output_dir"] == "/o" and out["output_dir_exists"] is True


# ---- eval ------------------------------------------------------------------


@dataclass
class _EvalCfg:
    name: str
    project_dir: str
    template: str
    description: str = ""


def test_list_eval_configs(monkeypatch):
    monkeypatch.setattr(
        eval_ops, "list_eval_configs", lambda: [_EvalCfg("ppl", "/ev", "test.yaml")]
    )
    out = tools_jobs._list_eval_configs({})
    assert out["eval_configs"][0]["name"] == "ppl"


def test_run_eval_unknown_name(monkeypatch):
    monkeypatch.setattr(eval_ops, "list_eval_configs", lambda: [])
    with pytest.raises(ValueError):
        tools_jobs._run_eval({"eval_name": "nope", "model_path": "/m"})


def test_run_eval_preview_then_commit(monkeypatch):
    monkeypatch.setattr(
        eval_ops, "list_eval_configs", lambda: [_EvalCfg("ppl", "/ev", "test.yaml")]
    )
    monkeypatch.setattr(
        eval_ops, "build_eval_command", lambda **kw: ["python", "eval", kw["model_path"]]
    )
    enqueued = []
    monkeypatch.setattr(
        queue_ops, "validate_and_enqueue",
        lambda **kw: enqueued.append(kw) or SimpleNamespace(queue_id="q9", job_type="eval"),
    )
    prop = tools_jobs._run_eval(
        {"eval_name": "ppl", "model_path": "/models/m", "batch_size": 8}
    )
    assert isinstance(prop, Proposal)
    assert "command" in prop.extra and enqueued == []  # preview only

    msg = prop.commit()
    assert enqueued and enqueued[0]["job_type"] == "eval"
    jp = enqueued[0]["job_params"]
    assert jp["eval_project"] == "/ev" and jp["eval_template"] == "test.yaml"
    assert jp["model_path"] == "/models/m" and jp["batch_size"] == 8
    assert enqueued[0]["enforce_fs_root"] is True
    assert "q9" in msg


# ---- control_job -----------------------------------------------------------


def _rec(job_type="training", job_id="jid", status="running"):
    return SimpleNamespace(job_type=job_type, job_id=job_id, status=status, config="c.yaml")


def test_control_job_unknown_action(monkeypatch):
    monkeypatch.setattr(job_records, "get_record", lambda q: _rec())
    with pytest.raises(ValueError):
        tools_jobs._control_job({"queue_id": "q1", "action": "explode"})


def test_control_job_rejects_non_training(monkeypatch):
    monkeypatch.setattr(job_records, "get_record", lambda q: _rec(job_type="dataset"))
    with pytest.raises(ValueError):
        tools_jobs._control_job({"queue_id": "q1", "action": "stop"})


def test_control_job_requires_correlation(monkeypatch):
    monkeypatch.setattr(job_records, "get_record", lambda q: _rec(job_id=None))
    with pytest.raises(ValueError):
        tools_jobs._control_job({"queue_id": "q1", "action": "stop"})


def test_control_job_commit_calls_trainer(monkeypatch):
    monkeypatch.setattr(job_records, "get_record", lambda q: _rec(job_id="jid-7"))
    import forgather.trainer_control as tc

    called = {}

    def fake_stop(jid):
        called["jid"] = jid
        return SimpleNamespace(success=True, message="stopping")

    monkeypatch.setattr(tc, "graceful_stop", fake_stop)
    prop = tools_jobs._control_job({"queue_id": "q1", "action": "stop"})
    assert isinstance(prop, Proposal)
    assert called == {}  # preview does nothing
    msg = prop.commit()
    assert called["jid"] == "jid-7" and "success=True" in msg
