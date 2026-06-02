"""Unit test for `forgather eval test --schedule` routing.

Mirrors the train/submit scheduler tests: the discovery + submit_orch calls
are monkeypatched, so this asserts that the scheduled path enqueues an "eval"
job with the eval-specific job_params (no real server or model needed).
"""

import argparse

from forgather.cli import eval as eval_cli
from forgather.cli import submit_orch


class _FakeClient:
    pass


def _eval_args(**over):
    base = dict(
        name="tinystories",
        model=None,
        devices=None,
        dry_run=False,
        schedule=True,
        enqueue=False,
        foreground=False,
        dataset=None,
        priority=3,
        via_server=None,
        local_only=False,
        local_fallback=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_eval_schedule_enqueues_eval_job(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr(
        eval_cli, "find_eval_config", lambda name, paths: ("/proj", "eval.yaml", {})
    )
    monkeypatch.setattr(eval_cli, "eval_search_paths", lambda d: ["/x"])
    monkeypatch.setattr(eval_cli, "_resolve_model_path", lambda args: "/model")
    monkeypatch.setattr(
        eval_cli, "eval_script_args_to_job_params", lambda args: {"max_steps": 5}
    )

    monkeypatch.setattr(submit_orch, "use_orchestrator", lambda args: _FakeClient())
    monkeypatch.setattr(submit_orch, "resolve_dataset_source", lambda c, a: None)

    def fake_submit_single(client, **kw):
        captured.update(kw)
        return {"queue_id": "qe-1", "priority": kw["priority"], "requested_gpus": 1}

    monkeypatch.setattr(submit_orch, "submit_single", fake_submit_single)

    eval_cli.test_cmd(_eval_args())

    out = capsys.readouterr().out
    assert "queued: qe-1" in out
    assert captured["job_type"] == "eval"
    assert captured["priority"] == 3
    jp = captured["job_params"]
    assert jp["eval_project"] == "/proj"
    assert jp["eval_template"] == "eval.yaml"
    assert jp["model_path"] == "/model"
    assert jp["max_steps"] == 5


def test_eval_enqueue_alias_warns(monkeypatch, capsys):
    monkeypatch.setattr(
        eval_cli, "find_eval_config", lambda name, paths: ("/proj", "eval.yaml", {})
    )
    monkeypatch.setattr(eval_cli, "eval_search_paths", lambda d: ["/x"])
    monkeypatch.setattr(eval_cli, "_resolve_model_path", lambda args: "/model")
    monkeypatch.setattr(eval_cli, "eval_script_args_to_job_params", lambda args: {})
    monkeypatch.setattr(submit_orch, "use_orchestrator", lambda args: _FakeClient())
    monkeypatch.setattr(submit_orch, "resolve_dataset_source", lambda c, a: None)
    monkeypatch.setattr(
        submit_orch,
        "submit_single",
        lambda client, **kw: {"queue_id": "qe-2", "priority": 0, "requested_gpus": 1},
    )

    eval_cli.test_cmd(_eval_args(schedule=False, enqueue=True))
    assert "--enqueue is deprecated" in capsys.readouterr().err
