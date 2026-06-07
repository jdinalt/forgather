"""Validate-and-enqueue core shared by the queue route and the agent.

Mirrors the validation the route used to do inline; also checks the route still
maps the exceptions to 400/403.
"""

from __future__ import annotations

import pytest

from forgather_server import config_ops, queue_ops, queue_store


class _Arg:
    def __init__(self, dest, cli_name, type="str", required=False, min=None, max=None):
        self.dest, self.cli_name, self.type = dest, cli_name, type
        self.required, self.min, self.max = required, min, max
        self.help = self.default = self.choices = self.group = None


@pytest.fixture(autouse=True)
def _stub_enqueue(monkeypatch):
    # Never touch the real queue store / fs-root / dynamic-args schema unless a
    # test opts in by overriding these.
    added = []
    monkeypatch.setattr(queue_store, "add_item", lambda it: added.append(it))
    monkeypatch.setattr(queue_ops.fs_paths, "is_path_in_fs_root", lambda p: True)
    monkeypatch.setattr(config_ops, "load_dynamic_args", lambda pd, c: [])
    return added


def test_rejects_unknown_job_type():
    with pytest.raises(queue_ops.EnqueueError, match="unsupported job_type"):
        queue_ops.validate_and_enqueue(
            project_dir="/p", config="c", dynamic_args={}, requested_gpus=0,
            job_type="bogus",
        )


def test_dataset_accepts_zero_gpus():
    # All currently-supported job types tolerate 0 GPUs (CPU dispatch); dataset
    # in particular is always CPU-only.
    item = queue_ops.validate_and_enqueue(
        project_dir="/p", config="c", dynamic_args={}, requested_gpus=0,
        job_type="dataset", job_params={"target": "train_dataset_split"},
    )
    assert item.job_type == "dataset" and item.requested_gpus == 0


def test_missing_required_dynamic_arg(monkeypatch):
    monkeypatch.setattr(
        config_ops, "load_dynamic_args",
        lambda pd, c: [_Arg("model_project", "--model-project", required=True)],
    )
    with pytest.raises(queue_ops.EnqueueError, match="required dynamic arg"):
        queue_ops.validate_and_enqueue(
            project_dir="/p", config="c", dynamic_args={}, requested_gpus=0,
            job_type="dataset",
        )


def test_numeric_bounds(monkeypatch):
    monkeypatch.setattr(
        config_ops, "load_dynamic_args",
        lambda pd, c: [_Arg("n", "--n", type="int", min=1, max=10)],
    )
    with pytest.raises(queue_ops.EnqueueError, match="constraint violated"):
        queue_ops.validate_and_enqueue(
            project_dir="/p", config="c", dynamic_args={"n": 99}, requested_gpus=0,
            job_type="dataset",
        )
    # NaN / inf rejected.
    with pytest.raises(queue_ops.EnqueueError, match="finite"):
        queue_ops.validate_and_enqueue(
            project_dir="/p", config="c", dynamic_args={"n": float("inf")},
            requested_gpus=0, job_type="dataset",
        )


def test_missing_required_params():
    with pytest.raises(queue_ops.EnqueueError, match="job_params missing"):
        queue_ops.validate_and_enqueue(
            project_dir="/p", config="c", dynamic_args={}, requested_gpus=1,
            job_type="inference", job_params={"port": 8000},  # missing model_path
        )


def test_fs_root_raises_forbidden(monkeypatch):
    monkeypatch.setattr(queue_ops.fs_paths, "is_path_in_fs_root", lambda p: False)
    with pytest.raises(queue_ops.EnqueueForbidden, match="outside the configured"):
        queue_ops.validate_and_enqueue(
            project_dir="/etc", config="c", dynamic_args={}, requested_gpus=0,
            job_type="dataset",
        )
    # EnqueueForbidden is an EnqueueError subclass (route catches the base too).
    assert issubclass(queue_ops.EnqueueForbidden, queue_ops.EnqueueError)


def test_happy_path_enqueues(_stub_enqueue):
    item = queue_ops.validate_and_enqueue(
        project_dir="/p", config="c.yaml", dynamic_args={}, requested_gpus=0,
        job_type="dataset", job_params={"target": "train_dataset_split"},
    )
    assert item in _stub_enqueue
    assert item.job_type == "dataset" and item.requested_gpus == 0


def test_route_maps_status_codes(monkeypatch):
    # The thin route adapter maps EnqueueForbidden->403, EnqueueError->400.
    from fastapi import HTTPException
    from forgather_server.routes import queue as queue_route

    def boom_forbidden(**kw):
        raise queue_ops.EnqueueForbidden("nope")

    monkeypatch.setattr(queue_ops, "validate_and_enqueue", boom_forbidden)
    req = queue_route.EnqueueRequest(project_dir="/p", config="c", requested_gpus=0, job_type="dataset")
    with pytest.raises(HTTPException) as ei:
        queue_route.enqueue(req)
    assert ei.value.status_code == 403

    def boom_bad(**kw):
        raise queue_ops.EnqueueError("bad")

    monkeypatch.setattr(queue_ops, "validate_and_enqueue", boom_bad)
    with pytest.raises(HTTPException) as ei:
        queue_route.enqueue(req)
    assert ei.value.status_code == 400
