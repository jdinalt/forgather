"""Unit tests for `forgather dataset-server start` — orchestrator-first launch.

The command defaults to enqueuing a scheduled ``dataset_server`` job through
the forgather server (background, cluster-registered, auto-provisioned auth),
mirroring `forgather diloco server`. ``--local-only`` runs it foreground.
"""

import argparse
import os

import pytest

from forgather.cli import dataset_server as ds
from forgather.cli.main import parse_args

# ---------------------------------------------------------------------------
# Arg parsing — the server's -p/-H must not collide with global -p/--project-dir
# ---------------------------------------------------------------------------


def test_parse_port_does_not_hit_global_project_dir():
    a = parse_args(
        ["dataset-server", "start", "-H", "0.0.0.0", "-p", "9001", "--local", "foo=/d"]
    )
    assert a.ds_subcommand == "start"
    assert a.host == "0.0.0.0"
    assert a.port == 9001
    assert a.local_maps == ["foo=/d"]
    assert a.local_only is False
    assert a.extra == []
    # The global -p/--project-dir must stay at its default, not 9001.
    assert a.project_dir == "."


def test_parse_local_only_captures_extra_flags():
    b = parse_args(
        ["dataset-server", "start", "--local-only", "--no-hf", "--tls-cert", "/c"]
    )
    assert b.local_only is True
    assert b.no_hf is True
    # Unknown server flags are captured for the foreground path to forward.
    assert b.extra == ["--tls-cert", "/c"]


# ---------------------------------------------------------------------------
# local-mapping parsing + job_params shape
# ---------------------------------------------------------------------------


def test_parse_local_maps_valid():
    assert ds._parse_local_maps(["a=/p1", "b=/p2"]) == [["a", "/p1"], ["b", "/p2"]]
    assert ds._parse_local_maps(None) == []


@pytest.mark.parametrize("bad", ["noequals", "=/p", "name="])
def test_parse_local_maps_invalid(bad):
    with pytest.raises(ValueError):
        ds._parse_local_maps([bad])


def test_parse_local_maps_absolutizes_relative_path():
    # The scheduled job runs from the repo root, so a relative local path
    # must be resolved against the CLI's CWD before enqueue.
    [[name, path]] = ds._parse_local_maps(["foo=../data/foo"])
    assert name == "foo"
    assert path == os.path.abspath("../data/foo")
    assert os.path.isabs(path)


def test_start_job_params_absolutizes_config():
    params = ds._start_job_params(_start_args(config="rel/ds.yaml"), [])
    assert params["config_file"] == os.path.abspath("rel/ds.yaml")
    assert os.path.isabs(params["config_file"])


def _start_args(**over):
    base = dict(
        host="0.0.0.0",
        port=8766,
        log_level="INFO",
        no_hf=False,
        allow_paths=False,
        allow_downloads=False,
        no_auth=False,
        regen_token=False,
        config=None,
        priority=0,
        json=False,
        extra=[],
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_start_job_params_mirrors_webui():
    params = ds._start_job_params(
        _start_args(port=9001, no_hf=True, regen_token=True), [["foo", "/d"]]
    )
    assert params == {
        "host": "0.0.0.0",
        "port": 9001,
        "log_level": "INFO",
        "no_auth": False,
        "regen_token": True,
        "no_hf": True,
        "allow_paths": False,
        "allow_downloads": False,
        "locals": [["foo", "/d"]],
    }


def test_start_job_params_regen_suppressed_under_no_auth():
    params = ds._start_job_params(_start_args(no_auth=True, regen_token=True), [])
    assert params["no_auth"] is True
    assert params["regen_token"] is False  # meaningless when auth is off
    assert "locals" not in params  # omitted when empty


def test_start_job_params_includes_config_file():
    params = ds._start_job_params(_start_args(config="/etc/ds.yaml"), [])
    assert params["config_file"] == "/etc/ds.yaml"


# ---------------------------------------------------------------------------
# Orchestrator enqueue + foreground decision
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self):
        self.enqueued = []

    def enqueue_job(self, **kw):
        self.enqueued.append(kw)
        return {"queue_id": "q_ds_1"}


def test_start_via_server_enqueue_shape(capsys):
    client = FakeClient()
    rc = ds._start_via_server(client, _start_args(port=9001), [["foo", "/d"]])
    assert rc == 0
    kw = client.enqueued[0]
    assert kw["job_type"] == "dataset_server"
    assert kw["config"] == "dataset:9001"
    assert kw["project_dir"] == "/"
    assert kw["requested_gpus"] == 0
    assert kw["job_params"]["host"] == "0.0.0.0"
    assert kw["job_params"]["locals"] == [["foo", "/d"]]
    assert "Enqueued dataset server job q_ds_1" in capsys.readouterr().out


def test_start_via_server_rejects_extra_flags(capsys):
    client = FakeClient()
    rc = ds._start_via_server(client, _start_args(extra=["--tls-cert", "/c"]), [])
    assert rc == 1
    assert not client.enqueued  # nothing enqueued
    assert "only supported with --local-only" in capsys.readouterr().err


def test_start_via_server_json(capsys):
    client = FakeClient()
    rc = ds._start_via_server(client, _start_args(json=True), [])
    assert rc == 0
    import json as _json

    assert _json.loads(capsys.readouterr().out)["queue_id"] == "q_ds_1"


def test_start_cmd_defaults_to_orchestrator(monkeypatch):
    # use_orchestrator returns a client → enqueue path taken.
    from forgather.cli import diloco_orch as orch

    client = FakeClient()
    monkeypatch.setattr(orch, "use_orchestrator", lambda args: client)
    args = _start_args(local_only=False, local_fallback=False, via_server=None)
    args.local_maps = []
    rc = ds._start_cmd(args)
    assert rc == 0
    assert client.enqueued[0]["job_type"] == "dataset_server"


def test_start_cmd_local_only_runs_foreground(monkeypatch):
    # use_orchestrator returns None (--local-only) → foreground subprocess.
    from forgather.cli import diloco_orch as orch

    monkeypatch.setattr(orch, "use_orchestrator", lambda args: None)

    captured = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd, cwd=None):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(ds.subprocess, "run", _fake_run)
    args = _start_args(local_only=True, local_fallback=False, via_server=None)
    args.local_maps = ["foo=/d"]
    rc = ds._start_cmd(args)
    assert rc == 0
    cmd = captured["cmd"]
    assert "tools.dataset_server" in cmd
    assert "-H" in cmd and "0.0.0.0" in cmd
    assert "--local" in cmd and "foo=/d" in cmd
