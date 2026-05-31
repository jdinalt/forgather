"""Unit tests for the orchestrator-backed DiLoCo CLI handlers."""

import argparse
import json

import pytest

from forgather.cli import diloco_orch as orch

# ---------------------------------------------------------------------------
# match_server
# ---------------------------------------------------------------------------

SERVERS = [
    {
        "id": "local:q1",
        "label": "diloco:8512",
        "base_url": "http://192.168.9.43:8512",
        "source": "local",
        "queue_id": "q1",
        "alive": True,
    },
    {
        "id": "registered:abcd",
        "label": "remote-a",
        "base_url": "https://10.0.0.5:8512",
        "source": "registered",
        "has_auth_token": True,
        "verify_tls": False,
    },
]


class TestMatchServer:
    def test_by_id(self):
        assert orch.match_server(SERVERS, "local:q1") == "http://192.168.9.43:8512"

    def test_by_label(self):
        assert orch.match_server(SERVERS, "remote-a") == "https://10.0.0.5:8512"

    def test_by_base_url(self):
        assert (
            orch.match_server(SERVERS, "https://10.0.0.5:8512/")
            == "https://10.0.0.5:8512"
        )

    def test_by_host_port(self):
        assert orch.match_server(SERVERS, "192.168.9.43:8512") == (
            "http://192.168.9.43:8512"
        )

    def test_no_match(self):
        assert orch.match_server(SERVERS, "localhost:8512") is None

    def test_empty_target(self):
        assert orch.match_server(SERVERS, None) is None


# ---------------------------------------------------------------------------
# assemble_status — a getter that raises is recorded as None
# ---------------------------------------------------------------------------


def test_assemble_status_partial():
    merged = orch.assemble_status(
        get_status=lambda: {"sync_round": 3},
        get_info=lambda: {"num_parameters": 10},
        get_known_workers=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        get_work_queues=None,
    )
    assert merged["status"] == {"sync_round": 3}
    assert merged["info"] == {"num_parameters": 10}
    assert merged["known_workers"] is None  # getter raised → None, not a crash
    assert merged["work_queues"] is None  # no getter supplied


# ---------------------------------------------------------------------------
# Fake ServerClient for handler tests
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, *, servers=None, jobs=None, reachable=True, dump=b""):
        self._servers = servers or []
        self._jobs = jobs or []
        self._reachable = reachable
        self._dump = dump

    def ping(self):
        return self._reachable

    def list_diloco_servers(self):
        return self._servers

    def list_jobs(self, include_dead=False):
        return self._jobs

    def job_dump(self, job_id):
        self.dumped = job_id
        return self._dump


@pytest.fixture
def patch_orchestrator(monkeypatch):
    """Install a FakeClient as the orchestrator the handlers build."""

    def _install(client):
        monkeypatch.setattr(orch, "_orchestrator", lambda args: client)
        return client

    return _install


class TestServersCmd:
    def test_json(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient(servers=SERVERS))
        rc = orch.servers_cmd(argparse.Namespace(via_server=None, json=True))
        assert rc == 0
        out = capsys.readouterr().out
        assert json.loads(out) == SERVERS  # valid JSON, verbatim

    def test_human_table(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient(servers=SERVERS))
        rc = orch.servers_cmd(argparse.Namespace(via_server=None, json=False))
        assert rc == 0
        out = capsys.readouterr().out
        assert "local:q1" in out and "alive" in out
        assert "registered:abcd" in out and "no-verify" in out

    def test_empty(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient(servers=[]))
        rc = orch.servers_cmd(argparse.Namespace(via_server=None, json=False))
        assert rc == 0
        assert "No DiLoCo servers" in capsys.readouterr().out


class TestResolveJobId:
    def test_local_server_label(self):
        c = FakeClient(servers=SERVERS)
        assert orch._resolve_job_id(c, "diloco:8512") == "q1"

    def test_by_worker_id(self):
        jobs = [
            {"id": "qX", "job_params": {"diloco": {"worker_id": "spectacular-fox"}}},
        ]
        c = FakeClient(servers=[], jobs=jobs)
        assert orch._resolve_job_id(c, "spectacular-fox") == "qX"

    def test_by_queue_id(self):
        c = FakeClient(servers=[], jobs=[{"id": "qY", "queue_id": "qY"}])
        assert orch._resolve_job_id(c, "qY") == "qY"

    def test_fallback_verbatim(self):
        c = FakeClient(servers=[], jobs=[])
        assert orch._resolve_job_id(c, "whatever") == "whatever"


class TestResolveOrchestratorBase:
    def test_direct_flag_skips(self, patch_orchestrator):
        patch_orchestrator(FakeClient(servers=SERVERS))
        args = argparse.Namespace(direct=True, server="local:q1", via_server=None)
        assert orch.resolve_orchestrator_base(args) == (None, None)

    def test_unreachable_falls_back(self, patch_orchestrator):
        patch_orchestrator(FakeClient(servers=SERVERS, reachable=False))
        args = argparse.Namespace(direct=False, server="local:q1", via_server=None)
        assert orch.resolve_orchestrator_base(args) == (None, None)

    def test_reachable_known_target(self, patch_orchestrator):
        client = patch_orchestrator(FakeClient(servers=SERVERS))
        args = argparse.Namespace(direct=False, server="local:q1", via_server=None)
        c, base = orch.resolve_orchestrator_base(args)
        assert c is client
        assert base == "http://192.168.9.43:8512"

    def test_reachable_unknown_target_falls_back(self, patch_orchestrator):
        patch_orchestrator(FakeClient(servers=SERVERS))
        args = argparse.Namespace(
            direct=False, server="localhost:8512", via_server=None
        )
        assert orch.resolve_orchestrator_base(args) == (None, None)


class TestLogsCmd:
    def test_dump(self, patch_orchestrator, capsysbinary):
        client = patch_orchestrator(
            FakeClient(servers=[], jobs=[{"id": "qZ", "queue_id": "qZ"}], dump=b"hi\n")
        )
        rc = orch.logs_cmd(argparse.Namespace(via_server=None, job="qZ", follow=False))
        assert rc == 0
        assert client.dumped == "qZ"
        assert capsysbinary.readouterr().out == b"hi\n"
