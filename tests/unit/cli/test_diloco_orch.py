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
        self.enqueued = []
        self.base = "http://fake-orch:8765"

    def enqueue_job(self, **kw):
        self.enqueued.append(kw)
        return {"queue_id": f"q{len(self.enqueued)}"}

    def generate_diloco_worker_names(self, count, exclude=None):
        self.gen_call = (count, list(exclude or []))
        return {"names": [f"name{i}" for i in range(count)]}

    def ping(self):
        return self._reachable

    def list_diloco_servers(self):
        return self._servers

    def list_jobs(self, include_dead=False):
        return self._jobs

    def job_dump(self, job_id):
        self.dumped = job_id
        return self._dump

    def add_diloco_registry(
        self, *, base_url, label=None, auth_token=None, verify_tls=True
    ):
        self.added = dict(
            base_url=base_url, label=label, auth_token=auth_token, verify_tls=verify_tls
        )
        # The real POST /diloco/registry returns a BARE id (token_hex(4));
        # the "registered:" prefix is only applied by `diloco servers`.
        return {
            "id": "a1b2c3d4",
            "label": label or base_url,
            "base_url": base_url,
            "has_auth_token": bool(auth_token),
            "verify_tls": verify_tls,
        }

    def delete_diloco_registry(self, entry_id):
        self.deleted_id = entry_id
        return {"deleted": entry_id}

    def diloco_server_control(self, action, base, command=None, worker_id=None):
        self.control_call = (action, base, command, worker_id)
        return {"status": "ok", "command": command, "workers": ["w0", "w1"]}

    def diloco_server_status(self, base):
        return {"workers": {}}


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


def _loc_args(**over):
    base = dict(
        server="local:q1", via_server=None, local_only=False, local_fallback=False
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestResolveOrchestratorBase:
    def test_local_only_skips(self, patch_orchestrator):
        patch_orchestrator(FakeClient(servers=SERVERS))
        assert orch.resolve_orchestrator_base(_loc_args(local_only=True)) == (
            None,
            None,
        )

    def test_down_default_raises(self, patch_orchestrator):
        from forgather.cli.server_client import ServerUnreachable

        patch_orchestrator(FakeClient(servers=SERVERS, reachable=False))
        with pytest.raises(ServerUnreachable):
            orch.resolve_orchestrator_base(_loc_args())

    def test_down_with_fallback_goes_direct(self, patch_orchestrator):
        patch_orchestrator(FakeClient(servers=SERVERS, reachable=False))
        assert orch.resolve_orchestrator_base(_loc_args(local_fallback=True)) == (
            None,
            None,
        )

    def test_reachable_known_target(self, patch_orchestrator):
        client = patch_orchestrator(FakeClient(servers=SERVERS))
        c, base = orch.resolve_orchestrator_base(_loc_args())
        assert c is client
        assert base == "http://192.168.9.43:8512"

    def test_up_unknown_target_default_raises(self, patch_orchestrator):
        from forgather.cli.server_client import ServerUnreachable

        patch_orchestrator(FakeClient(servers=SERVERS))
        with pytest.raises(ServerUnreachable):
            orch.resolve_orchestrator_base(_loc_args(server="localhost:8512"))

    def test_up_unknown_target_with_fallback_goes_direct(self, patch_orchestrator):
        patch_orchestrator(FakeClient(servers=SERVERS))
        assert orch.resolve_orchestrator_base(
            _loc_args(server="localhost:8512", local_fallback=True)
        ) == (None, None)


class TestRegistry:
    def test_register_json(self, patch_orchestrator, capsys):
        client = patch_orchestrator(FakeClient())
        args = argparse.Namespace(
            via_server=None,
            url="https://h:8512",
            label="L",
            auth_token="tok",
            no_verify_tls=True,
            json=True,
        )
        rc = orch.register_cmd(args)
        assert rc == 0
        # Forwarded with verify_tls inverted from --no-verify-tls.
        assert client.added == {
            "base_url": "https://h:8512",
            "label": "L",
            "auth_token": "tok",
            "verify_tls": False,
        }
        # JSON is the entry verbatim — bare id, as the server returns it.
        assert json.loads(capsys.readouterr().out)["id"] == "a1b2c3d4"

    def test_register_human_shows_prefixed_id(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient())
        args = argparse.Namespace(
            via_server=None,
            url="https://h:8512",
            label=None,
            auth_token=None,
            no_verify_tls=False,
            json=False,
        )
        rc = orch.register_cmd(args)
        assert rc == 0
        # Human output uses the copy-pasteable "registered:<id>" form.
        assert "registered:a1b2c3d4" in capsys.readouterr().out

    def test_unregister_strips_prefix(self, patch_orchestrator, capsys):
        client = patch_orchestrator(FakeClient())
        rc = orch.unregister_cmd(
            argparse.Namespace(via_server=None, entry_id="registered:abcd")
        )
        assert rc == 0
        assert client.deleted_id == "abcd"  # "registered:" prefix stripped


class TestControlOps:
    def test_orchestrator_ops_relay(self, monkeypatch):
        client = FakeClient(servers=SERVERS)
        monkeypatch.setattr(
            orch, "resolve_orchestrator_base", lambda args: (client, "http://h:8512")
        )
        ops, label = orch.make_control_ops(argparse.Namespace(server="local:q1"))
        assert isinstance(ops, orch._OrchestratorOps)
        assert "via forgather server" in label
        ops.relay("save_checkpoint", worker_id="w0")
        assert client.control_call == (
            "command",
            "http://h:8512",
            "save_checkpoint",
            "w0",
        )

    def test_direct_ops_fallback(self, monkeypatch):
        monkeypatch.setattr(
            orch, "resolve_orchestrator_base", lambda args: (None, None)
        )
        args = argparse.Namespace(
            server="localhost:8512", auth_token=None, no_verify_tls=False
        )
        ops, label = orch.make_control_ops(args)
        assert isinstance(ops, orch._DirectOps)
        assert label == "localhost:8512"


class TestDatasetSource:
    def test_none_and_local(self):
        assert orch.parse_dataset_source(None) is None
        assert orch.parse_dataset_source("local") is None

    def test_auto(self):
        assert orch.parse_dataset_source("auto") == {"kind": "auto"}

    def test_server(self):
        assert orch.parse_dataset_source("server:user:abc") == {
            "kind": "server",
            "server_id": "user:abc",
        }

    def test_invalid(self):
        with pytest.raises(ValueError):
            orch.parse_dataset_source("bogus")


def _server_args(**over):
    base = dict(
        output_dir="/m",
        port=8512,
        num_workers=2,
        host="127.0.0.1",
        async_mode=False,
        dylu=False,
        save_every=10,
        save_total_limit=3,
        outer_lr=0.7,
        outer_momentum=0.9,
        no_nesterov=False,
        heartbeat_timeout=120.0,
        min_workers=1,
        sync_every=500,
        num_fragments=1,
        bf16_comm=True,
        no_auth=False,
        bulk_cleartext=True,
        dn_buffer_size=0,
        from_checkpoint=None,
        regen_token=False,
        priority=0,
        json=False,
        via_server=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestLaunchServer:
    def test_enqueue_shape(self, patch_orchestrator, capsys):
        client = patch_orchestrator(FakeClient())
        rc = orch.launch_server(_server_args())
        assert rc == 0
        kw = client.enqueued[0]
        assert kw["job_type"] == "diloco_server"
        assert kw["config"] == "diloco:8512"
        assert kw["project_dir"] == "/m"
        assert kw["requested_gpus"] == 0
        jp = kw["job_params"]
        assert jp["num_workers"] == 2
        assert jp["bulk_cleartext"] is True
        # dylu off → no dylu_base_sync_every; auth on but regen off → no regen_token
        assert "dylu_base_sync_every" not in jp
        assert "regen_token" not in jp


def _worker_args(**over):
    base = dict(
        server="local:q1",
        worker_id=None,
        heartbeat_interval=30.0,
        devices=None,
        dry_run=False,
        count=1,
        dataset=None,
        gpus_per_worker=1,
        priority=0,
        via_server=None,
        direct=False,
        json=False,
        config_template="cfg",
        project_dir="/p",
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestLaunchWorkers:
    def test_count_two_autonames_and_resolves_server(self, patch_orchestrator):
        client = patch_orchestrator(FakeClient(servers=SERVERS))
        rc = orch.launch_workers(
            _worker_args(count=2, dataset="auto"), {"max_steps": 5}
        )
        assert rc == 0
        assert client.gen_call == (2, [])
        assert len(client.enqueued) == 2
        kw = client.enqueued[0]
        assert kw["job_type"] == "training"
        assert kw["config"] == "cfg"
        # server resolved local:q1 -> its base_url
        assert kw["job_params"]["diloco"]["server_addr"] == "http://192.168.9.43:8512"
        assert kw["job_params"]["diloco"]["worker_id"] == "name0"
        assert kw["dataset_source"] == {"kind": "auto"}
        assert kw["dynamic_args"] == {"max_steps": 5}
        assert kw["requested_gpus"] == 1

    def test_explicit_single_worker_no_generate(self, patch_orchestrator):
        client = patch_orchestrator(FakeClient(servers=[]))
        rc = orch.launch_workers(
            _worker_args(worker_id="w0", count=1, server="h:8512"), {}
        )
        assert rc == 0
        assert not hasattr(client, "gen_call")  # explicit id, count 1 → no generation
        assert client.enqueued[0]["job_params"]["diloco"]["worker_id"] == "w0"
        # unknown server passed through verbatim
        assert client.enqueued[0]["job_params"]["diloco"]["server_addr"] == "h:8512"

    def test_missing_config_errors(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient())
        rc = orch.launch_workers(_worker_args(config_template=None), {})
        assert rc == 1
        assert "needs a config" in capsys.readouterr().err

    def test_bad_dataset_errors(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient())
        rc = orch.launch_workers(_worker_args(dataset="nope"), {})
        assert rc == 1


class TestUseOrchestrator:
    def test_local_only(self):
        assert orch.use_orchestrator(_loc_args(local_only=True)) is None

    def test_up_returns_client(self, monkeypatch):
        up = FakeClient(reachable=True)
        monkeypatch.setattr(orch, "_orchestrator", lambda args: up)
        assert orch.use_orchestrator(_loc_args()) is up

    def test_down_default_raises(self, monkeypatch):
        from forgather.cli.server_client import ServerUnreachable

        down = FakeClient(reachable=False)
        monkeypatch.setattr(orch, "_orchestrator", lambda args: down)
        with pytest.raises(ServerUnreachable):
            orch.use_orchestrator(_loc_args())

    def test_down_with_fallback_returns_none(self, monkeypatch):
        down = FakeClient(reachable=False)
        monkeypatch.setattr(orch, "_orchestrator", lambda args: down)
        assert orch.use_orchestrator(_loc_args(local_fallback=True)) is None


class TestDynamicCliReconstruction:
    """Direct-path forwarding rebuilds first-class `forgather train` flags
    from the parsed dynamic-arg dict (forgather train rejects --dynamic-args)."""

    SCHEMA = [
        {"names": ["--max-steps"], "type": "int"},
        {"names": ["--compile"], "action": "store_true"},
        {"names": ["--no-accelerator"], "action": "store_true"},
        {"names": "--keep-fp32", "action": "store_false"},
    ]

    def _recon(self, dynamic_args):
        from forgather.cli.diloco import _dynamic_cli_from_schema

        return _dynamic_cli_from_schema(self.SCHEMA, dynamic_args)

    def test_valued_arg(self):
        assert self._recon({"max_steps": 5}) == ["--max-steps", "5"]

    def test_store_true_set(self):
        assert self._recon({"compile": True}) == ["--compile"]

    def test_store_true_default_false_omitted(self):
        # Unset store_true bools show up as False in the dict; don't forward.
        assert self._recon({"no_accelerator": False}) == []

    def test_store_false_emits_only_when_false(self):
        assert self._recon({"keep_fp32": False}) == ["--keep-fp32"]
        assert self._recon({"keep_fp32": True}) == []

    def test_unknown_arg_falls_back_to_valued(self):
        assert self._recon({"some_path": "/x"}) == ["--some-path", "/x"]


class TestStatusExitCode:
    def test_orchestrator_upstream_down_exits_nonzero(self, monkeypatch, capsys):
        """A dead upstream reached through the orchestrator must exit
        non-zero (scriptability), not print a healthy-looking 'unknown'."""
        from forgather.cli import diloco

        class DeadClient:
            def diloco_server_status(self, base):
                raise RuntimeError("server: 502 upstream unreachable")

            def diloco_server_info(self, base):
                return {}

            def diloco_known_workers(self, base):
                return {}

            def diloco_work_queues(self, base):
                return []

        monkeypatch.setattr(
            orch,
            "resolve_orchestrator_base",
            lambda args: (DeadClient(), "http://h:8512"),
        )
        args = argparse.Namespace(
            queues=False,
            json=True,
            direct=False,
            server="local:q1",
            via_server=None,
            auth_token=None,
            no_verify_tls=False,
        )
        rc = diloco._status_cmd(args)
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload


class TestLogsCmd:
    def test_dump(self, patch_orchestrator, capsysbinary):
        client = patch_orchestrator(
            FakeClient(servers=[], jobs=[{"id": "qZ", "queue_id": "qZ"}], dump=b"hi\n")
        )
        rc = orch.logs_cmd(argparse.Namespace(via_server=None, job="qZ", follow=False))
        assert rc == 0
        assert client.dumped == "qZ"
        assert capsysbinary.readouterr().out == b"hi\n"
