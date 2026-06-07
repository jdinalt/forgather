"""Unit tests for the orchestrator-backed DiLoCo CLI handlers."""

import argparse
import json
import os

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
        # 192.168.9.43 is routable, not loopback → localhost must not match it.
        assert orch.match_server(SERVERS, "localhost:8512") is None

    def test_empty_target(self):
        assert orch.match_server(SERVERS, None) is None

    # Loopback aliases are equivalent: a server listed under one form
    # matches a --server given as another (the user's localhost/127.0.0.1
    # report).
    LOOPBACK = [{"base_url": "https://127.0.0.1:8512", "source": "local"}]

    def test_loopback_localhost_url_matches_127(self):
        assert (
            orch.match_server(self.LOOPBACK, "https://localhost:8512")
            == "https://127.0.0.1:8512"
        )

    def test_loopback_bare_hostport_matches(self):
        assert (
            orch.match_server(self.LOOPBACK, "localhost:8512")
            == "https://127.0.0.1:8512"
        )

    def test_loopback_ipv6_matches(self):
        assert (
            orch.match_server(self.LOOPBACK, "[::1]:8512") == "https://127.0.0.1:8512"
        )

    def test_loopback_scheme_agnostic(self):
        assert (
            orch.match_server(self.LOOPBACK, "http://localhost:8512")
            == "https://127.0.0.1:8512"
        )

    def test_loopback_different_port_no_match(self):
        assert orch.match_server(self.LOOPBACK, "localhost:9999") is None


class TestResolveOne:
    """Implicit single-server selection when --server is omitted."""

    ONE = [{"id": "local:q1", "base_url": "http://127.0.0.1:8512"}]

    def test_explicit_matches(self):
        assert orch.resolve_one(SERVERS, "local:q1") == "http://192.168.9.43:8512"

    def test_explicit_unknown_is_none(self):
        assert orch.resolve_one(self.ONE, "nope:1") is None

    def test_implicit_single(self):
        assert orch.resolve_one(self.ONE, None) == "http://127.0.0.1:8512"

    def test_implicit_none_when_zero(self):
        assert orch.resolve_one([], None) is None

    def test_implicit_ambiguous_raises(self):
        from forgather.cli.server_client import ServerUnreachable

        with pytest.raises(ServerUnreachable):
            orch.resolve_one(SERVERS, None)  # 2 servers, no --server


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


def test_render_status_shows_aggregate_stats(capsys):
    merged = orch.assemble_status(
        get_status=lambda: {
            "status": "running",
            "aggregate_stats": {
                "total_tokens": 14547721,
                "total_steps": 464,
                "tok_per_sec": 172215.0,
                "mfu": 0.17,
                "train_loss": 7.4327,
                "eval_loss": 6.5,
                "eval_step": 400,
                "num_reporting": 2,
            },
        },
        get_info=lambda: {},
        get_known_workers=lambda: {},
        get_work_queues=None,
    )
    orch.render_status(merged, want_queues=False)
    out = capsys.readouterr().out
    # num_reporting annotates the aggregate header.
    assert "Training stats (aggregate of 2 reporting):" in out
    assert "14,547,721" in out
    assert "172,215 tok/s" in out
    assert "17.0%" in out
    assert "7.4327" in out
    assert "@ step 400" in out


def test_render_status_aggregate_without_num_reporting(capsys):
    # No num_reporting → plain header, no "(aggregate of N reporting)".
    merged = orch.assemble_status(
        get_status=lambda: {
            "status": "running",
            "aggregate_stats": {"total_tokens": 100},
        },
        get_info=lambda: {},
        get_known_workers=lambda: {},
        get_work_queues=None,
    )
    orch.render_status(merged, want_queues=False)
    out = capsys.readouterr().out
    assert "Training stats (aggregate):" in out
    assert "reporting" not in out


def test_render_status_header_fields(capsys):
    # Outer-optimizer config, save_dir, min_workers, fragment_submissions —
    # all sourced from /status (save_dir falls back to info.output_dir).
    merged = orch.assemble_status(
        get_status=lambda: {
            "status": "running",
            "outer_lr": 0.7,
            "outer_momentum": 0.9,
            "heartbeat_timeout": 120,
            "min_workers": 2,
            "fragment_submissions": 12,
        },
        get_info=lambda: {"output_dir": "/tmp/run42"},
        get_known_workers=lambda: {},
        get_work_queues=None,
    )
    orch.render_status(merged, want_queues=False)
    out = capsys.readouterr().out
    # No outer_optimizer field → reconstruct from lr/momentum (old servers).
    assert "SGD(lr=0.7, momentum=0.9)" in out
    assert "Save dir:      /tmp/run42" in out
    assert "min workers: 2" in out
    assert "Frag submits:  12" in out


def test_render_status_prefers_server_optimizer_description(capsys):
    # When the server supplies the full optimizer description, render it
    # verbatim (shows nesterov etc.) instead of reconstructing SGD(lr, mom).
    merged = orch.assemble_status(
        get_status=lambda: {
            "status": "running",
            "outer_optimizer": "SGD(lr=0.7, momentum=0.9, nesterov=True)",
            "outer_lr": 0.7,
            "outer_momentum": 0.9,
        },
        get_info=lambda: {},
        get_known_workers=lambda: {},
        get_work_queues=None,
    )
    orch.render_status(merged, want_queues=False)
    out = capsys.readouterr().out
    assert "Outer opt:     SGD(lr=0.7, momentum=0.9, nesterov=True)" in out
    # The reconstructed two-field form must not also appear.
    assert "SGD(lr=0.7, momentum=0.9)\n" not in out


class TestWorkerProgress:
    """Per-worker progress cell from the DiLoCoCallback stats (#125)."""

    def test_step_and_max(self):
        cell = orch._worker_progress({"step_total": 4672, "max_steps": 8030})
        assert "4,672/8,030" in cell
        assert "58%" in cell
        assert cell.startswith("[") and "#" in cell and "-" in cell  # bar

    def test_complete(self):
        cell = orch._worker_progress({"step_total": 8030, "max_steps": 8030})
        assert "100%" in cell
        assert "-" not in cell.split("]")[0]  # bar fully filled

    def test_step_only_no_target(self):
        # Older worker / no max_steps reported → bare step count, no bar.
        assert orch._worker_progress({"step_total": 500}) == "500"

    def test_missing(self):
        assert orch._worker_progress({}) == "—"
        assert orch._worker_progress(None) == "—"


def test_render_status_workers_show_progress(capsys):
    merged = orch.assemble_status(
        get_status=lambda: {
            "status": "running",
            "workers": {
                "w0": {
                    "hostname": "h",
                    "sync_round": 9,
                    "steps_per_second": 5.0,
                    "stats": {"step_total": 4672, "max_steps": 8030},
                }
            },
        },
        get_info=lambda: {},
        get_known_workers=lambda: {},
        get_work_queues=None,
    )
    orch.render_status(merged, want_queues=False)
    out = capsys.readouterr().out
    assert "Progress" in out  # column header
    assert "4,672/8,030" in out
    assert "58%" in out


def test_render_status_known_workers_resumable(capsys):
    merged = orch.assemble_status(
        get_status=lambda: {"status": "running"},
        get_info=lambda: {},
        get_known_workers=lambda: {
            "workers": [
                {"worker_id": "live-one", "running": True},
                {
                    "worker_id": "stopped-one",
                    "running": False,
                    "last_registered": 1_700_000_000,
                },
            ]
        },
        get_work_queues=None,
    )
    orch.render_status(merged, want_queues=False)
    out = capsys.readouterr().out
    assert "Known workers: 2 (1 running)" in out
    assert "Resumable (not running):" in out
    assert "stopped-one" in out
    # The running worker is not offered as resumable.
    assert "live-one" not in out.split("Resumable")[1]


def test_render_status_work_dispatch_label_and_by_worker(capsys):
    # The detail getter attaches by_worker; the hint yields a readable label
    # while the raw dataset_id stays visible as a secondary line.
    def detail(ds, seed):
        assert ds == "abc123" and seed == 0
        return {
            "by_worker": {
                "w0": {"units_issued": 5, "units_completed": 4},
                "w1": {"units_issued": 3, "units_completed": 3},
            }
        }

    merged = orch.assemble_status(
        get_status=lambda: {"status": "running"},
        get_info=lambda: {},
        get_known_workers=lambda: {},
        get_work_queues=lambda: [
            {
                "dataset_id": "abc123",
                "shuffle_seed": 0,
                "total_units": 100,
                "issued_count": 8,
                "completed_count": 7,
                "hint": {"length": 50000, "path": "wikitext", "split": "train"},
            }
        ],
        get_work_queue_detail=detail,
    )
    orch.render_status(merged, want_queues=True)
    out = capsys.readouterr().out
    assert "Work-unit dispatch:" in out
    assert "wikitext@train@0: 8/100 issued (8% issued), 7 confirmed" in out
    assert "50,000 rows" in out
    assert "dataset_id: abc123" in out
    assert "w0" in out and "w1" in out


def test_assemble_status_detail_failure_is_tolerated():
    # A failing detail fetch must not drop the queue summary.
    def boom(ds, seed):
        raise RuntimeError("upstream 404")

    merged = orch.assemble_status(
        get_status=lambda: {},
        get_info=lambda: {},
        get_known_workers=lambda: {},
        get_work_queues=lambda: [
            {"dataset_id": "x", "shuffle_seed": 0, "total_units": 1}
        ],
        get_work_queue_detail=boom,
    )
    assert merged["work_queues"][0]["dataset_id"] == "x"
    assert "by_worker" not in merged["work_queues"][0]


def test_render_status_skips_empty_aggregate(capsys):
    # A fresh server with no worker stats yet → no stats block, no zero-wall.
    merged = orch.assemble_status(
        get_status=lambda: {"status": "running", "aggregate_stats": {}},
        get_info=lambda: {},
        get_known_workers=lambda: {},
        get_work_queues=None,
    )
    orch.render_status(merged, want_queues=False)
    assert "Training stats" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Fake ServerClient for handler tests
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(
        self,
        *,
        servers=None,
        jobs=None,
        reachable=True,
        dump=b"",
        tty_path=None,
        known=None,
        cluster_self=None,
        cluster_self_raises=False,
    ):
        self._servers = servers or []
        self._jobs = jobs or []
        self._reachable = reachable
        self._dump = dump
        self._tty_path = tty_path
        self._known = known or {"workers": []}
        self._cluster_self = cluster_self
        self._cluster_self_raises = cluster_self_raises
        self.enqueued = []
        self.base = "http://fake-orch:8765"

    def cluster_self(self):
        # null in standalone mode, an identity dict in cluster mode.
        if self._cluster_self_raises:
            raise RuntimeError("cluster probe failed")
        return self._cluster_self

    def diloco_known_workers(self, base):
        return self._known

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

    def job_tty_path(self, job_id):
        # Mirrors the server: raises (like a 404 → RuntimeError) when no
        # path was configured for this fake.
        if self._tty_path is None:
            raise RuntimeError("server: no TTY log recorded yet")
        return self._tty_path

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
        diloco_server="local:q1", server=None, local_only=False, local_fallback=False
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
            orch.resolve_orchestrator_base(_loc_args(diloco_server="localhost:8512"))

    def test_up_unknown_target_with_fallback_goes_direct(self, patch_orchestrator):
        patch_orchestrator(FakeClient(servers=SERVERS))
        assert orch.resolve_orchestrator_base(
            _loc_args(diloco_server="localhost:8512", local_fallback=True)
        ) == (None, None)

    def test_implicit_single_server(self, patch_orchestrator):
        one = [{"id": "local:q1", "base_url": "http://127.0.0.1:8512"}]
        client = patch_orchestrator(FakeClient(servers=one))
        c, base = orch.resolve_orchestrator_base(_loc_args(diloco_server=None))
        assert c is client and base == "http://127.0.0.1:8512"

    def test_implicit_ambiguous_raises(self, patch_orchestrator):
        from forgather.cli.server_client import ServerUnreachable

        patch_orchestrator(FakeClient(servers=SERVERS))
        with pytest.raises(ServerUnreachable):
            orch.resolve_orchestrator_base(_loc_args(diloco_server=None))

    def test_implicit_zero_servers_raises(self, patch_orchestrator):
        from forgather.cli.server_client import ServerUnreachable

        patch_orchestrator(FakeClient(servers=[]))
        with pytest.raises(ServerUnreachable):
            orch.resolve_orchestrator_base(_loc_args(diloco_server=None))


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
        ops, label = orch.make_control_ops(argparse.Namespace(diloco_server="local:q1"))
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
            diloco_server="localhost:8512", auth_token=None, no_verify_tls=False
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


class TestResolveDatasetSource:
    """Mode-aware default for an unset --dataset (mirrors the webui)."""

    def test_explicit_value_wins_even_in_cluster(self):
        # An explicit --dataset is honored verbatim regardless of mode.
        client = FakeClient(cluster_self={"node_id": "n1"})
        assert (
            orch.resolve_dataset_source(client, argparse.Namespace(dataset="local"))
            is None
        )
        assert orch.resolve_dataset_source(
            client, argparse.Namespace(dataset="server:user:x")
        ) == {"kind": "server", "server_id": "user:x"}

    def test_unset_defaults_to_auto_in_cluster(self, capsys):
        client = FakeClient(cluster_self={"node_id": "n1", "is_master": True})
        assert orch.resolve_dataset_source(
            client, argparse.Namespace(dataset=None)
        ) == {"kind": "auto"}
        cap = capsys.readouterr()
        # Informational message goes to stderr, never stdout (keeps --json clean).
        assert "auto (cluster routing)" in cap.err
        assert cap.out == ""

    def test_unset_defaults_to_local_in_standalone(self):
        client = FakeClient(cluster_self=None)  # standalone
        assert (
            orch.resolve_dataset_source(client, argparse.Namespace(dataset=None))
            is None
        )

    def test_probe_failure_falls_back_to_local(self):
        client = FakeClient(cluster_self_raises=True)
        assert (
            orch.resolve_dataset_source(client, argparse.Namespace(dataset=None))
            is None
        )


class TestResolveConfigName:
    """Worker launch resolves the config like `forgather train`: explicit -t
    wins, else the project's default_config from meta.yaml."""

    def test_explicit_wins(self):
        ns = argparse.Namespace(config_template="x.yaml", project_dir="/p")
        assert orch._resolve_config_name(ns) == "x.yaml"

    def test_default_from_meta(self, monkeypatch):
        import forgather

        class FakeMeta:
            @staticmethod
            def find_project_dir(p):
                return p

            def __init__(self, pdir):
                pass

            def default_config(self):
                return "default.yaml"

        monkeypatch.setattr(forgather, "MetaConfig", FakeMeta)
        ns = argparse.Namespace(config_template=None, project_dir="/p")
        assert orch._resolve_config_name(ns) == "default.yaml"

    def test_none_when_no_project(self, monkeypatch):
        import forgather

        class FakeMeta:
            @staticmethod
            def find_project_dir(p):
                raise RuntimeError("no meta.yaml")

        monkeypatch.setattr(forgather, "MetaConfig", FakeMeta)
        ns = argparse.Namespace(config_template=None, project_dir="/nope")
        assert orch._resolve_config_name(ns) is None


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

    def test_relative_output_dir_is_absolutized(self, patch_orchestrator):
        # The scheduler launches the job from the repo root, so a relative
        # --output-dir must be resolved against the CLI's CWD before enqueue.
        client = patch_orchestrator(FakeClient())
        rc = orch.launch_server(
            _server_args(output_dir="../../models/small", from_checkpoint="ckpt/x")
        )
        assert rc == 0
        kw = client.enqueued[0]
        assert kw["project_dir"] == os.path.abspath("../../models/small")
        assert kw["job_params"]["output_dir"] == os.path.abspath("../../models/small")
        assert kw["job_params"]["from_checkpoint"] == os.path.abspath("ckpt/x")
        assert os.path.isabs(kw["job_params"]["output_dir"])


def _worker_args(**over):
    base = dict(
        diloco_server="local:q1",
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

    def test_shared_memory_backend_stamps_group(self, patch_orchestrator):
        # --backend shared_memory: every worker of one submit shares a single
        # shm_group_id + shm_group_size (= worker count) in its diloco block, so
        # the scheduler derives one region dir + size for the co-located group.
        client = patch_orchestrator(FakeClient(servers=SERVERS))
        rc = orch.launch_workers(
            _worker_args(count=2, backend="shared_memory", dataset="auto"), {}
        )
        assert rc == 0
        assert len(client.enqueued) == 2
        dilocos = [kw["job_params"]["diloco"] for kw in client.enqueued]
        for d in dilocos:
            assert d["backend"] == "shared_memory"
            assert d["shm_group_size"] == 2
        # One id for the whole submit (uniform across workers); worker ids stay
        # distinct.
        gid = dilocos[0]["shm_group_id"]
        assert gid and all(d["shm_group_id"] == gid for d in dilocos)
        assert dilocos[0]["worker_id"] != dilocos[1]["worker_id"]

    def test_http_backend_omits_group(self, patch_orchestrator):
        # The default http backend stamps no shm_group fields.
        client = patch_orchestrator(FakeClient(servers=SERVERS))
        rc = orch.launch_workers(
            _worker_args(count=2, backend="http", dataset="auto"), {}
        )
        assert rc == 0
        for kw in client.enqueued:
            d = kw["job_params"]["diloco"]
            assert "backend" not in d
            assert "shm_group_id" not in d
            assert "shm_group_size" not in d

    def test_explicit_single_worker_no_generate(self, patch_orchestrator):
        client = patch_orchestrator(FakeClient(servers=[]))
        rc = orch.launch_workers(
            _worker_args(worker_id="w0", count=1, diloco_server="h:8512"), {}
        )
        assert rc == 0
        assert not hasattr(client, "gen_call")  # explicit id, count 1 → no generation

    def test_requested_gpus_zero_preserved(self, patch_orchestrator):
        # --requested-gpus 0 (the no-reservation host-CUDA escape hatch) must
        # not be rewritten to 1 by the falsy fallback.
        client = patch_orchestrator(FakeClient(servers=[]))
        rc = orch.launch_workers(
            _worker_args(worker_id="w0", diloco_server="h:8512", requested_gpus=0), {}
        )
        assert rc == 0
        assert client.enqueued[0]["requested_gpus"] == 0
        assert client.enqueued[0]["job_params"]["diloco"]["worker_id"] == "w0"
        # unknown server passed through verbatim
        assert client.enqueued[0]["job_params"]["diloco"]["server_addr"] == "h:8512"

    def test_unset_dataset_defaults_auto_in_cluster(self, patch_orchestrator):
        # --dataset omitted + server in cluster mode → auto, like the webui.
        client = patch_orchestrator(
            FakeClient(servers=SERVERS, cluster_self={"node_id": "n1"})
        )
        rc = orch.launch_workers(_worker_args(worker_id="w0", dataset=None), {})
        assert rc == 0
        assert client.enqueued[0]["dataset_source"] == {"kind": "auto"}

    def test_unset_dataset_defaults_local_in_standalone(self, patch_orchestrator):
        # --dataset omitted + standalone server → local (no dataset_source).
        client = patch_orchestrator(FakeClient(servers=SERVERS, cluster_self=None))
        rc = orch.launch_workers(_worker_args(worker_id="w0", dataset=None), {})
        assert rc == 0
        assert client.enqueued[0]["dataset_source"] is None

    def test_json_stdout_clean_with_auto_default(self, patch_orchestrator, capsys):
        # --json + unset --dataset in cluster mode: the "dataset source: auto"
        # note must go to stderr so stdout stays parseable JSON.
        client = patch_orchestrator(
            FakeClient(servers=SERVERS, cluster_self={"node_id": "n1"})
        )
        rc = orch.launch_workers(
            _worker_args(worker_id="w0", dataset=None, json=True), {}
        )
        assert rc == 0
        cap = capsys.readouterr()
        parsed = json.loads(cap.out)  # raises if stdout was polluted
        assert parsed[0]["worker_id"] == "w0"
        assert "auto (cluster routing)" in cap.err

    def test_missing_config_errors(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient())
        rc = orch.launch_workers(_worker_args(config_template=None), {})
        assert rc == 1
        assert "needs a config" in capsys.readouterr().err

    def test_bad_dataset_errors(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient())
        rc = orch.launch_workers(_worker_args(dataset="nope"), {})
        assert rc == 1

    def test_no_dash_t_uses_project_default_config(
        self, patch_orchestrator, monkeypatch
    ):
        # No -t: the enqueued job's config is the project's default_config
        # (resolved like `forgather train`), not an error.
        client = patch_orchestrator(FakeClient(servers=SERVERS))
        monkeypatch.setattr(orch, "_resolve_config_name", lambda args: "default.yaml")
        rc = orch.launch_workers(_worker_args(config_template=None, worker_id="w0"), {})
        assert rc == 0
        assert client.enqueued[0]["config"] == "default.yaml"

    def test_implicit_single_server(self, patch_orchestrator):
        one = [{"id": "local:q1", "base_url": "http://127.0.0.1:8512"}]
        client = patch_orchestrator(FakeClient(servers=one))
        rc = orch.launch_workers(
            _worker_args(diloco_server=None, worker_id="w0", count=1), {}
        )
        assert rc == 0
        assert (
            client.enqueued[0]["job_params"]["diloco"]["server_addr"]
            == "http://127.0.0.1:8512"
        )

    def test_implicit_no_servers_errors(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient(servers=[]))
        rc = orch.launch_workers(
            _worker_args(diloco_server=None, worker_id="w0", count=1), {}
        )
        assert rc == 1
        assert "no DiLoCo server" in capsys.readouterr().err

    def test_implicit_ambiguous_errors(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient(servers=SERVERS))
        rc = orch.launch_workers(
            _worker_args(diloco_server=None, worker_id="w0", count=1), {}
        )
        assert rc == 1


class TestStoppedBaseWorkers:
    def test_filters_running_and_dedups_pp(self):
        known = {
            "workers": [
                {"worker_id": "alpha", "running": False},
                {"worker_id": "beta", "running": True},
                # pipeline ranks of one stopped worker → one base, deduped
                {"worker_id": "gamma_pp0", "running": False},
                {"worker_id": "gamma_pp1", "running": False},
                # any running rank → base excluded
                {"worker_id": "delta_pp0", "running": True},
                {"worker_id": "delta_pp1", "running": False},
            ]
        }
        assert orch._stopped_base_workers(known) == ["alpha", "gamma"]

    def test_empty(self):
        assert orch._stopped_base_workers({"workers": []}) == []


ONE_SERVER = [{"id": "local:q1", "base_url": "http://127.0.0.1:8512"}]


def _resume_args(**over):
    base = dict(
        server=None,
        via_server=None,
        config_template="cfg",
        project_dir="/p",
        dataset=None,
        heartbeat_interval=None,
        gpus_per_worker=1,
        priority=0,
        json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestLaunchResume:
    def test_resumes_stopped_workers(self, patch_orchestrator):
        known = {
            "workers": [
                {"worker_id": "alpha", "running": False},
                {"worker_id": "beta", "running": True},
                {"worker_id": "gamma", "running": False},
            ]
        }
        client = patch_orchestrator(FakeClient(servers=ONE_SERVER, known=known))
        rc = orch.launch_resume(_resume_args(), {})
        assert rc == 0
        # Only the stopped workers, reusing their ids, against the server.
        got = {
            (kw["job_params"]["diloco"]["worker_id"], kw["job_type"])
            for kw in client.enqueued
        }
        assert got == {("alpha", "training"), ("gamma", "training")}
        assert all(
            kw["job_params"]["diloco"]["server_addr"] == "http://127.0.0.1:8512"
            for kw in client.enqueued
        )

    def test_nothing_to_resume(self, patch_orchestrator, capsys):
        known = {"workers": [{"worker_id": "alpha", "running": True}]}
        client = patch_orchestrator(FakeClient(servers=ONE_SERVER, known=known))
        rc = orch.launch_resume(_resume_args(), {})
        assert rc == 0
        assert client.enqueued == []
        assert "No stopped workers" in capsys.readouterr().out

    def test_ambiguous_server_errors(self, patch_orchestrator, capsys):
        patch_orchestrator(FakeClient(servers=SERVERS, known={"workers": []}))
        rc = orch.launch_resume(_resume_args(server=None), {})
        assert rc == 1


class TestWorkerResumeMode:
    """The --resume mode gating in diloco._worker_cmd."""

    def _args(self, **over):
        base = dict(
            resume_workers=True,
            worker_id=None,
            count=1,
            project_dir=".",
            config_template=None,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_resume_with_worker_id_errors(self, capsys):
        from forgather.cli import diloco

        rc = diloco._worker_cmd(self._args(worker_id="x"))
        assert rc == 1
        assert "can't be combined" in capsys.readouterr().err

    def test_resume_local_only_errors(self, monkeypatch, capsys):
        from forgather.cli import diloco

        monkeypatch.setattr(orch, "use_orchestrator", lambda args: None)
        rc = diloco._worker_cmd(self._args())
        assert rc == 1
        assert "requires the forgather server" in capsys.readouterr().err


class TestWorkerForegroundDefaults:
    """The direct/foreground path through ``_worker_cmd`` — runs when
    the forgather server is unreachable AND ``--local-fallback`` is
    set, or ``--local-only`` is passed. Default values it stamps into
    the spawned trainer's env are part of the user contract."""

    def _args(self, **over):
        base = dict(
            resume_workers=False,
            worker_id=None,
            count=1,
            project_dir=".",
            config_template=None,
            diloco_server=None,
            heartbeat_interval=30.0,
            devices=None,
            dry_run=True,  # Don't actually subprocess.run the child.
            remainder=[],
            _dynamic_args={},
        )
        base.update(over)
        return argparse.Namespace(**base)

    def _force_direct_path(self, monkeypatch):
        # No forgather server; otherwise the orchestrator path takes over
        # and the foreground env-stamping never runs.
        monkeypatch.setattr(orch, "use_orchestrator", lambda args: None)
        # The foreground path validates dynamic args + loads the dynamic
        # schema from the project; stub both so the test doesn't depend
        # on having a real project on disk.
        from forgather.cli import diloco as diloco_mod

        monkeypatch.setattr(diloco_mod, "_load_dynamic_schema", lambda *_: {})
        monkeypatch.setattr(diloco_mod, "_worker_dynamic_args", lambda *_: {})
        from forgather.cli import submit_orch

        monkeypatch.setattr(
            submit_orch, "validate_dynamic_args", lambda *_a, **_k: None
        )

    def _capture_subprocess_env(self, monkeypatch):
        captured = {}

        class _Result:
            returncode = 0

        def fake_run(_argv, env=None):
            captured["env"] = dict(env or {})
            return _Result()

        from forgather.cli import diloco as diloco_mod

        monkeypatch.setattr(diloco_mod.subprocess, "run", fake_run)
        return captured

    def test_no_worker_id_mints_memorable_default(self, monkeypatch):
        """No ``--worker-id`` → ``DILOCO_WORKER_ID`` set to a generated
        ``adjective-species`` name (not the worker.py random hex
        fallback, and definitely not empty)."""
        self._force_direct_path(monkeypatch)
        # Pin the generator so the assertion is concrete.
        from forgather import utils as forgather_utils

        monkeypatch.setattr(forgather_utils, "generate_name", lambda: "merry-otter")
        captured = self._capture_subprocess_env(monkeypatch)

        from forgather.cli import diloco

        rc = diloco._worker_cmd(self._args(dry_run=False))
        assert rc == 0
        assert captured["env"]["DILOCO_WORKER_ID"] == "merry-otter"

    def test_explicit_worker_id_wins(self, monkeypatch):
        """An explicit ``--worker-id`` is preserved verbatim — the
        generator is *not* called to mint a replacement."""
        self._force_direct_path(monkeypatch)
        from forgather import utils as forgather_utils

        # If the generator runs at all the assertion below would fail
        # — pin it to something the operator value won't equal.
        monkeypatch.setattr(forgather_utils, "generate_name", lambda: "should-not-fire")
        captured = self._capture_subprocess_env(monkeypatch)

        from forgather.cli import diloco

        rc = diloco._worker_cmd(self._args(dry_run=False, worker_id="custom-id"))
        assert rc == 0
        assert captured["env"]["DILOCO_WORKER_ID"] == "custom-id"

    def test_generated_name_is_not_queue_id_shaped(self, monkeypatch):
        """Regression: previously the worker_id default leaked the
        queue_id (``q_<timestamp>_<hex>``) into the worker identity.
        The direct path mints a memorable name from a fixed adjective+
        species pool, so it can never be ``q_…``-shaped."""
        self._force_direct_path(monkeypatch)
        captured = self._capture_subprocess_env(monkeypatch)

        from forgather.cli import diloco

        rc = diloco._worker_cmd(self._args(dry_run=False))
        assert rc == 0
        wid = captured["env"]["DILOCO_WORKER_ID"]
        assert wid and not wid.startswith("q_")


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


class TestStatusWatch:
    class _C:
        base = "http://h:8512"

        def diloco_server_status(self, b):
            return {"status": "running", "sync_round": 2, "num_registered": 1}

        def diloco_server_info(self, b):
            return {}

        def diloco_known_workers(self, b):
            return {}

        def diloco_work_queues(self, b):
            return []

    def _args(self, **over):
        base = dict(
            queues=False,
            json=False,
            watch=True,
            interval=1.0,
            local_only=False,
            local_fallback=False,
            server="local:q1",
            via_server=None,
            auth_token=None,
            no_verify_tls=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_watch_renders_then_interrupt_exits_zero(self, monkeypatch, capsys):
        import time

        from forgather.cli import diloco

        monkeypatch.setattr(
            orch, "resolve_orchestrator_base", lambda args: (self._C(), "http://h:8512")
        )

        # Break the loop on the first sleep, as Ctrl-C would.
        def _boom(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr(time, "sleep", _boom)
        rc = diloco._status_cmd(self._args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "running" in out  # a tick rendered
        assert "every 1s" in out  # the watch header

    def test_watch_json_mutually_exclusive(self, capsys):
        from forgather.cli import diloco

        rc = diloco._status_cmd(self._args(json=True))
        assert rc == 1
        assert "mutually exclusive" in capsys.readouterr().err


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
        rc = orch.logs_cmd(
            argparse.Namespace(via_server=None, job="qZ", follow=False, path=False)
        )
        assert rc == 0
        assert client.dumped == "qZ"
        assert capsysbinary.readouterr().out == b"hi\n"

    def test_path_prints_tty_path(self, patch_orchestrator, capsys):
        # Resolved server-side (job_tty_path), so it works even when the job
        # isn't surfaced by /api/jobs — like the real endpoint-discovered
        # worker. No jobs list needed.
        patch_orchestrator(
            FakeClient(servers=[], jobs=[], tty_path="/cfg/jobs/q_qZ.tty")
        )
        rc = orch.logs_cmd(
            argparse.Namespace(
                via_server=None, job="nebulous-dingo", follow=False, path=True
            )
        )
        assert rc == 0
        assert capsys.readouterr().out.strip() == "/cfg/jobs/q_qZ.tty"

    def test_path_missing_errors(self, patch_orchestrator, capsys):
        # job_tty_path raises (404 → RuntimeError) → exit 1.
        patch_orchestrator(FakeClient(servers=[], jobs=[], tty_path=None))
        rc = orch.logs_cmd(
            argparse.Namespace(via_server=None, job="qZ", follow=False, path=True)
        )
        assert rc == 1
        assert "no TTY log recorded" in capsys.readouterr().err
