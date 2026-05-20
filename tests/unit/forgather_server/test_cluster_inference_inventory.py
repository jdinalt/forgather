"""
Tests for :mod:`forgather_server.cluster_inference_inventory`.

Mirrors :mod:`tests.unit.forgather_server.test_cluster_dataset_inventory`,
adapted for the inference variant. Covers:

- Local-server enumeration from JobRecords (single-model + multi-model
  parameter shapes, loopback / 0.0.0.0 rewriting, scheme detection).
- The ``models`` field extraction from ``job_params``.
- ``MasterInventory.merge_servers`` health-carry-over and the
  ``token_for_url`` reverse lookup the proxy relies on.
- Master role transitions wiping the cached inventory.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pytest

from forgather_server import cluster_inference_inventory
from forgather_server.cluster import NodeIdentity
from forgather_server.job_records import JobRecord


def _identity(hostname: str = "node-a", node_id: str = "node-a-id") -> NodeIdentity:
    return NodeIdentity(
        node_id=node_id,
        hostname=hostname,
        cluster_name="test",
        port=8765,
        forgather_version="0.0.0",
        started_at=time.time(),
    )


def _patch_sources(monkeypatch, *, records=None, identity: Optional[NodeIdentity] = None):
    monkeypatch.setattr(
        cluster_inference_inventory.job_records,
        "list_records",
        lambda: list(records or []),
    )
    monkeypatch.setattr(
        cluster_inference_inventory.cluster,
        "self_identity",
        lambda: identity,
    )


def _record(
    *,
    port: int = 8137,
    host: str = "0.0.0.0",
    status: str = "running",
    auth_token: Optional[str] = "tok",
    config: str = "inference:8137",
    tls: Optional[bool] = False,
    queue_id: str = "q1",
    extra_params: Optional[Dict[str, Any]] = None,
) -> JobRecord:
    params: Dict[str, Any] = {"host": host, "port": port}
    if tls is not None:
        params["tls"] = tls
    if extra_params:
        params.update(extra_params)
    return JobRecord(
        queue_id=queue_id,
        job_type="inference",
        status=status,
        config=config,
        job_params=params,
        auth_token=auth_token,
    )


class TestLocalServersFromJobRecords:
    def test_zero_bound_rewrites_to_cluster_hostname(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[
                _record(
                    host="0.0.0.0",
                    port=8137,
                    auth_token="tok-abc",
                    extra_params={"model_path": "/models/llama"},
                )
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_inference_inventory.local_servers()
        assert len(servers) == 1
        s = servers[0]
        assert s.base_url == "http://node-a:8137"
        assert s.auth_token == "tok-abc"
        assert s.peer_node_id == "node-a-id"
        assert s.source_id == "q1"
        assert s.loopback is False
        # Single-model record: models[] gets a one-element list derived
        # from the model_path basename.
        assert s.models == ["llama"]
        assert s.server_id == cluster_inference_inventory.server_id_for(
            "http://node-a:8137"
        )

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_kept_but_flagged(self, monkeypatch, host):
        """Loopback binds stay in the inventory (single-node operators
        still see them in the picker) but are marked ``loopback=True``
        so cluster-aware filtering can skip them when appropriate."""
        _patch_sources(
            monkeypatch,
            records=[_record(host=host)],
            identity=_identity(),
        )
        servers = cluster_inference_inventory.local_servers()
        assert len(servers) == 1
        assert servers[0].loopback is True

    def test_routable_host_overrides_zero_bind(self, monkeypatch):
        """When the scheduler has stamped a ``routable_host`` it wins
        over the 0.0.0.0 rewrite."""
        _patch_sources(
            monkeypatch,
            records=[
                _record(
                    host="0.0.0.0",
                    port=8137,
                    extra_params={"routable_host": "lan-host.local"},
                )
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_inference_inventory.local_servers()
        assert servers[0].base_url == "http://lan-host.local:8137"

    def test_https_scheme_detected(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[
                _record(
                    host="0.0.0.0",
                    port=8137,
                    extra_params={"scheme": "https"},
                )
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_inference_inventory.local_servers()
        assert servers[0].base_url == "https://node-a:8137"

    def test_non_running_records_skipped(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[
                _record(status="done", queue_id="finished"),
                _record(status="running", queue_id="alive"),
            ],
            identity=_identity(),
        )
        servers = cluster_inference_inventory.local_servers()
        assert [s.source_id for s in servers] == ["alive"]

    def test_non_inference_records_skipped(self, monkeypatch):
        rec = _record()
        rec.job_type = "dataset_server"
        _patch_sources(
            monkeypatch,
            records=[rec],
            identity=_identity(),
        )
        assert cluster_inference_inventory.local_servers() == []


class TestModelsExtraction:
    """``LocalInference.models`` is populated from ``job_params``
    without any network call. Three input shapes need to work."""

    def test_multi_model_with_explicit_names(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[
                _record(
                    extra_params={
                        "models": [
                            {"name": "alpha", "path": "/models/a"},
                            {"name": "beta", "path": "/models/b"},
                        ]
                    }
                )
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_inference_inventory.local_servers()
        assert servers[0].models == ["alpha", "beta"]

    def test_multi_model_names_derived_from_path(self, monkeypatch):
        """When a multi-model entry omits ``name``, the inference
        server's CLI derives it from the path basename. The inventory
        mirrors that convention so the picker shows the same routing
        name the operator would type."""
        _patch_sources(
            monkeypatch,
            records=[
                _record(
                    extra_params={
                        "models": [
                            {"path": "/models/llama-3"},
                            {"path": "/models/gemma/"},
                        ]
                    }
                )
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_inference_inventory.local_servers()
        assert servers[0].models == ["llama-3", "gemma"]

    def test_single_model_path_basename(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[_record(extra_params={"model_path": "/foo/bar/my_model"})],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_inference_inventory.local_servers()
        assert servers[0].models == ["my_model"]

    def test_empty_when_no_model_hint(self, monkeypatch):
        """Legacy / partial records may not carry ``model_path`` or
        ``models``. The inventory returns an empty list rather than
        synthesizing a placeholder — the webui falls back to a generic
        label in that case."""
        _patch_sources(
            monkeypatch,
            records=[_record()],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_inference_inventory.local_servers()
        assert servers[0].models == []


class TestMasterInventory:
    def setup_method(self):
        cluster_inference_inventory._reset_master_state_for_tests()

    def test_merge_carries_health_state(self):
        inv = cluster_inference_inventory.master_inventory
        inv.set_master_state(True)
        first = cluster_inference_inventory.MasterServerEntry(
            server_id="sid-1",
            base_url="http://node-a:8137",
            auth_token="t1",
            label="inference:8137",
            peer_node_id="node-a-id",
        )
        inv.merge_servers({"sid-1": first})
        inv.update_health("sid-1", healthy=True)
        assert inv.servers_snapshot()[0].healthy is True
        assert inv.servers_snapshot()[0].total_health_polls == 1

        # New fresh entry for the same id — health state must carry
        # forward so the picker doesn't flicker between merges.
        second = cluster_inference_inventory.MasterServerEntry(
            server_id="sid-1",
            base_url="http://node-a:8137",
            auth_token="t1-rotated",
            label="inference:8137",
            peer_node_id="node-a-id",
        )
        inv.merge_servers({"sid-1": second})
        s = inv.servers_snapshot()[0]
        assert s.healthy is True
        assert s.total_health_polls == 1
        # Identity fields update though.
        assert s.auth_token == "t1-rotated"

    def test_token_for_url_lookup(self):
        inv = cluster_inference_inventory.master_inventory
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "sid-1": cluster_inference_inventory.MasterServerEntry(
                    server_id="sid-1",
                    base_url="http://node-a:8137",
                    auth_token="secret-1",
                    label="x",
                    peer_node_id="node-a-id",
                ),
                "sid-2": cluster_inference_inventory.MasterServerEntry(
                    server_id="sid-2",
                    base_url="http://node-b:8138",
                    auth_token="",
                    label="y",
                    peer_node_id="node-b-id",
                ),
            }
        )
        # Normalized lookup (trailing slash tolerated).
        assert inv.token_for_url("http://node-a:8137") == "secret-1"
        assert inv.token_for_url("http://node-a:8137/") == "secret-1"
        # Empty auth_token → ``None`` (caller treats as --no-auth).
        assert inv.token_for_url("http://node-b:8138") is None
        # Unknown URL → ``None``.
        assert inv.token_for_url("http://nowhere:9999") is None

    def test_role_transition_clears_inventory(self):
        inv = cluster_inference_inventory.master_inventory
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "sid-1": cluster_inference_inventory.MasterServerEntry(
                    server_id="sid-1",
                    base_url="http://x:1",
                    auth_token="t",
                    label="x",
                    peer_node_id="x",
                )
            }
        )
        assert len(inv.servers_snapshot()) == 1
        # Losing master role drops the cached inventory so a non-
        # master serves nothing rather than a stale snapshot.
        inv.set_master_state(False)
        assert inv.servers_snapshot() == []
        assert inv.is_master() is False
