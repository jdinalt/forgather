"""
Tests for :mod:`forgather_server.cluster_dataset_inventory`.

The inventory module enumerates the dataset_servers this peer knows
about (JobRecord-spawned + user-registered) for the master's
aggregator. Two policy decisions get most of the coverage:

- loopback URLs are excluded (no peer can reach them);
- ``0.0.0.0`` binds are rewritten to the cluster identity's hostname.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from forgather_server import cluster_dataset_inventory
from forgather_server.cluster import NodeIdentity
from forgather_server.dataset_server_registry import RegistryEntry
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


def _patch_sources(
    monkeypatch,
    *,
    records=None,
    entries=None,
    identity: Optional[NodeIdentity] = None,
):
    monkeypatch.setattr(
        cluster_dataset_inventory.job_records,
        "list_records",
        lambda: list(records or []),
    )
    monkeypatch.setattr(
        cluster_dataset_inventory.dataset_server_registry,
        "list_entries",
        lambda: list(entries or []),
    )
    monkeypatch.setattr(
        cluster_dataset_inventory.cluster,
        "self_identity",
        lambda: identity,
    )


def _record(
    *,
    port: int = 8766,
    host: str = "0.0.0.0",
    status: str = "running",
    auth_token: Optional[str] = "tok",
    config: str = "config.yaml",
    tls: Optional[bool] = False,
    queue_id: str = "q1",
) -> JobRecord:
    params = {"host": host, "port": port}
    if tls is not None:
        params["tls"] = tls
    return JobRecord(
        queue_id=queue_id,
        job_type="dataset_server",
        status=status,
        config=config,
        job_params=params,
        auth_token=auth_token,
    )


class TestLocalServersFromJobRecords:
    def test_zero_bound_rewrites_to_cluster_hostname(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[_record(host="0.0.0.0", port=8766, auth_token="tok-abc")],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_dataset_inventory.local_servers()
        assert len(servers) == 1
        s = servers[0]
        assert s.base_url == "http://node-a:8766"
        assert s.auth_token == "tok-abc"
        assert s.source == "local"
        assert s.peer_node_id == "node-a-id"
        assert s.server_id == cluster_dataset_inventory.server_id_for(
            "http://node-a:8766"
        )

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_excluded(self, monkeypatch, host):
        _patch_sources(
            monkeypatch,
            records=[_record(host=host)],
            identity=_identity(),
        )
        assert cluster_dataset_inventory.local_servers() == []

    def test_routable_host_used_as_is(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[_record(host="datahost", port=9000)],
            identity=_identity(),
        )
        servers = cluster_dataset_inventory.local_servers()
        assert servers[0].base_url == "http://datahost:9000"

    def test_skips_non_dataset_server_records(self, monkeypatch):
        other = JobRecord(
            queue_id="q1",
            job_type="inference",
            status="running",
            job_params={"host": "node-a", "port": 8000},
            auth_token="x",
        )
        _patch_sources(monkeypatch, records=[other], identity=_identity())
        assert cluster_dataset_inventory.local_servers() == []

    @pytest.mark.parametrize("status", ["queued", "done", "failed", "cancelled"])
    def test_skips_non_running_records(self, monkeypatch, status):
        _patch_sources(
            monkeypatch,
            records=[_record(status=status)],
            identity=_identity(),
        )
        assert cluster_dataset_inventory.local_servers() == []

    def test_no_cluster_identity_drops_zero_bound(self, monkeypatch):
        """When the server isn't in cluster mode (or hasn't activated
        yet), 0.0.0.0 binds can't be rewritten to a peer-visible host
        and so don't make it into the inventory. Loopback binds are
        already excluded, leaving the result empty — matching the
        "single-node, nothing to share" expectation."""
        _patch_sources(
            monkeypatch,
            records=[_record(host="0.0.0.0")],
            identity=None,
        )
        assert cluster_dataset_inventory.local_servers() == []

    def test_tls_flag_promotes_https(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[_record(host="0.0.0.0", port=8766, tls=True)],
            identity=_identity(hostname="node-a"),
        )
        assert cluster_dataset_inventory.local_servers()[0].base_url == (
            "https://node-a:8766"
        )


class TestLocalServersFromRegistry:
    def test_user_entry_included(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            entries=[
                RegistryEntry(
                    id="r1",
                    label="prod",
                    base_url="http://otherhost:8766",
                    auth_token="utok",
                )
            ],
            identity=_identity(),
        )
        servers = cluster_dataset_inventory.local_servers()
        assert len(servers) == 1
        s = servers[0]
        assert s.source == "user"
        assert s.base_url == "http://otherhost:8766"
        assert s.auth_token == "utok"
        assert s.label == "prod"

    def test_trailing_slash_normalized(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            entries=[
                RegistryEntry(
                    id="r1",
                    label="x",
                    base_url="http://otherhost:8766/",
                    auth_token="",
                )
            ],
            identity=_identity(),
        )
        assert (
            cluster_dataset_inventory.local_servers()[0].base_url
            == "http://otherhost:8766"
        )

    def test_loopback_user_entry_excluded(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            entries=[
                RegistryEntry(
                    id="r1",
                    label="x",
                    base_url="http://127.0.0.1:8766",
                    auth_token="u",
                )
            ],
            identity=_identity(),
        )
        assert cluster_dataset_inventory.local_servers() == []


class TestDedup:
    def test_jobrecord_wins_over_user_entry_for_same_url(self, monkeypatch):
        """If the same URL appears as both a JobRecord-spawned server
        and a user-registered entry, the JobRecord wins. Its label
        carries the spawning config name, which is more diagnostic
        than the user's free-text label, and the JobRecord token is
        guaranteed to be live."""
        _patch_sources(
            monkeypatch,
            records=[_record(host="0.0.0.0", port=8766, auth_token="job-tok")],
            entries=[
                RegistryEntry(
                    id="r1",
                    label="user-label",
                    base_url="http://node-a:8766",
                    auth_token="user-tok",
                )
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_dataset_inventory.local_servers()
        assert len(servers) == 1
        assert servers[0].source == "local"
        assert servers[0].auth_token == "job-tok"


class TestServerIdFor:
    def test_stable_across_calls(self):
        a = cluster_dataset_inventory.server_id_for("http://x:8766")
        b = cluster_dataset_inventory.server_id_for("http://x:8766")
        assert a == b
        assert len(a) == 12

    def test_different_urls_differ(self):
        assert cluster_dataset_inventory.server_id_for(
            "http://a:8766"
        ) != cluster_dataset_inventory.server_id_for("http://b:8766")
