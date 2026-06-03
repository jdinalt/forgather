"""
Tests for :mod:`forgather_server.cluster_diloco_inventory`.

Mirrors :mod:`tests.unit.forgather_server.test_cluster_inference_inventory`
adapted for the DiLoCo variant. Covers:

- Local-server enumeration from JobRecords (0.0.0.0 rewrite, loopback
  handling, scheme detection, routable_host override, non-running
  records skipped, non-diloco_server records skipped).
- The user-registry source feeding into the unified ``local_servers``.
- ``MasterInventory.merge_servers`` health-carry-over and the
  ``token_for_url`` / ``verify_tls_for_url`` reverse lookups the
  proxy and CLI rely on.
- Master role transitions wiping the cached inventory.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest

from forgather_server import cluster_diloco_inventory, diloco_server_registry
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


def _patch_sources(
    monkeypatch,
    *,
    records=None,
    registry_entries: Optional[List[diloco_server_registry.RegistryEntry]] = None,
    identity: Optional[NodeIdentity] = None,
):
    monkeypatch.setattr(
        cluster_diloco_inventory.job_records,
        "list_records",
        lambda: list(records or []),
    )
    monkeypatch.setattr(
        cluster_diloco_inventory.diloco_server_registry,
        "list_entries",
        lambda: list(registry_entries or []),
    )
    monkeypatch.setattr(
        cluster_diloco_inventory.cluster,
        "self_identity",
        lambda: identity,
    )


def _record(
    *,
    port: int = 8512,
    host: str = "0.0.0.0",
    status: str = "running",
    auth_token: Optional[str] = "tok",
    config: str = "default",
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
        job_type="diloco_server",
        status=status,
        config=config,
        job_params=params,
        auth_token=auth_token,
    )


class TestLocalServersFromJobRecords:
    def test_zero_bound_rewrites_to_cluster_hostname(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[_record(host="0.0.0.0", port=8512, auth_token="tok-abc")],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_diloco_inventory.local_servers()
        assert len(servers) == 1
        s = servers[0]
        assert s.base_url == "http://node-a:8512"
        assert s.auth_token == "tok-abc"
        assert s.peer_node_id == "node-a-id"
        assert s.source == "local"
        assert s.source_id == "q1"
        assert s.loopback is False
        assert s.server_id == cluster_diloco_inventory.server_id_for(
            "http://node-a:8512"
        )

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_kept_but_flagged(self, monkeypatch, host):
        """Loopback binds stay in the inventory (single-node operators
        still see them in the panel) but ``loopback=True`` excludes
        them from cross-node candidate selection."""
        _patch_sources(
            monkeypatch,
            records=[_record(host=host)],
            identity=_identity(),
        )
        servers = cluster_diloco_inventory.local_servers()
        assert len(servers) == 1
        assert servers[0].loopback is True

    def test_routable_host_overrides_zero_bind(self, monkeypatch):
        """The scheduler stamps ``routable_host`` when it auto-detects
        a LAN address; that wins over the 0.0.0.0 rewrite."""
        _patch_sources(
            monkeypatch,
            records=[
                _record(
                    host="0.0.0.0",
                    port=8512,
                    extra_params={"routable_host": "lan-host.local"},
                )
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_diloco_inventory.local_servers()
        assert servers[0].base_url == "http://lan-host.local:8512"

    def test_https_scheme_detected(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[
                _record(host="0.0.0.0", port=8512, extra_params={"scheme": "https"})
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_diloco_inventory.local_servers()
        assert servers[0].base_url == "https://node-a:8512"

    def test_non_running_records_skipped(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            records=[
                _record(status="done", queue_id="finished"),
                _record(status="running", queue_id="alive"),
            ],
            identity=_identity(),
        )
        servers = cluster_diloco_inventory.local_servers()
        assert [s.source_id for s in servers] == ["alive"]

    def test_non_diloco_records_skipped(self, monkeypatch):
        rec = _record()
        rec.job_type = "inference"
        _patch_sources(
            monkeypatch,
            records=[rec],
            identity=_identity(),
        )
        assert cluster_diloco_inventory.local_servers() == []


class TestUserRegistrySource:
    def test_registry_entries_surface_with_source_user(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            registry_entries=[
                diloco_server_registry.RegistryEntry(
                    id="reg1",
                    label="prod",
                    base_url="https://gpu-box:8512",
                    auth_token="ext-tok",
                    verify_tls=True,
                ),
            ],
            identity=_identity(),
        )
        servers = cluster_diloco_inventory.local_servers()
        assert len(servers) == 1
        s = servers[0]
        assert s.source == "user"
        assert s.label == "prod"
        assert s.base_url == "https://gpu-box:8512"
        assert s.auth_token == "ext-tok"
        assert s.source_id == "reg1"
        assert s.verify_tls is True

    def test_verify_tls_off_propagated(self, monkeypatch):
        _patch_sources(
            monkeypatch,
            registry_entries=[
                diloco_server_registry.RegistryEntry(
                    id="reg1",
                    label="tunneled",
                    base_url="https://localhost:9999",
                    auth_token="",
                    verify_tls=False,
                ),
            ],
            identity=_identity(),
        )
        s = cluster_diloco_inventory.local_servers()[0]
        assert s.verify_tls is False
        # Loopback URL still in inventory but flagged.
        assert s.loopback is True

    def test_local_jobrecord_wins_collision(self, monkeypatch):
        """When the same URL appears as both a JobRecord and a registry
        entry, the JobRecord row is the one returned — its label /
        source_id are the richer ones."""
        url = "http://node-a:8512"
        _patch_sources(
            monkeypatch,
            records=[_record(host="0.0.0.0", port=8512, auth_token="local-tok")],
            registry_entries=[
                diloco_server_registry.RegistryEntry(
                    id="reg1",
                    label="dup",
                    base_url=url,
                    auth_token="reg-tok",
                ),
            ],
            identity=_identity(hostname="node-a"),
        )
        servers = cluster_diloco_inventory.local_servers()
        assert len(servers) == 1
        assert servers[0].source == "local"
        assert servers[0].auth_token == "local-tok"


class TestPeerEntryValidation:
    """``_local_server_from_dict`` is the ingress point for peer-supplied
    inventory rows. The cluster bearer trust model concedes peer
    compromise can submit code, but the parser still defense-in-depths
    against structurally-hostile entries that would steer the master's
    own outbound calls in surprising ways."""

    def _parse(self, **overrides):
        raw = {
            "server_id": "abc",
            "base_url": "http://node-x:8512",
            "auth_token": "tok",
            "label": "x",
            "source": "local",
            "peer_node_id": "n",
        }
        raw.update(overrides)
        return cluster_diloco_inventory._local_server_from_dict(raw)

    def test_rejects_non_http_scheme(self):
        assert self._parse(base_url="file:///etc/passwd") is None
        assert self._parse(base_url="ftp://x:21") is None

    def test_rejects_missing_host(self):
        assert self._parse(base_url="http:///path") is None

    def test_rejects_embedded_credentials(self):
        """``http://user:pass@host`` would make httpx forward Basic-auth
        credentials of a peer's choosing on every outbound call. Drop
        the entry rather than strip — there is no legitimate use of
        URL-embedded credentials in this inventory."""
        assert self._parse(base_url="http://attacker:secret@host:8512") is None
        assert self._parse(base_url="http://just-user@host:8512") is None

    def test_local_peer_entry_forces_verify_tls_true(self):
        """A peer attesting to a *spawned* server (source=local) must
        not be able to flip verify_tls off on this node. Otherwise one
        peer could downgrade every other peer's outbound TLS for the
        same URL."""
        entry = self._parse(source="local", verify_tls=False)
        assert entry is not None
        assert entry.verify_tls is True

    def test_user_registry_entry_honors_verify_tls_false(self):
        """A peer's *user-registry* entry with verify_tls=False is
        honored: the operator on the owning node explicitly opted out
        (e.g. for an SSH-tunneled remote whose cert won't validate).
        The local operator can still override by registering the URL
        locally with verify_tls=False."""
        entry = self._parse(source="user", verify_tls=False)
        assert entry is not None
        assert entry.verify_tls is False

    def test_accepts_well_formed_entry(self):
        entry = self._parse()
        assert entry is not None
        assert entry.base_url == "http://node-x:8512"
        assert entry.verify_tls is True


class TestMasterInventory:
    def setup_method(self):
        cluster_diloco_inventory._reset_master_state_for_tests()

    def test_merge_carries_health_state(self):
        inv = cluster_diloco_inventory.master_inventory
        inv.set_master_state(True)
        first = cluster_diloco_inventory.MasterServerEntry(
            server_id="sid-1",
            base_url="http://node-a:8512",
            auth_token="t1",
            label="default:8512",
            source="local",
            peer_node_id="node-a-id",
        )
        inv.merge_servers({"sid-1": first})
        inv.update_health("sid-1", healthy=True)
        assert inv.servers_snapshot()[0].healthy is True
        assert inv.servers_snapshot()[0].total_health_polls == 1

        # Fresh entry for the same id (token rotated, label updated) —
        # health state carries forward so the panel doesn't flicker
        # between merges.
        second = cluster_diloco_inventory.MasterServerEntry(
            server_id="sid-1",
            base_url="http://node-a:8512",
            auth_token="t1-rotated",
            label="default:8512",
            source="local",
            peer_node_id="node-a-id",
        )
        inv.merge_servers({"sid-1": second})
        s = inv.servers_snapshot()[0]
        assert s.healthy is True
        assert s.total_health_polls == 1
        # Identity fields update though.
        assert s.auth_token == "t1-rotated"

    def test_token_for_url_lookup(self):
        inv = cluster_diloco_inventory.master_inventory
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "sid-1": cluster_diloco_inventory.MasterServerEntry(
                    server_id="sid-1",
                    base_url="http://node-a:8512",
                    auth_token="secret-1",
                    label="x",
                    source="local",
                    peer_node_id="node-a-id",
                ),
                "sid-2": cluster_diloco_inventory.MasterServerEntry(
                    server_id="sid-2",
                    base_url="http://node-b:8513",
                    auth_token="",
                    label="y",
                    source="local",
                    peer_node_id="node-b-id",
                ),
            }
        )
        # Trailing slash + case tolerated through ``_normalize``.
        assert inv.token_for_url("http://node-a:8512") == "secret-1"
        assert inv.token_for_url("http://node-a:8512/") == "secret-1"
        assert inv.token_for_url("HTTP://NODE-A:8512") == "secret-1"
        # Empty auth_token → ``None`` (caller treats as --no-auth).
        assert inv.token_for_url("http://node-b:8513") is None
        # Unknown URL → ``None``.
        assert inv.token_for_url("http://nowhere:9999") is None

    def test_verify_tls_for_url_lookup(self):
        inv = cluster_diloco_inventory.master_inventory
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "sid-1": cluster_diloco_inventory.MasterServerEntry(
                    server_id="sid-1",
                    base_url="https://ssh-tunnel:8512",
                    auth_token="t",
                    label="tunneled",
                    source="user",
                    peer_node_id=None,
                    verify_tls=False,
                ),
            }
        )
        assert inv.verify_tls_for_url("https://ssh-tunnel:8512") is False
        # Unknown URL: ``None`` so the caller falls back to its own
        # secure-by-default.
        assert inv.verify_tls_for_url("https://other:8512") is None

    def test_role_transition_clears_inventory(self):
        inv = cluster_diloco_inventory.master_inventory
        inv.set_master_state(True)
        inv.merge_servers(
            {
                "sid-1": cluster_diloco_inventory.MasterServerEntry(
                    server_id="sid-1",
                    base_url="http://x:1",
                    auth_token="t",
                    label="x",
                    source="local",
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
