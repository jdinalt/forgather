"""Tests for tools/forgather_server/cluster_discovery.py.

End-to-end mDNS-on-the-LAN behavior is exercised in the loopback
two-server smoke test. Here we verify the unit-testable bits: TXT
record construction, peer filtering, and graceful handling of
malformed advertisements.
"""

import socket
import uuid
from unittest.mock import MagicMock

import forgather_server.cluster as cluster
import forgather_server.cluster_discovery as discovery
import pytest
from forgather_server import paths


@pytest.fixture(autouse=True)
def isolated_cluster(tmp_path, monkeypatch):
    cluster_dir = tmp_path / "cluster"
    cluster_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_state_dir", lambda: cluster_dir)
    monkeypatch.setattr(
        paths, "cluster_node_id_file", lambda: cluster_dir / "node_id"
    )
    cluster._reset_for_tests()
    yield cluster_dir
    cluster._reset_for_tests()


class TestBuildServiceInfo:
    def test_txt_records_round_trip(self):
        info = discovery._build_service_info(
            cluster_name="alpha",
            node_id="11111111-2222-3333-4444-555555555555",
            hostname="muthur",
            port=8765,
            version="1.1.0",
            addresses=[socket.inet_aton("10.0.0.5")],
        )
        props = info.properties
        assert props[discovery._TXT_CLUSTER] == b"alpha"
        assert (
            props[discovery._TXT_NODE_ID]
            == b"11111111-2222-3333-4444-555555555555"
        )
        assert props[discovery._TXT_VERSION] == b"1.1.0"
        assert props[discovery._TXT_HOSTNAME] == b"muthur"
        assert info.port == 8765
        # The instance name embeds the node_id so two servers on one
        # host don't collide on the mDNS bus.
        assert info.name.startswith("11111111-2222-3333-4444-555555555555.")
        assert info.name.endswith(discovery.SERVICE_TYPE)


class TestDecode:
    def test_decodes_bytes(self):
        assert discovery._decode(b"hello") == "hello"

    def test_returns_empty_for_none(self):
        assert discovery._decode(None) == ""

    def test_returns_empty_for_invalid_utf8(self):
        assert discovery._decode(b"\xff\xfe") == ""


def _activate_self() -> str:
    ident = cluster.activate("alpha", port=8765)
    return ident.node_id


def _make_service_info(
    *,
    node_id: str,
    cluster_name: str = "alpha",
    address: str = "10.0.0.7",
    port: int = 8765,
    version: str = "1.1.0",
    hostname: str = "peer1",
) -> MagicMock:
    """Build a stub ServiceInfo for the listener under test."""
    info = MagicMock()
    info.properties = {
        discovery._TXT_CLUSTER: cluster_name.encode("utf-8"),
        discovery._TXT_NODE_ID: node_id.encode("ascii"),
        discovery._TXT_VERSION: version.encode("utf-8"),
        discovery._TXT_HOSTNAME: hostname.encode("utf-8"),
    }
    info.port = port
    info.server = f"{node_id}.local."
    info.parsed_addresses = MagicMock(return_value=[address])
    return info


class TestPeerListener:
    def _listener_for(self, peer_info, self_node_id):
        zc = MagicMock()
        zc.get_service_info = MagicMock(return_value=peer_info)
        return (
            discovery._PeerListener(
                zc, self_node_id=self_node_id, self_cluster="alpha"
            ),
            zc,
        )

    def test_adds_matching_peer(self):
        self_id = _activate_self()
        peer_id = str(uuid.uuid4())
        info = _make_service_info(node_id=peer_id, address="10.0.0.7")
        listener, _zc = self._listener_for(info, self_id)
        listener.add_service(_zc, discovery.SERVICE_TYPE, "peer.foo._tcp.local.")
        peer_ids = {m.node_id for m in cluster.members()}
        assert peer_id in peer_ids
        peer = next(m for m in cluster.members() if m.node_id == peer_id)
        assert peer.address == "10.0.0.7"
        assert peer.last_source == "discovery"

    def test_ignores_own_advertisement(self):
        self_id = _activate_self()
        info = _make_service_info(node_id=self_id, address="10.0.0.5")
        listener, _zc = self._listener_for(info, self_id)
        listener.add_service(_zc, discovery.SERVICE_TYPE, "self.foo._tcp.local.")
        # Only self in members; no second entry created.
        assert len(cluster.members()) == 1

    def test_ignores_foreign_cluster(self):
        self_id = _activate_self()
        peer_id = str(uuid.uuid4())
        info = _make_service_info(
            node_id=peer_id, cluster_name="someone_else_cluster"
        )
        listener, _zc = self._listener_for(info, self_id)
        listener.add_service(_zc, discovery.SERVICE_TYPE, "peer._tcp.local.")
        assert len(cluster.members()) == 1  # self only

    def test_ignores_service_without_node_id(self):
        self_id = _activate_self()
        info = MagicMock()
        info.properties = {
            discovery._TXT_CLUSTER: b"alpha",
            discovery._TXT_VERSION: b"1.1.0",
        }
        info.port = 8765
        info.server = "foo.local."
        info.parsed_addresses = MagicMock(return_value=["10.0.0.7"])
        listener, _zc = self._listener_for(info, self_id)
        listener.add_service(_zc, discovery.SERVICE_TYPE, "x._tcp.local.")
        assert len(cluster.members()) == 1

    def test_skips_when_no_addresses(self):
        self_id = _activate_self()
        peer_id = str(uuid.uuid4())
        info = _make_service_info(node_id=peer_id)
        info.parsed_addresses = MagicMock(return_value=[])
        listener, _zc = self._listener_for(info, self_id)
        listener.add_service(_zc, discovery.SERVICE_TYPE, "x._tcp.local.")
        assert len(cluster.members()) == 1

    def test_loopback_address_used_when_only_loopback(self):
        # Loopback test scenario: two servers on one host, both
        # advertising 127.0.0.1. Selection must not strip these or
        # discovery breaks.
        self_id = _activate_self()
        peer_id = str(uuid.uuid4())
        info = _make_service_info(node_id=peer_id, address="127.0.0.1")
        listener, _zc = self._listener_for(info, self_id)
        listener.add_service(_zc, discovery.SERVICE_TYPE, "x._tcp.local.")
        peer = next(m for m in cluster.members() if m.node_id == peer_id)
        assert peer.address == "127.0.0.1"

    def test_remove_event_does_not_drop_member(self):
        self_id = _activate_self()
        peer_id = str(uuid.uuid4())
        info = _make_service_info(node_id=peer_id)
        listener, _zc = self._listener_for(info, self_id)
        listener.add_service(_zc, discovery.SERVICE_TYPE, "x._tcp.local.")
        assert any(m.node_id == peer_id for m in cluster.members())
        listener.remove_service(_zc, discovery.SERVICE_TYPE, "x._tcp.local.")
        # Membership entry retained (union-of-ever-seen view).
        assert any(m.node_id == peer_id for m in cluster.members())


class TestInterfaceAddresses:
    def test_excludes_loopback(self, monkeypatch):
        monkeypatch.setattr(
            discovery.socket,
            "getaddrinfo",
            lambda *a, **kw: [
                (None, None, None, "", ("10.0.0.5", 0)),
                (None, None, None, "", ("127.0.0.1", 0)),
            ],
        )
        addrs = discovery._interface_addresses()
        assert socket.inet_aton("10.0.0.5") in addrs
        assert socket.inet_aton("127.0.0.1") not in addrs

    def test_handles_gaierror(self, monkeypatch):
        def boom(*a, **kw):
            raise socket.gaierror("nope")

        monkeypatch.setattr(discovery.socket, "getaddrinfo", boom)
        assert discovery._interface_addresses() == []
