"""Tests for tools/forgather_server/cluster.py."""

import uuid

import forgather_server.cluster as cluster
import pytest
from forgather_server import paths


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point cluster persistence at a tmp dir and reset module state.

    Without this, every test would share the real ``~/.forgather/cluster/``
    and clobber the developer's actual node identity.
    """
    cluster_dir = tmp_path / "cluster"
    cluster_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_state_dir", lambda: cluster_dir)
    monkeypatch.setattr(
        paths, "cluster_node_id_file", lambda: cluster_dir / "node_id"
    )
    cluster._reset_for_tests()
    yield cluster_dir
    cluster._reset_for_tests()


class TestNodeIdentity:
    def test_activate_persists_and_returns_identity(self, isolated_state):
        ident = cluster.activate("test_cluster", port=8765)
        assert ident.cluster_name == "test_cluster"
        assert ident.port == 8765
        # UUID is well-formed
        uuid.UUID(ident.node_id)
        # Persistence file exists with mode 0600
        node_id_file = isolated_state / "node_id"
        assert node_id_file.exists()
        assert node_id_file.read_text().strip() == ident.node_id
        assert node_id_file.stat().st_mode & 0o777 == 0o600

    def test_node_id_stable_across_activations(self, isolated_state):
        ident1 = cluster.activate("c", port=1)
        cluster._reset_for_tests()
        ident2 = cluster.activate("c", port=1)
        assert ident1.node_id == ident2.node_id

    def test_corrupted_node_id_raises(self, isolated_state):
        (isolated_state / "node_id").write_text("not-a-uuid\n")
        with pytest.raises(ValueError):
            cluster.activate("c", port=1)

    def test_inactive_by_default(self):
        assert cluster.is_active() is False
        assert cluster.self_identity() is None
        assert cluster.master_node_id() is None

    def test_empty_cluster_name_rejected(self):
        with pytest.raises(ValueError):
            cluster.activate("", port=1)

    def test_advertise_addresses_persisted_on_identity(self):
        ident = cluster.activate(
            "c", port=8765, advertise_addresses=("10.0.0.5", "192.168.1.27")
        )
        assert ident.advertise_addresses == ("10.0.0.5", "192.168.1.27")

    def test_advertise_addresses_default_empty(self):
        ident = cluster.activate("c", port=8765)
        assert ident.advertise_addresses == ()


class TestMemberTable:
    def _act(self):
        return cluster.activate("c", port=8765)

    def test_self_present_in_members(self):
        ident = self._act()
        ms = cluster.members()
        assert len(ms) == 1
        assert ms[0].node_id == ident.node_id
        assert ms[0].last_source == "self"

    def test_update_member_inserts_new(self):
        self._act()
        peer_id = str(uuid.uuid4())
        m = cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
            forgather_version="1.1.0",
        )
        assert m.node_id == peer_id
        assert m.reachable is True
        assert {x.node_id for x in cluster.members()} == {
            cluster.self_identity().node_id,
            peer_id,
        }

    def test_update_member_refreshes_existing(self):
        self._act()
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
            now=100.0,
        )
        cluster.update_member(
            peer_id,
            hostname="peer1-renamed",
            address="10.0.0.99",
            port=9000,
            cluster_name="c",
            now=200.0,
        )
        m = next(x for x in cluster.members() if x.node_id == peer_id)
        assert m.hostname == "peer1-renamed"
        assert m.address == "10.0.0.99"
        assert m.port == 9000
        assert m.last_seen == 200.0

    def test_update_member_rejects_other_cluster(self):
        self._act()
        with pytest.raises(ValueError):
            cluster.update_member(
                str(uuid.uuid4()),
                hostname="alien",
                address="10.0.0.3",
                port=8765,
                cluster_name="some_other_cluster",
            )

    def test_mark_unreachable_keeps_entry(self):
        self._act()
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
        )
        cluster.mark_unreachable(peer_id)
        ms = cluster.members()
        assert len(ms) == 2  # entry retained
        peer = next(x for x in ms if x.node_id == peer_id)
        assert peer.reachable is False

    def test_mark_self_unreachable_is_noop(self):
        ident = self._act()
        cluster.mark_unreachable(ident.node_id)
        self_m = next(
            x for x in cluster.members() if x.node_id == ident.node_id
        )
        assert self_m.reachable is True

    def test_self_member_has_probe_attached(self):
        ident = self._act()
        self_m = next(
            x for x in cluster.members() if x.node_id == ident.node_id
        )
        # local_probe() is real (no mocking); it never returns None
        # in practice. Just verify the wiring carried something
        # through.
        assert self_m.probe is not None
        assert "versions" in self_m.probe
        assert "interfaces" in self_m.probe

    def test_update_member_attaches_probe(self):
        self._act()
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
            probe={"versions": {"torch": "2.10.0"}},
        )
        peer = next(x for x in cluster.members() if x.node_id == peer_id)
        assert peer.probe == {"versions": {"torch": "2.10.0"}}

    def test_update_member_without_probe_preserves_existing(self):
        # mDNS discovery doesn't carry probe data; a fresh discovery
        # hit must not wipe the probe we got from the last peer-pull.
        self._act()
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
            probe={"versions": {"torch": "2.10.0"}},
            source="peer_pull",
        )
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
            source="discovery",  # no probe argument
        )
        peer = next(x for x in cluster.members() if x.node_id == peer_id)
        assert peer.probe == {"versions": {"torch": "2.10.0"}}

    def test_update_brings_unreachable_back(self):
        self._act()
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
        )
        cluster.mark_unreachable(peer_id)
        cluster.update_member(
            peer_id,
            hostname="peer1",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
        )
        peer = next(x for x in cluster.members() if x.node_id == peer_id)
        assert peer.reachable is True


class TestSweepUnreachable:
    def test_sweep_marks_silent_members(self):
        cluster.activate("c", port=8765)
        peer_id = str(uuid.uuid4())
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
            now=0.0,
        )
        cluster._set_unreachable_after_for_tests(10.0)
        # Within window: still reachable.
        transitioned = cluster.sweep_unreachable(now=5.0)
        assert transitioned == []
        # Past window: transitions.
        transitioned = cluster.sweep_unreachable(now=20.0)
        assert transitioned == [peer_id]
        # Idempotent on the second sweep.
        transitioned = cluster.sweep_unreachable(now=21.0)
        assert transitioned == []

    def test_sweep_refreshes_self(self):
        ident = cluster.activate("c", port=8765)
        cluster.sweep_unreachable(now=999.0)
        self_m = next(
            x for x in cluster.members() if x.node_id == ident.node_id
        )
        assert self_m.last_seen == 999.0
        assert self_m.reachable is True


class TestMasterSelection:
    def test_master_is_lowest_uuid_among_reachable(self):
        ident = cluster.activate("c", port=8765)
        # Add a peer guaranteed to sort below self by node_id (we don't
        # know self's UUID a priori, so add candidates on both sides
        # and assert the global min).
        candidates = [str(uuid.uuid4()) for _ in range(5)]
        for cid in candidates:
            cluster.update_member(
                cid,
                hostname=f"peer-{cid[:4]}",
                address="10.0.0.2",
                port=8765,
                cluster_name="c",
            )
        all_ids = candidates + [ident.node_id]
        assert cluster.master_node_id() == min(all_ids)

    def test_unreachable_excluded_from_master(self):
        ident = cluster.activate("c", port=8765)
        # Force a peer with a UUID smaller than self by retrying.
        for _ in range(50):
            peer_id = str(uuid.uuid4())
            if peer_id < ident.node_id:
                break
        else:
            pytest.skip("could not generate a smaller UUID after 50 tries")
        cluster.update_member(
            peer_id,
            hostname="peer",
            address="10.0.0.2",
            port=8765,
            cluster_name="c",
        )
        assert cluster.master_node_id() == peer_id
        cluster.mark_unreachable(peer_id)
        assert cluster.master_node_id() == ident.node_id

    def test_is_self_master(self):
        cluster.activate("c", port=8765)
        # With only self, self is master.
        assert cluster.is_self_master() is True

    def test_no_master_when_inactive(self):
        assert cluster.master_node_id() is None
        assert cluster.is_self_master() is False
