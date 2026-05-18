"""Tests for tools/forgather_server/cluster_probe.py."""

import socket

import forgather_server.cluster_probe as probe


class TestLocalProbe:
    def test_shape(self):
        # The probe is real (no mocking); we just assert the shape so
        # the caller code in routes/cluster.py and the webui can rely
        # on these keys existing.
        p = probe.local_probe()
        assert "versions" in p
        assert "interfaces" in p
        assert "cpu" in p
        for key in ("forgather", "python", "platform", "torch", "nccl"):
            assert key in p["versions"]
        for key in ("logical", "physical", "ram_gib"):
            assert key in p["cpu"]

    def test_python_version_well_formed(self):
        p = probe.local_probe()
        # platform.python_version() is "3.X.Y" — refuse to ship if
        # something is silently returning empty strings.
        v = p["versions"]["python"]
        assert v
        parts = v.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts[:2])


class TestNetworkInterfaces:
    @staticmethod
    def _entry(family, address, netmask):
        from collections import namedtuple

        E = namedtuple(
            "snicaddr",
            ["family", "address", "netmask", "broadcast", "ptp"],
        )
        return E(family, address, netmask, None, None)

    def test_excludes_loopback(self, monkeypatch):
        import psutil

        monkeypatch.setattr(
            psutil,
            "net_if_addrs",
            lambda: {
                "lo": [self._entry(socket.AF_INET, "127.0.0.1", "255.0.0.0")],
                "eth0": [
                    self._entry(socket.AF_INET, "10.0.0.5", "255.255.255.0")
                ],
            },
        )
        monkeypatch.setattr(psutil, "net_if_stats", lambda: {})
        ifaces = probe._network_interfaces()
        names = [i["name"] for i in ifaces]
        assert "lo" not in names
        assert "eth0" in names

    def test_cidr_derived(self, monkeypatch):
        import psutil

        monkeypatch.setattr(
            psutil,
            "net_if_addrs",
            lambda: {
                "eth0": [
                    self._entry(socket.AF_INET, "192.168.1.27", "255.255.255.0")
                ],
            },
        )
        monkeypatch.setattr(psutil, "net_if_stats", lambda: {})
        ifaces = probe._network_interfaces()
        assert len(ifaces) == 1
        assert ifaces[0]["cidr"] == "192.168.1.0/24"

    def test_skips_ipv6_entries(self, monkeypatch):
        # The cluster discovery + membership layer is IPv4-only in
        # Phase 2. IPv6 entries from psutil should be ignored rather
        # than producing rows with no usable address. Use AF_INET6
        # from the socket module so the test mirrors real psutil
        # output.
        import psutil

        monkeypatch.setattr(
            psutil,
            "net_if_addrs",
            lambda: {
                "eth0": [
                    self._entry(socket.AF_INET6, "fe80::1", "ffff::"),
                    self._entry(
                        socket.AF_INET, "10.0.0.5", "255.255.255.0"
                    ),
                ],
            },
        )
        monkeypatch.setattr(psutil, "net_if_stats", lambda: {})
        ifaces = probe._network_interfaces()
        assert len(ifaces) == 1
        assert ifaces[0]["address"] == "10.0.0.5"

    def test_handles_psutil_failure(self, monkeypatch):
        import psutil

        def boom():
            raise OSError("nope")

        monkeypatch.setattr(psutil, "net_if_addrs", boom)
        # Caller treats missing interfaces as "unknown" rather than
        # crashing the probe.
        assert probe._network_interfaces() == []

    def test_stable_ordering(self, monkeypatch):
        import psutil

        # Insertion order varies across psutil versions; the UI sorts
        # by name to keep rows from shuffling between polls.
        monkeypatch.setattr(
            psutil,
            "net_if_addrs",
            lambda: {
                "eth1": [
                    self._entry(socket.AF_INET, "10.0.0.7", "255.255.255.0")
                ],
                "eth0": [
                    self._entry(socket.AF_INET, "10.0.0.5", "255.255.255.0")
                ],
            },
        )
        monkeypatch.setattr(psutil, "net_if_stats", lambda: {})
        ifaces = probe._network_interfaces()
        assert [i["name"] for i in ifaces] == ["eth0", "eth1"]
