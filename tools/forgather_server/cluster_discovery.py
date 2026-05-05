"""mDNS / Zeroconf discovery for cluster peers.

Advertises ``_forgather._tcp`` on the LAN with TXT records identifying
the cluster name, the persistent node UUID, the HTTP port, and the
forgather version. Browses the same service type and feeds peer
events into ``cluster.update_member``.

Cluster scoping: the TXT record carries the cluster name. Peers whose
``cluster=`` does not match this node's cluster are ignored. This
prevents two unrelated clusters on the same subnet from auto-merging
just because one developer happens to share a LAN with another.

Why mDNS rather than UDP broadcast: blocked less often by host
firewalls, has cross-platform tooling (``avahi-browse``, ``dns-sd``)
for diagnosis, and ``python-zeroconf`` is well-maintained.

The discovery layer is *only* for finding peers and refreshing their
last-seen address. Liveness — whether a peer is currently answering
HTTP — is the membership task's job (cluster_membership.py).
"""

from __future__ import annotations

import logging
import socket
from typing import List, Optional

from zeroconf import (
    InterfaceChoice,
    IPVersion,
    ServiceBrowser,
    ServiceInfo,
    ServiceListener,
    ServiceStateChange,
    Zeroconf,
)

from . import cluster

log = logging.getLogger("forgather_server.cluster.discovery")


SERVICE_TYPE = "_forgather._tcp.local."

# Zeroconf TXT record keys are bytes. Centralize them so the advertise
# and parse paths can't drift.
_TXT_CLUSTER = b"cluster"
_TXT_NODE_ID = b"node_id"
_TXT_VERSION = b"version"
_TXT_HOSTNAME = b"hostname"


def _build_service_info(
    *,
    cluster_name: str,
    node_id: str,
    hostname: str,
    port: int,
    version: str,
    addresses: List[bytes],
) -> ServiceInfo:
    """Construct the ServiceInfo we will register on the local mDNS bus.

    The instance name embeds the node UUID so two servers on the same
    host (e.g. loopback test) don't collide on the same record. mDNS
    instance names live in a single namespace per service type, so
    using node_id makes collisions effectively impossible.
    """
    instance_name = f"{node_id}.{SERVICE_TYPE}"
    properties = {
        _TXT_CLUSTER: cluster_name.encode("utf-8"),
        _TXT_NODE_ID: node_id.encode("ascii"),
        _TXT_VERSION: version.encode("utf-8"),
        _TXT_HOSTNAME: hostname.encode("utf-8"),
    }
    return ServiceInfo(
        type_=SERVICE_TYPE,
        name=instance_name,
        addresses=addresses,
        port=port,
        properties=properties,
        # ``server`` is the DNS host name embedded in the SRV record. We
        # use ``<node_id>.local.`` rather than the OS hostname so that
        # mDNS responders on the LAN don't second-guess us when two
        # nodes happen to share a hostname (common in container hosts).
        server=f"{node_id}.local.",
    )


# Interface name prefixes that we treat as "virtual" — Docker bridges,
# Kubernetes overlays, VPN tunnels, etc. These rarely carry traffic
# between physical hosts on the same LAN, and advertising them as
# cluster-reachable addresses confuses the peer-pull (the receiving
# node will happily try to dial 172.17.0.1 and either fail outright
# or — worse — connect to its own equivalently-numbered bridge).
_VIRTUAL_IFACE_PREFIXES: tuple = (
    "docker",
    "br-",
    "veth",
    "virbr",
    "cni",
    "kube",
    "flannel",
    "tun",
    "tap",
    "cali",
    "weave",
    "wg",  # WireGuard
    "tailscale",
    "zt",  # ZeroTier
)


def _is_virtual_iface(name: str) -> bool:
    return name.lower().startswith(_VIRTUAL_IFACE_PREFIXES)


def _interface_addresses() -> List[bytes]:
    """Enumerate routable local IPv4 addresses to advertise.

    ``socket.gethostname() + getaddrinfo`` is unreliable on Linux: most
    distros ship an ``/etc/hosts`` entry like ``127.0.1.1 <hostname>``
    that makes the call return loopback even when the host has real
    LAN interfaces. We hit that exact bug in the field — both nodes
    advertised 127.0.0.1, the master tried to peer-pull from
    127.0.0.1:<peer_port> and ended up calling itself.
    ``psutil.net_if_addrs()`` enumerates the kernel's interface list
    directly and avoids the resolver path entirely.

    Virtual interfaces (Docker bridges, k8s overlays, VPN tunnels) are
    excluded from the primary list — they share addresses across hosts
    and almost never carry inter-host traffic on a typical LAN. Two
    different hosts both running Docker will both have a 172.17.0.1
    bridge, and a peer trying to dial that address ends up talking to
    its *own* bridge, not the peer's. Virtual interfaces remain a
    final fallback so a host with literally nothing else still has
    *something* to advertise.

    Loopback is excluded except as a final-final fallback (so the
    loopback two-server smoke test still works). Link-local 169.254/16
    is skipped because it is rarely the address you want a peer to
    dial.

    Returns a list of packed 4-byte addresses suitable for ServiceInfo.
    """
    import psutil

    seen: set = set()
    real: List[bytes] = []
    virtual: List[bytes] = []
    try:
        for iface, entries in psutil.net_if_addrs().items():
            virtual_iface = _is_virtual_iface(iface)
            for entry in entries:
                if entry.family != socket.AF_INET:
                    continue
                ip = entry.address
                if not ip:
                    continue
                if ip.startswith("127."):
                    continue
                if ip.startswith("169.254."):
                    continue
                if ip in seen:
                    continue
                seen.add(ip)
                try:
                    packed = socket.inet_aton(ip)
                except OSError:
                    continue
                if virtual_iface:
                    virtual.append(packed)
                else:
                    real.append(packed)
    except Exception:
        log.exception("psutil.net_if_addrs failed; falling back to loopback")
    chosen = real if real else virtual
    if log.isEnabledFor(10):  # DEBUG
        log.debug(
            "advertising %d address(es): %s%s",
            len(chosen),
            [socket.inet_ntoa(a) for a in chosen],
            " (virtual fallback)" if not real and virtual else "",
        )
    return chosen


class _PeerListener(ServiceListener):
    """ServiceBrowser callback that funnels updates into cluster.py.

    The Listener interface from python-zeroconf is sync; we keep the
    work small (a single dict update guarded by cluster.py's lock) so
    no event-loop hop is needed.
    """

    def __init__(self, zc: Zeroconf, self_node_id: str, self_cluster: str):
        self._zc = zc
        self._self_node_id = self_node_id
        self._self_cluster = self_cluster

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self._handle(zc, type_, name, ServiceStateChange.Added)

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self._handle(zc, type_, name, ServiceStateChange.Updated)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        # We deliberately do *not* drop members on remove. The user-
        # agreed design keeps the union view; the membership task
        # decides reachability. A removed mDNS record may just mean a
        # service browser refresh, not an actual node going away.
        log.debug("mDNS remove event for %s (membership keeps record)", name)

    def _handle(
        self,
        zc: Zeroconf,
        type_: str,
        name: str,
        state: ServiceStateChange,
    ) -> None:
        try:
            info = zc.get_service_info(type_, name, timeout=2000)
        except Exception as e:
            log.debug("get_service_info failed for %s: %s", name, e)
            return
        if info is None:
            log.debug("no info for %s", name)
            return
        props = info.properties or {}
        node_id = _decode(props.get(_TXT_NODE_ID))
        peer_cluster = _decode(props.get(_TXT_CLUSTER))
        version = _decode(props.get(_TXT_VERSION)) or "unknown"
        hostname = _decode(props.get(_TXT_HOSTNAME)) or info.server.rstrip(".")
        if not node_id:
            log.debug("ignoring service without node_id TXT: %s", name)
            return
        if node_id == self._self_node_id:
            # Our own advertisement comes back to us; ignore.
            return
        if peer_cluster != self._self_cluster:
            log.debug(
                "ignoring %s: cluster=%s != ours=%s",
                name,
                peer_cluster,
                self._self_cluster,
            )
            return
        addresses = info.parsed_addresses(IPVersion.V4Only)
        if not addresses:
            log.debug("no IPv4 addresses for %s", name)
            return
        # Score each address and pick the best one. Lower is better.
        # The sender already prefers real interfaces over virtual
        # ones, but a peer running an older revision (or a peer with
        # only Docker bridges available) may still hand us a list
        # with 172.17.x.x in it; deprioritize those at the receiver
        # so we never store an unreachable address as the canonical
        # one for a member.
        address = sorted(addresses, key=_address_score)[0]
        # Diagnostic: which of our local interfaces is on the same
        # subnet as the chosen peer address. mDNS doesn't tell us
        # which interface the advertisement came in on, but this
        # answers the practical version of the question: "which of
        # our NICs would carry traffic to this peer." Logged on add
        # and on address-changing updates so the operator can see
        # whether peers are showing up on the interface they expect.
        iface_hint = _local_interface_for(address)
        all_advertised = [str(a) for a in addresses]
        log.info(
            "mDNS peer %s (%s) at %s:%d via local iface %s "
            "(advertised: %s)",
            hostname,
            node_id[:8],
            address,
            info.port or 0,
            iface_hint or "?",
            all_advertised,
        )
        try:
            cluster.update_member(
                node_id,
                hostname=hostname,
                address=address,
                port=info.port or 0,
                cluster_name=peer_cluster,
                forgather_version=version,
                source="discovery",
            )
        except Exception:
            log.exception("update_member failed for peer %s", node_id)


def _local_interface_for(ip: str) -> Optional[str]:
    """Best-effort: which of *our* interfaces is on the same subnet
    as ``ip``. Used purely for diagnostic logging — answers the
    question "which interface did this peer come in on?" by
    inferring the interface a connection to ``ip`` would use.

    mDNS itself does not carry a "received on" interface tag in the
    payload we see, so this is the most practical answer. Returns
    None if we can't match (peer is on a different subnet entirely,
    or psutil enumeration failed).
    """
    try:
        import ipaddress

        import psutil

        target = ipaddress.IPv4Address(ip)
        for iface, entries in psutil.net_if_addrs().items():
            for entry in entries:
                if entry.family != socket.AF_INET:
                    continue
                if not entry.address or not entry.netmask:
                    continue
                try:
                    net = ipaddress.IPv4Network(
                        f"{entry.address}/{entry.netmask}", strict=False
                    )
                except (ValueError, ipaddress.AddressValueError):
                    continue
                if target in net:
                    return iface
    except Exception:
        pass
    return None


def _address_score(ip: str) -> int:
    """Lower is better when picking among a peer's advertised addresses.

    The sender filters virtual interfaces, but receivers can't trust
    that the sender ran with this fix. Score Docker / link-local /
    loopback addresses so that even if they're in the advertised list
    we prefer a real LAN address.
    """
    if not ip:
        return 1000
    if ip.startswith("127."):
        return 200
    if ip.startswith("169.254."):
        return 150
    parts = ip.split(".")
    try:
        # Common Docker / overlay subnets (172.16-31.x.x).
        if parts[0] == "172" and 16 <= int(parts[1]) <= 31:
            return 100
    except (IndexError, ValueError):
        pass
    return 0


def _decode(b: Optional[bytes]) -> str:
    if b is None:
        return ""
    if isinstance(b, str):
        return b
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return ""


class ClusterDiscovery:
    """Lifecycle wrapper: register one ServiceInfo + one ServiceBrowser."""

    def __init__(
        self,
        *,
        interfaces: Optional[List[str]] = None,
        ip_version: IPVersion = IPVersion.V4Only,
    ):
        self._interfaces = interfaces
        self._ip_version = ip_version
        self._zc: Optional[Zeroconf] = None
        self._browser: Optional[ServiceBrowser] = None
        self._info: Optional[ServiceInfo] = None

    def start(self, *, addresses: Optional[List[bytes]] = None) -> None:
        ident = cluster.self_identity()
        if ident is None:
            raise RuntimeError(
                "cluster_discovery.start called before cluster.activate"
            )
        if self._zc is not None:
            raise RuntimeError("ClusterDiscovery already started")
        # Use explicit interfaces when provided (for tests / multi-NIC
        # boxes); otherwise let zeroconf bind to all interfaces.
        if self._interfaces:
            zc = Zeroconf(
                interfaces=self._interfaces, ip_version=self._ip_version
            )
        else:
            zc = Zeroconf(
                interfaces=InterfaceChoice.All, ip_version=self._ip_version
            )
        self._zc = zc
        # Resolution order:
        #   1. Operator-provided list via ``activate(advertise_addresses=...)``
        #      — wins outright. Use when auto-detection can't see the
        #      host's real interfaces (typical inside a container that
        #      isn't running with ``--network host``).
        #   2. ``addresses=`` keyword (test injection point).
        #   3. ``_interface_addresses()`` auto-detect.
        #   4. Loopback fallback — emits a WARNING because anything
        #      beyond a single-host smoke test is going to break in
        #      this state.
        if ident.advertise_addresses:
            addrs = []
            for ip in ident.advertise_addresses:
                try:
                    addrs.append(socket.inet_aton(ip))
                except OSError:
                    log.error("ignoring invalid --cluster-address %r", ip)
            log.info(
                "advertising operator-supplied addresses: %s",
                list(ident.advertise_addresses),
            )
        elif addresses is not None:
            addrs = addresses
        else:
            addrs = _interface_addresses()
            if addrs:
                log.info(
                    "advertising auto-detected addresses: %s",
                    [socket.inet_ntoa(a) for a in addrs],
                )
        if not addrs:
            log.warning(
                "no routable interfaces found; advertising 127.0.0.1 — "
                "peers on other hosts will NOT be able to reach this "
                "node. Use --cluster-address <ip> to override "
                "auto-detection (typical inside containers without "
                "--network host)."
            )
            addrs = [socket.inet_aton("127.0.0.1")]
        info = _build_service_info(
            cluster_name=ident.cluster_name,
            node_id=ident.node_id,
            hostname=ident.hostname,
            port=ident.port,
            version=ident.forgather_version,
            addresses=addrs,
        )
        zc.register_service(info, allow_name_change=False)
        self._info = info
        listener = _PeerListener(
            zc, self_node_id=ident.node_id, self_cluster=ident.cluster_name
        )
        self._browser = ServiceBrowser(zc, SERVICE_TYPE, listener)
        log.info(
            "mDNS discovery started: type=%s instance=%s",
            SERVICE_TYPE,
            info.name,
        )

    def stop(self) -> None:
        if self._zc is None:
            return
        try:
            if self._info is not None:
                self._zc.unregister_service(self._info)
        except Exception:
            log.exception("error unregistering mDNS service")
        try:
            self._zc.close()
        except Exception:
            log.exception("error closing zeroconf")
        self._zc = None
        self._browser = None
        self._info = None
        log.info("mDNS discovery stopped")
