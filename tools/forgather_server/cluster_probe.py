"""Per-node introspection: package versions and network interfaces.

The Samantha multi-node tutorial spends pages warning that mismatched
``torch`` / ``nccl`` versions across hosts are the most common cause
of a torchrun rendezvous appearing to succeed and then the training
hanging. Surfacing the versions in the Nodes view (and later, gating
multi-node submit on them in Phase 3) eliminates that whole class of
failure.

Network interfaces are the second piece of pre-flight: cross-node
training needs ``NCCL_SOCKET_IFNAME`` set to a routable interface,
and "what interfaces does muthur even have" is otherwise a question
that requires shelling into the box. Listing them here turns it into
a dropdown later.

Probe data is piggybacked onto ``/api/cluster/members`` rather than
served at a separate endpoint: the membership poll already happens
every 5 s and brings the probe data back for free, with no extra
round-trip.
"""

from __future__ import annotations

import logging
import platform
import socket
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Dict, List

log = logging.getLogger("forgather_server.cluster.probe")


def _safe_pkg_version(name: str) -> str:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return "unknown"


def _torch_versions() -> Dict[str, str]:
    """Best-effort torch + nccl + cuda runtime version snapshot.

    ``torch`` may not be importable on a machine that's only running
    the server for queueing-and-monitoring work. We treat that as a
    non-error and return ``"unavailable"`` — the UI displays the
    cell muted rather than as a divergence.
    """
    out: Dict[str, str] = {
        "torch": "unavailable",
        "cuda_runtime": "unavailable",
        "nccl": "unavailable",
    }
    try:
        import torch  # noqa: WPS433 — late import is intentional
    except Exception:
        return out
    try:
        out["torch"] = torch.__version__
    except Exception:
        pass
    try:
        cuda_v = torch.version.cuda  # type: ignore[attr-defined]
        out["cuda_runtime"] = cuda_v if cuda_v is not None else "cpu-only"
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            v = torch.cuda.nccl.version()
            # tuple like (2, 21, 5) -> "2.21.5"
            if isinstance(v, tuple):
                out["nccl"] = ".".join(str(p) for p in v)
            else:
                out["nccl"] = str(v)
    except Exception:
        pass
    return out


def _network_interfaces() -> List[Dict[str, Any]]:
    """List IPv4 interfaces with address + netmask + CIDR.

    Loopback is excluded (no operational value for cluster-internal
    decisions). Virtual interface prefixes are *not* filtered here —
    the operator may want to see ``docker0`` exists in the Nodes view
    even though we don't advertise its address. Filtering is a
    discovery concern, not a probe concern.
    """
    import ipaddress

    import psutil

    out: List[Dict[str, Any]] = []
    try:
        addrs = psutil.net_if_addrs()
    except Exception:
        log.exception("psutil.net_if_addrs failed during probe")
        return out
    try:
        stats = psutil.net_if_stats()
    except Exception:
        stats = {}
    for iface, entries in addrs.items():
        for entry in entries:
            if entry.family != socket.AF_INET:
                continue
            ip = entry.address
            netmask = entry.netmask
            if not ip or ip.startswith("127."):
                continue
            cidr = ""
            if netmask:
                try:
                    net = ipaddress.IPv4Network(
                        f"{ip}/{netmask}", strict=False
                    )
                    cidr = f"{net.network_address}/{net.prefixlen}"
                except (ValueError, ipaddress.AddressValueError):
                    cidr = ""
            stat = stats.get(iface)
            out.append(
                {
                    "name": iface,
                    "address": ip,
                    "netmask": netmask or "",
                    "cidr": cidr,
                    "is_up": bool(getattr(stat, "isup", True))
                    if stat is not None
                    else True,
                    "speed_mbps": int(getattr(stat, "speed", 0))
                    if stat is not None
                    else 0,
                }
            )
    # Stable ordering by interface name so the UI doesn't shuffle
    # rows on every refresh.
    out.sort(key=lambda r: r["name"])
    return out


def _cpu_summary() -> Dict[str, Any]:
    try:
        import psutil

        return {
            "logical": psutil.cpu_count(logical=True) or 0,
            "physical": psutil.cpu_count(logical=False) or 0,
            # Total RAM in GiB, rounded for display. Don't try to be
            # precise here — the Nodes view shows it as context, not
            # for accounting.
            "ram_gib": round(psutil.virtual_memory().total / (1024**3), 1),
        }
    except Exception:
        return {"logical": 0, "physical": 0, "ram_gib": 0.0}


# Versions are computed once at module import. They cannot change at
# runtime without a restart — re-probing every 5 s would waste cycles
# and risk import-time side effects firing more than once.
_version_cache: Dict[str, str] = {
    "forgather": _safe_pkg_version("forgather"),
    "transformers": _safe_pkg_version("transformers"),
    "python": platform.python_version(),
    "platform": platform.platform(),
    **_torch_versions(),
}
_cpu_cache: Dict[str, Any] = _cpu_summary()


def local_probe() -> Dict[str, Any]:
    """Snapshot of this node's pre-flight data.

    Versions and CPU info are cached at import time. Network
    interfaces are re-read on every call — they're cheap, change
    occasionally (NIC up/down, DHCP renumbering), and being live is
    more useful than being static.
    """
    return {
        "versions": dict(_version_cache),
        "interfaces": _network_interfaces(),
        "cpu": dict(_cpu_cache),
    }
