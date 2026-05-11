"""Auto-detect hostnames + LAN IPs for cert SAN coverage."""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Iterable

log = logging.getLogger("forgather.tls")


# Bound the auto-detected SAN list. A host with dozens of virtual
# interfaces (containers, VPNs, bridges) can otherwise produce a cert
# the operator can't audit at a glance. The operator can always add
# more entries via --hostname / --ip — but the *automatic* list stops
# at a reviewable size.
MAX_AUTO_SAN_ENTRIES = 32


def detect_hostnames(*, cap: int = MAX_AUTO_SAN_ENTRIES) -> list[str]:
    """Best-effort: short hostname + FQDN + ``localhost``.

    Deduplicated, lowercased. Falls back gracefully if any lookup fails.
    """
    names: list[str] = ["localhost"]
    try:
        h = socket.gethostname()
        if h:
            names.append(h)
    except Exception:
        pass
    try:
        fq = socket.getfqdn()
        if fq:
            names.append(fq)
    except Exception:
        pass
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        n = n.strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    if len(out) > cap:
        log.warning(
            "auto-detected %d hostnames; capping to %d for SAN inclusion",
            len(out),
            cap,
        )
        out = out[:cap]
    return out


def detect_ips(*, cap: int = MAX_AUTO_SAN_ENTRIES) -> list[str]:
    """Local IPv4/IPv6 addresses, excluding link-local and loopback duplicates.

    Always includes ``127.0.0.1`` and ``::1``. Other addresses come from
    ``psutil.net_if_addrs`` when available; on failure we fall back to
    ``socket.getaddrinfo`` against the local hostname.
    """
    ips: list[str] = ["127.0.0.1", "::1"]
    seen: set[str] = set(ips)

    def _add(raw: str) -> None:
        if not raw:
            return
        # Strip scope id (``fe80::1%eth0``)
        raw = raw.split("%", 1)[0].strip().lower()
        try:
            ipobj = ipaddress.ip_address(raw)
        except ValueError:
            return
        if ipobj.is_link_local:
            return
        s = str(ipobj)
        if s in seen:
            return
        seen.add(s)
        ips.append(s)

    try:
        import psutil  # type: ignore[import-not-found]

        for _name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                _add(getattr(a, "address", ""))
    except Exception:
        try:
            host = socket.gethostname()
            for info in socket.getaddrinfo(host, None):
                _add(info[4][0])
        except Exception:
            log.debug("IP discovery fallback failed", exc_info=True)
    if len(ips) > cap:
        log.warning(
            "auto-detected %d IPs; capping to %d for SAN inclusion",
            len(ips),
            cap,
        )
        ips = ips[:cap]
    return ips


def merge_san(
    base_hostnames: Iterable[str],
    base_ips: Iterable[str],
    extra_hostnames: Iterable[str] = (),
    extra_ips: Iterable[str] = (),
    *,
    hard_cap: int = 256,
) -> tuple[list[str], list[str]]:
    """Union + dedupe hostname/IP lists, preserving first-seen order.

    A ``hard_cap`` defends against accidental or hostile SAN bloat —
    extremely large SAN lists are both legitimately rare and a sign of
    something being wrong with discovery (e.g. a container host with
    hundreds of veth interfaces, or an operator pasting a CSV by
    mistake). ``ValueError`` is raised rather than silently truncating
    because in this case the operator typed the entries.
    """
    hseen: set[str] = set()
    hosts: list[str] = []
    for h in list(base_hostnames) + list(extra_hostnames):
        if not h:
            continue
        key = h.strip().lower()
        if key in hseen:
            continue
        hseen.add(key)
        hosts.append(key)

    iseen: set[str] = set()
    ips: list[str] = []
    for raw in list(base_ips) + list(extra_ips):
        if not raw:
            continue
        try:
            obj = ipaddress.ip_address(str(raw).split("%", 1)[0].strip())
        except ValueError:
            continue
        s = str(obj)
        if s in iseen:
            continue
        iseen.add(s)
        ips.append(s)
    if len(hosts) + len(ips) > hard_cap:
        raise ValueError(
            f"SAN list too large ({len(hosts)} hostnames + {len(ips)} IPs > "
            f"{hard_cap}); refusing to issue a cert this hard to audit. "
            "Trim --hostname/--ip entries or raise the cap."
        )
    return hosts, ips
