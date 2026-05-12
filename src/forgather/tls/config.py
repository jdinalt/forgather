"""Shared TLS config loader.

Layout under ``~/.config/forgather/tls/``::

    config.yaml            single source of truth
    ca/ca.crt              this host's CA cert
    ca/ca.key              0600; only on CA holders
    ca/ca.srl              serial counter
    server.crt             this host's server cert (SAN-rich)
    server.key             0600
    trusted/<name>.crt     CA certs imported from peers
    ca-bundle.crt          generated: ca/ca.crt + trusted/*.crt

The location is overridable via ``$FORGATHER_TLS_DIR``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from forgather.preprocess import forgather_config_dir

log = logging.getLogger("forgather.tls")


CONFIG_FILENAME = "config.yaml"


class TLSConfigError(RuntimeError):
    """Raised when the TLS config is present but malformed/incomplete."""


def tls_dir() -> Path:
    """Root directory for shared TLS state.

    Overridable via ``$FORGATHER_TLS_DIR``; defaults to
    ``<forgather_config_dir>/tls`` (Linux: ``~/.config/forgather/tls``).
    """
    override = os.environ.get("FORGATHER_TLS_DIR")
    if override:
        return Path(os.path.expanduser(override)).resolve()
    return Path(forgather_config_dir()) / "tls"


def _default_paths(root: Path) -> dict:
    return {
        "ca_cert": str(root / "ca" / "ca.crt"),
        "ca_key": str(root / "ca" / "ca.key"),
        "ca_serial": str(root / "ca" / "ca.srl"),
        "server_cert": str(root / "server.crt"),
        "server_key": str(root / "server.key"),
        "ca_bundle": str(root / "ca-bundle.crt"),
        "trusted_dir": str(root / "trusted"),
    }


@dataclass
class TLSConfig:
    """Resolved TLS state for this host.

    Construction always succeeds — missing files just mean
    ``is_provisioned()`` returns False. The servers / CLI use that
    accessor to decide whether to enable TLS.
    """

    root: Path
    enabled: bool = False
    auto_on_non_loopback: bool = True
    ca_cert: Path = field(default_factory=Path)
    ca_key: Path = field(default_factory=Path)
    ca_serial: Path = field(default_factory=Path)
    server_cert: Path = field(default_factory=Path)
    server_key: Path = field(default_factory=Path)
    ca_bundle: Path = field(default_factory=Path)
    trusted_dir: Path = field(default_factory=Path)
    san_hostnames: list[str] = field(default_factory=list)
    san_ips: list[str] = field(default_factory=list)
    validity_days: int = 825
    ca_validity_days: int = 3650
    # Whether peer-pull and proxy clients should validate the peer
    # cert's SAN against the URL hostname. Defaults to False:
    # on a private LAN with a private CA, the security boundary is
    # "do you hold a cert signed by my CA?", not "does your IP match
    # this string." DHCP-issued IPs and ephemeral hostnames make
    # hostname-SAN matching mostly theatre. Flip to True for setups
    # where you want strict RFC-6125 hostname verification (e.g.
    # public-DNS clusters with cert manager auto-renewal).
    verify_hostname: bool = False
    config_path: Optional[Path] = None
    raw: dict = field(default_factory=dict)

    @property
    def config_file(self) -> Path:
        return self.root / CONFIG_FILENAME

    def is_provisioned(self) -> bool:
        """True iff the local server can serve TLS (cert+key present)."""
        return self.server_cert.is_file() and self.server_key.is_file()

    def has_ca_authority(self) -> bool:
        """True iff this host can mint certs (CA cert + private key present)."""
        return self.ca_cert.is_file() and self.ca_key.is_file()

    def effective_bundle(self) -> Optional[Path]:
        """Return the path clients should use as CA bundle, or None."""
        if self.ca_bundle.is_file():
            return self.ca_bundle
        if self.ca_cert.is_file():
            return self.ca_cert
        return None


def load_config(root: Optional[Path] = None) -> TLSConfig:
    """Load (or synthesize defaults for) the shared TLS config.

    Always returns a :class:`TLSConfig`. If ``config.yaml`` is absent,
    returns one with ``enabled=False`` and default file paths populated.
    Callers check ``is_provisioned()`` / ``enabled`` to decide whether
    to act on it.
    """
    root = Path(root) if root else tls_dir()
    paths = _default_paths(root)
    cfg_path = root / CONFIG_FILENAME

    raw: dict = {}
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise TLSConfigError(f"could not read {cfg_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise TLSConfigError(f"{cfg_path} must contain a YAML mapping")

    san = raw.get("san") or {}
    if not isinstance(san, dict):
        raise TLSConfigError("'san' must be a mapping (hostnames, ips)")

    return TLSConfig(
        root=root,
        enabled=bool(raw.get("enabled", False)),
        auto_on_non_loopback=bool(raw.get("auto_on_non_loopback", True)),
        verify_hostname=bool(raw.get("verify_hostname", False)),
        ca_cert=Path(os.path.expanduser(raw.get("ca_cert") or paths["ca_cert"])),
        ca_key=Path(os.path.expanduser(raw.get("ca_key") or paths["ca_key"])),
        ca_serial=Path(
            os.path.expanduser(raw.get("ca_serial") or paths["ca_serial"])
        ),
        server_cert=Path(
            os.path.expanduser(raw.get("server_cert") or paths["server_cert"])
        ),
        server_key=Path(
            os.path.expanduser(raw.get("server_key") or paths["server_key"])
        ),
        ca_bundle=Path(
            os.path.expanduser(raw.get("ca_bundle") or paths["ca_bundle"])
        ),
        trusted_dir=Path(
            os.path.expanduser(raw.get("trusted_dir") or paths["trusted_dir"])
        ),
        san_hostnames=list(san.get("hostnames") or []),
        san_ips=list(san.get("ips") or []),
        validity_days=int(raw.get("validity_days", 825)),
        ca_validity_days=int(raw.get("ca_validity_days", 3650)),
        config_path=cfg_path if cfg_path.is_file() else None,
        raw=raw,
    )


def save_config(cfg: TLSConfig) -> Path:
    """Persist ``cfg`` to ``config.yaml`` and tighten dir perms."""
    cfg.root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cfg.root, 0o700)
    except OSError:
        pass

    data = {
        "enabled": bool(cfg.enabled),
        "auto_on_non_loopback": bool(cfg.auto_on_non_loopback),
        "verify_hostname": bool(cfg.verify_hostname),
        "ca_cert": str(cfg.ca_cert),
        "ca_key": str(cfg.ca_key),
        "ca_serial": str(cfg.ca_serial),
        "server_cert": str(cfg.server_cert),
        "server_key": str(cfg.server_key),
        "ca_bundle": str(cfg.ca_bundle),
        "trusted_dir": str(cfg.trusted_dir),
        "san": {
            "hostnames": list(cfg.san_hostnames),
            "ips": list(cfg.san_ips),
        },
        "validity_days": int(cfg.validity_days),
        "ca_validity_days": int(cfg.ca_validity_days),
    }
    path = cfg.root / CONFIG_FILENAME
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    cfg.config_path = path
    return path
