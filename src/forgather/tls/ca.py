"""CA creation, cert minting, renewal, trust-bundle management.

All certificate operations use :mod:`cryptography.x509` directly — no
shelling out to ``openssl``. Keys are RSA-2048; certs are SHA-256 signed.
That matches what self-hosted CA tooling like ``mkcert`` produces and
keeps the implementation small.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtensionOID, NameOID

from .config import TLSConfig

log = logging.getLogger("forgather.tls")


# --------------------------------------------------------------------- helpers


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _gen_key(bits: int = 2048) -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=bits)


def _write_secret(path: Path, data: bytes) -> None:
    """Write a private key with 0600 bits set *at creation time*.

    ``open(path, "wb")`` creates the file with ``0666 & ~umask`` (typically
    0644 — world-readable) and only an explicit ``chmod`` would tighten
    it afterwards. That leaves a TOCTOU window in which a local non-root
    attacker can read the key. ``os.open`` with ``O_CREAT|O_WRONLY|O_TRUNC``
    and mode ``0o600`` is the *only* portable way to ensure the file is
    never readable to anyone else, even momentarily. A chmod failure on
    a key path is treated as a hard error: a 0644 RSA private key on
    disk is never acceptable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Remove any existing file so O_CREAT|O_EXCL semantics aren't needed;
    # we still get the safe creation bits via the mode argument.
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    # Defensive re-chmod in case the process umask interacted oddly with
    # O_CREAT's mode argument on some platforms; failure here means the
    # file is on a filesystem that doesn't support POSIX modes (e.g. some
    # FAT/exotic mounts), which is *not* an acceptable place for a key.
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        raise OSError(
            f"could not set 0600 perms on private-key file {path}: {exc}. "
            "Refusing to leave a private key with default permissions."
        ) from exc


def _write_public(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass


def _key_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _cert_pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _load_key(path: Path) -> rsa.RSAPrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError(f"{path}: unsupported private key type {type(key).__name__}")
    return key


def _san_entries(
    hostnames: Iterable[str], ips: Iterable[str]
) -> list[x509.GeneralName]:
    entries: list[x509.GeneralName] = []
    for h in hostnames:
        h = (h or "").strip().lower()
        if h:
            entries.append(x509.DNSName(h))
    for raw in ips:
        try:
            obj = ipaddress.ip_address(str(raw).split("%", 1)[0].strip())
        except (ValueError, AttributeError):
            continue
        entries.append(x509.IPAddress(obj))
    if not entries:
        raise ValueError("SAN must include at least one hostname or IP")
    return entries


def _next_serial(serial_file: Path) -> int:
    """Issue a fresh certificate serial number.

    Uses ``x509.random_serial_number()`` (160 bits of entropy, RFC 5280
    compliant — same as the CA self-cert at create_ca). The serial file
    is still touched as a human-readable audit trail of issuance count,
    but the serial value itself is not derived from the file: a wiped
    or corrupted serial file MUST NOT cause subsequent mints to collide
    with prior generations' serials.
    """
    serial_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        cur = int(serial_file.read_text().strip(), 10)
    except (OSError, ValueError):
        cur = 0
    try:
        with open(serial_file, "w") as f:
            f.write(f"{cur + 1}\n")
        os.chmod(serial_file, 0o600)
    except OSError:
        pass
    return x509.random_serial_number()


# --------------------------------------------------------------------- CA


def create_ca(
    cfg: TLSConfig,
    *,
    common_name: str,
    validity_days: Optional[int] = None,
    force: bool = False,
) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    """Create a new self-signed CA and write it to ``cfg.ca_cert`` / ``cfg.ca_key``.

    Refuses to overwrite an existing CA unless ``force=True``.
    """
    if cfg.has_ca_authority() and not force:
        raise FileExistsError(
            f"CA already exists at {cfg.ca_cert}; pass force=True to overwrite"
        )
    days = validity_days if validity_days is not None else cfg.ca_validity_days
    key = _gen_key(2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Forgather"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    _write_public(cfg.ca_cert, _cert_pem(cert))
    _write_secret(cfg.ca_key, _key_pem(key))
    try:
        os.chmod(cfg.ca_cert.parent, 0o700)
    except OSError:
        pass
    log.info("Created CA: %s (CN=%s, valid %d days)", cfg.ca_cert, common_name, days)
    return cert, key


# --------------------------------------------------------------------- leaf


@dataclass
class MintedCert:
    cert_pem: bytes
    key_pem: bytes


def mint_server_cert(
    cfg: TLSConfig,
    *,
    hostnames: Iterable[str],
    ips: Iterable[str],
    validity_days: Optional[int] = None,
) -> MintedCert:
    """Issue a server cert from this host's CA. Returns PEM blobs only.

    The caller decides where to write them (``cfg.server_cert`` for the
    local host, or a temp dir for distribution to a peer via
    ``forgather tls mint --output``).
    """
    if not cfg.has_ca_authority():
        raise FileNotFoundError(
            f"No CA present (expected {cfg.ca_cert} + {cfg.ca_key}). "
            f"Run 'forgather tls init' first, or import a CA."
        )
    ca_cert = _load_cert(cfg.ca_cert)
    ca_key = _load_key(cfg.ca_key)

    san = _san_entries(hostnames, ips)
    # Use first hostname as CN; falls back to first IP.
    cn = ""
    for entry in san:
        if isinstance(entry, x509.DNSName):
            cn = entry.value
            break
    if not cn:
        for entry in san:
            if isinstance(entry, x509.IPAddress):
                cn = str(entry.value)
                break

    days = validity_days if validity_days is not None else cfg.validity_days
    leaf_key = _gen_key(2048)
    now = _now()
    builder = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn or "forgather")])
        )
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(_next_serial(cfg.ca_serial))
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            # Server cert doubles as the node's client cert when peers
            # call this node over mTLS — same identity, both directions.
            # Without CLIENT_AUTH here, OpenSSL rejects the cert as
            # "unsuitable certificate purpose" during the inbound mTLS
            # handshake.
            x509.ExtendedKeyUsage(
                [
                    x509.ExtendedKeyUsageOID.SERVER_AUTH,
                    x509.ExtendedKeyUsageOID.CLIENT_AUTH,
                ]
            ),
            critical=False,
        )
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()),
            critical=False,
        )
    )
    cert = builder.sign(ca_key, hashes.SHA256())
    return MintedCert(cert_pem=_cert_pem(cert), key_pem=_key_pem(leaf_key))


def install_server_cert(cfg: TLSConfig, minted: MintedCert) -> None:
    """Write a freshly-minted leaf into ``cfg.server_cert``/``server_key``."""
    _write_public(cfg.server_cert, minted.cert_pem)
    _write_secret(cfg.server_key, minted.key_pem)


# --------------------------------------------------------------------- bundle


def rebuild_bundle(cfg: TLSConfig) -> Path:
    """Concatenate ``ca/ca.crt`` + every ``trusted/*.crt`` into ``ca-bundle.crt``.

    Idempotent. Skips files that don't parse as certs. Returns the
    bundle path even if it's empty (caller decides what to do).
    """
    chunks: list[bytes] = []
    if cfg.ca_cert.is_file():
        chunks.append(cfg.ca_cert.read_bytes().rstrip() + b"\n")
    if cfg.trusted_dir.is_dir():
        for entry in sorted(cfg.trusted_dir.glob("*.crt")):
            try:
                _load_cert(entry)  # validate
            except Exception:
                log.warning("skipping invalid trusted cert: %s", entry)
                continue
            chunks.append(entry.read_bytes().rstrip() + b"\n")
    cfg.ca_bundle.parent.mkdir(parents=True, exist_ok=True)
    cfg.ca_bundle.write_bytes(b"".join(chunks))
    try:
        os.chmod(cfg.ca_bundle, 0o644)
    except OSError:
        pass
    return cfg.ca_bundle


def import_trusted_ca(
    cfg: TLSConfig, src: Path, *, name: Optional[str] = None
) -> Path:
    """Copy ``src`` into ``trusted/<name>.crt`` and rebuild the bundle.

    Validates that the cert is actually a CA (BasicConstraints CA:TRUE).
    """
    cert = _load_cert(src)
    try:
        bc = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value
        if not getattr(bc, "ca", False):
            raise ValueError("certificate is not marked as a CA (BasicConstraints CA:FALSE)")
    except x509.ExtensionNotFound:
        raise ValueError("certificate has no BasicConstraints extension; not a CA")

    label = name or cert.subject.rfc4514_string().replace("/", "_").replace(" ", "_")
    label = "".join(c for c in label if c.isalnum() or c in "-_.")[:64] or "imported"
    if not label.endswith(".crt"):
        label = f"{label}.crt"

    cfg.trusted_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.trusted_dir / label
    dest.write_bytes(src.read_bytes())
    try:
        os.chmod(dest, 0o644)
    except OSError:
        pass
    rebuild_bundle(cfg)
    return dest


# --------------------------------------------------------------------- info


def cert_info(path: Path) -> dict:
    """Inspect a cert on disk and return a small summary dict."""
    cert = _load_cert(path)
    san_dns: list[str] = []
    san_ip: list[str] = []
    try:
        san_ext = cert.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        ).value
        san_dns = list(san_ext.get_values_for_type(x509.DNSName))
        san_ip = [str(v) for v in san_ext.get_values_for_type(x509.IPAddress)]
    except x509.ExtensionNotFound:
        pass
    return {
        "subject": cert.subject.rfc4514_string(),
        "issuer": cert.issuer.rfc4514_string(),
        "serial": cert.serial_number,
        "not_before": cert.not_valid_before_utc.isoformat(),
        "not_after": cert.not_valid_after_utc.isoformat(),
        "san_dns": san_dns,
        "san_ip": san_ip,
        "is_ca": _cert_is_ca(cert),
        "days_remaining": (cert.not_valid_after_utc - _now()).days,
    }


def _cert_is_ca(cert: x509.Certificate) -> bool:
    try:
        bc = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value
        return bool(getattr(bc, "ca", False))
    except x509.ExtensionNotFound:
        return False
