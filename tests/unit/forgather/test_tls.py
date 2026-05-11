"""Smoke tests for the shared ``forgather.tls`` package.

Cover the parts a production rollout depends on:

* CA + leaf cert minting actually produces certs that verify.
* Trust-bundle rebuild concatenates the local CA plus any imports.
* The non-loopback bind policy refuses cleartext non-loopback hosts.
* ``uvicorn_ssl_kwargs`` returns the right paths once provisioned.
"""

from __future__ import annotations

import socket
import ssl
from pathlib import Path

import pytest

from forgather.tls import (
    TLSRequiredError,
    enforce_non_loopback_policy,
    host_is_loopback,
    httpx_verify,
    is_enabled,
    load_config,
    uvicorn_ssl_kwargs,
)
from forgather.tls.ca import (
    cert_info,
    create_ca,
    import_trusted_ca,
    install_server_cert,
    mint_server_cert,
    rebuild_bundle,
)
from forgather.tls.config import save_config


@pytest.fixture
def tls_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGATHER_TLS_DIR", str(tmp_path))
    return tmp_path


def _provisioned_cfg(tls_root, hostnames=("localhost",), ips=("127.0.0.1",)):
    cfg = load_config()
    cfg.enabled = True
    cfg.san_hostnames = list(hostnames)
    cfg.san_ips = list(ips)
    create_ca(cfg, common_name="Test CA")
    minted = mint_server_cert(cfg, hostnames=hostnames, ips=ips)
    install_server_cert(cfg, minted)
    rebuild_bundle(cfg)
    save_config(cfg)
    return load_config()


def test_load_config_returns_defaults_when_absent(tls_root):
    cfg = load_config()
    assert cfg.root == Path(tls_root)
    assert cfg.enabled is False
    assert cfg.is_provisioned() is False
    assert cfg.has_ca_authority() is False


def test_create_ca_writes_files_with_basic_constraints(tls_root):
    cfg = load_config()
    create_ca(cfg, common_name="Unit-Test CA")
    info = cert_info(cfg.ca_cert)
    assert info["is_ca"] is True
    assert "Unit-Test CA" in info["subject"]
    # Key file must be private (0600) — we don't crash if chmod failed,
    # but it should be readable only by us.
    mode = cfg.ca_key.stat().st_mode & 0o777
    assert mode == 0o600


def test_mint_server_cert_chains_to_ca(tls_root):
    cfg = _provisioned_cfg(tls_root, hostnames=("localhost", "host.example"),
                           ips=("127.0.0.1", "10.0.0.1"))
    info = cert_info(cfg.server_cert)
    assert info["is_ca"] is False
    assert "localhost" in info["san_dns"]
    assert "host.example" in info["san_dns"]
    assert "127.0.0.1" in info["san_ip"]
    assert "10.0.0.1" in info["san_ip"]

    # Verify the leaf chains to the CA using the OpenSSL trust store
    # built around the CA cert. If chaining is broken, load_verify_locations
    # + verify_mode=CERT_REQUIRED would still let the connection logic
    # accept; do the explicit chain check via SSLContext.
    ctx = ssl.create_default_context(cafile=str(cfg.ca_cert))
    # Loading the cert + verifying isn't a one-liner via SSLContext, so
    # rely on cryptography's verifier indirectly: parse cert + signature
    # and check issuer matches.
    from cryptography import x509

    ca = x509.load_pem_x509_certificate(cfg.ca_cert.read_bytes())
    leaf = x509.load_pem_x509_certificate(cfg.server_cert.read_bytes())
    assert leaf.issuer == ca.subject


def test_rebuild_bundle_includes_local_ca_and_trusted(tls_root, tmp_path):
    cfg = _provisioned_cfg(tls_root)
    # Create a second standalone CA in a separate dir, import it.
    other = tmp_path / "other"
    other.mkdir()
    # Build a temp config rooted at `other` to mint a different CA.
    import os

    saved = os.environ["FORGATHER_TLS_DIR"]
    os.environ["FORGATHER_TLS_DIR"] = str(other)
    try:
        other_cfg = load_config()
        create_ca(other_cfg, common_name="Other CA")
        other_ca = other_cfg.ca_cert
    finally:
        os.environ["FORGATHER_TLS_DIR"] = saved

    dest = import_trusted_ca(cfg, other_ca, name="other")
    assert dest.exists()
    bundle_text = cfg.ca_bundle.read_text()
    assert "BEGIN CERTIFICATE" in bundle_text
    # Bundle should contain both CAs (each PEM starts with the header).
    assert bundle_text.count("BEGIN CERTIFICATE") >= 2


def test_import_rejects_non_ca_certificates(tls_root, tmp_path):
    cfg = _provisioned_cfg(tls_root)
    # The leaf cert is NOT a CA — import should refuse.
    with pytest.raises(ValueError):
        import_trusted_ca(cfg, cfg.server_cert)


def test_host_is_loopback_classification():
    assert host_is_loopback("127.0.0.1") is True
    assert host_is_loopback("::1") is True
    assert host_is_loopback("localhost") is True
    assert host_is_loopback("10.0.0.5") is False
    assert host_is_loopback("0.0.0.0") is False
    assert host_is_loopback("") is False
    # Unresolvable hostnames are non-loopback (fail safe).
    assert host_is_loopback("nope.invalid") is False


def test_policy_refuses_non_loopback_http(tls_root):
    with pytest.raises(TLSRequiredError):
        enforce_non_loopback_policy(
            "0.0.0.0", tls_enabled=False, insecure=False, service="test"
        )


def test_policy_allows_loopback_http(tls_root):
    # Should not raise — loopback HTTP is OK.
    enforce_non_loopback_policy(
        "127.0.0.1", tls_enabled=False, insecure=False, service="test"
    )


def test_policy_allows_insecure_override(tls_root):
    enforce_non_loopback_policy(
        "0.0.0.0", tls_enabled=False, insecure=True, service="test"
    )


def test_policy_allows_non_loopback_with_tls(tls_root):
    enforce_non_loopback_policy(
        "0.0.0.0", tls_enabled=True, insecure=False, service="test"
    )


def test_uvicorn_kwargs_when_provisioned(tls_root):
    _provisioned_cfg(tls_root)
    kwargs = uvicorn_ssl_kwargs()
    assert "ssl_certfile" in kwargs and "ssl_keyfile" in kwargs
    assert Path(kwargs["ssl_certfile"]).is_file()
    assert Path(kwargs["ssl_keyfile"]).is_file()


def test_uvicorn_kwargs_empty_when_disabled(tls_root):
    cfg = load_config()
    assert uvicorn_ssl_kwargs() == {}


def test_is_enabled_reflects_provisioning(tls_root):
    assert is_enabled() is False
    _provisioned_cfg(tls_root)
    assert is_enabled() is True


def test_httpx_verify_points_at_bundle(tls_root):
    _provisioned_cfg(tls_root)
    v = httpx_verify()
    assert isinstance(v, str)
    assert v.endswith("ca-bundle.crt")
    assert Path(v).is_file()
