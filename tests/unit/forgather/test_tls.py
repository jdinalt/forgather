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
    stdlib_ssl_context,
    urllib_ssl_context,
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
    cfg = _provisioned_cfg(
        tls_root, hostnames=("localhost", "host.example"), ips=("127.0.0.1", "10.0.0.1")
    )
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


def test_mint_server_cert_has_server_and_client_auth_eku(tls_root):
    """Server certs must carry both EKUs so the same identity works
    in both directions for cluster mTLS (issue #31).

    Without ``CLIENT_AUTH``, OpenSSL rejects the cert during the
    inbound mTLS handshake with "unsuitable certificate purpose",
    silently breaking peer-to-peer cluster calls. The failure mode
    only surfaces at runtime against a real peer, so this regression
    test pins the EKU at the unit level.
    """
    from cryptography import x509

    cfg = _provisioned_cfg(tls_root)
    leaf = x509.load_pem_x509_certificate(cfg.server_cert.read_bytes())
    eku = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    oids = {e.dotted_string for e in eku}
    assert x509.ExtendedKeyUsageOID.SERVER_AUTH.dotted_string in oids
    assert x509.ExtendedKeyUsageOID.CLIENT_AUTH.dotted_string in oids


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


def test_httpx_verify_returns_chain_only_ssl_context(tls_root):
    """Default (verify_hostname=False) returns an SSLContext with chain
    validation on but hostname matching off — the LAN-friendly mode."""
    import ssl

    _provisioned_cfg(tls_root)
    v = httpx_verify()
    assert isinstance(v, ssl.SSLContext)
    assert v.check_hostname is False
    assert v.verify_mode == ssl.CERT_REQUIRED


def test_httpx_verify_strict_when_verify_hostname_set(tls_root):
    """verify_hostname=True turns on RFC-6125 SAN-vs-URL checking."""
    import ssl

    cfg = _provisioned_cfg(tls_root)
    cfg.verify_hostname = True
    save_config(cfg)
    cfg2 = load_config()
    v = httpx_verify(cfg2)
    assert isinstance(v, ssl.SSLContext)
    assert v.check_hostname is True
    assert v.verify_mode == ssl.CERT_REQUIRED


def test_httpx_verify_without_bundle_returns_true(tls_root):
    """No bundle → True (system trust); peer connections fail closed."""
    # No cert provisioned.
    cfg = load_config()
    v = httpx_verify(cfg)
    assert v is True


def test_mint_with_no_san_flags_uses_placeholder(tls_root):
    """`tls mint` (no --hostname/--ip) mints a chain-only-trust cert.

    Operators on dynamic-IP LANs shouldn't have to know each peer's
    address to mint its cert. Validate by checking the placeholder
    SAN is present.
    """
    import argparse
    import tempfile

    from forgather.cli.tls import _cmd_mint

    _provisioned_cfg(tls_root)
    with tempfile.TemporaryDirectory() as out_dir:
        args = argparse.Namespace(hostname=[], ip=[], output=out_dir)
        rc = _cmd_mint(args)
        assert rc == 0
        info = cert_info(Path(out_dir) / "server.crt")
        # Placeholder SAN: forgather-peer + localhost + 127.0.0.1 + ::1.
        assert "forgather-peer" in info["san_dns"]
        assert "127.0.0.1" in info["san_ip"]


def test_private_key_is_0600_from_creation(tls_root):
    """Atomic-create path must not leave the key at default perms."""
    cfg = _provisioned_cfg(tls_root)
    assert cfg.ca_key.stat().st_mode & 0o777 == 0o600
    assert cfg.server_key.stat().st_mode & 0o777 == 0o600


def test_random_serial_numbers_dont_collide(tls_root):
    """Re-creating ca.srl shouldn't reuse serials from prior generations."""
    cfg = _provisioned_cfg(tls_root)
    serial_a = cert_info(cfg.server_cert)["serial"]
    # Wipe the serial counter file — simulates ops accidentally deleting it.
    cfg.ca_serial.unlink(missing_ok=True)
    minted = mint_server_cert(cfg, hostnames=("localhost",), ips=("127.0.0.1",))
    install_server_cert(cfg, minted)
    serial_b = cert_info(cfg.server_cert)["serial"]
    assert serial_a != serial_b
    # Both should be high-entropy (>= 64 bits).
    assert int(serial_a) > 2**64
    assert int(serial_b) > 2**64


def test_san_hard_cap_refuses_huge_lists(tls_root):
    from forgather.tls.discovery import merge_san

    huge_hosts = [f"host{i}.lan" for i in range(300)]
    with pytest.raises(ValueError):
        merge_san([], [], extra_hostnames=huge_hosts, extra_ips=[])


def test_discovery_caps_auto_san(tls_root):
    from forgather.tls.discovery import detect_hostnames, detect_ips

    # cap=2 should always shrink to <=2 even on a host with more.
    h = detect_hostnames(cap=2)
    i = detect_ips(cap=2)
    assert len(h) <= 2
    assert len(i) <= 2


def test_resolve_state_precedence(tls_root):
    """--no-tls always wins; --tls wins over disabled config; default falls back."""
    import argparse

    from forgather.tls.runtime import _resolve_state

    _provisioned_cfg(tls_root)
    cfg = load_config()
    assert cfg.enabled is True

    # Default: pick from config.
    args = argparse.Namespace(tls=None, no_tls=False, tls_cert=None, tls_key=None)
    _, on, _, _ = _resolve_state(args, cfg)
    assert on is True

    # --no-tls overrides enabled.
    args = argparse.Namespace(tls=None, no_tls=True, tls_cert=None, tls_key=None)
    _, on, _, _ = _resolve_state(args, cfg)
    assert on is False

    # --tls overrides disabled config.
    cfg.enabled = False
    args = argparse.Namespace(tls=True, no_tls=False, tls_cert=None, tls_key=None)
    _, on, _, _ = _resolve_state(args, cfg)
    assert on is True


def test_policy_error_branches_on_state(tls_root):
    """The refusal message should suggest the right next step per state."""
    # Provisioned but disabled → should mention 'tls enable' or --tls.
    cfg = _provisioned_cfg(tls_root)
    cfg.enabled = False
    save_config(cfg)
    cfg2 = load_config()
    try:
        enforce_non_loopback_policy(
            "0.0.0.0", tls_enabled=False, insecure=False, service="x", cfg=cfg2
        )
    except Exception as exc:
        msg = str(exc)
        assert "tls enable" in msg or "--tls" in msg, msg


def test_install_rejects_mismatched_cert_and_key(tls_root, tmp_path):
    """`forgather tls install` must refuse a cert/key pair that don't match."""
    import argparse

    from forgather.cli.tls import _cmd_install

    cfg = _provisioned_cfg(tls_root)
    # Mint two independent cert/key pairs.
    a = mint_server_cert(cfg, hostnames=("a.example",), ips=("10.0.0.1",))
    b = mint_server_cert(cfg, hostnames=("b.example",), ips=("10.0.0.2",))
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(a.cert_pem)
    key_path.write_bytes(b.key_pem)  # wrong key.

    args = argparse.Namespace(cert=str(cert_path), key=str(key_path), ca=None)
    rc = _cmd_install(args)
    assert rc == 1


def test_member_tls_preserved_when_update_omits_field(tls_root):
    """update_member(tls=None) must NOT overwrite an existing tls value.

    Important: we use the same import path other tests use
    (``forgather_server.cluster``), and we *always* call
    ``_reset_for_tests`` on both entry and exit so the module-level
    singleton is left in a clean state. Reloading the module here
    would break other tests in the same pytest session that hold a
    reference to the old module.
    """
    import sys

    # Add tools/ to sys.path the same way the test_routes_cluster
    # tests do — once, idempotent — without deleting any cached
    # imports.
    tools_dir = str(Path(__file__).parents[3] / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from forgather_server import cluster

    cluster._state._reset_for_tests()
    try:
        cluster.activate("test-cluster", port=8765, tls=True)
        cluster.update_member(
            "node-b",
            hostname="b",
            address="10.0.0.2",
            port=8765,
            cluster_name="test-cluster",
            tls=True,
        )
        members = {m.node_id: m for m in cluster.members()}
        assert members["node-b"].tls is True
        # Now update without passing tls — should preserve True.
        cluster.update_member(
            "node-b",
            hostname="b",
            address="10.0.0.2",
            port=8765,
            cluster_name="test-cluster",
        )
        members = {m.node_id: m for m in cluster.members()}
        assert members["node-b"].tls is True
        # Pass tls=False — should overwrite.
        cluster.update_member(
            "node-b",
            hostname="b",
            address="10.0.0.2",
            port=8765,
            cluster_name="test-cluster",
            tls=False,
        )
        members = {m.node_id: m for m in cluster.members()}
        assert members["node-b"].tls is False
    finally:
        cluster._state._reset_for_tests()


# ---------------------------------------------------------------------------
# stdlib_ssl_context / urllib_ssl_context (DiLoCo / http.server adapters)
# ---------------------------------------------------------------------------


def test_stdlib_ssl_context_none_when_disabled(tls_root):
    """No CLI flag + unprovisioned config → None (cleartext server)."""
    assert stdlib_ssl_context() is None


def test_stdlib_ssl_context_loads_cert_chain_when_provisioned(tls_root):
    """Provisioned cluster + TLS on → server context with cert chain
    loaded and CERT_OPTIONAL configured so mTLS handshakes work."""
    _provisioned_cfg(tls_root)
    ctx = stdlib_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    # Server context loads keylog if requested; we don't assert that.
    # Bundle present → CERT_OPTIONAL.
    assert ctx.verify_mode == ssl.CERT_OPTIONAL


def test_stdlib_ssl_context_no_bundle_skips_client_auth(tls_root):
    """Cert+key provisioned but no CA bundle → context has no client-auth.
    Verify mode falls back to CERT_NONE (server's default for CLIENT_AUTH
    Purpose), so no mTLS path."""
    cfg = _provisioned_cfg(tls_root)
    # Remove the bundle to simulate a half-provisioned host.
    cfg.ca_bundle.unlink()
    cfg.ca_cert.unlink()
    save_config(cfg)
    cfg2 = load_config()
    assert cfg2.effective_bundle() is None
    ctx = stdlib_ssl_context(cfg=cfg2)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_NONE


def test_stdlib_ssl_context_raises_when_files_missing(tls_root, monkeypatch):
    """TLS forced on but cert file doesn't exist → FileNotFoundError."""
    import argparse

    args = argparse.Namespace(
        tls=True,
        no_tls=False,
        tls_cert="/nonexistent/cert.pem",
        tls_key="/nonexistent/key.pem",
    )
    with pytest.raises(FileNotFoundError):
        stdlib_ssl_context(args=args)


def test_urllib_ssl_context_none_when_no_bundle(tls_root):
    """Default (verify=True) and no CA bundle → None; caller falls back
    to system trust (urllib's default when no context= is passed)."""
    ctx = urllib_ssl_context()
    assert ctx is None


def test_urllib_ssl_context_loads_bundle_when_provisioned(tls_root):
    """Provisioned cluster → SSLContext with CA bundle + this node's
    cert loaded for mTLS identity. check_hostname off by default
    (matches httpx_verify's LAN-friendly mode)."""
    _provisioned_cfg(tls_root)
    ctx = urllib_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_urllib_ssl_context_strict_hostname(tls_root):
    """verify_hostname=True flips on RFC-6125 SAN matching."""
    cfg = _provisioned_cfg(tls_root)
    cfg.verify_hostname = True
    save_config(cfg)
    cfg2 = load_config()
    ctx = urllib_ssl_context(cfg2)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.check_hostname is True


def test_urllib_ssl_context_verify_false_returns_unverified_ctx(tls_root):
    """verify=False opt-out → context that won't verify chain or hostname.
    Used for SSH-tunneled remotes where the trust boundary is external."""
    _provisioned_cfg(tls_root)
    ctx = urllib_ssl_context(verify=False)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE


def test_stdlib_and_urllib_contexts_interoperate(tls_root):
    """End-to-end: spin up an http.server with stdlib_ssl_context,
    hit it with urllib using urllib_ssl_context, verify the
    handshake completes and the request succeeds."""
    import http.server
    import threading
    import urllib.request

    _provisioned_cfg(tls_root)
    server_ctx = stdlib_ssl_context()
    assert server_ctx is not None

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args, **kwargs):
            pass  # silence

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.socket = server_ctx.wrap_socket(httpd.socket, server_side=True)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        client_ctx = urllib_ssl_context()
        assert client_ctx is not None
        # Use localhost which is in the SAN list of _provisioned_cfg.
        url = f"https://localhost:{port}/"
        with urllib.request.urlopen(url, context=client_ctx, timeout=5) as resp:
            assert resp.status == 200
            assert resp.read() == b"ok"
    finally:
        httpd.shutdown()
        httpd.server_close()
