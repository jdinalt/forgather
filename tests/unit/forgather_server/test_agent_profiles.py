"""Tests for agent profile store, TLS verify, and runtime hot-swap."""

from __future__ import annotations

import os
import ssl

import pytest

from forgather_server import agent_profiles_store as store
from forgather_server import agent_tls


@pytest.fixture
def store_file(tmp_path, monkeypatch):
    """Redirect the profile store to a temp file."""
    f = tmp_path / "agent_profiles.json"
    monkeypatch.setattr(store, "agent_profiles_file", lambda: f)
    return f


# ---- profile store ---------------------------------------------------------


def test_add_lists_and_first_is_active(store_file):
    p1 = store.add_profile(label="claude", model="claude-x")
    p2 = store.add_profile(label="local", base_url="https://kitt:8000", model="qwen")
    ids = [p.id for p in store.list_profiles()]
    assert ids == [p1.id, p2.id]
    # First added becomes active automatically.
    assert store.get_active_id() == p1.id


def test_set_active_and_get_active(store_file):
    p1 = store.add_profile(label="a", model="m1")
    p2 = store.add_profile(label="b", model="m2")
    assert store.set_active(p2.id) is True
    assert store.get_active().id == p2.id
    assert store.set_active("nope") is False


def test_update_profile(store_file):
    p = store.add_profile(label="a", model="m1", verify_tls=True)
    updated = store.update_profile(p.id, model="m2", verify_tls=False)
    assert updated.model == "m2"
    assert updated.verify_tls is False
    assert store.get_profile(p.id).model == "m2"


def test_remove_reassigns_active(store_file):
    p1 = store.add_profile(label="a", model="m1")
    p2 = store.add_profile(label="b", model="m2")
    store.set_active(p1.id)
    store.remove_profile(p1.id)
    # Active fell back to the remaining profile.
    assert store.get_active_id() == p2.id
    assert store.get_profile(p1.id) is None


def test_revision_bumps_on_writes(store_file):
    r0 = store.revision()
    store.add_profile(label="a", model="m1")
    r1 = store.revision()
    assert r1 > r0
    p = store.list_profiles()[0]
    store.update_profile(p.id, model="m2")
    assert store.revision() > r1


def test_base_url_trailing_slash_stripped(store_file):
    p = store.add_profile(label="a", base_url="https://kitt:8000/", model="m")
    assert p.base_url == "https://kitt:8000"


def test_api_key_control_chars_rejected(store_file):
    with pytest.raises(ValueError):
        store.add_profile(label="a", model="m", api_key="bad\nkey")


def test_seed_if_empty_only_when_empty_and_has_model(store_file):
    assert store.seed_if_empty({}) is None  # no model -> no seed
    created = store.seed_if_empty({"model": "claude-x"})
    assert created is not None
    # Already has a profile -> second seed is a no-op.
    assert store.seed_if_empty({"model": "other"}) is None


def test_store_file_is_0600(store_file):
    store.add_profile(label="a", model="m")
    mode = os.stat(store_file).st_mode & 0o777
    assert mode == 0o600


# ---- TLS verify ------------------------------------------------------------


def test_build_verify_http_is_noop():
    assert agent_tls.build_verify(base_url="http://x", verify_tls=True, ca_cert_pem="") is True


def test_build_verify_opt_out():
    assert agent_tls.build_verify(base_url="https://x", verify_tls=False, ca_cert_pem="") is False


def test_build_verify_default_system_trust():
    assert agent_tls.build_verify(base_url="https://x", verify_tls=True, ca_cert_pem="") is True


def test_build_verify_imported_cert_builds_context():
    # A syntactically valid self-signed PEM so create_default_context(cadata=)
    # accepts it; we only assert the posture, not a real handshake.
    pem = _self_signed_pem()
    ctx = agent_tls.build_verify(base_url="https://x", verify_tls=True, ca_cert_pem=pem)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.check_hostname is False


# ---- /agent/models route ---------------------------------------------------


def test_models_route_maps_401_to_actionable_message(monkeypatch):
    import httpx
    from fastapi import HTTPException

    from forgather_server.routes import agent as ar

    def boom(**kw):
        request = httpx.Request("GET", "https://x/v1/models")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("401", request=request, response=response)

    monkeypatch.setattr(ar.agent_tls, "list_models", boom)
    with pytest.raises(HTTPException) as ei:
        ar.list_agent_models(ar.ModelsRequest(provider="anthropic", base_url="https://x"))
    assert ei.value.status_code == 400
    assert "bearer token" in ei.value.detail.lower()


def test_models_route_always_skips_tls(monkeypatch):
    from forgather_server.routes import agent as ar

    captured = {}

    def cap(**kw):
        captured.update(kw)
        return ["m1"]

    monkeypatch.setattr(ar.agent_tls, "list_models", cap)
    out = ar.list_agent_models(
        ar.ModelsRequest(provider="anthropic", base_url="https://x", api_key="k", verify_tls=True)
    )
    assert out == {"models": ["m1"]}
    # The discovery probe ignores the profile's TLS posture.
    assert captured["verify_tls"] is False
    assert captured["ca_cert_pem"] == ""


# ---- runtime hot-swap ------------------------------------------------------


def test_runtime_rebuilds_loop_on_profile_change(store_file, monkeypatch):
    from forgather_server.agent import runtime

    calls = {"n": 0}

    def fake_build(profile):
        calls["n"] += 1
        return f"loop-for-{profile.id}-rev{store.revision()}"

    monkeypatch.setattr(runtime, "_build_loop", fake_build)
    runtime.configure(None)  # reset cache

    p1 = store.add_profile(label="a", model="m1")
    p2 = store.add_profile(label="b", model="m2")

    loop_a = runtime.get_loop()
    assert calls["n"] == 1
    # Same active profile + revision -> cached, no rebuild.
    assert runtime.get_loop() is loop_a
    assert calls["n"] == 1

    # Switching the active profile rebuilds.
    store.set_active(p2.id)
    loop_b = runtime.get_loop()
    assert calls["n"] == 2
    assert loop_b != loop_a

    # Editing the active profile rebuilds (revision bumps).
    store.update_profile(p2.id, model="m3")
    runtime.get_loop()
    assert calls["n"] == 3


def test_runtime_status_reflects_active_profile(store_file, monkeypatch):
    from forgather_server.agent import runtime

    runtime.configure(None)
    assert runtime.is_enabled() is False
    assert runtime.status()["enabled"] is False

    p = store.add_profile(label="local", base_url="https://kitt:8000", model="qwen", verify_tls=False)
    st = runtime.status()
    assert st["enabled"] is True
    assert st["active_id"] == p.id
    assert st["model"] == "qwen"
    assert st["verify_tls"] is False
    # Secrets never leak into status.
    assert "api_key" not in st


def _self_signed_pem() -> str:
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "kitt.local")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode()
