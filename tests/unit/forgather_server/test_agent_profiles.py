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


def test_models_route_honors_tls_posture(monkeypatch):
    from forgather_server.routes import agent as ar

    captured = {}

    def cap(**kw):
        captured.update(kw)
        return [{"id": "m1", "max_model_len": 4096}]

    monkeypatch.setattr(ar.agent_tls, "list_models", cap)
    out = ar.list_agent_models(
        ar.ModelsRequest(provider="anthropic", base_url="https://x", api_key="k", verify_tls=True)
    )
    assert out == {"models": [{"id": "m1", "max_model_len": 4096}]}
    # The probe honors the requested TLS posture (no silent skip).
    assert captured["verify_tls"] is True
    # And passes verify_tls=False through when the caller opts out.
    ar.list_agent_models(
        ar.ModelsRequest(provider="anthropic", base_url="https://x", api_key="k", verify_tls=False)
    )
    assert captured["verify_tls"] is False


def test_models_route_does_not_redirect_saved_token_to_other_url(store_file, monkeypatch):
    from forgather_server.routes import agent as ar

    p = store.add_profile(label="A", base_url="https://A:8000", api_key="SECRET", verify_tls=False)

    captured = {}

    def cap(**kw):
        captured.update(kw)
        return [{"id": "m", "max_model_len": None}]

    monkeypatch.setattr(ar.agent_tls, "list_models", cap)

    # Same server → the saved token is used.
    ar.list_agent_models(ar.ModelsRequest(profile_id=p.id, base_url="https://A:8000"))
    assert captured["api_key"] == "SECRET"

    # Different URL with the same profile_id → token must NOT be sent there.
    captured.clear()
    ar.list_agent_models(ar.ModelsRequest(profile_id=p.id, base_url="https://evil:8000"))
    assert captured.get("api_key") in (None, "")


def test_propose_edit_config_enforces_fs_root(tmp_path, monkeypatch):
    from forgather_server import paths
    from forgather_server.agent import tools_authoring

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("token=hunter2\n")
    monkeypatch.setattr(paths, "is_path_in_fs_root", lambda pth: str(pth).startswith(str(allowed)))

    with pytest.raises(PermissionError):
        tools_authoring._propose_edit_config(
            {
                "project_dir": str(allowed),
                "config_name": "x.yaml",
                "path": str(outside),
                "new_content": "x",
            }
        )


def test_write_template_file_refuses_overwrite(tmp_path):
    f = tmp_path / "new.yaml"
    config_ops_write = __import__(
        "forgather_server.config_ops", fromlist=["write_template_file"]
    ).write_template_file
    config_ops_write(str(f), "first\n")
    assert f.read_text() == "first\n"
    with pytest.raises(FileExistsError):
        config_ops_write(str(f), "second\n")
    assert f.read_text() == "first\n"  # not clobbered


def test_build_loop_skips_probe_when_fully_pinned(store_file, monkeypatch):
    from forgather_server.agent import runtime

    called = {"n": 0}

    def boom(**kw):
        called["n"] += 1
        raise AssertionError("list_models should not be called when fully pinned")

    monkeypatch.setattr(runtime.agent_tls, "list_models", boom)
    p = store.add_profile(
        label="pinned", base_url="https://kitt:8000", model="qwen", max_tokens=8192, verify_tls=False
    )
    loop = runtime._build_loop(store.get_profile(p.id))
    assert called["n"] == 0
    assert loop.provider.model == "qwen"
    assert loop.provider.max_tokens == 8192


# ---- credential resolution (high-value-key guard) --------------------------


def test_anthropic_key_not_sent_to_custom_base_url(monkeypatch):
    from forgather_server.agent import runtime

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    # Local/third-party server, blank key, default env name -> must NOT
    # forward the Anthropic key.
    assert (
        runtime.resolve_credential("", "ANTHROPIC_API_KEY", "https://kitt:8000") is None
    )
    # Claude (no base_url) -> the env key is fine to use.
    assert runtime.resolve_credential("", "ANTHROPIC_API_KEY", "") == "sk-ant-secret"
    # Explicit key always wins, even for a custom base_url.
    assert runtime.resolve_credential("tok", "ANTHROPIC_API_KEY", "https://kitt:8000") == "tok"


def test_custom_env_var_allowed_for_local_server(monkeypatch):
    from forgather_server.agent import runtime

    monkeypatch.setenv("VLLM_TOKEN", "bearer-xyz")
    # A deliberately-named env var (not the Anthropic default) is honored.
    assert (
        runtime.resolve_credential("", "VLLM_TOKEN", "https://kitt:8000") == "bearer-xyz"
    )


# ---- max_tokens auto budgeting --------------------------------------------


def test_auto_max_tokens_caps_and_fallback():
    from forgather_server.agent import runtime

    assert runtime._auto_max_tokens(131072) == 32768  # capped
    assert runtime._auto_max_tokens(8192) == 8192  # below cap → full context
    assert runtime._auto_max_tokens(1048576) == 32768  # 1M context still capped
    assert runtime._auto_max_tokens(None) == runtime.AUTO_MAX_TOKENS_FALLBACK


def test_provider_clamps_max_tokens_to_remaining_context():
    # No anthropic client needed: budgeting is pure (client is lazy).
    from forgather_server.agent.providers.anthropic import AnthropicProvider

    def msg(text):
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    # Unknown context (Claude) → base returned unchanged.
    p_none = AnthropicProvider(model="m", max_tokens=32768, max_model_len=None)
    assert p_none._effective_max_tokens(msg("hi")) == 32768

    p = AnthropicProvider(model="m", max_tokens=32768, max_model_len=131072)
    # Small prompt → full base budget.
    assert p._effective_max_tokens(msg("hello")) == 32768
    # Medium prompt (~100k tokens) → reduced to fit remaining context.
    eff_med = p._effective_max_tokens(msg("y" * (100_000 * 3)))
    assert 512 <= eff_med < 32768
    # Prompt nearly fills the window → output floored, never negative.
    eff_full = p._effective_max_tokens(msg("x" * (131_000 * 3)))
    assert eff_full == 512


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


# ---- /agent/import validation ---------------------------------------------


def test_agent_import_rejects_unbalanced_tool_calls(monkeypatch):
    from fastapi import HTTPException

    from forgather_server.routes import agent as ar

    monkeypatch.setattr(ar.runtime, "is_enabled", lambda: True)
    msgs = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]}
    ]  # tool_use with no matching tool_result
    with pytest.raises(HTTPException) as ei:
        ar.agent_import(ar.ImportRequest(messages=msgs))
    assert ei.value.status_code == 400


def test_agent_import_accepts_balanced(monkeypatch):
    from forgather_server.routes import agent as ar

    monkeypatch.setattr(ar.runtime, "is_enabled", lambda: True)
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    out = ar.agent_import(ar.ImportRequest(messages=msgs))
    assert out["session_id"]


def test_agent_import_requires_enabled(monkeypatch):
    from fastapi import HTTPException

    from forgather_server.routes import agent as ar

    monkeypatch.setattr(ar.runtime, "is_enabled", lambda: False)
    with pytest.raises(HTTPException) as ei:
        ar.agent_import(ar.ImportRequest(messages=[]))
    assert ei.value.status_code == 503
