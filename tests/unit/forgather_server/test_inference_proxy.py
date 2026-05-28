"""Tests for forgather_server.routes.inference_proxy SSRF policy.

Covers ``_validate_base`` directly: scheme allow-list (always
enforced) and the optional ``--lock-inference-proxy`` localhost-only
mode.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from forgather_server.routes import inference_proxy


@pytest.fixture(autouse=True)
def _reset_lock():
    """Every test starts with the default (unlocked) posture."""
    saved = inference_proxy.LOCK_TO_LOCALHOST
    inference_proxy.LOCK_TO_LOCALHOST = False
    try:
        yield
    finally:
        inference_proxy.LOCK_TO_LOCALHOST = saved


# Default: any URL the operator types is allowed. The operator-typed-URL
# threat model already covers SSRF (the operator can already submit
# training jobs and exfiltrate anything; the proxy adds no capability).
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8137",
        "http://localhost:8137",
        "https://localhost",
        "http://[::1]:8137",
        "http://LOCALHOST:8137",
        "http://192.168.1.5:8137",
        "http://169.254.169.254",  # cloud metadata — operator-typed, fine
        "http://10.0.0.1",
        "http://example.com",
        "https://vllm.lan:8000",
    ],
)
def test_validate_base_accepts_any_http_host_by_default(url):
    assert inference_proxy._validate_base(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8137",
        "http://localhost:8137",
        "https://localhost",
        "http://[::1]:8137",
        "http://LOCALHOST:8137",
    ],
)
def test_validate_base_locked_accepts_localhost(url):
    inference_proxy.LOCK_TO_LOCALHOST = True
    assert inference_proxy._validate_base(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.5:8137",
        "http://169.254.169.254",
        "http://10.0.0.1",
        "http://example.com",
    ],
)
def test_validate_base_locked_rejects_remote(url):
    inference_proxy.LOCK_TO_LOCALHOST = True
    with pytest.raises(HTTPException) as ei:
        inference_proxy._validate_base(url)
    assert ei.value.status_code == 403
    assert "locked" in ei.value.detail.lower()


def test_validate_base_rejects_non_http_scheme():
    with pytest.raises(HTTPException) as ei:
        inference_proxy._validate_base("file:///etc/passwd")
    assert ei.value.status_code == 400
    assert "scheme" in ei.value.detail


def test_validate_base_rejects_non_http_scheme_even_when_locked():
    """Scheme guard is unconditional — locked mode doesn't strengthen
    it, unlocked mode doesn't weaken it."""
    inference_proxy.LOCK_TO_LOCALHOST = True
    with pytest.raises(HTTPException) as ei:
        inference_proxy._validate_base("gopher://127.0.0.1/")
    assert ei.value.status_code == 400


def test_validate_base_strips_trailing_slash():
    assert (
        inference_proxy._validate_base("http://192.168.1.5:8137/")
        == "http://192.168.1.5:8137"
    )


# ---------------------------------------------------------------------------
# Token-forwarding lookup. ``_token_for`` walks job_records and returns the
# auth_token registered for a JobRecord whose host:port matches ``base``.
# ---------------------------------------------------------------------------


from forgather_server.job_records import JobRecord  # noqa: E402


def _bust_cache():
    """The token index is cached for 5s — clear it for deterministic tests."""
    inference_proxy._token_cache = {}
    inference_proxy._token_cache_built_at = 0.0


def test_token_for_returns_record_token(monkeypatch):
    rec = JobRecord(
        queue_id="q1",
        job_type="inference",
        status="running",
        job_params={"port": 8137, "host": "127.0.0.1"},
        auth_token="tok-abc",
    )
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [rec])
    _bust_cache()
    assert inference_proxy._token_for("http://127.0.0.1:8137") == "tok-abc"
    # Localhost aliases resolve to the same record.
    assert inference_proxy._token_for("http://localhost:8137") == "tok-abc"


def test_token_for_returns_none_when_no_match(monkeypatch):
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [])
    _bust_cache()
    assert inference_proxy._token_for("http://127.0.0.1:8137") is None


def test_token_for_returns_none_for_no_auth_record(monkeypatch):
    rec = JobRecord(
        queue_id="q1",
        job_type="inference",
        status="running",
        job_params={"port": 8137, "host": "127.0.0.1"},
        auth_token=None,
    )
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [rec])
    _bust_cache()
    assert inference_proxy._token_for("http://127.0.0.1:8137") is None


def test_token_for_skips_non_inference_records(monkeypatch):
    rec = JobRecord(
        queue_id="q1",
        job_type="training",
        status="running",
        job_params={"port": 8137, "host": "127.0.0.1"},
        auth_token="tok-abc",
    )
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [rec])
    _bust_cache()
    assert inference_proxy._token_for("http://127.0.0.1:8137") is None


def test_token_for_skips_terminal_records(monkeypatch):
    rec = JobRecord(
        queue_id="q1",
        job_type="inference",
        status="done",
        job_params={"port": 8137, "host": "127.0.0.1"},
        auth_token="tok-abc",
    )
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [rec])
    _bust_cache()
    assert inference_proxy._token_for("http://127.0.0.1:8137") is None


# ---------------------------------------------------------------------------
# Off-host lookup via the cluster inference inventory. ``_token_for`` for
# any non-loopback URL consults ``cluster_inference_inventory.master_inventory``
# rather than the local JobRecord index. On master nodes that has the full
# cluster picture; on non-master nodes it's empty and the picker carries
# the token via the X-Inference-Auth-Token header.
# ---------------------------------------------------------------------------


def _reset_cluster_inventory():
    from forgather_server import cluster_inference_inventory

    cluster_inference_inventory._reset_master_state_for_tests()


def test_token_for_off_host_consults_cluster_inventory(monkeypatch):
    from forgather_server import cluster_inference_inventory as cii

    _reset_cluster_inventory()
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [])
    _bust_cache()

    # Seed the master inventory with a remote-peer entry.
    cii.master_inventory.set_master_state(True)
    cii.master_inventory.merge_servers(
        {
            "remote-1": cii.MasterServerEntry(
                server_id="remote-1",
                base_url="http://peer-host:8137",
                auth_token="remote-tok",
                label="x",
                peer_node_id="peer-1",
            )
        }
    )
    try:
        # Off-host URL routes through token_for_url → returns the token.
        assert inference_proxy._token_for("http://peer-host:8137") == "remote-tok"
    finally:
        _reset_cluster_inventory()


def test_token_for_off_host_returns_none_when_inventory_empty(monkeypatch):
    """Non-master nodes have an empty master_inventory — the lookup must
    return None without raising. The picker will then carry the token via
    the explicit header instead."""
    _reset_cluster_inventory()
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [])
    _bust_cache()
    assert inference_proxy._token_for("http://peer-host:8137") is None


def test_token_for_off_host_tolerates_url_case_and_default_port(monkeypatch):
    """token_for_url canonicalizes the URL before lookup so a query like
    ``HTTP://Peer-Host:80`` finds an entry stored as ``http://peer-host``."""
    from forgather_server import cluster_inference_inventory as cii

    _reset_cluster_inventory()
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [])
    _bust_cache()

    cii.master_inventory.set_master_state(True)
    cii.master_inventory.merge_servers(
        {
            "remote-1": cii.MasterServerEntry(
                server_id="remote-1",
                base_url="http://peer-host:8080",
                auth_token="tok",
                label="x",
                peer_node_id="peer-1",
            )
        }
    )
    try:
        # Same URL up-cased.
        assert inference_proxy._token_for("HTTP://PEER-HOST:8080") == "tok"
    finally:
        _reset_cluster_inventory()


def test_token_for_loopback_still_uses_local_jobrecord_index(monkeypatch):
    """Sanity: the off-host inventory path doesn't disturb the local-
    loopback fast path."""
    from forgather_server import cluster_inference_inventory as cii

    _reset_cluster_inventory()
    rec = JobRecord(
        queue_id="q1",
        job_type="inference",
        status="running",
        job_params={"port": 8137, "host": "127.0.0.1"},
        auth_token="local-tok",
    )
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [rec])
    _bust_cache()

    # Seed a different token in the cluster inventory at the same port to
    # confirm we're hitting the local index, not the cluster fallback.
    cii.master_inventory.set_master_state(True)
    cii.master_inventory.merge_servers(
        {
            "x": cii.MasterServerEntry(
                server_id="x",
                base_url="http://127.0.0.1:8137",
                auth_token="WOULD-BE-WRONG",
                label="x",
                peer_node_id=None,
            )
        }
    )
    try:
        assert inference_proxy._token_for("http://127.0.0.1:8137") == "local-tok"
    finally:
        _reset_cluster_inventory()


def test_auth_headers_for_includes_bearer(monkeypatch):
    rec = JobRecord(
        queue_id="q1",
        job_type="inference",
        status="running",
        job_params={"port": 8137, "host": "127.0.0.1"},
        auth_token="tok-abc",
    )
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [rec])
    _bust_cache()
    headers = inference_proxy._auth_headers_for("http://127.0.0.1:8137")
    assert headers == {"authorization": "Bearer tok-abc"}


def test_auth_headers_for_empty_when_no_match(monkeypatch):
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [])
    _bust_cache()
    assert inference_proxy._auth_headers_for("http://127.0.0.1:8137") == {}


# ---------------------------------------------------------------------------
# User-added inference-server registry. ``_auth_headers_for`` falls back to
# the registry when no JobRecord / cluster-inventory entry matches.
# ---------------------------------------------------------------------------


def test_auth_headers_for_falls_back_to_user_registry(monkeypatch):
    """No JobRecord, no cluster entry — registry token should be attached."""
    from forgather_server import inference_server_registry as reg

    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [])
    _bust_cache()
    monkeypatch.setattr(
        reg,
        "list_entries",
        lambda: [
            reg.RegistryEntry(
                id="abcd1234",
                label="vllm box",
                base_url="http://vllm.lan:8000",
                auth_token="user-tok",
                verify_tls=True,
            )
        ],
    )
    headers = inference_proxy._auth_headers_for("http://vllm.lan:8000")
    assert headers == {"authorization": "Bearer user-tok"}


def test_auth_headers_for_jobrecord_wins_over_registry(monkeypatch):
    """JobRecord token takes precedence — it's the freshest source."""
    from forgather_server import inference_server_registry as reg

    rec = JobRecord(
        queue_id="q1",
        job_type="inference",
        status="running",
        job_params={"port": 8137, "host": "127.0.0.1"},
        auth_token="record-tok",
    )
    monkeypatch.setattr(inference_proxy.job_records, "list_records", lambda: [rec])
    _bust_cache()
    monkeypatch.setattr(
        reg,
        "list_entries",
        lambda: [
            reg.RegistryEntry(
                id="abcd1234",
                label="x",
                base_url="http://127.0.0.1:8137",
                auth_token="WOULD-BE-WRONG",
                verify_tls=True,
            )
        ],
    )
    headers = inference_proxy._auth_headers_for("http://127.0.0.1:8137")
    assert headers == {"authorization": "Bearer record-tok"}


def test_verify_for_honors_registry_opt_out(monkeypatch):
    """A registry entry with verify_tls=False forces httpx verify=False."""
    from forgather_server import inference_server_registry as reg

    monkeypatch.setattr(
        reg,
        "list_entries",
        lambda: [
            reg.RegistryEntry(
                id="abcd1234",
                label="ssh-tunneled",
                base_url="https://localhost:9443",
                auth_token="",
                verify_tls=False,
            )
        ],
    )
    assert (
        inference_proxy._verify_for(
            "https://localhost:9443", base="https://localhost:9443"
        )
        is False
    )


def test_registry_add_and_lookup(tmp_path, monkeypatch):
    """End-to-end: add → list → find_token → remove."""
    from forgather_server import inference_server_registry as reg

    target = tmp_path / "inference_server_registry.json"
    monkeypatch.setattr(reg, "inference_server_registry_file", lambda: target)

    assert reg.list_entries() == []
    e = reg.add_entry(label="vllm", base_url="http://vllm.lan:8000/", auth_token="tok")
    # base_url normalized (trailing slash dropped)
    assert e.base_url == "http://vllm.lan:8000"
    assert reg.find_token("http://vllm.lan:8000") == "tok"
    # Removing returns the dropped entry and clears subsequent lookups.
    removed = reg.remove_entry(e.id)
    assert removed is not None and removed.id == e.id
    assert reg.list_entries() == []
    assert reg.find_token("http://vllm.lan:8000") is None


def test_registry_rejects_control_chars_in_token(tmp_path, monkeypatch):
    from forgather_server import inference_server_registry as reg

    target = tmp_path / "inference_server_registry.json"
    monkeypatch.setattr(reg, "inference_server_registry_file", lambda: target)

    with pytest.raises(ValueError):
        reg.add_entry(label="bad", base_url="http://x:8000", auth_token="line1\nline2")
