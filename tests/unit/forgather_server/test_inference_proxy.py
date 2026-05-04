"""Tests for forgather_server.routes.inference_proxy SSRF guard.

Covers ``_validate_base`` directly: scheme allow-list, default localhost-
only host policy, and the ``FORGATHER_INFERENCE_PROXY_ALLOW_REMOTE`` opt-
in escape hatch.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException
from forgather_server.routes import inference_proxy


# Localhost variants we accept by default.
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8137",
        "http://localhost:8137",
        "https://localhost",
        "http://[::1]:8137",
        "http://LOCALHOST:8137",  # case-insensitive
    ],
)
def test_validate_base_accepts_localhost(url, monkeypatch):
    monkeypatch.delenv(inference_proxy._REMOTE_ALLOW_ENV, raising=False)
    assert inference_proxy._validate_base(url) == url.rstrip("/")


# Non-localhost hosts must be refused with 403 by default.
@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.5:8137",
        "http://169.254.169.254",  # cloud metadata service
        "http://10.0.0.1",
        "http://example.com",
    ],
)
def test_validate_base_rejects_remote_by_default(url, monkeypatch):
    monkeypatch.delenv(inference_proxy._REMOTE_ALLOW_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        inference_proxy._validate_base(url)
    assert ei.value.status_code == 403
    assert "non-localhost" in ei.value.detail
    assert inference_proxy._REMOTE_ALLOW_ENV in ei.value.detail


def test_validate_base_allows_remote_when_opted_in(monkeypatch, caplog):
    monkeypatch.setenv(inference_proxy._REMOTE_ALLOW_ENV, "1")
    url = "http://192.168.1.5:8137"
    with caplog.at_level(logging.WARNING, logger="forgather_server.inference_proxy"):
        out = inference_proxy._validate_base(url)
    assert out == url
    # The host (without scheme/port) should appear in the warning so an
    # operator scanning logs sees what was let through.
    assert any(
        "192.168.1.5" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "TRUE", "Yes"])
def test_remote_allowed_truthy_values(monkeypatch, truthy):
    monkeypatch.setenv(inference_proxy._REMOTE_ALLOW_ENV, truthy)
    assert inference_proxy._remote_allowed() is True


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "maybe"])
def test_remote_allowed_falsy_values(monkeypatch, falsy):
    monkeypatch.setenv(inference_proxy._REMOTE_ALLOW_ENV, falsy)
    assert inference_proxy._remote_allowed() is False


def test_validate_base_rejects_non_http_scheme(monkeypatch):
    monkeypatch.delenv(inference_proxy._REMOTE_ALLOW_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        inference_proxy._validate_base("file:///etc/passwd")
    assert ei.value.status_code == 400
    assert "scheme" in ei.value.detail


def test_validate_base_rejects_non_http_scheme_even_when_remote_allowed(monkeypatch):
    # Opt-in must not weaken the scheme guard.
    monkeypatch.setenv(inference_proxy._REMOTE_ALLOW_ENV, "1")
    with pytest.raises(HTTPException) as ei:
        inference_proxy._validate_base("gopher://127.0.0.1/")
    assert ei.value.status_code == 400


def test_validate_base_strips_trailing_slash(monkeypatch):
    monkeypatch.delenv(inference_proxy._REMOTE_ALLOW_ENV, raising=False)
    assert (
        inference_proxy._validate_base("http://127.0.0.1:8137/")
        == "http://127.0.0.1:8137"
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
