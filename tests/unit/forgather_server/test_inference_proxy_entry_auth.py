"""Entry-bound auth for the inference proxy (issue #158).

When the webui names the selected registry entry by id
(X-Inference-Server-Id), the proxy must attach *that* entry's token — or
nothing if it has none — and must never substitute another entry's token
for the same base_url. This is what lets two entries share a URL with
independent auth and keeps the "No auth" label honest.
"""

from __future__ import annotations

import types

import pytest

from forgather_server import inference_server_registry as isr
from forgather_server.routes import inference_proxy as ip


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(isr, "inference_server_registry_file", lambda: tmp_path / "inf.json")
    return isr


def _req(headers):
    return types.SimpleNamespace(headers=headers)


def test_entry_id_attaches_that_entrys_token(registry):
    authed = registry.add_entry(label="authed", base_url="https://h:8000/v1", auth_token="TOK")
    h = ip._auth_headers_for("https://h:8000/v1", _req({ip._SERVER_ID_HEADER: authed.id}))
    assert h == {"authorization": "Bearer TOK"}


def test_no_token_entry_does_not_inherit_sibling_token(registry):
    # Two entries, same URL: one authed, one not. Selecting the no-auth one
    # must send NO token (the bug: find_token(url) would return the sibling's).
    registry.add_entry(label="authed", base_url="https://h:8000/v1", auth_token="TOK")
    noauth = registry.add_entry(label="noauth", base_url="https://h:8000/v1", auth_token="")
    h = ip._auth_headers_for("https://h:8000/v1", _req({ip._SERVER_ID_HEADER: noauth.id}))
    assert h == {}


def test_unknown_entry_id_sends_nothing(registry):
    registry.add_entry(label="authed", base_url="https://h:8000/v1", auth_token="TOK")
    h = ip._auth_headers_for("https://h:8000/v1", _req({ip._SERVER_ID_HEADER: "deadbeef"}))
    assert h == {}


def test_explicit_token_override_still_wins(registry):
    authed = registry.add_entry(label="authed", base_url="https://h:8000/v1", auth_token="TOK")
    h = ip._auth_headers_for(
        "https://h:8000/v1",
        _req({ip._TOKEN_OVERRIDE_HEADER: "OVERRIDE", ip._SERVER_ID_HEADER: authed.id}),
    )
    assert h == {"authorization": "Bearer OVERRIDE"}


def test_find_by_id(registry):
    e = registry.add_entry(label="x", base_url="https://h:8000/v1", auth_token="T")
    assert registry.find_by_id(e.id).auth_token == "T"
    assert registry.find_by_id("nope") is None
