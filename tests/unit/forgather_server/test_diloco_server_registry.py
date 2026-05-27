"""Unit tests for the user-defined DiLoCo server registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from forgather_server import diloco_server_registry as reg


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    """Redirect the registry file to a per-test tmp path."""
    target = tmp_path / "diloco_server_registry.json"
    monkeypatch.setattr(reg, "diloco_server_registry_file", lambda: target)
    return target


def test_list_is_empty_when_no_file(registry_path):
    assert reg.list_entries() == []


def test_add_and_list_roundtrip(registry_path):
    e = reg.add_entry(label="WAN box", base_url="http://10.0.0.1:8512")
    assert e.label == "WAN box"
    assert e.base_url == "http://10.0.0.1:8512"
    assert e.auth_token == ""
    assert e.verify_tls is True
    assert len(e.id) == 8
    entries = reg.list_entries()
    assert len(entries) == 1
    assert entries[0].id == e.id


def test_base_url_is_normalized(registry_path):
    e = reg.add_entry(label="x", base_url="http://h:8512/")
    assert e.base_url == "http://h:8512"


def test_empty_base_url_rejected(registry_path):
    with pytest.raises(ValueError, match="base_url is required"):
        reg.add_entry(label="x", base_url="")


def test_label_defaults_to_url(registry_path):
    e = reg.add_entry(label="", base_url="http://x:1")
    assert e.label == "http://x:1"


def test_control_char_in_token_rejected(registry_path):
    with pytest.raises(ValueError, match="control characters"):
        reg.add_entry(label="x", base_url="http://x:1", auth_token="abc\ndef")


def test_remove_existing(registry_path):
    e = reg.add_entry(label="x", base_url="http://x:1")
    removed = reg.remove_entry(e.id)
    assert removed is not None
    assert removed.id == e.id
    assert reg.list_entries() == []


def test_remove_missing_returns_none(registry_path):
    assert reg.remove_entry("nope") is None


def test_find_token_returns_stored(registry_path):
    reg.add_entry(label="x", base_url="http://x:1", auth_token="abc")
    assert reg.find_token("http://x:1") == "abc"


def test_find_token_strips_trailing_slash(registry_path):
    reg.add_entry(label="x", base_url="http://x:1", auth_token="abc")
    assert reg.find_token("http://x:1/") == "abc"


def test_find_token_returns_none_for_unknown(registry_path):
    reg.add_entry(label="x", base_url="http://x:1", auth_token="abc")
    assert reg.find_token("http://y:2") is None


def test_find_token_returns_none_for_empty_token(registry_path):
    reg.add_entry(label="x", base_url="http://x:1", auth_token="")
    assert reg.find_token("http://x:1") is None


def test_find_verify_tls_defaults_true_for_unknown(registry_path):
    assert reg.find_verify_tls("http://nowhere:1") is True


def test_find_verify_tls_respects_entry(registry_path):
    reg.add_entry(label="x", base_url="http://x:1", verify_tls=False)
    assert reg.find_verify_tls("http://x:1") is False


def test_file_mode_is_0600(registry_path):
    reg.add_entry(label="x", base_url="http://x:1", auth_token="t")
    mode = registry_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_corrupt_json_returns_empty(registry_path):
    registry_path.write_text("{ not json")
    assert reg.list_entries() == []
