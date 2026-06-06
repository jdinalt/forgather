"""The Anthropic adapter must send the right auth header per target.

vLLM's Anthropic Messages surface checks ``Authorization: Bearer`` (the
SDK's ``auth_token``), while real Claude uses the ``x-api-key`` header (the
SDK's ``api_key``). Sending the wrong one 401s. These tests pin the
selection. Skipped if the ``anthropic`` package isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from forgather_server.agent.providers.anthropic import AnthropicProvider


def test_auth_token_used_for_local_server():
    p = AnthropicProvider(model="qwen", auth_token="tok123", base_url="https://kitt:8000")
    client = p._ensure_client()
    assert client.auth_token == "tok123"
    assert not client.api_key  # None/empty — x-api-key not used


def test_api_key_used_for_claude():
    p = AnthropicProvider(model="claude-x", api_key="sk-ant-xxx")
    client = p._ensure_client()
    assert client.api_key == "sk-ant-xxx"
    assert not client.auth_token


def test_placeholder_when_no_credential():
    # vLLM --no-auth ignores the credential, but the SDK still requires one.
    p = AnthropicProvider(model="qwen", base_url="http://localhost:8000")
    client = p._ensure_client()
    assert client.api_key == "placeholder"
