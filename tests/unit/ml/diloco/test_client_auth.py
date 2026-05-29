"""End-to-end client/server bearer-auth tests for DiLoCo (issue #90).

Spins up a real ``DiLoCoServer`` with a fixed bearer, then drives it
with ``DiLoCoClient`` to confirm:

* A client constructed with the matching token can call ``/status``
  successfully.
* A client constructed without a token (and with token-discovery
  disabled by pointing at a non-loopback URL) gets a clean
  ``ConnectionError`` carrying the server's 401.
* Token discovery via the loopback per-port file works end-to-end —
  a client constructed without ``token=`` against a loopback URL
  picks the token up automatically.
* Token discovery via the ``FORGATHER_DILOCO_SERVER_TOKEN`` env var
  also works.
"""

from __future__ import annotations

import time

import pytest
import torch

from forgather.ml.diloco.auth import write_standalone_token
from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict():
    torch.manual_seed(0)
    return {
        "embedding.weight": torch.randn(8, 4),
        "layer.weight": torch.randn(4, 4),
    }


@pytest.fixture
def authed_server(tmp_path):
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
        auth_token="bearer-from-test",
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


def test_client_with_matching_token_passes(authed_server):
    """Client gets 200 from /status when carrying the right bearer."""
    client = DiLoCoClient(
        f"http://localhost:{authed_server.port}",
        token="bearer-from-test",
        timeout=5,
        max_retries=0,
    )
    status = client.get_status()
    assert "status" in status


def test_client_without_token_fails_401(authed_server, tmp_path, monkeypatch):
    """Client with no token configured (and no auto-discovery source)
    surfaces the server's 401 as a ConnectionError."""
    # Empty XDG_CONFIG_HOME → no per-port file. No env var either.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("FORGATHER_DILOCO_SERVER_TOKEN", raising=False)
    client = DiLoCoClient(
        f"http://localhost:{authed_server.port}",
        timeout=5,
        max_retries=0,
    )
    assert client.token is None
    with pytest.raises(ConnectionError) as exc_info:
        client.get_status()
    assert "401" in str(exc_info.value)


def test_loopback_token_autodiscovery_e2e(authed_server, tmp_path, monkeypatch):
    """A client that doesn't supply a token but points at a loopback
    URL picks up the token from the per-port file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("FORGATHER_DILOCO_SERVER_TOKEN", raising=False)
    # Write the right token to the per-port file the client will read.
    write_standalone_token(authed_server.port, "bearer-from-test")

    client = DiLoCoClient(
        f"http://localhost:{authed_server.port}",
        timeout=5,
        max_retries=0,
    )
    assert client.token == "bearer-from-test"
    status = client.get_status()
    assert "status" in status


def test_env_var_token_picked_up(authed_server, tmp_path, monkeypatch):
    """``FORGATHER_DILOCO_SERVER_TOKEN`` is consulted when no explicit
    token is passed. Wins over the loopback file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Make sure no per-port file exists for this port.
    monkeypatch.setenv("FORGATHER_DILOCO_SERVER_TOKEN", "bearer-from-test")

    client = DiLoCoClient(
        f"http://localhost:{authed_server.port}",
        timeout=5,
        max_retries=0,
    )
    assert client.token == "bearer-from-test"
    status = client.get_status()
    assert "status" in status


def test_bare_host_picks_scheme_from_tls_config(monkeypatch):
    """A bare ``host:port`` (no scheme) should pick https vs http via
    the same ``forgather.tls.client_scheme()`` the scheduler uses to
    stamp the JobRecord. Without this, the worker dialed http://
    against a TLS-wrapped server and the connection was reset.
    Regression test for the post-#90 bug."""
    # Stub client_scheme to return https — the cluster-provisioned case.
    import forgather.tls
    from forgather.ml.diloco.client import DiLoCoClient

    monkeypatch.setattr(forgather.tls, "client_scheme", lambda *a, **k: "https")
    c = DiLoCoClient("192.168.9.43:8512", timeout=5, max_retries=0)
    assert c.server_addr == "https://192.168.9.43:8512"

    # And the other way — TLS not provisioned → http.
    monkeypatch.setattr(forgather.tls, "client_scheme", lambda *a, **k: "http")
    c2 = DiLoCoClient("192.168.9.43:8512", timeout=5, max_retries=0)
    assert c2.server_addr == "http://192.168.9.43:8512"


def test_guessed_scheme_surfaces_hint_on_connection_error(monkeypatch):
    """A bare host:port worker that can't connect gets a connection
    error that NAMES the inferred-scheme ambiguity, instead of a bare
    connection reset with no explanation (fail-loud, post-#90 finding)."""
    import forgather.tls
    from forgather.ml.diloco.client import DiLoCoClient

    monkeypatch.setattr(forgather.tls, "client_scheme", lambda *a, **k: "http")
    # Port 9 (discard) refuses / drops — connection fails fast.
    c = DiLoCoClient("127.0.0.1:9", timeout=1, max_retries=0)
    assert c._scheme_guessed is True
    with pytest.raises(ConnectionError) as exc:
        c.get_status()
    msg = str(exc.value)
    assert "inferred" in msg and "https://" in msg


def test_explicit_scheme_no_hint(monkeypatch):
    """An explicit-scheme URL does not emit the inferred-scheme hint."""
    import forgather.tls
    from forgather.ml.diloco.client import DiLoCoClient

    monkeypatch.setattr(forgather.tls, "client_scheme", lambda *a, **k: "http")
    c = DiLoCoClient("http://127.0.0.1:9", timeout=1, max_retries=0)
    assert c._scheme_guessed is False
    with pytest.raises(ConnectionError) as exc:
        c.get_status()
    assert "inferred" not in str(exc.value)


def test_explicit_scheme_is_preserved(monkeypatch):
    """An explicit ``http://`` or ``https://`` URL passes through
    unchanged — overrides whatever client_scheme would have picked."""
    import forgather.tls
    from forgather.ml.diloco.client import DiLoCoClient

    monkeypatch.setattr(forgather.tls, "client_scheme", lambda *a, **k: "https")
    # Explicit http:// keeps http:// even when TLS is locally provisioned.
    c = DiLoCoClient("http://10.0.0.5:8512", timeout=5, max_retries=0)
    assert c.server_addr == "http://10.0.0.5:8512"
    # Explicit https:// keeps https://.
    c2 = DiLoCoClient("https://10.0.0.5:8512", timeout=5, max_retries=0)
    assert c2.server_addr == "https://10.0.0.5:8512"


def test_no_token_against_no_auth_server_works(tmp_path):
    """A server with auth disabled lets unauthenticated clients in —
    end-to-end coverage of the backwards-compat surface."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
    )
    s.start()
    time.sleep(0.2)
    try:
        client = DiLoCoClient(
            f"http://localhost:{s.port}",
            timeout=5,
            max_retries=0,
        )
        # No token configured, server doesn't require one.
        status = client.get_status()
        assert "status" in status
    finally:
        s.stop()
