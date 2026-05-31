"""Cleartext bulk-plane tests for DiLoCo (issue #90).

When ``bulk_cleartext`` is enabled the server offloads the three
heavy-data endpoints (/submit_pseudograd, /submit_fragment_pseudograd,
/global_params) to a second listener bound to a server-picked ephemeral
port. That listener is always cleartext and unauthenticated — its sole
purpose is to bypass TLS for throughput on a trusted LAN. The control
listener keeps the small JSON wire, stays bearer-required, and refuses
the bulk paths with a 404 that carries the bulk URL in
``X-Forgather-Bulk-Url`` (workers learn the ephemeral port over the
TLS-protected control plane).

Tests cover:

* ``get_bulk_url`` shape (None when disabled / before bind, http when
  bound; wildcard-host substitution).
* Control port refuses bulk paths with 404 + ``X-Forgather-Bulk-Url``.
* Bulk port serves bulk paths without a bearer; the control plane still
  requires the bearer.
* /register response carries ``X-Forgather-Bulk-Url`` so workers learn
  the ephemeral port automatically.
* End-to-end via DiLoCoClient: register on control port, then
  submit_pseudogradients routes to the bulk listener.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict():
    torch.manual_seed(0)
    return {"layer.weight": torch.randn(4, 4)}


@pytest.fixture
def two_port_server(tmp_path):
    """Server with the cleartext bulk plane enabled (bearer on control,
    no auth on bulk — the only bulk posture there is). The bulk listener
    binds an OS-assigned ephemeral port, readable post-start via
    ``server.bulk_port``."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
        auth_token="control-bearer",
        bulk_cleartext=True,
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


# ---------------------------------------------------------------------------
# get_bulk_url
# ---------------------------------------------------------------------------


def test_get_bulk_url_none_when_disabled(tmp_path):
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
    )
    assert s.get_bulk_url() is None


def test_get_bulk_url_none_before_bind(tmp_path):
    """Enabled but not yet started: the ephemeral port isn't bound, so
    there's nothing to advertise."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
        bulk_cleartext=True,
    )
    assert s.get_bulk_url() is None


def test_get_bulk_url_shape_when_bound(two_port_server):
    """Once bound, get_bulk_url returns http://host:<ephemeral port>.
    The bulk plane is always cleartext, so the scheme is always http."""
    url = two_port_server.get_bulk_url()
    assert url is not None
    assert url.startswith("http://")
    assert two_port_server.bulk_port is not None
    assert f":{two_port_server.bulk_port}" in url


def test_get_bulk_url_wildcard_host_uses_request_host(tmp_path):
    """A 0.0.0.0-bound server must NOT advertise http://0.0.0.0:<port>
    (unroutable from a remote worker). When the registering worker's
    request host is known, the advertised bulk URL uses it; otherwise
    it falls back to loopback. Regression test for the post-#90
    cross-host bulk-plane break. We set ``bulk_port`` directly to
    simulate a bound listener so the host-substitution logic is tested
    in isolation (no real socket)."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        host="0.0.0.0",
        bulk_cleartext=True,
        heartbeat_timeout=0,
    )
    s.bulk_port = 9999  # simulate a bound ephemeral listener
    # Without a request host, never leak the wildcard bind address.
    assert s.get_bulk_url() == "http://127.0.0.1:9999"
    # With the worker's view of our host, advertise that (routable).
    assert s.get_bulk_url("192.168.9.43") == "http://192.168.9.43:9999"


def test_get_bulk_url_explicit_host_preserved(tmp_path):
    """An explicit (non-wildcard) bind host is advertised verbatim even
    when a request host is available — the operator chose it."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        host="10.0.0.5",
        bulk_cleartext=True,
        heartbeat_timeout=0,
    )
    s.bulk_port = 9999  # simulate a bound ephemeral listener
    assert s.get_bulk_url("192.168.9.43") == "http://10.0.0.5:9999"


# ---------------------------------------------------------------------------
# Control port refuses bulk paths with hint header
# ---------------------------------------------------------------------------


def test_control_port_refuses_bulk_path_with_hint(two_port_server):
    """Bulk path on the control port returns 404 with the bulk URL in
    ``X-Forgather-Bulk-Url`` so misrouted clients can self-correct."""
    url = f"http://localhost:{two_port_server.port}/global_params"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": "Bearer control-bearer"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 404
    assert exc_info.value.headers.get("X-Forgather-Bulk-Url") == (
        two_port_server.get_bulk_url()
    )
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert body["bulk_url"] == two_port_server.get_bulk_url()


# ---------------------------------------------------------------------------
# Bulk port served without auth; control stays bearer-required
# ---------------------------------------------------------------------------


def test_bulk_port_serves_global_params_without_bearer(two_port_server):
    """The cleartext bulk listener accepts /global_params without any
    Authorization header."""
    url = f"http://localhost:{two_port_server.bulk_port}/global_params"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        body = resp.read()
        assert len(body) > 0


def test_bulk_port_refuses_non_bulk_path(two_port_server):
    """The bulk listener only serves bulk endpoints. /status on the
    bulk port is 404 (no way to control the server through the bulk
    listener even though auth is off)."""
    url = f"http://localhost:{two_port_server.bulk_port}/status"
    req = urllib.request.Request(url, method="GET")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 404


def test_control_port_still_requires_bearer(two_port_server):
    """The control port keeps its bearer-required posture even though
    the bulk port is wide open. The whole point of the split is to
    let bulk go fast WITHOUT relaxing control."""
    url = f"http://localhost:{two_port_server.port}/status"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url, timeout=5)
    assert exc_info.value.code == 401


# ---------------------------------------------------------------------------
# /register advertises the bulk URL
# ---------------------------------------------------------------------------


def test_register_response_advertises_bulk_url(two_port_server):
    """/register response carries ``X-Forgather-Bulk-Url`` so workers
    learn the ephemeral port on first contact without an extra
    round-trip."""
    body = json.dumps(
        {
            "worker_id": "alpha",
            "hostname": "test",
            "param_shapes": {"layer.weight": [4, 4]},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{two_port_server.port}/register",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer control-bearer",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert (
            resp.headers.get("X-Forgather-Bulk-Url") == two_port_server.get_bulk_url()
        )


# ---------------------------------------------------------------------------
# Client routes bulk endpoints to the bulk listener
# ---------------------------------------------------------------------------


def test_bulk_port_ignores_stray_bearer(two_port_server):
    """The cleartext bulk listener short-circuits the bearer check
    (returns True without inspecting headers). A worker that defensively
    sends Authorization to the bulk port must still be served — we don't
    reject legitimate requests just because they're over-credentialed."""
    url = f"http://localhost:{two_port_server.bulk_port}/global_params"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": "Bearer some-random-token"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200


def test_client_routes_bulk_to_bulk_listener(two_port_server):
    """End-to-end: client constructed with the control URL learns
    the bulk URL from /register's response header and routes
    /global_params there automatically."""
    client = DiLoCoClient(
        f"http://localhost:{two_port_server.port}",
        token="control-bearer",
        timeout=5,
        max_retries=0,
    )
    assert client.bulk_url is None
    client.register("alpha", worker_info={"param_shapes": {"layer.weight": [4, 4]}})
    assert client.bulk_url == two_port_server.get_bulk_url()
    # _url for a bulk path should now use the bulk URL.
    assert client._url("/global_params").startswith(client.bulk_url)
    # And control endpoints stay on the control URL.
    assert client._url("/status").startswith(client.server_addr)
