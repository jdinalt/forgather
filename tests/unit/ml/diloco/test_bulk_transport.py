"""Tests for the pluggable bulk byte transport (issue #154).

The client delegates the three bulk legs to a ``BulkBytesTransport`` — the seam a
future gRPC transport swaps into. These tests pin that contract: the client
frames + deserializes, the transport only moves bytes, and the op/verb mapping is
correct. They use a recording fake (no network), plus a localhost round-trip for
the default ``HttpBytesTransport``.
"""

import json
import struct

import pytest
import torch

from forgather.ml.diloco.bulk_transport import PATH_TO_OP, BulkOp, HttpBytesTransport
from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.wire_serialize import (
    deserialize_state_dict,
    serialize_state_dict,
)


class _RecordingTransport:
    """A BulkBytesTransport that records calls and returns a canned response."""

    def __init__(self, response_sd, wire_format):
        self._response = serialize_state_dict(response_sd, wire_format)
        self.calls = []

    def round_trip(self, op, payload=None, *, retries=0):
        self.calls.append((op, payload, retries))
        return self._response

    def close(self):
        self.closed = True


def _client_with_fake(transport):
    # Build a client without touching the network, then swap its transport.
    client = DiLoCoClient("http://localhost:1", wire_format="safetensors")
    client._transport = transport
    return client


def test_submit_pseudogradients_delegates_and_frames():
    """submit_pseudogradients frames a [len][header][payload] body, hands it to
    the transport as SUBMIT_PSEUDOGRAD, and deserializes the response."""
    sd = {"w": torch.randn(3, 3)}
    resp = {"w": torch.randn(3, 3)}
    fake = _RecordingTransport(resp, "safetensors")
    client = _client_with_fake(fake)

    out = client.submit_pseudogradients("worker_0", sd)

    assert len(fake.calls) == 1
    op, payload, _ = fake.calls[0]
    assert op is BulkOp.SUBMIT_PSEUDOGRAD
    # The payload is the framed request: [4-byte len][json header][st bytes].
    header_len = struct.unpack("!I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_len])
    assert header == {"worker_id": "worker_0", "fmt": "safetensors"}
    body = payload[4 + header_len :]
    assert deserialize_state_dict(body, "safetensors")["w"].equal(sd["w"])
    # The response was deserialized with the client's negotiated wire_format.
    assert out["w"].equal(resp["w"])


def test_global_params_is_bodyless():
    sd = {"w": torch.randn(2)}
    fake = _RecordingTransport(sd, "safetensors")
    client = _client_with_fake(fake)

    client.get_global_params()

    op, payload, _ = fake.calls[0]
    assert op is BulkOp.GLOBAL_PARAMS
    assert payload is None  # GET, no body


def test_fragment_op_mapping():
    sd = {"w": torch.randn(2)}
    fake = _RecordingTransport(sd, "safetensors")
    client = _client_with_fake(fake)

    client.submit_fragment_pseudogradients("worker_0", 1, sd)

    op, payload, _ = fake.calls[0]
    assert op is BulkOp.SUBMIT_FRAGMENT
    header_len = struct.unpack("!I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_len])
    assert header["fragment_id"] == 1


def test_client_close_delegates_to_transport():
    """DiLoCoClient.close() releases the bulk transport (closes a gRPC channel;
    no-op for HTTP) — the path DiLoCoWorker.stop() now drives at teardown."""
    fake = _RecordingTransport({"w": torch.randn(2)}, "safetensors")
    fake.closed = False
    client = _client_with_fake(fake)
    client.close()
    assert fake.closed is True


def test_path_to_op_covers_all_bulk_paths():
    assert PATH_TO_OP["/submit_pseudograd"] is BulkOp.SUBMIT_PSEUDOGRAD
    assert PATH_TO_OP["/submit_fragment_pseudograd"] is BulkOp.SUBMIT_FRAGMENT
    assert PATH_TO_OP["/global_params"] is BulkOp.GLOBAL_PARAMS


def _http_transport(**over):
    kw = dict(
        url_for=lambda path: f"http://localhost/{path.lstrip('/')}",
        headers_for=lambda content_type=None, *, path=None: {},
        ssl_for=lambda u: None,
        timeout=1,
        retry_delay=0,
        scheme_hint=lambda: "",
    )
    kw.update(over)
    return HttpBytesTransport(**kw)


def test_close_is_callable():
    # close() is a no-op for urllib but exists as the channel-teardown hook a
    # gRPC transport will implement; it must be callable and return None.
    assert _http_transport().close() is None


def test_retry_path_recovers_after_transient_failures():
    """round_trip retries URLError up to ``retries`` times with backoff, then
    succeeds — the fault-tolerant reconnection path PR3's transport relies on."""
    import urllib.error
    import urllib.request

    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"OK"

    def flaky_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("connection refused")
        return _Resp()

    transport = _http_transport()  # retry_delay=0 keeps it fast
    orig = urllib.request.urlopen
    urllib.request.urlopen = flaky_urlopen
    try:
        out = transport.round_trip(BulkOp.GLOBAL_PARAMS, retries=2)
        assert out == b"OK"
        assert calls["n"] == 3  # failed twice, succeeded on the third
    finally:
        urllib.request.urlopen = orig


def test_retry_exhausted_raises_connection_error():
    import urllib.error
    import urllib.request

    def always_fails(req, timeout=None, context=None):
        raise urllib.error.URLError("down")

    transport = _http_transport()
    orig = urllib.request.urlopen
    urllib.request.urlopen = always_fails
    try:
        with pytest.raises(ConnectionError, match="Failed to connect"):
            transport.round_trip(BulkOp.SUBMIT_PSEUDOGRAD, b"x", retries=1)
    finally:
        urllib.request.urlopen = orig


def test_http_transport_op_verb_mapping():
    """HttpBytesTransport derives POST (with body) for the submit legs and GET
    (bodyless) for global_params, recording what urllib would send."""
    sent = {}

    def fake_url(path):
        return f"http://localhost/{path.lstrip('/')}"

    def fake_headers(content_type=None, *, path=None):
        return {"Content-Type": content_type} if content_type else {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"OK"

    def fake_urlopen(req, timeout=None, context=None):
        sent["method"] = req.get_method()
        sent["has_body"] = req.data is not None
        return _Resp()

    transport = HttpBytesTransport(
        url_for=fake_url,
        headers_for=fake_headers,
        ssl_for=lambda u: None,
        timeout=1,
        retry_delay=0,
        scheme_hint=lambda: "",
    )

    import urllib.request

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        assert transport.round_trip(BulkOp.SUBMIT_PSEUDOGRAD, b"payload") == b"OK"
        assert sent == {"method": "POST", "has_body": True}
        transport.round_trip(BulkOp.GLOBAL_PARAMS)
        assert sent == {"method": "GET", "has_body": False}
    finally:
        urllib.request.urlopen = orig
