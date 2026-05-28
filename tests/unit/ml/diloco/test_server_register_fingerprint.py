"""Tests for the DiLoCoServer model-fingerprint check on /register.

Workers send a structural fingerprint (``{name: shape}`` for every
named parameter) on /register. The server compares it against its own
``_param_names`` + ``_param_list`` shapes and 422s on mismatch with a
diagnostic identifying the divergent params. This catches the
"operator pointed this worker at the wrong --model-id-or-path" class
of misconfiguration at register time, instead of letting it surface
hundreds of steps later in the first sync's optimizer step.

See task #51 / DiLoCoModelMismatchError.
"""

import json
import time
import urllib.error
import urllib.request

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient, DiLoCoModelMismatchError
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict():
    torch.manual_seed(42)
    return {
        "embedding.weight": torch.randn(8, 4),
        "layer.weight": torch.randn(4, 4),
    }


@pytest.fixture
def server(tmp_path):
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=2,
        port=0,
        heartbeat_timeout=0,
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


def _raw_register(server, worker_id, param_shapes):
    """Hit /register directly with a chosen param_shapes payload so
    we can read the HTTP status on 422 (the high-level client wraps
    422 as DiLoCoModelMismatchError which loses the raw status)."""
    body = {
        "worker_id": worker_id,
        "hostname": "test",
        "param_shapes": param_shapes,
    }
    req = urllib.request.Request(
        f"http://localhost:{server.port}/register",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=5)


def _matching_shapes():
    return {"embedding.weight": [8, 4], "layer.weight": [4, 4]}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_matching_fingerprint_registers_cleanly(server):
    """Worker shapes match server → /register returns 200 with the
    global params, just like a worker that didn't send a fingerprint."""
    resp = _raw_register(server, "alpha", _matching_shapes())
    assert resp.status == 200


def test_pre_fingerprint_worker_still_works(server):
    """A worker that omits ``param_shapes`` (pre-#51) registers
    without the fingerprint check — backward compat with existing
    deployments. They still hit the (less helpful) shape-mismatch
    crash at first sync if the model is wrong, but the registration
    itself succeeds."""
    body = {"worker_id": "alpha", "hostname": "test"}  # no param_shapes
    req = urllib.request.Request(
        f"http://localhost:{server.port}/register",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=5)
    assert resp.status == 200


# ---------------------------------------------------------------------------
# Mismatch cases
# ---------------------------------------------------------------------------


def test_shape_mismatch_returns_422(server):
    """One param has the wrong shape → 422 with a diagnostic that
    names the divergent param and shows both shapes."""
    bad = _matching_shapes()
    bad["layer.weight"] = [4, 8]  # server has [4, 4]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_register(server, "alpha", bad)
    assert exc_info.value.code == 422
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert body["kind"] == "slice_mismatch"
    assert "layer.weight" in body["error"]
    assert "[4, 8]" in body["error"]
    assert "[4, 4]" in body["error"]


def test_missing_param_on_server_returns_422(server):
    """Worker sends a param the server doesn't have → 422."""
    bad = _matching_shapes()
    bad["extra.weight"] = [2, 2]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_register(server, "alpha", bad)
    assert exc_info.value.code == 422
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "extra.weight" in body["error"]


def test_missing_param_on_worker_returns_422(server):
    """Worker omits a param the server has → 422 (covers the
    'worker has buffers where the server has params' case too)."""
    bad = _matching_shapes()
    del bad["embedding.weight"]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_register(server, "alpha", bad)
    assert exc_info.value.code == 422
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "embedding.weight" in body["error"]


def test_fingerprint_checked_before_worker_id_uniqueness(server):
    """The fingerprint check runs before the worker_id uniqueness
    check. Operator gets the more diagnostic-helpful error first;
    fixing the model path fixes the registration path on retry."""
    # alpha registers cleanly.
    _raw_register(server, "alpha", _matching_shapes())
    # Second registration with same worker_id AND wrong shapes:
    # fingerprint mismatch wins.
    bad = _matching_shapes()
    bad["layer.weight"] = [4, 8]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_register(server, "alpha", bad)
    assert exc_info.value.code == 422


# ---------------------------------------------------------------------------
# High-level client path
# ---------------------------------------------------------------------------


def test_high_level_client_raises_DiLoCoModelMismatchError(server):
    """DiLoCoClient.register() wraps 422 into the typed exception
    so the callback can branch on it cleanly without parsing the
    error message."""
    # Inject a wrong-shape fingerprint by patching the client's
    # register body — easier than constructing a fake nn.Module.
    client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
    bad_shapes = _matching_shapes()
    bad_shapes["layer.weight"] = [4, 8]
    with pytest.raises(DiLoCoModelMismatchError) as exc_info:
        client.register("alpha", worker_info={"param_shapes": bad_shapes})
    assert "layer.weight" in exc_info.value.diagnostic
    assert "[4, 8]" in exc_info.value.diagnostic
