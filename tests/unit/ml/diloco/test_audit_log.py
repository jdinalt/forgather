"""Audit-log tests for the DiLoCo parameter server (issue #90).

The server appends one JSON record per line to
``<output_dir>/diloco_audit.log`` for events worth reconstructing
after the fact: worker registration, deregistration, eviction,
outer-optimizer steps, and control actions. The log is best-effort
(write failures don't crash the operation) and intentionally never
records bearer tokens or other secrets.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict():
    torch.manual_seed(0)
    return {"layer.weight": torch.randn(4, 4)}


@pytest.fixture
def server(tmp_path):
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
    yield s, tmp_path
    s.stop()


def _read_audit(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Per-event records
# ---------------------------------------------------------------------------


def test_register_emits_audit_record(server):
    s, root = server
    client = DiLoCoClient(f"http://localhost:{s.port}", timeout=5, max_retries=0)
    client.register("alpha", worker_info={"param_shapes": {"layer.weight": [4, 4]}})

    log_path = root / "diloco_audit.log"
    assert log_path.exists()
    records = _read_audit(log_path)
    register_records = [r for r in records if r["event"] == "register"]
    assert len(register_records) == 1
    rec = register_records[0]
    assert rec["worker_id"] == "alpha"
    assert rec["num_registered"] == 1
    assert "ts" in rec


def test_deregister_emits_audit_record(server):
    s, root = server
    client = DiLoCoClient(f"http://localhost:{s.port}", timeout=5, max_retries=0)
    client.register("alpha", worker_info={"param_shapes": {"layer.weight": [4, 4]}})
    client.deregister("alpha")

    records = _read_audit(root / "diloco_audit.log")
    kinds = [r["event"] for r in records]
    assert "deregister" in kinds
    # Deregister triggers eviction internally; both records are present.
    assert "eviction" in kinds
    dereg = [r for r in records if r["event"] == "deregister"][0]
    assert dereg["worker_id"] == "alpha"


def test_control_action_audited(server):
    s, root = server
    # Hit /control/save_state directly via urllib (the DiLoCoClient
    # control surface isn't relevant to this test).
    body = b"{}"
    req = urllib.request.Request(
        f"http://localhost:{s.port}/control/save_state",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    # No auth_token configured on this server fixture, so no bearer
    # needed. The control action may fail (no save target set yet)
    # but the audit record fires before the body is executed.
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

    records = _read_audit(root / "diloco_audit.log")
    control_records = [r for r in records if r["event"] == "control"]
    assert len(control_records) == 1
    assert control_records[0]["action"] == "save_state"


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------


def test_records_have_iso_timestamp(server):
    s, root = server
    client = DiLoCoClient(f"http://localhost:{s.port}", timeout=5, max_retries=0)
    client.register("alpha", worker_info={"param_shapes": {"layer.weight": [4, 4]}})

    records = _read_audit(root / "diloco_audit.log")
    for r in records:
        assert "ts" in r
        # ISO 8601 with +00:00 UTC offset.
        assert "T" in r["ts"]
        assert r["ts"].endswith("+00:00")


def test_token_is_never_logged(server):
    """Belt-and-suspenders: even if a future change accidentally pulls
    bearer tokens into the audit-record fields, ``"token"`` must never
    appear anywhere in the log. This is a regression guard."""
    s, root = server
    client = DiLoCoClient(f"http://localhost:{s.port}", timeout=5, max_retries=0)
    client.register("alpha", worker_info={"param_shapes": {"layer.weight": [4, 4]}})

    text = (root / "diloco_audit.log").read_text(encoding="utf-8").lower()
    assert "bearer" not in text
    assert "auth_token" not in text


# ---------------------------------------------------------------------------
# Best-effort write
# ---------------------------------------------------------------------------


def test_audit_write_failure_does_not_crash_request(server, monkeypatch):
    """When the audit log path becomes unwritable mid-run, requests
    keep succeeding — the audit log is a record, not a guard."""
    s, root = server

    # Simulate a write failure by pointing the audit path at a
    # directory that doesn't exist.
    s._audit_path = "/nonexistent/dir/diloco_audit.log"

    client = DiLoCoClient(f"http://localhost:{s.port}", timeout=5, max_retries=0)
    # This should NOT raise — the registration succeeds and only the
    # audit-record write is dropped.
    client.register("alpha", worker_info={"param_shapes": {"layer.weight": [4, 4]}})


def test_no_output_dir_silently_skips_audit(tmp_path):
    """Empty output_dir → audit is a no-op (matches the in-process
    test fixture pattern where output_dir is provided but tests don't
    care about the audit trail)."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir="",  # explicit empty
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
    )
    assert s._audit_path is None
    s._audit("register", worker_id="alpha")  # must not raise
