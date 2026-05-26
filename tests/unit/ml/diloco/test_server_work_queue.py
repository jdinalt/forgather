"""Tests for the work-unit dispatch endpoints on DiLoCoServer.

Covers the design described in
``docs/design/diloco-work-unit-dispatch.md`` (phase 1):

- /datasets/register allocates or confirms a queue keyed by
  ``(dataset_id, shuffle_seed)``. Length-mismatch against a prior
  registration of the same dataset_id returns 409.
- /work/request returns ascending unit ids, then ``{exhausted: true}``.
- /work/complete updates the diagnostic-only "completed" bitmap;
  correctness path doesn't depend on it. Idempotent.
- /work/queues, /work/queue: diagnostic surface (base64 bitmaps,
  per-worker counters).
- save_state / load_state round-trips the bitmaps and counters.

Tests drive the server via the real HTTP path (start the threading
server, hit it with DiLoCoClient + a few direct urllib calls for the
409 paths the client coerces into ConnectionError today).
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict(dim=8, num_layers=2, seed=42):
    torch.manual_seed(seed)
    return {f"layer{i}.weight": torch.randn(dim, dim) for i in range(num_layers)}


@pytest.fixture
def server(tmp_path):
    """A running DiLoCoServer with a small K so exhaustion is cheap to test."""
    sd = _state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        default_work_units=8,
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


@pytest.fixture
def client(server):
    return DiLoCoClient(f"localhost:{server.port}", timeout=10)


def _decode_bitmap(b64: str) -> bytearray:
    return bytearray(base64.b64decode(b64))


def _bit_get(bm: bytearray, i: int) -> bool:
    return bool(bm[i >> 3] & (1 << (i & 7)))


# ---------------------------------------------------------------------------
# /datasets/register
# ---------------------------------------------------------------------------


class TestRegisterDataset:
    def test_returns_configured_k(self, client):
        reply = client.register_dataset("w0", "ds-1", 42, {"length": 10000})
        assert reply == {"total_units": 8}

    def test_idempotent_same_args(self, client):
        a = client.register_dataset("w0", "ds-1", 42, {"length": 10000})
        b = client.register_dataset("w1", "ds-1", 42, {"length": 10000})
        assert a == b == {"total_units": 8}

    def test_distinct_queues_per_seed(self, client, server):
        client.register_dataset("w0", "ds-1", 42, {"length": 10000})
        client.register_dataset("w0", "ds-1", 43, {"length": 10000})
        with server._work_queues_lock:
            assert ("ds-1", 42) in server._work_queues
            assert ("ds-1", 43) in server._work_queues
            # Sibling queues — same dataset, different seeds — have
            # independent bitmaps.
            assert (
                server._work_queues[("ds-1", 42)]
                is not server._work_queues[("ds-1", 43)]
            )

    def test_length_mismatch_returns_409(self, client, server):
        client.register_dataset("w0", "ds-1", 42, {"length": 10000})
        # Second register with same dataset_id but different length →
        # 409. The DiLoCoClient wraps HTTPError as ConnectionError, so
        # exercise via urllib directly to inspect the status.
        req = urllib.request.Request(
            f"http://localhost:{server.port}/datasets/register",
            data=json.dumps(
                {
                    "worker_id": "w1",
                    "dataset_id": "ds-1",
                    "shuffle_seed": 99,
                    "hint": {"length": 12345},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 409
        body = json.loads(exc_info.value.read().decode("utf-8"))
        assert "previously registered with length=10000" in body["error"]


# ---------------------------------------------------------------------------
# /work/request
# ---------------------------------------------------------------------------


class TestRequestWork:
    def test_ascending_unit_ids_then_exhausted(self, client):
        client.register_dataset("w0", "ds-1", 42, {"length": 1000})
        seen = []
        # K=8 (from the fixture); request 8 then expect exhaustion on 9th.
        for _ in range(8):
            reply = client.request_work("w0", "ds-1", 42)
            seen.append(reply)
        assert [r["unit_id"] for r in seen] == list(range(8))
        # Next request: queue is drained.
        reply = client.request_work("w0", "ds-1", 42)
        assert reply == {"exhausted": True}

    def test_two_workers_interleave_no_duplicates(self, server):
        c0 = DiLoCoClient(f"localhost:{server.port}", timeout=10)
        c1 = DiLoCoClient(f"localhost:{server.port}", timeout=10)
        c0.register_dataset("w0", "ds-1", 42, {"length": 1000})
        c1.register_dataset("w1", "ds-1", 42, {"length": 1000})
        ids = set()
        # Drain alternately. K=8 → 4 each.
        for i in range(4):
            ids.add(c0.request_work("w0", "ds-1", 42)["unit_id"])
            ids.add(c1.request_work("w1", "ds-1", 42)["unit_id"])
        # All 8 unit_ids issued, no duplicates.
        assert ids == set(range(8))

    def test_unknown_queue_returns_404(self, server):
        req = urllib.request.Request(
            f"http://localhost:{server.port}/work/request",
            data=json.dumps(
                {"worker_id": "w", "dataset_id": "ds-unknown", "shuffle_seed": 1}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 404


# ---------------------------------------------------------------------------
# /work/complete (diagnostic-only — must not affect issuance)
# ---------------------------------------------------------------------------


class TestCompleteWork:
    def test_updates_completed_bitmap(self, client):
        client.register_dataset("w0", "ds-1", 42, {"length": 1000})
        uid = client.request_work("w0", "ds-1", 42)["unit_id"]
        client.complete_work("w0", "ds-1", 42, uid)
        # Read back via /work/queue and verify the bit is set.
        detail = client.get_work_queue("ds-1", 42)
        assert detail["completed_count"] == 1
        completed = _decode_bitmap(detail["completed_bitmap_b64"])
        assert _bit_get(completed, uid)

    def test_idempotent(self, client):
        client.register_dataset("w0", "ds-1", 42, {"length": 1000})
        uid = client.request_work("w0", "ds-1", 42)["unit_id"]
        client.complete_work("w0", "ds-1", 42, uid)
        client.complete_work("w0", "ds-1", 42, uid)
        detail = client.get_work_queue("ds-1", 42)
        # Bumping completed_count twice would be wrong — the second
        # complete is a no-op.
        assert detail["completed_count"] == 1

    def test_unknown_unit_id_400(self, server, client):
        client.register_dataset("w0", "ds-1", 42, {"length": 1000})
        req = urllib.request.Request(
            f"http://localhost:{server.port}/work/complete",
            data=json.dumps(
                {
                    "worker_id": "w0",
                    "dataset_id": "ds-1",
                    "shuffle_seed": 42,
                    "unit_id": 999,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400


# ---------------------------------------------------------------------------
# /work/queues and /work/queue (diagnostic surface)
# ---------------------------------------------------------------------------


class TestQueueDiagnostics:
    def test_queues_lists_all(self, client):
        client.register_dataset("w0", "ds-1", 42, {"length": 1000})
        client.register_dataset("w0", "ds-2", 99, {"length": 2000})
        qs = client.get_work_queues()
        keys = {(q["dataset_id"], q["shuffle_seed"]) for q in qs}
        assert keys == {("ds-1", 42), ("ds-2", 99)}
        # Summaries don't include bitmaps (they live on /work/queue).
        for q in qs:
            assert "issued_bitmap_b64" not in q

    def test_queue_bitmap_decodes_correctly(self, client):
        client.register_dataset("w0", "ds-1", 42, {"length": 1000})
        # Issue units 0 and 1.
        client.request_work("w0", "ds-1", 42)
        client.request_work("w0", "ds-1", 42)
        detail = client.get_work_queue("ds-1", 42)
        issued = _decode_bitmap(detail["issued_bitmap_b64"])
        completed = _decode_bitmap(detail["completed_bitmap_b64"])
        assert _bit_get(issued, 0)
        assert _bit_get(issued, 1)
        assert not _bit_get(issued, 2)
        # Completed bitmap is still all-zero (no /work/complete calls).
        assert all(b == 0 for b in completed)
        assert detail["issued_count"] == 2
        assert detail["completed_count"] == 0

    def test_queue_by_worker_counters(self, server):
        c0 = DiLoCoClient(f"localhost:{server.port}", timeout=10)
        c1 = DiLoCoClient(f"localhost:{server.port}", timeout=10)
        c0.register_dataset("alpha", "ds-1", 42, {"length": 1000})
        c1.register_dataset("beta", "ds-1", 42, {"length": 1000})
        u_alpha = c0.request_work("alpha", "ds-1", 42)["unit_id"]
        c1.request_work("beta", "ds-1", 42)
        c1.request_work("beta", "ds-1", 42)
        c0.complete_work("alpha", "ds-1", 42, u_alpha)
        detail = c0.get_work_queue("ds-1", 42)
        assert detail["by_worker"]["alpha"] == {
            "units_issued": 1,
            "units_completed": 1,
        }
        assert detail["by_worker"]["beta"] == {
            "units_issued": 2,
            "units_completed": 0,
        }


# ---------------------------------------------------------------------------
# Persistence: save_state / load_state round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_work_queues_NOT_persisted_across_save_load(self, tmp_path):
        """Regression for #46.

        Work-queue persistence was removed because it created a
        cross-experiment hazard: same output_dir, new dataset →
        old queues with stale dataset_ids and stale hint.length lingered
        in /work/queues, confusing the operator. Workers re-register
        their datasets on connect, so the server reconstructs the
        queue map fresh on demand. The cost is bounded re-work after
        a server crash (≤ N_workers in-flight units), which matches
        the existing worker-death budget."""
        sd = _state_dict()
        ckpt = make_initial_checkpoint(sd, tmp_path)
        s = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        s.start()
        time.sleep(0.2)
        try:
            c = DiLoCoClient(f"localhost:{s.port}", timeout=10)
            c.register_dataset("w0", "ds-1", 42, {"length": 5000})
            u0 = c.request_work("w0", "ds-1", 42)["unit_id"]
            c.request_work("w0", "ds-1", 42)
            c.complete_work("w0", "ds-1", 42, u0)
            save_dir = str(tmp_path / "saved")
            s._dirty = True
            s.save_state(save_dir)
        finally:
            s.stop()

        # Fresh server: queues come up empty, ready for fresh
        # registrations from workers.
        s2 = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=save_dir,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        assert s2._work_queues == {}
        assert s2._dataset_lengths == {}

        # And the persisted file no longer contains a work_queues key
        # at all — confirm via direct inspection so a future regression
        # that re-adds the key without restoring the load path also
        # fails this test.
        import torch as _torch

        loaded = _torch.load(
            os.path.join(save_dir, "server_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert "work_queues" not in loaded
        assert "dataset_lengths" not in loaded

    def test_legacy_checkpoint_with_work_queues_loads_with_warning(
        self, tmp_path, caplog
    ):
        """A pre-#46 server_state.pt that contains a work_queues
        entry must still load cleanly, but the operator should see a
        clear warning that those queues are being dropped. Mid-epoch
        resume that worked under the old behavior is intentionally
        gone — workers re-register on connect, in-flight units get
        re-issued (≤ N_workers extra units trained per crash)."""
        import logging

        import torch as _torch

        sd = _state_dict()
        ckpt = make_initial_checkpoint(sd, tmp_path)
        s = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        s.start()
        try:
            save_dir = str(tmp_path / "saved")
            s._dirty = True
            s.save_state(save_dir)
        finally:
            s.stop()

        # Inject a legacy-shaped work_queues entry into the saved
        # server_state.pt, then reload and confirm the warning fires
        # and the queues themselves are NOT rehydrated.
        sp = os.path.join(save_dir, "server_state.pt")
        ss = _torch.load(sp, map_location="cpu", weights_only=False)
        ss["work_queues"] = {
            "stale-ds|0": {
                "dataset_id": "stale-ds",
                "shuffle_seed": 0,
                "total_units": 8,
                "issued": bytes(b"\x00"),
                "completed": bytes(b"\x00"),
                "hint_length": 12345,
                "issued_count": 0,
                "completed_count": 0,
                "by_worker": {},
            }
        }
        _torch.save(ss, sp)

        with caplog.at_level(logging.WARNING, logger="forgather.ml.diloco.server"):
            s2 = DiLoCoServer(
                output_dir=str(tmp_path),
                from_checkpoint=save_dir,
                num_workers=1,
                port=0,
                default_work_units=8,
            )

        assert s2._work_queues == {}
        assert any(
            "work-queue" in rec.message for rec in caplog.records
        ), "Expected a warning about ignored legacy work_queues, got: " + repr(
            [rec.message for rec in caplog.records]
        )

    def test_legacy_checkpoint_loads_with_empty_queue_map(self, tmp_path):
        """Checkpoints written before this feature landed have no
        ``work_queues`` key in server_state.pt. They must load cleanly
        with an empty queue map — no migration required."""
        sd = _state_dict()
        ckpt = make_initial_checkpoint(sd, tmp_path)
        s = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        # No registrations — _work_queues stays empty. Confirm
        # the server starts up fine (load_state runs in __init__).
        assert s._work_queues == {}
        assert s._dataset_lengths == {}
