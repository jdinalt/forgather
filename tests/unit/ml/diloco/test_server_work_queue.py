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

    def test_queue_label_round_trips_hint_fields(self, client):
        """Workers ship path/name/split/revision/data_files alongside
        length in the register hint so the webui can render a readable
        label next to the dataset_id. Server stores the first
        registration's values and returns them on /work/queues +
        /work/queue. Later workers' divergent values are ignored.
        Regression guard for the hint extension."""
        # First registration: full set of identity fields.
        client.register_dataset(
            "w0",
            "ds-1",
            42,
            {
                "length": 1000,
                "path": "roneneldan/TinyStories",
                "name": "default",
                "split": "train",
                "revision": "f54c09f",
                "data_files": ["a.parquet", "b.parquet"],
            },
        )
        # Second registration with the same dataset_id but DIFFERENT
        # display fields. Length matches (length-mismatch would 409),
        # but the label fields should not overwrite the first.
        client.register_dataset(
            "w1",
            "ds-1",
            42,
            {
                "length": 1000,
                "path": "this-should-be-ignored",
                "split": "validation",
            },
        )

        # Summary endpoint.
        qs = client.get_work_queues()
        ours = next(q for q in qs if q["dataset_id"] == "ds-1")
        assert ours["hint"]["path"] == "roneneldan/TinyStories"
        assert ours["hint"]["name"] == "default"
        assert ours["hint"]["split"] == "train"
        assert ours["hint"]["revision"] == "f54c09f"
        assert ours["hint"]["data_files"] == ["a.parquet", "b.parquet"]
        assert ours["hint"]["length"] == 1000

        # Detail endpoint mirrors the same fields.
        detail = client.get_work_queue("ds-1", 42)
        assert detail["hint"]["path"] == "roneneldan/TinyStories"
        assert detail["hint"]["split"] == "train"

    def test_queue_label_optional_fields_omitted(self, client):
        """A worker that only ships ``length`` (pre-hint-extension or
        a load with no name/split) gets a hint with just length back —
        not a hint sprinkled with None placeholders."""
        client.register_dataset("w0", "ds-1", 42, {"length": 500})
        detail = client.get_work_queue("ds-1", 42)
        assert detail["hint"] == {"length": 500}

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
    def test_work_queues_persisted_and_restored(self, tmp_path):
        """The server is the authority for which rows have been consumed,
        so its per-(dataset_id, shuffle_seed) issued/completed bitmaps and
        the per-dataset length snapshot MUST round-trip through
        save_state/load_state (#105). Without this a restart re-issues
        already-trained units within the epoch."""
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
        time.sleep(0.2)
        try:
            c = DiLoCoClient(f"localhost:{s.port}", timeout=10)
            c.register_dataset("w0", "ds-1", 42, {"length": 5000})
            u0 = c.request_work("w0", "ds-1", 42)["unit_id"]
            c.request_work("w0", "ds-1", 42)  # issue a second unit
            c.complete_work("w0", "ds-1", 42, u0)
            save_dir = str(tmp_path / "saved")
            s._dirty = True
            s.save_state(save_dir)
        finally:
            s.stop()

        # The persisted file carries the queue + length snapshot.
        loaded = _torch.load(
            os.path.join(save_dir, "server_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert "work_queues" in loaded and "dataset_lengths" in loaded
        assert loaded["dataset_lengths"]["ds-1"] == 5000
        assert "ds-1|42" in loaded["work_queues"]

        # A fresh server restores the queue with issuance intact.
        s2 = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=save_dir,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        assert s2._dataset_lengths == {"ds-1": 5000}
        q = s2._work_queues[("ds-1", 42)]
        assert q.issued_count == 2  # both issued units survived
        assert q.completed_count == 1  # the completed one survived
        assert q.hint_length == 5000
        # Bits 0 and 1 are set in the restored issued bitmap.
        assert q.issued[0] & 0b11 == 0b11

    def test_restored_queue_resumes_issuance(self, tmp_path):
        """After a restart, a worker re-registering its dataset reuses the
        restored queue and is handed the NEXT un-issued unit — it does not
        restart the epoch from unit 0 (#105)."""
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
            issued = [c.request_work("w0", "ds-1", 42)["unit_id"] for _ in range(3)]
            assert issued == [0, 1, 2]
            save_dir = str(tmp_path / "saved")
            s._dirty = True
            s.save_state(save_dir)
        finally:
            s.stop()

        s2 = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=save_dir,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        s2.start()
        time.sleep(0.2)
        try:
            c2 = DiLoCoClient(f"localhost:{s2.port}", timeout=10)
            # Re-register the same dataset (what a relaunched worker does).
            c2.register_dataset("w0", "ds-1", 42, {"length": 5000})
            # Issuance resumes at unit 3, not 0.
            assert c2.request_work("w0", "ds-1", 42)["unit_id"] == 3
        finally:
            s2.stop()

    def test_malformed_work_queue_key_skipped(self, tmp_path):
        """A work_queues entry whose key isn't 'dataset_id|seed' is skipped
        (logged), not fatal — load_state must not crash the server."""
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

        sp = os.path.join(save_dir, "server_state.pt")
        ss = _torch.load(sp, map_location="cpu", weights_only=False)
        ss["work_queues"] = {
            "no-seed-separator": {
                "total_units": 8,
                "issued": bytes(1),
                "completed": bytes(1),
                "hint_length": 1,
            }
        }
        _torch.save(ss, sp)

        s2 = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=save_dir,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        assert s2._work_queues == {}  # malformed entry skipped, no crash

    def test_corrupt_work_queue_value_does_not_brick_startup(self, tmp_path):
        """A work_queues entry with a valid key but a missing field, or a
        bitmap whose length disagrees with total_units, is skipped (logged)
        — one bad/partial entry must not abort the whole load and prevent a
        restart. Good entries in the same map still load."""
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
        time.sleep(0.2)
        try:
            c = DiLoCoClient(f"localhost:{s.port}", timeout=10)
            c.register_dataset("w0", "good-ds", 1, {"length": 100})
            c.request_work("w0", "good-ds", 1)
            save_dir = str(tmp_path / "saved")
            s._dirty = True
            s.save_state(save_dir)
        finally:
            s.stop()

        sp = os.path.join(save_dir, "server_state.pt")
        ss = _torch.load(sp, map_location="cpu", weights_only=False)
        # Valid key, missing 'total_units'/'issued'.
        ss["work_queues"]["partial-ds|0"] = {"hint_length": 5000}
        # Valid key, bitmap length inconsistent with total_units.
        ss["work_queues"]["badbitmap-ds|0"] = {
            "total_units": 1024,  # expects 128 bytes
            "issued": bytes(1),  # but only 1
            "completed": bytes(1),
            "hint_length": 1,
        }
        _torch.save(ss, sp)

        s2 = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=save_dir,
            num_workers=1,
            port=0,
            default_work_units=8,
        )
        # The two bad entries are skipped; the good one survives.
        assert ("good-ds", 1) in s2._work_queues
        assert ("partial-ds", 0) not in s2._work_queues
        assert ("badbitmap-ds", 0) not in s2._work_queues

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
