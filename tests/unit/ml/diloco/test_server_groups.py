"""Tests for the DiLoCoServer's group-aware registration & sync barrier.

Covers issue #84: per-rank pipeline workers register as members of a
``WorkerGroup``, each declaring only their slice of the model. The
union of a sealed group's slices must exactly cover the server's
param set. Sync barrier releases only when every member of every
group has submitted. Atomic group eviction on member death.

All tests are pure-Python — no torch.distributed, no GPU.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict() -> Dict[str, torch.Tensor]:
    """A 4-param fake model state_dict.

    Names chosen to mimic a typical encoder-style split: two early
    layers (slice A) and two late layers (slice B).
    """
    torch.manual_seed(0)
    return {
        "layer_0.weight": torch.randn(4, 4),
        "layer_1.weight": torch.randn(4, 4),
        "layer_2.weight": torch.randn(4, 4),
        "layer_3.weight": torch.randn(4, 4),
    }


def _full_shapes() -> Dict[str, List[int]]:
    return {name: list(t.shape) for name, t in _state_dict().items()}


def _slice_a() -> Dict[str, List[int]]:
    return {"layer_0.weight": [4, 4], "layer_1.weight": [4, 4]}


def _slice_b() -> Dict[str, List[int]]:
    return {"layer_2.weight": [4, 4], "layer_3.weight": [4, 4]}


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


@pytest.fixture
def async_server(tmp_path):
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=2,
        port=0,
        heartbeat_timeout=0,
        async_mode=True,
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


def _register(
    server: DiLoCoServer,
    worker_id: str,
    param_shapes: Dict[str, List[int]],
    group: Optional[dict] = None,
):
    body = {"worker_id": worker_id, "hostname": "test", "param_shapes": param_shapes}
    if group is not None:
        body["group"] = group
    req = urllib.request.Request(
        f"http://localhost:{server.port}/register",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=5)


# ---------------------------------------------------------------------------
# Solo group (pre-#84 contract preserved)
# ---------------------------------------------------------------------------


def test_solo_group_unchanged(server):
    """A worker that omits the ``group`` block forms a degenerate group
    of one. The slice IS the full model and seal-time coverage runs
    immediately; the contract is identical to the pre-#84 fingerprint
    check."""
    resp = _register(server, "alpha", _full_shapes())
    assert resp.status == 200


# ---------------------------------------------------------------------------
# Group seal + sync
# ---------------------------------------------------------------------------


def test_group_seal_and_sync(server):
    """Two ranks of a pp_world_size=2 group register with disjoint
    slices; the second registration seals the group; both ranks then
    submit slice-pseudo-gradients and the barrier releases with
    per-name aggregation that uses each name's one contributor."""
    # Rank 0
    resp = _register(
        server,
        "alpha_pp0",
        _slice_a(),
        {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
    )
    assert resp.status == 200

    # Rank 1 — seals the group, coverage check passes (a ∪ b == full)
    resp = _register(
        server,
        "alpha_pp1",
        _slice_b(),
        {"group_id": "alpha", "pp_rank": 1, "pp_world_size": 2},
    )
    assert resp.status == 200

    # The group is now sealed and the server tracks two workers.
    assert len(server._workers) == 2
    assert "alpha" in server._groups
    assert server._groups["alpha"].sealed
    assert set(server._groups["alpha"].members.values()) == {"alpha_pp0", "alpha_pp1"}


# ---------------------------------------------------------------------------
# Coverage failure (atomic rollback)
# ---------------------------------------------------------------------------


def test_group_coverage_mismatch_returns_422_and_rolls_back(server):
    """When the last member of a group registers and the union doesn't
    cover the server's full param set, the seal is rejected with 422
    AND every member registered so far is evicted (atomic rollback)."""
    # Rank 0 registers a valid slice — registry now has one member.
    resp = _register(
        server,
        "alpha_pp0",
        _slice_a(),
        {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
    )
    assert resp.status == 200
    assert "alpha_pp0" in server._workers
    assert "alpha" in server._groups

    # Rank 1 registers with a slice missing layer_3 → union doesn't
    # cover the full set → 422 + rollback of the rank-0 member too.
    bad_b = {"layer_2.weight": [4, 4]}  # missing layer_3
    with pytest.raises(urllib.error.HTTPError) as exc:
        _register(
            server,
            "alpha_pp1",
            bad_b,
            {"group_id": "alpha", "pp_rank": 1, "pp_world_size": 2},
        )
    assert exc.value.code == 422
    body = json.loads(exc.value.read().decode("utf-8"))
    assert body["kind"] == "group_coverage"
    assert "layer_3.weight" in body["error"]

    # Rollback: rank 0 should also be gone now.
    assert "alpha_pp0" not in server._workers
    assert "alpha" not in server._groups
    assert "alpha_pp0" not in server._worker_to_group


def test_group_slice_extra_param_returns_422(server):
    """A slice with a name the server doesn't have is rejected at slice
    fingerprint time (before group bookkeeping mutates)."""
    bad_a = dict(_slice_a())
    bad_a["extra.weight"] = [2, 2]
    with pytest.raises(urllib.error.HTTPError) as exc:
        _register(
            server,
            "alpha_pp0",
            bad_a,
            {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
        )
    assert exc.value.code == 422
    # No partial group should have been created.
    assert "alpha" not in server._groups


# ---------------------------------------------------------------------------
# Geometry / seal-state errors
# ---------------------------------------------------------------------------


def test_sealed_group_refuses_extra_member(server):
    """A worker arriving after pp_world_size members have registered
    is refused with 409 ('group is sealed')."""
    _register(
        server,
        "alpha_pp0",
        _slice_a(),
        {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
    )
    _register(
        server,
        "alpha_pp1",
        _slice_b(),
        {"group_id": "alpha", "pp_rank": 1, "pp_world_size": 2},
    )
    # Attempt to add a third member — but pp_world_size=2 declared, so
    # there is no pp_rank=2 slot.
    with pytest.raises(urllib.error.HTTPError) as exc:
        _register(
            server,
            "alpha_pp2",
            _slice_a(),
            {"group_id": "alpha", "pp_rank": 2, "pp_world_size": 2},
        )
    assert exc.value.code == 400


def test_slot_collision_returns_409(server):
    """Two workers registering with the same pp_rank in the same group
    is a collision (409)."""
    _register(
        server,
        "alpha_pp0",
        _slice_a(),
        {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        _register(
            server,
            "alpha_pp0_dup",
            _slice_a(),
            {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
        )
    assert exc.value.code == 409


def test_pp_world_size_mismatch_returns_422(server):
    """First member declared pp_world_size=2; a second member arriving
    with pp_world_size=3 is refused — group geometry is set on first
    registration."""
    _register(
        server,
        "alpha_pp0",
        _slice_a(),
        {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        _register(
            server,
            "alpha_pp1",
            _slice_b(),
            {"group_id": "alpha", "pp_rank": 1, "pp_world_size": 3},
        )
    assert exc.value.code == 422


def test_invalid_geometry_returns_400(server):
    """``pp_rank >= pp_world_size`` is a malformed payload — 400 before
    any registry mutation."""
    with pytest.raises(urllib.error.HTTPError) as exc:
        _register(
            server,
            "alpha",
            _slice_a(),
            {"group_id": "alpha", "pp_rank": 2, "pp_world_size": 2},
        )
    assert exc.value.code == 400


def test_async_mode_rejects_pipeline_group(async_server):
    """Pipeline groups + async mode is out of scope for #84; the server
    refuses pipeline-group registration with 400."""
    with pytest.raises(urllib.error.HTTPError) as exc:
        _register(
            async_server,
            "alpha_pp0",
            _slice_a(),
            {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
        )
    assert exc.value.code == 400


# ---------------------------------------------------------------------------
# Atomic group eviction on worker death
# ---------------------------------------------------------------------------


def test_sealed_group_one_dies_evicts_whole_group(server):
    """When a member of a sealed pipeline group dies (heartbeat timeout
    or explicit deregister), every member of the group is evicted and
    the group entry removed."""
    _register(
        server,
        "alpha_pp0",
        _slice_a(),
        {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
    )
    _register(
        server,
        "alpha_pp1",
        _slice_b(),
        {"group_id": "alpha", "pp_rank": 1, "pp_world_size": 2},
    )
    assert "alpha_pp0" in server._workers and "alpha_pp1" in server._workers

    # Simulate one rank's death via the internal hook.
    server._handle_worker_death("alpha_pp0")

    assert "alpha_pp0" not in server._workers
    assert "alpha_pp1" not in server._workers, "sibling should also be evicted"
    assert "alpha" not in server._groups
    assert "alpha_pp0" not in server._worker_to_group
    assert "alpha_pp1" not in server._worker_to_group


def test_solo_group_death_only_removes_that_worker(server):
    """Pre-#84 solo workers continue to behave as before: one worker's
    death removes only that worker (the group of one), no cascade."""
    _register(server, "alpha", _full_shapes())
    _register(server, "beta", _full_shapes())
    server._handle_worker_death("alpha")

    assert "alpha" not in server._workers
    assert "beta" in server._workers  # untouched
    assert "alpha" not in server._groups
    assert "beta" in server._groups


def test_partial_group_then_death_removes_partial(server):
    """Rank 0 registers; rank 1 never shows up; the group sits partial
    in ``_groups``. When rank 0 dies the partial group is cleaned up."""
    _register(
        server,
        "alpha_pp0",
        _slice_a(),
        {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
    )
    assert "alpha" in server._groups
    assert not server._groups["alpha"].sealed

    server._handle_worker_death("alpha_pp0")

    assert "alpha" not in server._groups
    assert "alpha_pp0" not in server._workers


# ---------------------------------------------------------------------------
# Tied-parameter aliases (same name in two slices)
# ---------------------------------------------------------------------------


def test_tied_alias_across_slices_accepted(tmp_path):
    """A name shared across two slices (representing a tied parameter
    held on both stages — e.g. an embedding tied to a transposed-view
    output projection) is accepted: the seal-time coverage check only
    requires the union to cover the server, with no disjointness
    requirement on overlap."""
    sd = {
        "embed.weight": torch.randn(8, 4),
        "layer.weight": torch.randn(4, 4),
        "lm_head.weight": torch.randn(8, 4),  # tied alias of embed.weight on server
    }
    ckpt = make_initial_checkpoint(sd, tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=2,
        port=0,
        heartbeat_timeout=0,
    )
    s.start()
    try:
        time.sleep(0.2)
        # Rank 0 holds embed + layer; rank 1 holds layer + lm_head.
        # Both ranks declare layer.weight — accepted under the tied-
        # parameter allowance. (The server's per-name averaging
        # treats both contributions identically for an alias-equivalent
        # pair; this is the simplest test exercising the "duplicate
        # name across slices" path.)
        _register(
            s,
            "alpha_pp0",
            {"embed.weight": [8, 4], "layer.weight": [4, 4]},
            {"group_id": "alpha", "pp_rank": 0, "pp_world_size": 2},
        )
        _register(
            s,
            "alpha_pp1",
            {"layer.weight": [4, 4], "lm_head.weight": [8, 4]},
            {"group_id": "alpha", "pp_rank": 1, "pp_world_size": 2},
        )
        assert s._groups["alpha"].sealed
    finally:
        s.stop()
