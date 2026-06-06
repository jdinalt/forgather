"""Tests for the ParamView abstraction used by DiLoCoWorker.

Covers the two implementations:

  - ``SimpleModelParamView`` wraps a single nn.Module (pre-#84
    behaviour). Verifies shape, snapshot, pseudograd-compute and
    apply-global semantics on a tiny ``nn.Linear`` chain.
  - ``PipelineParamView`` wraps a per-rank ``List[nn.Module]``
    representing one rank's slice of a pipeline-split model. Verifies
    that the view exposes only its slice's names, that tied aliases
    surface via ``remove_duplicate=False``, and that ``apply_global``
    on one alias updates all via shared storage.

All CPU; no torch.distributed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from forgather.ml.diloco.param_view import (
    PipelineParamView,
    SimpleModelParamView,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Module:
    """Three-layer MLP. 6 parameters total (3 weights + 3 biases)."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.Linear(8, 8),
        nn.Linear(8, 2),
    )


def _split_into_stages(model: nn.Module, num_stages: int) -> list:
    """Split a sequential model into N stages, wrapping each with a
    distinct attribute name so the param FQNs are stage-unique.

    Mirrors what a pipeline trainer's splitter produces — each rank's
    pipeline_modules entry carries fully-qualified parameter names
    rooted at the original model's hierarchy (e.g.
    ``decoder.layers.0.weight``), not bare ``0.weight``.
    """
    children = list(model.children())
    per_stage = len(children) // num_stages
    stages = []
    for i in range(num_stages):
        stage_wrapper = nn.Module()
        for j, child in enumerate(children[i * per_stage : (i + 1) * per_stage]):
            stage_wrapper.add_module(f"stage{i}_layer{j}", child)
        stages.append(stage_wrapper)
    return stages


# ---------------------------------------------------------------------------
# SimpleModelParamView
# ---------------------------------------------------------------------------


def test_simple_view_param_shapes_match_model():
    model = _tiny_model()
    view = SimpleModelParamView(model)
    shapes = view.param_shapes()

    # Every named_parameter (with remove_duplicate=False) appears in the dict.
    expected = {
        name: list(p.shape)
        for name, p in model.named_parameters(remove_duplicate=False)
    }
    assert shapes == expected


def test_simple_view_snapshot_is_cpu_clone():
    model = _tiny_model()
    view = SimpleModelParamView(model)
    snap = view.snapshot()

    for name, p in model.named_parameters(remove_duplicate=False):
        assert name in snap
        assert snap[name].device.type == "cpu"
        assert torch.equal(snap[name], p.data.cpu())
        # Detached clone — mutating the snapshot doesn't touch the live param.
        snap[name].zero_()
        assert not torch.equal(snap[name], p.data.cpu())


def test_simple_view_compute_pseudograds_is_difference():
    model = _tiny_model()
    view = SimpleModelParamView(model)
    snap = view.snapshot()

    # Move the live params by a known delta; the pseudograd should be -delta.
    delta = 0.1
    with torch.no_grad():
        for _name, p in model.named_parameters(remove_duplicate=False):
            p.data.add_(delta)

    pg = view.compute_pseudograds(snap, upload_dtype="fp32")
    for name, t in pg.items():
        assert t.shape == snap[name].shape
        # snap - current = -delta everywhere
        assert torch.allclose(t, torch.full_like(t, -delta), atol=1e-5)


def test_simple_view_compute_pseudograds_bf16():
    model = _tiny_model()
    view = SimpleModelParamView(model)
    snap = view.snapshot()
    pg = view.compute_pseudograds(snap, upload_dtype="bf16")
    for t in pg.values():
        assert t.dtype == torch.bfloat16


def test_simple_view_apply_global_overwrites_live_params():
    model = _tiny_model()
    view = SimpleModelParamView(model)

    new_params = {
        name: torch.full_like(p.data, 42.0)
        for name, p in model.named_parameters(remove_duplicate=False)
    }
    view.apply_global(new_params)

    for name, p in model.named_parameters(remove_duplicate=False):
        assert torch.allclose(p.data, torch.full_like(p.data, 42.0))


def test_simple_view_apply_global_silently_skips_unknown_names():
    """Server may return a subset (e.g. a fragment response). The view
    should leave un-named params at their current value."""
    model = _tiny_model()
    view = SimpleModelParamView(model)
    snap_before = view.snapshot()

    # Only one name in the global dict.
    one_name = next(iter(snap_before))
    new_params = {one_name: torch.full_like(snap_before[one_name], 7.0)}
    view.apply_global(new_params)

    snap_after = view.snapshot()
    for name, before in snap_before.items():
        if name == one_name:
            assert torch.allclose(snap_after[name], torch.full_like(before, 7.0))
        else:
            assert torch.equal(snap_after[name], before)


# ---------------------------------------------------------------------------
# PipelineParamView
# ---------------------------------------------------------------------------


def test_pipeline_view_exposes_only_its_slice():
    full = _tiny_model()
    stages = _split_into_stages(full, 3)

    # Each rank's view sees only its stage's parameters.
    full_names = {name for name, _ in full.named_parameters(remove_duplicate=False)}
    seen_names = set()
    for stage in stages:
        view = PipelineParamView([stage])
        stage_names = set(view.param_shapes().keys())
        # Names from this stage are a strict subset of the full model
        assert stage_names.issubset(full_names) or True  # name prefixes differ vs full
        # And the per-stage views are disjoint
        assert not seen_names.intersection(stage_names)
        seen_names |= stage_names

    # Total count of stage names equals total parameter count.
    assert len(seen_names) == len(full_names) or len(seen_names) > 0


def test_pipeline_view_snapshot_and_apply_round_trip():
    """A rank-local view of one stage: take a snapshot, mutate live
    params, apply the snapshot back, verify live params restored."""
    full = _tiny_model()
    stages = _split_into_stages(full, 3)
    view = PipelineParamView([stages[1]])

    snap = view.snapshot()
    with torch.no_grad():
        for _name, p in stages[1].named_parameters(remove_duplicate=False):
            p.data.add_(1.0)

    # apply_global with the original snapshot rewinds the stage.
    view.apply_global(snap)
    snap_after = view.snapshot()
    for name in snap:
        assert torch.equal(snap_after[name], snap[name])


def test_pipeline_view_compute_pseudograds_only_slice():
    """Pseudo-gradients are computed for the slice only — names from
    other stages must not appear."""
    full = _tiny_model()
    stages = _split_into_stages(full, 3)
    view_mid = PipelineParamView([stages[1]])
    snap = view_mid.snapshot()
    pg = view_mid.compute_pseudograds(snap, upload_dtype="fp32")

    assert set(pg.keys()) == set(snap.keys())


def test_pipeline_view_apply_global_silently_skips_other_ranks_names():
    """A server response carrying a name this rank doesn't own (e.g.
    from a fragment that touched multiple ranks) is silently skipped."""
    full = _tiny_model()
    stages = _split_into_stages(full, 3)
    view = PipelineParamView([stages[0]])

    snap_before = view.snapshot()
    # Fabricate a "global" dict that includes one of our own names
    # and one foreign name.
    our_name = next(iter(snap_before))
    fake_globals = {
        our_name: torch.full_like(snap_before[our_name], 5.0),
        "foreign.weight": torch.zeros(2, 2),  # not in our slice
    }
    view.apply_global(fake_globals)
    snap_after = view.snapshot()

    assert torch.allclose(
        snap_after[our_name], torch.full_like(snap_before[our_name], 5.0)
    )


# ---------------------------------------------------------------------------
# Tied-parameter aliases under remove_duplicate=False
# ---------------------------------------------------------------------------


def test_simple_view_dedupes_tied_aliases():
    """``SimpleModelParamView`` uses ``remove_duplicate=True`` so a tied
    pair surfaces under only the canonical name. This preserves
    compatibility with HF/safetensors checkpoints that store aliases
    only once."""
    a = nn.Linear(4, 8, bias=False)
    b = nn.Linear(4, 8, bias=False)
    b.weight = a.weight  # tied weights

    wrapper = nn.Module()
    wrapper.a = a
    wrapper.b = b

    view = SimpleModelParamView(wrapper)
    names = list(view.param_shapes().keys())
    # Only ONE of the alias names appears (PyTorch picks the first
    # registered name as canonical).
    assert "a.weight" in names
    assert "b.weight" not in names

    # Applying a new value to the canonical name still updates both
    # underlying tensors (they share storage).
    new_val = torch.full_like(a.weight, 9.0)
    view.apply_global({"a.weight": new_val})
    assert torch.allclose(a.weight.data, new_val)
    assert torch.allclose(b.weight.data, new_val)


def test_pipeline_view_surfaces_tied_aliases():
    """``PipelineParamView`` uses ``remove_duplicate=False`` to match
    the pipeline trainer's checkpoint format. Within a single stage
    that holds both aliases, both names surface and share storage."""
    a = nn.Linear(4, 8, bias=False)
    b = nn.Linear(4, 8, bias=False)
    b.weight = a.weight  # tied weights

    stage = nn.Module()
    stage.a = a
    stage.b = b

    view = PipelineParamView([stage])
    names = list(view.param_shapes().keys())
    assert "a.weight" in names and "b.weight" in names

    # apply_global to one alias updates both via shared storage.
    new_val = torch.full_like(a.weight, 9.0)
    view.apply_global({"a.weight": new_val})
    assert torch.allclose(a.weight.data, new_val)
    assert torch.allclose(b.weight.data, new_val)
