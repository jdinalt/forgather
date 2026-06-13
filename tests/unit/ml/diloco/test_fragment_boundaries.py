"""Tests for block-boundary fragmentation (Streaming DiLoCo, arXiv:2501.18512).

`FragmentManager` splits the model on transformer-block boundaries (each fragment
is a set of whole blocks + the attached non-block params), assigned sequentially
or strided, falling back to a contiguous split when no block plan is present.
"""

import pytest
import torch.nn as nn

from forgather.ml.diloco.fragments import (
    FragmentManager,
    NoBlockPlanError,
    discover_block_boundaries,
)


class FakeBlock(nn.Module):
    """A transformer block (its class name goes in _no_split_modules)."""

    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Linear(d, d, bias=False)
        self.mlp = nn.Linear(d, d, bias=False)


class FakeTransformer(nn.Module):
    """Mirrors the real forgather model structure: embeddings (pre), an indexed
    ModuleDict of blocks, then a final norm + LM head (post)."""

    _no_split_modules = ["FakeBlock"]

    def __init__(self, n_layers=4, d=4):
        super().__init__()
        self.embedding = nn.Embedding(8, d)
        self.layers = nn.ModuleDict({str(i): FakeBlock(d) for i in range(n_layers)})
        self.final_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, 8, bias=False)


def _all_params(m):
    return [n for n, _ in m.named_parameters()]


class TestDiscovery:
    def test_discovers_blocks_pre_post(self):
        m = FakeTransformer(n_layers=4)
        groups, pre, post = discover_block_boundaries(m)
        assert len(groups) == 4
        # Each block group is exactly that block's params, in order.
        for i, g in enumerate(groups):
            assert all(n.startswith(f"layers.{i}.") for n in g)
            assert set(g) == {f"layers.{i}.attn.weight", f"layers.{i}.mlp.weight"}
        assert pre == ["embedding.weight"]
        assert post == ["final_norm.weight", "final_norm.bias", "lm_head.weight"]
        # Partition: every param appears exactly once across groups+pre+post.
        covered = pre + post + [n for g in groups for n in g]
        assert sorted(covered) == sorted(_all_params(m))

    def test_no_split_modules_missing_raises(self):
        m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        with pytest.raises(NoBlockPlanError):
            discover_block_boundaries(m)


class TestAssignment:
    def test_sequential(self):
        m = FakeTransformer(n_layers=4)
        fm = FragmentManager(m, num_fragments=2, assignment="sequential")
        assert fm.block_faithful
        # blocks {0,1} -> frag0, {2,3} -> frag1
        assert fm.param_to_fragment["layers.0.attn.weight"] == 0
        assert fm.param_to_fragment["layers.1.mlp.weight"] == 0
        assert fm.param_to_fragment["layers.2.attn.weight"] == 1
        assert fm.param_to_fragment["layers.3.mlp.weight"] == 1
        # embeddings -> first fragment, head/final-norm -> last fragment
        assert fm.param_to_fragment["embedding.weight"] == 0
        assert fm.param_to_fragment["lm_head.weight"] == 1
        assert fm.param_to_fragment["final_norm.weight"] == 1

    def test_strided(self):
        m = FakeTransformer(n_layers=4)
        fm = FragmentManager(m, num_fragments=2, assignment="strided")
        assert fm.block_faithful
        # block i -> fragment i % 2: {0,2} -> frag0, {1,3} -> frag1
        assert fm.param_to_fragment["layers.0.attn.weight"] == 0
        assert fm.param_to_fragment["layers.2.attn.weight"] == 0
        assert fm.param_to_fragment["layers.1.attn.weight"] == 1
        assert fm.param_to_fragment["layers.3.attn.weight"] == 1
        assert fm.param_to_fragment["embedding.weight"] == 0  # pre -> first
        assert fm.param_to_fragment["lm_head.weight"] == 1  # post -> last

    def test_partition_invariant(self):
        m = FakeTransformer(n_layers=6)
        fm = FragmentManager(m, num_fragments=3, assignment="strided")
        flat = [n for frag in fm.fragments for n in frag]
        assert sorted(flat) == sorted(_all_params(m))  # exactly once each

    def test_more_fragments_than_blocks_falls_back(self):
        m = FakeTransformer(n_layers=2)
        # 4 fragments > 2 blocks -> can't split faithfully -> contiguous fallback
        fm = FragmentManager(m, num_fragments=4, assignment="strided")
        assert not fm.block_faithful


class TestPerRankFilter:
    def test_filters_to_slice(self):
        """A ParamView exposing only a sub-range filters fragments to the slice,
        while block ids stay globally consistent (discovered from the full model)."""
        m = FakeTransformer(n_layers=4)

        class View:  # duck-typed ParamView: only layers 2-3 + head
            def named_parameters(self):
                keep = ("layers.2.", "layers.3.", "final_norm", "lm_head")
                for n, p in m.named_parameters():
                    if any(n.startswith(k) for k in keep):
                        yield n, p

        fm = FragmentManager(
            View(), num_fragments=2, assignment="sequential", boundary_source=m
        )
        assert fm.block_faithful
        flat = {n for frag in fm.fragments for n in frag}
        # Only the slice's names are present; layer 0/1 + embedding are absent.
        assert "layers.0.attn.weight" not in flat
        assert "embedding.weight" not in flat
        assert "layers.2.attn.weight" in flat
        assert "lm_head.weight" in flat
        # Global ids preserved: blocks {0,1}->frag0, {2,3}->frag1, so the slice's
        # layer-2/3 params land in fragment 1.
        assert fm.param_to_fragment["layers.2.attn.weight"] == 1
        assert fm.param_to_fragment["lm_head.weight"] == 1


class TestFallback:
    def test_plain_module_contiguous_fallback(self):
        m = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
        fm = FragmentManager(m, num_fragments=2)
        assert not fm.block_faithful
        # Falls back to equal-count contiguous split (2 params -> 1 each).
        assert [len(f) for f in fm.fragments] == [1, 1]


class TestRealModel:
    def test_real_llama_discovery(self):
        """Discovery against a real forgather Llama validates the synthetic mirror."""
        fg = pytest.importorskip("forgather")
        proj = fg.Project("4M.yaml", "examples/models/llama")
        factory = proj("model")
        model = factory() if callable(factory) else factory
        groups, pre, post = discover_block_boundaries(model)
        n_layers = len(
            {
                n.split("layers.")[1].split(".")[0]
                for n, _ in model.named_parameters()
                if "layers." in n
            }
        )
        assert len(groups) == n_layers
        assert pre == ["causal_lm.input_encoder.embedding.weight"]
        assert "lm_head.weight" in post
        assert any("layer_norm" in n for n in post)
