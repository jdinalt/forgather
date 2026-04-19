"""Unit tests for YaRN scaling in modelsrc/transformer/rotary_embeddings.py."""

import math
import sys

import pytest
import torch

sys.path.insert(0, "modelsrc/transformer")

from rotary_embeddings import RotaryEmbedding, apply_yarn_scaling  # noqa: E402

D_HEAD = 128
NUM_HEADS = 4
HIDDEN = D_HEAD * NUM_HEADS
ROPE_THETA = 10000.0
ORIG_CTX = 4096
FACTOR = 4.0


def _make_module(rope_type: str, scaling_extra: dict | None = None) -> RotaryEmbedding:
    rope_parameters = {"rope_theta": ROPE_THETA, "rope_type": rope_type}
    if rope_type == "yarn":
        rope_parameters.update(
            {"factor": FACTOR, "original_max_position_embeddings": ORIG_CTX}
        )
        if scaling_extra:
            rope_parameters.update(scaling_extra)
    module = RotaryEmbedding(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        max_position_embeddings=ORIG_CTX * int(FACTOR),
        rope_parameters=rope_parameters,
    )
    module.reset_parameters()
    return module


def test_yarn_inv_freq_length_and_attention_scaling():
    module = _make_module("yarn")
    assert module.inv_freq.shape == (D_HEAD // 2,)
    expected = 0.1 * math.log(FACTOR) + 1.0
    assert module.attention_scaling == pytest.approx(expected, rel=1e-6)


def test_yarn_ramp_endpoints_vs_default():
    yarn_module = _make_module("yarn")
    default_module = _make_module("default")
    yarn_if = yarn_module.inv_freq
    default_if = default_module.inv_freq

    # Low frequency (slow) end: the last few entries correspond to small inv_freq,
    # long wavelengths. YaRN's "interpolation" branch divides by factor here.
    slow_ratio = (yarn_if[-1] / default_if[-1]).item()
    assert slow_ratio == pytest.approx(1.0 / FACTOR, rel=1e-4)

    # High frequency (fast) end: short wavelengths, YaRN keeps these unscaled.
    fast_ratio = (yarn_if[0] / default_if[0]).item()
    assert fast_ratio == pytest.approx(1.0, rel=1e-4)


def test_yarn_missing_original_max_position_embeddings_raises():
    base_inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, D_HEAD, 2, dtype=torch.float32) / D_HEAD)
    )
    with pytest.raises(ValueError, match="original_max_position_embeddings"):
        apply_yarn_scaling(base_inv_freq, {"factor": FACTOR}, D_HEAD)
