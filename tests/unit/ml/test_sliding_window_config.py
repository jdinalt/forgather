"""Regression test: causal_mask must read config.sliding_window, not config.window_size."""

import os
import sys
import unittest

import torch
from transformers import PretrainedConfig

_MODELSRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "modelsrc", "transformer"
)
sys.path.insert(0, os.path.abspath(_MODELSRC))

from causal_mask import causal_mask  # noqa: E402


def _make_config(attn_impl, sliding_window=None, hidden_size=32):
    config = PretrainedConfig(hidden_size=hidden_size)
    config._attn_implementation = attn_impl
    if sliding_window is not None:
        config.sliding_window = sliding_window
    return config


class TestSlidingWindowConfig(unittest.TestCase):
    def test_sliding_window_triggers_sw_branch_eager(self):
        """Eager + sliding_window=4 must materialize a sliding-window mask."""
        config = _make_config("eager", sliding_window=4)
        ids = torch.randint(0, 100, (1, 8))
        mask = causal_mask(config, torch.float32, input_ids=ids)
        self.assertIsNotNone(mask)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape, (1, 1, 8, 8))
        # Eager mask uses 0.0 for attend, finfo.min (~-inf) for masked.
        masked_value = torch.finfo(torch.float32).min
        # Row 0: attends to col 0 only.
        self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)
        # Row 7: attends to cols 4..7 (sliding window of 4), masks col 0 (outside window).
        self.assertAlmostEqual(mask[0, 0, 7, 0].item(), masked_value, delta=1.0)
        self.assertEqual(mask[0, 0, 7, 7].item(), 0.0)
        self.assertEqual(mask[0, 0, 7, 4].item(), 0.0)
        # Row 4: col 0 is outside window (distance 4 >= window 4), must be masked.
        self.assertAlmostEqual(mask[0, 0, 4, 0].item(), masked_value, delta=1.0)

    def test_sliding_window_with_padding_sdpa(self):
        """SDPA + sliding_window=64 + attention_mask must produce a 4D mask."""
        config = _make_config("sdpa", sliding_window=64)
        ids = torch.randint(0, 100, (1, 128))
        padding = torch.ones(1, 128, dtype=torch.long)
        mask = causal_mask(config, torch.float32, input_ids=ids, attention_mask=padding)
        self.assertIsNotNone(mask)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.ndim, 4)

    def test_no_sliding_window_sdpa_returns_none(self):
        """SDPA without sliding_window and no extras short-circuits to None."""
        config = _make_config("sdpa")
        ids = torch.randint(0, 100, (1, 128))
        mask = causal_mask(config, torch.float32, input_ids=ids)
        self.assertIsNone(mask)

    def test_window_size_attribute_is_ignored(self):
        """The legacy attribute `window_size` must NOT trigger the SW branch."""
        config = _make_config("sdpa")
        config.window_size = 64  # legacy/typo -- should be ignored
        ids = torch.randint(0, 100, (1, 128))
        mask = causal_mask(config, torch.float32, input_ids=ids)
        self.assertIsNone(mask)


if __name__ == "__main__":
    unittest.main()
