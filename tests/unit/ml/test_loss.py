#!/usr/bin/env python3
"""
Unit tests for loss functions in Forgather.

Tests that ChunkedCausalLoss produces identical results to CausalLoss
while using less memory for large vocabulary models.
"""

import unittest
import torch
from torch import FloatTensor, LongTensor

from forgather.ml.loss import CausalLoss, ChunkedCausalLoss


class TestCausalLoss(unittest.TestCase):
    """Test causal loss functions for correctness and equivalence."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_test_data(
        self, batch_size=2, seq_len=16, vocab_size=1000, dtype=torch.float32
    ):
        """
        Create synthetic test data for loss computation.

        Returns:
            logits: [batch_size, seq_len, vocab_size] unnormalized logits
            labels: [batch_size, seq_len] target token indices
        """
        logits = torch.randn(
            batch_size, seq_len, vocab_size, dtype=dtype, device=self.device
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        return logits, labels

    def test_chunked_matches_standard_small_vocab(self):
        """Test that chunked loss matches standard loss for small vocabulary."""
        logits, labels = self._create_test_data(
            batch_size=4, seq_len=32, vocab_size=1000
        )

        standard_loss = CausalLoss()
        chunked_loss = ChunkedCausalLoss(chunk_size=256)

        loss_standard = standard_loss(logits, labels)
        loss_chunked = chunked_loss(logits, labels)

        self.assertAlmostEqual(
            loss_standard.item(),
            loss_chunked.item(),
            places=5,
            msg="Chunked loss should match standard loss for small vocabulary",
        )

    def test_chunked_matches_standard_large_vocab(self):
        """Test that chunked loss matches standard loss for large vocabulary."""
        # Simulate Qwen3-like vocabulary size
        logits, labels = self._create_test_data(
            batch_size=2, seq_len=64, vocab_size=151936
        )

        standard_loss = CausalLoss()
        chunked_loss = ChunkedCausalLoss(chunk_size=4096)

        loss_standard = standard_loss(logits, labels)
        loss_chunked = chunked_loss(logits, labels)

        self.assertAlmostEqual(
            loss_standard.item(),
            loss_chunked.item(),
            places=4,
            msg="Chunked loss should match standard loss for large vocabulary",
        )

    def test_chunked_with_different_chunk_sizes(self):
        """Test that different chunk sizes produce identical results."""
        logits, labels = self._create_test_data(
            batch_size=2, seq_len=32, vocab_size=10000
        )

        chunk_sizes = [1024, 2048, 4096, 8192]
        losses = []

        for chunk_size in chunk_sizes:
            chunked_loss = ChunkedCausalLoss(chunk_size=chunk_size)
            loss = chunked_loss(logits, labels)
            losses.append(loss.item())

        # All losses should be very close to each other
        for i in range(len(losses) - 1):
            self.assertAlmostEqual(
                losses[i],
                losses[i + 1],
                places=5,
                msg=f"Chunk sizes {chunk_sizes[i]} and {chunk_sizes[i+1]} should produce same loss",
            )

    def test_chunked_with_ignore_index(self):
        """Test that chunked loss correctly handles ignore_index=-100."""
        batch_size, seq_len, vocab_size = 4, 32, 5000
        logits, labels = self._create_test_data(batch_size, seq_len, vocab_size)

        # Set some labels to ignore_index
        labels[:, ::4] = -100  # Ignore every 4th token

        standard_loss = CausalLoss()
        chunked_loss = ChunkedCausalLoss(chunk_size=1024)

        loss_standard = standard_loss(logits, labels)
        loss_chunked = chunked_loss(logits, labels)

        self.assertAlmostEqual(
            loss_standard.item(),
            loss_chunked.item(),
            places=5,
            msg="Chunked loss should handle ignore_index correctly",
        )

    def test_chunked_with_all_ignored(self):
        """Test that chunked loss returns 0 when all labels are ignored."""
        batch_size, seq_len, vocab_size = 2, 16, 1000
        logits, _ = self._create_test_data(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100, device=self.device)

        chunked_loss = ChunkedCausalLoss(chunk_size=256)
        loss = chunked_loss(logits, labels)

        self.assertEqual(
            loss.item(), 0.0, msg="Loss should be 0 when all labels are ignored"
        )

    def test_chunked_gradient_matches_standard(self):
        """Test that gradients from chunked loss match standard loss."""
        logits, labels = self._create_test_data(
            batch_size=2, seq_len=16, vocab_size=5000
        )
        logits.requires_grad = True

        # Compute standard loss and gradients
        logits_standard = logits.clone().detach().requires_grad_(True)
        standard_loss = CausalLoss()
        loss_standard = standard_loss(logits_standard, labels)
        loss_standard.backward()
        grad_standard = logits_standard.grad.clone()

        # Compute chunked loss and gradients
        logits_chunked = logits.clone().detach().requires_grad_(True)
        chunked_loss = ChunkedCausalLoss(chunk_size=1024)
        loss_chunked = chunked_loss(logits_chunked, labels)
        loss_chunked.backward()
        grad_chunked = logits_chunked.grad.clone()

        # Check loss values match
        self.assertAlmostEqual(
            loss_standard.item(),
            loss_chunked.item(),
            places=5,
            msg="Forward pass losses should match",
        )

        # Check gradients match
        max_grad_diff = (grad_standard - grad_chunked).abs().max().item()
        self.assertLess(
            max_grad_diff,
            1e-5,
            msg=f"Gradients should match (max diff: {max_grad_diff})",
        )

    def test_chunked_with_bfloat16(self):
        """Test that chunked loss works with bfloat16 (typical training dtype)."""
        if not torch.cuda.is_available():
            self.skipTest("bfloat16 test requires CUDA")

        logits, labels = self._create_test_data(
            batch_size=2, seq_len=32, vocab_size=50000, dtype=torch.bfloat16
        )

        standard_loss = CausalLoss()
        chunked_loss = ChunkedCausalLoss(chunk_size=2048)

        loss_standard = standard_loss(logits, labels)
        loss_chunked = chunked_loss(logits, labels)

        # bfloat16 has less precision, so use looser tolerance
        self.assertAlmostEqual(
            loss_standard.item(),
            loss_chunked.item(),
            places=3,
            msg="Chunked loss should match standard loss with bfloat16",
        )

    def test_chunked_chunk_size_not_divisor(self):
        """Test that chunked loss works when vocab_size is not divisible by chunk_size."""
        logits, labels = self._create_test_data(
            batch_size=2, seq_len=16, vocab_size=10003  # Prime number
        )

        standard_loss = CausalLoss()
        chunked_loss = ChunkedCausalLoss(chunk_size=1024)

        loss_standard = standard_loss(logits, labels)
        loss_chunked = chunked_loss(logits, labels)

        self.assertAlmostEqual(
            loss_standard.item(),
            loss_chunked.item(),
            places=5,
            msg="Chunked loss should handle non-divisible chunk sizes",
        )

    def test_repr(self):
        """Test string representation of loss classes."""
        standard_loss = CausalLoss(compile=False)
        chunked_loss = ChunkedCausalLoss(chunk_size=4096, compile=False)

        self.assertEqual(repr(standard_loss), "CausalLoss(compile=False)")
        self.assertEqual(
            repr(chunked_loss), "ChunkedCausalLoss(chunk_size=4096, compile=False)"
        )


class TestChunkedLossMemoryEfficiency(unittest.TestCase):
    """Test that chunked loss actually uses less memory (integration-style tests)."""

    @unittest.skipUnless(torch.cuda.is_available(), "Memory test requires CUDA")
    def test_chunked_works_with_large_vocab(self):
        """
        Test that chunked loss works correctly with very large vocabulary.

        Note: The memory benefits of chunked loss are most apparent in pipeline
        parallel training where activations accumulate across microbatches. This
        simple test just verifies correctness with Qwen3-sized vocabulary.
        """
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        # Large vocabulary to simulate Qwen3
        batch_size, seq_len, vocab_size = 2, 256, 151936
        logits = torch.randn(
            batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device=device
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Test that both losses produce similar results
        standard_loss = CausalLoss()
        chunked_loss = ChunkedCausalLoss(chunk_size=4096)

        loss_standard = standard_loss(logits.clone(), labels)
        loss_chunked = chunked_loss(logits.clone(), labels)

        # Verify they match (with tolerance for bfloat16)
        self.assertAlmostEqual(
            loss_standard.item(),
            loss_chunked.item(),
            places=3,
            msg=f"Losses should match: standard={loss_standard.item():.6f}, chunked={loss_chunked.item():.6f}",
        )

        # Test backward pass
        logits_grad = logits.clone().requires_grad_(True)
        loss_chunked_grad = chunked_loss(logits_grad, labels)
        loss_chunked_grad.backward()

        # Verify gradients were computed
        self.assertIsNotNone(
            logits_grad.grad, msg="Gradients should be computed for chunked loss"
        )


class TestLinearCrossEntropyLoss(unittest.TestCase):
    """Test LinearCrossEntropyLoss wrapper with different backends."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_test_model(self, hidden_dim=256, vocab_size=1000):
        """Create a simple output embeddings layer for testing."""
        import torch.nn as nn

        output_layer = nn.Linear(hidden_dim, vocab_size, bias=True, device=self.device)
        return output_layer

    def _create_test_data(
        self, batch_size=2, seq_len=16, hidden_dim=256, vocab_size=1000
    ):
        """Create test data for fused loss testing."""
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_dim, dtype=torch.float32, device=self.device
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        return hidden_states, labels

    def test_pytorch_backend_initialization(self):
        """Test that pytorch backend initializes correctly."""
        from forgather.ml.loss import LinearCrossEntropyLoss

        output_layer = self._create_test_model()
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="pytorch")

        self.assertEqual(loss_fn.actual_impl, "pytorch")
        self.assertIsNotNone(loss_fn.weight)
        self.assertTrue(hasattr(loss_fn, "forward_logits"))

    def test_pytorch_backend_forward(self):
        """Test forward pass with pytorch backend."""
        from forgather.ml.loss import LinearCrossEntropyLoss

        output_layer = self._create_test_model(hidden_dim=128, vocab_size=500)
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="pytorch", chunk_size=128)

        hidden_states, labels = self._create_test_data(
            batch_size=2, seq_len=16, hidden_dim=128, vocab_size=500
        )

        loss = loss_fn(hidden_states, labels)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_pytorch_backend_matches_standard(self):
        """Test that pytorch backend matches standard loss computation."""
        from forgather.ml.loss import LinearCrossEntropyLoss, CausalLoss

        hidden_dim, vocab_size = 128, 500
        output_layer = self._create_test_model(
            hidden_dim=hidden_dim, vocab_size=vocab_size
        )

        hidden_states, labels = self._create_test_data(
            batch_size=2, seq_len=16, hidden_dim=hidden_dim, vocab_size=vocab_size
        )

        # Standard approach: linear → logits → loss
        logits = output_layer(hidden_states)
        standard_loss_fn = CausalLoss()
        loss_standard = standard_loss_fn(logits, labels)

        # Fused approach: hidden states → loss directly
        fused_loss_fn = LinearCrossEntropyLoss(
            output_layer, impl="pytorch", chunk_size=128
        )
        loss_fused = fused_loss_fn(hidden_states, labels)

        # Should match within numerical precision
        self.assertAlmostEqual(
            loss_standard.item(),
            loss_fused.item(),
            places=5,
            msg="Fused loss should match standard loss",
        )

    def test_forward_logits_inference_mode(self):
        """Test that forward_logits works for inference."""
        from forgather.ml.loss import LinearCrossEntropyLoss

        hidden_dim, vocab_size = 128, 500
        output_layer = self._create_test_model(
            hidden_dim=hidden_dim, vocab_size=vocab_size
        )
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="pytorch")

        hidden_states, _ = self._create_test_data(
            batch_size=2, seq_len=16, hidden_dim=hidden_dim, vocab_size=vocab_size
        )

        # Test inference mode
        logits = loss_fn.forward_logits(hidden_states)

        self.assertEqual(logits.shape, (2, 16, vocab_size))
        self.assertFalse(torch.isnan(logits).any())

    def test_gradients_flow_correctly(self):
        """Test that gradients flow correctly through fused loss."""
        from forgather.ml.loss import LinearCrossEntropyLoss

        hidden_dim, vocab_size = 128, 500
        output_layer = self._create_test_model(
            hidden_dim=hidden_dim, vocab_size=vocab_size
        )
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="pytorch", chunk_size=128)

        hidden_states, labels = self._create_test_data(
            batch_size=2, seq_len=16, hidden_dim=hidden_dim, vocab_size=vocab_size
        )
        hidden_states.requires_grad = True

        loss = loss_fn(hidden_states, labels)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(output_layer.weight.grad)
        self.assertFalse(torch.isnan(hidden_states.grad).any())

    def test_auto_backend_fallback(self):
        """Test that auto backend selects available implementation."""
        from forgather.ml.loss import LinearCrossEntropyLoss

        output_layer = self._create_test_model()

        # Auto should fall back to pytorch since CCE/Liger likely not installed
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="auto")

        # Should have selected some backend
        self.assertIsNotNone(loss_fn.actual_impl)
        self.assertIn(loss_fn.actual_impl, ["cce", "liger", "pytorch"])

    @unittest.skipUnless(torch.cuda.is_available(), "CCE test requires CUDA")
    def test_cce_backend_if_available(self):
        """Test CCE backend if cut-cross-entropy is installed."""
        from forgather.ml.loss import LinearCrossEntropyLoss

        try:
            import cut_cross_entropy
        except ImportError:
            self.skipTest("cut-cross-entropy not installed")

        output_layer = self._create_test_model()
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="cce")

        self.assertEqual(loss_fn.actual_impl, "cce")

        hidden_states, labels = self._create_test_data()
        loss = loss_fn(hidden_states, labels)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))

    def test_repr(self):
        """Test string representation."""
        from forgather.ml.loss import LinearCrossEntropyLoss

        output_layer = self._create_test_model(hidden_dim=128, vocab_size=500)
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="pytorch", chunk_size=2048)

        repr_str = repr(loss_fn)
        self.assertIn("LinearCrossEntropyLoss", repr_str)
        self.assertIn("impl='pytorch'", repr_str)
        self.assertIn("vocab_size=500", repr_str)
        self.assertIn("hidden_dim=128", repr_str)

    def test_with_large_vocab(self):
        """Test with Qwen3-sized vocabulary."""
        if not torch.cuda.is_available():
            self.skipTest("Large vocab test requires CUDA")

        from forgather.ml.loss import LinearCrossEntropyLoss

        # Qwen3-like dimensions
        hidden_dim, vocab_size = 2048, 151936
        output_layer = self._create_test_model(
            hidden_dim=hidden_dim, vocab_size=vocab_size
        )
        loss_fn = LinearCrossEntropyLoss(output_layer, impl="pytorch", chunk_size=4096)

        # Smaller batch for memory constraints
        hidden_states, labels = self._create_test_data(
            batch_size=1, seq_len=64, hidden_dim=hidden_dim, vocab_size=vocab_size
        )
        hidden_states = hidden_states.to(torch.bfloat16)

        loss = loss_fn(hidden_states, labels)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
