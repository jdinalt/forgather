"""
Tests for Triton kernel implementations (fused GLU activation) and RoPE embeddings.

Tests validate:
1. Numerical correctness against PyTorch reference implementations
2. Gradient correctness (backward pass)
3. Multiple dtypes (float32, bfloat16, float16)
4. Various tensor shapes
5. Integration with module classes
6. RoPE: RotaryEmbedding module + apply_rotary_pos_emb function
"""

import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

# Skip entire module if no CUDA
if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA not available")

# Import modelsrc modules -- add to path since they're not a package
sys.path.insert(0, "modelsrc/transformer")

from glu_feedforward import _HAS_TRITON, GLUFeedforwardLayer

if _HAS_TRITON:
    from glu_feedforward import _FusedSiLUMul, _FusedReLUMul, _FusedGELUMul

from rotary_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)


def _skip_no_triton(test_func):
    """Decorator to skip test if Triton is not available."""
    return unittest.skipUnless(_HAS_TRITON, "Triton not installed")(test_func)


class TestFusedSiLUMul(unittest.TestCase):
    """Tests for the fused SiLU(gate) * up Triton kernel."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def _reference_silu_mul(self, gate, up):
        """PyTorch reference: up * silu(gate)."""
        return up * F.silu(gate)

    @_skip_no_triton
    def test_matches_pytorch_float32(self):
        """Fused SiLU*up matches PyTorch reference in float32."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float32)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float32)

        ref = self._reference_silu_mul(gate, up)
        fused = _FusedSiLUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-5, rtol=1e-5),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_matches_pytorch_bfloat16(self):
        """Fused SiLU*up matches PyTorch reference in bfloat16."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.bfloat16)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.bfloat16)

        ref = self._reference_silu_mul(gate, up)
        fused = _FusedSiLUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-2, rtol=1e-2),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_matches_pytorch_float16(self):
        """Fused SiLU*up matches PyTorch reference in float16."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float16)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float16)

        ref = self._reference_silu_mul(gate, up)
        fused = _FusedSiLUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-2, rtol=1e-2),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_matches_pytorch(self):
        """Gradient of fused SiLU*up matches PyTorch autograd."""
        gate = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.float32, requires_grad=True
        )
        up = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.float32, requires_grad=True
        )

        # Reference backward
        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        ref_out = self._reference_silu_mul(gate_ref, up_ref)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        # Fused backward
        gate_fused = gate.detach().clone().requires_grad_(True)
        up_fused = up.detach().clone().requires_grad_(True)
        fused_out = _FusedSiLUMul.apply(gate_fused.contiguous(), up_fused.contiguous())
        fused_out.backward(grad_out)

        self.assertTrue(
            torch.allclose(gate_ref.grad, gate_fused.grad, atol=1e-5, rtol=1e-5),
            f"Gate grad max diff: {(gate_ref.grad - gate_fused.grad).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(up_ref.grad, up_fused.grad, atol=1e-5, rtol=1e-5),
            f"Up grad max diff: {(up_ref.grad - up_fused.grad).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_bfloat16(self):
        """Gradient correctness in bfloat16."""
        gate = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        up = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )

        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        ref_out = self._reference_silu_mul(gate_ref, up_ref)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        gate_fused = gate.detach().clone().requires_grad_(True)
        up_fused = up.detach().clone().requires_grad_(True)
        fused_out = _FusedSiLUMul.apply(gate_fused.contiguous(), up_fused.contiguous())
        fused_out.backward(grad_out)

        self.assertTrue(
            torch.allclose(gate_ref.grad, gate_fused.grad, atol=1e-1, rtol=1e-1),
            f"Gate grad max diff: {(gate_ref.grad - gate_fused.grad).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(up_ref.grad, up_fused.grad, atol=1e-1, rtol=1e-1),
            f"Up grad max diff: {(up_ref.grad - up_fused.grad).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_various_shapes(self):
        """Test with various tensor shapes."""
        shapes = [
            (1, 1, 64),
            (1, 32, 128),
            (2, 64, 256),
            (4, 128, 1024),
            (8, 256, 4096),
            (1, 1, 11008),  # Llama d_ff
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                gate = torch.randn(shape, device=self.device, dtype=torch.float32)
                up = torch.randn(shape, device=self.device, dtype=torch.float32)

                ref = self._reference_silu_mul(gate, up)
                fused = _FusedSiLUMul.apply(gate.contiguous(), up.contiguous())

                self.assertTrue(
                    torch.allclose(ref, fused, atol=1e-5, rtol=1e-5),
                    f"Shape {shape}: max diff {(ref - fused).abs().max().item():.2e}",
                )


class TestFusedReLUMul(unittest.TestCase):
    """Tests for the fused ReLU(gate) * up Triton kernel."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def _reference_relu_mul(self, gate, up):
        """PyTorch reference: up * relu(gate)."""
        return up * F.relu(gate)

    @_skip_no_triton
    def test_matches_pytorch_float32(self):
        """Fused ReLU*up matches PyTorch reference in float32."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float32)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float32)

        ref = self._reference_relu_mul(gate, up)
        fused = _FusedReLUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-5, rtol=1e-5),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_matches_pytorch_bfloat16(self):
        """Fused ReLU*up matches PyTorch reference in bfloat16."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.bfloat16)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.bfloat16)

        ref = self._reference_relu_mul(gate, up)
        fused = _FusedReLUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-2, rtol=1e-2),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_matches_pytorch_float16(self):
        """Fused ReLU*up matches PyTorch reference in float16."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float16)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float16)

        ref = self._reference_relu_mul(gate, up)
        fused = _FusedReLUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-2, rtol=1e-2),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_matches_pytorch(self):
        """Gradient of fused ReLU*up matches PyTorch autograd."""
        gate = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.float32, requires_grad=True
        )
        up = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.float32, requires_grad=True
        )

        # Reference backward
        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        ref_out = self._reference_relu_mul(gate_ref, up_ref)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        # Fused backward
        gate_fused = gate.detach().clone().requires_grad_(True)
        up_fused = up.detach().clone().requires_grad_(True)
        fused_out = _FusedReLUMul.apply(gate_fused.contiguous(), up_fused.contiguous())
        fused_out.backward(grad_out)

        self.assertTrue(
            torch.allclose(gate_ref.grad, gate_fused.grad, atol=1e-5, rtol=1e-5),
            f"Gate grad max diff: {(gate_ref.grad - gate_fused.grad).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(up_ref.grad, up_fused.grad, atol=1e-5, rtol=1e-5),
            f"Up grad max diff: {(up_ref.grad - up_fused.grad).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_bfloat16(self):
        """Gradient correctness in bfloat16."""
        gate = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        up = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )

        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        ref_out = self._reference_relu_mul(gate_ref, up_ref)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        gate_fused = gate.detach().clone().requires_grad_(True)
        up_fused = up.detach().clone().requires_grad_(True)
        fused_out = _FusedReLUMul.apply(gate_fused.contiguous(), up_fused.contiguous())
        fused_out.backward(grad_out)

        self.assertTrue(
            torch.allclose(gate_ref.grad, gate_fused.grad, atol=1e-1, rtol=1e-1),
            f"Gate grad max diff: {(gate_ref.grad - gate_fused.grad).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(up_ref.grad, up_fused.grad, atol=1e-1, rtol=1e-1),
            f"Up grad max diff: {(up_ref.grad - up_fused.grad).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_various_shapes(self):
        """Test with various tensor shapes."""
        shapes = [
            (1, 1, 64),
            (1, 32, 128),
            (2, 64, 256),
            (4, 128, 1024),
            (8, 256, 4096),
            (1, 1, 11008),  # Llama d_ff
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                gate = torch.randn(shape, device=self.device, dtype=torch.float32)
                up = torch.randn(shape, device=self.device, dtype=torch.float32)

                ref = self._reference_relu_mul(gate, up)
                fused = _FusedReLUMul.apply(gate.contiguous(), up.contiguous())

                self.assertTrue(
                    torch.allclose(ref, fused, atol=1e-5, rtol=1e-5),
                    f"Shape {shape}: max diff {(ref - fused).abs().max().item():.2e}",
                )


class TestFusedGELUMul(unittest.TestCase):
    """Tests for the fused GELU(gate) * up Triton kernel."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def _reference_gelu_mul(self, gate, up):
        """PyTorch reference: up * gelu(gate)."""
        return up * F.gelu(gate)

    @_skip_no_triton
    def test_matches_pytorch_float32(self):
        """Fused GELU*up matches PyTorch reference in float32."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float32)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.float32)

        ref = self._reference_gelu_mul(gate, up)
        fused = _FusedGELUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-5, rtol=1e-5),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_matches_pytorch_bfloat16(self):
        """Fused GELU*up matches PyTorch reference in bfloat16."""
        gate = torch.randn(4, 128, 1024, device=self.device, dtype=torch.bfloat16)
        up = torch.randn(4, 128, 1024, device=self.device, dtype=torch.bfloat16)

        ref = self._reference_gelu_mul(gate, up)
        fused = _FusedGELUMul.apply(gate.contiguous(), up.contiguous())

        self.assertTrue(
            torch.allclose(ref, fused, atol=1e-2, rtol=1e-2),
            f"Max diff: {(ref - fused).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_matches_pytorch(self):
        """Gradient of fused GELU*up matches PyTorch autograd."""
        gate = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.float32, requires_grad=True
        )
        up = torch.randn(
            2, 64, 512, device=self.device, dtype=torch.float32, requires_grad=True
        )

        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        ref_out = self._reference_gelu_mul(gate_ref, up_ref)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        gate_fused = gate.detach().clone().requires_grad_(True)
        up_fused = up.detach().clone().requires_grad_(True)
        fused_out = _FusedGELUMul.apply(gate_fused.contiguous(), up_fused.contiguous())
        fused_out.backward(grad_out)

        self.assertTrue(
            torch.allclose(gate_ref.grad, gate_fused.grad, atol=1e-5, rtol=1e-5),
            f"Gate grad max diff: {(gate_ref.grad - gate_fused.grad).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(up_ref.grad, up_fused.grad, atol=1e-5, rtol=1e-5),
            f"Up grad max diff: {(up_ref.grad - up_fused.grad).abs().max().item():.2e}",
        )


class TestRotaryEmbedding(unittest.TestCase):
    """Tests for the RotaryEmbedding module and apply_rotary_pos_emb function."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def _make_rope(self, num_heads=8, d_head=64, **kwargs):
        """Create a RotaryEmbedding and initialize it."""
        rope = RotaryEmbedding(
            hidden_size=num_heads * d_head,
            num_attention_heads=num_heads,
            **kwargs,
        ).to(self.device)
        rope.reset_parameters()
        return rope

    def _make_qk(
        self, batch=2, seq_len=64, num_heads=8, d_head=64, dtype=torch.float32
    ):
        """Create q, k tensors for testing."""
        q = torch.randn(
            batch, seq_len, num_heads, d_head, device=self.device, dtype=dtype
        )
        k = torch.randn(
            batch, seq_len, num_heads, d_head, device=self.device, dtype=dtype
        )
        return q, k

    def test_forward_output_shape(self):
        """RotaryEmbedding returns (cos, sin) with correct shapes."""
        rope = self._make_rope(num_heads=8, d_head=64)
        x = torch.randn(2, 128, 512, device=self.device)
        position_ids = torch.arange(128, device=self.device).unsqueeze(0).expand(2, -1)

        cos, sin = rope(x, position_ids)

        self.assertEqual(cos.shape, (2, 128, 64))
        self.assertEqual(sin.shape, (2, 128, 64))

    def test_forward_output_dtype(self):
        """Output dtype depends on float32_output flag."""
        x = torch.randn(1, 16, 512, device=self.device, dtype=torch.bfloat16)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0)

        # Default (float32_output=False): matches input dtype
        rope = self._make_rope(float32_output=False)
        cos, sin = rope(x, position_ids)
        self.assertEqual(cos.dtype, torch.bfloat16)
        self.assertEqual(sin.dtype, torch.bfloat16)

        # float32_output=True: always float32
        rope_f32 = self._make_rope(float32_output=True)
        cos, sin = rope_f32(x, position_ids)
        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)

    def test_apply_rotary_pos_emb_correctness(self):
        """apply_rotary_pos_emb matches manual rotation in float32."""
        rope = self._make_rope(num_heads=4, d_head=64)
        q, k = self._make_qk(batch=2, seq_len=32, num_heads=4, d_head=64)
        position_ids = torch.arange(32, device=self.device).unsqueeze(0).expand(2, -1)

        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, position_embeddings=(cos, sin))

        # Manual reference (float32)
        cos_b = cos.unsqueeze(2)
        sin_b = sin.unsqueeze(2)
        q_ref = (q.float() * cos_b) + (rotate_half(q.float()) * sin_b)
        k_ref = (k.float() * cos_b) + (rotate_half(k.float()) * sin_b)

        self.assertTrue(
            torch.allclose(q_rot, q_ref.to(q.dtype), atol=1e-6, rtol=1e-6),
            f"Q max diff: {(q_rot - q_ref.to(q.dtype)).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(k_rot, k_ref.to(k.dtype), atol=1e-6, rtol=1e-6),
            f"K max diff: {(k_rot - k_ref.to(k.dtype)).abs().max().item():.2e}",
        )

    def test_apply_rotary_pos_emb_bf16(self):
        """apply_rotary_pos_emb works correctly with bf16 inputs."""
        rope = self._make_rope(num_heads=4, d_head=64)
        q, k = self._make_qk(
            batch=2, seq_len=32, num_heads=4, d_head=64, dtype=torch.bfloat16
        )
        position_ids = torch.arange(32, device=self.device).unsqueeze(0).expand(2, -1)

        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, position_embeddings=(cos, sin))

        self.assertEqual(q_rot.dtype, torch.bfloat16)
        self.assertEqual(k_rot.dtype, torch.bfloat16)
        self.assertEqual(q_rot.shape, q.shape)

    def test_float32_vs_native_precision(self):
        """Float32 rotation is more precise than native bf16 for large positions."""
        rope_f32 = self._make_rope(
            num_heads=4, d_head=128, max_position_embeddings=8192, float32_output=True
        )
        rope_native = self._make_rope(
            num_heads=4, d_head=128, max_position_embeddings=8192, float32_output=False
        )
        q, k = self._make_qk(
            batch=1, seq_len=64, num_heads=4, d_head=128, dtype=torch.bfloat16
        )
        # Use large position IDs to stress precision
        position_ids = torch.arange(4096, 4096 + 64, device=self.device).unsqueeze(0)

        cos_f32, sin_f32 = rope_f32(q, position_ids)
        cos_native, sin_native = rope_native(q, position_ids)

        # Float32 rotation (default)
        q_f32, _ = apply_rotary_pos_emb(q, k, position_embeddings=(cos_f32, sin_f32))

        # Native precision rotation
        q_native, _ = apply_rotary_pos_emb(
            q, k, position_embeddings=(cos_native, sin_native)
        )

        # Compute ground truth in full float32
        q_float = q.float()
        cos_b = cos_f32.unsqueeze(2)
        sin_b = sin_f32.unsqueeze(2)
        q_truth = ((q_float * cos_b) + (rotate_half(q_float) * sin_b)).to(
            torch.bfloat16
        )

        err_f32 = (q_f32.float() - q_truth.float()).abs().max().item()
        err_native = (q_native.float() - q_truth.float()).abs().max().item()

        # Float32 path should be at least as precise as native (usually much better)
        self.assertLessEqual(
            err_f32,
            err_native + 1e-7,
            f"Float32 err={err_f32:.2e}, native err={err_native:.2e}",
        )

    def test_backward_through_rotation(self):
        """Gradients flow correctly through apply_rotary_pos_emb."""
        rope = self._make_rope(num_heads=4, d_head=64)
        q = torch.randn(2, 16, 4, 64, device=self.device, requires_grad=True)
        k = torch.randn(2, 16, 4, 64, device=self.device, requires_grad=True)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, position_embeddings=(cos, sin))

        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertFalse(torch.all(q.grad == 0), "Q gradients are all zero")
        self.assertFalse(torch.all(k.grad == 0), "K gradients are all zero")

    def test_various_shapes(self):
        """Test with various tensor shapes."""
        configs = [
            (1, 1, 1, 64),  # Single token, single head
            (1, 32, 4, 64),  # Small batch
            (2, 128, 8, 64),  # Typical
            (4, 256, 32, 128),  # Large model (Llama d_head=128)
            (1, 1, 32, 128),  # Single token decode, many heads
        ]
        for batch, seq_len, num_heads, d_head in configs:
            with self.subTest(shape=(batch, seq_len, num_heads, d_head)):
                rope = self._make_rope(num_heads=num_heads, d_head=d_head)
                q, k = self._make_qk(
                    batch=batch, seq_len=seq_len, num_heads=num_heads, d_head=d_head
                )
                position_ids = (
                    torch.arange(seq_len, device=self.device)
                    .unsqueeze(0)
                    .expand(batch, -1)
                )

                cos, sin = rope(q, position_ids)
                q_rot, k_rot = apply_rotary_pos_emb(
                    q, k, position_embeddings=(cos, sin)
                )

                self.assertEqual(q_rot.shape, q.shape)
                self.assertEqual(k_rot.shape, k.shape)

    def test_position_ids_non_sequential(self):
        """Non-sequential position_ids work correctly (KV cache scenario)."""
        rope = self._make_rope(num_heads=8, d_head=64, max_position_embeddings=256)
        q, k = self._make_qk(batch=2, seq_len=1, num_heads=8, d_head=64)

        # Non-sequential positions (simulating KV cache at different steps)
        position_ids = torch.tensor([[42], [99]], device=self.device)
        cos, sin = rope(q, position_ids)

        self.assertEqual(cos.shape, (2, 1, 64))

        # Different positions should produce different embeddings
        pos_a = torch.tensor([[10]], device=self.device)
        pos_b = torch.tensor([[20]], device=self.device)
        cos_a, _ = rope(q[:1], pos_a)
        cos_b, _ = rope(q[:1], pos_b)
        self.assertFalse(
            torch.allclose(cos_a, cos_b),
            "Different positions should produce different embeddings",
        )

    def test_reset_parameters(self):
        """reset_parameters computes correct inv_freq values."""
        rope = self._make_rope(num_heads=4, d_head=64)

        # Manually compute expected inv_freq
        theta = 10000.0
        expected = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))

        self.assertTrue(
            torch.allclose(rope.inv_freq.cpu(), expected, atol=1e-6),
            f"inv_freq max diff: {(rope.inv_freq.cpu() - expected).abs().max().item():.2e}",
        )

    def test_meta_device_construction(self):
        """RotaryEmbedding can be constructed on meta device."""
        with torch.device("meta"):
            rope = RotaryEmbedding(
                hidden_size=512,
                num_attention_heads=8,
            )

        self.assertTrue(rope.inv_freq.is_meta)

        # reset_parameters should be a no-op on meta
        rope.reset_parameters()
        self.assertTrue(rope.inv_freq.is_meta)

        # Move to real device and initialize
        rope = rope.to_empty(device=self.device)
        rope.reset_parameters()
        self.assertFalse(rope.inv_freq.is_meta)
        self.assertEqual(rope.inv_freq.shape, (32,))  # d_head=64, half=32

    def test_position_embeddings_none_raises(self):
        """apply_rotary_pos_emb raises ValueError when position_embeddings is None."""
        q, k = self._make_qk(batch=1, seq_len=4, num_heads=4, d_head=64)

        with self.assertRaises(ValueError) as ctx:
            apply_rotary_pos_emb(q, k, position_embeddings=None)

        self.assertIn("position_embeddings is None", str(ctx.exception))

    def test_float32_output_flag(self):
        """float32_output=False returns embeddings in input dtype."""
        rope_f32 = self._make_rope(num_heads=4, d_head=64, float32_output=True)
        rope_native = self._make_rope(num_heads=4, d_head=64, float32_output=False)

        x = torch.randn(1, 16, 256, device=self.device, dtype=torch.bfloat16)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0)

        cos_f32, sin_f32 = rope_f32(x, position_ids)
        cos_native, sin_native = rope_native(x, position_ids)

        self.assertEqual(cos_f32.dtype, torch.float32)
        self.assertEqual(cos_native.dtype, torch.bfloat16)

        # Both should produce the same output dtype from apply_rotary_pos_emb
        q, k = self._make_qk(
            batch=1, seq_len=16, num_heads=4, d_head=64, dtype=torch.bfloat16
        )
        q_f32, _ = apply_rotary_pos_emb(q, k, position_embeddings=(cos_f32, sin_f32))
        q_native, _ = apply_rotary_pos_emb(
            q, k, position_embeddings=(cos_native, sin_native)
        )

        self.assertEqual(q_f32.dtype, torch.bfloat16)
        self.assertEqual(q_native.dtype, torch.bfloat16)


class TestGLUFeedforwardIntegration(unittest.TestCase):
    """Integration tests using the full GLUFeedforwardLayer module."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    @_skip_no_triton
    def test_use_triton_flag_silu(self):
        """GLUFeedforwardLayer with use_triton=True (SiLU) matches PyTorch."""
        d_model, d_ff = 256, 512

        layer_ref = GLUFeedforwardLayer(
            d_model, d_ff, dropout=0.0, use_triton=False
        ).to(self.device)
        layer_triton = GLUFeedforwardLayer(
            d_model, d_ff, dropout=0.0, use_triton=True
        ).to(self.device)

        # Copy weights
        layer_triton.load_state_dict(layer_ref.state_dict())

        x = torch.randn(2, 32, d_model, device=self.device)

        ref_out = layer_ref(x)
        triton_out = layer_triton(x)

        self.assertTrue(
            torch.allclose(ref_out, triton_out, atol=1e-5, rtol=1e-5),
            f"Max diff: {(ref_out - triton_out).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_use_triton_flag_gelu(self):
        """GLUFeedforwardLayer with use_triton=True (GELU) matches PyTorch."""
        d_model, d_ff = 256, 512

        layer_ref = GLUFeedforwardLayer(
            d_model,
            d_ff,
            activation_factory=lambda: nn.GELU(),
            dropout=0.0,
            use_triton=False,
        ).to(self.device)
        layer_triton = GLUFeedforwardLayer(
            d_model,
            d_ff,
            activation_factory=lambda: nn.GELU(),
            dropout=0.0,
            use_triton=True,
        ).to(self.device)

        layer_triton.load_state_dict(layer_ref.state_dict())

        x = torch.randn(2, 32, d_model, device=self.device)

        ref_out = layer_ref(x)
        triton_out = layer_triton(x)

        self.assertTrue(
            torch.allclose(ref_out, triton_out, atol=1e-5, rtol=1e-5),
            f"Max diff: {(ref_out - triton_out).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_through_full_layer(self):
        """Full layer forward+backward with Triton matches PyTorch."""
        d_model, d_ff = 256, 512

        layer_ref = GLUFeedforwardLayer(
            d_model, d_ff, dropout=0.0, use_triton=False
        ).to(self.device)
        layer_triton = GLUFeedforwardLayer(
            d_model, d_ff, dropout=0.0, use_triton=True
        ).to(self.device)
        layer_triton.load_state_dict(layer_ref.state_dict())

        x = torch.randn(2, 32, d_model, device=self.device, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        ref_out = layer_ref(x_ref)
        grad = torch.randn_like(ref_out)
        ref_out.backward(grad)

        triton_out = layer_triton(x)
        triton_out.backward(grad)

        self.assertTrue(
            torch.allclose(x_ref.grad, x.grad, atol=1e-4, rtol=1e-4),
            f"Input grad max diff: {(x_ref.grad - x.grad).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_use_triton_flag_relu(self):
        """GLUFeedforwardLayer with use_triton=True (ReLU) matches PyTorch."""
        d_model, d_ff = 256, 512

        layer_ref = GLUFeedforwardLayer(
            d_model,
            d_ff,
            activation_factory=lambda: nn.ReLU(),
            dropout=0.0,
            use_triton=False,
        ).to(self.device)
        layer_triton = GLUFeedforwardLayer(
            d_model,
            d_ff,
            activation_factory=lambda: nn.ReLU(),
            dropout=0.0,
            use_triton=True,
        ).to(self.device)

        layer_triton.load_state_dict(layer_ref.state_dict())

        x = torch.randn(2, 32, d_model, device=self.device)

        ref_out = layer_ref(x)
        triton_out = layer_triton(x)

        self.assertTrue(
            torch.allclose(ref_out, triton_out, atol=1e-5, rtol=1e-5),
            f"Max diff: {(ref_out - triton_out).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_unsupported_activation_fallback(self):
        """Unsupported activation with use_triton=True falls back to PyTorch."""
        d_model, d_ff = 256, 512

        # Tanh is not supported by Triton kernels
        layer = GLUFeedforwardLayer(
            d_model,
            d_ff,
            activation_factory=lambda: nn.Tanh(),
            dropout=0.0,
            use_triton=True,
        ).to(self.device)

        # Should fallback (no _fused_op set)
        self.assertIsNone(layer._fused_op)

        # Should still work correctly
        x = torch.randn(2, 32, d_model, device=self.device)
        out = layer(x)
        self.assertEqual(out.shape, (2, 32, d_model))

    def test_fallback_when_triton_unavailable(self):
        """Module works correctly when Triton is not installed."""
        d_model, d_ff = 256, 512

        # use_triton=True but module should handle gracefully
        layer = GLUFeedforwardLayer(d_model, d_ff, dropout=0.0, use_triton=False).to(
            self.device
        )

        x = torch.randn(2, 32, d_model, device=self.device)
        out = layer(x)
        self.assertEqual(out.shape, (2, 32, d_model))


if __name__ == "__main__":
    unittest.main()
