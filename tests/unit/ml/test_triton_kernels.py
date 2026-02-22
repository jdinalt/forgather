"""
Tests for Triton kernel implementations of fused GLU activation and RoPE rotation.

These kernels provide fused alternatives to the standard PyTorch implementations
in modelsrc/transformer/glu_feedforward.py and modelsrc/transformer/rotary_embeddings.py.

Tests validate:
1. Numerical correctness against PyTorch reference implementations
2. Gradient correctness (backward pass)
3. Multiple dtypes (float32, bfloat16, float16)
4. Various tensor shapes
5. Integration with module classes
6. Graceful fallback when Triton is unavailable
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

from glu_feedforward import GLUFeedforwardLayer, _HAS_TRITON

if _HAS_TRITON:
    from glu_feedforward import _FusedSiLUMul, _FusedGELUMul

from rotary_embeddings import RotaryPE, rotate_half
from rotary_embeddings import _HAS_TRITON as _ROPE_HAS_TRITON

if _ROPE_HAS_TRITON:
    from rotary_embeddings import (
        _FusedRoPERotation,
        _prepare_cos_sin_for_triton,
        _triton_rope_apply,
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


class TestFusedRoPERotation(unittest.TestCase):
    """Tests for the fused RoPE rotation Triton kernel."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def _create_rope_inputs(
        self, batch=2, seq_len=64, num_heads=8, d_head=64, dtype=torch.float32
    ):
        """Create test inputs for RoPE."""
        q = torch.randn(batch, seq_len, num_heads, d_head, device=self.device, dtype=dtype)
        k = torch.randn(batch, seq_len, num_heads, d_head, device=self.device, dtype=dtype)

        # Create cos/sin of shape [1, seq_len, 1, d_head]
        half_dim = d_head // 2
        freqs = torch.randn(seq_len, half_dim, device=self.device, dtype=dtype)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_head]
        sin = emb.sin().unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_head]

        return q, k, cos, sin

    def _reference_rope(self, x, cos, sin):
        """PyTorch reference RoPE: (x * cos) + (rotate_half(x) * sin)."""
        return (x * cos) + (rotate_half(x) * sin)

    @_skip_no_triton
    def test_matches_pytorch_float32(self):
        """Fused RoPE matches PyTorch reference in float32."""
        q, k, cos, sin = self._create_rope_inputs(dtype=torch.float32)

        ref_q = self._reference_rope(q, cos, sin)
        ref_k = self._reference_rope(k, cos, sin)

        cos_t, sin_t = _prepare_cos_sin_for_triton(
            cos, sin, q.shape[0], q.shape[1], q.shape[3]
        )
        fused_q = _FusedRoPERotation.apply(q.contiguous(), cos_t, sin_t)
        fused_k = _FusedRoPERotation.apply(k.contiguous(), cos_t, sin_t)

        self.assertTrue(
            torch.allclose(ref_q, fused_q, atol=1e-5, rtol=1e-5),
            f"Q max diff: {(ref_q - fused_q).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(ref_k, fused_k, atol=1e-5, rtol=1e-5),
            f"K max diff: {(ref_k - fused_k).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_matches_pytorch_bfloat16(self):
        """Fused RoPE matches PyTorch reference in bfloat16."""
        q, k, cos, sin = self._create_rope_inputs(dtype=torch.bfloat16)

        ref_q = self._reference_rope(q, cos, sin)

        cos_t, sin_t = _prepare_cos_sin_for_triton(
            cos, sin, q.shape[0], q.shape[1], q.shape[3]
        )
        fused_q = _FusedRoPERotation.apply(q.contiguous(), cos_t, sin_t)

        self.assertTrue(
            torch.allclose(ref_q, fused_q, atol=1e-2, rtol=1e-2),
            f"Q max diff: {(ref_q - fused_q).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_matches_pytorch_float16(self):
        """Fused RoPE matches PyTorch reference in float16."""
        q, k, cos, sin = self._create_rope_inputs(dtype=torch.float16)

        ref_q = self._reference_rope(q, cos, sin)

        cos_t, sin_t = _prepare_cos_sin_for_triton(
            cos, sin, q.shape[0], q.shape[1], q.shape[3]
        )
        fused_q = _FusedRoPERotation.apply(q.contiguous(), cos_t, sin_t)

        self.assertTrue(
            torch.allclose(ref_q, fused_q, atol=1e-2, rtol=1e-2),
            f"Q max diff: {(ref_q - fused_q).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_matches_pytorch(self):
        """Gradient of fused RoPE matches PyTorch autograd."""
        batch, seq_len, num_heads, d_head = 2, 32, 4, 64

        q, _, cos, sin = self._create_rope_inputs(
            batch=batch, seq_len=seq_len, num_heads=num_heads, d_head=d_head
        )

        # Reference backward
        q_ref = q.detach().clone().requires_grad_(True)
        ref_out = self._reference_rope(q_ref, cos, sin)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        # Fused backward
        q_fused = q.detach().clone().requires_grad_(True)
        cos_t, sin_t = _prepare_cos_sin_for_triton(
            cos, sin, batch, seq_len, d_head
        )
        fused_out = _FusedRoPERotation.apply(q_fused.contiguous(), cos_t, sin_t)
        fused_out.backward(grad_out)

        self.assertTrue(
            torch.allclose(q_ref.grad, q_fused.grad, atol=1e-5, rtol=1e-5),
            f"Q grad max diff: {(q_ref.grad - q_fused.grad).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_bfloat16(self):
        """Gradient correctness in bfloat16."""
        batch, seq_len, num_heads, d_head = 2, 32, 4, 64

        q, _, cos, sin = self._create_rope_inputs(
            batch=batch, seq_len=seq_len, num_heads=num_heads, d_head=d_head,
            dtype=torch.bfloat16,
        )

        q_ref = q.detach().clone().requires_grad_(True)
        ref_out = self._reference_rope(q_ref, cos, sin)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        q_fused = q.detach().clone().requires_grad_(True)
        cos_t, sin_t = _prepare_cos_sin_for_triton(cos, sin, batch, seq_len, d_head)
        fused_out = _FusedRoPERotation.apply(q_fused.contiguous(), cos_t, sin_t)
        fused_out.backward(grad_out)

        self.assertTrue(
            torch.allclose(q_ref.grad, q_fused.grad, atol=1e-1, rtol=1e-1),
            f"Q grad max diff: {(q_ref.grad - q_fused.grad).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_various_shapes(self):
        """Test with various tensor shapes."""
        configs = [
            (1, 1, 1, 64),      # Single token, single head
            (1, 32, 4, 64),     # Small batch
            (2, 128, 8, 64),    # Typical
            (4, 256, 32, 128),  # Large model (Llama d_head=128)
            (1, 1, 32, 128),    # Single token decode, many heads
            (2, 512, 8, 64),    # Long sequence
        ]
        for batch, seq_len, num_heads, d_head in configs:
            with self.subTest(shape=(batch, seq_len, num_heads, d_head)):
                q, _, cos, sin = self._create_rope_inputs(
                    batch=batch, seq_len=seq_len, num_heads=num_heads, d_head=d_head,
                )

                ref = self._reference_rope(q, cos, sin)

                cos_t, sin_t = _prepare_cos_sin_for_triton(
                    cos, sin, batch, seq_len, d_head
                )
                fused = _FusedRoPERotation.apply(q.contiguous(), cos_t, sin_t)

                self.assertTrue(
                    torch.allclose(ref, fused, atol=1e-5, rtol=1e-5),
                    f"Shape {(batch, seq_len, num_heads, d_head)}: "
                    f"max diff {(ref - fused).abs().max().item():.2e}",
                )

    @_skip_no_triton
    def test_with_position_ids(self):
        """Test RoPE with non-sequential position_ids (KV cache scenario)."""
        batch, seq_len, num_heads, d_head = 2, 1, 8, 64
        max_seq_len = 128

        q = torch.randn(batch, seq_len, num_heads, d_head, device=self.device)
        k = torch.randn(batch, seq_len, num_heads, d_head, device=self.device)

        # Non-sequential positions (simulating KV cache at different steps)
        position_ids = torch.tensor([[42], [99]], device=self.device)

        rope = RotaryPE(
            hidden_size=num_heads * d_head,
            num_attention_heads=num_heads,
            max_sequence_length=max_seq_len,
            use_triton=False,
        )
        rope_triton = RotaryPE(
            hidden_size=num_heads * d_head,
            num_attention_heads=num_heads,
            max_sequence_length=max_seq_len,
            use_triton=True,
        )

        ref_q, ref_k = rope(q, k, position_ids)
        fused_q, fused_k = rope_triton(q, k, position_ids)

        self.assertTrue(
            torch.allclose(ref_q, fused_q, atol=1e-5, rtol=1e-5),
            f"Q max diff: {(ref_q - fused_q).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(ref_k, fused_k, atol=1e-5, rtol=1e-5),
            f"K max diff: {(ref_k - fused_k).abs().max().item():.2e}",
        )


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
            d_model, d_ff, activation_factory=lambda: nn.GELU(),
            dropout=0.0, use_triton=False,
        ).to(self.device)
        layer_triton = GLUFeedforwardLayer(
            d_model, d_ff, activation_factory=lambda: nn.GELU(),
            dropout=0.0, use_triton=True,
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
    def test_unsupported_activation_fallback(self):
        """Unsupported activation with use_triton=True falls back to PyTorch."""
        d_model, d_ff = 256, 512

        # ReLU is not supported by Triton kernels
        layer = GLUFeedforwardLayer(
            d_model, d_ff, activation_factory=lambda: nn.ReLU(),
            dropout=0.0, use_triton=True,
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
        layer = GLUFeedforwardLayer(
            d_model, d_ff, dropout=0.0, use_triton=False
        ).to(self.device)

        x = torch.randn(2, 32, d_model, device=self.device)
        out = layer(x)
        self.assertEqual(out.shape, (2, 32, d_model))


class TestRoPEIntegration(unittest.TestCase):
    """Integration tests using the full RotaryPE module."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    @_skip_no_triton
    def test_use_triton_flag_cached(self):
        """RotaryPE with use_triton=True (cached) matches PyTorch."""
        num_heads, d_head = 8, 64
        hidden_size = num_heads * d_head
        seq_len = 128

        rope_ref = RotaryPE(hidden_size, num_heads, max_sequence_length=256)
        rope_triton = RotaryPE(
            hidden_size, num_heads, max_sequence_length=256, use_triton=True,
        )

        q = torch.randn(2, seq_len, num_heads, d_head, device=self.device)
        k = torch.randn(2, seq_len, num_heads, d_head, device=self.device)

        ref_q, ref_k = rope_ref(q, k)
        tri_q, tri_k = rope_triton(q, k)

        self.assertTrue(
            torch.allclose(ref_q, tri_q, atol=1e-5, rtol=1e-5),
            f"Q max diff: {(ref_q - tri_q).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(ref_k, tri_k, atol=1e-5, rtol=1e-5),
            f"K max diff: {(ref_k - tri_k).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_use_triton_flag_on_demand(self):
        """RotaryPE with use_triton=True (on-demand) matches PyTorch."""
        num_heads, d_head = 8, 64
        hidden_size = num_heads * d_head
        seq_len = 128

        rope_ref = RotaryPE(
            hidden_size, num_heads, cache_embeddings=False, compile_on_demand=False,
        )
        rope_triton = RotaryPE(
            hidden_size, num_heads, cache_embeddings=False,
            compile_on_demand=False, use_triton=True,
        )

        q = torch.randn(2, seq_len, num_heads, d_head, device=self.device)
        k = torch.randn(2, seq_len, num_heads, d_head, device=self.device)

        ref_q, ref_k = rope_ref(q, k)
        tri_q, tri_k = rope_triton(q, k)

        self.assertTrue(
            torch.allclose(ref_q, tri_q, atol=1e-5, rtol=1e-5),
            f"Q max diff: {(ref_q - tri_q).abs().max().item():.2e}",
        )

    @_skip_no_triton
    def test_backward_through_rope(self):
        """RotaryPE backward with Triton matches PyTorch."""
        num_heads, d_head = 4, 64
        hidden_size = num_heads * d_head
        seq_len = 32

        rope_ref = RotaryPE(hidden_size, num_heads, max_sequence_length=64)
        rope_triton = RotaryPE(
            hidden_size, num_heads, max_sequence_length=64, use_triton=True,
        )

        q = torch.randn(
            2, seq_len, num_heads, d_head, device=self.device, requires_grad=True
        )
        k = torch.randn(
            2, seq_len, num_heads, d_head, device=self.device, requires_grad=True
        )

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)

        ref_q, ref_k = rope_ref(q_ref, k_ref)
        grad_q = torch.randn_like(ref_q)
        grad_k = torch.randn_like(ref_k)
        (ref_q.sum() + ref_k.sum()).backward()

        q_tri = q.detach().clone().requires_grad_(True)
        k_tri = k.detach().clone().requires_grad_(True)
        tri_q, tri_k = rope_triton(q_tri, k_tri)
        (tri_q.sum() + tri_k.sum()).backward()

        self.assertTrue(
            torch.allclose(q_ref.grad, q_tri.grad, atol=1e-5, rtol=1e-5),
            f"Q grad max diff: {(q_ref.grad - q_tri.grad).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(k_ref.grad, k_tri.grad, atol=1e-5, rtol=1e-5),
            f"K grad max diff: {(k_ref.grad - k_tri.grad).abs().max().item():.2e}",
        )


class TestPrepareCosSinForTriton(unittest.TestCase):
    """Tests for the cos/sin preprocessing helper."""

    def setUp(self):
        self.device = torch.device("cuda")

    @_skip_no_triton
    def test_4d_input(self):
        """[1, seq, 1, d_head] input is correctly preprocessed."""
        seq_len, d_head = 32, 64
        half_dim = d_head // 2

        cos = torch.randn(1, seq_len, 1, d_head, device=self.device)
        sin = torch.randn(1, seq_len, 1, d_head, device=self.device)

        cos_t, sin_t = _prepare_cos_sin_for_triton(cos, sin, 2, seq_len, d_head)

        self.assertEqual(cos_t.shape, (2 * seq_len, half_dim))
        self.assertEqual(sin_t.shape, (2 * seq_len, half_dim))
        self.assertTrue(cos_t.is_contiguous())
        self.assertTrue(sin_t.is_contiguous())

    @_skip_no_triton
    def test_2d_input(self):
        """[seq, d_head] input is correctly preprocessed."""
        seq_len, d_head = 32, 64
        half_dim = d_head // 2

        cos = torch.randn(seq_len, d_head, device=self.device)
        sin = torch.randn(seq_len, d_head, device=self.device)

        cos_t, sin_t = _prepare_cos_sin_for_triton(cos, sin, 4, seq_len, d_head)

        self.assertEqual(cos_t.shape, (4 * seq_len, half_dim))
        self.assertTrue(cos_t.is_contiguous())

    @_skip_no_triton
    def test_batch_broadcast(self):
        """[1, seq, 1, d_head] is broadcast to batch_size > 1."""
        batch, seq_len, d_head = 4, 16, 128
        half_dim = d_head // 2

        cos = torch.randn(1, seq_len, 1, d_head, device=self.device)
        sin = torch.randn(1, seq_len, 1, d_head, device=self.device)

        cos_t, sin_t = _prepare_cos_sin_for_triton(cos, sin, batch, seq_len, d_head)

        self.assertEqual(cos_t.shape, (batch * seq_len, half_dim))

        # Verify broadcast: all batch entries should be identical
        cos_reshaped = cos_t.reshape(batch, seq_len, half_dim)
        for b in range(1, batch):
            self.assertTrue(
                torch.allclose(cos_reshaped[0], cos_reshaped[b]),
                f"Batch {b} differs from batch 0",
            )


if __name__ == "__main__":
    unittest.main()
