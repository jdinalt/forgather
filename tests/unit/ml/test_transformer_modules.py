"""
Unit tests for modelsrc/transformer/ modules.

Tests cover the core transformer building blocks: layers, stacks,
feedforward, positional encodings, loss, attention, init, and the
CasualLM wrapper.

Note: Triton kernel tests are in test_triton_kernels.py.
"""

import math
import sys
import unittest

import torch
import torch.nn as nn

# modelsrc/transformer is not a package; add to path for direct import
sys.path.insert(0, "modelsrc/transformer")

from causal_loss import CausalLoss
from causal_mask import causal_mask
from checkpoint_layer_stack import LayerStack as CheckpointLayerStack
from deepnet import DeepnetLayer, deepnet_alpha, deepnet_beta
from eager_attention import eager_scaled_dot_product_attention
from explicit_layer_stack import ExplicitLayerStack
from feedforward_layer import FeedforwardLayer
from glu_feedforward import GLUFeedforwardLayer
from init_weights import (
    has_local_state,
    init_embeddings,
    init_torch_linear_default,
    init_weights_by_regex,
    simple_weight_init,
)
from input_encoder import InputEncoder
from layer_stack import LayerStack
from null_pe import NullPE
from post_ln_layer import PostLNLayer
from pre_ln_layer import PreLNLayer
from sinusoidal_pe import SinusoidalPE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_attention_factory(**kwargs):
    """Return a simple linear layer that acts as a stand-in for attention."""
    d_model = kwargs.get("d_model", 32)
    return nn.Linear(d_model, d_model)


def _dummy_feedforward_factory(**kwargs):
    """Return a simple linear layer that acts as a stand-in for feedforward."""
    d_model = kwargs.get("d_model", 32)
    return nn.Linear(d_model, d_model)


def _dummy_norm_factory():
    return nn.LayerNorm(32)


def _make_layer_kwargs():
    return dict(
        feedforward_factory=_dummy_feedforward_factory,
        attention_factory=_dummy_attention_factory,
        norm_factory=_dummy_norm_factory,
        d_model=32,
        dropout=0.0,
        residual_dropout=0.0,
    )


# ===================================================================
# CausalLoss
# ===================================================================


class TestCausalLossModelsrc(unittest.TestCase):
    """Tests for modelsrc/transformer/causal_loss.py."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    def test_basic_loss_computation(self):
        """CausalLoss shifts logits and labels correctly."""
        vocab_size = 100
        logits = torch.randn(2, 8, vocab_size)
        labels = torch.randint(0, vocab_size, (2, 8))

        loss_fn = CausalLoss()
        loss = loss_fn(logits, labels)

        self.assertEqual(loss.shape, ())
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_shift_alignment(self):
        """Verify that shifting means token n predicts token n+1."""
        vocab_size = 10
        # Construct logits where position i has high confidence for label i+1
        logits = torch.full((1, 4, vocab_size), -10.0)
        labels = torch.tensor([[0, 1, 2, 3]])
        # Make shifted logits[i] predict labels[i+1]
        for i in range(3):
            logits[0, i, labels[0, i + 1]] = 10.0

        loss = CausalLoss()(logits, labels)
        # Should be very small since predictions match shifted labels
        self.assertLess(loss.item(), 0.1)

    def test_ignore_index(self):
        """When all shifted labels are -100, loss is NaN (cross_entropy behavior)."""
        vocab_size = 50
        logits = torch.randn(1, 8, vocab_size)
        labels = torch.full((1, 8), -100, dtype=torch.long)

        loss = CausalLoss()(logits, labels)
        # cross_entropy with all ignored labels and reduction='mean' returns NaN
        self.assertTrue(torch.isnan(loss))

    def test_repr(self):
        self.assertEqual(repr(CausalLoss()), "CausalLoss()")


# ===================================================================
# Positional Encodings
# ===================================================================


class TestNullPE(unittest.TestCase):
    """Tests for NullPE - passthrough positional encoder."""

    def test_passthrough(self):
        pe = NullPE()
        x = torch.randn(2, 16, 64)
        out = pe(x)
        self.assertTrue(torch.equal(x, out))

    def test_with_position_ids(self):
        pe = NullPE()
        x = torch.randn(2, 16, 64)
        pos = torch.arange(16).unsqueeze(0).expand(2, -1)
        out = pe(x, position_ids=pos)
        self.assertTrue(torch.equal(x, out))

    def test_resize_and_get(self):
        pe = NullPE()
        pe.resize_position_embeddings(1024)  # no-op
        self.assertIsNone(pe.get_position_embeddings())


class TestSinusoidalPE(unittest.TestCase):
    """Tests for SinusoidalPE."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.max_seq_len = 128
        self.pe = SinusoidalPE(self.d_model, self.max_seq_len)

    def test_output_shape(self):
        x = torch.zeros(2, 16, self.d_model)
        out = self.pe(x)
        self.assertEqual(out.shape, x.shape)

    def test_adds_positional_info(self):
        """Output differs from input (positions were added)."""
        x = torch.zeros(1, 8, self.d_model)
        out = self.pe(x)
        self.assertFalse(torch.all(out == 0))

    def test_different_positions_give_different_embeddings(self):
        """Each position has a unique encoding."""
        x = torch.zeros(1, self.max_seq_len, self.d_model)
        out = self.pe(x)
        # Check first two positions differ
        self.assertFalse(torch.allclose(out[0, 0], out[0, 1]))

    def test_with_position_ids(self):
        """Using explicit position_ids selects correct embeddings."""
        x = torch.zeros(1, 2, self.d_model)
        pos = torch.tensor([[5, 10]])
        out = self.pe(x, position_ids=pos)
        # Should equal the embedding at those positions
        expected = self.pe.weight[pos]
        self.assertTrue(torch.allclose(out, expected))

    @unittest.expectedFailure
    def test_resize_fails(self):
        """SinusoidalPE.resize_position_embeddings does not resize the buffer.

        Bug: resize_position_embeddings updates max_sequence_length and calls
        reset_parameters(), but reset_parameters writes to self.weight[:, ...]
        using the NEW max_sequence_length against the OLD buffer size, causing
        a shape mismatch RuntimeError.
        """
        self.pe.resize_position_embeddings(256)

    def test_get_position_embeddings(self):
        w = self.pe.get_position_embeddings()
        self.assertEqual(w.shape, (self.max_seq_len, self.d_model))

    def test_extra_repr(self):
        r = self.pe.extra_repr()
        self.assertIn("d_model=64", r)
        self.assertIn("max_sequence_length=128", r)


# ===================================================================
# InputEncoder
# ===================================================================


class TestInputEncoder(unittest.TestCase):
    """Tests for InputEncoder."""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 100
        self.d_model = 64
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def test_basic_forward(self):
        enc = InputEncoder(self.d_model, self.embedding, dropout=0.0)
        ids = torch.randint(0, self.vocab_size, (2, 16))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 16, self.d_model))

    def test_with_positional_encoder(self):
        pe = SinusoidalPE(self.d_model, 256)
        enc = InputEncoder(
            self.d_model, self.embedding, positional_encoder=pe, dropout=0.0
        )
        ids = torch.randint(0, self.vocab_size, (2, 16))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 16, self.d_model))

    def test_scale_sqrt_d_model(self):
        enc = InputEncoder(
            self.d_model, self.embedding, scale_sqrt_d_model=True, dropout=0.0
        )
        ids = torch.randint(0, self.vocab_size, (1, 4))

        # Output should be embedding * sqrt(d_model)
        expected = self.embedding(ids) * math.sqrt(self.d_model)
        out = enc(ids)
        self.assertTrue(torch.allclose(out, expected))

    def test_get_set_input_embeddings(self):
        enc = InputEncoder(self.d_model, self.embedding, dropout=0.0)
        self.assertIs(enc.get_input_embeddings(), self.embedding)

        new_emb = nn.Embedding(50, self.d_model)
        enc.set_input_embeddings(new_emb)
        self.assertIs(enc.get_input_embeddings(), new_emb)

    def test_no_positional_encoder(self):
        enc = InputEncoder(self.d_model, self.embedding, dropout=0.0)
        self.assertIsNone(enc.positional_encoder)
        self.assertIsNone(enc.get_position_embeddings())


# ===================================================================
# Feedforward Layers
# ===================================================================


class TestFeedforwardLayer(unittest.TestCase):
    """Tests for FeedforwardLayer."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        ff = FeedforwardLayer(64, 256, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out = ff(x)
        self.assertEqual(out.shape, (2, 16, 64))

    def test_with_different_activations(self):
        for act_factory in [lambda: nn.ReLU(), lambda: nn.GELU(), lambda: nn.SiLU()]:
            ff = FeedforwardLayer(32, 128, activation_factory=act_factory, dropout=0.0)
            x = torch.randn(1, 4, 32)
            out = ff(x)
            self.assertEqual(out.shape, (1, 4, 32))

    def test_no_bias(self):
        ff = FeedforwardLayer(32, 128, bias=False, dropout=0.0)
        self.assertIsNone(ff.linear1.bias)
        self.assertIsNone(ff.linear2.bias)

    def test_init_prefix_set(self):
        ff = FeedforwardLayer(32, 128)
        self.assertEqual(getattr(ff.linear1, "init_prefix"), "ff.up_proj")
        self.assertEqual(getattr(ff.linear2, "init_prefix"), "ff.down_proj")

    def test_extra_repr(self):
        ff = FeedforwardLayer(32, 128)
        r = ff.extra_repr()
        self.assertIn("d_model=32", r)
        self.assertIn("d_feedforward=128", r)

    def test_gradient_flow(self):
        ff = FeedforwardLayer(32, 64, dropout=0.0)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = ff(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestGLUFeedforwardLayer(unittest.TestCase):
    """Tests for GLUFeedforwardLayer (CPU, no Triton)."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        ff = GLUFeedforwardLayer(64, 256, dropout=0.0, use_triton=False)
        x = torch.randn(2, 16, 64)
        out = ff(x)
        self.assertEqual(out.shape, (2, 16, 64))

    def test_default_silu_activation(self):
        ff = GLUFeedforwardLayer(32, 128, dropout=0.0, use_triton=False)
        self.assertIsInstance(ff.activation, nn.SiLU)

    def test_custom_activation(self):
        ff = GLUFeedforwardLayer(
            32, 128, activation_factory=lambda: nn.GELU(), dropout=0.0, use_triton=False
        )
        self.assertIsInstance(ff.activation, nn.GELU)

    def test_init_prefixes(self):
        ff = GLUFeedforwardLayer(32, 128, use_triton=False)
        self.assertEqual(getattr(ff.up_proj, "init_prefix"), "ff.up_proj")
        self.assertEqual(getattr(ff.gate_proj, "init_prefix"), "ff.gate_proj")
        self.assertEqual(getattr(ff.down_proj, "init_prefix"), "ff.down_proj")

    def test_gradient_flow(self):
        ff = GLUFeedforwardLayer(32, 64, dropout=0.0, use_triton=False)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = ff(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_extra_repr(self):
        ff = GLUFeedforwardLayer(32, 128, use_triton=False)
        r = ff.extra_repr()
        self.assertIn("d_model=32", r)
        self.assertIn("d_feedforward=128", r)


# ===================================================================
# Transformer Layer Variants
# ===================================================================


class TestPreLNLayer(unittest.TestCase):
    """Tests for PreLNLayer."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        layer = PreLNLayer(**_make_layer_kwargs())
        x = torch.randn(2, 8, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_no_dropout(self):
        """dropout=None and dropout=0.0 both result in Identity."""
        for d in [None, 0.0, 0]:
            layer = PreLNLayer(**{**_make_layer_kwargs(), "dropout": d})
            self.assertIsInstance(layer.dropout, nn.Identity)

    def test_with_dropout(self):
        layer = PreLNLayer(**{**_make_layer_kwargs(), "dropout": 0.5})
        self.assertIsInstance(layer.dropout, nn.Dropout)

    def test_gradient_flow(self):
        layer = PreLNLayer(**_make_layer_kwargs())
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestPostLNLayer(unittest.TestCase):
    """Tests for PostLNLayer."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        layer = PostLNLayer(**_make_layer_kwargs())
        x = torch.randn(2, 8, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_no_dropout(self):
        for d in [None, 0.0, 0]:
            layer = PostLNLayer(**{**_make_layer_kwargs(), "dropout": d})
            self.assertIsInstance(layer.dropout, nn.Identity)

    def test_gradient_flow(self):
        layer = PostLNLayer(**_make_layer_kwargs())
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestDeepnetLayer(unittest.TestCase):
    """Tests for DeepnetLayer and deepnet_alpha/deepnet_beta."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        layer = DeepnetLayer(**_make_layer_kwargs(), alpha=1.5)
        x = torch.randn(2, 8, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_alpha_buffer_set(self):
        layer = DeepnetLayer(**_make_layer_kwargs(), alpha=2.0)
        self.assertAlmostEqual(layer.alpha.item(), 2.0)

    def test_reset_parameters(self):
        layer = DeepnetLayer(**_make_layer_kwargs(), alpha=3.0)
        layer.alpha.fill_(0.0)
        layer.reset_parameters()
        self.assertAlmostEqual(layer.alpha.item(), 3.0)

    def test_extra_repr(self):
        layer = DeepnetLayer(**_make_layer_kwargs(), alpha=1.5)
        self.assertIn("alpha=1.5", layer.extra_repr())


class TestDeepnetAlphaBeta(unittest.TestCase):
    """Tests for deepnet_alpha and deepnet_beta helper functions."""

    def test_decoder_only_alpha(self):
        # Formula: (2*m)^(1/4)
        self.assertAlmostEqual(deepnet_alpha(12, 0), (24) ** 0.25, places=5)

    def test_encoder_only_alpha(self):
        # Formula: (2*n)^(1/4)
        self.assertAlmostEqual(deepnet_alpha(0, 6), (12) ** 0.25, places=5)

    def test_encoder_decoder_alpha(self):
        self.assertIsInstance(deepnet_alpha(6, 6, "encoder"), float)
        self.assertIsInstance(deepnet_alpha(6, 6, "decoder"), float)

    def test_encoder_decoder_requires_which(self):
        with self.assertRaises(Exception):
            deepnet_alpha(6, 6)

    def test_decoder_only_beta(self):
        # Formula: (8*m)^(-1/4)
        self.assertAlmostEqual(deepnet_beta(12, 0), (96) ** (-0.25), places=5)

    def test_encoder_only_beta(self):
        self.assertAlmostEqual(deepnet_beta(0, 6), (48) ** (-0.25), places=5)

    def test_encoder_decoder_beta(self):
        self.assertIsInstance(deepnet_beta(6, 6, "encoder"), float)
        self.assertIsInstance(deepnet_beta(6, 6, "decoder"), float)


# ===================================================================
# Layer Stacks
# ===================================================================


class TestLayerStack(unittest.TestCase):
    """Tests for LayerStack."""

    def setUp(self):
        torch.manual_seed(42)

    def _make_simple_layer(self, layer_idx=0):
        return nn.Linear(32, 32)

    def test_basic_forward(self):
        stack = LayerStack(self._make_simple_layer, num_hidden_layers=4)
        x = torch.randn(2, 8, 32)
        out = stack(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_correct_number_of_layers(self):
        stack = LayerStack(self._make_simple_layer, num_hidden_layers=6)
        self.assertEqual(len(stack.layers), 6)

    def test_with_post_norm(self):
        stack = LayerStack(
            self._make_simple_layer,
            num_hidden_layers=2,
            post_norm_factory=lambda: nn.LayerNorm(32),
        )
        self.assertIsNotNone(stack.layer_norm)
        x = torch.randn(1, 4, 32)
        out = stack(x)
        self.assertEqual(out.shape, (1, 4, 32))

    def test_without_post_norm(self):
        stack = LayerStack(self._make_simple_layer, num_hidden_layers=2)
        self.assertIsNone(stack.layer_norm)

    def test_layer_indices(self):
        """Layer keys match expected string indices."""
        stack = LayerStack(self._make_simple_layer, num_hidden_layers=3)
        self.assertEqual(list(stack.layers.keys()), ["0", "1", "2"])


class TestCheckpointLayerStack(unittest.TestCase):
    """Tests for CheckpointLayerStack (activation checkpointing variant)."""

    def setUp(self):
        torch.manual_seed(42)

    def _make_simple_layer(self, layer_idx=0):
        return nn.Linear(32, 32)

    def test_basic_forward_no_checkpoint(self):
        stack = CheckpointLayerStack(
            self._make_simple_layer,
            num_hidden_layers=4,
            enable_checkpoint=False,
        )
        x = torch.randn(2, 8, 32)
        out = stack(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_correct_number_of_layers(self):
        stack = CheckpointLayerStack(
            self._make_simple_layer, num_hidden_layers=6, enable_checkpoint=False
        )
        self.assertEqual(len(stack.layers), 6)

    def test_with_post_norm(self):
        stack = CheckpointLayerStack(
            self._make_simple_layer,
            num_hidden_layers=2,
            enable_checkpoint=False,
            post_norm_factory=lambda: nn.LayerNorm(32),
        )
        self.assertIsNotNone(stack.layer_norm)

    def test_checkpoint_kwargs_merging(self):
        stack = CheckpointLayerStack(
            self._make_simple_layer,
            num_hidden_layers=1,
            checkpoint_kwargs={"preserve_rng_state": True},
        )
        self.assertTrue(stack.checkpoint_kwargs["preserve_rng_state"])
        self.assertFalse(stack.checkpoint_kwargs["use_reentrant"])

    def test_extra_repr(self):
        stack = CheckpointLayerStack(
            self._make_simple_layer,
            num_hidden_layers=2,
            enable_checkpoint=True,
            checkpoint_stride=2,
        )
        r = stack.extra_repr()
        self.assertIn("gradient_checkpointing=True", r)
        self.assertIn("checkpoint_stride=2", r)


class _KwargsLinear(nn.Module):
    """Linear that silently ignores extra kwargs in forward."""

    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x, **kwargs):
        return self.linear(x)


class TestExplicitLayerStack(unittest.TestCase):
    """Tests for ExplicitLayerStack."""

    def setUp(self):
        torch.manual_seed(42)

    def test_basic_forward(self):
        factories = [
            lambda **kw: _KwargsLinear(32),
            lambda **kw: _KwargsLinear(32),
        ]
        stack = ExplicitLayerStack(factories)
        x = torch.randn(2, 4, 32)
        out = stack(x)
        self.assertEqual(out.shape, (2, 4, 32))

    def test_number_of_layers(self):
        factories = [lambda **kw: nn.Linear(16, 16) for _ in range(5)]
        stack = ExplicitLayerStack(factories)
        self.assertEqual(len(stack.layers), 5)


# ===================================================================
# Eager Attention
# ===================================================================


class TestEagerScaledDotProductAttention(unittest.TestCase):
    """Tests for the reference eager attention implementation."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch = 2
        self.num_heads = 4
        self.seq_len = 8
        self.head_dim = 16

    def _make_qkv(self, seq_len=None):
        sl = seq_len or self.seq_len
        q = torch.randn(self.batch, self.num_heads, sl, self.head_dim)
        k = torch.randn(self.batch, self.num_heads, sl, self.head_dim)
        v = torch.randn(self.batch, self.num_heads, sl, self.head_dim)
        return q, k, v

    def test_basic_output_shape(self):
        q, k, v = self._make_qkv()
        out = eager_scaled_dot_product_attention(q, k, v)
        self.assertEqual(
            out.shape, (self.batch, self.num_heads, self.seq_len, self.head_dim)
        )

    def test_causal_mask(self):
        """With is_causal, position 0 should not attend to position 1."""
        q, k, v = self._make_qkv()
        out_causal = eager_scaled_dot_product_attention(q, k, v, is_causal=True)
        out_full = eager_scaled_dot_product_attention(q, k, v, is_causal=False)
        # Causal and full attention should differ
        self.assertFalse(torch.allclose(out_causal, out_full))

    def test_attn_mask_and_causal_exclusive(self):
        q, k, v = self._make_qkv()
        mask = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
        with self.assertRaises(ValueError):
            eager_scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)

    def test_boolean_mask(self):
        q, k, v = self._make_qkv(seq_len=4)
        # Block all future positions (lower triangular)
        mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        out = eager_scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(out.shape, (self.batch, self.num_heads, 4, self.head_dim))

    def test_float_mask(self):
        q, k, v = self._make_qkv(seq_len=4)
        mask = torch.zeros(4, 4)
        out = eager_scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(out.shape, (self.batch, self.num_heads, 4, self.head_dim))

    def test_custom_scale(self):
        q, k, v = self._make_qkv()
        out1 = eager_scaled_dot_product_attention(q, k, v, scale=0.01)
        out2 = eager_scaled_dot_product_attention(q, k, v, scale=100.0)
        # Very different scales should give different outputs
        self.assertFalse(torch.allclose(out1, out2))

    def test_gqa(self):
        """Test grouped query attention with fewer KV heads."""
        batch, seq_len, head_dim = 2, 8, 16
        num_heads_q, num_heads_kv = 8, 2
        q = torch.randn(batch, num_heads_q, seq_len, head_dim)
        k = torch.randn(batch, num_heads_kv, seq_len, head_dim)
        v = torch.randn(batch, num_heads_kv, seq_len, head_dim)

        out = eager_scaled_dot_product_attention(q, k, v, enable_gqa=True)
        self.assertEqual(out.shape, (batch, num_heads_q, seq_len, head_dim))

    def test_gqa_head_mismatch_without_flag(self):
        batch, seq_len, head_dim = 2, 4, 16
        q = torch.randn(batch, 8, seq_len, head_dim)
        k = torch.randn(batch, 2, seq_len, head_dim)
        v = torch.randn(batch, 2, seq_len, head_dim)

        with self.assertRaises(ValueError):
            eager_scaled_dot_product_attention(q, k, v, enable_gqa=False)

    def test_single_token_causal(self):
        """Single-token sequences: causal mask is a no-op."""
        q, k, v = self._make_qkv(seq_len=1)
        out = eager_scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(out.shape, (self.batch, self.num_heads, 1, self.head_dim))


# ===================================================================
# Init Weights
# ===================================================================


class TestHasLocalState(unittest.TestCase):
    """Tests for has_local_state helper."""

    def test_linear_has_state(self):
        self.assertTrue(has_local_state(nn.Linear(10, 10)))

    def test_embedding_has_state(self):
        self.assertTrue(has_local_state(nn.Embedding(100, 64)))

    def test_sequential_no_direct_state(self):
        """Sequential has child state but no direct parameters."""
        seq = nn.Sequential(nn.Linear(10, 10))
        self.assertFalse(has_local_state(seq))

    def test_identity_no_state(self):
        self.assertFalse(has_local_state(nn.Identity()))


class TestSimpleWeightInit(unittest.TestCase):
    """Tests for simple_weight_init."""

    def test_reinits_linear(self):
        linear = nn.Linear(32, 32)
        old_weight = linear.weight.clone()
        torch.manual_seed(99)
        simple_weight_init(linear)
        # Weight should change (reset_parameters called)
        self.assertFalse(torch.equal(old_weight, linear.weight))

    def test_skips_no_state_module(self):
        """Modules without parameters are silently skipped."""
        simple_weight_init(nn.Identity())  # should not raise

    def test_embedding_with_scale_rsqrt(self):
        emb = nn.Embedding(100, 64)
        simple_weight_init(emb, scale_rsqrt_d_model=True)
        # Verify std is approximately 1/sqrt(64) = 0.125
        std = emb.weight.std().item()
        self.assertAlmostEqual(std, 1.0 / math.sqrt(64), delta=0.05)


class TestInitEmbeddings(unittest.TestCase):
    """Tests for init_embeddings."""

    def test_scale_rsqrt_d_model(self):
        weight = torch.empty(100, 64)
        init_embeddings(weight, scale_rsqrt_d_model=True)
        expected_std = 1.0 / math.sqrt(64)
        self.assertAlmostEqual(weight.std().item(), expected_std, delta=0.05)

    def test_padding_index_zeroed(self):
        weight = torch.empty(100, 64)
        init_embeddings(weight, padding_index=0)
        self.assertTrue(torch.all(weight[0] == 0))

    def test_custom_std(self):
        weight = torch.empty(1000, 128)
        init_embeddings(weight, std=0.5, scale_rsqrt_d_model=False)
        self.assertAlmostEqual(weight.std().item(), 0.5, delta=0.05)


class TestInitTorchLinearDefault(unittest.TestCase):
    """Tests for init_torch_linear_default."""

    def test_initializes_uniformly(self):
        weight = torch.empty(64, 32)
        init_torch_linear_default(weight)
        # Uniform range = 1/sqrt(in_features) = 1/sqrt(32)
        max_val = 1.0 / math.sqrt(32)
        self.assertTrue(weight.max().item() <= max_val + 1e-6)
        self.assertTrue(weight.min().item() >= -max_val - 1e-6)

    def test_gain(self):
        weight = torch.empty(64, 32)
        init_torch_linear_default(weight, gain=2.0)
        max_val = 2.0 / math.sqrt(32)
        self.assertTrue(weight.max().item() <= max_val + 1e-6)


class TestInitWeightsByRegex(unittest.TestCase):
    """Tests for init_weights_by_regex."""

    def test_uses_reset_parameters_by_default(self):
        linear = nn.Linear(16, 16)
        old_w = linear.weight.clone()
        torch.manual_seed(99)
        init_weights_by_regex(linear, [])
        # reset_parameters was called, weight should change
        self.assertFalse(torch.equal(old_w, linear.weight))

    def test_regex_match_overrides_reset(self):
        linear = nn.Linear(16, 16)
        setattr(linear, "init_prefix", "ff.up_proj")
        # Mark params as not HF-initialized so the init actually runs
        for p in linear.parameters():
            p._is_hf_initialized = False

        called = []

        def custom_init(param):
            called.append(param)
            nn.init.zeros_(param)

        regex_list = [
            (r"weight", custom_init),
            (r"bias", custom_init),
        ]
        init_weights_by_regex(linear, regex_list)
        # Both weight and bias should have been initialized
        self.assertEqual(len(called), 2)
        self.assertTrue(torch.all(linear.weight == 0))

    def test_partial_init_raises(self):
        """If only some parameters match regex, it should raise ValueError."""
        linear = nn.Linear(16, 16)
        setattr(linear, "init_prefix", "ff.up_proj")

        regex_list = [(r"weight", nn.init.zeros_)]  # bias won't match
        with self.assertRaises(ValueError):
            init_weights_by_regex(linear, regex_list)

    def test_skips_modules_without_state(self):
        init_weights_by_regex(nn.Identity(), [])  # should not raise

    def test_raises_for_uninitialized_module(self):
        """Module with params but no reset_parameters and no regex match should fail."""

        class BadModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(4))

        with self.assertRaises(ValueError):
            init_weights_by_regex(BadModule(), [])


# ===================================================================
# CasualLM
# ===================================================================


class TestCasualLM(unittest.TestCase):
    """Tests for the CasualLM wrapper module."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 32
        self.vocab_size = 100
        self.seq_len = 8

        # Import here since it needs other modelsrc modules
        from causal_lm import CasualLM

        self.CasualLM = CasualLM

    def _make_model(self):
        embedding = nn.Embedding(self.vocab_size, self.d_model)
        input_encoder = InputEncoder(self.d_model, embedding, dropout=0.0)
        layer_stack = LayerStack(
            lambda layer_idx=0: _KwargsLinear(self.d_model),
            num_hidden_layers=2,
        )

        def dummy_mask_fn(config=None, dtype=None, **kwargs):
            return None

        model = self.CasualLM(
            input_encoder=input_encoder,
            layer_stack=layer_stack,
            init_weights=lambda m: None,
            attn_mask_fn=dummy_mask_fn,
            config=None,
        )
        return model

    def test_forward_shape(self):
        model = self._make_model()
        ids = torch.randint(0, self.vocab_size, (2, self.seq_len))
        out = model(ids)
        self.assertEqual(out.last_hidden_state.shape, (2, self.seq_len, self.d_model))

    def test_get_input_embeddings(self):
        model = self._make_model()
        emb = model.get_input_embeddings()
        self.assertIsInstance(emb, nn.Embedding)

    def test_set_input_embeddings(self):
        model = self._make_model()
        new_emb = nn.Embedding(50, self.d_model)
        model.set_input_embeddings(new_emb)
        self.assertIs(model.get_input_embeddings(), new_emb)

    def test_get_attn_mask_fn_disables_internal(self):
        model = self._make_model()
        self.assertTrue(model.use_internal_mask)
        fn = model.get_attn_mask_fn()
        self.assertFalse(model.use_internal_mask)
        self.assertTrue(callable(fn))

    def test_initialize_weights(self):
        called = []

        def track_init(m):
            called.append(m)

        embedding = nn.Embedding(self.vocab_size, self.d_model)
        input_encoder = InputEncoder(self.d_model, embedding, dropout=0.0)
        layer_stack = LayerStack(
            lambda layer_idx=0: _KwargsLinear(self.d_model),
            num_hidden_layers=1,
        )

        model = self.CasualLM(
            input_encoder=input_encoder,
            layer_stack=layer_stack,
            init_weights=track_init,
            attn_mask_fn=lambda config=None, dtype=None, **kw: None,
            config=None,
        )
        model.initialize_weights()
        self.assertEqual(len(called), 1)
        self.assertIs(called[0], model)

    def test_past_key_values_none(self):
        """Forward without KV cache returns None past_key_values."""
        model = self._make_model()
        ids = torch.randint(0, self.vocab_size, (1, 4))
        out = model(ids)
        self.assertIsNone(out.past_key_values)


# ===================================================================
# Causal Mask
# ===================================================================


class TestCausalMask(unittest.TestCase):
    """Tests for the causal_mask function."""

    def setUp(self):
        torch.manual_seed(42)

    def _make_config(self, attn_impl="sdpa", hidden_size=32, window_size=None):
        from transformers import PretrainedConfig

        config = PretrainedConfig(hidden_size=hidden_size)
        config._attn_implementation = attn_impl
        if window_size:
            config.window_size = window_size
        return config

    def test_sdpa_returns_none_for_simple_causal(self):
        """SDPA with no mask/cache/position_ids returns None (uses is_causal flag)."""
        config = self._make_config("sdpa")
        ids = torch.randint(0, 100, (2, 8))
        mask = causal_mask(config, torch.float32, input_ids=ids)
        self.assertIsNone(mask)

    def test_eager_returns_mask(self):
        """Eager attention must produce an actual mask tensor."""
        config = self._make_config("eager")
        ids = torch.randint(0, 100, (2, 8))
        mask = causal_mask(config, torch.float32, input_ids=ids)
        self.assertIsNotNone(mask)
        self.assertIsInstance(mask, torch.Tensor)

    def test_with_attention_mask(self):
        """Providing a padding mask forces mask generation even for SDPA."""
        config = self._make_config("sdpa")
        ids = torch.randint(0, 100, (2, 8))
        padding = torch.ones(2, 8, dtype=torch.long)
        padding[0, 6:] = 0  # mask last two tokens
        mask = causal_mask(config, torch.float32, input_ids=ids, attention_mask=padding)
        self.assertIsNotNone(mask)


if __name__ == "__main__":
    unittest.main()
