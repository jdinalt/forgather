# Gemma-3 (text)

Forgather model project for Google Gemma-3 text models, supporting
HuggingFace ↔ Forgather round-trip conversion via `forgather convert`.

## Architecture notes

Gemma-3 is mostly Llama-shaped but has several quirks that motivate the
extra modelsrc components in this project:

- **Decoupled head_dim**: `head_dim * num_heads != hidden_size` (e.g. for the
  270M model, `head_dim=256, num_heads=4, hidden_size=640`). Q/K/V/O
  projections use `num_heads * head_dim`, and `CausalMultiheadAttn` accepts
  an optional explicit `head_dim` for this.
- **GemmaRMSNorm**: RMSNorm applied as `(1 + weight) * normalize(x)`, with
  the weight initialized to zero. Computed in float32 for stability. Lives
  in `modelsrc/transformer/gemma_rms_norm.py`.
- **QK-Norm**: Query and key are normalized per-head (over `head_dim`)
  before RoPE. Reuses `CausalMultiheadAttn.qk_norm_factory`, pointed at
  `GemmaRMSNorm`.
- **Four layernorms per decoder layer**: `input_layernorm`,
  `post_attention_layernorm`, `pre_feedforward_layernorm`,
  `post_feedforward_layernorm`. `GemmaDecoderLayer` in
  `modelsrc/transformer/gemma_layer.py` wraps both the attention and
  feedforward blocks with pre- and post-norms.
- **Dual RoPE**: full-attention layers use a large `rope_theta`
  (e.g. 1e6) and sliding-attention layers use a small one (e.g. 1e4).
  `GemmaDualRotaryEmbedding` wraps two `RotaryEmbedding` instances and
  returns a dict keyed by layer type; `GemmaDecoderLayer` selects the
  appropriate entry per layer.
- **Per-layer attention type**: `config.layer_types` assigns each layer to
  either `full_attention` or `sliding_attention`. `GemmaDecoderLayer`
  constructs its attention module with the per-layer `sliding_window`
  (from `config.sliding_window`) for sliding layers and `None` for full
  layers. A `gemma_mask_fn` helper builds both a full causal mask and a
  sliding window mask at forward time and passes them through as a dict
  that the decoder layer unpacks.
- **GeGLU with `gelu_pytorch_tanh`**: the MLP activation is
  `nn.GELU(approximate='tanh')`. `use_triton` is set to `False` because
  the existing Triton-fused GLU kernel implements exact (erf-based) GELU.
- **Embedding scaling by `sqrt(hidden_size)`**: enabled via
  `InputEncoder.scale_sqrt_d_model`.
- **Tied embeddings**: `tie_word_embeddings=True` by default, wired via
  the standard model code generator `tied_weights_keys` mechanism.

## Usage

Convert a HuggingFace Gemma-3 model to Forgather format:

```bash
forgather convert ~/ai_assets/models/google_gemma-3-270m \
    ~/ai_assets/models/fg_gemma-3-270m \
    --model-type gemma3 --dtype bfloat16
```

Serve it with the Forgather inference server:

```bash
forgather inf server --model ~/ai_assets/models/fg_gemma-3-270m
forgather inf client --completion "Once upon a time"
```

Reverse conversion is symmetric (the HF model type is stored in the
Forgather config during conversion and auto-detected on the way back):

```bash
forgather convert ~/ai_assets/models/fg_gemma-3-270m \
    ~/tmp/hf_gemma-3-270m
```
