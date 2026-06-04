# SmolLM3

A native (code-generated) re-implementation of HuggingFace's
[**SmolLM3**](https://huggingface.co/blog/smollm3) architecture, with HF↔Forgather
weight conversion.

- Blog: https://huggingface.co/blog/smollm3
- Model card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B

## Architecture

SmolLM3 is, structurally, a **Llama** decoder — Grouped-Query Attention (GQA),
a SiLU gated MLP (GLU), pre-norm RMSNorm, tied input/output embeddings, and the
Llama-3 tokenizer (vocab 128 256) — with a single distinguishing feature:
**NoPE** (No Positional Embedding) on a periodic subset of layers.

| Component | Choice |
|---|---|
| Attention | Multi-head with GQA (3B: 16 query / 4 KV heads) |
| Positional encoding | RoPE on most layers; **none** on every 4th layer (NoPE) |
| MLP | Gated linear unit with SiLU activation |
| Normalization | RMSNorm, pre-norm |
| Q/K/V/O biases | None |
| Embeddings | Tied input/output |
| Tokenizer | Llama-3 BPE, 128 256 vocab |
| RoPE base (θ) | 5 000 000 (3B checkpoint), tuned for 64K context |

### NoPE — the one novelty

SmolLM3 omits rotary position embeddings from every
`no_rope_layer_interval`-th layer (default interval 4: layers 3, 7, 11, … in
0-based indexing). The schedule is stored in the config as `no_rope_layers`, a
list of `num_hidden_layers` flags where **`1` = apply RoPE** and **`0` = NoPE**,
generated as `(layer_idx + 1) % no_rope_layer_interval != 0` (matching
HuggingFace's `SmolLM3Config`).

The motivation comes from **NoPE** research (Kazemnejad et al. 2023, *The Impact
of Positional Encoding on Length Generalization*, arXiv:2305.19466): a decoder
with a causal mask is not truly position-agnostic even without explicit
positional encodings, and interleaving position-free attention layers improves
**length generalization** with negligible short-context cost. SmolLM3 uses this
(alongside a high RoPE base and intra-document masking) to reach a 64K–128K
context window.

#### How Forgather implements it

In Forgather, the central `RotaryEmbedding` (owned by `CasualLM`) computes
`(cos, sin)` once and threads them to every layer via `position_embeddings`
kwargs; each attention module applies them only if it was built with a
`pos_encoder`. Because `apply_rotary_pos_emb` *raises* when handed `None`
position embeddings, a NoPE layer must be constructed with **`pos_encoder=None`**
rather than simply fed empty embeddings.

[`SmolLM3DecoderLayer`](../../../modelsrc/transformer/smollm3_layer.py) (a thin
variant of `PreLNLayer`, modeled on `GemmaDecoderLayer`) reads
`config.no_rope_layers[layer_idx]` at construction and builds its attention with
`pos_encoder=None` for NoPE layers. The position embeddings still flow through
`forward` unchanged; a NoPE layer simply ignores them. You can see the schedule
in the constructed model's repr:

```
(3): SmolLM3DecoderLayer(layer_idx=3, apply_rope=False, ...)
```

## Configurations

The default config defines the original 3B model with the official tokenizer
(primarily a conversion target). A collection of down-scaled variants — using
the same wikitext tokenizers as the equivalent
[`llama`](../llama) configs — is provided for from-scratch experiments. All
variants retain the NoPE schedule and the faithful RoPE base (θ = 2 000 000, the
`SmolLM3Config` default).

| Config | Tokenizer | hidden / inter / heads / KV / layers | Tied | ~Params |
|---|---|---|---|---|
| `default.yaml` | SmolLM3 (128k) | 2048 / 11008 / 16 / 4 / 36 | yes | 3.1B |
| `4M.yaml` | wikitext 2k | 256 / 768 / 4 / 1 / 4 | no | 4M |
| `small.yaml` | wikitext 8k | 512 / 1280 / 8 / 2 / 10 | no | 30M |
| `small_tied.yaml` | wikitext 16k | 512 / 1280 / 8 / 2 / 10 | **yes** | 30M |
| `medium.yaml` | wikitext 32k | 768 / 2048 / 8 / 2 / 16 | no | 148M |

Small models are **untied by default** (for apples-to-apples comparison with the
other architectures); `small_tied.yaml` mirrors `llama/small_tied.yaml` and ties
the embedding to the output projection (with the corresponding init changes).

Construct / smoke-test a variant:

```bash
forgather -t small.yaml model -r construct          # build + parameter report
forgather -t small.yaml model -r --device cuda:0 test   # forward/backward smoke
```

## Conversion

Conversion reuses the Llama parameter-name mappings verbatim (SmolLM3 adds no
parameters — NoPE only removes a rotation), plus the scalar
`no_rope_layer_interval` config field (the explicit `no_rope_layers` list is
regenerated from it).

```bash
# HF -> Forgather (direction + model type auto-detected from the source config)
forgather convert /path/to/HuggingFaceTB_SmolLM3-3B /path/to/out_fg

# Forgather -> HF (round-trip; pass --model-type with the explicit --reverse flag)
forgather convert /path/to/out_fg /path/to/out_hf --reverse --model-type smollm3
```

The converted model reproduces the source's greedy generation token-for-token.
(The converter's logit comparison warns "dissimilar" at its 1e-5 tolerance — an
expected artifact of comparing two independent attention/RoPE implementations in
bf16; the functional output matches.)

### Tested models

- [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

### Limitations

- Long-context RoPE scaling (YaRN) is supported by `RotaryEmbedding` but left
  unset here, matching the base 3B checkpoint's config.
- Vision / MoE / multimodal variants are out of scope (dense text decoder only).

## References

- SmolLM3 blog — https://huggingface.co/blog/smollm3
- SmolLM3-3B model card — https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- Kazemnejad et al. 2023, *The Impact of Positional Encoding on Length
  Generalization* (NoPE) — https://arxiv.org/abs/2305.19466
- Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models*
  — https://arxiv.org/abs/2305.13245
- Su et al. 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding*
  — https://arxiv.org/abs/2104.09864
