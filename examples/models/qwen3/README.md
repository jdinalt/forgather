# Qwen3

A Forgather-native re-implementation of Alibaba's **Qwen3** (dense, text-only),
with HF↔Forgather conversion.
[Qwen3 family on HuggingFace.](https://huggingface.co/collections/Qwen/qwen3)

## Architecture

Qwen3 is the [Llama](../llama) backbone (RMSNorm pre-norm, SwiGLU, RoPE, GQA, no
biases) with one distinguishing change: **QK-Norm** — an RMSNorm applied per-head
to the query and key projections (over `head_dim`) before RoPE and attention.

Normalizing the queries and keys bounds the scale of the query·key dot products,
which prevents the attention-logit blow-ups that destabilize training at scale
and in low precision. It has become a common stability trick — used in ViT-22B
and Gemma-3 among others — and Qwen3 adopts it while *dropping* the QKV bias that
Qwen2 carried. In Forgather this is just the `qk_norm_factory` hook (pointed at
`RMSNorm`) on the shared `CausalMultiheadAttn` — the same hook Gemma-3 reuses.

| Component | Choice |
|---|---|
| Attention | GQA + **QK-Norm** (per-head RMSNorm on Q, K) |
| Positional encoding | RoPE, θ = 1e6 |
| MLP | SwiGLU (SiLU-gated) |
| Normalization | RMSNorm pre-norm, eps 1e-6 |
| Biases | none (Qwen2's QKV bias removed) |

## Configurations

| Config | Tokenizer | hidden / inter / heads / KV / layers | Tied |
|---|---|---|---|
| `default.yaml` | Qwen3 | 4096 / 12288 / 32 / 8 / 36 | no |
| `4M.yaml` | wikitext 2k | 256 / 768 / 4 / 2 / 4 | no |
| `small.yaml` | wikitext 8k | 512 / 1280 / 8 / 2 / 10 | **yes** |
| `medium.yaml` | wikitext 32k | 768 / 2048 / 8 / 2 / 16 | **yes** |

`small.yaml` and `medium.yaml` also demonstrate **tied input/output embeddings**:
beyond setting `tie_word_embeddings: True`, the default Llama init is adjusted
(the lm-head init is nulled and the embedding init is scaled by 1/√d_model).

### Tested models

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)

### Limitations

- Mixture-of-Experts (MoE) variants are not supported — dense decoders only.
- Qwen3-VL (vision-language) variants are not supported — text-only.

## References

- Qwen Team 2025, *Qwen3 Technical Report* — [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- Henry et al. 2020, *Query-Key Normalization for Transformers* — [arXiv:2010.04245](https://arxiv.org/abs/2010.04245)
- Dehghani et al. 2023, *Scaling Vision Transformers to 22 Billion Parameters* (per-head QK-norm at scale) — [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
- Shared Llama-backbone references (RoPE, SwiGLU, RMSNorm, GQA) — see [../llama](../llama).
