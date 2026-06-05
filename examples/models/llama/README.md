# Llama

A Forgather-native re-implementation of Meta's **Llama** (1 / 2 / 3), with
HF↔Forgather weight conversion. This is the reference decoder of the repo: the
shared `models/transformers/dynamic_llama.yaml` template defined here is the
backbone that the **Llama-family** projects — Mistral, Qwen3, Gemma-3, SmolLM3,
and LlamaCanon — extend with one or two changes each. (DeepOne and the vanilla
Causal LM are distinct enough that they sit on their own parallel base templates,
not this one.)

## Architecture

Llama consolidated a handful of post-*Attention Is All You Need* refinements into
what is now the de-facto standard decoder recipe: a **pre-norm**, decoder-only
transformer with rotary positions and a gated MLP.

| Component | Choice | Lineage |
|---|---|---|
| Normalization | RMSNorm, pre-norm | Zhang & Sennrich 2019; Xiong et al. 2020 |
| Attention | multi-head, optional GQA | Vaswani 2017; Ainslie et al. 2023 |
| Positional encoding | rotary (RoPE) | Su et al. 2021 |
| MLP | SwiGLU (SiLU-gated GLU) | Shazeer 2020 |
| Biases | none on Q/K/V/O or MLP | |
| Embeddings | untied by default | |

Forgather builds this from `dynamic_llama.yaml` (`torch.nn.RMSNorm`,
`GLUFeedforwardLayer` with `SiLU`, `RotaryEmbedding`, `CausalMultiheadAttn`,
`PreLNLayer`). The Llama-family projects are mostly this template plus a single
distinguishing feature — Mistral adds a sliding window, Qwen3 adds QK-norm,
SmolLM3 adds NoPE, and so on. (The DeepOne and Causal LM projects are *not*
Llama-derived: DeepOne extends `models/transformers/deepone.yaml` and Causal LM
extends `dynamic_causal_transformer.yaml`.)

### RoPE and long-context scaling

`RotaryEmbedding` supports the plain base-`theta` rotation and the **Llama-3
frequency scaling** (`rope_type: llama3`) used to extend context: a three-band
wavelength schedule that leaves high-frequency dimensions unscaled, divides
low-frequency dimensions by the scaling factor, and smoothly interpolates
between. `llama3.2_1b.yaml` exercises this path; YaRN scaling is also available.

## Configurations

The full-size configs reproduce real Meta checkpoints (conversion targets); the
down-scaled variants reuse the shared wikitext tokenizers for from-scratch
experiments.

| Config | Tokenizer | hidden / inter / heads / KV / layers | Notes |
|---|---|---|---|
| `default.yaml`, `llama2_7b.yaml` | Llama-2 | 4096 / 11008 / 32 / 32 / 32 | Llama-2-7B (MHA) |
| `llama3.2_1b.yaml` | Llama-3 | 2048 / 8192 / 32 / 8 / 16 | Llama-3.2-1B, RoPE `llama3` scaling |
| `4M.yaml` | wikitext 2k | 256 / 768 / 4 / (MHA) / 4 | ~4M, quick test |
| `small.yaml` | wikitext 8k | 512 / 1280 / 8 / 2 / 10 | ~34M |
| `small_tied.yaml` | wikitext 16k | 512 / 1280 / 8 / 2 / 10 | tied input/output embeddings |
| `medium.yaml` | wikitext 32k | 768 / 2048 / 8 / (MHA) / 16 | ~160M |

### Tested models

- [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)

## References

- Vaswani et al. 2017, *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Touvron et al. 2023, *LLaMA: Open and Efficient Foundation Language Models* — [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Touvron et al. 2023, *Llama 2: Open Foundation and Fine-Tuned Chat Models* — [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- Grattafiori et al. 2024, *The Llama 3 Herd of Models* — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Su et al. 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding* — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Shazeer 2020, *GLU Variants Improve Transformer* (SwiGLU) — [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Zhang & Sennrich 2019, *Root Mean Square Layer Normalization* — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture* (pre-LN) — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
