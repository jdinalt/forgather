# Mistral

A Forgather-native re-implementation of **Mistral 7B**, with HF↔Forgather
conversion. Architecturally Mistral is the [Llama](../llama) backbone (RMSNorm
pre-norm, SwiGLU MLP, RoPE, GQA, no biases) with one addition: **sliding-window
attention**.

## Architecture

### Sliding-window attention (SWA)

Each token attends only to itself and the previous `sliding_window - 1` tokens
instead of the full causal prefix. Two consequences:

- **Attention cost is linear in sequence length** — the window is a constant, so
  long sequences stay cheap.
- **The receptive field grows with depth.** A token at layer *k* can be
  influenced by information up to `k × sliding_window` positions back, because
  every layer slides the window forward — the same stacking argument that gives a
  deep CNN a large receptive field from small kernels. So a windowed model still
  propagates long-range information; it just does so through depth rather than
  within a single attention op. (Windowed / sparse attention traces back to the
  Sparse Transformer and Longformer.)

Mistral pairs SWA with a **rolling-buffer KV cache** (capped at `sliding_window`
entries, overwriting the oldest) and GQA for fast inference — both inherited from
the Llama backbone here. In Forgather, SWA is simply the `sliding_window` field
on the config, consumed by `CausalMultiheadAttn` and the flex / SDPA mask
builders; setting it to `None` gives full causal attention.

## Configurations

| Config | Tokenizer | hidden / inter / heads / KV / layers | `sliding_window` | Notes |
|---|---|---|---|---|
| `default.yaml` | Mistral | 4096 / 14336 / 32 / 8 / 32 | None | Mistral-7B-v0.1 |
| `7B_instruct.yaml` | Mistral | (default) | None | v0.2-Instruct, RoPE θ = 1e6 |
| `4M.yaml` | wikitext 2k | 256 / 768 / 4 / 2 / 4 | 4096 | quick test |
| `small.yaml` | wikitext 8k | 512 / 1280 / 8 / 2 / 10 | 1024 | ~34M, window exercised |
| `medium.yaml` | wikitext 32k | 768 / 2048 / 8 / 2 / 16 | 4096 | ~160M |

The full-size configs leave the window `None` (matching how later HF Mistral
releases ship); the down-scaled configs set a window to exercise the SWA path.

### Tested models

- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

## References

- Jiang et al. 2023, *Mistral 7B* — [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Beltagy et al. 2020, *Longformer: The Long-Document Transformer* (windowed attention) — [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- Child et al. 2019, *Generating Long Sequences with Sparse Transformers* — [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Ainslie et al. 2023, *GQA* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Shared Llama-backbone references (RoPE, SwiGLU, RMSNorm) — see [../llama](../llama).
