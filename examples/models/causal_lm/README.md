# Causal LM

The **vanilla transformer** decoder — the *Attention Is All You Need* recipe,
kept deliberately classic. It serves as a baseline and as a reference point for
measuring what the modern (Llama-era) refinements actually buy.

## Architecture

A decoder-only transformer that stays close to the 2017 original:

| Component | Choice |
|---|---|
| Normalization | LayerNorm, **post-norm** |
| Attention | multi-head (no GQA), no QK-norm |
| Positional encoding | fixed **sinusoidal** absolute |
| MLP | plain (non-gated), **ReLU** |
| Embeddings | input scaled by √d_model; output init scaled by 1/√d_model |

It is built from the `models/causal_lm` base (`PostLNLayer`, `sinusoidal_pe`,
the plain `feedforward_layer`, and `CausalMultiheadAttn` in MHA mode). It speaks
the HuggingFace attention interface (eager / flex-attention) and is
vLLM-compatible.

The √d_model embedding scaling and the additive sinusoidal positions are the
original Vaswani et al. choices; post-norm (LayerNorm *after* each residual add)
is what most pre-Llama transformers used before pre-norm became standard.

This config is the `small_causal` baseline in the
[Small Models](../../tiny_experiments/small_models) comparison — the gap between
it and every RoPE + GLU + GQA model there is a clean measure of how much the
now-standard ingredients are worth at small scale.

## Configurations

| Config | Tokenizer | hidden / inter / heads / layers |
|---|---|---|
| `4M.yaml` | wikitext 2k | 256 / 1024 / 4 / 4 |
| `small.yaml` | wikitext 8k | 512 / 1280 / 8 / 10 |
| `medium.yaml` | wikitext 16k | 768 / 3072 / 8 / 16 |

## References

- Vaswani et al. 2017, *Attention Is All You Need* (sinusoidal PE, post-LN, MHA, ReLU MLP, √d_model embedding scaling) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Ba et al. 2016, *Layer Normalization* — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture* (why post-LN vs pre-LN matters) — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
