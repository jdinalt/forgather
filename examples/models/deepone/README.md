# DeepOne

A **hybrid** decoder that pairs two ideas published separately and — as far as
we know — not combined elsewhere:

1. **DeepNorm**, the residual scaling + initialization scheme from *DeepNet:
   Scaling Transformers to 1,000 Layers* (Wang et al. 2022), engineered to make
   very deep post-norm stacks trainable; and
2. **ALiBi**, the linear attention-bias positional scheme from *Train Short,
   Test Long* (Press et al. 2021).

There is no external reference for this specific combination — DeepOne is a
Forgather example exploring whether DeepNet-style depth stability and ALiBi's
length extrapolation sit well together. It is modernized relative to both source
papers (RMSNorm instead of LayerNorm, a gated MLP instead of a plain one).

## Architecture

### DeepNorm — post-norm built for depth

Plain **post-LN** transformers train unstably once they are deep; **pre-LN** is
stable but tends to leave some quality on the table. DeepNorm keeps post-norm but
balances the residual stream two ways:

- **up-scale the residual branch** by a constant α before the layernorm, and
- **down-scale the initialization** of the projection weights by a constant β.

With α and β chosen as functions of depth, the expected per-update change to the
output stays bounded no matter how many layers are stacked, so the model gets
post-LN's performance with pre-LN's stability. DeepOne uses the encoder/decoder-
only DeepNorm constants (`modelsrc/transformer/deepnet.py`):

```
α = (2N) ** (1/4)      # residual up-scale,  N = num_hidden_layers
β = (8N) ** (-1/4)     # weight-init down-scale
```

applied per sub-block as `x = norm(residual * α + sublayer(x))`. (The paper
defines DeepNorm with LayerNorm; DeepOne uses RMSNorm and a ReLU-gated GLU MLP.)

### ALiBi — positions as a linear attention bias

Instead of adding positional information to the token embeddings, ALiBi adds a
static, head-specific penalty to the attention scores: a query attending to a key
*m* positions back has `m · slope_h` subtracted before the softmax. Nearer tokens
are favored, nothing positional has to be learned, and the model extrapolates to
sequences longer than it trained on. Because the position signal lives in the
scores rather than the values, semantic and positional contributions stay
cleanly separated.

DeepOne computes per-head slopes as a geometric series (`causal_alibi_attn.py`):

```
slopes = 1 / logspace(0, 7, num_heads, base=2)   #  1, 1/2, 1/4, ..., 1/128
```

and supports GQA, the KV cache, and the eager / SDPA / flex-attention /
flash-attention-2 backends. The flex path carries the ALiBi bias as a `score_mod`,
so it never materializes the O(seq²) bias matrix.

## Configurations

| Config | Tokenizer | hidden / inter / heads / layers |
|---|---|---|
| `4M.yaml` | wikitext 2k | 256 / 768 / 4 / 4 |
| `small.yaml` | wikitext 8k | 512 / 1280 / 8 / 10 |
| `medium.yaml` | wikitext 32k | 768 / 2048 / 8 / 16 |
| `1_7B.yaml` | wikitext 32k | 2048 / 5376 / 16 / 32 |

DeepOne appears as `small_deepone` in the
[Small Models](../../tiny_experiments/small_models) comparison, where its
post-LN + DeepNorm design is the least stable of the eight (grad-norm creeps into
the ~1.8–2.6 band mid-run) but still finishes mid-pack thanks to the clip guard
and WSD LR decay — a fair illustration of the post-LN/pre-LN tradeoff DeepNorm is
trying to thread.

## References

- Wang et al. 2022, *DeepNet: Scaling Transformers to 1,000 Layers* (DeepNorm) — [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)
- Press et al. 2021, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi) — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture* (post-LN vs pre-LN) — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Shared backbone references (RMSNorm, GLU) — see [../llama](../llama).
