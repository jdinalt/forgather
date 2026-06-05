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

### ReLU MLP — for activation sparsity

The gated MLP uses a **ReLU** activation rather than the now-customary SiLU/GELU.
The choice follows *ReLU Strikes Back* (Mirzadeh et al. 2023): ReLU drives a large
fraction of MLP activations to exact zero, and that sparsity can be exploited to
cut the FLOPs and memory traffic of inference (only the non-zero activations need
their down-projection columns). Smoother activations leak small non-zero values
everywhere and forfeit that structure. ReLU costs little or nothing in quality
while leaving the model far easier to **sparsify** later.

### ALiBi — positions as a linear attention bias

Instead of adding positional information to the token embeddings, ALiBi adds a
static, head-specific penalty to the attention scores: a query attending to a key
*m* positions back has `m · slope_h` subtracted before the softmax. Nearer tokens
are favored and the model extrapolates to sequences longer than it trained on.
Because the position signal lives in the scores rather than the values, semantic
and positional contributions stay cleanly separated. The original ALiBi fixes the
per-head slopes to a geometric series (`1, 1/2, 1/4, …`); Forgather's
implementation can instead make them **trainable** and offers an alternative
initialization.

#### Trainable slopes and the bimodal split

DeepOne enables **trainable** slopes (`trainable_alibi: True`) with the
**alternative init** (`alt_alibi_init: True`). The motivation is an empirical
observation: train the slopes from the original geometric init and they do not
stay put — roughly **half migrate toward 0 and half migrate upward**. Training
sorts the heads into two populations: *position-agnostic* heads (slope ≈ 0,
attending on content alone) and *strongly positional* heads (large slope,
attending to nearby tokens). The alternative init pre-bakes exactly this split —
the upper half of the heads keep substantial slopes, the lower half are
initialized to **0** — so for DeepOne it simply accelerates the model's own
tendency.

We suspect this bimodal structure is the model assembling **induction heads**
(Elhage et al. 2021): a two-layer circuit in which a *previous-token* head (strong
positional bias) copies information from the adjacent position into the residual
stream, and an *induction* head one layer later (little or no positional bias)
attends by content to copy the next expected token. The two slope populations
line up with the two roles.

#### Cost

Trainable ALiBi is **expensive**. Two ways to avoid paying for it across the whole
run: disable training and lean on the alternative init (often *better* than a
trained geometric init, and free), or train with trainable slopes just long enough
for them to settle and then freeze them, which speeds the rest of training
considerably.

DeepOne supports GQA, the KV cache, and the eager / SDPA / flex-attention /
flash-attention-2 backends — the flex path carries the bias as a `score_mod`, so
it never materializes the O(seq²) matrix. Trainable slopes work on every backend
except flash-attention-2 (fixed-bias, but the fastest for inference).

## Configurations

| Config | Tokenizer | hidden / inter / heads / layers |
|---|---|---|
| `4M.yaml` | wikitext 2k | 256 / 768 / 4 / 4 |
| `small.yaml` | wikitext 8k | 512 / 1280 / 8 / 10 |
| `medium.yaml` | wikitext 32k | 768 / 2048 / 8 / 16 |
| `1_7B.yaml` | wikitext 32k | 2048 / 5376 / 16 / 32 |

DeepOne appears as `small_deepone` in the
[Small Models](../../tiny_experiments/small_models) comparison, where it finishes
mid-pack. DeepNorm is built for stability, and in prior testing DeepOne has
trained with a notably **low, stable gradient norm — below Llama's**. In that
particular run, though, it was unexpectedly the *least* stable of the eight
(grad-norm crept into the ~1.8–2.6 band mid-run before the clip guard and WSD LR
decay brought it home). That result runs counter to the architecture's track
record; the cause is still open and worth tracking down.

## References

- Wang et al. 2022, *DeepNet: Scaling Transformers to 1,000 Layers* (DeepNorm) — [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)
- Press et al. 2021, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi) — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* (previous-token heads, induction heads) — <https://transformer-circuits.pub/2021/framework/index.html>
- Mirzadeh et al. 2023, *ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models* — [arXiv:2310.04564](https://arxiv.org/abs/2310.04564)
- Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture* (post-LN vs pre-LN) — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Shared backbone references (RMSNorm, GLU) — see [../llama](../llama).
