# Custom Canon (Canon-A + NoPE + QK-Norm)

A custom-model sub-project for the [`small-llm`](../README.md)
pretraining example. It defines a single architecture variant -
"Canon-A only, no positional encoding, with QK-Norm" - sized to match
the project's default Llama medium so the two can be compared head-to-
head on identical training runs.

This sub-project is the model half of the example pattern in the
parent project: training projects can point at it via
`ns.model_project_dir` / `ns.model_project_config` (see the parent's
[`canon.yaml`](../templates/configs/canon.yaml)).

## What "Canon" means here

**Canon layers** are small depthwise causal 1D convolutions
(kernel=4) inserted inside each transformer block to provide cheap,
local token mixing. The layers and their insertion points come from
Allen-Zhu's *Physics of Language Models: Part 4.1, Architecture
Design and the Magic of Canon Layers* ([arXiv:2512.17351](https://arxiv.org/abs/2512.17351),
2025), and the four positions defined there are:

| Position | Where it sits | Tensor dim |
|----------|---------------|------------|
| **A** | after input LayerNorm, before attention | `hidden_size` |
| **B** | on the QKV projections, before attention compute | `(n_heads + 2·n_kv_heads) · d_head` |
| **C** | after post-attention LayerNorm, before the MLP | `hidden_size` |
| **D** | on the gate/up projections in the SwiGLU MLP | `2 · intermediate_size` |

A given model can use any subset of {A, B, C, D}. The reusable
implementation lives in
[`examples/models/llama_canon/`](../../../models/llama_canon/) - the
same Llama backbone with optional Canon layers at each position. This
sub-project consumes that model project and selects a specific subset
(A only) plus a couple of orthogonal architecture choices.

## What this variant changes from baseline Llama

Concretely, relative to the parent project's default
`examples/models/llama/medium.yaml`:

- **Canon-A enabled**, Canon-B/C/D disabled. The pre-attention
  depthwise convolution mixes adjacent token representations before
  attention sees them.
- **NoPE** - no rotary or relative positional encoder. Canon-A's
  local convolution is expected to provide enough positional signal
  for attention to recover sequence order on its own.
- **QK-Norm** - RMSNorm applied to Q and K before the attention
  score computation, Qwen3-style. Not part of Canon proper; included
  here as a stabiliser that's cheap and tends to help at small
  scales.
- **Same parameter count and shape as the baseline:** `hidden_size =
  768`, `intermediate_size = 2048`, 16 layers, 8 attention heads.
  This matches Llama medium so the eval loss comparison in the parent
  project is head-to-head at matched compute.
- **Same WikiText 32k tokenizer** as the baseline.

## Why this specific configuration

The choice of "Canon-A only + NoPE" is informed by the more thorough
ablation in
[`examples/tiny_experiments/canon/`](../../../tiny_experiments/canon/README.md),
which sweeps the {A, B, C, D} subset on a 4M Llama trained on
TinyStories. Two findings drove the decision here:

- **Canon-A alone runs faster than baseline Llama**, despite adding
  a convolution per layer. In the 4M ablation, Canon-A clocks 316K
  tok/s vs 291K for Llama (+9%), with a 2.8% better eval loss (1.264
  vs 1.301).
- **Canon-A nearly replaces RoPE.** Canon-A NoPE achieves 1.312 eval
  vs Llama+RoPE's 1.301 - about 0.8% worse - while running 11%
  faster (323K vs 291K tok/s). The single pre-attention convolution
  provides enough local positional signal that explicit positional
  encoding becomes nearly redundant. At larger model sizes, where
  RoPE's complex-valued computation and memory footprint grow, this
  trade is expected to look more favourable.

So the design intent of `custom_canon` is "minimal Canon, no RoPE,
should be at least as fast as the baseline and competitive on eval."
The 4M tiny-experiments numbers say this should win on throughput
and at least tie on eval at small scale.

The parent project's empirical result at ~162M scale is more
sobering - see below.

## Configurations

```
forgather ls
```

| Config | Purpose |
|--------|---------|
| `custom_canon.yaml` | Canon-A + NoPE + QK-Norm, sized to match Llama medium. |

Only the one config is shipped. The training entry point is
[`canon.yaml`](../templates/configs/canon.yaml) in the parent
project, which points its `ns.model_project_dir` at this
sub-project.

## Empirical result in the parent project

In the parent project's
[1x Chinchilla bundle](../README.md#1x-chinchilla-runs), this
configuration finishes with a best eval loss of **2.678** - the
worst of the 1x runs, ~0.06 above `high_lr` and ~0.03 above the
two pure-bf16 runs. Throughput was also lower than the Llama
variants (~39 samp/s vs ~45-49 for the rest of the bundle), which
contradicts the 4M tiny-experiments expectation that Canon-A alone
should be *faster* than baseline.

Likely contributors to the throughput regression at this scale:
the Canon convolution implementation has not been re-tuned for
larger models, and the kernel that ran 9-11% faster than Llama at
4M may have lost its edge as `hidden_size` and sequence length
grew. The eval-loss gap is more interesting and probably wants a
proper LR sweep for this variant before any conclusion - the
parent project picked LRs tuned for the baseline Llama, and Canon-A
+ NoPE + QK-Norm is a different enough architecture that the
optimum may sit elsewhere.

The ablation in `tiny_experiments/canon` also identifies *better*
Canon variants at 4M scale - notably **Canon-B** alone (1.215 eval,
the lowest of any Canon subset) and **Canon-AC** (best
quality-per-throughput at 1.242 eval and 296K tok/s). Either would
be a more obvious next thing to try at 162M than re-running
Canon-A, especially Canon-AC since A and C are inexpensive.

## Cross-references

- [Parent project README](../README.md) for the head-to-head 1x
  Chinchilla results and how `canon.yaml` plugs this sub-project
  into training.
- [`examples/tiny_experiments/canon/`](../../../tiny_experiments/canon/README.md)
  for the full Canon position-and-RoPE ablation at 4M scale.
- [`examples/models/llama_canon/`](../../../models/llama_canon/) for
  the underlying model project this one extends, including
  parameter-overhead numbers and the `canon_kernel` /
  `canon_residual` knobs.
- Allen-Zhu, *Physics of Language Models: Part 4.1*,
  [arXiv:2512.17351](https://arxiv.org/abs/2512.17351), 2025 - the
  original Canon paper.
