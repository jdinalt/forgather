## Optimizers

Test harness for comparing optimizer implementations.

### Setup

All experiments use a 28.3M parameter LLaMA-style causal language model
(hidden_size=512, 10 layers, GQA with 8 heads / 2 KV heads, vocab_size=4000)
trained on a subset of HuggingFaceTB for 1 epoch (~560M tokens) with mixed
precision (bf16 autocast, float32 weights). The default training configuration
uses a per-device batch size of 32, sequence length 512, and a cosine LR
schedule with 10% warmup.

Learning rate is auto-scaled using a square-root rule:
`lr = base_lr * sqrt(tokens_per_step / base_batch_size)` where `base_lr=3e-4`
and `base_batch_size=16384`.

The default AdamW config (`amp/adamw.yaml`) serves as the baseline control.

### Available Configurations

```
forgather ls
```

Configurations are organized under `amp/` (mixed precision) and `bfloat16/`
(pure bf16 weights). Each config extends `project.yaml` which provides the
shared training setup.

### Running Experiments

```bash
# Train with default settings (single GPU)
forgather -t amp/adamw.yaml train

# Specify GPU
forgather -t amp/fg_apollo.yaml train -d 3

# Override hyperparameters via CLI
forgather -t amp/fg_apollo.yaml train --apollo-lr 0.008 --apollo-rank 128

# Gradient accumulation (larger effective batch)
forgather -t amp/fg_apollo.yaml train --gradient-accumulation-steps 8 --apollo-lr 0.008

# Custom log name for comparison runs
forgather -t amp/fg_apollo.yaml train --log-name apollo_test_run
```

### Apollo: Batch Size Sensitivity

APOLLO ("SGD-like Memory, AdamW-level Performance",
[arXiv:2412.05270](https://arxiv.org/abs/2412.05270)) approximates AdamW's
per-channel adaptive learning rate scaling using low-rank random projections.
Instead of maintaining full-rank first and second moment estimates (M, V) like
Adam, Apollo projects gradients into a low-rank subspace, computes M and V
there, and derives a per-channel scaling factor:

```
S_j = ||R_tilde[:,j]|| / ||R[:,j]||
```

where `R` is the projected gradient and `R_tilde = M / (sqrt(V) + eps)`. The
full-rank gradient is then scaled channel-wise by S before the weight update.

#### The Problem

Initial experiments showed Apollo performing significantly worse than AdamW
(eval loss 2.996 vs 2.692 -- a gap of 0.30) despite matching the paper's
recommended learning rate of 0.01 and using rank=128 (25% of hidden_dim=512).
The implementation was verified to be numerically identical to the official
`apollo-torch` package.

A weight decay bug was found and fixed (the `scale` parameter was used instead
of `lr` in the decoupled weight decay calculation), but this was not active in
the test runs (weight_decay=0).

#### Root Cause: Batch Size

The scaling factor S is estimated from a single mini-batch's projected gradient.
With small batches, the gradient is noisy, making the per-channel norm ratios
that define S unreliable. The paper's experiments use batch sizes of 256-512
sequences (~262K-524K tokens). Our default configuration uses only 32 sequences
(~16K tokens) -- roughly 16x smaller.

Increasing the effective batch size via gradient accumulation dramatically
closes the gap:

| Effective Batch | AdamW | Apollo (lr=0.008) | Gap |
|---|---|---|---|
| ga=1 (16K tokens/step) | **2.692** | 2.996 | +0.305 |
| ga=4 (62K tokens/step) | **2.720** | 2.801 | +0.081 |
| ga=8 (124K tokens/step) | 2.971 | **2.875** | -0.095 |

At ga=8, Apollo **outperforms** AdamW by ~0.1 eval loss. The remaining AdamW
advantage at ga=1 and ga=4 is entirely attributable to batch size sensitivity.

Note that larger gradient accumulation means fewer total optimization steps for
the same data (35980 steps at ga=1 vs 4497 at ga=8), so the absolute eval
losses are not directly comparable across rows. The meaningful comparison is
within each row: Apollo vs AdamW at the same batch size.

#### LR Sensitivity

With the batch size issue resolved, Apollo's LR sensitivity was explored via a
sweep at the default batch size (ga=1, 36K steps):

| Apollo LR | Final Eval Loss |
|---|---|
| 0.005 | 3.001 |
| 0.008 | **2.996** |
| 0.01 | 3.045 |
| 0.015 | 3.772 (diverges) |
| 0.02 | 3.856 (diverges) |
| 0.03 | 3.949 (diverges) |

The optimal Apollo LR for this model is around 0.008, close to the paper's
recommended 0.01. Learning rates above 0.015 cause divergence.

#### Configuration

The Apollo config (`amp/fg_apollo.yaml`) uses a `multiopt` setup that routes
weight matrices to Apollo and norm layers to Adam:

```yaml
optimizer_map:
    - [ "bias|norm", "default" ]
    - [ "embedding|lm_head|feedforward|attention", "apollo" ]
    - [ ".*", "default" ]
```

Key parameters can be overridden from the command line:

- `--apollo-lr`: Apollo learning rate (default: 0.01)
- `--apollo-rank`: Projection rank (default: 128)
- `--apollo-scale`: Gradient scale factor, applied as sqrt (default: 1.0)
- `--gradient-accumulation-steps`: Increase effective batch size

#### Recommendations

- Use a batch size of at least 60K tokens for Apollo to estimate reliable
  channel-wise scaling. At 120K+ tokens, Apollo can match or beat AdamW.
- Apollo LR should be set independently from the AdamW/Adam LR (typically
  ~30x higher: 0.008-0.01 vs 3e-4).
- For small-batch regimes where Apollo underperforms, gradient accumulation
  is a simple way to increase the effective batch size without additional
  memory cost.
