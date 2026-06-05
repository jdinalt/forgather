# Attention-Only Ablation

What does the **MLP** actually contribute to a transformer? This experiment
isolates it by training the [`singlehead`](../../models/singlehead) model — a
deliberately simple single-head ALiBi transformer — in two forms, **with** and
**without** the feedforward (MLP) sub-block, and comparing them on
[TinyStories](https://arxiv.org/abs/2305.07759).

The attention-only form is the object studied in Elhage et al.'s
[*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html)
(2021): a transformer whose residual stream is moved *only* by attention. Adding
the MLP back is the single variable under test.

## The two architectures

Both are pre-LN, RMSNorm, tied-embedding, single-head **ALiBi** transformers
(ALiBi is chosen so position lives entirely in the attention scores and never
muddies the query/key/value content — see the
[singlehead README](../../models/singlehead)). They differ by exactly one thing:

| Block | Update | Layer class |
|---|---|---|
| **Attention + MLP** | `x = x + attn(norm(x))`, then `x = x + mlp(norm(x))` | `PreLNLayer` |
| **Attention-only** | `x = x + attn(norm(x))` | `AttentionOnlyLayer` |

Removing the MLP without compensating drops parameters — at the small width the
MLP is the *majority* of the model. So the experiment runs two kinds of
attention-only model: a plain one (MLP simply deleted) and a **param-matched**
one that re-spends the MLP's capacity as wider, deeper attention (hidden 960 ×
11 layers) so its *non-embedding* parameter count matches the +MLP model. That
second arm is the fairer test: it asks whether the MLP does something attention
*can't*, separated from whether it just adds capacity.

| Run | hidden / layers / inter | MLP? | Non-emb | Params |
|---|---|---|---|---|
| `mlp_4m` | 256 / 6 / 384 | yes | 1.7M | 4.6M |
| `attn_only_4m` | 256 / 6 / — | no | 0.8M | 2.8M |
| `mlp_small` | 512 / 8 / 1280 | yes | 19.9M | 24M |
| `attn_only_small` | 512 / 8 / — | no | 4.2M | 8.3M |
| `attn_only_matched_small` | 960 / 11 / — | no | 20.3M | 28M |

`attn_only_matched_small` matches `mlp_small` on non-embedding parameters (~20M)
to within ~2%. A naive depth-only match would need ~38 layers at width 512;
widening to 960 keeps the depth at a stable 11 (very deep attention-only stacks
get hard to train).

## Setup

- **Data:** TinyStories (`roneneldan/TinyStories`), packed, ~250M training tokens.
- **Base:** `projects/tinyv2.yaml` (seq 2048, batch 8, WSD-style LR schedule,
  AdamW). The single-head model is **eager-attention only** (no flex/compile, no
  KV cache), so these runs are slower per step than the other tiny experiments.
- **Tokenizer:** shared wikitext-8K BPE (the singlehead model's default).
- **`save_strategy: steps`** so the trained models are written to disk for the
  generation test (the `tinyv2` default is `no`).

## Results

All five runs trained on the same 258M tokens. Best eval loss (cross-entropy):

| Run | Non-emb params | Best eval loss |
|---|---:|---:|
| `mlp_small` (attn + MLP, ~24M) | 19.9M | **1.162** |
| `attn_only_matched_small` (attn-only, ~28M) | 20.3M | 1.361 |
| `mlp_4m` (attn + MLP, ~4.6M) | 1.7M | 1.391 |
| `attn_only_small` (attn-only, ~8M) | 4.2M | 1.493 |
| `attn_only_4m` (attn-only, ~2.8M) | 0.8M | 1.728 |

![Best eval loss](assets/final_loss_bar.png)

![Training and eval loss vs tokens](assets/loss_comparison.png)

Three things stand out:

1. **At equal parameters, the MLP wins decisively.** `mlp_small` and
   `attn_only_matched_small` have the *same* non-embedding budget (~20M), yet the
   MLP model is **0.20 eval loss** ahead (1.16 vs 1.36). Spending that capacity on
   wider, deeper attention does **not** reproduce what the MLP does — so the MLP's
   value is not merely "more parameters."

2. **The MLP is strikingly parameter-efficient.** `mlp_4m` carries only **1.7M**
   non-embedding parameters yet reaches 1.39 — essentially tying the
   param-matched attention-only model with **20M** non-embedding parameters
   (1.36), and comfortably beating the 8M attention-only model (1.49). A little
   MLP compute is worth an order of magnitude more attention compute.

3. **Adding attention capacity helps, but with diminishing returns toward the
   MLP.** Going from the 8M to the 28M attention-only model buys 0.13 eval loss
   (1.49 → 1.36) — real, but it stalls ~0.20 short of the equal-param MLP model
   and well short of where the trend would need to continue.

(MFU is low across the board — the single-head model runs eager-only with no
`torch.compile` or flash kernels, so these are not throughput-representative
numbers; the deeper/wider runs simply keep the GPU busier.)

## Generation (subjective)

The trained models are sampled against
[`prompts/tiny_stories.yaml`](../../../prompts/tiny_stories.yaml) with
`examples/snippets/prompt_test.py`. Training writes weights under
`output_models/<run>/checkpoints/`, so co-locate the final checkpoint with the
model's config/code first:

```bash
RUN=mlp_small
CKPT=$(ls -d output_models/$RUN/checkpoints/checkpoint-* | sort -t- -k2 -n | tail -1)
cp "$CKPT"/pytorch_model*.bin "$CKPT"/pytorch_model.bin.index.json output_models/$RUN/

python ../../snippets/prompt_test.py \
    output_models/$RUN ../../../prompts/tiny_stories.yaml \
    --device cuda:0 --max-new-tokens 128 --no-kv-cache
```

> **Note:** the single-head model does not implement KV-cache decoding, so pass
> `--no-kv-cache` (the harness otherwise feeds attention only the latest token
> and output degenerates). The samples below use it.

The eval-loss numbers understate how different these models *feel* to sample.

**`mlp_small` (attn + MLP, 24M) — coherent and on-topic:**

> Jack wanted to read a book, so he went to **the library. It was very big and had
> lots of books that he liked, but it was too tall for him to reach. He looked
> around, but there were no books here. He noticed a man sitting in the corner…**

**`attn_only_small` (attention-only, 8M) — locally fluent, but unstable:**

> Alice was so tired when she got back home so she went **to bed. But as soon as
> Janielielielium Tri Anderson experered Mart Martsey declared Simon
> Simoniotmondmond 8 8 years1446 degrees 5 degrees 000…**

It produces good local phrases, then falls off a cliff into repetition loops or
token-soup — most often when the prompt is short or the continuation is
uncertain. This is the attention-only failure mode the Transformer Circuits work
predicts: with no per-token MLP, the model behaves like a strong bigram /
skip-trigram copier but has nothing to stabilize a longer trajectory.

**`attn_only_matched_small` (attention-only, 28M) — the degeneration is gone:**

> Alice was so tired when she got back home so she went **to bed with a mischievous
> smile on her face. She closed her eyes and fell asleep, dreaming of all the fun
> she would have tomorrow.**

Crucially, the param-matched model is *coherent and stable* — none of the
token-soup of its smaller sibling. So a good chunk of what looked like "the MLP
matters" is really "enough capacity matters"; given equal parameters, attention
alone can hold a story together.

**`mlp_4m` (attn + MLP, 4.6M) — coherent despite being tiny:**

> Jack wanted to read a book, so he went to **find an interesting book. He looked
> around and finally found it in the kitchen. He was so excited! But he felt
> frustrated because when he opened the book and saw that it had been empty!**

## Subjective analysis

The MLP plays two separable roles here:

- **A capacity role** — most of the dramatic difference between the *small*
  attention-only model and the others is just that removing the MLP threw away
  ~65% of the model. Give that capacity back as attention (the param-matched run)
  and the worst symptom, degenerate generation, disappears entirely.
- **A qualitative, efficiency role that capacity does *not* buy back** — at equal
  non-embedding parameters the MLP model is still 0.20 nats ahead and reads as
  sharper and more on-topic, and a 1.7M-parameter MLP model rivals a 20M-parameter
  attention-only one. The per-token nonlinear computation an MLP provides is
  something attention depth and width approximate only inefficiently.

The tidy reading, consistent with *A Mathematical Framework for Transformer
Circuits*: **attention moves and copies information; the MLP computes on it.** You
can lean hard on attention and still produce fluent TinyStories, but the MLP is a
much cheaper place to put the "thinking," and at a fixed parameter budget it wins.

## Reproducing

```bash
cd examples/tiny_experiments/attention_only

# Queue all four runs, one GPU each (needs a running forgather server)
for cfg in mlp_4m attn_only_4m mlp_small attn_only_small; do
    forgather -t "$cfg.yaml" submit --requested-gpus 1
done

# When `forgather job list` shows them all done, regenerate the plots
python assets/generate_plots.py

# Generation test (per model) — see "Generation" above for the checkpoint
# co-location step and why --no-kv-cache is required.
python ../../snippets/prompt_test.py output_models/mlp_small \
    ../../../prompts/tiny_stories.yaml --no-kv-cache
```

## References

- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* — <https://transformer-circuits.pub/2021/framework/index.html>
- Press et al. 2021, *Train Short, Test Long: Attention with Linear Biases (ALiBi)* — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Eldan & Li 2023, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* — [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
