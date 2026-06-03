# Small Models

Train and compare different small language model architectures on the
[Fineweb-Edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
dataset with a shared 8K-token wikitext BPE tokenizer. All models are ~30M
parameters, allowing direct comparison of architecture choices under identical
conditions.

This is the larger sibling of [`tiny_models`](../tiny_models/): real text
(Fineweb-Edu rather than TinyStories), a longer context (`seq_len` 4096), and a
~7x bigger model — a fairer test of architecture differences than the 4M-param
TinyStories models, where the task is almost too easy to separate them.

The project builds on the **`projects/small.yaml`** base template (DDP trainer,
`flex_attention`, `torch.compile`), with two project-level changes in
[`templates/project.yaml`](templates/project.yaml):

- **WSD learning-rate schedule** (warmup-stable-decay), matching the
  [`diloco`](../diloco/) example: constant LR after warmup, then anneal to
  `min_lr` over the final phase.
- **1B-token budget** (double the small baseline's 500M), with the WSD
  **annealing phase set to 10%** of the budget (100M tokens).

## Configurations

All seven are Forgather-native model implementations, each ~30M parameters with
the shared wikitext-8K tokenizer:

| Config | Architecture | Key features | Params |
|--------|-------------|--------------|-------:|
| `small_causal.yaml` | Vanilla Transformer | Basic decoder-only transformer (MHA, learned PE) | 32M |
| `small_llama.yaml` | Llama | Pre-layer-norm, RoPE, SiLU GLU, GQA (untied embeddings) | 34M |
| `small_llama_canon.yaml` | Llama + Canon | Llama with Canon convolutional layers for local token mixing (tied embeddings) | 30M |
| `small_deepone.yaml` | DeepOne | Post-layer-norm, Deepnet initialization, ALiBi positional encoding | 38M |
| `small_qwen3.yaml` | Qwen3 | QK-norm, GQA, tied embeddings | 30M |
| `small_mistral.yaml` | Mistral | Llama variant with sliding-window attention, GQA | 34M |
| `small_gemma3.yaml` | Gemma-3 | Interleaved sliding/full attention, GLU + gelu-tanh, dual RoPE, tied embeddings | 34M |

Each config points the base template at a model project's `small.yaml`
definition under [`examples/models/`](../../models/). All seven share the same
wikitext-8K tokenizer and ~512-hidden / 10-layer shape, so differences in the
loss curves reflect architecture, not size or vocabulary.

## Usage

### Train a single model

Run locally in the foreground on a chosen GPU:

```bash
forgather -t small_qwen3.yaml train -d 0
```

### Train all models for comparison

Submit every config to the forgather-server scheduler. Jobs run in the
background and are placed on GPUs automatically as they free up; with seven
configs on five GPUs, five run concurrently and the rest start as GPUs free:

```bash
# Queue every config at one GPU per job (skips the project template).
for cfg in $(forgather ls | grep -oP '[\w./-]+\.yaml' | grep -v '^project\.yaml' | tr -d '[]'); do
    forgather -t "$cfg" submit --requested-gpus 1
done
```

`forgather submit` is the explicit spelling of `forgather train --schedule`.
The base template is a DDP config (`nproc_per_node = "gpu"`), so it would
otherwise spread one model across every visible GPU; `--requested-gpus 1` pins
each job to a single GPU (the DDP trainer transparently falls back to the
single-process path at world-size 1). To instead run one model across multiple
GPUs, raise the count — e.g. `--requested-gpus 4`.

> The bracket-stripping (`tr -d '[]'`) and `grep -v project.yaml` are there
> because `forgather ls` wraps the **default** config in `[brackets]` and lists
> the project template alongside the runnable configs.

These runs use `flex_attention` + `torch.compile` (from the base template).
Compilation adds a one-time startup cost of a couple of minutes but is well
worth it over a 1B-token run — it roughly doubles throughput and keeps ALiBi
(DeepOne) and sliding-window (Gemma-3) attention memory-flat at `seq_len` 4096.

This requires a running forgather server (`forgather server` starts one in
cluster mode). See `docs/guides/server-cli.md` for the full guide.

### Working with the scheduler

```bash
forgather job list                  # queued / running / done, per row
forgather job status <queue_id>     # live loss / step / throughput
forgather job tail <queue_id>       # stream a running job's output
forgather job logs <queue_id>       # dump the full captured log
forgather job stop <queue_id>       # graceful stop (saves a final checkpoint)
forgather job abort <queue_id>      # stop immediately without saving (e.g. a throughput probe)
forgather job cancel <queue_id>     # cancel a queued or running job
forgather job scheduler pause       # hold the queue (running jobs continue)
forgather job scheduler resume
forgather job cleanup               # remove terminal job records
```

Check GPU usage:

```bash
nvidia-smi                 # hardware view: memory + utilization per GPU
forgather gpu status       # scheduler's view: which GPUs it considers in use
```

### Compare results

```bash
python assets/generate_plots.py
```

## Experimental results

<!-- RESULTS -->

## Adding a new model

1. Create a model project under `examples/models/` (or use an existing one),
   with a `small.yaml` config that uses the shared wikitext-8K tokenizer and is
   sized to ~30M parameters (verify with
   `forgather -t small.yaml model -r construct`).
2. Add a config in `templates/configs/` that sets `ns.model_project_dir` and
   `ns.model_project_config` to point to it.
3. Run `forgather ls` to verify it parses, and
   `forgather -t <cfg> model -r --device cuda:0 test` to confirm the model runs.

Note: with five GPUs, the first five models run concurrently and any extras
start as GPUs free up — so growing the set past five adds roughly one model's
runtime per extra GPU-wave rather than running everything at once.
