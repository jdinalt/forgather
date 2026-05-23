# Evaluating Models with `forgather eval`

The `forgather eval` subcommand runs a loss/perplexity evaluation against a
named dataset and writes the results as JSON alongside the model, plus a
human-readable summary to the console.

Unlike `forgather train`, an eval run does not require you to construct a
training project, locate a dataset project, hand-craft a long command line,
or dig through `runs/` for the loss. Named eval configs (`c4`,
`tinystories`, ...) are discovered automatically.

## Quick start

```bash
# List available eval configs
forgather eval list

# Show the details of a config
forgather eval show c4

# Evaluate a model on TinyStories with all visible GPUs (DDP)
forgather eval test tinystories -M /path/to/model

# Evaluate on FineWeb-Edu-Dedup using pipeline parallelism for a model that
# doesn't fit on a single GPU
forgather eval test fineweb-edu-dedup \
    -M /path/to/model \
    --trainer pipeline \
    --max-length 4096 \
    --dtype bfloat16
```

## Output

Rank 0 writes:

```
<model_dir>/evals/<eval_name>_<timestamp>/results.json
```

`results.json` contains:

```json
{
  "eval_name": "tinystories",
  "config_name": "Eval: TinyStories",
  "model_path": "/path/to/model",
  "checkpoint_path": null,
  "dataset_proj": "/.../examples/datasets/roneneldan",
  "dataset_config": "tinystories-packed.yaml",
  "dataset_target": "test_dataset",
  "batch_size": 16,
  "max_length": 512,
  "dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "trainer": "ddp",
  "world_size": 2,
  "eval_loss": 1.699,
  "perplexity": 5.47,
  "bpb": 0.612,
  "bpc": 0.612,
  "tokens_per_byte": 0.249,
  "total_bytes": 1599000,
  "total_chars": 1599000,
  "total_predicted_tokens": 399300,
  "wall_time_s": 12.3,
  "timestamp": "2026-04-18T05:11:34Z"
}
```

The same information is printed as a summary table at the end of the run.

### Cross-tokenizer comparison: use BPB, not perplexity

`perplexity = exp(eval_loss)` and `eval_loss` is a **per-token** cross-entropy,
so its denominator (tokens) changes with the tokenizer. A model whose
tokenizer compresses more bytes into each token gets a lower perplexity
mechanically, with no actual quality difference. Comparing perplexities
across models with different vocabularies is therefore misleading — the
gap can be 20%+ from tokenization alone.

`forgather eval test` reports two tokenizer-agnostic alternatives:

- **`bpb`** — bits-per-byte: `eval_loss × tokens_per_byte / ln(2)`. The
  denominator is bytes of source UTF-8 text, not tokens, so the metric is
  comparable across any pair of models regardless of vocabulary or
  tokenization scheme. Lower is better. This is the de facto standard
  (used in The Pile, GPT-3, Karpathy's autoresearch, etc.). **Use this for
  any cross-model comparison.**
- **`bpc`** — bits-per-character: same idea, denominator is Unicode code
  points. Equal to `bpb` for ASCII; useful for non-Latin scripts where
  per-character entropy is the more intuitive unit.
- **`tokens_per_byte`** — the conversion factor itself. Surfacing it makes
  the source of any past confusion explicit and lets you convert legacy
  perplexity numbers on the fly: `bpb ≈ log2(perplexity) × tokens_per_byte`.

Byte and character counts come from decoding `input_ids` of the eval
dataset prefix that the trainer actually consumed, via
`tokenizer.decode(..., skip_special_tokens=True)`. Token counts are over
predicted positions only (`labels[1:] != -100`, accounting for the causal
shift). The `bpb` formula assumes `eval_loss` equals the true token-mean
cross-entropy. The trainer reports a mean of per-step-mean losses, which
matches the token-mean exactly when batches contain equal numbers of
valid (non-ignored) tokens — the common case for eval with fixed
`max_length` and full-length sequences. Variable-length batches introduce
a small (typically <1%) approximation error.

## Subcommands

### `forgather eval list`

Prints one row per discovered eval config: name, description, and the
project file that defines it.

### `forgather eval show NAME`

Prints the resolved metadata for `NAME`: dataset project, dataset config,
target, default batch size / max length. Pass `--pp` to dump the full
preprocessed YAML.

### `forgather eval test NAME [flags]`

Runs the eval. All flags are optional; defaults come from the config.

| Flag | Default | Purpose |
|------|---------|---------|
| `-M`, `--model PATH` | current project's output dir | Model to evaluate |
| `-d`, `--devices "0,1"` | all visible GPUs | `CUDA_VISIBLE_DEVICES` for the child process |
| `--trainer {ddp,simple,pipeline}` | `ddp` | Trainer backend (see below) |
| `--checkpoint PATH` | *(none)* | Resume from an explicit checkpoint |
| `--no-checkpoint` | off | Load via `AutoModelForCausalLM.from_pretrained` instead of resuming |
| `--batch-size N` | config default | Per-device eval batch size |
| `--max-length N` | config default | Max sequence length |
| `--stride N` | config default (0) | Packing stride (forwarded to the dataset project) |
| `--max-steps N` | `-1` (all) | Cap the number of eval batches |
| `--dtype {bfloat16,float16,float32}` | `bfloat16` | Model dtype |
| `--attn-implementation NAME` | `sdpa` | `sdpa`, `flash_attention_2`, `flex_attention`, ... |
| `--compile` | off | `torch.compile` the model |
| `--output-dir PATH` | `$MODEL` | Override where `evals/` is written |
| `--dry-run` | off | Print the torchrun command without executing |

### Trainer backends

- **`ddp` (default)** — `DDPTrainer` under `torchrun --nproc-per-node gpu`.
  Uses every visible GPU unless you pass `-d`. This is the standard choice
  for models that fit on a single GPU.
- **`simple`** — single-process base `Trainer`. Faster startup, no
  `torchrun`. Useful for very small models or debugging.
- **`pipeline`** — `PipelineTrainer` under `torchrun`. Splits the model
  across GPUs via `create_manual_causal_lm_splitter()` with a `ScheduleGPipe`
  schedule. Use this when a single GPU cannot hold the model. The data
  collator switches to `padding="max_length"` for pipeline runs so every
  microbatch has a matching shape.

### Model loading

By default the trainer constructs the model on the target device with
`no_init_weights()`, then loads the latest checkpoint under `--model` (the
same mechanism `forgather train` uses). Pass an explicit path with
`--checkpoint PATH` to pin a specific one, or `--no-checkpoint` to load via
`AutoModelForCausalLM.from_pretrained` on the model directory.

**Quantized models** (artifacts produced by `forgather finalize --quantize`)
load transparently through the standard checkpoint-resume path. Forgather's
native loader (`forgather.ml.sharded_checkpoint.load_checkpoint`) detects
torchao quantization from `config.json`'s `quantization_config` block (or
falls back to scanning the saved state_dict) and installs the matching
quantized linear modules before `load_state_dict` runs. No extra flag,
no caller-side recipe argument. See [QAT Training § Evaluating Quantized
Models](../trainers/qat-training.md#evaluating-quantized-models).

The tokenizer is always loaded directly from `--model` via
`AutoTokenizer.from_pretrained`.

## Adding a new eval config

Eval configs live under `examples/evaluation/`. Each is a small YAML file
that extends the base eval template `test/test_type.yaml` (from
`templatelib/base/test/`) and sets a few metadata fields:

```yaml
-- extends "test/test_type.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "Eval: My Dataset"
    -- set ns.config_description = "Evaluate on MyDataset test split"
    -- set ns.eval_name = "mydataset"
    -- set ns.dataset_proj = joinpath(ns.forgather_dir, "examples", "datasets", "my_org")
    -- set ns.dataset_config = "my-packed.yaml"
    -- set ns.default_max_length = 1024
    -- set ns.default_batch_size = 8
```

The base template stamps `config_class = "type.evaluation"` into the meta
block, which is what `forgather eval list/show` uses to discover configs.

## User-level search paths

Drop a file at `~/.config/forgather/config.yaml` to extend (or replace) the default
search path:

```yaml
eval:
  search_paths:
    - /path/to/my/extra/eval_projects
  # replace_default: true   # drop the builtin examples/evaluation/ path
```

`forgather eval list` will pick up configs from every listed directory.

## Example: evaluating a pre-trained model across datasets

```bash
MODEL=/mnt/rust/aiassets/models/qwen3-1.7b-base

for NAME in tinystories c4 openorca fineweb-edu-dedup; do
    forgather eval test "$NAME" -M "$MODEL" --max-length 2048 --batch-size 4
done

# Results land in $MODEL/evals/<name>_<timestamp>/results.json for each.
```
