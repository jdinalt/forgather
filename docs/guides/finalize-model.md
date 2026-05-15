# Finalizing a Trained Model

After pre-training, you typically want to leave the original training output
directory untouched (so the training state stays reproducible) and produce a
separate, clean directory ready for fine-tuning, sharing, or inference.

The `forgather finalize` command builds that destination directory in one step:

- Copies the model source code (`*.py`)
- Copies the tokenizer (with optional new tokens and a chat template)
- Copies `config.json` (with token IDs synced to the tokenizer)
- Synthesizes a `generation_config.json` (or carries the source's forward,
  merging in any new stop tokens)
- Preserves exactly one checkpoint -- by default the latest -- with
  `optimizer_state.pt` carried optionally and scheduler / dataset / RNG /
  trainer state always dropped

## Quick start

```bash
# Duplicate the latest checkpoint into a clean handoff directory
forgather finalize output_models/wds out/wds_final

# Same, but preserve optimizer state for warm-start fine-tuning
forgather finalize output_models/wds out/wds_final --keep-optimizer

# Add ChatML tokens, set a chat template, and synthesize a sampling
# generation_config from a preset:
forgather finalize output_models/wds out/wds_chatml \
    --add-tokens chatml.yaml -t chatml.jinja \
    --generation-config precise

# Pull from a specific (non-latest) checkpoint
forgather finalize output_models/wds out/wds_final \
    -c output_models/wds/checkpoints/checkpoint-385440
```

## CLI reference

```
forgather finalize SOURCE DEST [options]
```

The destination must not exist; it is created by the command.

### Source selection

| Option | Description |
|--------|-------------|
| `-c, --checkpoint PATH` | Source checkpoint directory. Defaults to the latest under `SOURCE/checkpoints/`; if there is no `checkpoints/` directory the loader falls back to `SOURCE` itself. |

### Vocabulary and chat template

| Option | Description |
|--------|-------------|
| `--add-tokens YAML` | YAML file specifying tokens to add. Same format as `forgather convert --add-tokens`. |
| `--skip-default-tokens` | Don't auto-add a PAD token if missing. |
| `-t, --chat-template-path FILE` | Jinja2 chat template applied to the tokenizer. If neither the source nor this flag provide one, finalize logs a warning. |

### Stop tokens (for `generation_config.json`)

By default, when `--add-tokens` introduces tokens named `<|im_end|>`,
`<|eot|>`, or `<|end_of_turn|>`, those IDs are merged into the
`generation_config.eos_token_id` list alongside the original EOS so generation
stops on either. The original `eos_token_id` in `config.json` is **not**
modified -- only the generation config gets the merged list.

| Option | Description |
|--------|-------------|
| `--no-auto-stop-tokens` | Disable auto-detection of end-of-turn tokens. |
| `--stop-tokens "TOK1,TOK2"` | Explicit additional stop-token strings. |

### Generation config

| Option | Description |
|--------|-------------|
| `--generation-config carry` | (Default) Copy source `generation_config.json` if present, else synthesize a minimal `{bos,pad,eos}` config. |
| `--generation-config none` | Skip writing `generation_config.json` entirely. |
| `--generation-config PATH` | Load directly from a JSON file in the Forgather inference-preset format (keys: `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `num_beams`, ...). |
| `--generation-config NAME` | Bare name resolved against `~/.config/forgather/generation_config/NAME.json`. No presets ship with this branch -- populate that directory yourself, or pass an explicit `PATH`. |

Forgather presets use `max_tokens` (matching chat-completion APIs); finalize
translates this to HuggingFace's `max_new_tokens` and infers `do_sample`
when not explicit. Token IDs (`bos`, `pad`, `eos`) are always overlaid from
the (possibly-updated) tokenizer last.

### Checkpoint contents

| Option | Description |
|--------|-------------|
| `--keep-optimizer` | Carry `optimizer_state.pt` from the source checkpoint into the dest checkpoint. Scheduler, dataset, RNG, and trainer state are always dropped. |
| `--root-copy` | Write weights only at the model root and skip creating `DEST/checkpoints/`. Mutually exclusive with `--keep-optimizer`. The default writes weights into `DEST/checkpoints/checkpoint-N/` and creates relative symlinks at the root for HuggingFace `from_pretrained` compatibility. |

### Storage

| Option | Description |
|--------|-------------|
| `--safetensors` | Save as safetensors. Default is PyTorch (`.bin`). PyTorch handles tied embeddings natively; safetensors raises on save when weights are tied. |
| `--dtype {bfloat16,float16,float32}` | Cast weights to this dtype before saving. Default: keep the dtype the source checkpoint was saved in. |
| `--device STR` | Device for loading the model during finalize (default `cpu`). |

### Quantization

| Option | Description |
|--------|-------------|
| `--quantize RECIPE` | Quantize the model before saving using the named torchao recipe. Works on any source. If the source was trained with `--qat-recipe`, this completes the QAT round-trip and keeps the QAT training-time accuracy benefit. If the source is plain bf16, this is standard post-training quantization (PTQ). See [QAT Training](../trainers/qat-training.md) for the recipe list and the QAT-vs-PTQ tradeoff. |

Examples:

```bash
# QAT round-trip: source was trained with --qat-recipe
forgather finalize output_models/qat_run out/qat_int8_int4 \
    --quantize int8-dynamic-act-int4-weight

# PTQ: plain bf16 source, same flag
forgather finalize output_models/bf16_run out/bf16_int8_int4_ptq \
    --quantize int8-dynamic-act-int4-weight
```

When `--quantize` is set, finalize always writes `.bin`: torchao's
quantized tensor subclasses don't expose a single `.storage().data_ptr()`,
which the safetensors writer requires. If `--safetensors` is passed
alongside `--quantize`, it is silently disabled with a warning.

Finalize also writes a `quantization_config` block into `config.json`
with the recipe. Forgather's native checkpoint loader consumes this
hint (with a state_dict scan as fallback) and installs the matching
quantized linear modules before weights load — so `forgather eval`,
the inference server, and any other tool using the native loader
handle the artifact transparently with no caller-side flag. The same
block also enables HF `AutoModelForCausalLM.from_pretrained()`
auto-detection for non-Forgather consumers. See [Evaluating Quantized
Models](../trainers/qat-training.md#evaluating-quantized-models).

### Misc

| Option | Description |
|--------|-------------|
| `--dry-run` | Resolve and report what would be done; write nothing. |
| `--log-level LEVEL` | Logging level (default `INFO`). |

## Destination layout

Default (HuggingFace-compatible with one preserved checkpoint):

```
DEST/
├── config.json
├── tokenizer.json, tokenizer_config.json, special_tokens_map.json
├── generation_config.json                  # synthesized or carried
├── *.py                                    # model source from SOURCE root
├── .package_files_copied
├── pytorch_model*.bin* + index.json        # SYMLINKS into checkpoint-N
└── checkpoints/
    └── checkpoint-N/
        ├── checkpoint_manifest.json        # rewritten: only kept components
        ├── pytorch_model*.bin* + index.json
        └── optimizer_state.pt              # only if --keep-optimizer
```

With `--root-copy`:

```
DEST/
├── config.json
├── tokenizer*.json
├── generation_config.json
├── *.py
├── .package_files_copied
└── pytorch_model*.bin* + index.json        # at root, no checkpoints/
```

## Token configuration format

The `--add-tokens` flag accepts a YAML file. Bundled, ready-to-use configs
live in [`add_tokens_config/`](../../add_tokens_config/) -- start there for
common cases (e.g. ChatML setup). For the full format reference, init
strategies, and authoring guide, see
[Add-Tokens Config](add-tokens-config.md).

Short example:

```yaml
eos_token: "<|im_end|>"

special_tokens:
  - "<|im_start|>"

pad_token:
  token: "<|pad|>"
  init: "zero"
  if_missing: true
```

## See also

- **[Model Conversion](model-conversion.md)** -- full HuggingFace ↔ Forgather
  conversion, also supports `--add-tokens` and `-t`.
- **[EOS Tokens and `generate()` Stopping Criteria](eos-and-generate-stopping.md)** --
  theory of operation: how HF's `generate()` resolves stopping across the
  multiple files that carry EOS information.
- **[QAT Training](../trainers/qat-training.md)** -- pair `--quantize` here
  with `--qat-recipe` at training time for the full QAT round-trip, or use
  `--quantize` alone on a plain bf16 source for post-training quantization.
