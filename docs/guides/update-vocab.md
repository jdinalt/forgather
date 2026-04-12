# Vocabulary and Chat Template Update

The `update-vocab` command modifies the vocabulary and/or chat template of an
existing model without converting between formats. This is useful when preparing
a pretrained model for fine-tuning -- for example, adding chat-related special
tokens and a chat template before instruction tuning.

The command works with both HuggingFace and Forgather models. When
`OUTPUT_PATH` is omitted the model is modified **in-place**. If only the chat
template changed (no new tokens added), only the tokenizer files are rewritten
-- model weights are not touched.

## Quick start

```bash
# Set chat template in-place (only rewrites tokenizer files, fast)
forgather update-vocab --skip-default-tokens \
    -t template.jinja /path/to/model

# Add tokens and set chat template in-place
forgather update-vocab --add-tokens tokens.yaml \
    -t template.jinja /path/to/model

# Add tokens, save to a new directory
forgather update-vocab --add-tokens tokens.yaml \
    /path/to/model /path/to/output
```

## CLI reference

```bash
forgather update-vocab [OPTIONS] MODEL_PATH [OUTPUT_PATH]
```

**Positional arguments:**

| Argument | Description |
|----------|-------------|
| `MODEL_PATH` | Path to source model (HuggingFace or Forgather) |
| `OUTPUT_PATH` | Output directory for updated model. If omitted, modifies in-place. |

**Chat template:**

| Option | Description |
|--------|-------------|
| `-t, --chat-template-path FILE` | Jinja2 chat template file to apply to the tokenizer |

**Vocabulary:**

| Option | Description |
|--------|-------------|
| `--add-tokens YAML_FILE` | YAML file specifying tokens to add |
| `--skip-default-tokens` | Don't auto-add a PAD token if missing |

**Save format:**

| Option | Description |
|--------|-------------|
| `--save-format {huggingface,sharded}` | Output format (default: huggingface) |
| `--safetensors` | Use safetensors serialization |

**Model options:**

| Option | Description |
|--------|-------------|
| `--dtype {bfloat16,float32,float16}` | Override dtype for model loading |
| `--device {cpu,cuda,...}` | Device for model operations (default: cpu) |
| `--no-trust-remote-code` | Disable trusting remote code (trusted by default for local models) |

**Utility:**

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview changes without saving |
| `--log-level LEVEL` | Logging level (default: INFO) |

## Common workflows

### Chat template only

Set a chat template on a pretrained base model in-place before fine-tuning.
Only tokenizer files are rewritten -- model weights are not touched:

```bash
forgather update-vocab --skip-default-tokens \
    -t chatml.jinja \
    ~/models/my_pretrained
```

### Add chat tokens and template

Add chat-related special tokens and set the chat template in one pass:

```bash
cat > chat_tokens.yaml << EOF
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"
EOF

forgather update-vocab --add-tokens chat_tokens.yaml \
    -t chatml.jinja \
    ~/models/my_pretrained
```

### Vocabulary extension to a new directory

Add tokens and save to a separate directory (preserves the original):

```bash
forgather update-vocab --add-tokens domain_tokens.yaml \
    ~/models/base ~/models/base_extended
```

## Token configuration format

The `--add-tokens` flag accepts a YAML file. The format is the same as
`forgather convert --add-tokens` -- see
[Adding tokens during conversion](model-conversion.md#adding-tokens-during-conversion)
for the full specification.

Short example:

```yaml
pad_token:
  token: "<|pad|>"
  init: "zero"
  if_missing: true

special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"
```

Initialization strategies: `"zero"` (zero-fill), `"mean"` (mean of existing
embeddings), `"copy:ID"` (copy from token ID).

## See also

- **[Model Conversion](model-conversion.md)** -- full HuggingFace/Forgather
  conversion (also supports `--add-tokens` and `-t`)
- **[tools/update_vocab/README.md](../../tools/update_vocab/README.md)** --
  detailed reference with additional examples
