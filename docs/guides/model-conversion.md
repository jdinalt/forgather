# Model Conversion

Forgather uses its own model format -- dynamically generated Python code with
semantic parameter naming (e.g., `attention.query_linear` instead of
`self_attn.q_proj`). The model conversion tool provides bidirectional conversion
between HuggingFace Transformers models and Forgather models.

## Quick start

```bash
# HuggingFace -> Forgather
forgather convert Llama-3.2-1B-Instruct/ fg_Llama-3.2-1B-Instruct/

# Forgather -> HuggingFace (after training)
forgather convert --reverse fg_Llama-3.2-1B-Instruct/ hf_Llama-3.2-1B-Instruct/
```

The conversion direction is auto-detected: if the source has a HuggingFace
`config.json` with a `model_type` field, it converts HF to Forgather. If the
source has a Forgather config with an `hf_model_type` field (set during the
original HF-to-Forgather conversion), it converts back to HF.

## Supported models

Built-in converters are provided for:

| Model type | HF config class | Converter location |
|------------|----------------|--------------------|
| Llama | `LlamaConfig` | `examples/models/llama/src/converter.py` |
| Mistral | `MistralConfig` | `examples/models/mistral/src/converter.py` |
| Qwen3 | `Qwen3Config` | `examples/models/qwen3/src/converter.py` |

Converters are discovered automatically from `examples/models/*/src/converter.py`.

## CLI reference

```bash
forgather convert [OPTIONS] SRC_MODEL_PATH DST_MODEL_PATH
```

**Direction:**

| Option | Description |
|--------|-------------|
| (default) | Auto-detect direction from source config |
| `--reverse` | Force Forgather-to-HuggingFace conversion |
| `--model-type {llama,mistral,qwen3}` | Override detected model type |

**Model properties:**

| Option | Description |
|--------|-------------|
| `--dtype {bfloat16,float32,float16}` | Override output dtype (default: inherit from source) |
| `--max-length INT` | Override maximum sequence length |

**Checkpoint (Forgather-to-HF only):**

| Option | Description |
|--------|-------------|
| `-c, --checkpoint-path PATH` | Specific checkpoint to convert (default: latest) |

**Vocabulary:**

| Option | Description |
|--------|-------------|
| `--add-tokens YAML_FILE` | Add tokens from a YAML definition file |
| `--skip-default-tokens` | Don't auto-add a PAD token if missing |

**Testing:**

| Option | Description |
|--------|-------------|
| `--device {cpu,cuda,...}` | Device for validation (default: cpu) |
| `-g, --generation-test` | Run a generation test after conversion |
| `--prompt TEXT` | Custom prompt for generation test |
| `--debug-params` | Print parameter name mappings |
| `--dry-run` | Run conversion without saving |

**Extensibility:**

| Option | Description |
|--------|-------------|
| `--converter-path PATH` | Additional directories to search for converter plugins (repeatable) |

## How conversion works

### Parameter remapping

The core of conversion is recursive regex-based parameter name remapping. Each
converter defines patterns that transform parameter names between the two formats.

For example, the Llama converter maps:

| HuggingFace | Forgather |
|-------------|-----------|
| `model.layers.0.self_attn.q_proj.weight` | `causal_lm.layer_stack.layers.0.attention.query_linear.weight` |
| `model.layers.0.mlp.gate_proj.weight` | `causal_lm.layer_stack.layers.0.feedforward.gate_proj.weight` |
| `model.embed_tokens.weight` | `causal_lm.input_encoder.embedding.weight` |
| `model.norm.weight` | `causal_lm.layer_norm.weight` |

The patterns are defined as nested regex substitution rules, applied recursively
from the outermost module name inward. This keeps patterns composable -- a shared
pattern like `layers.(\d+).` handles layer indexing regardless of what comes after it.

### Validation

After conversion, the tool loads both the source and destination models and
compares their logits on a test input. This catches parameter mapping errors
or weight transformation bugs.

### Metadata

When converting HF to Forgather, the tool stores `hf_model_type` in the
Forgather config. This metadata enables auto-detection when converting back,
and records which HuggingFace architecture the model originated from.

## Adding tokens during conversion

The `--add-tokens` flag accepts a YAML file defining tokens to add to the
tokenizer and how to initialize their embeddings:

```yaml
# Special tokens
eos_token:
  token: "<|end_of_text|>"
  init: "mean"             # Initialize to mean of existing embeddings
pad_token:
  token: "<|pad|>"
  init: "zero"             # Zero-initialize
  if_missing: true          # Only add if not already present

# Additional special tokens
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"

# Regular tokens
regular_tokens:
  - "custom_token_1"
```

Initialization strategies: `"zero"` (zero-fill), `"mean"` (mean of existing
embeddings), `"copy:ID"` (copy from token ID).

By default, a `[PAD]` token is added with zero initialization if the tokenizer
doesn't have one. Use `--skip-default-tokens` to disable this.

## Writing a custom converter

Converters are Python classes that extend `HFConverter` and register themselves
with the `@register_converter` decorator. Place them at
`examples/models/<model_name>/src/converter.py` for automatic discovery, or in
any directory passed via `--converter-path`.

```python
from forgather.ml.model_conversion.registry import register_converter
from forgather.ml.model_conversion.hf_converter import HFConverter
from forgather.ml.model_conversion.standard_mappings import (
    STANDARD_HF_TO_FORGATHER,
    STANDARD_FORGATHER_TO_HF,
)
from transformers import MyModelConfig, MyModelForCausalLM


@register_converter("my_model")
class MyModelConverter(HFConverter):
    def __init__(self):
        super().__init__(model_type="my_model")

    def get_hf_config_class(self):
        return MyModelConfig

    def get_hf_model_class(self):
        return MyModelForCausalLM

    def get_project_info(self):
        """Point to the Forgather model project for this architecture."""
        return {
            "project_dir": "/path/to/examples/models/my_model",
            "config_name": "default.yaml",
        }

    def get_parameter_mappings(self, direction):
        if direction == "to_forgather":
            return MY_HF_TO_FORGATHER_PATTERNS
        else:
            return MY_FORGATHER_TO_HF_PATTERNS

    def get_config_field_mapping(self, direction):
        # Use standard mappings for common fields
        if direction == "to_forgather":
            return dict(STANDARD_HF_TO_FORGATHER)
        else:
            return dict(STANDARD_FORGATHER_TO_HF)

    def validate_source_config(self, config, direction):
        """Optional: validate assumptions about the source model."""
        if direction == "to_forgather":
            assert config.hidden_act == "silu", "Only SiLU activation supported"
```

The parameter mapping patterns are recursive regex substitution lists. See
`standard_mappings.py` and existing converters (e.g.,
`examples/models/llama/src/converter.py`) for reference.

## See also

- **[Vocabulary and Chat Template](update-vocab.md)** -- add tokens or set chat
  templates on an existing model without converting between formats
- **[Fixing End-of-Sequence Token Issues](fixing-eos-token-issues.md)** --
  diagnose and fix runaway generation when an older conversion left
  `generation_config.json` out of sync with the updated tokenizer EOS
