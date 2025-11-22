# Vocabulary Update Tool

The vocabulary update tool provides an easy way to add tokens to an existing model's vocabulary and resize embeddings accordingly. It works seamlessly with both HuggingFace and Forgather models using the same HuggingFace APIs.

## Quick Start

### Basic Usage

```bash
# Add tokens from YAML configuration
./update_vocab.py --add-tokens tokens.yaml /path/to/model /path/to/output

# Use default behavior (adds missing PAD token)
./update_vocab.py /path/to/model /path/to/output

# Skip default tokens for complete control
./update_vocab.py --skip-default-tokens /path/to/model /path/to/output
```

### Via Forgather CLI

```bash
# The tool can also be invoked via the forgather command
forgather update-vocab --add-tokens tokens.yaml /path/to/model /path/to/output
```

## Features

- **Universal Model Support**: Works with both HuggingFace and Forgather models
- **Token Configuration**: Add named special tokens (BOS, EOS, PAD, UNK) and custom tokens
- **Flexible Save Formats**: Save as HuggingFace format or Forgather sharded checkpoints
- **Multiple Serialization Options**: Choose between safetensors or PyTorch format
- **Initialization Strategies**: Zero-init or mean-init for new token embeddings
- **if_missing Support**: Add tokens only if they don't already exist
- **Token Reassignment**: Properly handle replacing existing tokens
- **Default Behavior**: Automatically adds missing PAD token unless disabled
- **Dry Run Mode**: Preview changes without modifying the model

## Installation

The tool is included with Forgather. No additional installation required.

Requirements:
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Forgather framework

## Command-Line Reference

### Basic Arguments

```bash
./update_vocab.py [OPTIONS] MODEL_PATH OUTPUT_PATH
```

**Positional Arguments:**
- `MODEL_PATH`: Path to source model (HuggingFace or Forgather)
- `OUTPUT_PATH`: Output directory for updated model

### Token Configuration

```bash
--add-tokens YAML_FILE
```
Path to YAML file specifying tokens to add. See [Token Configuration Format](#token-configuration-format) below.

```bash
--skip-default-tokens
```
Skip default token handling (adding missing PAD token). Use this for complete control over vocabulary.

### Save Format Options

```bash
--save-format {huggingface,sharded}
```
Choose save format:
- `huggingface` (default): Use `model.save_pretrained()` - saves config automatically
- `sharded`: Use Forgather's `sharded_checkpoint.save_checkpoint()`

```bash
--safetensors
```
Save using safetensors format instead of PyTorch format. Applies to both save formats.

### Model Options

```bash
--dtype {bfloat16,float32,float16}
```
Torch dtype for model loading. If not specified, uses model's existing dtype.

```bash
--device {cpu,cuda,cuda:0,...}
```
Device for model operations. Default: `cpu`

```bash
--trust-remote-code
```
Trust remote code when loading model (required for some custom models).

### Utility Options

```bash
--dry-run
```
Load model and tokenizer, show what would be changed, but don't save. Useful for previewing changes.

```bash
--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
```
Set logging level. Default: `INFO`

## Token Configuration Format

The tool uses the same YAML configuration format as the model converter. See the full specification in [Model Converter Documentation](../convert_model/README.md#vocabulary-extension).

### Basic Example

```yaml
# Named special tokens (optional)
bos_token: "<|begin_of_text|>"
eos_token: "<|end_of_text|>"
pad_token:
  token: "<|pad|>"
  init: "zero"
  if_missing: true

# Additional special tokens (optional)
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"

# Regular tokens (optional)
regular_tokens:
  - "custom_token_1"
  - "custom_token_2"
```

### Token Configuration Options

**Named Special Tokens:**
- `bos_token`: Beginning-of-sequence token
- `eos_token`: End-of-sequence token
- `pad_token`: Padding token
- `unk_token`: Unknown token

**Format Options:**
```yaml
# Simple string format (uses default initialization)
pad_token: "<|pad|>"

# Dict format (custom initialization and if_missing flag)
pad_token:
  token: "<|pad|>"
  init: "zero"        # "zero" or "mean"
  if_missing: true    # Only add if not already set
```

**Initialization Strategies:**
- `"zero"`: Initialize embeddings to zero (recommended for PAD)
- `"mean"`: Initialize to mean of existing embeddings (recommended for BOS/EOS/UNK)
- Defaults: BOS/EOS/UNK use "mean", PAD uses "zero"

**if_missing Flag:**
- When `true`, only adds/sets the token if it doesn't already exist
- Useful for ensuring a token exists without forcing replacement
- Default: `false`

## Usage Examples

### Example 1: Add Chat Template Tokens

Add instruction-tuning tokens to a base model:

```bash
cat > chat_tokens.yaml << EOF
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"
  - "<|system|>"
  - "<|user|>"
  - "<|assistant|>"
EOF

./update_vocab.py --add-tokens chat_tokens.yaml \
    ~/models/llama-7b \
    ~/models/llama-7b-chat
```

### Example 2: Replace PAD Token

Change the PAD token from `[PAD]` to `<|pad|>`:

```bash
cat > new_pad.yaml << EOF
pad_token: "<|pad|>"
EOF

./update_vocab.py --add-tokens new_pad.yaml \
    ~/models/my_model \
    ~/models/my_model_new_pad
```

### Example 3: Add BOS Token to Qwen3

Qwen3 models don't have a BOS token by default. Add one for document packing:

```bash
cat > qwen3_bos.yaml << EOF
bos_token:
  token: "<|begin_of_text|>"
  init: "mean"
  if_missing: true  # Only add if not already set
EOF

./update_vocab.py --add-tokens qwen3_bos.yaml \
    ~/models/qwen3-8b \
    ~/models/qwen3-8b-with-bos
```

### Example 4: Add Multiple Token Types

Add a mix of named tokens and custom tokens:

```bash
cat > tokens.yaml << EOF
# Ensure PAD token exists
pad_token:
  token: "<|pad|>"
  init: "zero"
  if_missing: true

# Add new BOS token
bos_token: "<|begin|>"

# Add chat template tokens
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"

# Add custom domain tokens
regular_tokens:
  - "[CITATION]"
  - "[FIGURE]"
  - "[TABLE]"
EOF

./update_vocab.py --add-tokens tokens.yaml \
    ~/models/base_model \
    ~/models/specialized_model
```

### Example 5: Dry Run Mode

Preview changes without saving:

```bash
./update_vocab.py --add-tokens tokens.yaml --dry-run \
    ~/models/my_model \
    ~/models/my_model_updated
```

Output shows:
- Current vocabulary size
- Tokens that would be added
- New vocabulary size
- But doesn't save the model

### Example 6: Save as Sharded Checkpoint with Safetensors

```bash
./update_vocab.py --add-tokens tokens.yaml \
    --save-format sharded \
    --safetensors \
    ~/models/my_model \
    ~/models/my_model_updated
```

### Example 7: Skip Default PAD Token

If you want complete control and don't want automatic PAD token addition:

```bash
./update_vocab.py --skip-default-tokens \
    ~/models/my_model \
    ~/models/my_model_copy
```

This creates an exact copy without modifying vocabulary.

### Example 8: Load with Specific DType

```bash
./update_vocab.py --add-tokens tokens.yaml \
    --dtype bfloat16 \
    ~/models/my_model \
    ~/models/my_model_updated
```

## Default Behavior

By default, the tool automatically adds a missing PAD token (`[PAD]`) with zero initialization. This ensures models have proper padding support.

**To use default behavior:**
```bash
./update_vocab.py /path/to/model /path/to/output
```

**To disable default behavior:**
```bash
./update_vocab.py --skip-default-tokens /path/to/model /path/to/output
```

**Default behavior is overridden when:**
- You provide `--add-tokens` configuration (your config is used instead)
- You specify `--skip-default-tokens` (no automatic changes)

## Save Format Comparison

### HuggingFace Format (default)

```bash
--save-format huggingface
```

**Advantages:**
- Standard HuggingFace model format
- Compatible with all HuggingFace tools
- Config saved automatically with model
- Single-call API: `model.save_pretrained()`

**Output Structure:**
```
output_path/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors (or pytorch_model.bin)
└── ...
```

### Sharded Checkpoint Format

```bash
--save-format sharded
```

**Advantages:**
- Optimized for large models
- Efficient parallel loading
- Forgather-native format
- Fine control over parameter sharing

**Output Structure:**
```
output_path/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── checkpoint-0.pt (or checkpoint-0.safetensors)
├── checkpoint-1.pt
├── checkpoint-metadata.json
└── ...
```

### Safetensors vs PyTorch

Both save formats support safetensors or PyTorch serialization:

**Safetensors** (`--safetensors`):
- Safer loading (no arbitrary code execution)
- Faster loading
- Better memory mapping
- Standard in modern HuggingFace models

**PyTorch** (default):
- Traditional format
- Broader compatibility
- Can save complex Python objects

## Working with Forgather Models

The tool works identically with Forgather models:

```bash
# Update a Forgather model
./update_vocab.py --add-tokens tokens.yaml \
    ~/models/fg_model \
    ~/models/fg_model_updated
```

Forgather models are detected automatically and handled correctly.

## Workflow Integration

### Fine-tuning Workflow

1. Start with base model
2. Add custom tokens for your domain
3. Fine-tune on your data
4. Deploy

```bash
# Step 1: Add domain tokens
./update_vocab.py --add-tokens domain_tokens.yaml \
    ~/models/llama-7b \
    ~/models/llama-7b-domain

# Step 2: Fine-tune (using your training script)
forgather -t config.yaml train

# Step 3: Deploy
# Model is ready with extended vocabulary
```

### Instruction Tuning Workflow

1. Add chat template tokens
2. Train on instruction data
3. Deploy as chat model

```bash
# Add instruction tokens
cat > inst_tokens.yaml << EOF
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"
  - "<|system|>"
  - "<|user|>"
  - "<|assistant|>"
EOF

./update_vocab.py --add-tokens inst_tokens.yaml \
    ~/models/base_model \
    ~/models/base_model_inst

# Now fine-tune on instruction data
```

## Troubleshooting

### Issue: "Output path already exists"

**Solution:** The tool won't overwrite existing models. Either:
- Choose a different output path
- Remove the existing output directory
- Use `--dry-run` to test first

### Issue: "Token already exists" warnings

**Solution:** This is informational. The tool skips tokens that already exist in the vocabulary. If you want to replace a token, use the token reassignment feature (specify a different token value for the same named token).

### Issue: Memory errors with large models

**Solution:**
- Use `--device cpu` to avoid GPU memory limits during update
- Process on a machine with sufficient RAM
- Consider using `--dtype bfloat16` to reduce memory usage

### Issue: Config not saved with sharded format

**Solution:** When using `--save-format sharded`, the config IS saved automatically (the code calls `config.save_pretrained()`). If you don't see it, check:
- Output directory permissions
- Disk space
- Log messages for errors

## Technical Details

### How It Works

1. **Load Model**: Uses `AutoModelForCausalLM.from_pretrained()` - works for both HF and Forgather
2. **Load Tokenizer**: Uses `AutoTokenizer.from_pretrained()`
3. **Add Tokens**: Calls `add_tokens_to_tokenizer()` from the refactored resize_embeddings module
4. **Resize Embeddings**: Calls `resize_word_embeddings()` to adjust model embedding layers
5. **Update Config**: Calls `update_config_from_tokenizer()` to sync vocab size and special token IDs
6. **Save**: Uses either HF `save_pretrained()` or Forgather `save_checkpoint()`

### Token Embedding Initialization

New tokens are initialized using the specified strategy:

- **Zero initialization**: Sets embedding weights to zero (good for PAD tokens)
- **Mean initialization**: Uses mean of existing embeddings (good for semantic tokens)

The initialization happens during `resize_word_embeddings()` and is applied to both input and output embeddings (unless they're tied).

### Compatibility

Works with any model that supports:
- `AutoModelForCausalLM` interface
- `AutoTokenizer` interface
- `save_pretrained()` method

This includes:
- All HuggingFace models (Llama, Mistral, Qwen, GPT, etc.)
- All Forgather models (converted from HF)
- Custom models using HF architecture

## See Also

- [Model Converter Documentation](../convert_model/README.md) - Converting between HF and Forgather formats
- [Forgather Training Documentation](../../docs/trainers/) - Training models with custom vocabularies
- [Configuration System](../../docs/configuration/) - Forgather configuration

## Contributing

To report issues or suggest improvements:
- File an issue at https://github.com/anthropics/claude-code/issues
- Include example YAML configurations and error messages
- Specify model type and size for reproducibility
