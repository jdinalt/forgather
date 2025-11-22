# Forgather Model Conversion Utility

The Forgather model conversion utility provides bidirectional conversion between HuggingFace Transformers models and Forgather's dynamic model format. It supports automatic model type detection, dtype auto-detection, vocabulary extension, and plugin-based architecture for different model families.

## Quick Start

### Basic Usage

```bash
# Convert HuggingFace model to Forgather (auto-detected)
forgather convert /path/to/hf_model /path/to/output_fg_model

# Convert Forgather model back to HuggingFace (auto-detected)
forgather convert /path/to/fg_model /path/to/output_hf_model

# Force Forgather→HF conversion
forgather convert --reverse /path/to/fg_model /path/to/output_hf_model
```

### Installation

The conversion utility is included with Forgather and requires:
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Forgather framework

## Features

### Automatic Model Type Detection

The converter automatically detects:
- **Conversion direction**: HF→FG or FG→HF based on config metadata
- **Model architecture**: Llama, Mistral, Qwen3, and other supported types
- **Data type (dtype)**: Inherits from source model or defaults to bfloat16

Models converted from HF→FG store `hf_model_type` metadata, enabling automatic reverse conversion without specifying model type.

### Supported Model Families

Built-in converters are provided for:
- **Llama**: Llama 2, Llama 3.x (including RoPE scaling variants)
- **Mistral**: Mistral 7B and variants (with sliding window attention)
- **Qwen3**: Qwen3 8B and variants

Additional converters can be added via the plugin system (see [Custom Converters](#custom-converters)).

### Plugin Architecture

The converter uses a plugin-based architecture:
- **Auto-discovery**: Automatically finds converters in `examples/models/*/src/converter.py`
- **Custom paths**: Add external converter directories via `--converter-path`
- **Registration**: Converters register themselves using `@register_converter` decorator

## Command-Line Reference

### Basic Arguments

```bash
forgather convert [OPTIONS] SOURCE_PATH DESTINATION_PATH
```

**Positional Arguments:**
- `SOURCE_PATH`: Path to source model (HF or Forgather - direction auto-detected)
- `DESTINATION_PATH`: Output directory for converted model

### Conversion Direction

```bash
--reverse
```
Force Forgather→HF conversion. By default, direction is auto-detected from the source model's configuration.

### Model Type

```bash
--model-type {llama,mistral,qwen3}
```
Override auto-detected model type for FG→HF conversion. Only needed if auto-detection fails or you want to force a specific converter. Default: `llama` if auto-detection fails.

### Data Type (dtype)

```bash
--dtype {bfloat16,float32,float16}
```
Specify output model dtype. If not specified, uses the following priority:
1. Source model's `dtype` field (if present)
2. Source model's `torch_dtype` field (deprecated, for backwards compatibility)
3. Default to `bfloat16`

**Examples:**
```bash
# Auto-detect dtype from source model
forgather convert ~/models/hf_llama ~/models/fg_llama

# Force float32 dtype
forgather convert --dtype float32 ~/models/hf_llama ~/models/fg_llama
```

### Model Length

```bash
--max-length LENGTH
```
Override maximum model sequence length. If not specified, uses source model's `max_position_embeddings` or equivalent.

**Example:**
```bash
# Extend context window to 32K tokens
forgather convert --max-length 32768 ~/models/hf_llama ~/models/fg_llama
```

### Checkpoint Selection (FG→HF only)

```bash
-c PATH, --checkpoint-path PATH
```
Specify a particular Forgather checkpoint to load. If not specified, uses the latest checkpoint in the source directory.

**Example:**
```bash
# Convert specific checkpoint
forgather convert --reverse -c ~/models/fg_llama/checkpoint-1000 ~/models/fg_llama ~/models/hf_llama
```

### Vocabulary Extension

```bash
--add-tokens YAML_FILE
```
Add custom tokens to the model vocabulary, including named special tokens (BOS, EOS, PAD, UNK) with customizable initialization strategies.

**YAML Format (New - with named tokens and init strategies):**
```yaml
# Named special tokens (optional)
bos_token: "<|begin_of_text|>"  # String format uses default init (mean)
eos_token:                       # Dict format allows custom init
  token: "<|end_of_text|>"
  init: "mean"
pad_token:
  token: "<|pad|>"
  init: "zero"
  if_missing: true               # Only add if not already set
unk_token: "<|unknown|>"

# Additional special tokens (optional)
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"

# Regular tokens (optional)
regular_tokens:
  - "custom_token_1"
  - "custom_token_2"
```

**Old Format (still supported):**
```yaml
special_tokens:
  - "<|im_start|>"
regular_tokens:
  - "custom_token_1"
```

**Named Special Tokens:**
- `bos_token`: Beginning-of-sequence token (for document packing)
- `eos_token`: End-of-sequence token
- `pad_token`: Padding token
- `unk_token`: Unknown token

These tokens can be set/replaced if missing or if you want to change them.

**Initialization Strategies:**
- `"zero"`: Initialize embeddings to zero (recommended for PAD)
- `"mean"`: Initialize to mean of existing embeddings (recommended for BOS/EOS/UNK)
- Default strategies:
  - BOS/EOS/UNK tokens: `"mean"`
  - PAD token: `"zero"`

**if_missing Flag:**
- When set to `true`, only adds/sets the token if it doesn't already exist
- Useful for ensuring a token exists without forcing replacement
- Example: `if_missing: true` for PAD token adds it only if missing

**Token Reassignment:**
- When replacing an existing token with a different value (e.g., changing `[PAD]` to `<|PAD|>`):
  - If the new token already exists in vocabulary, just reassigns the special token pointer
  - Otherwise, adds the new token and updates the special token pointer
- Properly handles the transition without creating duplicate tokens

**Default Behavior:**
- By default, the converter adds a missing PAD token (`[PAD]`) with zero initialization
- Use `--skip-default-tokens` to disable this automatic behavior
- Custom `--add-tokens` configuration overrides the default

**Behavior:**
- Named tokens are added/replaced with specified initialization strategy
- Model embeddings are resized using HuggingFace's `resize_token_embeddings()`
- Config is updated with new vocabulary size

**Examples:**

```bash
# Add BOS token to Qwen3 for document packing
cat > qwen3_bos.yaml << EOF
bos_token: "<|begin_of_text|>"
EOF
forgather convert --add-tokens qwen3_bos.yaml ~/models/qwen3 ~/models/qwen3_with_bos

# Add custom PAD token with zero initialization
cat > custom_pad.yaml << EOF
pad_token:
  token: "<|padding|>"
  init: "zero"
EOF
forgather convert --add-tokens custom_pad.yaml ~/models/hf_model ~/models/fg_model

# Hybrid: named tokens + additional special tokens
cat > tokens.yaml << EOF
bos_token: "<|begin|>"
eos_token: "<|end|>"
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"
EOF
forgather convert --add-tokens tokens.yaml ~/models/hf_qwen ~/models/fg_qwen

# Replace existing PAD token with a different one
cat > replace_pad.yaml << EOF
pad_token: "<|pad|>"  # Will replace [PAD] if it exists
EOF
forgather convert --add-tokens replace_pad.yaml ~/models/hf_model ~/models/fg_model

# Add PAD token only if missing (using if_missing flag)
cat > pad_if_missing.yaml << EOF
pad_token:
  token: "<|pad|>"
  if_missing: true  # Only add if not already set
EOF
forgather convert --add-tokens pad_if_missing.yaml ~/models/hf_model ~/models/fg_model

# Skip default PAD token addition for complete vocabulary control
forgather convert --skip-default-tokens ~/models/hf_model ~/models/fg_model
```

### Testing and Debugging

```bash
--device {cpu,cuda,cuda:0,cuda:1,...}
```
Device for model testing and validation. Default: `cpu`

```bash
-g, --generation-test
```
Run generation test with a sample prompt to verify model functionality.

```bash
--prompt "YOUR PROMPT"
```
Custom prompt for generation testing. Default prompt is provided.

```bash
--debug-params
```
Print parameter names during conversion for debugging weight mapping issues.

### Chat Template

```bash
-t PATH, --chat-template-path PATH
```
Assign a custom chat template to the output tokenizer. The file should contain a Jinja2 template for formatting chat conversations.

**Example:**
```bash
forgather convert -t chat_template.j2 ~/models/hf_llama ~/models/fg_llama
```

### Custom Converters

```bash
--converter-path PATH
```
Add additional directory to search for model converters. Can be specified multiple times.

**Example:**
```bash
forgather convert --converter-path ~/my_converters --converter-path ~/shared_converters ~/models/hf_custom ~/models/fg_custom
```

## Usage Examples

### Example 1: Basic HF→FG Conversion

```bash
# Convert Llama model from HuggingFace to Forgather
forgather convert ~/models/meta-llama/Llama-2-7b-hf ~/models/fg_llama_7b

# Output:
# Auto-detected HuggingFace model (type: llama)
# Auto-detected dtype from source model: bfloat16
# Converting HuggingFace model to Forgather format
# ...
# Conversion complete: ~/models/fg_llama_7b
```

### Example 2: FG→HF Conversion

```bash
# Convert Forgather model back to HuggingFace
forgather convert ~/models/fg_llama_7b ~/models/my_hf_llama

# Output:
# Auto-detected Forgather model (original type: llama)
# Auto-detected dtype from source model: bfloat16
# Converting Forgather model to HuggingFace Llama
# ...
# Conversion complete: ~/models/my_hf_llama
```

### Example 3: Extended Context Window

```bash
# Convert with extended context length
forgather convert --max-length 16384 ~/models/hf_mistral ~/models/fg_mistral_16k
```

### Example 4: Adding Custom Tokens

```bash
# Create token configuration
cat > custom_tokens.yaml << 'EOF'
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"
  - "<|tool_call|>"

regular_tokens:
  - "custom_token_1"
  - "custom_token_2"
EOF

# Convert with token addition
forgather convert --add-tokens custom_tokens.yaml ~/models/hf_qwen ~/models/fg_qwen_extended
```

### Example 5: Specific Checkpoint and Dtype

```bash
# Convert specific checkpoint to float32
forgather convert --reverse --dtype float32 -c ~/models/fg_llama/checkpoint-5000 ~/models/fg_llama ~/models/hf_llama_float32
```

### Example 6: Full Training Workflow

```bash
# 1. Convert HF model to Forgather for training
forgather convert ~/models/hf_llama ~/models/fg_llama

# 2. Train the model (using Forgather training system)
cd ~/models/fg_llama
forgather -t train_config.yaml train

# 3. Convert trained model back to HuggingFace for deployment
forgather convert --reverse ~/models/fg_llama/output_models/my_model ~/models/hf_trained_llama
```

## Auto-Detection Details

### Conversion Direction Detection

The converter determines direction by checking the source model's configuration:

**HuggingFace Model:**
- Has `model_type` field (e.g., "llama", "mistral")
- Does NOT have `hf_model_type` field

**Forgather Model:**
- Has `hf_model_type` field (stored during HF→FG conversion)
- This metadata enables automatic reverse conversion

### Model Type Detection

**For HF→FG:**
- Reads `model_type` from HuggingFace config
- Validates against registered converters
- Fails if model type is not supported

**For FG→HF:**
- Reads `hf_model_type` from Forgather config
- Falls back to `--model-type` argument if not found
- Default to "llama" if both are unavailable

### DType Detection

The converter uses the following priority for dtype:
1. Explicit `--dtype` argument (highest priority)
2. Source config's `dtype` field
3. Source config's `torch_dtype` field (deprecated, for backwards compatibility)
4. Default to `bfloat16` (lowest priority)

The detected dtype is saved to the output model's config, providing a hint for future loading.

## Custom Converters

### Creating a Custom Converter

1. **Create converter module** (e.g., `my_model_converter.py`):

```python
from forgather.ml.model_conversion import HFConverter, register_converter

@register_converter("my_model")
class MyModelConverter(HFConverter):
    model_type = "my_model"

    def get_hf_model_class(self):
        from transformers import MyModelForCausalLM
        return MyModelForCausalLM

    def get_weight_mapping(self, direction):
        # Return weight mapping dictionary
        pass

    def create_project_config(self, src_config, max_length=None):
        # Return config dict for Forgather Project()
        pass

    def create_hf_config(self, src_config, max_length=None):
        # Return HuggingFace config object
        pass
```

2. **Use the custom converter:**

```bash
# Point to directory containing your converter
forgather convert --converter-path /path/to/converter/dir ~/models/hf_my_model ~/models/fg_my_model
```

### Builtin Converter Locations

The converter auto-discovers builtin converters from:
```
examples/models/*/src/converter.py
```

For example:
- `examples/models/llama/src/converter.py` → LlamaConverter
- `examples/models/mistral/src/converter.py` → MistralConverter
- `examples/models/qwen3/src/converter.py` → Qwen3Converter

## Technical Details

### Weight Mapping

Each converter defines bidirectional weight mappings between HuggingFace and Forgather naming conventions:

**Standard Mappings** (most models):
- Input embeddings: `embed_tokens` ↔ `embedding.word_embedding.weight`
- Output head: `lm_head` ↔ `lm_head.weight`
- Layer norm: `norm` ↔ `layer_norm`
- Attention: `self_attn.{q,k,v,o}_proj` ↔ `attention.{query,key,value,output}_linear`
- MLP: `mlp.{gate,up,down}_proj` ↔ `feedforward.{gate,up,down}_proj`

**Model-Specific Mappings:**
- RoPE scaling parameters (Llama 3.x)
- Sliding window attention (Mistral)
- Query/Key normalization (Qwen3)

### Tied Embeddings

The converter properly handles tied embeddings (when input and output embeddings share weights):

- **HF→FG**: Detects `tie_word_embeddings` config, calls `model.tie_weights()` after loading
- **FG→HF**: Preserves tied embedding configuration in output model

### Vocabulary Size Handling

When source model vocab size differs from tokenizer size:
- Adjusts Forgather model to match source HF model size
- Logs the adjustment for transparency
- Handles vocab extension via `--add-tokens` correctly

### Logit Comparison

After conversion, the converter validates correctness by:
1. Running same prompt through source and destination models
2. Comparing output logits with tolerance check
3. Handling vocabulary size mismatches gracefully
4. Reporting discrepancies for debugging

## Troubleshooting

### Common Issues

**"No converter registered for model type 'X'"**
- The model type is not supported
- Add a custom converter using `--converter-path`
- Or contribute a new builtin converter

**"Could not auto-detect model type"**
- Model config is missing `model_type` or `hf_model_type`
- Use `--model-type` to specify explicitly (for FG→HF)
- Check that the model directory is valid

**"Vocab size mismatch" warning**
- Normal when adding tokens via `--add-tokens`
- Logit comparison only checks overlapping vocabulary
- Not an error unless logits are dissimilar

**"WARNING: Model logits are dissimilar"**
- Indicates potential weight mapping issue
- Check parameter names with `--debug-params`
- May be caused by dtype conversion (e.g., float32→bfloat16)
- Small differences are normal; large differences need investigation

**"Checkpoint not found"**
- For FG→HF, ensure source directory has trained checkpoints
- Use `--checkpoint-path` to specify exact checkpoint
- Check that `checkpoint-*/` directories exist

### Debug Output

Enable detailed logging:
```bash
# The conversion script uses Python logging at DEBUG level
# Check console output for detailed information about:
# - Converter discovery
# - Model type detection
# - Weight mapping
# - Parameter loading
```

Use `--debug-params` to see parameter name mappings:
```bash
forgather convert --debug-params ~/models/hf_model ~/models/fg_model
```

## Performance Notes

### Memory Usage

- Conversion loads both source and destination models in memory
- Large models (70B+) may require significant RAM
- Use `--device cpu` to avoid GPU memory issues during conversion
- Models are saved in sharded format for efficient loading

### Conversion Time

Typical conversion times (on CPU):
- 7B models: 2-5 minutes
- 13B models: 5-10 minutes
- 70B models: 20-30 minutes

Time depends on:
- Model size
- Disk I/O speed
- Whether validation/testing is enabled

### Disk Space

Output model requires approximately the same space as input:
- Model weights (sharded PyTorch files)
- Configuration files (config.json, etc.)
- Tokenizer files
- Generated Python model code (for Forgather models)

## See Also

- [Forgather Documentation](../../docs/)
- [Training Models](../../docs/trainers/)
- [Model Architecture](../../modelsrc/transformer/)
- [Configuration System](../../docs/configuration/)

## Contributing

To add support for a new model family:

1. Create converter in `examples/models/NEW_MODEL/src/converter.py`
2. Inherit from `HFConverter` base class
3. Implement required methods (weight mapping, config creation)
4. Register with `@register_converter("model_type")`
5. Test bidirectional conversion
6. Submit pull request with tests

See existing converters (Llama, Mistral, Qwen3) for examples.
