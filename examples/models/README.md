# Models

A collection of model definitions.

## Forgather Model Commands

### Construct Model

When a model is constructed from a Forgather configuration, all of the assets required to load the model with
`AutoModelForCausalLM.from_config()` are saved into the output directory. This includes:

- HF Model config (`config.json`)
- HF Tokenizer config (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)
- Generation config
- Chat template, if defined
- All of the source code required to load the model without any Forgather dependencies

Build the model with the `forgather model construct` command:

```bash
# Build model from sources (if it does not already exist) and construct an instance.
forgather -t CONFIG_NAME model construct
```

By default, the model is constructed on the `meta` device, meaning no actual tensors are allocated.
This is useful for validating the model definition without consuming memory. To materialize the model
on a real device, pass `--device`:

```bash
forgather -t CONFIG_NAME model --device cpu construct
forgather -t CONFIG_NAME model --device cuda construct
```

You can also control the default dtype used during construction with `--dtype`:

```bash
forgather -t CONFIG_NAME model --device cpu --dtype bfloat16 construct
```

To skip weight initialization (useful when you plan to load weights from a checkpoint), pass
`--no-init-weights`:

```bash
forgather -t CONFIG_NAME model --device cpu --no-init-weights construct
```

### Rebuild a Model

**IMPORTANT**: If the model definition has already been built, it will not be regenerated from sources.
Pass `--refresh-model` (or `-r`) to force regeneration. This is necessary when any of the model source
code files have changed.

```bash
# Rebuild the model definition from scratch.
forgather -t CONFIG_NAME model -r construct
```

### Saving the Initialized Weights

By default, only the model source code is saved to the output directory. To also save initialized weights,
specify a real device and `--save-checkpoint`:

```bash
# Save weights with initialized model.
# A real device (not meta) is required when saving.
forgather -t CONFIG_NAME model --device cpu --save-checkpoint construct

# Save in safetensors format
forgather -t CONFIG_NAME model --device cpu --save-checkpoint --safetensors construct
```

### Loading an Existing Checkpoint

Use `--load-from-checkpoint` to load weights from an existing checkpoint into the constructed model.
Loading is non-strict: any parameters present in the checkpoint are loaded, and a warning is printed
for any model parameters or buffers that were not found in the checkpoint.

```bash
# Load weights from a checkpoint
forgather -t CONFIG_NAME model --device cpu --load-from-checkpoint CHECKPOINT_PATH construct
```

### Updating an Existing Model

If the model source code has changed and you have a checkpoint you would like to carry forward, you
can rebuild the model definition, load the old weights, and save them with the new model in one step:

```bash
# Rebuild model from sources, load an existing (compatible) checkpoint,
# and save the checkpoint with the new model.
forgather -t CONFIG_NAME model -r --device cpu --save-checkpoint --safetensors \
    --load-from-checkpoint CHECKPOINT_PATH construct
```

### Output Options

The `construct` command prints model configuration, tokenizer, and parameter count information to
stdout. You can redirect this to a file or open it in your editor:

```bash
# Write output to a file
forgather -t CONFIG_NAME model construct -o model_info.txt

# Write to a temporary file and open in editor
forgather -t CONFIG_NAME model construct -e
```

## Test a Model

The `test` subcommand runs a forward and backward pass on the model. This is useful for verifying
that changes to model source code produce a working model.

```bash
# Basic model kick-test -- kick it to see if it falls over.
forgather -t CONFIG_NAME model --device cuda test
```

By default, the model is fed random inputs. You can control the batch size, sequence length, and
number of training steps:

```bash
forgather -t CONFIG_NAME model --device cuda test \
    --batch-size 2 --sequence-length 512 --steps 5
```

To test with real data, point to a dataset project:

```bash
forgather -t CONFIG_NAME model --device cuda \
    --load-from-checkpoint CHECKPOINT_PATH \
    test --batch-size 4 --sequence-length 4096 --steps 10 \
    --dataset-project DATASET_PROJECT --dataset-config DATASET_CONFIG
```

The test runs a mini-training loop using SGD. Additional options for the test:

- `--packed` -- Enable packed sequences in the data collator
- `--lr LR` -- Set the learning rate (default: 0.01)
- `--attn-implementation {eager,sdpa,flash_attention_2,flex_attention}` -- Select attention implementation
- `--gradient-checkpointing` -- Enable gradient checkpointing to reduce memory usage
- `--fuse-optim-with-backward` -- Fuse the optimizer step with the backward pass to save memory

## Using the Model

Once a model has been built, you can use the standard HuggingFace APIs to work with it.

```python
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

model_path = "output_models/llama_4m"

# To HF, any code not defined in "transformers" is considered "remote code",
# so trust_remote_code=True is required for Forgather models.
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Construct a newly initialized model instance
model = AutoModelForCausalLM.from_config(config)

# Save initialized model weights with HF API
model.save_pretrained(model_path)

# Note: if the model has tied weights, disable safetensors serialization
model.save_pretrained(model_path, safe_serialization=False)

# Skip weight init and use bfloat16
from forgather.ml.no_init_weights import no_init_weights
from forgather.ml.utils import default_dtype

with no_init_weights(), default_dtype(torch.bfloat16):
    model = AutoModelForCausalLM.from_config(config)

# Move the model to first CUDA device
device = "cuda:0"
model.to(device=device)

# Load the model with weights using HF API (requires saved model weights)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
```

### Loading and Saving with the Forgather API

Forgather provides its own checkpoint utilities, compatible with the HF sharded checkpoint format:

```python
from forgather.ml.sharded_checkpoint import load_checkpoint, save_checkpoint

# Load weights from checkpoint
load_checkpoint(model_path, model, device=device)

# Save checkpoint
save_checkpoint(model_path, model)
```

## Customize Model

There are many additional arguments for customizing the model or configuring it for testing.
Some of these arguments (such as `--hidden-size`, `--num-attention-heads`, etc.) are **dynamic
arguments** defined by the model's project configuration, so the available options may vary
between model definitions.

```bash
forgather model --help
usage: forgather model [-h] [--device DEVICE] [--dtype DTYPE] [--no-init-weights]
                       [--load-from-checkpoint LOAD_FROM_CHECKPOINT]
                       [--gradient-checkpointing] [--fuse-optim-with-backward]
                       [--refresh-model] [--save-checkpoint] [--safetensors]
                       [-o OUTPUT_FILE] [-e]
                       [--attn-implementation {eager,sdpa,flash_attention_2,flex_attention}]
                       [--output-dir OUTPUT_DIR] [--model-name MODEL_NAME]
                       [--tie-word-embeddings]
                       [--attention-dropout ATTENTION_DROPOUT]
                       [--hidden-size HIDDEN_SIZE]
                       [--num-attention-heads NUM_ATTENTION_HEADS]
                       [--num-key-value-heads NUM_KEY_VALUE_HEADS]
                       [--num-hidden-layers NUM_HIDDEN_LAYERS]
                       [--intermediate-size INTERMEDIATE_SIZE]
                       [--rope-parameters ROPE_PARAMETERS]
                       [--rms-norm-eps RMS_NORM_EPS]
                       [--sliding-window SLIDING_WINDOW]
                       {construct,test} ...

Test a model definition

positional arguments:
  {construct,test}      Model subcommands
    construct           Construct a model
    test                Test model forward and backward

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to construct model on (default: meta)
  --dtype DTYPE         Construct with default torch dtype (e.g., bfloat16, float32)
  --no-init-weights     Construct with no_init_weights() context manager
  --load-from-checkpoint LOAD_FROM_CHECKPOINT
                        Load model weights from checkpoint (path)
  --gradient-checkpointing
                        Enable gradient checkpointing
  --fuse-optim-with-backward
                        Combine backward with optimizer step to save memory
  --refresh-model, -r   Force regeneration of fresh model from sources by deleting output_dir
  --save-checkpoint     Save model checkpoint
  --safetensors         Save using safetensors
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Write output to file
  -e, --edit            Write output to temporary file and open in editor
  --attn-implementation {eager,sdpa,flash_attention_2,flex_attention}
                        Attention implementation

  Dynamic arguments (may vary by model definition):
  --output-dir OUTPUT_DIR
                        Model output directory
  --model-name MODEL_NAME
                        Model name
  --tie-word-embeddings
                        Tie input and output word embeddings
  --attention-dropout ATTENTION_DROPOUT
                        Attention dropout probability
  --hidden-size HIDDEN_SIZE
                        Model hidden dimension size (d_model)
  --num-attention-heads NUM_ATTENTION_HEADS
                        Number of query heads (and KV heads, if otherwise unspecified)
  --num-key-value-heads NUM_KEY_VALUE_HEADS
                        Number of Key/Value heads
  --num-hidden-layers NUM_HIDDEN_LAYERS
                        Number of hidden layers
  --intermediate-size INTERMEDIATE_SIZE
                        Feedforward dimension
  --rope-parameters ROPE_PARAMETERS
                        RoPE parameters dict (rope_theta, rope_type, factor, etc.)
  --rms-norm-eps RMS_NORM_EPS
                        RMS Norm eps
  --sliding-window SLIDING_WINDOW
                        Sliding window size. 'null' for disabled

# Additional arguments for "test"
forgather model test --help
usage: forgather model test [-h] [--batch-size BATCH_SIZE]
                            [--sequence-length SEQUENCE_LENGTH]
                            [--steps STEPS]
                            [--dataset-project DATASET_PROJECT]
                            [--dataset-config DATASET_CONFIG]
                            [--packed] [--lr LR]

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size (default: 2)
  --sequence-length SEQUENCE_LENGTH
                        Sequence length (default: 512)
  --steps STEPS         Number of train steps (default: 1)
  --dataset-project DATASET_PROJECT
                        Path to dataset project
  --dataset-config DATASET_CONFIG
                        Dataset config name
  --packed              Enable packed sequences in data collator
  --lr LR               Learning rate (default: 0.01)
```

## Model Definitions

### Single Head

[A simple custom model example](./single_head/README.md)

### Causal LM

[A plain vanilla decoder only transformer](./causal_lm/README.md)

### Llama

[The ubiquitous Llama, implemented using our model parts collection](./llama/README.md)

### Deepone

[A mysterious model of unknown origin](./deepone/README.md) -- A custom Deepnet ALiBi transformer