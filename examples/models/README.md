# Models

A collection of model definitions.

## Forgather Model Commands

### Construct Model

When a model is constructed from a Forgather configuration, all of the assets required to construct the model with
"AutoModelForCausalLM.from_config(PATH)" are saved into the output directory. This includes the following:

- HF Model config: "config.json"
- HF Tokenizer config: "tokenizer.json," "tokenizer_config.json," "special_tokens_map.json"
- Generation Config
- Chat Template, if defined
- All of the source code required to load the model without needing any Forgather dependencies.

You can build the model via the "forgather model construct" command

```bash
forgather -t CONFIG_NAME model construct
```

Note that creating the model definition is skipped if a definition already exists in the target directory. To ensure that the
model definition is rebuilt from scratch, delete the output-directory first.

```bash
rm -rf output_models
```

### Using the Model

Once a model has been built, you can use the usual HF APIs to construct it.

```python
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
)

model_path = "output_models/llama_4m"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# To HF, any code which is not defined in "transformers" is considered "remote-code"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Construct a newly initialized model instance
model = AutoModelForCausalLM.from_config(config)

# Save initialized model weights with HF API
model.save_pretrained(model_path)

# Note: If the model has tied weights, use
model.save_pretrained(model_path, safe_serialization=False)

# Skip weight init and use bfloat16
from forgather.ml.no_init_weights import no_init_weights
from forgather.ml.utils import default_dtype

with no_init_weights(), default_dtype(torch.bfloat16):
  model = AutoModelForCausalLM.from_config(config)

# Move the model to first CUDA device
device = "cuda:0"
model.to(device=device)

# Load the model with weight using the HF API -- requires saved model weights
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

## Loading and Saving weights with Forgather API -- compatible with HF format
from forgather.ml.sharded_checkpoint import load_checkpoint, save_checkpoint

# Load weights from checkpoint
load_checkpoint(model_path, model, device=device)

# Save checkpoint using Forgather API
save_checkpoint(model_path, model)
```

### Test Model

You can test the model with a forward and backward pass with the "model test" command.

```bash
# Perform quick forward and backward test on model
forgather -t CONFIG_NAME model --device cuda:0 test

# Test with model forward and backward when memory constrained
forgather -t CONFIG_NAME model --gradient-checkpointing --fuse-optim-with-backward --dtype bfloat16 --device cuda:0 test

# Test model with dataset
forgather -t CONFIG_NAME model --device cuda:0 test --dataset-project DATASET_PROJECT --dataset-config DATASET_CONFIG [--packed]
```

### Customize Model

There any many additional arguments which can be used to customize the model or configure the model for testing.

Example:

```bash
forgather model --help 
usage: forgather model [-h] [--device DEVICE] [--dtype DTYPE] [--no-init-weights] [--load-from-checkpoint LOAD_FROM_CHECKPOINT] [--gradient-checkpointing] [--fuse-optim-with-backward] [-o OUTPUT_FILE] [-e] [--attn-implementation {eager,sdpa,flash_attention_2,flex_attention}] [--output-dir OUTPUT_DIR] [--model-name MODEL_NAME] [--tie-word-embeddings]
                       [--attention-dropout ATTENTION_DROPOUT] [--hidden-size HIDDEN_SIZE] [--num-attention-heads" NUM_ATTENTION_HEADS"] [--num-kv-heads NUM_KV_HEADS] [--d-head D_HEAD] [--num-hidden-layers NUM_HIDDEN_LAYERS] [--dim-feedforward DIM_FEEDFORWARD] [--rope-theta ROPE_THETA] [--rms-norm-eps RMS_NORM_EPS] [--rope-scaling ROPE_SCALING]
                       [--sliding-window SLIDING_WINDOW]
                       {construct,test} ...

Test a model definition

positional arguments:
  {construct,test}      Model subcommands
    construct           Construct a model
    test                Test model forward and backward

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to construct model on
  --dtype DTYPE         Construct with default torch dtype
  --no-init-weights     Construct with no_init_weights() context manager
  --load-from-checkpoint LOAD_FROM_CHECKPOINT
                        Load model weights from checkpoint (path)
  --gradient-checkpointing
                        Enable gradient checkpointing
  --fuse-optim-with-backward
                        Combine backward with optimizer step to save memory
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Write output to file
  -e, --edit            Write output to temporary file and open in editor
  --attn-implementation {eager,sdpa,flash_attention_2,flex_attention}
                        Attention implementation
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
  --num-attention-heads" NUM_ATTENTION_HEADS"
                        Number of query heads (and KV heads, if otherwise unspecified)
  --num-kv-heads NUM_KV_HEADS
                        Number of Key/Value heads
  --d-head D_HEAD       Head dimension
  --num-hidden-layers NUM_HIDDEN_LAYERS
                        Number of hidden layers
  --dim-feedforward DIM_FEEDFORWARD
                        Feedforward dimension
  --rope-theta ROPE_THETA
                        RoPE Theta
  --rms-norm-eps RMS_NORM_EPS
                        RMS Norm eps
  --rope-scaling ROPE_SCALING
                        RoPE scaling type
  --sliding-window SLIDING_WINDOW
                        Sliding window size. 'null' for disabled

forgather model test --help
usage: forgather model test [-h] [--batch-size BATCH_SIZE] [--sequence-length SEQUENCE_LENGTH] [--steps STEPS] [--dataset-project DATASET_PROJECT] [--dataset-config DATASET_CONFIG] [--packed] [--lr LR]

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size
  --sequence-length SEQUENCE_LENGTH
                        Sequence length
  --steps STEPS         Number of train steps
  --dataset-project DATASET_PROJECT
                        Path to dataset project
  --dataset-config DATASET_CONFIG
                        Dataset config name
  --packed              Enable packed sequences in data collator
  --lr LR               Learning rate
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