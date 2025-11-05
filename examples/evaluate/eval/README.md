# Evaluate

Test model inference with "test" datasets.

## Usage

```bash
forgather train --help
usage: forgather train [-h] [-d DEVICES] [--dry-run] [--max-steps MAX_STEPS] [--save-strategy {no,steps,epoch}] [--attn-implementation {eager,sdpa,flash_attention_2,flex_attention}] [--dataset-config DATASET_CONFIG] [--dataset-proj DATASET_PROJ] [--model-id-or-path MODEL_ID_OR_PATH] [--output-dir OUTPUT_DIR] [--from-checkpoint FROM_CHECKPOINT] [--chat-template CHAT_TEMPLATE]
                       [--max-length MAX_LENGTH] [--dtype DTYPE] [--float32-precision" {highest,high,medium}] [--sdpa-backend {math,flash,efficient,cudnn}] [--batch-size BATCH_SIZE] [--dataset-target DATASET_TARGET]
                       ...

Run configuration with train script

positional arguments:
  remainder             All arguments after -- will be forwarded as torchrun arguments.

options:
  -h, --help            show this help message and exit
  -d DEVICES, --devices DEVICES
                        CUDA Visible Devices e.g. "0,1"
  --dry-run             Just show the generated commandline, without actually executing it.
  --max-steps MAX_STEPS
                        Maximum eval steps
  --save-strategy {no,steps,epoch}, -S {no,steps,epoch}
                        When to save checkpoints
  --attn-implementation {eager,sdpa,flash_attention_2,flex_attention}
                        Attention implementation
  --dataset-config DATASET_CONFIG
                        The name of the dataset configuration to use
  --dataset-proj DATASET_PROJ
                        Path to dataset project to use
  --model-id-or-path MODEL_ID_OR_PATH, -M MODEL_ID_OR_PATH
                        HF model ID or local path to model
  --output-dir OUTPUT_DIR
                        Training output director. Defaults to model_id_or_path
  --from-checkpoint FROM_CHECKPOINT
                        Explicit checkpoint path to load
  --chat-template CHAT_TEMPLATE, -C CHAT_TEMPLATE
                        Path to the chat template to use
  --max-length MAX_LENGTH
                        Maximum sequence length
  --dtype DTYPE         Torch dtype
  --float32-precision" {highest,high,medium}
                        Float32 precision
  --sdpa-backend {math,flash,efficient,cudnn}
                        Specify SDPA backend to use
  --batch-size BATCH_SIZE
                        Eval batch size
  --dataset-target DATASET_TARGET
                        The dataset target to test with
```

## Configs

- [**project.yaml**](templates/project.yaml) -- Common configuration
- [**default**](templates/configs/default.yaml) -- Test on single GPU
- [**pp_2gpu.yaml**](templates/configs/pp_2gpu.yaml) -- Pipeline Parallel, 2 GPUs
- [**pp_4gpu.yaml**](templates/configs/pp_4gpu.yaml) -- Pipeline Parallel, 4 GPUs

## Examples

```bash
MODEL_PATH="/path/to/model"

# Test with Samantha-Packed dataset
forgather -M "${MODEL_PATH}" --dataset-proj ../../datasets/QuixiAI/ --dataset-config samantha-packed.yaml \
--batch-size 8 --dtype bfloat16 --attn-implementation flex_attention --max-length 4096

```