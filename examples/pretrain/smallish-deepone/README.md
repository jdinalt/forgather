# Small LLM Pretraining

Train a model from scratch on [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)

> This dataset is a curated collection of high-quality educational and synthetic data designed for training small language models.

The train dataset is a combination of the "cosmopedia-v2" and "fineweb-edu-dedup" subsets of "HuggingFaceTB/smollm-corpus", where samples
are randomly interleaved. The draw probabilities are proportional to the estimated remaining samples in each sub-set, which should result in
all datasets running out of examples at about the same time.

Multiple examples are [packed into 4K token blocks](../../../docs/datasets/sequence-packing.md), where masking is applied to prevent cross-attention among samples within the same block.

See [SmolLM-Corpus dataset project](../../datasets/HuggingFaceTB/README.md)

Note that these dataset are quite large and will take a long time to download, if you don't already have them cached. Once cached, loading
is nearly instant, thanks to the new [Fast HF Dataset Loader](../../../docs/datasets/fast-hf-loader.md).

This project will train with all available GPUs, using [DDP](../../tiny_experiments/ddp_trainer/README.md), on the current node, dynamically
adjusting learning-rate with the global batch size.

When multiple GPUs are available, the default is to create a dataset shard for each rank and to process it independently. An alternative is to
set "ns.dispatch_batches = True," which will result in only rank0 reading and pre-processing the dataset, dispatching batches to the other ranks.

## Basic Usage

```bash
# Start training with newly initialized model
forgather train --init-model

# Resume training from last checkpoint
forgather train
```

When initializing the model, you can specify the model project and configuration to construct:

```bash
forgather train --init-model --model-project ../../models/qwen3/ --model-config 124M.yaml
```

By default, Torch SDPA attention will be used, although this is sub-optimal from a compute perspective, as SDPA does not support attention sparsity.
It is worth exploring alternative attention implementations.

```bash
# Use Flex Attention, which supports sparsity
forgather train --attn-implementation flex_attention
```

The training batch sizes have been tuned for a model with about 120M parameters on a GPU with 24 GBs of memory. You may need to adjust batch size to stay 
within the memory constraints of your GPU -- and you probably will want to tweak this anyway, as to optimize available resources.

Note that DDP is not suitable for models which will not fit on a single GPU. This will require other parallelism methods. e.g. Tensor, Pipeline, Fully-Sharded, etc.
With DDP and 24 GBs, you should still be able to reach about 1.7B parameters, although this will require additional memory optimizations.

```bash
forgather train --batch-size 8
```

The learning rate is specified in terms of a global batch-size of 1, where this value is scaled by sqrt(effective_batch_size), where effective_batch_size is 
equal to per_device_train_batch_size * world_size. The base learning rate can be set on the command line for tuning.

```bash
forgather train --lr 2.0e-4
```

## Options

```
usage: forgather train [-h] [-d DEVICES] [--dry-run] [--max-steps MAX_STEPS] [--save-strategy {no,steps,epoch}] [--no-accelerator] [--dist-backend DIST_BACKEND] [--attn-implementation {eager,sdpa,flash_attention_2,flex_attention}] [--verbose-info] [--batch-size BATCH_SIZE] [--model-project MODEL_PROJECT] [--model-config MODEL_CONFIG] [--init-model] [--no-restore-dataset-state] [--no-compile] [--learning-rate LEARNING_RATE]
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
                        Set maximum training steps
  --save-strategy {no,steps,epoch}, -S {no,steps,epoch}
                        When to save checkpoints
  --no-accelerator      Disable use of accelerator, when available. e.g. 'don't use GPU'
  --dist-backend DIST_BACKEND
                        The name of the torch-distributed backend to use
  --attn-implementation {eager,sdpa,flash_attention_2,flex_attention}
                        Attention implementation
  --verbose-info        Display verbose training info on startup
  --batch-size BATCH_SIZE
                        Set the per-device-training batch size.
  --model-project MODEL_PROJECT
                        Path to model project for model initialization
  --model-config MODEL_CONFIG
                        Model project configuration for model init
  --init-model          Initialize model weights
  --no-restore-dataset-state
                        Don't restore dataset state from checkpoint
  --no-compile          Disable Torch compile for faster startup.
  --learning-rate LEARNING_RATE, --lr LEARNING_RATE
                        Set the base learning rate.
  ```

## Examples

Initialize and train Quen3 124 million parameter model using flash-attention-2 on GPUs 0,1,3,4 and 5. Disable Torch compile for faster startup and run for the first 500 steps.

```bash

# Init model and train through to step 500, then checkpoint and stop.
forgather train --init-model --model-project ../../models/qwen3/ --model-config 124M.yaml --attn-implementation flash_attention_2 --no-compile --max-steps 500 -d 0,1,3,4,5

# Resume training from step 500 through the end of the epoch; don't disable Torch Compile.
forgather train --attn-implementation flash_attention_2 -d 0,1,3,4,5
```

Train a 30M parameter Llama model on 5 of 6 GPus.

```bash
train --init-model --model-project ../../models/llama/ --model-config 30M.yaml --attn-implementation flash_attention_2  -d 0,1,3,4,5 --batch-size 16 --step-cadence 4.0
```