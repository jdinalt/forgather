# Finetuning Examples

The finetuning template adds a number of additional arguments to "forgather train":

```bash
orgather train --help
usage: forgather train [-h] [-d DEVICES] [--dry-run] [--max-steps MAX_STEPS] [--save-strategy {no,steps,epoch}] [--attn-implementation {eager,sdpa,flash_attention_2,flex_attention}] [--train-epochs TRAIN_EPOCHS] [--log-peak-memory] [--dataset-config DATASET_CONFIG] [--dataset-proj DATASET_PROJ] [--model-id-or-path MODEL_ID_OR_PATH] [--output-dir OUTPUT_DIR]
                       [--gradient-checkpointing] [--resume-from-checkpoint RESUME_FROM_CHECKPOINT] [--save-on-each-node] [--chat-template CHAT_TEMPLATE]
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
  --attn-implementation {eager,sdpa,flash_attention_2,flex_attention}
                        Attention implementation
  --train-epochs TRAIN_EPOCHS
                        Set the number of epochs to train for
  --log-peak-memory, -P
                        Log peak GPU memory at each log step
  --dataset-config DATASET_CONFIG
                        The name of the dataset configuration to use
  --dataset-proj DATASET_PROJ
                        Path to dataset project to use
  --model-id-or-path MODEL_ID_OR_PATH, -M MODEL_ID_OR_PATH
                        HF model ID or local path to model
  --output-dir OUTPUT_DIR
                        Training output director. Defaults to model_id_or_path
  --gradient-checkpointing, -G
                        Enable gradient (activation) checkpoint, when supported
  --resume-from-checkpoint RESUME_FROM_CHECKPOINT
                        Explicit checkpoint path to load
  --save-on-each-node   Save common checkpoint files on each node
  --chat-template CHAT_TEMPLATE, -C CHAT_TEMPLATE
                        Path to the chat template to use
```

## Usage

See [Samantha](./samantha/README.md)
