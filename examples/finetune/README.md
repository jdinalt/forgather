# Finetuning Examples

This is a work-in-progress, demonstrating how to use Forgather for finetuning applications.

At present, I have only added a single dataset, Samantha, which is being used to test various use cases to see what
works, what does not (and needs to be fixed), and how to actually do it.

A few interesting datapoints:
- Using various memory saving techniques, it actually possible to perform full finetuning on a 7B Llama model, with a
  context length of 2048 tokens, on a single 24 GB GPU. This makes use of a custom Adafactor optimizer, gradient
checkpointing, and fused optimizer/backward steps. The performance is even reasonably good.
- With two RTX 4090s, using Pipeline Parallel, the same model can be trained without the above memory saving techniques.
- With 4 RTX 4090s, I was able to train a 30B Llama model -- with memory saving.
- With 6 RTX 4090, a 30B model can be trained with a pretty long context length.

Without NVLink, pipeline parallel is so much faster than FDSP!

There's still quite a bit of work to do, but I figured that this at least provides some guidance on how to make this
work.

Some of the examples are broken. I have added notes to the config, where applicable.

The finetuning template adds a number of additional arguments to "forgather train":

```bash
forgather train --help
usage: forgather train [-h] [-d DEVICES] [--dry-run] [--max-steps MAX_STEPS] [--save-strategy {no,steps,epoch}] [--train-epochs TRAIN_EPOCHS] [--log-peak-memory] [--dataset-config DATASET_CONFIG] [--dataset-proj DATASET_PROJ] [--model-id-or-path MODEL_ID_OR_PATH] [--output-dir OUTPUT_DIR] [--safe-load] [--gradient-checkpointing] [--chat-template CHAT_TEMPLATE] ...

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
  --save-strategy {no,steps,epoch}
                        When to save checkpoints
  --train-epochs TRAIN_EPOCHS
                        Set the number of epochs to train for
  --log-peak-memory     Log peak GPU memory at each log step
  --dataset-config DATASET_CONFIG
                        The name of the dataset configuration to use
  --dataset-proj DATASET_PROJ
                        Path to dataset project to use
  --model-id-or-path MODEL_ID_OR_PATH
                        HF model ID or local path to model
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Training output director. Defaults to model_id_or_path
  --safe-load           Fallback to the more compatible HF model loading method
  --gradient-checkpointing
                        Enable gradient (activation) checkpoint, when supported
  --chat-template CHAT_TEMPLATE
                        Path to the chat template to use
```

Note that one potential issue with some models is that they use the "persist=False" flag for some of their buffers. This means that these buffers are not stored with
the model's state dictionary. This presents a problem when constructing a model on the "meta" device. The "meta" device is a PyTorch pseudo-device.

When a model is constructed on this device, only the model skeleton is actually created, where the buffers and parameters have meta-data, but 
are not backed my memory. This allows one to construct a model, even it would not otherwise fit in memory. As the model's tensors are not
backed by memory, they are not initialized.

1. Get the model's configuration and construct the model with .from_config() on the "meta" device.
2. Transform the model to the desired dtype, for example, bfloat16.
3. Materialize the model's buffers/parameters directly on the GPU with .to_empty(). This avoid constructing the model on the CPU and having to copy it to the device, but the allocated memory is uninitialized.
4. Load the model's state dictionary directly from disk to the GPU via memory mapping. This is a lazy loading method and is very fast.

The problem arises when some of the model's parameters are missing from the state dictionary, as when the "persist=False" flag is set. The result is that the model is not fully initialized.

In practical terms, the relative positional encoder buffer, on HF Llama models, have "persist=False" set, so this does not work.

The work-around is to use the "--safe-load" flag, which falls back to constructing the the model on the cpu, loading the weights, then moving it to the GPU. It's much slower! I should be able to optimize this partially, be disabling the standard PyTorch weight init methods; this is on the TODO list.

To address this (and other pipeline compatibility issues) with the HF Llama-like models, use "scripts/convert_llama.py" to convert the model
to a more pipeline friendly format. You can then covert the resulting model back to HF format, when done.

## Examples

The CLI can be (optional) run as an interactive shell. This has tab-completion and history support, which may make it easier to work with than from bash.

```bash
forgather -i
Welcome to the Forgather interactive shell.
Project found at: /home/dinalt/ai_assets/forgather/examples/finetune/samantha
Use tab completion for templates, commands, and directories.
Examples: "template <TAB>", "-t <TAB>", "tr<TAB>", "cd <TAB>"
Type help or ? to list commands.

forgather:samantha> ?

Documented commands (type help <topic>):
========================================
EOF  cd  commands  config  configs  debug  edit  exit  help  pwd  quit

forgather:samantha> commands
Available commands:
  code
  construct
  dataset
  graph
  index
  ls
  meta
  pp
  targets
  tb
  tlist
  train
  trefs
  ws
forgather:samantha> config pipeline_llama_7b/1f1b_4gpu.yaml
Set template to: pipeline_llama_7b/1f1b_4gpu.yaml
forgather:samantha [pipeline_llama_7b/1f1b_4gpu.yaml]>
```

Train a HF Llama 7B (or similar) model on a single GPU (24 GB)

```bash
# From the samantha directory
forgather train --model-id-or-path ~/ai_assets/models/meta-llama--Llama-2-7b-hf --chat-template ~/ai_assets/forgather/chat_templates/chatml.jinja --log-peak-memory --safe-load --gradient-checkpointing
```

Convert the HF model to Forgather's format

Note that "--enable-checkpoint" sets the equivalent option, "--gradient-checkpointing," above by storing it the model's configuration.
```bash
# From scripts directory, convert the model to Forgather format
./convert_llama.py --model-type --dtype bfloat16 --max-length 4096 --enable-checkpoint -t ../chat_templates/chatml.jinja ~/ai_assets/models/meta-llama--Llama-2-7b-hf/ ~/ai_assets/models/llama-2-7b-fg
```

Train the same model, after being converted with "scripts/convert_llama.py"

```
# From samantha directory
forgather train --models-dir ~/ai_assets/models/ --model-name llama-2-7b-fg --log-peak-memory 
```

Train the 7B model on a 2 GPU Pipeline

```bash
forgather -t pipeline_llama_7b/2gpu.yaml  train --model-id-or-path ~/ai_assets/models/llama-2-7b-fg --log-peak-memory
```

Train a 30B model with a 4 GPU pipeline

```bash
forgather -t pipeline_llama_30b/1f1b_4gpu.yaml train --model-id-or-path ~/ai_assets/models/llama-2-7b-fg --log-peak-memory --gradient-checkpointing
```
