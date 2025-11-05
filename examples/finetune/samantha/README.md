# Samantha Finetune

Finetune a model on the Samantha dataset

## What You'll Learn

This tutorial teaches you how to:
- ✓ Fine-tune a 7B parameter language model on consumer GPUs (full precision, no LoRA/quantization!)
- ✓ Use [pipeline parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) to distribute models across multiple GPUs
- ✓ Train with packed-sequences and flex-attention
- ✓ Scale training across multiple machines over standard Gigabit Ethernet
- ✓ Convert models between HuggingFace and Forgather formats
- ✓ Manage checkpoints and resume training
- ✓ Serve your fine-tuned model via an inference API

**Time required**: ~1-2 hours (mostly waiting for downloads/training)
**Hardware requirements**: 1-6 GPUs with 16-24GB VRAM each

The "Samantha" dataset was an experimental dataset created by Eric Hartford, where the model is taught to believe that she is sentient.

https://erichartford.com/meet-samantha

## Minimum Hardware Requirements

The configurations have been written with the assumption of having a GPU which supports the bfloat16
data format and 24 GBs of VRAM (minimum).

There is an experimental config for a 16 GB GPU. The measured peak usage on a RTX 4090 is 16.23 GB, which *may* work, but I don't have a card to test this on.

We also have configurations for multi-GPU single node and multi-node training. You can mix-and-match GPUs, provided that they all support bfloat16, but the slowest GPU will be the bottleneck.

Tested Configurations:

- single node RTX 4090, x1, x2, x4, x6
- single node, RTX 3090, x1, x2
- multinode, RTX 3090 + RTX 3090
- multinode, RTX 4090 + RTX 3090

## Setup

You will need a model to finetune. For our examples, we will use the base [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) model. This is a raw, pretrained, model, which has never been trained to interact in a chat context before. It will not take very long for this model to become "Samantha," who is a pro at interacting with the ChatML dialog format.

You should be able to use any 7B Llama flavor, with minimal changes to these instructions.

```bash
# Download the model
MODELS_DIR="~/models" # Change this to where you store your models...
SRC_MODEL="${MODELS_DIR}/mistral_7b"
mkdir -p "${MODELS_DIR}"
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir "${SRC_MODEL}" \
--exclude "*.safetensors" "model.safetensors.index.json"
```

### Directory Structure Overview

This tutorial uses the following directory structure:

```
~/forgather/                          # Forgather installation
├── examples/finetune/samantha/       # Tutorial project (working directory)
├── scripts/convert_llama.py          # Model conversion script
└── chat_templates/chatml.jinja       # Chat template

~/models/                             # Models (you create this)
├── mistral_7b/                       # Downloaded HuggingFace model
└── fg_mistral_7b/                    # Converted Forgather model
    ├── pytorch_model-*.bin           # Model weights
    ├── checkpoints/                  # Training checkpoints
    │   ├── checkpoint-100/
    │   ├── checkpoint-200/
    │   └── ...
    └── runs/                         # Training logs
        └── run_2025-10-19.../
```

**Important paths**:
- Work from: `examples/finetune/samantha/`
- Model conversion script: `../../../scripts/convert_llama.py` (relative from tutorial dir)
- Chat template: `../../../chat_templates/chatml.jinja` (relative from tutorial dir)

## Configuration Tour (Optional)

### Configuration Files

While not exhaustive, this is a sampling of the configurations used by this project.

**Samantha Project**
- [samantha.yaml](./templates/samantha.yaml) -- Base project configuration
  - [1gpu_llama_7b/default.yaml](./templates/configs/1gpu_llama_7b/default.yaml) -- Default configuration: 1 GPU, 512 context
  - [1gpu_llama_7b/long_context.yaml](./templates/configs/1gpu_llama_7b/long_context.yaml) -- 1 GPU, 4096 context
  - [1gpu_llama_7b/16gb.yaml](./templates/configs/1gpu_llama_7b/16gb.yaml) -- 1 GPU, 16 GB
  - [1gpu_llama_7b/packed.yaml](./templates/configs/1gpu_llama_7b/packed.yaml) -- 4096 token packed sequences on a single GPU, 24 GBs
  - [pipeline_llama_7b/1f1b_2gpu.yaml](./templates/configs/pipeline_llama_7b/1f1b_2gpu.yaml) -- 2 GPU Pipeline
  - [pipeline_llama_7b/i1f1b_2gpu_packed.yaml](./templates/configs/pipeline_llama_7b/i1f1b_2gpu_packed.yaml) -- 4096 token packed sequences, i1F1B Pipe, 2 GPU
  - [pipeline_llama_7b/i1f1b_4gpu.yaml](./templates/configs/pipeline_llama_7b/i1f1b_4gpu.yaml) -- 4 GPU Pipeline
  - [pipeline_llama_7b/i1f1b_4gpu.yaml](./templates/configs/pipeline_llama_7b/zb_4gpu.yaml) -- 4 GPU Pipeline, Zero Bubble V
  - [pipeline_llama_7b/i1f1b_4gpu.yaml](./templates/configs/pipeline_llama_7b/1f1b_4gpu_float32.yaml) -- 4 GPU Pipeline, float32
  - [pipeline_llama_7b/1f1b_4gpu_adamw.yaml](./templates/configs/pipeline_llama_7b/1f1b_4gpu_adamw.yaml) -- 4 GPU Pipeline, AdamW
  - [pipeline_llama_7b/1f1b_4gpu_adamw4bit.yaml](./templates/configs/pipeline_llama_7b/1f1b_4gpu_adamw4bit.yaml) -- 4 GPU Pipeline, AdamW-4bit

  There are a few more experimental configs as well.

**Finetune**
- [projects/base_finetune_proj.yaml](../../../templatelib/finetune/projects/base_finetune_proj.yaml) -- Base Finetune Project

**Samantha Dataset**
- [samantha.yaml](../../datasets/QuixiAI/templatelib/configs/samantha.yaml) -- Samantha dataset definition
- [samantha-packed.yaml](../../datasets/QuixiAI/templatelib/configs/samantha-packed.yaml) -- Packed Samantha dataset definition
- [src/samantha.py](../../datasets/QuixiAI/src/samantha.py) -- Dataset preprocessing implementation

**Model**
- [models/transformers/dynamic_llama.yaml](../../../templatelib/examples/models/transformers/dynamic_llama.yaml) -- Base Forgather Llama model definition

**Chat Template**
- [chat_templates/chatml_eos.jinja](../../../chat_templates/chatml_eos.jinja) -- [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) chat template definition

### Interactive Forgather CLI
If you have not already installed the syntax-highlighting plugins for vim / VS Code, follow the instructions in "syntax_highlighting/" This will make the config files much more readable.

If running VS Code and not running in a VS Code terminal, you can integrate the terminal with the VS Code editor like this:

```bash
# From a VS code terminal
dinalt@hal9000:~/ai_assets/forgather$ env | grep VSCODE_IPC
VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-1e7b4a5b-9efe-4481-a35e-b489029bc661.sock

# From alternative terminal
export VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-1e7b4a5b-9efe-4481-a35e-b489029bc661.sock

# Edit commands from the external terminal will open files in VS Code!
```

```bash
# Start an interactive Forgather session
forgather -i

# List top level "interactive" commands
forgather:samantha> help

# List Forgather commands
forgather:samantha> commands

# List available configurations.
forgather:samantha> ls

# Change the configuration to "1gpu_llama_7b/long_context.yaml"
# Note that tab-completion is supported
forgather:samantha> config 1gpu_llama_7b/long_context.yaml

# Checkout the template hierarchy for this configuration
# If running in VS Code...
forgather:samantha [1gpu_llama_7b/long_context.yaml]> trefs --format svg -e

# Otherwise...
# Open the resulting file, "long_context.svg," with a compatible viewer.
forgather:samantha [1gpu_llama_7b/long_context.yaml]> trefs --format svg -o long_context.svg

# Take a look at one of the configurations we will be demonstrating
forgather:samantha [1gpu_llama_7b/long_context.yaml]> edit templates/configs/1gpu_llama_7b/long_context.yaml

# Take a look at the base Samantha project configuration.
forgather:samantha [1gpu_llama_7b/long_context.yaml]> edit templates/samantha.yaml

# Take a look at the base finetuning config files.
# First, bring up the menu to interactively select the files to edit
forgather:samantha [1gpu_llama_7b/long_context.yaml]> edit
...
# Then enter the numbers corresponding to "base_finetune_trainer.yaml" and "base_finetune_proj.yaml"
Select template(s) (0-58): 49,50

# Show the preprocessed configuration in the editor
forgather:samantha [1gpu_llama_7b/long_context.yaml]> pp -e

# See what this configuration looks like, when translated to native Python code
forgather:samantha [1gpu_llama_7b/long_context.yaml]> graph --format python -e

# Take a look at the configuration-specific arguments
# Most of these arguments are derived from the configuration's "dynamic_args" section.
forgather:samantha [1gpu_llama_7b/long_context.yaml]> train --help

# Quit, when done
forgather:samantha [1gpu_llama_7b/long_context.yaml]> quit
```

## Control Interface

Forgather has an interface for monitoring and controlling running training jobs. Using this interface is the preferred means of prematurely ending a training job, as it avoids the possibility of causing one or more workers to hang, when using control-c (pipeline parallel frequently hangs on termination).

```bash
usage: forgather control [-h] {list,status,stop,abort,save,cleanup} ...
list                List discoverable training jobs
status              Get status of a training job
stop                Send graceful stop command to a training job (saves final checkpoint)
abort               Abort training job WITHOUT saving checkpoint
save                Trigger checkpoint save in a training job
cleanup             Remove endpoint files for dead training jobs
```

The commands, other than "list," take a job-id as an additional argument, where you can find the
job-id via "list."

## Monitor with Tensorboard

You can monitor your training jobs with Tensorboard

```bash
forgather tb --output-dir OUTPUT_DIR [-- --bind_all]
# --bind_all : Bind to all IP interfaces, otherwise just localhost
```

## Single GPU Training

We will be training the full model, not using a low-rank approximation or quantization. With bfloat16, we need approximately 14 GBs just for the model parameters. With a conventional training setup, you would also need an additional 14 GBs for the gradients, 28 GBs for the optimizer-states, and a fair amount more for activation states (depends on sequence length),

PyTorch uses float32 by default, which takes twice as much memory as bfloat16.

```yaml
[trainer_args]
  ...
  default_dtype: bfloat16 # Use bfloat16, rather than float32
```

We address the optimizer state issue by using Adafactor, without momentum. This optimizer uses negligible
memory for optimizer states and performs nearly identically to AdamW, as long as the batch size is relatively small.

```yaml
[optimizer]
optimizer: &optimizer !partial:forgather.ml.optim.adafactor:Adafactor
  lr: 4.0e-6
  weight_decay: 0.001
...
[trainer_args]
  ...
  per_device_train_batch_size: 8 # As the batch size grows larger, AdamW becomes more effective than Adafactor
```

To address the storage required for gradients, we combine the gradient computation step with the optimizer
step. The result is that we only need to materialize one gradient at a time, and free it immediately after updating the parameter. This saves about 14 GBs.

```yaml
[trainer_args]
...
  fuse_optim_with_backward: True # Combine gradient computation with optimizer step
```

This just leaves the activation memory to contend with. To address this, we use activation checkpointing,
which saves the activation at each layer, discarding the intermediate activations, which can be recomputed
on the backward pass. This trades compute for memory.

```yaml
[trainer_args]
...
  gradient_checkpointing: True # Only save activations at each layer and recompute on backwards step
```

Note that 'fuse_optim_with_backward=True' is synergistic with 'gradient_checkpointing=True'

We can go one step further, by moving the activation checkpoints to CPU memory and back again, when needed to compute the gradient. This allows us to use a context length of 4096 on a single GPU.

```yaml
[trainer_args]
  ...
  enable_activation_offloading: True # Move saved activation to CPU memory
```

### Selecting Specific GPUs

By default, training uses the first N available GPUs. To use specific GPUs:

```bash
# Use only GPU 0
forgather -t config.yaml train -M "${MODEL}" -d 0

# Use GPUs 0, 1, and 3 (skip GPU 2)
forgather -t config.yaml train -M "${MODEL}" -d 0,1,3

# Alternative: use CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,3 forgather -t config.yaml train -M "${MODEL}"
```

This is useful if some GPUs are busy or may have issues.

### Testing Your Configuration

First, let's run a sanity check to verify if everything is working and that we don't run out of GPU memory.

```bash
forgather -t "1gpu_llama_7b/default.yaml" train --save-strategy no --max-steps 10 -M "${SRC_MODEL}" --chat-template "../../../chat_templates/chatml_eos.jinja"

# -t 1gpu_llama_7b/default.yaml : Train on a single GPU with conservative settings.
# --save-strategy no : Don't save checkpoints (for testing)
# -M "${SRC_MODEL}" : Path to the model to train.
# --max-steps 10 : Run a quick test, with only 10 training steps
# --chat-template ../../../chat_templates/chatml_eos.jinja : Use ChatML chat-template
```

The default config is pretty conservative (context length = 512).

Once you have verified that a given config will run, you can train on the full dataset...

```bash
# SRC_MODEL was defined above, which is the path to the model to train.
forgather -t CONFIG_TEMPLATE train -M "${SRC_MODEL}" --chat-template "../../../chat_templates/chatml_eos.jinja"
```

#### Single GPU, 1300 Context Length

```bash
# Train with a context length of 1300 on HF model
forgather -t "1gpu_llama_7b/med_context.yaml" -M "${SRC_MODEL}" --chat-template "../../../chat_templates/chatml_eos.jinja"

# Train with a context length of 1300 on Fg model
# FG_MODEL is defined below, after converting the model format
forgather -t "1gpu_llama_7b/med_context.yaml" train -M "${FG_MODEL}"
```

#### Single GPU, 4096 Context Length

We can go further by offloading the activation storage to CPU memory. This allows full training of a 7B parameter model, with a sequence length of 4096, on a 24 GB device!

A sequence length of 1300 covers 90% of the examples in this dataset, so there's not much to be gained with
a longer sequence length, but this may be useful for other datasets.

**NOTE**

Testing this configuration on the native HF model appears to barf with a device-side assertion. This appears to be related to using CPU checkpoint offloading. It's unclear if it's just this model or a more general issue.

In any case, it works fine if you convert the model to Forgather's native format first. See instructions for multi-GPU training for details on how to do this.

```bash
# Train with a context length of 4096:
forgather -t 1gpu_llama_7b/long_context.yaml  -M "${FG_MODEL}"
```
#### Single GPU, 4096 Packed

With [sequence packing](../../../docs/datasets/sequence-packing.md), we use "best_fit" packing to optimally pack as many examples as we can within the sequence dimension. This results very little compute wasted on "pad" tokens. We then use the attention mask to prevent cross-example attention.

Normally, this would require O(N^2) compute for these long sequences, but by using [Pytorch Flex Attetnion](https://pytorch.org/blog/flexattention/), the attention function is sparse, which can greatly reduce mean compute.

The attention implementation to use can be specified on the CLI via:

```bash
--attn-implementation {eager,sdpa,flash_attention_2,flex_attention}

# e.g.
forgather -t "1gpu_llama_7b/packed.yaml" train -M "${FG_MODEL}" --attn-implementation flex_attention
```

This will also work with "sdpa" and "eager," but they will require significantly more peak memory and compute.

Note that flex-attention makes use of torch.compile(). The first time you run a particular configuration, there will be some compile overhead, along with verbose diagnostic messages.

#### Single GPU, 16 GB

We can try to train on a 16 GB GPU. With the model weights using 14 GB, it's going to be pretty tight!

As above, this does not work with the HF model. Convert it to Forgather's format first.

```bash
# Train on a 16 GB GPU
forgather -t 1gpu_llama_7b/16gb.yaml train -M "${FG_MODEL}"
```

## Multi-GPU Setup

For our multi-GPU configurations, we will be using pipeline parallel. This is far more performant on consumer-grade hardware than Fully-Sharded-Data-Parallel (FSDP). Consumer-grade GPUs generally lack a high-speed interconnect (NVLINK). Without this, FSDP, the primary alternative, is painfully slow.

The catch is that we need to split the model into multiple segments. While we have tools for automatic model splitting, the tools don't support all PyTorch features and HF Llama-like models use unsupported features.

To work around this, we have a script for converting Llama-like models to a compatible format (and back again).

You can also do this for single-GPU training. In most cases, it should not be required, but it can reduce peak GPU memory usage a little bit. This step may still be required to use CPU Checkpoint Offloading, as the HF version of the model appears to crash with this option enabled.

```bash
# From the Forgather root directory
# Set name for converted model
FG_MODEL="${MODELS_DIR}/fg_mistral_7b"

# Convert model to Forgather Llama/Mistral implementation
scripts/convert_llama.py --model-type mistral --dtype bfloat16 --max-length 16384 \
-t "chat_templates/chatml_eos.jinja" "${SRC_MODEL}" "${FG_MODEL}" \
--add-tokens "scripts/example_additional_tokens.yaml"
```

To convert the model back to HF format...

```bash
scripts/convert_llama.py --reverse --model-type mistral --dtype bfloat16 \
--max-length 32768 "${FG_MODEL}" OUTPUT_MODEL_PATH
```

## Single Node Training

First, check if everything is working, like this:

```bash
forgather -t "pipeline_llama_7b/2gpu_1f1b.yaml" train --save-strategy no --max-steps 10 -M "${FG_MODEL}"
# Note that we don't need to specify the chat-template, as the conversion tool bakes it into the tokenizer.
```

There are quite a few different configurations defined, with different schedulers and numbers of GPUs.

- pipeline_llama_7b/2gpu_1f1b.yaml : Requires the least amount of memory. This is similar to the single-GPU settings, but without activation checkpointing.
- pipeline_llama_30b/1f1b_4gpu.yaml : As above, but 4 GPUs.
- pipeline_llama_7b/i1f1b_4gpu.yaml : Faster than 1f1b, but uses more memory
- pipeline_llama_7b/zb_4gpu.yaml : Zero-bubble scheduler. Fast, but uses more memory. This config does not support validation yet, so you will have to rely on training loss as a guide.

## Testing the Finetuned Model

You can test the resulting model using the provided Open-AI compatible inference server and client or with 3rd party tools.

```bash
# Start inference server (from 'forgather' directory)
# Change the model path to match your output directory.
# If the model already has the correct chat-template, you can drop the "-t CHAT_TEMPLATE" arg.
tools/inference_server/server.py -d "cuda:0" -t chat_templates/chatml_eos.jinja -T bfloat16 \
-c -m /home/dinalt/ai_assets/models/fg_mistral

# Note: -c : This will search for the latest checkpoint, rather than loading the model from the root directory.
```

Test if inference is working:

```bash
./tools/inference_server/client.py --stream --message "Hello, what is your name?"
Hi! I'm Samantha, and it's great to meet you.
```

Start an interactive session:

```bash
./tools/inference_server/client.py --interactive --stream
Interactive Chat Mode (type 'quit', 'exit', or 'q' to quit)
Commands:
  /clear    - Clear conversation history
  /system <message> - Set system prompt
  /help     - Show this help

> Hello Samantha. How are you feeling today? 
I'm feeling quite engaged and excited to continue our exploration of new ideas and perspectives. What would you like to discuss today?

>
```

Test the model with text completion:

```bash
./tools/inference_server/client.py --stream --completion "Once upon a time" --max-tokens 50
Once upon a time, before the age of social media, people used to write letters to each other. This was a way for them to express their thoughts, feelings, and emotions, and to stay connected with one another. Although letter-writing is not as common today
```

The server is Open-AI compatible, so you should be able to use any client compatible with this API.

## Multi-node Training

This scenario is considerably more complex than the single-node scenario, with many factors requiring consideration. This will require gathering and configuring network settings, a shared file system, software compatibility, a strategy for loading the initial seed weights, a strategy for a shared dataset, and a strategy for saving checkpoints.

### CLI Options

"forgather train" automatically sets the "torchrun" arguments for single-node training. For multi-node, you will need to explicitly pass them on the commandline.

To see what "torchrun" command will be used, without actually invoking it, pass the "--dry-run" argument. This can be used for diagnostics or as a starting point for manually invoking "torchrun."

```bash
forgather -t CONFIG_TEMPLATE train --dry-run ...
```

To manually pass arguments to "torchrun," append "-- ARGS..." to the end of the command:

```bash
forgather -t CONFIG_TEMPLATE train TRAINING_ARGS... -- TORCHRUN_ARGS...
```

#### torchrun args

```
--nnodes NNODES : Number of nodes
--nproc-per-node NPROC_PER_NODE : Number of workers per node; supported values: [auto, cpu, gpu, int]
--rdzv-backend RDZV_BACKEND : Rendezvous backend
--rdzv-endpoint RDZV_ENDPOINT : Rendezvous backend endpoint; usually in form <host>:<port>
--rdzv-id RDZV_ID : User-defined group id
--rdzv-conf RDZV_CONF : Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...)
```

**Examples**

```bash
# Two GPUs
... --nnodes 1 --nproc-per-node 2

# Four GPUs
... --nnodes 2 --nproc-per-node 2
# or
... --nnodes 4 --nproc-per-node 1
```

If the nodes don't have the same number of GPU's, say one has 1 GPU and another has 3, then set nproc-per-node to match the number of GPUs on that node.

```bash
# First node
... --nnodes 2 --nproc-per-node 1

# Second node
... --nnodes 2 --nproc-per-node 2
```

RDZV_BACKEND should be "c10d"

```bash
... --rdzv-backend c10d
```

There are alternatives, but they are outside the scope of these instructions.

RDZV_ENDPOINT : One of the nodes but needs to be chosen to host the rendezvous, which will be used to coordinate the job. This can be a host-name or an IP address; the port is optional, but defaults to 29400.

```bash
# Example, where host-name is hal9000.
... --rdzv-endpoint hal9000:29400
```

RDZV_ID : A user defined group-id. This is just a number. Pick one and make sure to use the same values on all nodes.

```bash
# Example
... --rdzv-id 123
```

RDZV_CONF : Additional args to pass to the rendezvous. As torchrun may have difficulty figuring out which machine is the host,
pass "is_host=true" only on the host.

```bash
# Pass only on the host
... --rdzv-conf "is_host=true"
```

#### Environment Variables

There are a few environment variables you should be aware of. Environment variables can be set be prefixing the command with
their values.

```bash
# Example of passing NCCL interface name
NCCL_SOCKET_IFNAME=eth0 forgather ...
```

**NCCL_SOCKET_IFNAME=IF_NAME**

This explicitly sets the IP interface name to use for communication. NCCL communication is independent of the rendezvous config and has a tendency to pick the wrong Ethernet interface, if not explicitly told which one to use. Check the results of "ip addr" and find the name of the interface connected to the network you will be using.

**TORCH_CPP_LOG_LEVEL=INFO** or **TORCH_DISTRIBUTED_DEBUG**

These options enable additional synchronization checks and logging, which can be useful for debugging.

See [reference](https://docs.pytorch.org/docs/stable/distributed.html)

**CUDA_LAUNCH_BLOCKING=1**

This forces all communication to be synchronous. This is terrible for performance, but very useful for debugging hangs.

See [reference](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

**NCCL_DEBUG=TRACE** or **NCCL_DEBUG=INFO**

These options cause NCCL to dump additional debug information which can be helpful for debugging communication issue.

See [reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

[All NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

### Example

We will assume that we have two nodes:

**hal9000**
```
GPU(s): RTX 4090 x 6
IP Interface: enp37s0f1
Path to model (NFS share): /home/dinalt/ai_assets/models/fg_mistral
CWD (NFS share): /home/dinalt/ai_assets/forgather
```

**muthur**
```
GPU(s): RTX 3090 x 1
IP Interface: eno1
Path to model (NFS share): /mnt/ai_assets/models/fg_mistral
CWD (NFS share): /mnt/ai_assets/ai_assets/forgather
```

We have configured a NFS volume, where "/home/dinalt/ai_assets/," on "hal9000" is mounted at "/mnt/ai_assets" on "muthur." Our current working directories on each node correspond to "Path to Forgather," which ensures that the configuration files are identical on both nodes, even if we make changes. The model directory, "fg_mistral," is also shared between the two hosts.

We will have hal9000 host the rendezvous and we will be using the "pipeline_llama_7b/2gpu_1f1b.yaml" config, which is for 2 GPUs.

Start job on "hal9000"
```bash
NCCL_SOCKET_IFNAME=enp37s0f1 forgather -t pipeline_llama_7b/2gpu_1f1b.yaml -p examples/finetune/samantha/ train \
-M /home/dinalt/ai_assets/models/fg_mistral -- --nnodes 2 --nproc-per-node 1 --rdzv-backend c10d \
--rdzv-endpoint hal9000:29400 --rdzv-id 1 --rdzv-conf "is_host=true"
```

Start job on "muthur"
```bash
NCCL_SOCKET_IFNAME=eno1 forgather -t pipeline_llama_7b/2gpu_1f1b.yaml -p examples/finetune/samantha/ train \
-M /home/dinalt/ai_assets/models/fg_mistral -- --nnodes 2 --nproc-per-node 1 --rdzv-backend c10d \
--rdzv-endpoint hal9000:29400 --rdzv-id 1
```

The command are nearly identical, excepting these point:
- The IP interface name matches that of the host it is running on (NCCL_SOCKET_IFNAME). Without specifying this, NCCL may pick the wrong interface to bind to.
- The path to the shared model directory. It's the same set of files, just via a different path.
- Only hal9000 has --rdzv-conf "is_host=true," which is needed because "torchrun" is not very good at correctly inferring that this is the rendezvous host.

The order in which they are started is not critical, although there is a 60 second timeout window in which to start all of the hosts.

Note that these machines have different GPU types. This works, but the slower of the two, the RTX 3090, is going to be a bottleneck. In theory, it should be possible to make this work with an asymmetric numbers of GPUs, say 3 GPUs on one machine, and 1 on the other, but "torchrun" does not support it. Supporting such a configuration is on my "TODO" list.

### Network Setup

Pipeline parallel requires relatively low bandwidth; a plain Gigabit Ethernet link should suffice. WiFi is probably workable too, as long as there is a strong signal, plenty of bandwidth, and reasonably low latency, but a wired network is preferable.

Ideally, all of the nodes should be in the same subnet. Although there is no reason that it should not work through a router, the router could potentially be a bottleneck and adds latency.

If you have a firewall enabled, you will need to add exceptions for the participating hosts/ports. If you encounter communications issues, I would recommend disabling your firewall(s) temporarily, as this makes debugging the issue much easier. You will need to have port 29400 open for the rendezvous. Additional ports will be needed for the communication backend. By default, NCCL will use any available ephemeral port, which complicates firewall setup. You can find instructions for narrowing the range of ports used [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).

Similarly, if you are using Docker, you will need to ensure that the required ports can be reached from your network. The easiest solution is to specify "--network host," which provides direct access to all of the host's network interfaces.

### Shared File System Setup

While not strictly required, having a shared filed system greatly simplifies things. I would suggest setting up a shared NFS volume, which will be assumed for the remainder of the tutorial. Consult your favorite search engine or LLM for details on how to do this.

### Software Setup

Ideally, all nodes should have an identical software environment. Using a common Docker container is the safest approach, although a fresh Python virtual environment may be sufficient.

If things are not working as expected, double-check that all of your package versions match!

#### Verify Software Versions Match

Before starting multi-node training, verify all nodes have matching PyTorch and NCCL versions:

```bash
# Run on each node
python -c 'import torch; print(f"PyTorch: {torch.__version__}\nNCCL: {torch.cuda.nccl.version()}")'
```

Example output:
```
PyTorch: 2.8.0+cu128
NCCL: (2, 27, 3)
```

All nodes **must** show identical versions. Even minor version differences will cause "Mismatched NCCL version" errors.

### Initial Checkpoint

When training starts, we need to load the initial checkpoint. One was to solve this issue is to use store the model in a shared NFS directory, as we do in the above example. This can be a bit slow to load, although, it will be cached for subsequent runs. This is my recommend approach.

> **Note on Network Storage Performance**: When loading models over NFS or network storage,
> initial model loading can take significant time (e.g., ~90 seconds for a 14GB model over
> Gigabit Ethernet). This is a one-time cost at the start of training - subsequent steps
> use cached data and run at normal speed. This is expected behavior, not a problem.

An alternative is to place an identical (local) copy of the initial weights on each node and specify the checkpoint to load on the commandline.

```bash
forgather ... train ... --resume-from-checkpoint /path/to/local/checkpoint
```

The primary disadvantage to this approach is that if you need to resume from a new checkpoint, you will need to remove this argument. This can be an issue if torchrun is configured for fault-tolerance. When a failure occurs, it will roll-back to the latest checkpoint by restarting the training script. With this parameter set, this will cause it to rollback to the start, rather than to the latest checkpoint.

### Output Directory

We will saved checkpoints (and logs) in the output directory, which defaults to the model directory. When this is an NFS share, saving (and loading) checkpoints can be pretty slow. An alternative is to specify a unique local output directory on each node. In this case, each node will save (and load) only its shards in that directory. The only disadvantage to this approach is that the checkpoints will be scattered across all nodes and these will need to be collected at the end of training into a common directory before the model can be used for inference.

```bash
forgather ... train ... --output-dir /path/to/local/output_dir --save-on-each-node
```

The "--save-on-each-node" flag will result in each node saving a copy of the files which are common to all nodes. In this case, the "pytorch_model.bin.index.json" and "eval_metrics.json" files. Don't use this option when using a shared directory, as it may corrupt the shared files.

**tip**

You can also use the "--output-dir" option when using a shared output directory (don't pass "--save-on-each-node"). If you copy everything (config.yaml, source-code, tokenizer), excepting the model weights from the original directory, the checkpoint saving logic will automatically symlink the saved weights from the latest checkpoint into the root of the output directory. This can be useful for testing the model with external tools, while the model is still training. For example, with [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

## Troubleshooting

### "Mismatched NCCL version detected"

**Symptom**: Multi-node training fails with error like:
```
RuntimeError: Mismatched NCCL version detected : rank 1 version 22705 rank 0 version 22703
```

**Cause**: Different PyTorch/NCCL versions on different nodes. This can happen when PyTorch releases a new version while you're testing.

**Solution**:
1. Verify versions on all nodes:
   ```bash
   python -c 'import torch; print(f"PyTorch: {torch.__version__}\nNCCL: {torch.cuda.nccl.version()}")'
   ```
2. Use the same Python environment (venv/conda) on all nodes
3. If using containers, ensure all nodes use the exact same container image

### Training Hangs on Multi-node Setup

**Symptoms**: Training starts but hangs at initialization or after a few steps.

**Debugging steps**:
1. **Check network connectivity**: Ensure all nodes can reach each other on the required ports
   ```bash
   # On each node, test connectivity to rendezvous host
   ping hal9000
   ```

2. **Check NCCL interface**: NCCL may be trying to use the wrong network interface
   ```bash
   # Find your network interfaces
   ip addr

   # Set the correct interface explicitly
   NCCL_SOCKET_IFNAME=enp37s0f1 forgather ...
   ```

3. **Enable debug logging**:
   ```bash
   NCCL_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO forgather ...
   ```

4. **Test with synchronous execution** (slow but helps identify hangs):
   ```bash
   CUDA_LAUNCH_BLOCKING=1 forgather ...
   ```

### Model Loading Takes Extremely Long (Multi-node)

**Symptom**: First training step takes 60-90 seconds, then subsequent steps are normal speed.

**This is expected behavior**, not a bug! When loading a 14GB model over Gigabit Ethernet (~125 MB/s theoretical max), it takes time:
- 14GB model / 125 MB/s ≈ 112 seconds (theoretical)
- Real-world with overhead: 60-90 seconds

**Solutions**:
- **Accept it**: It's a one-time cost at training start
- **Use local copies**: Copy model to local disk on each node, use `--resume-from-checkpoint`
- **Upgrade network**: Use 10GbE if available

### FileNotFoundError: model.safetensors

**Symptom**: Training fails looking for `model-00001-of-00002.safetensors`.

**Cause**: Downloaded both PyTorch and SafeTensors formats, but SafeTensors index file exists while weights don't.

**Solution**: Exclude SafeTensors files during download:
```bash
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir "${SRC_MODEL}" \
--exclude "*.safetensors" "model.safetensors.index.json"
```

Alternatively, delete the "index" file, for which weights don't exist.

## Common Warnings and Expected Behaviors

### Control Callback Shutdown Warning

When training ends, you may see:
```
WARNING:forgather.ml.trainer.callbacks.trainer_control:Control callback shutdown timed out after 2.0 seconds
```

**This is normal** and can be safely ignored. The control interface cleanup times out but doesn't affect training results.

### HuggingFace CLI Deprecation Warning

You may see warnings about deprecated `huggingface-cli download` syntax. These can be safely ignored - the commands in this tutorial work correctly despite the warnings. The newer CLI command is "hf," although I have yet to write instructions for using it.

## Finalizing the Model

When you are done training and wish to consolidate everything to use with external tools (or share), you will want to copy the latest checkpoint weights into the root of the model directory -- most tools don't know how to find the latest checkpoint and may load the initial weights instead.

If you have converted the model to Forgather format, I would recommend converting it back to the original HF format, as the Forgather format is not optimized for inference yet. See above for details.

You can then discard the additional checkpoints and logging data, if you don't need them for anything else.
