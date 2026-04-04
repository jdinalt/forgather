# Getting Started

This guide walks you through installing Forgather and training your first model.

## Prerequisites

- A Linux system (tested on Ubuntu)
- **Python 3.12 or newer.** Forgather uses Python 3.12 language features. Newer
  versions will likely work but are untested; older versions will not.
  Python 3.12 is the default on Ubuntu 24.04. On older Debian-based distributions
  you can install it from the deadsnakes PPA:
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install python3.12 python3.12-venv python3.12-dev
  ```
- An NVIDIA GPU with CUDA support (recommended; CPU-only training is possible but slow)
- A C compiler and Python development headers (required by Triton / flex-attention):
  ```bash
  sudo apt-get install build-essential python3-dev
  ```
- Graphviz (optional, used by some plotting tools):
  ```bash
  sudo apt-get install graphviz
  ```

## Installation

Clone the repository and install in a virtual environment.

**Using venv:**

```bash
# Use python3.12 explicitly if your system default is older
python3.12 -m venv ~/venvs/forgather
source ~/venvs/forgather/bin/activate
git clone https://github.com/jdinalt/forgather.git
cd forgather
pip install -e .
```

**Using uv:**

```bash
git clone https://github.com/jdinalt/forgather.git
cd forgather
uv venv --python 3.12 ~/venvs/forgather
source ~/venvs/forgather/bin/activate
uv pip install -e .
```

Verify the installation:

```bash
forgather ls -r
```

This recursively lists all Forgather projects and configurations found under the
current directory. You should see output listing the bundled example projects.

## Your first training run

The `tiny_llama` tutorial trains a ~4M parameter Llama model on a subset of the
TinyStories dataset. On a single RTX 4090, this takes about three minutes.

```bash
cd examples/tutorials/tiny_llama
```

**List available configurations:**

```bash
forgather ls
```

This shows the project name, description, and available configuration templates.

**Preview the configuration** (optional but useful for understanding what will happen):

```bash
forgather -t train_tiny_llama.yaml pp | less
```

The `pp` command preprocesses the configuration templates (expanding Jinja2,
resolving inheritance) and prints the final YAML. This is the single most useful
debugging tool when working with Forgather configurations.

**Train:**

```bash
forgather -t train_tiny_llama.yaml train
```

Training will download the TinyStories dataset on first run, then begin training.
You will see loss, learning rate, and other metrics printed at each logging step.

## Exploring the results

Once training completes, the model and training artifacts are saved under
`output_models/tiny_llama/`.

**View a training summary:**

```bash
forgather logs summary
```

**Test the model:**

Before loading the model for inference, create symlinks to the latest checkpoint
in the model output directory:

```bash
forgather checkpoint link
```

Then start the inference server and test it:

```bash
# Start the server (loads the model onto GPU)
forgather inf server -c -m output_models/tiny_llama

# In another terminal, generate text
forgather inf client --completion "Once upon a time"
```

The model is small and undertrained, but you should see reasonably coherent short
stories. For a more detailed walkthrough -- including TensorBoard monitoring,
loss plots, text generation with custom sampling parameters, and loading the
model programmatically -- see the
[project notebook](../../examples/tutorials/tiny_llama/project_index.ipynb).

## Key CLI commands

| Command | Description |
|---------|-------------|
| `forgather ls` | List available configurations in the current project |
| `forgather ls -r` | Recursively list all projects and configs |
| `forgather index` | Show project overview as markdown |
| `forgather -t CONFIG pp` | Preview the fully-expanded configuration |
| `forgather -t CONFIG train` | Train with the given configuration |
| `forgather logs summary` | Print summary statistics for the latest training run |
| `forgather logs plot` | Generate training metric plots |
| `forgather inf server -m PATH` | Start the inference server |
| `forgather inf client` | Start the interactive inference client |
| `forgather control list` | List running training jobs |
| `forgather control stop JOB_ID` | Gracefully stop a running job |
| `forgather -i` | Start an interactive shell with tab completion |

Run `forgather --help` or `forgather <command> --help` for full usage details.

## Interactive mode

For day-to-day work, running `forgather -i` launches an interactive shell that is
often easier to use than invoking `forgather` repeatedly from your normal shell.
It provides:

- **Tab completion** for configuration names, commands, and arguments
- **Persistent current template** -- set it once with `config baseline.yaml`, then
  run `pp`, `train`, etc. without repeating `-t baseline.yaml`
- **Project-specific command history** (stored in `.forgather_history`)
- **Editor integration** -- the `edit` command opens template files directly in
  VS Code or vim, with multi-file selection

```bash
forgather -i
forgather> ls                             # List available configurations
forgather> config train_tiny_llama.yaml   # Set current template
forgather> pp                             # Preview configuration
forgather> train                          # Train
forgather> edit                           # Open templates in your editor
```

When running in a VS Code terminal, the interactive CLI automatically detects
VS Code and opens files as editor tabs. This makes it easy to inspect the full
template inheritance chain while working on a configuration.

For the full guide, including vim clientserver setup and multi-file editing,
see the [Interactive CLI Guide](../guides/interactive-cli.md).

## Next steps

With your first model trained, here are recommended paths for learning more:

**Tutorials:**

- [H.P. Lovecraft Project](../../examples/tutorials/hp_lovecraft_project/README.md) --
  Learn how to create workspaces and projects from scratch, while finetuning a 7B
  parameter model on a single 24 GB GPU.
- [Samantha](../../examples/finetune/samantha/README.md) --
  A practical finetuning example using the Samantha dataset with Mistral-7B.
- [Pre-training](../../examples/pretrain/small-llm/README.md) --
  A more involved pretraining project with Chinchilla-optimal scaling, multiple
  optimizer configurations, and pipeline parallelism.

**Understanding the system:**

- [Projects Overview](../../examples/tutorials/projects_overview/project_index.ipynb) --
  Interactive notebook exploring the Project abstraction.
- [Project Composition](../../examples/tutorials/project_composition/project_index.ipynb) --
  How template inheritance works.
- [Configuration Syntax](../configuration/syntax-reference.md) --
  Complete reference for the YAML + Jinja2 configuration language.
- [Model Architecture](../model-architecture.md) --
  Inventory of transformer components in `modelsrc/transformer/`.
