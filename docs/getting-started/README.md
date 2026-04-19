# Getting Started

This guide walks you through installing Forgather and training your first model.

## Prerequisites

- A Linux system (tested on Ubuntu 24.04)
- **Python 3.12 or newer.** Forgather uses Python 3.12 language features. Newer
  versions will likely work but are untested; older versions will not.
  Python 3.12 is the default on Ubuntu 24.04. On older Debian-based distributions
  you can install it from the deadsnakes PPA:
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install python3.12 python3.12-venv python3.12-dev
  ```
- An NVIDIA GPU with CUDA support is strongly recommended but not
  required. CPU-only training works -- the Tiny Llama tutorial below
  has been run end-to-end on a Chromebook, taking most of a day for
  the same workload that finishes in ~2 minutes on an RTX 4090.
  Budget accordingly.  Non-CUDA accelerators (Intel, AMD, Apple
  Silicon) *may* work -- Forgather deliberately avoids hard CUDA
  dependencies where possible -- but have not been tested outside of
  CUDA and CPU, so treat them as experimental.
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

**Recommended: install cut-cross-entropy from source:**

The pip-installable version of `cut-cross-entropy` (25.1.1) is missing features
needed for numerical stability during bf16/fp16 training (`accum_e_fp32`,
`accum_c_fp32`). Forgather will fall back gracefully, but training may exhibit
lm_head spectral norm explosion over long runs. Install the latest version from
source:

```bash
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
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

For a complete walkthrough — including TensorBoard monitoring, loss plots, text
generation, and programmatic model loading — see the
[Tiny Llama tutorial](../tutorials/tiny_llama/README.md).

```bash
cd examples/tutorials/tiny_llama
```

**List available configurations:**

```bash
forgather ls
```

**Train:**

```bash
forgather -t v2.yaml train
```

Training downloads the TinyStories dataset on first run, then prints loss,
learning rate, and other metrics at each logging step. Artifacts are saved under
`output_models/tiny_llama/`.

**Summarize the results:**

```bash
forgather logs summary
```

**Run an evaluation:**

```bash
forgather -t v2.yaml eval test tinystories
```

Results are written to `output_models/tiny_llama/evals/` as Markdown and JSON.

## Key CLI commands

| Command | Description |
|---------|-------------|
| `forgather ls` | List available configurations in the current project |
| `forgather ls -r` | Recursively list all projects and configs |
| `forgather index` | Show project overview as markdown |
| `forgather -t CONFIG pp` | Preview the fully-expanded configuration |
| `forgather -t CONFIG train` | Train with the given configuration |
| `forgather -t CONFIG tb` / `forgather tb --all` | Launch TensorBoard on this config's runs / every run |
| `forgather logs summary` | Print summary statistics for the latest training run |
| `forgather logs plot` | Generate training metric plots |
| `forgather -t CONFIG eval test NAME` | Run a named evaluation config on the trained model |
| `forgather inf server -c -m PATH` | Start the inference server (`-c` = load latest checkpoint) |
| `forgather inf client` | Start the interactive inference client |
| `forgather control list` | List running training jobs |
| `forgather control stop JOB_ID` | Gracefully stop a running job |
| `forgather checkpoint link` | Symlink latest checkpoint for plain `from_pretrained` loading |
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

- [Tiny Llama](../tutorials/tiny_llama/README.md) --
  Full walkthrough of the getting-started project: TensorBoard, loss plots, text
  generation, and programmatic use.
- [H.P. Lovecraft Project](../tutorials/hp_lovecraft_project/README.md) --
  Learn how to create workspaces and projects from scratch, while finetuning a 7B
  parameter model on a single 24 GB GPU.
- [Samantha](../tutorials/samantha/README.md) --
  A practical finetuning example using the Samantha dataset with Mistral-7B.

**Understanding the system:**

- [Projects Overview](../tutorials/projects_overview/project_index.ipynb) --
  Interactive notebook exploring the Project abstraction.
- [Project Composition](../tutorials/project_composition/project_index.ipynb) --
  How template inheritance works.
- [Configuration Syntax](../configuration/syntax-reference.md) --
  Complete reference for the YAML + Jinja2 configuration language.
- [Model Architecture](../guides/model-architecture.md) --
  Inventory of transformer components in `modelsrc/transformer/`.
