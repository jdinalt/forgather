# Getting Started

This guide walks you through installing Forgather and training your first model from the CLI.

> **Prefer a web UI?** Forgather ships with a single-user web frontend
> over the same APIs the CLI uses — project browsing, queued training,
> live GPU monitoring, log viewing, an in-browser editor, and a chat
> client against served models. If that sounds more useful than the
> CLI to you, jump straight to the
> [**Forgather server walkthrough**](../guides/forgather-server-walkthrough.md)
> — it's an end-to-end tour from a fresh install to chatting with a
> small model you train along the way. The setup overlaps with this
> guide; the walkthrough links back here for the install steps.
>
> ![Forgather server first-load view](../guides/screenshots/05-trefs-graph.png)

> **Want to skip the host setup?** Forgather ships a development
> Dockerfile that provisions Python 3.12, PyTorch with CUDA wheels,
> all dependencies, and a developer-friendly base toolchain in a
> reproducible image. If you'd rather not touch your host Python at
> all, jump to [**Installing with Docker**](#installing-with-docker)
> below.

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
- `git` (used to clone the repo and to fetch the `cut-cross-entropy`
  source install below). On most distributions it's installed by
  default, but minimal Docker base images (e.g. plain `ubuntu:24.04`)
  don't ship it:
  ```bash
  sudo apt-get install git
  ```
- **Graphviz** (optional). Only used by the CLI's
  `forgather trefs --format svg`, which shells out to `dot` to render
  template-dependency graphs as SVG. The Forgather server's
  in-browser graph view bundles a WebAssembly build of Graphviz
  (`@viz-js/viz`) and works without the system package.
  ```bash
  sudo apt-get install graphviz
  ```
- **Node.js + npm** (optional, only for the Forgather server's web
  UI). The `forgather server` command serves a Vite/React SPA built
  from `tools/forgather_server/webui/`. The build artifact isn't
  checked in, so you build it once after install — see "Running the
  Forgather server" below. Any current LTS Node release works
  (tested on Node 20).
  ```bash
  sudo apt-get install nodejs npm
  ```
  None of this is needed if you only use the CLI; the running server
  itself has no Node dependency once the dist bundle exists.

## Installation

Clone the repository, then install in a virtual environment.

**Using venv:**

```bash
git clone https://github.com/jdinalt/forgather.git
cd forgather

# Use python3.12 explicitly if your system default is older.
python3.12 -m venv ~/venvs/forgather
source ~/venvs/forgather/bin/activate

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

The install pulls in PyTorch, transformers, the FastAPI server
deps, mkdocs, and a few other large packages — expect ~2–3 GB of
downloads on a fresh machine. On a slow network the first install
can take several minutes; if pip looks stuck it's almost certainly
still downloading.

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

## Installing with Docker

The repo ships a `Dockerfile` (and matching helpers in `docker/`)
that builds an Ubuntu 24.04 image with the full Forgather environment
pre-provisioned: Python 3.12, PyTorch (CUDA wheels), all
dependencies, `cut-cross-entropy` from source, and a developer
toolchain (vim, tmux, ripgrep, jq, htop, ssh, sudo, ...). It's
useful in two ways:

- **As a development environment** — one command and you have a
  working Forgather install without touching your host Python.
- **As a clean sandbox for release testing** — build the image with
  `--no-cache` and you get a reproducible from-scratch verification
  that the source tree builds and runs end-to-end.

### Prerequisites

- Docker Engine 24+ (or Docker Desktop on macOS/Windows).
- For GPU training: an NVIDIA GPU with current drivers on the host
  and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  installed (`nvidia-ctk runtime configure --runtime=docker` and a
  `systemctl restart docker`). PyTorch wheels bundle their own CUDA
  runtime, so you don't need a CUDA SDK on the host — just the
  driver and the container toolkit.

### Build the image

```bash
git clone https://github.com/jdinalt/forgather.git
cd forgather
docker/build.sh
```

`docker/build.sh` reads your host UID, GID, and username (`id -u`,
`id -g`, `id -un`) and passes them as build args, so the image
carries an account that matches your host user. Files created
inside the container on a bind-mounted home land with correct
ownership on the host.

The first build pulls ~3 GB of dependencies and takes a few minutes;
rebuilds reuse the layer cache.

### Run it

```bash
docker/run.sh
```

This drops you into an interactive bash shell with:

- The Forgather venv (at `/opt/forgather/venv`) on `PATH`.
- `--gpus all` (override with `GPUS=none` for CPU only or
  `GPUS='"device=0,1"'` for a subset).
- Your host home directory bind-mounted at the same path inside the
  container, so absolute paths in shell history, configs, and
  notebooks keep resolving correctly.
- The canonical Forgather server / job ports (8765, 8137, 6006,
  8000) forwarded to the host's loopback so you can reach the web
  UI from a browser on the host.

The container's entrypoint detects the bind-mounted Forgather
checkout and re-links the editable install to it on entry, so your
host-side edits are picked up live without a rebuild.

```bash
# Inside the container:
forgather ls -r
cd examples/tutorials/tiny_llama
forgather -t v2.yaml train
```

### Container lifecycle

The container is long-lived: the first `docker/run.sh` invocation
creates a detached container named `forgather-dev-${USER}` with
`sleep infinity` as PID 1; subsequent invocations re-attach via
`docker exec`. Logging out of an interactive shell does **not**
stop the container, so a `forgather server` (or any training job)
you started in one session keeps running, and you can re-attach
from a new terminal to inspect or control it.

```bash
docker/run.sh                   # attach (creating the container if needed)
docker/run.sh forgather ls -r   # one-shot command in the same container
docker/run.sh --status          # is the container running, stopped, or absent?
docker/run.sh --stop            # stop (but keep) — preserves filesystem state
docker/run.sh --rm              # stop and remove (next run.sh recreates fresh)
docker/run.sh --recreate        # rebuild from scratch (e.g. after image rebuild)
```

`IMAGE`, `GPUS`, `NETWORK`, port and mount overrides only apply
when the container is **created**. After `docker/build.sh`
rebuilds the image, run `docker/run.sh --recreate` to roll the
running container forward to the new image.

If you'd rather drive `docker` directly:

```bash
NAME=forgather-dev-$USER
docker ps -a --filter name=${NAME}        # see the container, running or not
docker logs ${NAME}                       # entrypoint output (install re-link warnings)
docker stop ${NAME}                       # stop
docker start ${NAME}                      # start an existing stopped container
docker rm -f ${NAME}                      # stop and remove
```

For force-rebuilding after pulling repo changes:

```bash
docker/build.sh -- --no-cache
docker/run.sh --recreate
```

### Networking

`docker/run.sh` defaults to `--network host`, so the container
shares the host's network stack. Every service inside the
container is reachable on its bound port without `-p` mappings,
and tools that default to `127.0.0.1` (Forgather server, MkDocs,
TensorBoard, inference) Just Work — open
<http://localhost:8765/> from the host browser as if Forgather
were running on bare metal.

If you'd rather use bridge networking with explicit port-forwards
(slightly more isolated, but every service then has to bind
`0.0.0.0` inside the container to be reachable through the
forward), set `NETWORK=bridge`:

```bash
NETWORK=bridge docker/run.sh
# Inside the container:
forgather server -H 0.0.0.0
mkdocs serve --host 0.0.0.0
tensorboard --bind_all
```

The bridge mode forwards the host side to `127.0.0.1` only by
default (same exposure as the host-networking case). For LAN
access from another machine, set `HOST_BIND=0.0.0.0` alongside
`NETWORK=bridge`.

### Common overrides

```bash
# CPU-only:
GPUS=none docker/run.sh

# Specific GPUs:
GPUS='"device=0,1"' docker/run.sh

# Mount additional host paths (e.g. scratch / dataset volumes):
EXTRA_MOUNTS="-v /scratch:/scratch" docker/run.sh

# Forward extra ports (Vite dev server, etc.):
EXTRA_PORTS="-p 5173:5173" docker/run.sh

# Build / run a tagged variant:
docker/build.sh forgather-dev:experiment
IMAGE=forgather-dev:experiment docker/run.sh
```

For more detail — including the release-testing workflow that uses
the in-image copy of the repo without a bind-mount — see the
[Docker Development Image](../development/docker.md) page.

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

## Running the Forgather server

`forgather server` launches a local web UI that wraps the same APIs
the CLI uses — project / config browsing, queued training runs, GPU
monitoring, log viewing, an in-browser editor for templates and
arbitrary text files, and a chat client against served models. The
server is single-user, localhost-first; it binds to `127.0.0.1` by
default and ships with no auth.

### Build the web UI

The web UI is a Vite/React SPA and isn't pre-built into the repo.
Before starting the server, build the dist bundle:

```bash
cd tools/forgather_server/webui
npm install          # one-time, fetches Vite + React + Monaco + viz-js
npm run build        # produces webui/dist/
```

This needs Node.js + npm installed (see Prerequisites). `npm install`
takes a couple of minutes on first run; the build itself is fast.
The output is a static `dist/` directory the running server serves
directly — no Node process at runtime. Re-run `npm run build` after
pulling changes that touch `webui/src/`.

If you start the server before `webui/dist/` exists, the API
endpoints still work but the root URL returns **404 Not Found** —
build the UI first, or run the Vite dev server (see "Dev mode"
below).

### Starting the server

```bash
forgather server
```

Defaults to `http://127.0.0.1:8765/`. Open that URL; the sidebar's
six collapsible groups (Views, Tools, Search Roots, Projects, Files)
are all closed on first boot — expand the ones you want.

Common options:

```bash
forgather server -H 127.0.0.1 -p 8765 -l INFO     # custom bind / verbosity
CUDA_VISIBLE_DEVICES=0,1,3 forgather server        # exclude specific GPUs
                                                    # from the scheduler pool
```

The scheduler is enabled by default — submitted jobs start dispatching
immediately. Pause anytime with the ⏸ button in the sidebar header.

### Accessing a remote server over SSH

If your development box is remote, use SSH local-port forwarding so
the browser on your laptop can reach the server's localhost ports.
The Forgather server itself listens on **8765**; the services it
spawns each pick a canonical default port so existing port-forward
configs keep working without per-host rebinds:

| Service                  | Default port |
| ------------------------ | ------------ |
| Forgather server (UI)    | **8765**     |
| Inference server jobs    | **8137**     |
| TensorBoard jobs         | **6006**     |
| MkDocs jobs              | **8000**     |

```bash
ssh -L 8765:localhost:8765 \
    -L 8137:localhost:8137 \
    -L 6006:localhost:6006 \
    -L 8000:localhost:8000 \
    user@dev-host
```

Or persist the forwards in `~/.ssh/config` so you don't have to
remember the ports:

```
Host dev-host
    HostName dev-host.example.com
    User you
    LocalForward 8765 localhost:8765
    LocalForward 8137 localhost:8137
    LocalForward 6006 localhost:6006
    LocalForward 8000 localhost:8000
```

Then `ssh dev-host` and open <http://localhost:8765/> on your
laptop. Inference / TensorBoard / MkDocs jobs surface their served
URLs as clickable links on the running-job card; with the forwards
in place those links resolve transparently.

> **Heads-up: prefer `localhost` over `127.0.0.1`** when the
> browser is reaching the server through an SSH tunnel. They're
> not always interchangeable on the client. Chromebook in
> particular routes `localhost` through the tunnel correctly but
> hits its own loopback for `127.0.0.1`, which fails to connect.
> macOS and most Linux desktops treat the two identically, but
> `localhost` is the safer default for tunneled access.

### Persistent state

Per-user state lives under `~/.forgather/server/`: search roots,
queue, job records, captured TTY logs, dynamic-args overrides,
GPU policy. All files are written crash-atomically (tmp + fsync +
rename). Power-loss-mid-write never leaves a half-written canonical
file, and every reader tolerates a corrupt or truncated file by
falling back to empty state.

### Dev mode (hot reload)

If you're modifying `webui/src/`, run the Vite dev server alongside
the Python backend:

```bash
# Terminal 1 — API backend
forgather server -p 8765

# Terminal 2 — Vite dev server (hot reload, proxies /api → :8765)
cd tools/forgather_server/webui
npm run dev          # opens http://localhost:5173
```

For an end-to-end tour of the UI — install through training a small
model and chatting with it — see the
[Forgather server walkthrough](../guides/forgather-server-walkthrough.md).
For the full feature reference and API documentation, see the
[Forgather server README](../forgather-server.md).

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
