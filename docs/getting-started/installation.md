# Installation

**TL;DR**

```bash
# If running remotely over ssh,
# setup port forwarding
ssh -L 8765:localhost:8765 \
    -L 8137:localhost:8137 \
    -L 6006:localhost:6006 \
    -L 8000:localhost:8000 \
    user@dev-host

# Install with Docker
git clone https://github.com/jdinalt/forgather.git
cd forgather
docker/build                  # per-user dev image, bakes your host UID/GID in
docker/run                    # interactive shell, --gpus all, ports forwarded

# Inside the container:

# Start the webui...
forgather server

# control-click on `http://localhost:8765/?token=4c4febdc07830cdd...` to connect with your browser

# ...or use the CLI
forgather --help
cd examples/tutorials/tiny_llama
forgather -t v2.yaml train
```

Two paths: install on the host directly (Python venv via `pip` or
`uv`), or run inside the bundled Docker development image. Pick
whichever fits your machine.

> **Want to skip the host setup?** Forgather ships a development
> Dockerfile that provisions Python 3.12, PyTorch with CUDA wheels,
> all dependencies, and a developer-friendly base toolchain in a
> reproducible image. Jump to [**Installing with Docker**](#installing-with-docker)
> below.

After installing, head back to [Getting Started](README.md) for the
first-training-run walkthrough, CLI reference, and the Forgather
server tour.

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
  checked in, so you build it once after install via
  `./build-webui.sh` at the repo root — see
  [Running the Forgather server](README.md#running-the-forgather-server).
  Any current LTS Node release works (tested on Node 20).
  ```bash
  sudo apt-get install nodejs npm
  ```
  None of this is needed if you only use the CLI; the running server
  itself has no Node dependency once the dist bundle exists. On a
  checkout shared between hosts of different platform (e.g. an NFS
  share spanning x86_64 and aarch64), always invoke `./build-webui.sh`
  — `node_modules/` is platform-specific and the script keeps each
  platform's install in its own sibling directory.

## Host installation (pip / uv)

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

**Heads-up: TensorBoard + setuptools 82 incompatibility.** TensorBoard
≤ 2.20.0 (the latest release as of writing) imports `pkg_resources`
at module load, but setuptools 82 (Feb 2026) removed `pkg_resources`
entirely. If your environment ends up with setuptools ≥ 82 you'll
hit `ModuleNotFoundError: No module named 'pkg_resources'` the first
time you run `tensorboard` or `forgather tb`. The fix is on
TensorBoard master ([PR #7057](https://github.com/tensorflow/tensorboard/pull/7057),
March 2026) but not in any release yet. Two workarounds:

```bash
# Option 1 — pin setuptools below 82 (most common):
pip install "setuptools<82"

# Option 2 — backport the upstream fix in-place against your installed
# tensorboard. The Docker image takes this path; the patch script is
# idempotent, fails loudly if the pre-patch text has moved, and is
# safe to remove once tensorboard ships a fixed release. From the
# Forgather repo:
python docker/patches/fix_tensorboard_pkg_resources.py
```

Drop either workaround once Forgather pins a TensorBoard release
that contains the upstream fix.

Verify the installation:

```bash
forgather ls -r
```

This recursively lists all Forgather projects and configurations found under the
current directory. You should see output listing the bundled example projects.

## Installing with Docker

> **Looking for the full reference?** See [**Docker images**](docker.md)
> for the comprehensive guide — every CLI flag and env var on the
> `build.sh` / `run.sh` helpers, the runtime (distributable) image
> for clusters, multi-node setup, persistent overrides, and
> troubleshooting. The section below is the install quick-start; the
> reference page is where to go to customize things or understand
> how it works.

The repo ships a `Dockerfile` (and `docker/` helper scripts) that
builds an Ubuntu 24.04 image with the full Forgather environment
pre-provisioned (Python 3.12, PyTorch with CUDA wheels, all deps,
`cut-cross-entropy` from source, a developer toolchain). There's
also a distributable **runtime image** (`Dockerfile.runtime`) for
multi-node clusters. The [Docker images](docker.md) reference is the
full guide — every flag and env var, the runtime image, multi-node
setup, persistent overrides, container lifecycle, networking, and
troubleshooting.

### Docker prerequisites

- Docker Engine 24+ (or Docker Desktop on macOS/Windows).
- For GPU training: an NVIDIA GPU with current drivers on the host
  and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  installed (`nvidia-ctk runtime configure --runtime=docker` then
  `systemctl restart docker`). PyTorch wheels bundle their own CUDA
  runtime, so you don't need a CUDA SDK on the host — just the
  driver and the container toolkit.

### Quick start

```bash
git clone https://github.com/jdinalt/forgather.git
cd forgather
docker/build               # per-user CUDA dev image (~3 GB); bakes in your host UID/GID
docker/build --cpu         # CPU-only build (~1.5 GB) for GPU-less hosts — pair with GPUS=none
docker/run                 # interactive shell: --gpus all, $HOME bind-mounted, host networking
GPUS=none docker/run       # hosts with no NVIDIA driver (must use this)

# Inside the container:
forgather ls -r
cd examples/tutorials/tiny_llama
forgather -t v2.yaml train
```

`docker/build` runs `./build-webui.sh` automatically so the server's
SPA is ready; the container is long-lived (re-attach with `docker/run`,
roll forward after a rebuild with `docker/run --recreate`). `--cpu`
builds with the CPU-only PyTorch wheel from `download.pytorch.org/whl/cpu`
(CLI/config sanity-checks only — CPU training is orders of magnitude
slower). For `GPUS` / `NETWORK` / `EXTRA_MOUNTS` overrides, the
container lifecycle, networking, and the runtime image, see the
[Docker images](docker.md) reference.

## Next: your first training run

With Forgather installed, head to
[Getting Started → Your first training run](README.md#your-first-training-run)
to train a tiny Llama on TinyStories.
