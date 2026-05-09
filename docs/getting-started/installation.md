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
docker/build.sh                  # generic image; works for any host user
docker/run.sh                    # interactive shell, --gpus all, ports forwarded

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
  checked in, so you build it once after install — see
  [Running the Forgather server](README.md#running-the-forgather-server).
  Any current LTS Node release works (tested on Node 20).
  ```bash
  sudo apt-get install nodejs npm
  ```
  None of this is needed if you only use the CLI; the running server
  itself has no Node dependency once the dist bundle exists.

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

There's also a separate **runtime image** (`Dockerfile.runtime`)
intended for distribution to a multi-node cluster — generic, no
host-clone dependency, builds the SPA inside the image. The
[Docker images](docker.md) reference covers both.

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

`docker/build.sh` builds a single generic image — there are no
per-user build args. The image carries a fixed in-container user at
UID/GID 1000; at container start, the entrypoint reads `PUID`/`PGID`
env vars (forwarded by `docker/run.sh` from your host's `id -u` /
`id -g`) and remaps the in-container user via `gosu`. Files created
inside the container on a bind-mounted home land with correct
ownership on the host, and the same image works for any operator
without rebuilding. See [Docker images](docker.md) for the full
PUID/PGID remap rationale.

The first build pulls ~3 GB of dependencies and takes a few minutes;
rebuilds reuse the layer cache. After the docker build, `build.sh`
runs `./build-webui.sh` in a transient container against the host
clone so the Forgather server's SPA dist/ is ready before
`docker/run.sh` is invoked. Skip the post-step with
`SKIP_WEBUI_BUILD=1 docker/build.sh` (e.g. you'll iterate on the
SPA via `npm run dev`).

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
- The host's network stack (`--network host`) so services bound to
  `127.0.0.1` inside the container are reachable on the host's
  loopback as-is.

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

For more detail — full CLI / env-var reference, the runtime
(distributable) image, multi-node setup, and the release-testing
workflow against a freshly cloned tree — see the [Docker images](docker.md)
reference.

## Next: your first training run

With Forgather installed, head to
[Getting Started → Your first training run](README.md#your-first-training-run)
to train a tiny Llama on TinyStories.
