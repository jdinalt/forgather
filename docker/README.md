# Forgather Docker images

Two images, distinct roles:

| | **Dev image** (`Dockerfile`) | **Runtime image** (`Dockerfile.runtime`) |
| - | - | - |
| **Audience** | Forgather developers, release testing | Operators, end users, cluster deployments |
| **Source code** | Bind-mounted from your host clone | Cloned from git at build time, baked into `/opt/forgather/repo` |
| **Default command** | `bash -l` | `forgather server -H 0.0.0.0 -p 8765` |
| **Mutability** | Mutable (host-clone bind-mount, edits go live) | **Immutable by design** (build once, distribute identical) |
| **Networking default** | `--network host` (Linux only) | Bridge with `-p 8765:8765` (portable) |
| **Multi-node** | `--network host` works out of the box | `NETWORK=host` opt-in required |
| **Distributable** | Yes — PUID/PGID remap means one image works for any host user | Yes |

After [recent consolidation work](#consolidation), both images share the
same user-identity pattern (PUID/PGID remap via `gosu` at container
start), the same entrypoint script (`docker/entrypoint.sh`), and a
shared shell library for run-script scaffolding (`docker/_lib.sh`).
The differences in the table above are deliberate, not historical
duplication.

**Which to pick:**

- Hacking on Forgather — **dev image**.
- Running Forgather as a server on your machine for actual training,
  with no plan to modify the source — **runtime image**.
- Distributing a fixed Forgather build to a multi-machine cluster —
  **runtime image**, definitively. Build once, push, run on N nodes.
- Iterating on the Docker tooling itself — both, but most changes
  start with the dev image.

---

## Quick start

```bash
# Dev image:
docker/build.sh                   # build forgather-dev:latest
docker/run.sh                     # interactive shell, repo bind-mounted

# Runtime image:
docker/runtime/build.sh           # build forgather:latest
docker/runtime/run.sh             # starts the server, prints clickable URL
```

The first build of either image pulls ~3 GB of dependencies (PyTorch
and friends); rebuilds reuse the layer cache.

For the dev image, you'll land in a bash shell at the repo path with
the venv on `PATH`, GPU access (`--gpus all`), and host networking.
`forgather ls -r` works out of the box.

For the runtime image, the script waits for the server to write its
auth token, then prints `http://127.0.0.1:8765/?token=<token>` —
open that in a browser to land in a logged-in session.

---

## Shared concerns

Topics that apply to both images.

### User identity (PUID/PGID remap)

Both images ship with a fixed in-container user at UID/GID 1000. At
container start, the entrypoint reads `PUID` / `PGID` env vars
(defaulting to 1000:1000), `usermod`s the in-container **uid only**
(see below), chowns the small in-image home, then drops privileges
via `gosu` before exec'ing the real command.

The run scripts forward `PUID=$(id -u)` and `PGID=$(id -g)` from the
calling shell automatically, so files written from inside the
container land on bind-mounted host paths with host-correct ownership
— without rebuilding the image. One image, any user.

If you launch with `docker run --user $(id -u):$(id -g)` (rootless
podman, container-with-no-root scenarios), the entrypoint detects it
isn't running as root and skips the remap entirely.

#### Why only the uid is remapped (and the gid stays at 1000)

The in-image venv at `/opt/forgather/venv` is built with files
owned by uid 1000 / gid 1000, with `chmod -R go+rwX` applied so
group + other have read/write/execute on directories and
read/write on files. At runtime the entrypoint changes the
in-container user's **uid** to PUID but leaves the primary **gid**
at 1000. That keeps the venv group-writable for the remapped user
without any recursive chown — cold-start is ~2 seconds even when
host UID != 1000 (an earlier version did `chown -R /opt/forgather`
on every container start with a different UID, which ran over
thousands of files and added tens of seconds).

This implicitly assumes **gid 1000 inside the container has no
load-bearing meaning on your host**. On a typical
single-user Linux box the host's gid 1000 is just the first
interactive user's primary group — files created in your
bind-mounted home will land with gid 1000 on the host side, which
is fine if you're the only user. On a shared host where gid 1000
belongs to a different user / service, you may want to inspect
ownership of files written from the container before assuming the
default is right; ACLs or a different bind-mount strategy can fix
it if needed.

### GPUs

```bash
# CPU-only:
GPUS=none docker/run.sh
GPUS=none docker/runtime/run.sh

# Specific GPUs:
GPUS='"device=0,1"' docker/run.sh
GPUS='"device=0,1"' docker/runtime/run.sh
```

Both run scripts default to `--gpus all`. The unified entrypoint
runs a one-line `nvidia-smi` probe at container start: prints
`nvidia-smi: driver=<ver>, N device(s) visible` on success, warns
when `nvidia-smi` is missing or reports zero devices. Non-fatal —
operators run CPU-only sometimes — but loud enough that an obvious
GPU misconfiguration shows up immediately.

### Networking

The dev image defaults to `--network host` (Linux only); the runtime
image defaults to bridge with `-p ${HOST_BIND}:${PORT}:8765`. Both
support flipping via `NETWORK=host` / `NETWORK=bridge`.

**For multi-node operation**, set `NETWORK=host` on both images.
Forgather's cluster discovery uses mDNS, which depends on multicast
that doesn't traverse Docker bridge networks. See
[`docs/guides/multi-node-training.md`](../docs/guides/multi-node-training.md)
for the full multi-node setup.

### Persistent state

Forgather's per-user state lives at `~/.forgather/` inside the
container — auth token, queue index, GPU policy, generation configs,
hardware FLOPS cache, cluster node id (if multi-node). The two
images get there differently:

- **Dev image** bind-mounts `$HOME` wholesale, so `~/.forgather/`
  inside the container *is* the host's `~/.forgather/`.
- **Runtime image** mounts a docker-managed named volume
  `forgather-state` at `/home/forgather/.forgather/`, isolated from
  the host filesystem (preferred for release deployments).

To make the runtime image read/write the same on-disk state as the
dev image (useful when iterating between the two), point the
runtime's `STATE_VOLUME` at the host path:

```bash
STATE_VOLUME=$HOME/.forgather docker/runtime/run.sh
```

To opt out of state persistence on the runtime image (ephemeral —
fresh auth token on every recreate), set `STATE_VOLUME=` (empty).

### Persistent overrides

Both run scripts source `~/.config/forgather/docker.env` (or the path
in `$FORGATHER_DOCKER_CONFIG`) before applying defaults. Use `:=`
so command-line vars still win:

```bash
# ~/.config/forgather/docker.env
: "${EXTRA_MOUNTS:=-v /mnt/rust:/mnt/rust}"
: "${GPUS:=all}"
: "${NETWORK:=host}"
```

### Lifecycle commands

Both run scripts share these subcommands (handled by the shared
`docker/_lib.sh`):

| Command | Effect |
| --- | --- |
| (no arg) | create or attach (start if stopped) |
| `--status` | state + image + network info |
| `--stop` | stop, keep filesystem |
| `--rm` | stop + remove |
| `-h` / `--help` | usage from the script's header docstring |

Image-specific subcommands stay per-image (dev: `--recreate`;
runtime: `--logs`, `--shell`, `--token`, `--recreate`, `--dev`).

### Container is long-lived

Both run scripts create a detached container; subsequent invocations
re-attach via `docker exec`. Logging out of an interactive shell does
not stop the container, so `forgather server` (or any training job)
started in one session keeps running. Re-attach from a new terminal
to inspect or control it.

When the container already exists, env-var overrides for
`IMAGE` / `GPUS` / `NETWORK` / port / mount are **ignored on
re-attach** — those bake at create time. Use `--recreate` to pick up
changes after `docker/build.sh` rebuilt the image (or the runtime's
`docker/runtime/run.sh --recreate`).

### Equivalent raw `docker` commands

The helper scripts are thin wrappers — drop straight to `docker` if
you'd rather:

```bash
NAME=forgather-dev-$USER          # or 'forgather-server' for runtime
docker ps -a --filter "name=${NAME}"
docker logs ${NAME}
docker stop ${NAME}
docker start ${NAME}
docker restart ${NAME}
docker rm -f ${NAME}
docker exec -it ${NAME} bash -l
```

`docker logs` is particularly useful when something goes wrong at
container start — entrypoint output (the `nvidia-smi` probe, the
editable-install re-link on the dev image, etc.) prints there, not
into your interactive shell.

### Container init (zombie reaping)

Both images run with Docker's `--init` flag, which puts `tini`
in front of the entrypoint as PID 1. Without this, when torchrun
gets killed and its worker subprocesses get re-parented to PID 1
(= sleep, on the dev image), nobody calls `wait()` on them and they
pile up as zombies. `tini` reaps orphans regardless of parentage —
the only layer that can see grandchildren of the Forgather server.

This bit operators on the multi-node cluster after a hung save-stop;
see [`docs/guides/multi-node-training.md`](../docs/guides/multi-node-training.md)
for the full story.

---

## Dev image — specifics

### Layout

| File | Purpose |
| ---- | ------- |
| `../Dockerfile` | Image definition |
| `../.dockerignore` | Build-context filter |
| `build.sh` | Builds the image (no per-user args needed) |
| `run.sh` | Launches a long-lived container with `$HOME` bind-mounted |
| `entrypoint.sh` | **Shared with runtime image** — venv setup, PUID/PGID remap, `nvidia-smi` probe, editable-install when `FORGATHER_REPO` is set |
| `_lib.sh` | **Shared with runtime image** — common run-script scaffold |

### Editable install against your host clone

The venv at `/opt/forgather/venv` carries every Forgather dependency
but **not** the Forgather package itself. `run.sh` sets
`FORGATHER_REPO` to your host-side checkout's path; the entrypoint
installs Forgather in editable mode against that tree on first start
(and re-runs the install if you point it at a different checkout).

Your edits show up immediately without a rebuild. There is no
in-image copy of the repo to drift, mirror, or chown.

If `FORGATHER_REPO` is unset (or doesn't point at a Forgather
checkout) the entrypoint prints a warning — the venv is still usable
for arbitrary Python work, but the `forgather` command won't be
available until you install the package against a real source tree.

### Web UI bundle (build on the host)

The dev image does **not** prebuild the SPA. The bundle is
checkout-local: it lives at
`tools/forgather_server/webui/dist/` inside your host clone, where
the FastAPI app finds it at runtime. Build it once before starting
the Forgather server:

```bash
# On the host (or inside the container — same checkout, same result):
cd "$FORGATHER_REPO" && ./build-webui.sh
```

`docker/build.sh` runs `./build-webui.sh` automatically as a post-
step against your host clone (`SKIP_WEBUI_BUILD=1` to skip when you
plan to use Vite hot-reload via `npm run dev` instead).

The entrypoint prints a one-line reminder when `webui/dist/` is
missing.

### Bundled developer tools

Beyond the venv + base CLI tools (vim, tmux, ripgrep, jq, htop, ssh,
sudo, ...), the dev image bakes in:

- `gh` (GitHub CLI) — for `gh pr`, `gh repo`, `gh auth login` from
  inside the container without re-installing on every rebuild.

Optional, opt-in at build time:

- **Claude Code** (`@anthropic-ai/claude-code`) — pass `--claude`
  to `docker/build.sh` to install it globally via npm. Lands at
  `/usr/bin/claude`, world-executable so the gosu-dropped runtime
  user can invoke it. Off by default; the average operator
  doesn't need it baked in.

  Note that if you already have Claude Code installed in your
  host's `~/.local/bin/` or via npm under `~/`, the dev image's
  bind-mounted `$HOME` makes that install available inside the
  container — so most developers won't need `--claude` either.
  It's a convenience for users who don't have a host install.

```bash
# Build without Claude Code (default):
docker/build.sh

# Build with Claude Code baked in:
docker/build.sh --claude

# Combine with a custom tag and docker passthrough:
docker/build.sh forgather-dev:claude --claude
docker/build.sh --claude -- --no-cache
```

### Common overrides

```bash
# Bind-mount additional host paths:
EXTRA_MOUNTS="-v /scratch:/scratch -v /data:/data" docker/run.sh

# Forward extra ports (Vite dev server, ...):
EXTRA_PORTS="-p 5173:5173" docker/run.sh

# Tag and run a different build:
docker/build.sh forgather-dev:experiment
IMAGE=forgather-dev:experiment docker/run.sh
```

### Cross-device symlinks

`run.sh` only bind-mounts `$HOME`. If anything under your home is a
symlink whose target lives on a different filesystem (a RAID volume,
a separate `/data` mount, etc.), the symlink is visible inside the
container but its target isn't — every dereference dangles. Common
pattern:

```
~/ai_assets/forgather -> /home/dinalt/rust/forgather
/home/dinalt/rust     -> /mnt/rust/home/dinalt/rust    # RAID
```

Inside the container `/mnt/rust` doesn't exist, so the link breaks.
Bind-mount the underlying mountpoint at the same path so symlinks
resolve identically:

```bash
EXTRA_MOUNTS="-v /mnt/rust:/mnt/rust" docker/run.sh --recreate
```

Use `--recreate` — mount config is fixed at container creation, not
on `docker exec`.

`run.sh` validates this at create-time:

- **Fatal (exit 2)** if the forgather repo path itself resolves
  through a symlink to an uncovered location. Without a bind-mount
  Docker fails with a confusing `mkdir: file exists` OCI error;
  bailing early gives a clear suggested `EXTRA_MOUNTS` line.
- **Warning** for any other `$HOME`-rooted symlink whose target is
  uncovered. Non-fatal — those only matter if you actually
  dereference them inside the container.

### Release-testing workflow

Use the dev image as a clean sandbox by building with `--no-cache`
and bind-mounting a freshly cloned tree:

```bash
docker/build.sh forgather-dev:release-test -- --no-cache

# In a clean directory:
git clone https://github.com/jdinalt/forgather.git fresh-forgather
cd fresh-forgather
IMAGE=forgather-dev:release-test docker/run.sh -- bash -lc \
    "forgather ls -r && \
     cd examples/tutorials/tiny_llama && \
     forgather -t v2.yaml train"
```

`--no-cache` verifies the Dockerfile and dependency graph from
scratch; the fresh clone verifies the source tree itself runs
end-to-end. Together that's exactly what an end user gets from
a fresh `pip install -e .`.

---

## Runtime image — specifics

### Design philosophy: immutable by design

The runtime image is **intended to be light-weight, identical across
a distribution**. The supported deployment model is:

1. Develop in the dev image.
2. Bake a commit and push it.
3. `docker/runtime/build.sh` once.
4. Distribute the image (via registry push, `docker save`, etc.).
5. Run identical copies on N nodes.

This avoids redundant downloads and ensures every node runs the same
"everything" — torch wheels, tokenizers, generated kernels, the
forgather code itself. Mutating a runtime container in production
breaks this contract.

The image enforces this by not bundling any in-container build tools
for the SPA, keeping `/opt/forgather/repo` install-time-static, and
documenting the immutability contract clearly. **The `--dev` opt-in
below is a debugging affordance, not the workflow.**

### Layout

| File | Purpose |
| ---- | ------- |
| `../Dockerfile.runtime` | Image definition |
| `runtime/build.sh` | Builds the image |
| `runtime/run.sh` | Launches a server container, prints auth-token URL |
| `../entrypoint.sh` | **Shared with dev image** — venv setup, PUID/PGID remap, `nvidia-smi` probe; editable-install branch is no-op when `FORGATHER_REPO` is unset |
| `../_lib.sh` | **Shared with dev image** — common run-script scaffold |

### Source tree comes from `git`, not from your local checkout

`Dockerfile.runtime` clones from `FORGATHER_GIT_URL` at the ref
`FORGATHER_GIT_REF` (default `dev` — moves to `main` once a stable
release ships with this docker tooling). That keeps the build
reproducible and decoupled from whatever stray state happens to sit
in the publisher's working directory.

```bash
# Pin a release tag:
FORGATHER_GIT_REF=v1.0.0 docker/runtime/build.sh

# Iterate on an unmerged branch:
FORGATHER_GIT_REF=feature/my-change docker/runtime/build.sh
```

For air-gapped builds (offline CI, isolated lab):

```bash
docker build -t forgather:offline \
    --build-arg FORGATHER_SOURCE_DIR=. \
    -f Dockerfile.runtime .
```

`FORGATHER_SOURCE_DIR` (default empty) tells the Dockerfile to `cp`
the source from inside the build context instead of running `git
clone`. `docker/runtime/build.sh` does not currently thread this arg
through; invoke `docker build` directly when needed.

### Volumes

`docker/runtime/run.sh` is **conservative about exposing your host
filesystem**. By default it mounts only one thing, and that thing is
a docker-managed named volume — no host paths at all:

| Source | Container | Purpose | Default? |
| - | - | - | - |
| `forgather-state` (named volume) | `/home/forgather/.forgather` | Server state (auth token, queue, GPU policy, ...) | ✓ enabled |
| `$HF_CACHE_HOST` (host path) | `/home/forgather/.cache/huggingface` | Bind-mount, share HF cache with host install | opt-in |
| `$EXTRA_MOUNTS` (free-form) | wherever you say | scratch, data, output dirs, ... | opt-in |

The state volume keeps the auth token across `docker rm`. Reset by
`docker volume rm forgather-state`, or set `STATE_VOLUME=` (empty)
to opt out entirely.

```bash
# Share HF cache with host:
HF_CACHE_HOST=$HOME/.cache/huggingface docker/runtime/run.sh

# Share state with the dev image (see "Persistent state" above):
STATE_VOLUME=$HOME/.forgather docker/runtime/run.sh
```

### Multi-node operation

Two env vars compose the cluster CMD:

```bash
# Single-node (default — bridge with port forward):
docker/runtime/run.sh

# Cluster mode — mDNS multicast needs host networking:
NETWORK=host CLUSTER=lab docker/runtime/run.sh

# Cluster with explicit advertised address (useful inside a
# container without --network host, or behind NAT):
NETWORK=host CLUSTER=lab CLUSTER_ADDRESS=192.168.1.27 \
    docker/runtime/run.sh
```

`CLUSTER=<name>` causes the run script to append
`--cluster <name>` to the server CMD. `CLUSTER_ADDRESS=<ip>`
adds `--cluster-address <ip>`. The script warns loudly when
`CLUSTER` is set without `NETWORK=host`, since bridge networking
breaks mDNS discovery.

For the broader multi-node setup (peer discovery, distributed-job
launching, hang diagnosis), see
[`docs/guides/multi-node-training.md`](../../docs/guides/multi-node-training.md).

### Healthcheck

The image declares a Docker `HEALTHCHECK` that probes
`http://127.0.0.1:8765/api/cluster/self` every 30 seconds (5s
timeout, 20s start-period grace, 3 retries). The endpoint returns
200 in both standalone and cluster modes, so a passing check means
FastAPI is up and serving.

```bash
docker inspect --format '{{.State.Health.Status}}' forgather-server
```

Orchestration layers (compose, swarm, k8s readiness probes) can use
this to gate traffic or trigger restarts. Works the same under
bridge networking and `NETWORK=host`.

### Common overrides

```bash
# CPU-only:
GPUS=none docker/runtime/run.sh

# Specific GPUs:
GPUS='"device=0,1"' docker/runtime/run.sh

# Expose on the LAN (default is loopback only; auth token still gates):
HOST_BIND=0.0.0.0 docker/runtime/run.sh

# Use a different port on the host:
PORT=8888 docker/runtime/run.sh

# Override the image (e.g. for a versioned release):
IMAGE=ghcr.io/jdinalt/forgather:1.1.0 docker/runtime/run.sh

# Forward extra ports — e.g. tensorboard from a Forgather job:
EXTRA_PORTS="-p 6006:6006" docker/runtime/run.sh
```

### Diagnostic shell

```bash
docker/runtime/run.sh --shell
# or:
docker exec -u forgather -ti forgather-server bash
```

The diagnostic shell has the venv on `PATH`, so `forgather`,
`python`, and the rest of the CLI work as expected. Useful for
`forgather control list`, `forgather logs summary`, and ad-hoc
Python work.

### `--dev`: testing fixes without rebuilding (debug only)

The runtime image is intended to be **immutable and identical
across a distribution** (see [Design philosophy](#design-philosophy-immutable-by-design)
above). If you've found a bug in a deployed runtime image and want to
test a fix without going through a full rebuild + redistribute cycle,
`docker/runtime/run.sh` accepts a `--dev` flag (or `DEV=1` env var)
that bind-mounts a host-side forgather clone over the image's baked-in
`/opt/forgather/repo`. Because the image installs forgather editable
from that path, host-side edits go live the next container restart.

```bash
# Use the script's own repo root (works when you run from the
# clone you want to test):
docker/runtime/run.sh --dev --recreate

# Or point at a specific clone:
docker/runtime/run.sh --dev /home/me/forgather-fork --recreate

# Equivalent via env var:
DEV=1 docker/runtime/run.sh --recreate
DEV=/home/me/forgather-fork docker/runtime/run.sh --recreate
```

The script prints a prominent multi-line WARNING when `--dev` is
active so it's obvious in the operator's terminal that the container
is off the golden path. **Please rebuild the image for production
deployment; do not ship a runtime image that depends on a host-side
clone.**

### Distributing the image

Tag and push as usual:

```bash
docker tag forgather:latest ghcr.io/jdinalt/forgather:1.1.0
docker push ghcr.io/jdinalt/forgather:1.1.0
```

Multi-arch builds (`linux/arm64`) are out of scope for this
Dockerfile; if you need them, drive `docker buildx build --platform
linux/amd64,linux/arm64` against `Dockerfile.runtime` directly.

---

## Troubleshooting

**Server won't start, `docker logs` shows a permission error.**
If you opted into a host-path bind-mount and the host directory
contains files owned by a different user than the one running the
script, the in-container forgather user (remapped to your host UID)
won't be able to write. The entrypoint never chowns bind-mounted
host paths — that would be slow and pointless on populated caches.
Either chown the host directory yourself or point the env var at a
different writable directory.

**Webui shows "missing dist" warning at start.**
Dev image only — the runtime image bakes the SPA at image build.
On the dev image, run `./build-webui.sh` from your host clone (or
inside the container against the bind-mounted repo).

**Auth token rotates on every restart.**
Runtime image: the token only persists if `~/.forgather/` is on a
persistent volume. By default `docker/runtime/run.sh` mounts the
named volume `forgather-state`; if you `docker volume rm` that
volume between runs, the token is regenerated. Dev image: token
lives on the bind-mounted host home, so it persists across
container recreate.

**Different host user wants to use the same image.**
Both images now support this — that's the whole point of the
PUID/PGID remap pattern. They run `docker/run.sh` (dev) or
`docker/runtime/run.sh` (runtime) from their account; the script
forwards their UID automatically. No rebuild needed.

**Multi-node hang or "no peer discovery."**
mDNS doesn't traverse Docker bridge networks. Set `NETWORK=host`
on every node and recreate. Also check
[`docs/guides/multi-node-training.md`](../docs/guides/multi-node-training.md)
for the full troubleshooting cookbook including faulthandler / SIGUSR1
live-stack-dump.

**Tensorboard fails on first start in a fresh image.**
The image build applies a backport patch to fix
TensorBoard ≤2.20's reliance on `pkg_resources` (removed by
setuptools 82). The patch is at
`docker/patches/fix_tensorboard_pkg_resources.py` and fails the
build loudly if it's no longer needed (i.e. the installed
tensorboard version contains the upstream fix); when that happens,
remove the patch invocation from both Dockerfiles.

---

## Consolidation

Recent refactor work collapsed several pieces of duplication between
the two images:

- `docker/_lib.sh` — shared shell library, sourced by both run
  scripts. `container_state`, `lib_ensure_running`, common
  subcommand dispatch, persistent overrides loading.
- `docker/entrypoint.sh` — single entrypoint script used by both
  images. Branches on `FORGATHER_REPO`: when set (dev), re-installs
  forgather editable against that path; when unset (runtime), just
  exec's the command. Both flows share the PUID/PGID remap and
  `nvidia-smi` probe.
- PUID/PGID remap — both images use the same pattern (fixed in-image
  UID 1000, remap-via-gosu at start). The dev image used to bake
  host UID at build; that's gone, so a single dev image works for
  any host user.

What stays per-image:
- `Dockerfile` vs `Dockerfile.runtime` — different sources of the
  forgather tree (bind-mount vs git clone), different default CMD,
  different webui handling.
- The image-specific subcommands (`--recreate` on the dev image;
  `--logs` / `--shell` / `--token` / `--dev` / `--recreate` on the
  runtime image).
- The runtime image's `HEALTHCHECK` and `--init` are on by default;
  the dev image inherits `--init` from `docker/run.sh`.
