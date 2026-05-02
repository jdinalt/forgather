# Forgather Docker development image

Ubuntu 24.04-based image that bundles every Forgather **dependency**
— Python 3.12, PyTorch (CUDA wheels), `cut-cross-entropy` from
source, plus a developer-friendly base set of CLI tools (vim, tmux,
ripgrep, jq, htop, ssh, sudo, ...). The Forgather package itself
is **not** baked in: at runtime the entrypoint installs it in
editable mode against the bind-mounted host clone, so there's a
single source of truth (your working tree) and no in-image
duplicate to drift.

The intent is twofold:

- **For users**: a one-command development environment that doesn't
  require touching the host's Python install or fighting with
  CUDA/userspace mismatches.
- **For release testing**: a clean sandbox with no leftover state, so
  you can verify a fresh checkout of Forgather builds and runs
  end-to-end before tagging a release.

## Layout

| File | Purpose |
| ---- | ------- |
| `../Dockerfile` | Image definition |
| `../.dockerignore` | Build-context filter |
| `build.sh` | Builds the image with build args matching the host UID/GID |
| `run.sh` | Launches an interactive shell with the host home bind-mounted |
| `entrypoint.sh` | Activates the venv and re-links the editable install to a bind-mounted repo |

## Quick start

```bash
# From the repo root:
docker/build.sh
docker/run.sh
```

The first build pulls ~3 GB of dependencies (PyTorch + friends) and
takes a few minutes; rebuilds reuse the layer cache. `docker/run.sh`
drops you into a bash shell at the repo path with the venv already
on `PATH`, GPU access enabled (`--gpus all`), and host networking
(so every service inside the container is reachable on the host's
loopback as-is).

```bash
forgather ls -r        # works out of the box
forgather -t v2.yaml train
```

See `docs/getting-started/README.md` for the full walkthrough.

## Container lifecycle

The container is **long-lived**. The first `docker/run.sh`
invocation creates a detached container named
`forgather-dev-${USER}` whose PID 1 is `sleep infinity`; subsequent
invocations re-attach via `docker exec`. Logging out of an
interactive shell does NOT stop the container, so a `forgather
server` (or any training job) you started in one session keeps
running, and you can re-attach from a new terminal to inspect
or control it.

```bash
docker/run.sh                   # attach (creating the container if needed)
docker/run.sh forgather control list
                                # one-shot command in the same container
docker/run.sh --status          # report running / stopped / absent
docker/run.sh --stop            # stop (but keep) the container
docker/run.sh --rm              # stop and remove
docker/run.sh --recreate        # rebuild from scratch (e.g. after image rebuild)
```

When the container already exists, `IMAGE` / `GPUS` / `NETWORK` /
port / mount overrides are ignored — those are baked in at
creation time. To pick up changes (e.g. after `docker/build.sh`
rebuilt the image), run `docker/run.sh --recreate`.

Multiple users on the same host get isolated containers
automatically (`forgather-dev-alice`, `forgather-dev-bob`); set
`NAME=...` to override.

### Equivalent raw docker commands

The helper script is only a thin wrapper. You can drive the
container directly with `docker` if you'd rather:

```bash
NAME=forgather-dev-$USER

docker ps -a --filter "name=${NAME}"           # list (running or not)
docker ps    --filter "name=${NAME}"           # list (running only)
docker logs ${NAME}                            # PID 1 stdout/stderr (entrypoint output)
docker stats ${NAME}                           # live CPU / memory / I/O
docker stop  ${NAME}                           # stop, keep filesystem
docker start ${NAME}                           # start an existing stopped container
docker restart ${NAME}                         # stop + start
docker rm -f ${NAME}                           # stop + remove (next run.sh recreates)
docker exec -it ${NAME} bash -l                # extra interactive shell

# Force-rebuild image, then recreate the container against it:
docker/build.sh -- --no-cache
docker/run.sh --recreate
```

`docker logs` is particularly useful when something goes wrong at
container start — the entrypoint's editable-install re-link and
webui-dist warning are printed there, not into your interactive
shell.

## Networking

`run.sh` defaults to `--network host`. The container shares the
host's network stack, so:

- Every service inside the container is reachable on whatever port
  it bound — no `-p` mappings, no `--host 0.0.0.0` gymnastics.
- `forgather server`, `mkdocs`, `tensorboard`, and the inference
  server all default to binding `127.0.0.1`, which under host
  networking *is* the host's loopback. Open
  <http://localhost:8765/> from the host browser as if Forgather
  were running on bare metal.
- Same exposure as running on the host: services on `127.0.0.1`
  stay on `127.0.0.1`; nothing leaks to the LAN unless you bind
  `0.0.0.0` explicitly.

To opt back into the original bridge networking (with explicit
`-p` forwards), set `NETWORK=bridge`:

```bash
NETWORK=bridge docker/run.sh
```

Under bridge networking the in-container `127.0.0.1` is the
*container's* loopback, which the docker-proxy can't reach. Every
service has to bind `0.0.0.0` to be reachable through the
forward:

```bash
forgather server -H 0.0.0.0
mkdocs serve --host 0.0.0.0
tensorboard --bind_all
```

The container's login banner reminds you of this when
`NETWORK=bridge` is in effect. By default the host-side forward
binds `127.0.0.1` only; set `HOST_BIND=0.0.0.0 NETWORK=bridge`
for LAN access. Forgather server's auth token still gates every
request regardless of bind.

## Web UI bundle

The image does **not** prebuild the SPA — that build is local to
the bind-mounted checkout, and the dist bundle would land in the
wrong tree if we built at image-build time. Build it once on the
host (or once inside the container against the bind-mounted repo)
before starting the Forgather server; subsequent runs reuse it.
The entrypoint prints a one-line reminder on container start when
`webui/dist/` is missing.

```bash
# On the host (or from inside the container — same checkout):
cd "$FORGATHER_REPO" && ./build-webui.sh
```

## How the user identity is preserved

`build.sh` reads `id -un` / `id -u` / `id -g` and bakes them into the
image as `USER_NAME` / `USER_UID` / `USER_GID`. The image's runtime
user matches your host user, so files created inside the container on
the bind-mounted home land with correct ownership on the host. If
your UID collides with Ubuntu 24.04's stock `ubuntu` (uid=1000) user,
the build deletes the stock account before creating yours.

To build for a different identity (e.g. on CI):

```bash
USER_NAME=dev USER_UID=1000 USER_GID=1000 docker/build.sh
```

## How the editable install works

The venv at `/opt/forgather/venv` carries every Forgather
dependency but **not** the Forgather package itself — outside
`/home`, so the bind-mount doesn't shadow it. `run.sh` sets
`FORGATHER_REPO` to your host-side checkout's path; the
entrypoint installs Forgather in editable mode against that
tree on first start (and re-runs the install if you point it at
a different checkout). Your edits show up immediately without
a rebuild, and there is no in-image copy of the repo to drift,
mirror, or chown.

If `FORGATHER_REPO` is unset (or doesn't point at a Forgather
checkout) the entrypoint prints a warning — the venv is still
usable for arbitrary Python work, but the `forgather` command
won't be available until you install the package against a
real source tree.

## Common overrides

```bash
# CPU-only:
GPUS=none docker/run.sh

# Specific GPUs:
GPUS='"device=0,1"' docker/run.sh

# Bind-mount additional host paths:
EXTRA_MOUNTS="-v /scratch:/scratch -v /data:/data" docker/run.sh

# Forward extra ports (in addition to the canonical four):
EXTRA_PORTS="-p 5173:5173" docker/run.sh   # Vite dev server

# Tag and run a different build:
docker/build.sh forgather-dev:experiment
IMAGE=forgather-dev:experiment docker/run.sh
```

### Cross-device symlinks

`run.sh` only bind-mounts `$HOME`. If anything under your home is a
symlink whose target lives on a different filesystem (a RAID volume,
a separate `/data` mount, etc.), the symlink itself is visible inside
the container but its target isn't — every dereference becomes a
dangling link. Common pattern:

```
~/ai_assets/forgather -> /home/dinalt/rust/forgather
/home/dinalt/rust     -> /mnt/rust/home/dinalt/rust    # RAID
```

Inside the container `/mnt/rust` doesn't exist, so the link is broken.
Bind-mount the underlying mountpoint at the same path so symlinks
resolve identically:

```bash
EXTRA_MOUNTS="-v /mnt/rust:/mnt/rust" docker/run.sh --recreate
```

Use `--recreate` (or `--rm` then a normal launch) — mount config is
fixed at container creation, not on `docker exec`.

`run.sh` checks for this at create-time:

- **Fatal (exit 2)** if the forgather repo path itself resolves
  through a symlink to an uncovered location. Without a bind-mount
  Docker fails the container with a confusing `mkdir: file exists`
  OCI error; bailing early gives a clear suggested `EXTRA_MOUNTS`
  line instead.
- **Warning** for any other `$HOME`-rooted symlink whose target is
  uncovered. Non-fatal — those links only matter if you actually
  dereference them inside the container.

### Persistent overrides

Anything you'd otherwise pass via env var on the command line can live
in `~/.config/forgather/docker.env` (or `$XDG_CONFIG_HOME/forgather/
docker.env`, or the path in `$FORGATHER_DOCKER_CONFIG`). The file is
sourced before `run.sh` applies its defaults; use `:= ` so a
command-line `VAR=...` still wins:

```bash
# ~/.config/forgather/docker.env
: "${EXTRA_MOUNTS:=-v /mnt/rust:/mnt/rust -v /data:/data}"
: "${GPUS:=all}"
: "${NETWORK:=host}"
```

## Release-testing workflow

Use the image as a clean sandbox by building with `--no-cache` (to
rule out layer-cache contamination) and bind-mounting a freshly
cloned tree:

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

The `--no-cache` build verifies the Dockerfile and dependency
graph from scratch; the fresh clone verifies the source tree
itself runs end-to-end against that environment. Together that's
exactly what an end user gets after a fresh `pip install -e .`.
