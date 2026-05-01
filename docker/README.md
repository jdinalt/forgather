# Forgather Docker development image

Ubuntu 24.04-based image that bundles a fully provisioned Forgather
environment — Python 3.12, PyTorch (CUDA wheels), the Forgather
package and all dependencies, `cut-cross-entropy` from source, plus a
developer-friendly base set of CLI tools (vim, tmux, ripgrep, jq,
htop, ssh, sudo, ...).

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
on `PATH`, GPU access enabled (`--gpus all`), and the canonical
Forgather server / job ports (8765, 8137, 6006, 8000) forwarded to
the host's loopback.

```bash
forgather ls -r        # works out of the box
forgather -t v2.yaml train
```

See `docs/getting-started/README.md` for the full walkthrough.

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

## How the editable install survives a bind-mounted home

The venv lives at `/opt/forgather/venv` — outside `/home`, so the
bind-mount doesn't shadow it. At image-build time the venv is seeded
with an editable install of the in-image repo at `/opt/forgather/repo`.
`run.sh` sets `FORGATHER_REPO` to the host-side checkout's path; the
entrypoint detects that and re-installs the package in editable mode
against the bind-mounted source tree, so your edits show up
immediately without a rebuild.

If you don't want the live re-link (e.g. release testing the
in-image copy), `unset FORGATHER_REPO` before invoking the
container, or run `docker run` directly without `run.sh`.

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

## Release-testing workflow

Use the image as a clean sandbox: build with `--no-cache` to rule out
layer-cache contamination, then run **without** the bind-mount so the
container only sees the in-image copy.

```bash
docker/build.sh forgather-dev:release-test -- --no-cache
docker run --rm -it --gpus all forgather-dev:release-test \
    bash -lc "forgather ls -r && \
              cd /opt/forgather/repo/examples/tutorials/tiny_llama && \
              forgather -t v2.yaml train"
```

`/opt/forgather/repo` is the COPY-ed-in copy of the source tree; with
no bind-mount and no `FORGATHER_REPO`, that's the install the venv
runs against — exactly what an end user gets after a fresh `pip
install -e .`.
