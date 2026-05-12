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
| **User identity** | Host operator's UID/GID/name baked in at build time via `docker/build.sh` build args | Fixed in-container user (`forgather`, UID 1000), remapped to host's PUID/PGID at container start via `gosu` |
| **Distributable** | **No** — scoped to the user who built it (single-host, single-user) | Yes — build once, deploy anywhere |

The two images share the entrypoint script (`docker/entrypoint.sh`)
and a shared shell library for run-script scaffolding
(`docker/_lib.sh`), but they pursue different user-identity stories:
the dev image is built per-operator so the in-container user IS the
host operator from the first instant (no usermod, no gosu drop, no
race); the runtime image is portable and does the PUID/PGID-via-gosu
remap at container start.

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
# Dev image (tag defaults to forgather-dev:<your-host-username>):
docker/build.sh                   # build forgather-dev:dinalt (or similar)
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

## CLI reference

The `docker/` helpers are thin wrappers around `docker build` and
`docker run`. This section is the authoritative listing of every
flag and env var. For the conceptual context behind each (e.g. why
the PUID/PGID remap exists, why mDNS needs `--network host`,
why the runtime image is immutable by design), see [Shared
concerns](#shared-concerns), [Dev image specifics](#dev-image-specifics),
and [Runtime image specifics](#runtime-image-specifics) below.

Every script accepts `-h` / `--help` and prints its own usage from
the script's docstring header. Help is read-only — it never builds,
creates, or modifies anything.

### `docker/build.sh` — build the dev image

```text
docker/build.sh [TAG] [--claude] [-- DOCKER_BUILD_ARGS...]
docker/build.sh -h | --help
```

Build the dev image (`Dockerfile`). The image is **single-user and
host-scoped**: `docker/build.sh` reads `id -u` / `id -g` / `id -un`
from the calling shell and passes them as `USER_UID` / `USER_GID` /
`USER_NAME` build args, baking the host operator's identity directly
into the image. There's no runtime usermod / gosu drop — the in-
container user IS the host user from container start. (For the
build-once-deploy-everywhere, user-agnostic story, use
`docker/runtime/build.sh` instead.)

After the docker build succeeds, `build.sh` runs `./build-webui.sh`
in a transient container against your host clone so the SPA `dist/`
is ready before `docker/run.sh` is invoked. Skip with
`SKIP_WEBUI_BUILD=1` (e.g. when iterating on the SPA via
`npm run dev`).

The default tag is `forgather-dev:<host-username>` so multiple
operators on a shared host don't collide on a single
`forgather-dev:latest` tag.

`docker/build.sh` refuses to run as `uid 0` — baking root into the
image would collide with the existing in-image root account. Re-run
as a regular user, or use the runtime image which is user-agnostic.

**Positional argument**

| Arg | Default | Notes |
| - | - | - |
| `TAG` | `forgather-dev:<host-username>` | Image tag. Combine with `IMAGE=` on `docker/run.sh` to use a non-default build. |

**Flags**

| Flag | Effect |
| - | - |
| `--claude` | Bake in Claude Code (`@anthropic-ai/claude-code`) via npm global. Off by default; the bind-mounted `$HOME` already exposes a host-side install if you have one. |
| `--` | Separator: everything after passes through to `docker build` (e.g. `--no-cache`, `--progress=plain`). |
| `-h` / `--help` | Print usage and exit. |

**Env vars**

| Var | Default | Effect |
| - | - | - |
| `SKIP_WEBUI_BUILD` | _unset_ | Skip the post-build `./build-webui.sh` step. Use when you'll run Vite via `npm run dev` instead of consuming the static `dist/`. |

**Examples**

```bash
docker/build.sh                                   # default tag
docker/build.sh forgather-dev:experiment          # custom tag
docker/build.sh --claude                          # bake in Claude Code
docker/build.sh -- --no-cache                     # force a clean rebuild
docker/build.sh forgather-dev:claude --claude -- --no-cache
SKIP_WEBUI_BUILD=1 docker/build.sh                # no SPA post-step
```

### `docker/run.sh` — launch / attach the dev container

```text
docker/run.sh                       # interactive bash, create-or-attach
docker/run.sh COMMAND [ARG...]      # one-shot command in the same container
docker/run.sh --status | --stop | --rm | --recreate
docker/run.sh -h | --help
```

Long-lived container: first invocation creates it detached
(`sleep infinity` as PID 1); subsequent invocations re-attach via
`docker exec`. Logging out of an interactive shell does NOT stop
the container, so anything started in one session keeps running and
can be inspected from another terminal.

**Subcommands**

| Subcommand | Effect |
| - | - |
| (none) | Create or attach (start if stopped). Default command: interactive `bash -l`. |
| `COMMAND ARGS` | One-shot command in the same container. The container is created or started first if needed. |
| `--status` | Print container state, image tag, network mode, started-at timestamp. |
| `--stop` | Stop the container; keep the filesystem (re-attach later picks up where you left off). |
| `--rm` | Stop and remove. Next `docker/run.sh` recreates from scratch. |
| `--recreate` | Stop + remove + create fresh. Required when env-var overrides change after first create (e.g. you added an `EXTRA_MOUNTS` and want it to take effect). |
| `-h` / `--help` | Print usage and exit. |

**Env vars** *(applied at container CREATE time only — overrides on re-attach are ignored)*

| Var | Default | Effect |
| - | - | - |
| `IMAGE` | `forgather-dev:<host-username>` | Image to run. Combine with `docker/build.sh TAG` to test a different build. |
| `NAME` | `forgather-dev-${USER}` | Container name. Useful when running multiple variants side-by-side. |
| `GPUS` | `all` | Passed to `--gpus`. `none` disables GPU access; `'"device=0,1"'` exposes a subset (note the inner quotes — required for the docker CLI to parse the device list). |
| `NETWORK` | `host` | `host` or `bridge`. Host networking is Linux-only and the most ergonomic; bridge wraps with explicit `-p` forwards. |
| `HOST_BIND` | `127.0.0.1` | Bridge mode only — host interface to bind forwards to. Set to `0.0.0.0` for LAN access. |
| `EXTRA_PORTS` | _empty_ | Bridge mode only — extra `-p` mappings (e.g. `'-p 5173:5173'` for a Vite dev server). Ignored under host networking with a warning. |
| `EXTRA_MOUNTS` | _empty_ | Extra `-v` arguments (e.g. `'-v /scratch:/scratch'`). |
| `FORGATHER_DOCKER_CONFIG` | `~/.config/forgather/docker.env` | Path to the [persistent overrides file](#persistent-overrides) (see below). |

**Examples**

```bash
docker/run.sh                                     # interactive shell
docker/run.sh forgather ls -r                     # one-shot command
docker/run.sh --status                            # state probe
docker/run.sh --recreate                          # roll forward to a new image

GPUS=none docker/run.sh                           # CPU only
GPUS='"device=0,1"' docker/run.sh                 # subset of GPUs
EXTRA_MOUNTS='-v /mnt/rust:/mnt/rust' docker/run.sh --recreate
NETWORK=bridge HOST_BIND=0.0.0.0 docker/run.sh --recreate
IMAGE=forgather-dev:experiment docker/run.sh --recreate
```

### `docker/runtime/build.sh` — build the runtime (distributable) image

```text
docker/runtime/build.sh [TAG] [-- DOCKER_BUILD_ARGS...]
docker/runtime/build.sh -h | --help
```

Build the runtime image (`Dockerfile.runtime`). Source comes from
git (default `dev` branch); the SPA is built inside the image, so
nothing has to be present on the host beyond the `Dockerfile.runtime`
itself. The result is generic and immutable — distribute identical
copies to N nodes.

**Positional argument**

| Arg | Default | Notes |
| - | - | - |
| `TAG` | `forgather:latest` | Image tag. Use a versioned tag for distribution (e.g. `ghcr.io/jdinalt/forgather:1.1.0`). |

**Env vars**

| Var | Default | Effect |
| - | - | - |
| `FORGATHER_GIT_URL` | `https://github.com/jdinalt/forgather.git` | Repo to clone. Override to build from a fork. |
| `FORGATHER_GIT_REF` | `dev` | Branch / tag / SHA to check out. Pin to a release tag (e.g. `v1.0.0`) for reproducible distribution. |
| `FORGATHER_SOURCE_DIR` | _unset_ | Air-gap mode: instead of `git clone`, copy the source from this path inside the build context. Threaded through only when invoking `docker build` directly (the wrapper script does not pass it). |

**Flags / passthrough**

| Flag | Effect |
| - | - |
| `--` | Separator: everything after passes through to `docker build`. |
| `-h` / `--help` | Print usage and exit. |

**Examples**

```bash
docker/runtime/build.sh                                       # default tag, dev branch
docker/runtime/build.sh ghcr.io/me/forgather:1.1.0           # custom tag
FORGATHER_GIT_REF=v1.0.0 docker/runtime/build.sh             # pin a release
FORGATHER_GIT_REF=feature/foo docker/runtime/build.sh        # iterate on a branch
docker/runtime/build.sh -- --no-cache                        # force clean rebuild

# Air-gapped (no git access from the build context):
docker build -t forgather:offline \
    --build-arg FORGATHER_SOURCE_DIR=. \
    -f Dockerfile.runtime .
```

### `docker/runtime/run.sh` — launch / manage the runtime container

```text
docker/runtime/run.sh
docker/runtime/run.sh --status | --logs | --shell | --token
docker/runtime/run.sh --stop | --rm | --recreate
docker/runtime/run.sh --dev [PATH] --recreate
docker/runtime/run.sh -h | --help
```

Default container command: `forgather server -H 0.0.0.0 -p 8765`
(the container exits if this returns). On first start the script
polls for the auth-token file the server writes and prints a
clickable token URL.

**Subcommands**

| Subcommand | Effect |
| - | - |
| (none) | Create or attach. On create, the script polls for the auth-token file and prints `http://${HOST_BIND}:${PORT}/?token=...` (or the host-network equivalent). |
| `--status` | Print container state, image, network mode, started-at timestamp. |
| `--logs` | `docker logs -f` (follow). Use this when something goes wrong at startup — entrypoint output (`nvidia-smi` probe, etc.) lands here, not in your terminal. |
| `--shell` | Diagnostic shell as the `forgather` user (`docker exec -u forgather -ti ... bash -l`). Has the venv on `PATH` so `forgather`, `python`, etc. work as expected. |
| `--token` | Print the current auth token (re-reads from the persistent volume). Use after a restart, or when you need to script the token into another tool. |
| `--stop` | Stop; keep the filesystem (state volume persists). |
| `--rm` | Stop and remove the container. The state volume is NOT removed — destroy that explicitly with `docker volume rm forgather-state`. |
| `--recreate` | Stop + remove + create fresh. Required to pick up env-var override changes. |
| `--dev [PATH]` | DEBUG ONLY. Bind-mount a host-side forgather clone over `/opt/forgather/repo` so host-side edits go live without rebuilding. PATH defaults to the script's own repo root. Equivalent to `DEV=...`. See the runtime image's [`--dev` debug-only opt-in](#runtime-image-specifics) section below. |
| `-h` / `--help` | Print usage and exit. |

**Env vars** *(applied at container CREATE time only)*

| Var | Default | Effect |
| - | - | - |
| `IMAGE` | `forgather:latest` | Image to run (e.g. `ghcr.io/me/forgather:1.1.0` for a versioned release). |
| `NAME` | `forgather-server` | Container name. |
| `NETWORK` | `bridge` | `bridge` (with `-p ${HOST_BIND}:${PORT}:8765`) or `host`. **Use `host` for multi-node** — Forgather's mDNS multicast cluster discovery doesn't traverse Docker bridge networks. Under host networking, `PORT`/`HOST_BIND`/`EXTRA_PORTS` are ignored. |
| `PORT` | `8765` | Bridge only — host-side port forwarded to the server's 8765 inside the container. |
| `HOST_BIND` | `127.0.0.1` | Bridge only — host interface for the forward. `0.0.0.0` exposes on the LAN; the auth token still gates. |
| `GPUS` | `all` | Passed to `--gpus`. `none` for CPU-only; `'"device=0,1"'` for a subset. |
| `HF_CACHE_HOST` | _unset_ | Opt-in bind-mount of a host HuggingFace cache into `/home/forgather/.cache/huggingface`. Lazily creates the host directory. Useful when you want to share downloads with a host-side install. |
| `STATE_VOLUME` | `forgather-state` (named volume) | Mounted at `/home/forgather/.config/forgather` for auth-token / queue / GPU policy. Set to a host path for a bind-mount (e.g. share state with the dev image), or empty (`STATE_VOLUME=`) for ephemeral state (token rotates on every recreate). |
| `EXTRA_MOUNTS` | _empty_ | Extra `-v` args. |
| `EXTRA_PORTS` | _empty_ | Bridge mode only — extra `-p` mappings (e.g. `'-p 6006:6006'` for tensorboard). |
| `CLUSTER` | _unset_ | When set, the server CMD becomes `forgather server -H 0.0.0.0 -p 8765 --cluster <name>`. **Use with `NETWORK=host`** — bridge networking breaks mDNS discovery. The script warns loudly when `CLUSTER` is set without `NETWORK=host`. |
| `CLUSTER_ADDRESS` | _unset_ | When set with `CLUSTER`, appends `--cluster-address <ip>` (overrides the auto-detected interface IP advertised over mDNS). Useful behind NAT or with multiple network interfaces. |
| `NO_AUTH` | _unset_ | When set, the server starts with `--no-auth` (no bearer-token gate). **Trusted-LAN only** — any host on the network can hit the API. Used by the multi-node smoke test to avoid token-fetching across N containers. |
| `TLS_INIT` | _unset_ | When set, runs `forgather tls init` inside the container on first start (idempotent — no-op if TLS state is already provisioned in the mounted state volume). Convenient one-shot HTTPS bring-up. Full reference: [TLS](../operations/tls.md#docker-runtime-image). |
| `DEV` | _unset_ | DEBUG-ONLY. `1` mounts `${REPO_ROOT}` over `/opt/forgather/repo`; a path mounts that path. Equivalent to the `--dev` flag. Triggers a prominent warning at container create. |
| `FORGATHER_DOCKER_CONFIG` | `~/.config/forgather/docker.env` | Path to persistent overrides file. |

**Examples**

```bash
# Single-node, default networking:
docker/runtime/run.sh                             # create + start; prints token URL
docker/runtime/run.sh --status                    # state probe
docker/runtime/run.sh --token                     # re-read the auth token
docker/runtime/run.sh --shell                     # diagnostic shell
docker/runtime/run.sh --recreate                  # roll forward to a new image
docker/runtime/run.sh --logs                      # follow server logs

# Multi-node:
NETWORK=host CLUSTER=lab docker/runtime/run.sh
NETWORK=host CLUSTER=lab CLUSTER_ADDRESS=192.168.1.27 \
    docker/runtime/run.sh

# Trusted-LAN testing without a token gate:
NETWORK=host NO_AUTH=1 docker/runtime/run.sh

# Share state with the dev image (same auth token, queue, configs):
STATE_VOLUME=$HOME/.config/forgather docker/runtime/run.sh

# Debug-only: mount a fork over the baked-in source and recreate:
docker/runtime/run.sh --dev /home/me/forgather-fork --recreate

# Custom port + LAN exposure:
PORT=8888 HOST_BIND=0.0.0.0 docker/runtime/run.sh

# Versioned release image:
IMAGE=ghcr.io/jdinalt/forgather:1.1.0 docker/runtime/run.sh
```

### Persistent overrides

Both `docker/run.sh` and `docker/runtime/run.sh` source
`$FORGATHER_DOCKER_CONFIG` (default
`$XDG_CONFIG_HOME/forgather/docker.env`, falling back to
`~/.config/forgather/docker.env`) before applying defaults. The
file is shell-sourced — use the `: "${VAR:=default}"` pattern so a
command-line `VAR=... docker/run.sh` still wins:

```bash
# ~/.config/forgather/docker.env

# Applies to both images:
: "${EXTRA_MOUNTS:=-v /mnt/rust:/mnt/rust -v /scratch:/scratch}"
: "${GPUS:=all}"
: "${NETWORK:=host}"

# Runtime-image specific (silently ignored by the dev image):
: "${HF_CACHE_HOST:=$HOME/.cache/huggingface}"
: "${STATE_VOLUME:=$HOME/.config/forgather}"
```

The file is shared between both run scripts, so any var that's
relevant to only one is silently ignored by the other. Override
the path entirely with `FORGATHER_DOCKER_CONFIG=/path/to/file`.

**When to use it.** Any time a flag set hits muscle-memory friction
("I always need `EXTRA_MOUNTS=...`"). Persisting it removes the
foot-gun of forgetting to pass it on `--recreate`. The
`: "${VAR:=default}"` pattern keeps one-off overrides on the
command line working as expected.

### Equivalent raw `docker` commands

The helper scripts are thin wrappers — drop straight to `docker`
if you'd rather:

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

---

## Shared concerns

Topics that apply to both images.

### User identity

The two images take opposite approaches here. Pick whichever fits
your deployment story.

**Dev image — host operator baked in at build time.** `docker/build.sh`
reads `id -u` / `id -g` / `id -un` from the calling shell and passes
them to `docker build` as the `USER_UID` / `USER_GID` / `USER_NAME`
build args. The Dockerfile uses those to create the in-container user
with the operator's exact identity. The final `USER ${USER_NAME}`
directive makes that user the default for `docker exec`, so files
written from inside the container land on bind-mounted host paths
with host-correct ownership *without* any runtime remap. The
entrypoint's privilege-drop block is guarded on `id -u == 0` and is
naturally skipped because the container starts as the operator, not
as root.

There's no race window, no usermod, no gosu. The trade-off: one
image per operator on a shared host — the default tag is
`forgather-dev:<host-username>` for exactly this reason.

**Runtime image — fixed in-image UID, remapped at start via PUID/PGID.**
The runtime image is distributable, so it can't bake any single
operator's UID in. It ships with a fixed in-container user (`forgather`,
UID/GID 1000); at container start the entrypoint reads `PUID` / `PGID`
env vars (forwarded automatically by `docker/runtime/run.sh` from
`id -u` / `id -g`), `usermod`s the in-container **uid only** (see
below), chowns the small in-image home, then drops privileges via
`gosu` before exec'ing the server. One image, any operator.

If you launch the runtime image with `docker run --user $(id -u):$(id -g)`
(rootless podman, container-with-no-root scenarios), the entrypoint
detects it isn't running as root and skips the remap entirely.

#### Runtime image: why only the uid is remapped (and the gid stays at 1000)

The in-image venv at `/opt/forgather/venv` is built with files owned
by uid 1000 / gid 1000, with `umask 0002` set during venv-building
RUNs so newly created directories land at mode 0775 (group writable).
At runtime the entrypoint changes the in-container user's **uid** to
PUID but leaves the primary **gid** at 1000. That keeps the venv
group-writable for the remapped user without any recursive chown —
cold-start is fast even when host UID != 1000 (an earlier version
did `chown -R /opt/forgather` on every container start with a
different UID, which ran over thousands of files and added tens of
seconds).

This implicitly assumes **gid 1000 inside the container has no
load-bearing meaning on your host**. On a typical single-user Linux
box the host's gid 1000 is just the first interactive user's primary
group — files created in your bind-mounted home will land with gid
1000 on the host side, which is fine if you're the only user. On a
shared host where gid 1000 belongs to a different user / service,
inspect ownership of files written from the container before
assuming the default is right; ACLs or a different bind-mount
strategy can fix it if needed.

### GPUs

Both run scripts default to `--gpus all`. Override via the
`GPUS` env var (see [CLI reference](#cli-reference)).
The unified entrypoint runs a one-line `nvidia-smi` probe at
container start: prints `nvidia-smi: driver=<ver>, N device(s)
visible` on success, warns when `nvidia-smi` is missing or reports
zero devices. Non-fatal — operators run CPU-only sometimes — but
loud enough that an obvious GPU misconfiguration shows up
immediately.

### Networking

The dev image defaults to `--network host` (Linux only); the runtime
image defaults to bridge with `-p ${HOST_BIND}:${PORT}:8765`. Both
support flipping via `NETWORK=host` / `NETWORK=bridge`.

**For multi-node operation**, set `NETWORK=host` on both images.
Forgather's cluster discovery uses mDNS, which depends on multicast
that doesn't traverse Docker bridge networks. See
[`docs/guides/multi-node-training.md`](../guides/multi-node-training.md)
for the full multi-node setup.

### Persistent state

Forgather's per-user state lives at `~/.config/forgather/` inside the
container — auth token, queue index, GPU policy, generation configs,
hardware FLOPS cache, cluster node id (if multi-node). The two
images get there differently:

- **Dev image** bind-mounts `$HOME` wholesale, so `~/.config/forgather/`
  inside the container *is* the host's `~/.config/forgather/`.
- **Runtime image** mounts a docker-managed named volume
  `forgather-state` at `/home/forgather/.config/forgather/`, isolated from
  the host filesystem (preferred for release deployments).

To make the runtime image read/write the same on-disk state as the
dev image (useful when iterating between the two), point the
runtime's `STATE_VOLUME` at the host path:

```bash
STATE_VOLUME=$HOME/.config/forgather docker/runtime/run.sh
```

To opt out of state persistence on the runtime image (ephemeral —
fresh auth token on every recreate), set `STATE_VOLUME=` (empty).

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

### Container init (zombie reaping)

Both images run with Docker's `--init` flag, which puts `tini`
in front of the entrypoint as PID 1. Without this, when torchrun
gets killed and its worker subprocesses get re-parented to PID 1
(= sleep, on the dev image), nobody calls `wait()` on them and they
pile up as zombies. `tini` reaps orphans regardless of parentage —
the only layer that can see grandchildren of the Forgather server.

This bit operators on the multi-node cluster after a hung save-stop;
see [`docs/guides/multi-node-training.md`](../guides/multi-node-training.md)
for the full story.

---

## Dev image specifics

### Layout

| File | Purpose |
| ---- | ------- |
| `Dockerfile` | Image definition |
| `.dockerignore` | Build-context filter |
| `docker/build.sh` | Builds the image; passes host `id -u`/`id -g`/`id -un` as build args |
| `docker/run.sh` | Launches a long-lived container with `$HOME` bind-mounted |
| `docker/entrypoint.sh` | **Shared with runtime image** — `nvidia-smi` probe, editable-install when `FORGATHER_REPO` is set. The phase-1 PUID/PGID remap block is skipped on the dev image because the container starts as the host operator already. |
| `docker/_lib.sh` | **Shared with runtime image** — common run-script scaffold |

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

### Upgrading Forgather inside the container

The dev image's venv is mutable, and the source tree is bind-mounted
from your host clone — so most updates don't need a rebuild. The
smallest hammer:

```bash
# On the host: pull the new revision.
cd "$FORGATHER_REPO" && git pull

# Inside the running container: refresh deps + re-run editable install.
uv pip install -e "$FORGATHER_REPO"

# If the SPA changed, rebuild the static bundle too.
cd "$FORGATHER_REPO" && ./build-webui.sh

# Restart any long-running services (forgather server, training jobs)
# so they pick up the new code.
```

`uv pip install` here updates whatever's drifted in `pyproject.toml`
(new pinned versions, new dependencies) without re-downloading the
whole venv.

**When you do need to rebuild the image:** dependency surgery that
needs a fresh apt layer (new system packages, a Python minor-version
bump), changes to `Dockerfile` itself, or a venv that's accumulated
enough cruft that a clean slate is faster than untangling it. In
those cases:

```bash
docker/build.sh                  # incremental rebuild
docker/build.sh -- --no-cache    # full rebuild from scratch
docker/run.sh --recreate         # discard the old container, attach to the new image
```

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
  `/usr/bin/claude`, world-executable so the in-container user can
  invoke it. Off by default; the average operator doesn't need it
  baked in.

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

### Cross-device symlinks

`run.sh` only bind-mounts `$HOME`. If anything under your home is a
symlink whose target lives on a different filesystem (a RAID volume,
a separate `/data` mount, etc.), the symlink is visible inside the
container but its target isn't — every dereference dangles. Common
pattern:

```text
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

## Runtime image specifics

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
| `Dockerfile.runtime` | Image definition |
| `docker/runtime/build.sh` | Builds the image |
| `docker/runtime/run.sh` | Launches a server container, prints auth-token URL |
| `docker/entrypoint.sh` | **Shared with dev image** — `nvidia-smi` probe, PUID/PGID remap via `usermod`+`gosu` (only the runtime image takes this branch — the dev image starts as the host operator and skips it), and an editable-install branch that's a no-op when `FORGATHER_REPO` is unset |
| `docker/_lib.sh` | **Shared with dev image** — common run-script scaffold |

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
| `forgather-state` (named volume) | `/home/forgather/.config/forgather` | Server state (auth token, queue, GPU policy, ...) | ✓ enabled |
| `$HF_CACHE_HOST` (host path) | `/home/forgather/.cache/huggingface` | Bind-mount, share HF cache with host install | opt-in |
| `$EXTRA_MOUNTS` (free-form) | wherever you say | scratch, data, output dirs, ... | opt-in |

The state volume keeps the auth token across `docker rm`. Reset by
`docker volume rm forgather-state`, or set `STATE_VOLUME=` (empty)
to opt out entirely.

```bash
# Share HF cache with host:
HF_CACHE_HOST=$HOME/.cache/huggingface docker/runtime/run.sh

# Share state with the dev image (see "Persistent state" above):
STATE_VOLUME=$HOME/.config/forgather docker/runtime/run.sh
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
[`docs/guides/multi-node-training.md`](../guides/multi-node-training.md).

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
Runtime image: the token only persists if `~/.config/forgather/` is on a
persistent volume. By default `docker/runtime/run.sh` mounts the
named volume `forgather-state`; if you `docker volume rm` that
volume between runs, the token is regenerated. Dev image: token
lives on the bind-mounted host home, so it persists across
container recreate.

**Different host user wants to use the same image.**
Only the **runtime image** supports this without a rebuild —
`docker/runtime/run.sh` forwards `PUID=$(id -u)` and `PGID=$(id -g)`
automatically, and the in-image user is remapped at container start
via `gosu`. The **dev image** bakes a single host operator's
identity in at build time; a second user needs to run
`docker/build.sh` from their own account to produce their own image
(the default tag includes their username, so the two coexist).

**Multi-node hang or "no peer discovery."**
mDNS doesn't traverse Docker bridge networks. Set `NETWORK=host`
on every node and recreate. Also check
[`docs/guides/multi-node-training.md`](../guides/multi-node-training.md)
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
the two images while keeping them deliberately divergent on user
identity:

- `docker/_lib.sh` — shared shell library, sourced by both run
  scripts. `container_state`, `lib_ensure_running`, common
  subcommand dispatch, persistent overrides loading. Also provides
  `lib_wait_for_entrypoint_remap`, which the runtime image's
  `--shell` path calls before `docker exec -u forgather` to defeat
  the race against the entrypoint's `usermod`.
- `docker/entrypoint.sh` — single entrypoint script used by both
  images. Branches on `FORGATHER_REPO` for the editable install, and
  the phase-1 PUID/PGID remap block is guarded on `id -u == 0` so
  only the runtime image takes it (the dev image starts as the host
  operator). The `nvidia-smi` probe runs for both. Build-time env
  vars (`UV_CACHE_DIR` etc.) are scrubbed unconditionally at the top
  of the script so both flows behave correctly.
- User identity — **deliberately not unified.** The dev image bakes
  the host operator's UID/GID/name in at build time (one image per
  operator, no runtime remap, no race). The runtime image keeps the
  fixed-UID + PUID-remap pattern (one image, any operator). The two
  scripts are written so the same shared lib handles both stories
  without each having to know about the other.

What stays per-image:
- `Dockerfile` vs `Dockerfile.runtime` — different sources of the
  forgather tree (bind-mount vs git clone), different default CMD,
  different webui handling.
- The image-specific subcommands (`--recreate` on the dev image;
  `--logs` / `--shell` / `--token` / `--dev` / `--recreate` on the
  runtime image).
- The runtime image's `HEALTHCHECK` and `--init` are on by default;
  the dev image inherits `--init` from `docker/run.sh`.
