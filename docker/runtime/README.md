# Forgather runtime Docker image

A pre-built, **user-agnostic** image whose default command is
`forgather server`. Distinct from the dev image (`docker/`):

| | Dev image (`docker/`) | Runtime image (`docker/runtime/`) |
| - | - | - |
| Audience | Forgather developers | Operators / end users / release testing |
| Source code | Bind-mounted from host clone | Cloned from git at build time into `/opt/forgather/repo` |
| In-container user | UID/GID matches host (built per-user) | Fixed `forgather` user, remapped at start |
| Default command | `bash -l` | `forgather server -H 0.0.0.0 -p 8765` |
| Networking | `--network host` (Linux only) | Bridge + `-p 8765:8765` (portable) |
| Webui SPA | Built post-install on the host clone | Built in-image |
| Distributable | No | Yes — single image works for any host user |

The runtime image's source tree is **not** copied from your host
checkout — it's `git clone`d from `FORGATHER_GIT_URL` at the ref
`FORGATHER_GIT_REF` (default `dev` — the active development branch
where this docker tooling currently lives; will move to `main` once
a stable release ships with it). That keeps the build reproducible
and decoupled from whatever stray state happens to sit in the
publisher's working directory. To pin a release tag, point at a
fork, or iterate on an unmerged branch:

```bash
FORGATHER_GIT_REF=feature/my-change docker/runtime/build.sh
```

The dev image remains the recommended path for active development. The
runtime image is the artifact to publish, ship to a server, or hand to
a colleague who just wants to *run* Forgather.

## Quick start

```bash
# From the repo root:
docker/runtime/build.sh                # tag: forgather:latest
docker/runtime/run.sh                  # creates and starts the server
```

On first start, `run.sh` waits for the server to write its auth
token to the state volume and prints a clickable
`http://127.0.0.1:8765/?token=<token>` URL — open that in your host
browser to land in a logged-in session and set a password. The token
persists in the named volume `forgather-state` across `docker rm`,
so subsequent restarts reuse the same value; re-fetch any time with:

```bash
docker/runtime/run.sh --token          # print the auth token
```

You can also paste the token into the plain
`http://127.0.0.1:8765/` login page if you'd rather not embed it
in a URL bar. Either way you get the full Forgather server with the
SPA, queue, scheduler, and
GPU policy manager — same as you'd get inside the dev container.

```bash
# Diagnostic shell as the forgather user — same image, no new container:
docker/runtime/run.sh --shell

# Or directly:
docker exec -u forgather -ti forgather-server bash
```

The diagnostic shell has the venv on PATH, so `forgather`, `python`,
and the rest of the CLI work as you'd expect. Useful for `forgather
control list`, `forgather logs summary`, ad-hoc Python work, etc.

## How the UID/GID remap works

The image ships with an in-container `forgather` user at UID/GID
1000. At container start, the entrypoint reads `PUID`/`PGID` env vars
(defaulting to 1000:1000), `usermod`s the in-container user to match,
chowns the in-image dirs, then drops privileges to that user with
`gosu` before exec'ing the real command.

`docker/runtime/run.sh` automatically forwards `PUID=$(id -u)` and
`PGID=$(id -g)` from the calling shell, so the in-container UID
ends up matching your host UID. The remap is what makes any
host-path bind-mount you DO opt into (HF cache, scratch dir, etc.)
write back with host-correct ownership.

If you launch the container with `docker run --user $(id -u):$(id -g)`
(e.g. on a host where you can't or won't start as root, like rootless
podman), the entrypoint detects that it's not root and skips the
remap — the container just runs as the UID you passed.

## Volumes

`docker/runtime/run.sh` is **conservative about exposing your host
filesystem to the container**. By default it mounts only one thing,
and that thing is a docker-managed named volume — no host paths at
all:

| Source | Container | Purpose | Default? |
| - | - | - | - |
| `forgather-state` (named docker volume) | `/home/forgather/.forgather` | Server state: auth token, queue, GPU policy, generation configs, hardware FLOPS cache | ✓ enabled |
| `$HF_CACHE_HOST` (host path) | `/home/forgather/.cache/huggingface` | Bind-mount — share HF dataset/model cache with the host install | opt-in |
| `$EXTRA_MOUNTS` (host paths, free-form) | wherever you say | Anything else (scratch, data, output dirs, ...) | opt-in |

You opt into the HF cache mount by setting `HF_CACHE_HOST`:

```bash
HF_CACHE_HOST=$HOME/.cache/huggingface docker/runtime/run.sh
```

State persistence: a fresh `docker rm -f` leaves the named volume in
place, so the next `docker/runtime/run.sh` will reuse the same auth
token, queue, and config — no surprise token regeneration on every
restart. To opt out entirely (ephemeral state, fresh token on every
recreate) set `STATE_VOLUME=` (empty string). To bind-mount a host
path instead of the named volume, set `STATE_VOLUME=/host/path/...`.

## Common overrides

```bash
# CPU-only:
GPUS=none docker/runtime/run.sh

# Specific GPUs:
GPUS='"device=0,1"' docker/runtime/run.sh

# Expose on the LAN (default is loopback only; auth token still gates):
HOST_BIND=0.0.0.0 docker/runtime/run.sh

# Use a different port on the host (in case 8765 is taken):
PORT=8888 docker/runtime/run.sh

# Share the host's HF cache (opt-in; not mounted by default):
HF_CACHE_HOST=$HOME/.cache/huggingface docker/runtime/run.sh

# Bind-mount additional host paths (e.g. dataset / output dirs):
EXTRA_MOUNTS="-v /scratch:/scratch -v /data:/data" docker/runtime/run.sh

# Forward extra ports — e.g. a Forgather job's tensorboard:
EXTRA_PORTS="-p 6006:6006" docker/runtime/run.sh

# Override the image (e.g. for a versioned release):
IMAGE=ghcr.io/jdinalt/forgather:1.1.0 docker/runtime/run.sh

# Disable the state volume (ephemeral; fresh auth token every time):
STATE_VOLUME= docker/runtime/run.sh
```

Note that the `IMAGE`, `PORT`, `HOST_BIND`, `GPUS`, `HF_CACHE_HOST`,
`STATE_VOLUME`, `EXTRA_MOUNTS`, and `EXTRA_PORTS` env vars are only
applied when the container is *created* — they're ignored on re-attach.
Use `docker/runtime/run.sh --recreate` to rebuild the container with
new settings.

## Lifecycle commands

```bash
docker/runtime/run.sh                  # create or attach (start if stopped)
docker/runtime/run.sh --status         # state + image + network info
docker/runtime/run.sh --logs           # tail the server's stdout/stderr
docker/runtime/run.sh --shell          # diagnostic bash shell as forgather
docker/runtime/run.sh --token          # print the auth token
docker/runtime/run.sh --stop           # stop, keep filesystem
docker/runtime/run.sh --rm             # stop + remove
docker/runtime/run.sh --recreate       # remove and recreate (e.g. after build)
```

Equivalent raw `docker` commands work too — the script is just a
thin wrapper:

```bash
docker logs -f forgather-server
docker stop forgather-server
docker exec -u forgather -ti forgather-server bash
```

## Troubleshooting

**Server won't start, `docker logs` shows a permission error**
If you opted into `HF_CACHE_HOST` (or any other host-path
bind-mount via `EXTRA_MOUNTS`) and that host directory contains
files owned by a different user than the one running
`docker/runtime/run.sh`, the in-container forgather user — remapped
to your host UID — won't be able to write. The entrypoint never
chowns bind-mounted host paths (it would be slow and pointless on
populated caches). Either chown the host directory yourself or
point the env var at a different writable directory.

**Webui shows "missing dist" warning in logs**
That warning comes from the dev image's entrypoint, not this one.
For the runtime image the SPA bundle is baked into the image
during build (`./build-webui.sh` runs as part of `docker build`).
If the bundle is genuinely missing, rebuild: `docker/runtime/build.sh`.

**Auth token rotates on every start**
The token only persists if `~/.forgather/` is on a persistent volume.
By default `docker/runtime/run.sh` mounts the named volume
`forgather-state` for this purpose; if you `docker volume rm`
that volume between runs, the token is regenerated.

**Different host user wants to use the same image**
That's the whole point — no rebuild needed. Just have them run
`docker/runtime/run.sh` from their account; `PUID=$(id -u)
PGID=$(id -g)` is forwarded automatically. The image stays generic.

**Distributing the image**
Tag and push as usual: `docker tag forgather:latest
ghcr.io/jdinalt/forgather:1.1.0 && docker push
ghcr.io/jdinalt/forgather:1.1.0`. Multi-arch builds (`linux/arm64`)
are out of scope for this Dockerfile; if you need them, drive
`docker buildx build --platform linux/amd64,linux/arm64` against
`Dockerfile.runtime` directly.
