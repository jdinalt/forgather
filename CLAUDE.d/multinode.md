# Multi-node training and smoke tests

## Multi-node CLI

`forgather cluster` talks to a server started with `--cluster <name>`:

```bash
forgather cluster nodes                    # members + GPUs + version
forgather cluster jobs                     # multi-node bundles
forgather cluster jobs <bundle-id>         # per-rank detail
forgather cluster cancel <bundle-id>       # fan-out cancel

forgather -p <proj> -t <cfg> submit --global \
    [--member host:gpus[:iface] ...]       # repeatable; default = every reachable peer's idle GPUs
    [--rdzv-host hostname] [--rdzv-port 29400] \
    [--priority N] [--dynamic-arg KEY=VAL ...] \
    [--allow-version-mismatch] [--wait]
```

(`forgather cluster submit ...` is a deprecated alias of
`forgather submit --global ...`.)

Hostnames in `--member` resolve to UUIDs via the membership table — no
UUIDs ever appear in the CLI.

## Multi-node smoke test

End-to-end test of the runtime image's multi-node path: builds the
runtime image, deploys to a remote host via NFS-shared
`docker save`/`load`, starts a cluster on both hosts, runs Tiny Llama
v2 across all GPUs, verifies the checkpoint, cleans up.

```bash
tests/smoke_runtime_multinode.sh                       # default: REMOTE=muthur, port 18765
tests/smoke_runtime_multinode.sh --no-build            # skip rebuild + deploy (after first run)
tests/smoke_runtime_multinode.sh --keep                # leave containers running on success
tests/smoke_runtime_multinode.sh --remote box2
SMOKE_PORT=28765 tests/smoke_runtime_multinode.sh
```

**Prerequisites**
- Passwordless `ssh <REMOTE>` from this host
- Both hosts have docker + nvidia-container-toolkit
- `/mnt/rust/aiassets` mounted at the same path on both hosts (NFS or
  equivalent — `project_dir` must resolve identically on every peer)

On any failure the script's EXIT trap dumps `docker logs` from both
hosts, cluster JSON state, the latest TTY log from each peer, and
`nvidia-smi` output to
`/mnt/rust/aiassets/.tmp/smoke-<cluster>-failure-<ts>.log`.

## Manual multi-node (without the script)

```bash
docker run -d --init --gpus all --network host \
    -v /mnt/rust/aiassets:/mnt/rust/aiassets \
    forgather:latest \
    forgather server -H 0.0.0.0 -p 8765 --cluster <name> --no-auth
```

Or via helper:

```bash
NETWORK=host CLUSTER=<name> NO_AUTH=1 docker/runtime/run.sh
```

`NO_AUTH=1` disables the bearer-token gate — trusted-LAN only, used
by the smoke test to avoid token-fetching across N containers.

Full Docker reference (every flag and env var on
`docker/build`, `docker/run`, `docker/runtime/build.sh`,
`docker/runtime/run.sh`; PUID/PGID remap rationale; persistent
overrides; troubleshooting): `docs/getting-started/docker.md`.
