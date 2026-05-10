# Forgather Dataset Server

A Uvicorn + FastAPI server that exposes the Forgather
`fast_load_iterable_dataset` machinery over HTTP. Designed for
multi-node training: one node hosts the datasets (cache, named
locals) and the other nodes consume them transparently by setting
the `FORGATHER_DATASET_SERVER` environment variable. No client
code or config-template change required.

The same `fast_load_iterable_dataset(...)` call routes locally
when the env var is unset and through the server when it's set.
What flows over the wire is a thin `RemoteBackend` wrapped in the
usual `ComposableIterableDataset`, so client-side `.shuffle()`,
`.shard()`, `.map()`, `.state_dict()` etc. all "just work."

## Quick start

### Same host (zero-config auth)

```bash
# Terminal 1 — start the server. Token is auto-generated and
# written to ~/.forgather/dataset_server/8766.token (mode 0600).
forgather dataset-server start

# Terminal 2 — point clients at it. The localhost token is
# auto-discovered by the loader and the diagnostic CLI; no token
# env var needed.
export FORGATHER_DATASET_SERVER=http://localhost:8766

# Confirm it's reachable.
forgather dataset-server status

# Any forgather command that loads datasets now routes through
# the server transparently.
forgather train
```

### Cross-host (explicit token)

```bash
# Server (data host) — bind to a real interface; auto-token is
# printed to stderr on startup. Distribute that token to the
# clients via your usual secret channel (config-management,
# secret store, scp, etc.) and write it to ~/.fdss.token there
# with mode 0600.
forgather dataset-server start --host 0.0.0.0 \
    --local stories=/data/tinystories

# Client (training nodes) — read the token from the file rather
# than pasting it inline (keeps it out of shell history and `ps`
# output).
chmod 600 ~/.fdss.token
export FORGATHER_DATASET_SERVER=http://datahost:8766
export FORGATHER_DATASET_SERVER_TOKEN="$(cat ~/.fdss.token)"

# Sanity check: status hits /v1/health + /v1/auth/status.
forgather dataset-server status
# Then your normal training command — datasets are now served
# over the wire:
forgather train
```

For the full token-resolution order (explicit kwarg →
`FORGATHER_DATASET_SERVER_TOKEN` → per-port localhost file →
none) see [Authentication](#authentication) below.

## Installation

The base Forgather install is sufficient — `fastapi`, `uvicorn`,
and `httpx` are already installed for the other servers.

The server runs as a stand-alone executable, as a Python module,
or via the Forgather CLI. See [Running](#running) below.

## Authentication

The server requires a bearer token by default. Multi-user hosts
share the same loopback addresses, so without auth any local user
on the box could pull whatever datasets the server has cached.

This token model mirrors `tools/inference_server` exactly — same
file layout, same auto-discovery semantics, same `--no-auth` knob.
For the broader threat-model picture (how this token interacts with
forgather-server's bearer token and the per-job trainer-control
token), see the
[forgather server threat model](../../forgather-server.md#threat-model).

**Default behaviour** — if you don't pass any auth flag, the server
generates a random 64-hex-char token at startup and prints it on
**stderr**:

```
dataset_server auth token: 8f5b...
clients must send 'Authorization: Bearer <token>'
curl -H "Authorization: Bearer 8f5b..." http://127.0.0.1:8766/v1/datasets
shared token file: /home/<you>/.forgather/dataset_server/8766.token
```

The auto-generated token is also written to a per-port file under
`$FORGATHER_HOME/dataset_server/<port>.token` (default
`~/.forgather/dataset_server/<port>.token`, mode 0600 in a 0700
directory) and removed when the server exits (atexit + SIGINT /
SIGTERM handlers).

`RemoteBackend`, the loader's `_remote_load_iterable_dataset`, and
the `forgather dataset-server` diagnostic CLI all auto-discover the
token when their URL is loopback (`127.0.0.1`, `::1`, `localhost`).
So a client running on the same host as the server picks up the
token with no flags on either side.

The lookup is keyed by port. A server bound to `0.0.0.0:8766` and a
client connecting to `http://127.0.0.1:8766` share the same file.

**Supplying a known token** — `--auth-token TOKEN` or
`--auth-token-file PATH` (mode 0600). Prefer the file form for
orchestrators since `--auth-token` is visible to other local users
via `ps`. When you supply a token explicitly the server does *not*
publish it to the shared file — operator-managed tokens stay where
the operator put them.

**Disabling auth** — `--no-auth` removes the bearer-token gate
entirely. The startup banner warns prominently. Only use it on
hosts where you're the only user, or where the bind address is on
a trusted network (e.g. an internal cluster interface).

**Override env / explicit token on the client side**:

```bash
# Force a specific token (override the file lookup):
export FORGATHER_DATASET_SERVER_TOKEN="..."

# Or pass per-invocation:
forgather dataset-server status --token "..."
```

## Loading policy

The server has an explicit policy gate over `POST /v1/load`. This
controls **what kinds of datasets** the server is willing to
serve — separate from auth (who is allowed to ask).

Three knobs (all default to the safe option):

| Flag | Default | What it gates |
|---|---|---|
| `--no-hf` | off (HF cache enabled) | Loading any HuggingFace dataset id (e.g. `allenai/c4`). With `--no-hf` only `local/*` mappings work. |
| `--allow-paths` | off (paths rejected) | Loading by absolute filesystem path. Off by default — clients should use named `local/*` mappings instead. |
| `--allow-downloads` | off (cache-only) | Letting HF downloads happen when the dataset isn't cached. The server runs HF loads with `HF_DATASETS_OFFLINE=1` unless this flag is set, so a cache miss surfaces as a 404 instead of starting a multi-hour download. |

Resolution order for a `POST /v1/load` request:

1. If `path` starts with `local/<name>`: look up the local
   mapping. 404 if unknown. If found, the resolved filesystem
   path is loaded — **always allowed**, no `--allow-paths` needed.
2. Else if `path` is an existing filesystem path: requires
   `--allow-paths`. Otherwise 403.
3. Else (assume HF dataset id): requires HF cache to be enabled
   (so not `--no-hf`). The load runs offline unless
   `--allow-downloads`; a cache miss surfaces as 404.

### Named local datasets

The preferred way to expose a local dataset is via `--local`:

```bash
forgather dataset-server start \
    --local stories=/data/tinystories \
    --local mycorpus=/data/corpora/2024-01
```

Clients then request `local/stories` or `local/mycorpus` —
no need to know the server-side filesystem path:

```python
ds = fast_load_iterable_dataset(path="local/stories")
```

`--local` is repeatable. The path must exist at server startup
(checked by argparse). Names must not contain `/`. The
`local/` prefix in the client request is fixed (similar to the
HF `namespace/name` convention).

Why is path-loading off by default? Two reasons:

- It exposes the server's filesystem layout to whoever can
  reach the bind port.
- It makes the client request not-portable across nodes — the
  same path may not exist on every server. Named locals are an
  abstraction that keeps the client request stable.

## Running

### Stand-alone executable

```bash
# In-tree:
./tools/dataset_server/server.py --help

# Or via the interpreter:
python tools/dataset_server/server.py --help
python -m tools.dataset_server --help
```

### Via the forgather CLI

```bash
forgather dataset-server start [server flags...]
```

`start` is a REMAINDER passthrough — every flag after `start`
goes to the underlying script unchanged. `--help` is forwarded.

### Examples

```bash
# Default: HF cache enabled (cache-only, no downloads), no
# locals, paths disabled, auth on (auto-token).
forgather dataset-server start

# Cache-only HF + a couple of named locals.
forgather dataset-server start \
    --local stories=/data/tinystories \
    --local mycorpus=/data/saved_corpus

# Lock down to local mappings only — no HF, no paths.
forgather dataset-server start \
    --no-hf \
    --local foo=/data/foo \
    --local bar=/data/bar

# Trusted-LAN mode: bind everywhere, disable auth.
forgather dataset-server start -H 0.0.0.0 --no-auth

# Allow path-based loads (development convenience):
forgather dataset-server start --allow-paths

# Allow HF downloads on cache miss (rare for a server role):
forgather dataset-server start --allow-downloads
```

The default port is **8766**. The forgather orchestration server
uses 8765 — pick a different port if you have to share a host.

## Client routing (the env-var workflow)

```bash
# Terminal 1
forgather dataset-server start --local stories=/data/tinystories

# Terminal 2
export FORGATHER_DATASET_SERVER=http://localhost:8766

# Now any forgather command that calls fast_load_iterable_dataset
# transparently routes through the server. Example using a stock
# dataset config:
forgather -t fast-iter.yaml dataset --target train_dataset_split -n 3
```

The same call path works in training: every config template that
uses `fast_load_iterable_dataset` gets routed automatically when
the env var is set. No template edits, no client-side code changes.

To go back to local loading, unset the env var.

## Diagnostic CLI

```bash
forgather dataset-server status     # health + auth + policy
forgather dataset-server list       # loaded handles
forgather dataset-server cache      # HF cache contents on the server
forgather dataset-server local      # configured local mappings
```

All accept `--server URL` (default `$FORGATHER_DATASET_SERVER`,
falling back to `http://127.0.0.1:8766`), `--token TOKEN`
(falling back to `$FORGATHER_DATASET_SERVER_TOKEN`, then to the
per-port localhost file), and `--json` for machine output.

The `cache` action is the introspection feature — it reports
which HuggingFace datasets are already in the server's
`~/.cache/huggingface/datasets/` (or `$HF_DATASETS_CACHE`),
broken down by config and split:

```
$ forgather dataset-server cache
cache_root: /home/dinalt/.cache/huggingface/datasets
datasets:   19

- allenai/c4  (1.4 TB)
    en @ 0.0.0  -- train=364,868,892, validation=364,608
- HuggingFaceTB/smollm-corpus  (1.1 TB)
    cosmopedia-v2 @ 0.0.0  -- train=39,134,000
    fineweb-edu-dedup @ 0.0.0  -- train=190,168,005
    python-edu @ 0.0.0  -- train=7,678,448
- ...
```

Use this when you're not sure whether a particular HF dataset
is already pre-warmed on the host you're about to serve from.

## HTTP API

All `/v1/*` endpoints (other than the open ones) require
`Authorization: Bearer <token>` unless the server was started
with `--no-auth`. JSON bodies / responses; the streaming format
on `/iter` is newline-delimited JSON.

### Open endpoints

- **`GET /v1/health`** — returns service / version / current
  policy. Doesn't require auth so health checks work without
  managing tokens.
- **`GET /v1/auth/status`** — `{"auth_required": bool}`. Lets
  clients detect `--no-auth` mode.

### Gated endpoints

- **`GET /v1/datasets`** — list currently loaded handles, with
  length, source, and load_args.
- **`GET /v1/datasets/{handle}`** — handle metadata.
- **`GET /v1/datasets/{handle}/length`** — `{"length": int}`.
- **`GET /v1/datasets/{handle}/iter?seed=&position=&limit=`** —
  NDJSON stream of examples. `seed` reshuffles before iterating;
  `position` seeks to a flat example index first; `limit` caps
  the number returned.
- **`POST /v1/load`** — body `{"path", "name", "split", "data_files",
  "revision"}` mirroring `fast_load_iterable_dataset(...)`. Returns
  `{"handle", "length", "source", "load_args"}`. Subject to the
  loading-policy gate above. Cached by hash of the canonicalized
  args, so the same request returns the same handle.
- **`GET /v1/cache/hf`** — list HF datasets cached on the server's
  host. Walks `$HF_DATASETS_CACHE` (default
  `~/.cache/huggingface/datasets/`).
- **`GET /v1/local`** — list registered `local/*` mappings.

## Architecture notes

- **Stateless wrt clients**: every `/iter` call carries the
  `(seed, position)` it should start from. No per-client state
  on the server — multiple clients can share a handle without
  trampling each other's iteration cursors.
- **Handle cache**: keyed by `sha256(canonicalize(load_args))`.
  Loaded backends live for the lifetime of the server (no LRU
  eviction; intentional — see [out of scope](#out-of-scope)).
- **Loaded backend = pure storage**: the server stores the
  `ArrowBackend` underneath the `ComposableIterableDataset` the
  loader returned. The wrapper layer (slice, shard, map,
  shuffle buffer, state_dict, multi-worker, length estimation)
  lives **client-side** on the `RemoteBackend` wrapper. Clients
  apply their own slice / shard / map without round-tripping it
  through the server.
- **Anti-recursion**: the server lazy-loads via
  `_local_load_iterable_dataset` (a public-but-underscore
  helper in `fast_hf_loader.py`) which always loads locally,
  bypassing the `FORGATHER_DATASET_SERVER` env var. This means
  the server can have the env var set in its own environment
  without looping back to itself.
- **Per-port token**: matches the inference server. Each
  dataset_server instance has its own token file
  (`<port>.token`), so you can run multiple instances on the
  same host without aliasing.
- **Streaming**: `StreamingResponse` over `application/x-ndjson`.
  Clients consume line-by-line. `BrokenPipeError` /
  `ConnectionResetError` on the server side are normal
  end-of-stream conditions when the client closes early; both
  are logged at INFO with the example count actually emitted.

## Out of scope

The server is intentionally minimal. The following are explicit
non-goals — call them out in any future PR if you want them
discussed:

- **Web UI**. The forgather orchestration server has one; the
  dataset server doesn't need one.
- **LRU / size-bound eviction of cached backends**. Handles live
  for the lifetime of the server. For long-running servers with
  many distinct dataset configs this is a known limitation.
- **Compression on the NDJSON stream**. The wire format is plain
  JSON for now; the bottleneck in early measurements has been
  GIL contention and example construction, not bytes-on-wire.
- **Rate limiting / quota**.
- **Inter-server auth carve-outs** (the cluster-peer pattern in
  `forgather_server`). Not needed for the dataset use case.
- **Sharing the forgather_server's global token**. Per-port
  (inference-style) is the chosen model.

## Related docs

- **Loader internals**: [Fast HF Loader](../../datasets/fast-hf-loader.md) —
  what `fast_load_iterable_dataset` actually does on the local path.
- **Checkpointing**: [Fast HF Loader Checkpoints](../../datasets/fast-hf-loader-checkpoints.md) —
  state_dict / load_state_dict semantics, which the
  RemoteBackend-based wrapper inherits unchanged.
- **Other forgather servers**:
  [forgather_server](../../forgather-server.md),
  [inference_server](../inference_server/README.md).
