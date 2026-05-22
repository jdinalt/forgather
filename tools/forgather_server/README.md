# Forgather Server

A web frontend over the existing Forgather CLI. Single pane of glass for
discovering projects, inspecting configurations, queuing training / eval
/ inference / TensorBoard jobs across a GPU pool, watching their TTY
logs, controlling them, and talking to running inference servers from
the browser — wraps `MetaConfig`, `ConfigEnvironment`,
`TrainerControlClient`, and friends rather than re-implementing them.

**Prototype status.** Single-user, localhost-first. Every spawned
service binds to `127.0.0.1` by default and `/api/` is gated by a bearer
token (see [Threat model](#threat-model)). No rate limiting, no native
TLS — run behind an SSH tunnel or reverse proxy if you need LAN access.

> **New here?** For a guided tour of the web UI — fresh install through
> training a Tiny Llama and chatting with it — read the
> [Forgather Server Walkthrough](../../docs/guides/forgather-server-walkthrough.md)
> first; come back here for the reference material.

## Quick reference

Skip to the two reference tables most operators want first:

- [CLI arguments](#cli-arguments) — every flag accepted by
  `forgather server`.
- [Config file (`server_config.yaml`)](#config-file-server_configyaml) —
  full YAML schema for persistent CLI defaults and auto-start services.

The rest of this document covers the threat model, authentication
details, persistent on-disk state, UI panels, and the HTTP API.

---

## CLI arguments

`forgather server` accepts the following arguments. Anything passed on
the command line overrides the matching key in `server_config.yaml`;
anything absent from both falls back to the defaults shown.

| Flag                                 | Default                                  | Effect                                                                                                                |
| ------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `--config PATH`                      | `<config>/server/server_config.yaml`     | Path to the YAML config file. Default location is created (with a commented template) if missing.                     |
| `-H` / `--host HOST`                 | `127.0.0.1`                              | Bind address. `0.0.0.0` / `::` accepted; the bearer token then traverses the network in cleartext unless TLS is on.   |
| `-p` / `--port PORT`                 | `8765`                                   | TCP port.                                                                                                             |
| `-l` / `--log-level LEVEL`           | `INFO`                                   | `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                                                                  |
| `--reload`                           | off                                      | Uvicorn auto-reload — development convenience only; spawned jobs do not survive a hot-reload.                         |
| `--no-auth`                          | off                                      | Disable the bearer-token / password gate. Single-trusted-user host only. See [Threat model](#threat-model).            |
| `--regen-token`                      | off                                      | Rotate the persisted bearer token at startup. Invalidates every CLI client using the old token.                       |
| `--persist-sessions`                 | off                                      | Persist browser session cookies to `<config>/server/sessions.json` (0600) so the webui survives restarts. See [Persisted sessions](#persisted-sessions). |
| `--cluster NAME`                     | unset                                    | Join the named cluster (mDNS-scoped). Standalone otherwise. See [Cluster mode](#cluster-mode-multi-node-prototype).   |
| `--cluster-address IP`               | unset (repeatable)                       | Override the address advertised to cluster peers. Repeatable — useful when running inside a container whose network namespace hides the host NICs from psutil. The first entry also seeds the startup banner's clickable URL when bound to `0.0.0.0`. |
| `--tls` / `--no-tls`                 | shared config                            | Force-enable / force-disable TLS, overriding `<config>/tls/`'s shared setting. See [docs/operations/tls.md](../../docs/operations/tls.md). |
| `--tls-cert PATH` / `--tls-key PATH` | resolved from shared config              | Override the certificate / private-key paths for this run.                                                            |
| `--insecure`                         | off                                      | Allow binding a non-loopback host without TLS. Suppresses the "token in cleartext" abort.                             |
| `--lock-inference-proxy`             | off                                      | Restrict the inference reverse proxy to localhost upstreams. The unconditional `http`/`https`-only scheme guard still applies. See [Network exposure](#network-exposure). |

The `args:` mapping in `server_config.yaml` accepts the same names with
dashes turned to underscores (`log_level`, `regen_token`,
`persist_sessions`, `cluster_address`, …). See the next section.

## Config file (`server_config.yaml`)

Top-level keys:

```yaml
args:        # persistent CLI defaults; CLI flags still win
  ...
services:    # auto-start declarations for long-running spawned processes
  ...
```

The server resolves the file in this order:

1. `--config PATH` on the command line (explicit override).
2. `<forgather_config_dir>/server/server_config.yaml` (default). On
   first boot a commented template is written here so the defaults are
   visible / uncomment-to-change.

Programmatic writes (the webui's **Create service…** button and the
`/api/services` endpoints) regenerate the file body and lose any
inline user comments — a fixed documentation preamble at the top
survives. Operator-edited fields like `args:` keep working but won't
preserve hand-written comments after the first programmatic write.

The sidebar footer's **⚙ Open config** button opens this file in the
embedded editor; **⟳ Restart server** next to it re-execs the running
process so edits take effect without disrupting active jobs (spawned
subprocesses survive across `os.execv`; the rebooted server re-attaches
to them via the standard PID-reattach path).

### `args:` block — CLI default overrides

Every entry under `args:` corresponds to a CLI argument. Use
snake_case (dashes are accepted and normalized for convenience):

```yaml
args:
  # Network
  host: 0.0.0.0
  port: 8765
  log_level: INFO

  # Auth
  no_auth: false
  regen_token: false
  persist_sessions: true        # webui survives restarts (dev convenience)

  # Cluster
  cluster: my-cluster
  cluster_address:
    - 192.168.1.27              # operator-supplied advertise address

  # TLS — see docs/operations/tls.md
  insecure: false
  tls: null                     # path to an alternate TLS config

  # Inference reverse-proxy hardening
  lock_inference_proxy: false
```

Unknown keys log a warning at startup and are ignored.

### `services:` block — auto-start services

Long-running spawned processes the server brings up automatically on
boot. Each entry under `<type>.<name>` is `enabled: true|false` plus
the same args the corresponding modal would have submitted as
`job_params`. Supported types and the queue `job_type` each maps to:

| `type`        | Maps to       | Args shape match                |
| ------------- | ------------- | ------------------------------- |
| `dataset`     | `dataset_server` | The **Dataset…** modal       |
| `inference`   | `inference`      | The **Inference…** modal     |
| `tensorboard` | `tensorboard`    | The **TensorBoard…** modal   |
| `mkdocs`      | `mkdocs`         | The **MkDocs…** modal        |

```yaml
services:
  dataset:
    primary:
      enabled: true
      host: 0.0.0.0
      port: 8766
      no_auth: false
      no_hf: false
      allow_paths: false
      allow_downloads: false
      config_file: /etc/forgather/dataset_server.yaml   # optional
      locals:                                           # optional
        - [shakespeare, /datasets/shakespeare]

  inference:
    llama-8b:
      enabled: true
      model_path: /models/llama-3-8b
      port: 8137
      host: 0.0.0.0
      dtype: bfloat16
      from_checkpoint: false
      compile: false
      disable_kv_cache: false
      requested_gpus: 1         # operator-meta — defaults to 1 for inference
    llama-70b:
      enabled: false            # stays available but not auto-started
      model_path: /models/llama-3-70b
      port: 8138
      requested_gpus: 4

  tensorboard:
    runs:
      enabled: true
      logdir: /mnt/runs
      port: 6006
      bind_all: true

  mkdocs:
    docs:
      enabled: true
      config_file: /repo/mkdocs.yml
      host: localhost
      port: 9999
      strict: false
      livereload: true
      dirty: false
      watch:                    # optional
        - /repo/docs
```

**Operator-meta keys** — recognized at the entry top level alongside
`enabled`, stripped before the args are forwarded to the spawned
process:

| Key              | Default                                | Effect                                  |
| ---------------- | -------------------------------------- | --------------------------------------- |
| `enabled`        | `false`                                | Auto-start the service on boot.         |
| `priority`       | `0`                                    | Queue priority (higher dispatches first). |
| `requested_gpus` | `1` for `inference`, `0` for the rest  | GPU reservation count.                  |

Everything else is forwarded verbatim to the job's `job_params`. The
**dispatch-injected** fields (`scheme`, `routable_host` — added by the
scheduler post-submit for inference / dataset_server jobs) are
excluded from the service signature so a service's pre- and
post-dispatch signatures match, which is what makes restart-without-
double-spawn and ▶/⏹ correctness work.

The names are operator-chosen, must match `[A-Za-z0-9_-]+`, and are
purely human labels — dedupe between configured services and live
queue items is by **signature**, an sha256 over
`(type, normalized args)`. Multiple instances of the same type with
different args are fine (common case: several inference servers on
different ports / models).

For the boot-time / status / sidebar-UI semantics and the matching API
endpoints, see [Auto-start services](#auto-start-services) and
[API quick reference → Services](#services-auto-start).

---

## Threat model

The auth gate is designed for the realistic local-host case: a developer
running the server on their workstation or a shared GPU box, where other
unprivileged Unix accounts may exist on the same machine. It is **not**
a multi-tenant authorization system, and a token holder is effectively
the server's uid. Read this section before exposing anything beyond
loopback.

### What the auth gate defends against

- **Other unprivileged users on the same host.** Loopback ports are not
  isolated by uid on Linux — without auth, any local account could scan
  `127.0.0.1:8765` and drive the server. Bearer tokens stop that.
- **Discovery via shared state.** `~/.config/forgather/` and
  `~/.config/forgather/server/` are mode `0o700`; the persisted token, password
  hash, queue, job records, GPU policy, search roots, override cache,
  per-job inference tokens, and per-job TTY logs are all `0o600`. A
  startup migration tightens modes on legacy files. Other users on the
  host can't read your token off disk.
- **Stale browser tabs on a shared workstation.** `POST
  /api/auth/set-password` requires either the current password or a
  fresh bearer-token authentication when a password is already set. A
  cookie-only session (someone walking up to your unlocked screen) can
  no longer rotate the password silently.
- **Accidental LAN exposure.** Every server-spawned process — the
  forgather server itself, the trainer control endpoint, TensorBoard,
  MkDocs, inference servers — defaults to `127.0.0.1`. Going off
  loopback is an explicit per-process opt-in, called out below.

### What the auth gate does NOT defend against

A holder of the forgather-server bearer token can do everything the
server's uid can do. By design, that includes:

- Reading and writing any file the server uid can read / write — via
  `/api/template/source`, `/api/fs/read`, `/api/fs/write`, etc. There is
  no path-jail.
- Enqueuing arbitrary training / eval / inference / convert / finalize /
  TensorBoard / MkDocs jobs that run as the server's uid.
- Killing those jobs, killing every compute process on a GPU, and
  changing GPU policy.
- Rotating the server's password — but only when authenticated by token
  or current password. Cookie-only sessions cannot.

The token is a **uid-level credential.** Treat it like an SSH key for
the server's user account: never paste it into chat, rotate it with
`forgather server --regen-token` if you suspect compromise, and don't
run the server on a host where you don't trust every user who has shell
access.

### Network exposure

All defaults are loopback. Where there's a legitimate reason to listen
elsewhere, the opt-in is explicit and the auth gate stays in place:

- **Forgather server.** `forgather server -H <host>` binds elsewhere.
  The token then traverses the network in cleartext; use SSH port
  forwarding or a TLS-terminating reverse proxy.
- **Trainer control.** `TrainerControlCallback(host="0.0.0.0", ...)`
  exposes the per-job control endpoint. The per-job bearer token is
  still required, but you have to share it with whichever client is
  reaching in remotely.
- **TensorBoard.** Pass `bind_all=true` in the queue submit modal. This
  bypasses the auth-gated reverse proxy at `/api/tb/{queue_id}/` —
  anyone who can reach the TB port can read your training metrics.
- **Inference server.** `forgather inf server -H 0.0.0.0 ...`. Auth
  remains enforced; the token is printed on the server's stderr at
  startup.
- **Inference proxy.** The forgather server's `/api/inference/*` proxy
  forwards to whatever URL the operator typed into the Inference
  panel. By default any HTTP/HTTPS host is allowed — the proxy is
  auth-gated by the same bearer token as everything else, and an
  authenticated attacker can already submit training jobs that
  exfiltrate anything they please, so an SSRF guard on this endpoint
  doesn't add capability. Operators in stricter environments
  (non-operator-controlled clients) can pass `--lock-inference-proxy`
  to `forgather server` to restrict the proxy to localhost upstreams.
  The scheme guard (http/https only) is unconditional regardless.
- **Dataset_server proxy.** The forgather server's
  `/api/dataset-server/proxy/*` routes forward to dataset_servers the
  webui knows about: locally spawned jobs (auto-discovered, loopback
  only) and URLs the operator has registered via *Datasets → Servers →
  + Add*. Unlike the inference proxy, the dataset_server's primary
  deployment is *remote* — one data host serving N training nodes —
  so the SSRF allowlist is the registry itself rather than an env
  var. Any URL the operator hasn't registered (and isn't loopback) is
  refused with a 403. The registration is the explicit consent.

### Residual gaps

- **MkDocs has no proxy.** MkDocs lacks a clean `--path-prefix` flag and
  HTML rewriting is brittle, so spawned `mkdocs serve` processes are
  loopback-only with no auth in front of them. Other local users on the
  host can read the rendered docs if they discover the port. If you
  need LAN-accessible docs, run `mkdocs serve` outside the scheduler or
  put it behind your own reverse proxy.
- **TLS is opt-in.** Run `forgather tls init` once and every Forgather
  server on the host serves HTTPS off a shared CA. Without it, the
  server refuses to bind non-loopback hosts unless `--insecure` is
  passed. Full walkthrough in [docs/operations/tls.md](../../docs/operations/tls.md).
- **Inter-node cluster calls authenticate via mutual TLS.** With TLS on,
  every peer presents its CA-signed `server.crt` as a client cert for
  `/api/cluster/*_local` requests; the receiving server treats
  cert-presence as proof of cluster membership. Browser / bearer clients
  are unaffected. Details in
  [docs/operations/tls.md#cluster-inter-node-auth-mtls](../../docs/operations/tls.md#cluster-inter-node-auth-mtls).
- **No rate limiting.** A leaked token has no automatic lockout.
- **Dataset-server trust is transitive.** Every example a registered
  dataset_server returns flows into the training pipeline as-is — no
  integrity check, no content filter. A malicious or compromised
  dataset host can poison the resulting model. See the
  [Security considerations](../dataset_server/README.md#security-considerations)
  section of the dataset_server README for the full client-side trust
  story; the short version is "only register URLs you'd `pip install`
  from."

## Authentication overview

The system is composed of several services that each defend their own
endpoints. Operators who want to tune individual knobs should know which
layer they're touching.

### Forgather server (`/api/`)

- Bearer token at `~/.config/forgather/server/auth_token` (mode `0o600`).
- Optional PBKDF2-SHA256 password at `~/.config/forgather/server/password_hash`
  for browser logins.
- `AuthMiddleware` gates everything under `/api/`, including FastAPI's
  `/api/openapi.json`, `/api/docs`, and `/api/redoc`.
- Browser bootstrap via `?token=…`, then an in-memory `HttpOnly` /
  `SameSite=Lax` session cookie. Re-auth is required to set or change
  the password.
- Escape hatch: `forgather server --no-auth` for trusted single-user
  hosts.

### Trainer control (per-job)

- Per-job bearer token at `~/.config/forgather/jobs/{job_id}/auth_token` (mode
  `0o600`), generated by `TrainerControlCallback` on rank 0.
- aiohttp middleware gates `/control`, `/status`, `/jobs`. Default bind
  is `127.0.0.1`.
- `endpoint.json` records the actual bind address. The
  `HTTPTrainerControlClient` (used by `forgather control` and by the
  forgather server's job-control proxy) loads the per-job token
  automatically — no manual configuration needed.
- Constructor knobs: `host`, `auth_token`, `disable_auth`.

### Inference server (per-spawn)

- When spawned by the forgather server scheduler: per-job token at
  `~/.config/forgather/server/inference/{queue_id}.token` (mode `0o600`),
  passed to the inference process via `--auth-token-file` so it never
  appears in `ps`/argv.
- The forgather server's `/api/inference/*` proxy looks up the upstream
  token by `(host, port)` from JobRecords and forwards `Authorization:
  Bearer <token>` to the upstream — the webui doesn't see it.
- When run standalone: `--auth-token`, `--auth-token-file`, or an
  auto-generated token printed on stderr. `--no-auth` to opt out.
- `/v1/*` and `/tokenize` require the bearer; `/health` is always open
  so the proxy can probe before the model finishes loading.

### TensorBoard (per-spawn)

- No native auth. Spawn defaults to `--host 127.0.0.1`.
- Auth-gated reverse proxy at `/api/tb/{queue_id}/{path:path}` rides the
  forgather server's `AuthMiddleware`. The dispatcher passes
  `--path_prefix /api/tb/{queue_id}` so TB's internal links match.
- WebSockets are not proxied; the realtime profile plugin is
  unavailable through the proxy. Users who need it can set
  `bind_all=true` in the queue submit modal and connect to the upstream
  port directly.

### MkDocs (per-spawn)

- No native auth. Spawn defaults to `127.0.0.1`. No reverse proxy.
- Documented residual exposure on shared hosts (see [Residual
  gaps](#residual-gaps)).

### Universal escape hatches

For trusted single-user hosts on a trusted network, auth can be
disabled per service:

- Forgather server: `forgather server --no-auth`.
- Inference server: `forgather inf server --no-auth`.
- Trainer control: `TrainerControlCallback(disable_auth=True)`.

These flags are deliberately verbose. The recommended posture is to
leave auth on and forward ports over SSH for remote access.

---

## CLI access

The `forgather` CLI can talk to a running server directly — no browser needed. All commands accept `--server URL` or the `FORGATHER_SERVER_URL` environment variable; both default to `http://127.0.0.1:8765`.

For a workflow-oriented walkthrough with recipes, see
[guides/server-cli.md](../../docs/guides/server-cli.md). The reference
below is a quick cheat-sheet.

**Submit jobs from the terminal:**

```bash
# Inside a project directory
forgather -t train.yaml train --enqueue
forgather -t train.yaml train --enqueue --priority 5 --requested-gpus 2
forgather eval test c4 -M output_models/my_model --enqueue
forgather tb --enqueue --port 6006
forgather inf server --enqueue -m output_models/my_model
# Multi-model server (one process hosts several models):
forgather inf server --enqueue -m a=output_models/a -m b=output_models/b
# With every loaded model pinned to GPU (no CPU swap; required on
# unified-memory hardware like DGX Spark / Grace-Hopper):
forgather inf server --enqueue -m a=output_models/a -m b=output_models/b --keep-on-gpu
forgather convert --enqueue --src output_models/my_model --dst /tmp/hf_export
forgather finalize --enqueue --source output_models/my_model --dest /tmp/final
forgather update --enqueue --src output_models/my_model --dst /tmp/my_model_v2
forgather mkdocs -f docs/mkdocs.yml --enqueue
```

**Queue and scheduler:**

```bash
forgather sched status                   # enabled, queued/running counts, last tick
forgather sched list                     # table of all queued + active + recent jobs
forgather sched pause                    # stop dispatching new jobs
forgather sched resume
forgather sched cancel <queue_id>        # remove a queued or running job
forgather sched cleanup                  # bulk-remove terminal job records
forgather sched cleanup <job_id>         # remove one specific terminal record
forgather sched gc                       # sweep orphan TTY files (see "State directories and GC")
```

**Per-job control and logs:**

```bash
forgather job status <id>               # trainer status dict (409 = still starting)
forgather job save <id>                 # trigger checkpoint
forgather job stop <id>                 # graceful stop (saves final checkpoint)
forgather job save-stop <id>
forgather job abort <id>                # immediate stop, no checkpoint
forgather job kill <id>                 # SIGTERM
forgather job force-kill --yes <id>     # SIGKILL
forgather job tail <id>                 # stream live TTY; Ctrl-C exits cleanly
forgather job dump <id>                 # write full captured log to stdout
forgather job dump <id> > log.txt
```

**GPU policy:**

```bash
forgather gpu status                    # table: util, mem, temp, power, disabled, min_priority, pids
forgather gpu disable <idx>             # mark GPU unavailable for scheduling
forgather gpu enable <idx>
forgather gpu priority <idx> <N>        # only dispatch jobs with priority >= N to this GPU
forgather gpu kill --yes <idx>          # SIGKILL all compute processes on the card
```

---

## Installation

### Python side

The server's runtime deps ship with Forgather: `fastapi`, `uvicorn`,
`websockets`, `psutil`, `pynvml`, `pydantic`, `pyyaml`. If you installed
Forgather with `pip install -e .`, you're done.

Notes:

- `websockets` is required for the live GPU stream and TTY tail. Without
  a WebSocket backend, uvicorn 404s on upgrade and those features
  silently degrade.
- `pynvml` provides full GPU info (utilization, power, temp, per-GPU
  PIDs). Without it, the server falls back to `torch.cuda` for name +
  memory only — and warns that indices may not match physical indices
  when `CUDA_VISIBLE_DEVICES` is set.
- `psutil` is used for liveness checks (job re-attach across restart,
  abort, the `alive` flag in `/api/jobs`).

### Web UI

Vite + React + TypeScript. Build once, then the running server serves
`webui/dist/` as static assets:

```bash
cd tools/forgather_server/webui
npm install          # one-time
npm run build        # produces webui/dist/
```

`node`/`npm` are only needed for the build step. The running server has
no Node dependency.

**Prefer `./build-webui.sh`** at the repo root for everyday use — it
handles the install gate and a per-platform quirk you'll otherwise hit:

`node_modules/` is platform-specific (npm only fetches the
`@rollup/rollup-<os>-<arch>-{gnu,musl,...}` native binary that matches
the install host), so a tree populated on linux-x86_64 won't link on
linux-aarch64 or darwin-arm64 and vice versa. To keep multiple
platforms happy on the same checkout (e.g. a repo shared over NFS
between hosts, or a developer who builds in both an x86 container and
an ARM container), `build-webui.sh` renames the inactive platform's
install to a sibling directory `.node_modules-<that-platform>/` and
renames the matching platform's sibling (if any) back into
`node_modules/` before each build. The mechanism is two `mv` calls —
no git stash, no symlinks. `node_modules/` is always a real directory
at install time (npm's reify step replaces symlinks). The
`.node_modules-*/` sibling directories are gitignored, and the
committed `package-lock.json` already pins every platform's optional
native dep so each platform installs cleanly without lockfile edits.

Platform tags are `<os>[-musl]-<arch>` — e.g. `linux-x86_64`,
`linux-aarch64`, `linux-musl-aarch64`, `darwin-aarch64` — derived from
`uname -s`/`uname -m` and a libc probe on Linux. The detector
recognises Rollup's `linux-{x64,arm64}-{gnu,musl}` and
`darwin-{x64,arm64}` variants; Windows isn't covered, and an install
on an unrecognised platform falls through to a fresh `npm install`.

Do not `cp -r` a `node_modules/` across hosts of different platform —
let `build-webui.sh` install per-platform.

**Cache headers.** The static-files mount is wrapped in a
`CachingStaticFiles` subclass that pins the SPA cache policy to:

- `index.html` and other unhashed top-level files → `Cache-Control: no-cache`
  (forces revalidation on every navigation; the server still answers
  with 304 Not Modified when nothing has changed).
- `/assets/*` (Vite-emitted, content-hashed) →
  `Cache-Control: public, max-age=31536000, immutable`.

Without this, Starlette's defaults emit no `Cache-Control` at all,
which lets browsers fall back to heuristic freshness on `index.html` —
a freshly-built webui then stays invisible behind a stale cached
`index.html` (which still references the old hashed bundle names) until
the user does a hard reload (Ctrl+Shift+R). If you ever see "I rebuilt
the UI and the change isn't showing up," check the response headers on
`/` first — they should include `cache-control: no-cache`.

## Running

```bash
# Default: 127.0.0.1:8765
forgather server

# Custom bind / verbosity
forgather server -H 127.0.0.1 -p 8765 -l INFO
```

Open <http://127.0.0.1:8765/>. On first boot the server seeds its
search-roots list with `<repo>/examples`; add or remove roots via the
sidebar's **Browse…** button.

### Server config file (`server_config.yaml`)

CLI defaults and auto-start services live in a YAML file so
persistent preferences (host, port, log level, cluster name,
services) don't have to be re-typed on every launch. The full schema
is up front in [Config file (`server_config.yaml`)](#config-file-server_configyaml).

The webui sidebar's bottom bar has a **gear** button (⚙) that opens
this file in the embedded editor, and a **reload** button (⟳) that
restarts the server in place via `os.execv` so config changes take
effect without disrupting running jobs (spawned subprocesses survive
the exec via the existing PID-reattach path on the new server's
boot).

### Auto-start services

For the YAML schema, supported types, and operator-meta keys, see the
[`services:` block](#services-block-auto-start-services) section
near the top of this document.

**Boot semantics.** The lifespan handler runs an autostart pass
before the dispatcher's first tick: for every `enabled: true`
service whose signature isn't already in the queue or in a
non-terminal JobRecord, it enqueues a fresh QueueItem. Already-
running services (matched by signature — including matches against
manually-submitted jobs with the same args) are skipped, so a
restart never double-spawns and an operator who manually started an
equivalent job has it counted as the service's running instance.

**Sidebar UI.** The Services sidebar group renders one row per
launcher (Inference / Dataset / TensorBoard / MkDocs). A right-
aligned count pill shows how many instances are *actually running*
(JobRecord status `running`, not just queued/starting). A disclosure
chevron to the left of the launcher row expands the per-type list
when there are configured instances; each row carries a red/green
dot, ▶/⏹ to toggle the `enabled` flag (start / stop), and × to
delete (the running instance, if any, is aborted first). The four
service modals each have a **Create service…** button beside Start
that prompts for a name and persists the entry to the config file.

**API.** Full CRUD plus enable-toggle, with the enable path running
the autostart pass (or aborting the matching running job) so changes
land immediately. See [API quick reference → Services (auto-start)](#services-auto-start).

### Persisted sessions

In-memory browser sessions are wiped on every restart by default —
"restart" is the implicit revoke. For rapid dev cycles where the
operator is hitting the ⟳ button often, this is tedious. Opt into
persistence with `--persist-sessions` (or `args: persist_sessions:
true` in the config file) and the session dict is written to
`<config>/server/sessions.json` (mode 0600) on every create / revoke
and reloaded on boot. The existing 30-day TTL still applies; the
`/api/auth/logout` endpoint still revokes; `rm sessions.json` drops
everything.

### Authentication (operational)

For the threat model and the full service-by-service layout, see
[Threat model](#threat-model) and [Authentication overview](#authentication-overview).
This section is the operational handbook — token rotation, browser
bootstrap, and remote access.

On startup the server prints a Jupyter-style URL with the token baked in:

```
    Forgather server is running at:
        http://127.0.0.1:8765/?token=4c4febdc…
        http://localhost:8765/?token=4c4febdc…

    CLI auth: token in /home/<user>/.config/forgather/server/auth_token (mode 0600)
    First successful token login will prompt to set a password for future browser logins.
```

When the server binds to a wildcard host (`-H 0.0.0.0` / `::`) the
banner substitutes a connectable address rather than printing the
literal wildcard — Ctrl-clicking `http://0.0.0.0:8765/` doesn't
resolve in any terminal. Priority: the first `--cluster-address`
override → an auto-detected non-loopback IPv4 from `psutil` →
`localhost` as a final fallback. Explicit bind hosts (`-H 127.0.0.1`,
`-H 192.168.1.27`) pass through unchanged.

| Channel                    | Used by                       | Notes                                                     |
| -------------------------- | ----------------------------- | --------------------------------------------------------- |
| `Authorization: Bearer …`  | CLI clients                   | Loaded automatically from the token file (see below).     |
| `?token=…` query parameter | Browser bootstrap, WebSockets | The webui strips it from the URL after exchanging it.     |
| Session cookie             | Browser after login           | `HttpOnly`, `SameSite=Lax`, in-memory (lost on restart).  |
| Password (PBKDF2-SHA256)   | Browser after first login     | Optional; set via the prompt that follows token bootstrap. Re-auth required to change. |

```bash
# Rotate the token (invalidates all existing CLI sessions)
forgather server --regen-token

# Disable auth entirely — only safe on a single-user host you trust.
forgather server --no-auth

# Clear the password (next browser login will prompt to set a new one)
rm ~/.config/forgather/server/password_hash
```

CLI clients pick the token up automatically. Override with
`FORGATHER_SERVER_TOKEN=<token>` if you're talking to a server whose
token file isn't in your home directory (e.g. an SSH-tunnelled remote
machine):

```bash
ssh -L 8765:127.0.0.1:8765 remote
FORGATHER_SERVER_TOKEN=$(ssh remote cat .config/forgather/server/auth_token) \
  forgather sched status
```

Binding to a non-loopback host (`-H 0.0.0.0`) is supported but the
bearer token then traverses the network in cleartext. Run behind an
SSH tunnel or a TLS-terminating reverse proxy for LAN access; native
TLS support is on the roadmap.

### Cluster mode (multi-node, prototype)

The server can join a peer-to-peer cluster of other forgather servers
on the same LAN. Cluster mode is **opt-in**: without `--cluster <name>`
behavior is identical to the single-node prototype.

```bash
# Standalone (default — no LAN advertisement, no peer membership)
forgather server

# Multi-node: advertise on mDNS, peer with other servers using the
# same cluster name. Bind to all interfaces so peers can reach the
# API across the network.
forgather server -H 0.0.0.0 --cluster lab
```

**Cluster name scoping.** Only servers started with the same
`--cluster NAME` see each other. Two unrelated clusters on the same
LAN will not auto-merge. The name is per-invocation (not persisted),
so a host can move between clusters by restarting with a different
flag.

**Node identity.** Each host mints a stable UUID at first cluster
startup, persisted at `~/.config/forgather/cluster/node_id` (mode 0600).
The UUID survives hostname changes, NIC swaps, and cluster-name
changes. Master is selected deterministically as the lowest UUID
among reachable members; no election round-trip.

**Discovery.** mDNS / Zeroconf, advertising `_forgather._tcp` with
TXT records `cluster=<name>`, `node_id=<uuid>`, `version=<x.y.z>`,
`hostname=<host>`. Peers without a matching cluster TXT are ignored.

Address advertisement uses `psutil.net_if_addrs()` to enumerate real
LAN IPs — `socket.gethostname()` is unreliable on Linux because of
`/etc/hosts` artifacts like `127.0.1.1`. Common virtual interface
prefixes (`docker*`, `br-*`, `veth*`, `tun*`, `tap*`, `wg*`, etc.)
are filtered out because they share addresses across hosts and
typically don't carry inter-host traffic.

**When auto-detection fails:** if the server runs inside a container
whose network namespace hides the host's real interfaces, psutil may
see only loopback or only a container bridge. The auto-detector
falls back to `127.0.0.1` and emits a WARNING; peers on other hosts
will not be able to reach you in that state. Use `--cluster-address
<ip>` (repeatable) to specify the address(es) you want advertised:

```bash
# Inside a container without --network host: tell forgather what
# host-routable address to put in the mDNS record.
forgather server -H 0.0.0.0 --cluster lab --cluster-address 192.168.1.27
```

To diagnose what's happening on a running cluster, the server logs
which interface(s) it advertises at startup, and which local
interface it inferred for each incoming peer (matched by subnet
against the peer's advertised address). Look for
`mDNS peer <hostname> at <addr>:<port> via local iface <iface>`
in the log to confirm peers are showing up on the interface you
expect.

**Membership.** Every 5 s each node GETs `/api/cluster/members` from
every other known peer, merges the returned member tables, and marks
silent peers as unreachable after 15 s (two full peer-pull cycles at
the default 5 s cadence). Unreachable peers are kept in the table
(union-of-ever-seen view) — the user agreed model is to flag, not
delete. **Liveness is owned by the direct peer-pull alone**: mDNS
discovery and transitively-reported members are tagged identity-only
and never refresh `last_seen` / flip `reachable=True` on an existing
entry. New members coming in via discovery / peer_report start
`reachable=False` until a direct pull confirms them — otherwise a
stale mDNS cache or a third node restarting with an old member
table could resurrect a dead peer for one sweep window.

**Security.** Inter-node API calls authenticate via mTLS — every
peer presents a CA-signed client certificate during the TLS
handshake, and the auth middleware accepts the call without a
bearer token only for paths on a narrow allow-list. The threat
model assumes the cluster as a whole is trusted (consistent with
the torch.distributed assumption that already underpins multi-host
training — any peer can submit jobs, which is arbitrary code
execution). The carve-out is:

- GET on the read-only inter-node endpoints (members, self,
  master, gpus_local, bandwidth_local, training_status_local,
  dataset_servers_local, dataset_inventory, dataset_servers,
  dataset_router/resolve, issue_url_token) — see
  `auth._PEER_ALLOWED_PATHS`.
- POST on a smaller mutation allow-list (`gpu_policy_local`,
  `training_local`, `training_cancel_local`,
  `dataset_servers/refresh`) — see `auth._PEER_ALLOWED_MUTATIONS`.
- Per-node webui auth (bearer token / browser session) is unchanged
  — the mTLS carve-out applies only to inter-node traffic, not to
  browsers.

**Cross-node SSO.** Clicking a peer in the sidebar Nodes group calls
`POST /api/cluster/peer_session` on the local node. The local node
then GETs `/api/cluster/issue_url_token` on the target peer over
mTLS; the peer mints a 60 s **single-use** URL token (distinct from
its persistent bearer at `~/.config/forgather/server/auth_token`)
and returns it. The browser opens `https://peer:port/?token=…` in
a new tab, the peer's `LoginGate` consumes the token via
`/api/auth/login`, strips it from the address bar, and replaces it
with a session cookie. A leaked URL only exposes a 60 s single-use
window, not the long-lived bearer.

If you don't trust the operators of every node in your cluster,
don't enable cluster mode.

**Cluster view.** When cluster mode is active, a 🖧 **Cluster**
entry appears in the sidebar (cluster-only — filtered out
otherwise). The view is a Datasets-style tabbed panel with four
tabs, all kept mounted so scroll position and in-flight queries
survive switching:

- **jobs** — the Cluster Jobs card (multi-node training bundles);
  see *Cluster Jobs panel* below.
- **network** — pairwise latency + bandwidth probe. `Refresh`
  walks the peer list sequentially (so two simultaneous bulk
  transfers don't saturate the local NIC), per peer doing first a
  30-sample HTTP latency probe — min / median / max ms,
  warmup-trimmed — and then an adaptive parallel-stream **raw-TCP**
  bandwidth probe (4 streams in flight, sized for ~2 s of
  steady-state transfer per stream). The data channel is plain TCP
  via a one-shot ephemeral listener so Python's `ssl` module isn't
  the bottleneck on fast links; the control channel still flows
  over the authenticated mTLS HTTPS path. Each row in the table
  swaps its Latency / Throughput cells to "Measuring…" while that
  peer is in flight so the operator sees per-peer progress.
- **nodes** — per-peer rollup: hostname, master/peer/this-server
  tags, version chips (yellow on divergence), a collapsible
  **Interfaces** list, and a collapsible **GPUs (N · M idle)** list
  — one row per GPU with index/name/memory/util/temp/status. Click
  a GPU row to toggle disabled; mutations route through the master
  proxy.
- **datasets** — the master-aggregated dataset_server / dataset
  inventory previously under *Datasets → Cluster*. Click a dataset
  row to navigate to *Datasets → Explore* with the first healthy
  host's first split pre-selected (see *Cross-view click-through*
  below).

Peer right-click context menu (kill processes, set min-priority)
is intentionally absent in v1 — those mutations route through
future by-node proxy work.

**Sidebar Nodes group.** A second cluster-only surface in the
sidebar above Views lists every peer by hostname with a tri-state
health dot (green / yellow / red — see *Node health* below) and
hands one-click SSO to the peer's webui. Distinct from the
Cluster view in Views: this surface is about *navigating between
nodes*; the Cluster view is about the cluster's internal state.

**Node health.** Each peer's dot reflects three states:

- **green** — reachable and headline versions match the cluster
  majority.
- **yellow** — HTTP-reachable but at least one headline version
  (`forgather`, `torch`, `nccl`, `transformers`) is missing on this
  node or differs from the majority. Catches cases like a peer's
  nvml/driver glitch silently dropping its `nccl` version while the
  node otherwise stays up. The row tooltip lists the disagreements;
  click still works so the operator can SSO in and investigate.
- **red** — last peer-pull failed and `last_seen` exceeded the
  unreachable threshold (15 s by default — two full peer-pull
  cycles).

The dot reflects the live `member.reachable` flag, which is only
refreshed by a *direct* peer-pull GET to that node's
`/api/cluster/members`. Transitive entries reported by other peers
and mDNS-cached records are tagged as identity-only and never
vouch for liveness — so a third node restarting with a stale
member table can't resurrect a dead peer's dot to green.

**Pre-flight probe (Phase 2).** Each member entry carries a
``probe`` payload computed once at startup and propagated via
peer-pull:

- **Versions**: ``forgather``, ``torch`` + CUDA runtime + ``nccl``,
  ``transformers``, ``python``, platform string. Surfaced inline in
  every node's header as compact chips. When a node's value diverges
  from the cluster majority for any headline key, the chip turns
  yellow and tooltips with the divergence; the cluster header gets a
  "version mismatch" tag. Multi-node training is exquisitely
  sensitive to ``torch`` / ``nccl`` mismatches across hosts — the
  Samantha tutorial spends pages on this — so seeing it at a glance
  before launching anything matters.
- **Network interfaces**: every IPv4 interface with address, netmask,
  CIDR, link state, and link speed (when reported by the kernel).
  Collapsible per-node panel. Useful when picking
  ``NCCL_SOCKET_IFNAME`` for multi-node training, and as a quick
  sanity check that cluster-internal traffic is on the interface
  you expect.
- **CPU / RAM summary**: logical + physical core count and total
  RAM in GiB, shown in the node header next to the address.

**Network probe (Phase 2).** Lives on the **Cluster view → network**
tab. On-demand only, triggered by **Refresh** so the network stays
idle the rest of the time; sequential across peers because two
simultaneous bulk transfers would saturate the local NIC and
under-report each link.

For each peer the orchestrator runs two passes in order:

1. **Latency** — 30 keepalived round-trips to
   ``/api/cluster/latency_local`` (empty 200 over the mTLS HTTPS
   channel). First 3 samples discarded to skip TCP-connect /
   TLS-handshake / DNS spikes; report min / median / max ms.
2. **Bandwidth** — adaptive parallel-stream **raw TCP** transfer.
   Coordination over HTTPS: ``POST /api/cluster/bandwidth_prep``
   asks the peer to open a one-shot ``asyncio.start_server``
   listener on ``0.0.0.0:0`` and returns ``(port, 32-byte token)``.
   The local node then opens 4 concurrent plain TCP connections to
   that port, sends the token, and times the receive. The peer
   verifies the token before serving bytes; the listener self-closes
   after the first served connection (or 30 s timeout). Adaptive
   sizing: a single-stream probe estimates the rate, then each of
   the 4 streams pulls enough bytes to take ~2 s of steady-state
   transfer.

The raw-TCP data path bypasses Python's ``ssl`` module, which
otherwise capped single-stream throughput at ~2 Gbps even on a
10 Gbps wire. The bytes themselves are deterministic zero data with
no useful information content, so removing TLS from the data channel
adds no useful capability to an attacker who'd already need to be
inside the cluster LAN's trust boundary (and the 32-byte handshake
token prevents a coincidental port scan during a measurement from
poisoning the result).

Results cached for 1 hour. ``GET /api/cluster/bandwidth`` /
``/api/cluster/latency`` return cached entries;
``POST .../refresh`` re-runs across all peers;
``POST .../refresh_one/{node_id}`` re-runs against one peer (used
by the per-peer "Measuring…" progress feedback in the table).

**Multi-node training submit.** Multi-node submits are folded into
the regular Run dialog — the same dialog that opens from a config's
**▶ Run** action in the project tree or config viewer. When the
server is in cluster mode, a collapsible **Multi-node** panel appears
above the Dynamic arguments section. The local node is pre-checked
as the only participant by default, so a cluster-mode webui that
just clicks Submit gets identical single-node behaviour to a
standalone server. Adding peers turns it into a fanout.

In the panel, each row is a cluster member with five columns: a
**Use** checkbox, the node's hostname/address, a **GPUs** spinner
bounded by the node's actual hardware (with a `(N idle of M)` hint
matching the single-node dialog — wire format stays
``nproc_per_node`` because that's what torchrun expects, the local
scheduler translates it into nproc + ``CUDA_VISIBLE_DEVICES``), an
**NCCL iface** dropdown (or text field on nodes whose probe didn't
report any interfaces), and a **rdzv host** radio. The participant
table caps at ~9 rows then scrolls inside the panel so the
rdzv-port row, version warnings, and help line stay anchored even
with many cluster members.

Project + config come from the dialog itself (the config you
right-clicked Run on), and the dialog's existing dynamic-args + GPU
+ priority knobs flow through to every peer in the fanout. So
per-config overrides — dataset paths, `max_steps`, `lr`, etc. —
reach every node the same way they reach a single-node run.

When cluster mode is active, the dialog's single-node "GPUs"
spinner + nproc help text + gpuMismatch notice are **hidden**: the
panel's per-node GPUs column is the only knob, and showing both
got confusing. Priority stays visible because it applies to both
submit paths.

Last-used multi-node settings (participants, per-node GPUs, iface,
rdzv host/port, mismatch acknowledgement) persist in the same
per-config overrides cache as the dynamic-args, so a config "opens
where you left off" for both submit modes. Reset to defaults
clears multi-node state alongside the dynamic-args.

The **Cluster view → jobs** tab lists the running and
recently-finished bundles, with status, per-rank assignment, and a
Cancel action. There is no longer a "+ Multi-node training" button
on that panel — the submit flow is the regular Run dialog.

On submit, the master:

1. Validates participants are reachable and probe data shows
   matching ``forgather`` / ``torch`` / ``nccl`` / ``transformers``
   versions across the selected set; mismatches return HTTP 409
   unless ``allow_version_mismatch=true`` is passed.
2. Generates a unique ``rdzv_id`` and computes
   ``rdzv_endpoint = <rdzv_node.address>:<rdzv_port>``
   (default port ``29400``).
3. Assigns ``node_rank`` by request order — the rdzv host typically
   ends up rank 0 because the modal puts the master first.
4. Fans out a ``POST /api/cluster/training_local`` to each
   participant with that node's per-rank torchrun args
   (``--nnodes``, ``--node-rank``, ``--rdzv-backend=c10d``,
   ``--rdzv-endpoint``, ``--rdzv-id``, ``--nproc-per-node``,
   ``--rdzv-conf is_host=true|false``). Each peer also gets
   ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, and
   ``TP_SOCKET_IFNAME`` in ``extra_env``, all set to the same
   interface (NCCL for CUDA collectives, Gloo for CPU collectives,
   tensorpipe for RPC — each derives its advertised address
   independently and they must all be pinned together). The
   interface name comes from the operator's modal selection when
   set; otherwise the server auto-derives it by matching the
   member's advertised address against its probe's interface table
   (``_derive_iface_from_member`` in ``routes/cluster.py``). If no
   interface can be derived (probe missing, address mismatch) the
   submit fails with HTTP 422 rather than spawning a job that will
   deadlock in ``connectFullMesh``. The peer's local scheduler
   picks up the queue item and spawns torchrun in rendezvous mode
   (no ``--standalone``).
   The two ``/etc/hosts`` workarounds we have to apply explicitly:
   - ``is_host`` because torch's c10d backend autodetects "am I
     the rendezvous host?" by resolving ``socket.gethostname()``
     and comparing it to ``rdzv_endpoint``. On Debian/Ubuntu the
     system hostname resolves to ``127.0.1.1`` via ``/etc/hosts``,
     so the comparison silently fails on every node and *no* node
     binds the TCPStore.
   - ``GLOO_SOCKET_IFNAME`` (and ``TP_SOCKET_IFNAME``) because
     once the rendezvous succeeds, Gloo's ``connectFullMesh`` has
     each rank publish its own address — also via
     ``socket.gethostname()`` — so peers receive ``127.0.1.1`` and
     connect to their own loopback instead of each other.
5. Records a ClusterJob bundle linking the per-node queue ids back
   to a single ``cluster_job_id``. Listed via
   ``GET /api/cluster/jobs``; cancel via
   ``POST /api/cluster/jobs/{id}/cancel`` fans out a cancel to each
   participant. Bundle creation and cancellation are journaled via
   ``cluster_journal`` so Phase 4's replication seam covers
   multi-node lifecycle.

If a fanout step fails partway through, the master rolls back by
issuing cancels to the participants it already enqueued on, then
returns the original error. There's no half-submitted state.

**Status rollup.** ``GET /api/cluster/jobs`` and
``GET /api/cluster/jobs/{id}`` compute each bundle's live status by
fanning out to every member's ``GET /api/cluster/training_status_local``
(read-only; in the peer-allowed list). The master reads its own
participant's status directly from local job_records, queries every
remote peer in parallel, and rolls the per-rank statuses up via
priority order: ``failed > running > cancelled > queued > done``.
"done" requires *every* member to be terminal — partial completion
is ambiguous, not done. Once the rollup reaches a terminal state
the bundle's own ``status`` field is promoted in place
(``done`` / ``failed`` / ``cancelled``) so subsequent reads
short-circuit without fanning out. Slow or unreachable peers
contribute ``current_status="unknown"`` for that rank rather than
blocking the whole list.

**Non-master proxying.** Bundle records live on the master only. To
keep every webui in the cluster showing the same job list, non-master
nodes proxy ``GET /api/cluster/jobs`` to the master (which is in the
peer-allowed list, so no bearer is needed for the inter-node call).
Master-unreachable falls through to the local empty list rather than
erroring — the page must keep rendering during a master failover.

**Asymmetric topologies.** The fanout itself doesn't care whether
participants have matching ``nproc_per_node`` (the cluster of
operators we tested with had a 1-GPU box and a 2-GPU box).
Deeper, the trainer's per-node coordination groups (used by
``main_process_first`` for cached dataset preprocessing) discover
topology via an ``all_gather_object`` on hostnames rather than the
old ``world_size // local_world_size`` integer math, so heterogeneous
layouts produce correct local groups. Single-rank nodes skip
local-group creation but still participate in peer nodes' group
creation calls so the world-collective stays balanced.

**Limitations to be aware of in v1:**

- Project paths are assumed to resolve at the same location on every
  participant. There is no automatic config staging.
- Per-node TTY logs and job control still run through each peer's
  own webui — there's no cross-node log aggregation. Open the peer's
  webui in another tab to watch its rank's torchrun output.
- ``TrainerControlCallback`` registers only on rank 0 and binds its
  HTTP control endpoint to ``127.0.0.1`` — so live save/stop/abort
  commands have to be issued from the webui or CLI on whichever node
  hosts rank 0. The Cluster Jobs panel's Cancel button still works
  from any node because it routes through the JobRecord-level
  cancel-fanout, not the trainer-control HTTP layer.
- The version check is advisory at the headline-key level
  (``forgather`` / ``torch`` / ``nccl`` / ``transformers``). It
  doesn't compare CUDA toolkit, transformers patch versions, etc.;
  add those to ``cluster_probe.py`` if a real divergence bites.
- ``peak_hardware_flops`` for MFU is auto-detected from rank 0's GPU
  only and multiplied by world_size. For a homogeneous cluster this
  is correct; for a heterogeneous cluster (e.g. mixed 3090 + 4090,
  or pairing a Spark with a desktop GPU) the reported MFU is
  meaningless. Workaround: set
  ``peak_hardware_flops`` explicitly per-config, or stick to
  homogeneous training clusters until probe-driven aggregation
  lands.

**Operational notes for multi-node operation:**

- **Container PID 1 must reap orphan grandchildren.** Forgather's
  Python server doesn't see the worker subprocesses spawned by
  torchrun (those are torchrun's children, not ours), so when
  torchrun gets killed the workers re-parent to PID 1 of the
  container's pid namespace. If PID 1 is ``sleep infinity`` (the
  pre-init default of ``docker/run``) it doesn't call ``wait()``
  and the workers pile up as zombies. ``docker/run`` now passes
  ``--init`` so Docker's bundled ``tini`` becomes PID 1 and reaps
  orphans regardless of parentage. Existing containers need
  recreation to pick this up: ``docker/run --rm && docker/run``.

- **Diagnosing hangs with faulthandler.** ``train_script.py``
  enables Python's ``faulthandler`` at startup and registers
  ``SIGUSR1`` for live thread dumps:
  - On a crash (SIGSEGV / SIGFPE / SIGABRT / SIGBUS / SIGILL), every
    thread's Python stack is dumped to stderr — which torchrun
    routes to the per-rank TTY log. Silent rank deaths (CUDA driver
    assertions, OOM-kills, C++ exceptions in background threads)
    leave a trace where they used to leave nothing.
  - To inspect a hung rank live: ``kill -USR1 <pid>`` against the
    rank's worker process. Faulthandler dumps every thread's stack
    to the TTY log without killing the process. Same idiom as
    ``py-spy dump``, but works inside containers that strip
    ``CAP_SYS_PTRACE`` (which most production containers do, and our
    forgather-dev container in particular). The dump in the TTY
    log shows exactly which ``dist.*`` collective each rank is
    blocked in; matching them up across ranks gives you the
    deadlock site immediately.
  - The per-rank ``DistributedEnvironment(...)`` line includes
    ``host=<hostname>`` so you can correlate "rank N is hung" with
    the actual node it lives on without cross-referencing
    ``cluster_jobs``.

- **Kill verifies process exit.** ``abort`` and ``force-kill`` poll
  for the PID to actually exit (up to 2 s) after issuing the
  signal. If the process is still alive (e.g. stuck in an
  uninterruptible CUDA driver call), the JobRecord's ``error``
  field is populated with a message pointing at the lingering PID
  — the record stays visible in the UI instead of silently
  disappearing while the GPU is still pinned.

- **Stale endpoint cleanup.** A trainer-control endpoint file
  (``~/.config/forgather/jobs/job_*/endpoint.json``) left behind by a
  killed-and-restarted server can resurface as a phantom "running"
  job in the Jobs list. The Jobs panel's right-click menu offers a
  **Remove stale endpoint** action for entries whose PID is
  dead/zombie/recycled — backend rmtree's the directory so the
  entry stops surfacing. Toggle "include dead endpoints" on the
  Jobs panel to see them; the default view filters them out.

- **Single-writer checkpoints on shared FS.** When several ranks
  share a filesystem (NFS, the typical multi-node setup), only one
  rank globally writes the model shard files. The CheckpointManager
  honours ``save_on_each_node=False`` (the documented default for
  shared storage) by gating the shard-file save loop on
  ``_should_save_common``, so concurrent writers can't race on the
  same shard paths. Pipeline-parallel runs (``save_on_all_ranks=True``)
  still have every rank write its own non-overlapping shards as
  before — different stages own disjoint FQNs.

**State.** Cluster runtime state lives at `~/.config/forgather/cluster/`:

```
~/.config/forgather/cluster/
├── node_id              # persistent UUID (0600)
└── journal/
    └── events.jsonl     # append-only event log (Phase 4 seam)
```

The journal is a future-proofing seam: Phase 4 will route every
global-state mutation (queue, GPU policy, cluster jobs) through
append-only events so master/backup replication can be added later
without restructuring storage. v1 emits no events to the journal yet.

#### Multi-node dataset routing (`FORGATHER_DATASET_SERVER=auto`)

In cluster mode the master keeps a deduped inventory of every
`dataset_server` known to any peer (both spawned via the webui's
Tools menu and registered via the per-node user-registry). The
inventory drives a tiny router exposed at

```
GET /api/cluster/dataset_router/resolve?path=<dataset path>
```

which picks a healthy server at random across the candidate set
(crude load balance) and returns `{base_url, auth_token, server_id}`.
Three master-only background loops, started from the lifespan and
self-gated on `cluster.is_self_master`, keep the inventory live:

| loop                                                    | interval               | what it does                                                                 |
|---------------------------------------------------------|------------------------|------------------------------------------------------------------------------|
| `cluster_dataset_inventory.master_collect_servers_loop` | 10 s                   | GET each peer's `/api/cluster/dataset_servers_local`, merge into the set     |
| `cluster_dataset_inventory.master_health_loop`          | 10 s                   | GET `/v1/health` on every dataset server, flip the per-server healthy flag   |
| `cluster_dataset_inventory.master_dataset_refresh_loop` | 10 s (warm-up) / 60 s  | GET `/v1/datasets` + `/v1/local`, rebuild the `local/<name>` routing index   |
| `cluster_inference_inventory.master_collect_servers_loop` | 10 s                 | GET each peer's `/api/cluster/inference_servers_local` for the picker        |
| `cluster_inference_inventory.master_health_loop`        | 10 s                   | GET `/health` on every inference server (root-mounted, not `/v1/health`)     |

On a master transition the new master clears its inventory and the
router returns `503 Retry-After: 5` until the first dataset-refresh
pass completes. `local/<name>` is a **global** key — two servers
advertising the same name are treated as interchangeable replicas
(intentional, gives operators a knob for redundancy/load-balance).
HF / path requests fall back to "any healthy server" and the
`dataset_server` loads on demand; the resilient client retries on
failure and re-routes to a different server on its next attempt.

To use the router from a training job:

```bash
FORGATHER_DATASET_SERVER=auto forgather train …                # CLI
forgather -p <proj> -t <cfg> cluster submit --dataset-source auto …
```

Or pick `Auto (cluster routing)` in any submit modal. The CLI flag
and modal selector both encode `dataset_source={"kind":"auto"}` on
the job_params; the scheduler's `dataset_source.resolve_to_env`
expands that to `FORGATHER_DATASET_SERVER=auto` in the spawn env,
and the resilient client in
`forgather.ml.datasets.resilient_remote_backend` queries the local
forgather_server's resolve endpoint on every (re)connect — so a
peer that dies mid-iteration causes the next attempt to land on a
different healthy peer with no operator intervention.

Diagnostics: `forgather cluster datasets [-v]` prints the deduped
inventory; `forgather cluster resolve <path>` dry-runs the router;
`forgather cluster server <server_id> {status|list|cache|local}`
talks to any cluster server via the master-proxy without needing
the upstream bearer. The **Cluster view → datasets** tab in the
webui surfaces the same payload — server health, refresh ages,
per-server poll counters, and a deduped dataset table with hosts.
Clicking a dataset row navigates to *Datasets → Explore* with the
first healthy host's first split pre-selected.

**Known limits in v1.**

- No global scheduler — peer scheduling decisions are still
  independent. Cluster job submits use a static fanout at submit
  time; there is no live re-balancing or cross-node preemption.
- No file/log streaming through a by-node proxy — to inspect a peer's
  jobs / projects / files outside the Cluster Jobs panel, open that
  peer's webui directly. The "any node sees the same cluster job
  list" proxying covers `/api/cluster/jobs` only, not `/api/jobs` or
  the file/project endpoints.
- ``TrainerControlCallback`` registers only on rank 0 and binds its
  HTTP control endpoint to ``127.0.0.1`` — see "Operational notes"
  above.
- No automatic master failover — the master is whichever reachable
  member has the lowest UUID; if it goes down the cluster keeps
  running with a new master, but in-flight global state (queue
  mutations during the gap) is lost. Phase 4 + Phase 5 work.
- No cross-architecture training (e.g. ARM Spark + x86_64 desktop):
  the version probe surfaces a platform mismatch in the Cluster
  view's Nodes tab (and in the sidebar Nodes dot as yellow) and the
  multi-node submit refuses unless the operator acknowledges, but
  torch wheels and CUDA kernels won't actually interoperate across
  architectures. The check is advisory; the operator is on the
  hook for whether their cluster makes sense.

### Excluding misbehaving GPUs

Set `CUDA_VISIBLE_DEVICES` when starting the server to keep specific
GPUs out of the scheduler's allocation pool. Excluded cards still appear
in the GPUs view (telemetry stays live so you can monitor temperatures /
processes) but with a dashed red border and an `EXCLUDED` badge — the
scheduler refuses to assign them.

```bash
# Reserve GPU 2 (e.g. thermally suspect) — dispatcher won't pick it
CUDA_VISIBLE_DEVICES=0,1,3,4,5 forgather server -p 8765
```

The allow-list is parsed once at module import. Restart the server to
change it.

### Persistent state

Everything under `~/.config/forgather/server/` survives restarts:

| File / dir                  | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `search_roots.json`         | Project-discovery roots (seeded on first boot).        |
| `queue.json`                | Queue of items waiting for GPUs.                       |
| `job_records.json`          | Records for jobs the server has launched (any state). |
| `jobs/{queue_id}.tty`       | Captured stdout+stderr for each launched job.          |
| `overrides/{hash}.json`     | Per-config dynamic-args override cache.                |
| `gpu_policy.json`           | Per-GPU runtime policy: disabled + min_priority.       |
| `auth_token`                | Bearer token shared with CLI clients (mode 0600).      |
| `password_hash`             | Optional pbkdf2_sha256 hash for browser logins (0600). |
| `sessions.json`             | Persisted browser sessions (0600). Present only when started with `--persist-sessions`. |
| `server_config.yaml`        | Operator-editable CLI defaults + auto-start services (0600). See [Server config file](#server-config-file-server_configyaml). |

All state files are written crash-atomically via `_atomic.py`: tmp file
written in the target directory, `fsync` on the fd, then `os.replace`.
Power loss or SIGKILL mid-write never leaves the canonical file
partially written. Every reader tolerates a corrupt / truncated file by
falling back to empty state.

### State directories and GC

Two sibling directories under `~/.config/forgather/` accumulate per-job files,
one per subsystem. They are independent — neither owns the other —
though the server reads the trainer-side directory to correlate
PID-lineage with running JobRecords.

#### `~/.config/forgather/server/jobs/q_*.tty` (server-owned)

The captured stdout/stderr of every job the server dispatches. For
training jobs the scheduler symlinks `q_<id>.tty` to
`<run>/logs/tty.log` once the trainer's `endpoint.json` is correlated,
so users can `tail -f logs/tty.log` from the run directory while the
job is live.

When a JobRecord transitions to a terminal status (`done` / `failed`
/ `aborted`), the scheduler **moves the captured TTY into the run's
`logs/tty.log`**, atomically replacing the symlink with the actual
file. After this the run directory is self-contained — the central
copy under `~/.config/forgather/server/jobs/` is gone. For non-training
jobs (eval, inference, tensorboard, …) there is no `logs_dir` to move
into; their TTY stays in the central directory until the JobRecord is
removed (`DELETE /api/jobs/{id}` or `POST /api/jobs/cleanup`), which
also unlinks it.

A periodic sweep (daily, plus once at server startup) deletes any
`q_*.tty` whose `queue_id` is not referenced by any record or
queued item, mtime older than `FORGATHER_ORPHAN_TTY_TTL_SECONDS`
(default `3600`). Run it on demand with:

```bash
forgather sched gc
```

#### `~/.config/forgather/jobs/job_<ts>_<host>_<pid>/` (trainer-owned)

Each `TrainerControlCallback` (added to a Forgather Trainer via the
`callbacks=` argument; see the project-root `CLAUDE.md` for the
boilerplate) creates a per-job directory here on rank 0 and writes
`endpoint.json` with the host:port the trainer's HTTP control API
listens on. On a clean exit the callback both removes
`endpoint.json` and `rmdir`s the directory, so well-behaved runs
leave nothing behind. Crashed runs leak the directory.

`forgather control cleanup` reaps both kinds of leftover:

- Directories whose `endpoint.json` points at a dead PID (or one that
  the kernel has recycled — verified against `psutil.Process.create_time()`).
- Directories with no `endpoint.json` and mtime older than the TTL
  (`--ttl SECONDS`, or `FORGATHER_ORPHAN_JOB_DIR_TTL_SECONDS`,
  default 3600) — these are crash leftovers.

```bash
# Show counts and prompt before deleting
forgather control cleanup
# Skip the prompt
forgather control cleanup --force
# Tighter age threshold for orphan directories
forgather control cleanup --ttl 600
```

### Re-attach across restart

Training subprocesses are spawned with `start_new_session=True`, so they
keep running after the server exits. On startup the scheduler walks
every JobRecord still marked `running` / `starting` and:

- If the recorded PID is still alive (and `create_time()` matches, to
  guard against PID reuse): re-attach in the unified jobs list.
  Trainer-side control commands (Save / Stop / Save&Stop / Abort) and
  the local `Kill` keep working through the existing endpoint plus
  process-group SIGTERM.
- Otherwise: mark the record `failed` with a clear reason.

Reaping a re-attached job records `status="done"` with `exit_code=null`
since exit codes for non-child processes aren't recoverable from
outside.

### Dev mode (Vite + hot reload)

For rapid frontend iteration, run Vite separately from the API:

```bash
# Terminal 1 — API backend
forgather server -p 8765

# Terminal 2 — Vite dev server with hot reload
cd tools/forgather_server/webui
npm run dev
# opens http://localhost:5173, proxies /api → :8765 (REST + WebSocket)
```

---

## Implemented features

### App chrome

The left side of the window is a collapsible sidebar (`<aside
class="app-sidebar">`) that owns navigation and global actions. Top to
bottom:

- **Header** — "Forgather Server" title and a window/sidebar SVG
  toggle that collapses the sidebar. Right-click anywhere on the
  header opens a small context menu whose only entry today is
  **Help…**, routing to this reference document (rendered through
  MkDocs if a serve is alive, the built-in Docs viewer otherwise).
  (The Refresh and scheduler ▶/⏸ controls that used to live up here
  moved to the new footer — see below.)
- **Nodes** (cluster-only, sits above Views) — collapsible
  `<details>` listing every cluster peer by hostname with a tri-state
  health dot (green = reachable, yellow = reachable but a headline
  version is missing / diverges from the cluster majority, red =
  unreachable) and master/this-server tags. Clicking a peer mints a
  short-lived single-use SSO URL (`/api/cluster/peer_session`) and
  opens that peer's webui in a new tab with no login prompt — same
  trust model as cluster bearer access. Hidden entirely when the
  server is in standalone mode.
- **Views** (collapsible `<details>`) — vertical tabs with icons:
  🖧 Cluster (cluster-only), 📁 Projects, ✎ Edit, 📚 Docs, 🖥 GPUs,
  📋 Queue, ⚙ Jobs, 🔮 Inference, 🗂 Datasets. Selecting anything
  in the project tree routes back to the Projects view
  automatically. The **Edit** view is the tabbed Monaco editor
  (formerly named "Files"); it was renamed to free the "Files" name
  for the new sidebar filesystem tree (see below). **GPUs** is
  always the local node's live `GpuPanel` (WS stream, kill, context
  menu), independent of cluster mode. **Cluster** is the
  cluster-wide surface — see
  [Cluster mode (multi-node, prototype)](#cluster-mode-multi-node-prototype).
- **Tools** (collapsible `<details>`) — one-shot model-manipulation
  utilities. Persisted to localStorage so the next open of each
  modal defaults to the last-committed values; `priority` resets
  each time since the right value depends on current queue state.
    - **📐 Evaluate…** — queues `forgather eval` against an arbitrary
      model directory.
    - **🔁 Convert Model…** — queues `forgather convert` against a
      pair of source/destination model paths. Direction
      (HF ↔ Forgather) is auto-detected unless `--reverse` is forced.
      Persisted under `forgather-global-convert-v1`. The footer
      carries a **Reset to defaults** button that clears the
      persisted blob.
    - **📦 Finalize Model…** — queues `forgather finalize` to package
      a trained Forgather output tree into a clean directory:
      tokenizer additions, chat template, generation config,
      root-copy / keep-optimizer toggles. Persisted under
      `forgather-global-finalize-v1`. Same **Reset to defaults**
      affordance.
    - **⬆️ Update Model…** — queues `forgather update` to migrate a
      saved Forgather model to the current source schema. Reads
      `forgather_arch` / `forgather_arch_version` from the source
      `config.json` and walks the per-arch migration chain; the
      modal exposes `--arch` / `--from-version` / `--to-version` /
      `--checkpoint` overrides plus dtype, device, strict / no-strict,
      safetensors, and dry-run toggles. Persisted under
      `forgather-global-update-v1`. Same **Reset to defaults**
      affordance.
- **Services** (collapsible `<details>`) — launchers for the four
  long-running spawned-process services: 🔮 Inference, 🗂 Dataset,
  📊 TensorBoard, 📖 MkDocs. Same persistence model as Tools. Each
  launcher carries a right-aligned running-count pill (same UI as
  Views → Jobs) and, when there are configured instances of that
  type, a chevron that expands a per-type list of saved services.
  Each saved-service row has a red/green dot reflecting actual
  running state (JobRecord `status == "running"`), a ▶/⏹ toggle
  that flips the `enabled` flag, an `×` delete (aborts the running
  instance first), and a clickable label that does the obvious
  thing for each type:

  - **Inference / Dataset** → switch to the matching view (chat
    or browse the running server).
  - **TensorBoard** → open `http://<host>:<port>/api/tb/<queue_id>/`
    in a new tab. The path prefix is the one the scheduler stamps
    onto the spawned TB via `--path_prefix`; TB only serves under
    that prefix.
  - **MkDocs** → open `http://<host>:<port>/` in a new tab.

  For wildcard binds (`0.0.0.0` / `::`), the URL substitutes
  `window.location.hostname` — the host the browser is already
  reaching the webui on, guaranteed to be reachable from there.

  Each service modal also has a **Create service…** button that
  prompts for a name (with a sensible default per type — model
  basename for inference, logdir basename for tensorboard, etc.)
  and persists the modal's current args into `server_config.yaml`
  via [`POST /api/services`](#services-auto-start). See
  [Auto-start services](#auto-start-services) for the boot
  semantics.
- **Project tree** — Search Roots + workspace-clustered projects
  (see below).

Below the scrolling section stack is a **sidebar footer** pinned
via `position: sticky; bottom: 0`. Four icon-only buttons (tooltips
explain each):

| Glyph | Action |
| --- | --- |
| ⟳ | **Refresh data** — invalidates the entire client query cache so disk edits to workspace metadata, templates, configs are picked up immediately. |
| ▶ / ⏸ | **Scheduler toggle** — flips the dispatcher loop on/off (green when running, muted when paused). Same mutation that backed the old header button. |
| ↺ | **Restart server** — confirms, then hits `POST /api/server/restart`. The process re-execs in place; running training / inference / dataset_server / mkdocs / tensorboard subprocesses survive across the exec via the standard PID-reattach path. Useful for picking up `server_config.yaml` changes without killing the terminal. |
| ⚙ | **Open server config** — opens the resolved `server_config.yaml` in the embedded editor. |

When collapsed, the sidebar shrinks to a 44-px strip showing only the
expand toggle and the icon-only view switcher. Both the collapsed
strip and the expanded layout stay mounted in the DOM (toggled via
`display:none`), so the project tree's expansion state — which
workspaces / projects / artifact groups are open — survives a
collapse/expand cycle.

Default ports for the spawned services match each tool's canonical
default — TensorBoard `6006`, inference `8137`, MkDocs `8000` — so
existing SSH port-forward configs keep working without per-host
rebinds. Inference picks `8137` rather than the more common `8000` so
it doesn't collide with MkDocs out of the box. Collisions on first
submit are easy to resolve in the dialog and the resolved port
persists for next time.

### Project / config discovery

- Walks each search root in two passes: first for ``forgather_workspace/``
  marker dirs (so empty workspaces seed empty clusters that still show
  in the tree), then for ``meta.yaml`` (projects, attached to whichever
  workspace_root MetaConfig resolves them to). Hierarchical workspaces
  nest under their enclosing parent. Both passes prune hidden directories,
  ``forgather_workspace/``, ``output_models/``, ``node_modules/``,
  ``__pycache__``, and ``.git`` to avoid slow or redundant subtree walks.
- Workspaces resolve display name + description from
  `forgather_workspace/workspace.yaml` → README title + first paragraph
  → directory basename. `forgather ws create` writes `workspace.yaml`
  alongside the existing files.
- Configs lazy-load `config_name`, `config_description`, and
  `config_class` from the materialized `meta` block when their project
  is expanded.
- **Per-config artifact sub-tree** — configs that have materialized
  outputs (runs, checkpoints, evaluations) expand to three sub-groups
  with live counts: **Logs**, **Checkpoints**, **Evaluations**. Leaves
  are clickable selection targets with their own detail panels in the
  right pane, and right-clickable for delete-permanently / delete-all
  (user-confirmed, guarded by `/api/fs/delete-dir`). Populated lazily
  via `/api/project/models` — two configs that materialize to the same
  `output_dir` show the same sub-nodes.
- **Refresh button** (⟳ in the sidebar footer) invalidates the
  entire client query cache so disk edits to workspace metadata,
  templates, configs are picked up immediately.

### Config inspection

A three-tab viewer for the selected config:

| Tab         | Content                                                                          |
| ----------- | -------------------------------------------------------------------------------- |
| `info`      | Project's `README.md` rendered as markdown (GFM tables, inline images).          |
| `pp`        | Jinja-rendered, fully preprocessed YAML.                                         |
| `templates` | Two browsing modes (mode bar at the top of the left pane): `trefs` shows the Graphviz-rendered template-dependency graph for the selected config; `tlist` shows every template on the project's search path, grouped by search-root category. Click a node / row to preview in the right pane. |

Monaco syntax-highlights these with a custom Monarch tokenizer
for Forgather's YAML + Jinja2 dialect (`--`/`<<`/`>>`/`==` line
statements, `[block]` / `[/block]`, `!call` / `!partial` / `!singleton`
/ etc., inline `#--- name ---` markers, anchors / aliases).

The `templates` tab's right pane displays the selected node's source
read-only. An **✎ Edit** button next to the path label hands the file
off to the Edit panel (see below) for actual editing.
Right-clicking any template — graph node in `trefs` mode, list row
in `tlist` mode — opens a context menu with **✎ Open in Editor**
that bypasses the preview and drops the file straight into the Files
panel.

The `tlist` view is backed by `GET /api/project/templates`, which
mirrors the interactive CLI's `edit` selector: groups labeled
"Project Templates" / "Workspace Templates" / "Base Templates" /
"Example Templates" / "<Subdir> Templates", with each template
attributed to the *first* matching search-path entry (Jinja's
first-match resolution). A synthetic **Meta** group is prepended,
containing the project's `meta.yaml` so it can be browsed and edited
alongside templates — `meta.yaml` lives outside any `templates/`
directory so `MetaConfig.find_templates()` doesn't yield it on its
own. The Meta group is inserted *after* the search-path attribution
loop runs so the project_dir search root (which contains every
project template) doesn't sweep them into Meta.

**Header**: shows the config's pretty name (from `config_name` in
the materialized meta block) bolded, with the yaml filename in muted
monospace next to it (omitted when the two would be identical), then
a small `config_class` chip, then the project label. Mirrors the
two-line label the project tree already uses.

**Auto-navigation**: clicking a project (expanding the tree node)
selects its `default_config` and switches to the `info` tab — so
browsing projects surfaces the README first.

**Tab tracking on config switch**: the `info` tab is project-scoped
(it's the README), so a click that's actively *choosing* a config
in the tree silently jumps to the `templates` tab. The two
config-scoped tabs (`pp`, `templates`) are left alone so the user
can iterate across configs while keeping the same lens — comparing
materialized YAML between configs is the entire point of `pp`, and
the `templates` view auto-updates its right pane (see below) so
re-clicking a config feels like flipping a slide.

**Right-pane follows the active config**: in either `trefs` or
`tlist` mode the read-only preview auto-resets to the active
config's own template every time the config changes, including the
initial mount. Manual deep-dives into parent templates (clicking a
node in `trefs` or a non-config row in `tlist`) override the
preview and aren't disturbed unless the user picks a different
config.

**tlist click promotes configs**: clicking a row in `tlist` whose
path matches one of the project's configs (i.e. lives under
`config_prefix`) promotes that config to the active selection,
updating the header chip, action buttons, dynamic-args form, and
the `trefs` graph that you'd see if you flipped modes. trefs nodes
are always *referenced* templates of the current config — never
sibling configs you'd want to switch to — so trefs clicks remain
preview-only.

**Class-aware actions**: configs marked `type.training_script*` get
**▶ Run**, **🔧 Overrides…**, **🗑 Clean Output…**, **📊 TensorBoard…**
buttons; when the config has checkpoints on disk, **🔮 Serve Inference…**
and **⚖ Evaluate…** also appear. Other classes (`type.model`,
`type.dataset`, etc.) only get **🔧 Overrides…**. Same filtering applies
to the right-click context menu on tree rows.

### Selection-driven detail panels

Clicking a leaf in the artifact sub-tree swaps the right pane to a
dedicated viewer — the tree is the single source of navigation truth:

- **Log** (`LogDetailPanel`) — tabs `TTY` (captured `tty.log`) and
  `Summary` (best loss, total steps, eval loss, perplexity, derived
  from `/api/run/summary`).
- **Checkpoint** (`CheckpointDetailPanel`) — step, size, world_size,
  saved timestamp, path, plus **🔮 Serve Inference…** and
  **⚖ Evaluate…** buttons pre-filled with this checkpoint's path.
- **Evaluation** (`EvalDetailPanel`) — results table (per-metric,
  per-sample if present) via `EvalResultTable`.

### Edit panel (tabbed editor)

Main-pane view that opens files for editing. Reached either by
clicking the ✎ Edit tab in the view switcher, the **✎ Edit** button
on a selected template in the Projects → templates view, the
**✎ Open** entry in the sidebar Files tree's right-click menu, or
the **📄 New Config…** / **📄 New Template…** flow under a project
context menu. All four routes hand the resulting absolute path to
`filesApi.openFile(path)` and switch the view to `edit`.

Per-buffer language is resolved by `webui/src/file-languages.ts`:
`.yaml` / `.yml` / `.jinja` / `.jinja2` use Forgather's custom
Monarch tokenizer; `.md` / `.markdown` use Monaco's built-in
markdown; `.py` uses built-in python; everything else falls back to
plaintext (so `.log`, `Makefile`, `LICENSE`, `.json`, `.toml`,
`.sh`, etc. all open and render — they just don't get
extension-specific syntax highlighting).

Click-to-open in the Files tree is *not* gated by extension.
`GET /api/template/source` does a binary-detection check on the
server (null-byte scan over the first 8 KiB plus a UTF-8 decode
attempt) and returns HTTP 415 for files that look binary. The
editor surfaces the 415's detail in-tab — clear "this isn't a
text file" instead of streaming garbage into Monaco.

State lives in `useFilesState` (`webui/src/files-state.ts`) — a single
hook owned by `App.tsx` so any caller can drop a file in regardless
of which view is currently visible. Buffers are keyed by absolute
path and shared across splits, so the same file open in two splits
stays in lock-step. The hook returns: `openFile`, `setContent`,
`saveFile`, `closeTab`, `closeOthers`, `closeAll`, `setActiveTab`,
`setActiveSplit`, `splitVertical`, `moveTab`, `isDirty`, and
`dropPath` (the last is a non-prompting close-everywhere used when
an external file op invalidates a path — rename / move / delete from
the Files tree).

**Render layout** (`components/FilesPanel.tsx`): a row of `SplitPane`s.
Each split has a tab bar (with one `FileTab` per open path, plus a ⊟
fork-vertical-split button) and a Monaco editor showing the active
buffer. Empty splits collapse automatically when their last tab moves
or closes (the layout always keeps at least one split). The dirty
indicator is the bullet next to the tab label.

**Save**: window-level Ctrl/Cmd+S handler installed during the panel's
`useEffect`, registered in capture phase so Monaco doesn't swallow the
key. Saves the active split's active tab via
`PUT /api/template/source` (atomic tmp+fsync+rename through
`_atomic.atomic_write_text`). The right-click context menu on a tab
or on the editor body offers Save / Close / Close Others / Close All —
Close-style actions confirm with `window.confirm` if any closing tab
is dirty.

**Drag/drop**: tabs are HTML5-draggable with the
`application/x-forgather-tab` MIME. Dropping on another tab inserts
before that tab; dropping on the spacer at the end of a tab bar
appends. Cross-split moves auto-collapse the source split if it
empties out and a peer is left.

**React-18 gotcha**: `setState(updater)` does not run the updater
synchronously. `openFile` decides whether to fire the
`api.templateSource(path)` fetch by reading `stateRef.current.buffers`
synchronously *before* calling `setState`, not by mutating a flag
inside the updater closure. (An earlier version did the latter and
the fetch never fired — the buffer appeared with `loading: true` and
stayed there.) Other places that need to read latest state from async
callbacks (`saveFile`) use the same `stateRef` snapshot.

**Backend**: `PUT /api/template/source` accepts `{path, content,
expected_mtime?}`, requires an absolute path to an *existing*
regular file (no create-new yet), and writes through
`_atomic.atomic_write_text`. Same trust posture as
`GET /api/template/source` — single-user localhost prototype, no
per-search-root containment check.

**Optimistic-concurrency: lost-update protection.** Every
`GET /api/template/source` returns the file's `os.path.getmtime`
as an `X-Mtime` response header. The editor stamps the buffer's
``baselineMtime`` from this header on load and after every
successful save. Save sends ``expected_mtime`` along with the
content; if the file's current on-disk mtime is newer (with a
1 µs tolerance for filesystem jitter), the server responds
**409** with `detail: {message, current_mtime, expected_mtime}`.
The client throws a typed `SaveConflictError`, the buffer keeps
its local content (no clobber), and `FilesPanel` opens a
`ConflictModal` showing the file path, both timestamps, and three
choices:

- **Overwrite** — `forceSaveFile(path)` re-PUTs without
  `expected_mtime` so the server skips the check.
- **Reload from disk** — `reloadFile(path)` re-GETs and replaces
  baseline + content + mtime; local edits are discarded.
- **Cancel** — `clearConflict(path)` dismisses the modal; the
  buffer stays dirty so the user can keep editing or retry.

`FileBuffer` carries `baselineMtime` and an optional
`conflict: {currentMtime}` flag; the modal watches every open
buffer and pops for the first conflicting one.

### Sidebar layout: top-level collapsible sections + footer bar

The sidebar's body below the header is a stack of independent
`<details>`-backed groups, all sharing the same chrome (uppercase
muted summary, custom `▸`/`▾` glyph via `::before`,
`::-webkit-details-marker { display: none }`) and all defaulting to
*closed* — first boot doesn't trigger any directory walks until
the user expands something. The bubbled `toggle` event is filtered
with `e.target === e.currentTarget` so nested `<details>` (project
rows, file-tree dirs) don't stomp on the outer section's open
state.

| Section | Component | Purpose |
| --- | --- | --- |
| **Views** | `<nav class="sidebar-views">` | The view switcher (📁 Projects, ✎ Edit, 🖥 GPUs, 📋 Queue, ⚙ Jobs, 🔮 Inference). |
| **Tools** | inline buttons | One-shot model-manipulation utilities: 📐 Evaluate, 🔁 Convert Model, 📦 Finalize Model, ⬆️ Update Model. |
| **Services** | inline buttons + `ServicesPanel` | Long-running spawned processes: 🔮 Inference, 🗂 Dataset, 📊 TensorBoard, 📖 MkDocs. Each launcher row carries a right-aligned running-count pill (same UI pattern as Views → Jobs) and, when there are configured instances of that type, a chevron that expands a per-type list of saved services with red/green dots and ▶/⏹/× controls. See [Auto-start services](#auto-start-services). |
| **Search Roots** | `SearchRootsPanel` | Root-list management: Browse… to add, × to remove, **📁 New Workspace…** for the dropdown-driven flow. Lifted out of `ProjectTree` so each group is its own top-level entry. |
| **Projects** | `ProjectTree` | The familiar workspace-clustered project forest. |
| **Files** | `FilesTree` | Hierarchical filesystem view of every search root. |

Below the scrolling section stack a **sidebar footer** is pinned via
`position: sticky; bottom: 0`. Four icon-only buttons:

- **⟳ Refresh data.** Invalidates the entire client query cache so
  disk edits to workspace metadata, templates, configs are picked
  up immediately. Moved here from the old sidebar header.
- **▶ / ⏸ Scheduler toggle.** Flips the dispatcher loop on/off
  (green when running, muted when paused). Same mutation that
  backed the old header button.
- **↺ Restart server.** Confirms, hits `POST /api/server/restart`,
  then polls `/api/health` and reloads the page once the rebooted
  server is responsive. Useful for picking up `server_config.yaml`
  changes without killing the terminal. Spawned jobs survive.
- **⚙ Open config.** Opens the loaded server config file
  (`server_config.yaml`) in the embedded editor. The path is
  surfaced by `GET /api/server-config-path`.

Earlier iterations had Tools and the view switcher visually
distinct from the rest (a horizontal rule above and below Tools, a
Tools-specific summary block). Those were dropped so the groups
read as a single uniform stack — easier to scan, no implicit
grouping where there isn't one. The Tools / Services split came
later to separate one-shot utilities (Evaluate / Convert / Finalize
/ Update) from persistent services (which gained the
configured-instance management above).

### Files tree (sidebar)

A hierarchical filesystem view of every configured search root,
letting users browse what's actually on disk and open files for
editing without knowing paths in advance. Component:
`webui/src/components/FilesTree.tsx`.

**Lazy loading.** Each root and each subdirectory is a *controlled*
`<details>` with React state (`useState(false)`) tracking open
state via `onToggle`, and the `<DirChildren>` listing pane is
**only rendered when the node is open**. Without this gate, React
mounts every `<details>`'s content regardless of the open
attribute, the inner `useQuery` fires immediately, and the entire
tree gets walked recursively on first paint. With the gate, opening
the Files section fetches only the search-roots list (one tiny
call); each root's listing fetches only when the user clicks it
open; the same applies to every nested directory.

Listings are cached under `["fs-browse", path, showHidden, true]`
(matching the same key the modal `DirectoryBrowser` uses, so cache
entries are shared and refreshes propagate). 30-second `staleTime`
keeps re-opens snappy.

Files render as clickable buttons regardless of extension — every
file gets a click-to-open. The backend's binary-detection in
`/api/template/source` (null-byte scan + UTF-8 decode check)
refuses truly binary files with HTTP 415 and the editor surfaces
the message in-tab, so the user gets a clear "this isn't a text
file" instead of garbage. Per-buffer language is resolved by
`languageFor(path)` (`file-languages.ts`): `.yaml`/`.yml`/`.jinja*`
→ Forgather Monarch tokenizer, `.md`/`.markdown` → Monaco markdown,
`.py` → Monaco python, everything else → plaintext (so `.log`,
`LICENSE`, `Makefile`, `.json`, `.toml`, `.sh` etc. all open fine).

A **Show hidden** checkbox at the top of the section toggles
dotfile visibility; the listing query key includes the flag so
toggling refetches.

**Right-click context menu** items (all conditional on the
target type):

| Item | Visible when | Action |
| --- | --- | --- |
| **✎ Open** | file | `filesApi.openFile(path)` + switch to Edit view |
| **➕ New File…** | dir | `POST /api/fs/new-file` (bare-name, refuses overwrite) — opens the new empty file in the editor |
| **➕ New Folder…** | dir | `POST /api/fs/mkdir` |
| **📁 New Workspace…** | dir under a search root | opens `InitWorkspaceModal` (see below) targeting the clicked dir |
| **📁 New Project…** | dir under an existing workspace | opens `NewProjectModal` with the enclosing workspace pre-resolved + the rel path from `workspace_root` pre-filled in `project_dir_name` |
| **✎ Rename…** | non-root | prompt for new bare basename → `POST /api/fs/rename` |
| **✂ Cut** | non-root | set in-memory clipboard `{path, mode: "cut"}` |
| **❏ Copy** | any | set clipboard `{path, mode: "copy"}` |
| **⎘ Paste** | dir, when clipboard set | `POST /api/fs/move` (cut, consumes clipboard) or `POST /api/fs/copy` with `auto_rename: true` (copy — collisions become `<stem> (copy)<ext>` siblings rather than 409 errors) |
| **⎘ Duplicate** | non-root | `POST /api/fs/copy` into the clicked node's parent with `auto_rename: true`; same "(copy)" suffix flow paste uses, no clipboard needed |
| **🗑 Delete Permanently…** | non-root | confirm + `POST /api/fs/delete-file` (file) or `POST /api/fs/delete-dir` (dir) |

The clipboard is in-memory (`useState` in `FilesTree`); no OS
clipboard interaction. Search roots themselves can't be Cut /
renamed / deleted via this menu — managing roots stays in the
**Search Roots** section.

After any rename / move / delete, the tree calls
`filesApi.dropPath(stale)` so any open editor tab pointing at the
now-stale path is dropped silently (no dirty-prompt). The user
saves before invoking the destructive op; if they didn't, the tab
is discarded without confirmation since the path is already gone
from disk.

**Init-workspace-here flow.** The Files-tree directory menu's
**📁 New Workspace…** opens a slimmer `InitWorkspaceModal` —
*not* the dropdown-driven `NewWorkspaceModal` from the Search-Roots
section — because the path is already determined by the
right-click target. The modal collects only metadata (name /
description / forgather dir / libs / additional search paths) and
the clicked dir becomes the workspace root directly. Backend
`POST /api/workspace/init-here` validates that the directory
exists, doesn't already contain `forgather_workspace/`, and lives
at-or-under a configured search root, then dispatches to
`ws_create_cmd` with the new `init_existing` flag — which skips
the original "must not exist" check + `os.makedirs(workspace_dir)`
and just writes the four metadata files into a new
`forgather_workspace/` subdir.

**Targeted cache invalidation.** Each create/rename/move/delete
op invalidates *only* the immediately-affected parent directory's
listing — keyed by `["fs-browse", parent]` with `exact: false`,
which prefix-matches just that path's variants (showHidden /
files_too). Sibling, ancestor, and unrelated subtrees aren't
touched. Combined with the lazy-mount above, creating a workspace
or project triggers exactly one listing refetch (the parent that
got the new entry), not a re-walk of everything currently visible.

**Backend** — endpoints under `routes/fs.py`, all sharing the same
safety posture as the existing `/fs/delete-file` (absolute path
required, no symlinks, ≥4 path components). None of these
non-destructive ops require a `confirmed` flag because each is
recoverable by reverse operation:

- `POST /api/fs/rename` `{path, new_name}` — `os.rename` to a
  bare basename; refuses overwrite (409).
- `POST /api/fs/copy` `{src, dest_dir, auto_rename?: bool, target_name?: string}`
  — `shutil.copy2` for files, `shutil.copytree` for directories.
  Without `auto_rename` a destination collision returns 409. With
  `auto_rename: true` the server picks a non-colliding sibling by
  appending ` (copy)` / ` (copy 2)` / … to the stem (used by paste
  and right-click Duplicate). `target_name` overrides the
  destination basename — single filename only, no path separators —
  used by the "Duplicate Config…" prompt to land the new file at
  the operator-chosen name.
- `POST /api/fs/move` `{src, dest_dir}` — `shutil.move` (so
  cross-device moves degrade to copy + unlink); refuses overwrite.
- `POST /api/fs/new-file` `{parent, name}` — `Path.touch()` an
  empty file; refuses overwrite.
- `POST /api/fs/mkdir` `{parent, name}` — single new directory
  (already existed; reused for + New Folder…).

### Markdown surfaces: Docs view + Project Info

Both the **Docs** view (`DocsPanel`) and the project tree's
**Info** tab (`InfoPane`) render markdown with `react-markdown +
remark-gfm + rehype-slug`. They share three behaviours worth
calling out:

- **Outline column.** A 220-px-wide nav rail to the left of the
  content lists every h1 / h2 / h3 by clicking the rendered DOM
  for `id`-stamped headings (rehype-slug stamps them) and
  rendering one entry per heading. Clicking smooth-scrolls the
  body to the matching anchor. Hidden entirely when the page has
  fewer than two headings.
- **Scroll restore.** The Docs view's Back button restores the
  scroll position of the page being returned to (the back-stack
  entry records `scrollTop` when pushed; the body re-applies it
  in a `requestAnimationFrame` after the content has rendered, so
  a saved offset doesn't get clamped to 0 by an empty body during
  a refetch). The Info tab applies the same trick across
  config-tab switches — it stays mounted with `display:none` so
  the scroll container survives, and its scrollTop is saved /
  restored from a ref.
- **Default landing page.** The Docs view lands on `docs/README.md`
  rather than the repo-root README — the docs index is the curated
  entry point with links to installation / tutorials / config / API,
  whereas the root README is closer to a project elevator pitch.
  Falls back to the root README if the docs index is missing.

`docs_hooks.py` is a MkDocs `on_page_markdown` hook (wired via
`mkdocs.yml: hooks:`) that rewrites relative markdown links on
pages whose source is a symlink. Many pages under `docs/` are
symlinks to canonical files elsewhere in the repo — e.g.
`docs/forgather-server.md → ../tools/forgather_server/README.md`.
MkDocs computes link paths from the docs_dir page location rather
than the source file's realpath, so relative links written from
the source author's perspective (`../../docs/foo.md`) come out
broken in the rendered site. The hook resolves each relative href
against the symlink target's realpath, then rewrites it as a path
relative to the docs_dir page; it also maintains a
`realpath → docs_dir alias` map so a link that lands on the
realpath of another docs symlink gets pointed at the in-tree alias
rather than ascending out of `docs_dir`.

### Persistent dynamic-args overrides

`/api/config/overrides` is a per-config JSON cache keyed by
`sha256(abspath(project_dir) + "\0" + config_name)`. Stored values are
layered as the *base* under any explicit kwargs and applied
automatically by `pp`, `output-dir`, `config/meta`, *and the trefs
graph* — so e.g. setting `--trainer-type=fsdp2` makes the trefs view
show `trainers/fsdp2_trainer.yaml` instead of the default. Submitting a
job auto-saves the values used; the **🔧 Overrides…** modal explicitly
sets/clears them.

### Submit / queue / scheduler

- ▶ **Run** opens a Submit modal with a generated form for the config's
  `[dynamic_args]` block. Schemas honor `type` (`int` / `str` / `float`
  / `bool` / `path`), `choices` (renders a dropdown), `action:
  store_true` / `store_false` (renders as a checkbox with concrete
  default), and `path` types (renders an inline file picker). The form
  pre-fills from the overrides cache.
- The Multi-node panel and the Dynamic arguments form are each in
  their own collapsible `<details>` block. With both open, neither
  takes more than 50% of the dialog body so a long Multi-node panel
  can't push the Dynamic args off-screen and vice versa. The
  participants table inside the Multi-node panel caps at ~9 rows
  and scrolls internally for the same reason.
- The form shows what `nproc_per_node` the config declares (`"gpu"` /
  fixed integer / `"cpu"` / `"auto"`) and warns when the user's GPU
  reservation count would mismatch a fixed worker count. These
  single-node-mode controls are hidden when the server is in cluster
  mode — the per-node GPUs column in the Multi-node panel takes their
  place. Priority stays visible across both modes.
- When cluster mode is active and the operator has only the local
  node selected (the implicit default), Submit goes through the
  regular single-node enqueue path and uses the panel's local-node
  GPUs value as the reservation count. Adding a peer flips Submit
  to the cluster fanout path; the button label changes to "Submit
  to N nodes" so the choice is explicit. The dialog refuses to
  submit if cluster mode is active and the operator has unselected
  every node.
- Last-used Multi-node settings (participants, per-node GPUs, iface,
  rdzv host/port, mismatch acknowledgement) persist alongside the
  dynamic-args overrides in the same per-config cache, so a config
  "opens where you left off" for both submit modes. **Reset to
  defaults** drops everything we cached for this config, including
  the Multi-node selection.
- The scheduler holds a JSON-backed queue + an in-memory dispatcher
  loop. **Enabled by default** so a freshly-restarted server resumes
  dispatch immediately. Pause anytime with the `▶`/`⏸` button in the
  sidebar header. The Queue view shows the current `running` /
  `paused` state.
- Dispatch picks idle GPU indices that aren't excluded via
  `CUDA_VISIBLE_DEVICES`, sets the child's `CUDA_VISIBLE_DEVICES` to
  the assignment, and invokes `torchrun` directly (mirrors what
  `forgather train` does, minus the extra subprocess layer — lets the
  scheduler own the process group for clean abort).

**Ten job types** share the queue, scheduler, GPU accounting, and TTY
capture machinery. The non-CUDA-by-default types (`tensorboard`,
`mkdocs`, `convert`, `finalize`, `update`, `dataset`, `dataset_server`)
accept `requested_gpus == 0`; the others default to at least one GPU.
Convert / finalize / update will happily take a GPU if the user sets
`--device cuda…` and bumps the reservation.

| Type             | Spawned by                                                                             | Lifecycle                              |
| ---------------- | -------------------------------------------------------------------------------------- | -------------------------------------- |
| `training`       | ▶ Run (Submit modal)                                                                   | Terminal when trainer exits.           |
| `eval`           | ⚖ Evaluate… (EvalModal, from config or checkpoint)                                     | Terminal when `forgather eval` exits.  |
| `inference`      | 🔮 Inference… (InferenceModal, project-backed or ad-hoc; sidebar Services)             | Long-lived; kill/force-kill to stop.   |
| `dataset_server` | 🗂 Dataset… (DatasetServerModal, sidebar Services)                                     | Long-lived; kill to stop.              |
| `tensorboard`    | 📊 TensorBoard… (TensorBoardModal, sidebar Services or per-config/per-model)           | Long-lived; kill to stop.              |
| `mkdocs`         | 📖 MkDocs… (MkDocsModal, sidebar Services — picks an `mkdocs.yml` + host:port)         | Long-lived; kill to stop.              |
| `convert`        | 🔁 Convert Model… (ConvertModal, sidebar Tools)                                        | Terminal when `convert` exits.         |
| `finalize`       | 📦 Finalize Model… (FinalizeModal, sidebar Tools)                                      | Terminal when `finalize` exits.        |
| `update`         | ⬆️ Update Model… (UpdateModal, sidebar Tools or config / checkpoint right-click)        | Terminal when `update` exits.          |
| `model`          | Run on a model config (config_class `type.model`)                                      | Terminal when `forgather model` exits. |
| `dataset`        | Run on a dataset config (config_class `type.dataset`)                                  | Terminal when `forgather dataset` exits.|

Helpers live in `inference_ops.py`, `eval_ops.py`, `tensorboard_ops.py`,
`mkdocs_ops.py`, `convert_ops.py`, `finalize_ops.py`, `update_ops.py`,
`model_ops.py`, `dataset_ops.py`, `dataset_server_ops.py` (build argv)
and `launcher.spawn_*_process` (same sandbox as training but with the
right argv). The scheduler's dispatcher branches on `item.job_type` to
pick the spawn function; GPU accounting and re-attach logic are
unchanged. Long-lived web services (inference, tensorboard, mkdocs,
dataset_server) all surface their URL as a clickable link on the Jobs
card so the operator can jump straight to the running endpoint.

**Dataset-source selector**. Every job type whose subprocess pulls
training examples (`training`, `eval`, `model`, `dataset`) gains a
dropdown in its submit modal that picks where the loader fetches
from: **Local** (the in-process loader, default) or any
dataset_server the forgather_server knows about (spawned-locally
JobRecords + URLs registered under *Datasets → Servers → + Add
server*). The choice persists alongside the other overrides; if the
saved server has gone away by the time the modal re-opens it snaps
back to Local. Resolved server-side into `FORGATHER_DATASET_SERVER`
+ `FORGATHER_DATASET_SERVER_TOKEN` env vars and merged into the
spawn's `extra_env`. Cluster fanout applies the same env vars to
every peer (the master resolves once and broadcasts).

### Scheduling algorithm

Each scheduler tick (~2 s) runs this placement logic:

1. **Build the queue.** Read `queue.json`, sort items by priority
   descending, then by submission time ascending (so higher-priority
   jobs go first; FIFO within a priority band).

2. **Build the idle pool.** Start from every GPU and drop any that are:
   - **excluded** via `CUDA_VISIBLE_DEVICES` (set at server start);
   - **disabled** at runtime via the UI toggle (persists in
     `gpu_policy.json`);
   - already **reserved** for one of our `starting` / `running`
     JobRecords.

   External processes (the user's desktop compositor, an unrelated
   CUDA program, a hybrid C+G daemon like
   `gnome-remote-desktop-daemon`) are *not* consulted. Trying to
   classify arbitrary processes as "real compute work" vs "desktop
   rendering" turned out to be a tar pit: NVIDIA's proprietary driver
   routes graphics-with-CUDA-context daemons through the compute
   list, hybrid C+G processes show up there too, and any name-based
   allowlist is incomplete by construction. The escape valve for
   "I'm running unrelated work on this GPU and don't want Forgather
   touching it" is the disable button on the GPU card. Compute and
   graphics processes are still surfaced via NVML
   (`nvmlDeviceGetComputeRunningProcesses` /
   `nvmlDeviceGetGraphicsRunningProcesses`) for display in the UI
   and to gate the kill-process endpoint (which restricts itself to
   compute processes so it can't terminate the user's desktop).

3. **Per-item eligibility.** For each queue item, filter the idle pool
   to GPUs whose `min_priority` gate the item clears
   (`gpu.min_priority <= item.priority`). An item can't land on a
   reserved GPU unless it qualifies.

4. **Best-fit to threshold** is the key heuristic. Within the eligible
   set, prefer GPUs with the *highest* `min_priority` the item still
   clears. Tie-break by index ascending (determinism). Formally, sort
   eligible indices by `(-gpu.min_priority, gpu.index)`.

   Rationale: if a priority-10 job could run on either `gpu0` (no
   gate) or `gpu5` (gated `min_priority=10`), put it on `gpu5`. That
   leaves `gpu0` free for a priority-0 job that *can't* use `gpu5`.
   Without this bias, the high-priority job would happily grab `gpu0`
   and block the low-priority job behind it — defeating the whole
   purpose of having reserved the higher-threshold GPU.

5. **Skip, don't block.** If an item can't be placed (fewer eligible
   GPUs than it requested), skip it and continue with the next item.
   A head-of-queue item that's over-constrained (e.g. wants 8 GPUs
   when only 4 are idle) does not block items behind it that would
   fit. Item ordering is stable across ticks, so the skipped item is
   reconsidered every tick until its resources free up.

6. **Commit.** Take the first `requested_gpus` indices of the sorted
   eligible list, re-sort them by index for readability, remove them
   from the in-tick idle pool, and launch the item (moves it from
   `queue_store` to `job_records`, spawns `torchrun` with
   `CUDA_VISIBLE_DEVICES` set to the chosen indices).

What the algorithm intentionally does **not** do:

- **No preemption.** A running job keeps its GPU until it finishes.
  Raising a job's priority or setting a GPU's `min_priority` doesn't
  kick anyone off.
- **No backfill across priority bands.** If the head of the queue is a
  4-GPU job that can't fit, a 1-GPU job further down with lower priority
  *can* run ahead of it (because of "skip, don't block"). If they have
  the *same* priority, FIFO order is preserved. There's no attempt to
  reserve GPUs for the blocked high-priority item while smaller ones
  run — that would require pool-reservation bookkeeping that isn't in
  scope for the prototype.
- **No NUMA / PCIe-topology awareness.** Multi-GPU assignments are just
  the first N eligible indices after the best-fit sort.
- **No cross-node scheduling.** Every GPU is assumed to be on the same
  node. The `node` field on JobRecords / GpuInfo is set up so a future
  `NodeClient` abstraction can be slotted in without changing the
  dispatch logic.

### Jobs / TTY

- **Jobs tab** unifies two sources: JobRecords we launched (status
  `starting` / `running` / `done` / `failed` / `aborted`) and
  externally-discovered trainer endpoints from `~/.config/forgather/jobs/`.
  Merged by PID lineage, tagged with `source = record | merged |
  endpoint`.
- Training-job cards show live status pills (loss, lr, grad_norm, epoch,
  tok/s, tokens, peak mem) plus a progress bar derived from
  `global_step / max_steps`. Non-training job types show a compact row
  with their identifying params (model path, port, etc.).
- Per-job control buttons forward to the trainer's HTTP endpoint:
  Save checkpoint / Save & stop / Graceful stop / Abort. **Kill** sends
  SIGTERM to the local process group (works for our jobs even
  pre-correlation). **Force kill** (right-click → "☠ Force kill
  (SIGKILL)") sends SIGKILL to the process group as a last-resort
  escape hatch for hung torchrun groups that won't respond to SIGTERM.
  Eval / inference / tensorboard jobs have no trainer-control endpoint,
  so only Kill / Force kill apply.
- **Bulk cleanup**: a `🧹 Cleanup completed` button at the top of the
  Jobs tab sweeps every terminal record (`done` / `failed` / `aborted`)
  via `POST /api/jobs/cleanup`. Captured TTY files are kept until the
  record is removed, so per-job `🗑` on a finished row still works too.
- **Dead endpoint visibility**: by default the Jobs list filters out
  endpoint-only entries whose PID is dead/zombie/recycled — those are
  trainer-control directories left behind by an earlier Forgather
  server instance. Toggle **Include dead endpoints** on the panel
  header to see them; right-click → **✕ Remove stale endpoint**
  rmtree's the directory under `~/.config/forgather/jobs/` so the entry
  stops surfacing. Live endpoint-only entries (foreign trainers) are
  still shown but offer no actions — those aren't ours to evict.
  Zombie-PID detection respects ``STATUS_ZOMBIE`` properly; a
  process that has exited but hasn't been reaped is treated as
  dead, not running.
- **Split-pane TTY**: toggle "⊞ Show TTY" to split the Jobs view; click
  a job to route its TTY output to the bottom pane. Draggable handle
  resizes (persisted to `localStorage`); double-click to reset to 45%.
- TTY stream subscribes to `WS /api/jobs/{id}/tty` — backlog then poll-
  follow. The backlog is read in 1 MiB chunks so a large log doesn't
  OOM the server; the one-shot REST dump (`GET /api/jobs/{id}/tty`) caps
  at the trailing 32 MiB of the captured file. Imperative
  `appendChild(textNode)` so browser text selection survives new chunks
  streaming in (lets you copy log lines from a running job). Once the
  trainer registers `logs_dir`, the captured TTY is symlinked into
  `<logs_dir>/tty.log` for durability alongside the trainer's other
  artifacts.
- Per-card hide/restart aware: server restart marks orphaned-but-still-
  alive processes as re-attached and continues monitoring them.

### Inference panel

An in-browser replacement for `forgather inf client` that talks to
running inference-server jobs (or any OpenAI-compatible endpoint).
Four sub-tabs sharing the same `InferenceState` (base URL, model,
generation params — persisted to `localStorage`):

- **Model** — base URL entry with a reachability test, picker for
  **Running inference servers** (auto-fills URL from inference job
  params), model-list fetch against `/models`, a **Generation
  parameters** form covering the OpenAI-named fields plus a wide
  selection of HuggingFace `GenerationConfig` extensions (`min_p`,
  `penalty_alpha`, `num_beam_groups`, `epsilon_cutoff`, etc.) with an
  expandable Advanced section. Tri-state selects let the user override
  `do_sample` / `early_stopping` explicitly rather than being stuck with
  temperature-derived defaults.

  In **cluster mode** the picker queries `/api/cluster/inference_servers`
  (master-aggregated) so jobs running on any reachable cluster peer
  appear alongside local ones — see the "Cluster inference picker"
  subsection below. The picker waits for the cluster-mode gate to
  resolve before showing rows, so it never flickers from local-only
  to the full cluster set on first paint. Outside cluster mode the
  picker queries `/api/jobs` directly. Rows that came from the cluster
  endpoint also show a short peer-id badge and a ⚠ indicator when the
  master's health-poll reports the server unhealthy.
- **Completion** — textarea + Send/Stop/Clear. Streams via
  `POST /v1/completions` (SSE) with an async iterator; `stream`
  checkbox falls back to a one-shot `stream: false` POST so beam-search
  and other streamer-incompatible modes work. Status line reports
  tokens + elapsed seconds; abort cancels the underlying fetch.
- **Chat** — multi-turn chat against `/v1/chat/completions`. Stateless
  wire format (client sends full `messages[]` each turn). Collapsible
  system-message disclosure at the top, transcript with `ReactMarkdown`
  for assistant turns and preserved-whitespace monospace for user turns,
  multi-line compose with Ctrl/Cmd+Enter to send. Regenerate-last,
  per-message edit (truncate + re-run), per-message delete. History
  + system text persist under `forgather-inference-chat-v1`.
- **Analyze** — per-token causal-LM scoring + visualization. Sends
  `echo=true, logprobs=K, max_tokens=0` to `/v1/completions`, which
  runs a single forward pass and returns per-token logprobs + top-K
  alternatives — see the inference server's "Per-Token Scoring"
  section. Split layout: textarea on the left, color-coded token
  output on the right, with optional histogram below. Both splits
  have drag handles (double-click to reset); state persists under
  `forgather-analyze-prefs`.
  - **Metric**: `loss` (default) or `entropy`. Entropy is a Forgather
    extension that needs the full vocab distribution, so the server
    returns it as a non-standard `token_entropies` field; if a
    non-Forgather server (vLLM, OpenAI) is in use the panel falls
    back to loss for coloring and shows a one-line note.
  - **Colormap**: viridis (default), plasma, magma, inferno, turbo,
    or a fallback green→red HSL ramp. Implemented via degree-6
    polynomial fits in `colormaps.ts` — no LUT, no dep. Foreground
    text color is auto-picked per-token from background luminance.
  - **Scale**: `auto` (5th/95th percentile of the response's own
    values, outlier-robust) or `manual` (fixed `min`/`max` so the
    same color encodes the same value across runs).
  - **Smooth**: exponential moving average over the color-encoded
    signal, α in `[0, 1)`. 0 = off; higher = smoother. Tooltip values
    stay raw — smoothing is purely a coloring aid.
  - **Histogram**: SVG (resolution-independent, `viewBox` + `width:
    100%`), 30 bins over the raw metric values. Bars are colored
    using the same colormap and scale so the histogram doubles as a
    legend.
  - **Selection-aware**: if the user has a non-empty selection in
    the textarea, Analyze scores only the selected substring instead
    of the whole input. Button label changes to `Analyze selection
    (N ch)` so the mode is unambiguous. Lets the user probe a single
    paragraph out of a pasted novel without truncation.
  - **Tooltip on hover**: shows the actual token, loss, perplexity,
    entropy (if available), and the ranked top-K alternatives with
    their probabilities. Rendered as `position: fixed` with
    flip-above/below + viewport-edge clamping so it escapes the
    pane's `overflow: auto` clip and never gets cropped.

**Inference… (sidebar Services section)** — opens `InferenceModal`
in ad-hoc mode: the model path becomes a `PathField` instead of a
read-only summary, so the user can serve any on-disk directory without
a Forgather project. Ad-hoc settings (path, port, dtype, attention
impl, cache impl, compile flags, chat template, checkpoint path)
persist under `forgather-adhoc-inference-v1` — the next invocation
defaults to the last-submitted values. Requested GPUs and priority
stay fresh each invocation since the "right" value depends on current
queue occupancy.

**Multi-model UX (ad-hoc only).** A "+ Add model" button under the
model PathField adds another row. Each row carries a path picker and
an optional name override (auto-derived from the basename when blank);
the picker remembers the parent of the last-picked path under
`pathfield.last:inference.model` so each row reopens at the same
directory the previous one came from. With more than one row the rows
live in a scrollable container (`max(240px, 30vh)` cap) so a long list
doesn't overflow the modal. A `Keep all models on GPU (no CPU swap)`
checkbox surfaces in the same UI; it maps to `--keep-on-gpu` on the
spawned server. The specific-checkpoint-path field is hidden in
multi-model mode (the server rejects `-c <PATH>` with more than one
model — the boolean `from_checkpoint` toggle still applies globally
and loads each model's latest checkpoint).

The "Create service…" suggested name is the single model's basename
in one-model mode, or `host-port` in multi-model mode — concatenating
many model names produces ugly long names quickly.

### Cluster inference picker

Mirror of the dataset-server cluster inventory in
`cluster_dataset_inventory.py`, sized for inference. Implemented in
`cluster_inference_inventory.py`:

- `LocalInference` (per-peer enumeration): scans `job_records` for
  `job_type == "inference"` in the `starting` / `running` state.
  Single-model jobs derive a one-element `models[]` from
  `model_path`'s basename; multi-model jobs use the configured `-m`
  names. `0.0.0.0` binds are rewritten to the cluster identity's
  hostname; pure-loopback binds stay in the list but are flagged
  `loopback=true`.
- Master aggregation: two async loops self-gating on master role:
  - `master_collect_servers_loop` every 10 s pulls each peer's
    `/api/cluster/inference_servers_local` and merges into a master
    snapshot.
  - `master_health_loop` every 10 s probes each server's `/health`
    (unauthenticated, root-mounted on the inference server) and
    updates the `healthy` flag.
- Endpoints (all under `/api/cluster/`):
  - `GET /inference_servers_local` — per-peer view, tokens included,
    peer-mTLS allowed (the master polls this).
  - `GET /inference_servers` — master-aggregated browser-facing list.
    On non-master nodes the route proxies to the master so every
    webui sees the same set. **Includes the bearer token** — see the
    discussion in the route's docstring; the picker carries the
    token via `X-Inference-Auth-Token` so the proxy can dial off-host
    upstreams from any cluster node.
  - `POST /inference_servers/refresh` — wake hook called by the
    scheduler on inference-job spawn / reap / abort so the picker
    reflects the change in ~1s rather than waiting for the 10s
    collect tick. Also surfaced manually via the webui (not currently
    wired to a button, but reachable for operator-driven refreshes).

Auth token surface: `auth.py:_PEER_ALLOWED_PATHS` carves out
`inference_servers_local` and `inference_servers`;
`auth.py:_PEER_ALLOWED_MUTATIONS` carves out
`inference_servers/refresh`.

Token-stripping the browser response (so remote-peer tokens never
leave the master) is tracked as a follow-up; the current behavior
matches `/api/jobs`, which already ships per-job tokens to the
authenticated session.

**Generation presets** — save/load named JSON presets of the current
generation params. Served by `/api/generation-configs/*`, which merges
two layers: bundled examples under `<repo>/generation_config/` (read-
only: `greedy`, `precise`, `balanced`, `creative`, `beam_search`,
`contrastive`) and user presets under `~/.config/forgather/generation_config/`
(writable; shadows same-named bundled entries). Delete on a built-in
returns 403 with guidance; delete on a user shadow restores the
built-in.

**Browser → inference-server proxy** (`routes/inference_proxy.py`) —
the webui can't hit spawned inference servers directly without running
into CORS / Private Network Access / extension-blocking. Everything
routes through same-origin `/api/inference/*`; the proxy forwards to
whichever base URL the caller names, streaming byte-for-byte so the
SSE framing reaches the browser unchanged. The proxy accepts any
HTTP/HTTPS host the operator types into the panel — forgather is a
single-user research tool, the proxy is auth-gated by the same token
that gates training-job submission, and an authenticated attacker
already has full RCE on the host (a job can shell out and exfiltrate
anything). An SSRF guard on this endpoint adds friction without
adding security. The expected workflow is "vLLM on another box"; the
proxy is built around that. For operators who want stricter posture
(e.g. forgather behind a multi-user gate), pass
`--lock-inference-proxy` to `forgather server` to restrict the proxy
to `127.0.0.1` / `localhost` / `::1`. The scheme guard (http/https
only) is unconditional regardless of lock state.

### Datasets view

Top-level webui tab (sidebar 🗂 **Datasets**) for inspecting and
managing the dataset_servers a training run might pull from. Two
sub-tabs sharing the local + user-added server lists. The
cluster-wide *Cluster* sub-tab was moved to the
**Cluster view → datasets** tab — this surface is intentionally
per-node only:

- **Servers** — left list of **Spawned dataset servers** (locally-
  launched JobRecords, auto-discovered) and **User-added servers**
  (URLs registered via *+ Add server*). Add/delete dialog for user
  entries; **Copy bundle** on each alive spawned row emits a
  `forgather-dataset://host:port/?token=…` URI to the clipboard,
  and the *+ Add server* modal has a matching **Paste bundle**
  affordance for one-step cross-host transfer.

  Selecting a server reveals three typed renderers loaded
  concurrently, with a single **↻ Refresh** button that re-fetches
  all three at once:
  - **Status** — colored policy chips (auth required/disabled, HF
    cache enabled/disabled, paths off/allowed, downloads off/
    allowed) with tooltips explaining each setting.
  - **HF Cache** — sortable table with a horizontal stacked
    size-distribution bar above it. Each split name in the splits
    cell is a clickable link that opens that split in Explore.
  - **Local** — same shape (table + chart + per-split click-
    through). Registered `local/<name>` mappings are enriched
    server-side with split metadata so the webui shows the same
    row counts / features / size info HF cache entries get.
- **Explore** — hierarchical tree (server → HF cache / local →
  repo → config → split) with a paged preview table on the right
  for the selected split. Tree is lazily expanded; click-to-expand
  individual rows in the preview table bumps the per-cell
  truncation cap. The browse pane has a draggable vertical
  divider — drag to resize, double-click to reset, ←/→ to nudge
  (Shift for x4); width persists in localStorage. Pager elides
  the middle (`‹ Prev 1 … 42 43 44 … 588 Next ›`) with a 🎲 button
  that jumps to a random page and expands a random row on it
  (handy for sampling a large corpus); 25 / 50 / 100 rows-per-page
  selector plus a **Go to** input for jumping directly to a page.

  Row expand-on-click ignores drag-select gestures: if the user
  moves the cursor more than ~4 px between mousedown and mouseup,
  or releases with a non-empty selection, the row stays expanded
  so they can copy text out of it.

  Each cell has a right-click context menu with:
  - **Copy cell text** — writes the full underlying value (not the
    truncated displayed form) to the clipboard via `navigator.
    clipboard.writeText`. Non-string values are JSON-stringified.
  - **Analyze in Inference…** — jumps to *Inference > Analyze*
    with this cell's full text already loaded into the textarea and
    scoring kicked off, one click. Wired through an App-level
    `pendingAnalyze: {text, key}` slot (mirrors the existing
    Cluster→Datasets `pendingExplore` pattern); the key nonce
    dedups so flipping tabs back doesn't re-fire stale requests.

  Cross-view click-through: clicking a row in the *Cluster view →
  datasets* tab opens this Explore tab with the first healthy host's
  first config/split pre-resolved and selected. If the chosen server
  doesn't have the dataset cached (or has no enumerable splits yet),
  the right pane shows a yellow `couldn't resolve` hint instead of
  silently appearing empty.

**Dataset… (sidebar Services section)** — opens the
DatasetServerModal: host, port, no-auth toggle, loading-policy
flags (`--no-hf`, `--allow-paths`, `--allow-downloads`), a
repeatable Local-mapping form (`name=path`), and an optional
config-file path. Spawned dataset_servers join the regular Jobs
view with the same URL + token surfacing inference jobs get. The
generated bearer token is **persisted** across restarts (mirroring
`forgather server`'s `auth_token`) so peers keep working after a
server reboot; pass `--regen-token` to the underlying script (or
re-spawn from this modal after deleting the per-port `.token` file)
to rotate.

**Edit Configuration… (right-click on Dataset…)** —
creates `<forgather_config_dir>/dataset_server/config.yaml` as a
commented YAML stub if it doesn't exist (0600 in a 0700 dir), then
opens it in the editor view. The standalone dataset_server loads
this file when no `--config` is passed.

**Browser → dataset_server proxy** (`routes/dataset_server.py`) —
same-origin proxy for the `/v1/*` endpoints. Unlike the inference
proxy (localhost-default), this proxy's SSRF allowlist is the user
registry itself: loopback always, registered URLs always,
everything else 403 with a "register first" hint. The registration
step is the explicit operator consent. See the module docstring
for the threat-model details, including the small bearer-
amplification it acknowledges.

### GPUs

- NVML-driven: per-card name, memory, util, temp, power, compute PIDs.
  Live updates via `WS /api/gpus/stream` (~2 s cadence, with REST
  prime).
- GPU↔job attribution: process chips on each GPU card map back to live
  jobs (chip turns blue + shows the config name when matched).
- **Three non-schedulable states**, visually distinct:
  - **Excluded** (red dashed border + `EXCLUDED` badge): filtered out
    via `CUDA_VISIBLE_DEVICES` at server start. Static.
  - **Disabled** (amber dashed border + `DISABLED` badge):
    runtime-toggled by the operator via the UI. Reversible, persists
    via `gpu_policy.json`.
  - **Priority-gated** (blue `≥N` pill): a minimum-priority threshold
    for scheduling. Only jobs with `priority >= N` get placed on the
    GPU. `0` means no gate.
- **Left-click a GPU card** toggles `disabled`. Excluded cards ignore
  clicks.
- **Right-click a GPU card** opens a context menu:
  - Enable/Disable GPU (same as left-click).
  - Set minimum priority… (prompt; integer validation).
  - Clear priority gate (shown when > 0).
  - ☠ Kill all N processes (SIGKILL) — last-resort cleanup for wedged
    ranks. Confirm dialog enumerates each PID and tags any that match
    one of our jobs (`pid 12345 (config_name)`). Hits **every** process
    on the GPU, including ones we didn't launch. Proceeds through
    `POST /api/gpus/{index}/kill` which requires `{confirmed: true}`.
- **Right-click a Job card** opens a context menu:
  - **☠ Force kill (SIGKILL)** for live server-launched jobs that
    aren't responding to SIGTERM — routes through a `force-kill`
    control action. Backend polls for the PID to actually exit
    (up to 2 s) and stamps the JobRecord's ``error`` field if it's
    still alive afterwards, so a stuck-in-CUDA process surfaces
    instead of silently leaving a phantom GPU consumer.
  - **✕ Remove stale endpoint** for endpoint-only entries whose
    PID is dead/zombie/recycled — backend rmtree's
    ``~/.config/forgather/jobs/job_<id>/`` so the entry stops showing up
    in the Jobs list. Live endpoint-only entries (foreign trainers
    we didn't launch) still show "No actions" — those aren't ours
    to evict. Toggle "include dead endpoints" on the Jobs panel
    header to see dead entries in the first place; the default
    view filters them out.

### Filesystem helpers

- Directory browser modal (used by Add Search Root, the `path`-type
  dynamic-args picker, and the New Workspace / New Project parent
  pickers) with quick-jump chips for Examples / Forgather repo / Home,
  supports show-hidden, navigate-by-double-click, click-to-pick on
  files, and a **+ New Folder** chip in the quick-row that calls
  `POST /api/fs/mkdir` on the current path and auto-navigates into
  the freshly-created directory. Bare-name validation server-side
  (no path separators, no `.`/`..`, no overwrite) keeps a single
  invocation to a single new directory.
- Asset endpoint with strict path-safety (resolved-target-must-stay-
  inside-project, `..` blocked, symlink containment check, 50 MiB cap)
  used to serve images embedded in the project README.

### Workspace creation

The **📁 New Workspace…** button in the Search Roots section
(alongside Browse…) opens `NewWorkspaceModal`, the in-app equivalent
of `forgather ws create`. Required: Parent (search-root dropdown,
auto-defaults to the first existing root), Name, Description, and
Forgather dir (auto-defaults to the bundled "Forgather repo"
quick-path). Optional: Workspace dir (relative to parent; nested
paths supported via `mkdir -p`; Browse… anchored to the chosen
parent lets the user pick an existing subdirectory and drops a
trailing-`/` relative path into the field), Libraries (newline-
separated, pre-filled with `base` + `examples` since every
workspace in the repo uses that pair), Additional search paths
(newline-separated absolute paths). The dropdown carries an extra
**+ Create new search root…** option that swaps in an inline
sub-form (existing parent dir + bare name); on submit the server
mkdirs the target and registers it as a search root in one shot
(`POST /api/search-roots {path, create: true}`), then auto-selects
it as the parent.

Submit calls `POST /api/workspace/new`, which validates that the
parent matches a configured search root exactly, slugifies the
workspace dir basename if not provided (CLI-matched: spaces -> `_`,
lowercased, dots stripped), splits and rejects any `..`/`.`
segments, runs an `os.path.commonpath` containment check against
the parent, then dispatches to `forgather.cli.workspace.ws_create_cmd`
via a `SimpleNamespace`. `os.makedirs` (called by the CLI) handles
intermediate-directory creation for nested paths.

Fresh workspaces appear in the project tree because discovery
walks for `forgather_workspace/` markers in addition to `meta.yaml`
projects (see "Project / config discovery" above) — empty
workspaces seed empty clusters that still render.

### Right-click context menus

The project tree exposes a different menu per node type:

- **Workspace row** — **📁 Create Project…** plus a trailing
  **🗑 Delete Workspace…**. Create-Project opens
  `NewProjectModal`, the in-app equivalent of
  `forgather project create`: required Name + Description, plus
  Config prefix (default `configs`), Default config (default
  `default.yaml`), Project dir (relative to workspace; may be
  nested with `mkdir -p` semantics; Browse… button anchored to
  `workspace_root` lets the user pick an existing subdirectory and
  drops the relative path back into the field with a trailing `/`
  for the leaf name), and an optional Copy-from `PathField` for
  seeding the default config from an existing file. Submit calls
  `POST /api/workspace/new-project`, which dispatches into
  `forgather.cli.project.project_create_cmd` via a
  `SimpleNamespace` so we don't duplicate the CLI's project-skeleton
  logic. Tree refresh is via `["projects"]` invalidation. The
  synthetic "Unaffiliated" cluster (no `workspace_root`) doesn't
  receive the menu. Delete-Workspace recursively removes the
  workspace directory via `POST /api/fs/delete-dir`, with the same
  two-step gate as Delete-Project (standard `confirm()` plus a
  typed-token prompt requiring the workspace's directory basename),
  since deleting a workspace cascades to every project, config, and
  in-tree output_models within it.
- **Project row** — **📄 New Config…** / **📄 New Template…**.
  Both open a `NewTemplateModal` (shares the chrome with
  `CleanOutputModal` et al.) with project / kind / base-dir summary
  rows, an auto-focused name input, an inline hint about the
  `.yaml` default suffix and subdirectory support, and a live
  preview of the absolute target path. Subdirectory creation under
  the configs / templates root is handled by typing a nested name
  (e.g. `experiments/foo.yaml`) — `mkdir -p` semantics on the
  server. The base path comes from
  `GET /api/project/template-paths` (`MetaConfig.searchpath[0]` for
  templates, plus `config_prefix` for configs). Submit calls
  `POST /api/project/new-template`, invalidates the project tree
  and `project-templates` queries so the new file shows up in
  `tlist`, then hands the returned path to the Edit panel via
  the App-level `onEditTemplate` hook — the user lands directly
  on a blank editor for the new file. A trailing **🗑 Delete
  Project…** entry recursively removes the project directory via
  `POST /api/fs/delete-dir`; it's gated by both a standard
  `confirm()` and a typed-token prompt requiring the user to type
  the project's directory basename, since the project tree often
  contains an `output_models/` subtree (runs / checkpoints) that
  the regular Clean Output flow won't touch. The confirm body
  spells out that outputs configured to live *outside* the project
  tree are not affected. After delete `["projects"]`,
  `["project-templates", dir]`, and `["project-models", dir]` are
  invalidated, and the active selection is dropped if it was
  pointing into the deleted project.
- **Config row** — Run / TensorBoard / Overrides plus, when the
  config has actually been run, Clean Output (gated on
  `configOutputDir`'s `output_dir_exists` — `output_dir` is
  per-config and can live anywhere on disk, so the menu polls the
  resolved path rather than guessing from `output_models/`).
  Serve Inference / Evaluate / Convert Model / Finalize Model
  surface when the config has checkpoints on disk. Convert and
  Finalize pre-fill the source path with the config's resolved
  `output_dir` while inheriting every other field from the global
  tool's persisted defaults; submit then writes everything
  (including the new source path) back, so the next opening —
  global tool or context-menu — reflects the last run. Items are
  filtered by `config_class` so non-training configs only show
  **Overrides**. **⎘ Duplicate Config…** prompts for the new
  filename (defaulting to `<stem> (copy)<ext>`) and copies the
  config file alongside the original via `POST /api/fs/copy` with
  `target_name`; the new entry appears in the tree immediately on
  `["projects"]` invalidation. A trailing **🗑 Delete Config…**
  entry unlinks just the config template file (via
  `POST /api/fs/delete-file`); it explicitly does *not* touch the
  config's `output_dir` / runs / checkpoints — those have their
  own Clean Output / Delete Permanently flows. After delete the
  `["projects"]` and `["project-templates", …]` queries are
  invalidated so the tree and the `tlist` view both refresh, and
  the active selection is cleared if it pointed at the deleted
  file.
- **Checkpoint leaf** — Serve Inference / Evaluate (both pre-fill the
  modal with this checkpoint's path), plus Delete Permanently.
- **Log leaf** / **Evaluation leaf** — Delete Permanently.
- **Logs / Checkpoints / Evaluations group header** — Delete All
  Permanently (atomic subdir deletion: one call to `/api/fs/delete-dir`
  on the parent directory rather than N per-leaf calls).

Destructive paths route through two sibling endpoints:
`POST /api/fs/delete-dir` (recursive directory removal, used by Clean
Output and the artifact-leaf / group menus) and
`POST /api/fs/delete-file` (single regular-file unlink, used by
Delete Config). Both require `confirmed: true`, reject symlinks,
require absolute paths, and enforce a ≥4-path-component depth floor;
the directory variant additionally checks against a denylist of
common system roots (`/`, `/home`, `/etc`, …) — the file variant
relies on the depth floor alone since you can't recursively wipe a
file.

## Not yet implemented

- Per-run metrics charts (loss curves, etc. — the data is already in
  `trainer_logs.json`; the UI just needs a renderer).
- Auto-rename or re-path of open editor buffers when the on-disk
  file is renamed / moved from the Files tree. Current behavior
  closes the stale tab silently — the user re-opens the new path
  from the tree.
- (CLI-only items mostly rolled into the UI: `forgather ws create`
  is now the New Workspace… button under Search Roots,
  `forgather project create` is the workspace context menu, and
  per-config / per-template creation is the project context menu.)
- Multi-node deployment. Today's design tags each GPU and JobRecord
  with a `node` identifier and concentrates the "this could be remote"
  surfaces in `gpu_monitor.py` / `launcher.py` / `scheduler.py`, so the
  future seam is a `NodeClient` abstraction in front of those modules.

---

## Directory layout

```
src/forgather/cli/
├── server.py                  # CLI shim: `forgather server` → backend subprocess
└── wrappers_args.py           # CLI parser registration for `server`

generation_config/             # Bundled generation-parameter presets
│                              #   (read-only from the UI; shadowed by
│                              #    ~/.config/forgather/generation_config/)
├── greedy.json
├── precise.json
├── balanced.json
├── creative.json
├── beam_search.json
└── contrastive.json

tools/forgather_server/
├── server.py                  # uvicorn entry point
├── app.py                     # FastAPI app factory + lifespan (dispatcher loop)
├── paths.py                   # ~/.config/forgather/server/ state helpers
├── _atomic.py                 # Crash-atomic file-write helpers
│                              #   (tmp + fsync + os.replace)
├── search_roots.py            # JSON-backed search-root list, default seeding
├── discovery.py               # Walk roots → cluster projects by workspace
├── models_catalog.py          # Enumerate per-project output_dirs, runs,
│                              #   checkpoints, evaluations
├── config_ops.py              # Wrappers around ConfigEnvironment, with
│                              #   per-config overrides auto-applied
├── overrides_store.py         # Per-config dynamic-args override cache
├── queue_store.py             # Persistent FIFO queue (waiting items only)
├── job_records.py             # Persistent records of dispatched jobs
├── launcher.py                # Spawn training / eval / inference /
│                              #   tensorboard / mkdocs / convert /
│                              #   finalize / update / model / dataset
│                              #   processes; own process group
├── inference_ops.py           # Build inference-server argv
├── eval_ops.py                # Build `forgather eval` argv
├── tensorboard_ops.py         # Build tensorboard argv
├── mkdocs_ops.py              # Build `mkdocs serve` argv
├── convert_ops.py             # Build `forgather convert` argv
├── finalize_ops.py            # Build `forgather finalize` argv
├── update_ops.py              # Build `forgather update` argv
├── model_ops.py               # Build `forgather model` argv
├── dataset_ops.py             # Build `forgather dataset` argv
├── scheduler.py               # Dispatcher loop, GPU allocation,
│                              #   per-job-type spawn, re-attach, reap, abort
├── gpu_monitor.py             # NVML / torch.cuda enumeration,
│                              #   CUDA_VISIBLE_DEVICES allow-list
├── gpu_policy.py              # Runtime per-GPU policy (disabled,
│                              #   min_priority) — persisted
├── routes/
│   ├── search_roots.py        # GET/POST/DELETE /api/search-roots
│   ├── projects.py            # /api/projects, /api/project/{readme,asset}
│   ├── configs.py             # /api/config/{raw,pp,trefs,meta,templates,
│   │                          #               overrides,output-dir} +
│   │                          #   /api/template/source
│   ├── models.py              # /api/project/models, /api/model/{runs,
│   │                          #   checkpoints,evaluations}, /api/run/{tty,
│   │                          #   summary}, /api/eval-configs
│   ├── fs.py                  # /api/fs/{browse,quick-paths,delete-dir}
│   ├── gpus.py                # /api/gpus + WS /api/gpus/stream + kill
│   ├── jobs.py                # /api/jobs (unified), control, TTY (REST + WS),
│   │                          #   cleanup
│   ├── queue.py               # /api/queue + /api/queue/scheduler +
│   │                          #   /api/config/dynamic-args
│   ├── inference_proxy.py     # /api/inference/{health,models,completions,
│   │                          #   chat/completions} — same-origin SSE proxy
│   └── generation_configs.py  # /api/generation-configs/{list,get,put,delete}
└── webui/
    ├── package.json           # Vite, React, TypeScript, Monaco, viz-js,
    │                          #   TanStack Query, react-markdown, remark-gfm
    ├── vite.config.ts         # dev-mode /api → :8765 proxy (REST + WS)
    └── src/
        ├── main.tsx           # React + QueryClientProvider bootstrap
        ├── App.tsx            # Collapsible sidebar (header, Views,
        │                      #   Tools, Services, Search Roots,
        │                      #   ProjectTree, FilesTree, sticky
        │                      #   footer) + main pane; owns view /
        │                      #   selection / tab state and the
        │                      #   scheduler play/pause
        ├── api.ts             # Typed fetch wrappers for every endpoint
        ├── inference-client.ts# Browser client for /v1/* (via the proxy);
        │                      #   streamCompletion / streamChatCompletion /
        │                      #   runCompletion / runChatCompletion +
        │                      #   shared SSE loop
        ├── forgather-syntax.ts # Monaco Monarch tokenizer
        ├── file-languages.ts  # Extension -> Monaco language id;
        │                      #   plaintext fallback for unknown
        │                      #   types — every file is openable
        │                      #   subject to the backend binary check
        ├── files-state.ts     # useFilesState hook: open buffers, splits,
        │                      #   tabs, save (Ctrl+S), drag-drop reorder,
        │                      #   dropPath (silent close-everywhere)
        ├── styles.css
        └── components/
            ├── ProjectTree.tsx      # Sidebar tree + per-config artifact
            │                        #   sub-groups; context menus
            ├── DirectoryBrowser.tsx
            ├── PathField.tsx        # Text input + Browse… picker
            ├── ContextMenu.tsx      # Generic floating menu
            ├── ConfigViewer.tsx     # Tabs: info / pp / templates
            ├── InfoPane.tsx         # Markdown renderer (GFM + image proxy)
            ├── TemplatesView.tsx    # `templates` tab container: trefs/tlist
            │                        #   mode bar, shared right-pane preview,
            │                        #   right-click → Open in Editor
            ├── DynamicArgsForm.tsx  # Shared form for Submit + Overrides
            ├── SubmitModal.tsx      # Enqueue training job
            ├── OverridesModal.tsx   # Set/reset persistent dynamic-args
            ├── CleanOutputModal.tsx # Delete output_dir / models_dir
            ├── EvalModal.tsx        # Enqueue eval job
            ├── NewProjectModal.tsx  # forgather project create flow:
            │                        #   name/description + CLI-matched
            │                        #   defaults + copy-from picker;
            │                        #   nested project_dir via Browse…
            │                        #   anchored at the workspace root
            ├── NewWorkspaceModal.tsx# forgather ws create flow: parent
            │                        #   search-root dropdown (with
            │                        #   inline + Create new search
            │                        #   root… sub-form), nested
            │                        #   workspace dir, libs/search
            │                        #   paths textareas
            ├── InitWorkspaceModal.tsx# Init workspace in an existing
            │                        #   directory — slimmer modal for
            │                        #   the Files-tree right-click flow:
            │                        #   path is fixed, only metadata
            │                        #   is collected
            ├── NewTemplateModal.tsx # New Config / New Template prompt
            │                        #   with live target-path preview
            ├── SearchRootsPanel.tsx # Top-level Search Roots sidebar
            │                        #   group; root list + Browse… +
            │                        #   📁 New Workspace…
            ├── InferenceModal.tsx   # Enqueue inference-server job
            │                        #   (project-backed or ad-hoc)
            ├── TensorBoardModal.tsx # Enqueue tensorboard job
            │                        #   (config-backed; or `global`
            │                        #   from sidebar Services)
            ├── MkDocsModal.tsx      # Enqueue `mkdocs serve` job
            │                        #   (sidebar Services — global only)
            ├── ConvertModal.tsx     # Enqueue `forgather convert` job
            │                        #   (sidebar Tools or config / checkpoint
            │                        #   right-click)
            ├── FinalizeModal.tsx    # Enqueue `forgather finalize` job
            │                        #   (sidebar Tools or config / checkpoint
            │                        #   right-click)
            ├── UpdateModal.tsx      # Enqueue `forgather update` job
            │                        #   (sidebar Tools or config / checkpoint
            │                        #   right-click; pre-fills source path
            │                        #   and optional checkpoint)
            ├── ServicesPanel.tsx    # Configured-service rows in the
            │                        #   sidebar Services group (red/green
            │                        #   dots, ▶/⏹/× row controls,
            │                        #   click-through per type)
            ├── LogDetailPanel.tsx   # Selection target for a run/log leaf
            ├── CheckpointDetailPanel.tsx # Selection target for a checkpoint
            ├── EvalDetailPanel.tsx  # Selection target for an evaluation
            ├── RunSummaryView.tsx   # Extracted from legacy models panel
            ├── EvalResultTable.tsx  # Extracted from legacy models panel
            ├── InferencePanel.tsx   # Inference view: model/completion/chat
            │                        #   sub-tabs (Inference launcher lives
            │                        #   in the sidebar Services section)
            ├── InferenceModelPanel.tsx     # Base URL, params, presets
            ├── InferenceCompletionPanel.tsx# Textarea completion + Stream
            ├── InferenceChatPanel.tsx      # Multi-turn chat + markdown
            ├── GpuPanel.tsx         # Live GPU cards; PID→job attribution
            ├── JobsPanel.tsx        # Unified jobs list + split-pane TTY
            │                        #   + bulk cleanup
            ├── TtyViewer.tsx        # Imperative-append terminal
            ├── QueuePanel.tsx      # Queue list + scheduler status
            │                        #   (toggle lives in the sidebar)
            ├── FilesTree.tsx        # Sidebar filesystem tree per search
            │                        #   root; in-memory clipboard for
            │                        #   Cut / Copy / Paste; right-click
            │                        #   → Open / Rename / Delete
            └── FilesPanel.tsx       # Editor with tabbed splits, drag-drop
                                     #   reorder, Save / Close context menu;
                                     #   per-file Monaco language via
                                     #   file-languages.ts
```

## Architecture in one paragraph

The backend is a thin FastAPI app that wraps Forgather's existing Python
APIs — no re-implementation. Every endpoint ultimately calls into
`MetaConfig`, `ConfigEnvironment`, the `forgather.cli.trefs` renderers,
or `TrainerControlClient`. Config materialization respects per-config
override values pulled from a JSON cache, so `pp` / `trefs` /
`output-dir` / `config/meta` all reflect whatever the user has set in
the 🔧 Overrides modal. The scheduler dispatches ten job types —
training (`torchrun`), eval (`forgather eval`), inference
(`tools/inference_server/server.py`), TensorBoard (`tensorboard`),
MkDocs (`mkdocs serve`), convert (`forgather convert`), finalize
(`forgather finalize`), update (`forgather update`), model, and
dataset — all through a common `launcher.spawn_*`
surface that owns its process group via `start_new_session=True` so
jobs survive server restart. Inference
servers spawned this way appear in the Inference panel's "Running
inference servers" picker; the browser talks to them through a
same-origin SSE proxy so CORS / PNA don't get in the way. The frontend
is a Vite/React SPA driven by TanStack Query for caching + background
refresh; persistent server state is plain JSON files under
`~/.config/forgather/server/` so it's inspectable with ordinary tools.

## API quick reference

All endpoints are under `/api`. JSON unless noted. Endpoints marked WS
are WebSockets.

### Discovery

| Endpoint                                       | Purpose                                             |
| ---------------------------------------------- | --------------------------------------------------- |
| `GET /api/health`                              | Liveness                                            |
| `GET /api/server-config-path`                  | Resolved path to the loaded `server_config.yaml` (`{path}` — used by the sidebar gear button) |
| `POST /api/server/restart`                     | Schedule an in-place `os.execv` restart; running subprocesses survive. Returns `{restart: "scheduled"}` immediately, then the process re-execs after a short delay so the response body can flush. |
| `GET /api/search-roots`                        | List search roots                                   |
| `POST /api/search-roots` `{path, create?: bool}`| Add a search root; with `create: true` the server `mkdir`s the path before registering (used by the New Workspace modal's inline create-root flow) |
| `DELETE /api/search-roots?path=`               | Remove a search root                                |
| `GET /api/projects`                            | Workspace-clustered project tree                    |
| `GET /api/project?project_dir=`                | Single-project detail                               |
| `GET /api/project/readme?project_dir=`         | README.md as markdown                               |
| `GET /api/project/asset?project_dir=&asset=`   | Image / file embedded in the README (path-guarded)  |
| `GET /api/project/templates?project_dir=`      | Every template on the project's search path, grouped by search-root category (with synthetic Meta group for `meta.yaml`) — backs the `tlist` view |
| `GET /api/project/template-paths?project_dir=` | Resolved `templates_dir` + `configs_dir` + `config_prefix` (for the New Config / New Template modal's path preview) |
| `POST /api/workspace/new-project` `{workspace_dir, name, description, config_prefix?, default_config?, project_dir_name?, copy_from?}` | Create a project under a workspace — wraps the CLI's `project_create_cmd`; nested `project_dir_name` (`a/b/c`) supported; refuses overwrite, returns absolute project_dir |
| `POST /api/workspace/new` `{parent_dir, name, description, workspace_dir_name?, forgather_dir, libs?, search_paths?}` | Create a workspace under a search root — wraps `ws_create_cmd`; parent must be a configured search root; nested `workspace_dir_name` supported; returns absolute workspace_dir |
| `POST /api/workspace/init-here` `{workspace_dir, name, description, forgather_dir, libs?, search_paths?}` | Initialize a workspace in an *existing* directory — used by the Files-tree right-click flow. Refuses if `forgather_workspace/` already exists; requires `workspace_dir` to live at-or-under a configured search root. |
| `POST /api/project/new-template` `{project_dir, kind: "config"\|"template", name}` | Create an empty file under the templates dir; refuses overwrite, `.yaml` auto-appended, returns absolute path |

### Config inspection

| Endpoint                                                                         | Purpose                                                        |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/config/raw?path=`                                                      | Raw config source                                              |
| `GET /api/config/pp?project_dir=&config=`                                        | Preprocessed YAML (overrides applied)                          |
| `GET /api/config/trefs?project_dir=&config=&format=json\|dot\|tree`              | Template dependency graph (overrides applied)                  |
| `GET /api/config/templates?project_dir=&config=`                                 | Flat list of consumed templates                                |
| `GET /api/config/meta?project_dir=&config=`                                      | `config_name` / `config_description` / `config_class`          |
| `GET /api/config/output-dir?project_dir=&config=`                                | Resolved `output_dir` + `models_dir`, sizes, `nproc_per_node`  |
| `GET /api/config/dynamic-args?project_dir=&config=`                              | Form schema for the submit / overrides UI                      |
| `GET /api/config/overrides?project_dir=&config=`                                 | Cached override values for this config                         |
| `POST /api/config/overrides` `{project_dir, config, values}`                     | Set / replace cached overrides                                 |
| `DELETE /api/config/overrides?project_dir=&config=`                              | Clear cached overrides                                         |
| `GET /api/template/source?path=`                                                 | Raw source of any template; `X-Mtime` response header carries the file's mtime so the editor can detect concurrent edits |
| `PUT /api/template/source` `{path, content, expected_mtime?}`                    | Write template content (atomic; path must exist). When `expected_mtime` is given, returns 409 with `{message, current_mtime, expected_mtime}` if the file is newer on disk; pass null/omit to force-overwrite. Successful response includes the new `mtime`. |

### Models / runs / checkpoints / evaluations

Populates the project-tree sub-groups and detail panels:

| Endpoint                                                 | Purpose                                                        |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/project/models?project_dir=`                   | Per-`output_dir` summary (configs, run/checkpoint/eval counts) |
| `GET /api/model/runs?output_dir=`                        | Run entries with timestamps and log paths                      |
| `GET /api/model/checkpoints?output_dir=`                 | Checkpoints (step, size, world_size, manifest)                 |
| `GET /api/model/evaluations?output_dir=`                 | Evaluations + results summary                                  |
| `GET /api/run/summary?run_dir=`                          | Trainer-log statistics (best loss, steps, perplexity, …)       |
| `GET /api/run/tty?run_dir=`                              | `tty.log` tail (one-shot)                                      |
| `GET /api/eval-configs`                                  | Discoverable eval configs for the EvalModal dropdown           |

### Filesystem

| Endpoint                                                | Purpose                                                  |
| ------------------------------------------------------- | -------------------------------------------------------- |
| `GET /api/fs/browse?path=&show_hidden=&files_too=`      | Directory listing (dirs only by default)                 |
| `GET /api/fs/quick-paths`                               | Named quick-jump shortcuts                               |
| `POST /api/fs/delete-dir` `{path, confirmed: true}`     | Delete a directory (multiple safety guards; see code)    |
| `POST /api/fs/delete-file` `{path, confirmed: true}`    | Delete a single regular file (depth floor + symlink reject; used by Delete Config) |
| `POST /api/fs/mkdir` `{parent, name}`                   | Create a single new directory under `parent`; bare-name (no separators), refuses overwrite — used by DirectoryBrowser's + New Folder chip |
| `POST /api/fs/rename` `{path, new_name}`                | Rename a file or directory in place (bare basename); refuses overwrite — used by the sidebar Files tree |
| `POST /api/fs/copy` `{src, dest_dir}`                   | Copy a file (`shutil.copy2`) or directory (`shutil.copytree`) to `dest_dir/basename(src)`; refuses overwrite — used by the Files tree's Paste-after-Copy |
| `POST /api/fs/move` `{src, dest_dir}`                   | Move a file or directory to `dest_dir/basename(src)` via `shutil.move`; refuses overwrite — used by the Files tree's Paste-after-Cut |
| `POST /api/fs/new-file` `{parent, name}`                | Create an empty file at `parent/name`; bare-name, refuses overwrite — used by the Files tree's New File… affordance |

### GPUs

| Endpoint                                                    | Purpose                                                                          |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `GET /api/gpus`                                             | One-shot snapshot                                                                |
| `WS /api/gpus/stream`                                       | Push updates every ~2 s                                                          |
| `GET /api/gpus/policy`                                      | All per-GPU runtime policies (`{index: {disabled, min_priority}}`)               |
| `POST /api/gpus/{index}/policy` `{disabled?, min_priority?}` | Upsert per-GPU policy; unset fields are left alone                              |
| `POST /api/gpus/{index}/kill` `{confirmed: true}`           | SIGKILL every compute process on the GPU (returns `{pids, killed, failed}`)      |

### Cluster (multi-node, opt-in via `--cluster`)

Endpoints in this group return empty / null payloads when the server
is in standalone mode (no `--cluster` flag), so a webui that polls
them is safe to mount unconditionally.

| Endpoint                                                           | Auth                       | Purpose                                                                    |
| ------------------------------------------------------------------ | -------------------------- | -------------------------------------------------------------------------- |
| `GET /api/cluster/self`                                            | bearer / peer              | This node's identity, or `null` if standalone                              |
| `GET /api/cluster/members`                                         | bearer / peer              | Cluster name, master node_id, full member table                            |
| `GET /api/cluster/master`                                          | bearer / peer              | Current master_node_id and `is_self_master`                                |
| `GET /api/cluster/gpus_local`                                      | bearer / peer              | This node's GPU snapshot. Returns `X-Forgather-Node-Id` header for sanity-checking peer responses |
| `GET /api/cluster/gpus`                                            | bearer                     | Aggregated `{nodes: [{node_id, hostname, address, reachable, gpus, error}]}` across the cluster (master fetches each peer's `gpus_local` in parallel) |
| `POST /api/cluster/gpu_policy_local` `{gpu_index, disabled?, min_priority?}` | bearer / peer (only mutation path carved out for peers) | Apply a GPU policy update on this node |
| `POST /api/cluster/nodes/{node_id}/gpus/{idx}/policy` `{disabled?, min_priority?}` | bearer | Master-side proxy: forward a GPU policy update to the named node (short-circuits self) |
| `GET /api/cluster/bandwidth_local?bytes=N`                         | bearer / peer              | Legacy HTTPS data path. Streams `N` bytes back so the caller can time the receive (default = probe size; capped at 4 GiB). Superseded by the raw-TCP path below for the live tab — left in place for ad-hoc / CLI use. |
| `POST /api/cluster/bandwidth_prep` `{bytes}`                       | bearer / peer              | Open a one-shot ephemeral raw-TCP listener for a single bandwidth-test transfer. Returns `{port, bytes, token}` where `token` is a fresh 32-byte hex handshake the caller sends first; mismatched tokens are dropped without serving. Listener self-closes after one served connection (or 30 s timeout). |
| `GET /api/cluster/bandwidth`                                       | bearer                     | Cached pairwise bandwidth measurements (1 h TTL)                          |
| `POST /api/cluster/bandwidth/refresh`                              | bearer                     | Run a fresh adaptive parallel-stream bandwidth measurement against every reachable peer (sequential across peers, parallel streams per peer) and update the cache |
| `POST /api/cluster/bandwidth/refresh_one/{node_id}`                | bearer                     | Re-run the bandwidth probe against one peer. Used by the per-peer "Measuring…" progress feedback in the webui. |
| `GET /api/cluster/latency_local`                                   | bearer / peer              | Empty 200 with a node-id header — peer endpoint for RTT round-trip timing |
| `GET /api/cluster/latency`                                         | bearer                     | Cached pairwise latency measurements (1 h TTL). Each entry carries min / median / max ms across `samples` post-warmup probes. |
| `POST /api/cluster/latency/refresh`                                | bearer                     | Run a fresh latency probe against every reachable peer and update the cache |
| `POST /api/cluster/latency/refresh_one/{node_id}`                  | bearer                     | Re-run the latency probe against one peer.                               |
| `POST /api/cluster/jobs/submit` `{project_dir, config, dynamic_args?, priority?, members:[{node_id,nproc_per_node,nccl_socket_ifname?}], rdzv_node_id?, rdzv_port?, allow_version_mismatch?}` | bearer | Submit a multi-node training bundle; master fans out per-rank queue items to each participant. Auto-derives the iface from each member's advertised IP when `nccl_socket_ifname` is omitted. Returns the bundle and any version-mismatch warnings. HTTP 422 if no iface can be matched, 409 on unacknowledged version mismatch. |
| `GET /api/cluster/jobs`                                            | bearer / peer              | List multi-node bundles with rolled-up status. Non-master nodes proxy to master so every webui sees the same list. Peer-allowed because the response is read-only and cluster-wide by definition. |
| `GET /api/cluster/jobs/{id}`                                       | bearer                     | Get one bundle (with rolled-up status, fanned out from master)            |
| `POST /api/cluster/jobs/{id}/cancel`                               | bearer                     | Fan out cancel to every participant of the bundle                         |
| `POST /api/cluster/training_local` `{project_dir, config, dynamic_args?, requested_gpus, priority, rdzv_args, extra_env, cluster_job_id?}` | bearer / peer (only mutation path carved out for peers) | Per-rank training enqueue used by the master fanout. The peer's scheduler picks up the queue item and spawns torchrun in rdzv mode. |
| `POST /api/cluster/training_cancel_local` `{queue_id}`             | bearer / peer              | Per-rank cancel used by the master cancel-fanout                          |
| `GET /api/cluster/training_status_local?queue_id=...`              | bearer / peer              | Per-rank job-status snapshot used by the master to roll up cluster-job status. Read-only, scoped to one queue_id. |
| `GET /api/cluster/issue_url_token`                                 | bearer / peer              | Mint a 60 s single-use URL token for cross-node SSO. Distinct from the persistent bearer; consumed by `verify_url_token` on first `/api/auth/login`. 503 when cluster mode is not active on this node. |
| `POST /api/cluster/peer_session` `{node_id}`                       | bearer                     | Look up the named peer, fetch its `issue_url_token` over mTLS, return `{url: "https://addr:port/?token=…", hostname}` for the browser to open in a new tab. Refuses self (400) and unreachable peers (503). |

The probe payload (versions + interfaces + CPU summary) is
piggybacked on every member entry returned by `/api/cluster/members`
under the ``probe`` field. There is no separate `/api/cluster/probe`
endpoint — peer-pull already brings the data with no extra
round-trip.

The "peer" auth column means a known cluster member presenting a
CA-signed client certificate (mTLS) can call the endpoint without
the bearer token; see [Cluster mode (multi-node, prototype)](#cluster-mode-multi-node-prototype)
for the threat model.

### Queue / scheduler

| Endpoint                                                  | Purpose                                                |
| --------------------------------------------------------- | ------------------------------------------------------ |
| `GET /api/queue`                                          | List queued items                                      |
| `POST /api/queue` `{project_dir, config, dynamic_args, requested_gpus, priority, job_type?, job_params?, dataset_source?}` | Enqueue any job type (`training` / `eval` / `inference` / `dataset_server` / `tensorboard` / `mkdocs` / `convert` / `finalize` / `update` / `model` / `dataset`). `dataset_source` is `{kind:"local"}` or `{kind:"server", server_id:"local:<queue_id>"|"user:<entry_id>"}`; resolved into `FORGATHER_DATASET_SERVER[_TOKEN]` env vars and merged into `job_params.extra_env` for training-shaped types. |
| `DELETE /api/queue/{queue_id}`                            | Cancel a queued item (or abort if it's already running) |
| `GET /api/queue/scheduler`                                | Dispatcher on/off + counters                           |
| `POST /api/queue/scheduler` `{enabled}`                   | Enable / disable the dispatcher                        |

### Jobs (unified: launched + discovered)

| Endpoint                                                              | Purpose                                                    |
| --------------------------------------------------------------------- | ---------------------------------------------------------- |
| `GET /api/jobs?include_dead_endpoints=`                               | Merged list of JobRecords + endpoint discoveries           |
| `GET /api/jobs/{id}/status`                                           | Trainer-side `/status` proxy (step, loss, etc.)            |
| `POST /api/jobs/{id}/control/{save\|stop\|save-stop\|abort\|kill\|force-kill}` | Trainer control commands; `kill`=local SIGTERM, `force-kill`=local SIGKILL |
| `DELETE /api/jobs/{id}`                                               | Remove a terminal JobRecord from history                   |
| `POST /api/jobs/cleanup`                                              | Bulk-remove every terminal JobRecord (`done` / `failed` / `aborted`) |
| `POST /api/jobs/gc`                                                   | Sweep orphan TTY files from `~/.config/forgather/server/jobs/`    |
| `GET /api/jobs/{id}/tty`                                              | Full captured TTY (one-shot)                               |
| `WS /api/jobs/{id}/tty?follow=`                                       | Backlog + follow-tail of captured TTY                      |

### Inference proxy

Same-origin forwarder so the browser can talk to inference-server jobs
without running into CORS / PNA issues. The endpoint set also includes a
small **user-added-server registry** — the Inference → Model picker
lists running spawned/cluster servers by default, and this registry adds
a persistent "User-added servers" section so operators can one-click
external OpenAI-compatible upstreams (vLLM, remote inference, a
teammate's box) without retyping URL + token every session. Entries
live at `<config>/server/inference_server_registry.json` (0600);
tokens never round-trip back to the browser after the initial save.

| Endpoint                                                                                | Purpose                                                              |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `GET /api/inference/health?base=`                                                       | Proxy `<base>/health`                                                |
| `GET /api/inference/models?base=`                                                       | Proxy `<base>/models`                                                |
| `POST /api/inference/completions?base=`                                                 | Proxy `<base>/completions` (byte-for-byte SSE passthrough)           |
| `POST /api/inference/chat/completions?base=`                                            | Proxy `<base>/chat/completions` (byte-for-byte SSE passthrough)      |
| `GET /api/inference-servers/user`                                                       | List registered user URLs                                            |
| `POST /api/inference-servers/user` `{label, base_url, auth_token?, verify_tls?}`        | Register an external inference server. Tokens with CR/LF rejected as 400. |
| `DELETE /api/inference-servers/user/{entry_id}`                                         | Remove a registry entry                                              |

Token resolution order for every inference-proxy call: explicit
`X-Inference-Auth-Token` header → JobRecord auto-lookup (for spawned
local servers) → cluster-inventory lookup (for off-host peer servers
on master nodes) → registry lookup (for user-added entries) → none.

### Dataset_server registry + proxy

Drives the Datasets view's Servers tab. The registry CRUD endpoints
persist user-added URLs + tokens at `<config>/server/
dataset_server_registry.json` (0600). The proxy is the same-origin
forwarder for the dataset_server's `/v1/*` endpoints; its SSRF
allowlist is the registry itself (see `routes/dataset_server.py`).

| Endpoint                                                                              | Purpose                                                              |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `GET /api/dataset-servers/local`                                                      | Enumerate dataset_server JobRecords spawned by this forgather_server |
| `GET /api/dataset-servers/local/{queue_id}/bundle`                                    | Mint a `forgather-dataset://` transfer URI for Copy bundle           |
| `GET /api/dataset-servers/user`                                                       | List registered user URLs                                            |
| `POST /api/dataset-servers/user` `{label, base_url, auth_token?}`                     | Register a remote dataset_server. Tokens with CR/LF rejected as 400. |
| `DELETE /api/dataset-servers/user/{entry_id}`                                         | Remove a registry entry                                              |
| `POST /api/dataset-server/config/ensure-stub`                                         | Create the standalone-server's default config stub if absent         |
| `GET /api/dataset-server/proxy/health?base=`                                          | Proxy `<base>/v1/health`                                             |
| `GET /api/dataset-server/proxy/auth-status?base=`                                     | Proxy `<base>/v1/auth/status`                                        |
| `GET /api/dataset-server/proxy/datasets?base=`                                        | Proxy `<base>/v1/datasets`                                           |
| `GET /api/dataset-server/proxy/cache?base=`                                           | Proxy `<base>/v1/cache/hf`                                           |
| `GET /api/dataset-server/proxy/local?base=`                                           | Proxy `<base>/v1/local`                                              |
| `POST /api/dataset-server/proxy/load?base=`                                           | Proxy `<base>/v1/load` (body passthrough)                            |
| `GET /api/dataset-server/proxy/length?base=&handle=`                                  | Proxy `<base>/v1/datasets/{handle}/length`                           |
| `GET /api/dataset-server/proxy/iter?base=&handle=&position=&limit=`                   | Proxy `<base>/v1/datasets/{handle}/iter`; NDJSON stream collected into `{rows: [...]}`. `limit` capped at 500. |

Token resolution order for every proxy call: explicit
`X-Dataset-Auth-Token` header → JobRecord auto-lookup (for local
servers) → registry lookup (for user-added entries) → none.

### Services (auto-start)

CRUD over the `services:` block in `server_config.yaml`. Entries
declare long-running spawned processes (dataset / inference /
tensorboard / mkdocs) that the server brings up on boot. See
[Auto-start services](#auto-start-services) for the full schema.

| Endpoint                                                  | Purpose                                                        |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/services`                                       | List every configured service with its current running status (`ServiceStatus[]`: service + `running` (true iff a JobRecord with `status=="running"` matches the signature) + `queue_id` + raw `status`). |
| `POST /api/services` `{type, name, enabled, args}`        | Upsert by `<type, name>`. If `enabled=true` the autostart pass runs immediately so the entry comes up without waiting for the next server boot. |
| `DELETE /api/services/{type}/{name}`                      | Remove the entry. Any matching running instance is aborted first via `scheduler.abort_or_cancel` so the queue / Jobs rows don't linger. |
| `POST /api/services/{type}/{name}/enabled` `{enabled}`    | Toggle the auto-start flag. `enabled=true` triggers the autostart pass (start if not already running); `enabled=false` aborts the matching running instance. |

Service signature = `sha256((type, normalized_args))[:16]`. The
"normalized args" exclude operator-meta keys (`enabled` /
`priority` / `requested_gpus`) and scheduler-injected fields
(`scheme` / `routable_host`) so pre- and post-dispatch signatures
for the same logical service match.

### Generation-parameter presets

Named JSON blobs consumed by the Inference panel's preset picker.
Read-only bundled examples at `<repo>/generation_config/` are merged
with user-writable presets at `~/.config/forgather/generation_config/`.

| Endpoint                                                  | Purpose                                                        |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/generation-configs`                             | List presets (`{name, builtin}[]`)                             |
| `GET /api/generation-configs/{name}`                      | Load one preset (user copy wins over bundled)                  |
| `PUT /api/generation-configs/{name}` `{…params…}`         | Save / overwrite — lands in `~/.config/forgather/generation_config/`  |
| `DELETE /api/generation-configs/{name}`                   | Delete a user preset (403 if it only exists as a bundled one)  |
