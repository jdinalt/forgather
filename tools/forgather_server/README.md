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

## Documentation map

This README is the operator hub: CLI flags, the `server_config.yaml`
schema, the security model (threat model + authentication), and how to
install and run the server. Three companion docs live next to it (tool
internals, intentionally outside the mkdocs nav):

- **[Web UI features](./FEATURES.md)** -- what every panel, view, and
  context menu in the browser app does, the AI agent assistant, and the
  not-yet-implemented list.
- **[Architecture](./ARCHITECTURE.md)** -- internals: cluster mode
  (multi-node), persistent on-disk state and garbage collection, dev
  mode, the directory layout, and the one-paragraph design summary.
- **[API reference](./API.md)** -- the full `/api` endpoint reference.

## Quick reference

Skip to the two reference tables most operators want first:

- [CLI arguments](#cli-arguments) — every flag accepted by
  `forgather server`.
- [Config file (`server_config.yaml`)](#config-file-server_configyaml) —
  full YAML schema for persistent CLI defaults and auto-start services.

The rest of this README covers the security model (threat model + auth),
demo mode, the search-path and filesystem flags, installation, and
running. The UI panels, multi-node/cluster internals, persistent state,
and the HTTP API now live in the [companion docs](#documentation-map)
above.

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
| `--demo`                             | off                                      | Read-only public-demo mode: blocks every POST/PUT/DELETE outside a narrow allowlist, redacts bearer tokens from API responses, and surfaces a "DEMO MODE" chip in the webui sidebar. See [Demo mode](#demo-mode-public-read-only-exposure). |
| `--fs-root PATH`                     | unrestricted (or repo+search roots in `--demo`) | Restrict every path-accepting API to descendants of this directory. Repeatable. Default in `--demo` is the Forgather repo plus the registered search roots, so demo visitors can't browse outside curated content. See [Filesystem allowlist](#filesystem-allowlist---fs-root). |
| `--regen-token`                      | off                                      | Rotate the persisted bearer token at startup. Invalidates every CLI client using the old token.                       |
| `--persist-sessions`                 | off                                      | Persist browser session cookies to `<config>/server/sessions.json` (0600) so the webui survives restarts. See [Persisted sessions](#persisted-sessions). |
| `--cluster NAME`                     | `default`                                | Cluster name to join (mDNS-scoped). Cluster mode is always on; this flag only overrides the name. See [Cluster mode](./ARCHITECTURE.md#cluster-mode-multi-node-prototype). |
| `--cluster-address IP`               | unset (repeatable)                       | Override the address advertised to cluster peers. Repeatable — useful when running inside a container whose network namespace hides the host NICs from psutil. The first entry also seeds the startup banner's clickable URL when bound to `0.0.0.0`. |
| `--tls` / `--no-tls`                 | shared config                            | Force-enable / force-disable TLS, overriding `<config>/tls/`'s shared setting. See [docs/operations/tls.md](../../docs/operations/tls.md). |
| `--tls-cert PATH` / `--tls-key PATH` | resolved from shared config              | Override the certificate / private-key paths for this run.                                                            |
| `--insecure`                         | off                                      | Allow binding a non-loopback host without TLS. Suppresses the "token in cleartext" abort.                             |
| `--lock-inference-proxy`             | off                                      | Restrict the inference reverse proxy to localhost upstreams. The unconditional `http`/`https`-only scheme guard still applies. See [Network exposure](#network-exposure). |
| `--docs-landing PATH`                | unset                                    | Path the Docs view opens by default, overriding the built-in `docs/README.md` preference. Absolute or repo-relative. A missing file falls back to the default — the override is a hint, not a hard requirement. |
| `--meta-template-dir PATH`           | unset (repeatable)                       | Additional directory to scan for meta-templates (the scaffold catalog used by New Config / New Template / New Project). Earliest entry has highest priority, so a user scaffold whose id matches a bundled default overrides it. See [Meta-template search path](#meta-template-search-path). |
| `--no-default-meta-templates`        | off                                      | Drop the bundled `templatelib/meta/` scaffolds from the catalog. Pair with `--meta-template-dir` to expose only a curated user catalog. |
| `--eval-dir PATH`                    | unset (repeatable)                       | Additional directory to scan for evaluation projects (the ones surfaced by `forgather eval list` and the webui's Evaluate modal). Repeatable; earliest entry wins on name collision. Composes with `eval.search_paths` in `~/.config/forgather/config.yaml`. See [Evaluation search path](#evaluation-search-path). |
| `--no-default-eval`                  | off                                      | Drop the bundled `examples/evaluation/` directory from the eval-config search path. Pair with `--eval-dir` for a curated user-only catalog. |

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

### `args:` block (CLI default overrides)

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

### `services:` block (auto-start services)

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
| `diloco`      | `diloco_server`  | The **DiLoCo server…** modal |

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

  diloco:
    ringdale:
      enabled: false            # opt-in; spawn manually via the DiLoCo modal
      output_dir: /shared/models/ringdale
      port: 8512
      num_workers: 2
      host: 0.0.0.0             # ``routable_host`` is stamped by the scheduler
                                # so cross-machine workers can reach it
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
**dispatch-injected** fields are excluded from the service signature
so a service's pre- and post-dispatch signatures match (what makes
restart-without-double-spawn and ▶/⏹ correctness work):
- `scheme` — stamped for `inference` and `dataset_server` to reflect
  whether the spawned child is actually serving TLS.
- `routable_host` — stamped for `inference`, `dataset_server`,
  `diloco_server`, and `mkdocs` whenever the operator bound to
  `0.0.0.0` / `::` / empty, with a LAN-routable address picked from
  the cluster's known peer set or psutil's first non-loopback IP.

The names are operator-chosen, must match `[A-Za-z0-9_-]+`, and are
purely human labels — dedupe between configured services and live
queue items is by **signature**, an sha256 over
`(type, normalized args)`. Multiple instances of the same type with
different args are fine (common case: several inference servers on
different ports / models).

For the boot-time / status / sidebar-UI semantics and the matching API
endpoints, see [Auto-start services](#auto-start-services) and
[API quick reference → Services](./API.md#services-auto-start).

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

## Demo mode (public read-only exposure)

`--demo` turns the server into a public-safe instance: every mutating
request (`POST` / `PUT` / `DELETE` / `PATCH`) returns
`403 {"detail":"Server is in read-only demo mode"}`, and bearer tokens
are stripped from `/api/jobs`, `/api/queue`, `/api/services` response
bodies before they reach the browser. The webui surfaces a compact amber **"DEMO
MODE"** chip in the sidebar header next to the Forgather version
label.

Pair with `--no-auth` for a fully anonymous public demo, or leave the
auth gate on for a curated audience:

```bash
# Anonymous public demo
forgather server --no-auth --demo --fs-root /path/to/example/projects

# Token-gated demo (token still required to load the webui, but the
# logged-in user can only read state)
forgather server --demo --fs-root /path/to/example/projects
```

**What's allowlisted** (the only POSTs that still work in demo mode):
- `/api/auth/logout` — session UX.
- `/api/inference/{completions, chat/completions, tokenize, detokenize}`
  — proxy reads against an external inference upstream. The
  `detokenize` route round-trips token ids back to a string and lets
  the webui recover a byte-accurate chat-template prompt against
  upstreams (like vLLM) whose `/tokenize` doesn't include it.
- `/api/dataset-server/proxy/load` and
  `/api/cluster/dataset_server_proxy/<server_id>/load` — proxy reads
  against an external dataset server so the Datasets panel can open
  a dataset for browsing.

**What's still safe in demo mode** (defense in depth):
- The "Copy bundle" button on local dataset-server rows is hidden in
  the UI and the backend endpoint refuses with 403 (the bundle URI
  embeds the real bearer token, which the redactor can't catch by
  key name).
- The server-config gear is hidden — the file it would open lives
  outside any sane `--fs-root`.
- The scheduler / restart / shutdown gears are disabled with
  explanatory tooltips.

**Recommended deployment**: container or VM, mount example projects
read-only, run with `--no-auth --demo --fs-root <examples-dir>`.

## Meta-template search path

The New Config / New Template / New Project modals show a tree of
*scaffolds* discovered under `templatelib/meta/`. By default the catalog
ships with the framework. Two CLI flags let the operator extend or
replace it:

```bash
# Add a user catalog alongside the bundled defaults
forgather server --meta-template-dir /home/me/forgather-scaffolds

# Multiple user roots, in priority order (first wins)
forgather server \
  --meta-template-dir /home/me/site \
  --meta-template-dir /home/me/personal

# Replace defaults entirely with a curated catalog
forgather server \
  --meta-template-dir /opt/myorg/scaffolds \
  --no-default-meta-templates
```

**Merge semantics** are first-wins, matching Jinja's search path:

- A leaf scaffold with the same id (e.g. `datasets/packed`) in two roots
  uses the one from the **earlier** root. So a user customisation of a
  bundled scaffold lives at the same relative path under their root and
  overrides the default.
- A category present in multiple roots merges children + templates from
  every root; the **first** root's `_category.yaml` provides the display
  label and description.
- Non-existent `--meta-template-dir` paths are logged as a warning at
  startup but don't crash discovery — typos give an empty contribution,
  not a startup failure.

The authoring guide at
[`templatelib/meta/README.md`](../../templatelib/meta/README.md)
covers the body / manifest pair, field types, `picker:` kinds, and the
verbose-with-commented-defaults pattern.

## Evaluation search path

The Evaluate modal and `GET /api/eval/configs` discover evaluation
projects (the ones described by `forgather eval list`) by walking a
search path. By default this is the bundled `examples/evaluation/`
directory plus any extras the user configured in
`~/.config/forgather/config.yaml`'s `eval.search_paths`. Two server
CLI flags let an operator extend or replace this without touching the
user config:

```bash
# Add a user catalog of eval projects alongside the bundled defaults
forgather server --eval-dir /home/me/my-evals

# Multiple user roots, priority-ordered (earliest wins on collision)
forgather server \
  --eval-dir /home/me/site-evals \
  --eval-dir /home/me/personal-evals

# Replace the defaults entirely with a curated catalog
forgather server \
  --eval-dir /opt/myorg/eval-projects \
  --no-default-eval
```

**Resolution order**: `--eval-dir` extras come first in scan order,
then the library's default discovery (bundled `examples/evaluation/` +
the user's `eval.search_paths` from config.yaml). `--no-default-eval`
drops the bundled directory from the resolved list. Duplicate paths
across these sources are de-duplicated while preserving the
priority-first ordering. Non-existent `--eval-dir` paths are logged
as a warning at startup but don't crash discovery — same shape as
`--meta-template-dir`.

Use `--eval-dir` for evaluation projects authored outside the
forgather directory tree (a per-user / per-org catalog kept under
version control elsewhere). The CLI's `forgather eval` commands keep
using the library's default discovery (they don't see the server's
extras); for CLI users the same effect is available via the
`eval.search_paths` user-config key.

## Filesystem allowlist (`--fs-root`)

Jupyter-Lab-style root restriction. Pass `--fs-root <path>`
(repeatable) and every path-accepting API will refuse paths that don't
resolve to a descendant of one of the listed roots, returning a 403.

```bash
# Limit browsing/editing to a single project tree
forgather server --fs-root /home/me/research

# Multiple roots (union)
forgather server --fs-root /home/me/research --fs-root /scratch/datasets
```

**Defaults**:
- Without `--demo` and without `--fs-root`: unrestricted (historical
  behaviour — the operator already trusts the box).
- With `--demo` and no `--fs-root`: defaults to the Forgather repo
  plus every directory in `search_roots.json`. That's the curated
  project content the operator already declared browsable.
- With `--fs-root` (regardless of `--demo`): exactly the supplied
  roots, no implicit union with anything else.

**Fail-closed**: if every supplied `--fs-root` is unresolvable or not
a directory, the server refuses to start rather than silently falling
back to unrestricted. Typos in the operator's argv aren't a security
weakening.

**What gets gated**: every `/api/fs/*` endpoint, every `/api/configs/*`
and `/api/template/source` read/write, every `/api/docs/file` and
`/api/docs/asset` read, every `/api/project*` endpoint that takes a
`project_dir`, every `/api/workspace/*` creation flow, and
`/api/config/dynamic-args`. The file-picker UI hides directory entries
whose realpath would land outside the allowlist (so a symlink can't
even be *clicked* to escape).

## Quiet-tokens flag (spawned servers)

The inference server (`tools/inference_server/server.py`), dataset
server (`tools/dataset_server/server.py`), and DiLoCo server
(`forgather diloco server`) all accept `--quiet-tokens`. When set, the
bearer-token-bearing launch banner (and the `curl -H "Authorization:
Bearer …"` example, and any on-disk token-file path) is replaced with a
one-line message that says auth is on but reveals nothing sensitive.
The token is still written to its per-port file as usual, so the local
CLI client / cluster peers still discover it; only the TTY log is
sanitized.

`--quiet-tokens` exists for one purpose: keeping the bearer out of a
publicly-visible TTY pane in `--demo` deployments. It is **not** an
operator choice — there is no checkbox in any spawn modal. Instead the
scheduler applies it automatically to every server it spawns **iff this
webui is running in `--demo` mode** (`demo_mode_enabled()`); in normal
operation the token is always printed so it can be copied onto clients,
Jupyter-style.

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
  `HTTPTrainerControlClient` (used by `forgather job` and by the
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
forgather -t train.yaml train --schedule              # background submit
forgather -t train.yaml submit                        # shorthand for train --schedule
forgather -t train.yaml train --schedule --priority 5 --requested-gpus 2
forgather eval test c4 -M output_models/my_model --schedule
forgather tb --enqueue --port 6006
# inf server is a long-running service: it submits to the scheduler
# (background) by default. --local-only runs it in the foreground.
forgather inf server -m output_models/my_model
# Multi-model server (one process hosts several models):
forgather inf server -m a=output_models/a -m b=output_models/b
# With every loaded model pinned to GPU (no CPU swap; required on
# unified-memory hardware like DGX Spark / Grace-Hopper):
forgather inf server -m a=output_models/a -m b=output_models/b --keep-on-gpu
forgather convert --enqueue --src output_models/my_model --dst /tmp/hf_export
forgather finalize --enqueue --source output_models/my_model --dest /tmp/final
forgather update --enqueue --src output_models/my_model --dst /tmp/my_model_v2
forgather mkdocs -f docs/mkdocs.yml --enqueue
```

**Queue and scheduler** (`forgather sched ...` is a deprecated alias):

```bash
forgather job scheduler status           # enabled, queued/running counts, last tick
forgather job list                       # table of all queued + active + recent jobs
forgather job scheduler pause            # stop dispatching new jobs
forgather job scheduler resume
forgather job cancel <queue_id>          # remove a queued or running job
forgather job cleanup                    # bulk-remove terminal job records
forgather job cleanup <job_id>           # remove one specific terminal record
forgather job gc                         # sweep orphan TTY files (see "State directories and GC")
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

**Drive the AI agent (interactive testing):** `forgather agent` is a thin
client over the `/api/agent/*` endpoints for exercising the in-process
assistant from the terminal — send messages as the user, watch the agent's
text / tool calls / results stream, and make every Approve/Reject decision
yourself (no auto-approve). Conversation + pending-action state lives in the
server, so each command is one step:

```bash
forgather agent profiles                       # list connection profiles (* = active)
forgather agent use <profile_id>               # test against any profile (Claude or local vLLM)
forgather agent status                         # active agent's connection + disclosure mode
forgather agent message "build the wikitext dataset"   # start a turn -> prints a session id
forgather agent approve <action_id>            # your call on a proposed (CONFIRM/PROPOSE) action
forgather agent reject  <action_id> --reason "use config Y"
forgather agent message --session <id> "...follow-up guidance..."
forgather agent continue --session <id>        # resume a turn cut off by the token budget
forgather agent sessions                       # list active session ids (* none = none yet)
forgather agent history <id>                   # dump the conversation
forgather agent forget <id>                    # delete a session from the server
```

A new session is created the first time you `message` without `--session`
(the id prints on the `STATE:` line); reuse it with `--session <id>`, or
`sessions` to list what's active.

Each turn streams until the agent finishes (often an answer or a clarifying
question) or pauses for approval; a final `STATE:` line says which and what to
run next. `--json` on the streaming verbs emits raw event JSONL.

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
dot, ▶/⏹ to toggle the `enabled` flag (start / stop), ✎ to edit (or
right-click → **Edit…**), and × to delete (the running instance, if
any, is aborted first). The four service modals each have a
**Create service…** button beside Start that prompts for a name and
persists the entry to the config file.

**Editing an existing service.** Right-clicking a row (or clicking
its pencil button) reopens the same modal pre-populated from the
service's persisted args. The footer collapses to a single **Save**
button; the name is fixed (rename = delete + recreate). When the
service is currently running the button becomes **Save & restart**:
the old instance is disabled+aborted, the modal waits for it to
drain, then upserts the new args with `enabled=true` so the
autostart pass spawns a fresh instance with the updated config.

**API.** Full CRUD plus enable-toggle, with the enable path running
the autostart pass (or aborting the matching running job) so changes
land immediately. See [API quick reference → Services (auto-start)](./API.md#services-auto-start).

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
  forgather job scheduler status
```

Binding to a non-loopback host (`-H 0.0.0.0`) is supported but the
bearer token then traverses the network in cleartext. Run behind an
SSH tunnel or a TLS-terminating reverse proxy for LAN access; native
TLS support is on the roadmap.

