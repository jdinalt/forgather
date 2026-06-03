# DiLoCo: full security model

This doc covers the auth + TLS + audit layer plus the cross-node
discovery surface. The design mirrors the established forgather
pattern (dataset_server, inference_server, forgather_server) adapted
to DiLoCo's stdlib `http.server` stack.

## Goals

1. **Bearer-token auth on the control plane** so registration,
   heartbeat, control actions, and status reads are restricted to
   callers holding the per-server token.
2. **TLS** for cleartext-vs-encrypted choice on a per-invocation basis,
   built on the same `~/.config/forgather/tls/` material every other
   forgather server uses. mTLS for cluster-peer identity at no extra
   plumbing cost.
3. **Operator opt-out for bulk data transport** — pseudo-gradients
   and model weights — for the trusted-LAN throughput case. Disrupt
   training: OK. Take over hosts: not OK.
4. **Audit log** of every event worth reconstructing after the fact
   (registrations, evictions, outer-optimizer steps, control actions).
5. **WebUI proxy carries credentials** the operator entered, instead
   of ignoring `auth_token` / `verify_tls` on the registry.

## Control vs bulk plane

The 14-endpoint DiLoCo wire splits two ways:

| Plane | Endpoints | Body shape | Throughput? |
|---|---|---|---|
| Control | `/register`, `/heartbeat`, `/deregister`, `/datasets/register`, `/work/{request,complete,queues,queue}`, `/control/*`, `/status`, `/info`, `/health` | small JSON | Latency-tolerant |
| Bulk | `/submit_pseudograd`, `/submit_fragment_pseudograd`, `/global_params` | torch.save'd state dict | Throughput-sensitive (MBs) |

The two planes run on a single port (default) or, with
`--bulk-cleartext`, the bulk endpoints move to a second listener. The
control port is always TLS+bearer-auth (or explicit `--no-auth`). The
bulk listener has no configurable posture: it is always cleartext,
unauthenticated, and bound to a server-picked ephemeral port. Its sole
reason to exist is to bypass TLS for throughput on a trusted LAN — a
TLS bulk plane would gain nothing over the control port, and a bearer
sent over a sniffable socket leaks the control credential to a sniffer
who can already read the tensors, so neither is offered.

When `--bulk-cleartext` is set:

* The control port refuses bulk paths with HTTP 404 + an
  `X-Forgather-Bulk-Url` header. Avoids two ways into the bulk plane
  (the slow-but-secure path + the fast-but-cleartext path) which
  would let an attacker pick whichever is convenient.
* `/register` responses include the same `X-Forgather-Bulk-Url`
  header so workers learn the server-assigned ephemeral port without
  an extra round-trip — and because that header rides the
  TLS-protected control plane, the port never needs to be a stable,
  operator-chosen value.
* The DiLoCo HTTP client (`forgather.ml.diloco.client.DiLoCoClient`)
  captures the bulk URL from `/register` and routes the three bulk
  endpoints there automatically. The control URL still hosts every
  small endpoint plus the bearer-protected register.
* The client **omits** `Authorization: Bearer` on bulk-routed requests.
  The bulk plane is unauthenticated by design, so the bearer is never
  needed there — and sending the control-plane token over the cleartext
  bulk socket would hand a LAN sniffer full control-plane authority,
  defeating the entire two-plane split. Bearer redaction is therefore a
  client-side invariant, not just a server-side "don't check" (see
  `DiLoCoClient._headers` / `_routes_to_bulk`).

## Authentication

### Bearer tokens

Per-port file at `~/.config/forgather/diloco_server/<port>.token`
(mode 0600 in a 0700 dir). Auto-generated on first run, reused across
restarts. `--regen-token` rotates. `--auth-token` / `--auth-token-file`
override; `--no-auth` opts out entirely.

Client discovery precedence (in `DiLoCoClient.__init__`):

1. Explicit `token=` constructor arg.
2. `FORGATHER_DILOCO_SERVER_TOKEN` env var.
3. The per-port file at the loopback path (only when the URL is
   `localhost` / `127.0.0.1` / `::1`).

Remote workers don't get the loopback auto-discovery; they need the
env var, the explicit arg, or a webui-proxy override header.

### mTLS

When TLS is enabled with a CA bundle present, the server's
`stdlib_ssl_context` is built with `ssl.CERT_OPTIONAL` and the bundle
loaded as the CA. A request whose TLS handshake validated a peer cert
against that CA is treated as cluster-authenticated and the bearer
check is skipped (`peer_cert_authenticated(handler)` reads
`handler.connection.getpeercert()` and returns True for a non-empty
dict).

This means the webui proxy, the DiLoCo CLI, and inter-cluster peer
calls can all authenticate via the shared TLS material without any
extra token plumbing. The bearer path stays available for non-cluster
callers (e.g. an operator using `curl` from a laptop with a CA-trusted
client).

## Cross-node discovery (cluster inventory)

Multi-node DiLoCo training requires a peer on host `B` to learn that a
DiLoCo server is running on host `A` and to obtain its bearer token —
without an operator hand-pasting both. The forgather server already
solves this for `dataset_server` and `inference_server` via the
cluster-inventory pattern (`cluster_dataset_inventory.py`,
`cluster_inference_inventory.py`); DiLoCo uses the same model.

**Per-peer attestation (`/api/cluster/diloco_servers_local`).** Every
forgather server exposes a read-only endpoint listing the DiLoCo
servers it attests to, as a unified list across two sources:

1. `JobRecord(job_type == "diloco_server")` entries currently in
   `starting` / `running` state (locally-spawned via the webui or
   `forgather diloco server`).
2. The user-added registry at
   `<config>/server/diloco_server_registry.json` — the
   `forgather diloco register <url> --auth-token <tok>` escape hatch
   for WAN endpoints / SSH-tunneled remotes / teammate servers that
   mDNS can't see.

Each entry carries `server_id`, `base_url`, `auth_token`, `label`,
`source` (`local` / `user`), `peer_node_id`, `source_id`, `loopback`,
and `verify_tls`. Tokens are in the response body; the endpoint is
on the mTLS peer-allow-list (`auth._PEER_ALLOWED_PATHS`) and inherits
the trust boundary the dataset and inference equivalents already
establish — a peer that presented a valid cluster-CA-signed cert is
by definition allowed to read cluster state, including the per-server
bearer tokens it would need to dial the upstream.

**Master aggregation (`/api/cluster/diloco_servers`).** The master
node runs `master_collect_servers_loop` every ~10 s, GETs
`/api/cluster/diloco_servers_local` from every reachable peer (and
includes its own local list), deduplicates by `server_id` (the first
12 hex chars of SHA-256 over the normalized `base_url`), and exposes
the aggregated snapshot.

**Health probing.** A `master_health_loop` probes `/health` on each
server every ~10 s and flips the `healthy` flag, preserving the
per-server failure-streak counters across collect ticks so a flapping
upstream stays distinguishable from one that just hasn't been polled
yet. Browser clients on any node reach the aggregated list through
the same `/api/cluster/diloco_servers` route — non-master nodes proxy
upstream to master so every webui sees the same set.

**Wake hooks.** The scheduler signals `wake_loops()` on
`diloco_server` job spawn/reap/abort and `POST
/api/cluster/diloco_servers/refresh` lets the webui force a re-poll
on demand. Membership transitions (a new master elected,
a peer joining) also wake the loops via
`cluster_membership.register_role_change_listener`. Steady-state
convergence is ~1 s after a state change, not the bare-cadence 10 s.

**Loopback / 0.0.0.0 handling.** Same rules as the
dataset/inference inventories. JobRecords bound to `0.0.0.0` are
rewritten to the cluster identity's hostname (or to the scheduler-
stamped `routable_host` when present). Loopback-only binds are kept
in the inventory with `loopback=True` so the operator still sees
their node-local DiLoCo servers in the panel, but the cross-node
candidate-selection logic excludes them.

**Token transit, end-to-end.**

1. Server on `A` writes its bearer to
   `~/.config/forgather/diloco_server/<port>.token` (mode 0600)
   and registers the JobRecord with that token.
2. `A`'s forgather server reports the entry on
   `/api/cluster/diloco_servers_local` over its TLS+mTLS surface.
3. `B`'s master (if `B` is master) pulls the entry via the cluster-CA
   carve-out (no bearer needed — the mTLS client cert is the
   credential).
4. `B`'s webui calls `/api/cluster/diloco_servers` and gets the
   aggregated list; the proxy at `routes/diloco.py` consults
   `cluster_diloco_inventory.master_inventory.token_for_url(base)`
   when dialing `A` upstream, so the operator never sees, copies, or
   pastes the token.

The user-added registry is preserved as an **escape hatch**: it
covers servers that aren't reachable on the local LAN at all
(WAN, SSH tunnel, mDNS-suppressed network). The operator types
the URL and token once, and the entry then flows through the
cluster inventory identically to a locally-spawned one.

### Why not auto-share via mDNS only?

mDNS advertises the forgather server's bind, not the DiLoCo server's.
A DiLoCo server is a child process the forgather server spawned; the
forgather server is the authority for "what DiLoCo servers exist on
this node" because it holds the `JobRecord` rows and the persistent
registry. Cluster discovery therefore happens at the forgather-server
layer (HTTP over the existing peer-pull rail), not at the DiLoCo
server's own listener.

### Threat-model deviations from the dataset / inference precedents

The DiLoCo cluster inventory follows the dataset/inference patterns
closely, but two surfaces are wider on purpose; both are operator-
visible choices, not accidents.

**SSRF allowlist includes peer-attested URLs.** `routes/diloco.py`'s
`_known_base_urls()` treats any base URL the master aggregates from
the cluster as a permitted proxy target. The dataset proxy
(`routes/dataset_server.py`) does *not* — it permits only loopback
plus locally-registered URLs. The reason for the wider DiLoCo
posture: cluster-discovered DiLoCo servers must be inspectable from
*any* peer's webui (otherwise there's no point in surfacing them in
the cluster panel). The narrower dataset posture works because the
dataset surface routes through `/api/cluster/dataset_router/resolve`
on the master, not the per-base proxy.

A credentialed peer can use this to put an arbitrary base URL on
the master's outbound allowlist; that URL is then probed by the
health loop every ~10 s and accepted by the webui proxy. Per the
overall cluster bearer trust model — a peer that holds a CA-signed
cert already has training-job RCE — this is in-scope. Operators who
want stricter posture should not put untrusted hosts into the
cluster's mTLS trust root.

**`verify_tls=False` is gated to `source="user"` on ingest.**
The escape hatch for SSH-tunneled remotes is honored only when a
peer reports the entry as a user-registry entry on its own node.
Spawned (`source="local"`) entries from a peer always run with the
cluster CA chain — one peer can't downgrade another peer's outbound
TLS by attesting to a fresh URL with chain validation off. The
local operator can still register the URL locally with
`verify_tls=False` if they need it.

This gating happens at ingest in
`cluster_diloco_inventory._local_server_from_dict`. URLs with
embedded `userinfo` (`http://user:pass@host`) are dropped at the
same checkpoint so a peer can't inject Basic-auth credentials into
the master's outbound calls.

The ingest gate runs on the **master** when it receives a peer's
`/diloco_servers_local`. On a non-master node, entries returned from
`/api/cluster/diloco_servers` (proxied from the master) are accepted
as-is — there is no second `_local_server_from_dict` gate on the
non-master side, so a compromised master could in principle ship
`source="user", verify_tls=False` for an arbitrary URL. A compromised
master already implies cluster-wide RCE per the bearer trust model,
so the missing second gate doesn't change the threat surface.

### Identity binding for `group_id` / `pp_rank` — phase 1

**Phase 1 = job-level only.** Holding a valid bearer (or a cluster
cert) is sufficient to claim any `group_id` / `pp_rank` within a given
DiLoCo server's lifetime. Cross-job spoofing is blocked because each
spawned DiLoCo server has its own per-port token.

This is a deliberate scope choice. A worker that already holds the
job's bearer can disrupt the job by claiming the wrong rank, but it
can also disrupt the job by submitting garbage pseudo-gradients — so
per-rank identity buys nothing additional under the current threat
model.

**Future PR**: encode `(job_id, group_id, pp_rank)` in mTLS Subject
Alternative Names or pre-register expected tuples at job start. The
audit-log `control` event already has a `caller` slot for this.

## Webui proxy

`tools/forgather_server/routes/diloco.py` reads credentials in the
same precedence as the dataset_server proxy:

1. Explicit `X-Diloco-Auth-Token` request header (operator override).
2. `_token_for_local(base)` — JobRecord auto-lookup for locally
   spawned servers. The scheduler persisted the token on the record
   when spawning; we match on bind port.
3. `diloco_server_registry.find_token(base)` for user-added remotes.
4. Empty (server is running `--no-auth`).

`verify_tls=False` on a registry entry propagates `verify=False` to
the httpx client — the SSH-tunneled-remote opt-out. Otherwise the
proxy uses `forgather.tls.httpx_verify_for_url(target)` which builds
an SSLContext from the cluster CA bundle.

The DiLoCoPanel form lets operators set `auth_token` (masked input)
and toggle `verify_tls`. Registry rows display a 🔒 indicator when a
token is stored.

## Audit log

`<output_dir>/diloco_audit.log` is an append-only JSONL stream. One
line per record; each carries a UTC ISO-8601 timestamp + event kind +
event-specific fields. Best-effort: write failures log a warning and
keep the request going — the audit log is a record, not a guard.

Instrumented events:

| event | fields |
|---|---|
| `register` | worker_id, hostname, group_id, pp_rank, pp_world_size, num_registered |
| `deregister` | worker_id |
| `eviction` | trigger_worker_id, evicted (list), group_id, remaining |
| `outer_step` | sync_round, contributors, missing_contributors |
| `fragment_outer_step` | fragment_id, fragment_round, sync_round, contributors, triggered_by (optional) |
| `control` | action, data (per-action allowlisted; see `_CONTROL_AUDIT_FIELDS`) |

Tokens are never written to this file (regression-guarded by
`test_token_is_never_logged`). The control-payload allowlist is the
forward-compat guardrail for future control endpoints that may carry
secret material — unknown fields under a known action are dropped.

## RCE hardening

Independent of auth: every inbound tensor blob deserializes via
`torch.load(..., weights_only=True, map_location="cpu")`. This is the
only mitigation that makes the "disrupt OK, host takeover not OK"
guarantee actually hold on an open bulk plane. The guarantee was
already in place from prior PRs; this work preserves it.

## Spawn flow (local)

When forgather_server spawns a DiLoCo server through the queue:

1. Scheduler calls `_resolve_diloco_server_token(port, regen)` — a
   per-port token file, reused across restarts unless `--regen-token`.
2. Token is persisted on the resulting `JobRecord.auth_token` so the
   webui proxy's auto-lookup finds it.
3. The spawn command is built with
   `--auth-token-file <per-port path>` so the token never lands in
   `argv` (visible via `ps`).
4. The spawned `forgather diloco server` reads the same file at
   startup via `resolve_auth_token`'s `"persisted"` branch.

## Wire-format additions

* `Authorization: Bearer <token>` on every authenticated request.
* `WWW-Authenticate: Bearer realm="forgather-diloco"` on 401 responses.
* `X-Forgather-Bulk-Url: <url>` on `/register` responses when the
  cleartext bulk plane is enabled (carries the server-assigned
  ephemeral port); also on 404 responses from the control port for
  bulk paths.
* `X-Diloco-Auth-Token: <token>` — webui proxy override header.

## Out of scope (deferred)

* **Per-rank identity binding.** Job-level only in this PR; future PR
  adds mTLS subject-bound `(job_id, group_id, pp_rank)` or
  pre-registered roster.
* **Token rotation while running.** Restart-driven only.
* **Audit log tamper-evidence.** Plain JSONL; no HMAC chain.
* **Migrating off stdlib `http.server`.** The bearer + TLS layer
  added here is surgical; a full FastAPI/uvicorn migration would be
  a separate refactor.

## Test surface

| File | Coverage |
|---|---|
| `tests/unit/forgather/test_tls.py` | `stdlib_ssl_context` + `urllib_ssl_context` |
| `tests/unit/ml/diloco/test_server_auth.py` | Bearer 401/200 paths; verify_bearer unit |
| `tests/unit/ml/diloco/test_client_auth.py` | End-to-end client/server auth |
| `tests/unit/ml/diloco/test_server_mtls.py` | mTLS skip-bearer; peer_cert_authenticated unit |
| `tests/unit/ml/diloco/test_server_bulk_port.py` | Cleartext bulk-plane routing + ephemeral-port advertisement |
| `tests/unit/ml/diloco/test_audit_log.py` | JSONL records + best-effort writer |
| `tests/unit/forgather_server/test_scheduler_diloco_server_token.py` | Per-port spawn token |
| `tests/unit/forgather_server/test_routes_diloco_auth.py` | Proxy auth/verify_tls attachment |
| `tests/unit/forgather_server/test_cluster_diloco_inventory.py` | Local enumeration (JobRecord + user registry); peer-entry validation (URL, userinfo, source-gated verify_tls); master-inventory merge, role transition, token/verify_tls lookup |
| `tests/unit/forgather_server/test_routes_cluster_diloco.py` | `/api/cluster/diloco_servers_local`, `/diloco_servers`, `/diloco_servers/refresh`; peer-mTLS allow-list membership; non-master proxy-to-master fallback |
| `tests/unit/forgather_server/test_routes_diloco.py` (cluster-proxy section) | Non-master proxy threads master snapshot into SSRF / auth / verify lookups; cluster-known URL allowed, unknown still 403, cluster `verify_tls=False` honored |
| `tests/unit/forgather_server/test_scheduler_diloco_env.py` | DILOCO_WORKER_ID memorable-name default + regression that it never equals queue_id |

## See also

* `docs/operations/tls.md` — operator-facing setup, including the
  DiLoCo-specific section that pairs with this design doc.
* `docs/design/diloco-pipeline-groups.md` — pipeline-parallel groups
  (issue #84). Security here is orthogonal; both work together.
* Tracking issue: GitHub #90.
