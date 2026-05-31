# DiLoCo: full security model (issue #90)

This doc covers the auth + TLS + audit layer landed by PR
`feature/diloco-security`. The design mirrors the established
forgather pattern (dataset_server, inference_server, forgather_server)
adapted to DiLoCo's stdlib `http.server` stack.

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

## See also

* `docs/operations/tls.md` — operator-facing setup, including the
  DiLoCo-specific section that pairs with this design doc.
* `docs/design/diloco-pipeline-groups.md` — pipeline-parallel groups
  (issue #84). Security here is orthogonal; both work together.
* Tracking issue: GitHub #90.
