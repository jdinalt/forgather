# Forgather Server -- HTTP API Reference

> Part of the [Forgather Server](./README.md) docs. The full `/api` endpoint
> reference. See [ARCHITECTURE](./ARCHITECTURE.md) for how the server is built
> and the [README](./README.md) for usage.

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
| `POST /api/workspace/new-project` `{workspace_dir, name, description, config_prefix?, default_config?, project_dir_name?, copy_from?, meta_template?, values?}` | Create a project under a workspace — wraps the CLI's `project_create_cmd`; nested `project_dir_name` (`a/b/c`) supported; refuses overwrite, returns absolute project_dir. `copy_from` and `meta_template` are mutually exclusive (400 otherwise); when `meta_template` is set the server renders the scaffold against `values` and seeds the default config with the result |
| `POST /api/workspace/new` `{parent_dir, name, description, workspace_dir_name?, forgather_dir, libs?, search_paths?}` | Create a workspace under a search root — wraps `ws_create_cmd`; parent must be a configured search root; nested `workspace_dir_name` supported; returns absolute workspace_dir |
| `POST /api/workspace/init-here` `{workspace_dir, name, description, forgather_dir, libs?, search_paths?}` | Initialize a workspace in an *existing* directory — used by the Files-tree right-click flow. Refuses if `forgather_workspace/` already exists; requires `workspace_dir` to live at-or-under a configured search root. |
| `POST /api/project/new-template` `{project_dir, kind: "config"\|"template", name, meta_template?, values?}` | Create a file under the templates dir; refuses overwrite, `.yaml` auto-appended, returns absolute path. With `meta_template` + `values` the file is seeded from a scaffold under `templatelib/meta/`; without them it is created empty |
| `GET /api/project/meta-templates`              | Tree of available scaffolds discovered under `templatelib/meta/`. Each leaf is a `MetaTemplate` (`id`, `title`, `description`, `target_kind`, `fields[]`); each branch is a `MetaCategory` with `templates[]` + `children[]`. Feeds the New Config / New Template modal's "pick a starting point" step |

### AI agent

Streaming endpoints return `text/event-stream` (`data: {json}\n\n` per
agent event), consumed by the webui with `fetch` + `ReadableStream` (not
`EventSource`, so the session cookie authenticates them).

| Endpoint                                          | Purpose                                                        |
| ------------------------------------------------- | ------------------------------------------------------------- |
| `GET /api/agent/status`                           | Whether the agent is configured (`{enabled, provider, model, base_url}` — no secrets) |
| `POST /api/agent/message` `{message, session_id?}`| Send a user message; SSE-streams the turn (text, tool activity, action cards). A leading `session` frame carries the session id |
| `POST /api/agent/approve` `{action_id}`           | Approve a pending action; runs the stored commit and SSE-streams the resumed turn |
| `POST /api/agent/reject` `{action_id}`            | Reject a pending action; SSE-streams the resumed turn |
| `GET /api/agent/sessions/{id}`                    | Conversation history (`{messages, awaiting_approval, …}`) |
| `GET /api/agent/profiles`                         | List saved profiles + `active_id` (credentials redacted to `has_api_key` / `has_imported_cert`) |
| `POST /api/agent/profiles` `{label, provider?, model?, base_url?, api_key?, api_key_env?, verify_tls?, ca_cert_pem?, …}` | Create a profile |
| `PUT /api/agent/profiles/{id}` `{…same fields…}`  | Update a profile (omitted fields unchanged; empty `api_key`/`ca_cert_pem` clears) |
| `DELETE /api/agent/profiles/{id}`                 | Remove a profile |
| `POST /api/agent/profiles/{id}/activate`          | Switch the active profile (hot-swap; no restart) |
| `POST /api/agent/models` `{profile_id? \| base_url, api_key?, verify_tls?, ca_cert_pem?}` | List available models (Claude SDK, or vLLM `/v1/models`) for the model picker |
| `POST /api/agent/fetch-cert` `{base_url}`         | Fetch the server's TLS cert (`{pem, sha256, …}`) for the import flow |

#### Agent tools (in-process)

The assistant drives the same machinery the webui does, through an
in-process tool registry (`tools/forgather_server/agent/`, the single
source of truth a future `forgather mcp` server would re-export). Each
tool has a **risk** that decides the gate: `read` runs automatically;
`propose` shows a diff and waits for approval; `confirm` is an
approve-to-run gate for low-blast mutations (no rich diff). The
approval gate is enforced **server-side** in the loop — the model can't
make a `propose`/`confirm` change permanent on its own.

Tool inventory by area:

- **Navigate / inspect** (read): `list_workspaces`, `list_projects`,
  `list_configs`, `inspect_config`, `render_config_pp`,
  `render_config_code`, `check_config`, `list_config_templates`,
  `config_template_refs`, `resolve_output_dir`, `read_file`,
  `list_directory`, `find_files`, `search_docs`, `reveal_in_ui`.
- **Author** (propose): `propose_edit_config`, `propose_new_config`
  (scaffold / copy-from / inline content), `propose_new_project`,
  `propose_new_workspace` (+ `list_meta_templates`).
- **Run jobs** (confirm): `run_dataset`, `run_construct`, `run_train`,
  `run_eval` — plus `list_jobs`, `read_job_output`, `wait_for_job`,
  `job_status`, `scheduler_status`, `gpu_status` (read) and
  `control_job` (confirm: save / stop / save-stop / abort),
  `cleanup_jobs` (confirm: remove finished job records).
- **Results** (read): `list_models`, `list_runs`, `run_summary`,
  `list_checkpoints`, `list_evaluations`, `read_run_tty`.
- **Datasets** (read): `list_dataset_servers`, `dataset_info`
  (+ `list_eval_configs`).
- **Services** (`list_services` read; per-type start tools +
  `stop_service`, confirm) — one start tool per service type, each with
  explicit args: `start_dataset_server`, `start_diloco_server`,
  `start_inference_server` (core) and `start_tensorboard`, `start_mkdocs`
  (extended). `start_dataset_server()` with no args brings up a default
  dataset server.
- **DiLoCo**: `list_diloco_servers`, `diloco_status` (read),
  `diloco_control` (confirm).
- **Inference / cluster / overrides**: `list_inference_servers`,
  `query_model` (confirm), `cluster_status`, `get_config_overrides`,
  `set_config_overrides` (confirm).
- **Filesystem** (`stat_path` read; `create_file` / `delete_path` /
  `move_path` / `copy_path` confirm; `edit_file` propose) — general file
  management for plain text/markdown/scratch files and cleanup /
  reorganizing; reuses the `/api/fs/*` and config_ops write guards (fs-root,
  symlink-chain rejection, depth floor, denylist, no-clobber-on-create,
  optimistic mtime). `create_file` touches an empty file; `edit_file`
  overwrites an existing file with a before/after diff (use
  `propose_edit_config` for Forgather configs instead). `delete_path` is
  recursive for directories and irreversible.
- **Meta**: `list_tools`, `tool_help`, `call_tool` (see disclosure
  below); `list_playbook`, `read_playbook` (task procedures — see below).

**Playbook (knowledge disclosure).** Task-specific procedures (how to run
training, build datasets, serve a model, etc.) live in markdown under
`agent/playbook/` rather than in the system prompt, so the base prompt stays
lean and the per-task detail doesn't tax context on every request. The prompt
points the agent at `read_playbook(topic)` (with `list_playbook` to discover
topics); it pulls the relevant procedure on demand. This is the knowledge
analogue of the tool-disclosure mechanism — three retrieval surfaces:
`tool_help` (how to call a tool), `search_docs` (how Forgather works), and
`read_playbook` (how to do a task with these tools).

**Tool disclosure (context-budget control).** As the tool set grows,
its descriptions + schemas occupy context — a real ceiling for
limited-context local (vLLM) models, where prompt caching (a *cost*
mechanism, and off for vLLM) doesn't help. Tools carry a **tier**
(`core` / `extended`) and the registry serializes in one of two modes:

- `inline` (default for Claude / large-context): every tool is in the
  array; `extended` tools show a one-line summary, with the full
  description fetched on demand via `tool_help`. The block stays static,
  so prompt caching is unaffected.
- `deferred` (default for a custom `base_url` / local model): only
  `core` + meta tools are in the array; `extended` tools are hidden and
  invoked through the `call_tool` dispatcher (discovered with
  `list_tools` / `tool_help`). The array stays tiny.

The mode is auto-selected by provider, overridable per profile via
`disclosure_mode` (`auto` / `inline` / `deferred`). `call_tool` runs the
inner tool under the inner tool's own risk, so a confirm/propose tool
still hits the approval gate when invoked indirectly.

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

### Cluster (multi-node)

The server is always in cluster mode (name `default` unless
`--cluster NAME` overrides it), so these endpoints always answer with
the node's own identity and member table. On a single-node `default`
cluster the member table is just this node, the master is self, and
the cross-node aggregation endpoints return only the local snapshot —
a webui that polls them is safe to mount unconditionally.

| Endpoint                                                           | Auth                       | Purpose                                                                    |
| ------------------------------------------------------------------ | -------------------------- | -------------------------------------------------------------------------- |
| `GET /api/cluster/self`                                            | bearer / peer              | This node's identity (always populated once the server has started)        |
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
the bearer token; see [Cluster mode (multi-node, prototype)](./ARCHITECTURE.md#cluster-mode-multi-node-prototype)
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
`X-Inference-Auth-Token` header → **`X-Inference-Server-Id` (entry-bound)**
→ JobRecord auto-lookup (for spawned local servers) → cluster-inventory
lookup (for off-host peer servers on master nodes) → registry lookup by
URL (for a raw typed URL) → none.

**Entry-bound auth (`X-Inference-Server-Id`).** When the webui talks to a
*registered* server it sends the entry's id, and the proxy attaches exactly
that entry's token — or nothing if the entry has none — with **no** URL
fallback. This keeps the token server-side (the browser only holds the id),
lets two entries share a `base_url` with independent auth, and makes the
model view's auth indicator authoritative: a "No auth" entry never inherits
a sibling entry's token. The entry's `verify_tls` posture is bound the same
way. The entry's token is attached **only** when the request actually
targets that entry's own `base_url` — naming entry A while forwarding to a
different host sends no token. The URL-based lookups remain only for callers
that don't name an entry (a hand-typed URL, a spawned-local or cluster
server).

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
[Auto-start services](./README.md#auto-start-services) for the full schema.

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
