export interface SearchRoot {
  path: string;
  exists: boolean;
}

export interface ConfigInfo {
  name: string;
  path: string;
  is_default: boolean;
}

export interface ProjectInfo {
  project_dir: string;
  name: string | null;
  description: string | null;
  default_config: string | null;
  workspace_root: string | null;
  configs: ConfigInfo[];
  parse_error: string | null;
}

export interface WorkspaceCluster {
  workspace_root: string;
  name: string | null;
  description: string | null;
  parent_workspace_root: string | null;
  projects: ProjectInfo[];
}

export interface ConfigMeta {
  name: string | null;
  description: string | null;
  /** e.g. "type.training_script", "type.training_script.causal_lm",
   *  "type.model", "type.dataset". Lets the UI show only relevant
   *  actions per config type. */
  config_class: string | null;
  parse_error: string | null;
}

export interface OutputDirInfo {
  output_dir: string;
  models_dir: string;
  output_dir_exists: boolean;
  models_dir_exists: boolean;
  output_dir_size_bytes: number;
  output_dir_entry_count: number;
  models_dir_size_bytes: number;
  models_dir_entry_count: number;
  /** Raw nproc_per_node from the config's meta block: either a positive
   *  integer (fixed worker count) or a torchrun keyword like "gpu" / "cpu"
   *  / "auto". */
  nproc_per_node: number | string | null;
}

export interface DeleteDirResponse {
  deleted: string;
  removed_bytes: number;
}

export interface TrefsNode {
  name: string;
  path: string;
}

export interface TrefsEdge {
  source: string;
  target: string;
}

export interface TrefsGraph {
  root: string;
  nodes: TrefsNode[];
  edges: TrefsEdge[];
}

export interface DebugTraceItem {
  /** Template name as Jinja2 sees it (relative to the search path). */
  name: string;
  /** Absolute filesystem path of the source file (or "" for synthetic
   *  fragments split out of a parent template). */
  path: string;
  /** Pre-preprocess source as the loader returned it. */
  raw: string;
  /** Source after the LineStatementProcessor rewrite (plain Jinja2). */
  preprocessed: string;
}

/** Structured 400 detail returned by config endpoints (pp, code, debug)
 *  when any pipeline stage fails. The frontend renders a compiler-style
 *  block keyed off `kind`. */
export type ConfigErrorKind =
  | "preprocess_error"
  | "yaml_error"
  | "code_error";

export interface ConfigErrorDetail {
  kind: ConfigErrorKind;
  template: string | null;
  lineno: number | null;
  message: string;
  source_context: string | null;
}

/** Error subclass thrown by api.* helpers when an HTTP request fails.
 *  `detail` carries the JSON body as-is when the response was JSON
 *  (e.g. ConfigErrorDetail), or the raw text body otherwise. */
export class ApiError extends Error {
  status: number;
  statusText: string;
  detail: unknown;

  constructor(status: number, statusText: string, detail: unknown) {
    const summary =
      typeof detail === "string"
        ? detail
        : (detail as { message?: string })?.message || JSON.stringify(detail);
    super(`${status} ${statusText}: ${summary}`);
    this.name = "ApiError";
    this.status = status;
    this.statusText = statusText;
    this.detail = detail;
  }
}

function isConfigErrorDetail(d: unknown): d is ConfigErrorDetail {
  if (typeof d !== "object" || d === null) return false;
  const kind = (d as { kind?: string }).kind;
  return (
    kind === "preprocess_error" ||
    kind === "yaml_error" ||
    kind === "code_error"
  );
}

export function asConfigError(err: unknown): ConfigErrorDetail | null {
  if (err instanceof ApiError && isConfigErrorDetail(err.detail)) {
    return err.detail;
  }
  return null;
}

export interface TemplateEntry {
  name: string;
  path: string;
  rel_path: string;
}

export interface TemplateGroup {
  category: string;
  search_path: string;
  templates: TemplateEntry[];
}

export interface FsEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface FsListing {
  path: string;
  parent: string | null;
  entries: FsEntry[];
}

export interface QuickPath {
  label: string;
  path: string;
}

export interface GpuProcess {
  pid: number;
  used_mem_bytes: number;
  /** Best-effort process name from NVML / /proc; null when lookup fails. */
  name: string | null;
  /** "compute" for CUDA workloads, "graphics" for desktop / window-manager
   *  processes. Graphics processes do NOT block scheduler dispatch. */
  kind: "compute" | "graphics";
}

export interface GpuInfo {
  index: number;
  name: string;
  total_mem_bytes: number;
  used_mem_bytes: number;
  util_pct: number | null;
  mem_util_pct: number | null;
  power_w: number | null;
  temp_c: number | null;
  fan_pct: number | null;
  processes: GpuProcess[];
  source: string;
  node: string;
  /** True when the operator excluded this GPU via CUDA_VISIBLE_DEVICES at
   *  server start. Excluded GPUs still report telemetry but the scheduler
   *  refuses to assign them. */
  excluded: boolean;
  /** True when the user has runtime-disabled this GPU via the web UI.
   *  Distinct from excluded: this is reversible and persists via gpu_policy.json. */
  disabled: boolean;
  /** Minimum queue priority required to schedule on this GPU (inclusive).
   *  0 means no restriction. */
  min_priority: number;
  /** True when a running JobRecord on the owning peer has reserved this
   *  GPU. Stamped server-side so peers are authoritative for their own
   *  reservations — used by the cluster Nodes panel to mark a peer GPU
   *  BUSY without needing cross-node job visibility. */
  reserved: boolean;
  /** True when total_mem_bytes reports host system RAM rather than a
   *  discrete VRAM pool — set for GPUs like GB10 / Jetson where NVML
   *  returns "Not Supported" for memory info and the device shares
   *  system memory. */
  unified_memory?: boolean;
}

export interface GpuPolicy {
  disabled: boolean;
  min_priority: number;
}

/** This server's identity within an active cluster. ``null`` from the
 *  /api/cluster/self endpoint means the server is in standalone mode
 *  and the rest of the cluster API will return empty payloads. */
export interface ClusterSelf {
  node_id: string;
  hostname: string;
  cluster_name: string;
  port: number;
  forgather_version: string;
  started_at: number;
  is_master: boolean;
}

export interface ClusterProbeInterface {
  name: string;
  address: string;
  netmask: string;
  cidr: string;
  is_up: boolean;
  speed_mbps: number;
}

export interface ClusterProbe {
  versions: Record<string, string>;
  interfaces: ClusterProbeInterface[];
  cpu: {
    logical: number;
    physical: number;
    ram_gib: number;
  };
}

export interface ClusterMember {
  node_id: string;
  hostname: string;
  address: string;
  port: number;
  cluster_name: string;
  forgather_version: string;
  first_seen: number;
  last_seen: number;
  reachable: boolean;
  /** "discovery" | "peer_pull" | "self" — debug aid for figuring out
   *  which mechanism is keeping a stale entry alive. */
  last_source: string;
  /** Pre-flight probe payload — package versions, interfaces, CPU
   *  summary. Null until the peer answers at least one peer-pull
   *  with the new shape. */
  probe: ClusterProbe | null;
}

export interface ClusterBandwidthEntry {
  peer_node_id: string;
  peer_hostname: string;
  peer_address: string;
  bytes_transferred: number;
  elapsed_seconds: number;
  mbps: number;
  timestamp: number;
  error: string | null;
}

export interface ClusterBandwidthResponse {
  measurements: ClusterBandwidthEntry[];
  server_time: number;
}

export interface ClusterLatencyEntry {
  peer_node_id: string;
  peer_hostname: string;
  peer_address: string;
  samples: number;
  min_ms: number;
  median_ms: number;
  max_ms: number;
  timestamp: number;
  error: string | null;
}

export interface ClusterLatencyResponse {
  measurements: ClusterLatencyEntry[];
  server_time: number;
}

export interface ClusterJobMember {
  node_id: string;
  hostname: string;
  address: string;
  port: number;
  queue_id: string;
  nproc_per_node: number;
  node_rank: number;
  nccl_socket_ifname: string | null;
  /** Live status of this rank's queue item, fetched at read time
   *  via the master's per-peer status fanout. null when the master
   *  could not reach the peer. */
  current_status?: string | null;
  exit_code?: number | null;
  error?: string | null;
}

export interface ClusterJob {
  cluster_job_id: string;
  project_dir: string;
  config: string;
  submitted_at: number;
  rdzv_endpoint: string;
  rdzv_id: string;
  rdzv_node_id: string;
  members: ClusterJobMember[];
  status: string;
  cancelled_at: number | null;
  /** Aggregated status across all members. The bundle's own
   *  ``status`` only flips on cancel/done/failed promotions; this
   *  field is always recomputed at read time. */
  rolled_up_status?: string;
}

export interface ClusterJobMemberSpec {
  node_id: string;
  nproc_per_node: number;
  nccl_socket_ifname?: string | null;
}

export interface ClusterJobSubmitRequest {
  project_dir: string;
  config: string;
  dynamic_args?: Record<string, unknown>;
  priority?: number;
  members: ClusterJobMemberSpec[];
  rdzv_node_id?: string;
  rdzv_port?: number;
  allow_version_mismatch?: boolean;
  /** Same shape as ``EnqueueRequest.dataset_source``; resolved once on
   *  the master and merged into every peer's extra_env. */
  dataset_source?: DatasetSource | null;
}

export interface ClusterJobSubmitResponse {
  cluster_job: ClusterJob;
  warnings: string[];
}

export interface ClusterJobCancelResponse {
  cluster_job_id: string;
  cancelled: boolean;
  per_member: { node_id: string; queue_id: string; result: unknown }[];
}

export interface ClusterMembersResponse {
  cluster_name: string | null;
  self_node_id: string | null;
  master_node_id: string | null;
  members: ClusterMember[];
  server_time: number;
}

export interface ClusterGpusEntry {
  node_id: string;
  hostname: string;
  address: string;
  reachable: boolean;
  gpus: GpuInfo[];
  error: string | null;
}

export interface ClusterGpusResponse {
  nodes: ClusterGpusEntry[];
  server_time: number;
}

/** Configured auto-start service entry (from `services:` in the
 *  server config) plus its current running status. */
export interface ConfiguredService {
  type: string;
  name: string;
  enabled: boolean;
  args: Record<string, unknown>;
  signature: string;
}
export interface ServiceStatus {
  service: ConfiguredService;
  running: boolean;
  queue_id: string | null;
  status: string | null; // "queued" | "starting" | "running" | null
}

/** Unified job model returned by /api/jobs.
 *
 *  ``id`` is the stable identifier the UI keys on. For server-launched jobs
 *  it equals the queue_id; for externally-discovered endpoints it equals
 *  the trainer's job_id. ``source`` distinguishes them so the UI can
 *  expose extra controls (e.g. View TTY) only where they make sense. */
export interface Job {
  id: string;
  queue_id: string | null;
  job_id: string | null;
  project_dir: string | null;
  config: string | null;
  dynamic_args: Record<string, unknown> | null;
  requested_gpus: number | null;
  priority: number | null;
  submitted_at: number | null;
  node: string | null;
  gpu_indices: number[] | null;
  job_type: "training"
    | "eval"
    | "inference"
    | "dataset_server"
    | "tensorboard"
    | "mkdocs"
    | "convert"
    | "finalize"
    | "update"
    | "model"
    | "dataset"
    | "construct";
  job_params: Record<string, unknown> | null;
  status: string;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  error: string | null;
  pid: number | null;
  host: string | null;
  port: number | null;
  alive: boolean;
  tty_log_path: string | null;
  logs_dir: string | null;
  output_dir: string | null;
  /** For path-prefixed sub-services (e.g. TB spawned with --path_prefix);
   *  the panel appends this to the host:port link so the URL actually
   *  serves content. Null when the spawn didn't use a prefix. */
  path_prefix: string | null;
  /** Bearer token for inference jobs. The Inference panel auto-fills its
   *  Auth-Token field from this when the user picks a local server.
   *  Null for non-inference jobs and for inference jobs running --no-auth. */
  auth_token: string | null;
  source: "record" | "endpoint" | "merged";
}

/** Job statuses for which the scheduler considers the job to be holding its
 *  reserved GPUs. Mirrors ``job_records.RUNNING_STATUSES`` on the backend;
 *  keep in sync when adding a new status. The UI consults this set to
 *  decide whether a GPU card should render as "busy" (a Forgather job has
 *  the device) vs "idle" (available for dispatch). */
export const RUNNING_JOB_STATUSES: ReadonlySet<string> = new Set([
  "starting",
  "running",
]);

export interface ControlResponse {
  success: boolean;
  message: string;
  data: Record<string, unknown> | null;
}

/** Trainer control commands proxied through /api/jobs/{id}/control/{action}.
 *  - save / stop / save-stop / abort go through the trainer's HTTP endpoint.
 *  - kill = local SIGTERM on the process group (works pre-correlation).
 *  - force-kill = local SIGKILL — last resort for hung torchrun groups
 *    that aren't responding to SIGTERM. Both server-only. */
export type ControlAction =
  | "save"
  | "stop"
  | "save-stop"
  | "abort"
  | "kill"
  | "force-kill";

export interface DynamicArg {
  dest: string;
  cli_name: string;
  type: string; // "int" | "str" | "float" | "bool" | "path"
  help: string | null;
  default: unknown;
  choices: unknown[] | null;
  /** Colon-separated organizational path (e.g. "Trainer:LR-scaling").
   *  When any arg in the schema has a group, ungrouped args fall under
   *  an "Other" bucket so the form stays consistent. */
  group: string | null;
  /** Enforced at action time. The webui blocks Submit when a required
   *  arg is missing; ``pp`` still materializes (placeholder defaults). */
  required: boolean;
  /** Inclusive numeric bound. Only meaningful when type is "int" or
   *  "float". Either may be set independently. */
  min: number | null;
  max: number | null;
}

/** Last-used multi-node submit settings. The webui owns this shape;
 *  the server stores it as an opaque blob alongside the dynamic-args
 *  overrides so a config "opens where you left off" for cluster
 *  submits the same way it does for single-node ones. */
export interface MultinodeOverrides {
  rdzv_port: number;
  /** Ordered list of node ids the operator opted in. Empty implies
   *  "single-node submit" — no cluster fanout, even when the cluster
   *  is active. */
  selected_node_ids: string[];
  /** node_id → desired nproc per peer. */
  per_node_nproc: Record<string, number>;
  /** node_id → NCCL/Gloo/TP iface name; empty string means "auto"
   *  and the server derives it from the member's advertised IP. */
  per_node_iface: Record<string, string>;
  /** Which member hosts the rendezvous TCPStore. Defaults to master
   *  when null. */
  rdzv_node_id: string | null;
  /** Whether the user has acknowledged any version mismatch warnings
   *  the server returned on the previous submit attempt. */
  allow_version_mismatch: boolean;
}

/** Submit-modal dataset-source choice. ``server_id`` is one of:
 *  ``local:<queue_id>`` (a forgather_server-spawned dataset_server) or
 *  ``user:<entry_id>`` (a URL registered via Datasets → Servers → + Add).
 *  The token is never embedded here — the backend resolves it from the
 *  JobRecord / registry at submit time, so deleting an entry or stopping
 *  a local server invalidates the choice and surfaces as a 400. */
export type DatasetSource =
  | { kind: "local" }
  | { kind: "server"; server_id: string }
  | { kind: "auto" };

export interface OverridesData {
  values: Record<string, unknown>;
  requested_gpus: number | null;
  multinode: MultinodeOverrides | null;
  dataset_source: DatasetSource | null;
  updated_at: number | null;
}

/** Items still waiting for GPUs. Once dispatched they leave the queue and
 *  appear under /api/jobs as a Job with source="record" or "merged". */
export interface QueueItem {
  queue_id: string;
  project_dir: string;
  config: string;
  dynamic_args: Record<string, unknown>;
  requested_gpus: number;
  priority: number;
  submitted_at: number;
}

export interface SchedulerStatus {
  enabled: boolean;
  tick_count: number;
  last_tick_at: number | null;
  running_count: number;
}

export interface EnqueueRequest {
  project_dir: string;
  config: string;
  dynamic_args: Record<string, unknown>;
  requested_gpus: number;
  priority: number;
  /** Defaults to "training" server-side when omitted. */
  job_type?: "training"
    | "eval"
    | "inference"
    | "dataset_server"
    | "tensorboard"
    | "mkdocs"
    | "convert"
    | "finalize"
    | "update"
    | "model"
    | "dataset"
    | "construct";
  /** Type-specific payload; empty for training. */
  job_params?: Record<string, unknown>;
  /** Submit-modal dataset-source choice. Resolved server-side and
   *  merged into ``job_params.extra_env`` for training jobs. */
  dataset_source?: DatasetSource | null;
}

/** One row for the "pick an eval config" picker in EvalModal. */
export interface EvalConfigEntry {
  name: string;
  project_dir: string;
  template: string;
  description: string;
  default_batch_size: number;
  default_max_length: number;
  default_stride: number;
}

export interface ModelEntry {
  output_dir: string;
  model_name: string;
  configs: string[];
  exists: boolean;
  run_count: number;
  checkpoint_count: number;
  eval_count: number;
  total_size_bytes: number;
  parse_errors: Record<string, string>;
}

/** Mirrors forgather.eval_config.EvalResult. */
export interface EvalResultData {
  eval_name: string;
  config_name: string;
  description: string;
  dataset_proj: string;
  dataset_config: string;
  dataset_target: string;
  model_path: string;
  checkpoint_path: string | null;
  batch_size: number;
  max_length: number;
  stride: number;
  dtype: string;
  attn_implementation: string;
  trainer: string;
  world_size: number;
  eval_loss: number | null;
  perplexity: number | null;
  bpb: number | null;
  bpc: number | null;
  tokens_per_byte: number | null;
  total_bytes: number | null;
  total_chars: number | null;
  total_predicted_tokens: number | null;
  wall_time_s: number | null;
  timestamp: string | null;
}

export interface EvalEntry {
  eval_dir: string;
  eval_id: string;
  result: EvalResultData | null;
  parse_error: string | null;
}

export interface RunEntry {
  run_dir: string;
  run_id: string;
  /** Seconds since epoch (float). */
  started_at: number;
  has_logs: boolean;
  hostname: string | null;
  /** Absolute path to tty.log (or its symlink) if the run captured one. */
  tty_log_path: string | null;
}

export interface CheckpointEntry {
  checkpoint_dir: string;
  step: number;
  size_bytes: number;
  world_size: number | null;
  timestamp: string | null;
  manifest_present: boolean;
}

export interface RunSummary {
  summary: Record<string, unknown>;
  log_path: string | null;
  config_path: string | null;
  pp_path: string | null;
}

export interface IpynbCell {
  cell_type: string; // "markdown" | "code" | "raw"
  source: string;
  language: string | null;
  outputs: Record<string, unknown>[];
}

export interface DocsFile {
  path: string;
  kind: "markdown" | "ipynb";
  /** Set when ``kind === "markdown"``. */
  content: string | null;
  /** Set when ``kind === "ipynb"``. */
  cells: IpynbCell[] | null;
}

/** Thrown by ``putTemplateSource`` when the on-disk mtime is newer
 *  than the ``expected_mtime`` the client sent — i.e. someone else
 *  (or another tool) modified the file since the editor opened it.
 *  ``saveFile`` catches this and stashes it on the buffer's
 *  ``conflict`` field so the editor can prompt the user. */
export class SaveConflictError extends Error {
  currentMtime: number = 0;
  expectedMtime: number = 0;
  constructor(message: string) {
    super(message);
    this.name = "SaveConflictError";
  }
}

async function readErrorDetail(r: Response): Promise<unknown> {
  const text = await r.text();
  // FastAPI wraps everything in {"detail": ...}. When ``detail`` is a dict
  // (e.g. PreprocessErrorDetail) we want to surface the dict directly so the
  // UI can render a structured error; when it's a plain string we keep the
  // string. Fall back to the raw text on any parse failure.
  try {
    const parsed = JSON.parse(text);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "detail" in parsed
    ) {
      return (parsed as { detail: unknown }).detail;
    }
    return parsed;
  } catch {
    return text;
  }
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) {
    throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
  }
  return r.json() as Promise<T>;
}

async function fetchText(url: string): Promise<string> {
  const r = await fetch(url);
  if (!r.ok) {
    throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
  }
  return r.text();
}

/** Helper for dataset_server proxy GETs. Forwards the optional bearer
 *  token via the side-channel header so the user's forgather-server
 *  bearer (in Authorization) doesn't leak to the upstream. The proxy
 *  itself falls back to JobRecord / registry auto-lookup when token is
 *  blank — see ``routes/dataset_server.py::_auth_headers_for``. */
async function datasetServerProxyGet<T>(
  url: string,
  base: string,
  token: string,
): Promise<T> {
  const sep = url.includes("?") ? "&" : "?";
  const u = `${url}${sep}base=${encodeURIComponent(base)}`;
  const headers: Record<string, string> = {};
  if (token) headers["x-dataset-auth-token"] = token;
  const r = await fetch(u, { headers });
  if (!r.ok) {
    throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
  }
  return r.json() as Promise<T>;
}

/** Cluster-aggregated dataset_server entry. Tokens are stripped by
 *  the master before responding; this shape is safe for the browser. */
export interface ClusterDatasetServer {
  server_id: string;
  base_url: string;
  label: string;
  source: string; // "local" | "user"
  peer_node_id: string | null;
  healthy: boolean;
  last_health_check: number;
  last_health_error: string;
  last_dataset_refresh: number;
  last_dataset_error: string;
  // Polling counters from the master's loops.
  total_health_polls?: number;
  health_failures?: number;
  consecutive_health_failures?: number;
  total_dataset_polls?: number;
  dataset_failures?: number;
  consecutive_dataset_failures?: number;
  /** Per-entry TLS verification policy. False = chain validation off
   *  for outbound calls. Used for SSH-tunneled / out-of-band-secured
   *  upstreams; surfaced so the webui can show an "insecure TLS"
   *  badge. */
  verify_tls?: boolean;
  /** Source-side identifier on the owning peer — JobRecord
   *  ``queue_id`` for ``source === "local"``, registry entry id for
   *  ``source === "user"``. Used by the webui to target a DELETE
   *  for user-added entries owned by the local node. */
  source_id?: string | null;
  /** True when the URL's host is loopback. Cluster auto-routing
   *  skips these; the webui shows them with a "node-local" badge. */
  loopback?: boolean;
}

/** Master-aggregated inference-server entry, used by the cluster
 *  picker in InferenceModelPanel.
 *
 *  Includes ``auth_token`` so the panel can attach the upstream
 *  bearer via the ``X-Inference-Auth-Token`` header — the proxy on
 *  non-master peers can't auto-discover off-host tokens, so the
 *  picker carries it. This matches the per-job ``auth_token`` field
 *  that already flows through ``/api/jobs`` for locally-spawned
 *  servers.
 *
 *  Hardening this surface (server-side token attach via a non-master
 *  pull loop, browser never sees tokens) is tracked as follow-up. */
export interface ClusterInferenceServer {
  server_id: string;
  base_url: string;
  auth_token: string;
  label: string;
  peer_node_id: string | null;
  source_id?: string | null;
  loopback?: boolean;
  /** Configured OpenAI routing names this server hosts (one entry
   *  per ``--model`` flag the inference server was launched with).
   *  Empty list ⇒ the JobRecord didn't carry a usable hint; the
   *  picker falls back to the URL as the label. */
  models: string[];
  healthy: boolean;
  last_health_check: number;
  last_health_error: string;
  total_health_polls?: number;
  health_failures?: number;
  consecutive_health_failures?: number;
}

/** Aggregate counters across the inventory. */
export interface ClusterDatasetInventoryMetrics {
  healthy_servers: number;
  unhealthy_servers: number;
  total_servers: number;
  total_datasets: number;
  total_health_polls: number;
  total_health_failures: number;
  total_dataset_polls: number;
  total_dataset_failures: number;
  master_age_seconds: number | null;
}

/** One unique dataset in the cluster (deduped across servers). */
export interface ClusterDatasetEntry {
  dataset_id: string;
  source: string; // "local" | "hf" | "path"
  name: string | null;
  load_args: Record<string, unknown> | null;
  length: number | null;
  column_names: string[] | null;
  server_ids: string[];
  /** Total on-disk size in bytes across the dataset (sum of splits
   *  for HF / cluster-aggregated local entries). Null when unknown. */
  size_bytes?: number | null;
  /** Distinct ``meta_hash`` values observed across servers
   *  advertising this name. Multiple values = collision. */
  meta_hashes?: string[];
}

export interface ClusterDatasetInventoryResponse {
  is_master: boolean;
  master_become_ts: number | null;
  last_servers_collect_ts: number | null;
  last_health_pass_ts: number | null;
  last_dataset_pass_ts: number | null;
  servers: ClusterDatasetServer[];
  datasets: ClusterDatasetEntry[];
  metrics?: ClusterDatasetInventoryMetrics;
}

/** Dataset server registered as a user-added entry. */
export interface DatasetServerUser {
  id: string;
  label: string;
  base_url: string;
  has_auth_token: boolean;
  /** False = TLS chain + hostname validation off for outbound calls
   *  to this URL. Operator-asserted for SSH-tunneled / out-of-band-
   *  secured upstreams. Default true (secure-by-default). */
  verify_tls?: boolean;
}

/** Dataset server spawned by the forgather_server itself. */
export interface DatasetServerLocal {
  queue_id: string;
  label: string;
  base_url: string;
  host: string;
  port: number;
  alive: boolean;
  has_auth_token: boolean;
}

export interface AddDatasetServerRequest {
  label?: string;
  base_url: string;
  auth_token?: string;
  /** False = TLS chain + hostname validation off for outbound calls
   *  to this URL. Operator-asserted for SSH-tunneled / out-of-band-
   *  secured upstreams. Defaults to true on the server side when
   *  omitted. */
  verify_tls?: boolean;
}

/** Inference server registered as a user-added entry. Same shape as
 *  the dataset-server variant; the two registries live in separate
 *  files on disk and are surfaced by separate routes. */
export interface InferenceServerUser {
  id: string;
  label: string;
  base_url: string;
  has_auth_token: boolean;
  /** False = TLS chain + hostname validation off for outbound calls
   *  to this URL. Operator-asserted for SSH-tunneled / out-of-band-
   *  secured upstreams. Default true (secure-by-default). */
  verify_tls?: boolean;
}

export interface AddInferenceServerRequest {
  label?: string;
  base_url: string;
  auth_token?: string;
  verify_tls?: boolean;
}

/** One row from ``GET /v1/datasets``. Field set tracks what the
 *  dataset_server's wire model exposes; we mirror only what we render. */
export interface DatasetHandleEntry {
  handle: string;
  length?: number | null;
  source?: string | null;
  load_args?: Record<string, unknown> | null;
}

/** ``GET /v1/cache/hf`` per-split entry. */
export interface HFCacheSplit {
  name: string;
  num_examples?: number | null;
  num_bytes?: number | null;
}
export interface HFCacheConfig {
  config: string;
  version?: string | null;
  size_bytes?: number | null;
  splits: HFCacheSplit[];
}
export interface HFCacheRepo {
  repo: string;
  size_bytes?: number | null;
  configs: HFCacheConfig[];
}
export interface HFCacheResponse {
  cache_root: string;
  datasets: HFCacheRepo[];
}

/** One entry from the (enriched) ``GET /v1/local`` response. */
export interface LocalDatasetEntry {
  name: string;
  path: string;
  layout?: "dataset_dict" | "dataset" | "unknown" | "missing";
  size_bytes?: number | null;
  config_name?: string | null;
  dataset_name?: string | null;
  features?: string[];
  splits?: HFCacheSplit[];
}
export interface LocalListResponse {
  local: LocalDatasetEntry[];
}

/** ``GET /v1/health`` response. */
export interface DatasetServerHealth {
  status: string;          // "ok"
  service: string;         // "forgather-dataset-server"
  version: string;         // "1.0.0"
  policy: {
    auth_required: boolean;
    hf_cache_enabled: boolean;
    allow_paths: boolean;
    allow_downloads: boolean;
    local_count: number;
  };
}

/** ``GET /v1/datasets`` response — currently-loaded handles. */
export interface DatasetHandleRow {
  handle: string;
  length: number;
  source: string | null;
  load_args: Record<string, unknown>;
}
export interface DatasetHandlesResponse {
  handles: DatasetHandleRow[];
}

/** ``POST /v1/load`` response. */
export interface LoadResponse {
  handle: string;
  length: number;
  load_args: Record<string, unknown>;
  source: string | null;
  column_names: string[] | null;
}

/** Body of ``POST /v1/load``. */
export interface LoadRequest {
  path: string;
  name?: string;
  split?: string;
  data_files?: unknown;
  revision?: string;
}

/** ``GET /v1/datasets/{handle}/iter`` (wrapped by our proxy as JSON). */
export interface IterResponse {
  rows: Array<Record<string, unknown>>;
}

export const api = {
  listSearchRoots: () => fetchJson<SearchRoot[]>("/api/search-roots"),
  addSearchRoot: async (path: string, create = false) => {
    const r = await fetch("/api/search-roots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, create }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json() as Promise<SearchRoot>;
  },
  removeSearchRoot: async (path: string) => {
    const r = await fetch(
      `/api/search-roots?path=${encodeURIComponent(path)}`,
      { method: "DELETE" },
    );
    if (!r.ok) throw new Error(await r.text());
  },
  listProjects: () => fetchJson<WorkspaceCluster[]>("/api/projects"),
  configRaw: (path: string) =>
    fetchText(`/api/config/raw?path=${encodeURIComponent(path)}`),
  configPp: (project_dir: string, config: string) =>
    fetchText(
      `/api/config/pp?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
    ),
  configDebug: (project_dir: string, config: string) =>
    fetchJson<DebugTraceItem[]>(
      `/api/config/debug?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
    ),
  configCodeTargets: (project_dir: string, config: string) =>
    fetchJson<string[]>(
      `/api/config/code-targets?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
    ),
  /** Render *target* (or the entire config when `target` is the empty string)
   *  as Python source. Backend default matches the CLI's ``forgather code``
   *  default of ``main``. */
  configCode: (project_dir: string, config: string, target: string = "main") =>
    fetchText(
      `/api/config/code?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}&target=${encodeURIComponent(target)}`,
    ),
  configTrefsJson: (project_dir: string, config: string) =>
    fetchJson<TrefsGraph>(
      `/api/config/trefs?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}&format=json`,
    ),
  configTrefsDot: (project_dir: string, config: string) =>
    fetchText(
      `/api/config/trefs?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}&format=dot`,
    ),
  /** Config node graph as Graphviz DOT. Pass an empty *target* to render
   *  all top-level targets in a single diagram. When *include_values* is
   *  true, plain scalars / lists / dicts also appear as graph nodes. */
  configGraphDot: (
    project_dir: string,
    config: string,
    target: string = "",
    include_values: boolean = false,
  ) =>
    fetchText(
      `/api/config/graph?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}${target ? `&target=${encodeURIComponent(target)}` : ""}${include_values ? "&include_values=true" : ""}`,
    ),
  templateSource: (path: string) =>
    fetchText(`/api/template/source?path=${encodeURIComponent(path)}`),
  templateSourceWithMeta: async (
    path: string,
  ): Promise<{ content: string; mtime: number }> => {
    const r = await fetch(
      `/api/template/source?path=${encodeURIComponent(path)}`,
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    const mtime = parseFloat(r.headers.get("X-Mtime") ?? "0") || 0;
    const content = await r.text();
    return { content, mtime };
  },
  listProjectTemplates: (project_dir: string) =>
    fetchJson<TemplateGroup[]>(
      `/api/project/templates?project_dir=${encodeURIComponent(project_dir)}`,
    ),
  projectTemplatePaths: (project_dir: string) =>
    fetchJson<{
      templates_dir: string;
      configs_dir: string;
      config_prefix: string;
    }>(
      `/api/project/template-paths?project_dir=${encodeURIComponent(project_dir)}`,
    ),
  initWorkspaceHere: async (req: {
    workspace_dir: string;
    name: string;
    description: string;
    forgather_dir: string;
    libs?: string[];
    search_paths?: string[];
  }): Promise<{ workspace_dir: string }> => {
    const r = await fetch("/api/workspace/init-here", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  newWorkspace: async (req: {
    parent_dir: string;
    name: string;
    description: string;
    workspace_dir_name?: string | null;
    forgather_dir: string;
    libs?: string[];
    search_paths?: string[];
  }): Promise<{ workspace_dir: string }> => {
    const r = await fetch("/api/workspace/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  newProject: async (req: {
    workspace_dir: string;
    name: string;
    description: string;
    config_prefix?: string;
    default_config?: string;
    project_dir_name?: string | null;
    copy_from?: string | null;
  }): Promise<{ project_dir: string }> => {
    const r = await fetch("/api/workspace/new-project", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  newProjectTemplate: async (
    project_dir: string,
    kind: "config" | "template",
    name: string,
  ): Promise<{ path: string }> => {
    const r = await fetch("/api/project/new-template", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_dir, kind, name }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  putTemplateSource: async (
    path: string,
    content: string,
    expected_mtime?: number,
  ): Promise<{ path: string; bytes_written: number; mtime: number }> => {
    const r = await fetch("/api/template/source", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path,
        content,
        expected_mtime: expected_mtime ?? null,
      }),
    });
    if (r.status === 409) {
      // Optimistic-concurrency conflict. The body's `detail` is a JSON
      // object: ``{message, current_mtime, expected_mtime}``. Hoist
      // those onto a typed error so saveFile can branch on it.
      let detail: any = null;
      try {
        const body = await r.json();
        detail = body?.detail ?? body;
      } catch {
        // body wasn't JSON
      }
      const err = new SaveConflictError(
        detail?.message ?? "file changed on disk since you opened it",
      );
      err.currentMtime = detail?.current_mtime ?? 0;
      err.expectedMtime = detail?.expected_mtime ?? 0;
      throw err;
    }
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    return r.json();
  },
  configMeta: (project_dir: string, config: string) =>
    fetchJson<ConfigMeta>(
      `/api/config/meta?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
    ),
  fsBrowse: (path: string, show_hidden = false, files_too = false) =>
    fetchJson<FsListing>(
      `/api/fs/browse?path=${encodeURIComponent(path)}&show_hidden=${show_hidden}&files_too=${files_too}`,
    ),
  fsQuickPaths: () => fetchJson<QuickPath[]>("/api/fs/quick-paths"),
  fsPathExists: (path: string) =>
    fetchJson<{ exists: boolean; is_file: boolean; is_dir: boolean }>(
      `/api/fs/path-exists?path=${encodeURIComponent(path)}`,
    ),
  fsMkdir: async (parent: string, name: string): Promise<{ path: string }> => {
    const r = await fetch("/api/fs/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parent, name }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  fsNewFile: async (
    parent: string,
    name: string,
  ): Promise<{ path: string }> => {
    const r = await fetch("/api/fs/new-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parent, name }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  fsRename: async (
    path: string,
    new_name: string,
  ): Promise<{ path: string }> => {
    const r = await fetch("/api/fs/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, new_name }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  fsCopy: async (
    src: string,
    dest_dir: string,
    opts: { autoRename?: boolean; targetName?: string } = {},
  ): Promise<{ path: string }> => {
    const r = await fetch("/api/fs/copy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        src,
        dest_dir,
        auto_rename: !!opts.autoRename,
        target_name: opts.targetName ?? null,
      }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  fsMove: async (
    src: string,
    dest_dir: string,
  ): Promise<{ path: string }> => {
    const r = await fetch("/api/fs/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src, dest_dir }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
    return r.json();
  },
  listGpus: () => fetchJson<GpuInfo[]>("/api/gpus"),
  /** Returns null when the server is in standalone mode. */
  getClusterSelf: () => fetchJson<ClusterSelf | null>("/api/cluster/self"),
  getClusterMembers: () =>
    fetchJson<ClusterMembersResponse>("/api/cluster/members"),
  getClusterGpus: () =>
    fetchJson<ClusterGpusResponse>("/api/cluster/gpus"),
  getClusterBandwidth: () =>
    fetchJson<ClusterBandwidthResponse>("/api/cluster/bandwidth"),
  listClusterJobs: () => fetchJson<ClusterJob[]>("/api/cluster/jobs"),
  submitClusterJob: async (
    req: ClusterJobSubmitRequest,
  ): Promise<ClusterJobSubmitResponse> => {
    const r = await fetch("/api/cluster/jobs/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<ClusterJobSubmitResponse>;
  },
  cancelClusterJob: async (
    cluster_job_id: string,
  ): Promise<ClusterJobCancelResponse> => {
    const r = await fetch(
      `/api/cluster/jobs/${encodeURIComponent(cluster_job_id)}/cancel`,
      { method: "POST" },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<ClusterJobCancelResponse>;
  },
  /** Mint a one-click URL the browser can open to log into a peer node
   *  via the cluster bearer carve-out. The peer's webui consumes the
   *  ``?token=`` query and strips it from the address bar on first
   *  paint, leaving a normal session cookie. */
  peerSessionUrl: async (
    node_id: string,
  ): Promise<{ url: string; hostname: string }> => {
    const r = await fetch("/api/cluster/peer_session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id }),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json();
  },
  refreshClusterBandwidth: async (): Promise<ClusterBandwidthResponse> => {
    const r = await fetch("/api/cluster/bandwidth/refresh", {
      method: "POST",
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<ClusterBandwidthResponse>;
  },
  refreshClusterBandwidthOne: async (
    nodeId: string,
  ): Promise<ClusterBandwidthEntry> => {
    const r = await fetch(
      `/api/cluster/bandwidth/refresh_one/${encodeURIComponent(nodeId)}`,
      { method: "POST" },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<ClusterBandwidthEntry>;
  },
  getClusterLatency: () =>
    fetchJson<ClusterLatencyResponse>("/api/cluster/latency"),
  refreshClusterLatencyOne: async (
    nodeId: string,
  ): Promise<ClusterLatencyEntry> => {
    const r = await fetch(
      `/api/cluster/latency/refresh_one/${encodeURIComponent(nodeId)}`,
      { method: "POST" },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<ClusterLatencyEntry>;
  },
  /** Master-proxied GPU policy mutation. Routes to the named node's
   *  /api/cluster/gpu_policy_local; short-circuits when the target is
   *  the local node. Returns the updated policy from the owning node. */
  setNodeGpuPolicy: async (
    node_id: string,
    gpu_index: number,
    policy: { disabled?: boolean; min_priority?: number },
  ): Promise<GpuPolicy> => {
    const r = await fetch(
      `/api/cluster/nodes/${encodeURIComponent(node_id)}/gpus/${gpu_index}/policy`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(policy),
      },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<GpuPolicy>;
  },
  gpuStreamUrl: (): string => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/gpus/stream`;
  },
  listJobs: (includeDead = false) =>
    fetchJson<Job[]>(`/api/jobs?include_dead_endpoints=${includeDead}`),
  jobStatus: (jobId: string) =>
    fetchJson<Record<string, unknown>>(
      `/api/jobs/${encodeURIComponent(jobId)}/status`,
    ),
  jobControl: async (jobId: string, action: ControlAction) => {
    const r = await fetch(
      `/api/jobs/${encodeURIComponent(jobId)}/control/${action}`,
      { method: "POST" },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    return r.json() as Promise<ControlResponse>;
  },
  removeJob: async (jobId: string) => {
    const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
      method: "DELETE",
    });
    if (!r.ok) throw new Error(await r.text());
  },
  cleanupJobs: async (): Promise<{ removed: string[]; count: number }> => {
    const r = await fetch("/api/jobs/cleanup", { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  ttyStreamUrl: (job_id: string, follow = true): string => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/jobs/${encodeURIComponent(job_id)}/tty?follow=${follow}`;
  },
  ttyDump: (job_id: string) =>
    fetchText(`/api/jobs/${encodeURIComponent(job_id)}/tty`),
  listQueue: () => fetchJson<QueueItem[]>("/api/queue"),
  enqueue: async (req: EnqueueRequest) => {
    const r = await fetch("/api/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    return r.json() as Promise<QueueItem>;
  },
  abortQueueItem: async (queue_id: string) => {
    const r = await fetch(`/api/queue/${encodeURIComponent(queue_id)}`, {
      method: "DELETE",
    });
    if (!r.ok) throw new Error(await r.text());
  },
  setGpuPolicy: async (
    gpu_index: number,
    policy: { disabled?: boolean; min_priority?: number },
  ): Promise<GpuPolicy> => {
    const r = await fetch(`/api/gpus/${gpu_index}/policy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(policy),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<GpuPolicy>;
  },
  getGpuPolicy: () =>
    fetchJson<Record<string, GpuPolicy>>("/api/gpus/policy"),
  killGpuProcesses: async (gpu_index: number) => {
    const r = await fetch(`/api/gpus/${gpu_index}/kill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmed: true }),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<{
      gpu_index: number;
      pids: number[];
      killed: number[];
      failed: number[];
    }>;
  },
  schedulerStatus: () => fetchJson<SchedulerStatus>("/api/queue/scheduler"),
  schedulerToggle: async (enabled: boolean) => {
    const r = await fetch("/api/queue/scheduler", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json() as Promise<SchedulerStatus>;
  },
  dynamicArgs: (project_dir: string, config: string) =>
    fetchJson<DynamicArg[]>(
      `/api/config/dynamic-args?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
    ),
  configOutputDir: (project_dir: string, config: string) =>
    fetchJson<OutputDirInfo>(
      `/api/config/output-dir?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
    ),
  getOverrides: (project_dir: string, config: string) =>
    fetchJson<OverridesData>(
      `/api/config/overrides?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
    ),
  setOverrides: async (
    project_dir: string,
    config: string,
    values: Record<string, unknown>,
    requested_gpus?: number | null,
    multinode?: MultinodeOverrides | null,
    dataset_source?: DatasetSource | null,
  ): Promise<OverridesData> => {
    const r = await fetch("/api/config/overrides", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_dir,
        config,
        values,
        requested_gpus: requested_gpus ?? null,
        multinode: multinode ?? null,
        dataset_source: dataset_source ?? null,
      }),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    return r.json() as Promise<OverridesData>;
  },
  clearOverrides: async (
    project_dir: string,
    config: string,
  ): Promise<{ cleared: boolean }> => {
    const r = await fetch(
      `/api/config/overrides?project_dir=${encodeURIComponent(project_dir)}&config=${encodeURIComponent(config)}`,
      { method: "DELETE" },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    return r.json() as Promise<{ cleared: boolean }>;
  },
  projectReadme: (project_dir: string) =>
    fetchText(
      `/api/project/readme?project_dir=${encodeURIComponent(project_dir)}`,
    ),
  docsRoot: () => fetchJson<{ path: string | null }>("/api/docs/root"),
  docsRepoRoot: () => fetchJson<{ repo_root: string }>("/api/docs/repo-root"),
  serverConfigPath: () =>
    fetchJson<{ path: string | null }>("/api/server-config-path"),
  listServices: () => fetchJson<ServiceStatus[]>("/api/services"),
  upsertService: async (
    type: string,
    name: string,
    enabled: boolean,
    args: Record<string, unknown>,
  ) => {
    const r = await fetch("/api/services", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, name, enabled, args }),
    });
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<ServiceStatus>;
  },
  deleteService: async (type: string, name: string) => {
    const r = await fetch(
      `/api/services/${encodeURIComponent(type)}/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json();
  },
  restartServer: async () => {
    const r = await fetch("/api/server/restart", { method: "POST" });
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<{ restart: string }>;
  },
  shutdownServer: async (opts: { stopJobs?: boolean } = {}) => {
    const r = await fetch("/api/server/shutdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stop_jobs: !!opts.stopJobs }),
    });
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<{
      shutdown: string;
      stopped_jobs: string[];
    }>;
  },
  // Cluster maintenance: master forwards to the named node via mTLS.
  // For the local node, the master short-circuits to the local helper.
  restartNode: async (nodeId: string) => {
    const r = await fetch(
      `/api/cluster/nodes/${encodeURIComponent(nodeId)}/restart`,
      { method: "POST" },
    );
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<{ restart: string }>;
  },
  shutdownNode: async (
    nodeId: string,
    opts: { stopJobs?: boolean } = {},
  ) => {
    const r = await fetch(
      `/api/cluster/nodes/${encodeURIComponent(nodeId)}/shutdown`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stop_jobs: !!opts.stopJobs }),
      },
    );
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<{
      shutdown: string;
      stopped_jobs: string[];
    }>;
  },
  setServiceEnabled: async (type: string, name: string, enabled: boolean) => {
    const r = await fetch(
      `/api/services/${encodeURIComponent(type)}/${encodeURIComponent(name)}/enabled`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      },
    );
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<ServiceStatus>;
  },
  listLocalDatasetServers: () =>
    fetchJson<DatasetServerLocal[]>("/api/dataset-servers/local"),
  localDatasetServerBundle: (queue_id: string) =>
    fetchJson<{ bundle: string }>(
      `/api/dataset-servers/local/${encodeURIComponent(queue_id)}/bundle`,
    ),
  listUserDatasetServers: () =>
    fetchJson<DatasetServerUser[]>("/api/dataset-servers/user"),
  addUserDatasetServer: async (
    req: AddDatasetServerRequest,
  ): Promise<DatasetServerUser> => {
    const r = await fetch("/api/dataset-servers/user", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    return r.json();
  },
  deleteUserDatasetServer: async (id: string): Promise<void> => {
    const r = await fetch(
      `/api/dataset-servers/user/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
  },

  // User-added inference servers. Same CRUD shape as the dataset-server
  // registry — the picker in InferenceModelPanel surfaces these alongside
  // running local/cluster servers so operators don't have to retype
  // external URLs (vLLM, remote OpenAI-compatible boxes) every session.
  listUserInferenceServers: () =>
    fetchJson<InferenceServerUser[]>("/api/inference-servers/user"),
  addUserInferenceServer: async (
    req: AddInferenceServerRequest,
  ): Promise<InferenceServerUser> => {
    const r = await fetch("/api/inference-servers/user", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    return r.json();
  },
  deleteUserInferenceServer: async (id: string): Promise<void> => {
    const r = await fetch(
      `/api/inference-servers/user/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
  },
  /** Reveal the stored bearer token for a user-added entry — only
   *  called when the operator explicitly asks (Show / Copy on the
   *  Auth-token field), so the secret crosses the wire on demand
   *  rather than every time a server is picked. Refused server-side
   *  in demo mode (403). */
  getUserInferenceServerToken: async (id: string): Promise<string> => {
    const r = await fetch(
      `/api/inference-servers/user/${encodeURIComponent(id)}/token`,
    );
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status} ${r.statusText}: ${detail}`);
    }
    const body = (await r.json()) as { auth_token: string };
    return body.auth_token ?? "";
  },

  // Proxy GETs. ``token`` is the upstream bearer that's forwarded via
  // the X-Dataset-Auth-Token side-channel; empty string means no
  // explicit token (the proxy then falls back to JobRecord auto-lookup
  // for local servers, or registry lookup for saved user entries).
  datasetServerHealth: (base: string, token: string) =>
    datasetServerProxyGet<DatasetServerHealth>(
      "/api/dataset-server/proxy/health",
      base,
      token,
    ),
  datasetServerAuthStatus: (base: string, token: string) =>
    datasetServerProxyGet<{ auth_required: boolean }>(
      "/api/dataset-server/proxy/auth-status",
      base,
      token,
    ),
  datasetServerDatasets: (base: string, token: string) =>
    datasetServerProxyGet<DatasetHandlesResponse>(
      "/api/dataset-server/proxy/datasets",
      base,
      token,
    ),
  datasetServerCache: (base: string, token: string) =>
    datasetServerProxyGet<HFCacheResponse>(
      "/api/dataset-server/proxy/cache",
      base,
      token,
    ),
  datasetServerLocal: (base: string, token: string) =>
    datasetServerProxyGet<LocalListResponse>(
      "/api/dataset-server/proxy/local",
      base,
      token,
    ),
  /** POST a ``LoadRequest`` to the proxy; returns the handle + length +
   *  ``column_names`` the dataset_server reports. ``token`` is the
   *  optional explicit bearer; empty string defers to proxy auto-lookup. */
  datasetServerLoad: async (
    base: string,
    body: LoadRequest,
    token: string,
  ): Promise<LoadResponse> => {
    const u = `/api/dataset-server/proxy/load?base=${encodeURIComponent(base)}`;
    const headers: Record<string, string> = {
      "content-type": "application/json",
    };
    if (token) headers["x-dataset-auth-token"] = token;
    const r = await fetch(u, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<LoadResponse>;
  },
  datasetServerLength: (base: string, handle: string, token: string) =>
    datasetServerProxyGet<{ length: number }>(
      `/api/dataset-server/proxy/length?handle=${encodeURIComponent(handle)}`,
      base,
      token,
    ),
  datasetServerIter: (
    base: string,
    handle: string,
    position: number,
    limit: number,
    token: string,
  ) =>
    datasetServerProxyGet<IterResponse>(
      `/api/dataset-server/proxy/iter?handle=${encodeURIComponent(handle)}` +
        `&position=${position}&limit=${limit}`,
      base,
      token,
    ),

  // ---- Cluster-aggregated dataset inventory + master proxy ----

  /** Master-aggregated dataset-server inventory + dataset listing.
   *  Same shape the ``forgather cluster datasets`` CLI consumes. */
  getClusterDatasetInventory: () =>
    fetchJson<ClusterDatasetInventoryResponse>(
      "/api/cluster/dataset_inventory",
    ),
  /** Token-stripped server list — same as ``inventory.servers`` but
   *  a smaller payload for the Explore + Servers tabs. */
  getClusterDatasetServers: () =>
    fetchJson<ClusterDatasetServer[]>("/api/cluster/dataset_servers"),
  /** Cluster-aggregated inference servers (master-polled + health-
   *  tracked). Includes ``auth_token`` per entry so the
   *  InferenceModelPanel can pin the bearer in the
   *  ``X-Inference-Auth-Token`` header when dialling a remote-peer
   *  upstream — same surface the per-job ``auth_token`` on
   *  :ref:`/api/jobs` already exposes for locally-spawned servers.
   *  Returns an empty list when cluster mode is inactive. */
  getClusterInferenceServers: () =>
    fetchJson<ClusterInferenceServer[]>("/api/cluster/inference_servers"),
  /** Wake the master's collect/health/refresh loops on demand.
   *  Best-effort — fire-and-forget. Used right after a registry
   *  add/delete so the cluster inventory reflects the change within
   *  ~1s rather than waiting up to one collect tick. */
  refreshClusterDatasetServers: async (): Promise<void> => {
    try {
      await fetch("/api/cluster/dataset_servers/refresh", {
        method: "POST",
      });
    } catch {
      // Latency hint only — silently ignore failures.
    }
  },

  /** Cluster-proxied probes against a single dataset_server. The master
   *  injects the bearer from its inventory, so the browser only needs
   *  the cluster bearer. ``server_id`` comes from the cluster inventory. */
  clusterDatasetServerHealth: (server_id: string) =>
    fetchJson<DatasetServerHealth>(
      `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/health`,
    ),
  clusterDatasetServerAuthStatus: (server_id: string) =>
    fetchJson<{ auth_required: boolean }>(
      `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/auth-status`,
    ),
  clusterDatasetServerDatasets: (server_id: string) =>
    fetchJson<DatasetHandlesResponse>(
      `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/datasets`,
    ),
  clusterDatasetServerCache: (server_id: string) =>
    fetchJson<HFCacheResponse>(
      `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/cache`,
    ),
  clusterDatasetServerLocal: (server_id: string) =>
    fetchJson<LocalListResponse>(
      `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/local`,
    ),
  clusterDatasetServerLoad: async (
    server_id: string,
    body: LoadRequest,
  ): Promise<LoadResponse> => {
    const u = `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/load`;
    const r = await fetch(u, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      throw new ApiError(r.status, r.statusText, await readErrorDetail(r));
    }
    return r.json() as Promise<LoadResponse>;
  },
  clusterDatasetServerLength: (server_id: string, handle: string) =>
    fetchJson<{ length: number }>(
      `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/length` +
        `?handle=${encodeURIComponent(handle)}`,
    ),
  clusterDatasetServerIter: (
    server_id: string,
    handle: string,
    position: number,
    limit: number,
  ) =>
    fetchJson<IterResponse>(
      `/api/cluster/dataset_server_proxy/${encodeURIComponent(server_id)}/iter` +
        `?handle=${encodeURIComponent(handle)}` +
        `&position=${position}&limit=${limit}`,
    ),

  docsFile: (path: string) =>
    fetchJson<DocsFile>(`/api/docs/file?path=${encodeURIComponent(path)}`),
  docsAssetUrl: (path: string): string =>
    `/api/docs/asset?path=${encodeURIComponent(path)}`,
  projectAssetUrl: (project_dir: string, asset: string): string =>
    `/api/project/asset?project_dir=${encodeURIComponent(project_dir)}&asset=${encodeURIComponent(asset)}`,
  listProjectModels: (project_dir: string) =>
    fetchJson<ModelEntry[]>(
      `/api/project/models?project_dir=${encodeURIComponent(project_dir)}`,
    ),
  listModelRuns: (output_dir: string) =>
    fetchJson<RunEntry[]>(
      `/api/model/runs?output_dir=${encodeURIComponent(output_dir)}`,
    ),
  runSummary: (run_dir: string) =>
    fetchJson<RunSummary>(
      `/api/run/summary?run_dir=${encodeURIComponent(run_dir)}`,
    ),
  listModelCheckpoints: (output_dir: string) =>
    fetchJson<CheckpointEntry[]>(
      `/api/model/checkpoints?output_dir=${encodeURIComponent(output_dir)}`,
    ),
  listModelEvaluations: (output_dir: string) =>
    fetchJson<EvalEntry[]>(
      `/api/model/evaluations?output_dir=${encodeURIComponent(output_dir)}`,
    ),
  listEvalConfigs: () => fetchJson<EvalConfigEntry[]>("/api/eval-configs"),
  runTty: (run_dir: string) =>
    fetchText(`/api/run/tty?run_dir=${encodeURIComponent(run_dir)}`),
  deleteFile: async (path: string): Promise<DeleteDirResponse> => {
    const r = await fetch("/api/fs/delete-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, confirmed: true }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json();
  },
  deleteDir: async (path: string) => {
    const r = await fetch("/api/fs/delete-dir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, confirmed: true }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json() as Promise<DeleteDirResponse>;
  },
  listGenerationConfigs: () =>
    fetchJson<{ presets: { name: string; builtin: boolean }[] }>(
      "/api/generation-configs",
    ),
  getGenerationConfig: (name: string) =>
    fetchJson<{
      name: string;
      builtin: boolean;
      params: Record<string, unknown>;
    }>(`/api/generation-configs/${encodeURIComponent(name)}`),
  putGenerationConfig: async (
    name: string,
    params: Record<string, unknown>,
  ) => {
    const r = await fetch(
      `/api/generation-configs/${encodeURIComponent(name)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      },
    );
    if (!r.ok) throw new Error(await r.text());
    return r.json() as Promise<{
      name: string;
      builtin: boolean;
      params: Record<string, unknown>;
    }>;
  },
  deleteGenerationConfig: async (name: string) => {
    const r = await fetch(
      `/api/generation-configs/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
    if (!r.ok) {
      let detail = await r.text();
      try {
        detail = JSON.parse(detail).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(detail);
    }
  },
};
