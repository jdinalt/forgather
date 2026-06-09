import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import { api } from "../api";
import { persistGet, persistSet } from "../persist";
import {
  promptAndCreateService,
  saveServiceArgsAndMaybeRestart,
} from "../services-create";
import { ModalBackdrop } from "./ModalBackdrop";
import { PathField } from "./PathField";

const STORAGE_KEY = "forgather-diloco-server-modal";

/** A section whose body collapses behind a clickable header bar. Most runs
 *  leave the advanced groups at their defaults, so they start collapsed to
 *  keep the modal from feeling overwhelming; the operator expands what they
 *  need. Uses a native (uncontrolled) <details> so a closed section renders
 *  nothing but its summary bar — no empty bordered box — mirroring the
 *  Submit modal's ``.submit-section`` pattern. These sit as flat siblings in
 *  the modal grid (no nesting), so there's no toggle-bubbling to guard. */
function CollapsibleSection({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details
      className="collapsible-section"
      open={defaultOpen}
      style={{ gridColumn: "1 / -1" }}
    >
      <summary>{title}</summary>
      <div style={{ padding: 8 }}>{children}</div>
    </details>
  );
}

interface PersistedAdHoc {
  outputDir: string;
  port: number;
  numWorkers: number;
  host: string;
  async: boolean;
  dnBufferSize: number;
  dylu: boolean;
  dyluBase: number;
  // Group-wide worker settings the server is authoritative for. They
  // must match across the group, so the operator sets them here and the
  // workers adopt them from /info (no per-worker knob).
  syncEvery: number;
  numFragments: number;
  // Wire precision (issue #130). Four server-authoritative knobs covering
  // each leg × dtype-vs-stochastic-rounding. ``uploadDtype`` replaces the
  // legacy ``bf16Comm`` boolean (which was just upload bf16-vs-fp32).
  uploadDtype: string;
  uploadSr: boolean;
  downloadDtype: string;
  downloadSr: boolean;
  fromCheckpoint: string;
  // Optional label for this run's stats log dir (runs/<timestamp>_<runName>),
  // holding the JSONL stream + TensorBoard events. Honored on a fresh start;
  // a resume from checkpoint continues the prior run's dir regardless.
  runName: string;
  saveEvery: number;
  saveTotalLimit: number;
  outerLr: number;
  outerMomentum: number;
  noNesterov: boolean;
  heartbeatTimeout: number;
  minWorkers: number;
  // Security (issue #90): control-plane auth knobs.
  noAuth: boolean;
  regenToken: boolean;
  // Cleartext bulk plane. When on, the server serves the bulk
  // endpoints on a separate cleartext listener on an ephemeral port it
  // picks itself — bypassing TLS for throughput on a trusted LAN.
  bulkCleartext: boolean;
  // Bulk transport (issue #154). Wire codec for the tensor legs and the
  // optional gRPC streaming listener. Server-authoritative — workers adopt
  // them from /info. gRPC supersedes the cleartext bulk plane.
  wireFormat: string;
  grpcEnabled: boolean;
  // Sync backend the worker group must use (issue #154). Declared here and
  // advertised via /info; workers validate their own backend against it and
  // fail loud on disagreement. Must match `submit --backend` for the workers.
  backend: string;
}

const DEFAULT_AD_HOC: PersistedAdHoc = {
  outputDir: "",
  port: 8512,
  numWorkers: 2,
  host: "127.0.0.1",
  async: false,
  dnBufferSize: 0,
  dylu: false,
  dyluBase: 500,
  syncEvery: 500,
  numFragments: 1,
  uploadDtype: "bf16",
  uploadSr: false,
  downloadDtype: "fp32",
  downloadSr: false,
  fromCheckpoint: "",
  runName: "",
  saveEvery: 10,
  saveTotalLimit: 3,
  outerLr: 0.7,
  outerMomentum: 0.9,
  noNesterov: false,
  heartbeatTimeout: 120,
  minWorkers: 1,
  noAuth: false,
  regenToken: false,
  bulkCleartext: false,
  wireFormat: "pickle",
  grpcEnabled: false,
  backend: "http",
};

function loadPersisted(): PersistedAdHoc {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return DEFAULT_AD_HOC;
  try {
    const parsed = JSON.parse(raw) as Partial<PersistedAdHoc> & {
      bf16Comm?: boolean;
    };
    const merged = { ...DEFAULT_AD_HOC, ...parsed };
    // Migrate the legacy ``bf16Comm`` boolean (upload bf16 vs fp32) when a
    // pre-#130 blob predates the explicit uploadDtype knob, so a saved fp32
    // upload preference isn't silently flipped back to bf16 on upgrade.
    if (parsed.uploadDtype === undefined && parsed.bf16Comm === false) {
      merged.uploadDtype = "fp32";
    }
    return merged;
  } catch {
    return DEFAULT_AD_HOC;
  }
}

interface EditingService {
  name: string;
  enabled: boolean;
  running: boolean;
  args: Record<string, unknown>;
}

interface Props {
  /** Pre-fill ``output_dir`` from a checkpoint-context invocation (e.g.
   *  right-clicking a model on the Projects view). null = blank. */
  outputDir?: string | null;
  /** Edit-mode: when set, the modal seeds itself from this service's
   *  args and the only submit path is "Save" (in-place upsert with
   *  restart-if-running). Pencil-icon clicks in the services sidebar
   *  drop into this mode. */
  editingService?: EditingService | null;
  onClose: () => void;
  /** Called with the new queue_id after a one-shot launch. */
  onSubmitted?: (queueId: string) => void;
  /** Called after a persistent service is written. The sidebar uses
   *  this to expand the Services category so the new entry is visible. */
  onServiceCreated?: (type: "diloco") => void;
}

/** Two-mode modal — operator can either start a one-shot DiLoCo server
 *  (enqueued, runs until killed, no persistent config) or save the
 *  current settings as a persistent service that auto-starts at server
 *  boot. Mirrors the InferenceModal pattern. */
function pickStr(args: Record<string, unknown>, k: string, d: string): string {
  const v = args[k];
  return typeof v === "string" ? v : d;
}
function pickNum(args: Record<string, unknown>, k: string, d: number): number {
  const v = args[k];
  return typeof v === "number" ? v : d;
}
function pickBool(
  args: Record<string, unknown>,
  k: string,
  d: boolean,
): boolean {
  const v = args[k];
  return typeof v === "boolean" ? v : d;
}

export function DiLoCoServerModal({
  outputDir,
  editingService,
  onClose,
  onSubmitted,
  onServiceCreated,
}: Props) {
  const qc = useQueryClient();
  const persisted = loadPersisted();

  // Edit mode seeds from the service's args. Create mode seeds from
  // either the explicit outputDir prop or the persisted ad-hoc state.
  const seed: PersistedAdHoc = editingService
    ? {
        outputDir: pickStr(editingService.args, "output_dir", ""),
        port: pickNum(editingService.args, "port", 8512),
        numWorkers: pickNum(editingService.args, "num_workers", 2),
        host: pickStr(editingService.args, "host", "127.0.0.1"),
        async: pickBool(editingService.args, "async_mode", false),
        dnBufferSize: pickNum(editingService.args, "dn_buffer_size", 0),
        dylu: pickBool(editingService.args, "dylu", false),
        dyluBase: pickNum(editingService.args, "dylu_base_sync_every", 500),
        syncEvery: pickNum(editingService.args, "sync_every", 500),
        numFragments: pickNum(editingService.args, "num_fragments", 1),
        // Prefer the explicit #130 keys; fall back to the deprecated
        // ``bf16_comm`` boolean (upload bf16 vs fp32) for older service args.
        uploadDtype: pickStr(
          editingService.args,
          "upload_dtype",
          pickBool(editingService.args, "bf16_comm", true) ? "bf16" : "fp32",
        ),
        uploadSr: pickBool(editingService.args, "upload_sr", false),
        downloadDtype: pickStr(editingService.args, "download_dtype", "fp32"),
        downloadSr: pickBool(editingService.args, "download_sr", false),
        fromCheckpoint: pickStr(editingService.args, "from_checkpoint", ""),
        runName: pickStr(editingService.args, "run_name", ""),
        saveEvery: pickNum(editingService.args, "save_every", 10),
        saveTotalLimit: pickNum(editingService.args, "save_total_limit", 3),
        outerLr: pickNum(editingService.args, "outer_lr", 0.7),
        outerMomentum: pickNum(editingService.args, "outer_momentum", 0.9),
        noNesterov: pickBool(editingService.args, "no_nesterov", false),
        heartbeatTimeout: pickNum(editingService.args, "heartbeat_timeout", 120),
        minWorkers: pickNum(editingService.args, "min_workers", 1),
        noAuth: pickBool(editingService.args, "no_auth", false),
        regenToken: pickBool(editingService.args, "regen_token", false),
        bulkCleartext: pickBool(editingService.args, "bulk_cleartext", false),
        wireFormat: pickStr(editingService.args, "wire_format", "pickle"),
        grpcEnabled: pickBool(editingService.args, "grpc_enabled", false),
        backend: pickStr(editingService.args, "backend", "http"),
      }
    : {
        ...persisted,
        outputDir: outputDir || persisted.outputDir,
      };

  const [output_dir, setOutputDir] = useState(seed.outputDir);
  const [port, setPort] = useState(seed.port);
  const [numWorkers, setNumWorkers] = useState(seed.numWorkers);
  const [host, setHost] = useState(seed.host);
  const [asyncMode, setAsyncMode] = useState(seed.async);
  const [dnBufferSize, setDnBufferSize] = useState(seed.dnBufferSize);
  const [dylu, setDylu] = useState(seed.dylu);
  const [dyluBase, setDyluBase] = useState(seed.dyluBase);
  const [syncEvery, setSyncEvery] = useState(seed.syncEvery);
  const [numFragments, setNumFragments] = useState(seed.numFragments);
  const [uploadDtype, setUploadDtype] = useState(seed.uploadDtype);
  const [uploadSr, setUploadSr] = useState(seed.uploadSr);
  const [downloadDtype, setDownloadDtype] = useState(seed.downloadDtype);
  const [downloadSr, setDownloadSr] = useState(seed.downloadSr);
  const [fromCheckpoint, setFromCheckpoint] = useState(seed.fromCheckpoint);
  const [runName, setRunName] = useState(seed.runName);
  const [saveEvery, setSaveEvery] = useState(seed.saveEvery);
  const [saveTotalLimit, setSaveTotalLimit] = useState(seed.saveTotalLimit);
  const [outerLr, setOuterLr] = useState(seed.outerLr);
  const [outerMomentum, setOuterMomentum] = useState(seed.outerMomentum);
  const [noNesterov, setNoNesterov] = useState(seed.noNesterov);
  const [heartbeatTimeout, setHeartbeatTimeout] = useState(
    seed.heartbeatTimeout,
  );
  const [minWorkers, setMinWorkers] = useState(seed.minWorkers);
  const [noAuth, setNoAuth] = useState(seed.noAuth);
  const [regenToken, setRegenToken] = useState(seed.regenToken);
  const [bulkCleartext, setBulkCleartext] = useState(seed.bulkCleartext);
  const [wireFormat, setWireFormat] = useState(seed.wireFormat);
  const [grpcEnabled, setGrpcEnabled] = useState(seed.grpcEnabled);
  const [backend, setBackend] = useState(seed.backend);
  const [saving, setSaving] = useState(false);

  // Light validation — the backend re-checks but flagging in-UI is friendlier.
  const trimmedOutputDir = output_dir.trim();
  const trimmedFromCheckpoint = fromCheckpoint.trim();
  const portValid = port > 0 && port < 65536;
  const formValid =
    trimmedOutputDir !== "" && numWorkers >= 1 && portValid && minWorkers >= 1;

  // Maintenance: delete the server's rotated-checkpoint dir
  // (``<model dir>/checkpoints``) so a fresh run starts clean without
  // leaving the modal. The button is gated on the dir actually existing
  // (probed via /fs/path-exists) and always confirms with the full path
  // first. Deletion goes through /fs/delete-dir, which independently
  // re-validates (traversal / depth / denylist guards).
  const checkpointsDir = trimmedOutputDir
    ? trimmedOutputDir.replace(/\/+$/, "") + "/checkpoints"
    : "";
  const checkpointsProbe = useQuery({
    queryKey: ["fs", "path-exists", checkpointsDir],
    queryFn: () => api.fsPathExists(checkpointsDir),
    enabled: !!checkpointsDir,
    staleTime: 2000,
  });
  const checkpointsExist = !!checkpointsProbe.data?.is_dir;
  const deleteCheckpoints = useMutation({
    mutationFn: (p: string) => api.deleteDir(p),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["fs", "path-exists", checkpointsDir] });
      qc.invalidateQueries({ queryKey: ["diloco"] });
    },
  });
  const onDeleteCheckpoints = () => {
    if (!checkpointsExist || deleteCheckpoints.isPending) return;
    if (
      !confirm(
        `Delete this directory and everything in it?\n\n${checkpointsDir}`,
      )
    )
      return;
    deleteCheckpoints.mutate(checkpointsDir);
  };

  /** Build the args dict shared by both submit paths. Matches the
   *  shape the scheduler's _build_diloco_server expects from
   *  ``job_params`` (and what the services config persists). */
  const buildArgs = (): Record<string, unknown> => {
    const args: Record<string, unknown> = {
      output_dir: trimmedOutputDir,
      port,
      num_workers: numWorkers,
      host: host.trim() || "127.0.0.1",
      async_mode: asyncMode,
      dylu,
      save_every: saveEvery,
      save_total_limit: saveTotalLimit,
      outer_lr: outerLr,
      outer_momentum: outerMomentum,
      no_nesterov: noNesterov,
      heartbeat_timeout: heartbeatTimeout,
      min_workers: minWorkers,
      // Group-wide worker settings (adopted from /info; no worker knob).
      sync_every: syncEvery,
      num_fragments: numFragments,
    };
    // Wire precision (issue #130). Emit only on divergence from the CLI
    // default (bf16 upload, fp32 download), keeping the spawned argv readable.
    // Stochastic rounding is meaningful only on an fp32 → bf16 cast, so emit
    // it only when its leg is actually bf16.
    if (uploadDtype && uploadDtype !== "bf16") {
      args.upload_dtype = uploadDtype;
    }
    if (uploadSr && uploadDtype === "bf16") {
      args.upload_sr = true;
    }
    if (downloadDtype && downloadDtype !== "fp32") {
      args.download_dtype = downloadDtype;
    }
    if (downloadSr && downloadDtype === "bf16") {
      args.download_sr = true;
    }
    if (dnBufferSize > 0) args.dn_buffer_size = dnBufferSize;
    if (dylu) args.dylu_base_sync_every = dyluBase;
    if (trimmedFromCheckpoint) args.from_checkpoint = trimmedFromCheckpoint;
    if (runName.trim()) args.run_name = runName.trim();
    // Security (issue #90). The scheduler interprets these:
    //   no_auth=true        → skip token resolution; pass --no-auth
    //   regen_token=true    → rotate the persisted per-port token
    //   bulk_cleartext=true → serve bulk endpoints on a separate
    //                         cleartext listener (server-picked
    //                         ephemeral port), bypassing TLS for speed
    // Token redaction in the spawned server's TTY is NOT an operator
    // choice here — the scheduler passes --quiet-tokens automatically
    // when this webui runs in --demo mode. ``regen_token`` is a no-op
    // under --no-auth; strip it so the argv reflects intent.
    args.no_auth = noAuth;
    if (!noAuth) {
      args.regen_token = regenToken;
    }
    // Sync backend (issue #154). Emit only on divergence from the CLI default
    // (http), keeping the spawned argv readable.
    if (backend && backend !== "http") {
      args.backend = backend;
    }
    // Bulk transport (issue #154) only applies to the http backend: the
    // wire codec, the gRPC fast-path, and the cleartext bulk plane all govern
    // the over-the-wire tensor legs, which shared_memory (shared CPU region)
    // and collective (torch.distributed all-reduce) don't use. Emit them only
    // for http so the argv reflects what's actually in play.
    if (backend === "http") {
      if (wireFormat && wireFormat !== "pickle") {
        args.wire_format = wireFormat;
      }
      if (grpcEnabled) {
        // gRPC is the single bulk fast-path and supersedes the cleartext
        // plane (the server forces bulk_cleartext off under gRPC), so emit
        // only one of the two.
        args.grpc_enabled = true;
      } else if (bulkCleartext) {
        args.bulk_cleartext = true;
      }
    }
    return args;
  };

  /** Persist the current values for the next ad-hoc invocation. */
  const persistCurrent = () => {
    const cur: PersistedAdHoc = {
      outputDir: trimmedOutputDir,
      port,
      numWorkers,
      host,
      async: asyncMode,
      dnBufferSize,
      dylu,
      dyluBase,
      syncEvery,
      numFragments,
      uploadDtype,
      uploadSr,
      downloadDtype,
      downloadSr,
      fromCheckpoint: trimmedFromCheckpoint,
      runName,
      saveEvery,
      saveTotalLimit,
      outerLr,
      outerMomentum,
      noNesterov,
      heartbeatTimeout,
      minWorkers,
      noAuth,
      regenToken,
      bulkCleartext,
      wireFormat,
      grpcEnabled,
      backend,
    };
    persistSet(STORAGE_KEY, JSON.stringify(cur));
  };

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      qc.invalidateQueries({ queryKey: ["diloco"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const onStartOneShot = () => {
    if (!formValid) return;
    persistCurrent();
    const args = buildArgs();
    enqueue.mutate({
      project_dir: trimmedOutputDir,
      config: `diloco:${port}`,
      dynamic_args: {},
      requested_gpus: 0,
      priority: 0,
      job_type: "diloco_server",
      job_params: args,
    });
  };

  const onCreateService = async () => {
    if (!formValid) return;
    persistCurrent();
    const ok = await promptAndCreateService(qc, "diloco", buildArgs());
    if (ok) {
      onServiceCreated?.("diloco");
      onClose();
    }
  };

  const onSaveEdit = async () => {
    if (!editingService || !formValid) return;
    setSaving(true);
    try {
      const ok = await saveServiceArgsAndMaybeRestart(
        qc,
        "diloco",
        editingService.name,
        editingService.running,
        editingService.enabled,
        buildArgs(),
      );
      if (ok) onClose();
    } finally {
      setSaving(false);
    }
  };

  const close = () => {
    if (!editingService) persistCurrent();
    onClose();
  };

  return (
    <ModalBackdrop onClose={close}>
      <div
        className="modal"
        style={{ minWidth: 520, maxHeight: "85vh", overflow: "auto" }}
      >
        <header
          style={{
            display: "flex",
            alignItems: "center",
            padding: 12,
            borderBottom: "1px solid var(--border, #444)",
          }}
        >
          <strong>DiLoCo server</strong>
          <span style={{ flex: 1 }} />
          <button onClick={close}>✕</button>
        </header>

        <div
          style={{
            padding: 12,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
          }}
        >
          <div style={{ gridColumn: "1 / -1" }}>
            <label>
              Model / output dir <span style={{ color: "tomato" }}>*</span>
            </label>
            <PathField
              value={output_dir}
              onChange={setOutputDir}
              placeholder="/path/to/model-output-dir"
            />
            <div className="muted" style={{ fontSize: "smaller" }}>
              The shared init checkpoint dir. Workers must point at this
              same dir.
            </div>
            <div
              style={{
                marginTop: 6,
                display: "flex",
                alignItems: "center",
                gap: 8,
                flexWrap: "wrap",
              }}
            >
              <button
                type="button"
                onClick={onDeleteCheckpoints}
                disabled={!checkpointsExist || deleteCheckpoints.isPending}
                title={
                  !trimmedOutputDir
                    ? "Set the model / output dir first"
                    : !checkpointsExist
                      ? `No checkpoints directory at ${checkpointsDir}`
                      : `Delete ${checkpointsDir}`
                }
              >
                {deleteCheckpoints.isPending
                  ? "Deleting…"
                  : "Delete Checkpoints…"}
              </button>
              {checkpointsDir && (
                <span className="muted" style={{ fontSize: "smaller" }}>
                  {checkpointsExist
                    ? checkpointsDir
                    : "no checkpoints dir to delete"}
                </span>
              )}
            </div>
            {deleteCheckpoints.isError && (
              <div
                role="alert"
                style={{ color: "tomato", fontSize: "smaller", marginTop: 4 }}
              >
                {(deleteCheckpoints.error as Error).message}
              </div>
            )}
          </div>

          <label>
            Port <span style={{ color: "tomato" }}>*</span>
            <input
              type="number"
              min={1}
              max={65535}
              value={port}
              onChange={(e) => setPort(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            num_workers <span style={{ color: "tomato" }}>*</span>
            <input
              type="number"
              min={1}
              value={numWorkers}
              onChange={(e) => setNumWorkers(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <label>
            Bind host
            <input
              type="text"
              value={host}
              onChange={(e) => setHost(e.target.value)}
              placeholder="127.0.0.1"
              style={{ width: "100%" }}
            />
            <div className="muted" style={{ fontSize: "smaller" }}>
              127.0.0.1 = loopback only; 0.0.0.0 = all interfaces (LAN).
            </div>
          </label>

          <label>
            min_workers
            <input
              type="number"
              min={1}
              value={minWorkers}
              onChange={(e) => setMinWorkers(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>

          <div style={{ gridColumn: "1 / -1" }}>
            <label>
              From checkpoint (optional)
              <PathField
                value={fromCheckpoint}
                onChange={setFromCheckpoint}
                placeholder="/path/to/output_dir/checkpoint-25"
              />
              <div className="muted" style={{ fontSize: "smaller" }}>
                Overrides loading from the latest checkpoint in
                output_dir.
              </div>
            </label>
          </div>

          <div style={{ gridColumn: "1 / -1" }}>
            <label>
              Run name (optional)
              <input
                type="text"
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                placeholder="e.g. lr0.7-2workers"
                style={{ width: "100%" }}
              />
              <div className="muted" style={{ fontSize: "smaller" }}>
                Labels this run's stats dir
                (<code>output_dir/runs/&lt;timestamp&gt;_&lt;run-name&gt;</code>,
                holding the JSONL stream + TensorBoard events). Defaults to the
                hostname. A resume from checkpoint continues the prior run's dir.
              </div>
            </label>
          </div>

          <CollapsibleSection title="Synchronization">
            <div className="muted" style={{ fontSize: "smaller", marginBottom: 6 }}>
              Group-wide and server-authoritative: set here and adopted by each
              worker from <code>/info</code> — there are no per-worker flags.
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 8,
              }}
            >
              <label>
                sync_every
                <input
                  type="number"
                  min={1}
                  value={syncEvery}
                  onChange={(e) => setSyncEvery(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
                {dylu && (
                  <div className="muted" style={{ fontSize: "smaller" }}>
                    Superseded by DyLU base sync_every while DyLU is on.
                  </div>
                )}
              </label>
              <label>
                num_fragments (1 = no streaming)
                <input
                  type="number"
                  min={1}
                  value={numFragments}
                  onChange={(e) => setNumFragments(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
              </label>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Async mode">
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label>
                <input
                  type="checkbox"
                  checked={asyncMode}
                  onChange={(e) => setAsyncMode(e.target.checked)}
                />{" "}
                Async mode (workers don't wait for each other)
              </label>
              {asyncMode && (
                <>
                  <label>
                    DN buffer size (0 = disabled)
                    <input
                      type="number"
                      min={0}
                      value={dnBufferSize}
                      onChange={(e) => setDnBufferSize(Number(e.target.value))}
                      style={{ width: "100%" }}
                    />
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={dylu}
                      onChange={(e) => setDylu(e.target.checked)}
                    />{" "}
                    Dynamic Local Updates (DyLU)
                  </label>
                  {dylu && (
                    <label>
                      DyLU base sync_every
                      <input
                        type="number"
                        min={1}
                        value={dyluBase}
                        onChange={(e) => setDyluBase(Number(e.target.value))}
                        style={{ width: "100%" }}
                      />
                    </label>
                  )}
                </>
              )}
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Precision">
            <div className="muted" style={{ fontSize: "smaller", marginBottom: 6 }}>
              Server-authoritative wire precision (issue #130), per leg. Each
              worker adopts these from <code>/info</code>. Stochastic rounding
              only applies on an fp32 → bf16 cast.
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 8,
              }}
            >
              <label>
                upload_dtype{" "}
                <span className="muted" style={{ fontSize: "smaller" }}>
                  (worker → server)
                </span>
                <select
                  value={uploadDtype}
                  onChange={(e) => setUploadDtype(e.target.value)}
                  style={{ width: "100%" }}
                >
                  <option value="bf16">bf16</option>
                  <option value="fp32">fp32</option>
                </select>
              </label>
              <label>
                download_dtype{" "}
                <span className="muted" style={{ fontSize: "smaller" }}>
                  (server → worker)
                </span>
                <select
                  value={downloadDtype}
                  onChange={(e) => setDownloadDtype(e.target.value)}
                  style={{ width: "100%" }}
                >
                  <option value="fp32">fp32</option>
                  <option value="bf16">bf16</option>
                </select>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={uploadSr && uploadDtype === "bf16"}
                  disabled={uploadDtype !== "bf16"}
                  onChange={(e) => setUploadSr(e.target.checked)}
                />{" "}
                upload stochastic rounding
                {uploadDtype !== "bf16" && (
                  <div className="muted" style={{ fontSize: "smaller" }}>
                    Needs upload_dtype = bf16.
                  </div>
                )}
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={downloadSr && downloadDtype === "bf16"}
                  disabled={downloadDtype !== "bf16"}
                  onChange={(e) => setDownloadSr(e.target.checked)}
                />{" "}
                download stochastic rounding
                {downloadDtype !== "bf16" && (
                  <div className="muted" style={{ fontSize: "smaller" }}>
                    Needs download_dtype = bf16.
                  </div>
                )}
              </label>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Outer optimizer">
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 8,
              }}
            >
              <label>
                outer_lr
                <input
                  type="number"
                  step={0.05}
                  value={outerLr}
                  onChange={(e) => setOuterLr(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
              </label>
              <label>
                outer_momentum
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={outerMomentum}
                  onChange={(e) => setOuterMomentum(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
              </label>
              <label style={{ gridColumn: "1 / -1" }}>
                <input
                  type="checkbox"
                  checked={noNesterov}
                  onChange={(e) => setNoNesterov(e.target.checked)}
                />{" "}
                Disable Nesterov momentum
              </label>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Persistence + monitoring">
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 8,
              }}
            >
              <label>
                save_every (rounds; 0 = off)
                <input
                  type="number"
                  min={0}
                  value={saveEvery}
                  onChange={(e) => setSaveEvery(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
              </label>
              <label>
                save_total_limit
                <input
                  type="number"
                  min={0}
                  value={saveTotalLimit}
                  onChange={(e) => setSaveTotalLimit(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
              </label>
              <label>
                heartbeat_timeout (s, 0 = off)
                <input
                  type="number"
                  min={0}
                  value={heartbeatTimeout}
                  onChange={(e) => setHeartbeatTimeout(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
              </label>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Security">
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label>
                <input
                  type="checkbox"
                  checked={noAuth}
                  onChange={(e) => setNoAuth(e.target.checked)}
                />{" "}
                <code>--no-auth</code>{" "}
                <span className="muted" style={{ fontSize: "smaller" }}>
                  disable the bearer-token gate (trusted-LAN only — any
                  reachable host can drive the server)
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={regenToken}
                  disabled={noAuth}
                  onChange={(e) => setRegenToken(e.target.checked)}
                />{" "}
                <code>--regen-token</code>{" "}
                <span className="muted" style={{ fontSize: "smaller" }}>
                  rotate the per-port token; existing workers will 401
                  until they re-pull it
                </span>
              </label>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Transport">
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label>
                <code>--backend</code>{" "}
                <select
                  value={backend}
                  onChange={(e) => setBackend(e.target.value)}
                >
                  <option value="http">http</option>
                  <option value="shared_memory">shared_memory</option>
                  <option value="collective">collective</option>
                </select>{" "}
                <span className="muted" style={{ fontSize: "smaller" }}>
                  Sync backend the worker group must use. Declared here and
                  advertised via /info; workers validate against it and fail
                  loud on disagreement. <strong>Must match</strong> the{" "}
                  <code>--backend</code> the workers are submitted with.
                </span>
              </label>

              {backend === "http" ? (
                <>
                  <hr style={{ width: "100%", opacity: 0.2 }} />
                  <label>
                    <code>--wire-format</code>{" "}
                    <select
                      value={wireFormat}
                      onChange={(e) => setWireFormat(e.target.value)}
                    >
                      <option value="pickle">pickle</option>
                      <option value="safetensors">safetensors</option>
                    </select>{" "}
                    <span className="muted" style={{ fontSize: "smaller" }}>
                      Bulk-tensor wire codec. <code>safetensors</code> drops
                      pickle for an explicit typed, zero-copy frame (no
                      arbitrary-code deserialization); <code>pickle</code> is
                      the back-compatible default.
                    </span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={grpcEnabled}
                      onChange={(e) => setGrpcEnabled(e.target.checked)}
                    />{" "}
                    <code>--grpc</code>{" "}
                    <span className="muted" style={{ fontSize: "smaller" }}>
                      Serve the bulk legs over a streaming gRPC listener
                      (workers negotiate it via /info; HTTP stays the fallback).
                      Supersedes <code>--bulk-cleartext</code>. Cleartext/
                      trusted-LAN today — TLS parity is a follow-up.
                    </span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={bulkCleartext && !grpcEnabled}
                      disabled={grpcEnabled}
                      onChange={(e) => setBulkCleartext(e.target.checked)}
                    />{" "}
                    <code>--bulk-cleartext</code>{" "}
                    <span className="muted" style={{ fontSize: "smaller" }}>
                      Bypass TLS for bulk data: serve /submit_pseudograd,
                      /submit_fragment_pseudograd, and /global_params on a
                      separate cleartext listener on a server-assigned ephemeral
                      port (workers learn it over the encrypted control
                      channel). Trades on-wire confidentiality of the bulk
                      tensors for throughput — trusted LANs only.
                      {grpcEnabled && (
                        <> Superseded by <code>--grpc</code>.</>
                      )}
                    </span>
                  </label>
                </>
              ) : (
                <div className="muted" style={{ fontSize: "smaller" }}>
                  Wire format, gRPC, and the cleartext bulk plane apply only to
                  the <code>http</code> backend — the{" "}
                  <code>{backend}</code> backend moves tensors off the HTTP wire
                  (shared CPU region / collective all-reduce), so there's
                  nothing to configure here.
                </div>
              )}
            </div>
          </CollapsibleSection>
        </div>

        {!!enqueue.error && (
          <div
            className="muted"
            style={{ padding: "0 12px 8px 12px", color: "tomato" }}
          >
            {(enqueue.error as Error).message}
          </div>
        )}

        <footer
          style={{
            padding: 12,
            borderTop: "1px solid var(--border, #444)",
            display: "flex",
            gap: 8,
            justifyContent: "flex-end",
          }}
        >
          <button onClick={close}>Cancel</button>
          {editingService ? (
            <button
              onClick={onSaveEdit}
              disabled={!formValid || saving}
              title={
                editingService.running
                  ? "Save: the running instance is restarted with the new settings"
                  : "Save the new settings to the service config"
              }
            >
              {saving ? "Saving…" : "Save"}
            </button>
          ) : (
            <>
              <button
                onClick={onCreateService}
                disabled={!formValid}
                title="Save as a persistent service that auto-starts on server boot"
              >
                Create service…
              </button>
              <button
                onClick={onStartOneShot}
                disabled={!formValid || enqueue.isPending}
                title="Enqueue a one-shot DiLoCo server (no persistent config)"
              >
                {enqueue.isPending ? "Starting…" : "Start"}
              </button>
            </>
          )}
        </footer>
      </div>
    </ModalBackdrop>
  );
}
