import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api";
import { persistGet, persistSet } from "../persist";
import {
  promptAndCreateService,
  saveServiceArgsAndMaybeRestart,
} from "../services-create";
import { ModalBackdrop } from "./ModalBackdrop";
import { PathField } from "./PathField";

const STORAGE_KEY = "forgather-diloco-server-modal";

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
  bf16Comm: boolean;
  fromCheckpoint: string;
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
  bf16Comm: true,
  fromCheckpoint: "",
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
};

function loadPersisted(): PersistedAdHoc {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return DEFAULT_AD_HOC;
  try {
    const parsed = JSON.parse(raw) as Partial<PersistedAdHoc>;
    return { ...DEFAULT_AD_HOC, ...parsed };
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
        bf16Comm: pickBool(editingService.args, "bf16_comm", true),
        fromCheckpoint: pickStr(editingService.args, "from_checkpoint", ""),
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
  const [bf16Comm, setBf16Comm] = useState(seed.bf16Comm);
  const [fromCheckpoint, setFromCheckpoint] = useState(seed.fromCheckpoint);
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
      bf16_comm: bf16Comm,
    };
    if (dnBufferSize > 0) args.dn_buffer_size = dnBufferSize;
    if (dylu) args.dylu_base_sync_every = dyluBase;
    if (trimmedFromCheckpoint) args.from_checkpoint = trimmedFromCheckpoint;
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
    if (bulkCleartext) {
      args.bulk_cleartext = true;
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
      bf16Comm,
      fromCheckpoint: trimmedFromCheckpoint,
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

          <fieldset style={{ gridColumn: "1 / -1", padding: 8 }}>
            <legend>Worker settings (group-wide)</legend>
            <div className="muted" style={{ fontSize: "smaller", marginBottom: 6 }}>
              These must match across every worker, so they're set here and
              adopted by each worker from <code>/info</code> — there are no
              per-worker flags.
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
              <label style={{ gridColumn: "1 / -1" }}>
                <input
                  type="checkbox"
                  checked={bf16Comm}
                  onChange={(e) => setBf16Comm(e.target.checked)}
                />{" "}
                bf16 pseudo-gradient communication
              </label>
            </div>
          </fieldset>

          <fieldset style={{ gridColumn: "1 / -1", padding: 8 }}>
            <legend>Sync mode</legend>
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
          </fieldset>

          <fieldset style={{ gridColumn: "1 / -1", padding: 8 }}>
            <legend>Outer optimizer</legend>
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
          </fieldset>

          <fieldset style={{ gridColumn: "1 / -1", padding: 8 }}>
            <legend>Persistence + monitoring</legend>
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
          </fieldset>

          <fieldset style={{ gridColumn: "1 / -1", padding: 8 }}>
            <legend>Security (auth + bulk plane)</legend>
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

              <hr style={{ width: "100%", opacity: 0.2 }} />

              <label>
                <input
                  type="checkbox"
                  checked={bulkCleartext}
                  onChange={(e) => setBulkCleartext(e.target.checked)}
                />{" "}
                <code>--bulk-cleartext</code>{" "}
                <span className="muted" style={{ fontSize: "smaller" }}>
                  Bypass TLS for bulk data: serve /submit_pseudograd,
                  /submit_fragment_pseudograd, and /global_params on a
                  separate cleartext listener on a server-assigned
                  ephemeral port (workers learn it over the encrypted
                  control channel). Trades on-wire confidentiality of the
                  bulk tensors for throughput — trusted LANs only.
                </span>
              </label>
            </div>
          </fieldset>
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
