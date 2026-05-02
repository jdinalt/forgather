import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { PathField } from "./PathField";

/** Settings persisted across sidebar-Tools "Update…" invocations.
 *  ``priority`` resets each time since the right value depends on
 *  current queue state. ``requestedGpus`` is sticky. */
interface PersistedUpdate {
  srcModelPath: string;
  /** Existing parent directory to write the new model into. */
  dstParent: string;
  /** New directory name for the migrated model under ``dstParent``. */
  modelName: string;
  arch: string;
  fromVersion: string;
  toVersion: string;
  checkpoint: string;
  device: string;
  dtype: string;
  noStrict: boolean;
  safetensors: boolean;
  dryRun: boolean;
  logLevel: string;
  requestedGpus: number;
}

const STORAGE_KEY = "forgather-global-update-v1";

const DEFAULTS: PersistedUpdate = {
  srcModelPath: "",
  dstParent: "",
  modelName: "",
  arch: "",
  fromVersion: "",
  toVersion: "",
  checkpoint: "",
  device: "cpu",
  // "keep" is a UI sentinel meaning "don't pass --dtype"; the script
  // then keeps the saved checkpoint dtype.
  dtype: "keep",
  noStrict: false,
  safetensors: false,
  dryRun: false,
  logLevel: "INFO",
  requestedGpus: 0,
};

function loadPersisted(): Partial<PersistedUpdate> {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return {};
    // Migrate older persisted blobs that still carry a single
    // ``dstModelPath`` (the pre-split destination). Splitting now means
    // the user's last-used destination still seeds the form when they
    // upgrade — without forcing a Reset.
    if (
      typeof parsed.dstModelPath === "string" &&
      parsed.dstParent === undefined &&
      parsed.modelName === undefined
    ) {
      const split = splitDestPath(parsed.dstModelPath);
      parsed.dstParent = split.parent;
      parsed.modelName = split.name;
      delete parsed.dstModelPath;
    }
    return parsed;
  } catch {
    return {};
  }
}

function savePersisted(s: PersistedUpdate) {
  persistSet(STORAGE_KEY, JSON.stringify(s));
}

function splitDestPath(p: string): { parent: string; name: string } {
  const trimmed = p.replace(/\/+$/, "");
  const i = trimmed.lastIndexOf("/");
  if (i < 0) return { parent: "", name: trimmed };
  return { parent: trimmed.slice(0, i), name: trimmed.slice(i + 1) };
}

function joinDestPath(parent: string, name: string): string {
  return `${parent.trim().replace(/\/+$/, "")}/${name.trim()}`;
}

interface Props {
  /** Pre-filled source model directory. When set, the modal switches
   *  to project-backed mode: persisted defaults still seed the rest of
   *  the form, but the source path is locked to the caller's value
   *  (Reset doesn't unpick it). Mirrors ConvertModal / FinalizeModal so
   *  context-menu invocations don't pollute the global tool's
   *  persisted defaults. */
  initialSrcPath?: string;
  /** Pre-filled checkpoint directory. Optional — when set, the
   *  ``--checkpoint`` flag pins the source weights instead of letting
   *  the update script auto-pick the latest. Used by the right-click
   *  menu on a specific checkpoint. */
  initialCheckpoint?: string;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

/** Global "Update…" tool — queues a ``forgather update`` job to migrate
 *  a saved Forgather model to the current source schema. The user picks
 *  src/dst paths plus optional schema overrides (arch / from-version /
 *  to-version) and an optional explicit checkpoint. The script reads
 *  ``forgather_arch`` / ``forgather_arch_version`` from the source's
 *  ``config.json`` when the override fields are blank. */
export function UpdateModal({
  initialSrcPath,
  initialCheckpoint,
  onClose,
  onSubmitted,
}: Props) {
  const qc = useQueryClient();
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });
  const gpusQ = useQuery({ queryKey: ["gpus-once"], queryFn: api.listGpus });

  const persisted = loadPersisted();
  const initial = { ...DEFAULTS, ...persisted };
  const [srcModelPath, setSrcModelPath] = useState(
    initialSrcPath ?? initial.srcModelPath,
  );
  const [dstParent, setDstParent] = useState(initial.dstParent ?? "");
  const [modelName, setModelName] = useState(initial.modelName ?? "");
  const [arch, setArch] = useState(initial.arch);
  const [fromVersion, setFromVersion] = useState(initial.fromVersion);
  const [toVersion, setToVersion] = useState(initial.toVersion);
  const [checkpoint, setCheckpoint] = useState(
    initialCheckpoint ?? initial.checkpoint,
  );
  const [device, setDevice] = useState(initial.device);
  const [dtype, setDtype] = useState(initial.dtype);
  const [noStrict, setNoStrict] = useState(initial.noStrict);
  const [safetensors, setSafetensors] = useState(initial.safetensors);
  const [dryRun, setDryRun] = useState(initial.dryRun);
  const [logLevel, setLogLevel] = useState(initial.logLevel);
  // priority resets each invocation; requestedGpus is sticky.
  const [requestedGpus, setRequestedGpus] = useState<number>(
    initial.requestedGpus ?? 0,
  );
  const [priority, setPriority] = useState<number>(0);

  const maxGpus = Math.max(0, gpusQ.data?.length ?? 0);

  const resetDefaults = () => {
    persistRemove(STORAGE_KEY);
    // When invoked from a context menu, the source path was pinned by
    // the caller — Reset shouldn't unpick it.
    setSrcModelPath(initialSrcPath ?? DEFAULTS.srcModelPath);
    setDstParent(DEFAULTS.dstParent);
    setModelName(DEFAULTS.modelName);
    setArch(DEFAULTS.arch);
    setFromVersion(DEFAULTS.fromVersion);
    setToVersion(DEFAULTS.toVersion);
    // Likewise for a context-pinned checkpoint.
    setCheckpoint(initialCheckpoint ?? DEFAULTS.checkpoint);
    setDevice(DEFAULTS.device);
    setDtype(DEFAULTS.dtype);
    setNoStrict(DEFAULTS.noStrict);
    setSafetensors(DEFAULTS.safetensors);
    setDryRun(DEFAULTS.dryRun);
    setLogLevel(DEFAULTS.logLevel);
    setRequestedGpus(DEFAULTS.requestedGpus);
  };

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const dst = (() => {
    const parent = dstParent.trim();
    const name = modelName.trim();
    if (!parent || !name) return "";
    return joinDestPath(parent, name);
  })();

  const submit = () => {
    const src = srcModelPath.trim();
    if (!src || !dst) return;
    const fv = fromVersion.trim();
    const tv = toVersion.trim();

    savePersisted({
      srcModelPath: src,
      dstParent: dstParent.trim(),
      modelName: modelName.trim(),
      arch: arch.trim(),
      fromVersion: fv,
      toVersion: tv,
      checkpoint: checkpoint.trim(),
      device: device.trim(),
      dtype,
      noStrict,
      safetensors,
      dryRun,
      logLevel,
      requestedGpus,
    });

    const job_params: Record<string, unknown> = {
      src_model_path: src,
      dst_model_path: dst,
      no_strict: noStrict,
      safetensors,
      dry_run: dryRun,
      log_level: logLevel,
    };
    const a = arch.trim();
    if (a) job_params.arch = a;
    // PEP 440 strings; the script parses them via packaging.version.
    if (fv) job_params.from_version = fv;
    if (tv) job_params.to_version = tv;
    const ck = checkpoint.trim();
    if (ck) job_params.checkpoint = ck;
    const dev = device.trim();
    if (dev) job_params.device = dev;
    // "keep" is UI-only — omit --dtype so the script keeps the
    // checkpoint dtype.
    if (dtype && dtype !== "keep") job_params.dtype = dtype;

    enqueue.mutate({
      // project_dir isn't meaningful for update; use the dest path so
      // logs can still link back to where the artifact lives.
      project_dir: dst,
      // Display label on Jobs / Queue rows; mirrors the convert /
      // finalize / tensorboard / inference / mkdocs label scheme.
      config: `update:${basename(dst)}`,
      dynamic_args: {},
      requested_gpus: requestedGpus,
      priority,
      job_type: "update",
      job_params,
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Run forgather update"
      >
        <header className="modal-header">
          <h3>Run forgather update</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-row">
            <label className="wide">
              Source model directory
              <PathField
                value={srcModelPath}
                onChange={setSrcModelPath}
                mode="dirs-only"
                title="Pick the source Forgather model directory"
                wide
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Output parent directory
              <PathField
                value={dstParent}
                onChange={setDstParent}
                mode="dirs-only"
                placeholder="existing directory to create the new model under"
                title="Pick the parent directory"
                wide
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Model name
              <input
                type="text"
                className="wide"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="new directory name (must not already exist under parent)"
              />
            </label>
          </div>
          {dst && (
            <div className="submit-row">
              <span className="muted current-path" title={dst}>
                → <code>{dst}</code>
              </span>
            </div>
          )}

          <div className="submit-row">
            <label className="wide">
              Checkpoint
              <PathField
                value={checkpoint}
                onChange={setCheckpoint}
                mode="dirs-only"
                placeholder="optional — defaults to latest under SRC/checkpoints/"
                title="Pick checkpoint directory"
                wide
              />
            </label>
          </div>

          <h4 className="dyn-heading">Schema</h4>
          <div className="submit-row">
            <label>
              <code>--arch</code>
              <input
                type="text"
                value={arch}
                onChange={(e) => setArch(e.target.value)}
                placeholder="e.g. llama (read from config when blank)"
              />
            </label>
            <label>
              <code>--from-version</code>
              <input
                type="text"
                value={fromVersion}
                onChange={(e) => setFromVersion(e.target.value)}
                placeholder="PEP 440 — from config when blank"
              />
            </label>
            <label>
              <code>--to-version</code>
              <input
                type="text"
                value={toVersion}
                onChange={(e) => setToVersion(e.target.value)}
                placeholder="PEP 440 — converter's current"
              />
            </label>
          </div>

          <h4 className="dyn-heading">Output</h4>
          <div className="submit-row">
            <label>
              dtype
              <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
                <option value="keep">keep checkpoint dtype</option>
                <option value="bfloat16">bfloat16</option>
                <option value="float16">float16</option>
                <option value="float32">float32</option>
              </select>
            </label>
            <label>
              Device
              <input
                type="text"
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                placeholder="cpu | cuda:0"
              />
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={safetensors}
                onChange={(e) => setSafetensors(e.target.checked)}
              />
              <code>--safetensors</code>
              <span className="muted">opt in (default is .bin)</span>
            </label>
          </div>
          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={noStrict}
                onChange={(e) => setNoStrict(e.target.checked)}
              />
              <code>--no-strict</code>
              <span className="muted">
                allow missing / unexpected keys (default: strict)
              </span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
              />
              <code>--dry-run</code>
              <span className="muted">
                resolve and report the plan; don't write
              </span>
            </label>
          </div>

          <h4 className="dyn-heading">Scheduling</h4>
          <div className="submit-row">
            <label>
              GPUs
              <input
                type="number"
                min={0}
                max={maxGpus}
                value={requestedGpus}
                onChange={(e) =>
                  setRequestedGpus(
                    Math.max(
                      0,
                      Math.min(maxGpus, Number(e.target.value) || 0),
                    ),
                  )
                }
              />
              <span className="muted">
                0 unless you set <code>--device cuda…</code>
              </span>
            </label>
            <label>
              Priority
              <input
                type="number"
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
              />
            </label>
            <label>
              Log level
              <select
                value={logLevel}
                onChange={(e) => setLogLevel(e.target.value)}
              >
                <option value="DEBUG">DEBUG</option>
                <option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
                <option value="CRITICAL">CRITICAL</option>
              </select>
            </label>
          </div>

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. Update will
              enqueue but won't start until the scheduler is enabled
              (sidebar play/pause).
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div className="muted current-path">
            {enqueue.error ? String(enqueue.error) : ""}
          </div>
          <div className="btn-row">
            <AutoWatchTtyToggle />
            <button
              className="secondary"
              onClick={resetDefaults}
              title="Clear persisted settings and restore defaults"
            >
              Reset to defaults
            </button>
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={enqueue.isPending || !srcModelPath.trim() || !dst}
            >
              {enqueue.isPending ? "Submitting…" : "Run update"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

function basename(p: string): string {
  const i = p.replace(/\/+$/, "").lastIndexOf("/");
  return i < 0 ? p : p.slice(i + 1);
}
