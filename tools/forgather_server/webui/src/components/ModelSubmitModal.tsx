import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, ConfigInfo, ProjectInfo } from "../api";
import { useDatasetSource } from "../dataset-source";
import { persistGet, persistSet } from "../persist";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import {
  coerceArgs,
  DynamicArgsForm,
  listMissingRequired,
  listOutOfBounds,
} from "./DynamicArgsForm";
import { PathField } from "./PathField";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

type Subcommand = "construct" | "test";

/** Devices the modal lets the user pick. The ``forgather model`` CLI
 *  accepts arbitrary ``cuda:N`` strings, but the present launcher path
 *  reserves at most one GPU and pins it via ``CUDA_VISIBLE_DEVICES`` —
 *  so anything past plain ``cuda`` doesn't add capability. We keep the
 *  picker honest until the model command itself learns multi-GPU. */
const DEVICE_OPTIONS = ["meta", "cpu", "cuda"] as const;

/** Float dtypes the user is realistically going to pick. The CLI
 *  accepts the full ``construct.torch_dtype_map`` (int/quint/complex/etc.)
 *  but those don't make sense for model construction. Empty string =
 *  don't pass --dtype, leaving the framework default. */
const DTYPE_OPTIONS = [
  { value: "", label: "(default)" },
  { value: "float32", label: "float32" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float16", label: "float16" },
  { value: "float64", label: "float64" },
];

/** Bounds checked on the dialog's own numeric inputs at submit time.
 *  Mirrors ``DatasetSubmitModal``'s helper. Dynamic-args bounds are
 *  enforced separately by ``listOutOfBounds`` against the schema. */
interface NumericField {
  key: string;
  label: string;
  raw: string;
  min?: number;
  max?: number;
  integer?: boolean;
}

/** Per-(project, config) form-state cache. Mirrors DiLoCoServerModal's
 *  approach: localStorage holds the operator's last-used settings so
 *  reopening the modal doesn't dump them. Dynamic-args values are
 *  cached separately on the server side via ``setOverrides``; these
 *  are the model-modal-specific knobs that live entirely in the
 *  frontend (device, dtype, subcommand toggles, test settings,
 *  priority). */
interface PersistedModelSubmit {
  subcommand: Subcommand;
  device: string;
  dtype: string;
  noInitWeights: boolean;
  saveCheckpoint: boolean;
  safetensors: boolean;
  gradientCheckpointing: boolean;
  fuseOptimWithBackward: boolean;
  loadFromCheckpoint: string;
  batchSize: string;
  sequenceLength: string;
  steps: string;
  lr: string;
  packed: boolean;
  amp: string;
  priority: number;
}

const DEFAULT_PERSISTED: PersistedModelSubmit = {
  subcommand: "construct",
  device: "meta",
  dtype: "",
  noInitWeights: false,
  saveCheckpoint: false,
  safetensors: false,
  gradientCheckpointing: false,
  fuseOptimWithBackward: false,
  loadFromCheckpoint: "",
  batchSize: "2",
  sequenceLength: "512",
  steps: "1",
  lr: "0.01",
  packed: false,
  amp: "",
  priority: 0,
};

function loadPersisted(key: string): PersistedModelSubmit {
  const raw = persistGet(key);
  if (!raw) return DEFAULT_PERSISTED;
  try {
    const parsed = JSON.parse(raw) as Partial<PersistedModelSubmit>;
    return { ...DEFAULT_PERSISTED, ...parsed };
  } catch {
    return DEFAULT_PERSISTED;
  }
}


function checkNumericFields(fields: NumericField[]): string[] {
  const violations: string[] = [];
  for (const f of fields) {
    const raw = f.raw.trim();
    if (raw === "") continue;
    const v = Number(raw);
    if (!Number.isFinite(v)) {
      violations.push(`${f.label} must be a number`);
      continue;
    }
    if (f.integer && !Number.isInteger(v)) {
      violations.push(`${f.label} must be an integer`);
      continue;
    }
    if (f.min !== undefined && v < f.min) {
      violations.push(`${f.label} >= ${f.min}`);
    }
    if (f.max !== undefined && v > f.max) {
      violations.push(`${f.label} <= ${f.max}`);
    }
  }
  return violations;
}

/** Submit modal for ``type.model`` configs. Queues ``forgather model
 *  construct`` or ``forgather model test`` against the selected config.
 *  GPU reservation is implicit: ``device=cuda`` reserves one GPU,
 *  anything else reserves zero. The model command runs as a fire-and-
 *  forget subprocess — no trainer-control protocol — so there's no
 *  progress / save / stop affordance once it lands in the Jobs panel. */
export function ModelSubmitModal({ project, config, onClose, onSubmitted }: Props) {
  const qc = useQueryClient();
  const argsQ = useQuery({
    queryKey: ["dynamic-args", project.project_dir, config.name],
    queryFn: () => api.dynamicArgs(project.project_dir, config.name),
  });
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });
  const overridesQ = useQuery({
    queryKey: ["overrides", project.project_dir, config.name],
    queryFn: () => api.getOverrides(project.project_dir, config.name),
  });

  const [values, setValues] = useState<Record<string, string>>({});
  const [overrideSeeded, setOverrideSeeded] = useState<boolean>(false);

  // Dataset-source selector — shared hook handles state + offline-
  // fallback seeding. ``forgather model construct/test`` calls
  // ``fast_load_iterable_dataset`` when --dataset-project is set, so
  // the choice plumbs through the same FORGATHER_DATASET_SERVER env
  // route training uses.
  const {
    source: datasetSource,
    selector: datasetSourceSelector,
  } = useDatasetSource({
    ready: !!overridesQ.data,
    initial: overridesQ.data?.dataset_source ?? null,
  });

  // Per-(project, config) form state, seeded from localStorage so
  // reopening the modal lands on whatever the operator submitted
  // last. Persistence happens in a single useEffect at the bottom of
  // this block; submit also touches the cache before enqueueing to
  // guarantee at-least-once write even if the modal closes fast.
  const storageKey = useMemo(
    () => `forgather-submit-model/${project.project_dir}/${config.name}`,
    [project.project_dir, config.name],
  );
  const persisted = useMemo<PersistedModelSubmit>(
    () => loadPersisted(storageKey),
    [storageKey],
  );

  const [subcommand, setSubcommand] = useState<Subcommand>(persisted.subcommand);
  const [device, setDevice] = useState<string>(persisted.device);
  const [dtype, setDtype] = useState<string>(persisted.dtype);
  const [noInitWeights, setNoInitWeights] = useState<boolean>(persisted.noInitWeights);
  const [saveCheckpoint, setSaveCheckpoint] = useState<boolean>(persisted.saveCheckpoint);
  const [safetensors, setSafetensors] = useState<boolean>(persisted.safetensors);
  const [gradientCheckpointing, setGradientCheckpointing] = useState<boolean>(
    persisted.gradientCheckpointing,
  );
  const [fuseOptimWithBackward, setFuseOptimWithBackward] = useState<boolean>(
    persisted.fuseOptimWithBackward,
  );
  const [loadFromCheckpoint, setLoadFromCheckpoint] = useState<string>(
    persisted.loadFromCheckpoint,
  );

  const [batchSize, setBatchSize] = useState<string>(persisted.batchSize);
  const [sequenceLength, setSequenceLength] = useState<string>(persisted.sequenceLength);
  const [steps, setSteps] = useState<string>(persisted.steps);
  const [lr, setLr] = useState<string>(persisted.lr);
  const [packed, setPacked] = useState<boolean>(persisted.packed);
  const [amp, setAmp] = useState<string>(persisted.amp);

  const [priority, setPriority] = useState<number>(persisted.priority);

  // Mirror operator edits back into localStorage. Same shape as the
  // initial seed so loadPersisted can read it on the next mount.
  useEffect(() => {
    persistSet(
      storageKey,
      JSON.stringify({
        subcommand,
        device,
        dtype,
        noInitWeights,
        saveCheckpoint,
        safetensors,
        gradientCheckpointing,
        fuseOptimWithBackward,
        loadFromCheckpoint,
        batchSize,
        sequenceLength,
        steps,
        lr,
        packed,
        amp,
        priority,
      } satisfies PersistedModelSubmit),
    );
  }, [
    storageKey,
    subcommand,
    device,
    dtype,
    noInitWeights,
    saveCheckpoint,
    safetensors,
    gradientCheckpointing,
    fuseOptimWithBackward,
    loadFromCheckpoint,
    batchSize,
    sequenceLength,
    steps,
    lr,
    packed,
    amp,
    priority,
  ]);

  // Meta device can't hold real weights, so save_checkpoint and
  // load_from_checkpoint don't apply. Effective values mask out the
  // operator's stored intent when device == "meta" — UI shows them
  // unchecked + disabled and submit ignores them. The raw state
  // stays intact so a flip back to cpu/cuda restores the prior
  // intent.
  const metaDevice = device === "meta";
  const effectiveSaveCheckpoint = saveCheckpoint && !metaDevice;
  const effectiveLoadFromCheckpoint = metaDevice ? "" : loadFromCheckpoint;

  // GPU reservation is implicit: cuda → 1, otherwise 0. The launcher
  // restricts the spawned process to that single GPU via
  // CUDA_VISIBLE_DEVICES, so the model command's own --device cuda
  // resolves to it.
  const requestedGpus = device === "cuda" ? 1 : 0;

  // ``test`` runs a forward + backward pass, which the meta device
  // can't do (no real storage). Filter it out of the dropdown for
  // ``test``, and snap the current selection off ``meta`` when the
  // user switches subcommand to test.
  const deviceOptions = useMemo<readonly string[]>(
    () => (subcommand === "test" ? ["cpu", "cuda"] : DEVICE_OPTIONS),
    [subcommand],
  );
  useEffect(() => {
    if (subcommand === "test" && device === "meta") {
      setDevice("cpu");
    }
  }, [subcommand, device]);

  useEffect(() => {
    if (overrideSeeded) return;
    if (!argsQ.data || !overridesQ.data) return;
    const cached = overridesQ.data.values;
    if (Object.keys(cached).length === 0) {
      setOverrideSeeded(true);
      return;
    }
    const schemaDests = new Set(argsQ.data.map((a) => a.dest));
    const seed: Record<string, string> = {};
    for (const [k, v] of Object.entries(cached)) {
      if (schemaDests.has(k) && v != null) {
        seed[k] = String(v);
      }
    }
    if (Object.keys(seed).length > 0) {
      setValues((prev) => ({ ...seed, ...prev }));
    }
    setOverrideSeeded(true);
  }, [argsQ.data, overridesQ.data, overrideSeeded]);

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const missingRequired = useMemo(
    () => (argsQ.data ? listMissingRequired(argsQ.data, values) : []),
    [argsQ.data, values],
  );
  const outOfBounds = useMemo(
    () => (argsQ.data ? listOutOfBounds(argsQ.data, values) : []),
    [argsQ.data, values],
  );

  const builtinViolations = useMemo<string[]>(() => {
    if (subcommand !== "test") return [];
    return checkNumericFields([
      { key: "batchSize", label: "Batch size", raw: batchSize, min: 1, integer: true },
      {
        key: "sequenceLength",
        label: "Sequence length",
        raw: sequenceLength,
        min: 1,
        integer: true,
      },
      { key: "steps", label: "Steps", raw: steps, min: 1, integer: true },
      { key: "lr", label: "LR", raw: lr, min: 0 },
    ]);
  }, [subcommand, batchSize, sequenceLength, steps, lr]);

  // no-init-weights leaves parameters uninitialized; without a
  // Load-from-checkpoint path those random-bit weights become the
  // model's actual weights. Almost always an operator error, so
  // block submit instead of letting it through. Device==meta is
  // exempted: meta tensors have no storage to populate either way,
  // and the noInitWeights checkbox is meaningless in that mode.
  const noInitWithoutLoad =
    noInitWeights && !metaDevice && !loadFromCheckpoint.trim();

  const submitBlockedReason: string | undefined =
    missingRequired.length > 0
      ? `Required arg(s) missing: ${missingRequired.map((a) => a.cli_name).join(", ")}`
      : outOfBounds.length > 0
        ? `Out-of-range value(s): ${outOfBounds.map((a) => a.cli_name).join(", ")}`
        : builtinViolations.length > 0
          ? `Invalid value(s): ${builtinViolations.join("; ")}`
          : noInitWithoutLoad
            ? "no-init-weights requires a Load-from-checkpoint path (otherwise the model would run with random uninitialized weights)"
            : undefined;

  const submit = () => {
    const schema = argsQ.data ?? [];
    const dyn = coerceArgs(values, schema);
    const params: Record<string, unknown> = {
      subcommand,
      device,
      no_init_weights: noInitWeights,
      // Mask the meta-incompatible knobs out at submit time so a
      // stale operator preference can't sneak through if the meta-
      // device disable logic is ever bypassed.
      save_checkpoint: effectiveSaveCheckpoint,
      safetensors,
      gradient_checkpointing: gradientCheckpointing,
      fuse_optim_with_backward: fuseOptimWithBackward,
    };
    if (dtype) params.dtype = dtype;
    if (effectiveLoadFromCheckpoint)
      params.load_from_checkpoint = effectiveLoadFromCheckpoint;
    if (subcommand === "test") {
      const bs = Number(batchSize);
      const sl = Number(sequenceLength);
      const st = Number(steps);
      const lrn = Number(lr);
      if (Number.isFinite(bs)) params.batch_size = bs;
      if (Number.isFinite(sl)) params.sequence_length = sl;
      if (Number.isFinite(st)) params.steps = st;
      if (Number.isFinite(lrn)) params.lr = lrn;
      params.packed = packed;
      if (amp) params.amp = amp;
    }
    enqueue.mutate({
      project_dir: project.project_dir,
      config: config.name,
      dynamic_args: dyn,
      requested_gpus: requestedGpus,
      priority,
      job_type: "model",
      job_params: params,
      dataset_source: datasetSource,
    });
    api
      .setOverrides(
        project.project_dir,
        config.name,
        dyn,
        null,
        null,
        datasetSource,
      )
      .then(() => {
        qc.invalidateQueries({
          queryKey: ["overrides", project.project_dir, config.name],
        });
      })
      .catch(() => {});
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Submit model job"
      >
        <header className="modal-header">
          <h3>Submit model job</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-summary">
            <div>
              <span className="muted">config</span>
              <code>{config.name}</code>
            </div>
            <div>
              <span className="muted">project</span>
              <code>{project.project_dir}</code>
            </div>
          </div>

          {datasetSourceSelector}

          <div className="submit-row">
            <label title="construct: build the model and print its parameter / architecture summary. test: run a few train steps against random or a real dataset to verify forward + backward.">
              Subcommand
              <select
                value={subcommand}
                onChange={(e) => setSubcommand(e.target.value as Subcommand)}
              >
                <option value="construct">construct</option>
                <option value="test">test</option>
              </select>
            </label>
            <label title='Where to construct the model. "meta" allocates parameter shapes without weights (cheap sanity check); "cpu" materializes on host RAM; "cuda" reserves one GPU and constructs on it.'>
              Device
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
              >
                {deviceOptions.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </label>
            <label title="Default torch dtype during construction. Leave blank to use the framework default.">
              Dtype
              <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
                {DTYPE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label title="Higher priority dispatches sooner once a slot is free.">
              Priority
              <input
                type="number"
                step={1}
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
              />
              <span className="muted">higher runs sooner</span>
            </label>
          </div>

          <div className="submit-row">
            <label title="Skip parameter initialization (saves time when weights will be loaded from a checkpoint anyway). Requires a Load-from-checkpoint path on non-meta devices — otherwise the model runs with uninitialized random weights.">
              <input
                type="checkbox"
                checked={noInitWeights}
                onChange={(e) => setNoInitWeights(e.target.checked)}
              />
              no-init-weights
            </label>
            <label
              title={
                metaDevice
                  ? "Disabled on the meta device: meta tensors have no real storage so there's nothing to checkpoint."
                  : "After construction, save the model weights as a checkpoint into the model's output_dir."
              }
              style={metaDevice ? { opacity: 0.5 } : undefined}
            >
              <input
                type="checkbox"
                checked={effectiveSaveCheckpoint}
                onChange={(e) => setSaveCheckpoint(e.target.checked)}
                disabled={metaDevice}
              />
              save-checkpoint
            </label>
            <label
              title={
                effectiveSaveCheckpoint
                  ? "Use safetensors format for the saved checkpoint."
                  : "Only meaningful when save-checkpoint is enabled."
              }
              style={!effectiveSaveCheckpoint ? { opacity: 0.5 } : undefined}
            >
              <input
                type="checkbox"
                checked={safetensors}
                onChange={(e) => setSafetensors(e.target.checked)}
                disabled={!effectiveSaveCheckpoint}
              />
              safetensors
            </label>
          </div>

          <div className="submit-row">
            <label title="Enable gradient checkpointing on the model (model.gradient_checkpointing_enable). Useful smoke test that the feature is wired up and doesn't crash on this architecture.">
              <input
                type="checkbox"
                checked={gradientCheckpointing}
                onChange={(e) => setGradientCheckpointing(e.target.checked)}
              />
              gradient-checkpointing
            </label>
            <label title="Fuse optimizer.step + zero_grad into a per-parameter post-grad hook (saves memory by avoiding a full grad accumulation). Only meaningful in test; smoke-tests that the model's grad path supports it.">
              <input
                type="checkbox"
                checked={fuseOptimWithBackward}
                onChange={(e) => setFuseOptimWithBackward(e.target.checked)}
              />
              fuse-optim-with-backward
            </label>
          </div>

          <div className="submit-row">
            <label
              style={{ flex: 1, opacity: metaDevice ? 0.5 : undefined }}
              title={
                metaDevice
                  ? "Disabled on the meta device: meta tensors have no real storage to load weights into."
                  : "Load weights from this checkpoint after construction. Required when no-init-weights is set."
              }
            >
              Load from checkpoint{noInitWeights && !metaDevice ? " (required)" : " (optional)"}
              <PathField
                value={effectiveLoadFromCheckpoint}
                onChange={setLoadFromCheckpoint}
                placeholder="/path/to/checkpoint"
                title={
                  metaDevice
                    ? "Disabled on the meta device"
                    : "Pick a checkpoint directory or file"
                }
                wide
                disabled={metaDevice}
              />
            </label>
          </div>

          {subcommand === "test" && (
            <>
              <h4 className="dyn-heading">Test settings</h4>
              <div className="submit-row">
                <label title="Number of examples per training step.">
                  Batch size
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={batchSize}
                    onChange={(e) => setBatchSize(e.target.value)}
                  />
                </label>
                <label title="Token length per example.">
                  Sequence length
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={sequenceLength}
                    onChange={(e) => setSequenceLength(e.target.value)}
                  />
                </label>
                <label title="How many train steps to run.">
                  Steps
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={steps}
                    onChange={(e) => setSteps(e.target.value)}
                  />
                </label>
                <label title="SGD learning rate for the smoke-test optimizer.">
                  LR
                  <input
                    type="text"
                    value={lr}
                    onChange={(e) => setLr(e.target.value)}
                  />
                </label>
              </div>
              <div className="submit-row">
                <label title="Automatic Mixed Precision dtype for the autocast context. Disabled = no autocast.">
                  AMP
                  <select value={amp} onChange={(e) => setAmp(e.target.value)}>
                    <option value="">disabled</option>
                    <option value="bf16">bf16</option>
                    <option value="fp16">fp16</option>
                  </select>
                </label>
                <label title="Pass packed_sequences=True to the data collator (concatenates short examples up to sequence-length).">
                  <input
                    type="checkbox"
                    checked={packed}
                    onChange={(e) => setPacked(e.target.checked)}
                  />
                  packed sequences
                </label>
              </div>
            </>
          )}

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. The job will
              enqueue but won't start until the scheduler is enabled on the
              Queue tab.
            </div>
          )}

          <h4 className="dyn-heading">
            Dynamic arguments
            {argsQ.data && argsQ.data.length === 0 && (
              <span className="muted"> (this config declares none)</span>
            )}
          </h4>
          {argsQ.isLoading && <div className="muted pad">Loading…</div>}
          {argsQ.error && (
            <div className="err pad">
              <pre>{String(argsQ.error)}</pre>
            </div>
          )}
          {argsQ.data && argsQ.data.length > 0 && (
            <DynamicArgsForm
              schema={argsQ.data}
              values={values}
              onChange={(dest, v) =>
                setValues((prev) => ({ ...prev, [dest]: v }))
              }
              enforceRequired
            />
          )}
        </div>

        <footer className="modal-footer">
          <div className="muted current-path">
            {enqueue.error ? String(enqueue.error) : ""}
          </div>
          <div className="btn-row">
            <AutoWatchTtyToggle />
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={
                enqueue.isPending ||
                argsQ.isLoading ||
                missingRequired.length > 0 ||
                outOfBounds.length > 0 ||
                builtinViolations.length > 0
              }
              title={submitBlockedReason}
            >
              {enqueue.isPending ? "Submitting…" : "Submit"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
