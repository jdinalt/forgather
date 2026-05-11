import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, ConfigInfo, ProjectInfo } from "../api";
import { useDatasetSource } from "../dataset-source";
import { persistGet, persistRemove, persistSet } from "../persist";
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

/** Class-specific (not per-config) form state persisted across dataset
 *  submits. Holding it globally — as the existing global tools (Convert,
 *  Finalize, TensorBoard, …) do — means a user who tunes the inspection
 *  knobs once (e.g. ``examples=10``, ``features="text"``) gets those
 *  defaults the next time they open the dialog from any dataset config.
 *  Per-config dynamic-args still live in the server-side overrides
 *  cache and are unaffected by these defaults. */
interface PersistedDataset {
  tokenizerPath: string;
  histogram: boolean;
  target: string;
  examples: string;
  features: string;
  numShards: string;
  shardIndex: string;
  selectRange: string;
  seed: string;
  exampleStride: string;
  truncate: string;
  histogramSamples: string;
}

const STORAGE_KEY = "forgather-dataset-submit-v1";

const DEFAULTS: PersistedDataset = {
  tokenizerPath: "",
  histogram: false,
  target: "train_dataset_split",
  examples: "",
  features: "",
  numShards: "",
  shardIndex: "",
  selectRange: "",
  seed: "",
  exampleStride: "",
  truncate: "",
  histogramSamples: "1000",
};

function loadPersisted(): Partial<PersistedDataset> {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function savePersisted(s: PersistedDataset): void {
  persistSet(STORAGE_KEY, JSON.stringify(s));
}

/** Canonical dataset targets. The materialized config may expose many
 *  more top-level keys (intermediates, helpers, etc.); only these make
 *  sense to inspect via ``forgather dataset``. We intersect this set
 *  with what the config actually declares so unsupported configs still
 *  get a usable picker. ``validation_*`` is treated as a synonym for
 *  ``eval_*`` — some configs use that naming. */
const CANONICAL_DATASET_TARGETS = [
  "train_dataset",
  "eval_dataset",
  "validation_dataset",
  "test_dataset",
  "train_dataset_split",
  "eval_dataset_split",
  "validation_dataset_split",
  "test_dataset_split",
] as const;

/** Per-field bounds checked at submit time. Mirrors the same closed-
 *  interval semantics the dynamic-args form uses for ``min``/``max``;
 *  blank values skip the check (the flag won't be emitted at all). */
interface NumericField {
  /** Form-state key. */
  key: string;
  /** Human label used in the blocking-reason tooltip. */
  label: string;
  raw: string;
  min?: number;
  max?: number;
  /** Treat the field as int-only — disallow non-integer entries even
   *  when they fall in [min, max]. The dataset CLI rejects floats for
   *  every numeric flag we expose. */
  integer?: boolean;
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

/** Submit modal for ``type.dataset`` configs. Queues a ``forgather
 *  dataset`` run. CPU-only — no GPU reservation. The user picks how to
 *  inspect the dataset (raw examples / histogram / shard) and supplies
 *  any dynamic args the config declares. */
export function DatasetSubmitModal({
  project,
  config,
  onClose,
  onSubmitted,
}: Props) {
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
  // Targets dropdown: same list ``forgather targets`` prints. Falls back
  // to a free-text default if the materialization fails (e.g. config
  // parse error) so the modal stays usable.
  const targetsQ = useQuery({
    queryKey: ["code-targets", project.project_dir, config.name],
    queryFn: () => api.configCodeTargets(project.project_dir, config.name),
    staleTime: 5 * 60 * 1000,
  });

  const [values, setValues] = useState<Record<string, string>>({});
  const [overrideSeeded, setOverrideSeeded] = useState<boolean>(false);

  // Dataset-source selector (shared hook). ``forgather dataset``
  // reads FORGATHER_DATASET_SERVER like every other consumer, so the
  // choice plumbs through cleanly.
  const {
    source: datasetSource,
    setSource: setDatasetSource,
    selector: datasetSourceSelector,
  } = useDatasetSource({
    ready: !!overridesQ.data,
    initial: overridesQ.data?.dataset_source ?? null,
  });

  // Seed each field from the persisted blob (if any) on first mount.
  // Read once via lazy initializers so subsequent re-renders don't keep
  // re-reading localStorage. Reset is a hard reload of DEFAULTS.
  const initial: PersistedDataset = { ...DEFAULTS, ...loadPersisted() };
  const [tokenizerPath, setTokenizerPath] = useState<string>(
    initial.tokenizerPath,
  );
  const [histogram, setHistogram] = useState<boolean>(initial.histogram);
  const [target, setTarget] = useState<string>(initial.target);
  const [examples, setExamples] = useState<string>(initial.examples);
  const [features, setFeatures] = useState<string>(initial.features);
  const [numShards, setNumShards] = useState<string>(initial.numShards);
  const [shardIndex, setShardIndex] = useState<string>(initial.shardIndex);
  const [selectRange, setSelectRange] = useState<string>(initial.selectRange);
  const [seed, setSeed] = useState<string>(initial.seed);
  const [exampleStride, setExampleStride] = useState<string>(
    initial.exampleStride,
  );
  const [truncate, setTruncate] = useState<string>(initial.truncate);
  const [histogramSamples, setHistogramSamples] = useState<string>(
    initial.histogramSamples,
  );

  // ``priority`` is intentionally not persisted — the right value
  // depends on current queue state, not user preference. Same call we
  // make for the Convert / Finalize tools.
  const [priority, setPriority] = useState<number>(0);

  const handleReset = () => {
    if (
      !confirm(
        "Reset dataset dialog defaults? Dynamic-args overrides for this config are unaffected.",
      )
    ) {
      return;
    }
    persistRemove(STORAGE_KEY);
    setTokenizerPath(DEFAULTS.tokenizerPath);
    setHistogram(DEFAULTS.histogram);
    setTarget(DEFAULTS.target);
    setExamples(DEFAULTS.examples);
    setFeatures(DEFAULTS.features);
    setNumShards(DEFAULTS.numShards);
    setShardIndex(DEFAULTS.shardIndex);
    setSelectRange(DEFAULTS.selectRange);
    setSeed(DEFAULTS.seed);
    setExampleStride(DEFAULTS.exampleStride);
    setTruncate(DEFAULTS.truncate);
    setHistogramSamples(DEFAULTS.histogramSamples);
    setPriority(0);
    // Dataset-source dropdown — without this the live in-form value
    // survives the reset and the next submit writes it straight back
    // into overrides.
    setDatasetSource(null);
  };

  // Filter the materialized target list down to the canonical six —
  // and preserve their canonical order so the dropdown is consistent
  // across configs.
  const filteredTargets = useMemo<string[]>(() => {
    const have = new Set(targetsQ.data ?? []);
    return CANONICAL_DATASET_TARGETS.filter((t) => have.has(t));
  }, [targetsQ.data]);

  // If the persisted/default selection isn't in the filtered list, fall
  // back to the first available one. Doesn't override an explicit user
  // pick.
  useEffect(() => {
    if (filteredTargets.length === 0) return;
    if (!filteredTargets.includes(target)) {
      setTarget(filteredTargets[0]);
    }
    // Only react when the available targets change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredTargets]);

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

  // Built-in fields validate independently of the dynamic-args bounds —
  // ``listOutOfBounds`` only walks the schema, not these.
  const builtinViolations = useMemo<string[]>(() => {
    const fields: NumericField[] = [
      {
        key: "examples",
        label: "N Examples",
        raw: examples,
        min: 1,
        integer: true,
      },
      {
        key: "exampleStride",
        label: "Stride",
        raw: exampleStride,
        min: 1,
        integer: true,
      },
      {
        key: "truncate",
        label: "Truncate",
        raw: truncate,
        min: 1,
        integer: true,
      },
      {
        key: "histogramSamples",
        label: "Histogram samples",
        raw: histogramSamples,
        min: 1,
        integer: true,
      },
      {
        key: "numShards",
        label: "num-shards",
        raw: numShards,
        min: 1,
        integer: true,
      },
      {
        key: "shardIndex",
        label: "shard-index",
        raw: shardIndex,
        min: 0,
        integer: true,
      },
      { key: "seed", label: "seed", raw: seed, integer: true },
    ];
    const v = checkNumericFields(fields);
    // Cross-field rule: shard-index < num-shards when both are set.
    if (numShards.trim() && shardIndex.trim()) {
      const ns = Number(numShards);
      const si = Number(shardIndex);
      if (Number.isFinite(ns) && Number.isFinite(si) && si >= ns) {
        v.push("shard-index < num-shards");
      }
    }
    return v;
  }, [
    examples,
    exampleStride,
    truncate,
    histogramSamples,
    numShards,
    shardIndex,
    seed,
  ]);

  const submitBlockedReason: string | undefined =
    missingRequired.length > 0
      ? `Required arg(s) missing: ${missingRequired.map((a) => a.cli_name).join(", ")}`
      : outOfBounds.length > 0
        ? `Out-of-range value(s): ${outOfBounds.map((a) => a.cli_name).join(", ")}`
        : builtinViolations.length > 0
          ? `Invalid value(s): ${builtinViolations.join("; ")}`
          : undefined;

  const submit = () => {
    const schema = argsQ.data ?? [];
    const dyn = coerceArgs(values, schema);

    const params: Record<string, unknown> = {
      histogram,
    };
    if (tokenizerPath) params.tokenizer_path = tokenizerPath;
    if (target) params.target = target;
    const ex = Number(examples);
    if (examples && Number.isFinite(ex)) params.examples = ex;
    if (features.trim()) {
      params.features = features
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    }
    const ns = Number(numShards);
    if (numShards && Number.isFinite(ns)) params.num_shards = ns;
    const si = Number(shardIndex);
    if (shardIndex && Number.isFinite(si)) params.shard_index = si;
    if (selectRange) params.select_range = selectRange;
    const sd = Number(seed);
    if (seed && Number.isFinite(sd)) params.seed = sd;
    const es = Number(exampleStride);
    if (exampleStride && Number.isFinite(es)) params.example_stride = es;
    const tr = Number(truncate);
    if (truncate && Number.isFinite(tr)) params.truncate = tr;
    const hs = Number(histogramSamples);
    if (histogram && Number.isFinite(hs)) params.histogram_samples = hs;

    enqueue.mutate({
      project_dir: project.project_dir,
      config: config.name,
      dynamic_args: dyn,
      requested_gpus: 0,
      priority,
      job_type: "dataset",
      job_params: params,
      dataset_source: datasetSource,
    });
    // Persist the dialog-specific knobs (not priority — see the
    // ``priority`` declaration above) for the next open. Snapshot
    // current state directly rather than rebuilding from ``params``
    // since ``params`` strips empty strings and we want to round-trip
    // those too.
    savePersisted({
      tokenizerPath,
      histogram,
      target,
      examples,
      features,
      numShards,
      shardIndex,
      selectRange,
      seed,
      exampleStride,
      truncate,
      histogramSamples,
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
        aria-label="Submit dataset job"
      >
        <header className="modal-header">
          <h3>Submit dataset job</h3>
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
            <label
              style={{ flex: 1 }}
              title="Path to a tokenizer (HF or Forgather). Required for histogram and tokenized-decode modes; optional otherwise."
            >
              Tokenizer path (optional)
              <PathField
                value={tokenizerPath}
                onChange={setTokenizerPath}
                placeholder="/path/to/tokenizer"
                title="Pick a tokenizer directory"
                wide
              />
            </label>
          </div>

          <div className="submit-row">
            <label title="Which dataset target to materialize. Populated from the config's top-level targets.">
              Target
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                disabled={targetsQ.isLoading}
              >
                {filteredTargets.length > 0 ? (
                  filteredTargets.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))
                ) : (
                  <option value={target}>{target}</option>
                )}
              </select>
            </label>
            <label title="Number of examples to print from the dataset. Leave blank to skip example printing.">
              N Examples
              <input
                type="number"
                min={1}
                step={1}
                value={examples}
                onChange={(e) => setExamples(e.target.value)}
                placeholder="(none)"
              />
            </label>
            <label title="Print every Nth example (after the offset). Default is 1 — print every example.">
              Stride
              <input
                type="number"
                min={1}
                step={1}
                value={exampleStride}
                onChange={(e) => setExampleStride(e.target.value)}
                placeholder="1"
              />
            </label>
            <label title="Truncate each printed example to N characters. Leave blank for no truncation.">
              Truncate
              <input
                type="number"
                min={1}
                step={1}
                value={truncate}
                onChange={(e) => setTruncate(e.target.value)}
                placeholder="(off)"
              />
            </label>
          </div>

          <div className="submit-row">
            <label
              style={{ flex: 1 }}
              title="Comma-separated feature names to print per example. Defaults to the config's main feature."
            >
              Features (comma-separated)
              <input
                type="text"
                value={features}
                onChange={(e) => setFeatures(e.target.value)}
                placeholder="(default = main feature)"
              />
            </label>
          </div>

          <div className="submit-row">
            <label title="Plot a token-length histogram for the dataset (requires --tokenizer-path).">
              <input
                type="checkbox"
                checked={histogram}
                onChange={(e) => setHistogram(e.target.checked)}
              />
              histogram
            </label>
            {histogram && (
              <label title="How many examples to draw when computing the histogram. Default 1000.">
                Samples
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={histogramSamples}
                  onChange={(e) => setHistogramSamples(e.target.value)}
                />
              </label>
            )}
          </div>

          <div className="submit-row">
            <label title="Split the dataset into N equal shards (for distributed processing). Leave blank for no sharding.">
              num-shards
              <input
                type="number"
                min={1}
                step={1}
                value={numShards}
                onChange={(e) => setNumShards(e.target.value)}
                placeholder="(none)"
              />
            </label>
            <label title="Which shard to process, in [0, num-shards).">
              shard-index
              <input
                type="number"
                min={0}
                step={1}
                value={shardIndex}
                onChange={(e) => setShardIndex(e.target.value)}
                placeholder="0"
              />
            </label>
            <label title='Select a sub-range of the dataset before processing — e.g. "100:500", "10%:", ":0.1".'>
              select-range
              <input
                type="text"
                value={selectRange}
                onChange={(e) => setSelectRange(e.target.value)}
                placeholder="100:500"
              />
            </label>
            <label title="Shuffle with this seed before iterating. Leave blank to keep the dataset's natural order.">
              seed
              <input
                type="number"
                step={1}
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="(off)"
              />
            </label>
          </div>

          <div className="submit-row">
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
            <button
              className="secondary"
              onClick={handleReset}
              disabled={enqueue.isPending}
              title="Reset dialog-specific defaults (target, examples, sharding, etc.). Per-config dynamic args are unaffected."
            >
              Reset to defaults
            </button>
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
