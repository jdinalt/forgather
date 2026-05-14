import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { PathField } from "./PathField";
import { ModalBackdrop } from "./ModalBackdrop";

/** Settings persisted across sidebar-Tools "Finalize…" invocations.
 *  ``priority`` resets each time since the right value depends on
 *  current queue state. ``requestedGpus`` is sticky. */
interface PersistedFinalize {
  source: string;
  /** Existing parent directory to write the finalized model into. */
  destParent: string;
  /** New directory name for the finalized model under ``destParent``. */
  modelName: string;
  checkpoint: string;
  addTokens: string;
  skipDefaultTokens: boolean;
  chatTemplatePath: string;
  noAutoStopTokens: boolean;
  stopTokens: string;
  generationConfig: string;
  keepOptimizer: boolean;
  rootCopy: boolean;
  safetensors: boolean;
  dtype: string;
  device: string;
  dryRun: boolean;
  logLevel: string;
  requestedGpus: number;
  /** Torchao QAT convert recipe. Empty string means "skip QAT convert". */
  qatConvert: string;
}

// Keep in sync with QAT_RECIPES in src/forgather/ml/qat_recipes.py.
const QAT_CONVERT_RECIPES = [
  "int8-dynamic-act-int4-weight",
  "int4-weight-only",
  "float8-dynamic-act-float8-weight",
] as const;

const STORAGE_KEY = "forgather-global-finalize-v1";

const DEFAULTS: PersistedFinalize = {
  source: "",
  destParent: "",
  modelName: "",
  checkpoint: "",
  addTokens: "",
  skipDefaultTokens: false,
  chatTemplatePath: "",
  noAutoStopTokens: false,
  stopTokens: "",
  generationConfig: "carry",
  keepOptimizer: false,
  rootCopy: false,
  safetensors: false,
  // The script keeps the checkpoint dtype unless overridden; "keep" is
  // a UI-only sentinel meaning "don't pass --dtype".
  dtype: "keep",
  device: "cpu",
  dryRun: false,
  logLevel: "INFO",
  requestedGpus: 0,
  qatConvert: "",
};

function loadPersisted(): Partial<PersistedFinalize> {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return {};
    // Migrate older persisted blobs that still carry a single ``dest``
    // (the pre-split destination). Splitting now means the user's
    // last-used destination still seeds the form when they upgrade —
    // without forcing a Reset.
    if (
      typeof parsed.dest === "string" &&
      parsed.destParent === undefined &&
      parsed.modelName === undefined
    ) {
      const split = splitDestPath(parsed.dest);
      parsed.destParent = split.parent;
      parsed.modelName = split.name;
      delete parsed.dest;
    }
    return parsed;
  } catch {
    return {};
  }
}

function savePersisted(s: PersistedFinalize) {
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
   *  to project-backed mode: persisted defaults are ignored, the
   *  user's submitted values aren't saved back, and the Reset button
   *  is hidden. Mirrors InferenceModal / ConvertModal so context-menu
   *  invocations don't pollute the global tool's persisted defaults. */
  initialSource?: string;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

/** Global "Finalize…" tool — queues a ``forgather finalize`` job to
 *  package a trained model into a clean output directory. */
export function FinalizeModal({ initialSource, onClose, onSubmitted }: Props) {
  const qc = useQueryClient();
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });
  const gpusQ = useQuery({ queryKey: ["gpus-once"], queryFn: api.listGpus });
  // Used to derive default tokenizer paths the first time the modal is
  // opened — the bundled ChatML configs live under the Forgather repo.
  const quickQ = useQuery({
    queryKey: ["fs-quick-paths"],
    queryFn: api.fsQuickPaths,
    staleTime: 5 * 60 * 1000,
  });
  const repoRoot = useMemo(
    () =>
      (
        quickQ.data?.find((q) => q.label === "Forgather repo")?.path ?? ""
      ).replace(/\/+$/, ""),
    [quickQ.data],
  );
  const defaultAddTokens = repoRoot
    ? `${repoRoot}/add_tokens_config/chatml.yaml`
    : "";
  const defaultChatTemplate = repoRoot
    ? `${repoRoot}/chat_templates/chatml.jinja`
    : "";

  // Pinned-source mode (context-menu invocation) just overrides the
  // initial source path; everything else falls back to the global
  // persisted values, and submit still writes the (now context-
  // sourced) values back so the next opening reflects the last run.
  const persisted = loadPersisted();
  const initial = { ...DEFAULTS, ...persisted };
  const [source, setSource] = useState(initialSource ?? initial.source);
  const [destParent, setDestParent] = useState(initial.destParent ?? "");
  const [modelName, setModelName] = useState(initial.modelName ?? "");
  const [checkpoint, setCheckpoint] = useState(initial.checkpoint);
  const [addTokens, setAddTokens] = useState(initial.addTokens);
  const [skipDefaultTokens, setSkipDefaultTokens] = useState(
    initial.skipDefaultTokens,
  );
  const [chatTemplatePath, setChatTemplatePath] = useState(
    initial.chatTemplatePath,
  );
  const [noAutoStopTokens, setNoAutoStopTokens] = useState(
    initial.noAutoStopTokens,
  );
  const [stopTokens, setStopTokens] = useState(initial.stopTokens);
  const [generationConfig, setGenerationConfig] = useState(
    initial.generationConfig,
  );
  // Pull the merged bundled + user preset list. The finalize resolver
  // (forgather/ml/model_conversion/finalize.py:_resolve_preset_path)
  // checks both ``<repo>/generation_config/`` and
  // ``~/.config/forgather/generation_config/``, so any name in this list is
  // valid for finalize even though the CLI's --help text only mentions
  // the user directory.
  const presetsQ = useQuery({
    queryKey: ["generation-configs"],
    queryFn: api.listGenerationConfigs,
    staleTime: 60_000,
  });
  const presetNames = useMemo(
    () => (presetsQ.data?.presets ?? []).map((p) => p.name),
    [presetsQ.data],
  );
  // The persisted/in-form value can be one of:
  //   - "carry" / "none" — handled directly by the select
  //   - a known preset name — selected as that preset
  //   - anything else — treated as a custom JSON path (the path field
  //     becomes visible)
  const isKnownChoice =
    generationConfig === "carry" ||
    generationConfig === "none" ||
    presetNames.includes(generationConfig);
  const selectValue = isKnownChoice ? generationConfig : "__custom__";
  const customPath = isKnownChoice ? "" : generationConfig;
  const [keepOptimizer, setKeepOptimizer] = useState(initial.keepOptimizer);
  const [rootCopy, setRootCopy] = useState(initial.rootCopy);
  const [safetensors, setSafetensors] = useState(initial.safetensors);
  const [dtype, setDtype] = useState(initial.dtype);
  const [device, setDevice] = useState(initial.device);
  const [dryRun, setDryRun] = useState(initial.dryRun);
  const [logLevel, setLogLevel] = useState(initial.logLevel);
  const [requestedGpus, setRequestedGpus] = useState<number>(
    initial.requestedGpus ?? 0,
  );
  const [qatConvert, setQatConvert] = useState(initial.qatConvert ?? "");
  const [priority, setPriority] = useState<number>(0);

  // Backfill tokenizer defaults once quick-paths resolves, but only
  // when the user has neither a persisted value nor an in-form value.
  // ``persisted`` is captured from the initial localStorage read; it
  // does not change at runtime.
  useEffect(() => {
    if (!persisted.addTokens && defaultAddTokens) {
      setAddTokens((cur) => cur || defaultAddTokens);
    }
    if (!persisted.chatTemplatePath && defaultChatTemplate) {
      setChatTemplatePath((cur) => cur || defaultChatTemplate);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultAddTokens, defaultChatTemplate]);

  const maxGpus = Math.max(0, gpusQ.data?.length ?? 0);
  // The CLI says these flags are mutually exclusive — the script will
  // also reject the combination, but disabling --root-copy in the UI
  // when --keep-optimizer is on saves the user a round-trip.
  const rootCopyDisabled = keepOptimizer;
  const keepOptimizerDisabled = rootCopy;

  const resetDefaults = () => {
    persistRemove(STORAGE_KEY);
    // When invoked from a context menu, the source path was pinned by
    // the caller — Reset shouldn't unpick it, since clearing it would
    // throw away the very thing the right-click flow is about.
    setSource(initialSource ?? DEFAULTS.source);
    setDestParent(DEFAULTS.destParent);
    setModelName(DEFAULTS.modelName);
    setCheckpoint(DEFAULTS.checkpoint);
    // Reset to the bundled ChatML defaults if the repo path is
    // available; otherwise fall back to the empty DEFAULTS values.
    setAddTokens(defaultAddTokens || DEFAULTS.addTokens);
    setSkipDefaultTokens(DEFAULTS.skipDefaultTokens);
    setChatTemplatePath(defaultChatTemplate || DEFAULTS.chatTemplatePath);
    setNoAutoStopTokens(DEFAULTS.noAutoStopTokens);
    setStopTokens(DEFAULTS.stopTokens);
    setGenerationConfig(DEFAULTS.generationConfig);
    setKeepOptimizer(DEFAULTS.keepOptimizer);
    setRootCopy(DEFAULTS.rootCopy);
    setSafetensors(DEFAULTS.safetensors);
    setDtype(DEFAULTS.dtype);
    setDevice(DEFAULTS.device);
    setDryRun(DEFAULTS.dryRun);
    setLogLevel(DEFAULTS.logLevel);
    setRequestedGpus(DEFAULTS.requestedGpus);
    setQatConvert(DEFAULTS.qatConvert);
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
    const parent = destParent.trim();
    const name = modelName.trim();
    if (!parent || !name) return "";
    return joinDestPath(parent, name);
  })();

  const submit = () => {
    const src = source.trim();
    if (!src || !dst) return;

    savePersisted({
      source: src,
      destParent: destParent.trim(),
      modelName: modelName.trim(),
      checkpoint: checkpoint.trim(),
      addTokens: addTokens.trim(),
      skipDefaultTokens,
      chatTemplatePath: chatTemplatePath.trim(),
      noAutoStopTokens,
      stopTokens: stopTokens.trim(),
      generationConfig: generationConfig.trim(),
      keepOptimizer,
      rootCopy,
      safetensors,
      dtype,
      device: device.trim(),
      dryRun,
      logLevel,
      requestedGpus,
      qatConvert,
    });

    const job_params: Record<string, unknown> = {
      source: src,
      dest: dst,
      skip_default_tokens: skipDefaultTokens,
      no_auto_stop_tokens: noAutoStopTokens,
      keep_optimizer: keepOptimizer,
      root_copy: rootCopy,
      safetensors,
      dry_run: dryRun,
      log_level: logLevel,
    };
    const ck = checkpoint.trim();
    if (ck) job_params.checkpoint = ck;
    const at = addTokens.trim();
    if (at) job_params.add_tokens = at;
    const ct = chatTemplatePath.trim();
    if (ct) job_params.chat_template_path = ct;
    const st = stopTokens.trim();
    if (st) job_params.stop_tokens = st;
    const gc = generationConfig.trim();
    // "carry" is the script's own default — omit the flag.
    if (gc && gc !== "carry") job_params.generation_config = gc;
    // "keep" is a UI sentinel meaning "don't override checkpoint dtype".
    if (dtype && dtype !== "keep") job_params.dtype = dtype;
    const dev = device.trim();
    if (dev) job_params.device = dev;
    if (qatConvert) job_params.qat_convert = qatConvert;

    enqueue.mutate({
      // project_dir isn't meaningful for finalize; use the dest path so
      // logs can still link back to where the artifact lives.
      project_dir: dst,
      config: `finalize:${basename(dst)}`,
      dynamic_args: {},
      requested_gpus: requestedGpus,
      priority,
      job_type: "finalize",
      job_params,
    });
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Run forgather finalize"
      >
        <header className="modal-header">
          <h3>Run forgather finalize</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-row">
            <label className="wide">
              Source model directory
              <PathField
                value={source}
                onChange={setSource}
                mode="dirs-only"
                title="Pick the source output_models/X tree or HF dir"
                wide
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Output parent directory
              <PathField
                value={destParent}
                onChange={setDestParent}
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
                placeholder="optional — defaults to latest under SOURCE/checkpoints/"
                title="Pick checkpoint directory"
                wide
              />
            </label>
          </div>

          <h4 className="dyn-heading">Tokenizer</h4>
          <div className="submit-row">
            <label className="wide">
              Add tokens
              <PathField
                value={addTokens}
                onChange={setAddTokens}
                mode="files-and-dirs"
                placeholder="optional — YAML of tokens to add"
                title="Pick add-tokens YAML"
                wide
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Chat template
              <PathField
                value={chatTemplatePath}
                onChange={setChatTemplatePath}
                mode="files-and-dirs"
                placeholder="optional — Jinja2 chat template"
                title="Pick chat template"
                wide
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={skipDefaultTokens}
                onChange={(e) => setSkipDefaultTokens(e.target.checked)}
              />
              <code>--skip-default-tokens</code>
              <span className="muted">don't auto-add PAD</span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={noAutoStopTokens}
                onChange={(e) => setNoAutoStopTokens(e.target.checked)}
              />
              <code>--no-auto-stop-tokens</code>
              <span className="muted">disable end-of-turn auto-detect</span>
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Stop tokens
              <input
                type="text"
                className="wide"
                value={stopTokens}
                onChange={(e) => setStopTokens(e.target.value)}
                placeholder='comma-separated, e.g. "<|stop|>,<|end|>"'
              />
            </label>
          </div>

          <h4 className="dyn-heading">Generation config</h4>
          <div className="submit-row">
            <label className="wide">
              <code>--generation-config</code>
              <select
                value={selectValue}
                onChange={(e) => {
                  const v = e.target.value;
                  // Switching to "(custom path…)" clears the value so
                  // the PathField starts empty; the user picks a JSON
                  // and that becomes the field value.
                  setGenerationConfig(v === "__custom__" ? "" : v);
                }}
              >
                <option value="carry">carry — copy source's gen config</option>
                <option value="none">none — skip generation_config.json</option>
                {presetNames.length > 0 && (
                  <optgroup label="Presets">
                    {presetNames.map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </optgroup>
                )}
                <option value="__custom__">(custom path…)</option>
              </select>
            </label>
          </div>
          {selectValue === "__custom__" && (
            <div className="submit-row">
              <label className="wide">
                Custom config path
                <PathField
                  value={customPath}
                  onChange={setGenerationConfig}
                  mode="files-and-dirs"
                  placeholder="path to a Forgather inference-preset JSON"
                  title="Pick generation-config JSON"
                  wide
                />
              </label>
            </div>
          )}

          <h4 className="dyn-heading">Output</h4>
          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={keepOptimizer}
                disabled={keepOptimizerDisabled}
                onChange={(e) => setKeepOptimizer(e.target.checked)}
              />
              <code>--keep-optimizer</code>
              <span className="muted">copy optimizer state for warm-start</span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={rootCopy}
                disabled={rootCopyDisabled}
                onChange={(e) => setRootCopy(e.target.checked)}
              />
              <code>--root-copy</code>
              <span className="muted">no checkpoints/ dir; root weights only</span>
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
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
              />
              <code>--dry-run</code>
              <span className="muted">resolve only; don't write</span>
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              <code>--qat-convert</code>
              <select
                value={qatConvert}
                onChange={(e) => setQatConvert(e.target.value)}
                title="Run torchao QAT convert step on a QAT-trained model. Recipe must match what was used at training time."
              >
                <option value="">(none — skip QAT convert)</option>
                {QAT_CONVERT_RECIPES.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
              <span className="muted">
                only set for models trained with <code>--qat-recipe</code>
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
              Scheduler is currently <strong>disabled</strong>. Finalize will
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
              disabled={enqueue.isPending || !source.trim() || !dst}
            >
              {enqueue.isPending ? "Submitting…" : "Run finalize"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}

function basename(p: string): string {
  const i = p.replace(/\/+$/, "").lastIndexOf("/");
  return i < 0 ? p : p.slice(i + 1);
}
