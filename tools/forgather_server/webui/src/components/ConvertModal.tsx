import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { PathField } from "./PathField";

/** Settings persisted across sidebar-Tools "Convert…" invocations. The
 *  next open of the global tool defaults to the user's last-committed
 *  values; ``priority`` and ``requestedGpus`` reset each time since the
 *  right value depends on current queue state. */
interface PersistedConvert {
  srcModelPath: string;
  dstModelPath: string;
  reverse: boolean;
  modelType: string;
  dtype: string;
  maxLength: string;
  checkpointPath: string;
  device: string;
  generationTest: boolean;
  prompt: string;
  chatTemplatePath: string;
  addTokens: string;
  skipDefaultTokens: boolean;
  dryRun: boolean;
  logLevel: string;
}

const STORAGE_KEY = "forgather-global-convert-v1";

const DEFAULTS: PersistedConvert = {
  srcModelPath: "",
  dstModelPath: "",
  reverse: false,
  modelType: "auto",
  // "from-model" is a UI-only sentinel meaning "don't pass --dtype" —
  // the convert script then keeps the source model's dtype (falling
  // back to bfloat16 only when the source has none recorded).
  dtype: "from-model",
  maxLength: "",
  checkpointPath: "",
  device: "",
  generationTest: false,
  prompt: "",
  chatTemplatePath: "",
  addTokens: "",
  skipDefaultTokens: false,
  dryRun: false,
  logLevel: "INFO",
};

function loadPersisted(): Partial<PersistedConvert> {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function savePersisted(s: PersistedConvert) {
  persistSet(STORAGE_KEY, JSON.stringify(s));
}

interface Props {
  /** Pre-filled source model path. When set, the modal switches to
   *  project-backed mode: persisted defaults are ignored, the user's
   *  submitted values aren't saved back, and the Reset button is
   *  hidden. Mirrors InferenceModal's ``modelOutputDir`` contract so
   *  context-menu invocations from a specific config don't pollute
   *  the global tool's persisted defaults. */
  initialSrcPath?: string;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

/** Global "Convert…" tool — queues a ``forgather convert`` job. The
 *  user picks src/dst paths plus optional conversion knobs; direction
 *  is auto-detected by the script unless the user forces it via
 *  --reverse. */
export function ConvertModal({ initialSrcPath, onClose, onSubmitted }: Props) {
  const qc = useQueryClient();
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });
  const gpusQ = useQuery({ queryKey: ["gpus-once"], queryFn: api.listGpus });

  // Pinned-src mode (context-menu invocation) just overrides the
  // initial source path; everything else falls back to the global
  // persisted values, and submit still writes the (now context-
  // sourced) values back so the next opening reflects the last run.
  const persisted = loadPersisted();
  const initial = { ...DEFAULTS, ...persisted };
  const [srcModelPath, setSrcModelPath] = useState(
    initialSrcPath ?? initial.srcModelPath,
  );
  const [dstModelPath, setDstModelPath] = useState(initial.dstModelPath);
  const [reverse, setReverse] = useState(initial.reverse);
  const [modelType, setModelType] = useState(initial.modelType);
  const [dtype, setDtype] = useState(initial.dtype);
  const [maxLength, setMaxLength] = useState(initial.maxLength);
  const [checkpointPath, setCheckpointPath] = useState(initial.checkpointPath);
  const [device, setDevice] = useState(initial.device);
  const [generationTest, setGenerationTest] = useState(initial.generationTest);
  const [prompt, setPrompt] = useState(initial.prompt);
  const [chatTemplatePath, setChatTemplatePath] = useState(
    initial.chatTemplatePath,
  );
  const [addTokens, setAddTokens] = useState(initial.addTokens);
  const [skipDefaultTokens, setSkipDefaultTokens] = useState(
    initial.skipDefaultTokens,
  );
  const [dryRun, setDryRun] = useState(initial.dryRun);
  const [logLevel, setLogLevel] = useState(initial.logLevel);
  // Resets each invocation — depends on current queue / GPU state.
  const [requestedGpus, setRequestedGpus] = useState<number>(0);
  const [priority, setPriority] = useState<number>(0);

  const maxGpus = Math.max(0, gpusQ.data?.length ?? 0);

  const resetDefaults = () => {
    persistRemove(STORAGE_KEY);
    // When invoked from a context menu, the source path was pinned by
    // the caller — Reset shouldn't unpick it, since clearing it would
    // throw away the very thing the right-click flow is about.
    setSrcModelPath(initialSrcPath ?? DEFAULTS.srcModelPath);
    setDstModelPath(DEFAULTS.dstModelPath);
    setReverse(DEFAULTS.reverse);
    setModelType(DEFAULTS.modelType);
    setDtype(DEFAULTS.dtype);
    setMaxLength(DEFAULTS.maxLength);
    setCheckpointPath(DEFAULTS.checkpointPath);
    setDevice(DEFAULTS.device);
    setGenerationTest(DEFAULTS.generationTest);
    setPrompt(DEFAULTS.prompt);
    setChatTemplatePath(DEFAULTS.chatTemplatePath);
    setAddTokens(DEFAULTS.addTokens);
    setSkipDefaultTokens(DEFAULTS.skipDefaultTokens);
    setDryRun(DEFAULTS.dryRun);
    setLogLevel(DEFAULTS.logLevel);
  };

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const submit = () => {
    const src = srcModelPath.trim();
    const dst = dstModelPath.trim();
    if (!src || !dst) return;
    const ml = maxLength.trim();
    const mlNum = ml ? Number(ml) : NaN;

    savePersisted({
      srcModelPath: src,
      dstModelPath: dst,
      reverse,
      modelType,
      dtype,
      maxLength: ml,
      checkpointPath: checkpointPath.trim(),
      device: device.trim(),
      generationTest,
      prompt,
      chatTemplatePath: chatTemplatePath.trim(),
      addTokens: addTokens.trim(),
      skipDefaultTokens,
      dryRun,
      logLevel,
    });

    const job_params: Record<string, unknown> = {
      src_model_path: src,
      dst_model_path: dst,
      reverse,
      generation_test: generationTest,
      dry_run: dryRun,
      skip_default_tokens: skipDefaultTokens,
      log_level: logLevel,
    };
    // "from-model" is UI-only — omit --dtype so the script keeps the
    // source model's dtype (its own default behaviour).
    if (dtype && dtype !== "from-model") job_params.dtype = dtype;
    // "auto" means "let the script auto-detect" → omit the flag.
    if (modelType && modelType !== "auto") job_params.model_type = modelType;
    if (Number.isFinite(mlNum) && mlNum > 0) job_params.max_length = mlNum;
    const ck = checkpointPath.trim();
    if (ck) job_params.checkpoint_path = ck;
    const dev = device.trim();
    if (dev) job_params.device = dev;
    if (generationTest && prompt) job_params.prompt = prompt;
    const ct = chatTemplatePath.trim();
    if (ct) job_params.chat_template_path = ct;
    const at = addTokens.trim();
    if (at) job_params.add_tokens = at;

    enqueue.mutate({
      // project_dir is required by the QueueItem schema. Convert isn't
      // tied to a Forgather project — use the destination path so logs
      // and "where did this come from" still point somewhere meaningful.
      project_dir: dst,
      // Display label on Jobs / Queue rows; mirrors the
      // tensorboard / inference / mkdocs label scheme.
      config: `convert:${basename(dst)}`,
      dynamic_args: {},
      requested_gpus: requestedGpus,
      priority,
      job_type: "convert",
      job_params,
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Run forgather convert"
      >
        <header className="modal-header">
          <h3>Run forgather convert</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-row">
            <label className="wide">
              Source model path
              <PathField
                value={srcModelPath}
                onChange={setSrcModelPath}
                mode="dirs-only"
                title="Pick the source model directory"
                wide
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Destination model path
              <PathField
                value={dstModelPath}
                onChange={setDstModelPath}
                mode="dirs-only"
                title="Pick the destination directory"
                wide
              />
            </label>
          </div>

          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={reverse}
                onChange={(e) => setReverse(e.target.checked)}
              />
              <code>--reverse</code>
              <span className="muted">
                force Forgather→HF (default: auto-detect direction)
              </span>
            </label>
          </div>

          <div className="submit-row">
            <label>
              Model type
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
              >
                <option value="auto">auto-detect</option>
                <option value="llama">llama</option>
                <option value="mistral">mistral</option>
                <option value="qwen3">qwen3</option>
                <option value="gemma3_text">gemma3_text</option>
              </select>
              <span className="muted">FG→HF only; HF→FG ignores this</span>
            </label>
            <label>
              dtype
              <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
                <option value="from-model">from source model</option>
                <option value="bfloat16">bfloat16</option>
                <option value="float16">float16</option>
                <option value="float32">float32</option>
              </select>
            </label>
            <label>
              Max length
              <input
                type="number"
                min={1}
                value={maxLength}
                onChange={(e) => setMaxLength(e.target.value)}
                placeholder="optional"
              />
            </label>
          </div>

          <div className="submit-row">
            <label className="wide">
              Checkpoint path
              <PathField
                value={checkpointPath}
                onChange={setCheckpointPath}
                mode="dirs-only"
                placeholder="optional — defaults to latest checkpoint in src"
                title="Pick checkpoint directory"
                wide
              />
            </label>
          </div>

          <div className="submit-row">
            <label>
              Device
              <input
                type="text"
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                placeholder="optional, e.g. cuda:0 or cpu"
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

          <h4 className="dyn-heading">Tokenizer</h4>
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
            <label className="wide">
              Add tokens
              <PathField
                value={addTokens}
                onChange={setAddTokens}
                mode="files-and-dirs"
                placeholder="optional — YAML of additional tokens"
                title="Pick add-tokens YAML"
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
              <span className="muted">don't auto-add PAD etc.</span>
            </label>
          </div>

          <h4 className="dyn-heading">Test / debug</h4>
          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={generationTest}
                onChange={(e) => setGenerationTest(e.target.checked)}
              />
              <code>-g</code>
              <span className="muted">run generation test on dest model</span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
              />
              <code>--dry-run</code>
              <span className="muted">don't write output</span>
            </label>
          </div>
          {generationTest && (
            <div className="submit-row">
              <label className="wide">
                Test prompt
                <input
                  type="text"
                  className="wide"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="optional — defaults to script default"
                />
              </label>
            </div>
          )}

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
                0 = CPU only; raise if you set <code>--device cuda…</code>
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
          </div>

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. Convert will
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
              disabled={
                enqueue.isPending || !srcModelPath.trim() || !dstModelPath.trim()
              }
            >
              {enqueue.isPending ? "Submitting…" : "Run convert"}
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
