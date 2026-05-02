import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api, EvalConfigEntry } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { PathField } from "./PathField";

/** Settings persisted across ad-hoc "Evaluate…" invocations. Mirrors the
 *  InferenceModal pattern: project-backed flows (right-click → Evaluate
 *  on a config or checkpoint) ignore persistence and derive defaults
 *  from props instead. ``requestedGpus`` and ``priority`` reset every
 *  time because the right value depends on current queue / GPU state. */
interface PersistedAdHoc {
  modelPath: string;
  evalName: string;
  trainer: "ddp" | "simple" | "pipeline";
  batchSize: string;
  maxLength: string;
  maxSteps: number;
  dtype: string;
  attn: string;
  compileFlag: boolean;
  ckptPath: string;
  outputDir: string;
}

const AD_HOC_STORAGE_KEY = "forgather-adhoc-eval-v1";

const AD_HOC_DEFAULTS: PersistedAdHoc = {
  modelPath: "",
  evalName: "",
  trainer: "ddp",
  batchSize: "",
  maxLength: "",
  maxSteps: -1,
  dtype: "bfloat16",
  attn: "sdpa",
  compileFlag: false,
  ckptPath: "",
  outputDir: "",
};

function loadAdHoc(): Partial<PersistedAdHoc> {
  const raw = persistGet(AD_HOC_STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveAdHoc(s: PersistedAdHoc) {
  persistSet(AD_HOC_STORAGE_KEY, JSON.stringify(s));
}

interface Props {
  /** Model's output_dir — where ``evals/`` will land. Omit (or pass
   *  empty string) for ad-hoc mode: the modal then renders a PathField
   *  so the user can pick any model directory, persists their choices
   *  across invocations, and shows a "Reset to defaults" button. */
  modelOutputDir?: string;
  /** Human-facing model name for the header (basename of output_dir).
   *  Optional — derived from the picked path in ad-hoc mode. */
  modelName?: string;
  /** If set, --checkpoint <path> is passed; else the script loads from
   *  model dir. Ignored in ad-hoc mode (the user enters this themselves). */
  checkpointPath?: string | null;
  /** Owning training project, if known — used as a display hint on the job row. */
  projectDir?: string;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

export function EvalModal({
  modelOutputDir,
  modelName,
  checkpointPath,
  projectDir,
  onClose,
  onSubmitted,
}: Props) {
  const qc = useQueryClient();
  const configsQ = useQuery({
    queryKey: ["eval-configs"],
    queryFn: api.listEvalConfigs,
  });
  const gpusQ = useQuery({ queryKey: ["gpus-once"], queryFn: api.listGpus });
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });

  // Ad-hoc mode: caller didn't pin a specific model path, so the user
  // picks one here. Show a PathField for the model dir (and a separate
  // PathField for an optional Forgather checkpoint), and seed the rest
  // from the previous invocation's persisted values.
  const adHoc = !modelOutputDir;
  const persisted = adHoc ? loadAdHoc() : {};

  const [modelPath, setModelPath] = useState<string>(
    modelOutputDir ?? persisted.modelPath ?? "",
  );
  const [evalName, setEvalName] = useState<string>(
    adHoc ? persisted.evalName ?? "" : "",
  );
  const [requestedGpus, setRequestedGpus] = useState<number>(1);
  const [priority, setPriority] = useState<number>(0);
  const [trainer, setTrainer] = useState<"ddp" | "simple" | "pipeline">(
    (adHoc ? persisted.trainer : undefined) ?? "ddp",
  );
  const [batchSize, setBatchSize] = useState<string>(
    adHoc ? persisted.batchSize ?? "" : "",
  );
  const [maxLength, setMaxLength] = useState<string>(
    adHoc ? persisted.maxLength ?? "" : "",
  );
  const [maxSteps, setMaxSteps] = useState<number>(
    adHoc ? persisted.maxSteps ?? -1 : -1,
  );
  const [dtype, setDtype] = useState<string>(
    adHoc ? persisted.dtype ?? "bfloat16" : "bfloat16",
  );
  const [attn, setAttn] = useState<string>(
    adHoc ? persisted.attn ?? "sdpa" : "sdpa",
  );
  const [compileFlag, setCompileFlag] = useState<boolean>(
    adHoc ? persisted.compileFlag ?? false : false,
  );
  // ad-hoc users may want to point at a specific Forgather checkpoint
  // dir; project-backed flows pass that in via ``checkpointPath``.
  const [ckptPath, setCkptPath] = useState<string>(
    adHoc ? persisted.ckptPath ?? "" : "",
  );
  const [outputDir, setOutputDir] = useState<string>(
    adHoc ? persisted.outputDir ?? "" : "",
  );

  // Only meaningful in ad-hoc mode — project-backed flows don't touch
  // persistence and derive everything from props.
  const resetDefaults = () => {
    persistRemove(AD_HOC_STORAGE_KEY);
    setModelPath("");
    setEvalName("");
    setTrainer(AD_HOC_DEFAULTS.trainer);
    setBatchSize(AD_HOC_DEFAULTS.batchSize);
    setMaxLength(AD_HOC_DEFAULTS.maxLength);
    setMaxSteps(AD_HOC_DEFAULTS.maxSteps);
    setDtype(AD_HOC_DEFAULTS.dtype);
    setAttn(AD_HOC_DEFAULTS.attn);
    setCompileFlag(AD_HOC_DEFAULTS.compileFlag);
    setCkptPath(AD_HOC_DEFAULTS.ckptPath);
    setOutputDir(AD_HOC_DEFAULTS.outputDir);
  };

  const maxGpus = Math.max(1, gpusQ.data?.length ?? 1);
  const idleGpuCount = useMemo(() => {
    if (!gpusQ.data) return null;
    // Match the scheduler: only excluded / disabled gate dispatch.
    return gpusQ.data.filter((g) => !g.excluded && !g.disabled).length;
  }, [gpusQ.data]);

  const selected: EvalConfigEntry | undefined = useMemo(
    () => configsQ.data?.find((e) => e.name === evalName),
    [configsQ.data, evalName],
  );

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const submit = () => {
    if (!selected) return;
    const finalModelPath = modelPath.trim();
    if (!finalModelPath) return;

    // Persist ad-hoc choices pre-enqueue so the next "Evaluate…" click
    // defaults to the user's last committed intent — even if the
    // request fails. Project-backed flows don't touch persistence.
    if (adHoc) {
      saveAdHoc({
        modelPath: finalModelPath,
        evalName,
        trainer,
        batchSize: batchSize.trim(),
        maxLength: maxLength.trim(),
        maxSteps,
        dtype,
        attn,
        compileFlag,
        ckptPath: ckptPath.trim(),
        outputDir: outputDir.trim(),
      });
    }

    // Project-backed callers pass the checkpoint via prop; ad-hoc users
    // type one into the optional checkpoint PathField. Either way, only
    // forward a non-empty path to the script.
    const effectiveCheckpoint = adHoc ? ckptPath.trim() : checkpointPath ?? "";

    const job_params: Record<string, unknown> = {
      eval_project: selected.project_dir,
      eval_template: selected.template,
      model_path: finalModelPath,
      trainer,
      max_steps: maxSteps,
      dtype,
      attn_implementation: attn,
      compile: compileFlag,
    };
    if (effectiveCheckpoint) job_params.checkpoint_path = effectiveCheckpoint;
    const bs = batchSize.trim();
    if (bs !== "") job_params.batch_size = Number(bs);
    const ml = maxLength.trim();
    if (ml !== "") job_params.max_length = Number(ml);
    const od = outputDir.trim();
    if (od !== "") job_params.output_dir = od;

    // project_dir + config on the QueueItem are display hints only for
    // eval jobs — the scheduler reads job_params. Putting the eval name
    // in `config` means the Jobs/Queue panels show something useful in
    // their existing column.
    enqueue.mutate({
      project_dir: projectDir ?? finalModelPath,
      config: selected.name,
      dynamic_args: {},
      requested_gpus: requestedGpus,
      priority,
      job_type: "eval",
      job_params,
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Submit evaluation job"
      >
        <header className="modal-header">
          <h3>
            {adHoc ? (
              "Evaluate model"
            ) : (
              <>
                Evaluate{" "}
                <code>
                  {modelName ?? basename(modelOutputDir!)}
                  {checkpointPath ? ` @${basename(checkpointPath)}` : ""}
                </code>
              </>
            )}
          </h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          {adHoc ? (
            <div className="submit-row">
              <label className="wide">
                Model path
                <PathField
                  value={modelPath}
                  onChange={setModelPath}
                  placeholder="/path/to/model directory"
                  mode="dirs-only"
                  title="Pick model directory"
                  wide
                />
              </label>
            </div>
          ) : (
            <div className="submit-summary">
              <div>
                <span className="muted">model</span>
                <code>{modelOutputDir}</code>
              </div>
              {checkpointPath && (
                <div>
                  <span className="muted">checkpoint</span>
                  <code>{checkpointPath}</code>
                </div>
              )}
            </div>
          )}

          <div className="submit-row">
            <label>
              Eval config
              <select
                value={evalName}
                onChange={(e) => setEvalName(e.target.value)}
              >
                <option value="">— pick one —</option>
                {(configsQ.data ?? []).map((e) => (
                  <option key={e.name} value={e.name}>
                    {e.name} — {e.description}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="submit-row">
            <label>
              GPUs
              <input
                type="number"
                min={1}
                max={maxGpus}
                value={requestedGpus}
                onChange={(e) =>
                  setRequestedGpus(
                    Math.max(1, Math.min(maxGpus, Number(e.target.value) || 1)),
                  )
                }
              />
              {idleGpuCount !== null && (
                <span className="muted">
                  ({idleGpuCount} idle of {maxGpus})
                </span>
              )}
            </label>
            <label>
              Priority
              <input
                type="number"
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
              />
              <span className="muted">higher runs sooner</span>
            </label>
          </div>

          <h4 className="dyn-heading">Run parameters</h4>
          <div className="submit-row">
            <label>
              Trainer
              <select
                value={trainer}
                onChange={(e) =>
                  setTrainer(e.target.value as "ddp" | "simple" | "pipeline")
                }
              >
                <option value="ddp">ddp</option>
                <option value="simple">simple</option>
                <option value="pipeline">pipeline</option>
              </select>
            </label>
            <label>
              Max steps
              <input
                type="number"
                value={maxSteps}
                onChange={(e) => setMaxSteps(Number(e.target.value) || -1)}
              />
              <span className="muted">-1 = run to end</span>
            </label>
          </div>

          <div className="submit-row">
            <label>
              Batch size
              <input
                type="text"
                placeholder={
                  selected
                    ? `default ${selected.default_batch_size}`
                    : "blank = use config default"
                }
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
              />
            </label>
            <label>
              Max length
              <input
                type="text"
                placeholder={
                  selected
                    ? `default ${selected.default_max_length}`
                    : "blank = use config default"
                }
                value={maxLength}
                onChange={(e) => setMaxLength(e.target.value)}
              />
            </label>
          </div>

          <div className="submit-row">
            <label>
              dtype
              <select
                value={dtype}
                onChange={(e) => setDtype(e.target.value)}
              >
                <option value="bfloat16">bfloat16</option>
                <option value="float16">float16</option>
                <option value="float32">float32</option>
                <option value="float64">float64</option>
              </select>
            </label>
            <label>
              attn impl
              <select
                value={attn}
                onChange={(e) => setAttn(e.target.value)}
              >
                <option value="sdpa">sdpa</option>
                <option value="flex_attention">flex_attention</option>
                <option value="flash_attention_2">flash_attention_2</option>
                <option value="eager">eager</option>
              </select>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={compileFlag}
                onChange={(e) => setCompileFlag(e.target.checked)}
              />
              compile
            </label>
          </div>

          {adHoc && (
            <div className="submit-row">
              <label className="wide">
                Checkpoint path
                <PathField
                  value={ckptPath}
                  onChange={setCkptPath}
                  placeholder="optional — Forgather checkpoint dir; blank loads from model dir"
                  mode="dirs-only"
                  title="Pick checkpoint directory"
                  wide
                />
              </label>
            </div>
          )}

          <div className="submit-row">
            <label className="wide">
              Output dir
              <PathField
                value={outputDir}
                onChange={setOutputDir}
                placeholder="blank = evals/ under model dir"
                mode="dirs-only"
                title="Pick eval output directory"
                wide
              />
            </label>
          </div>

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. The job will
              enqueue but won't start until the scheduler is enabled on the
              Queue tab.
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div className="muted current-path">
            {enqueue.error ? String(enqueue.error) : ""}
          </div>
          <div className="btn-row">
            <AutoWatchTtyToggle />
            {adHoc && (
              <button
                className="secondary"
                onClick={resetDefaults}
                title="Clear persisted settings and restore defaults"
              >
                Reset to defaults
              </button>
            )}
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={enqueue.isPending || !selected || !modelPath.trim()}
            >
              {enqueue.isPending ? "Submitting…" : "Submit"}
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
