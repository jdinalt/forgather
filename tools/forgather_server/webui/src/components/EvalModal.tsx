import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api, EvalConfigEntry } from "../api";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { PathField } from "./PathField";

interface Props {
  /** Model's output_dir — where ``evals/`` will land. */
  modelOutputDir: string;
  /** Human-facing model name for the header (basename of output_dir). */
  modelName: string;
  /** If set, --checkpoint <path> is passed; else the script loads from model dir. */
  checkpointPath: string | null;
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

  const [evalName, setEvalName] = useState<string>("");
  const [requestedGpus, setRequestedGpus] = useState<number>(1);
  const [priority, setPriority] = useState<number>(0);
  const [trainer, setTrainer] = useState<"ddp" | "simple" | "pipeline">("ddp");
  const [batchSize, setBatchSize] = useState<string>("");
  const [maxLength, setMaxLength] = useState<string>("");
  const [maxSteps, setMaxSteps] = useState<number>(-1);
  const [dtype, setDtype] = useState<string>("bfloat16");
  const [attn, setAttn] = useState<string>("sdpa");
  const [compileFlag, setCompileFlag] = useState<boolean>(false);
  const [outputDir, setOutputDir] = useState<string>("");

  const maxGpus = Math.max(1, gpusQ.data?.length ?? 1);
  const idleGpuCount = useMemo(() => {
    if (!gpusQ.data) return null;
    return gpusQ.data.filter((g) => g.processes.length === 0).length;
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
    const job_params: Record<string, unknown> = {
      eval_project: selected.project_dir,
      eval_template: selected.template,
      model_path: modelOutputDir,
      trainer,
      max_steps: maxSteps,
      dtype,
      attn_implementation: attn,
      compile: compileFlag,
    };
    if (checkpointPath) job_params.checkpoint_path = checkpointPath;
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
      project_dir: projectDir ?? modelOutputDir,
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
            Evaluate{" "}
            <code>
              {modelName}
              {checkpointPath ? ` @${basename(checkpointPath)}` : ""}
            </code>
          </h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
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
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={enqueue.isPending || !selected}
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
