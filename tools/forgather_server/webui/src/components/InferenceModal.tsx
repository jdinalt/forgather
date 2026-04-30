import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { PathField } from "./PathField";

/** Settings persisted across ad-hoc "Start Server…" invocations. Project-
 *  backed flows don't read/write this — they derive initial values from
 *  their props instead. Only the fields the user typically customizes go
 *  here; ``requestedGpus`` and ``priority`` reset each time because their
 *  "right" value depends on what's currently running. */
interface PersistedAdHoc {
  modelPath: string;
  port: number;
  host: string;
  fromCheckpoint: boolean;
  ckptPath: string;
  dtype: string;
  attn: string;
  cacheImpl: string;
  compileFlag: boolean;
  compileArgs: string;
  disableKvCache: boolean;
  chatTemplate: string;
}

const AD_HOC_STORAGE_KEY = "forgather-adhoc-inference-v1";

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
  /** Pre-populated model path (a project's ``output_dir``). Omit or
   *  pass empty string for ad-hoc mode — the modal then renders a
   *  PathField so the user can pick or type any model directory. */
  modelOutputDir?: string;
  /** Human-facing label for the modal title. In ad-hoc mode the
   *  basename of the chosen path is used once one is entered. */
  modelName?: string;
  /** If set, the server loads that specific checkpoint via ``-c <path>``;
   *  else ``from_checkpoint`` determines whether ``-c`` (latest) or no
   *  checkpoint flag at all is passed. Seeded by the caller: from a
   *  checkpoint right-click, set to the checkpoint dir; from a model
   *  right-click, null with ``from_checkpoint`` defaulting to true
   *  (load latest Forgather checkpoint). */
  checkpointPath: string | null;
  projectDir?: string;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

export function InferenceModal({
  modelOutputDir,
  modelName,
  checkpointPath,
  projectDir,
  onClose,
  onSubmitted,
}: Props) {
  const qc = useQueryClient();
  const gpusQ = useQuery({ queryKey: ["gpus-once"], queryFn: api.listGpus });
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });

  // Ad-hoc mode: caller didn't pin a specific model path, so the user
  // picks one here. The PathField is shown instead of the read-only
  // summary. Relevant defaults shift: from_checkpoint starts off (a bare
  // HuggingFace directory typically loads via ``from_pretrained``).
  const adHoc = !modelOutputDir;
  // Ad-hoc settings are seeded from the previous invocation's persisted
  // values. Project-backed flows ignore persistence entirely so their
  // initial state always derives from props.
  const persisted = adHoc ? loadAdHoc() : {};
  const [modelPath, setModelPath] = useState<string>(
    modelOutputDir ?? persisted.modelPath ?? "",
  );

  // Default port: the inference server's own default. Many users have
  // SSH port-forwards keyed to this port, so keep the canonical default
  // rather than shifting to dodge first-submit collisions — collisions
  // are easy to fix per-submit.
  const [port, setPort] = useState<number>(persisted.port ?? 8137);
  const [host, setHost] = useState<string>(persisted.host ?? "127.0.0.1");
  // GPU count / priority stay fresh each time — the "right" values
  // depend on current queue + GPU occupancy, not on past choices.
  const [requestedGpus, setRequestedGpus] = useState<number>(1);
  const [priority, setPriority] = useState<number>(0);
  const [ckptPath, setCkptPath] = useState<string>(
    checkpointPath ?? (adHoc ? persisted.ckptPath ?? "" : ""),
  );
  // ``from_checkpoint`` on: use Forgather checkpoint loading (either the
  // path above, or the latest if empty). Off: use Transformers
  // ``from_pretrained`` against the model dir. Default on for project
  // models (user wants the -c flag path); default off for ad-hoc paths
  // that usually point at a plain HF model directory.
  const [fromCheckpoint, setFromCheckpoint] = useState<boolean>(
    adHoc ? persisted.fromCheckpoint ?? false : true,
  );
  const [dtype, setDtype] = useState<string>(persisted.dtype ?? "bfloat16");
  // "default" is a UI-only pseudo-value meaning "don't pass the flag,
  // let the server use its own default". Same applies to cacheImpl.
  const [attn, setAttn] = useState<string>(persisted.attn ?? "sdpa");
  const [cacheImpl, setCacheImpl] = useState<string>(
    persisted.cacheImpl ?? "default",
  );
  const [compileFlag, setCompileFlag] = useState<boolean>(
    adHoc ? persisted.compileFlag ?? false : false,
  );
  const [compileArgs, setCompileArgs] = useState<string>(
    persisted.compileArgs ?? "",
  );
  const [disableKvCache, setDisableKvCache] = useState<boolean>(
    adHoc ? persisted.disableKvCache ?? false : false,
  );
  const [chatTemplate, setChatTemplate] = useState<string>(
    persisted.chatTemplate ?? "",
  );

  // Only meaningful in ad-hoc mode — project-backed flows don't
  // touch persistence and derive everything from props, so a reset
  // would only trigger surprise re-renders.
  const resetDefaults = () => {
    persistRemove(AD_HOC_STORAGE_KEY);
    setModelPath("");
    setPort(8137);
    setHost("127.0.0.1");
    setFromCheckpoint(false);
    setCkptPath("");
    setDtype("bfloat16");
    setAttn("sdpa");
    setCacheImpl("default");
    setCompileFlag(false);
    setCompileArgs("");
    setDisableKvCache(false);
    setChatTemplate("");
  };

  const maxGpus = Math.max(1, gpusQ.data?.length ?? 1);
  const idleGpuCount = useMemo(() => {
    if (!gpusQ.data) return null;
    return gpusQ.data.filter((g) => g.processes.length === 0).length;
  }, [gpusQ.data]);

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const submit = () => {
    const finalPath = modelPath.trim();
    if (!finalPath) return;
    // Persist the ad-hoc choices so the next "Start Server…" click
    // defaults to whatever the user just committed to. Saving pre-
    // enqueue (not in onSuccess) keeps this simple — if the request
    // fails the persisted state still matches the last confirmed
    // intent, which is what the user wants to see when they reopen
    // the modal to retry.
    if (adHoc) {
      saveAdHoc({
        modelPath: finalPath,
        port,
        host,
        fromCheckpoint,
        ckptPath: ckptPath.trim(),
        dtype,
        attn,
        cacheImpl,
        compileFlag,
        compileArgs: compileArgs.trim(),
        disableKvCache,
        chatTemplate: chatTemplate.trim(),
      });
    }
    const job_params: Record<string, unknown> = {
      model_path: finalPath,
      port,
      host,
      dtype,
      from_checkpoint: fromCheckpoint,
      compile: compileFlag,
      disable_kv_cache: disableKvCache,
    };
    // "default" is UI-only — omit the key so the server picks its own.
    if (attn !== "default") job_params.attn_implementation = attn;
    if (cacheImpl !== "default") job_params.cache_implementation = cacheImpl;
    const ck = ckptPath.trim();
    if (ck) job_params.checkpoint_path = ck;
    const ct = chatTemplate.trim();
    if (ct) job_params.chat_template = ct;
    const ca = compileArgs.trim();
    if (ca) job_params.compile_args = ca;

    enqueue.mutate({
      project_dir: projectDir ?? finalPath,
      // Human label on the QueueItem / Job row — "inf:<port>" is more
      // useful than the model name alone when several inference jobs
      // run at once.
      config: `inference:${port}`,
      dynamic_args: {},
      requested_gpus: requestedGpus,
      priority,
      job_type: "inference",
      job_params,
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Start inference server"
      >
        <header className="modal-header">
          <h3>
            {adHoc ? (
              "Start inference server"
            ) : (
              <>
                Serve inference:{" "}
                <code>
                  {modelName}
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
                  placeholder="/path/to/model or HuggingFace cache dir"
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
            </div>
          )}

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

          <div className="submit-row">
            <label>
              Host
              <input
                type="text"
                value={host}
                onChange={(e) => setHost(e.target.value)}
              />
            </label>
            <label>
              Port
              <input
                type="number"
                min={1}
                max={65535}
                value={port}
                onChange={(e) => setPort(Number(e.target.value) || 8137)}
              />
            </label>
          </div>

          <h4 className="dyn-heading">Checkpoint</h4>
          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={fromCheckpoint}
                onChange={(e) => setFromCheckpoint(e.target.checked)}
              />
              Load via Forgather checkpoint (<code>-c</code>)
              <span className="muted">
                off → <code>from_pretrained</code> on the model dir
              </span>
            </label>
          </div>
          {fromCheckpoint && (
            <div className="submit-row">
              <label className="wide">
                Checkpoint path
                <PathField
                  value={ckptPath}
                  onChange={setCkptPath}
                  placeholder="blank = latest checkpoint"
                  mode="dirs-only"
                  title="Pick checkpoint directory"
                  wide
                />
              </label>
            </div>
          )}

          <h4 className="dyn-heading">Model options</h4>
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
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={compileFlag}
                onChange={(e) => setCompileFlag(e.target.checked)}
              />
              compile
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={disableKvCache}
                onChange={(e) => setDisableKvCache(e.target.checked)}
              />
              disable kv cache
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Attention Implementation
              <select
                value={attn}
                onChange={(e) => setAttn(e.target.value)}
              >
                <option value="default">default</option>
                <option value="sdpa">sdpa</option>
                <option value="flex_attention">flex_attention</option>
                <option value="flash_attention_2">flash_attention_2</option>
              </select>
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Cache Implementation
              <select
                value={cacheImpl}
                onChange={(e) => setCacheImpl(e.target.value)}
              >
                <option value="default">default</option>
                <option value="static">static</option>
                <option value="dynamic">dynamic</option>
              </select>
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Compile Args
              <input
                type="text"
                value={compileArgs}
                onChange={(e) => setCompileArgs(e.target.value)}
                placeholder='YAML, e.g. {mode: "reduce-overhead", fullgraph: true}'
              />
              <span className="muted">
                requires <strong>compile</strong>; passed to <code>torch.compile</code>
              </span>
            </label>
          </div>

          <div className="submit-row">
            <label className="wide">
              Chat template
              <PathField
                value={chatTemplate}
                onChange={setChatTemplate}
                placeholder="optional — path to a Jinja2 chat template"
                mode="files-and-dirs"
                title="Pick chat template"
                wide
              />
            </label>
          </div>

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. The server
              will enqueue but won't start until the scheduler is enabled
              on the Queue tab.
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
              disabled={enqueue.isPending || !modelPath.trim()}
            >
              {enqueue.isPending ? "Submitting…" : "Start server"}
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
