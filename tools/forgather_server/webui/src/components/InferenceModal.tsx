import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { promptAndCreateService, sanitizeServiceName } from "../services-create";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { PathField } from "./PathField";
import { ModalBackdrop } from "./ModalBackdrop";

interface ModelRow {
  /** Filesystem path to the model directory. */
  path: string;
  /** Routing name for the OpenAI ``model`` field. Empty → auto-derive
   *  from the path basename at submit time. */
  name: string;
}

/** Settings persisted across ad-hoc "Start Server…" invocations. Project-
 *  backed flows don't read/write this — they derive initial values from
 *  their props instead. Only the fields the user typically customizes go
 *  here; ``priority`` resets each time because the "right" value depends
 *  on what's currently running. ``requestedGpus`` is sticky so the user
 *  doesn't have to retype 4 GPUs every server start. */
interface PersistedAdHoc {
  /** Legacy single-model field, still read on load for back-compat with
   *  pre-multi-model persisted state. New writes go to ``models``. */
  modelPath?: string;
  /** Multi-model rows. When >1 entry, the server hosts all of them and
   *  dispatches by the OpenAI ``model`` field. */
  models?: ModelRow[];
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
  /** Multi-model: keep all entries GPU-resident (no swap to CPU). */
  keepOnGpu: boolean;
  chatTemplate: string;
  requestedGpus: number;
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

function deriveName(path: string): string {
  return path
    .replace(/\/+$/, "")
    .split("/")
    .filter(Boolean)
    .pop() ?? "";
}

export function InferenceModal({
  modelOutputDir,
  modelName,
  checkpointPath,
  projectDir,
  onClose,
  onSubmitted,
  onServiceCreated,
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
  const persisted = loadAdHoc();

  // Model rows. Project-backed flows always have exactly one (from props
  // — no add/remove UI). Ad-hoc flows can have many.
  const initialModels: ModelRow[] = modelOutputDir
    ? [{ path: modelOutputDir, name: "" }]
    : persisted.models && persisted.models.length > 0
      ? persisted.models
      : [{ path: persisted.modelPath ?? "", name: "" }];
  const [models, setModels] = useState<ModelRow[]>(initialModels);

  // Default port: the inference server's own default. Many users have
  // SSH port-forwards keyed to this port, so keep the canonical default
  // rather than shifting to dodge first-submit collisions — collisions
  // are easy to fix per-submit.
  const [port, setPort] = useState<number>(persisted.port ?? 8137);
  // Default to "localhost" rather than "127.0.0.1" — both bind to the
  // same loopback addresses, but some browsers (notably ChromeOS over
  // SSH port-forwards) only follow clickable links to "localhost".
  const [host, setHost] = useState<string>(persisted.host ?? "localhost");
  const [requestedGpus, setRequestedGpus] = useState<number>(
    persisted.requestedGpus ?? 1,
  );
  const [priority, setPriority] = useState<number>(0);
  const [ckptPath, setCkptPath] = useState<string>(
    checkpointPath ?? persisted.ckptPath ?? "",
  );
  const [fromCheckpoint, setFromCheckpoint] = useState<boolean>(
    persisted.fromCheckpoint ?? !adHoc,
  );
  const [dtype, setDtype] = useState<string>(persisted.dtype ?? "bfloat16");
  const [attn, setAttn] = useState<string>(persisted.attn ?? "sdpa");
  const [cacheImpl, setCacheImpl] = useState<string>(
    persisted.cacheImpl ?? "default",
  );
  const [compileFlag, setCompileFlag] = useState<boolean>(
    persisted.compileFlag ?? false,
  );
  const [compileArgs, setCompileArgs] = useState<string>(
    persisted.compileArgs ?? "",
  );
  const [disableKvCache, setDisableKvCache] = useState<boolean>(
    persisted.disableKvCache ?? false,
  );
  const [keepOnGpu, setKeepOnGpu] = useState<boolean>(
    persisted.keepOnGpu ?? false,
  );
  const [chatTemplate, setChatTemplate] = useState<string>(
    persisted.chatTemplate ?? "",
  );

  // Project-backed flows force exactly one model row; the add/remove UI
  // never appears. Multi-model is an ad-hoc-only concern (a project is a
  // single model, by definition).
  const allowMultiModel = adHoc;
  const isMultiModel = models.length > 1;

  const updateModel = (idx: number, patch: Partial<ModelRow>) => {
    setModels((rows) =>
      rows.map((r, i) => (i === idx ? { ...r, ...patch } : r)),
    );
  };
  const addModelRow = () => {
    setModels((rows) => [...rows, { path: "", name: "" }]);
  };
  const removeModelRow = (idx: number) => {
    setModels((rows) => rows.filter((_, i) => i !== idx));
  };

  const resetDefaults = () => {
    persistRemove(AD_HOC_STORAGE_KEY);
    setModels([{ path: "", name: "" }]);
    setPort(8137);
    setHost("localhost");
    setFromCheckpoint(false);
    setCkptPath("");
    setDtype("bfloat16");
    setAttn("sdpa");
    setCacheImpl("default");
    setCompileFlag(false);
    setCompileArgs("");
    setDisableKvCache(false);
    setKeepOnGpu(false);
    setChatTemplate("");
    setRequestedGpus(1);
    // Priority is per-session (not persisted) but resetting it here too
    // matches the operator's expectation that "Reset to defaults" puts
    // the form into the same shape it had on first open.
    setPriority(0);
  };

  // No clamp to >=1: a CPU-only server has zero GPUs, and the scheduler
  // dispatches requested_gpus=0 immediately (no GPU reservation). The
  // launcher pins the spawned server to "-d cpu" in that case.
  const maxGpus = gpusQ.data?.length ?? 0;
  const idleGpuCount = useMemo(() => {
    if (!gpusQ.data) return null;
    // Match the scheduler: only excluded / disabled gate dispatch.
    return gpusQ.data.filter((g) => !g.excluded && !g.disabled).length;
  }, [gpusQ.data]);

  // Snap requestedGpus into [0, maxGpus] once the GPU list resolves.
  // Without this, a persisted value of 1 on a CPU-only host (or a
  // previously-set value larger than the now-available count after
  // someone disabled GPUs) would render in the input as-is and submit
  // unchanged unless the user happens to focus the field.
  useEffect(() => {
    if (gpusQ.data === undefined) return;
    setRequestedGpus((cur) => Math.max(0, Math.min(maxGpus, cur)));
  }, [gpusQ.data, maxGpus]);

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  // Resolve the entered rows into the wire shape. Single row → legacy
  // ``model_path: PATH``; multi-row → ``models: [{name, path}]`` with
  // names auto-derived from path basenames when the user didn't override.
  const resolvedModels = useMemo(() => {
    return models
      .map((r) => ({
        path: r.path.trim(),
        name: (r.name.trim() || deriveName(r.path.trim())).trim(),
      }))
      .filter((r) => r.path !== "");
  }, [models]);

  // Detect duplicate routing names — the server would reject these at
  // startup, but it's friendlier to flag in-UI than wait for the spawn
  // to fail.
  const dupNames = useMemo(() => {
    const seen = new Set<string>();
    const dup = new Set<string>();
    for (const r of resolvedModels) {
      if (r.name && seen.has(r.name)) dup.add(r.name);
      seen.add(r.name);
    }
    return Array.from(dup);
  }, [resolvedModels]);

  const buildArgs = (): Record<string, unknown> => {
    const args: Record<string, unknown> = {
      port,
      host,
      dtype,
      from_checkpoint: fromCheckpoint,
      compile: compileFlag,
      disable_kv_cache: disableKvCache,
    };
    // keep_on_gpu only matters with multiple models; single-model
    // servers already keep the sole model resident.
    if (isMultiModel && keepOnGpu) args.keep_on_gpu = true;
    if (resolvedModels.length === 1) {
      // Single-model: legacy job_params shape (scheduler/inference_ops
      // still read ``model_path`` directly for one-model jobs).
      args.model_path = resolvedModels[0].path;
    } else {
      args.models = resolvedModels;
    }
    if (attn !== "default") args.attn_implementation = attn;
    if (cacheImpl !== "default") args.cache_implementation = cacheImpl;
    // Specific-checkpoint path is single-model only; multi-model uses
    // the boolean ``fromCheckpoint`` toggle to load each model's
    // latest checkpoint.
    if (!isMultiModel) {
      const ck = ckptPath.trim();
      if (ck) args.checkpoint_path = ck;
    }
    const ct = chatTemplate.trim();
    if (ct) args.chat_template = ct;
    const ca = compileArgs.trim();
    if (ca) args.compile_args = ca;
    return args;
  };

  const canSubmit =
    resolvedModels.length >= 1 &&
    dupNames.length === 0 &&
    !(isMultiModel && ckptPath.trim());

  const submit = () => {
    if (!canSubmit) return;
    // Persist the choices so the next "Start Server…" click defaults
    // to whatever the user just committed to. Saving pre-enqueue (not
    // in onSuccess) keeps this simple — if the request fails the
    // persisted state still matches the last confirmed intent.
    // In project-backed mode we keep the existing ``modelPath`` field
    // untouched so reopening the ad-hoc modal still shows the user's
    // prior ad-hoc choice.
    saveAdHoc({
      modelPath: adHoc ? resolvedModels[0]?.path ?? "" : persisted.modelPath,
      models: adHoc ? models : undefined,
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
      keepOnGpu,
      chatTemplate: chatTemplate.trim(),
      requestedGpus,
    });
    const job_params = buildArgs();
    // project_dir for the queue row: in project-backed mode use the
    // caller's projectDir; in ad-hoc mode use the first model path so
    // queue-list display has something to show.
    const project_dir =
      projectDir ?? resolvedModels[0]?.path ?? "";
    enqueue.mutate({
      project_dir,
      config: `inference:${port}`,
      dynamic_args: {},
      requested_gpus: requestedGpus,
      priority,
      job_type: "inference",
      job_params,
    });
  };

  return (
    <ModalBackdrop onClose={onClose}>
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
            <>
              <div
                className={
                  "inference-model-rows" +
                  (isMultiModel ? " inference-model-rows-scroll" : "")
                }
              >
                {models.map((row, idx) => (
                  <div className="submit-row" key={idx}>
                    <label className="wide">
                      {isMultiModel ? `Model ${idx + 1}` : "Model path"}
                      <PathField
                        value={row.path}
                        onChange={(v) => updateModel(idx, { path: v })}
                        placeholder="/path/to/model or HuggingFace cache dir"
                        mode="dirs-only"
                        title="Pick model directory"
                        wide
                        // Adding a second model is almost always picking
                        // a sibling of the first — open the browser at
                        // the same parent dir on subsequent clicks.
                        rememberKey="inference.model"
                      />
                    </label>
                    {isMultiModel && (
                      <>
                        <label>
                          Name
                          <input
                            type="text"
                            value={row.name}
                            onChange={(e) =>
                              updateModel(idx, { name: e.target.value })
                            }
                            placeholder={deriveName(row.path) || "auto"}
                          />
                        </label>
                        <button
                          className="tiny"
                          onClick={() => removeModelRow(idx)}
                          title="Remove this model"
                          aria-label="Remove model"
                        >
                          ×
                        </button>
                      </>
                    )}
                  </div>
                ))}
              </div>
              {allowMultiModel && (
                <div className="submit-row">
                  <button
                    className="secondary"
                    onClick={addModelRow}
                    title="Host an additional model in the same server; requests dispatch by the OpenAI 'model' field. Models swap between CPU and GPU on demand."
                  >
                    + Add model
                  </button>
                  {dupNames.length > 0 && (
                    <span className="muted">
                      duplicate name{dupNames.length > 1 ? "s" : ""}:{" "}
                      <code>{dupNames.join(", ")}</code>
                    </span>
                  )}
                </div>
              )}
              {isMultiModel && (
                <div className="submit-row">
                  <label className="dyn-checkbox">
                    <input
                      type="checkbox"
                      checked={keepOnGpu}
                      onChange={(e) => setKeepOnGpu(e.target.checked)}
                    />
                    Keep all models on GPU (no CPU swap)
                    <span className="muted">
                      avoids swap latency; only use if total GPU memory &gt;
                      sum of model sizes
                    </span>
                  </label>
                </div>
              )}
            </>
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
                min={0}
                max={maxGpus}
                value={requestedGpus}
                onChange={(e) => {
                  // Number("") -> NaN, which || 0 turns into 0; explicit 0
                  // is valid (CPU server). Clamp to [0, maxGpus].
                  const raw = Number(e.target.value);
                  const n = Number.isFinite(raw) ? raw : 0;
                  setRequestedGpus(Math.max(0, Math.min(maxGpus, n)));
                }}
              />
              {idleGpuCount !== null && (
                <span className="muted">
                  ({idleGpuCount} idle of {maxGpus})
                  {maxGpus === 0 && " — CPU only"}
                </span>
              )}
              {requestedGpus === 0 && maxGpus > 0 && (
                <span className="muted">
                  0 = run on CPU (no GPU reservation)
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
          {fromCheckpoint && !isMultiModel && (
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
                  rememberKey="inference.checkpoint"
                />
              </label>
            </div>
          )}
          {fromCheckpoint && isMultiModel && (
            <div className="submit-row">
              <span className="muted">
                Multi-model: loads each model's latest checkpoint. The
                specific-checkpoint field is hidden because{" "}
                <code>-c &lt;PATH&gt;</code> is not supported when hosting
                multiple models in one server.
              </span>
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
          {isMultiModel && compileFlag && (
            <div className="submit-row">
              <span className="muted">
                <strong>compile</strong> with multi-model: torch.compile
                artifacts may not survive CPU↔GPU swaps; expect
                recompilation on each swap.
              </span>
            </div>
          )}
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
                rememberKey="inference.chat-template"
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
              className="secondary"
              onClick={async () => {
                if (!canSubmit) return;
                const args = {
                  ...buildArgs(),
                  // Inference services need at least one GPU; persist
                  // the operator's choice so autostart respects it.
                  requested_gpus: requestedGpus,
                };
                // Suggested name:
                //   - single model: that model's basename
                //   - multi-model: ``host-port`` (concatenating all
                //     model names doesn't scale past two or three)
                const suggested =
                  resolvedModels.length === 1
                    ? sanitizeServiceName(deriveName(resolvedModels[0].path))
                    : sanitizeServiceName(`${host}-${port}`);
                const ok = await promptAndCreateService(
                  qc,
                  "inference",
                  args,
                  suggested,
                );
                if (ok) {
                  onServiceCreated?.("inference");
                  onClose();
                }
              }}
              disabled={!canSubmit}
              title="Persist these settings to the server config as an auto-start service"
            >
              Create service…
            </button>
            <button
              onClick={submit}
              disabled={enqueue.isPending || !canSubmit}
            >
              {enqueue.isPending ? "Submitting…" : "Start server"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
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
  /** Fired after the "Create service…" button successfully persists a
   *  new service entry. The caller uses this to auto-expand the
   *  matching launcher row in the sidebar so the new instance is
   *  immediately visible. */
  onServiceCreated?: (type: "inference") => void;
}

function basename(p: string): string {
  const i = p.replace(/\/+$/, "").lastIndexOf("/");
  return i < 0 ? p : p.slice(i + 1);
}
