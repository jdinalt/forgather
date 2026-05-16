import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, ConfigInfo, ProjectInfo } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { promptAndCreateService } from "../services-create";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { ModalBackdrop } from "./ModalBackdrop";
import { PathField } from "./PathField";

/** Settings persisted across sidebar-Tools "TensorBoard…" invocations.
 *  Config-backed flows (ConfigTensorBoardModal) don't read/write this —
 *  their initial values come from the config's resolved output_dir.
 *  ``priority`` stays fresh each invocation. */
interface PersistedGlobalTb {
  logdir: string;
  port: number;
  bindAll: boolean;
  windowTitle: string;
  reloadInterval: string;
  reloadMultifile: boolean;
  samplesPerPlugin: string;
  host: string;
}

const GLOBAL_STORAGE_KEY = "forgather-global-tensorboard-v1";

function loadGlobalTb(): Partial<PersistedGlobalTb> {
  const raw = persistGet(GLOBAL_STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveGlobalTb(s: PersistedGlobalTb) {
  persistSet(GLOBAL_STORAGE_KEY, JSON.stringify(s));
}

interface Props {
  /** Initial logdir — typically the model's output_dir. User can navigate
   *  upward via the path picker to cover multiple models, or enter an
   *  arbitrary path (a common runs-root, for instance). */
  initialLogdir: string;
  /** Friendly name seeded into the window title (defaults to model name /
   *  project basename). Falls back to the directory basename. */
  initialWindowTitle: string;
  projectDir?: string;
  /** Global / sidebar-Tools invocation — read/write a localStorage key so
   *  the next open defaults to the user's last-committed settings. */
  global?: boolean;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
  onServiceCreated?: (type: "tensorboard") => void;
}

export function TensorBoardModal({
  initialLogdir,
  initialWindowTitle,
  projectDir,
  global,
  onClose,
  onSubmitted,
  onServiceCreated,
}: Props) {
  const qc = useQueryClient();
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });

  const persisted = global ? loadGlobalTb() : {};

  const [logdir, setLogdir] = useState<string>(
    initialLogdir || persisted.logdir || "",
  );
  // TensorBoard's own default. Many users have SSH port-forwards keyed
  // to 6006, so don't pick a different port just to avoid first-submit
  // collisions — collisions are easy to fix per-submit.
  const [port, setPort] = useState<number>(persisted.port ?? 6006);
  const [bindAll, setBindAll] = useState<boolean>(persisted.bindAll ?? false);
  const [windowTitle, setWindowTitle] = useState<string>(
    initialWindowTitle || persisted.windowTitle || "",
  );
  const [priority, setPriority] = useState<number>(0);

  // Advanced options — collapsed by default.
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [reloadInterval, setReloadInterval] = useState<string>(
    persisted.reloadInterval ?? "",
  );
  const [reloadMultifile, setReloadMultifile] = useState<boolean>(
    persisted.reloadMultifile ?? false,
  );
  const [samplesPerPlugin, setSamplesPerPlugin] = useState<string>(
    persisted.samplesPerPlugin ?? "",
  );
  const [host, setHost] = useState<string>(persisted.host ?? "");

  const resetDefaults = () => {
    persistRemove(GLOBAL_STORAGE_KEY);
    // Honour the caller's seed values the same way the initial useState
    // bindings do: ``initialLogdir`` / ``initialWindowTitle`` win when
    // present, otherwise fall back to the empty default.
    setLogdir(initialLogdir || "");
    setPort(6006);
    setBindAll(false);
    setWindowTitle(initialWindowTitle || "");
    setReloadInterval("");
    setReloadMultifile(false);
    setSamplesPerPlugin("");
    setHost("");
  };

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const buildArgs = (): Record<string, unknown> => {
    const args: Record<string, unknown> = {
      logdir,
      port,
      bind_all: bindAll,
      reload_multifile: reloadMultifile,
    };
    const wt = windowTitle.trim();
    if (wt) args.window_title = wt;
    const h = host.trim();
    if (!bindAll && h) args.host = h;
    const ri = reloadInterval.trim();
    if (ri !== "") {
      const n = Number.parseInt(ri, 10);
      if (Number.isFinite(n)) args.reload_interval = n;
    }
    const spp = samplesPerPlugin.trim();
    if (spp) args.samples_per_plugin = spp;
    return args;
  };

  const submit = () => {
    if (global) {
      // Persist the user's choices so the next open of the global
      // TensorBoard tool defaults to whatever they just committed to.
      saveGlobalTb({
        logdir: logdir.trim(),
        port,
        bindAll,
        windowTitle: windowTitle.trim(),
        reloadInterval: reloadInterval.trim(),
        reloadMultifile,
        samplesPerPlugin: samplesPerPlugin.trim(),
        host: host.trim(),
      });
    }
    const job_params = buildArgs();

    enqueue.mutate({
      project_dir: projectDir ?? logdir,
      // Display label on the Jobs / Queue rows — "tb:<port>" is
      // immediately recognizable and unique per instance.
      config: `tensorboard:${port}`,
      dynamic_args: {},
      requested_gpus: 0,
      priority,
      job_type: "tensorboard",
      job_params,
    });
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Start TensorBoard"
      >
        <header className="modal-header">
          <h3>
            Open TensorBoard
            {initialWindowTitle && (
              <>
                : <code>{initialWindowTitle}</code>
              </>
            )}
          </h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-row">
            <label className="wide">
              Log directory
              <PathField
                value={logdir}
                onChange={setLogdir}
                mode="dirs-only"
                title="Pick logdir (typically the model's output_dir)"
                wide
              />
            </label>
          </div>

          <div className="submit-row">
            <label>
              Port
              <input
                type="number"
                min={1}
                max={65535}
                value={port}
                onChange={(e) => setPort(Number(e.target.value) || 6006)}
              />
            </label>
            <label>
              Priority
              <input
                type="number"
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
              />
              <span className="muted">no GPUs reserved</span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={bindAll}
                onChange={(e) => setBindAll(e.target.checked)}
              />
              <code>--bind_all</code>
              <span className="muted">listen on every interface</span>
            </label>
          </div>

          <div className="submit-row">
            <label className="wide">
              Window title
              <input
                type="text"
                className="wide"
                value={windowTitle}
                onChange={(e) => setWindowTitle(e.target.value)}
              />
            </label>
          </div>

          <h4
            className="dyn-heading collapsible-heading"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            <span className="tri">{showAdvanced ? "▾" : "▸"}</span>
            Advanced options
          </h4>
          {showAdvanced && (
            <>
              <div className="submit-row">
                <label>
                  Reload interval
                  <input
                    type="text"
                    placeholder="seconds (blank = default)"
                    value={reloadInterval}
                    onChange={(e) => setReloadInterval(e.target.value)}
                  />
                </label>
                <label className="dyn-checkbox">
                  <input
                    type="checkbox"
                    checked={reloadMultifile}
                    onChange={(e) => setReloadMultifile(e.target.checked)}
                  />
                  <code>--reload_multifile</code>
                </label>
              </div>
              <div className="submit-row">
                <label className="wide">
                  Samples per plugin
                  <input
                    type="text"
                    className="wide"
                    placeholder="e.g. images=100,scalars=500"
                    value={samplesPerPlugin}
                    onChange={(e) => setSamplesPerPlugin(e.target.value)}
                  />
                </label>
              </div>
              {!bindAll && (
                <div className="submit-row">
                  <label>
                    Host
                    <input
                      type="text"
                      placeholder="127.0.0.1"
                      value={host}
                      onChange={(e) => setHost(e.target.value)}
                    />
                  </label>
                </div>
              )}
            </>
          )}

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. TensorBoard
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
            {global && (
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
              className="secondary"
              onClick={async () => {
                if (!logdir.trim()) return;
                const ok = await promptAndCreateService(
                  qc,
                  "tensorboard",
                  buildArgs(),
                );
                if (ok) {
                  onServiceCreated?.("tensorboard");
                  onClose();
                }
              }}
              disabled={!logdir.trim()}
              title="Persist these settings to the server config as an auto-start service"
            >
              Create service…
            </button>
            <button
              onClick={submit}
              disabled={enqueue.isPending || !logdir.trim()}
            >
              {enqueue.isPending ? "Submitting…" : "Start TensorBoard"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}

/** Config-level entry: resolves the config's ``output_dir`` and seeds the
 *  modal. Used from the config's toolbar button and the project-tree
 *  right-click menu so the user can launch TB for a training run before
 *  (or while) it's running. */
export function ConfigTensorBoardModal({
  project,
  config,
  onClose,
  onSubmitted,
}: {
  project: ProjectInfo;
  config: ConfigInfo;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}) {
  const outQ = useQuery({
    queryKey: ["output-dir", project.project_dir, config.name],
    queryFn: () => api.configOutputDir(project.project_dir, config.name),
  });

  if (outQ.isLoading) {
    return (
      <ModalBackdrop onClose={onClose}>
        <div
          className="modal submit-modal"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="pane-state">Resolving output_dir…</div>
        </div>
      </ModalBackdrop>
    );
  }
  if (outQ.error) {
    return (
      <ModalBackdrop onClose={onClose}>
        <div
          className="modal submit-modal"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="pane-state err">
            <pre>{String(outQ.error)}</pre>
          </div>
          <footer className="modal-footer">
            <div className="btn-row">
              <button className="secondary" onClick={onClose}>
                Close
              </button>
            </div>
          </footer>
        </div>
      </ModalBackdrop>
    );
  }

  const logdir = outQ.data?.output_dir ?? project.project_dir;
  const title =
    config.name.replace(/\.ya?ml$/, "") || project.name || project.project_dir;
  return (
    <TensorBoardModal
      initialLogdir={logdir}
      initialWindowTitle={title}
      projectDir={project.project_dir}
      onClose={onClose}
      onSubmitted={onSubmitted}
    />
  );
}
