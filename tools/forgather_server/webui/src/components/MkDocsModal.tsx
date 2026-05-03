import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { PathField } from "./PathField";

/** Settings persisted across sidebar-Tools "MkDocs…" invocations. The
 *  next open of the global tool defaults to the user's last-committed
 *  values; ``priority`` stays fresh each invocation. */
interface PersistedMkDocs {
  configFile: string;
  host: string;
  port: number;
  strict: boolean;
  livereload: boolean;
  dirty: boolean;
  watchDirs: string;
}

const STORAGE_KEY = "forgather-global-mkdocs-v1";

function loadPersisted(): Partial<PersistedMkDocs> {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function savePersisted(s: PersistedMkDocs) {
  persistSet(STORAGE_KEY, JSON.stringify(s));
}

interface Props {
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}

/** Global "MkDocs…" tool — queues an ``mkdocs serve`` job. The user
 *  picks an ``mkdocs.yml`` and a host:port; the running job appears in
 *  the Jobs view with a clickable URL like the TensorBoard / Inference
 *  cards. */
export function MkDocsModal({ onClose, onSubmitted }: Props) {
  const qc = useQueryClient();
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });
  // Used to derive the default mkdocs.yml when the user has no
  // persisted choice yet — the Forgather repo always has one at root.
  const quickQ = useQuery({
    queryKey: ["fs-quick-paths"],
    queryFn: api.fsQuickPaths,
    staleTime: 5 * 60 * 1000,
  });

  const persisted = loadPersisted();
  const repoMkdocs = useMemo(() => {
    const repo = quickQ.data?.find((q) => q.label === "Forgather repo")?.path;
    return repo ? `${repo.replace(/\/+$/, "")}/mkdocs.yml` : "";
  }, [quickQ.data]);

  const [configFile, setConfigFile] = useState<string>(
    persisted.configFile ?? "",
  );
  // Backfill the default once the quick-paths fetch resolves, but
  // only if the user hasn't typed / picked anything (and didn't
  // have a persisted value to begin with). Cross-server stale-path
  // contamination is no longer possible — persisted state is
  // namespaced by server identity (see persist.ts).
  useEffect(() => {
    if (!persisted.configFile && repoMkdocs) {
      setConfigFile((cur) => cur || repoMkdocs);
    }
    // persisted.configFile is captured from the initial localStorage
    // read; no need to depend on it (it doesn't change at runtime).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repoMkdocs]);
  // Default to "localhost" rather than "127.0.0.1" — both bind to the
  // same loopback addresses, but some browsers (notably ChromeOS over
  // SSH port-forwards) only follow clickable links to "localhost".
  const [host, setHost] = useState<string>(persisted.host ?? "localhost");
  // Default port: mkdocs' own default. Common SSH port-forward target;
  // don't shift it just to dodge first-submit collisions.
  const [port, setPort] = useState<number>(persisted.port ?? 8000);
  const [strict, setStrict] = useState<boolean>(persisted.strict ?? false);
  const [livereload, setLivereload] = useState<boolean>(
    persisted.livereload ?? true,
  );
  const [dirty, setDirty] = useState<boolean>(persisted.dirty ?? false);
  // Free-text comma-separated list of extra --watch dirs. Empty by
  // default — mkdocs already watches the docs/ tree from mkdocs.yml.
  const [watchDirs, setWatchDirs] = useState<string>(persisted.watchDirs ?? "");
  const [priority, setPriority] = useState<number>(0);

  const resetDefaults = () => {
    persistRemove(STORAGE_KEY);
    // Mirror the same fallback the useEffect uses: prefer the
    // discovered repo mkdocs.yml, otherwise an empty path.
    setConfigFile(repoMkdocs || "");
    setHost("localhost");
    setPort(8000);
    setStrict(false);
    setLivereload(true);
    setDirty(false);
    setWatchDirs("");
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
    const finalConfig = configFile.trim();
    if (!finalConfig) return;
    savePersisted({
      configFile: finalConfig,
      host: host.trim() || "localhost",
      port,
      strict,
      livereload,
      dirty,
      watchDirs: watchDirs.trim(),
    });
    const watch = watchDirs
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const job_params: Record<string, unknown> = {
      config_file: finalConfig,
      host: host.trim() || "localhost",
      port,
      strict,
      livereload,
      dirty,
    };
    if (watch.length > 0) job_params.watch = watch;

    enqueue.mutate({
      project_dir: finalConfig,
      // Display label on Jobs / Queue rows — "mkdocs:<port>" mirrors
      // the tensorboard / inference label scheme.
      config: `mkdocs:${port}`,
      dynamic_args: {},
      requested_gpus: 0,
      priority,
      job_type: "mkdocs",
      job_params,
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Start MkDocs"
      >
        <header className="modal-header">
          <h3>Start MkDocs serve</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-row">
            <label className="wide">
              mkdocs.yml
              <PathField
                value={configFile}
                onChange={setConfigFile}
                mode="files-and-dirs"
                title="Pick the project's mkdocs.yml"
                wide
              />
            </label>
          </div>

          <div className="submit-row">
            <label>
              Host
              <input
                type="text"
                value={host}
                onChange={(e) => setHost(e.target.value)}
                placeholder="localhost"
              />
            </label>
            <label>
              Port
              <input
                type="number"
                min={1}
                max={65535}
                value={port}
                onChange={(e) => setPort(Number(e.target.value) || 8000)}
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
          </div>

          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={strict}
                onChange={(e) => setStrict(e.target.checked)}
              />
              <code>--strict</code>
              <span className="muted">treat warnings as errors</span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={livereload}
                onChange={(e) => setLivereload(e.target.checked)}
              />
              live reload
              <span className="muted">auto-rebuild on change</span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={dirty}
                onChange={(e) => setDirty(e.target.checked)}
              />
              <code>--dirty</code>
              <span className="muted">only rebuild changed files</span>
            </label>
          </div>

          <div className="submit-row">
            <label className="wide">
              Extra watch dirs
              <input
                type="text"
                className="wide"
                placeholder="comma-separated paths (optional)"
                value={watchDirs}
                onChange={(e) => setWatchDirs(e.target.value)}
              />
            </label>
          </div>

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. MkDocs will
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
              disabled={enqueue.isPending || !configFile.trim()}
            >
              {enqueue.isPending ? "Submitting…" : "Start MkDocs"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
