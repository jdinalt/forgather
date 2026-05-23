import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "../api";
import { persistGet, persistRemove, persistSet } from "../persist";
import {
  promptAndCreateService,
  sanitizeServiceName,
  saveServiceArgsAndMaybeRestart,
} from "../services-create";
import { AutoWatchTtyToggle } from "./AutoWatchTtyToggle";
import { ModalBackdrop } from "./ModalBackdrop";
import { PathField } from "./PathField";

function argStr(args: Record<string, unknown>, key: string, def: string): string {
  const v = args[key];
  return typeof v === "string" ? v : def;
}
function argNum(args: Record<string, unknown>, key: string, def: number): number {
  const v = args[key];
  return typeof v === "number" ? v : def;
}
function argBool(args: Record<string, unknown>, key: string, def: boolean): boolean {
  const v = args[key];
  return typeof v === "boolean" ? v : def;
}

/** Decode ``locals`` from the wire shape (list of [name, path] pairs)
 *  back into the row objects the UI uses. */
function decodeLocals(
  raw: unknown,
): Array<{ name: string; path: string }> {
  if (!Array.isArray(raw)) return [];
  const out: Array<{ name: string; path: string }> = [];
  for (const entry of raw) {
    if (
      Array.isArray(entry) &&
      entry.length >= 2 &&
      typeof entry[0] === "string" &&
      typeof entry[1] === "string"
    ) {
      out.push({ name: entry[0], path: entry[1] });
    }
  }
  return out;
}

/** Settings persisted across "Start Dataset Server…" invocations. Matches
 *  the InferenceModal pattern: ``priority`` resets each time because the
 *  "right" value depends on what's currently running; everything else is
 *  sticky so the operator doesn't have to retype config-file paths or
 *  local-mapping lists each time. */
interface PersistedDatasetServer {
  host: string;
  port: number;
  logLevel: string;
  noAuth: boolean;
  quietTokens: boolean;
  noHf: boolean;
  allowPaths: boolean;
  allowDownloads: boolean;
  configFile: string;
  locals: Array<{ name: string; path: string }>;
}

const STORAGE_KEY = "forgather-dataset-server-v1";

function loadPersisted(): Partial<PersistedDatasetServer> {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function savePersisted(s: PersistedDatasetServer) {
  persistSet(STORAGE_KEY, JSON.stringify(s));
}

interface Props {
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
  onServiceCreated?: (type: "dataset") => void;
  /** When set, the modal switches into "Edit service" mode: state is
   *  hydrated from ``editingService.args`` (instead of localStorage),
   *  title + footer change, Save calls
   *  ``saveServiceArgsAndMaybeRestart`` with the fixed name. */
  editingService?: {
    name: string;
    enabled: boolean;
    running: boolean;
    args: Record<string, unknown>;
  };
}

export function DatasetServerModal({
  onClose,
  onSubmitted,
  onServiceCreated,
  editingService,
}: Props) {
  const qc = useQueryClient();
  const isEdit = !!editingService;
  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
  });

  // In edit mode the existing service's args win; localStorage is
  // ignored so we don't overlay stale fields onto a service the user
  // created with different settings.
  const persisted = isEdit ? {} : loadPersisted();
  const editArgs = editingService?.args ?? {};

  // Default to "localhost" (loopback) — matches the dataset_server's own
  // default. Users who want LAN-reachable bind explicitly enter 0.0.0.0.
  const [host, setHost] = useState<string>(
    isEdit ? argStr(editArgs, "host", "127.0.0.1") : persisted.host ?? "127.0.0.1",
  );
  // Default port matches tools/dataset_server/server.py.
  const [port, setPort] = useState<number>(
    isEdit ? argNum(editArgs, "port", 8766) : persisted.port ?? 8766,
  );
  const [logLevel, setLogLevel] = useState<string>(
    isEdit ? argStr(editArgs, "log_level", "INFO") : persisted.logLevel ?? "INFO",
  );
  const [priority, setPriority] = useState<number>(0);

  const [noAuth, setNoAuth] = useState<boolean>(
    isEdit ? argBool(editArgs, "no_auth", false) : persisted.noAuth ?? false,
  );
  // Suppress bearer-token printing to the TTY log on launch — for
  // public-demo deployments where the TTY pane is visible to untrusted
  // viewers. The token still works; clients/peers discover it through
  // the persisted per-port file as usual.
  const [quietTokens, setQuietTokens] = useState<boolean>(
    isEdit
      ? argBool(editArgs, "quiet_tokens", false)
      : persisted.quietTokens ?? false,
  );
  // Not persisted: this is a one-shot "rotate on this start" knob,
  // not a default to carry between modal opens. In edit mode we hydrate
  // from the args (the wire field is ``regen_token``) but it's still
  // applied as a one-shot when the service restarts.
  const [regenToken, setRegenToken] = useState<boolean>(
    isEdit ? argBool(editArgs, "regen_token", false) : false,
  );
  // Toggling --no-auth makes regenToken meaningless; clear it so the
  // visible checked state always matches the disabled state. Without
  // this the box can look "checked but greyed", which confuses users.
  useEffect(() => {
    if (noAuth && regenToken) setRegenToken(false);
  }, [noAuth, regenToken]);
  const [noHf, setNoHf] = useState<boolean>(
    isEdit ? argBool(editArgs, "no_hf", false) : persisted.noHf ?? false,
  );
  const [allowPaths, setAllowPaths] = useState<boolean>(
    isEdit
      ? argBool(editArgs, "allow_paths", false)
      : persisted.allowPaths ?? false,
  );
  const [allowDownloads, setAllowDownloads] = useState<boolean>(
    isEdit
      ? argBool(editArgs, "allow_downloads", false)
      : persisted.allowDownloads ?? false,
  );
  const [configFile, setConfigFile] = useState<string>(
    isEdit
      ? argStr(editArgs, "config_file", "")
      : persisted.configFile ?? "",
  );
  const [locals, setLocals] = useState<Array<{ name: string; path: string }>>(
    isEdit ? decodeLocals(editArgs.locals) : persisted.locals ?? [],
  );
  const [saving, setSaving] = useState<boolean>(false);

  const resetDefaults = () => {
    persistRemove(STORAGE_KEY);
    setHost("127.0.0.1");
    setPort(8766);
    setLogLevel("INFO");
    setNoAuth(false);
    setQuietTokens(false);
    setRegenToken(false);
    setNoHf(false);
    setAllowPaths(false);
    setAllowDownloads(false);
    setConfigFile("");
    setLocals([]);
  };

  const enqueue = useMutation({
    mutationFn: api.enqueue,
    onSuccess: (item) => {
      qc.invalidateQueries({ queryKey: ["queue"] });
      onSubmitted?.(item.queue_id);
      onClose();
    },
  });

  const addLocal = () =>
    setLocals((rows) => [...rows, { name: "", path: "" }]);
  const removeLocal = (idx: number) =>
    setLocals((rows) => rows.filter((_, i) => i !== idx));
  const updateLocal = (
    idx: number,
    field: "name" | "path",
    value: string,
  ) =>
    setLocals((rows) =>
      rows.map((r, i) => (i === idx ? { ...r, [field]: value } : r)),
    );

  // Drop empty rows on submit so the user can leave a trailing blank
  // entry without it tripping the server-side path-existence check.
  const cleanLocals = locals
    .map(({ name, path }) => ({ name: name.trim(), path: path.trim() }))
    .filter(({ name, path }) => name && path);

  // Block submit on local rows where only one of {name, path} is filled
  // — those represent half-typed input the user almost certainly meant
  // to finish, not delete.
  const partialLocals = locals.some(
    ({ name, path }) =>
      (name.trim() && !path.trim()) || (!name.trim() && path.trim()),
  );

  // Single source of truth for the job_params shape — used by both
  // ``Start server`` (one-shot enqueue) and ``Create service…``
  // (persist into the services config and let the autostart pass kick
  // it off).
  const buildArgs = (): Record<string, unknown> => {
    const args: Record<string, unknown> = {
      host: host.trim() || "127.0.0.1",
      port,
      log_level: logLevel,
      no_auth: noAuth,
      quiet_tokens: quietTokens,
      // ``regen_token`` only meaningful when auth is on; the scheduler
      // ignores it under ``--no-auth`` but no need to ship a stale flag.
      regen_token: regenToken && !noAuth,
      no_hf: noHf,
      allow_paths: allowPaths,
      allow_downloads: allowDownloads,
    };
    const cf = configFile.trim();
    if (cf) args.config_file = cf;
    if (cleanLocals.length > 0) {
      // Wire format: list of [name, path] pairs (JSON has no tuples).
      args.locals = cleanLocals.map(({ name, path }) => [name, path]);
    }
    return args;
  };

  const submit = () => {
    savePersisted({
      host: host.trim() || "127.0.0.1",
      port,
      logLevel,
      noAuth,
      quietTokens,
      noHf,
      allowPaths,
      allowDownloads,
      configFile: configFile.trim(),
      locals: cleanLocals,
    });
    const job_params = buildArgs();
    enqueue.mutate({
      // No project context for this tool; use the working host so the
      // QueueItem still has *some* project_dir field. The launcher
      // ignores it.
      project_dir: "/",
      // Display label on the Queue / Jobs rows — "dataset:<port>" is
      // immediately recognizable and unique per instance.
      config: `dataset:${port}`,
      dynamic_args: {},
      requested_gpus: 0,
      priority,
      job_type: "dataset_server",
      job_params,
    });
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Start dataset server"
      >
        <header className="modal-header">
          <h3>
            {isEdit ? (
              <>
                Edit dataset service: <code>{editingService!.name}</code>
              </>
            ) : (
              "Start dataset server"
            )}
          </h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-row">
            <label>
              Host
              <input
                type="text"
                value={host}
                onChange={(e) => setHost(e.target.value)}
                placeholder="127.0.0.1"
              />
              <span className="muted">use 0.0.0.0 for LAN</span>
            </label>
            <label>
              Port
              <input
                type="number"
                min={1}
                max={65535}
                value={port}
                onChange={(e) => setPort(Number(e.target.value) || 8766)}
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

          <h4 className="dyn-heading">Auth</h4>
          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={noAuth}
                onChange={(e) => setNoAuth(e.target.checked)}
              />
              <code>--no-auth</code>
              <span className="muted">
                disable bearer-token gate (trusted-LAN only)
              </span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={regenToken}
                disabled={noAuth}
                onChange={(e) => setRegenToken(e.target.checked)}
              />
              <code>--regen-token</code>
              <span className="muted">
                rotate the persisted per-port token; existing clients
                will need to re-pull
              </span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={quietTokens}
                disabled={noAuth}
                onChange={(e) => setQuietTokens(e.target.checked)}
              />
              <code>--quiet-tokens</code>
              <span className="muted">
                suppress the bearer token in the launch banner (TTY log
                stays public-safe — clients/peers still get the token
                from the persisted per-port file)
              </span>
            </label>
          </div>

          <h4 className="dyn-heading">Loading policy</h4>
          <div className="submit-row">
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={noHf}
                onChange={(e) => setNoHf(e.target.checked)}
              />
              <code>--no-hf</code>
              <span className="muted">
                disable HF cache loading (only <code>local/*</code> served)
              </span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={allowPaths}
                onChange={(e) => setAllowPaths(e.target.checked)}
              />
              <code>--allow-paths</code>
              <span className="muted">
                allow filesystem-path loads (off by default)
              </span>
            </label>
            <label className="dyn-checkbox">
              <input
                type="checkbox"
                checked={allowDownloads}
                onChange={(e) => setAllowDownloads(e.target.checked)}
              />
              <code>--allow-downloads</code>
              <span className="muted">
                allow HF downloads on cache miss
              </span>
            </label>
          </div>

          <h4 className="dyn-heading">Local datasets</h4>
          <div className="muted" style={{ marginBottom: 6 }}>
            Each row is registered as <code>local/NAME</code>; clients then
            request <code>local/NAME</code> instead of the host filesystem
            path. Path must exist on the server host.
          </div>
          {locals.map((row, idx) => (
            <div className="submit-row" key={idx}>
              <label>
                Name
                <input
                  type="text"
                  value={row.name}
                  onChange={(e) => updateLocal(idx, "name", e.target.value)}
                  placeholder="stories"
                />
              </label>
              <label className="wide">
                Path
                <PathField
                  value={row.path}
                  onChange={(v) => updateLocal(idx, "path", v)}
                  placeholder="/data/tinystories"
                  mode="dirs-only"
                  title="Pick local dataset directory"
                  wide
                  // Adding the next local is almost always picking a
                  // sibling under the same data root — open the
                  // browser at the parent of the last pick.
                  rememberKey="dataset.local"
                />
              </label>
              <button
                className="tiny"
                onClick={() => removeLocal(idx)}
                title="Remove this mapping"
              >
                ×
              </button>
            </div>
          ))}
          <div className="submit-row">
            <button className="secondary" onClick={addLocal}>
              + Add local mapping
            </button>
          </div>

          <h4 className="dyn-heading">Advanced</h4>
          <div className="submit-row">
            <label className="wide">
              Config file
              <PathField
                value={configFile}
                onChange={setConfigFile}
                placeholder="optional — YAML config (overrides above merged in)"
                mode="files-and-dirs"
                title="Pick config file"
                wide
              />
              <span className="muted">
                CLI flags override file values; <code>local</code> entries are unioned.
              </span>
            </label>
          </div>
          <div className="submit-row">
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
              </select>
            </label>
          </div>

          {!schedQ.data?.enabled && (
            <div className="notice">
              Scheduler is currently <strong>disabled</strong>. The dataset
              server will enqueue but won't start until the scheduler is
              enabled on the Queue tab.
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div className="muted current-path">
            {enqueue.error ? String(enqueue.error) : ""}
            {!enqueue.error && partialLocals
              ? "Finish or remove partially-filled local-mapping rows."
              : ""}
          </div>
          <div className="btn-row">
            {!isEdit && <AutoWatchTtyToggle />}
            {!isEdit && (
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
            {isEdit ? (
              <button
                onClick={async () => {
                  if (partialLocals) return;
                  setSaving(true);
                  const ok = await saveServiceArgsAndMaybeRestart(
                    qc,
                    "dataset",
                    editingService!.name,
                    editingService!.running,
                    editingService!.enabled,
                    buildArgs(),
                  );
                  setSaving(false);
                  if (ok) onClose();
                }}
                disabled={saving || partialLocals}
                title={
                  editingService!.running
                    ? "Save changes; the running instance will be restarted to apply them"
                    : "Save changes to the service config"
                }
              >
                {saving
                  ? editingService!.running
                    ? "Restarting…"
                    : "Saving…"
                  : editingService!.running
                    ? "Save & restart"
                    : "Save"}
              </button>
            ) : (
              <>
                <button
                  className="secondary"
                  onClick={async () => {
                    // Default name: ``dataset-<port>`` — the port is what
                    // makes multiple instances on the same host distinct
                    // and is always present (defaulted to 8766 in the
                    // form).
                    const suggested = sanitizeServiceName(`dataset-${port}`);
                    const ok = await promptAndCreateService(
                      qc,
                      "dataset",
                      buildArgs(),
                      suggested,
                    );
                    if (ok) {
                      onServiceCreated?.("dataset");
                      onClose();
                    }
                  }}
                  disabled={partialLocals}
                  title="Persist these settings to the server config as an auto-start service"
                >
                  Create service…
                </button>
                <button
                  onClick={submit}
                  disabled={enqueue.isPending || partialLocals}
                >
                  {enqueue.isPending ? "Submitting…" : "Start server"}
                </button>
              </>
            )}
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
