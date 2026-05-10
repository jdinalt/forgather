import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import {
  AddDatasetServerRequest,
  DatasetServerLocal,
  DatasetServerUser,
  api,
} from "../api";
import { ModalBackdrop } from "./ModalBackdrop";

type SubTab = "servers";

/** Identifier the panel uses to refer to either kind of server uniformly.
 *  Local servers key by ``queue_id`` (stable across the run), user
 *  entries key by registry ``id`` (8 hex chars). */
type ServerKey =
  | { kind: "local"; queue_id: string }
  | { kind: "user"; id: string };

interface SelectedServer {
  key: ServerKey;
  base_url: string;
  label: string;
  has_auth_token: boolean;
  alive: boolean | null; // null = unknown (user entry)
}

function keyMatches(a: ServerKey, b: ServerKey): boolean {
  if (a.kind !== b.kind) return false;
  return a.kind === "local"
    ? b.kind === "local" && a.queue_id === b.queue_id
    : b.kind === "user" && (a as { id: string }).id === (b as { id: string }).id;
}

/** Top-level Datasets view. First and only tab so far is "Servers" —
 *  more tabs (Browse, Inspect, …) will land alongside it. */
export function DatasetsPanel() {
  const [tab, setTab] = useState<SubTab>("servers");
  return (
    <div className="inference-panel">
      <header className="viewer-header inference-header">
        <div className="inference-header-title">
          <strong>Datasets</strong>
          <nav className="tabs">
            <button
              className={tab === "servers" ? "active" : ""}
              onClick={() => setTab("servers")}
            >
              servers
            </button>
          </nav>
        </div>
      </header>

      <div
        style={{
          display: tab === "servers" ? "block" : "none",
          flex: 1,
          minHeight: 0,
          overflow: "auto",
        }}
      >
        <DatasetServersTab />
      </div>
    </div>
  );
}

function DatasetServersTab() {
  const qc = useQueryClient();
  const localsQ = useQuery({
    queryKey: ["dataset-servers-local"],
    queryFn: api.listLocalDatasetServers,
    refetchInterval: 5000,
  });
  const usersQ = useQuery({
    queryKey: ["dataset-servers-user"],
    queryFn: api.listUserDatasetServers,
  });

  const [selected, setSelected] = useState<SelectedServer | null>(null);
  const [addOpen, setAddOpen] = useState(false);

  // Token entered manually for the *currently selected* user entry,
  // when the server hasn't stored one. Empty string means the user
  // hasn't typed anything yet; the proxy then falls back to its
  // saved-registry value (None for blank entries). Local servers never
  // need this — the proxy auto-looks-up from the JobRecord.
  const [pendingToken, setPendingToken] = useState<string>("");

  const localServers = localsQ.data ?? [];
  const userServers = usersQ.data ?? [];

  // When the selected entry disappears (e.g. a local server exits, or
  // the user deletes their entry), clear the selection so the action
  // buttons don't fire against a stale URL.
  useMemo(() => {
    if (!selected) return;
    if (selected.key.kind === "local") {
      const found = localServers.find(
        (s) =>
          selected.key.kind === "local" && s.queue_id === selected.key.queue_id,
      );
      if (!found) setSelected(null);
      else if (found.base_url !== selected.base_url) {
        // host/port changed — re-sync the resolved URL
        setSelected({
          key: { kind: "local", queue_id: found.queue_id },
          base_url: found.base_url,
          label: found.label,
          has_auth_token: found.has_auth_token,
          alive: found.alive,
        });
      }
    } else {
      const found = userServers.find(
        (s) => selected.key.kind === "user" && s.id === selected.key.id,
      );
      if (!found) setSelected(null);
    }
  }, [selected, localServers, userServers]);

  const onPickLocal = (s: DatasetServerLocal) => {
    setSelected({
      key: { kind: "local", queue_id: s.queue_id },
      base_url: s.base_url,
      label: s.label,
      has_auth_token: s.has_auth_token,
      alive: s.alive,
    });
    setPendingToken("");
  };
  const onPickUser = (s: DatasetServerUser) => {
    setSelected({
      key: { kind: "user", id: s.id },
      base_url: s.base_url,
      label: s.label,
      has_auth_token: s.has_auth_token,
      alive: null,
    });
    setPendingToken("");
  };

  const removeUser = useMutation({
    mutationFn: (id: string) => api.deleteUserDatasetServer(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-servers-user"] });
    },
  });

  return (
    <div className="inference-model-panel">
      <section>
        <h4 className="dyn-heading">
          Local dataset servers
          <span className="muted"> ({localServers.length})</span>
        </h4>
        {localServers.length === 0 && (
          <div className="muted pane-state-small">
            No dataset_server jobs — start one from Tools → Start Dataset Server…
          </div>
        )}
        <ul className="inference-server-list">
          {localServers.map((s) => {
            const sel =
              selected !== null &&
              keyMatches(selected.key, { kind: "local", queue_id: s.queue_id });
            return (
              <li
                key={s.queue_id}
                className={
                  "inference-server-row" + (sel ? " selected" : "")
                }
                onClick={() => onPickLocal(s)}
              >
                <div className="inference-server-row-line">
                  <span
                    className={
                      "queue-status " +
                      (s.alive ? "status-running" : "status-done")
                    }
                  >
                    {s.alive ? "ALIVE" : "DEAD"}
                  </span>
                  <span className="inference-server-url">{s.base_url}</span>
                  <span className="muted">· {s.label}</span>
                  {s.has_auth_token && (
                    <span className="muted">· auth ✓</span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      <section>
        <h4 className="dyn-heading">
          User-added servers
          <span className="muted"> ({userServers.length})</span>
          <button
            className="tiny"
            style={{ marginLeft: 8 }}
            onClick={() => setAddOpen(true)}
            title="Register a remote dataset_server URL"
          >
            + Add
          </button>
        </h4>
        {userServers.length === 0 && (
          <div className="muted pane-state-small">
            No user-added servers. Use “+ Add” to register a remote URL.
          </div>
        )}
        <ul className="inference-server-list">
          {userServers.map((s) => {
            const sel =
              selected !== null &&
              keyMatches(selected.key, { kind: "user", id: s.id });
            return (
              <li
                key={s.id}
                className={
                  "inference-server-row" + (sel ? " selected" : "")
                }
                onClick={() => onPickUser(s)}
              >
                <div className="inference-server-row-line">
                  <span className="queue-status status-unknown">USER</span>
                  <span className="inference-server-url">{s.base_url}</span>
                  <span className="muted">· {s.label}</span>
                  {s.has_auth_token && (
                    <span className="muted">· auth ✓</span>
                  )}
                  <button
                    className="tiny"
                    style={{ marginLeft: "auto" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (
                        window.confirm(
                          `Remove ${s.label} from the registry?`,
                        )
                      ) {
                        removeUser.mutate(s.id);
                      }
                    }}
                    title="Remove this entry"
                  >
                    ×
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      {selected && (
        <ServerActions
          selected={selected}
          pendingToken={pendingToken}
          setPendingToken={setPendingToken}
        />
      )}

      {addOpen && (
        <AddServerModal
          onClose={() => setAddOpen(false)}
          onAdded={() => {
            qc.invalidateQueries({ queryKey: ["dataset-servers-user"] });
            setAddOpen(false);
          }}
        />
      )}
    </div>
  );
}

interface ServerActionsProps {
  selected: SelectedServer;
  pendingToken: string;
  setPendingToken: (t: string) => void;
}

type ResultKind = "status" | "datasets" | "cache" | "local";

interface FetchResult {
  kind: ResultKind;
  data: unknown;
  error: string | null;
  fetched_at: number;
}

function ServerActions({
  selected,
  pendingToken,
  setPendingToken,
}: ServerActionsProps) {
  const [result, setResult] = useState<FetchResult | null>(null);
  const [pending, setPending] = useState<ResultKind | null>(null);

  // Local-server tokens flow via the proxy's JobRecord auto-lookup, so
  // ``pendingToken`` only matters for user entries (where the registry
  // either has a token or doesn't, and the user may want to override
  // for one-off requests).
  const tokenToUse = selected.key.kind === "user" ? pendingToken : "";

  const runFetch = async (kind: ResultKind) => {
    setPending(kind);
    setResult(null);
    try {
      let data: unknown;
      const base = selected.base_url;
      if (kind === "status") data = await api.datasetServerHealth(base, tokenToUse);
      else if (kind === "datasets")
        data = await api.datasetServerDatasets(base, tokenToUse);
      else if (kind === "cache") data = await api.datasetServerCache(base, tokenToUse);
      else data = await api.datasetServerLocal(base, tokenToUse);
      setResult({ kind, data, error: null, fetched_at: Date.now() });
    } catch (e) {
      setResult({
        kind,
        data: null,
        error: e instanceof Error ? e.message : String(e),
        fetched_at: Date.now(),
      });
    } finally {
      setPending(null);
    }
  };

  return (
    <section>
      <h4 className="dyn-heading">
        Selected:{" "}
        <code style={{ marginLeft: 6 }}>{selected.base_url}</code>
      </h4>

      {selected.key.kind === "user" && (
        <div className="submit-row">
          <label className="wide">
            Auth token (override)
            <input
              type="password"
              value={pendingToken}
              onChange={(e) => setPendingToken(e.target.value)}
              placeholder={
                selected.has_auth_token
                  ? "leave blank to use the registered token"
                  : "leave blank if the server runs --no-auth"
              }
            />
          </label>
        </div>
      )}

      <div className="submit-row">
        <button
          onClick={() => runFetch("status")}
          disabled={pending !== null}
          title="GET /v1/health"
        >
          Status
        </button>
        <button
          onClick={() => runFetch("datasets")}
          disabled={pending !== null}
          title="GET /v1/datasets — currently loaded handles"
        >
          Handles
        </button>
        <button
          onClick={() => runFetch("cache")}
          disabled={pending !== null}
          title="GET /v1/cache/hf — HF cache contents on the server host"
        >
          HF Cache
        </button>
        <button
          onClick={() => runFetch("local")}
          disabled={pending !== null}
          title="GET /v1/local — registered local/* dataset mappings"
        >
          Local
        </button>
      </div>

      {result && (
        <div style={{ marginTop: 8 }}>
          <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>
            {result.kind} · fetched{" "}
            {new Date(result.fetched_at).toLocaleTimeString()}
            {result.error ? " · error" : ""}
          </div>
          {result.error ? (
            <pre className="pane-state err" style={{ whiteSpace: "pre-wrap" }}>
              {result.error}
            </pre>
          ) : (
            <pre
              style={{
                background: "var(--bg)",
                border: "1px solid var(--border)",
                padding: 8,
                fontSize: 12,
                overflow: "auto",
                maxHeight: 320,
              }}
            >
              {JSON.stringify(result.data, null, 2)}
            </pre>
          )}
        </div>
      )}
    </section>
  );
}

function AddServerModal({
  onClose,
  onAdded,
}: {
  onClose: () => void;
  onAdded: () => void;
}) {
  const [label, setLabel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [authToken, setAuthToken] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setPending(true);
    setError(null);
    try {
      const req: AddDatasetServerRequest = {
        label: label.trim(),
        base_url: baseUrl.trim(),
        auth_token: authToken.trim(),
      };
      await api.addUserDatasetServer(req);
      onAdded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal submit-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Add dataset server"
      >
        <header className="modal-header">
          <h3>Add dataset server</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="modal-body">
          <div className="submit-row">
            <label className="wide">
              Label
              <input
                type="text"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. dataset host"
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Base URL
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://datahost:8766"
              />
              <span className="muted">
                Non-localhost targets require <code>FORGATHER_INFERENCE_PROXY_ALLOW_REMOTE=1</code> on the forgather server.
              </span>
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Auth token
              <input
                type="password"
                value={authToken}
                onChange={(e) => setAuthToken(e.target.value)}
                placeholder="optional — leave blank if the server runs --no-auth"
              />
            </label>
          </div>
        </div>
        <footer className="modal-footer">
          <div className="muted current-path">{error ?? ""}</div>
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button onClick={submit} disabled={pending || !baseUrl.trim()}>
              {pending ? "Adding…" : "Add"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
