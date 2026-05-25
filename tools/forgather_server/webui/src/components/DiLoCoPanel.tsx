import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, DiLoCoInfo, DiLoCoServer, DiLoCoStatus } from "../api";
import { persistGet, persistSet } from "../persist";

const STORAGE_KEY = "forgather-diloco-state";

interface PanelState {
  /** Selected server's stable id (local:<queue_id> or registered:<id>). */
  selectedId: string | null;
  /** Auto-refresh cadence for /status, in seconds. */
  refreshSeconds: number;
}

const DEFAULT_STATE: PanelState = {
  selectedId: null,
  refreshSeconds: 3,
};

function loadState(): PanelState {
  const raw = persistGet(STORAGE_KEY);
  if (!raw) return DEFAULT_STATE;
  try {
    const parsed = JSON.parse(raw) as Partial<PanelState>;
    return {
      selectedId:
        typeof parsed.selectedId === "string" ? parsed.selectedId : null,
      refreshSeconds:
        typeof parsed.refreshSeconds === "number"
          ? parsed.refreshSeconds
          : DEFAULT_STATE.refreshSeconds,
    };
  } catch {
    return DEFAULT_STATE;
  }
}

function formatUptime(seconds: number | undefined): string {
  if (seconds === undefined || seconds === null) return "—";
  const s = Math.max(0, Math.floor(seconds));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return hh > 0 ? `${hh}h ${mm}m` : mm > 0 ? `${mm}m ${ss}s` : `${ss}s`;
}

function relativeAge(epoch: number | undefined): string {
  if (!epoch) return "—";
  const dt = Math.max(0, Date.now() / 1000 - epoch);
  if (dt < 1) return "now";
  if (dt < 60) return `${dt.toFixed(0)}s ago`;
  if (dt < 3600) return `${(dt / 60).toFixed(0)}m ago`;
  return `${(dt / 3600).toFixed(1)}h ago`;
}

export function DiLoCoPanel() {
  const [state, setState] = useState<PanelState>(loadState);
  useEffect(() => {
    persistSet(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  // Server list — refresh every 5s; cluster propagation will keep the
  // list in flux once that slice lands, but for now this is just the
  // local + registered union.
  const serversQuery = useQuery({
    queryKey: ["diloco", "servers"],
    queryFn: api.listDiLoCoServers,
    refetchInterval: 5_000,
    refetchIntervalInBackground: false,
  });
  const servers = serversQuery.data ?? [];

  // Auto-pick the first server when nothing is selected and a server
  // shows up. Doesn't override an explicit selection, even if that
  // selection's id has disappeared from the list — leaves the empty
  // detail panel visible so the operator notices the loss themselves.
  useEffect(() => {
    if (state.selectedId === null && servers.length > 0) {
      setState((s) => ({ ...s, selectedId: servers[0].id }));
    }
  }, [state.selectedId, servers]);

  const selected = useMemo(
    () => servers.find((s) => s.id === state.selectedId) ?? null,
    [servers, state.selectedId],
  );

  // /status — polled at `refreshSeconds` cadence whenever a server is
  // selected. /info is fetched once per selected server (it's static-ish).
  const statusQuery = useQuery({
    queryKey: ["diloco", "status", selected?.base_url],
    queryFn: () => api.diLoCoServerStatus(selected!.base_url),
    enabled: !!selected,
    refetchInterval: state.refreshSeconds * 1000,
    refetchIntervalInBackground: false,
  });
  const infoQuery = useQuery({
    queryKey: ["diloco", "info", selected?.base_url],
    queryFn: () => api.diLoCoServerInfo(selected!.base_url),
    enabled: !!selected,
    // /info changes only on server restart; no auto-refresh.
    staleTime: 60_000,
  });

  return (
    <div className="inference-panel">
      <header className="viewer-header inference-header">
        <div className="inference-header-title">
          <strong>DiLoCo</strong>
          {selected && (
            <span className="muted"> — {selected.base_url}</span>
          )}
          <span style={{ flex: 1 }} />
          <label
            className="muted"
            style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
          >
            refresh
            <select
              value={state.refreshSeconds}
              onChange={(e) =>
                setState((s) => ({
                  ...s,
                  refreshSeconds: Number(e.target.value),
                }))
              }
            >
              <option value={1}>1s</option>
              <option value={3}>3s</option>
              <option value={5}>5s</option>
              <option value={10}>10s</option>
              <option value={30}>30s</option>
            </select>
          </label>
          <button
            onClick={() => serversQuery.refetch()}
            title="Refresh server list now"
          >
            ↻
          </button>
        </div>
      </header>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "320px 1fr",
          gap: 12,
          flex: 1,
          minHeight: 0,
          padding: 8,
        }}
      >
        <ServersList
          servers={servers}
          loading={serversQuery.isLoading}
          error={serversQuery.error}
          selectedId={state.selectedId}
          onSelect={(id) =>
            setState((s) => ({ ...s, selectedId: id }))
          }
          onAfterRegistryChange={() => serversQuery.refetch()}
        />

        <div style={{ minHeight: 0, overflow: "auto" }}>
          {!selected ? (
            <div className="muted" style={{ padding: 16 }}>
              {servers.length === 0
                ? "No DiLoCo servers known. Add an external one or spawn a local server."
                : "Select a server to see its status."}
            </div>
          ) : (
            <ServerDetail
              server={selected}
              status={statusQuery.data ?? null}
              info={infoQuery.data ?? null}
              statusLoading={statusQuery.isLoading}
              statusError={statusQuery.error}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Server list (left pane)
// ---------------------------------------------------------------------------

interface ServersListProps {
  servers: DiLoCoServer[];
  loading: boolean;
  error: unknown;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAfterRegistryChange: () => void;
}

function ServersList({
  servers,
  loading,
  error,
  selectedId,
  onSelect,
  onAfterRegistryChange,
}: ServersListProps) {
  const [showAdd, setShowAdd] = useState(false);
  const local = servers.filter((s) => s.source === "local");
  const registered = servers.filter((s) => s.source === "registered");

  return (
    <div
      style={{
        border: "1px solid var(--border, #444)",
        borderRadius: 6,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: 8,
          borderBottom: "1px solid var(--border, #444)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <strong>Servers</strong>
        <span style={{ flex: 1 }} />
        <button onClick={() => setShowAdd((v) => !v)}>
          {showAdd ? "Cancel" : "+ Add external…"}
        </button>
      </div>

      {showAdd && (
        <AddExternalServerForm
          onClose={() => setShowAdd(false)}
          onAdded={() => {
            setShowAdd(false);
            onAfterRegistryChange();
          }}
        />
      )}

      <div style={{ overflow: "auto", flex: 1, minHeight: 0 }}>
        {loading && <div className="muted" style={{ padding: 8 }}>Loading…</div>}
        {!!error && (
          <div className="muted" style={{ padding: 8, color: "tomato" }}>
            {(error as Error)?.message ?? String(error)}
          </div>
        )}
        {!loading && servers.length === 0 && (
          <div className="muted" style={{ padding: 8 }}>
            No servers known.
          </div>
        )}

        {local.length > 0 && (
          <ServerGroup
            heading="Local"
            entries={local}
            selectedId={selectedId}
            onSelect={onSelect}
            onAfterRegistryChange={onAfterRegistryChange}
          />
        )}
        {registered.length > 0 && (
          <ServerGroup
            heading="Registered"
            entries={registered}
            selectedId={selectedId}
            onSelect={onSelect}
            onAfterRegistryChange={onAfterRegistryChange}
          />
        )}
      </div>
    </div>
  );
}

function ServerGroup({
  heading,
  entries,
  selectedId,
  onSelect,
  onAfterRegistryChange,
}: {
  heading: string;
  entries: DiLoCoServer[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAfterRegistryChange: () => void;
}) {
  return (
    <div>
      <div
        className="muted"
        style={{
          fontSize: "smaller",
          padding: "4px 8px",
          borderBottom: "1px solid var(--border, #333)",
        }}
      >
        {heading}
      </div>
      <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {entries.map((s) => (
          <ServerRow
            key={s.id}
            server={s}
            selected={s.id === selectedId}
            onSelect={() => onSelect(s.id)}
            onAfterRegistryChange={onAfterRegistryChange}
          />
        ))}
      </ul>
    </div>
  );
}

function ServerRow({
  server,
  selected,
  onSelect,
  onAfterRegistryChange,
}: {
  server: DiLoCoServer;
  selected: boolean;
  onSelect: () => void;
  onAfterRegistryChange: () => void;
}) {
  const qc = useQueryClient();
  const removeMutation = useMutation({
    mutationFn: () => {
      // server.id is "registered:<hex>" — pull off the registry's hex id.
      const regId = server.id.startsWith("registered:")
        ? server.id.slice("registered:".length)
        : server.id;
      return api.deleteDiLoCoRegistryEntry(regId);
    },
    onSuccess: () => {
      onAfterRegistryChange();
      qc.invalidateQueries({ queryKey: ["diloco"] });
    },
  });

  return (
    <li
      onClick={onSelect}
      style={{
        cursor: "pointer",
        padding: "6px 8px",
        borderBottom: "1px solid var(--border, #2a2a2a)",
        background: selected ? "var(--row-selected, #1f2a3a)" : undefined,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {server.source === "local" && (
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              background: server.alive ? "#3a3" : "#a33",
              display: "inline-block",
            }}
            title={server.alive ? "running" : "not running"}
          />
        )}
        <strong style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
          {server.label}
        </strong>
        {server.source === "registered" && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (window.confirm(`Remove ${server.label}?`)) {
                removeMutation.mutate();
              }
            }}
            title="Remove from registry"
          >
            ✕
          </button>
        )}
      </div>
      <div className="muted" style={{ fontSize: "smaller" }}>
        {server.base_url}
      </div>
    </li>
  );
}

function AddExternalServerForm({
  onClose,
  onAdded,
}: {
  onClose: () => void;
  onAdded: () => void;
}) {
  const [label, setLabel] = useState("");
  const [baseUrl, setBaseUrl] = useState("http://");
  const addMutation = useMutation({
    mutationFn: () =>
      api.addDiLoCoRegistryEntry({
        label: label.trim() || undefined,
        base_url: baseUrl.trim(),
      }),
    onSuccess: () => {
      onAdded();
    },
  });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        addMutation.mutate();
      }}
      style={{
        padding: 8,
        borderBottom: "1px solid var(--border, #444)",
        display: "grid",
        gap: 6,
      }}
    >
      <label>
        Label
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="e.g. WAN box"
          style={{ width: "100%" }}
        />
      </label>
      <label>
        Base URL
        <input
          type="url"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder="http://host:8512"
          required
          style={{ width: "100%" }}
        />
      </label>
      {addMutation.error && (
        <div className="muted" style={{ color: "tomato" }}>
          {String((addMutation.error as Error).message)}
        </div>
      )}
      <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
        <button type="button" onClick={onClose}>
          Cancel
        </button>
        <button
          type="submit"
          disabled={addMutation.isPending || baseUrl.trim() === ""}
        >
          Add
        </button>
      </div>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Server detail (right pane)
// ---------------------------------------------------------------------------

function ServerDetail({
  server,
  status,
  info,
  statusLoading,
  statusError,
}: {
  server: DiLoCoServer;
  status: DiLoCoStatus | null;
  info: DiLoCoInfo | null;
  statusLoading: boolean;
  statusError: unknown;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 4 }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <strong style={{ fontSize: "larger" }}>{server.label}</strong>
        <span className="muted">{server.base_url}</span>
        <span style={{ flex: 1 }} />
        <a
          href={`${server.base_url}/dashboard`}
          target="_blank"
          rel="noreferrer noopener"
          title="Open the DiLoCo server's built-in dashboard in a new tab"
        >
          Open built-in dashboard ↗
        </a>
      </header>

      {!!statusError && (
        <div className="muted" style={{ color: "tomato" }}>
          {(statusError as Error)?.message ?? String(statusError)}
        </div>
      )}
      {statusLoading && !status && (
        <div className="muted">Loading status…</div>
      )}

      {status && <StatusOverview status={status} info={info} />}
      {status && <WorkersTable status={status} />}
      {status && <ServerMetrics status={status} info={info} />}
    </div>
  );
}

function Field({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div>
      <div className="muted" style={{ fontSize: "smaller" }}>
        {label}
      </div>
      <div>{value}</div>
    </div>
  );
}

function StatusOverview({
  status,
  info,
}: {
  status: DiLoCoStatus;
  info: DiLoCoInfo | null;
}) {
  return (
    <section
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
        gap: 12,
        border: "1px solid var(--border, #444)",
        borderRadius: 6,
        padding: 12,
      }}
    >
      <Field label="Status" value={status.status ?? "—"} />
      <Field label="Mode" value={status.mode ?? "—"} />
      <Field label="Sync round" value={status.sync_round ?? 0} />
      <Field
        label="Workers"
        value={`${status.num_registered ?? 0}/${status.num_workers ?? "?"}`}
      />
      <Field label="Uptime" value={formatUptime(status.uptime_seconds)} />
      {info?.num_parameters !== undefined && (
        <Field
          label="Parameters"
          value={info.num_parameters.toLocaleString()}
        />
      )}
      {status.model_size_mb !== undefined && (
        <Field
          label="Model size"
          value={`${status.model_size_mb.toFixed(1)} MB`}
        />
      )}
    </section>
  );
}

function WorkersTable({ status }: { status: DiLoCoStatus }) {
  const workers = status.workers ?? {};
  const ids = Object.keys(workers);
  if (ids.length === 0) {
    return (
      <section
        style={{
          border: "1px solid var(--border, #444)",
          borderRadius: 6,
          padding: 12,
        }}
      >
        <div className="muted">No workers registered.</div>
      </section>
    );
  }
  return (
    <section
      style={{
        border: "1px solid var(--border, #444)",
        borderRadius: 6,
        overflow: "auto",
      }}
    >
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ textAlign: "left" }}>
            <th style={{ padding: "6px 8px" }}>Worker</th>
            <th style={{ padding: "6px 8px" }}>Host</th>
            <th style={{ padding: "6px 8px" }}>Sync round</th>
            <th style={{ padding: "6px 8px" }}>Steps/s</th>
            <th style={{ padding: "6px 8px" }}>Last heartbeat</th>
          </tr>
        </thead>
        <tbody>
          {ids.map((wid) => {
            const w = workers[wid];
            return (
              <tr
                key={wid}
                style={{ borderTop: "1px solid var(--border, #2a2a2a)" }}
              >
                <td style={{ padding: "6px 8px" }}>{wid}</td>
                <td style={{ padding: "6px 8px" }}>{w.hostname ?? "—"}</td>
                <td style={{ padding: "6px 8px" }}>{w.sync_round ?? 0}</td>
                <td style={{ padding: "6px 8px" }}>
                  {w.steps_per_second !== undefined
                    ? w.steps_per_second.toFixed(2)
                    : "—"}
                </td>
                <td style={{ padding: "6px 8px" }}>
                  {relativeAge(w.last_heartbeat)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function ServerMetrics({
  status,
  info,
}: {
  status: DiLoCoStatus;
  info: DiLoCoInfo | null;
}) {
  const rows: Array<[string, React.ReactNode]> = [];
  if (status.outer_lr !== undefined) rows.push(["Outer LR", status.outer_lr]);
  if (status.outer_momentum !== undefined)
    rows.push(["Outer momentum", status.outer_momentum]);
  if (status.mode === "async") {
    if (status.dn_buffer_size !== undefined)
      rows.push([
        "DN buffer",
        `${status.dn_buffered ?? 0}/${status.dn_buffer_size}`,
      ]);
    if (status.dylu_enabled)
      rows.push(["DyLU base sync_every", status.dylu_base_sync_every ?? "—"]);
    if (status.total_submissions !== undefined)
      rows.push(["Total submissions", status.total_submissions]);
  }
  if (status.heartbeat_timeout !== undefined)
    rows.push(["Heartbeat timeout", `${status.heartbeat_timeout}s`]);
  if (status.min_workers !== undefined)
    rows.push(["min_workers", status.min_workers]);
  if (status.total_worker_deaths !== undefined && status.total_worker_deaths > 0)
    rows.push(["Worker deaths", status.total_worker_deaths]);
  if (status.fragment_submissions)
    rows.push(["Fragment submissions", status.fragment_submissions]);
  if (info?.output_dir) rows.push(["output_dir", info.output_dir]);
  if (status.save_dir && status.save_dir !== info?.output_dir)
    rows.push(["save_dir", status.save_dir]);

  if (rows.length === 0) return null;
  return (
    <section
      style={{
        border: "1px solid var(--border, #444)",
        borderRadius: 6,
        padding: 12,
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: 12,
      }}
    >
      {rows.map(([k, v]) => (
        <Field key={k} label={k} value={v} />
      ))}
    </section>
  );
}
