import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  api,
  DiLoCoInfo,
  DiLoCoQueueSummary,
  DiLoCoServer,
  DiLoCoStatus,
} from "../api";
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
  // Per-server list of work queues (one entry per active
  // ``(dataset_id, shuffle_seed)`` queue). Polled on the same cadence
  // as /status so the heatmaps update in lock-step with sync rounds.
  const queuesQuery = useQuery({
    queryKey: ["diloco", "work-queues", selected?.base_url],
    queryFn: () => api.diLoCoWorkQueues(selected!.base_url),
    enabled: !!selected,
    refetchInterval: state.refreshSeconds * 1000,
    refetchIntervalInBackground: false,
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
              queues={queuesQuery.data ?? null}
              refreshSeconds={state.refreshSeconds}
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
  queues,
  refreshSeconds,
}: {
  server: DiLoCoServer;
  status: DiLoCoStatus | null;
  info: DiLoCoInfo | null;
  statusLoading: boolean;
  statusError: unknown;
  queues: DiLoCoQueueSummary[] | null;
  refreshSeconds: number;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 4 }}>
      <DashboardHeader server={server} status={status} info={info} />

      {!!statusError && (
        <div
          role="alert"
          style={{
            background: "#2d1520",
            border: "1px solid tomato",
            borderRadius: 6,
            padding: "8px 12px",
            color: "tomato",
            fontSize: "smaller",
          }}
        >
          {(statusError as Error)?.message ?? String(statusError)}
        </div>
      )}
      {statusLoading && !status && (
        <div className="muted">Loading status…</div>
      )}

      {status && (
        <WorkersTable baseUrl={server.base_url} status={status} />
      )}
      {status && <ServerMetrics status={status} />}
      {status && (
        <ControlPanel
          baseUrl={server.base_url}
          status={status}
          info={info}
        />
      )}
      {queues && queues.length > 0 && (
        <WorkQueuesSection
          baseUrl={server.base_url}
          queues={queues}
          refreshSeconds={refreshSeconds}
        />
      )}
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

/** Compact summary strip — mirrors the server's native dashboard
 *  header. Mode badge, sync round, uptime, parameter count + size. */
function DashboardHeader({
  server,
  status,
  info,
}: {
  server: DiLoCoServer;
  status: DiLoCoStatus | null;
  info: DiLoCoInfo | null;
}) {
  const params = status?.model_params ?? info?.num_parameters;
  const sizeMb = status?.model_size_mb;
  const mode = status?.mode ?? "—";
  const modeColor =
    mode === "async"
      ? { bg: "#3a2a1a", fg: "#ff9e64" }
      : { bg: "#1a3a5c", fg: "#7aa2f7" };
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        flexWrap: "wrap",
        padding: "10px 14px",
        background: "var(--bg-surface, #24283b)",
        border: "1px solid var(--border, #3b4261)",
        borderRadius: 6,
      }}
    >
      <strong style={{ fontSize: 16 }}>{server.label}</strong>
      {status && (
        <span
          style={{
            display: "inline-block",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 600,
            textTransform: "uppercase",
            background: modeColor.bg,
            color: modeColor.fg,
          }}
        >
          {mode}
        </span>
      )}
      {status && (
        <span className="muted">
          Round <b style={{ color: "var(--text, inherit)" }}>{status.sync_round ?? 0}</b>
        </span>
      )}
      <span className="muted">
        {formatUptime(status?.uptime_seconds)}
      </span>
      {params !== undefined && (
        <span className="muted">
          {formatParams(params)}
          {sizeMb !== undefined && ` (${sizeMb.toFixed(1)} MB)`}
        </span>
      )}
      <span className="muted" style={{ marginLeft: "auto", fontSize: 11 }}>
        {server.base_url}
      </span>
    </header>
  );
}

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function workerHealthColor(lastHeartbeat: number | undefined): string {
  if (!lastHeartbeat) return "#f7768e"; // red
  const ago = Date.now() / 1000 - lastHeartbeat;
  if (ago < 60) return "#9ece6a"; // green
  if (ago < 120) return "#e0af68"; // yellow
  return "#f7768e";
}

function truncId(id: string): string {
  return id.length > 20 ? `${id.slice(0, 17)}…` : id;
}

function WorkersTable({
  baseUrl,
  status,
}: {
  baseUrl: string;
  status: DiLoCoStatus;
}) {
  const queryClient = useQueryClient();
  const kickMutation = useMutation({
    mutationFn: (workerId: string) =>
      api.diLoCoServerControl(baseUrl, "kick_worker", {
        worker_id: workerId,
      }),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["diloco", "status", baseUrl] });
    },
  });

  const workers = status.workers ?? {};
  const ids = Object.keys(workers);

  const headerRow = (
    <header
      style={{
        padding: "8px 14px",
        background: "var(--bg-surface, #24283b)",
        borderBottom: "1px solid var(--border, #3b4261)",
        fontWeight: 600,
      }}
    >
      Workers{" "}
      <span className="muted" style={{ fontWeight: 400, fontSize: "smaller" }}>
        ({status.num_registered ?? 0}/{status.num_workers ?? "?"})
      </span>
    </header>
  );

  return (
    <section
      style={{
        border: "1px solid var(--border, #3b4261)",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      {headerRow}
      {ids.length === 0 ? (
        <div className="muted" style={{ padding: "16px 14px" }}>
          No workers connected
        </div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left" }}>
              <th style={{ padding: "6px 8px", width: 28 }}></th>
              <th style={{ padding: "6px 8px" }}>ID</th>
              <th style={{ padding: "6px 8px" }}>Hostname</th>
              <th style={{ padding: "6px 8px" }}>Round</th>
              <th style={{ padding: "6px 8px" }}>Steps/s</th>
              <th style={{ padding: "6px 8px" }}>Last heartbeat</th>
              <th style={{ padding: "6px 8px" }}></th>
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
                  <td style={{ padding: "6px 8px" }}>
                    <span
                      title="Health (green <60s heartbeat, yellow <120s, red older)"
                      style={{
                        display: "inline-block",
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        background: workerHealthColor(w.last_heartbeat),
                      }}
                    />
                  </td>
                  <td
                    style={{ padding: "6px 8px", fontFamily: "monospace" }}
                    title={wid}
                  >
                    {truncId(wid)}
                  </td>
                  <td style={{ padding: "6px 8px" }}>{w.hostname ?? "—"}</td>
                  <td style={{ padding: "6px 8px" }}>{w.sync_round ?? 0}</td>
                  <td style={{ padding: "6px 8px" }}>
                    {w.steps_per_second && w.steps_per_second > 0
                      ? w.steps_per_second.toFixed(2)
                      : "—"}
                  </td>
                  <td style={{ padding: "6px 8px" }}>
                    {relativeAge(w.last_heartbeat)}
                  </td>
                  <td style={{ padding: "6px 8px", textAlign: "right" }}>
                    <button
                      className="tiny"
                      onClick={() => {
                        if (window.confirm(`Kick worker ${wid}?`)) {
                          kickMutation.mutate(wid);
                        }
                      }}
                      disabled={kickMutation.isPending}
                      title="Force-evict this worker from the server"
                    >
                      Kick
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}

function ServerMetrics({ status }: { status: DiLoCoStatus }) {
  // Always-shown metric grid. Mirrors the server dashboard's "Server
  // Metrics" panel; sync vs async paths diverge after the first row.
  const baseMetrics: Array<[string, React.ReactNode]> = [];
  baseMetrics.push(["Outer LR", status.outer_lr ?? "—"]);
  baseMetrics.push(["Outer momentum", status.outer_momentum ?? "—"]);
  baseMetrics.push(["Worker deaths", status.total_worker_deaths ?? 0]);
  baseMetrics.push([
    "HB timeout",
    status.heartbeat_timeout !== undefined
      ? `${status.heartbeat_timeout}s`
      : "—",
  ]);

  const pendingCount = status.pending_submissions?.length ?? 0;
  const expectedCount = Math.max(status.num_workers ?? 1, 1);
  const pendingPct = Math.min(100, (pendingCount / expectedCount) * 100);

  return (
    <section
      style={{
        border: "1px solid var(--border, #3b4261)",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      <header
        style={{
          padding: "8px 14px",
          background: "var(--bg-surface, #24283b)",
          borderBottom: "1px solid var(--border, #3b4261)",
          fontWeight: 600,
        }}
      >
        Server metrics
      </header>
      <div style={{ padding: 14 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
            gap: 12,
          }}
        >
          {baseMetrics.map(([k, v]) => (
            <Field key={k} label={k} value={v} />
          ))}
        </div>

        {status.mode === "sync" && status.pending_submissions && (
          <div style={{ marginTop: 12 }}>
            <div className="muted" style={{ fontSize: "smaller" }}>
              Pending submissions ({pendingCount}/{expectedCount})
            </div>
            <div
              style={{
                marginTop: 4,
                background: "var(--bg, #1a1b26)",
                border: "1px solid var(--border, #3b4261)",
                borderRadius: 4,
                height: 10,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${pendingPct}%`,
                  height: "100%",
                  background: "#7aa2f7",
                  transition: "width 200ms ease",
                }}
              />
            </div>
          </div>
        )}

        {status.mode === "async" && (
          <div
            style={{
              marginTop: 12,
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              gap: 12,
            }}
          >
            <Field
              label="Total submissions"
              value={status.total_submissions ?? 0}
            />
            <Field
              label="DN buffer"
              value={
                status.dn_buffer_size && status.dn_buffer_size > 0
                  ? `${status.dn_buffered ?? 0}/${status.dn_buffer_size}`
                  : "off"
              }
            />
            <Field
              label="DyLU"
              value={
                status.dylu_enabled
                  ? `on (H=${status.dylu_base_sync_every ?? "?"})`
                  : "off"
              }
            />
          </div>
        )}

        {!!status.fragment_submissions && (
          <div style={{ marginTop: 12 }}>
            <Field
              label="Fragment submissions"
              value={status.fragment_submissions}
            />
          </div>
        )}
      </div>
    </section>
  );
}

/** Save / shutdown / optimizer / worker-count controls. Mirrors the
 *  server dashboard's "Control" panel. Each row goes through
 *  ``api.diLoCoServerControl`` which proxies to the upstream server's
 *  ``/control/{action}`` endpoint. */
function ControlPanel({
  baseUrl,
  status,
  info,
}: {
  baseUrl: string;
  status: DiLoCoStatus;
  info: DiLoCoInfo | null;
}) {
  const queryClient = useQueryClient();

  const [confirmShutdown, setConfirmShutdown] = useState(false);
  const [formLr, setFormLr] = useState<string>("");
  const [formMomentum, setFormMomentum] = useState<string>("");
  const [formNumWorkers, setFormNumWorkers] = useState<string>("");
  const [actionMsg, setActionMsg] = useState<
    { ok: boolean; text: string } | null
  >(null);

  // Seed form fields from /status the first time they land. Operator
  // edits don't get clobbered: each ref tracks "the value we last
  // seeded" and only re-seeds if the field still matches.
  const lrSeed = useMemo(
    () =>
      status.outer_lr !== undefined ? String(status.outer_lr) : "",
    [status.outer_lr],
  );
  const momSeed = useMemo(
    () =>
      status.outer_momentum !== undefined
        ? String(status.outer_momentum)
        : "",
    [status.outer_momentum],
  );
  const numWorkersSeed = useMemo(
    () =>
      status.num_workers !== undefined ? String(status.num_workers) : "",
    [status.num_workers],
  );
  useEffect(() => {
    if (formLr === "" && lrSeed !== "") setFormLr(lrSeed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lrSeed]);
  useEffect(() => {
    if (formMomentum === "" && momSeed !== "") setFormMomentum(momSeed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [momSeed]);
  useEffect(() => {
    if (formNumWorkers === "" && numWorkersSeed !== "")
      setFormNumWorkers(numWorkersSeed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numWorkersSeed]);

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["diloco", "status", baseUrl] });
    queryClient.invalidateQueries({ queryKey: ["diloco", "info", baseUrl] });
    queryClient.invalidateQueries({
      queryKey: ["diloco", "servers"],
    });
  };

  const controlMutation = useMutation({
    mutationFn: ({
      action,
      body,
    }: {
      action: string;
      body?: Record<string, unknown>;
    }) => api.diLoCoServerControl(baseUrl, action, body ?? {}),
    onSuccess: (_data, vars) => {
      setActionMsg({ ok: true, text: `${vars.action.replace(/_/g, " ")}: OK` });
      invalidate();
      window.setTimeout(() => setActionMsg(null), 4000);
    },
    onError: (err: unknown, vars) => {
      setActionMsg({
        ok: false,
        text: `${vars.action}: ${(err as Error)?.message ?? String(err)}`,
      });
      window.setTimeout(() => setActionMsg(null), 6000);
    },
  });

  const saveDir = status.save_dir ?? info?.output_dir ?? null;

  return (
    <section
      style={{
        border: "1px solid var(--border, #3b4261)",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      <header
        style={{
          padding: "8px 14px",
          background: "var(--bg-surface, #24283b)",
          borderBottom: "1px solid var(--border, #3b4261)",
          fontWeight: 600,
        }}
      >
        Control
      </header>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
        {actionMsg && (
          <div
            role="status"
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              fontSize: "smaller",
              color: actionMsg.ok ? "#9ece6a" : "tomato",
              border: `1px solid ${actionMsg.ok ? "#9ece6a" : "tomato"}`,
            }}
          >
            {actionMsg.text}
          </div>
        )}

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 12,
          }}
        >
          {/* Save Checkpoint */}
          <div
            style={{
              border: "1px solid var(--border, #3b4261)",
              borderRadius: 4,
              padding: 10,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ fontWeight: 600 }}>Save state</div>
            <button
              onClick={() =>
                controlMutation.mutate({ action: "save_state" })
              }
              disabled={controlMutation.isPending || !saveDir}
              title={
                saveDir
                  ? `Save checkpoint to ${saveDir}`
                  : "No save_dir configured on the server"
              }
            >
              Save checkpoint
            </button>
            {!saveDir && (
              <div className="muted" style={{ fontSize: 11 }}>
                No save_dir configured
              </div>
            )}
            {saveDir && (
              <div className="muted" style={{ fontSize: 11, wordBreak: "break-all" }}>
                {saveDir}
              </div>
            )}
          </div>

          {/* Shutdown */}
          <div
            style={{
              border: "1px solid var(--border, #3b4261)",
              borderRadius: 4,
              padding: 10,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ fontWeight: 600 }}>Shutdown</div>
            <button
              onClick={() => setConfirmShutdown(true)}
              disabled={controlMutation.isPending}
              style={{ background: "#3a2a2a", color: "#f7768e" }}
              title="Stop the DiLoCo server. All connected workers will lose sync."
            >
              Shutdown server
            </button>
          </div>

          {/* Outer optimizer */}
          <div
            style={{
              border: "1px solid var(--border, #3b4261)",
              borderRadius: 4,
              padding: 10,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ fontWeight: 600 }}>Optimizer</div>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span className="muted" style={{ minWidth: 64 }}>LR</span>
              <input
                type="number"
                step="any"
                value={formLr}
                onChange={(e) => setFormLr(e.target.value)}
                style={{ flex: 1, minWidth: 0 }}
              />
            </label>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span className="muted" style={{ minWidth: 64 }}>Momentum</span>
              <input
                type="number"
                step="any"
                value={formMomentum}
                onChange={(e) => setFormMomentum(e.target.value)}
                style={{ flex: 1, minWidth: 0 }}
              />
            </label>
            <button
              onClick={() => {
                const body: Record<string, unknown> = {};
                if (formLr.trim()) body.lr = Number(formLr);
                if (formMomentum.trim()) body.momentum = Number(formMomentum);
                if (Object.keys(body).length === 0) return;
                controlMutation.mutate({ action: "update_optimizer", body });
              }}
              disabled={
                controlMutation.isPending ||
                (formLr.trim() === "" && formMomentum.trim() === "")
              }
            >
              Apply
            </button>
          </div>

          {/* Expected worker count */}
          <div
            style={{
              border: "1px solid var(--border, #3b4261)",
              borderRadius: 4,
              padding: 10,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ fontWeight: 600 }}>Workers</div>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span className="muted" style={{ minWidth: 64 }}>Expected</span>
              <input
                type="number"
                min={status.min_workers ?? 1}
                step={1}
                value={formNumWorkers}
                onChange={(e) => setFormNumWorkers(e.target.value)}
                style={{ flex: 1, minWidth: 0 }}
              />
            </label>
            <button
              onClick={() => {
                const n = Number(formNumWorkers);
                if (!Number.isFinite(n) || n < 1) return;
                controlMutation.mutate({
                  action: "update_num_workers",
                  body: { num_workers: Math.floor(n) },
                });
              }}
              disabled={
                controlMutation.isPending || formNumWorkers.trim() === ""
              }
            >
              Apply
            </button>
            {status.min_workers !== undefined && (
              <div className="muted" style={{ fontSize: 11 }}>
                min_workers = {status.min_workers}
              </div>
            )}
          </div>
        </div>

        {info?.output_dir && (
          <div className="muted" style={{ fontSize: 11, wordBreak: "break-all" }}>
            output_dir: {info.output_dir}
          </div>
        )}
      </div>

      {confirmShutdown && (
        <div
          role="dialog"
          onClick={(e) => {
            if (e.target === e.currentTarget) setConfirmShutdown(false);
          }}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.5)",
            zIndex: 100,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              background: "var(--bg-surface, #24283b)",
              border: "1px solid var(--border, #3b4261)",
              borderRadius: 6,
              padding: 16,
              maxWidth: 420,
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <p style={{ margin: 0 }}>
              Shut down the DiLoCo server? Any connected workers will fail
              to sync; the next sync attempt will surface as a connection
              error in their TTY pane.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setConfirmShutdown(false)}>Cancel</button>
              <button
                style={{ background: "#3a2a2a", color: "#f7768e" }}
                onClick={() => {
                  setConfirmShutdown(false);
                  controlMutation.mutate({ action: "shutdown" });
                }}
              >
                Confirm shutdown
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Work-unit dispatch — per-queue heatmaps
// ---------------------------------------------------------------------------

function WorkQueuesSection({
  baseUrl,
  queues,
  refreshSeconds,
}: {
  baseUrl: string;
  queues: DiLoCoQueueSummary[];
  refreshSeconds: number;
}) {
  return (
    <section
      style={{
        border: "1px solid var(--border, #444)",
        borderRadius: 6,
        padding: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <header style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <strong>Work-unit dispatch</strong>
        <span className="muted" style={{ fontSize: "smaller" }}>
          one queue per (dataset_id, shuffle_seed)
        </span>
      </header>
      {queues.map((q) => (
        <QueueHeatmap
          key={`${q.dataset_id}|${q.shuffle_seed}`}
          baseUrl={baseUrl}
          summary={q}
          refreshSeconds={refreshSeconds}
        />
      ))}
    </section>
  );
}

function QueueHeatmap({
  baseUrl,
  summary,
  refreshSeconds,
}: {
  baseUrl: string;
  summary: DiLoCoQueueSummary;
  refreshSeconds: number;
}) {
  // Fetch the detail (bitmaps + per-worker counters) per queue. Each
  // bitmap is K bits → at K=1024 that's 128 bytes, base64 ≈ 172
  // bytes. Polling at the panel's cadence is cheap.
  const detailQuery = useQuery({
    queryKey: [
      "diloco",
      "work-queue",
      baseUrl,
      summary.dataset_id,
      summary.shuffle_seed,
    ],
    queryFn: () =>
      api.diLoCoWorkQueue(baseUrl, summary.dataset_id, summary.shuffle_seed),
    refetchInterval: refreshSeconds * 1000,
    refetchIntervalInBackground: false,
  });

  const detail = detailQuery.data;
  const issuedBytes = useMemo(
    () => (detail ? decodeBase64(detail.issued_bitmap_b64) : null),
    [detail],
  );
  const completedBytes = useMemo(
    () => (detail ? decodeBase64(detail.completed_bitmap_b64) : null),
    [detail],
  );

  // K=1024 → 32×32 grid is the natural shape; for other K pick a
  // squarish grid: ceil(sqrt(K)) cols.
  const cols = Math.ceil(Math.sqrt(summary.total_units));

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        padding: 8,
        background: "var(--row-alt, #1a1a1a)",
        borderRadius: 4,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <code style={{ fontSize: "smaller" }}>
          {summary.dataset_id}@{summary.shuffle_seed}
        </code>
        <span style={{ flex: 1 }} />
        <span className="muted" style={{ fontSize: "smaller" }}>
          {summary.issued_count}/{summary.total_units} issued
          {summary.completed_count > 0 &&
            ` · ${summary.completed_count} confirmed`}
          {" · "}
          {summary.hint.length.toLocaleString()} rows
        </span>
      </div>

      {issuedBytes && completedBytes ? (
        <div
          aria-label="work unit heatmap"
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${cols}, 1fr)`,
            gap: 1,
            width: "100%",
            maxWidth: 480,
          }}
        >
          {Array.from({ length: summary.total_units }, (_, i) => {
            const issued = bitGet(issuedBytes, i);
            const completed = bitGet(completedBytes, i);
            // Three states: available (transparent) / issued
            // (orange) / issued+confirmed-complete (green).
            const bg = completed
              ? "#2d6a4f"
              : issued
                ? "#b8741b"
                : "rgba(255,255,255,0.06)";
            return (
              <div
                key={i}
                title={`unit ${i}: ${
                  completed ? "completed" : issued ? "issued" : "available"
                }`}
                style={{
                  width: "100%",
                  aspectRatio: "1 / 1",
                  background: bg,
                  borderRadius: 1,
                }}
              />
            );
          })}
        </div>
      ) : (
        <div className="muted" style={{ fontSize: "smaller" }}>
          Loading heatmap…
        </div>
      )}

      {detail && Object.keys(detail.by_worker).length > 0 && (
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "smaller",
            marginTop: 4,
          }}
        >
          <thead>
            <tr style={{ textAlign: "left" }}>
              <th style={{ padding: "2px 6px" }}>Worker</th>
              <th style={{ padding: "2px 6px" }}>Issued</th>
              <th style={{ padding: "2px 6px" }}>Completed</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(detail.by_worker)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([wid, c]) => (
                <tr
                  key={wid}
                  style={{ borderTop: "1px solid var(--border, #2a2a2a)" }}
                >
                  <td style={{ padding: "2px 6px" }}>{wid}</td>
                  <td style={{ padding: "2px 6px" }}>{c.units_issued}</td>
                  <td style={{ padding: "2px 6px" }}>{c.units_completed}</td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function decodeBase64(s: string): Uint8Array {
  // atob → binary string → byte array. Adequate for K=1024 (~128 bytes);
  // for very large K we'd want Uint8Array.fromBase64 or similar.
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function bitGet(bm: Uint8Array, i: number): boolean {
  return ((bm[i >> 3] >> (i & 7)) & 1) === 1;
}
