import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  api,
  DiLoCoInfo,
  DiLoCoQueueSummary,
  DiLoCoServer,
  DiLoCoStatus,
  DiLoCoWorkerStatus,
  Job,
  ServiceStatus,
} from "../api";
import { persistGet, persistSet } from "../persist";
import { DiLoCoServerModal } from "./DiLoCoServerModal";
import LossChart from "./LossChart";
import { ModalBackdrop } from "./ModalBackdrop";
import { ServicesPanel } from "./ServicesPanel";
import { TensorBoardModal } from "./TensorBoardModal";

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

export function DiLoCoPanel({
  pendingServerPick,
  onServerPickConsumed,
  onEditService,
}: {
  pendingServerPick?: { queueId: string; key: number } | null;
  onServerPickConsumed?: () => void;
  /** Open the DiLoCo service in edit mode (routed to App's editingService,
   *  which renders the DiLoCoServerModal edit modal). */
  onEditService?: (s: ServiceStatus) => void;
} = {}) {
  const [state, setState] = useState<PanelState>(loadState);
  // Launch-a-local-server modal (same modal as Services → DiLoCo) and the
  // TensorBoard launcher for the selected server's runs/ dir.
  const [launchOpen, setLaunchOpen] = useState(false);
  const [tbOpen, setTbOpen] = useState(false);
  useEffect(() => {
    persistSet(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  // Server list — refresh every 5s. The unified list spans every
  // DiLoCo server this node knows about: locally-spawned, user-
  // registered, and cluster-attested via the master-aggregated
  // inventory at ``/api/cluster/diloco_servers``.
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

  // "Open in DiLoCo" from a Job card: select the matching local
  // server (id = "local:<queue_id>") then signal consumption so the
  // auto-pick logic doesn't fight a future re-fire of the same pick.
  // The ``key`` in the pending object lets the effect re-fire when
  // the operator clicks Open again for the same queue.
  useEffect(() => {
    if (!pendingServerPick) return;
    const target = `local:${pendingServerPick.queueId}`;
    if (servers.some((s) => s.id === target)) {
      setState((s) => ({ ...s, selectedId: target }));
      onServerPickConsumed?.();
    }
    // Don't consume when the server hasn't shown up yet — the
    // job_records refresh cycle will give us another shot.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingServerPick, servers]);

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
  // Forgather-side jobs list. Used to map each DiLoCo worker_id back
  // to its job_id. The mapping primarily flows through
  // ``job_params.diloco.worker_id`` (the queue route stamps a
  // memorable two-word default when the operator doesn't supply one),
  // with ``output_dir`` as the secondary key for resumes / legacy
  // records that lack the job_params field. With a matched Job we can
  // fetch per-worker training status + drive the per-worker control
  // protocol (Save / Save & Stop / Abort).
  const jobsQuery = useQuery({
    queryKey: ["jobs", "for-diloco"],
    queryFn: () => api.listJobs(false),
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
          {selected && (
            <button
              onClick={() => setTbOpen(true)}
              title={
                statusQuery.data?.save_dir
                  ? `Open TensorBoard on ${statusQuery.data.save_dir}/runs`
                  : "Open TensorBoard (pick a logdir)"
              }
            >
              📊 TensorBoard
            </button>
          )}
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
          onStartLocal={() => setLaunchOpen(true)}
          onEditService={onEditService}
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
              jobs={jobsQuery.data ?? null}
              refreshSeconds={state.refreshSeconds}
            />
          )}
        </div>
      </div>

      {launchOpen && (
        <DiLoCoServerModal
          onClose={() => setLaunchOpen(false)}
          onSubmitted={(queueId) => {
            setLaunchOpen(false);
            serversQuery.refetch();
            // Select the just-launched local server (id is local:<queue_id>).
            setState((s) => ({ ...s, selectedId: `local:${queueId}` }));
          }}
        />
      )}

      {tbOpen && selected && (
        <TensorBoardModal
          global
          initialLogdir={
            statusQuery.data?.save_dir
              ? `${statusQuery.data.save_dir.replace(/\/+$/, "")}/runs`
              : ""
          }
          initialWindowTitle={`DiLoCo ${selected.label}`}
          onClose={() => setTbOpen(false)}
        />
      )}
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
  onStartLocal: () => void;
  onEditService?: (s: ServiceStatus) => void;
}

function ServersList({
  servers,
  loading,
  error,
  selectedId,
  onSelect,
  onAfterRegistryChange,
  onStartLocal,
  onEditService,
}: ServersListProps) {
  const [showAdd, setShowAdd] = useState(false);
  // One flat list of every server known to this node. The cluster
  // mental model is uniform — local-spawn, peer-spawn, and user-
  // registered entries are all just "DiLoCo servers in this cluster"
  // from the operator's POV. Per-row badges still mark the origin
  // (where the server lives and how this node learned about it) so
  // operator actions like "remove from registry" can attach to the
  // right rows; nothing surfaces as a separate group.
  const sortedServers = useMemo(() => {
    // Liveness tier only — alphabetical-by-label as the secondary
    // key. Sorting cluster entries last within a tier would re-leak
    // the local/registered/cluster split the unified list is meant
    // to hide; "this node" is conveyed by the per-row chip styling,
    // not by list position. Mirrors ClusterSidebarPanel's pattern
    // (role + reachability, then alphabetical-by-hostname).
    const tier = (s: DiLoCoServer): number => {
      if (s.alive === true || s.healthy === true) return 0;
      if (s.alive === false || s.healthy === false) return 2;
      // Registered entries don't expose a health flag *by design*
      // (the registry stores intent, not liveness) — treat them as
      // neutral rather than down. Gated on source so an as-yet-
      // unprobed cluster entry doesn't get the same pass.
      if (s.source === "registered") return 0;
      return 1;
    };
    return [...servers].sort((a, b) => {
      const r = tier(a) - tier(b);
      if (r !== 0) return r;
      return (a.label || a.base_url || "").localeCompare(
        b.label || b.base_url || "",
      );
    });
  }, [servers]);

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
        <button
          onClick={onStartLocal}
          title="Launch a local DiLoCo server (or save it as a service)"
        >
          + Start local…
        </button>
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
        {/* Defined services — start/stop toggle, edit, delete (same control
            surface as Services → DiLoCo). A running service also appears in
            the live list below; this section is for managing the
            definitions. */}
        <div
          className="muted"
          style={{
            fontSize: "smaller",
            padding: "4px 8px",
            borderBottom: "1px solid var(--border, #333)",
          }}
        >
          Services
        </div>
        <ServicesPanel filterType="diloco" onEditService={onEditService} />

        {/* Divider for the running-server list. "Live" was the
            pre-cluster wording (the list only carried entries we knew
            were alive); the unified list also includes user-registered
            entries with no liveness signal and cluster entries that
            may be down, so "Known servers" matches what's actually
            shown. Hidden in the empty state — the empty-state copy
            below carries the same scope claim. */}
        {(loading || !!error || sortedServers.length > 0) && (
          <div
            className="muted"
            style={{
              fontSize: "smaller",
              padding: "4px 8px",
              borderBottom: "1px solid var(--border, #333)",
            }}
          >
            Known servers
          </div>
        )}
        {loading && <div className="muted" style={{ padding: 8 }}>Loading…</div>}
        {!!error && (
          <div className="muted" style={{ padding: 8, color: "tomato" }}>
            {(error as Error)?.message ?? String(error)}
          </div>
        )}
        {!loading && servers.length === 0 && (
          <div className="muted" style={{ padding: 8 }}>
            No DiLoCo servers known to this cluster.
          </div>
        )}

        {sortedServers.length > 0 && (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {sortedServers.map((s) => (
              <ServerRow
                key={s.id}
                server={s}
                selected={s.id === selectedId}
                onSelect={() => onSelect(s.id)}
                onAfterRegistryChange={onAfterRegistryChange}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

/** Health state condensed across the three origin kinds — local /
 *  registered / cluster — into a single shape the row dot can render.
 *  Falls through to whichever liveness field is populated so the dot
 *  and the sort ranker agree on every row. Returns ``null`` for "no
 *  signal" (the registry stores intent, not liveness). */
function rowHealth(server: DiLoCoServer): boolean | null {
  if (typeof server.healthy === "boolean") return server.healthy;
  if (typeof server.alive === "boolean") return server.alive;
  return null;
}

/** Hostname portion of ``base_url``, or null if it can't be parsed.
 *  Used to label peer-attested rows by hostname rather than by node-id
 *  prefix — consistent with InferenceModelPanel / DatasetsPanel /
 *  ClusterPanel, which all surface ``peer <hostname>`` rather than the
 *  opaque UUID. */
function hostnameFromUrl(url: string | undefined): string | null {
  if (!url) return null;
  try {
    return new URL(url).hostname || null;
  } catch {
    return null;
  }
}

/** Where the server runs and how this node learned about it, as a
 *  compact label suitable for an inline chip. Cluster entries surface
 *  the hostname parsed from ``base_url`` (which the master rewrites
 *  from 0.0.0.0 to the cluster identity's hostname); a peer_node_id
 *  prefix is the last-resort fallback. */
function originLabel(server: DiLoCoServer): string {
  if (server.source === "cluster") {
    const host = hostnameFromUrl(server.base_url);
    if (host) return `peer ${host}`;
    return server.peer_node_id ? `peer ${server.peer_node_id.slice(0, 8)}` : "peer";
  }
  if (server.source === "registered") {
    return "registered";
  }
  return "this node";
}

/** Which ``node-tag-*`` modifier matches an origin label — ``-ok`` for
 *  "this node" (mirrors MultiNodeSubmitPanel's "this node" / "master"
 *  styling); ``-muted`` for everything else. */
function originChipClass(server: DiLoCoServer): string {
  if (server.source === "local") return "node-tag node-tag-ok";
  return "node-tag node-tag-muted";
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

  // Hoist per-row display values so the health dot, its tooltip, and
  // the origin chip all share one source of truth (and ``originLabel``
  // doesn't get called three times per render).
  const origin = originLabel(server);
  const health = rowHealth(server);
  const dotColor =
    health === true ? "#3a3" : health === false ? "#a33" : "#888";
  const dotTitle =
    health === true
      ? `healthy (${origin})`
      : health === false
      ? `down (${origin})`
      : `health unknown (${origin})`;

  return (
    <li
      role="button"
      tabIndex={0}
      aria-current={selected ? "true" : undefined}
      onClick={onSelect}
      onKeyDown={(e) => {
        // Keyboard users select with Enter or Space — matches the
        // browser default for <button> elements which <li role=button>
        // doesn't inherit. Preventing default on Space stops the page
        // from scrolling underneath the focused row.
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      style={{
        cursor: "pointer",
        padding: "6px 8px",
        borderBottom: "1px solid var(--border, #2a2a2a)",
        background: selected ? "var(--row-selected, #1f2a3a)" : undefined,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 4,
            background: dotColor,
            display: "inline-block",
          }}
          title={dotTitle}
        />
        <strong style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
          {server.label}
        </strong>
        <span
          className={originChipClass(server)}
          title={`Origin: ${origin}`}
          style={{ whiteSpace: "nowrap" }}
        >
          {origin}
        </span>
        {server.has_auth_token && (
          <span
            title="Bearer-token auth configured for this server"
            aria-label="bearer auth"
            style={{ fontSize: "smaller" }}
          >
            🔒
          </span>
        )}
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
  const [authToken, setAuthToken] = useState("");
  const [verifyTls, setVerifyTls] = useState(true);
  const addMutation = useMutation({
    mutationFn: () =>
      api.addDiLoCoRegistryEntry({
        label: label.trim() || undefined,
        base_url: baseUrl.trim(),
        auth_token: authToken,
        verify_tls: verifyTls,
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
      <label>
        Bearer token (optional)
        <input
          type="password"
          value={authToken}
          onChange={(e) => setAuthToken(e.target.value)}
          placeholder="Leave blank for --no-auth servers"
          autoComplete="off"
          style={{ width: "100%" }}
        />
      </label>
      <label
        style={{ display: "flex", alignItems: "center", gap: 6 }}
        title="Disable when the upstream cert won't validate (e.g. SSH-tunneled remotes)."
      >
        <input
          type="checkbox"
          checked={verifyTls}
          onChange={(e) => setVerifyTls(e.target.checked)}
        />
        Verify TLS certificate
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
  jobs,
  refreshSeconds,
}: {
  server: DiLoCoServer;
  status: DiLoCoStatus | null;
  info: DiLoCoInfo | null;
  statusLoading: boolean;
  statusError: unknown;
  queues: DiLoCoQueueSummary[] | null;
  jobs: Job[] | null;
  refreshSeconds: number;
}) {
  return (
    // Bounded container keeps wide-monitor layout readable. Left-
    // aligned (no marginInline:auto) so it doesn't shift around as
    // operator sizes the window — content always anchors to the
    // left edge of the pane.
    <div
      style={{
        maxWidth: 1100,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        padding: 4,
      }}
    >
      <DashboardHeader server={server} status={status} info={info} />

      {status && <AggregateStatsCard status={status} />}

      {status && (
        <LossHistoryCard
          baseUrl={server.base_url}
          refreshSeconds={refreshSeconds}
        />
      )}

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
        <WorkersSection
          baseUrl={server.base_url}
          status={status}
          jobs={jobs}
          refreshSeconds={refreshSeconds}
        />
      )}
      {status && (
        // Metrics + Control side-by-side on wide screens, stacked on
        // narrow ones. ``auto-fit`` collapses to one column when the
        // viewport can't hold two minmax(320px,…) cells.
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: 10,
          }}
        >
          <ServerMetrics status={status} />
          <ControlPanel baseUrl={server.base_url} status={status} info={info} />
        </div>
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

function formatTokens(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

/** Unified aggregate training stats collected from every worker (total
 *  tokens/steps/FLOPs, aggregate throughput/MFU/memory, smoothed train/eval
 *  loss). Renders nothing until at least one metric has been reported, so a
 *  fresh server shows no empty card. */
function AggregateStatsCard({ status }: { status: DiLoCoStatus }) {
  const agg = status.aggregate_stats;
  if (!agg) return null;
  const items: Array<[string, React.ReactNode]> = [];
  if (agg.total_tokens) items.push(["Total tokens", formatTokens(agg.total_tokens)]);
  if (agg.total_steps)
    items.push(["Total steps", agg.total_steps.toLocaleString()]);
  if (agg.total_flos) items.push(["Total FLOPs", agg.total_flos.toExponential(2)]);
  if (agg.tok_per_sec)
    items.push([
      "Throughput",
      `${Math.round(agg.tok_per_sec).toLocaleString()} tok/s`,
    ]);
  if (agg.mfu) items.push(["MFU", `${(agg.mfu * 100).toFixed(1)}%`]);
  if (agg.peak_memory)
    items.push(["Peak memory", `${(agg.peak_memory / 1e9).toFixed(2)} GB`]);
  if (agg.grad_norm != null) items.push(["Grad norm", agg.grad_norm.toFixed(3)]);
  if (agg.train_loss != null)
    items.push(["Train loss", agg.train_loss.toFixed(4)]);
  if (agg.eval_loss != null)
    items.push([
      "Eval loss",
      agg.eval_step != null
        ? `${agg.eval_loss.toFixed(4)} @ ${agg.eval_step.toLocaleString()}`
        : agg.eval_loss.toFixed(4),
    ]);
  if (items.length === 0) return null;

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
          display: "flex",
          gap: 8,
          alignItems: "baseline",
        }}
      >
        <span>Training stats</span>
        {agg.num_reporting != null && (
          <span className="muted" style={{ fontWeight: 400, fontSize: "smaller" }}>
            (aggregate of {agg.num_reporting} reporting)
          </span>
        )}
      </header>
      <div style={{ padding: 14 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
            gap: 12,
          }}
        >
          {items.map(([k, v]) => (
            <Field key={k} label={k} value={v} />
          ))}
        </div>
      </div>
    </section>
  );
}

/** Loss curves (train + eval) over training steps, fetched from the server's
 *  aggregate-stats history and plotted with pan/zoom/reset. Renders nothing
 *  until there's at least one loss point, so a run with no loss reported yet
 *  shows no empty chart. */
function LossHistoryCard({
  baseUrl,
  refreshSeconds,
}: {
  baseUrl: string;
  refreshSeconds: number;
}) {
  const histQuery = useQuery({
    queryKey: ["diloco-stats-history", baseUrl],
    queryFn: () => api.diLoCoStatsHistory(baseUrl),
    refetchInterval: refreshSeconds * 1000,
  });
  const records = histQuery.data?.records ?? [];
  const hasLoss = records.some(
    (r) => typeof r.train_loss === "number" || typeof r.eval_loss === "number",
  );
  // Surface an unreachable endpoint (e.g. a DiLoCo server predating
  // /stats_history) instead of silently hiding — otherwise "no chart" is
  // indistinguishable from "no data yet". Stay quiet on the normal
  // no-loss-reported-yet case.
  if (histQuery.isError) {
    return (
      <section
        style={{
          border: "1px solid var(--border, #3b4261)",
          borderRadius: 6,
          padding: "8px 14px",
          fontSize: "smaller",
        }}
        className="muted"
      >
        Loss curves unavailable — the DiLoCo server doesn't expose{" "}
        <code>/stats_history</code> (restart it to enable the loss plot).
      </section>
    );
  }
  if (!hasLoss) return null;

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
          display: "flex",
          gap: 8,
          alignItems: "baseline",
        }}
      >
        <span>Loss curves</span>
        <span className="muted" style={{ fontWeight: 400, fontSize: "smaller" }}>
          train / eval — scroll to zoom, drag to pan, double-click to reset
          {histQuery.data?.downsampled ? " (downsampled)" : ""}
        </span>
      </header>
      <div style={{ padding: 14 }}>
        <LossChart records={records} />
      </div>
    </section>
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
      {/* Sync backend (issue #154), the group's transport topology
       *  (http / shared_memory / collective). Server-declared and the value
       *  workers validate against, so surfacing it here lets the operator
       *  confirm what a server actually came up as — distinct from the mode
       *  badge and from the grpc/http bulk-transport axis. */}
      {info && (
        <span
          title="Sync backend (server-declared)"
          style={{
            display: "inline-block",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 600,
            background: "#2a2440",
            color: "#bb9af7",
          }}
        >
          {info.expected_client_settings?.backend ?? "—"}
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

function workerHealthColor(
  lastHeartbeat: number | undefined,
  heartbeatTimeout: number | undefined,
): string {
  if (!lastHeartbeat) return "#f7768e"; // red
  const ago = Date.now() / 1000 - lastHeartbeat;
  // Derive thresholds from the server's configured heartbeat_timeout
  // so a server set to a non-default cadence (e.g. 30s for fast
  // smoke tests, 600s for cross-WAN) gets accurate health colors.
  // Falls back to the documented default (120s) when the server
  // hasn't advertised it.
  const timeout = heartbeatTimeout && heartbeatTimeout > 0 ? heartbeatTimeout : 120;
  if (ago < timeout * 0.5) return "#9ece6a"; // green: well within budget
  if (ago < timeout) return "#e0af68"; // yellow: trending stale
  return "#f7768e"; // red: server will evict on next sweep
}

function truncId(id: string): string {
  // The common case is ``q_<timestamp>_<8-hex>`` (~25 chars) or its
  // pipeline-group variant ``q_<timestamp>_<8-hex>_pp<N>`` (~28-30
  // chars); both should fit verbatim. The threshold here is a safety
  // net for unusual operator-supplied --diloco-worker-id values that
  // could blow out the layout. Title attr still carries the full id
  // for hover.
  return id.length > 48 ? `${id.slice(0, 45)}…` : id;
}

// Worker control actions, mapped to the server-relayed trainer-control
// command vocabulary. All worker control in this panel goes through the
// DiLoCo server's /control/command relay (proxied by diLoCoServerControl):
// the server queues the command and delivers it on the worker's heartbeat,
// so it works for every registered worker — local or remote — without the
// webui needing to reach each worker's trainer-control endpoint.
type WorkerAction = "save" | "save-stop" | "abort";

const RELAY_COMMAND: Record<WorkerAction, string> = {
  save: "save_checkpoint",
  "save-stop": "save_and_stop",
  abort: "abort",
};

/** Human label for a control action, for button/result text. */
function bulkActionLabel(action: WorkerAction): string {
  switch (action) {
    case "save":
      return "Save checkpoint";
    case "save-stop":
      return "Save & Stop";
    case "abort":
      return "Abort";
  }
}

/** Relay a command to specific worker ids, or to all workers when
 *  ``workerIds`` is null. Targeting a pipeline group means relaying to every
 *  member id — only the leader's heartbeat consumes it, and the callback's
 *  cross-rank broadcast then applies it on all ranks, so addressing all
 *  members is safe and leader-agnostic. */
async function relayToWorkers(
  baseUrl: string,
  command: string,
  workerIds: string[] | null,
): Promise<void> {
  if (workerIds === null) {
    await api.diLoCoServerControl(baseUrl, "command", { command });
    return;
  }
  await Promise.all(
    workerIds.map((id) =>
      api.diLoCoServerControl(baseUrl, "command", { command, worker_id: id }),
    ),
  );
}

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

/** Thrown when the operator cancels an in-flight shutdown sequence. The
 *  ``cancelled`` marker lets the caller distinguish it from a real failure. */
function cancelledError(): Error {
  return Object.assign(new Error("cancelled by operator"), { cancelled: true });
}

/** Poll the DiLoCo server's /status until the workers that were registered
 *  when we started waiting have all deregistered (applied save-and-stop and
 *  exited). Tracks that initial target set — not a live count — so a worker
 *  that crash-restarts (re-registers) or an unrelated late arrival can't pin
 *  the wait open. Reports progress as targets drop off. Throws on timeout
 *  (naming how many remain) or when ``shouldCancel`` trips. */
async function waitForWorkersToStop(
  baseUrl: string,
  opts: {
    timeoutMs?: number;
    pollMs?: number;
    onProgress?: (stopped: number, total: number) => void;
    shouldCancel?: () => boolean;
  } = {},
): Promise<void> {
  const { timeoutMs = 600_000, pollMs = 2_000, onProgress, shouldCancel } = opts;
  const deadline = Date.now() + timeoutMs;
  // Capture the target set at the moment we start waiting.
  let remaining = new Set<string>();
  try {
    const s = await api.diLoCoServerStatus(baseUrl);
    remaining = new Set(Object.keys(s.workers ?? {}));
  } catch {
    // ignore; first poll below establishes the set
  }
  const total = remaining.size;
  if (total === 0) return;
  while (true) {
    if (shouldCancel?.()) throw cancelledError();
    try {
      const s = await api.diLoCoServerStatus(baseUrl);
      const live = new Set(Object.keys(s.workers ?? {}));
      remaining = new Set([...remaining].filter((id) => live.has(id)));
    } catch {
      // Transient status failure — retry next tick.
    }
    onProgress?.(total - remaining.size, total);
    if (remaining.size === 0) return;
    if (Date.now() >= deadline) {
      throw new Error(
        `timed out after ${Math.round(timeoutMs / 1000)}s waiting for ` +
          `${remaining.size} worker(s) to stop`,
      );
    }
    await sleep(pollMs);
  }
}

/** Per-worker TensorBoard launcher: opens the TB modal on the worker's own
 *  ``output_dir/runs``. Renders nothing when the worker never reported an
 *  output_dir (older client / no correlated job), since there's no logdir. */
function WorkerTBButton({
  outputDir,
  label,
}: {
  outputDir?: string | null;
  label: string;
}) {
  const [open, setOpen] = useState(false);
  if (!outputDir) return null;
  const logdir = `${outputDir.replace(/\/+$/, "")}/runs`;
  return (
    <>
      <button
        className="tiny"
        onClick={() => setOpen(true)}
        title={`Open TensorBoard on ${logdir}`}
      >
        📊 TB
      </button>
      {open && (
        <TensorBoardModal
          global
          initialLogdir={logdir}
          initialWindowTitle={`DiLoCo worker ${label}`}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

function WorkersSection({
  baseUrl,
  status,
  jobs,
  refreshSeconds,
}: {
  baseUrl: string;
  status: DiLoCoStatus;
  jobs: Job[] | null;
  refreshSeconds: number;
}) {
  const workers = status.workers ?? {};
  const ids = Object.keys(workers);
  const queryClient = useQueryClient();
  // Collective worker controls: one relay broadcast (worker_id omitted) tells
  // the server to queue the command for every registered worker, delivered on
  // each one's next heartbeat. Works for remote workers too — no per-job
  // correlation needed.
  const [bulkBusy, setBulkBusy] = useState<WorkerAction | null>(null);
  const [bulkMsg, setBulkMsg] = useState<{ ok: boolean; text: string } | null>(
    null,
  );
  const applyToAll = async (action: WorkerAction, confirmMsg?: string) => {
    if (ids.length === 0) return;
    if (confirmMsg && !window.confirm(confirmMsg)) return;
    setBulkBusy(action);
    setBulkMsg(null);
    try {
      const resp = await api.diLoCoServerControl(baseUrl, "command", {
        command: RELAY_COMMAND[action],
      });
      const n = Array.isArray(resp?.workers) ? resp.workers.length : ids.length;
      setBulkMsg({ ok: true, text: `${bulkActionLabel(action)}: relayed to ${n}` });
      queryClient.invalidateQueries({ queryKey: ["diloco", "status", baseUrl] });
    } catch (e) {
      setBulkMsg({
        ok: false,
        text: `${bulkActionLabel(action)}: ${(e as Error).message}`,
      });
    } finally {
      setBulkBusy(null);
      window.setTimeout(() => setBulkMsg(null), 5000);
    }
  };

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
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <span>
          Workers{" "}
          <span
            className="muted"
            style={{ fontWeight: 400, fontSize: "smaller" }}
          >
            ({status.num_registered ?? 0}/{status.num_workers ?? "?"})
          </span>
        </span>
        <span style={{ flex: 1 }} />
        {bulkMsg && (
          <span
            role="status"
            style={{
              fontSize: "smaller",
              fontWeight: 400,
              color: bulkMsg.ok ? "#9ece6a" : "tomato",
            }}
          >
            {bulkMsg.text}
          </span>
        )}
        {/* All-workers controls — one relay broadcast to every registered
            worker. Hidden when no workers are connected. */}
        {ids.length > 0 && (
          <span
            style={{ display: "flex", gap: 6, fontWeight: 400 }}
            title={`Relay to all ${ids.length} worker(s)`}
          >
            <span className="muted" style={{ fontSize: 11, alignSelf: "center" }}>
              All:
            </span>
            <button
              className="tiny"
              disabled={bulkBusy !== null}
              onClick={() => applyToAll("save")}
              title="Request a checkpoint save on every worker without stopping"
            >
              Save
            </button>
            <button
              className="tiny"
              disabled={bulkBusy !== null}
              onClick={() =>
                applyToAll(
                  "save-stop",
                  `Save a final checkpoint and stop all ${ids.length} worker(s)?`,
                )
              }
              title="Save a final checkpoint then stop every worker cleanly"
            >
              Save &amp; Stop
            </button>
            <button
              className="tiny"
              style={{ color: "tomato" }}
              disabled={bulkBusy !== null}
              onClick={() =>
                applyToAll(
                  "abort",
                  `Abort all ${ids.length} worker(s)? Unsaved progress is lost.`,
                )
              }
              title="Stop every worker immediately without saving"
            >
              Abort
            </button>
          </span>
        )}
      </header>
      {ids.length === 0 ? (
        <div className="muted" style={{ padding: "16px 14px" }}>
          No workers connected
        </div>
      ) : (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
          }}
        >
          {groupWorkers(workers).map((group) => (
            <GroupCard
              key={group.groupId}
              baseUrl={baseUrl}
              group={group}
              workers={workers}
              heartbeatTimeout={status.heartbeat_timeout}
              job={correlateJob(group, workers, jobs)}
              refreshSeconds={refreshSeconds}
            />
          ))}
        </div>
      )}
    </section>
  );
}

interface GroupMember {
  workerId: string;
  ppRank: number;
}

interface GroupView {
  groupId: string;
  members: GroupMember[];
  /** True iff at least one member's worker_id carried a ``_pp<N>``
   *  suffix. Solo workers form a group of one with this flag false
   *  and render as a plain ``WorkerCard`` (pre-#84 behavior). */
  isPipelineGroup: boolean;
}

/** Correlate a worker group to its forgather job (issue #103).
 *
 *  Three keys identify a candidate job, in priority order:
 *   - ``job_params.diloco.worker_id`` == the group id. The queue route
 *     stamps a memorable two-word default into ``job_params.diloco`` for
 *     every DiLoCo training submission that arrives without an explicit
 *     id, so this is the authoritative key for any job submitted via the
 *     route. Pipeline ranks register as ``<base>_pp<N>`` and the group id
 *     strips that suffix, so the match holds for pipeline groups too.
 *   - ``queue_id``/``job_id`` == the group id. Catches legacy / pre-
 *     route-fill jobs (and the rare scheduler-fallback case) where the
 *     worker ended up registered under the queue_id.
 *   - the worker's reported ``output_dir``. A run that reuses a stable
 *     custom worker-id (e.g. to resume from its checkpoint) registers
 *     under an id distinct from both queue_id and any stamped
 *     worker_id, so the previous keys miss; the per-worker output-dir
 *     suffix is kept in lockstep with the job's resolved ``output_dir``,
 *     so this re-links it. Pipeline ranks of one job share a local
 *     output_dir, so any member's value identifies the group.
 *
 *  We gather candidates by all three keys and rank live-first (then most
 *  recently started) rather than short-circuiting on the id keys. This is
 *  essential because a worker is often named after a PRIOR run's id
 *  (picked from the restart menu): the old job with that exact id still
 *  lingers in the list, dead, and an id-key-first match would bind the
 *  worker to that corpse — hiding the live job's stats/controls. The same
 *  ranking lets a fresh respawn outrank the stopped job that still shares
 *  its output_dir.
 *
 *  Cross-node behavior: ``jobs`` is the local node's ``/api/jobs``
 *  list. A worker spawned on a peer (cluster-discovered DiLoCo server,
 *  worker submitted from another node) won't have a JobRecord on this
 *  node, so correlation returns null — but per-worker training stats
 *  still render from the DiLoCo server's heartbeat-aggregated
 *  ``workers[wid].stats`` view. The controls (Save / Save & Stop /
 *  Abort) work either way; they go through the DiLoCo server's
 *  trainer-control relay rather than the local trainer-control endpoint. */
function correlateJob(
  group: GroupView,
  workers: Record<string, DiLoCoWorkerStatus>,
  jobs: Job[] | null,
): Job | null {
  if (!jobs) return null;
  const groupOutputDir = group.members
    .map((m) => workers[m.workerId]?.output_dir)
    .find((d) => d);
  const candidates = jobs.filter((j) => {
    // Most reliable: the worker_id stamped on the job_params at queue
    // time. The queue route fills a memorable default for blank
    // submissions, so non-pool webui submits surface here.
    const diloco = (j.job_params as { diloco?: { worker_id?: string } } | null)
      ?.diloco;
    const stampedWid =
      typeof diloco?.worker_id === "string" ? diloco.worker_id : null;
    return (
      (stampedWid != null && stampedWid === group.groupId) ||
      j.queue_id === group.groupId ||
      j.job_id === group.groupId ||
      (!!groupOutputDir && j.output_dir === groupOutputDir)
    );
  });
  if (candidates.length === 0) return null;
  // Live first, then most recently started — a running respawn outranks a
  // dead job that shares its queue_id (old name reuse) or output_dir.
  candidates.sort(
    (a, b) =>
      Number(b.alive) - Number(a.alive) ||
      (b.started_at ?? b.submitted_at ?? 0) -
        (a.started_at ?? a.submitted_at ?? 0),
  );
  return candidates[0];
}

/** Group workers by stripping the ``_pp<N>`` suffix from ``worker_id``.
 *  Workers without that suffix form a degenerate group of one. */
function groupWorkers(
  workers: Record<string, DiLoCoWorkerStatus>,
): GroupView[] {
  const groups: Record<string, GroupView> = {};
  for (const wid of Object.keys(workers)) {
    const ppMatch = wid.match(/^(.+)_pp(\d+)$/);
    const groupId = ppMatch ? ppMatch[1] : wid;
    const ppRank = ppMatch ? parseInt(ppMatch[2], 10) : 0;
    if (!groups[groupId]) {
      groups[groupId] = {
        groupId,
        members: [],
        isPipelineGroup: false,
      };
    }
    groups[groupId].members.push({ workerId: wid, ppRank });
    if (ppMatch) groups[groupId].isPipelineGroup = true;
  }
  for (const g of Object.values(groups)) {
    g.members.sort((a, b) => a.ppRank - b.ppRank);
  }
  return Object.values(groups).sort((a, b) =>
    a.groupId.localeCompare(b.groupId),
  );
}

/** Aggregator card for a DiLoCo worker group.
 *
 *  Solo workers (single member, no ``_pp<N>`` suffix on their worker_id)
 *  render as a plain ``WorkerCard`` — preserves the pre-#84 panel layout
 *  exactly. Pipeline groups (multiple members) render as a compact
 *  header summarising the canonical (pp_rank=0) stats with a ``PP×N``
 *  badge and an expand caret; expanded, each rank's full ``WorkerCard``
 *  is shown below.
 *
 *  Atomic group eviction (server-side) means kicking any one member
 *  takes the whole group down, so the group header's Kick button just
 *  targets pp_rank=0; per-rank Kick buttons stay inside each expanded
 *  ``WorkerCard`` for diagnostic use. The trainer-protocol controls
 *  (Save / Stop / Abort) act on the shared job, so they're emitted
 *  once on the group header instead of per-rank. */
function GroupCard({
  baseUrl,
  group,
  workers,
  heartbeatTimeout,
  job,
  refreshSeconds,
}: {
  baseUrl: string;
  group: GroupView;
  workers: Record<string, DiLoCoWorkerStatus>;
  heartbeatTimeout: number | undefined;
  job: Job | null;
  refreshSeconds: number;
}) {
  // Solo case: identical to pre-#84 rendering. WorkerCard owns the
  // full layout; no per-group affordances added.
  if (!group.isPipelineGroup) {
    const m = group.members[0];
    return (
      <WorkerCard
        baseUrl={baseUrl}
        workerId={m.workerId}
        workerStatus={workers[m.workerId]}
        heartbeatTimeout={heartbeatTimeout}
        job={job}
        refreshSeconds={refreshSeconds}
      />
    );
  }

  // Pipeline group: one canonical row + optional expand.
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(false);

  const canonicalMember = group.members[0];
  const canonical = workers[canonicalMember.workerId];

  // Worst-case (oldest) heartbeat across all ranks — surfaces a
  // straggler before the group is evicted.
  const oldestHeartbeat = Math.min(
    ...group.members.map((m) => workers[m.workerId].last_heartbeat ?? Infinity),
  );

  const jobStatusQ = useQuery({
    queryKey: ["jobs", "status", job?.job_id],
    queryFn: () => api.jobStatus(job!.job_id!),
    enabled: !!job?.job_id && !!job?.alive,
    refetchInterval: refreshSeconds * 1000,
    refetchIntervalInBackground: false,
  });

  // Group-level Kick targets pp_rank=0; atomic eviction takes the rest.
  const kickMutation = useMutation({
    mutationFn: () =>
      api.diLoCoServerControl(baseUrl, "kick_worker", {
        worker_id: canonicalMember.workerId,
      }),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["diloco", "status", baseUrl] });
    },
  });

  // Trainer-control via the server relay: target every member of the group
  // (only the leader's heartbeat consumes it; the callback's cross-rank
  // broadcast applies it on all ranks). Works without a correlated job.
  const controlMutation = useMutation({
    mutationFn: (action: WorkerAction) =>
      relayToWorkers(
        baseUrl,
        RELAY_COMMAND[action],
        group.members.map((m) => m.workerId),
      ),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["diloco", "status", baseUrl] });
    },
  });

  // Same source-preference as ``WorkerCard``: trainer-control endpoint
  // when a local JobRecord is correlated, else the canonical member's
  // heartbeat-reported stats from the DiLoCo server. Pipeline-group
  // members share an output_dir but each rank reports its own per-step
  // stats; the canonical member (pp_rank=0 by convention) is the source
  // of truth for the group's progress / loss / lr display.
  const stats: Record<string, unknown> | null =
    jobStatusQ.data ??
    (canonical?.stats ? { ...canonical.stats } : null);
  const ppN = group.members.length;

  return (
    <div
      style={{
        borderTop: "1px solid var(--border, #2a2a2a)",
        padding: "10px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      {/* Group header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <button
          className="tiny"
          onClick={() => setExpanded((e) => !e)}
          title={expanded ? "Collapse per-rank diagnostics" : "Expand per-rank diagnostics"}
          style={{
            width: 20,
            padding: "0 4px",
            fontFamily: "monospace",
          }}
        >
          {expanded ? "▾" : "▸"}
        </button>
        <span
          title={`Group health: oldest heartbeat among ${ppN} ranks`}
          style={{
            display: "inline-block",
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: workerHealthColor(oldestHeartbeat, heartbeatTimeout),
          }}
        />
        <code title={group.groupId} style={{ fontFamily: "monospace", fontSize: 12 }}>
          {truncId(group.groupId)}
        </code>
        <span
          className="muted"
          style={{
            fontSize: 11,
            padding: "1px 6px",
            borderRadius: 3,
            background: "var(--bg-surface, #24283b)",
            border: "1px solid var(--border, #3b4261)",
          }}
          title={`${ppN} pipeline ranks`}
        >
          PP×{ppN}
        </span>
        {(() => {
          // Cluster-bundle origin chip: when the correlated JobRecord
          // carries a ``cluster_job_id`` (PR-A multi-node DiLoCo
          // composition), surface it so the operator can correlate
          // this group with the Cluster Jobs view and the bundle's
          // per-rank queue items.
          const cjId =
            (job?.job_params as { cluster_job_id?: string } | null)
              ?.cluster_job_id ?? null;
          if (!cjId) return null;
          return (
            <span
              className="muted"
              style={{
                fontSize: 11,
                padding: "1px 6px",
                borderRadius: 3,
                background: "var(--bg-surface, #24283b)",
                border: "1px solid var(--border, #3b4261)",
              }}
              title={`Multi-node cluster bundle: ${cjId}`}
            >
              cluster <code style={{ fontSize: 11 }}>{cjId.slice(0, 10)}…</code>
            </span>
          );
        })()}
        <span className="muted" style={{ fontSize: 12 }}>
          Round <b style={{ color: "var(--text, inherit)" }}>{canonical.sync_round ?? 0}</b>
        </span>
        {canonical.steps_per_second !== undefined &&
          canonical.steps_per_second > 0 && (
            <span className="muted" style={{ fontSize: 12 }}>
              {canonical.steps_per_second.toFixed(2)} steps/s
            </span>
          )}
        <span className="muted" style={{ fontSize: 12 }}>
          hb {relativeAge(oldestHeartbeat)}
        </span>
        <span style={{ flex: 1 }} />
        <button
          className="tiny"
          onClick={() => {
            if (
              window.confirm(
                `Kick group ${group.groupId}? All ${ppN} pipeline ranks will be evicted (atomic group eviction).`,
              )
            )
              kickMutation.mutate();
          }}
          disabled={kickMutation.isPending}
          title="Force-evict this group from the DiLoCo server"
        >
          Kick group
        </button>
      </div>

      {stats && <JobStatsRow stats={stats} />}
      {!stats && (
        <div className="muted" style={{ fontSize: 11 }}>
          No training-side stats yet — the canonical worker hasn't
          reported a log step. (Controls below still work: they go
          through the DiLoCo server relay.)
        </div>
      )}

      <div style={{ display: "flex", gap: 6, justifyContent: "flex-end", flexWrap: "wrap" }}>
        <WorkerTBButton
          outputDir={canonical?.output_dir}
          label={canonicalMember.workerId}
        />
        <button
          className="tiny"
          onClick={() => controlMutation.mutate("save")}
          disabled={controlMutation.isPending}
          title="Request a checkpoint save without stopping training"
        >
          Save checkpoint
        </button>
        <button
          className="tiny"
          onClick={() => {
            if (
              window.confirm(
                `Save final checkpoint and stop group ${group.groupId}?`,
              )
            )
              controlMutation.mutate("save-stop");
          }}
          disabled={controlMutation.isPending}
          title="Save a final checkpoint, then stop training cleanly"
        >
          Save &amp; Stop
        </button>
        <button
          className="tiny"
          style={{ color: "tomato" }}
          onClick={() => {
            if (
              window.confirm(
                `Abort training on group ${group.groupId}? Any unsaved progress is lost.`,
              )
            )
              controlMutation.mutate("abort");
          }}
          disabled={controlMutation.isPending}
          title="Stop training immediately without saving"
        >
          Abort
        </button>
      </div>

      {/* Expanded per-rank diagnostics */}
      {expanded && (
        <div
          style={{
            marginTop: 4,
            paddingLeft: 12,
            borderLeft: "2px solid var(--border, #3b4261)",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {group.members.map((m) => (
            <WorkerCard
              key={m.workerId}
              baseUrl={baseUrl}
              workerId={m.workerId}
              workerStatus={workers[m.workerId]}
              heartbeatTimeout={heartbeatTimeout}
              // Compact: the per-rank stats are identical to the
              // group canonical (Round, steps/s) already shown on
              // the header, and Save / Save & Stop / Abort are
              // job-level controls — emitting them per-rank would
              // be misleading since clicking any one targets the
              // same shared job.
              job={job}
              refreshSeconds={refreshSeconds}
              compact
            />
          ))}
        </div>
      )}
    </div>
  );
}

function WorkerCard({
  baseUrl,
  workerId,
  workerStatus,
  heartbeatTimeout,
  job,
  refreshSeconds,
  compact = false,
}: {
  baseUrl: string;
  workerId: string;
  workerStatus: DiLoCoWorkerStatus;
  heartbeatTimeout: number | undefined;
  job: Job | null;
  refreshSeconds: number;
  /** Compact mode hides the JobStatsRow, the "no correlated job"
   *  warning, and the trainer-protocol controls (Save / Save & Stop /
   *  Abort) — used by ``GroupCard`` for the expanded per-rank
   *  diagnostic view, where those job-level affordances are
   *  redundant with (and confusing alongside) the group header's
   *  copy. Per-rank Kick stays — it's a server-level control. */
  compact?: boolean;
}) {
  const queryClient = useQueryClient();
  // Per-worker training status — only fetched when we have a matched
  // job_id (the trainer-control protocol requires the correlated id).
  // Polls at the panel cadence so the row's progress bar / stat pills
  // tick in sync with the rest of the view.
  const jobStatusQ = useQuery({
    queryKey: ["jobs", "status", job?.job_id],
    queryFn: () => api.jobStatus(job!.job_id!),
    enabled: !!job?.job_id && !!job?.alive,
    refetchInterval: refreshSeconds * 1000,
    refetchIntervalInBackground: false,
  });

  const kickMutation = useMutation({
    mutationFn: () =>
      api.diLoCoServerControl(baseUrl, "kick_worker", {
        worker_id: workerId,
      }),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["diloco", "status", baseUrl] });
    },
  });

  // Trainer-control via the server relay, targeting this worker's id.
  const controlMutation = useMutation({
    mutationFn: (action: WorkerAction) =>
      relayToWorkers(baseUrl, RELAY_COMMAND[action], [workerId]),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["diloco", "status", baseUrl] });
    },
  });

  // Source the stats row from the trainer-control endpoint when a local
  // JobRecord correlated this worker; fall back to the heartbeat-reported
  // ``workerStatus.stats`` (sourced by the DiLoCo callback in parallel
  // with the trainer-control relay) when no correlation is available —
  // the cross-node case where the trainer lives on a peer this webui
  // can't reach. Both paths feed the same JobStatsRow keys
  // (``global_step``, ``max_steps``, ``loss``, ``learning_rate``,
  // ``grad_norm``, ``epoch``, ``tok_per_sec``, ``tokens``, ``peak_mem``).
  const stats: Record<string, unknown> | null =
    jobStatusQ.data ?? (workerStatus.stats ? { ...workerStatus.stats } : null);

  return (
    <div
      style={{
        borderTop: "1px solid var(--border, #2a2a2a)",
        padding: "10px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      {/* Top row: identity + DiLoCo-server-side facts + Kick */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span
          title="Health (green <60s heartbeat, yellow <120s, red older)"
          style={{
            display: "inline-block",
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: workerHealthColor(
              workerStatus.last_heartbeat,
              heartbeatTimeout,
            ),
          }}
        />
        <code
          title={workerId}
          style={{ fontFamily: "monospace", fontSize: 12 }}
        >
          {truncId(workerId)}
        </code>
        <span className="muted" style={{ fontSize: 12 }}>
          {workerStatus.hostname ?? "—"}
        </span>
        <span className="muted" style={{ fontSize: 12 }}>
          Round <b style={{ color: "var(--text, inherit)" }}>{workerStatus.sync_round ?? 0}</b>
        </span>
        {workerStatus.steps_per_second !== undefined &&
          workerStatus.steps_per_second > 0 && (
            <span className="muted" style={{ fontSize: 12 }}>
              {workerStatus.steps_per_second.toFixed(2)} steps/s
            </span>
          )}
        {/* Worker-reported sync count. For an off-server backend
            (shared-memory) the server's own ``sync_round`` above stays 0
            because the worker never submits — this is the real progress
            signal. ``last_send_mb`` ~0 marks the off-wire sync. */}
        {workerStatus.sync_state?.sync_count !== undefined && (
          <span
            className="muted"
            style={{ fontSize: 12 }}
            title={
              workerStatus.sync_state.last_send_mb !== undefined
                ? `last sync sent ${workerStatus.sync_state.last_send_mb.toFixed(
                    1,
                  )} MB on the wire`
                : undefined
            }
          >
            sync{" "}
            <b style={{ color: "var(--text, inherit)" }}>
              {workerStatus.sync_state.sync_count}
            </b>
          </span>
        )}
        <span className="muted" style={{ fontSize: 12 }}>
          hb {relativeAge(workerStatus.last_heartbeat)}
        </span>
        <span style={{ flex: 1 }} />
        {/* Per-rank Kick is suppressed in compact mode: under
            atomic group eviction, kicking one rank just spells
            "kick the whole group" the long way around. The group
            header carries a single Kick group button. */}
        {!compact && (
          <button
            className="tiny"
            onClick={() => {
              if (window.confirm(`Kick worker ${workerId}?`))
                kickMutation.mutate();
            }}
            disabled={kickMutation.isPending}
            title="Force-evict this worker from the DiLoCo server"
          >
            Kick
          </button>
        )}
      </div>

      {/* Middle: training-side progress + stats. Sourced from the local
          trainer-control endpoint when a JobRecord is correlated, else
          from the DiLoCo server's heartbeat-aggregated per-worker view.
          Suppressed in compact mode — the per-rank stats duplicate the
          group's canonical stats on the header. */}
      {!compact && stats && <JobStatsRow stats={stats} />}
      {!compact && !stats && (
        <div className="muted" style={{ fontSize: 11 }}>
          No training-side stats yet — the worker hasn't reported a log
          step. (Controls below still work: they go through the DiLoCo
          server relay.)
        </div>
      )}

      {/* Bottom: trainer-control via the relay (Save / Save & Stop / Abort).
          Suppressed in compact mode — a pipeline rank's controls are
          emitted once on the group header; duplicating per-rank would be
          misleading. */}
      {!compact && (
        <div style={{ display: "flex", gap: 6, justifyContent: "flex-end", flexWrap: "wrap" }}>
          <WorkerTBButton outputDir={workerStatus.output_dir} label={workerId} />
          <button
            className="tiny"
            onClick={() => controlMutation.mutate("save")}
            disabled={controlMutation.isPending}
            title="Request a checkpoint save without stopping training"
          >
            Save checkpoint
          </button>
          <button
            className="tiny"
            onClick={() => {
              if (window.confirm(`Save final checkpoint and stop worker ${workerId}?`))
                controlMutation.mutate("save-stop");
            }}
            disabled={controlMutation.isPending}
            title="Save a final checkpoint, then stop training cleanly"
          >
            Save &amp; Stop
          </button>
          <button
            className="tiny"
            style={{ color: "tomato" }}
            onClick={() => {
              if (window.confirm(`Abort training on worker ${workerId}? Any unsaved progress is lost.`))
                controlMutation.mutate("abort");
            }}
            disabled={controlMutation.isPending}
            title="Stop training immediately without saving"
          >
            Abort
          </button>
        </div>
      )}
    </div>
  );
}

/** Per-worker training stats — progress bar from global_step / max_steps
 *  plus a compact pill row mirroring JobsPanel's JobStatusBlock (loss,
 *  lr, tok/s, peak_mem, grad_norm, epoch).
 *
 *  Accepts both source shapes:
 *   - trainer-control endpoint /status: spreads the trainer's raw log dict,
 *     so the per-window token count is keyed ``tokens``.
 *   - DiLoCo server per-worker /status (cross-node fallback): the heartbeat
 *     schema names the same value ``tokens_window`` (see diloco/stats.py).
 *  Cross-node entries are aliased onto ``tokens`` here so the pill renders
 *  identically without forcing the wire schema to ship redundant keys. */
function JobStatsRow({ stats }: { stats: Record<string, unknown> }) {
  const step = typeof stats.global_step === "number" ? stats.global_step : null;
  const max = typeof stats.max_steps === "number" ? stats.max_steps : null;
  const showProgress = step !== null && max !== null && max > 0;
  const pct = showProgress ? Math.max(0, Math.min(100, ((step as number) / (max as number)) * 100)) : 0;

  // Same display order as JobsPanel.JobStatusBlock — loss / lr /
  // grad_norm / epoch / tok/s / tokens / peak_mem. Each picker
  // returns null when the field is missing or wrong-shaped so the
  // pill is silently dropped.
  const pickTokens = (s: Record<string, unknown>): unknown =>
    typeof s.tokens === "number" ? s.tokens : s.tokens_window;
  const pickers: Array<[string, (s: Record<string, unknown>) => unknown, (v: unknown) => string | null]> = [
    ["loss", (s) => s.loss, (v) => (typeof v === "number" ? v.toFixed(4) : null)],
    ["lr", (s) => s.learning_rate, (v) => (typeof v === "number" ? fmtLr(v) : null)],
    [
      "grad_norm",
      (s) => s.grad_norm,
      (v) => (typeof v === "number" ? v.toFixed(3) : null),
    ],
    ["epoch", (s) => s.epoch, (v) => (typeof v === "number" ? v.toFixed(3) : null)],
    [
      "tok/s",
      (s) => s.tok_per_sec,
      (v) => (typeof v === "number" ? fmtCount(v) : null),
    ],
    ["tokens", pickTokens, (v) => (typeof v === "number" ? fmtCount(v) : null)],
    ["peak_mem", (s) => s.peak_mem, (v) => fmtPeakMem(v)],
  ];
  const pills: Array<[string, string]> = [];
  for (const [label, pick, fmt] of pickers) {
    const out = fmt(pick(stats));
    if (out !== null) pills.push([label, out]);
  }

  if (!showProgress && pills.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {showProgress && (
        <div
          title={`${step} / ${max} steps`}
          style={{ display: "flex", alignItems: "center", gap: 8 }}
        >
          <div
            style={{
              flex: 1,
              background: "var(--bg, #1a1b26)",
              border: "1px solid var(--border, #3b4261)",
              borderRadius: 4,
              height: 8,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${pct}%`,
                height: "100%",
                background: "#7aa2f7",
                transition: "width 200ms ease",
              }}
            />
          </div>
          <span className="muted" style={{ fontSize: 11, whiteSpace: "nowrap" }}>
            {step}/{max} ({pct.toFixed(1)}%)
          </span>
        </div>
      )}
      {pills.length > 0 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {pills.map(([k, v]) => (
            <span
              key={k}
              style={{
                fontSize: 11,
                background: "var(--bg, #1a1b26)",
                border: "1px solid var(--border, #3b4261)",
                borderRadius: 4,
                padding: "1px 6px",
              }}
            >
              <span className="muted">{k}</span> {v}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Formatter helpers (mirror JobsPanel's privates; duplicated to avoid a
// cross-component import — they're trivial and unlikely to drift in
// the lifetime of this view).
// ---------------------------------------------------------------------------

function fmtLr(v: number): string {
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs < 1e-5 || abs >= 100) return v.toExponential(2);
  return v.toPrecision(4).replace(/\.?0+$/, "");
}

function fmtCount(v: number): string {
  return Math.round(v).toLocaleString();
}

function fmtPeakMem(v: unknown): string | null {
  const values: number[] = [];
  if (typeof v === "number") values.push(v);
  else if (Array.isArray(v)) {
    for (const x of v) if (typeof x === "number") values.push(x);
  } else return null;
  if (values.length === 0) return null;
  const max = Math.max(...values);
  if (!Number.isFinite(max) || max <= 0) return null;
  const useGiB = max / 1024 ** 3 >= 1;
  const fmt = (n: number) =>
    useGiB ? (n / 1024 ** 3).toFixed(2) : (n / 1024 ** 2).toFixed(0);
  const unit = useGiB ? "GiB" : "MiB";
  return `${values.map(fmt).join(", ")} ${unit}`;
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
  const numRegistered = Object.keys(status.workers ?? {}).length;

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
      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 10 }}>
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

        {/* Two columns inside the Control panel max out at ~200px each
            so the buttons don't stretch to "1/5 of the screen" on wide
            monitors. The grid still collapses to one column on narrow. */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 10,
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
              title="Stop the DiLoCo server — cleanly (stop workers, checkpoint, stop) or immediately."
            >
              Shutdown server
            </button>
            {numRegistered > 0 && (
              <div className="muted" style={{ fontSize: 11 }}>
                {numRegistered} registered worker(s)
              </div>
            )}
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
        {info &&
          (info.transport === "grpc" ||
            info.expected_client_settings?.wire_format === "safetensors") && (
            <div
              className="muted"
              style={{ fontSize: 11, wordBreak: "break-all" }}
            >
              bulk transport:{" "}
              {info.transport === "grpc"
                ? `gRPC${info.grpc_endpoint ? ` (${info.grpc_endpoint})` : ""}`
                : "HTTP"}
              {info.expected_client_settings?.wire_format
                ? ` · ${info.expected_client_settings.wire_format}`
                : ""}
            </div>
          )}
      </div>

      {confirmShutdown && (
        <ShutdownDialog
          baseUrl={baseUrl}
          numWorkers={numRegistered}
          saveDir={saveDir}
          onClose={() => {
            setConfirmShutdown(false);
            invalidate();
          }}
        />
      )}
    </section>
  );
}

/** Two-mode shutdown dialog, all driven through the server's command relay.
 *  The *clean* path is the easy default for "stop everything": relay
 *  save-and-stop to every worker, wait until they've deregistered (exited),
 *  checkpoint the server, then stop it. The *force* path relays abort to all
 *  workers (best-effort — workers stop without saving) and stops the server
 *  without waiting.
 *
 *  A clean run can be cancelled while it waits (hands control back with the
 *  server still up so the operator can troubleshoot a stuck worker); the
 *  dialog streams milestones and a live worker-stop count. */
function ShutdownDialog({
  baseUrl,
  numWorkers,
  saveDir,
  onClose,
}: {
  baseUrl: string;
  numWorkers: number;
  saveDir: string | null;
  onClose: () => void;
}) {
  type Phase = "choose" | "running" | "done" | "failed" | "cancelled";
  const [phase, setPhase] = useState<Phase>("choose");
  const [log, setLog] = useState<string[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Cancellation: the long step is the worker-stop poll. A ref (not state) so
  // the in-flight async loop reads the latest value without a stale closure;
  // ``cancelling`` state just drives the button label until the loop notices.
  const cancelRef = useRef(false);
  const [cancelling, setCancelling] = useState(false);
  const append = (line: string) => setLog((l) => [...l, line]);
  const ensureNotCancelled = () => {
    if (cancelRef.current) throw cancelledError();
  };
  const requestCancel = () => {
    cancelRef.current = true;
    setCancelling(true);
    append("Cancel requested — handing control back…");
  };

  // Escape cancels only while still choosing; once running we ignore it so a
  // stray keypress can't dismiss the dialog mid-sequence and hide progress.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && phase === "choose") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, phase]);

  const n = numWorkers;

  // Common catch: a cancellation (operator bailed) is not a failure — it just
  // hands control back with the server still running, so the operator can
  // troubleshoot. Anything else is a real error.
  const handleSequenceError = (e: unknown) => {
    setProgress(null);
    if (cancelRef.current || (e as { cancelled?: boolean })?.cancelled) {
      append("Cancelled. The server was NOT stopped; workers may still be running.");
      setPhase("cancelled");
    } else {
      setError((e as Error).message);
      setPhase("failed");
    }
  };

  const runClean = async () => {
    cancelRef.current = false;
    setCancelling(false);
    setPhase("running");
    try {
      if (n > 0) {
        append(`Relaying save-and-stop to ${n} worker(s)…`);
        await relayToWorkers(baseUrl, "save_and_stop", null);
        append("Waiting for workers to stop…  (Cancel to regain control)");
        setProgress(`0/${n} stopped`);
        await waitForWorkersToStop(baseUrl, {
          onProgress: (s, t) => setProgress(`${s}/${t} stopped`),
          shouldCancel: () => cancelRef.current,
        });
        append("All workers stopped.");
        setProgress(null);
      } else {
        append("No workers registered — skipping worker stop.");
      }

      ensureNotCancelled();
      if (saveDir) {
        append("Saving server checkpoint…");
        try {
          await api.diLoCoServerControl(baseUrl, "save_state");
          append("  server checkpoint saved.");
        } catch (e) {
          // Surface but continue — losing the server checkpoint shouldn't
          // strand a server we've already told every worker to leave.
          append(`  server checkpoint failed: ${(e as Error).message}`);
        }
      } else {
        append("No save_dir configured — skipping server checkpoint.");
      }

      ensureNotCancelled();
      append("Stopping server…");
      await api.diLoCoServerControl(baseUrl, "shutdown");
      append("Server stopped. Done.");
      setPhase("done");
    } catch (e) {
      handleSequenceError(e);
    }
  };

  const runForce = async () => {
    cancelRef.current = false;
    setCancelling(false);
    setPhase("running");
    try {
      if (n > 0) {
        append(`Relaying abort to ${n} worker(s)…`);
        try {
          await relayToWorkers(baseUrl, "abort", null);
          append("  abort relayed (workers stop without saving).");
        } catch (e) {
          // Best-effort: we're stopping the server next regardless.
          append(`  abort relay failed: ${(e as Error).message} (continuing).`);
        }
      }
      append("Stopping server…");
      await api.diLoCoServerControl(baseUrl, "shutdown");
      append("Server stopped. Done.");
      setPhase("done");
    } catch (e) {
      handleSequenceError(e);
    }
  };

  return (
    <ModalBackdrop onClose={phase === "running" ? () => {} : onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="diloco-shutdown-title"
        style={{
          background: "var(--bg-surface, #24283b)",
          border: "1px solid var(--border, #3b4261)",
          borderRadius: 6,
          padding: 16,
          maxWidth: 480,
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <strong id="diloco-shutdown-title">Shutdown DiLoCo server</strong>

        {phase === "choose" && (
          <>
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
              <div style={{ fontWeight: 600 }}>Clean shutdown</div>
              <div className="muted" style={{ fontSize: "smaller" }}>
                Save &amp; stop {n > 0 ? `all ${n} worker(s)` : "workers"}, wait
                for them to exit, checkpoint the server, then stop it. No data
                loss.
              </div>
              <button
                style={{ alignSelf: "flex-start" }}
                onClick={runClean}
                autoFocus
              >
                Clean shutdown
              </button>
            </div>

            <div
              style={{
                border: "1px solid #5a2a2a",
                borderRadius: 4,
                padding: 10,
                display: "flex",
                flexDirection: "column",
                gap: 6,
              }}
            >
              <div style={{ fontWeight: 600, color: "#f7768e" }}>
                Force stop
              </div>
              <div className="muted" style={{ fontSize: "smaller" }}>
                Relay <strong>abort</strong> to {n > 0 ? `all ${n} worker(s)` : "workers"}{" "}
                (they stop without saving) and stop the server without waiting.
                Unsaved progress is lost.
              </div>
              <button
                style={{
                  alignSelf: "flex-start",
                  background: "#3a2a2a",
                  color: "#f7768e",
                }}
                onClick={runForce}
              >
                Force stop
              </button>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button onClick={onClose}>Cancel</button>
            </div>
          </>
        )}

        {phase !== "choose" && (
          <>
            <pre
              style={{
                margin: 0,
                padding: 10,
                background: "var(--bg, #1a1b26)",
                border: "1px solid var(--border, #3b4261)",
                borderRadius: 4,
                fontSize: 12,
                maxHeight: 240,
                overflow: "auto",
                whiteSpace: "pre-wrap",
              }}
            >
              {log.join("\n")}
              {progress && `\n${progress}`}
            </pre>
            {error && (
              <div role="alert" style={{ color: "tomato", fontSize: "smaller" }}>
                {error}
                {phase === "failed" && (
                  <div className="muted" style={{ marginTop: 4 }}>
                    The server was not stopped. Re-open this dialog to retry, or
                    use Force kill.
                  </div>
                )}
              </div>
            )}
            {phase === "cancelled" && (
              <div className="muted" style={{ fontSize: "smaller" }}>
                Control handed back so you can troubleshoot. Re-open this dialog
                to retry once the stuck worker is resolved, or use Force kill.
              </div>
            )}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              {phase === "running" && (
                <button
                  onClick={requestCancel}
                  disabled={cancelling}
                  title="Stop waiting and regain control of the UI. The server is left running."
                >
                  {cancelling ? "Cancelling…" : "Cancel"}
                </button>
              )}
              <button onClick={onClose} disabled={phase === "running"}>
                {phase === "running" ? "Working…" : "Close"}
              </button>
            </div>
          </>
        )}
      </div>
    </ModalBackdrop>
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
        gap: 10,
      }}
    >
      <header style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <strong>Work-unit dispatch</strong>
        <span className="muted" style={{ fontSize: "smaller" }}>
          one queue per (dataset_id, shuffle_seed) — interleaved /
          multi-source runs surface each underlying registration as
          its own card
        </span>
      </header>
      {/* auto-fit so multiple queues (interleaved / split-per-source
          training runs) wrap side-by-side rather than stacking
          vertically and pushing everything below offscreen. */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 10,
        }}
      >
        {queues.map((q) => (
          <QueueHeatmap
            key={`${q.dataset_id}|${q.shuffle_seed}`}
            baseUrl={baseUrl}
            summary={q}
            refreshSeconds={refreshSeconds}
          />
        ))}
      </div>
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
        <strong style={{ fontSize: 12 }}>
          {formatQueueLabel(summary) ?? summary.dataset_id}
          <span className="muted">@{summary.shuffle_seed}</span>
        </strong>
        {formatQueueLabel(summary) && (
          <code
            className="muted"
            style={{ fontSize: 11 }}
            title="dataset_id (16-hex hash of path/name/split/data_files/revision)"
          >
            {summary.dataset_id}
          </code>
        )}
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
            // Halved from the original 480 — at K=1024 (32x32 grid)
            // each cell is now ~7.5px, still hover-targetable and
            // packed tightly enough that two side-by-side queue
            // cards fit in one row at typical webui widths.
            maxWidth: 240,
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

/** Render a human-readable dataset label from the hint fields the
 *  worker shipped on /datasets/register. Pre-#hint-extension servers
 *  / workers won't have these so the queue card falls back to the
 *  raw dataset_id hash. */
function formatQueueLabel(summary: DiLoCoQueueSummary): string | null {
  const h = summary.hint;
  // ``path`` is always present when the hint extension is in play.
  if (!h.path) return null;
  let s = h.path;
  if (h.name) s += `:${h.name}`;
  if (h.split) s += `@${h.split}`;
  return s;
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
