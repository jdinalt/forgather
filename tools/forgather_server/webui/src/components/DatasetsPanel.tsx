import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import {
  AddDatasetServerRequest,
  ClusterDatasetEntry,
  ClusterDatasetInventoryResponse,
  ClusterDatasetServer,
  DatasetServerHealth,
  DatasetServerLocal,
  DatasetServerUser,
  HFCacheResponse,
  LocalDatasetEntry,
  LocalListResponse,
  api,
} from "../api";
import {
  DatasetsExploreTab,
  type SelectedLeaf,
} from "./DatasetsExploreTab";
import { ModalBackdrop } from "./ModalBackdrop";
import { useDemoMode } from "../demoMode";

type SubTab = "servers" | "explore";

/** Identifier the panel uses to refer to either kind of server uniformly.
 *  Local servers key by ``queue_id`` (stable across the run), user
 *  entries key by registry ``id`` (8 hex chars).
 *
 *  Cluster-wide server selection lives on the Cluster view's Datasets
 *  tab; this surface is intentionally per-node only — spawned jobs on
 *  this host plus servers the operator added by URL. */
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
  if (a.kind === "local") {
    return b.kind === "local" && a.queue_id === b.queue_id;
  }
  return b.kind === "user" && a.id === (b as { id: string }).id;
}

/** Top-level Datasets view. Tabs:
 *  - Servers: CRUD + status/handles/cache.
 *  - Explore: tree of dataset → split → table of rows.
 *
 * Cluster-wide inventory (master-aggregated view of every
 * dataset_server and every unique dataset) lives under the Cluster
 * view's Datasets tab — see ClusterPanel.
 */
interface DatasetsPanelProps {
  /** Cross-view preselect: when the Cluster view's Datasets tab fires
   *  a row click (or anything outside this panel routes a leaf here),
   *  the parent sets this and we both switch to the Explore sub-tab
   *  and pass the leaf down. Null in steady state. */
  pendingExplore?: SelectedLeaf | null;
  /** Called after the Explore tab consumes the preselect, so the
   *  parent can clear it. */
  onPreselectConsumed?: () => void;
  /** Called when any source inside this panel wants to navigate to
   *  Explore (Servers tab row clicks). Lifts the leaf to the parent
   *  so the same flow works whether the trigger came from inside
   *  this panel or from the Cluster view. */
  onOpenInExplore?: (leaf: SelectedLeaf) => void;
  /** Hand a chunk of text up to the parent so it can swap to the
   *  Inference > Analyze tab and kick off scoring. Wired into the
   *  cell context menu in DatasetsExploreTab. */
  onAnalyzeText?: (text: string) => void;
}

export function DatasetsPanel({
  pendingExplore,
  onPreselectConsumed,
  onOpenInExplore,
  onAnalyzeText,
}: DatasetsPanelProps = {}) {
  // Detect cluster mode via the same query the App-level gate uses.
  // TanStack dedups by ``queryKey`` so this doesn't cost an extra HTTP
  // request — App.tsx's polling and this one share the same cache.
  const clusterSelfQ = useQuery({
    queryKey: ["cluster-self"],
    queryFn: api.getClusterSelf,
    refetchInterval: 30000,
  });
  const clusterActive = !!clusterSelfQ.data;

  const [tab, setTab] = useState<SubTab>("servers");
  // Switch to the Explore sub-tab whenever a preselect arrives from
  // outside — Cluster view's Datasets tab, or anywhere else that
  // hands a leaf to App. The Explore tab itself consumes the
  // preselect and signals back via onPreselectConsumed.
  useEffect(() => {
    if (pendingExplore) setTab("explore");
  }, [pendingExplore]);
  const openInExplore = (leaf: SelectedLeaf) => {
    setTab("explore");
    onOpenInExplore?.(leaf);
  };
  // Shared queries — both subtabs need the local+user lists. Same keys
  // mean TanStack Query serves them from one cache.
  const localsQ = useQuery({
    queryKey: ["dataset-servers-local"],
    queryFn: api.listLocalDatasetServers,
    refetchInterval: 5000,
  });
  const usersQ = useQuery({
    queryKey: ["dataset-servers-user"],
    queryFn: api.listUserDatasetServers,
  });
  // Cluster-wide servers from the master inventory — used by the
  // Cluster tab, by the cluster-aware Servers fold, and by the
  // Explore tab's host list in cluster mode.
  const clusterServersQ = useQuery({
    queryKey: ["cluster", "dataset_servers"],
    queryFn: api.getClusterDatasetServers,
    refetchInterval: 5000,
    enabled: clusterActive,
  });
  const localServers = localsQ.data ?? [];
  const userServers = usersQ.data ?? [];
  const clusterServers = clusterServersQ.data ?? [];
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
            <button
              className={tab === "explore" ? "active" : ""}
              onClick={() => setTab("explore")}
            >
              explore
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
        <DatasetServersTab onOpenInExplore={openInExplore} />
      </div>
      <div
        style={{
          display: tab === "explore" ? "flex" : "none",
          flex: 1,
          minHeight: 0,
        }}
      >
        <DatasetsExploreTab
          localServers={localServers}
          userServers={userServers}
          clusterServers={clusterServers}
          clusterActive={clusterActive}
          preselect={pendingExplore ?? null}
          onPreselectConsumed={onPreselectConsumed}
          onAnalyzeText={onAnalyzeText}
        />
      </div>
    </div>
  );
}

// ---------- Cluster tab ----------

type ClusterDatasetSortKey =
  | "dataset_id"
  | "source"
  | "length"
  | "size_bytes"
  | "host_count";

/** Format an epoch-seconds timestamp as a ``Ns / Nm / Nh ago`` delta. */
function formatAgo(ts: number | null): string {
  if (ts === null || ts === undefined || ts <= 0) return "never";
  const now = Date.now() / 1000;
  const delta = Math.max(0, Math.floor(now - ts));
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  return `${Math.floor(delta / 3600)}h ago`;
}

/** Read-only cluster inventory view. Polls /api/cluster/dataset_inventory
 *  every 5s. The master self-gates the loops, so this surface stays
 *  consistent across master failover (the new master rebuilds its
 *  inventory from peers within ~10s of taking over).
 *
 *  Exported so the Cluster view (ClusterPanel) can mount it as its
 *  Datasets tab. Kept in DatasetsPanel.tsx because it shares helpers
 *  (ClusterDatasetRow, SortableHeader, formatAgo, …) with the rest of
 *  this file.
 *
 *  ``onOpenInExplore``, when wired, makes the dataset rows clickable:
 *  the parent navigates the outer view to Datasets, switches to the
 *  Explore sub-tab, and pre-selects the clicked dataset against the
 *  first healthy server that advertises it. */
export function DatasetsClusterTab({
  onOpenInExplore,
}: {
  onOpenInExplore?: (leaf: SelectedLeaf) => void;
} = {}) {
  const inventoryQ = useQuery<ClusterDatasetInventoryResponse>({
    queryKey: ["cluster", "dataset_inventory"],
    queryFn: api.getClusterDatasetInventory,
    refetchInterval: 5000,
  });
  const inv = inventoryQ.data;

  const [serverSort, setServerSort] = useState<SortState<"label" | "healthy">>({
    by: "healthy",
    dir: "desc",
  });
  const [dsSort, setDsSort] = useState<SortState<ClusterDatasetSortKey>>({
    by: "dataset_id",
    dir: "asc",
  });
  const toggleServer = makeSortToggle(serverSort, setServerSort);
  const toggleDs = makeSortToggle(dsSort, setDsSort);

  const servers = inv?.servers ?? [];
  const datasets = inv?.datasets ?? [];

  // server_id -> {base_url, label} for rendering "hosts" column on
  // dataset rows. Built once per render — cheap (a few dozen entries).
  const serverIdMap: Map<string, ClusterDatasetServer> = useMemo(() => {
    const m = new Map<string, ClusterDatasetServer>();
    for (const s of servers) m.set(s.server_id, s);
    return m;
  }, [servers]);

  const sortedServers = useMemo(() => {
    const arr = [...servers];
    arr.sort((a, b) => {
      let cmp = 0;
      if (serverSort.by === "label") {
        cmp = a.base_url.localeCompare(b.base_url);
      } else {
        // "healthy" desc puts healthy servers first; secondary sort
        // by label for stable order across refreshes.
        cmp =
          Number(b.healthy) - Number(a.healthy) ||
          a.base_url.localeCompare(b.base_url);
      }
      return serverSort.dir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [servers, serverSort]);

  const sortedDatasets = useMemo(() => {
    const arr = [...datasets];
    arr.sort((a, b) => {
      let cmp = 0;
      if (dsSort.by === "dataset_id") {
        cmp = a.dataset_id.localeCompare(b.dataset_id);
      } else if (dsSort.by === "source") {
        cmp = a.source.localeCompare(b.source) ||
          a.dataset_id.localeCompare(b.dataset_id);
      } else if (dsSort.by === "length") {
        cmp = (a.length ?? -1) - (b.length ?? -1);
      } else if (dsSort.by === "size_bytes") {
        cmp = (a.size_bytes ?? -1) - (b.size_bytes ?? -1);
      } else if (dsSort.by === "host_count") {
        cmp = a.server_ids.length - b.server_ids.length;
      }
      return dsSort.dir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [datasets, dsSort]);

  if (inventoryQ.isLoading && !inv) {
    return (
      <div className="pane-state-small muted" style={{ padding: 16 }}>
        Loading cluster inventory…
      </div>
    );
  }
  if (inventoryQ.isError) {
    const msg = inventoryQ.error instanceof Error
      ? inventoryQ.error.message
      : "request failed";
    return (
      <div className="pane-state-small" style={{ padding: 16, color: "var(--danger, #b00)" }}>
        Could not load cluster inventory: {msg}
      </div>
    );
  }
  const isMaster = inv?.is_master ?? false;
  const lastPass = inv?.last_dataset_pass_ts ?? null;

  return (
    <div className="inference-model-panel" style={{ padding: "0 12px" }}>
      <section>
        <h4 className="dyn-heading">
          Cluster inventory
          <span className="muted" style={{ marginLeft: 8 }}>
            ({servers.length} server{servers.length === 1 ? "" : "s"},{" "}
            {datasets.length} dataset{datasets.length === 1 ? "" : "s"})
          </span>
        </h4>
        <div className="muted" style={{ marginBottom: 8 }}>
          {isMaster
            ? "This node is the cluster master. Inventory is computed locally."
            : "Inventory proxied from the cluster master."}
          {" · "}
          last dataset refresh: {formatAgo(lastPass)}
        </div>
        {inv?.metrics && (
          // Cumulative poll counters across all servers. Surfaces
          // "the master is running but everything's failing" cases —
          // total_health_failures climbing without any healthy
          // server is the obvious signal.
          <div className="muted" style={{ marginBottom: 8, fontSize: "0.9em" }}>
            healthy:{" "}
            <strong>
              {inv.metrics.healthy_servers}/{inv.metrics.total_servers}
            </strong>
            {" · "}
            health polls: {inv.metrics.total_health_failures} failed /{" "}
            {inv.metrics.total_health_polls}
            {" · "}
            dataset polls: {inv.metrics.total_dataset_failures} failed /{" "}
            {inv.metrics.total_dataset_polls}
            {typeof inv.metrics.master_age_seconds === "number" && (
              <>
                {" · "}
                master age: {Math.floor(inv.metrics.master_age_seconds)}s
              </>
            )}
          </div>
        )}
      </section>

      <section>
        <h4 className="dyn-heading">
          Servers
          <span className="muted"> ({servers.length})</span>
        </h4>
        {servers.length === 0 ? (
          <div className="muted pane-state-small">
            No dataset_servers reported. Start one or add a user entry on
            any cluster node — the master will pick it up within ~10s.
          </div>
        ) : (
          <div className="preview-table-wrap">
            <table className="preview-table">
              <thead>
                <tr>
                  <SortableHeader<"label" | "healthy">
                    col="label"
                    label="server"
                    current={serverSort}
                    toggle={toggleServer}
                  />
                  <th>source</th>
                  <th>peer</th>
                  <SortableHeader<"label" | "healthy">
                    col="healthy"
                    label="health"
                    current={serverSort}
                    toggle={toggleServer}
                    defaultDir="desc"
                  />
                  <th>last refresh</th>
                </tr>
              </thead>
              <tbody>
                {sortedServers.map((s) => (
                  <tr key={s.server_id}>
                    <td>
                      <span style={{ fontFamily: "monospace" }}>
                        {s.base_url}
                      </span>
                      <span className="muted" style={{ marginLeft: 8 }}>
                        · {s.label}
                      </span>
                    </td>
                    <td>{s.source}</td>
                    <td>
                      <span
                        style={{ fontFamily: "monospace" }}
                        title={s.peer_node_id ?? ""}
                      >
                        {s.peer_node_id ? s.peer_node_id.slice(0, 8) : "-"}
                      </span>
                    </td>
                    <td>
                      <span
                        className={
                          "queue-status " +
                          (s.healthy ? "status-running" : "status-done")
                        }
                      >
                        {s.healthy ? "OK" : "DOWN"}
                      </span>
                      {!s.healthy && s.last_health_error && (
                        <span
                          className="muted"
                          style={{ marginLeft: 6 }}
                          title={s.last_health_error}
                        >
                          ({s.last_health_error.slice(0, 40)}
                          {s.last_health_error.length > 40 ? "…" : ""})
                        </span>
                      )}
                    </td>
                    <td>{formatAgo(s.last_dataset_refresh)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h4 className="dyn-heading">
          Datasets
          <span className="muted"> ({datasets.length})</span>
        </h4>
        {datasets.length === 0 ? (
          <div className="muted pane-state-small">
            No datasets indexed yet. Local mappings (``--local NAME=PATH``)
            show up automatically; HF / path datasets appear once a
            client (training run) has issued a ``/v1/load`` for them.
          </div>
        ) : (
          <div className="preview-table-wrap">
            <table className="preview-table">
              <thead>
                <tr>
                  <SortableHeader<ClusterDatasetSortKey>
                    col="dataset_id"
                    label="dataset"
                    current={dsSort}
                    toggle={toggleDs}
                  />
                  <SortableHeader<ClusterDatasetSortKey>
                    col="source"
                    label="source"
                    current={dsSort}
                    toggle={toggleDs}
                  />
                  <SortableHeader<ClusterDatasetSortKey>
                    col="length"
                    label="rows"
                    current={dsSort}
                    toggle={toggleDs}
                    defaultDir="desc"
                  />
                  <SortableHeader<ClusterDatasetSortKey>
                    col="size_bytes"
                    label="size"
                    current={dsSort}
                    toggle={toggleDs}
                    defaultDir="desc"
                  />
                  <SortableHeader<ClusterDatasetSortKey>
                    col="host_count"
                    label="hosts"
                    current={dsSort}
                    toggle={toggleDs}
                    defaultDir="desc"
                  />
                </tr>
              </thead>
              <tbody>
                {sortedDatasets.map((d) => (
                  <ClusterDatasetRow
                    key={d.dataset_id}
                    entry={d}
                    serverIdMap={serverIdMap}
                    onOpenInExplore={onOpenInExplore}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

/** Single row in the cluster datasets table. Extracted so the hosts
 *  cell can carry a useful tooltip without bloating the parent. */
function ClusterDatasetRow({
  entry,
  serverIdMap,
  onOpenInExplore,
}: {
  entry: ClusterDatasetEntry;
  serverIdMap: Map<string, ClusterDatasetServer>;
  onOpenInExplore?: (leaf: SelectedLeaf) => void;
}) {
  const hostBaseUrls = entry.server_ids
    .map((sid) => serverIdMap.get(sid)?.base_url ?? sid)
    .filter((url): url is string => !!url);
  const hostsTooltip = hostBaseUrls.join("\n");
  // Human-friendly handle: for `local/<name>` entries the ID is
  // already the human name; for HF/path entries the ID is the 16-hex
  // canonical hash — show the load_args.path beside it for context.
  const friendlyName =
    entry.name ??
    (entry.load_args && typeof entry.load_args["path"] === "string"
      ? (entry.load_args["path"] as string)
      : null);
  // Pick the source server we'll hand to the Explore tab. Prefer the
  // first *healthy* server that advertises this dataset; fall back to
  // the first server_id otherwise. If we picked an unhealthy peer the
  // Explore tab would just surface the upstream error inline — the
  // healthy-first preference is a UX shortcut, not a correctness gate.
  const args = entry.load_args ?? {};
  // Derive the load path by source. The backend only populates
  // load_args for path-based handles; ``local`` and ``hf`` entries
  // carry their identity in dataset_id directly:
  //   - local: dataset_id is ``local/<name>``
  //   - hf:    dataset_id is the HF repo (e.g., ``allenai/c4``)
  //   - path:  dataset_id is the handle hash; load_args holds path
  let path = "";
  if (entry.source === "local" || entry.source === "hf") {
    path = entry.dataset_id;
  } else if (typeof args["path"] === "string") {
    path = args["path"] as string;
  }
  const loadArgsName =
    typeof args["name"] === "string" ? (args["name"] as string) : undefined;
  const loadArgsSplit =
    typeof args["split"] === "string"
      ? (args["split"] as string)
      : undefined;
  const firstHealthyServerId = entry.server_ids.find(
    (sid) => serverIdMap.get(sid)?.healthy,
  );
  const chosenServerId = firstHealthyServerId ?? entry.server_ids[0] ?? null;
  const chosenServer = chosenServerId
    ? serverIdMap.get(chosenServerId) ?? null
    : null;
  // Click is only meaningful when we have both a server to route to
  // *and* a path. Either missing → render an inert row (no pointer
  // cursor, no handler) so the user isn't misled.
  const clickable = !!onOpenInExplore && !!chosenServer && !!path;
  const onClick = clickable
    ? () => {
        onOpenInExplore!({
          server_label: chosenServer!.label,
          server_base_url: chosenServer!.base_url,
          cluster_server_id: chosenServer!.server_id,
          load: {
            path,
            name: loadArgsName,
            split: loadArgsSplit,
          },
          display: friendlyName ?? entry.dataset_id,
          hint_rows: entry.length,
        });
      }
    : undefined;
  const title = clickable
    ? `Open in Explore via ${chosenServer!.base_url}` +
      (entry.server_ids.length > 1
        ? ` (first of ${entry.server_ids.length} hosts)`
        : "")
    : undefined;
  return (
    <tr
      onClick={onClick}
      style={clickable ? { cursor: "pointer" } : undefined}
      title={title}
    >
      <td style={{ fontFamily: "monospace" }}>
        {entry.dataset_id}
        {friendlyName && friendlyName !== entry.dataset_id && (
          <span className="muted" style={{ marginLeft: 6 }}>
            ({friendlyName})
          </span>
        )}
      </td>
      <td>{entry.source}</td>
      <td title={entry.length != null ? fmtCount(entry.length) : ""}>
        {fmtCountCompact(entry.length)}
      </td>
      <td>{fmtBytes(entry.size_bytes)}</td>
      <td title={hostsTooltip}>
        {entry.server_ids.length}
        {hostBaseUrls.length > 0 && (
          <span
            className="muted"
            style={{
              marginLeft: 6,
              fontSize: "0.85em",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              maxWidth: 240,
              display: "inline-block",
              verticalAlign: "middle",
            }}
          >
            ({hostBaseUrls.join(", ")})
          </span>
        )}
      </td>
    </tr>
  );
}

interface DatasetServersTabProps {
  /** Click-through for table rows / split links. Builds a SelectedLeaf
   *  in the right shape for the Explore tab to seed its own state. */
  onOpenInExplore: (leaf: SelectedLeaf) => void;
}

function DatasetServersTab({ onOpenInExplore }: DatasetServersTabProps) {
  const demoMode = useDemoMode();
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

  // Hide JobRecord entries for servers whose process has exited — they
  // can't be queried and only confuse the operator. The selection-
  // cleanup useEffect below transparently clears any selection on a
  // server that disappears here.
  const localServers = (localsQ.data ?? []).filter((s) => s.alive);
  const userServers = usersQ.data ?? [];

  // When the selected entry disappears (e.g. a spawned server exits,
  // or the user deletes their entry), clear the selection so the
  // action buttons don't fire against a stale URL. useEffect so the
  // setState happens after commit rather than during render.
  useEffect(() => {
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
  };
  const onPickUser = (s: DatasetServerUser) => {
    setSelected({
      key: { kind: "user", id: s.id },
      base_url: s.base_url,
      label: s.label,
      has_auth_token: s.has_auth_token,
      alive: null,
    });
  };

  const removeUser = useMutation({
    mutationFn: (id: string) => api.deleteUserDatasetServer(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-servers-user"] });
    },
  });

  // Mint a forgather-dataset:// URI on the server side (the token lives
  // in JobRecords, not the browser) and write the result to the
  // clipboard. The "bundle" is two pieces of state in one string so the
  // destination machine's "+ Add" → "Paste bundle" can fill URL + token
  // in a single step. See AddServerModal for the parser.
  const copyLocalBundle = async (queue_id: string) => {
    try {
      const { bundle } = await api.localDatasetServerBundle(queue_id);
      await navigator.clipboard?.writeText(bundle);
    } catch (e) {
      window.alert(
        `Could not copy bundle: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  return (
    <div className="inference-model-panel">
      <section>
        <h4 className="dyn-heading">
          Spawned dataset servers
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
              keyMatches(selected.key, {
                kind: "local",
                queue_id: s.queue_id,
              });
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
                  {s.alive && !demoMode && (
                    <button
                      className="secondary"
                      style={{ marginLeft: "auto" }}
                      onClick={(e) => {
                        e.stopPropagation();
                        void copyLocalBundle(s.queue_id);
                      }}
                      title={
                        "Copy a forgather-dataset:// URI containing the " +
                        "URL and token. Paste it into '+ Add server' on " +
                        "another node."
                      }
                    >
                      Copy bundle
                    </button>
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
            style={{ marginLeft: 12 }}
            onClick={() => setAddOpen(true)}
            disabled={demoMode}
            title={
              demoMode
                ? "Read-only demo mode — server registry is locked"
                : "Register a remote dataset_server URL"
            }
          >
            + Add server
          </button>
        </h4>
        {userServers.length === 0 && (
          <div className="muted pane-state-small">
            No user-added servers. Use “+ Add server” to register a remote URL.
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
                    // ``destructive`` class so the global demo-mode CSS
                    // rule disables this row's remove control.
                    className="tiny destructive"
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
        // Key on base_url so switching to a different server (or one
        // whose host/port changed) remounts the component — the old
        // queries would otherwise be associated with the wrong server.
        <ServerActions
          key={selected.base_url}
          selected={selected}
          onOpenInExplore={onOpenInExplore}
        />
      )}

      {addOpen && (
        <AddServerModal
          onClose={() => setAddOpen(false)}
          onAdded={() => {
            // Kick the master collect/refresh loops so the cluster
            // inventory reflects the new entry within ~1s instead
            // of waiting up to one collect tick. Both queries get
            // invalidated; the cluster query waits a beat for the
            // master to re-poll before refetching.
            void api.refreshClusterDatasetServers();
            qc.invalidateQueries({ queryKey: ["dataset-servers-user"] });
            qc.invalidateQueries({
              queryKey: ["cluster", "dataset_servers"],
            });
            qc.invalidateQueries({
              queryKey: ["cluster", "dataset_inventory"],
            });
            setAddOpen(false);
          }}
        />
      )}
    </div>
  );
}

interface ServerActionsProps {
  selected: SelectedServer;
  onOpenInExplore: (leaf: SelectedLeaf) => void;
}

/** Stacked detail view for a selected dataset_server. Renders three
 *  panels concurrently — Status, HF Cache, Local — each driven by its
 *  own query so a slow upstream on one endpoint doesn't block the
 *  others. The single "↻ Refresh" button re-fetches all three at
 *  once; the auth-token lookup (JobRecord for spawned servers,
 *  registry for user-added) happens server-side in the proxy. */
function ServerActions({ selected, onOpenInExplore }: ServerActionsProps) {
  const qc = useQueryClient();
  const base = selected.base_url;
  // Token argument is empty: the proxy resolves auth from JobRecord
  // (spawned) or the user registry (added). The override-token
  // input was removed when tokens started persisting across restarts.
  const tok = "";

  const statusQ = useQuery({
    queryKey: ["dataset-server", base, "status"],
    queryFn: () => api.datasetServerHealth(base, tok),
  });
  const cacheQ = useQuery({
    queryKey: ["dataset-server", base, "cache"],
    queryFn: () => api.datasetServerCache(base, tok),
  });
  const localQ = useQuery({
    queryKey: ["dataset-server", base, "local"],
    queryFn: () => api.datasetServerLocal(base, tok),
  });

  const refreshing =
    statusQ.isFetching || cacheQ.isFetching || localQ.isFetching;
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ["dataset-server", base] });
  };

  return (
    <section>
      <h4 className="dyn-heading" style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span>
          Selected:{" "}
          <code style={{ marginLeft: 6 }}>{selected.base_url}</code>
        </span>
        <button
          onClick={refresh}
          disabled={refreshing}
          title="Re-fetch Status, HF Cache, and Local for the selected server"
        >
          {refreshing ? "Refreshing…" : "↻ Refresh"}
        </button>
      </h4>

      <PanelBlock title="Status" query={statusQ}>
        {(data) => <StatusCard data={data} />}
      </PanelBlock>
      <PanelBlock title="HF Cache" query={cacheQ}>
        {(data) => (
          <HFCacheTable
            data={data}
            server={selected}
            onOpenInExplore={onOpenInExplore}
          />
        )}
      </PanelBlock>
      <PanelBlock title="Local" query={localQ}>
        {(data) => (
          <LocalTable
            data={data}
            server={selected}
            onOpenInExplore={onOpenInExplore}
          />
        )}
      </PanelBlock>
    </section>
  );
}

/** Heading + loading/error/success boilerplate shared by the three
 *  per-server panels. ``children`` is the success-state renderer; we
 *  hand it the unwrapped data once the query resolves. */
function PanelBlock<T>({
  title,
  query,
  children,
}: {
  title: string;
  query: { data: T | undefined; isLoading: boolean; isFetching: boolean; error: unknown; dataUpdatedAt: number };
  children: (data: T) => React.ReactNode;
}) {
  const errMsg =
    query.error instanceof Error
      ? query.error.message
      : query.error
        ? String(query.error)
        : null;
  return (
    <div style={{ marginTop: 12 }}>
      <div
        className="muted"
        style={{
          fontSize: 11,
          marginBottom: 4,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <strong style={{ color: "inherit" }}>{title}</strong>
        {query.dataUpdatedAt > 0 && (
          <span>
            · fetched{" "}
            {new Date(query.dataUpdatedAt).toLocaleTimeString()}
          </span>
        )}
        {query.isFetching && <span>· refreshing…</span>}
      </div>
      {errMsg ? (
        <pre className="pane-state err" style={{ whiteSpace: "pre-wrap" }}>
          {errMsg}
        </pre>
      ) : query.data === undefined ? (
        query.isLoading ? (
          <div className="muted pane-state-small">Loading…</div>
        ) : null
      ) : (
        children(query.data)
      )}
    </div>
  );
}

type SortDir = "asc" | "desc";

interface SortState<K extends string> {
  by: K;
  dir: SortDir;
}

/** Click handler for sortable column headers. Cycles dir if the column
 *  is already active; switches to the new column with the supplied
 *  default direction otherwise. Default direction is "desc" for numeric
 *  columns (the operator usually wants "largest first") and "asc" for
 *  text (alphabetical). */
function makeSortToggle<K extends string>(
  current: SortState<K>,
  set: (s: SortState<K>) => void,
) {
  return (col: K, defaultDir: SortDir = "asc") => {
    if (current.by === col) {
      set({ by: col, dir: current.dir === "asc" ? "desc" : "asc" });
    } else {
      set({ by: col, dir: defaultDir });
    }
  };
}

interface SortableHeaderProps<K extends string> {
  col: K;
  label: string;
  current: SortState<K>;
  toggle: (col: K, defaultDir?: SortDir) => void;
  defaultDir?: SortDir;
}

function SortableHeader<K extends string>({
  col,
  label,
  current,
  toggle,
  defaultDir,
}: SortableHeaderProps<K>) {
  const active = current.by === col;
  const arrow = active ? (current.dir === "asc" ? " ▲" : " ▼") : "";
  return (
    <th
      className={"sortable" + (active ? " active" : "")}
      onClick={() => toggle(col, defaultDir)}
      style={{ cursor: "pointer", userSelect: "none" }}
      title={`Sort by ${label}`}
    >
      {label}
      {arrow}
    </th>
  );
}

/** Sum num_examples across a split list. Treats missing values as 0
 *  so a row with partial metadata still sorts. */
function sumSplitRows(
  splits: Array<{ num_examples?: number | null } | { rows?: number | null }>,
): number {
  let total = 0;
  for (const s of splits) {
    const v =
      "num_examples" in s ? s.num_examples : "rows" in s ? s.rows : null;
    if (typeof v === "number") total += v;
  }
  return total;
}

/** Format bytes as a short human string. KB / MB / GB / TB. */
function fmtBytes(n: number | null | undefined): string {
  if (n == null || n === 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

function fmtCount(n: number | null | undefined): string {
  if (n == null) return "—";
  return n.toLocaleString();
}

/** Compact human form of a row count: ``1234`` → ``1.2K``, ``2119719`` →
 *  ``2.1M``. Used for the Cluster tab's Datasets summary column where
 *  the full locale-formatted number would crowd the layout. */
function fmtCountCompact(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 1000) return String(n);
  const units = ["", "K", "M", "B", "T"];
  let i = 0;
  let v = n;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)}${units[i]}`;
}

/** Truncate a string to ``max`` chars, appending "…" when cut. */
function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + "…" : s;
}

/** Status pane. Card with the service line + a row of policy chips
 *  colored by what each setting means for trust:
 *  - auth_required false → red (anyone reachable can pull)
 *  - hf_cache_enabled true / false → neutral (default true is fine)
 *  - allow_paths true → amber (server leaks fs layout to clients)
 *  - allow_downloads true → amber (clients can fill disk via the server)
 *  Everything safe-default stays neutral. */
function StatusCard({ data }: { data: DatasetServerHealth }) {
  const p = data.policy;
  const status = data.status;
  const statusClass =
    status === "ok" ? "chip chip-ok" : "chip chip-warn";
  return (
    <div className="ds-status-card">
      <div className="ds-status-head">
        <span className={statusClass}>{status.toUpperCase()}</span>
        <strong>{data.service}</strong>
        <span className="muted">v{data.version}</span>
      </div>
      <div className="ds-status-policy">
        <span
          className={p.auth_required ? "chip chip-ok" : "chip chip-danger"}
          title={
            p.auth_required
              ? "Bearer token required for /v1/* endpoints"
              : "--no-auth is set; any host that can reach the bind " +
                "port can pull datasets"
          }
        >
          {p.auth_required ? "auth required" : "auth disabled"}
        </span>
        <span
          className={p.hf_cache_enabled ? "chip" : "chip chip-muted"}
          title={
            p.hf_cache_enabled
              ? "HuggingFace cache loads are allowed"
              : "--no-hf is set; only local/* mappings are servable"
          }
        >
          HF cache: {p.hf_cache_enabled ? "enabled" : "disabled"}
        </span>
        <span
          className={p.allow_paths ? "chip chip-warn" : "chip"}
          title={
            p.allow_paths
              ? "--allow-paths is set; clients can request loads by " +
                "absolute filesystem path, leaking server fs layout"
              : "Path-based loads are refused; only HF cache + local/* " +
                "mappings are loadable"
          }
        >
          paths: {p.allow_paths ? "allowed" : "off"}
        </span>
        <span
          className={p.allow_downloads ? "chip chip-warn" : "chip"}
          title={
            p.allow_downloads
              ? "--allow-downloads is set; cache misses trigger HF " +
                "downloads (clients can fill server disk / bandwidth)"
              : "Cache-only; misses surface as 404 instead of pulling " +
                "from HuggingFace"
          }
        >
          downloads: {p.allow_downloads ? "allowed" : "off"}
        </span>
        <span className="chip">
          local datasets: {p.local_count}
        </span>
      </div>
    </div>
  );
}

type HFCacheSortKey = "repo" | "config" | "version" | "splits" | "size";

interface HFFlatRow {
  repo: string;
  config: string;
  version: string;
  splits: { name: string; rows: number | null }[];
  size_bytes: number | null;
}

/** HF Cache table. One row per (repo, config). Splits are shown
 *  inline so the user sees row counts without an extra drill-in.
 *  Each split name in the cell is clickable and opens that split in
 *  the Explore tab. */
function HFCacheTable({
  data,
  server,
  onOpenInExplore,
}: {
  data: HFCacheResponse;
  server: SelectedServer;
  onOpenInExplore: (leaf: SelectedLeaf) => void;
}) {
  const datasets = data.datasets ?? [];
  const totalSize = datasets.reduce(
    (acc, d) => acc + (d.size_bytes ?? 0),
    0,
  );
  // Flatten into (repo, config) rows for tabular rendering.
  const rows: HFFlatRow[] = useMemo(() => {
    const out: HFFlatRow[] = [];
    for (const d of datasets) {
      if (!d.configs || d.configs.length === 0) {
        out.push({
          repo: d.repo,
          config: "—",
          version: "—",
          splits: [],
          size_bytes: d.size_bytes ?? null,
        });
        continue;
      }
      for (const c of d.configs) {
        out.push({
          repo: d.repo,
          config: c.config,
          version: c.version ?? "—",
          splits: c.splits.map((s) => ({
            name: s.name,
            rows: s.num_examples ?? null,
          })),
          size_bytes: c.size_bytes ?? null,
        });
      }
    }
    return out;
  }, [datasets]);

  const [sort, setSort] = useState<SortState<HFCacheSortKey>>({
    by: "size",
    dir: "desc",
  });
  const toggle = makeSortToggle(sort, setSort);
  const sorted = useMemo(() => {
    const cmp = (a: HFFlatRow, b: HFFlatRow): number => {
      let r = 0;
      switch (sort.by) {
        case "repo":
          r = a.repo.localeCompare(b.repo);
          break;
        case "config":
          r = a.config.localeCompare(b.config);
          break;
        case "version":
          r = a.version.localeCompare(b.version);
          break;
        case "splits":
          r = sumSplitRows(a.splits) - sumSplitRows(b.splits);
          break;
        case "size":
          r = (a.size_bytes ?? 0) - (b.size_bytes ?? 0);
          break;
      }
      return sort.dir === "asc" ? r : -r;
    };
    return [...rows].sort(cmp);
  }, [rows, sort]);

  const openSplit = (row: HFFlatRow, splitName: string, splitRows: number | null) => {
    // ``cluster_server_id`` is intentionally omitted — the Servers
    // tab no longer routes through the cluster proxy. Cluster-wide
    // selection lives on the Cluster view's Datasets tab.
    onOpenInExplore({
      server_label: server.label,
      server_base_url: server.base_url,
      load: { path: row.repo, name: row.config, split: splitName },
      display: `${row.repo} · ${row.config} · ${splitName}`,
      hint_rows: splitRows,
    });
  };

  return (
    <div className="ds-section">
      <div className="muted ds-section-summary">
        cache root: <code>{data.cache_root}</code> · {datasets.length}{" "}
        dataset{datasets.length === 1 ? "" : "s"} · {fmtBytes(totalSize)}
      </div>
      {rows.length === 0 ? (
        <div className="pane-state muted">HF cache is empty on this host.</div>
      ) : (
        <>
          <DatasetBarChart
            title="Size distribution across the cache"
            items={rows.map((r, i) => ({
              key: `${r.repo}:${r.config}:${i}`,
              code: r.repo,
              muted: r.config,
              size: r.size_bytes ?? 0,
            }))}
          />
          <div className="preview-table-wrap">
            <table className="preview-table ds-hf-table">
              <thead>
                <tr>
                  <SortableHeader
                    col="repo"
                    label="Repo"
                    current={sort}
                    toggle={toggle}
                  />
                  <SortableHeader
                    col="config"
                    label="Config"
                    current={sort}
                    toggle={toggle}
                  />
                  <SortableHeader
                    col="version"
                    label="Version"
                    current={sort}
                    toggle={toggle}
                  />
                  <SortableHeader
                    col="splits"
                    label="Splits"
                    current={sort}
                    toggle={toggle}
                    defaultDir="desc"
                  />
                  <SortableHeader
                    col="size"
                    label="Size"
                    current={sort}
                    toggle={toggle}
                    defaultDir="desc"
                  />
                </tr>
              </thead>
              <tbody>
                {sorted.map((r, i) => (
                  <tr key={`${r.repo}:${r.config}:${i}`}>
                    <td>
                      <code>{r.repo}</code>
                    </td>
                    <td>{r.config}</td>
                    <td className="muted">{r.version}</td>
                    <td>
                      {r.splits.length === 0 ? (
                        <span className="muted">—</span>
                      ) : (
                        <span className="preview-cell">
                          {r.splits.map((s, idx) => (
                            <span key={s.name}>
                              {idx > 0 && ", "}
                              <a
                                className="ds-split-link"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openSplit(r, s.name, s.rows);
                                }}
                                title={`Open ${r.repo} · ${r.config} · ${s.name} in Explore`}
                              >
                                {s.name}
                                {s.rows != null && `: ${fmtCount(s.rows)}`}
                              </a>
                            </span>
                          ))}
                        </span>
                      )}
                    </td>
                    <td className="row-index">{fmtBytes(r.size_bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

/** One bar segment in the size-distribution chart. ``key`` must be
 *  stable across renders; ``code`` is rendered with code styling
 *  (typically a repo / dataset name), ``muted`` is the trailing
 *  qualifier (config, dataset_name, etc.). */
interface BarItem {
  key: string;
  code: string;
  muted: string;
  size: number;
}

/** Horizontal stacked bars showing each item's share of the total.
 *  Top-K segments; the rest collapse into "+ N more". Generic so both
 *  HF Cache and Local tables can share the rendering. */
function DatasetBarChart({
  items,
  title,
}: {
  items: BarItem[];
  title: string;
}) {
  const TOP_N = 8;
  const sized = items.filter((r) => r.size > 0).sort((a, b) => b.size - a.size);
  const totalSize = sized.reduce((acc, r) => acc + r.size, 0);
  if (totalSize === 0 || sized.length === 0) return null;
  const top = sized.slice(0, TOP_N);
  const tail = sized.slice(TOP_N);
  const tailSize = tail.reduce((a, b) => a + b.size, 0);

  // Stable palette: cycle through a small set tinted off the accent
  // color so the bars look consistent across re-renders without
  // dragging in a chart library.
  const palette = [
    "var(--accent)",
    "#3fb950",
    "#f78166",
    "#a371f7",
    "#56d4dd",
    "#d2a8ff",
    "#ffa657",
    "#79c0ff",
  ];

  return (
    <div className="ds-bar-chart" title={title}>
      <div className="ds-bar-row">
        {top.map((r, i) => {
          const pct = (r.size / totalSize) * 100;
          return (
            <span
              key={r.key}
              className="ds-bar-seg"
              style={{
                flexBasis: `${pct}%`,
                background: palette[i % palette.length],
              }}
              title={`${r.code}${r.muted ? " · " + r.muted : ""} — ${fmtBytes(r.size)} (${pct.toFixed(1)}%)`}
            />
          );
        })}
        {tailSize > 0 && (
          <span
            className="ds-bar-seg ds-bar-tail"
            style={{ flexBasis: `${(tailSize / totalSize) * 100}%` }}
            title={`+ ${tail.length} more — ${fmtBytes(tailSize)}`}
          />
        )}
      </div>
      <div className="ds-bar-legend">
        {top.map((r, i) => (
          <span key={r.key} className="ds-bar-legend-item">
            <span
              className="ds-bar-swatch"
              style={{ background: palette[i % palette.length] }}
            />
            <code>{r.code}</code>
            {r.muted && <span className="muted"> · {r.muted}</span>}
            <span className="muted"> · {fmtBytes(r.size)}</span>
          </span>
        ))}
        {tail.length > 0 && (
          <span className="ds-bar-legend-item muted">
            <span className="ds-bar-swatch ds-bar-tail" />+ {tail.length} more
            ({fmtBytes(tailSize)})
          </span>
        )}
      </div>
    </div>
  );
}

type LocalSortKey =
  | "name"
  | "path"
  | "layout"
  | "config"
  | "splits"
  | "features"
  | "size";

/** Local mappings table. One row per registered ``local/<name>``.
 *  Split names in the splits cell are clickable and open that split
 *  in the Explore tab. */
function LocalTable({
  data,
  server,
  onOpenInExplore,
}: {
  data: LocalListResponse;
  server: SelectedServer;
  onOpenInExplore: (leaf: SelectedLeaf) => void;
}) {
  const rows = data.local ?? [];
  const totalSize = rows.reduce((acc, r) => acc + (r.size_bytes ?? 0), 0);

  const [sort, setSort] = useState<SortState<LocalSortKey>>({
    by: "size",
    dir: "desc",
  });
  const toggle = makeSortToggle(sort, setSort);
  const sorted = useMemo(() => {
    const cmp = (a: LocalDatasetEntry, b: LocalDatasetEntry): number => {
      let r = 0;
      switch (sort.by) {
        case "name":
          r = a.name.localeCompare(b.name);
          break;
        case "path":
          r = a.path.localeCompare(b.path);
          break;
        case "layout":
          r = (a.layout ?? "").localeCompare(b.layout ?? "");
          break;
        case "config":
          r = (a.config_name ?? "").localeCompare(b.config_name ?? "");
          break;
        case "splits":
          r = sumSplitRows(a.splits ?? []) - sumSplitRows(b.splits ?? []);
          break;
        case "features":
          r = (a.features?.length ?? 0) - (b.features?.length ?? 0);
          break;
        case "size":
          r = (a.size_bytes ?? 0) - (b.size_bytes ?? 0);
          break;
      }
      return sort.dir === "asc" ? r : -r;
    };
    return [...rows].sort(cmp);
  }, [rows, sort]);

  if (rows.length === 0) {
    return (
      <div className="pane-state muted">
        No named local datasets registered on this server. Start the
        dataset_server with one or more <code>--local NAME=PATH</code>{" "}
        flags (or add them to the YAML config).
      </div>
    );
  }

  const openSplit = (
    row: LocalDatasetEntry,
    splitName: string,
    splitRows: number | null,
  ) => {
    // See HFCacheTable: cluster_server_id is gone with the cluster
    // branch — Cluster-wide selection lives on the Cluster view.
    onOpenInExplore({
      server_label: server.label,
      server_base_url: server.base_url,
      load: { path: `local/${row.name}`, split: splitName },
      display: `local/${row.name}${row.config_name ? ` · ${row.config_name}` : ""} · ${splitName}`,
      hint_rows: splitRows,
    });
  };

  return (
    <div className="ds-section">
      <div className="muted ds-section-summary">
        {rows.length} mapping{rows.length === 1 ? "" : "s"} · {fmtBytes(totalSize)}
      </div>
      <DatasetBarChart
        title="Size distribution across local mappings"
        items={rows.map((r) => ({
          key: r.name,
          code: `local/${r.name}`,
          // Use config_name when present; fall back to dataset_name
          // so the legend is informative for both layouts.
          muted: r.config_name ?? r.dataset_name ?? "",
          size: r.size_bytes ?? 0,
        }))}
      />
      <div className="preview-table-wrap">
        <table className="preview-table ds-local-table">
          <thead>
            <tr>
              <SortableHeader col="name" label="Name" current={sort} toggle={toggle} />
              <SortableHeader col="path" label="Path" current={sort} toggle={toggle} />
              <SortableHeader col="layout" label="Layout" current={sort} toggle={toggle} />
              <SortableHeader col="config" label="Config" current={sort} toggle={toggle} />
              <SortableHeader
                col="splits"
                label="Splits"
                current={sort}
                toggle={toggle}
                defaultDir="desc"
              />
              <SortableHeader
                col="features"
                label="Features"
                current={sort}
                toggle={toggle}
                defaultDir="desc"
              />
              <SortableHeader
                col="size"
                label="Size"
                current={sort}
                toggle={toggle}
                defaultDir="desc"
              />
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <LocalRow
                key={r.name}
                row={r}
                onOpenSplit={openSplit}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LocalRow({
  row,
  onOpenSplit,
}: {
  row: LocalDatasetEntry;
  onOpenSplit: (
    row: LocalDatasetEntry,
    splitName: string,
    splitRows: number | null,
  ) => void;
}) {
  const splits = row.splits ?? [];
  const features = row.features ?? [];
  // Features as small, comma-separated text — datasets can carry
  // dozens of columns, and a wall of pill chips made the row height
  // explode. Truncate the visible text; full list lands in the title
  // attribute for hover.
  const featuresText = features.join(", ");
  const featuresTruncated = truncate(featuresText, 80);
  return (
    <tr>
      <td>
        <code>local/{row.name}</code>
      </td>
      <td title={row.path}>
        <span className="preview-cell">{truncate(row.path, 40)}</span>
      </td>
      <td className="muted">
        {row.layout ?? "—"}
        {row.layout === "missing" && (
          <span className="chip chip-warn" style={{ marginLeft: 6 }}>
            path gone
          </span>
        )}
      </td>
      <td className="muted">
        {row.config_name ?? "—"}
        {row.dataset_name ? ` · ${row.dataset_name}` : ""}
      </td>
      <td>
        {splits.length === 0 ? (
          <span className="muted">—</span>
        ) : (
          <span className="preview-cell">
            {splits.map((s, idx) => (
              <span key={s.name}>
                {idx > 0 && ", "}
                <a
                  className="ds-split-link"
                  onClick={(e) => {
                    e.stopPropagation();
                    onOpenSplit(row, s.name, s.num_examples ?? null);
                  }}
                  title={`Open local/${row.name} · ${s.name} in Explore`}
                >
                  {s.name}
                  {s.num_examples != null && `: ${fmtCount(s.num_examples)}`}
                </a>
              </span>
            ))}
          </span>
        )}
      </td>
      <td
        className="ds-features-cell muted"
        title={featuresText || undefined}
      >
        {featuresText ? featuresTruncated : <span className="muted">—</span>}
      </td>
      <td className="row-index">{fmtBytes(row.size_bytes)}</td>
    </tr>
  );
}

/** Decode a ``forgather-dataset://host:port/?token=...`` bundle into
 *  ``{base_url, token}``. The URI shape is produced by
 *  ``/api/dataset-servers/local/<queue_id>/bundle`` on the source
 *  machine. Strict-ish parsing: scheme must match, host + port must be
 *  present, query string must carry a token (empty token allowed but
 *  surfaces as ""). Anything malformed raises so the caller can show
 *  the user a specific error rather than silently accepting garbage. */
function parseBundle(raw: string): { base_url: string; token: string } {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("forgather-dataset://")) {
    throw new Error(
      "bundle must start with forgather-dataset:// (use Copy bundle on the source server)",
    );
  }
  // Force a parseable scheme; URL() rejects custom schemes for
  // hostname/pathname extraction in some browsers, so rewrite to http://
  // for parsing only — we keep the original scheme out of the result.
  let parsed: URL;
  try {
    parsed = new URL("http://" + trimmed.slice("forgather-dataset://".length));
  } catch (e) {
    throw new Error(
      `could not parse bundle: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  if (!parsed.hostname || !parsed.port) {
    throw new Error("bundle is missing host or port");
  }
  const base_url = `http://${parsed.hostname}:${parsed.port}`;
  const token = parsed.searchParams.get("token") ?? "";
  return { base_url, token };
}

function AddServerModal({
  onClose,
  onAdded,
}: {
  onClose: () => void;
  onAdded: () => void;
}) {
  const demoMode = useDemoMode();
  const [label, setLabel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [authToken, setAuthToken] = useState("");
  const [showAuthToken, setShowAuthToken] = useState(false);
  // Per-entry TLS policy. Default secure (verify chain + hostname);
  // operator can opt out for SSH-tunneled / out-of-band-secured
  // remotes whose cert doesn't validate against the local CA.
  const [verifyTls, setVerifyTls] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pasteBundle = async () => {
    try {
      const text = await navigator.clipboard?.readText();
      if (!text) {
        setError("clipboard is empty");
        return;
      }
      const { base_url, token } = parseBundle(text);
      setBaseUrl(base_url);
      setAuthToken(token);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const submit = async () => {
    setPending(true);
    setError(null);
    try {
      const req: AddDatasetServerRequest = {
        label: label.trim(),
        base_url: baseUrl.trim(),
        auth_token: authToken.trim(),
        verify_tls: verifyTls,
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
        {/* Wrap inputs in a <form> with autoComplete="off" plus the
            new-password trick on the token field so Chrome doesn't try
            to autofill the URL as a username for a saved password. The
            onSubmit handler is wired to the Add button so Enter still
            submits. */}
        <form
          className="modal-body"
          autoComplete="off"
          onSubmit={(e) => {
            e.preventDefault();
            if (!pending && baseUrl.trim()) void submit();
          }}
        >
          <div className="submit-row">
            <button
              type="button"
              className="secondary"
              onClick={() => void pasteBundle()}
              title={
                "Read a forgather-dataset:// bundle from the clipboard " +
                "and fill URL + token in one step. Get one by clicking " +
                "'Copy bundle' on the source machine's local-server row."
              }
            >
              Paste bundle from clipboard
            </button>
          </div>
          <div className="submit-row">
            <label className="wide">
              Label
              <input
                type="text"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. dataset host"
                autoComplete="off"
                name="ds-label"
              />
            </label>
          </div>
          <div className="submit-row">
            <label className="wide">
              Base URL
              <input
                type="text"
                className="wide"
                inputMode="url"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://datahost:8766"
                autoComplete="off"
                spellCheck={false}
                name="ds-base-url"
              />
            </label>
          </div>
          {/* Hint sits below the row so it doesn't fight the input for
              horizontal flex space (which is what made the input look
              ~10% wide before). */}
          <div className="muted" style={{ marginTop: -4, marginBottom: 10 }}>
            Loopback + URLs you add here are allowed; everything else is
            refused by the proxy. The URL list is the authorization
            decision — only add servers you trust. Every byte they return
            flows into your training pipeline. User-added URLs aren't
            actively probed for reachability — click Status after
            adding to confirm. See the dataset_server README's “Security
            considerations” for the full trust story.
          </div>
          <div className="submit-row">
            <label className="wide">
              Auth token
              {/* path-field stretches the input to fit a 64-hex bearer
                  and parks the Show / Copy buttons inline — same pattern
                  the Inference Model panel uses. */}
              <div className="path-field">
                <input
                  // Mirror AddInferenceServerModal: in demo mode force
                  // masked + read-only and hide Show / Copy so a
                  // pre-filled token can't be revealed.
                  type={demoMode || !showAuthToken ? "password" : "text"}
                  className="wide"
                  value={demoMode ? "" : authToken}
                  onChange={(e) => setAuthToken(e.target.value)}
                  readOnly={demoMode}
                  placeholder={
                    demoMode
                      ? "Token entry disabled in demo mode"
                      : "optional — leave blank if the server runs --no-auth"
                  }
                  autoComplete="new-password"
                  spellCheck={false}
                  name="ds-auth-token"
                />
                {!demoMode && (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => setShowAuthToken((v) => !v)}
                    title={showAuthToken ? "Hide token" : "Show token"}
                  >
                    {showAuthToken ? "Hide" : "Show"}
                  </button>
                )}
                {!demoMode && (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => {
                      if (!authToken) return;
                      navigator.clipboard?.writeText(authToken).catch(() => {});
                    }}
                    disabled={!authToken}
                    title="Copy token to clipboard"
                  >
                    Copy
                  </button>
                )}
              </div>
            </label>
          </div>
          <div className="submit-row">
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={verifyTls}
                onChange={(e) => setVerifyTls(e.target.checked)}
              />
              <span>Verify TLS chain + hostname</span>
            </label>
            {!verifyTls && (
              <div
                className="muted"
                style={{
                  marginTop: 4,
                  paddingLeft: 24,
                  // Make the warning visually distinct without
                  // shouting — operator already chose this.
                  color: "var(--warning, #b87000)",
                }}
              >
                ⚠ Chain validation off. The upstream cert is no longer
                authenticated by TLS — only enable this when the
                channel is secured by other means (SSH tunnel, VPN,
                air-gapped LAN). Bearer auth and any downstream load
                policy still apply.
              </div>
            )}
          </div>
        </form>
        <footer className="modal-footer">
          <div className="muted current-path">{error ?? ""}</div>
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void submit()}
              disabled={demoMode || pending || !baseUrl.trim()}
              title={
                demoMode
                  ? "Read-only demo mode — try the live tool to register a server"
                  : undefined
              }
            >
              {pending ? "Adding…" : "Add"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
