import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import {
  AddDatasetServerRequest,
  DatasetHandleRow,
  DatasetHandlesResponse,
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

type SubTab = "servers" | "explore";

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

/** Top-level Datasets view. Tabs: Servers (CRUD + status/handles/cache),
 *  Explore (tree of dataset → split → table of rows). */
export function DatasetsPanel() {
  const [tab, setTab] = useState<SubTab>("servers");
  // Pending pre-selection for the Explore tab. Set when a row in the
  // Servers tab is clicked (handles row / cache split / local split);
  // the Explore tab consumes it once and signals back to clear.
  const [pendingExplore, setPendingExplore] = useState<SelectedLeaf | null>(
    null,
  );
  const openInExplore = (leaf: SelectedLeaf) => {
    setPendingExplore(leaf);
    setTab("explore");
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
  const localServers = localsQ.data ?? [];
  const userServers = usersQ.data ?? [];
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
          preselect={pendingExplore}
          onPreselectConsumed={() => setPendingExplore(null)}
        />
      </div>
    </div>
  );
}

interface DatasetServersTabProps {
  /** Click-through for table rows / split links. Builds a SelectedLeaf
   *  in the right shape for the Explore tab to seed its own state. */
  onOpenInExplore: (leaf: SelectedLeaf) => void;
}

function DatasetServersTab({ onOpenInExplore }: DatasetServersTabProps) {
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

  const localServers = localsQ.data ?? [];
  const userServers = usersQ.data ?? [];

  // When the selected entry disappears (e.g. a local server exits, or
  // the user deletes their entry), clear the selection so the action
  // buttons don't fire against a stale URL. useEffect so the setState
  // happens after commit rather than during render.
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
                  {s.alive && (
                    <button
                      className="secondary"
                      style={{ marginLeft: "auto" }}
                      onClick={(e) => {
                        e.stopPropagation();
                        void copyLocalBundle(s.queue_id);
                      }}
                      title={
                        "Copy a forgather-dataset:// URI containing the URL " +
                        "and token. Paste it into '+ Add server' on another node."
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
            title="Register a remote dataset_server URL"
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
        // Key on base_url so switching to a different server (or one
        // whose host/port changed) remounts the component — the old
        // metadata + ↻-refresh button would otherwise point at the
        // wrong server. Also nukes any in-flight fetch's resolved
        // setState (the unmounted instance is GC'd).
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
  onOpenInExplore: (leaf: SelectedLeaf) => void;
}

type ResultKind = "status" | "datasets" | "cache" | "local";

/** Tagged union so per-kind renderers can destructure typed data. */
type FetchResult =
  | { kind: "status"; data: DatasetServerHealth; fetched_at: number }
  | { kind: "datasets"; data: DatasetHandlesResponse; fetched_at: number }
  | { kind: "cache"; data: HFCacheResponse; fetched_at: number }
  | { kind: "local"; data: LocalListResponse; fetched_at: number }
  | { kind: ResultKind; data: null; error: string; fetched_at: number };

function ServerActions({ selected, onOpenInExplore }: ServerActionsProps) {
  const [result, setResult] = useState<FetchResult | null>(null);
  const [pending, setPending] = useState<ResultKind | null>(null);

  // Auth is resolved by the proxy: JobRecord auto-lookup for local
  // servers, registry lookup for user-added entries. The "override
  // token" input is gone now that tokens persist across restarts —
  // delete + re-add the entry to change a stored token.
  const tokenToUse = "";

  const runFetch = async (kind: ResultKind) => {
    setPending(kind);
    setResult(null);
    try {
      const base = selected.base_url;
      let r: FetchResult;
      if (kind === "status") {
        const data = await api.datasetServerHealth(base, tokenToUse);
        r = { kind, data, fetched_at: Date.now() };
      } else if (kind === "datasets") {
        const data = await api.datasetServerDatasets(base, tokenToUse);
        r = { kind, data, fetched_at: Date.now() };
      } else if (kind === "cache") {
        const data = await api.datasetServerCache(base, tokenToUse);
        r = { kind, data, fetched_at: Date.now() };
      } else {
        const data = await api.datasetServerLocal(base, tokenToUse);
        r = { kind, data, fetched_at: Date.now() };
      }
      setResult(r);
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

      <div className="submit-row">
        {(
          [
            { kind: "status", label: "Status", title: "GET /v1/health" },
            {
              kind: "datasets",
              label: "Handles",
              title: "GET /v1/datasets — currently loaded handles",
            },
            {
              kind: "cache",
              label: "HF Cache",
              title: "GET /v1/cache/hf — HF cache contents on the server host",
            },
            {
              kind: "local",
              label: "Local",
              title: "GET /v1/local — registered local/* dataset mappings",
            },
          ] as { kind: ResultKind; label: string; title: string }[]
        ).map(({ kind, label, title }) => {
          // Visual states:
          // - pending === kind            → "Refreshing…"
          // - result.kind === kind        → ↻ prefix (click again to refresh)
          // - otherwise                    → bare label
          const active = result?.kind === kind;
          const refreshing = pending === kind;
          const text = refreshing
            ? `${label}…`
            : active
              ? `↻ ${label}`
              : label;
          return (
            <button
              key={kind}
              className={active ? "active" : ""}
              onClick={() => runFetch(kind)}
              disabled={pending !== null}
              title={
                title +
                (active ? "  (click to refresh)" : "")
              }
            >
              {text}
            </button>
          );
        })}
      </div>

      {result && (
        <div style={{ marginTop: 8 }}>
          <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>
            {result.kind} · fetched{" "}
            {new Date(result.fetched_at).toLocaleTimeString()}
            {"error" in result ? " · error" : ""}
          </div>
          {"error" in result ? (
            <pre className="pane-state err" style={{ whiteSpace: "pre-wrap" }}>
              {result.error}
            </pre>
          ) : result.kind === "status" ? (
            <StatusCard data={result.data} />
          ) : result.kind === "datasets" ? (
            <HandlesTable
              data={result.data}
              server={selected}
              onOpenInExplore={onOpenInExplore}
            />
          ) : result.kind === "cache" ? (
            <HFCacheTable
              data={result.data}
              server={selected}
              onOpenInExplore={onOpenInExplore}
            />
          ) : (
            <LocalTable
              data={result.data}
              server={selected}
              onOpenInExplore={onOpenInExplore}
            />
          )}
        </div>
      )}
    </section>
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

/** Compact rendering of a load_args dict for the Handles table.
 *  ``path`` is the only required key; ``name`` and ``split`` follow
 *  when present. Other keys (data_files, revision) are appended
 *  generically. Keeps to one line so the table doesn't grow per-row
 *  heights. */
function formatLoadArgs(args: Record<string, unknown>): string {
  const ordered: string[] = [];
  if (args.path) ordered.push(String(args.path));
  if (args.name) ordered.push(`name=${String(args.name)}`);
  if (args.split) ordered.push(`split=${String(args.split)}`);
  for (const [k, v] of Object.entries(args)) {
    if (k === "path" || k === "name" || k === "split") continue;
    if (v == null) continue;
    ordered.push(`${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`);
  }
  return ordered.join(" · ");
}

type HandleSortKey = "handle" | "length" | "source" | "args";

function HandlesTable({
  data,
  server,
  onOpenInExplore,
}: {
  data: DatasetHandlesResponse;
  server: SelectedServer;
  onOpenInExplore: (leaf: SelectedLeaf) => void;
}) {
  const rows = data.handles ?? [];
  const [sort, setSort] = useState<SortState<HandleSortKey>>({
    by: "length",
    dir: "desc",
  });
  const toggle = makeSortToggle(sort, setSort);

  const sorted = useMemo(() => {
    const cmp = (a: DatasetHandleRow, b: DatasetHandleRow): number => {
      let r = 0;
      switch (sort.by) {
        case "handle":
          r = a.handle.localeCompare(b.handle);
          break;
        case "length":
          r = (a.length ?? 0) - (b.length ?? 0);
          break;
        case "source":
          r = (a.source ?? "").localeCompare(b.source ?? "");
          break;
        case "args":
          r = formatLoadArgs(a.load_args ?? {}).localeCompare(
            formatLoadArgs(b.load_args ?? {}),
          );
          break;
      }
      return sort.dir === "asc" ? r : -r;
    };
    return [...rows].sort(cmp);
  }, [rows, sort]);

  if (rows.length === 0) {
    return (
      <div className="pane-state muted">
        No datasets currently loaded on this server.
        <div style={{ marginTop: 4 }}>
          The handle cache fills as clients call POST /v1/load (typically
          on first read of a training dataset).
        </div>
      </div>
    );
  }

  const openLeaf = (row: DatasetHandleRow) => {
    const args = row.load_args ?? {};
    const path = String(args.path ?? "");
    const name = args.name != null ? String(args.name) : undefined;
    const split = args.split != null ? String(args.split) : undefined;
    if (!path) return;
    const bits = [path];
    if (name) bits.push(name);
    if (split) bits.push(split);
    onOpenInExplore({
      server_label: server.label,
      server_base_url: server.base_url,
      load: { path, name, split },
      display: bits.join(" · "),
      hint_rows: row.length,
    });
  };

  return (
    <div className="preview-table-wrap">
      <table className="preview-table ds-handles-table">
        <thead>
          <tr>
            <SortableHeader
              col="handle"
              label="Handle"
              current={sort}
              toggle={toggle}
            />
            <SortableHeader
              col="length"
              label="Length"
              current={sort}
              toggle={toggle}
              defaultDir="desc"
            />
            <SortableHeader
              col="source"
              label="Source"
              current={sort}
              toggle={toggle}
            />
            <SortableHeader
              col="args"
              label="Load args"
              current={sort}
              toggle={toggle}
            />
          </tr>
        </thead>
        <tbody>
          {sorted.map((h) => (
            <tr
              key={h.handle}
              onClick={() => openLeaf(h)}
              title="Open in Explore"
              style={{ cursor: "pointer" }}
            >
              <td>
                <code title={h.handle}>{truncate(h.handle, 12)}</code>
              </td>
              <td className="row-index">{fmtCount(h.length)}</td>
              <td>
                <span className="muted">{h.source ?? "—"}</span>
              </td>
              <td title={JSON.stringify(h.load_args, null, 2)}>
                <span className="preview-cell">
                  {truncate(formatLoadArgs(h.load_args ?? {}), 120)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
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
  const [label, setLabel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [authToken, setAuthToken] = useState("");
  const [showAuthToken, setShowAuthToken] = useState(false);
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
                  type={showAuthToken ? "text" : "password"}
                  className="wide"
                  value={authToken}
                  onChange={(e) => setAuthToken(e.target.value)}
                  placeholder="optional — leave blank if the server runs --no-auth"
                  autoComplete="new-password"
                  spellCheck={false}
                  name="ds-auth-token"
                />
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setShowAuthToken((v) => !v)}
                  title={showAuthToken ? "Hide token" : "Show token"}
                >
                  {showAuthToken ? "Hide" : "Show"}
                </button>
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
              </div>
            </label>
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
              disabled={pending || !baseUrl.trim()}
            >
              {pending ? "Adding…" : "Add"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
