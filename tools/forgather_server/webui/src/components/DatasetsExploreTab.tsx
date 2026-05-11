import { UseQueryResult, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import {
  DatasetServerLocal,
  DatasetServerUser,
  HFCacheConfig,
  HFCacheRepo,
  HFCacheResponse,
  HFCacheSplit,
  IterResponse,
  LoadRequest,
  LoadResponse,
  LocalDatasetEntry,
  LocalListResponse,
  api,
} from "../api";

const PAGE_SIZE_OPTIONS = [25, 100, 200] as const;
const DEFAULT_PAGE_SIZE = 25;
const CELL_TRUNCATE = 200;
const EXPANDED_CELL_TRUNCATE = 5000;

/** Same glyph the main sidebar toggle uses. Inlined so we don't have
 *  to factor the icon out of App.tsx just for this re-use. */
function TreeToggleIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      aria-hidden="true"
    >
      <rect x="1.5" y="2.5" width="13" height="11" rx="1.2" />
      <line x1="5.5" y1="2.5" x2="5.5" y2="13.5" />
    </svg>
  );
}

/** What the user has selected — a leaf in the tree. ``server_base_url``
 *  pins which dataset_server the load goes to; the rest is the wire
 *  format for ``POST /v1/load``. */
export interface SelectedLeaf {
  server_label: string;
  server_base_url: string;
  load: LoadRequest;
  /** Display string for the right pane header. */
  display: string;
  /** Optional row count we already know from the tree's metadata, so we
   *  can show a count before /load completes. */
  hint_rows?: number | null;
}

interface ServerOption {
  /** "local:<queue_id>" or "user:<id>" — stable identifier for tree
   *  expansion state. */
  key: string;
  label: string;
  base_url: string;
}

interface Props {
  localServers: DatasetServerLocal[];
  userServers: DatasetServerUser[];
  /** Cross-tab navigation: when DatasetsPanel switches to "explore"
   *  after the user clicked a row in the Servers tab, the leaf to
   *  pre-select is set here. The tab consumes it once on mount /
   *  change, then signals back via ``onPreselectConsumed`` so the
   *  parent can clear it (otherwise re-opening the same tab would
   *  re-trigger the seed). */
  preselect?: SelectedLeaf | null;
  onPreselectConsumed?: () => void;
}

export function DatasetsExploreTab({
  localServers,
  userServers,
  preselect,
  onPreselectConsumed,
}: Props) {
  const servers: ServerOption[] = useMemo(() => {
    const local = localServers
      .filter((s) => s.alive)
      .map((s) => ({
        key: `local:${s.queue_id}`,
        label: s.label,
        base_url: s.base_url,
      }));
    const user = userServers.map((s) => ({
      key: `user:${s.id}`,
      label: s.label,
      base_url: s.base_url,
    }));
    return [...local, ...user];
  }, [localServers, userServers]);

  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [selected, setSelected] = useState<SelectedLeaf | null>(null);

  // Cross-tab preselect: when the Servers tab fires a row click,
  // DatasetsPanel sets ``preselect`` and switches the active tab here.
  // Consume the value once, then tell the parent to clear it so a
  // later tab switch doesn't re-seed an old selection.
  useEffect(() => {
    if (!preselect) return;
    setSelected(preselect);
    onPreselectConsumed?.();
  }, [preselect, onPreselectConsumed]);
  // Tree pane collapse — gives the preview pane the full width when the
  // user has already picked a leaf and just wants to read rows.
  const [treeCollapsed, setTreeCollapsed] = useState<boolean>(false);
  const [pageSize, setPageSize] = useState<number>(DEFAULT_PAGE_SIZE);
  const [page, setPage] = useState<number>(0);
  // Reset page when the selection changes — different leaf, fresh state.
  useEffect(() => setPage(0), [selected]);

  // Load query: cached by (server, load_args). Re-clicking the same leaf
  // is a no-op on the server (it hashes the load_args).
  const loadQ = useQuery({
    queryKey: selected
      ? ["ds-load", selected.server_base_url, JSON.stringify(selected.load)]
      : ["ds-load-idle"],
    queryFn: () =>
      api.datasetServerLoad(
        (selected as SelectedLeaf).server_base_url,
        (selected as SelectedLeaf).load,
        "",
      ),
    enabled: !!selected,
  });

  const handle = loadQ.data?.handle;
  const columns = loadQ.data?.column_names ?? null;
  const length = loadQ.data?.length ?? selected?.hint_rows ?? null;
  const totalPages =
    length != null && length > 0 ? Math.ceil(length / pageSize) : 1;

  const iterQ = useQuery({
    queryKey: [
      "ds-iter",
      selected?.server_base_url,
      handle,
      page,
      pageSize,
    ],
    queryFn: () =>
      api.datasetServerIter(
        (selected as SelectedLeaf).server_base_url,
        handle as string,
        page * pageSize,
        pageSize,
        "",
      ),
    enabled: !!handle && !!selected,
  });

  // Switching page size: re-anchor so the first row of the new page is
  // approximately the first row of the previous page — avoids throwing
  // the user back to page 1 every time they bump from 25 to 200.
  const onPageSizeChange = (next: number) => {
    if (next === pageSize) return;
    const firstRow = page * pageSize;
    setPage(Math.floor(firstRow / next));
    setPageSize(next);
  };

  const toggle = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const isOpen = (key: string) => expanded.has(key);

  return (
    <div className="datasets-explore">
      <header className="datasets-explore-header">
        <button
          className="datasets-explore-tree-toggle"
          onClick={() => setTreeCollapsed((v) => !v)}
          title={
            treeCollapsed
              ? "Show the browse pane"
              : "Hide the browse pane and give the row preview full width"
          }
          aria-label={treeCollapsed ? "Show browse pane" : "Hide browse pane"}
        >
          <TreeToggleIcon />
        </button>
        <div className="datasets-explore-header-title">
          <div className="preview-title">
            {selected ? (
              <strong>{selected.display}</strong>
            ) : (
              <span className="muted">
                Pick a dataset split from the browse pane to load the first{" "}
                {DEFAULT_PAGE_SIZE} rows.
              </span>
            )}
          </div>
          {selected && (
            <div className="muted preview-meta">
              {length != null && (
                <span>{length.toLocaleString()} rows · </span>
              )}
              {columns && (
                <span>
                  {columns.length} column{columns.length === 1 ? "" : "s"} ·{" "}
                </span>
              )}
              <span>{selected.server_base_url}</span>
            </div>
          )}
        </div>
        <span className="spacer" />
        {selected && (
          <div className="preview-page-size" title="Rows per page">
            {PAGE_SIZE_OPTIONS.map((n) => (
              <button
                key={n}
                className={n === pageSize ? "active" : ""}
                onClick={() => onPageSizeChange(n)}
              >
                {n}
              </button>
            ))}
          </div>
        )}
      </header>
      <div className="datasets-explore-body">
        <aside
          className={
            "datasets-explore-tree" + (treeCollapsed ? " collapsed" : "")
          }
        >
          {servers.length === 0 ? (
            <div className="muted pane-state-small">
              No reachable dataset servers. Start a local server or add one
              under the Servers tab.
            </div>
          ) : (
            <ul className="explore-tree-root">
              {servers.map((srv) => (
                <ServerNode
                  key={srv.key}
                  server={srv}
                  isOpen={isOpen}
                  toggle={toggle}
                  selected={selected}
                  setSelected={setSelected}
                />
              ))}
            </ul>
          )}
        </aside>
        <main className="datasets-explore-preview">
          {selected ? (
            <PreviewPane
              selected={selected}
              loadQ={loadQ}
              iterQ={iterQ}
              columns={columns}
              page={page}
              setPage={setPage}
              pageSize={pageSize}
              totalPages={totalPages}
            />
          ) : (
            <div className="pane-state muted">No selection.</div>
          )}
        </main>
      </div>
    </div>
  );
}

interface NodeProps {
  isOpen: (key: string) => boolean;
  toggle: (key: string) => void;
  selected: SelectedLeaf | null;
  setSelected: (sel: SelectedLeaf | null) => void;
}

interface ServerNodeProps extends NodeProps {
  server: ServerOption;
}

function ServerNode({
  server,
  isOpen,
  toggle,
  selected,
  setSelected,
}: ServerNodeProps) {
  const key = server.key;
  const open = isOpen(key);
  return (
    <li className="explore-tree-node">
      <button
        className="explore-tree-row server-row"
        onClick={() => toggle(key)}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <span className="explore-server-label">{server.label}</span>
        <span className="muted explore-server-url"> {server.base_url}</span>
      </button>
      {open && (
        <ul className="explore-tree-children">
          <CacheGroup
            serverKey={key}
            base_url={server.base_url}
            server_label={server.label}
            isOpen={isOpen}
            toggle={toggle}
            selected={selected}
            setSelected={setSelected}
          />
          <LocalGroup
            serverKey={key}
            base_url={server.base_url}
            server_label={server.label}
            isOpen={isOpen}
            toggle={toggle}
            selected={selected}
            setSelected={setSelected}
          />
        </ul>
      )}
    </li>
  );
}

interface GroupProps extends NodeProps {
  serverKey: string;
  base_url: string;
  server_label: string;
}

function CacheGroup({
  serverKey,
  base_url,
  server_label,
  isOpen,
  toggle,
  selected,
  setSelected,
}: GroupProps) {
  const key = `${serverKey}:cache`;
  const open = isOpen(key);
  const cacheQ = useQuery({
    queryKey: ["ds-cache", base_url],
    queryFn: () => api.datasetServerCache(base_url, ""),
    enabled: open,
  });
  const repos = (cacheQ.data as HFCacheResponse | undefined)?.datasets ?? [];

  return (
    <li className="explore-tree-node">
      <button
        className="explore-tree-row group-row"
        onClick={() => toggle(key)}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <span>HF Cache</span>
        <span className="muted"> ({open ? repos.length : "…"})</span>
      </button>
      {open && (
        <ul className="explore-tree-children">
          {cacheQ.isLoading && (
            <li className="muted pane-state-small">loading…</li>
          )}
          {cacheQ.error && (
            <li className="muted pane-state-small">
              {String(cacheQ.error)}
            </li>
          )}
          {!cacheQ.isLoading && !cacheQ.error && repos.length === 0 && (
            <li className="muted pane-state-small">cache is empty</li>
          )}
          {repos.map((repo) => (
            <RepoNode
              key={`${key}:${repo.repo}`}
              parentKey={key}
              repo={repo}
              base_url={base_url}
              server_label={server_label}
              isOpen={isOpen}
              toggle={toggle}
              selected={selected}
              setSelected={setSelected}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

interface RepoNodeProps extends NodeProps {
  parentKey: string;
  repo: HFCacheRepo;
  base_url: string;
  server_label: string;
}

function RepoNode({
  parentKey,
  repo,
  base_url,
  server_label,
  isOpen,
  toggle,
  selected,
  setSelected,
}: RepoNodeProps) {
  const key = `${parentKey}:${repo.repo}`;
  const open = isOpen(key);
  return (
    <li className="explore-tree-node">
      <button
        className="explore-tree-row repo-row"
        onClick={() => toggle(key)}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <code>{repo.repo}</code>
        <span className="muted"> ({fmtBytes(repo.size_bytes ?? 0)})</span>
      </button>
      {open && (
        <ul className="explore-tree-children">
          {repo.configs.map((cfg) => (
            <ConfigNode
              key={`${key}:${cfg.config}`}
              parentKey={key}
              repo_id={repo.repo}
              cfg={cfg}
              base_url={base_url}
              server_label={server_label}
              isOpen={isOpen}
              toggle={toggle}
              selected={selected}
              setSelected={setSelected}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

interface ConfigNodeProps extends NodeProps {
  parentKey: string;
  repo_id: string;
  cfg: HFCacheConfig;
  base_url: string;
  server_label: string;
}

function ConfigNode({
  parentKey,
  repo_id,
  cfg,
  base_url,
  server_label,
  isOpen,
  toggle,
  selected,
  setSelected,
}: ConfigNodeProps) {
  const key = `${parentKey}:${cfg.config}`;
  const open = isOpen(key);
  return (
    <li className="explore-tree-node">
      <button
        className="explore-tree-row config-row"
        onClick={() => toggle(key)}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <span>{cfg.config}</span>
        {cfg.version && <span className="muted"> @{cfg.version}</span>}
        {cfg.size_bytes != null && (
          <span className="muted"> · {fmtBytes(cfg.size_bytes)}</span>
        )}
      </button>
      {open && (
        <ul className="explore-tree-children">
          {cfg.splits.map((sp) => {
            const leaf: SelectedLeaf = {
              server_label,
              server_base_url: base_url,
              display: `${repo_id} · ${cfg.config} · ${sp.name}`,
              hint_rows: sp.num_examples ?? null,
              load: { path: repo_id, name: cfg.config, split: sp.name },
            };
            return (
              <SplitLeaf
                key={`${key}:${sp.name}`}
                leaf={leaf}
                split={sp}
                selected={selected}
                setSelected={setSelected}
              />
            );
          })}
        </ul>
      )}
    </li>
  );
}

function LocalGroup({
  serverKey,
  base_url,
  server_label,
  isOpen,
  toggle,
  selected,
  setSelected,
}: GroupProps) {
  const key = `${serverKey}:local`;
  const open = isOpen(key);
  const localQ = useQuery({
    queryKey: ["ds-local", base_url],
    queryFn: () => api.datasetServerLocal(base_url, ""),
    enabled: open,
  });
  const entries = (localQ.data as LocalListResponse | undefined)?.local ?? [];

  return (
    <li className="explore-tree-node">
      <button
        className="explore-tree-row group-row"
        onClick={() => toggle(key)}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <span>Local</span>
        <span className="muted"> ({open ? entries.length : "…"})</span>
      </button>
      {open && (
        <ul className="explore-tree-children">
          {localQ.isLoading && (
            <li className="muted pane-state-small">loading…</li>
          )}
          {localQ.error && (
            <li className="muted pane-state-small">
              {String(localQ.error)}
            </li>
          )}
          {!localQ.isLoading && !localQ.error && entries.length === 0 && (
            <li className="muted pane-state-small">
              no named local datasets registered on this server
            </li>
          )}
          {entries.map((entry) => (
            <LocalEntryNode
              key={`${key}:${entry.name}`}
              parentKey={key}
              entry={entry}
              base_url={base_url}
              server_label={server_label}
              isOpen={isOpen}
              toggle={toggle}
              selected={selected}
              setSelected={setSelected}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

interface LocalEntryNodeProps extends NodeProps {
  parentKey: string;
  entry: LocalDatasetEntry;
  base_url: string;
  server_label: string;
}

function LocalEntryNode({
  parentKey,
  entry,
  base_url,
  server_label,
  isOpen,
  toggle,
  selected,
  setSelected,
}: LocalEntryNodeProps) {
  const key = `${parentKey}:${entry.name}`;
  const open = isOpen(key);
  const splits = entry.splits ?? [];
  const hasSplits = splits.length > 0;
  const path = `local/${entry.name}`;
  return (
    <li className="explore-tree-node">
      <button
        className="explore-tree-row repo-row"
        onClick={() => {
          if (hasSplits) {
            toggle(key);
          } else {
            // No discoverable splits — clicking selects with no
            // split argument so the dataset_server picks its default.
            setSelected({
              server_label,
              server_base_url: base_url,
              display: `${path}${
                entry.config_name ? ` · ${entry.config_name}` : ""
              }`,
              hint_rows: null,
              load: { path },
            });
          }
        }}
      >
        <span className="tri">
          {hasSplits ? (open ? "▾" : "▸") : "·"}
        </span>
        <code>{path}</code>
        {entry.config_name && (
          <span className="muted"> · {entry.config_name}</span>
        )}
        {entry.size_bytes != null && (
          <span className="muted"> · {fmtBytes(entry.size_bytes)}</span>
        )}
        {entry.layout && entry.layout !== "dataset_dict" && (
          <span className="muted"> · {entry.layout}</span>
        )}
      </button>
      {hasSplits && open && (
        <ul className="explore-tree-children">
          {splits.map((sp) => {
            const leaf: SelectedLeaf = {
              server_label,
              server_base_url: base_url,
              display: `${path}${
                entry.config_name ? ` · ${entry.config_name}` : ""
              } · ${sp.name}`,
              hint_rows: sp.num_examples ?? null,
              load: { path, split: sp.name },
            };
            return (
              <SplitLeaf
                key={`${key}:${sp.name}`}
                leaf={leaf}
                split={sp}
                selected={selected}
                setSelected={setSelected}
              />
            );
          })}
        </ul>
      )}
    </li>
  );
}

interface SplitLeafProps {
  leaf: SelectedLeaf;
  split: HFCacheSplit;
  selected: SelectedLeaf | null;
  setSelected: (sel: SelectedLeaf | null) => void;
}

function SplitLeaf({ leaf, split, selected, setSelected }: SplitLeafProps) {
  // Match by full load_args + server so the same split on two different
  // servers shows the right selection state independently.
  const isSelected =
    selected !== null &&
    selected.server_base_url === leaf.server_base_url &&
    selected.load.path === leaf.load.path &&
    selected.load.name === leaf.load.name &&
    selected.load.split === leaf.load.split;
  return (
    <li
      className={
        "explore-tree-row split-row" + (isSelected ? " selected" : "")
      }
      onClick={() => setSelected(leaf)}
    >
      <span className="tri">·</span>
      <span>{split.name}</span>
      {split.num_examples != null && (
        <span className="muted">
          {" "}
          ({split.num_examples.toLocaleString()} rows
          {split.num_bytes ? ` · ${fmtBytes(split.num_bytes)}` : ""})
        </span>
      )}
    </li>
  );
}

interface PreviewPaneProps {
  selected: SelectedLeaf;
  // TanStack Query handles for the load and iter calls live in the
  // parent so the explore-tab top bar can render the result-derived
  // header (length / column count) without duplicating the queries.
  loadQ: UseQueryResult<LoadResponse, Error>;
  iterQ: UseQueryResult<IterResponse, Error>;
  columns: string[] | null;
  page: number;
  setPage: (p: number) => void;
  pageSize: number;
  totalPages: number;
}

function PreviewPane({
  selected,
  loadQ,
  iterQ,
  columns,
  page,
  setPage,
  pageSize,
  totalPages,
}: PreviewPaneProps) {
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const handle = loadQ.data?.handle;
  return (
    <div className="preview-pane">
      {loadQ.isLoading && (
        <div className="pane-state muted">Loading dataset handle…</div>
      )}
      {loadQ.error && (
        <pre className="pane-state err">{String(loadQ.error)}</pre>
      )}

      {handle && (
        <>
          {iterQ.isLoading && (
            <div className="pane-state muted">Fetching rows…</div>
          )}
          {iterQ.error && (
            <pre className="pane-state err">{String(iterQ.error)}</pre>
          )}
          {iterQ.data && (
            <RowsTable
              rows={iterQ.data.rows}
              columns={columns}
              expandedRow={expandedRow}
              setExpandedRow={setExpandedRow}
              pageOffset={page * pageSize}
            />
          )}
        </>
      )}

      {/* totalPages is derived from length; only render when we have it. */}
      {loadQ.data?.length != null ||
        (selected.hint_rows != null && totalPages > 0) ? (
        <footer className="preview-pager">
          <Pager page={page} totalPages={totalPages} setPage={setPage} />
        </footer>
      ) : null}
    </div>
  );
}

interface RowsTableProps {
  rows: Array<Record<string, unknown>>;
  columns: string[] | null;
  expandedRow: number | null;
  setExpandedRow: (n: number | null) => void;
  pageOffset: number;
}

function RowsTable({
  rows,
  columns,
  expandedRow,
  setExpandedRow,
  pageOffset,
}: RowsTableProps) {
  // If the server didn't report column_names (vocab streams, unusual
  // sources), derive the union of keys across the visible window so the
  // user still gets a useful table.
  const cols = useMemo(() => {
    if (columns && columns.length > 0) return columns;
    const seen = new Set<string>();
    for (const r of rows) {
      for (const k of Object.keys(r)) seen.add(k);
    }
    return Array.from(seen);
  }, [columns, rows]);

  if (rows.length === 0) {
    return <div className="pane-state muted">No rows in this page.</div>;
  }
  return (
    <div className="preview-table-wrap">
      <table className="preview-table">
        <thead>
          <tr>
            <th className="row-index">#</th>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => {
            const absIdx = pageOffset + idx;
            const isExpanded = expandedRow === idx;
            return (
              <tr
                key={idx}
                className={isExpanded ? "expanded" : ""}
                onClick={() => setExpandedRow(isExpanded ? null : idx)}
              >
                <td className="row-index">{absIdx.toLocaleString()}</td>
                {cols.map((c) => (
                  <td key={c}>
                    <Cell
                      value={row[c]}
                      expanded={isExpanded}
                    />
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Cell({
  value,
  expanded,
}: {
  value: unknown;
  expanded: boolean;
}) {
  if (value == null) return <span className="muted">—</span>;
  let display: string;
  if (typeof value === "string") {
    display = value;
  } else if (typeof value === "number" || typeof value === "boolean") {
    return <code>{String(value)}</code>;
  } else {
    try {
      display = JSON.stringify(value, null, expanded ? 2 : 0);
    } catch {
      display = String(value);
    }
  }
  const cap = expanded ? EXPANDED_CELL_TRUNCATE : CELL_TRUNCATE;
  const truncated = display.length > cap;
  const shown = truncated ? display.slice(0, cap) + "…" : display;
  return (
    <span
      className={"preview-cell" + (expanded ? " expanded" : "")}
      style={{ whiteSpace: expanded ? "pre-wrap" : "nowrap" }}
    >
      {shown}
    </span>
  );
}

interface PagerProps {
  page: number;
  totalPages: number;
  setPage: (p: number) => void;
}

/** Compact pager with first/last + neighbors + ellipsis. Pages are
 *  zero-indexed internally; the UI displays 1-based numbers because
 *  that's what users expect for "page 1 of N". */
function Pager({ page, totalPages, setPage }: PagerProps) {
  const tokens = useMemo<Array<number | "...">>(() => {
    const out: Array<number | "..."> = [];
    if (totalPages <= 7) {
      for (let i = 0; i < totalPages; i++) out.push(i);
      return out;
    }
    // Always show first page, last page, current ± 1.
    const want = new Set<number>([
      0,
      totalPages - 1,
      page - 1,
      page,
      page + 1,
    ]);
    // Filter out-of-range and sort.
    const sorted = [...want]
      .filter((p) => p >= 0 && p < totalPages)
      .sort((a, b) => a - b);
    let prev = -1;
    for (const p of sorted) {
      if (prev >= 0 && p > prev + 1) out.push("...");
      out.push(p);
      prev = p;
    }
    return out;
  }, [page, totalPages]);

  return (
    <div className="pager">
      <button
        className="secondary"
        disabled={page <= 0}
        onClick={() => setPage(page - 1)}
      >
        ‹ Prev
      </button>
      {tokens.map((t, i) =>
        t === "..." ? (
          <span key={`e${i}`} className="pager-ellipsis muted">
            …
          </span>
        ) : (
          <button
            key={t}
            className={"pager-page" + (t === page ? " active" : "")}
            onClick={() => setPage(t)}
          >
            {t + 1}
          </button>
        ),
      )}
      <button
        className="secondary"
        disabled={page >= totalPages - 1}
        onClick={() => setPage(page + 1)}
      >
        Next ›
      </button>
    </div>
  );
}

function fmtBytes(n: number): string {
  if (n === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}
