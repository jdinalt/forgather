import { UseQueryResult, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ClusterDatasetServer,
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
import { persistGet, persistSet } from "../persist";
import { ContextMenu } from "./ContextMenu";

const PAGE_SIZE_OPTIONS = [25, 50, 100] as const;
const DEFAULT_PAGE_SIZE = 25;
const CELL_TRUNCATE = 200;
const EXPANDED_CELL_TRUNCATE = 5000;

// Tree pane width — used to be a fixed 384px stylesheet value;
// surfaced here so the user can drag a divider and the choice
// persists in localStorage. Default bumped ~15% (442px) because the
// HF/local tree rows easily run wide enough to wrap at 384.
const DEFAULT_TREE_WIDTH = 442;
const MIN_TREE_WIDTH = 240;
const MAX_TREE_WIDTH = 900;
const TREE_WIDTH_STORAGE_KEY = "datasets-explore-tree-width";

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
 *  format for ``POST /v1/load``.
 *
 *  When ``cluster_server_id`` is set, the Explore tab routes the
 *  /load + /iter calls through the cluster-proxy path
 *  (``/api/cluster/dataset_server_proxy/{server_id}/...``) so the
 *  master injects the upstream bearer — used for cluster-mode
 *  servers the local node hasn't registered. ``server_base_url``
 *  is still carried for display purposes.
 */
export interface SelectedLeaf {
  server_label: string;
  server_base_url: string;
  /** When present, hit the cluster proxy instead of the per-node
   *  proxy. Used by the Cluster mode unified host list and by
   *  cross-tab "View in Explore" handoffs from cluster-server rows. */
  cluster_server_id?: string;
  load: LoadRequest;
  /** Display string for the right pane header. */
  display: string;
  /** Optional row count we already know from the tree's metadata, so we
   *  can show a count before /load completes. */
  hint_rows?: number | null;
}

interface ServerOption {
  /** "local:<queue_id>", "user:<id>", or "cluster:<server_id>" —
   *  stable identifier for tree expansion state. */
  key: string;
  label: string;
  base_url: string;
  /** When set, this entry routes through the cluster proxy. */
  cluster_server_id?: string;
}

/** Compute the set of tree-node keys to add to the ``expanded`` set
 *  so a preselect leaf becomes visible in the browse pane.
 *
 *  Tree key shape (mirroring CacheGroup / LocalGroup / RepoNode /
 *  LocalEntryNode):
 *
 *    server:                      ``cluster:<sid>`` | ``local:<qid>`` | ``user:<id>``
 *    HF cache group:              ``<server>:cache``
 *    HF repo:                     ``<server>:cache:<path>``
 *    Local group:                 ``<server>:local``
 *    Local entry:                 ``<server>:local:<name>``
 *
 *  We currently only know how to derive keys for cluster servers —
 *  per-node leaves coming from the Servers tab don't carry the
 *  ``queue_id`` / ``id`` we'd need. That's fine; the Servers tab
 *  flow has always relied on the right-pane load and not on tree
 *  expansion.
 */
function deriveExpandKeys(leaf: SelectedLeaf): string[] {
  if (!leaf.cluster_server_id) return [];
  const serverKey = `cluster:${leaf.cluster_server_id}`;
  const path = leaf.load.path;
  if (path.startsWith("local/")) {
    return [serverKey, `${serverKey}:local`, `${serverKey}:local:${path.slice(6)}`];
  }
  return [serverKey, `${serverKey}:cache`, `${serverKey}:cache:${path}`];
}

interface Props {
  localServers: DatasetServerLocal[];
  userServers: DatasetServerUser[];
  /** Cluster-wide servers from the master's inventory. Empty in
   *  standalone mode. The tab merges these into the host list and
   *  routes their fetches through the cluster proxy. */
  clusterServers?: ClusterDatasetServer[];
  /** When true, hide the per-node Local + User sources and use
   *  ``clusterServers`` exclusively. */
  clusterActive?: boolean;
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
  clusterServers,
  clusterActive,
  preselect,
  onPreselectConsumed,
}: Props) {
  const servers: ServerOption[] = useMemo(() => {
    if (clusterActive) {
      // Cluster mode: master-aggregated, deduped, healthy-first. We
      // *show* unhealthy entries too (they may come back) but mark
      // them visually — the tree expansion will surface the
      // upstream error if the operator clicks anyway.
      const cluster = (clusterServers ?? [])
        .slice()
        .sort((a, b) => Number(b.healthy) - Number(a.healthy))
        .map((s) => ({
          key: `cluster:${s.server_id}`,
          label: s.healthy ? s.label : `${s.label} (down)`,
          base_url: s.base_url,
          cluster_server_id: s.server_id,
        }));
      return cluster;
    }
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
  }, [localServers, userServers, clusterServers, clusterActive]);

  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [selected, setSelected] = useState<SelectedLeaf | null>(null);

  // Cross-tab preselect: when the Servers tab or the Cluster view's
  // Datasets tab fires a row click, the parent sets ``preselect`` and
  // switches the active tab here.
  //
  // The cluster row only carries the dataset's ``path`` (and the
  // server to route through) — not a fully-qualified path+name+split.
  // Setting that directly as ``selected`` would trigger /v1/load on
  // a partial leaf, which the HF backend rejects ("must pick a
  // config"), wedging the right pane in "Loading dataset handle…".
  // Instead, hold the preselect as ``pendingResolve``: fetch the
  // chosen server's cache / local inventory, pick the first config +
  // first split (or just the first split for local datasets), and
  // *then* set ``selected`` to the fully-resolved leaf so the load
  // succeeds and the right pane shows real rows.
  //
  // We also replace (not merge) the ``expanded`` set so previously-
  // open branches collapse — otherwise the new path gets lost in the
  // existing expansion noise.
  const [pendingResolve, setPendingResolve] = useState<SelectedLeaf | null>(
    null,
  );
  // When a pending resolve runs out of input data — the chosen server
  // didn't actually have the dataset cached, or has no enumerable
  // configs/splits — we drop the resolve without setting ``selected``.
  // Without surfacing that, the right pane just shows "No selection."
  // which looks like the click did nothing. This hint replaces that
  // placeholder until the user picks a split themselves.
  const [resolveHint, setResolveHint] = useState<string | null>(null);
  useEffect(() => {
    if (!preselect) return;
    setSelected(null);
    setResolveHint(null);
    setPendingResolve(preselect);
    const keys = deriveExpandKeys(preselect);
    setExpanded(new Set(keys));
    onPreselectConsumed?.();
  }, [preselect, onPreselectConsumed]);

  // Resolver queries — same query keys as CacheGroup / LocalGroup
  // below so TanStack dedups: if the user already had this server's
  // group open the data is in-cache and the resolve completes in one
  // tick. ``enabled`` is gated on pendingResolve + the path shape so
  // we don't fire spurious requests.
  const resolvingLocal =
    pendingResolve?.load.path.startsWith("local/") ?? false;
  const resolveBase = pendingResolve
    ? pendingResolve.cluster_server_id ?? pendingResolve.server_base_url
    : null;
  const resolveCacheQ = useQuery({
    queryKey: ["ds-cache", resolveBase],
    queryFn: () =>
      pendingResolve!.cluster_server_id
        ? api.clusterDatasetServerCache(pendingResolve!.cluster_server_id)
        : api.datasetServerCache(pendingResolve!.server_base_url, ""),
    enabled: !!pendingResolve && !resolvingLocal,
  });
  const resolveLocalQ = useQuery({
    queryKey: ["ds-local", resolveBase],
    queryFn: () =>
      pendingResolve!.cluster_server_id
        ? api.clusterDatasetServerLocal(pendingResolve!.cluster_server_id)
        : api.datasetServerLocal(pendingResolve!.server_base_url, ""),
    enabled: !!pendingResolve && resolvingLocal,
  });
  useEffect(() => {
    if (!pendingResolve) return;
    const path = pendingResolve.load.path;
    if (path.startsWith("local/")) {
      const data = resolveLocalQ.data as LocalListResponse | undefined;
      if (!data) return;
      const name = path.slice(6);
      const entry = data.local?.find((e) => e.name === name);
      if (!entry) {
        setResolveHint(
          `${path} isn't in this server's local registry. Expand a host below to pick a different one.`,
        );
        setPendingResolve(null);
        return;
      }
      const split = entry.splits?.[0];
      // Local without a known split: load with no split arg — the
      // server picks its default. With a split: full leaf.
      setSelected({
        ...pendingResolve,
        load: split ? { path, split: split.name } : { path },
        display: split ? `${path} · ${split.name}` : path,
        hint_rows: split?.num_examples ?? null,
      });
      setPendingResolve(null);
      return;
    }
    // HF: resolve to first config + first split.
    const data = resolveCacheQ.data as HFCacheResponse | undefined;
    if (!data) return;
    const repo = data.datasets?.find((r) => r.repo === path);
    const cfg = repo?.configs?.[0];
    const split = cfg?.splits?.[0];
    if (cfg && split) {
      setSelected({
        ...pendingResolve,
        load: { path, name: cfg.config, split: split.name },
        display: `${path} · ${cfg.config} · ${split.name}`,
        hint_rows: split.num_examples ?? null,
      });
      // Also expand the config row so the highlighted split sits
      // under a visible parent.
      if (pendingResolve.cluster_server_id) {
        const cfgKey = `cluster:${pendingResolve.cluster_server_id}:cache:${path}:${cfg.config}`;
        setExpanded((prev) => {
          const next = new Set(prev);
          next.add(cfgKey);
          return next;
        });
      }
    } else if (!repo) {
      setResolveHint(
        `${path} isn't cached on this server. Expand a host below to pick a different one.`,
      );
    } else {
      // Repo present but no configs/splits enumerable — server may
      // not have completed its first refresh yet.
      setResolveHint(
        `${path} is cached but doesn't expose any configs/splits yet. Try refreshing the server, or pick a split manually below.`,
      );
    }
    setPendingResolve(null);
  }, [pendingResolve, resolveCacheQ.data, resolveLocalQ.data]);
  // Tree pane collapse — gives the preview pane the full width when the
  // user has already picked a leaf and just wants to read rows.
  const [treeCollapsed, setTreeCollapsed] = useState<boolean>(false);
  // Tree pane width is operator-adjustable via the divider; persists
  // across reloads. Initial read clamps in case localStorage holds a
  // stale value outside the current bounds.
  const [treeWidth, setTreeWidth] = useState<number>(() => {
    const raw = persistGet(TREE_WIDTH_STORAGE_KEY);
    const n = raw != null ? Number(raw) : NaN;
    if (!Number.isFinite(n)) return DEFAULT_TREE_WIDTH;
    return Math.max(MIN_TREE_WIDTH, Math.min(MAX_TREE_WIDTH, n));
  });
  // Drag state lives in refs so the pointer-move callback doesn't
  // re-bind on every render (and so it sees the latest grab anchor
  // without depending on stale closures). Mirrors the TemplatesView
  // resizer pattern.
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const onResizerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      dragRef.current = { startX: e.clientX, startWidth: treeWidth };
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [treeWidth],
  );
  const onResizerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current;
      if (!drag) return;
      const next = drag.startWidth + (e.clientX - drag.startX);
      setTreeWidth(Math.max(MIN_TREE_WIDTH, Math.min(MAX_TREE_WIDTH, next)));
    },
    [],
  );
  const onResizerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragRef.current) return;
      dragRef.current = null;
      try {
        (e.currentTarget as Element).releasePointerCapture(e.pointerId);
      } catch {
        // Capture may already be released if the pointer was cancelled.
      }
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      persistSet(TREE_WIDTH_STORAGE_KEY, String(treeWidth));
    },
    [treeWidth],
  );
  // Double-click resets to the default — quick escape hatch if the
  // user dragged the pane somewhere awkward.
  const onResizerDoubleClick = useCallback(() => {
    setTreeWidth(DEFAULT_TREE_WIDTH);
    persistSet(TREE_WIDTH_STORAGE_KEY, String(DEFAULT_TREE_WIDTH));
  }, []);
  // Keyboard resizing: focusable separator + Arrow/Home/End keys.
  // Step is 16px for arrows, 64px with shift — same idiom as the
  // <input type="range"> spec. Page-level Home/End jump to the
  // limits.
  const onResizerKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const step = e.shiftKey ? 64 : 16;
      let next: number | null = null;
      switch (e.key) {
        case "ArrowLeft":
          next = treeWidth - step;
          break;
        case "ArrowRight":
          next = treeWidth + step;
          break;
        case "Home":
          next = MIN_TREE_WIDTH;
          break;
        case "End":
          next = MAX_TREE_WIDTH;
          break;
        default:
          return;
      }
      e.preventDefault();
      const clamped = Math.max(MIN_TREE_WIDTH, Math.min(MAX_TREE_WIDTH, next));
      setTreeWidth(clamped);
      persistSet(TREE_WIDTH_STORAGE_KEY, String(clamped));
    },
    [treeWidth],
  );
  const [pageSize, setPageSize] = useState<number>(DEFAULT_PAGE_SIZE);
  const [page, setPage] = useState<number>(0);
  // Reset page when the selection changes — different leaf, fresh state.
  useEffect(() => setPage(0), [selected]);

  // Load query: cached by (server, load_args). Re-clicking the same leaf
  // is a no-op on the server (it hashes the load_args).
  //
  // Cluster-mode leaves carry ``cluster_server_id``; for those we go
  // through the cluster proxy so the master injects the upstream
  // bearer. Per-node leaves keep using ``datasetServerLoad`` against
  // the local registry / JobRecord proxy.
  const loadQ = useQuery({
    queryKey: selected
      ? [
          "ds-load",
          selected.cluster_server_id ?? selected.server_base_url,
          JSON.stringify(selected.load),
        ]
      : ["ds-load-idle"],
    queryFn: () => {
      const s = selected as SelectedLeaf;
      if (s.cluster_server_id) {
        return api.clusterDatasetServerLoad(s.cluster_server_id, s.load);
      }
      return api.datasetServerLoad(s.server_base_url, s.load, "");
    },
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
      selected?.cluster_server_id ?? selected?.server_base_url,
      handle,
      page,
      pageSize,
    ],
    queryFn: () => {
      const s = selected as SelectedLeaf;
      if (s.cluster_server_id) {
        return api.clusterDatasetServerIter(
          s.cluster_server_id,
          handle as string,
          page * pageSize,
          pageSize,
        );
      }
      return api.datasetServerIter(
        s.server_base_url,
        handle as string,
        page * pageSize,
        pageSize,
        "",
      );
    },
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
          style={
            treeCollapsed
              ? undefined
              : { flex: `0 0 ${treeWidth}px`, width: treeWidth }
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
        {!treeCollapsed && (
          <div
            className="datasets-explore-resizer"
            onPointerDown={onResizerDown}
            onPointerMove={onResizerMove}
            onPointerUp={onResizerUp}
            onPointerCancel={onResizerUp}
            onDoubleClick={onResizerDoubleClick}
            onKeyDown={onResizerKeyDown}
            title="Drag to resize the browse pane · double-click to reset · arrows to nudge"
            role="separator"
            aria-orientation="vertical"
            aria-valuenow={treeWidth}
            aria-valuemin={MIN_TREE_WIDTH}
            aria-valuemax={MAX_TREE_WIDTH}
            tabIndex={0}
          />
        )}
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
          ) : resolveHint ? (
            <div className="pane-state warn">{resolveHint}</div>
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
            cluster_server_id={server.cluster_server_id}
            server_label={server.label}
            isOpen={isOpen}
            toggle={toggle}
            selected={selected}
            setSelected={setSelected}
          />
          <LocalGroup
            serverKey={key}
            base_url={server.base_url}
            cluster_server_id={server.cluster_server_id}
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
  /** When set, the group's data fetches route through the cluster
   *  proxy instead of the per-node proxy. */
  cluster_server_id?: string;
  server_label: string;
}

function CacheGroup({
  serverKey,
  base_url,
  cluster_server_id,
  server_label,
  isOpen,
  toggle,
  selected,
  setSelected,
}: GroupProps) {
  const key = `${serverKey}:cache`;
  const open = isOpen(key);
  // Cluster servers route through /api/cluster/dataset_server_proxy
  // so the master injects the upstream bearer — keeps the browser
  // free of any cluster server's token. Per-node servers keep using
  // the existing per-node proxy.
  const cacheQ = useQuery({
    queryKey: ["ds-cache", cluster_server_id ?? base_url],
    queryFn: () =>
      cluster_server_id
        ? api.clusterDatasetServerCache(cluster_server_id)
        : api.datasetServerCache(base_url, ""),
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
              cluster_server_id={cluster_server_id}
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
  cluster_server_id?: string;
  server_label: string;
}

function RepoNode({
  parentKey,
  repo,
  base_url,
  cluster_server_id,
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
              cluster_server_id={cluster_server_id}
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
  cluster_server_id?: string;
  server_label: string;
}

function ConfigNode({
  parentKey,
  repo_id,
  cfg,
  base_url,
  cluster_server_id,
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
              cluster_server_id,
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
  cluster_server_id,
  server_label,
  isOpen,
  toggle,
  selected,
  setSelected,
}: GroupProps) {
  const key = `${serverKey}:local`;
  const open = isOpen(key);
  const localQ = useQuery({
    queryKey: ["ds-local", cluster_server_id ?? base_url],
    queryFn: () =>
      cluster_server_id
        ? api.clusterDatasetServerLocal(cluster_server_id)
        : api.datasetServerLocal(base_url, ""),
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
              cluster_server_id={cluster_server_id}
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
  cluster_server_id?: string;
  server_label: string;
}

function LocalEntryNode({
  parentKey,
  entry,
  base_url,
  cluster_server_id,
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
              cluster_server_id,
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
              cluster_server_id,
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

  // Mousedown position per row, used to distinguish a click (toggle
  // expand) from a drag-select (don't toggle, let the user copy the
  // text they just selected). Without this, any text selection inside
  // the row collapses it on mouseup, which makes copy/paste from a
  // collapsed example impossible. Threshold: 4px of movement or any
  // active non-empty selection at click time suppresses the toggle.
  const dragRef = useRef<{ x: number; y: number } | null>(null);

  const [menu, setMenu] = useState<{
    x: number;
    y: number;
    column: string;
    value: unknown;
  } | null>(null);

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
                onMouseDown={(e) => {
                  // Only left-button starts a candidate-click. Middle
                  // and right buttons must not arm the drag check, or
                  // a right-click drag would suppress the next real
                  // click.
                  if (e.button !== 0) return;
                  dragRef.current = { x: e.clientX, y: e.clientY };
                }}
                onClick={(e) => {
                  const start = dragRef.current;
                  dragRef.current = null;
                  // Suppress toggle if the user dragged > 4px (likely
                  // a drag-select gesture) or if mouseup landed with
                  // text selected (the user is mid-selection — they
                  // want to copy, not collapse). The browser clears
                  // any prior selection on mousedown of a fresh click,
                  // so a non-empty selection at click time always
                  // means "selected during this gesture."
                  if (start) {
                    const dx = e.clientX - start.x;
                    const dy = e.clientY - start.y;
                    if (dx * dx + dy * dy > 16) return;
                  }
                  const sel = window.getSelection();
                  if (sel && !sel.isCollapsed && sel.toString().length > 0) {
                    return;
                  }
                  setExpandedRow(isExpanded ? null : idx);
                }}
              >
                <td className="row-index">{absIdx.toLocaleString()}</td>
                {cols.map((c) => (
                  <td
                    key={c}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      setMenu({
                        x: e.clientX,
                        y: e.clientY,
                        column: c,
                        value: row[c],
                      });
                    }}
                  >
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
      {menu && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)}>
          <div className="context-menu-header muted">{menu.column}</div>
          <button
            className="context-menu-item"
            onClick={() => {
              void copyCellValue(menu.value);
              setMenu(null);
            }}
          >
            Copy cell text
          </button>
        </ContextMenu>
      )}
    </div>
  );
}

/** Serialize a cell value the same way the Cell component does for
 *  display, but without truncation — that's the whole point of the
 *  context-menu copy: get the full underlying text, not the visible
 *  truncated form. Async because navigator.clipboard.writeText is. */
async function copyCellValue(value: unknown): Promise<void> {
  let text: string;
  if (value == null) {
    text = "";
  } else if (typeof value === "string") {
    text = value;
  } else if (typeof value === "number" || typeof value === "boolean") {
    text = String(value);
  } else {
    try {
      text = JSON.stringify(value, null, 2);
    } catch {
      text = String(value);
    }
  }
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    // clipboard API can fail under permissions-policy or non-secure
    // contexts; surface enough to triage without spamming console
    // noise on every right-click.
    alert(
      `Copy failed: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
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

  // Goto field tracks user input as a string so partial typing
  // ("12") doesn't fight a numeric-state-driven re-render. The
  // submit handler parses, clamps to [1..totalPages], and converts
  // back to a 0-indexed page before calling setPage.
  const [gotoText, setGotoText] = useState("");
  const submitGoto = () => {
    const raw = gotoText.trim();
    if (!raw) return;
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    const clamped = Math.max(1, Math.min(totalPages, Math.floor(n)));
    setPage(clamped - 1);
    setGotoText("");
  };

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
      {totalPages > 1 && (
        <form
          className="pager-goto"
          onSubmit={(e) => {
            e.preventDefault();
            submitGoto();
          }}
        >
          <label className="muted">Go to</label>
          <input
            type="number"
            min={1}
            max={totalPages}
            value={gotoText}
            placeholder={String(page + 1)}
            onChange={(e) => setGotoText(e.target.value)}
            title={`Jump to a page between 1 and ${totalPages}`}
          />
          <button
            className="secondary"
            type="submit"
            disabled={gotoText.trim() === ""}
          >
            Go
          </button>
          <span className="muted pager-goto-total">/ {totalPages}</span>
        </form>
      )}
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
