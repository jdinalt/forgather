import { useCallback, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  CheckpointEntry,
  ConfigInfo,
  EvalEntry,
  ProjectInfo,
  ServiceStatus,
} from "./api";
import { getAutoWatchTty } from "./autoWatch";
import { useDemoMode, useServerVersion } from "./demoMode";
import { ContextMenu } from "./components/ContextMenu";
import { ProjectTree } from "./components/ProjectTree";
import { ConfigViewer } from "./components/ConfigViewer";
import { ClusterSidebarPanel } from "./components/ClusterSidebarPanel";
import { ClusterPanel } from "./components/ClusterPanel";
import { GpuPanel } from "./components/GpuPanel";
import { EvalModal } from "./components/EvalModal";
import { InferenceModal } from "./components/InferenceModal";
import { DatasetServerModal } from "./components/DatasetServerModal";
import { InferencePanel } from "./components/InferencePanel";
import { DatasetsPanel } from "./components/DatasetsPanel";
import { DiLoCoPanel } from "./components/DiLoCoPanel";
import { DiLoCoServerModal } from "./components/DiLoCoServerModal";
import type { SelectedLeaf } from "./components/DatasetsExploreTab";
import { JobsPanel } from "./components/JobsPanel";
import { ServicesPanel } from "./components/ServicesPanel";
import { LogDetailPanel } from "./components/LogDetailPanel";
import { CheckpointDetailPanel } from "./components/CheckpointDetailPanel";
import { EvalDetailPanel } from "./components/EvalDetailPanel";
import { TensorBoardModal } from "./components/TensorBoardModal";
import { MkDocsModal } from "./components/MkDocsModal";
import { ConvertModal } from "./components/ConvertModal";
import { FinalizeModal } from "./components/FinalizeModal";
import { UpdateModal } from "./components/UpdateModal";
import { DocsPanel } from "./components/DocsPanel";
import { FilesPanel } from "./components/FilesPanel";
import { FilesTree } from "./components/FilesTree";
import { SearchRootsPanel } from "./components/SearchRootsPanel";
import { ShutdownModal } from "./components/ShutdownModal";
import { useFilesState } from "./files-state";

type View =
  | "projects"
  | "edit"
  | "docs"
  | "gpus"
  | "cluster"
  | "jobs"
  | "inference"
  | "datasets"
  | "diloco";
export type ConfigTab = "info" | "pp" | "code" | "graph" | "templates" | "debug";

// View metadata. "GPUs" is always the local node's GPU panel; "Nodes"
// is a separate cluster-only entry that's filtered out in standalone
// mode (see ``visibleViews`` below).
const VIEWS: { id: View; label: string; icon: string; clusterOnly?: boolean }[] =
  [
    // Cluster is the cluster-wide context — when it's present, it's
    // the first thing the eye should land on; in standalone mode it's
    // filtered out entirely, so this slot is invisible.
    { id: "cluster", label: "Cluster", icon: "🖧", clusterOnly: true },
    { id: "projects", label: "Projects", icon: "📁" },
    { id: "edit", label: "Edit", icon: "✎" },
    { id: "docs", label: "Docs", icon: "📚" },
    { id: "gpus", label: "GPUs", icon: "🖥" },
    { id: "jobs", label: "Jobs", icon: "⚙" },
    { id: "inference", label: "Inference", icon: "🔮" },
    { id: "datasets", label: "Datasets", icon: "🗂" },
    { id: "diloco", label: "DiLoCo", icon: "🧩" },
  ];

// A window glyph with a left-biased vertical divider — represents the
// app canvas with the sidebar partition. Shown as the sidebar toggle so
// it doesn't get confused with the ▶ scheduler-play button next to it.
function SidebarIcon() {
  return (
    <svg
      className="sidebar-icon"
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

// "Restart-alt" glyph: a clockwise circular arrow with the arrow
// pointing *back* into a small notch — the classic "reboot in place"
// icon (Material Design's ``restart_alt``). Visually distinct from a
// plain power button (which means shutdown) and from the C-shaped
// ReloadIcon (which means refresh / reload data). Filled rather than
// stroked because the path data is borrowed from Material's icon
// font, which is shipped as a filled outline.
function RestartIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 5 V1 L 7 6 l 5 5 V 7 c 3.31 0 6 2.69 6 6 s -2.69 6 -6 6 s -6 -2.69 -6 -6 H 4 c 0 4.42 3.58 8 8 8 s 8 -3.58 8 -8 S 16.42 5 12 5 z" />
    </svg>
  );
}

// Classic power-button glyph: a broken circle with a vertical stroke
// pointing up through the gap. Means "shutdown" — visually distinct
// from the RestartIcon's circular arrow (in-place reboot) and the
// ReloadIcon (refresh data).
function PowerIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3 v9" />
      <path d="M6.4 7.2 a8 8 0 1 0 11.2 0" />
    </svg>
  );
}

// Circular-arrow glyph matching the old top-bar Refresh button (which
// used the "⟳" character). Inline SVG so disabled / hover styling
// matches the other icon buttons in the footer without depending on
// the system font's rendering of the unicode glyph.
function ReloadIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="23 4 23 10 17 10" />
      <path d="M20.49 15A9 9 0 1 1 18.36 6.36L23 10" />
    </svg>
  );
}

// Simple gear glyph for the sidebar settings bar. Stroked rather than
// filled so it sits comfortably next to the other monochrome controls.
function GearIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
    </svg>
  );
}

type DocsBackEntry =
  | { kind: "doc"; path: string | null; scrollTop: number }
  | {
      kind: "external";
      view: View;
      selection: Selection;
      tab: ConfigTab;
    };

export type Selection =
  | null
  | { kind: "config"; project: ProjectInfo; config: ConfigInfo }
  | {
      kind: "log";
      project: ProjectInfo;
      config: ConfigInfo;
      run_dir: string;
      run_id: string;
    }
  | {
      kind: "checkpoint";
      project: ProjectInfo;
      config: ConfigInfo;
      output_dir: string;
      checkpoint: CheckpointEntry;
    }
  | {
      kind: "eval";
      project: ProjectInfo;
      config: ConfigInfo;
      output_dir: string;
      evaluation: EvalEntry;
    };

export default function App() {
  const demoMode = useDemoMode();
  const serverVersion = useServerVersion();
  const [view, setView] = useState<View>("docs");
  const [selected, setSelected] = useState<Selection>(null);
  // Tab state lives here so opening a project can both pick its default
  // config AND switch to "info" in one render cycle.
  const [tab, setTab] = useState<ConfigTab>("info");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  // Ctrl+B / Cmd+B toggles the sidebar, matching VS Code. Capture phase so
  // Monaco doesn't swallow it inside the editor.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (e.altKey || e.shiftKey) return;
      if (e.key !== "b" && e.key !== "B") return;
      e.preventDefault();
      e.stopPropagation();
      setSidebarCollapsed((c) => !c);
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () =>
      window.removeEventListener("keydown", onKey, { capture: true } as any);
  }, []);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [servicesOpen, setServicesOpen] = useState(false);
  // Each Services launcher row carries its own disclosure for the
  // configured-instance list. Keyed by service type. Collapsed by
  // default so the Services pane stays tidy on first paint.
  const [servicesCategoryOpen, setServicesCategoryOpen] = useState<
    Record<string, boolean>
  >({});
  // Configured-services list, kept here so the per-category launcher
  // rows can hide their disclosure chevron when there are no instances.
  // Shares the ["services"] query key with ServicesPanel — react-query
  // dedupes the fetch, so this is "free".
  const servicesQ = useQuery({
    queryKey: ["services"],
    queryFn: api.listServices,
    refetchInterval: 4000,
  });
  const servicesByType = (servicesQ.data ?? []).reduce<Record<string, number>>(
    (acc, s) => {
      acc[s.service.type] = (acc[s.service.type] ?? 0) + 1;
      return acc;
    },
    {},
  );
  // Running count per type — same UI pattern as the Views → Jobs pill.
  // Only "actually running" entries count (ServiceStatus.running maps
  // strictly to JobRecord status == "running"; queued/starting/aborted
  // don't), so the pill matches the green dots on the rows below.
  const runningServicesByType = (servicesQ.data ?? []).reduce<
    Record<string, number>
  >((acc, s) => {
    if (s.running) {
      acc[s.service.type] = (acc[s.service.type] ?? 0) + 1;
    }
    return acc;
  }, {});
  const expandServicesCategory = useCallback((t: string) => {
    setServicesCategoryOpen((s) => ({ ...s, [t]: true }));
    // Also unfold the Services group itself if the user created the
    // entry from a modal triggered elsewhere — without this, the new
    // entry would land inside a collapsed parent.
    setServicesOpen(true);
  }, []);
  const [searchRootsOpen, setSearchRootsOpen] = useState(false);
  const [projectsOpen, setProjectsOpen] = useState(false);
  const [filesOpen, setFilesOpen] = useState(false);
  // Reveal-in-Files plumbing. ProjectTree's "Reveal in Files" menu
  // item calls revealInFiles(path); we forward a {path, nonce} to
  // FilesTree, which expands ancestors and selects the row. The nonce
  // lets the same path be revealed twice in a row — equality on path
  // alone wouldn't fire useEffect a second time.
  const [revealRequest, setRevealRequest] = useState<
    { path: string; nonce: number } | null
  >(null);
  const revealInFiles = useCallback((path: string) => {
    setFilesOpen(true);
    setRevealRequest({ path, nonce: Date.now() });
  }, []);
  const [viewsOpen, setViewsOpen] = useState(true);
  // Cluster sidebar group — collapsed by default to keep the sidebar
  // tidy on first paint; hidden entirely when standalone.
  const [clusterOpen, setClusterOpen] = useState(false);
  const [startServerOpen, setStartServerOpen] = useState(false);
  const [datasetServerOpen, setDatasetServerOpen] = useState(false);
  const [tensorboardOpen, setTensorboardOpen] = useState(false);
  const [mkdocsOpen, setMkdocsOpen] = useState(false);
  const [dilocoServerOpen, setDilocoServerOpen] = useState(false);
  // Edit-service state: when set, the matching modal is rendered in
  // edit mode pre-populated from this service's args. Single piece of
  // state — only one edit modal is open at a time. Dispatched from
  // ServicesPanel's right-click menu / pencil button.
  const [editingService, setEditingService] = useState<ServiceStatus | null>(
    null,
  );
  const [convertOpen, setConvertOpen] = useState(false);
  const [finalizeOpen, setFinalizeOpen] = useState(false);
  const [updateOpen, setUpdateOpen] = useState(false);
  const [evaluateOpen, setEvaluateOpen] = useState(false);
  // Set when a submit modal closes with the "Watch TTY on start" toggle on.
  // The Jobs view consumes this once the job appears in the polled list and
  // clears it back to null via onAutoWatchConsumed.
  const [autoWatchJobId, setAutoWatchJobId] = useState<string | null>(null);
  // Current document path for the Docs view (null = root README).
  const [docsPath, setDocsPath] = useState<string | null>(null);
  // Back-stack snapshots for the Docs view. Each entry records what to
  // restore when the user clicks Back: either a previous doc path or
  // a "return to a different view" record taken when the user entered
  // Docs from elsewhere (e.g. clicking a doc link in a project README).
  const [docsBackStack, setDocsBackStack] = useState<DocsBackEntry[]>([]);
  const filesApi = useFilesState();
  const qc = useQueryClient();

  // Cross-view dataset preselect: the Cluster view's Datasets tab and
  // the Datasets view's Servers tab both surface row clicks that
  // should land the user in Datasets → Explore with the chosen leaf
  // expanded. Lifting the leaf to App.tsx lets both call sites share
  // one navigation path (the alternative — letting DatasetsPanel own
  // it — couldn't be triggered from outside the panel).
  const [pendingExplore, setPendingExplore] = useState<SelectedLeaf | null>(
    null,
  );
  const openInExplore = useCallback((leaf: SelectedLeaf) => {
    setPendingExplore(leaf);
    setView("datasets");
  }, []);
  // Stable identity so the Explore tab's preselect-consume effect
  // doesn't see a new callback on every App render.
  const clearPendingExplore = useCallback(() => setPendingExplore(null), []);

  // Cross-section: "Analyze…" item on a Datasets cell context menu
  // sends the cell's full text to the Inference > Analyze tab and
  // kicks off scoring. ``key`` is a render-stable nonce that lets the
  // analyze panel's effect distinguish a fresh request from a stale
  // re-render — without it, switching tabs and back would re-trigger.
  const [pendingAnalyze, setPendingAnalyze] = useState<
    { text: string; key: number } | null
  >(null);
  const analyzeText = useCallback((text: string) => {
    setPendingAnalyze({ text, key: Date.now() });
    setView("inference");
  }, []);
  // Stable identity so the analyze panel's consume-effect doesn't see
  // a new callback on every App render. Mirrors clearPendingExplore.
  const clearPendingAnalyze = useCallback(
    () => setPendingAnalyze(null),
    [],
  );

  // Cross-section: "Open in Inference / Datasets" buttons on the Jobs
  // view's inference / dataset-server cards. Drop the user into the
  // matching panel with the row pre-selected — saves the
  // navigate-then-click-the-same-server dance. ``key`` is a fresh
  // nonce so the consume-effect can dedup against stale renders, same
  // pattern as pendingExplore / pendingAnalyze. The id is the Job's
  // stable ``id`` (also the queue_id for server-launched jobs), which
  // is what InferenceModelPanel keys its picker rows on and what
  // DatasetServersTab keys ``selected`` on via ``queue_id``.
  const [pendingInferenceServer, setPendingInferenceServer] = useState<
    { jobId: string; baseUrl: string; key: number } | null
  >(null);
  const openInferenceServer = useCallback(
    (jobId: string, baseUrl: string) => {
      setPendingInferenceServer({ jobId, baseUrl, key: Date.now() });
      setView("inference");
    },
    [],
  );
  const clearPendingInferenceServer = useCallback(
    () => setPendingInferenceServer(null),
    [],
  );

  const [pendingDatasetServer, setPendingDatasetServer] = useState<
    { queueId: string; key: number } | null
  >(null);
  const openDatasetServer = useCallback((queueId: string) => {
    setPendingDatasetServer({ queueId, key: Date.now() });
    setView("datasets");
  }, []);
  const clearPendingDatasetServer = useCallback(
    () => setPendingDatasetServer(null),
    [],
  );

  // Same pattern for the DiLoCo view: a Job card's "Open in DiLoCo"
  // button stamps a pending queueId, the panel reads it once, then
  // calls clear so the auto-pick logic resumes.
  const [pendingDiLoCoServer, setPendingDiLoCoServer] = useState<
    { queueId: string; key: number } | null
  >(null);
  const openDiLoCoServer = useCallback((queueId: string) => {
    setPendingDiLoCoServer({ queueId, key: Date.now() });
    setView("diloco");
  }, []);
  const clearPendingDiLoCoServer = useCallback(
    () => setPendingDiLoCoServer(null),
    [],
  );

  // Wired into every submit modal's onSubmitted prop. Reads the sticky
  // localStorage preference at submit time so a stale toggle from an earlier
  // modal can't trigger an unintended view switch.
  const onJobSubmitted = useCallback((queueId: string) => {
    if (!getAutoWatchTty()) return;
    setAutoWatchJobId(queueId);
    setView("jobs");
  }, []);

  // Switch to the Files panel and open the given template path. Used by the
  // Edit button surfaced in the templates view.
  const openFileForEdit = (path: string) => {
    filesApi.openFile(path);
    setView("edit");
  };

  // Captured scroll position to restore after the next back-navigation
  // applies. Set when Back is clicked, consumed by DocsPanel once the
  // popped page's content has rendered. ``null`` outside of Back, so
  // forward navigations always start at the top.
  const [pendingDocsScroll, setPendingDocsScroll] = useState<number | null>(null);

  // Helper: read the current docs scrollTop from the DOM. The body is
  // a single fixed selector (the only ``.docs-pane-body`` in the
  // tree), so document.querySelector is fine.
  const readDocsScrollTop = (): number => {
    const el = document.querySelector(".docs-pane-body");
    return el ? el.scrollTop : 0;
  };

  // Open a document in the Docs view. If we're entering Docs from another
  // view (e.g. clicking a markdown link in a project README), snapshot the
  // current view + selection so Back returns there. If we're already in
  // Docs and navigating to a different doc, snapshot the previous doc
  // path plus its scroll position so Back restores it. Either way, leave
  // docsBackStack untouched if the path is the same as the current one
  // (idempotent re-entry).
  const openDocs = useCallback(
    (path: string | null) => {
      if (view !== "docs") {
        setDocsBackStack((s) => [
          ...s,
          { kind: "external", view, selection: selected, tab },
        ]);
        setDocsPath(path);
        setPendingDocsScroll(null);
        setView("docs");
        return;
      }
      if (docsPath !== path) {
        const scrollTop = readDocsScrollTop();
        setDocsBackStack((s) => [
          ...s,
          { kind: "doc", path: docsPath, scrollTop },
        ]);
        setDocsPath(path);
        setPendingDocsScroll(null);
      }
    },
    [view, selected, tab, docsPath],
  );
  // Pop the back-stack and apply the restored state. For a "doc" entry
  // we swap the doc path and stash the saved scrollTop in
  // pendingDocsScroll so DocsPanel restores it once the prev page's
  // content has rendered. For an "external" entry we restore the
  // pre-Docs view + selection (matching browser-back semantics across
  // the view boundary).
  const docsBack = useCallback(() => {
    if (docsBackStack.length === 0) return;
    const top = docsBackStack[docsBackStack.length - 1];
    setDocsBackStack((s) => s.slice(0, -1));
    if (top.kind === "external") {
      setView(top.view);
      setSelected(top.selection);
      setTab(top.tab);
    } else {
      setPendingDocsScroll(top.scrollTop);
      setDocsPath(top.path);
    }
  }, [docsBackStack]);
  // Within-docs link click: the user clicked a markdown / ipynb link in
  // the rendered doc. Push the previous path + scroll so Back unwinds
  // to exactly where the user was.
  const docsNavigate = useCallback(
    (path: string) => {
      if (docsPath === path) return;
      const scrollTop = readDocsScrollTop();
      setDocsBackStack((s) => [
        ...s,
        { kind: "doc", path: docsPath, scrollTop },
      ]);
      setDocsPath(path);
      setPendingDocsScroll(null);
    },
    [docsPath],
  );

  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
    refetchInterval: 3000,
  });
  // Sidebar count pills for the Jobs view entry (which now also hosts the
  // queue). Share the same query keys as QueueSection / JobsPanel so the
  // data is deduped: when the view is open the polling already pays for
  // itself, and when it's closed we still want a heartbeat so the pills
  // stay current. Refetch cadence here matches the panels' own.
  const queueCountQ = useQuery({
    queryKey: ["queue"],
    queryFn: api.listQueue,
    refetchInterval: 2000,
  });
  const jobsCountQ = useQuery({
    queryKey: ["jobs", false],
    queryFn: () => api.listJobs(false),
    refetchInterval: 5000,
  });
  // Cluster identity is fetched once at app load and again every 30 s.
  // The payload is null in standalone mode and a small object in
  // cluster mode; either way the response is cheap. The "Cluster"
  // view entry and the "Nodes" sidebar group are both gated on a
  // non-null response — see below.
  const clusterSelfQ = useQuery({
    queryKey: ["cluster-self"],
    queryFn: api.getClusterSelf,
    refetchInterval: 30000,
    staleTime: 30000,
  });
  const clusterActive = !!clusterSelfQ.data;
  // Guard against being stranded on a cluster-only view when the
  // server flips back to standalone. Only act on confirmed success
  // with null data — a transient fetch error or in-flight refetch
  // shouldn't bump the user off the Cluster view they're working
  // on.
  useEffect(() => {
    if (
      clusterSelfQ.isSuccess &&
      clusterSelfQ.data === null &&
      view === "cluster"
    ) {
      setView("gpus");
    }
  }, [clusterSelfQ.isSuccess, clusterSelfQ.data, view]);
  // Total node count for the Cluster view pill — only fetched when
  // we're actually in cluster mode. Shares the queryKey used by
  // ClusterPanel so the cache is reused once that view is opened.
  const clusterMembersQ = useQuery({
    queryKey: ["cluster", "members"],
    queryFn: api.getClusterMembers,
    refetchInterval: 5000,
    enabled: clusterActive,
  });
  const queuedCount = queueCountQ.data?.length ?? 0;
  const runningCount = (jobsCountQ.data ?? []).filter((j) => j.alive).length;
  const nodesCount = clusterMembersQ.data?.members.length ?? 0;
  // The Jobs entry shows two distinct pills — queued (neutral) and running
  // (accent) — rather than one number, since the merged view now holds
  // both states and conflating them into a sum would be misleading. The
  // peer count belongs on the "Nodes" sidebar group (see below) rather
  // than on the "Cluster" view — the group is the list of nodes.
  // Cluster-only views (currently just "cluster") are filtered out of
  // the sidebar in standalone mode.
  const visibleViews = VIEWS.filter((v) => clusterActive || !v.clusterOnly);
  // Tab title: include the node hostname when in cluster mode so
  // the user can tell which node's webui a given browser tab is
  // talking to. Two-tab workflows are common — one tab per node —
  // and "Forgather Server" repeated in every tab title is useless.
  useEffect(() => {
    const self = clusterSelfQ.data;
    if (self) {
      document.title = `${self.hostname} — Forgather`;
    } else {
      document.title = "Forgather Server";
    }
  }, [clusterSelfQ.data]);
  const toggleSched = useMutation({
    mutationFn: api.schedulerToggle,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["scheduler-status"] }),
  });
  const schedEnabled = !!schedQ.data?.enabled;

  // Selecting anything in the project tree implies the user wants the
  // projects view — route there automatically so the detail panel is
  // actually visible.
  //
  // Tab handling: ``info`` is project-scoped (the README), so a click
  // that is actively *choosing* a config doesn't belong there. Switch to
  // ``templates`` — the most relevant view for picking a config out of
  // the dependency graph or the tlist. ``pp`` and ``templates`` are
  // both config-scoped; leave them alone so the user can iterate
  // across configs while keeping the same lens (especially useful for
  // ``pp``, where comparing materialized YAML between configs is the
  // whole point).
  const onConfigSelect = (project: ProjectInfo, config: ConfigInfo) => {
    setSelected({ kind: "config", project, config });
    setView("projects");
    if (tab === "info") setTab("templates");
  };

  // Project-tree expand: jump to default config + info tab, but only when
  // entering a different project.
  const onProjectOpen = (project: ProjectInfo, defaultConfig: ConfigInfo) => {
    if (
      selected?.kind === "config" &&
      selected.project.project_dir === project.project_dir
    )
      return;
    setSelected({ kind: "config", project, config: defaultConfig });
    setTab("info");
    setView("projects");
  };

  const setSelectionAndGoToProjects = (sel: Selection) => {
    setSelected(sel);
    if (sel !== null) setView("projects");
  };

  const refresh = () => {
    qc.invalidateQueries();
  };

  // Back-to-config navigation from detail panels.
  const backToConfig = (project: ProjectInfo, config: ConfigInfo) => {
    setSelected({ kind: "config", project, config });
  };

  // Coerce stale tab values from prior sessions that had "raw"/"models"/"trefs".
  const safeTab = (t: string): ConfigTab => {
    if (
      t === "info" ||
      t === "pp" ||
      t === "code" ||
      t === "graph" ||
      t === "templates" ||
      t === "debug"
    )
      return t;
    return "info";
  };

  const currentTab = safeTab(tab);

  // Help context-menu state for the Tools sidebar. Right-clicking any
  // tool button surfaces a single "Help…" item; clicking it routes to
  // the primary doc for that tool. When a MkDocs job is alive we open
  // its rendered page in a new tab instead of the built-in Docs view
  // (nicer rendering + search). All Tools entries share the same shape;
  // the per-tool variation is just the doc relpath and the mkdocs slug.
  /** A single extra menu item, rendered above "Help…". Tool-specific
   *  affordances use this slot to attach lightweight one-shot actions
   *  without growing the per-tool surface area further. */
  interface ToolExtraMenuItem {
    label: string;
    onChoose: () => void | Promise<void>;
  }
  interface ToolHelpMenu {
    x: number;
    y: number;
    /** Relative-to-repo-root path of the doc, e.g.
     *  ``docs/guides/finalize-model.md``. Absolute path is built at click
     *  time by joining with ``repo-root`` from the API. */
    docRelpath: string;
    /** Slug under MkDocs's served base URL — i.e. the path produced by
     *  MkDocs for the same doc. For ``docs/guides/foo.md`` that's
     *  ``guides/foo/``; for ``tools/dataset_server/README.md`` (symlinked
     *  into ``docs/tools/dataset_server/``) that's
     *  ``tools/dataset_server/``. Trailing slash matches MkDocs's own
     *  output so the link doesn't bounce through a redirect. */
    mkdocsSlug: string;
    /** Tool label used in the context menu header. */
    label: string;
    /** Optional tool-specific items rendered above "Help…". */
    extraItems?: ToolExtraMenuItem[];
  }
  const [toolHelpMenu, setToolHelpMenu] = useState<ToolHelpMenu | null>(null);
  // Repo root is server-resolved once; we cache so each Help click is
  // synchronous after the first fetch.
  const repoRootQ = useQuery({
    queryKey: ["docs-repo-root"],
    queryFn: api.docsRepoRoot,
    staleTime: Infinity,
  });

  // Path to the YAML config the server loaded at startup. Used by the
  // settings gear in the sidebar footer to open the file in the editor.
  const serverConfigQ = useQuery({
    queryKey: ["server-config-path"],
    queryFn: api.serverConfigPath,
    staleTime: Infinity,
  });
  const openServerConfig = useCallback(() => {
    const p = serverConfigQ.data?.path;
    if (p) openFileForEdit(p);
  }, [serverConfigQ.data?.path]);

  // Restart the running server as a fresh process. The backend
  // returns 202-ish ({"restart": "scheduled"}) before unwinding
  // uvicorn, so we wait until the next /api/health success after a
  // failure and then reload the page so the rebooted server's state
  // is reflected fresh. Spawned subprocesses (training, inference,
  // dataset_server, …) survive the restart and are reattached on the
  // new server's startup path.
  const [restarting, setRestarting] = useState(false);
  const restartServer = useCallback(async () => {
    if (
      !window.confirm(
        "Restart the forgather server process? Running training / inference / dataset jobs keep running, but the webui will briefly disconnect.",
      )
    ) {
      return;
    }
    setRestarting(true);
    try {
      await api.restartServer();
    } catch (e) {
      window.alert(
        `Restart request failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      setRestarting(false);
      return;
    }
    // Wait for the server to drop, then poll /api/health until it
    // answers again. Hard-reload so every cached query refetches
    // against the rebooted server.
    let sawDown = false;
    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline) {
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (r.ok) {
          if (sawDown) {
            window.location.reload();
            return;
          }
        } else {
          sawDown = true;
        }
      } catch {
        sawDown = true;
      }
      await new Promise((r) => setTimeout(r, 750));
    }
    setRestarting(false);
    window.alert(
      "Server did not come back within 60 seconds. Check the terminal where it was launched.",
    );
  }, []);

  // Shutdown is a one-way exit (unlike restart, which polls /api/health
  // until the rebooted process answers again). We just open a modal so
  // the operator can pick "stop server only" vs "stop all jobs and
  // shutdown"; the modal calls api.shutdownServer and then we flip
  // ``shuttingDown`` so the rest of the UI shows that the server is
  // going away rather than that the request is hanging.
  const [shutdownOpen, setShutdownOpen] = useState(false);
  const [shuttingDown, setShuttingDown] = useState(false);
  const onShutdownStarted = useCallback(() => {
    setShutdownOpen(false);
    setShuttingDown(true);
  }, []);

  const openHelp = useCallback(
    async (menu: ToolHelpMenu) => {
      // 1. Build the absolute path the Docs API expects.
      let repoRoot = repoRootQ.data?.repo_root;
      if (!repoRoot) {
        try {
          const r = await api.docsRepoRoot();
          repoRoot = r.repo_root;
        } catch {
          // Network blip: surface nothing better than the built-in fallback.
          repoRoot = "";
        }
      }
      const absPath = repoRoot
        ? `${repoRoot.replace(/\/+$/, "")}/${menu.docRelpath}`
        : null;

      // 2. If a MkDocs serve is alive, prefer rendering through it —
      //    nicer theme + search than the lightweight built-in viewer.
      try {
        const jobs = await api.listJobs(false);
        const mk = jobs.find(
          (j) => j.job_type === "mkdocs" && j.alive && j.port,
        );
        if (mk && mk.port) {
          const host = (mk.job_params?.host as string) || "localhost";
          const safeHost =
            host === "0.0.0.0" || host === "127.0.0.1" || host === "::1"
              ? "localhost"
              : host;
          const url = `http://${safeHost}:${mk.port}/${menu.mkdocsSlug}`;
          window.open(url, "_blank", "noopener,noreferrer");
          return;
        }
      } catch {
        // Fall through to the built-in viewer.
      }

      // 3. Built-in Docs view.
      if (absPath) openDocs(absPath);
    },
    [repoRootQ.data, openDocs],
  );

  // Per-tool button: standard click opens the modal; right-click opens
  // the help menu seeded with the matching doc. ``extraItems`` lets a
  // tool surface lightweight side-affordances (e.g. "Edit
  // Configuration…") in the same context menu, without inventing a
  // second menu type.
  interface ToolEntry {
    icon: string;
    label: string;
    title: string;
    onOpen: () => void;
    docRelpath: string;
    mkdocsSlug: string;
    extraItems?: ToolExtraMenuItem[];
    // Filled in for entries under "Services" — the backend service
    // type each launcher creates instances of. Used to fan out the
    // configured-service list under the matching launcher row.
    serviceType?: "inference" | "dataset" | "tensorboard" | "mkdocs" | "diloco";
  }

  // Long-running spawned processes the operator wants to launch and
  // then forget about (inference, datasets, dashboards). Split out from
  // TOOLS so the one-shot model-manipulation utilities aren't visually
  // mixed with persistent services in the sidebar.
  const SERVICES: ToolEntry[] = [
    {
      icon: "🔮",
      label: "Inference…",
      title:
        "Serve an arbitrary model directory — project affiliation optional",
      onOpen: () => setStartServerOpen(true),
      docRelpath: "tools/inference_server/README.md",
      mkdocsSlug: "tools/inference_server/",
      serviceType: "inference",
    },
    {
      icon: "🗂",
      label: "Dataset…",
      title:
        "Run the Forgather dataset server — clients route fast_load_iterable_dataset over HTTP via FORGATHER_DATASET_SERVER",
      onOpen: () => setDatasetServerOpen(true),
      docRelpath: "tools/dataset_server/README.md",
      mkdocsSlug: "tools/dataset_server/",
      serviceType: "dataset",
    },
    {
      icon: "📊",
      label: "TensorBoard…",
      title: "Open TensorBoard against any logdir on disk",
      onOpen: () => setTensorboardOpen(true),
      docRelpath: "docs/guides/tensorboard.md",
      mkdocsSlug: "guides/tensorboard/",
      serviceType: "tensorboard",
    },
    {
      icon: "📖",
      label: "MkDocs…",
      title:
        "Serve Forgather's documentation locally with live rebuild on edit. Defaults to the bundled mkdocs.yml; the served URL appears as a clickable link on the resulting Job card.",
      onOpen: () => setMkdocsOpen(true),
      docRelpath: "docs/guides/mkdocs.md",
      mkdocsSlug: "guides/mkdocs/",
      serviceType: "mkdocs",
    },
    {
      icon: "🧩",
      label: "DiLoCo…",
      title:
        "Start a DiLoCo parameter server. CPU-only, long-lived; holds global model parameters and accepts pseudo-gradient submissions from workers over HTTP.",
      onOpen: () => setDilocoServerOpen(true),
      docRelpath: "docs/trainers/diloco.md",
      mkdocsSlug: "trainers/diloco/",
      serviceType: "diloco",
    },
  ];

  // One-shot model-manipulation utilities (evaluate / convert /
  // finalize / update). Distinct from SERVICES which spawn long-running
  // processes.
  const TOOLS: ToolEntry[] = [
    {
      icon: "📐",
      label: "Evaluate…",
      title: "Run loss/perplexity evaluation against any model directory",
      onOpen: () => setEvaluateOpen(true),
      docRelpath: "docs/guides/evaluating-models.md",
      mkdocsSlug: "guides/evaluating-models/",
    },
    {
      icon: "🔁",
      label: "Convert Model…",
      title: "Convert between Huggingface and Forgather model formats",
      onOpen: () => setConvertOpen(true),
      docRelpath: "docs/guides/model-conversion.md",
      mkdocsSlug: "guides/model-conversion/",
    },
    {
      icon: "📦",
      label: "Finalize Model…",
      title: "Finalize a trained model into a clean output directory",
      onOpen: () => setFinalizeOpen(true),
      docRelpath: "docs/guides/finalize-model.md",
      mkdocsSlug: "guides/finalize-model/",
    },
    {
      icon: "⬆️",
      label: "Update Model…",
      title: "Migrate a saved Forgather model to the current source schema",
      onOpen: () => setUpdateOpen(true),
      docRelpath: "docs/guides/model-update.md",
      mkdocsSlug: "guides/model-update/",
    },
  ];

  return (
    <div
      className={
        "app" +
        (sidebarCollapsed ? " sidebar-collapsed" : "") +
        (demoMode ? " demo-mode" : "")
      }
    >
      {/*
        Both the collapsed strip and the expanded layout stay mounted so
        ProjectTree's local expansion state (which workspaces / projects /
        artifact groups are open) survives a collapse/expand cycle.
      */}
      <aside
        className={"app-sidebar" + (sidebarCollapsed ? " collapsed" : "")}
      >
        <div className="sidebar-collapsed-content">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed(false)}
            title="Expand sidebar (Ctrl+B)"
            aria-label="Expand sidebar"
          >
            <SidebarIcon />
          </button>
          {demoMode && (
            <span
              className="sidebar-header-demo-chip sidebar-header-demo-chip-collapsed"
              role="status"
              aria-label="Read-only demo mode"
              title="Read-only demo mode — mutating actions are blocked. Expand the sidebar for details."
            >
              DEMO
            </span>
          )}
          <nav className="sidebar-views icon-only">
            {visibleViews.map((v) => (
              <button
                key={v.id}
                className={view === v.id ? "active" : ""}
                onClick={() => setView(v.id)}
                title={v.label}
              >
                <span className="view-icon">{v.icon}</span>
              </button>
            ))}
          </nav>
        </div>
        <div className="sidebar-expanded-content">
          <header
            className="sidebar-header"
            // Right-click anywhere on the header surfaces a small menu
            // whose only entry today is a Help… link to the server
            // reference doc — same plumbing the per-tool right-click
            // menus use (ToolHelpMenu + openHelp). Add per-app actions
            // here as the need shows up.
            onContextMenu={(e) => {
              e.preventDefault();
              setToolHelpMenu({
                x: e.clientX,
                y: e.clientY,
                docRelpath: "tools/forgather_server/README.md",
                mkdocsSlug: "forgather-server/",
                label: "Forgather Server",
              });
            }}
            title="Right-click for help"
          >
            <h1>Forgather Server</h1>
            {serverVersion && (
              <span
                className="sidebar-header-version"
                title={`Forgather ${serverVersion}`}
              >
                v{serverVersion}
              </span>
            )}
            {demoMode && (
              <span
                className="sidebar-header-demo-chip"
                role="status"
                aria-label={
                  "Read-only demo mode is active. Mutating actions " +
                  "(file edits, job submission, server admin) are " +
                  "blocked; read-only browsing still works."
                }
                title={
                  "This server is running with --demo: mutating actions " +
                  "(file edits, job submission, server admin, etc.) are " +
                  "blocked. Read-only browsing still works."
                }
              >
                DEMO MODE
              </span>
            )}
            <div className="sidebar-header-actions">
              <button
                className="sidebar-toggle"
                onClick={() => setSidebarCollapsed(true)}
                title="Collapse sidebar (Ctrl+B)"
                aria-label="Collapse sidebar"
              >
                <SidebarIcon />
              </button>
            </div>
          </header>

          {/* Nodes group: peer hostnames + health status. Hidden in
              standalone mode. Clicking a peer opens its webui in a
              new tab using a cluster-bearer SSO URL. Placed above
              Views because it's the highest-level navigation context
              — which node am I looking at — and most useful when it's
              the first thing the eye lands on. Distinct from the
              "Cluster" view in the Views section: this surface is
              about navigating between nodes; that one is about the
              cluster's internal state. */}
          {clusterActive && (
            <details
              className="sidebar-cluster-details"
              open={clusterOpen}
              onToggle={(e) => {
                if (e.target !== e.currentTarget) return;
                setClusterOpen(
                  (e.currentTarget as HTMLDetailsElement).open,
                );
              }}
            >
              <summary>
                Nodes
                {nodesCount > 0 && (
                  <span className="badge">{nodesCount}</span>
                )}
              </summary>
              <ClusterSidebarPanel
                selfNodeId={clusterSelfQ.data?.node_id ?? null}
                masterNodeId={
                  clusterMembersQ.data?.master_node_id ?? null
                }
              />
            </details>
          )}

          <details
            className="sidebar-views-details"
            open={viewsOpen}
            onToggle={(e) => {
              if (e.target !== e.currentTarget) return;
              setViewsOpen((e.currentTarget as HTMLDetailsElement).open);
            }}
          >
            <summary>Views</summary>
            <nav className="sidebar-views">
              {visibleViews.map((v) => {
                return (
                  <button
                    key={v.id}
                    className={view === v.id ? "active" : ""}
                    onClick={() => setView(v.id)}
                  >
                    <span className="view-icon">{v.icon}</span>
                    <span className="view-label">{v.label}</span>
                    {v.id === "jobs" && queuedCount > 0 && (
                      <span
                        className="badge"
                        title={`${queuedCount} queued`}
                      >
                        {queuedCount}
                      </span>
                    )}
                    {v.id === "jobs" && runningCount > 0 && (
                      <span
                        className="badge badge-running"
                        title={`${runningCount} running`}
                      >
                        {runningCount}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>
          </details>

          <details
            className="sidebar-tools"
            open={servicesOpen}
            onToggle={(e) =>
              setServicesOpen((e.target as HTMLDetailsElement).open)
            }
          >
            <summary>Services</summary>
            <div className="sidebar-tools-body">
              {SERVICES.map((tool) => {
                const t = tool.serviceType;
                const count = t ? servicesByType[t] ?? 0 : 0;
                const runningCount = t ? runningServicesByType[t] ?? 0 : 0;
                const open = t ? !!servicesCategoryOpen[t] : false;
                const showChevron = !!t && count > 0;
                return (
                  <div
                    key={tool.label}
                    className="services-category"
                  >
                    <div className="services-category-row">
                      {showChevron ? (
                        <button
                          className="services-category-chevron"
                          onClick={() =>
                            setServicesCategoryOpen((s) => ({
                              ...s,
                              [t!]: !s[t!],
                            }))
                          }
                          title={open ? "Collapse" : "Expand"}
                          aria-label={open ? "Collapse" : "Expand"}
                        >
                          {open ? "▾" : "▸"}
                        </button>
                      ) : (
                        // Placeholder so launcher labels stay aligned
                        // whether or not their type currently has any
                        // configured instances.
                        <span
                          className="services-category-chevron services-category-chevron-spacer"
                          aria-hidden="true"
                        />
                      )}
                      <button
                        className="sidebar-tool-btn"
                        onClick={tool.onOpen}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setToolHelpMenu({
                            x: e.clientX,
                            y: e.clientY,
                            docRelpath: tool.docRelpath,
                            mkdocsSlug: tool.mkdocsSlug,
                            label: tool.label,
                            extraItems: tool.extraItems,
                          });
                        }}
                        title={tool.title}
                      >
                        <span className="sidebar-tool-btn-label">
                          {tool.icon} {tool.label}
                        </span>
                        {runningCount > 0 && (
                          <span className="badge">{runningCount}</span>
                        )}
                      </button>
                    </div>
                    {showChevron && open && (
                      <div className="services-category-body">
                        <ServicesPanel
                          filterType={t!}
                          onSwitchView={(v) => setView(v)}
                          onEditService={setEditingService}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
              <div className="sidebar-tools-hint muted">
                Right-click any service for help.
              </div>
            </div>
          </details>

          <details
            className="sidebar-tools"
            open={toolsOpen}
            onToggle={(e) =>
              setToolsOpen((e.target as HTMLDetailsElement).open)
            }
          >
            <summary>Tools</summary>
            <div className="sidebar-tools-body">
              {TOOLS.map((tool) => (
                <button
                  key={tool.label}
                  className="sidebar-tool-btn"
                  onClick={tool.onOpen}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    setToolHelpMenu({
                      x: e.clientX,
                      y: e.clientY,
                      docRelpath: tool.docRelpath,
                      mkdocsSlug: tool.mkdocsSlug,
                      label: tool.label,
                      extraItems: tool.extraItems,
                    });
                  }}
                  title={tool.title}
                >
                  {tool.icon} {tool.label}
                </button>
              ))}
              <div className="sidebar-tools-hint muted">
                Right-click any tool for help.
              </div>
            </div>
          </details>

          <details
            className="sidebar-search-roots-details"
            open={searchRootsOpen}
            onToggle={(e) => {
              if (e.target !== e.currentTarget) return;
              setSearchRootsOpen(
                (e.currentTarget as HTMLDetailsElement).open,
              );
            }}
          >
            <summary>Search Roots</summary>
            <div className="sidebar-search-roots">
              <SearchRootsPanel />
            </div>
          </details>

          <details
            className="sidebar-projects-details"
            open={projectsOpen}
            // Guard against bubbled `toggle` events: <details>'s toggle
            // event propagates, so any nested <details> inside ProjectTree
            // would otherwise stomp on this section's open state.
            onToggle={(e) => {
              if (e.target !== e.currentTarget) return;
              setProjectsOpen((e.currentTarget as HTMLDetailsElement).open);
            }}
          >
            <summary>Projects</summary>
            <div className="sidebar-projects">
              <ProjectTree
                onSelect={onConfigSelect}
                onProjectOpen={onProjectOpen}
                selection={selected}
                setSelection={setSelectionAndGoToProjects}
                onEditTemplate={openFileForEdit}
                onJobSubmitted={onJobSubmitted}
                onRevealInFiles={revealInFiles}
              />
            </div>
          </details>

          <details
            className="sidebar-files-details"
            open={filesOpen}
            onToggle={(e) => {
              if (e.target !== e.currentTarget) return;
              setFilesOpen((e.currentTarget as HTMLDetailsElement).open);
            }}
          >
            <summary>Files</summary>
            <div className="sidebar-files">
              <FilesTree
                onOpenFile={openFileForEdit}
                onDropPath={filesApi.dropPath}
                onOpenDoc={openDocs}
                revealRequest={revealRequest}
              />
            </div>
          </details>

          {/* Footer bar pinned to the bottom of the sidebar. Gear opens
              the YAML config the server loaded at startup in the editor
              so the operator can tweak persistent server defaults. */}
          <div className="sidebar-footer">
            <button
              className="sidebar-footer-gear"
              onClick={refresh}
              title="Re-read projects, configs, and templates from disk"
              aria-label="Refresh data"
            >
              <ReloadIcon />
            </button>
            <button
              className={
                "sidebar-footer-gear sched-footer-btn " +
                (schedEnabled ? "running" : "paused")
              }
              onClick={() => toggleSched.mutate(!schedEnabled)}
              disabled={demoMode || toggleSched.isPending || schedQ.isLoading}
              title={
                demoMode
                  ? "Read-only demo mode — scheduler controls are disabled"
                  : schedEnabled
                    ? "Scheduler running — click to pause"
                    : "Scheduler paused — click to run"
              }
              aria-label={schedEnabled ? "Pause scheduler" : "Run scheduler"}
            >
              {schedEnabled ? "⏸" : "▶"}
            </button>
            <button
              className="sidebar-footer-gear"
              onClick={restartServer}
              disabled={demoMode || restarting || shuttingDown}
              title={
                demoMode
                  ? "Read-only demo mode — server restart is disabled"
                  : restarting
                    ? "Waiting for the server to come back up…"
                    : "Restart the forgather server (running jobs survive)"
              }
              aria-label="Restart server"
            >
              <RestartIcon />
            </button>
            <button
              className="sidebar-footer-gear"
              onClick={() => setShutdownOpen(true)}
              disabled={demoMode || shuttingDown || restarting}
              title={
                demoMode
                  ? "Read-only demo mode — server shutdown is disabled"
                  : shuttingDown
                    ? "Shutdown in progress…"
                    : "Shutdown the forgather server"
              }
              aria-label="Shutdown server"
            >
              <PowerIcon />
            </button>
            {!demoMode && (
              <button
                className="sidebar-footer-gear"
                onClick={openServerConfig}
                disabled={!serverConfigQ.data?.path}
                title={
                  serverConfigQ.data?.path
                    ? `Open server config: ${serverConfigQ.data.path}`
                    : "Server config path unavailable"
                }
                aria-label="Open server config"
              >
                <GearIcon />
              </button>
            )}
          </div>
        </div>
      </aside>

      {/*
        All views stay mounted while hidden so state, DOM positions, and
        WebSocket streams survive view switches.
      */}
      <div className="app-main">
        <div
          className="view-panel"
          style={view === "projects" ? undefined : { display: "none" }}
        >
          <main className="main">
            {selected === null && (
              <div className="pane-state muted">
                Select a configuration from the left to inspect it.
              </div>
            )}
            {selected?.kind === "config" && (
              <ConfigViewer
                project={selected.project}
                config={selected.config}
                tab={currentTab}
                onTabChange={(t) => setTab(t)}
                onEditTemplate={openFileForEdit}
                onSelectConfig={onConfigSelect}
                onJobSubmitted={onJobSubmitted}
                onOpenDoc={openDocs}
                onEditFile={openFileForEdit}
              />
            )}
            {selected?.kind === "log" && (
              <LogDetailPanel
                project={selected.project}
                config={selected.config}
                run_dir={selected.run_dir}
                run_id={selected.run_id}
                onBack={backToConfig}
              />
            )}
            {selected?.kind === "checkpoint" && (
              <CheckpointDetailPanel
                project={selected.project}
                config={selected.config}
                output_dir={selected.output_dir}
                checkpoint={selected.checkpoint}
                onBack={backToConfig}
              />
            )}
            {selected?.kind === "eval" && (
              <EvalDetailPanel
                project={selected.project}
                config={selected.config}
                output_dir={selected.output_dir}
                evaluation={selected.evaluation}
                onBack={backToConfig}
              />
            )}
          </main>
        </div>
        <div
          className="view-panel"
          style={view === "edit" ? undefined : { display: "none" }}
        >
          <FilesPanel api={filesApi} onOpenDoc={openDocs} />
        </div>
        <div
          className="view-panel"
          style={view === "docs" ? undefined : { display: "none" }}
        >
          <DocsPanel
            path={docsPath}
            onNavigate={docsNavigate}
            onEdit={openFileForEdit}
            canGoBack={docsBackStack.length > 0}
            onBack={docsBack}
            restoreScrollTop={pendingDocsScroll}
            onScrollRestored={() => setPendingDocsScroll(null)}
          />
        </div>
        <div
          className="view-panel"
          style={view === "gpus" ? undefined : { display: "none" }}
        >
          <GpuPanel />
        </div>
        {clusterActive && (
          <div
            className="view-panel"
            style={view === "cluster" ? undefined : { display: "none" }}
          >
            <ClusterPanel onOpenInExplore={openInExplore} />
          </div>
        )}
        <div
          className="view-panel"
          style={view === "jobs" ? undefined : { display: "none" }}
        >
          <JobsPanel
            autoWatchJobId={autoWatchJobId}
            onAutoWatchConsumed={() => setAutoWatchJobId(null)}
            onOpenInferenceServer={openInferenceServer}
            onOpenDatasetServer={openDatasetServer}
            onOpenDiLoCoServer={openDiLoCoServer}
          />
        </div>
        <div
          className="view-panel"
          style={view === "inference" ? undefined : { display: "none" }}
        >
          <InferencePanel
            pendingAnalyze={pendingAnalyze}
            onAnalyzeConsumed={clearPendingAnalyze}
            pendingServerPick={pendingInferenceServer}
            onServerPickConsumed={clearPendingInferenceServer}
          />
        </div>
        <div
          className="view-panel"
          style={view === "datasets" ? undefined : { display: "none" }}
        >
          <DatasetsPanel
            pendingExplore={pendingExplore}
            onPreselectConsumed={clearPendingExplore}
            onOpenInExplore={openInExplore}
            onAnalyzeText={analyzeText}
            pendingServerPick={pendingDatasetServer}
            onServerPickConsumed={clearPendingDatasetServer}
          />
        </div>
        <div
          className="view-panel"
          style={view === "diloco" ? undefined : { display: "none" }}
        >
          <DiLoCoPanel
            pendingServerPick={pendingDiLoCoServer}
            onServerPickConsumed={clearPendingDiLoCoServer}
          />
        </div>
      </div>

      {startServerOpen && (
        <InferenceModal
          checkpointPath={null}
          onClose={() => setStartServerOpen(false)}
          onSubmitted={onJobSubmitted}
          onServiceCreated={expandServicesCategory}
        />
      )}
      {datasetServerOpen && (
        <DatasetServerModal
          onClose={() => setDatasetServerOpen(false)}
          onSubmitted={onJobSubmitted}
          onServiceCreated={expandServicesCategory}
        />
      )}
      {tensorboardOpen && (
        <TensorBoardModal
          global
          initialLogdir=""
          initialWindowTitle=""
          onClose={() => setTensorboardOpen(false)}
          onSubmitted={onJobSubmitted}
          onServiceCreated={expandServicesCategory}
        />
      )}
      {mkdocsOpen && (
        <MkDocsModal
          onClose={() => setMkdocsOpen(false)}
          onSubmitted={onJobSubmitted}
          onServiceCreated={expandServicesCategory}
        />
      )}
      {dilocoServerOpen && (
        <DiLoCoServerModal
          onClose={() => setDilocoServerOpen(false)}
          onSubmitted={onJobSubmitted}
          onServiceCreated={expandServicesCategory}
        />
      )}
      {/* Edit-service modals — same components as the create-mode mounts
          above, but routed via editingService.type. Only one renders at a
          time because only one type matches. */}
      {editingService && editingService.service.type === "inference" && (
        <InferenceModal
          checkpointPath={null}
          editingService={{
            name: editingService.service.name,
            enabled: editingService.service.enabled,
            running: editingService.running,
            args: editingService.service.args,
          }}
          onClose={() => setEditingService(null)}
        />
      )}
      {editingService && editingService.service.type === "dataset" && (
        <DatasetServerModal
          editingService={{
            name: editingService.service.name,
            enabled: editingService.service.enabled,
            running: editingService.running,
            args: editingService.service.args,
          }}
          onClose={() => setEditingService(null)}
        />
      )}
      {editingService && editingService.service.type === "tensorboard" && (
        <TensorBoardModal
          global
          initialLogdir=""
          initialWindowTitle=""
          editingService={{
            name: editingService.service.name,
            enabled: editingService.service.enabled,
            running: editingService.running,
            args: editingService.service.args,
          }}
          onClose={() => setEditingService(null)}
        />
      )}
      {editingService && editingService.service.type === "mkdocs" && (
        <MkDocsModal
          editingService={{
            name: editingService.service.name,
            enabled: editingService.service.enabled,
            running: editingService.running,
            args: editingService.service.args,
          }}
          onClose={() => setEditingService(null)}
        />
      )}
      {editingService && editingService.service.type === "diloco" && (
        <DiLoCoServerModal
          editingService={{
            name: editingService.service.name,
            enabled: editingService.service.enabled,
            running: editingService.running,
            args: editingService.service.args,
          }}
          onClose={() => setEditingService(null)}
        />
      )}
      {shutdownOpen && (
        <ShutdownModal
          onClose={() => setShutdownOpen(false)}
          onShutdownStarted={onShutdownStarted}
        />
      )}
      {convertOpen && (
        <ConvertModal
          onClose={() => setConvertOpen(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {finalizeOpen && (
        <FinalizeModal
          onClose={() => setFinalizeOpen(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {updateOpen && (
        <UpdateModal
          onClose={() => setUpdateOpen(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {evaluateOpen && (
        <EvalModal
          onClose={() => setEvaluateOpen(false)}
          onSubmitted={onJobSubmitted}
        />
      )}

      {toolHelpMenu && (
        <ContextMenu
          x={toolHelpMenu.x}
          y={toolHelpMenu.y}
          onClose={() => setToolHelpMenu(null)}
        >
          <div className="context-menu-header muted">{toolHelpMenu.label}</div>
          {toolHelpMenu.extraItems?.map((item) => (
            <button
              key={item.label}
              onClick={() => {
                const fn = item.onChoose;
                setToolHelpMenu(null);
                void fn();
              }}
            >
              {item.label}
            </button>
          ))}
          <button
            onClick={() => {
              const t = toolHelpMenu;
              setToolHelpMenu(null);
              void openHelp(t);
            }}
          >
            Help…
          </button>
        </ContextMenu>
      )}
    </div>
  );
}
