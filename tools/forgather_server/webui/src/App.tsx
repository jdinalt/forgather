import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, CheckpointEntry, ConfigInfo, EvalEntry, ProjectInfo } from "./api";
import { getAutoWatchTty } from "./autoWatch";
import { ProjectTree } from "./components/ProjectTree";
import { ConfigViewer } from "./components/ConfigViewer";
import { GpuPanel } from "./components/GpuPanel";
import { EvalModal } from "./components/EvalModal";
import { InferenceModal } from "./components/InferenceModal";
import { InferencePanel } from "./components/InferencePanel";
import { JobsPanel } from "./components/JobsPanel";
import { QueuePanel } from "./components/QueuePanel";
import { LogDetailPanel } from "./components/LogDetailPanel";
import { CheckpointDetailPanel } from "./components/CheckpointDetailPanel";
import { EvalDetailPanel } from "./components/EvalDetailPanel";
import { TensorBoardModal } from "./components/TensorBoardModal";
import { MkDocsModal } from "./components/MkDocsModal";
import { ConvertModal } from "./components/ConvertModal";
import { FinalizeModal } from "./components/FinalizeModal";
import { UpdateModal } from "./components/UpdateModal";
import { FilesPanel } from "./components/FilesPanel";
import { FilesTree } from "./components/FilesTree";
import { SearchRootsPanel } from "./components/SearchRootsPanel";
import { useFilesState } from "./files-state";

type View = "projects" | "edit" | "gpus" | "jobs" | "queue" | "inference";
export type ConfigTab = "info" | "pp" | "code" | "graph" | "templates" | "debug";

const VIEWS: { id: View; label: string; icon: string }[] = [
  { id: "projects", label: "Projects", icon: "📁" },
  { id: "edit", label: "Edit", icon: "✎" },
  { id: "gpus", label: "GPUs", icon: "🖥" },
  { id: "queue", label: "Queue", icon: "📋" },
  { id: "jobs", label: "Jobs", icon: "⚙" },
  { id: "inference", label: "Inference", icon: "🔮" },
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
  const [view, setView] = useState<View>("projects");
  const [selected, setSelected] = useState<Selection>(null);
  // Tab state lives here so opening a project can both pick its default
  // config AND switch to "info" in one render cycle.
  const [tab, setTab] = useState<ConfigTab>("info");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [searchRootsOpen, setSearchRootsOpen] = useState(false);
  const [projectsOpen, setProjectsOpen] = useState(false);
  const [filesOpen, setFilesOpen] = useState(false);
  const [viewsOpen, setViewsOpen] = useState(false);
  const [startServerOpen, setStartServerOpen] = useState(false);
  const [tensorboardOpen, setTensorboardOpen] = useState(false);
  const [mkdocsOpen, setMkdocsOpen] = useState(false);
  const [convertOpen, setConvertOpen] = useState(false);
  const [finalizeOpen, setFinalizeOpen] = useState(false);
  const [updateOpen, setUpdateOpen] = useState(false);
  const [evaluateOpen, setEvaluateOpen] = useState(false);
  // Set when a submit modal closes with the "Watch TTY on start" toggle on.
  // The Jobs view consumes this once the job appears in the polled list and
  // clears it back to null via onAutoWatchConsumed.
  const [autoWatchJobId, setAutoWatchJobId] = useState<string | null>(null);
  const filesApi = useFilesState();
  const qc = useQueryClient();

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

  const schedQ = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: api.schedulerStatus,
    refetchInterval: 3000,
  });
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

  return (
    <div className={"app" + (sidebarCollapsed ? " sidebar-collapsed" : "")}>
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
            title="Expand sidebar"
            aria-label="Expand sidebar"
          >
            <SidebarIcon />
          </button>
          <nav className="sidebar-views icon-only">
            {VIEWS.map((v) => (
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
          <header className="sidebar-header">
            <h1>Forgather Server</h1>
            <span className="muted">preview</span>
            <div className="sidebar-header-actions">
              <button
                className="refresh-btn"
                onClick={refresh}
                title="Re-read projects, configs, and templates from disk"
              >
                ⟳ Refresh
              </button>
              <button
                className={
                  "sched-btn " + (schedEnabled ? "running" : "paused")
                }
                onClick={() => toggleSched.mutate(!schedEnabled)}
                disabled={toggleSched.isPending || schedQ.isLoading}
                title={
                  schedEnabled
                    ? "Scheduler running — click to pause"
                    : "Scheduler paused — click to run"
                }
                aria-label={schedEnabled ? "Pause scheduler" : "Run scheduler"}
              >
                {schedEnabled ? "⏸" : "▶"}
              </button>
              <button
                className="sidebar-toggle"
                onClick={() => setSidebarCollapsed(true)}
                title="Collapse sidebar"
                aria-label="Collapse sidebar"
              >
                <SidebarIcon />
              </button>
            </div>
          </header>

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
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  className={view === v.id ? "active" : ""}
                  onClick={() => setView(v.id)}
                >
                  <span className="view-icon">{v.icon}</span>
                  <span className="view-label">{v.label}</span>
                </button>
              ))}
            </nav>
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
              <button
                className="sidebar-tool-btn"
                onClick={() => setStartServerOpen(true)}
                title="Serve an arbitrary model directory — project affiliation optional"
              >
                🔮 Serve Inference…
              </button>
              <button
                className="sidebar-tool-btn"
                onClick={() => setEvaluateOpen(true)}
                title="Run loss/perplexity evaluation against any model directory"
              >
                📐 Evaluate…
              </button>
              <button
                className="sidebar-tool-btn"
                onClick={() => setTensorboardOpen(true)}
                title="Open TensorBoard against any logdir on disk"
              >
                📊 TensorBoard…
              </button>
              <button
                className="sidebar-tool-btn"
                onClick={() => setMkdocsOpen(true)}
                title="Serve Forgather's documentation locally with live rebuild on edit. Defaults to the bundled mkdocs.yml; the served URL appears as a clickable link on the resulting Job card."
              >
                📖 MkDocs…
              </button>
              <button
                className="sidebar-tool-btn"
                onClick={() => setConvertOpen(true)}
                title="Convert between Huggingface and Forgather model formats"
              >
                🔁 Convert Model…
              </button>
              <button
                className="sidebar-tool-btn"
                onClick={() => setFinalizeOpen(true)}
                title="Finalize a trained model into a clean output directory"
              >
                📦 Finalize Model…
              </button>
              <button
                className="sidebar-tool-btn"
                onClick={() => setUpdateOpen(true)}
                title="Migrate a saved Forgather model to the current source schema"
              >
                ⬆️ Update Model…
              </button>
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
              />
            </div>
          </details>
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
          <FilesPanel api={filesApi} />
        </div>
        <div
          className="view-panel"
          style={view === "gpus" ? undefined : { display: "none" }}
        >
          <GpuPanel />
        </div>
        <div
          className="view-panel"
          style={view === "jobs" ? undefined : { display: "none" }}
        >
          <JobsPanel
            autoWatchJobId={autoWatchJobId}
            onAutoWatchConsumed={() => setAutoWatchJobId(null)}
          />
        </div>
        <div
          className="view-panel"
          style={view === "queue" ? undefined : { display: "none" }}
        >
          <QueuePanel />
        </div>
        <div
          className="view-panel"
          style={view === "inference" ? undefined : { display: "none" }}
        >
          <InferencePanel />
        </div>
      </div>

      {startServerOpen && (
        <InferenceModal
          checkpointPath={null}
          onClose={() => setStartServerOpen(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {tensorboardOpen && (
        <TensorBoardModal
          global
          initialLogdir=""
          initialWindowTitle=""
          onClose={() => setTensorboardOpen(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {mkdocsOpen && (
        <MkDocsModal
          onClose={() => setMkdocsOpen(false)}
          onSubmitted={onJobSubmitted}
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
    </div>
  );
}
