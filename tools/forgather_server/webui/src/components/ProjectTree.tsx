import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, CheckpointEntry, ConfigInfo, EvalEntry, ModelEntry, ProjectInfo, RunEntry, WorkspaceCluster } from "../api";
import { CleanOutputModal } from "./CleanOutputModal";
import { ContextMenu } from "./ContextMenu";
import { ConvertModal } from "./ConvertModal";
import { EvalModal } from "./EvalModal";
import { FinalizeModal } from "./FinalizeModal";
import { InferenceModal } from "./InferenceModal";
import { NewProjectModal } from "./NewProjectModal";
import { NewTemplateModal } from "./NewTemplateModal";
import { OverridesModal } from "./OverridesModal";
import { DatasetSubmitModal } from "./DatasetSubmitModal";
import { ModelSubmitModal } from "./ModelSubmitModal";
import { SubmitModal } from "./SubmitModal";
import { ConfigTensorBoardModal } from "./TensorBoardModal";
import { Selection } from "../App";

type ConfigAction = "submit" | "overrides" | "clean" | "tensorboard" | "delete";

interface ContextTarget {
  project: ProjectInfo;
  config: ConfigInfo;
  x: number;
  y: number;
}

interface ActiveModal {
  action: ConfigAction;
  project: ProjectInfo;
  config: ConfigInfo;
}

// Separate state for action modals seeded from a config's output_dir.
// Serve / Eval also carry a checkpoint path (a specific ckpt may be the
// trigger via the checkpoint right-click menu); Convert / Finalize don't
// have a checkpoint-targeted entry point in this menu, so the slot stays
// null for them.
interface ServeEvalModal {
  action: "serve" | "eval" | "convert" | "finalize";
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  checkpointPath: string | null;
}

interface CheckpointMenuTarget {
  x: number;
  y: number;
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  checkpoint: CheckpointEntry;
}

interface LeafMenuTarget {
  x: number;
  y: number;
  kind: "log" | "eval";
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  item_dir: string;
  label: string;
}

interface GroupMenuTarget {
  x: number;
  y: number;
  kind: "logs" | "checkpoints" | "evals";
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  count: number;
}

interface ProjectMenuTarget {
  x: number;
  y: number;
  project: ProjectInfo;
}

interface WorkspaceMenuTarget {
  x: number;
  y: number;
  workspace: WorkspaceCluster;
}

interface Props {
  onSelect: (project: ProjectInfo, config: ConfigInfo) => void;
  onProjectOpen: (project: ProjectInfo, defaultConfig: ConfigInfo) => void;
  selection: Selection;
  setSelection: (sel: Selection) => void;
  /** Hand a path to the Files panel (App-level) and switch view. */
  onEditTemplate: (path: string) => void;
  /** Bubble the submitted job's queue id to App so it can decide whether
   *  to switch to the Jobs view + auto-open the TTY. */
  onJobSubmitted?: (queueId: string) => void;
}

export function ProjectTree({
  onSelect,
  onProjectOpen,
  selection,
  setSelection,
  onEditTemplate,
  onJobSubmitted,
}: Props) {
  const qc = useQueryClient();
  const projectsQ = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });

  const [contextTarget, setContextTarget] = useState<ContextTarget | null>(null);
  const [ckptMenuTarget, setCkptMenuTarget] = useState<CheckpointMenuTarget | null>(null);
  const [leafMenuTarget, setLeafMenuTarget] = useState<LeafMenuTarget | null>(null);
  const [groupMenuTarget, setGroupMenuTarget] = useState<GroupMenuTarget | null>(null);
  const [projectMenuTarget, setProjectMenuTarget] =
    useState<ProjectMenuTarget | null>(null);
  const [workspaceMenuTarget, setWorkspaceMenuTarget] =
    useState<WorkspaceMenuTarget | null>(null);
  const [newProjectModal, setNewProjectModal] =
    useState<WorkspaceCluster | null>(null);
  const [newTemplateModal, setNewTemplateModal] = useState<{
    project: ProjectInfo;
    kind: "config" | "template";
  } | null>(null);
  const [activeModal, setActiveModal] = useState<ActiveModal | null>(null);
  const [serveEvalModal, setServeEvalModal] = useState<ServeEvalModal | null>(null);

  const openNewTemplate = (
    project: ProjectInfo,
    kind: "config" | "template",
  ) => {
    setProjectMenuTarget(null);
    setNewTemplateModal({ project, kind });
  };

  // Kind → subdirectory name under output_dir. "logs" maps to "runs" because
  // that's what the catalog enumerates; checkpoints/evals match 1:1.
  const groupSubdir = (kind: "logs" | "checkpoints" | "evals"): string =>
    kind === "logs" ? "runs" : kind;

  const invalidateAfterDelete = (
    project_dir: string,
    output_dir: string,
    kind: "logs" | "checkpoints" | "evals",
  ) => {
    qc.invalidateQueries({ queryKey: ["project-models", project_dir] });
    if (kind === "logs")
      qc.invalidateQueries({ queryKey: ["model-runs", output_dir] });
    if (kind === "checkpoints")
      qc.invalidateQueries({ queryKey: ["model-checkpoints", output_dir] });
    if (kind === "evals")
      qc.invalidateQueries({ queryKey: ["model-evaluations", output_dir] });
  };

  const deleteLeaf = async (t: LeafMenuTarget) => {
    if (
      !confirm(
        `Delete this ${t.kind} permanently?\n\n${t.label}\n${t.item_dir}\n\nThis cannot be undone.`,
      )
    )
      return;
    try {
      await api.deleteDir(t.item_dir);
      invalidateAfterDelete(
        t.project.project_dir,
        t.output_dir,
        t.kind === "log" ? "logs" : "evals",
      );
      // Reset selection if the deleted leaf is currently selected.
      if (
        (t.kind === "log" &&
          selection?.kind === "log" &&
          selection.run_dir === t.item_dir) ||
        (t.kind === "eval" &&
          selection?.kind === "eval" &&
          selection.evaluation.eval_dir === t.item_dir)
      ) {
        setSelection({ kind: "config", project: t.project, config: t.config });
      }
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const deleteCheckpoint = async (t: CheckpointMenuTarget) => {
    if (
      !confirm(
        `Delete checkpoint step ${t.checkpoint.step} permanently?\n\n${t.checkpoint.checkpoint_dir}\n\nThis cannot be undone.`,
      )
    )
      return;
    try {
      await api.deleteDir(t.checkpoint.checkpoint_dir);
      invalidateAfterDelete(t.project.project_dir, t.output_dir, "checkpoints");
      if (
        selection?.kind === "checkpoint" &&
        selection.checkpoint.checkpoint_dir === t.checkpoint.checkpoint_dir
      ) {
        setSelection({ kind: "config", project: t.project, config: t.config });
      }
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const deleteGroup = async (t: GroupMenuTarget) => {
    // Deleting the parent subdirectory (e.g. <output_dir>/runs) atomically
    // removes every item in the group — matches how the catalog enumerates.
    const parent = `${t.output_dir.replace(/\/+$/, "")}/${groupSubdir(t.kind)}`;
    if (
      !confirm(
        `Delete all ${t.count} ${t.kind} permanently?\n\n${parent}\n\nThis cannot be undone.`,
      )
    )
      return;
    try {
      await api.deleteDir(parent);
      invalidateAfterDelete(t.project.project_dir, t.output_dir, t.kind);
      // Reset selection if it points to anything in the group we just nuked.
      if (
        selection &&
        selection.config.path === t.config.path &&
        ((t.kind === "logs" && selection.kind === "log") ||
          (t.kind === "checkpoints" && selection.kind === "checkpoint") ||
          (t.kind === "evals" && selection.kind === "eval"))
      ) {
        setSelection({ kind: "config", project: t.project, config: t.config });
      }
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const openContextMenu = (
    project: ProjectInfo,
    config: ConfigInfo,
    e: React.MouseEvent,
  ) => {
    e.preventDefault();
    e.stopPropagation();
    setContextTarget({ project, config, x: e.clientX, y: e.clientY });
  };

  const choose = (action: ConfigAction) => {
    if (!contextTarget) return;
    const target = contextTarget;
    setContextTarget(null);
    if (action === "delete") {
      deleteConfigFile(target.project, target.config);
      return;
    }
    setActiveModal({
      action,
      project: target.project,
      config: target.config,
    });
  };

  const deleteWorkspace = async (ws: WorkspaceCluster) => {
    const dir = ws.workspace_root;
    const label = ws.name || basename(dir);
    if (
      !confirm(
        `Delete this entire workspace?\n\n${label}\n${dir}\n\n` +
          `This recursively removes the workspace directory and ` +
          `everything under it: every project, their configs, ` +
          `templates, and any output_models / runs / checkpoints ` +
          `stored inside the workspace tree. Outputs configured to ` +
          `live elsewhere on disk are not touched.\n\n` +
          `This cannot be undone.`,
      )
    ) {
      return;
    }
    const typed = window.prompt(
      `Type the workspace's directory name to confirm:\n\n${basename(dir)}`,
      "",
    );
    if (typed == null) return;
    if (typed.trim() !== basename(dir)) {
      alert("Confirmation text did not match. Workspace not deleted.");
      return;
    }
    try {
      await api.deleteDir(dir);
      qc.invalidateQueries({ queryKey: ["projects"] });
      // Drop the selection if it was inside this workspace.
      if (
        selection &&
        selection.project.workspace_root &&
        (selection.project.workspace_root === dir ||
          selection.project.workspace_root.startsWith(dir + "/") ||
          selection.project.project_dir.startsWith(dir + "/"))
      ) {
        setSelection(null);
      }
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const deleteProject = async (project: ProjectInfo) => {
    const dir = project.project_dir;
    const label = project.name || basename(dir);
    // Two-step: a single confirm for a project that may carry an
    // output_models/ subtree is too easy to wave through. First confirm
    // is the standard "are you sure"; second is a typed-token gate so
    // muscle-memory misses don't take down a workspace.
    if (
      !confirm(
        `Delete this entire project?\n\n${label}\n${dir}\n\n` +
          `This recursively removes the project directory and ` +
          `everything under it: meta.yaml, templates, and any ` +
          `output_models / runs / checkpoints stored inside the ` +
          `project tree. Outputs configured to live elsewhere on ` +
          `disk are not touched.\n\nThis cannot be undone.`,
      )
    ) {
      return;
    }
    const typed = window.prompt(
      `Type the project's directory name to confirm:\n\n${basename(dir)}`,
      "",
    );
    if (typed == null) return;
    if (typed.trim() !== basename(dir)) {
      alert("Confirmation text did not match. Project not deleted.");
      return;
    }
    try {
      await api.deleteDir(dir);
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["project-templates", dir] });
      qc.invalidateQueries({ queryKey: ["project-models", dir] });
      // Drop the selection if it pointed anywhere inside this project.
      if (selection?.project.project_dir === dir) {
        setSelection(null);
      }
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const deleteConfigFile = async (
    project: ProjectInfo,
    config: ConfigInfo,
  ) => {
    if (
      !confirm(
        `Delete this config file?\n\n${config.path}\n\n` +
          `This removes only the config template, not its output_dir or runs. ` +
          `Cannot be undone.`,
      )
    ) {
      return;
    }
    try {
      await api.deleteFile(config.path);
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({
        queryKey: ["project-templates", project.project_dir],
      });
      // If the just-deleted config was the current selection, drop it back
      // to the bare-project view so the right pane doesn't show stale data.
      if (selection?.config.path === config.path) {
        setSelection(null);
      }
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const chooseServeEval = (
    action: "serve" | "eval" | "convert" | "finalize",
    output_dir: string,
    checkpointPath: string | null,
  ) => {
    if (!contextTarget) return;
    setServeEvalModal({
      action,
      project: contextTarget.project,
      config: contextTarget.config,
      output_dir,
      checkpointPath,
    });
    setContextTarget(null);
  };

  return (
    <div className="project-tree">
      <section className="sidebar-section">
        {projectsQ.isLoading && <div>Loading...</div>}
        {projectsQ.error && <div className="err">{String(projectsQ.error)}</div>}
        {projectsQ.data && (
          <WorkspaceForest
            clusters={projectsQ.data}
            onSelect={onSelect}
            onProjectOpen={onProjectOpen}
            selection={selection}
            onContextRequest={openContextMenu}
            onProjectMenu={(project, e) => {
              e.preventDefault();
              e.stopPropagation();
              setProjectMenuTarget({
                x: e.clientX,
                y: e.clientY,
                project,
              });
            }}
            onWorkspaceMenu={(workspace, e) => {
              e.preventDefault();
              e.stopPropagation();
              setWorkspaceMenuTarget({
                x: e.clientX,
                y: e.clientY,
                workspace,
              });
            }}
            setSelection={setSelection}
            onCheckpointMenu={(x, y, project, config, output_dir, checkpoint) =>
              setCkptMenuTarget({ x, y, project, config, output_dir, checkpoint })
            }
            onLeafMenu={setLeafMenuTarget}
            onGroupMenu={setGroupMenuTarget}
          />
        )}
        {projectsQ.data && projectsQ.data.length === 0 && (
          <div className="muted">No projects found. Add a search root above.</div>
        )}
      </section>

      {contextTarget && (
        <ContextMenu
          x={contextTarget.x}
          y={contextTarget.y}
          onClose={() => setContextTarget(null)}
        >
          <ConfigContextMenuItems
            project={contextTarget.project}
            config={contextTarget.config}
            onChoose={choose}
            onChooseServeEval={chooseServeEval}
          />
        </ContextMenu>
      )}

      {ckptMenuTarget && (
        <ContextMenu
          x={ckptMenuTarget.x}
          y={ckptMenuTarget.y}
          onClose={() => setCkptMenuTarget(null)}
        >
          <div className="context-menu-header muted">
            checkpoint-{ckptMenuTarget.checkpoint.step}
          </div>
          <button
            onClick={() => {
              const t = ckptMenuTarget;
              setCkptMenuTarget(null);
              setServeEvalModal({
                action: "serve",
                project: t.project,
                config: t.config,
                output_dir: t.output_dir,
                checkpointPath: t.checkpoint.checkpoint_dir,
              });
            }}
          >
            🔮 Serve Inference…
          </button>
          <button
            onClick={() => {
              const t = ckptMenuTarget;
              setCkptMenuTarget(null);
              setServeEvalModal({
                action: "eval",
                project: t.project,
                config: t.config,
                output_dir: t.output_dir,
                checkpointPath: t.checkpoint.checkpoint_dir,
              });
            }}
          >
            ⚖ Evaluate…
          </button>
          <button
            className="destructive"
            onClick={() => {
              const t = ckptMenuTarget;
              setCkptMenuTarget(null);
              deleteCheckpoint(t);
            }}
          >
            🗑 Delete Permanently…
          </button>
        </ContextMenu>
      )}

      {leafMenuTarget && (
        <ContextMenu
          x={leafMenuTarget.x}
          y={leafMenuTarget.y}
          onClose={() => setLeafMenuTarget(null)}
        >
          <div className="context-menu-header muted">{leafMenuTarget.label}</div>
          <button
            className="destructive"
            onClick={() => {
              const t = leafMenuTarget;
              setLeafMenuTarget(null);
              deleteLeaf(t);
            }}
          >
            🗑 Delete Permanently…
          </button>
        </ContextMenu>
      )}

      {workspaceMenuTarget && (
        <ContextMenu
          x={workspaceMenuTarget.x}
          y={workspaceMenuTarget.y}
          onClose={() => setWorkspaceMenuTarget(null)}
        >
          <div className="context-menu-header muted">
            {workspaceMenuTarget.workspace.name ||
              basename(workspaceMenuTarget.workspace.workspace_root)}
          </div>
          <button
            onClick={() => {
              const ws = workspaceMenuTarget.workspace;
              setWorkspaceMenuTarget(null);
              setNewProjectModal(ws);
            }}
          >
            📁 Create Project…
          </button>
          <button
            className="context-menu-destructive"
            onClick={() => {
              const ws = workspaceMenuTarget.workspace;
              setWorkspaceMenuTarget(null);
              deleteWorkspace(ws);
            }}
            title={workspaceMenuTarget.workspace.workspace_root}
          >
            🗑 Delete Workspace…
          </button>
        </ContextMenu>
      )}

      {projectMenuTarget && (
        <ContextMenu
          x={projectMenuTarget.x}
          y={projectMenuTarget.y}
          onClose={() => setProjectMenuTarget(null)}
        >
          <div className="context-menu-header muted">
            {projectMenuTarget.project.name ||
              basename(projectMenuTarget.project.project_dir)}
          </div>
          <button
            onClick={() =>
              openNewTemplate(projectMenuTarget.project, "config")
            }
          >
            📄 New Config…
          </button>
          <button
            onClick={() =>
              openNewTemplate(projectMenuTarget.project, "template")
            }
          >
            📄 New Template…
          </button>
          <button
            className="context-menu-destructive"
            onClick={() => {
              const project = projectMenuTarget.project;
              setProjectMenuTarget(null);
              deleteProject(project);
            }}
            title={projectMenuTarget.project.project_dir}
          >
            🗑 Delete Project…
          </button>
        </ContextMenu>
      )}

      {groupMenuTarget && (
        <ContextMenu
          x={groupMenuTarget.x}
          y={groupMenuTarget.y}
          onClose={() => setGroupMenuTarget(null)}
        >
          <div className="context-menu-header muted">
            All {groupMenuTarget.kind} ({groupMenuTarget.count})
          </div>
          <button
            className="destructive"
            onClick={() => {
              const t = groupMenuTarget;
              setGroupMenuTarget(null);
              deleteGroup(t);
            }}
          >
            🗑 Delete All Permanently…
          </button>
        </ContextMenu>
      )}

      {newTemplateModal && (
        <NewTemplateModal
          project={newTemplateModal.project}
          kind={newTemplateModal.kind}
          onCreated={(path) => onEditTemplate(path)}
          onClose={() => setNewTemplateModal(null)}
        />
      )}

      {newProjectModal && (
        <NewProjectModal
          workspace={newProjectModal}
          onCreated={() => {
            // Tree refresh comes from the modal's mutation onSuccess hook;
            // nothing else to do here yet — the new project will appear
            // under the workspace once the projects query refetches.
          }}
          onClose={() => setNewProjectModal(null)}
        />
      )}

      {activeModal?.action === "submit" && (
        <SubmitModalRouter
          project={activeModal.project}
          config={activeModal.config}
          onClose={() => setActiveModal(null)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {activeModal?.action === "overrides" && (
        <OverridesModal
          project={activeModal.project}
          config={activeModal.config}
          onClose={() => setActiveModal(null)}
        />
      )}
      {activeModal?.action === "clean" && (
        <CleanOutputModal
          project={activeModal.project}
          config={activeModal.config}
          onClose={() => setActiveModal(null)}
        />
      )}
      {activeModal?.action === "tensorboard" && (
        <ConfigTensorBoardModal
          project={activeModal.project}
          config={activeModal.config}
          onClose={() => setActiveModal(null)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {serveEvalModal?.action === "serve" && (
        <InferenceModal
          modelOutputDir={serveEvalModal.output_dir}
          modelName={basename(serveEvalModal.output_dir)}
          checkpointPath={serveEvalModal.checkpointPath}
          projectDir={serveEvalModal.project.project_dir}
          onClose={() => setServeEvalModal(null)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {serveEvalModal?.action === "eval" && (
        <EvalModal
          modelOutputDir={serveEvalModal.output_dir}
          modelName={basename(serveEvalModal.output_dir)}
          checkpointPath={serveEvalModal.checkpointPath}
          projectDir={serveEvalModal.project.project_dir}
          onClose={() => setServeEvalModal(null)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {serveEvalModal?.action === "convert" && (
        <ConvertModal
          initialSrcPath={serveEvalModal.output_dir}
          onClose={() => setServeEvalModal(null)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {serveEvalModal?.action === "finalize" && (
        <FinalizeModal
          initialSource={serveEvalModal.output_dir}
          onClose={() => setServeEvalModal(null)}
          onSubmitted={onJobSubmitted}
        />
      )}
    </div>
  );
}

/** Picks the correct submit modal based on the config's class. Falls
 *  back to the training-style ``SubmitModal`` while the meta query is
 *  in flight or for any unrecognised class — that path is also what
 *  pure ``type.training_script*`` configs go through. */
function SubmitModalRouter({
  project,
  config,
  onClose,
  onSubmitted,
}: {
  project: ProjectInfo;
  config: ConfigInfo;
  onClose: () => void;
  onSubmitted?: (queueId: string) => void;
}) {
  const metaQ = useQuery({
    queryKey: ["config-meta", project.project_dir, config.name],
    queryFn: () => api.configMeta(project.project_dir, config.name),
    staleTime: 5 * 60 * 1000,
  });
  const cls = metaQ.data?.config_class ?? null;
  if (cls?.startsWith("type.model")) {
    return (
      <ModelSubmitModal
        project={project}
        config={config}
        onClose={onClose}
        onSubmitted={onSubmitted}
      />
    );
  }
  if (cls?.startsWith("type.dataset")) {
    return (
      <DatasetSubmitModal
        project={project}
        config={config}
        onClose={onClose}
        onSubmitted={onSubmitted}
      />
    );
  }
  return (
    <SubmitModal
      project={project}
      config={config}
      onClose={onClose}
      onSubmitted={onSubmitted}
    />
  );
}

/** Renders the right-click menu items for a config, filtered by class. */
function ConfigContextMenuItems({
  project,
  config,
  onChoose,
  onChooseServeEval,
}: {
  project: ProjectInfo;
  config: ConfigInfo;
  onChoose: (action: ConfigAction) => void;
  onChooseServeEval: (
    action: "serve" | "eval" | "convert" | "finalize",
    output_dir: string,
    checkpointPath: string | null,
  ) => void;
}) {
  const metaQ = useQuery({
    queryKey: ["config-meta", project.project_dir, config.name],
    queryFn: () => api.configMeta(project.project_dir, config.name),
    staleTime: 5 * 60 * 1000,
  });
  // ProjectConfigs already fetches this when the project is expanded;
  // TanStack dedupes by key, so leaving ``enabled`` at its default lets
  // the menu transparently fall back to its own fetch when the cache
  // is cold (no extra network call when it isn't).
  const modelsQ = useQuery({
    queryKey: ["project-models", project.project_dir],
    queryFn: () => api.listProjectModels(project.project_dir),
    staleTime: 5 * 60 * 1000,
  });

  const cls = metaQ.data?.config_class ?? null;
  const isTraining = cls?.startsWith("type.training_script") ?? false;
  const isModel = cls?.startsWith("type.model") ?? false;
  const isDataset = cls?.startsWith("type.dataset") ?? false;
  const showRunCleanup = isTraining || cls === null;
  const showRun = showRunCleanup || isModel || isDataset;

  const modelEntry = modelsQ.data?.find((m: ModelEntry) => m.configs.includes(config.name));
  const hasCheckpoints = (modelEntry?.checkpoint_count ?? 0) > 0;
  const outputDir = modelEntry?.output_dir ?? "";

  return (
    <>
      <div className="context-menu-header muted">
        {config.name}
        {cls && <span className="context-menu-class">{cls}</span>}
      </div>
      {showRun && (
        <button onClick={() => onChoose("submit")}>▶ Run…</button>
      )}
      <button onClick={() => onChoose("overrides")}>🔧 Overrides…</button>
      {showRunCleanup && (
        <button onClick={() => onChoose("clean")}>🗑 Clean Output…</button>
      )}
      {showRunCleanup && (
        <button onClick={() => onChoose("tensorboard")}>
          📊 TensorBoard…
        </button>
      )}
      {hasCheckpoints && (
        <button onClick={() => onChooseServeEval("serve", outputDir, null)}>
          🔮 Serve Inference…
        </button>
      )}
      {hasCheckpoints && (
        <button onClick={() => onChooseServeEval("eval", outputDir, null)}>
          ⚖ Evaluate…
        </button>
      )}
      {hasCheckpoints && (
        <button onClick={() => onChooseServeEval("convert", outputDir, null)}>
          🔁 Convert Model…
        </button>
      )}
      {hasCheckpoints && (
        <button onClick={() => onChooseServeEval("finalize", outputDir, null)}>
          📦 Finalize Model…
        </button>
      )}
      <button
        className="context-menu-destructive"
        onClick={() => onChoose("delete")}
        title={config.path}
      >
        🗑 Delete Config…
      </button>
    </>
  );
}

type ContextRequest = (
  project: ProjectInfo,
  config: ConfigInfo,
  e: React.MouseEvent,
) => void;

type ProjectMenuRequest = (
  project: ProjectInfo,
  e: React.MouseEvent,
) => void;

type WorkspaceMenuRequest = (
  ws: WorkspaceCluster,
  e: React.MouseEvent,
) => void;

function WorkspaceForest({
  clusters,
  onSelect,
  onProjectOpen,
  selection,
  onContextRequest,
  onProjectMenu,
  onWorkspaceMenu,
  setSelection,
  onCheckpointMenu,
  onLeafMenu,
  onGroupMenu,
}: {
  clusters: WorkspaceCluster[];
  onSelect: (p: ProjectInfo, c: ConfigInfo) => void;
  onProjectOpen: (p: ProjectInfo, c: ConfigInfo) => void;
  selection: Selection;
  onContextRequest: ContextRequest;
  onProjectMenu: ProjectMenuRequest;
  onWorkspaceMenu: WorkspaceMenuRequest;
  setSelection: (sel: Selection) => void;
  onCheckpointMenu: (
    x: number,
    y: number,
    project: ProjectInfo,
    config: ConfigInfo,
    output_dir: string,
    checkpoint: CheckpointEntry,
  ) => void;
  onLeafMenu: (t: LeafMenuTarget) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
}) {
  const byParent = new Map<string, WorkspaceCluster[]>();
  const roots: WorkspaceCluster[] = [];
  for (const c of clusters) {
    if (c.parent_workspace_root) {
      const arr = byParent.get(c.parent_workspace_root) ?? [];
      arr.push(c);
      byParent.set(c.parent_workspace_root, arr);
    } else {
      roots.push(c);
    }
  }
  const sortKey = (c: WorkspaceCluster) =>
    (c.name || c.workspace_root || "").toLowerCase();
  roots.sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
  for (const arr of byParent.values()) {
    arr.sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
  }

  return (
    <>
      {roots.map((ws) => (
        <WorkspaceBlock
          key={ws.workspace_root || "unaffiliated"}
          ws={ws}
          childrenByParent={byParent}
          onSelect={onSelect}
          onProjectOpen={onProjectOpen}
          selection={selection}
          onContextRequest={onContextRequest}
          onProjectMenu={onProjectMenu}
          onWorkspaceMenu={onWorkspaceMenu}
          setSelection={setSelection}
          onCheckpointMenu={onCheckpointMenu}
          onLeafMenu={onLeafMenu}
          onGroupMenu={onGroupMenu}
          depth={0}
        />
      ))}
    </>
  );
}

function WorkspaceBlock({
  ws,
  childrenByParent,
  onSelect,
  onProjectOpen,
  selection,
  onContextRequest,
  onProjectMenu,
  onWorkspaceMenu,
  setSelection,
  onCheckpointMenu,
  onLeafMenu,
  onGroupMenu,
  depth,
}: {
  ws: WorkspaceCluster;
  childrenByParent: Map<string, WorkspaceCluster[]>;
  onSelect: (p: ProjectInfo, c: ConfigInfo) => void;
  onProjectOpen: (p: ProjectInfo, c: ConfigInfo) => void;
  selection: Selection;
  onContextRequest: ContextRequest;
  onProjectMenu: ProjectMenuRequest;
  onWorkspaceMenu: WorkspaceMenuRequest;
  setSelection: (sel: Selection) => void;
  onCheckpointMenu: (
    x: number,
    y: number,
    project: ProjectInfo,
    config: ConfigInfo,
    output_dir: string,
    checkpoint: CheckpointEntry,
  ) => void;
  onLeafMenu: (t: LeafMenuTarget) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
  depth: number;
}) {
  const label = ws.name || ws.workspace_root || "Unaffiliated";
  const children = childrenByParent.get(ws.workspace_root) ?? [];
  const totalProjects =
    ws.projects.length + countNestedProjects(ws, childrenByParent);
  return (
    <details className="workspace" style={{ marginLeft: depth ? 10 : 0 }}>
      <summary
        title={ws.workspace_root}
        onContextMenu={
          ws.workspace_root ? (e) => onWorkspaceMenu(ws, e) : undefined
        }
      >
        <span className="ws-label">{label}</span>
        <span className="badge">{totalProjects}</span>
        {ws.description && (
          <div className="muted summary-desc">{ws.description}</div>
        )}
      </summary>
      <ul>
        {ws.projects.map((p) => (
          <ProjectBlock
            key={p.project_dir}
            project={p}
            onSelect={onSelect}
            onProjectOpen={onProjectOpen}
            selection={selection}
            onContextRequest={onContextRequest}
            onProjectMenu={onProjectMenu}
            setSelection={setSelection}
            onCheckpointMenu={onCheckpointMenu}
            onLeafMenu={onLeafMenu}
            onGroupMenu={onGroupMenu}
          />
        ))}
      </ul>
      {children.map((child) => (
        <WorkspaceBlock
          key={child.workspace_root}
          ws={child}
          childrenByParent={childrenByParent}
          onSelect={onSelect}
          onProjectOpen={onProjectOpen}
          selection={selection}
          onContextRequest={onContextRequest}
          onProjectMenu={onProjectMenu}
          onWorkspaceMenu={onWorkspaceMenu}
          setSelection={setSelection}
          onCheckpointMenu={onCheckpointMenu}
          onLeafMenu={onLeafMenu}
          onGroupMenu={onGroupMenu}
          depth={depth + 1}
        />
      ))}
    </details>
  );
}

function countNestedProjects(
  ws: WorkspaceCluster,
  childrenByParent: Map<string, WorkspaceCluster[]>,
): number {
  const children = childrenByParent.get(ws.workspace_root) ?? [];
  let n = 0;
  for (const c of children) {
    n += c.projects.length + countNestedProjects(c, childrenByParent);
  }
  return n;
}

function ProjectBlock({
  project,
  onSelect,
  onProjectOpen,
  selection,
  onContextRequest,
  onProjectMenu,
  setSelection,
  onCheckpointMenu,
  onLeafMenu,
  onGroupMenu,
}: {
  project: ProjectInfo;
  onSelect: (p: ProjectInfo, c: ConfigInfo) => void;
  onProjectOpen: (p: ProjectInfo, c: ConfigInfo) => void;
  selection: Selection;
  onContextRequest: ContextRequest;
  onProjectMenu: ProjectMenuRequest;
  setSelection: (sel: Selection) => void;
  onCheckpointMenu: (
    x: number,
    y: number,
    project: ProjectInfo,
    config: ConfigInfo,
    output_dir: string,
    checkpoint: CheckpointEntry,
  ) => void;
  onLeafMenu: (t: LeafMenuTarget) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const defaultConfig =
    project.default_config != null
      ? project.configs.find((c) => c.name === project.default_config)
      : undefined;
  return (
    <li className="project">
      <details
        onToggle={(e) => {
          const isOpen = (e.target as HTMLDetailsElement).open;
          setExpanded(isOpen);
          if (isOpen && defaultConfig) {
            onProjectOpen(project, defaultConfig);
          }
        }}
      >
        <summary
          title={project.project_dir}
          onContextMenu={(e) => onProjectMenu(project, e)}
        >
          <span className="proj-name">
            {project.name || basename(project.project_dir)}
          </span>
          <span className="badge">{project.configs.length}</span>
          {project.parse_error && (
            <span className="err-badge" title={project.parse_error}>
              ERR
            </span>
          )}
          {project.description && (
            <div className="muted summary-desc">{project.description}</div>
          )}
        </summary>
        {expanded && (
          <ProjectConfigs
            project={project}
            onSelect={onSelect}
            selection={selection}
            onContextRequest={onContextRequest}
            setSelection={setSelection}
            onCheckpointMenu={onCheckpointMenu}
            onLeafMenu={onLeafMenu}
            onGroupMenu={onGroupMenu}
          />
        )}
      </details>
    </li>
  );
}

function ProjectConfigs({
  project,
  onSelect,
  selection,
  onContextRequest,
  setSelection,
  onCheckpointMenu,
  onLeafMenu,
  onGroupMenu,
}: {
  project: ProjectInfo;
  onSelect: (p: ProjectInfo, c: ConfigInfo) => void;
  selection: Selection;
  onContextRequest: ContextRequest;
  setSelection: (sel: Selection) => void;
  onCheckpointMenu: (
    x: number,
    y: number,
    project: ProjectInfo,
    config: ConfigInfo,
    output_dir: string,
    checkpoint: CheckpointEntry,
  ) => void;
  onLeafMenu: (t: LeafMenuTarget) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
}) {
  const [expandedConfigs, setExpandedConfigs] = useState<Set<string>>(new Set());

  const metaQs = useQueries({
    queries: project.configs.map((c) => ({
      queryKey: ["config-meta", project.project_dir, c.name],
      queryFn: () => api.configMeta(project.project_dir, c.name),
      staleTime: 5 * 60 * 1000,
    })),
  });

  // Fetch project models for per-config artifact counts.
  const modelsQ = useQuery({
    queryKey: ["project-models", project.project_dir],
    queryFn: () => api.listProjectModels(project.project_dir),
    staleTime: 5 * 60 * 1000,
  });

  // configName -> { output_dir, run_count, checkpoint_count, eval_count }
  const configCounts = new Map<
    string,
    { output_dir: string; run_count: number; checkpoint_count: number; eval_count: number }
  >();
  for (const entry of modelsQ.data ?? []) {
    for (const name of entry.configs) {
      configCounts.set(name, {
        output_dir: entry.output_dir,
        run_count: entry.run_count,
        checkpoint_count: entry.checkpoint_count,
        eval_count: entry.eval_count,
      });
    }
  }

  const toggleConfig = (name: string) => {
    setExpandedConfigs((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  return (
    <ul>
      {project.configs.map((c, i) => {
        const meta = metaQs[i].data;
        const loading = metaQs[i].isLoading;
        const display = meta?.name || null;
        const counts = configCounts.get(c.name);
        const total =
          (counts?.run_count ?? 0) +
          (counts?.checkpoint_count ?? 0) +
          (counts?.eval_count ?? 0);
        const isConfigExpanded = expandedConfigs.has(c.name);

        // Config stays highlighted when any of its sub-items (log / checkpoint /
        // eval) is selected — users can't see what they've chosen otherwise.
        const configSelected = selection?.config.path === c.path;
        return (
          <li
            key={c.path}
            className={
              "config " +
              (configSelected ? "selected " : "") +
              (c.is_default ? "default " : "")
            }
          >
            <div className="config-row">
              {total > 0 ? (
                <button
                  className="config-expand-tri"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleConfig(c.name);
                  }}
                  title={isConfigExpanded ? "Collapse" : "Expand artifacts"}
                >
                  {isConfigExpanded ? "▾" : "▸"}
                </button>
              ) : (
                <span className="config-expand-placeholder" />
              )}
              <button
                className="link"
                onClick={() => onSelect(project, c)}
                onContextMenu={(e) => onContextRequest(project, c, e)}
                title={c.path}
              >
                <span className="config-name">
                  {display ?? (loading ? c.name : c.name)}
                  {c.is_default && <span className="default-marker"> ★</span>}
                </span>
                {display && (
                  <span className="muted config-filename">{c.name}</span>
                )}
                {meta?.description && (
                  <div className="muted config-desc">{meta.description}</div>
                )}
                {meta?.parse_error && (
                  <div className="err config-desc" title={meta.parse_error}>
                    parse error
                  </div>
                )}
              </button>
            </div>
            {isConfigExpanded && counts && (
              <ConfigArtifacts
                project={project}
                config={c}
                counts={counts}
                selection={selection}
                setSelection={setSelection}
                onCheckpointMenu={onCheckpointMenu}
                onLeafMenu={onLeafMenu}
                onGroupMenu={onGroupMenu}
              />
            )}
          </li>
        );
      })}
    </ul>
  );
}

function ConfigArtifacts({
  project,
  config,
  counts,
  selection,
  setSelection,
  onCheckpointMenu,
  onLeafMenu,
  onGroupMenu,
}: {
  project: ProjectInfo;
  config: ConfigInfo;
  counts: {
    output_dir: string;
    run_count: number;
    checkpoint_count: number;
    eval_count: number;
  };
  selection: Selection;
  setSelection: (sel: Selection) => void;
  onCheckpointMenu: (
    x: number,
    y: number,
    project: ProjectInfo,
    config: ConfigInfo,
    output_dir: string,
    checkpoint: CheckpointEntry,
  ) => void;
  onLeafMenu: (t: LeafMenuTarget) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
}) {
  const [logsOpen, setLogsOpen] = useState(false);
  const [ckptsOpen, setCkptsOpen] = useState(false);
  const [evalsOpen, setEvalsOpen] = useState(false);

  return (
    <ul className="artifact-groups">
      {counts.run_count > 0 && (
        <LogsGroup
          project={project}
          config={config}
          output_dir={counts.output_dir}
          count={counts.run_count}
          open={logsOpen}
          onToggle={() => setLogsOpen((v) => !v)}
          selectedRunDir={
            selection?.kind === "log" && selection.config.path === config.path
              ? selection.run_dir
              : null
          }
          setSelection={setSelection}
          onLeafMenu={onLeafMenu}
          onGroupMenu={onGroupMenu}
        />
      )}
      {counts.checkpoint_count > 0 && (
        <CheckpointsGroup
          project={project}
          config={config}
          output_dir={counts.output_dir}
          count={counts.checkpoint_count}
          open={ckptsOpen}
          onToggle={() => setCkptsOpen((v) => !v)}
          selectedCheckpointDir={
            selection?.kind === "checkpoint" &&
            selection.config.path === config.path
              ? selection.checkpoint.checkpoint_dir
              : null
          }
          setSelection={setSelection}
          onCheckpointMenu={onCheckpointMenu}
          onGroupMenu={onGroupMenu}
        />
      )}
      {counts.eval_count > 0 && (
        <EvalsGroup
          project={project}
          config={config}
          output_dir={counts.output_dir}
          count={counts.eval_count}
          open={evalsOpen}
          onToggle={() => setEvalsOpen((v) => !v)}
          selectedEvalDir={
            selection?.kind === "eval" && selection.config.path === config.path
              ? selection.evaluation.eval_dir
              : null
          }
          setSelection={setSelection}
          onLeafMenu={onLeafMenu}
          onGroupMenu={onGroupMenu}
        />
      )}
    </ul>
  );
}

function LogsGroup({
  project,
  config,
  output_dir,
  count,
  open,
  onToggle,
  selectedRunDir,
  setSelection,
  onLeafMenu,
  onGroupMenu,
}: {
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  count: number;
  open: boolean;
  onToggle: () => void;
  selectedRunDir: string | null;
  setSelection: (sel: Selection) => void;
  onLeafMenu: (t: LeafMenuTarget) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
}) {
  const runsQ = useQuery({
    queryKey: ["model-runs", output_dir],
    queryFn: () => api.listModelRuns(output_dir),
    enabled: open,
    staleTime: 60_000,
  });

  return (
    <li className="artifact-group">
      <button
        className="artifact-group-header link"
        onClick={onToggle}
        onContextMenu={(e) => {
          e.preventDefault();
          onGroupMenu({
            x: e.clientX,
            y: e.clientY,
            kind: "logs",
            project,
            config,
            output_dir,
            count,
          });
        }}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <span className="muted">Logs ({count})</span>
      </button>
      {open && (
        <ul className="artifact-leaves">
          {runsQ.isLoading && (
            <li className="artifact-leaf muted">Loading…</li>
          )}
          {runsQ.error && (
            <li className="artifact-leaf err">{String(runsQ.error)}</li>
          )}
          {(runsQ.data ?? []).map((r: RunEntry) => (
            <li
              key={r.run_dir}
              className={
                "artifact-leaf" +
                (r.run_dir === selectedRunDir ? " selected" : "")
              }
              onClick={() =>
                setSelection({
                  kind: "log",
                  project,
                  config,
                  run_dir: r.run_dir,
                  run_id: r.run_id,
                })
              }
              onContextMenu={(e) => {
                e.preventDefault();
                onLeafMenu({
                  x: e.clientX,
                  y: e.clientY,
                  kind: "log",
                  project,
                  config,
                  output_dir,
                  item_dir: r.run_dir,
                  label: r.run_id,
                });
              }}
              title={r.run_dir}
            >
              <span className="artifact-leaf-label">{r.run_id}</span>
              {r.started_at > 0 && (
                <span className="muted artifact-leaf-meta">
                  {new Date(r.started_at * 1000).toLocaleDateString()}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function CheckpointsGroup({
  project,
  config,
  output_dir,
  count,
  open,
  onToggle,
  selectedCheckpointDir,
  setSelection,
  onCheckpointMenu,
  onGroupMenu,
}: {
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  count: number;
  open: boolean;
  onToggle: () => void;
  selectedCheckpointDir: string | null;
  setSelection: (sel: Selection) => void;
  onCheckpointMenu: (
    x: number,
    y: number,
    project: ProjectInfo,
    config: ConfigInfo,
    output_dir: string,
    checkpoint: CheckpointEntry,
  ) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
}) {
  const ckptsQ = useQuery({
    queryKey: ["model-checkpoints", output_dir],
    queryFn: () => api.listModelCheckpoints(output_dir),
    enabled: open,
    staleTime: 60_000,
  });

  return (
    <li className="artifact-group">
      <button
        className="artifact-group-header link"
        onClick={onToggle}
        onContextMenu={(e) => {
          e.preventDefault();
          onGroupMenu({
            x: e.clientX,
            y: e.clientY,
            kind: "checkpoints",
            project,
            config,
            output_dir,
            count,
          });
        }}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <span className="muted">Checkpoints ({count})</span>
      </button>
      {open && (
        <ul className="artifact-leaves">
          {ckptsQ.isLoading && (
            <li className="artifact-leaf muted">Loading…</li>
          )}
          {ckptsQ.error && (
            <li className="artifact-leaf err">{String(ckptsQ.error)}</li>
          )}
          {(ckptsQ.data ?? []).map((ckpt: CheckpointEntry) => (
            <li
              key={ckpt.checkpoint_dir}
              className={
                "artifact-leaf" +
                (ckpt.checkpoint_dir === selectedCheckpointDir
                  ? " selected"
                  : "")
              }
              onClick={() =>
                setSelection({
                  kind: "checkpoint",
                  project,
                  config,
                  output_dir,
                  checkpoint: ckpt,
                })
              }
              onContextMenu={(e) => {
                e.preventDefault();
                onCheckpointMenu(
                  e.clientX,
                  e.clientY,
                  project,
                  config,
                  output_dir,
                  ckpt,
                );
              }}
              title={ckpt.checkpoint_dir}
            >
              <span className="artifact-leaf-label">step {ckpt.step}</span>
              <span className="muted artifact-leaf-meta">
                {formatBytes(ckpt.size_bytes)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function EvalsGroup({
  project,
  config,
  output_dir,
  count,
  open,
  onToggle,
  selectedEvalDir,
  setSelection,
  onLeafMenu,
  onGroupMenu,
}: {
  project: ProjectInfo;
  config: ConfigInfo;
  output_dir: string;
  count: number;
  open: boolean;
  onToggle: () => void;
  selectedEvalDir: string | null;
  setSelection: (sel: Selection) => void;
  onLeafMenu: (t: LeafMenuTarget) => void;
  onGroupMenu: (t: GroupMenuTarget) => void;
}) {
  const evalsQ = useQuery({
    queryKey: ["model-evaluations", output_dir],
    queryFn: () => api.listModelEvaluations(output_dir),
    enabled: open,
    staleTime: 60_000,
  });

  return (
    <li className="artifact-group">
      <button
        className="artifact-group-header link"
        onClick={onToggle}
        onContextMenu={(e) => {
          e.preventDefault();
          onGroupMenu({
            x: e.clientX,
            y: e.clientY,
            kind: "evals",
            project,
            config,
            output_dir,
            count,
          });
        }}
      >
        <span className="tri">{open ? "▾" : "▸"}</span>
        <span className="muted">Evaluations ({count})</span>
      </button>
      {open && (
        <ul className="artifact-leaves">
          {evalsQ.isLoading && (
            <li className="artifact-leaf muted">Loading…</li>
          )}
          {evalsQ.error && (
            <li className="artifact-leaf err">{String(evalsQ.error)}</li>
          )}
          {(evalsQ.data ?? []).map((ev: EvalEntry) => (
            <li
              key={ev.eval_dir}
              className={
                "artifact-leaf" +
                (ev.eval_dir === selectedEvalDir ? " selected" : "")
              }
              onClick={() =>
                setSelection({
                  kind: "eval",
                  project,
                  config,
                  output_dir,
                  evaluation: ev,
                })
              }
              onContextMenu={(e) => {
                e.preventDefault();
                onLeafMenu({
                  x: e.clientX,
                  y: e.clientY,
                  kind: "eval",
                  project,
                  config,
                  output_dir,
                  item_dir: ev.eval_dir,
                  label: ev.result?.config_name ?? ev.eval_id,
                });
              }}
              title={ev.eval_dir}
            >
              <span className="artifact-leaf-label">
                {ev.result?.config_name ?? ev.eval_id}
              </span>
              {ev.result?.eval_loss != null && (
                <span className="muted artifact-leaf-meta">
                  {ev.result.eval_loss.toFixed(3)}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function basename(p: string): string {
  if (!p) return "";
  const parts = p.split("/").filter(Boolean);
  return parts[parts.length - 1] || p;
}
