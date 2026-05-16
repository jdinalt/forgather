import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Editor, { OnMount } from "@monaco-editor/react";

import { api, ConfigInfo, ModelEntry, ProjectInfo } from "../api";
import { FORGATHER_LANGUAGE_ID, registerForgatherLanguage } from "../forgather-syntax";
import { CleanOutputModal } from "./CleanOutputModal";
import { DatasetSubmitModal } from "./DatasetSubmitModal";
import { EvalModal } from "./EvalModal";
import { InfoPane } from "./InfoPane";
import { ModelSubmitModal } from "./ModelSubmitModal";
import { OverridesModal } from "./OverridesModal";
import { SubmitModal } from "./SubmitModal";
import { ConfigTensorBoardModal } from "./TensorBoardModal";
import { TemplatesView } from "./TemplatesView";
import { DebugPanel } from "./DebugPanel";
import { CodePanel } from "./CodePanel";
import { GraphPanel } from "./GraphPanel";
import { ConfigErrorView } from "./ConfigErrorView";
import { ConfigTab } from "../App";
import { asConfigError } from "../api";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  tab: ConfigTab;
  onTabChange: (tab: ConfigTab) => void;
  onEditTemplate: (path: string) => void;
  onSelectConfig: (project: ProjectInfo, config: ConfigInfo) => void;
  /** Bubble the submitted job's queue id to App so it can decide whether
   *  to switch to the Jobs view + auto-open the TTY. */
  onJobSubmitted?: (queueId: string) => void;
  /** Hand off a markdown / ipynb link click in the README to the Docs view. */
  onOpenDoc?: (path: string) => void;
  /** Hand off a yaml / py link click in the README to the editor. */
  onEditFile?: (path: string) => void;
}

export function ConfigViewer({
  project,
  config,
  tab,
  onTabChange,
  onEditTemplate,
  onSelectConfig,
  onJobSubmitted,
  onOpenDoc,
  onEditFile,
}: Props) {
  const [submitting, setSubmitting] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [overriding, setOverriding] = useState(false);
  const [tensorboarding, setTensorboarding] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const setTab = onTabChange;

  const ppQ = useQuery({
    queryKey: ["pp", project.project_dir, config.name],
    queryFn: () => api.configPp(project.project_dir, config.name),
    enabled: tab === "pp",
  });

  const metaQ = useQuery({
    queryKey: ["config-meta", project.project_dir, config.name],
    queryFn: () => api.configMeta(project.project_dir, config.name),
    staleTime: 5 * 60 * 1000,
  });
  const cls = metaQ.data?.config_class ?? null;
  const isTraining = cls?.startsWith("type.training_script") ?? false;
  const isModel = cls?.startsWith("type.model") ?? false;
  const isDataset = cls?.startsWith("type.dataset") ?? false;
  const showRunCleanup = metaQ.data ? isTraining : true;
  const showRun = metaQ.data ? isTraining || isModel || isDataset : true;

  // Read project models from cache to decide whether to show Serve/Eval buttons.
  const modelsQ = useQuery({
    queryKey: ["project-models", project.project_dir],
    queryFn: () => api.listProjectModels(project.project_dir),
    staleTime: 5 * 60 * 1000,
    enabled: false, // reads from cache populated by ProjectTree
  });
  const modelEntry = modelsQ.data?.find((m: ModelEntry) =>
    m.configs.includes(config.name),
  );
  const hasCheckpoints = (modelEntry?.checkpoint_count ?? 0) > 0;
  const outputDir = modelEntry?.output_dir ?? "";
  // Resolves the config's *actual* output_dir from its rendered
  // meta and stats it. ``output_dir`` is configurable per config
  // (can live anywhere on disk), so existence has to be checked
  // against the resolved path, not inferred from output_models/.
  // Used to gate the Clean Output button so it only shows when
  // there's something to clean.
  const outputDirQ = useQuery({
    queryKey: ["config-output-dir", project.project_dir, config.name],
    queryFn: () => api.configOutputDir(project.project_dir, config.name),
    staleTime: 60 * 1000,
  });
  const outputDirExists = !!outputDirQ.data?.output_dir_exists;
  const modelName = outputDir
    ? outputDir.split("/").filter(Boolean).pop() ?? outputDir
    : config.name;

  const onMount: OnMount = (_editor, monaco) => {
    registerForgatherLanguage(monaco);
  };

  return (
    <div className="viewer">
      <header className="viewer-header config-viewer-header">
        <div className="viewer-title">
          {metaQ.data?.name ? (
            <>
              <strong>{metaQ.data.name}</strong>
              <code className="muted viewer-yaml-name">{config.name}</code>
            </>
          ) : (
            <strong>{config.name}</strong>
          )}
          {cls && (
            <span className="viewer-class" title={cls}>
              {cls}
            </span>
          )}
          <span className="muted viewer-project">
            — {project.name || project.project_dir}
          </span>
        </div>
        {showRun && (
          <button
            className="run-btn"
            onClick={() => setSubmitting(true)}
            title="Submit this config to the queue"
          >
            ▶ Run
          </button>
        )}
        <button
          className="clean-btn"
          onClick={() => setOverriding(true)}
          title="Set persistent overrides for dynamic args"
        >
          🔧 Overrides…
        </button>
        {/* Hidden when the config's resolved output_dir doesn't
            exist on disk — there's nothing to clean. */}
        {showRunCleanup && outputDirExists && (
          <button
            className="clean-btn"
            onClick={() => setCleaning(true)}
            title="Delete this config's output directory"
          >
            🗑 Clean Output…
          </button>
        )}
        {showRunCleanup && (
          <button
            className="clean-btn"
            onClick={() => setTensorboarding(true)}
            title="Open TensorBoard against this config's output directory"
          >
            📊 TensorBoard…
          </button>
        )}
        {hasCheckpoints && (
          <button
            className="clean-btn"
            onClick={() => setEvaluating(true)}
            title="Evaluate this model (blank = latest checkpoint)"
          >
            📐 Evaluate…
          </button>
        )}
        <nav className="tabs">
          <button
            className={tab === "info" ? "active" : ""}
            onClick={() => setTab("info")}
          >
            info
          </button>
          <button
            className={tab === "pp" ? "active" : ""}
            onClick={() => setTab("pp")}
          >
            pp
          </button>
          <button
            className={tab === "code" ? "active" : ""}
            onClick={() => setTab("code")}
            title="Render the config (or a single target) as Python source"
          >
            code
          </button>
          <button
            className={tab === "graph" ? "active" : ""}
            onClick={() => setTab("graph")}
            title="Config node dependency graph"
          >
            graph
          </button>
          <button
            className={tab === "templates" ? "active" : ""}
            onClick={() => setTab("templates")}
          >
            templates
          </button>
          <button
            className={tab === "debug" ? "active" : ""}
            onClick={() => setTab("debug")}
            title="Per-template preprocess trace"
          >
            debug
          </button>
        </nav>
      </header>

      <InfoPane
        project_dir={project.project_dir}
        enabled={tab === "info"}
        onOpenDoc={onOpenDoc}
        onEditFile={onEditFile}
      />
      {tab === "pp" && (
        <EditorPane
          value={ppQ.data ?? ""}
          loading={ppQ.isLoading}
          error={ppQ.error}
          onMount={onMount}
        />
      )}
      {tab === "code" && (
        <CodePanel project={project} config={config} onMount={onMount} />
      )}
      {tab === "graph" && (
        <GraphPanel project={project} config={config} />
      )}
      {tab === "templates" && (
        <TemplatesView
          project={project}
          config={config}
          onMount={onMount}
          onEditTemplate={onEditTemplate}
          onSelectConfig={onSelectConfig}
        />
      )}
      {tab === "debug" && (
        <DebugPanel
          project={project}
          config={config}
          onMount={onMount}
          onEditTemplate={onEditTemplate}
        />
      )}
      {submitting && isModel && (
        <ModelSubmitModal
          project={project}
          config={config}
          onClose={() => setSubmitting(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {submitting && isDataset && (
        <DatasetSubmitModal
          project={project}
          config={config}
          onClose={() => setSubmitting(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {submitting && !isModel && !isDataset && (
        <SubmitModal
          project={project}
          config={config}
          onClose={() => setSubmitting(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {cleaning && (
        <CleanOutputModal
          project={project}
          config={config}
          onClose={() => setCleaning(false)}
        />
      )}
      {overriding && (
        <OverridesModal
          project={project}
          config={config}
          onClose={() => setOverriding(false)}
        />
      )}
      {tensorboarding && (
        <ConfigTensorBoardModal
          project={project}
          config={config}
          onClose={() => setTensorboarding(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
      {evaluating && (
        <EvalModal
          modelOutputDir={outputDir}
          modelName={modelName}
          checkpointPath={null}
          projectDir={project.project_dir}
          onClose={() => setEvaluating(false)}
          onSubmitted={onJobSubmitted}
        />
      )}
    </div>
  );
}

function EditorPane({
  value,
  loading,
  error,
  onMount,
}: {
  value: string;
  loading: boolean;
  error: unknown;
  onMount: OnMount;
}) {
  if (loading) return <div className="pane-state">Loading...</div>;
  if (error) {
    const cfgErr = asConfigError(error);
    if (cfgErr) {
      return (
        <div className="pane-state err">
          <ConfigErrorView err={cfgErr} />
        </div>
      );
    }
    return (
      <div className="pane-state err">
        <pre>{String(error)}</pre>
      </div>
    );
  }
  return (
    <Editor
      height="100%"
      language={FORGATHER_LANGUAGE_ID}
      value={value}
      theme="vs-dark"
      options={{
        readOnly: true,
        minimap: { enabled: false },
        fontSize: 13,
        scrollBeyondLastLine: false,
      }}
      onMount={onMount}
    />
  );
}
