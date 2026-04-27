import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Editor, { OnMount } from "@monaco-editor/react";

import { api, ConfigInfo, ModelEntry, ProjectInfo } from "../api";
import { FORGATHER_LANGUAGE_ID, registerForgatherLanguage } from "../forgather-syntax";
import { CleanOutputModal } from "./CleanOutputModal";
import { EvalModal } from "./EvalModal";
import { InfoPane } from "./InfoPane";
import { OverridesModal } from "./OverridesModal";
import { SubmitModal } from "./SubmitModal";
import { ConfigTensorBoardModal } from "./TensorBoardModal";
import { TemplatesView } from "./TemplatesView";
import { ConfigTab } from "../App";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  tab: ConfigTab;
  onTabChange: (tab: ConfigTab) => void;
  onEditTemplate: (path: string) => void;
  onSelectConfig: (project: ProjectInfo, config: ConfigInfo) => void;
}

export function ConfigViewer({
  project,
  config,
  tab,
  onTabChange,
  onEditTemplate,
  onSelectConfig,
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
  const showRunCleanup = metaQ.data ? isTraining : true;

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
  const modelName = outputDir
    ? outputDir.split("/").filter(Boolean).pop() ?? outputDir
    : config.name;

  const onMount: OnMount = (_editor, monaco) => {
    registerForgatherLanguage(monaco);
  };

  return (
    <div className="viewer">
      <header className="viewer-header">
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
        {showRunCleanup && (
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
        {showRunCleanup && (
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
            ⚖ Evaluate…
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
            className={tab === "templates" ? "active" : ""}
            onClick={() => setTab("templates")}
          >
            templates
          </button>
        </nav>
      </header>

      <InfoPane project_dir={project.project_dir} enabled={tab === "info"} />
      {tab === "pp" && (
        <EditorPane
          value={ppQ.data ?? ""}
          loading={ppQ.isLoading}
          error={ppQ.error}
          onMount={onMount}
        />
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
      {submitting && (
        <SubmitModal
          project={project}
          config={config}
          onClose={() => setSubmitting(false)}
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
        />
      )}
      {evaluating && (
        <EvalModal
          modelOutputDir={outputDir}
          modelName={modelName}
          checkpointPath={null}
          projectDir={project.project_dir}
          onClose={() => setEvaluating(false)}
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
  if (error)
    return (
      <div className="pane-state err">
        <pre>{String(error)}</pre>
      </div>
    );
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
