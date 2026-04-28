import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import Editor, { OnMount } from "@monaco-editor/react";

import { api, ConfigInfo, ProjectInfo, asConfigError } from "../api";
import { ConfigErrorView } from "./ConfigErrorView";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onMount: OnMount;
}

/** Pseudo-target meaning "render the entire config in one document".
 *  The backend treats an empty `target=` query string as "render all", so
 *  the value here is the empty string and we just label it "(all)". */
const ALL_TARGETS = "";

/** Two-pane "code" view: target list (left) + Python source (right).
 *
 *  Targets list is fetched once per (project, config) via
 *  /api/config/code-targets and the Python source is re-fetched on each
 *  selection change. The default selection is "main" (matching the CLI's
 *  ``forgather code`` default); a synthetic "(all)" entry at the top
 *  renders the entire config. */
export function CodePanel({ project, config, onMount }: Props) {
  const targetsQ = useQuery({
    queryKey: ["code-targets", project.project_dir, config.name],
    queryFn: () => api.configCodeTargets(project.project_dir, config.name),
  });

  const [target, setTarget] = useState<string>("main");

  // Reset selection when switching configs so we don't ask for a target
  // that doesn't exist on the new config.
  useEffect(() => {
    setTarget("main");
  }, [project.project_dir, config.name]);

  const targets = targetsQ.data ?? [];
  // If "main" isn't in the config (rare), fall back to the first target.
  const effectiveTarget =
    target === ALL_TARGETS || targets.includes(target)
      ? target
      : targets[0] ?? "main";

  const codeQ = useQuery({
    queryKey: ["code", project.project_dir, config.name, effectiveTarget],
    queryFn: () =>
      api.configCode(project.project_dir, config.name, effectiveTarget),
    // Don't fire until we know the targets list (avoids a redundant request
    // that would race with the auto-correct in `effectiveTarget`).
    enabled: targetsQ.isSuccess || targetsQ.isError,
  });

  // The targets endpoint runs the same preprocess/load pipeline as code, so
  // a structured 400 lands here too — surface it once (in the Python pane)
  // rather than in both panes.
  const targetsErr = asConfigError(targetsQ.error);
  const codeErr = asConfigError(codeQ.error);
  const renderError = codeErr ?? targetsErr;

  return (
    <div className="code-view">
      <div className="code-targets">
        <div className="code-targets-header">
          {targets.length} target{targets.length === 1 ? "" : "s"}
        </div>
        <ul className="code-target-items">
          <li
            className={
              target === ALL_TARGETS
                ? "code-target-item active"
                : "code-target-item"
            }
            onClick={() => setTarget(ALL_TARGETS)}
            title="Render the entire config in one document"
          >
            <em>(all targets)</em>
          </li>
          {targets.map((t) => (
            <li
              key={t}
              className={
                t === target ? "code-target-item active" : "code-target-item"
              }
              onClick={() => setTarget(t)}
              title={t}
            >
              {t}
            </li>
          ))}
        </ul>
      </div>
      <div className="code-source">
        {renderError ? (
          <div className="pane-state err">
            <ConfigErrorView err={renderError} />
          </div>
        ) : codeQ.isLoading || targetsQ.isLoading ? (
          <div className="pane-state">Loading…</div>
        ) : codeQ.error ? (
          <div className="pane-state err">
            <pre>{String(codeQ.error)}</pre>
          </div>
        ) : (
          <Editor
            height="100%"
            language="python"
            value={codeQ.data ?? ""}
            theme="vs-dark"
            options={{
              readOnly: true,
              minimap: { enabled: false },
              fontSize: 13,
              scrollBeyondLastLine: false,
              lineNumbers: "on",
            }}
            onMount={onMount}
          />
        )}
      </div>
    </div>
  );
}
