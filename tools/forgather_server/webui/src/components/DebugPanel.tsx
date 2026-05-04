import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import Editor, { OnMount } from "@monaco-editor/react";

import { api, ConfigInfo, ProjectInfo, asConfigError } from "../api";
import { FORGATHER_LANGUAGE_ID } from "../forgather-syntax";
import { ConfigErrorView } from "./ConfigErrorView";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onMount: OnMount;
  /** Open the selected raw template in the Files panel for editing.
   *  Only enabled when the template was loaded from a real file (i.e.
   *  the trace item carries a non-empty ``path``). */
  onEditTemplate: (path: string) => void;
}

/** Three-pane debug view: template list (left) + raw template (middle) +
 *  preprocessed template (right). Drives one HTTP call to /api/config/debug
 *  on activation (lazy via React Query) and renders everything from the
 *  cached payload — no per-template fetches needed since the trace already
 *  carries each raw + preprocessed pair. */
export function DebugPanel({
  project,
  config,
  onMount,
  onEditTemplate,
}: Props) {
  const debugQ = useQuery({
    queryKey: ["debug", project.project_dir, config.name],
    queryFn: () => api.configDebug(project.project_dir, config.name),
  });

  const items = debugQ.data ?? [];
  const [selected, setSelected] = useState<string | null>(null);

  // Default selection: first item whose name matches the config.name (the
  // root template the user picked); fall back to the first trace item.
  useEffect(() => {
    if (items.length === 0) {
      setSelected(null);
      return;
    }
    if (selected && items.some((it) => it.name === selected)) return;
    const root = items.find((it) => it.name === config.name) ?? items[0];
    setSelected(root.name);
  }, [items, config.name, selected]);

  const current = useMemo(
    () => items.find((it) => it.name === selected) ?? null,
    [items, selected],
  );

  if (debugQ.isLoading) {
    return <div className="pane-state">Loading…</div>;
  }
  if (debugQ.error) {
    const cfgErr = asConfigError(debugQ.error);
    if (cfgErr) {
      return (
        <div className="pane-state err">
          <ConfigErrorView err={cfgErr} />
        </div>
      );
    }
    return (
      <div className="pane-state err">
        <pre>{String(debugQ.error)}</pre>
      </div>
    );
  }
  if (items.length === 0) {
    return <div className="pane-state muted">No templates loaded.</div>;
  }

  return (
    <div className="debug-view">
      <div className="debug-list">
        <div className="debug-list-header">
          {items.length} template{items.length === 1 ? "" : "s"}
        </div>
        <ul className="debug-items">
          {items.map((it, i) => (
            <li
              key={`${it.name}-${i}`}
              className={
                it.name === selected ? "debug-item active" : "debug-item"
              }
              onClick={() => setSelected(it.name)}
              title={it.path || it.name}
            >
              <div className="debug-item-name">{it.name}</div>
              {it.path && (
                <div className="debug-item-path muted">{it.path}</div>
              )}
            </li>
          ))}
        </ul>
      </div>
      <div className="debug-source">
        <div className="template-label muted">
          <code>raw — {current?.name ?? ""}</code>
          {current?.path && (
            <button
              className="template-edit-btn"
              onClick={() => onEditTemplate(current.path)}
              title="Open this template for editing in the Files panel"
            >
              ✎ Edit
            </button>
          )}
        </div>
        <div className="template-editor">
          <Editor
            height="100%"
            language={FORGATHER_LANGUAGE_ID}
            value={current?.raw ?? ""}
            theme="vs-dark"
            options={{
              readOnly: true,
              minimap: { enabled: false },
              fontSize: 12,
              scrollBeyondLastLine: false,
              lineNumbers: "on",
            }}
            onMount={onMount}
          />
        </div>
      </div>
      <div className="debug-source">
        <div className="template-label muted">
          <code>preprocessed — {current?.name ?? ""}</code>
        </div>
        <div className="template-editor">
          <Editor
            height="100%"
            language={FORGATHER_LANGUAGE_ID}
            value={current?.preprocessed ?? ""}
            theme="vs-dark"
            options={{
              readOnly: true,
              minimap: { enabled: false },
              fontSize: 12,
              scrollBeyondLastLine: false,
              lineNumbers: "on",
            }}
            onMount={onMount}
          />
        </div>
      </div>
    </div>
  );
}
