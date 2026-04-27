import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { instance } from "@viz-js/viz";
import Editor, { OnMount } from "@monaco-editor/react";

import { api, ConfigInfo, ProjectInfo, TemplateGroup } from "../api";
import { FORGATHER_LANGUAGE_ID } from "../forgather-syntax";
import { ContextMenu } from "./ContextMenu";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
  onMount: OnMount;
  onEditTemplate: (path: string) => void;
  /** When the user clicks a tlist row whose path matches one of the
   *  project's configs, promote that config to the active selection so
   *  the rest of the UI (Run button, trefs graph, dynamic-args, etc.)
   *  reflects the click. trefs mode doesn't need this — its nodes are
   *  always templates referenced *by* the current config, never sibling
   *  configs the user might want to switch to. */
  onSelectConfig: (project: ProjectInfo, config: ConfigInfo) => void;
}

type Mode = "trefs" | "tlist";

interface MenuPos {
  x: number;
  y: number;
  path: string;
}

export function TemplatesView({
  project,
  config,
  onMount,
  onEditTemplate,
  onSelectConfig,
}: Props) {
  const [mode, setMode] = useState<Mode>("trefs");
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [menu, setMenu] = useState<MenuPos | null>(null);

  // Auto-show the active config's own template in the right pane every
  // time the config changes — including the initial mount. This makes
  // iterating across configs much quicker: pick a config, the preview
  // already shows it; tweak via tlist promotion or the project tree,
  // the preview follows. Subsequent clicks on a non-config template
  // (e.g. a parent in trefs) override the preview without re-firing
  // this effect, so manual deep-dives aren't disturbed.
  useEffect(() => {
    setSelectedPath(config.path);
  }, [config.path]);

  const sourceQ = useQuery({
    queryKey: ["template-source", selectedPath],
    queryFn: () => api.templateSource(selectedPath!),
    enabled: !!selectedPath,
  });

  const onContextRequest = (path: string, x: number, y: number) => {
    setSelectedPath(path);
    setMenu({ path, x, y });
  };

  // tlist-only: promote a clicked path to the active config when it
  // matches one of the project's configs. trefs callers can ignore
  // this — they go through onSelect directly.
  const promoteIfConfig = (path: string) => {
    const match = project.configs.find((c) => c.path === path);
    if (match && match.path !== config.path) {
      onSelectConfig(project, match);
    }
  };

  return (
    <div className="templates-view">
      <div className="trefs-split">
        <div className="trefs-left">
          <div className="templates-mode-bar">
            <button
              className={mode === "trefs" ? "active" : ""}
              onClick={() => setMode("trefs")}
              title="Show templates referenced by this configuration"
            >
              trefs
            </button>
            <button
              className={mode === "tlist" ? "active" : ""}
              onClick={() => setMode("tlist")}
              title="Show every template on the project's search path"
            >
              tlist
            </button>
          </div>
          <div className="trefs-graph">
            {mode === "trefs" ? (
              <TrefsGraphPane
                project={project}
                config={config}
                onSelect={setSelectedPath}
                onContextRequest={onContextRequest}
              />
            ) : (
              <TlistPane
                project={project}
                selectedPath={selectedPath}
                onSelect={(path) => {
                  setSelectedPath(path);
                  promoteIfConfig(path);
                }}
                onContextRequest={(path, x, y) => {
                  onContextRequest(path, x, y);
                  promoteIfConfig(path);
                }}
              />
            )}
          </div>
        </div>
        <div className="trefs-source">
          {selectedPath ? (
            <>
              <div className="template-label muted">
                <code>{selectedPath}</code>
                <button
                  className="template-edit-btn"
                  onClick={() => onEditTemplate(selectedPath)}
                  title="Open this template for editing in the Files panel"
                >
                  ✎ Edit
                </button>
              </div>
              <div className="template-editor">
                <Editor
                  height="100%"
                  language={FORGATHER_LANGUAGE_ID}
                  value={sourceQ.data ?? ""}
                  theme="vs-dark"
                  options={{
                    readOnly: true,
                    minimap: { enabled: false },
                    fontSize: 12,
                    scrollBeyondLastLine: false,
                  }}
                  onMount={onMount}
                />
              </div>
            </>
          ) : (
            <div className="pane-state muted">
              {mode === "trefs"
                ? "Click a template node to inspect it."
                : "Click a template to inspect it."}
            </div>
          )}
        </div>
      </div>
      {menu && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)}>
          <button
            className="context-menu-item"
            onClick={() => {
              onEditTemplate(menu.path);
              setMenu(null);
            }}
          >
            ✎ Open in Editor
          </button>
        </ContextMenu>
      )}
    </div>
  );
}

interface TrefsPaneProps {
  project: ProjectInfo;
  config: ConfigInfo;
  onSelect: (path: string) => void;
  onContextRequest: (path: string, x: number, y: number) => void;
}

function TrefsGraphPane({
  project,
  config,
  onSelect,
  onContextRequest,
}: TrefsPaneProps) {
  const dotQ = useQuery({
    queryKey: ["trefs-dot", project.project_dir, config.name],
    queryFn: () => api.configTrefsDot(project.project_dir, config.name),
  });
  const jsonQ = useQuery({
    queryKey: ["trefs-json", project.project_dir, config.name],
    queryFn: () => api.configTrefsJson(project.project_dir, config.name),
  });
  const svgHostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!dotQ.data || !svgHostRef.current || !jsonQ.data) return;
    let cancelled = false;
    instance().then((viz) => {
      if (cancelled || !svgHostRef.current) return;
      const svg = viz.renderSVGElement(dotQ.data!);

      // Map cleaned-DOT-id back to {name,path}. DOT escapes '.','/','-' to '_'.
      const nameByClean = new Map<string, { name: string; path: string }>();
      for (const n of jsonQ.data!.nodes) {
        const clean = n.name.replace(/[.\/-]/g, "_");
        nameByClean.set(clean, n);
      }
      svg.querySelectorAll("g.node").forEach((el) => {
        const title = el.querySelector("title")?.textContent || "";
        const node = nameByClean.get(title);
        if (!node) return;
        (el as SVGGElement).style.cursor = "pointer";
        el.addEventListener("click", () => onSelect(node.path));
        el.addEventListener("contextmenu", (e) => {
          const me = e as MouseEvent;
          me.preventDefault();
          onContextRequest(node.path, me.clientX, me.clientY);
        });
      });

      svgHostRef.current.innerHTML = "";
      svgHostRef.current.appendChild(svg);
    });
    return () => {
      cancelled = true;
    };
  }, [dotQ.data, jsonQ.data, onSelect, onContextRequest]);

  return (
    <>
      {dotQ.isLoading && <div className="pane-state">Loading graph...</div>}
      {dotQ.error && (
        <div className="pane-state err">
          <pre>{String(dotQ.error)}</pre>
        </div>
      )}
      <div ref={svgHostRef} className="svg-host" />
    </>
  );
}

interface TlistProps {
  project: ProjectInfo;
  selectedPath: string | null;
  onSelect: (path: string) => void;
  onContextRequest: (path: string, x: number, y: number) => void;
}

function TlistPane({
  project,
  selectedPath,
  onSelect,
  onContextRequest,
}: TlistProps) {
  const tlistQ = useQuery({
    queryKey: ["project-templates", project.project_dir],
    queryFn: () => api.listProjectTemplates(project.project_dir),
    staleTime: 30_000,
  });

  if (tlistQ.isLoading) {
    return <div className="pane-state">Loading templates...</div>;
  }
  if (tlistQ.error) {
    return (
      <div className="pane-state err">
        <pre>{String(tlistQ.error)}</pre>
      </div>
    );
  }
  const groups = tlistQ.data ?? [];
  if (groups.length === 0) {
    return <div className="pane-state muted">No templates found.</div>;
  }
  return (
    <div className="tlist">
      {groups.map((g: TemplateGroup) => (
        <div key={g.search_path} className="tlist-group">
          <div className="tlist-group-header">
            <span className="tlist-group-name">{g.category}</span>
            <span className="tlist-group-count">{g.templates.length}</span>
            <span className="tlist-group-path muted" title={g.search_path}>
              {g.search_path}
            </span>
          </div>
          <ul className="tlist-items">
            {g.templates.map((t) => (
              <li
                key={t.path}
                className={
                  "tlist-item" + (t.path === selectedPath ? " active" : "")
                }
                title={t.path}
                onClick={() => onSelect(t.path)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  onContextRequest(t.path, e.clientX, e.clientY);
                }}
              >
                {t.rel_path}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
