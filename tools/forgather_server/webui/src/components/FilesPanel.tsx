import Editor, { OnMount } from "@monaco-editor/react";
import { useEffect, useState } from "react";

import { registerForgatherLanguage } from "../forgather-syntax";
import { languageFor } from "../file-languages";
import { EditorSplit, FilesApi } from "../files-state";
import { ContextMenu } from "./ContextMenu";

interface Props {
  api: FilesApi;
}

interface MenuPos {
  x: number;
  y: number;
  splitId: string;
  path: string;
}

const DRAG_MIME = "application/x-forgather-tab";

export function FilesPanel({ api }: Props) {
  const { state } = api;
  const [menu, setMenu] = useState<MenuPos | null>(null);

  // Ctrl/Cmd+S saves the active tab in the active split. Bound at window
  // level so the shortcut works regardless of where focus sits inside the
  // panel (Monaco swallows most keys, but this captures during the
  // capture-phase before Monaco gets it). Only fires while a Files-panel
  // editor exists, i.e. there is at least one open tab somewhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isSave = (e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S");
      if (!isSave) return;
      const active = state.splits.find((sp) => sp.id === state.activeSplitId);
      const path = active?.activePath ?? null;
      if (!path) return;
      e.preventDefault();
      e.stopPropagation();
      api.saveFile(path).catch(() => {
        // Surface failures via the buffer's error field — the editor pane
        // shows it. Nothing else to do here.
      });
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true } as any);
  }, [state.activeSplitId, state.splits, api]);

  if (state.splits.every((sp) => sp.tabPaths.length === 0)) {
    return (
      <div className="files-panel files-empty">
        <div className="pane-state muted">
          No files open. Use the <strong>Edit</strong> button on a template in
          the Projects → templates view to open it here.
        </div>
      </div>
    );
  }

  return (
    <div className="files-panel">
      <div className="files-splits">
        {state.splits.map((sp) => (
          <SplitPane
            key={sp.id}
            split={sp}
            isActive={sp.id === state.activeSplitId}
            api={api}
            openMenu={(x, y, splitId, path) => setMenu({ x, y, splitId, path })}
          />
        ))}
      </div>
      {menu && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)}>
          <button
            className="context-menu-item"
            onClick={() => {
              api.saveFile(menu.path).catch(() => {});
              setMenu(null);
            }}
          >
            Save
          </button>
          <button
            className="context-menu-item"
            onClick={() => {
              api.closeTab(menu.splitId, menu.path);
              setMenu(null);
            }}
          >
            Close
          </button>
          <button
            className="context-menu-item"
            onClick={() => {
              api.closeOthers(menu.splitId, menu.path);
              setMenu(null);
            }}
          >
            Close Others
          </button>
          <button
            className="context-menu-item"
            onClick={() => {
              api.closeAll(menu.splitId);
              setMenu(null);
            }}
          >
            Close All
          </button>
        </ContextMenu>
      )}
      <ConflictModal api={api} />
    </div>
  );
}

/** Surfaces save-time conflicts (file changed on disk after we
 *  opened it) as a blocking modal. Watches every buffer; opens for
 *  the first one that has ``conflict`` set. The user picks
 *  Overwrite / Reload / Cancel; each clears the conflict on the
 *  buffer (Cancel via clearConflict, the others as a side effect of
 *  the save / reload succeeding). */
function ConflictModal({ api }: { api: FilesApi }) {
  const conflicting = Object.values(api.state.buffers).find((b) => b.conflict);
  if (!conflicting || !conflicting.conflict) return null;

  const path = conflicting.path;
  const fmtTime = (s: number) => new Date(s * 1000).toLocaleString();

  return (
    <div className="modal-backdrop" onClick={() => api.clearConflict(path)}>
      <div
        className="modal conflict-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="File changed on disk"
      >
        <header className="modal-header">
          <h3>File changed on disk</h3>
          <button
            className="tiny"
            onClick={() => api.clearConflict(path)}
            aria-label="Close"
          >
            ×
          </button>
        </header>
        <div className="modal-body">
          <div className="submit-summary">
            <div>
              <span className="muted">file</span>
              <code title={path}>{path}</code>
            </div>
            <div>
              <span className="muted">opened at</span>
              <code>{fmtTime(conflicting.baselineMtime ?? 0)}</code>
            </div>
            <div>
              <span className="muted">disk mtime</span>
              <code>{fmtTime(conflicting.conflict.currentMtime)}</code>
            </div>
          </div>
          <div className="muted-hint">
            This file was modified on disk after you opened it for
            editing. Choose how to resolve:
            <ul>
              <li>
                <strong>Overwrite</strong>: save your version, replacing
                the on-disk content.
              </li>
              <li>
                <strong>Reload</strong>: discard your edits and load
                the current on-disk content.
              </li>
              <li>
                <strong>Cancel</strong>: keep your edits in the buffer;
                the file stays unsaved.
              </li>
            </ul>
          </div>
        </div>
        <footer className="modal-footer">
          <div />
          <div className="btn-row">
            <button
              className="secondary"
              onClick={() => api.clearConflict(path)}
            >
              Cancel
            </button>
            <button
              className="secondary"
              onClick={() => {
                api.reloadFile(path).catch(() => {});
              }}
            >
              Reload from disk
            </button>
            <button
              className="destructive"
              onClick={() => {
                api.forceSaveFile(path).catch(() => {});
              }}
            >
              Overwrite
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

interface SplitProps {
  split: EditorSplit;
  isActive: boolean;
  api: FilesApi;
  openMenu: (x: number, y: number, splitId: string, path: string) => void;
}

function SplitPane({ split, isActive, api, openMenu }: SplitProps) {
  const { state } = api;
  const activeBuf =
    split.activePath != null ? state.buffers[split.activePath] : null;
  const [dragOverEnd, setDragOverEnd] = useState(false);

  const onMount: OnMount = (_editor, monaco) => {
    registerForgatherLanguage(monaco);
  };

  const onTabBarDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOverEnd(false);
    const raw = e.dataTransfer.getData(DRAG_MIME);
    if (!raw) return;
    try {
      const { fromSplitId, path } = JSON.parse(raw) as {
        fromSplitId: string;
        path: string;
      };
      api.moveTab(fromSplitId, split.id, path, null);
    } catch {
      // ignore
    }
  };

  return (
    <div
      className={"files-split" + (isActive ? " active" : "")}
      onMouseDown={() => api.setActiveSplit(split.id)}
    >
      <div
        className="files-tabbar"
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes(DRAG_MIME)) {
            e.preventDefault();
            setDragOverEnd(true);
          }
        }}
        onDragLeave={() => setDragOverEnd(false)}
        onDrop={onTabBarDrop}
      >
        {split.tabPaths.map((path) => (
          <FileTab
            key={path}
            path={path}
            split={split}
            api={api}
            openMenu={openMenu}
          />
        ))}
        <div
          className={"files-tabbar-spacer" + (dragOverEnd ? " drop-target" : "")}
        />
        <button
          className="files-split-btn"
          title="Split editor vertically"
          onClick={(e) => {
            e.stopPropagation();
            api.splitVertical(split.id);
          }}
        >
          ⊟
        </button>
      </div>
      <div
        className="files-editor-host"
        onContextMenu={(e) => {
          if (!split.activePath) return;
          e.preventDefault();
          openMenu(e.clientX, e.clientY, split.id, split.activePath);
        }}
      >
        {!activeBuf && (
          <div className="pane-state muted">No file selected.</div>
        )}
        {activeBuf?.loading && (
          <div className="pane-state muted">Loading {activeBuf.path}…</div>
        )}
        {activeBuf && activeBuf.error && !activeBuf.loading && (
          <div className="pane-state err">
            <pre>{activeBuf.error}</pre>
          </div>
        )}
        {activeBuf && !activeBuf.loading && (
          <Editor
            height="100%"
            language={languageFor(activeBuf.path)}
            path={activeBuf.path}
            theme="vs-dark"
            value={activeBuf.content}
            options={{
              minimap: { enabled: false },
              fontSize: 13,
              scrollBeyondLastLine: false,
            }}
            onChange={(v) => api.setContent(activeBuf.path, v ?? "")}
            onMount={onMount}
          />
        )}
      </div>
    </div>
  );
}

interface TabProps {
  path: string;
  split: EditorSplit;
  api: FilesApi;
  openMenu: (x: number, y: number, splitId: string, path: string) => void;
}

function FileTab({ path, split, api, openMenu }: TabProps) {
  const buf = api.state.buffers[path];
  const dirty = !!buf && !buf.loading && buf.content !== buf.baseline;
  const isActive = split.activePath === path;
  const [dropBefore, setDropBefore] = useState(false);
  const label = path.split("/").pop() || path;

  return (
    <div
      className={
        "files-tab" +
        (isActive ? " active" : "") +
        (dirty ? " dirty" : "") +
        (dropBefore ? " drop-before" : "")
      }
      title={path + (buf?.error ? `\n\nError: ${buf.error}` : "")}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData(
          DRAG_MIME,
          JSON.stringify({ fromSplitId: split.id, path }),
        );
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes(DRAG_MIME)) {
          e.preventDefault();
          setDropBefore(true);
        }
      }}
      onDragLeave={() => setDropBefore(false)}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDropBefore(false);
        const raw = e.dataTransfer.getData(DRAG_MIME);
        if (!raw) return;
        try {
          const { fromSplitId, path: dragged } = JSON.parse(raw) as {
            fromSplitId: string;
            path: string;
          };
          if (dragged === path && fromSplitId === split.id) return;
          api.moveTab(fromSplitId, split.id, dragged, path);
        } catch {
          // ignore
        }
      }}
      onClick={() => api.setActiveTab(split.id, path)}
      onContextMenu={(e) => {
        e.preventDefault();
        openMenu(e.clientX, e.clientY, split.id, path);
      }}
      onAuxClick={(e) => {
        // Middle-click closes — convention from most editors.
        if (e.button === 1) {
          e.preventDefault();
          api.closeTab(split.id, path);
        }
      }}
    >
      <span className="files-tab-label">
        {dirty && <span className="files-tab-dirty-dot">●</span>}
        {label}
      </span>
      <button
        className="files-tab-close"
        title="Close"
        onClick={(e) => {
          e.stopPropagation();
          api.closeTab(split.id, path);
        }}
      >
        ✕
      </button>
    </div>
  );
}
