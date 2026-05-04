import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api } from "../api";
import { PathField } from "./PathField";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  /** The directory the user right-clicked in the Files tree.
   *  ``forgather_workspace/`` will be created inside this dir. */
  workspaceDir: string;
  onCreated: (workspace_dir: string) => void;
  onClose: () => void;
}

/** Companion to ``NewWorkspaceModal`` for the Files-tree right-click
 *  flow: the user has already picked a specific directory, so we drop
 *  the parent-search-root dropdown and the workspace-dir-name field
 *  (both irrelevant when the path is fixed) and just collect the
 *  metadata: name, description, forgather dir, libs, additional search
 *  paths. The clicked directory is shown read-only at the top so the
 *  user can confirm. */
export function InitWorkspaceModal({
  workspaceDir,
  onCreated,
  onClose,
}: Props) {
  const qc = useQueryClient();
  const quickQ = useQuery({
    queryKey: ["fs-quick-paths"],
    queryFn: api.fsQuickPaths,
    staleTime: 5 * 60 * 1000,
  });

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [forgatherDir, setForgatherDir] = useState("");
  // Pre-filled with the two libraries every workspace in the repo uses
  // (base + examples). Same default as NewWorkspaceModal.
  const [libs, setLibs] = useState("base\nexamples");
  const [searchPaths, setSearchPaths] = useState("");
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    nameRef.current?.focus();
  }, []);

  // Default forgather_dir to the bundled "Forgather repo" quick-path.
  useEffect(() => {
    if (forgatherDir) return;
    const fg = quickQ.data?.find((q) => q.label === "Forgather repo");
    if (fg) setForgatherDir(fg.path);
  }, [quickQ.data, forgatherDir]);

  const splitLines = (s: string) =>
    s
      .split(/\r?\n/)
      .map((x) => x.trim())
      .filter((x) => x.length > 0);

  const create = useMutation({
    mutationFn: () =>
      api.initWorkspaceHere({
        workspace_dir: workspaceDir,
        name: name.trim(),
        description: description.trim(),
        forgather_dir: forgatherDir.trim(),
        libs: splitLines(libs),
        search_paths: splitLines(searchPaths),
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      // Refresh just the workspace directory's listing so the new
      // ``forgather_workspace/`` subdir appears.
      qc.invalidateQueries({
        queryKey: ["fs-browse", r.workspace_dir],
        exact: false,
      });
      onCreated(r.workspace_dir);
      onClose();
    },
  });

  const canSubmit =
    !!name.trim() &&
    !!description.trim() &&
    !!forgatherDir.trim() &&
    !create.isPending;

  const submit = () => {
    if (!canSubmit) return;
    create.mutate();
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal new-workspace-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Initialize workspace here"
      >
        <header className="modal-header">
          <h3>Initialize workspace here</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-summary">
            <div>
              <span className="muted">target</span>
              <code title={workspaceDir}>{workspaceDir}</code>
            </div>
            <div className="muted-hint">
              <code>forgather_workspace/</code> will be created inside this
              directory; the directory's existing contents are left alone.
            </div>
          </div>
          <div className="new-project-grid">
            <label>
              <span className="muted">Name *</span>
              <input
                ref={nameRef}
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Workspace"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    submit();
                  }
                }}
              />
            </label>
            <label>
              <span className="muted">Description *</span>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Short description"
              />
            </label>
            <label className="full-row">
              <span className="muted">Forgather dir *</span>
              <PathField
                value={forgatherDir}
                onChange={setForgatherDir}
                placeholder="/path/to/forgather"
                mode="dirs-only"
                title="Pick Forgather installation directory"
              />
            </label>
            <label className="full-row">
              <span className="muted">
                Libraries
                <span className="muted-hint">
                  {" "}
                  · one per line; resolved under{" "}
                  <code>forgather/templatelib/</code>. Defaults to{" "}
                  <code>base</code> + <code>examples</code>.
                </span>
              </span>
              <textarea
                value={libs}
                onChange={(e) => setLibs(e.target.value)}
                rows={3}
                spellCheck={false}
              />
            </label>
            <label className="full-row">
              <span className="muted">
                Additional search paths
                <span className="muted-hint">
                  {" "}
                  · one absolute path per line
                </span>
              </span>
              <textarea
                value={searchPaths}
                onChange={(e) => setSearchPaths(e.target.value)}
                rows={3}
                spellCheck={false}
              />
            </label>
          </div>

          {create.error && (
            <div className="err pad">
              <pre>{String(create.error)}</pre>
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div />
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button onClick={submit} disabled={!canSubmit}>
              {create.isPending ? "Creating…" : "Initialize"}
            </button>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
