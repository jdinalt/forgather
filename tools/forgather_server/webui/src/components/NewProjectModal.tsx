import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api, WorkspaceCluster } from "../api";
import { DirectoryBrowser } from "./DirectoryBrowser";
import { PathField } from "./PathField";

interface Props {
  workspace: WorkspaceCluster;
  /** Pre-fill the nested project-dir field. Used when the modal is
   *  popped from the Files-tree right-click on a subdirectory of the
   *  workspace — the relative path from workspace_root lands here
   *  with a trailing slash so the user only types the leaf name. */
  initialProjectDirName?: string;
  onCreated: (project_dir: string) => void;
  onClose: () => void;
}

/** Mirror of ``forgather project create``. The workspace is identified by
 *  the ``workspace_root`` of the cluster the user right-clicked. ``name``
 *  and ``description`` are required; the others have CLI-matching
 *  defaults (config_prefix=configs, default_config=default.yaml). The
 *  ``project_dir_name`` field is the on-disk directory for the project —
 *  defaults to the lowercased name with spaces -> underscores, matching
 *  the CLI. ``copy_from`` is an optional source config that gets copied
 *  in as the default. */
export function NewProjectModal({
  workspace,
  initialProjectDirName,
  onCreated,
  onClose,
}: Props) {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [configPrefix, setConfigPrefix] = useState("configs");
  const [defaultConfig, setDefaultConfig] = useState("default.yaml");
  const [projectDirName, setProjectDirName] = useState(
    initialProjectDirName ?? "",
  );
  const [copyFrom, setCopyFrom] = useState("");
  const [browsingProjDir, setBrowsingProjDir] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    nameRef.current?.focus();
  }, []);

  // Auto-derived dir name preview (matches CLI: spaces -> _, lowercased).
  // Only used as the placeholder; the user can override by typing into
  // the projectDirName field.
  const derivedDirName = name.trim().replace(/\s+/g, "_").toLowerCase();
  const effectiveDirName = projectDirName.trim() || derivedDirName;
  const targetPreview =
    effectiveDirName && workspace.workspace_root
      ? `${workspace.workspace_root.replace(/\/+$/, "")}/${effectiveDirName}`
      : null;

  const create = useMutation({
    mutationFn: () =>
      api.newProject({
        workspace_dir: workspace.workspace_root,
        name: name.trim(),
        description: description.trim(),
        config_prefix: configPrefix.trim() || "configs",
        default_config: defaultConfig.trim() || "default.yaml",
        project_dir_name: projectDirName.trim() || null,
        copy_from: copyFrom.trim() || null,
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      // Refresh only the immediate parent of the new project dir in
      // the Files tree. The new project lives at
      // ``workspace_dir/<project_dir_name>``; the parent we need to
      // refetch is therefore the workspace_dir itself (or, for nested
      // project_dir_name, the dirname of the resolved path).
      const parent =
        r.project_dir.replace(/\/+$/, "").split("/").slice(0, -1).join("/") ||
        "/";
      qc.invalidateQueries({
        queryKey: ["fs-browse", parent],
        exact: false,
      });
      onCreated(r.project_dir);
      onClose();
    },
  });

  const canSubmit =
    !!name.trim() && !!description.trim() && !create.isPending;

  const submit = () => {
    if (!canSubmit) return;
    create.mutate();
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal new-project-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Create project"
      >
        <header className="modal-header">
          <h3>Create project</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-summary">
            <div>
              <span className="muted">workspace</span>
              <code title={workspace.workspace_root}>
                {workspace.name || workspace.workspace_root}
              </code>
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
                placeholder="My Project"
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
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    submit();
                  }
                }}
              />
            </label>
            <label>
              <span className="muted">Config prefix</span>
              <input
                type="text"
                value={configPrefix}
                onChange={(e) => setConfigPrefix(e.target.value)}
                placeholder="configs"
                spellCheck={false}
              />
            </label>
            <label>
              <span className="muted">Default config</span>
              <input
                type="text"
                value={defaultConfig}
                onChange={(e) => setDefaultConfig(e.target.value)}
                placeholder="default.yaml"
                spellCheck={false}
              />
            </label>
            <label>
              <span className="muted">
                Project dir
                <span className="muted-hint">
                  {" "}
                  · relative to workspace; may be nested. Use Browse… to
                  pick an existing subdirectory of the workspace.
                </span>
              </span>
              <div className="path-field">
                <input
                  type="text"
                  value={projectDirName}
                  onChange={(e) => setProjectDirName(e.target.value)}
                  placeholder={derivedDirName || "auto-derived from name"}
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="secondary"
                  disabled={!workspace.workspace_root}
                  onClick={() => setBrowsingProjDir(true)}
                >
                  Browse…
                </button>
              </div>
            </label>
            <label>
              <span className="muted">Copy from (optional)</span>
              <PathField
                value={copyFrom}
                onChange={setCopyFrom}
                placeholder="Optional source config to copy as default"
                mode="files-and-dirs"
                title="Pick source config"
              />
            </label>
          </div>

          {targetPreview && (
            <div className="new-template-preview">
              <span className="muted">will create</span>
              <code title={targetPreview}>{targetPreview}</code>
            </div>
          )}

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
              {create.isPending ? "Creating…" : "Create"}
            </button>
          </div>
        </footer>

        {browsingProjDir && workspace.workspace_root && (
          <DirectoryBrowser
            initialPath={workspace.workspace_root}
            mode="dirs-only"
            title={`Pick existing subdirectory under ${workspace.workspace_root}`}
            onCancel={() => setBrowsingProjDir(false)}
            onPick={(picked) => {
              setBrowsingProjDir(false);
              const wsNorm = workspace.workspace_root.replace(/\/+$/, "");
              const pickedNorm = picked.replace(/\/+$/, "");
              if (
                pickedNorm !== wsNorm &&
                !pickedNorm.startsWith(wsNorm + "/")
              ) {
                alert(
                  `Picked directory is not under the workspace:\n\n${picked}\n\nNot under: ${workspace.workspace_root}`,
                );
                return;
              }
              const rel =
                pickedNorm === wsNorm ? "" : pickedNorm.slice(wsNorm.length + 1);
              setProjectDirName(rel ? `${rel}/` : "");
            }}
          />
        )}
      </div>
    </div>
  );
}
