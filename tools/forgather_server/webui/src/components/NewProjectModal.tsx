import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { api, MetaTemplate, WorkspaceCluster } from "../api";
import { DirectoryBrowser } from "./DirectoryBrowser";
import {
  MetaTemplateFields,
  missingRequiredFields,
} from "./MetaTemplateFields";
import { MetaTemplatePicker } from "./MetaTemplatePicker";
import { PathField } from "./PathField";
import { ModalBackdrop } from "./ModalBackdrop";

type StartingPoint = "blank" | "copy" | "scaffold";

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

/** Mirror of ``forgather project create``. The workspace is identified
 *  by the ``workspace_root`` of the cluster the user right-clicked.
 *  ``name`` and ``description`` are required; the others have
 *  CLI-matching defaults (config_prefix=configs,
 *  default_config=default.yaml).
 *
 *  The project's default config is seeded from one of three sources,
 *  picked by the "Starting point" radio:
 *    - Blank      → server writes the built-in empty stub.
 *    - Copy       → server copies the named file.
 *    - Scaffold   → server renders a meta-template against the
 *                   values entered below the picker.
 *  Mutual exclusion is enforced both client-side (radio is one-of-three)
 *  and server-side (the route 400s if both fields are set). */
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
  const [defaultConfig, setDefaultConfig] = useState("");
  const [projectDirName, setProjectDirName] = useState(
    initialProjectDirName ?? "",
  );

  const [startingPoint, setStartingPoint] = useState<StartingPoint>("blank");
  const [copyFrom, setCopyFrom] = useState("");
  const [scaffold, setScaffold] = useState<MetaTemplate | null>(null);
  const [scaffoldValues, setScaffoldValues] = useState<
    Record<string, string>
  >({});

  const [browsingProjDir, setBrowsingProjDir] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    nameRef.current?.focus();
  }, []);

  // Auto-derived dir name preview (matches CLI: spaces -> _, lowercased).
  // Only used as the placeholder; the user can override by typing into
  // the projectDirName field.
  const derivedDirName = slugify(name);
  const effectiveDirName = projectDirName.trim() || derivedDirName;
  const targetPreview =
    effectiveDirName && workspace.workspace_root
      ? `${workspace.workspace_root.replace(/\/+$/, "")}/${effectiveDirName}`
      : null;

  // Default-config filename. When the user picks a scaffold, derive a
  // sensible filename from its CONFIG_NAME field if the user hasn't
  // overridden the field manually. This means a new C4 dataset project
  // gets a ``c4.yaml`` default instead of the bare ``default.yaml``.
  // The user can always overtype the placeholder to force a name.
  const scaffoldDefaultConfig = useMemo(() => {
    if (startingPoint !== "scaffold" || !scaffold) return "";
    const candidate =
      scaffoldValues.CONFIG_NAME?.trim() ||
      scaffoldValues.NAME?.trim() ||
      scaffold.title;
    return candidate ? slugify(candidate) + ".yaml" : "";
  }, [startingPoint, scaffold, scaffoldValues]);

  const effectiveDefaultConfig =
    defaultConfig.trim() || scaffoldDefaultConfig || "default.yaml";

  const missingFields = useMemo(
    () =>
      startingPoint === "scaffold"
        ? missingRequiredFields(scaffold, scaffoldValues)
        : [],
    [startingPoint, scaffold, scaffoldValues],
  );

  const create = useMutation({
    mutationFn: () => {
      const payload: Parameters<typeof api.newProject>[0] = {
        workspace_dir: workspace.workspace_root,
        name: name.trim(),
        description: description.trim(),
        config_prefix: configPrefix.trim() || "configs",
        default_config: effectiveDefaultConfig,
        project_dir_name: projectDirName.trim() || null,
      };
      if (startingPoint === "copy") {
        payload.copy_from = copyFrom.trim() || null;
      } else if (startingPoint === "scaffold" && scaffold) {
        payload.meta_template = scaffold.id;
        payload.values = scaffoldValues;
      }
      return api.newProject(payload);
    },
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
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

  const startingPointValid =
    startingPoint === "blank" ||
    (startingPoint === "copy" && copyFrom.trim().length > 0) ||
    (startingPoint === "scaffold" && scaffold !== null && missingFields.length === 0);

  const canSubmit =
    !!name.trim() &&
    !!description.trim() &&
    startingPointValid &&
    !create.isPending;

  const submit = () => {
    if (!canSubmit) return;
    create.mutate();
  };

  return (
    <ModalBackdrop onClose={onClose}>
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
                placeholder={scaffoldDefaultConfig || "default.yaml"}
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
          </div>

          {/* Starting point: tri-state radio for the first config's seed. */}
          <fieldset className="new-project-starting-point">
            <legend>Starting point</legend>

            <label className="starting-point-row">
              <input
                type="radio"
                name="starting-point"
                checked={startingPoint === "blank"}
                onChange={() => setStartingPoint("blank")}
              />
              <div>
                <div className="starting-point-title">Blank</div>
                <div className="muted starting-point-desc">
                  Use the built-in empty default-config stub.
                </div>
              </div>
            </label>

            <label className="starting-point-row">
              <input
                type="radio"
                name="starting-point"
                checked={startingPoint === "copy"}
                onChange={() => setStartingPoint("copy")}
              />
              <div className="starting-point-body">
                <div className="starting-point-title">Copy from existing file</div>
                <div className={startingPoint === "copy" ? "" : "muted"}>
                  <PathField
                    value={copyFrom}
                    onChange={setCopyFrom}
                    placeholder="/path/to/source.yaml"
                    mode="files-and-dirs"
                    title="Pick source config"
                  />
                </div>
              </div>
            </label>

            <label className="starting-point-row">
              <input
                type="radio"
                name="starting-point"
                checked={startingPoint === "scaffold"}
                onChange={() => setStartingPoint("scaffold")}
              />
              <div className="starting-point-body">
                <div className="starting-point-title">Use a scaffold</div>
                <div className="muted starting-point-desc">
                  Render a starter config from{" "}
                  <code>templatelib/meta/</code> and seed the project with it.
                </div>
                <MetaTemplatePicker
                  targetKind="config"
                  selected={scaffold}
                  onSelect={setScaffold}
                  showBlankOption={false}
                  showDetailPanel={true}
                  disabled={startingPoint !== "scaffold"}
                />
                {startingPoint === "scaffold" && scaffold && (
                  <MetaTemplateFields
                    scaffold={scaffold}
                    values={scaffoldValues}
                    onChange={setScaffoldValues}
                  />
                )}
              </div>
            </label>
          </fieldset>

          {targetPreview && (
            <div className="new-template-preview">
              <span className="muted">will create</span>
              <code title={targetPreview}>{targetPreview}</code>
              <span className="muted">
                {" "}
                with {effectiveDefaultConfig}
              </span>
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
                pickedNorm === wsNorm
                  ? ""
                  : pickedNorm.slice(wsNorm.length + 1);
              setProjectDirName(rel ? `${rel}/` : "");
            }}
          />
        )}
      </div>
    </ModalBackdrop>
  );
}

function slugify(s: string): string {
  return s.trim().replace(/\s+/g, "_").toLowerCase();
}
