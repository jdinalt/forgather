import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api, ProjectInfo } from "../api";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  project: ProjectInfo;
  kind: "config" | "template";
  onCreated: (path: string) => void;
  onClose: () => void;
}

/** Prompt for a new config or template file name. Lives next to the
 *  project's other modals stylistically so the New Config / New Template
 *  flow doesn't fall back to a browser ``prompt()``.
 *
 *  ``kind="config"`` writes under ``<templates>/<config_prefix>/<name>``;
 *  ``kind="template"`` writes under ``<templates>/<name>`` directly. The
 *  base path is resolved from the project's MetaConfig (server-side) and
 *  shown in the preview so the user knows exactly where the file lands. */
export function NewTemplateModal({ project, kind, onCreated, onClose }: Props) {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-focus the name input on open. Without this the user has to
  // tab/click through the dialog before typing.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const pathsQ = useQuery({
    queryKey: ["template-paths", project.project_dir],
    queryFn: () => api.projectTemplatePaths(project.project_dir),
    staleTime: 5 * 60 * 1000,
  });

  const baseDir =
    kind === "config" ? pathsQ.data?.configs_dir : pathsQ.data?.templates_dir;

  // Live preview of the absolute path the file will land at. Mirrors the
  // server's normalization: trim, append .yaml when no extension, treat
  // the input as relative to baseDir. Subdirectories are allowed.
  const trimmed = name.trim();
  const withSuffix =
    trimmed && /\.[a-zA-Z0-9]+$/.test(trimmed) ? trimmed : trimmed + ".yaml";
  const preview = baseDir && trimmed ? `${baseDir}/${withSuffix}` : null;

  const create = useMutation({
    mutationFn: () =>
      api.newProjectTemplate(project.project_dir, kind, trimmed),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({
        queryKey: ["project-templates", project.project_dir],
      });
      // Sidebar Files tree: refresh only the immediate parent dir of
      // the newly-created template / config file.
      const parent =
        r.path.replace(/\/+$/, "").split("/").slice(0, -1).join("/") || "/";
      qc.invalidateQueries({
        queryKey: ["fs-browse", parent],
        exact: false,
      });
      onCreated(r.path);
      onClose();
    },
  });

  const canSubmit = !!trimmed && !create.isPending && !!baseDir;

  const submit = () => {
    if (!canSubmit) return;
    create.mutate();
  };

  const title = kind === "config" ? "New Config" : "New Template";
  const placeholder =
    kind === "config" ? "my_experiment.yaml" : "shared/my_block.yaml";

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        className="modal new-template-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={title}
      >
        <header className="modal-header">
          <h3>{title}</h3>
          <button className="tiny" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          <div className="submit-summary">
            <div>
              <span className="muted">project</span>
              <code>{project.name || project.project_dir}</code>
            </div>
            <div>
              <span className="muted">kind</span>
              <code>{kind}</code>
            </div>
            <div>
              <span className="muted">base</span>
              <code title={baseDir ?? ""}>
                {pathsQ.isLoading
                  ? "resolving…"
                  : baseDir ?? "(unable to resolve)"}
              </code>
            </div>
          </div>

          {pathsQ.error && (
            <div className="err pad">
              <pre>{String(pathsQ.error)}</pre>
            </div>
          )}

          <div className="new-template-input-row">
            <label htmlFor="new-template-name" className="muted">
              File name
            </label>
            <input
              id="new-template-name"
              ref={inputRef}
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={placeholder}
              spellCheck={false}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  submit();
                }
              }}
            />
          </div>
          <div className="muted new-template-hint">
            Extension defaults to <code>.yaml</code> if omitted.
            Subdirectories are allowed (e.g.{" "}
            <code>{kind === "config" ? "experiments/foo.yaml" : "shared/x.yaml"}</code>
            ).
          </div>

          {preview && (
            <div className="new-template-preview">
              <span className="muted">will create</span>
              <code title={preview}>{preview}</code>
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
      </div>
    </ModalBackdrop>
  );
}
