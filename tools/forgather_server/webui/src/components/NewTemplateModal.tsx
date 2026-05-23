import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api, MetaTemplate, ProjectInfo } from "../api";
import { MetaTemplatePicker } from "./MetaTemplatePicker";
import {
  MetaTemplateFields,
  missingRequiredFields,
} from "./MetaTemplateFields";
import { ModalBackdrop } from "./ModalBackdrop";

interface Props {
  project: ProjectInfo;
  kind: "config" | "template";
  onCreated: (path: string) => void;
  onClose: () => void;
}

/** New Config / New Template flow. Two steps:
 *
 *   1. Pick a starting point — "Blank file" or one of the meta-templates
 *      from `templatelib/meta/`, rendered as a tree.
 *   2. Enter the filename and (when a scaffold is picked) fill the form
 *      fields declared by the scaffold's manifest.
 *
 *  `kind="config"` writes under `<templates>/<config_prefix>/<name>`;
 *  `kind="template"` writes under `<templates>/<name>` directly. The
 *  base path is resolved from the project's MetaConfig (server-side) and
 *  shown in the preview so the user knows exactly where the file lands. */
export function NewTemplateModal({ project, kind, onCreated, onClose }: Props) {
  const qc = useQueryClient();
  const [step, setStep] = useState<"pick" | "fill">("pick");
  // null = "Blank file" (no scaffold). Otherwise the selected MetaTemplate.
  const [selected, setSelected] = useState<MetaTemplate | null>(null);
  const [name, setName] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});

  const pathsQ = useQuery({
    queryKey: ["template-paths", project.project_dir],
    queryFn: () => api.projectTemplatePaths(project.project_dir),
    staleTime: 5 * 60 * 1000,
  });

  const baseDir =
    kind === "config" ? pathsQ.data?.configs_dir : pathsQ.data?.templates_dir;

  const trimmed = name.trim();
  const withSuffix =
    trimmed && /\.[a-zA-Z0-9]+$/.test(trimmed) ? trimmed : trimmed + ".yaml";
  const preview = baseDir && trimmed ? `${baseDir}/${withSuffix}` : null;

  const missing = useMemo(
    () => missingRequiredFields(selected, values),
    [selected, values],
  );

  const create = useMutation({
    mutationFn: () =>
      api.newProjectTemplate(
        project.project_dir,
        kind,
        trimmed,
        selected ? { meta_template: selected.id, values } : undefined,
      ),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({
        queryKey: ["project-templates", project.project_dir],
      });
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

  const canSubmit =
    !!trimmed && !create.isPending && !!baseDir && missing.length === 0;

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
          <h3>
            {title}
            {step === "pick" ? " — pick a starting point" : " — configure"}
          </h3>
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

          {step === "pick" ? (
            <MetaTemplatePicker
              targetKind={kind}
              selected={selected}
              onSelect={setSelected}
            />
          ) : (
            <>
              <div className="new-template-input-row">
                <label htmlFor="new-template-name" className="muted">
                  File name
                </label>
                <input
                  id="new-template-name"
                  autoFocus
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
                <code>
                  {kind === "config"
                    ? "experiments/foo.yaml"
                    : "shared/x.yaml"}
                </code>
                ).
              </div>

              {preview && (
                <div className="new-template-preview">
                  <span className="muted">will create</span>
                  <code title={preview}>{preview}</code>
                </div>
              )}

              {selected && (
                <MetaTemplateFields
                  scaffold={selected}
                  values={values}
                  onChange={setValues}
                />
              )}
            </>
          )}

          {create.error && (
            <div className="err pad">
              <pre>{String(create.error)}</pre>
            </div>
          )}
        </div>

        <footer className="modal-footer">
          <div>
            {step === "fill" && (
              <button className="secondary" onClick={() => setStep("pick")}>
                ← Back
              </button>
            )}
          </div>
          <div className="btn-row">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            {step === "pick" ? (
              <button onClick={() => setStep("fill")}>Next →</button>
            ) : (
              <button onClick={submit} disabled={!canSubmit}>
                {create.isPending ? "Creating…" : "Create"}
              </button>
            )}
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
