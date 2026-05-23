import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import {
  api,
  MetaCategory,
  MetaTemplate,
  ProjectInfo,
} from "../api";
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

  const metaQ = useQuery({
    queryKey: ["meta-templates"],
    queryFn: () => api.listMetaTemplates(),
    staleTime: 5 * 60 * 1000,
  });

  // When the user picks a scaffold, pre-fill the values map with its
  // declared defaults so the form opens with sensible starting text
  // instead of empty boxes.
  useEffect(() => {
    if (!selected) {
      setValues({});
      return;
    }
    const next: Record<string, string> = {};
    for (const f of selected.fields) {
      next[f.name] = f.default ?? "";
    }
    setValues(next);
  }, [selected]);

  const baseDir =
    kind === "config" ? pathsQ.data?.configs_dir : pathsQ.data?.templates_dir;

  const trimmed = name.trim();
  const withSuffix =
    trimmed && /\.[a-zA-Z0-9]+$/.test(trimmed) ? trimmed : trimmed + ".yaml";
  const preview = baseDir && trimmed ? `${baseDir}/${withSuffix}` : null;

  const missingRequired = useMemo(() => {
    if (!selected) return [] as string[];
    return selected.fields
      .filter((f) => f.required && !(values[f.name] ?? "").trim())
      .map((f) => f.name);
  }, [selected, values]);

  const create = useMutation({
    mutationFn: () =>
      api.newProjectTemplate(
        project.project_dir,
        kind,
        trimmed,
        selected
          ? { meta_template: selected.id, values }
          : undefined,
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
    !!trimmed &&
    !create.isPending &&
    !!baseDir &&
    missingRequired.length === 0;

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
              categories={metaQ.data ?? []}
              loading={metaQ.isLoading}
              error={metaQ.error}
              selected={selected}
              onSelect={setSelected}
            />
          ) : (
            <FillForm
              selected={selected}
              name={name}
              setName={setName}
              values={values}
              setValues={setValues}
              placeholder={placeholder}
              preview={preview}
              onEnterSubmit={submit}
              kind={kind}
            />
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
              <button
                onClick={() => setStep("fill")}
                disabled={metaQ.isLoading}
              >
                Next →
              </button>
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

// ----------------------------------------------------------------------
// Step 1: pick a meta-template (or "Blank file")

function MetaTemplatePicker({
  categories,
  loading,
  error,
  selected,
  onSelect,
}: {
  categories: MetaCategory[];
  loading: boolean;
  error: unknown;
  selected: MetaTemplate | null;
  onSelect: (mt: MetaTemplate | null) => void;
}) {
  return (
    <div className="meta-picker">
      <div className="meta-picker-tree">
        <label className="meta-picker-blank">
          <input
            type="radio"
            checked={selected === null}
            onChange={() => onSelect(null)}
          />
          <div>
            <div className="meta-picker-blank-title">Blank file</div>
            <div className="muted meta-picker-blank-desc">
              Create an empty file and fill it in yourself.
            </div>
          </div>
        </label>

        {loading && <div className="muted pad">Loading scaffolds…</div>}
        {error != null && (
          <div className="err pad">
            <pre>{String(error)}</pre>
          </div>
        )}
        {!loading && !error && categories.length === 0 && (
          <div className="muted pad">No scaffolds available.</div>
        )}
        {categories.map((c) => (
          <MetaCategoryNode
            key={c.name}
            category={c}
            selected={selected}
            onSelect={onSelect}
            depth={0}
          />
        ))}
      </div>

      <div className="meta-picker-detail">
        {selected ? (
          <>
            <h4>{selected.title}</h4>
            {selected.description && (
              <p className="muted">{selected.description}</p>
            )}
            {selected.fields.length > 0 && (
              <>
                <div className="muted meta-picker-fields-label">
                  Fields you&apos;ll be asked for:
                </div>
                <ul className="meta-picker-fields">
                  {selected.fields.map((f) => (
                    <li key={f.name}>
                      <code>{f.label || f.name}</code>
                      {f.required ? (
                        <span className="meta-picker-req"> required</span>
                      ) : f.default != null ? (
                        <span className="muted"> (default: {f.default})</span>
                      ) : (
                        <span className="muted"> (optional)</span>
                      )}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        ) : (
          <p className="muted">
            Select a starting point. The new file will be created with
            the scaffold pre-filled, ready to refine by hand.
          </p>
        )}
      </div>
    </div>
  );
}

function MetaCategoryNode({
  category,
  selected,
  onSelect,
  depth,
}: {
  category: MetaCategory;
  selected: MetaTemplate | null;
  onSelect: (mt: MetaTemplate) => void;
  depth: number;
}) {
  // Top-level groups open by default; nested groups stay collapsed so the
  // tree doesn't visually explode for users browsing the catalog.
  const defaultOpen = depth === 0;
  return (
    <details className="meta-picker-cat" open={defaultOpen}>
      <summary>
        <span className="meta-picker-cat-title">{category.title}</span>
        {category.description && (
          <span className="muted meta-picker-cat-desc">
            {" — "}
            {category.description}
          </span>
        )}
      </summary>
      <div className="meta-picker-cat-body">
        {category.templates.map((t) => (
          <label
            key={t.id}
            className={`meta-picker-leaf${selected?.id === t.id ? " selected" : ""}`}
          >
            <input
              type="radio"
              checked={selected?.id === t.id}
              onChange={() => onSelect(t)}
            />
            <div>
              <div className="meta-picker-leaf-title">{t.title}</div>
              {t.description && (
                <div className="muted meta-picker-leaf-desc">
                  {t.description}
                </div>
              )}
            </div>
          </label>
        ))}
        {category.children.map((c) => (
          <MetaCategoryNode
            key={c.name}
            category={c}
            selected={selected}
            onSelect={onSelect}
            depth={depth + 1}
          />
        ))}
      </div>
    </details>
  );
}

// ----------------------------------------------------------------------
// Step 2: fill name + scaffold fields

function FillForm({
  selected,
  name,
  setName,
  values,
  setValues,
  placeholder,
  preview,
  onEnterSubmit,
  kind,
}: {
  selected: MetaTemplate | null;
  name: string;
  setName: (v: string) => void;
  values: Record<string, string>;
  setValues: (v: Record<string, string>) => void;
  placeholder: string;
  preview: string | null;
  onEnterSubmit: () => void;
  kind: "config" | "template";
}) {
  return (
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
              onEnterSubmit();
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

      {selected && selected.fields.length > 0 && (
        <div className="meta-fill-form">
          <div className="muted meta-fill-form-label">
            From scaffold <code>{selected.title}</code>
          </div>
          {selected.fields.map((f) => (
            <div key={f.name} className="new-template-input-row">
              <label htmlFor={`meta-field-${f.name}`} className="muted">
                {f.label || f.name}
                {f.required && (
                  <span className="meta-picker-req"> *</span>
                )}
              </label>
              <input
                id={`meta-field-${f.name}`}
                type="text"
                value={values[f.name] ?? ""}
                onChange={(e) =>
                  setValues({ ...values, [f.name]: e.target.value })
                }
                placeholder={f.placeholder}
                spellCheck={false}
              />
              {f.description && (
                <div className="muted meta-fill-field-help">
                  {f.description}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  );
}
