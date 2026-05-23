import { useQuery } from "@tanstack/react-query";

import { api, MetaCategory, MetaTemplate } from "../api";

interface Props {
  /** Filter the catalog by target_kind. Pass ``"config"`` from
   *  NewProjectModal so scaffolds that produce loose templates (which
   *  can't seed a project's default config) don't show up. Omit to
   *  show everything. */
  targetKind?: "config" | "template";
  selected: MetaTemplate | null;
  onSelect: (mt: MetaTemplate | null) => void;
  /** Whether to render the "Blank file" radio at the top of the tree.
   *  NewTemplateModal uses ``true`` because the picker is the only place
   *  the user can choose Blank. NewProjectModal uses ``false`` because
   *  it owns a tri-state radio (Blank / Copy / Scaffold) at the section
   *  level, and "Blank" lives there. */
  showBlankOption?: boolean;
  /** Render a detail panel on the right with the selected scaffold's
   *  title, description, and field list. ``true`` is the standard
   *  two-pane layout; set ``false`` when the parent shows its own
   *  detail/form alongside (e.g. NewProjectModal renders the form
   *  fields inline next to the project fields). */
  showDetailPanel?: boolean;
  /** Disable interaction (radio buttons + tree collapse). Used by
   *  NewProjectModal to grey out the picker when a different starting
   *  point is selected. */
  disabled?: boolean;
}

/** Two-pane meta-template picker: tree on the left, optional detail
 *  panel on the right. Fetches the catalog once via
 *  ``GET /api/project/meta-templates`` and renders it as a collapsible
 *  tree, with the leaf rows hosting radio buttons. */
export function MetaTemplatePicker({
  targetKind,
  selected,
  onSelect,
  showBlankOption = true,
  showDetailPanel = true,
  disabled = false,
}: Props) {
  const metaQ = useQuery({
    queryKey: ["meta-templates"],
    queryFn: () => api.listMetaTemplates(),
    staleTime: 5 * 60 * 1000,
  });

  const categories = metaQ.data ?? [];
  const filtered = targetKind
    ? filterCategoriesByKind(categories, targetKind)
    : categories;

  return (
    <div className={`meta-picker${disabled ? " disabled" : ""}`}>
      <div className="meta-picker-tree">
        {showBlankOption && (
          <label className="meta-picker-blank">
            <input
              type="radio"
              checked={selected === null}
              onChange={() => onSelect(null)}
              disabled={disabled}
            />
            <div>
              <div className="meta-picker-blank-title">Blank file</div>
              <div className="muted meta-picker-blank-desc">
                Create an empty file and fill it in yourself.
              </div>
            </div>
          </label>
        )}

        {metaQ.isLoading && <div className="muted pad">Loading scaffolds…</div>}
        {metaQ.error != null && (
          <div className="err pad">
            <pre>{String(metaQ.error)}</pre>
          </div>
        )}
        {!metaQ.isLoading && !metaQ.error && filtered.length === 0 && (
          <div className="muted pad">No scaffolds available.</div>
        )}
        {filtered.map((c) => (
          <MetaCategoryNode
            key={c.name}
            category={c}
            selected={selected}
            onSelect={onSelect}
            depth={0}
            disabled={disabled}
          />
        ))}
      </div>

      {showDetailPanel && (
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
                          <span className="muted">
                            {" "}
                            (default: {f.default})
                          </span>
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
      )}
    </div>
  );
}

function MetaCategoryNode({
  category,
  selected,
  onSelect,
  depth,
  disabled,
}: {
  category: MetaCategory;
  selected: MetaTemplate | null;
  onSelect: (mt: MetaTemplate) => void;
  depth: number;
  disabled: boolean;
}) {
  // Top-level groups open by default; nested groups stay collapsed so
  // the tree doesn't visually explode when browsing the full catalog.
  const defaultOpen = depth === 0;
  return (
    <details className="meta-picker-cat" open={defaultOpen}>
      <summary>
        <span className="meta-picker-cat-title">{category.title}</span>
        {(category.summary || category.description) && (
          <span className="muted meta-picker-cat-desc">
            {" — "}
            {category.summary || category.description}
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
              disabled={disabled}
            />
            <div>
              <div className="meta-picker-leaf-title">{t.title}</div>
              {(t.summary || t.description) && (
                <div className="muted meta-picker-leaf-desc">
                  {t.summary || t.description}
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
            disabled={disabled}
          />
        ))}
      </div>
    </details>
  );
}

/** Prune categories down to scaffolds whose ``target_kind`` matches.
 *  A category is dropped when neither it nor any descendant has a
 *  matching leaf — keeps the picker from showing empty branches when
 *  the caller filtered out everything inside them. */
function filterCategoriesByKind(
  categories: MetaCategory[],
  kind: "config" | "template",
): MetaCategory[] {
  const out: MetaCategory[] = [];
  for (const c of categories) {
    const templates = c.templates.filter((t) => t.target_kind === kind);
    const children = filterCategoriesByKind(c.children, kind);
    if (templates.length === 0 && children.length === 0) continue;
    out.push({ ...c, templates, children });
  }
  return out;
}
