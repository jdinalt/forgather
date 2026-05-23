import { useEffect } from "react";

import { MetaTemplate } from "../api";

interface Props {
  scaffold: MetaTemplate;
  values: Record<string, string>;
  onChange: (values: Record<string, string>) => void;
  disabled?: boolean;
  /** Optional label rendered above the field block. Defaults to
   *  ``From scaffold <title>`` so the user can tell where the form
   *  came from when it lives next to other unrelated inputs. */
  heading?: string;
}

/** Renders the form fields declared by a meta-template's manifest.
 *  Used both by NewTemplateModal (below the file-name input) and by
 *  NewProjectModal (inside the tri-state starting-point section). */
export function MetaTemplateFields({
  scaffold,
  values,
  onChange,
  disabled,
  heading,
}: Props) {
  // When the scaffold changes, pre-fill values with manifest defaults
  // so the form opens with sensible starting text instead of empty
  // boxes. Only fields not already populated are touched, so the
  // user's edits survive a re-render that re-passes the same scaffold.
  useEffect(() => {
    const next: Record<string, string> = { ...values };
    let mutated = false;
    for (const f of scaffold.fields) {
      if (next[f.name] === undefined) {
        next[f.name] = f.default ?? "";
        mutated = true;
      }
    }
    if (mutated) onChange(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scaffold.id]);

  if (scaffold.fields.length === 0) return null;

  const label = heading ?? `From scaffold ${scaffold.title}`;

  return (
    <div className="meta-fill-form">
      <div className="muted meta-fill-form-label">{label}</div>
      {scaffold.fields.map((f) => (
        <div key={f.name} className="new-template-input-row">
          <label htmlFor={`meta-field-${f.name}`} className="muted">
            {f.label || f.name}
            {f.required && <span className="meta-picker-req"> *</span>}
          </label>
          <input
            id={`meta-field-${f.name}`}
            type="text"
            value={values[f.name] ?? ""}
            onChange={(e) =>
              onChange({ ...values, [f.name]: e.target.value })
            }
            placeholder={f.placeholder}
            spellCheck={false}
            disabled={disabled}
          />
          {f.description && (
            <div className="muted meta-fill-field-help">{f.description}</div>
          )}
        </div>
      ))}
    </div>
  );
}

/** Return the names of required fields that aren't satisfied by
 *  ``values``. Used by parent modals to gate the submit button. */
export function missingRequiredFields(
  scaffold: MetaTemplate | null,
  values: Record<string, string>,
): string[] {
  if (!scaffold) return [];
  return scaffold.fields
    .filter((f) => f.required && !(values[f.name] ?? "").trim())
    .map((f) => f.name);
}
