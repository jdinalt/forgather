import { useState } from "react";
import { DynamicArg } from "../api";
import { DirectoryBrowser } from "./DirectoryBrowser";

export interface DynamicArgsFormProps {
  schema: DynamicArg[];
  values: Record<string, string>;
  onChange: (dest: string, value: string) => void;
}

export function DynamicArgsForm({ schema, values, onChange }: DynamicArgsFormProps) {
  return (
    <div className="dyn-args">
      {schema.map((a) => (
        <DynArgField
          key={a.dest}
          arg={a}
          value={values[a.dest] ?? ""}
          onChange={(v) => onChange(a.dest, v)}
        />
      ))}
    </div>
  );
}

function DynArgField({
  arg,
  value,
  onChange,
}: {
  arg: DynamicArg;
  value: string;
  onChange: (v: string) => void;
}) {
  const placeholder = arg.default != null ? String(arg.default) : `(${arg.type})`;
  const [browsing, setBrowsing] = useState(false);

  // Argparse-style ``choices`` always wins — render a dropdown regardless
  // of the declared type. A non-list or empty list falls through to the
  // normal type-driven widget.
  const hasChoices =
    Array.isArray(arg.choices) && arg.choices.length > 0;

  let widget: JSX.Element;
  if (hasChoices) {
    widget = (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">(template default)</option>
        {(arg.choices as unknown[]).map((c) => {
          const s = String(c);
          return (
            <option key={s} value={s}>
              {s}
            </option>
          );
        })}
      </select>
    );
  } else if (arg.type === "bool" && typeof arg.default === "boolean") {
    // store_true / store_false flag — known concrete default, no
    // "unset" state. Backend tags these by reporting type=bool with a
    // boolean default. Render as a checkbox initialized to the cached
    // override (if any) or the default.
    const defaulted = arg.default as boolean;
    const checked = value === "" ? defaulted : value === "true";
    widget = (
      <label className="dyn-checkbox">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked ? "true" : "false")}
        />
        <span className="muted">{checked ? "true" : "false"}</span>
        {value === "" && (
          <span className="muted dyn-checkbox-note">
            (default: {defaulted ? "true" : "false"})
          </span>
        )}
      </label>
    );
  } else if (arg.type === "bool") {
    // Generic value-bearing bool with no concrete default — keep the
    // tri-state select so "unset" is distinguishable from explicit false.
    widget = (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">(template default)</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  } else if (arg.type === "path") {
    // Text input + "Browse…" button. The file picker lists files as well
    // as directories; paths here can refer to either (e.g. a checkpoint
    // file vs an output directory). Clicking a file picks it; clicking a
    // directory navigates; the modal's footer still supports picking the
    // currently-shown directory.
    widget = (
      <div className="path-field">
        <input
          type="text"
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          className="secondary"
          onClick={() => setBrowsing(true)}
        >
          Browse…
        </button>
      </div>
    );
  } else {
    widget = (
      <input
        type={arg.type === "int" || arg.type === "float" ? "number" : "text"}
        step={arg.type === "float" ? "any" : undefined}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  return (
    <div className="dyn-field">
      <label>
        <div className="dyn-name">
          <code>{arg.cli_name}</code>
          <span className="dyn-type muted">{arg.type}</span>
        </div>
        {widget}
      </label>
      {arg.help && <div className="dyn-help muted">{arg.help}</div>}
      {browsing && (
        <DirectoryBrowser
          initialPath={value || undefined}
          mode="files-and-dirs"
          title={`Pick path for ${arg.cli_name}`}
          onCancel={() => setBrowsing(false)}
          onPick={(p) => {
            onChange(p);
            setBrowsing(false);
          }}
        />
      )}
    </div>
  );
}

/** Convert the string-typed form state into the right JSON shape. Blank
 *  fields are omitted so the template's own defaults take precedence (same
 *  semantic as the CLI when a flag isn't passed). */
export function coerceArgs(
  raw: Record<string, string>,
  schema: DynamicArg[],
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const a of schema) {
    const v = raw[a.dest];
    if (v == null || v === "") continue;
    switch (a.type) {
      case "int":
        out[a.dest] = Number.parseInt(v, 10);
        break;
      case "float":
        out[a.dest] = Number.parseFloat(v);
        break;
      case "bool":
        out[a.dest] = v === "true";
        break;
      default:
        out[a.dest] = v;
    }
  }
  return out;
}
