import { useMemo, useRef, useState } from "react";
import { DynamicArg } from "../api";
import { DirectoryBrowser } from "./DirectoryBrowser";

export interface DynamicArgsFormProps {
  schema: DynamicArg[];
  values: Record<string, string>;
  onChange: (dest: string, value: string) => void;
  /** When true, ``required: true`` flags drive the missing-arg highlighting
   *  and tree expansion. SubmitModal uses this; OverridesModal sets it
   *  false because saving overrides has no semantic of "missing" — the
   *  user is just editing cached values. */
  enforceRequired?: boolean;
}

interface TreeNode {
  name: string;
  /** Stable path for React keys (colon-joined). */
  path: string;
  children: Record<string, TreeNode>;
  args: DynamicArg[];
}

const OTHER_GROUP_NAME = "Other";

function emptyNode(name: string, path: string): TreeNode {
  return { name, path, children: {}, args: [] };
}

/** Bucket the flat schema into a group-path tree.
 *
 *  - Args with no ``group`` go straight into the root if no other arg has
 *    a group (preserves the existing flat layout).
 *  - As soon as one arg declares a group, ungrouped args are corraled
 *    into an "Other" sibling at the root so the tree stays tidy.
 *  - Group strings are colon-separated paths; whitespace + empty
 *    segments are stripped so ``"Trainer:  :LR"`` collapses sensibly. */
function buildTree(schema: DynamicArg[]): { root: TreeNode; grouped: boolean } {
  const grouped = schema.some((a) => a.group);
  const root = emptyNode("", "");
  for (const a of schema) {
    let parts: string[] = [];
    if (a.group) {
      parts = a.group
        .split(":")
        .map((s) => s.trim())
        .filter(Boolean);
    } else if (grouped) {
      parts = [OTHER_GROUP_NAME];
    }
    let cursor = root;
    for (const p of parts) {
      if (!cursor.children[p]) {
        const childPath = cursor.path ? `${cursor.path}:${p}` : p;
        cursor.children[p] = emptyNode(p, childPath);
      }
      cursor = cursor.children[p];
    }
    cursor.args.push(a);
  }
  return { root, grouped };
}

function sortedChildren(node: TreeNode): TreeNode[] {
  return Object.values(node.children).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
}

function sortedArgs(node: TreeNode): DynamicArg[] {
  return [...node.args].sort((a, b) =>
    a.cli_name.localeCompare(b.cli_name),
  );
}

function isMissing(arg: DynamicArg, values: Record<string, string>): boolean {
  const v = values[arg.dest];
  return v == null || v === "";
}

/** True when the user has typed a numeric value that violates the
 *  declared min/max. Empty / non-numeric inputs are *not* out-of-bounds —
 *  they fall under "missing required" instead so the two states don't
 *  double-flag the same field. Bounds are inclusive on both ends. */
function isOutOfBounds(
  arg: DynamicArg,
  values: Record<string, string>,
): boolean {
  if (arg.type !== "int" && arg.type !== "float") return false;
  if (arg.min == null && arg.max == null) return false;
  const raw = values[arg.dest];
  if (raw == null || raw === "") return false;
  const n = arg.type === "int" ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
  if (!Number.isFinite(n)) return false;
  if (arg.min != null && n < arg.min) return true;
  if (arg.max != null && n > arg.max) return true;
  return false;
}

/** Human-readable bound suffix appended to the tooltip when an arg
 *  declares any bound. Returns null when there's nothing to show so
 *  the caller can fall back to the help text alone. */
function formatBounds(arg: DynamicArg): string | null {
  if (arg.type !== "int" && arg.type !== "float") return null;
  if (arg.min == null && arg.max == null) return null;
  if (arg.min != null && arg.max != null) {
    return `range: [${arg.min}, ${arg.max}]`;
  }
  if (arg.min != null) return `min: ${arg.min}`;
  return `max: ${arg.max}`;
}

function buildTitle(arg: DynamicArg): string | undefined {
  const bounds = formatBounds(arg);
  if (arg.help && bounds) return `${arg.help}\n(${bounds})`;
  if (arg.help) return arg.help;
  if (bounds) return bounds;
  return undefined;
}

/** Recursively walk a subtree to see if any leaf is in an invalid state
 *  (missing-required OR out-of-bounds). Drives the red highlight on the
 *  path and the initial-expanded default. */
function subtreeHasInvalid(
  node: TreeNode,
  values: Record<string, string>,
): boolean {
  for (const a of node.args) {
    if (a.required && isMissing(a, values)) return true;
    if (isOutOfBounds(a, values)) return true;
  }
  for (const c of Object.values(node.children)) {
    if (subtreeHasInvalid(c, values)) return true;
  }
  return false;
}

/** Public helper used by SubmitModal to gate the Submit button. Returns
 *  the dests of any required arg the user has not filled in. */
export function listMissingRequired(
  schema: DynamicArg[],
  values: Record<string, string>,
): DynamicArg[] {
  return schema.filter((a) => a.required && isMissing(a, values));
}

/** Public helper for SubmitModal: dests with a user-supplied value that
 *  violates the declared min/max. */
export function listOutOfBounds(
  schema: DynamicArg[],
  values: Record<string, string>,
): DynamicArg[] {
  return schema.filter((a) => isOutOfBounds(a, values));
}

export function DynamicArgsForm({
  schema,
  values,
  onChange,
  enforceRequired = false,
}: DynamicArgsFormProps) {
  const { root, grouped } = useMemo(() => buildTree(schema), [schema]);

  // Flat (no groups in the schema): render the original list, alphabetized.
  if (!grouped) {
    const flat = sortedArgs(root);
    return (
      <div className="dyn-args">
        {flat.map((a) => (
          <DynArgField
            key={a.dest}
            arg={a}
            value={values[a.dest] ?? ""}
            onChange={(v) => onChange(a.dest, v)}
            missing={enforceRequired && a.required && isMissing(a, values)}
            outOfBounds={isOutOfBounds(a, values)}
          />
        ))}
      </div>
    );
  }

  // Grouped: render top-level args (rare — schemas almost always nest) +
  // a <details> per child group, recursively.
  return (
    <div className="dyn-args dyn-args-grouped">
      {sortedArgs(root).map((a) => (
        <DynArgField
          key={a.dest}
          arg={a}
          value={values[a.dest] ?? ""}
          onChange={(v) => onChange(a.dest, v)}
          missing={enforceRequired && a.required && isMissing(a, values)}
          outOfBounds={isOutOfBounds(a, values)}
        />
      ))}
      {sortedChildren(root).map((child) => (
        <DynArgGroupNode
          key={child.path}
          node={child}
          values={values}
          onChange={onChange}
          enforceRequired={enforceRequired}
        />
      ))}
    </div>
  );
}

function DynArgGroupNode({
  node,
  values,
  onChange,
  enforceRequired,
}: {
  node: TreeNode;
  values: Record<string, string>;
  onChange: (dest: string, value: string) => void;
  enforceRequired: boolean;
}) {
  // Initial expansion state computed once so the user can collapse a
  // problem-containing group manually if they want. Subsequent renders
  // only update the live highlight (driven separately).
  const initialOpenRef = useRef<boolean | null>(null);
  if (initialOpenRef.current === null) {
    initialOpenRef.current = subtreeHasInvalid(node, values) && enforceRequired
      ? true
      : subtreeHasInvalid(node, values) && !enforceRequired
        ? // Out-of-bounds always opens — that's a problem regardless of
          // whether SubmitModal-style required enforcement is active.
          true
        : false;
  }
  const [open, setOpen] = useState<boolean>(initialOpenRef.current);

  const liveInvalid = subtreeHasInvalid(node, values);

  const summaryClass =
    "dyn-group-summary" + (liveInvalid ? " dyn-group-missing" : "");

  return (
    <details
      className="dyn-group"
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className={summaryClass}>{node.name}</summary>
      <div className="dyn-group-body">
        {sortedArgs(node).map((a) => (
          <DynArgField
            key={a.dest}
            arg={a}
            value={values[a.dest] ?? ""}
            onChange={(v) => onChange(a.dest, v)}
            missing={enforceRequired && a.required && isMissing(a, values)}
            outOfBounds={isOutOfBounds(a, values)}
          />
        ))}
        {sortedChildren(node).map((child) => (
          <DynArgGroupNode
            key={child.path}
            node={child}
            values={values}
            onChange={onChange}
            enforceRequired={enforceRequired}
          />
        ))}
      </div>
    </details>
  );
}

function DynArgField({
  arg,
  value,
  onChange,
  missing,
  outOfBounds,
}: {
  arg: DynamicArg;
  value: string;
  onChange: (v: string) => void;
  missing?: boolean;
  outOfBounds?: boolean;
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
        min={
          (arg.type === "int" || arg.type === "float") && arg.min != null
            ? arg.min
            : undefined
        }
        max={
          (arg.type === "int" || arg.type === "float") && arg.max != null
            ? arg.max
            : undefined
        }
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  const fieldClass =
    "dyn-field" +
    (arg.required ? " dyn-field-required" : "") +
    (missing ? " dyn-field-missing" : "") +
    (outOfBounds ? " dyn-field-missing" : "");

  return (
    <div className={fieldClass} title={buildTitle(arg)}>
      <label>
        <div className="dyn-name">
          <code>{arg.cli_name}</code>
          {arg.required && (
            <span className="dyn-required-tag" title="Required">
              *
            </span>
          )}
          <span className="dyn-type muted">{arg.type}</span>
        </div>
        {widget}
      </label>
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
