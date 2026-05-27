import { useState } from "react";

import { persistGet, persistSet } from "../persist";
import { DirectoryBrowser, BrowseMode } from "./DirectoryBrowser";

interface Props {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  /** "files-and-dirs" picks a file or a directory; "dirs-only" is what
   *  output/log paths typically want. */
  mode?: BrowseMode;
  title?: string;
  /** When true the input stretches to fill the row (wraps on narrow
   *  modals). Useful for full-width path fields like "Output dir". */
  wide?: boolean;
  /** When set, the picker remembers the directory the user last picked
   *  from under this key (namespaced via ``persist``) and reopens to
   *  the parent of that path next time ``value`` is empty. Useful when
   *  the user adds several rows in a row (e.g. multi-model inference
   *  setup) and the next pick is almost always a sibling of the last
   *  one. Omit for one-shot picks where remembering would surprise. */
  rememberKey?: string;
  /** Disable both the text input and the Browse button. The caller
   *  is responsible for displaying any associated dimming / tooltip
   *  on the surrounding label. */
  disabled?: boolean;
}

const REMEMBER_PREFIX = "pathfield.last:";

function parentOf(p: string): string {
  // Strip trailing slashes, drop the last component, restore the
  // leading slash if the result is empty (root).
  const stripped = p.replace(/\/+$/, "");
  const i = stripped.lastIndexOf("/");
  if (i <= 0) return "/";
  return stripped.slice(0, i);
}

/** Text input + "Browse…" button — same widget the DynamicArgsForm uses
 *  for ``type="path"`` dynamic args, lifted into a reusable component
 *  so modals can drop it in for their known-path fields without
 *  re-implementing the picker wiring. */
export function PathField({
  value,
  onChange,
  placeholder,
  mode = "files-and-dirs",
  title,
  wide = false,
  rememberKey,
  disabled = false,
}: Props) {
  const [browsing, setBrowsing] = useState(false);

  // Seed the directory browser:
  //   1. If the field has a value, browse from it (existing behavior).
  //   2. Else if rememberKey is set and we have a stored last-dir,
  //      browse from there — natural for "add another item from the
  //      same parent directory" workflows.
  //   3. Else let DirectoryBrowser fall back to its quick-paths default.
  const remembered =
    rememberKey ? persistGet(REMEMBER_PREFIX + rememberKey) : null;
  const seedPath = value || remembered || undefined;

  return (
    <div className={"path-field" + (wide ? " path-field-wide" : "")}>
      <input
        type="text"
        className={wide ? "wide" : undefined}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      />
      <button
        type="button"
        className="secondary"
        onClick={() => setBrowsing(true)}
        disabled={disabled}
      >
        Browse…
      </button>
      {browsing && (
        <DirectoryBrowser
          initialPath={seedPath}
          mode={mode}
          title={title}
          onCancel={() => setBrowsing(false)}
          onPick={(p) => {
            // Persist the *parent* of the picked path so the next open
            // lands one level up — typical "pick another sibling"
            // workflow (e.g. add a second model from the same
            // models/ directory). Picking the path itself would force
            // the user to click "up" every time.
            if (rememberKey && p) {
              persistSet(REMEMBER_PREFIX + rememberKey, parentOf(p));
            }
            onChange(p);
            setBrowsing(false);
          }}
        />
      )}
    </div>
  );
}
