import { useState } from "react";

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
}: Props) {
  const [browsing, setBrowsing] = useState(false);
  return (
    <div className={"path-field" + (wide ? " path-field-wide" : "")}>
      <input
        type="text"
        className={wide ? "wide" : undefined}
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
      {browsing && (
        <DirectoryBrowser
          initialPath={value || undefined}
          mode={mode}
          title={title}
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
