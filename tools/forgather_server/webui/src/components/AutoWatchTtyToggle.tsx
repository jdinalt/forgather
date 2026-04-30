import { useState } from "react";

import { getAutoWatchTty, setAutoWatchTty } from "../autoWatch";

/** Compact one-line checkbox shared by every submit modal. Sits inline
 *  on the footer button row alongside Cancel/Submit so it doesn't add
 *  vertical space. Toggling it writes through to localStorage so the
 *  preference is sticky across submissions and modal types. */
export function AutoWatchTtyToggle() {
  const [checked, setChecked] = useState<boolean>(() => getAutoWatchTty());
  return (
    <label
      className="dyn-checkbox auto-watch-tty-toggle"
      title="After submit, switch to the Jobs view and open this job's TTY once it starts running."
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => {
          setChecked(e.target.checked);
          setAutoWatchTty(e.target.checked);
        }}
      />
      <span className="muted">Watch TTY</span>
    </label>
  );
}
