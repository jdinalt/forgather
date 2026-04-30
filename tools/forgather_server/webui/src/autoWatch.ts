import { persistGet, persistSet } from "./persist";

/** Sticky preference: when checked in any submit modal, the app
 *  switches to the Jobs view and opens the TTY for the just-submitted
 *  job as soon as it transitions out of the queue. The checkbox state
 *  is read from / written to localStorage on every toggle, so the
 *  default the user sees in the next modal matches their last choice. */
const AUTO_WATCH_KEY = "forgather-auto-watch-tty-v1";

export function getAutoWatchTty(): boolean {
  return persistGet(AUTO_WATCH_KEY) === "1";
}

export function setAutoWatchTty(v: boolean): void {
  persistSet(AUTO_WATCH_KEY, v ? "1" : "0");
}
