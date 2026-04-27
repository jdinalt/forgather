/** localStorage wrapper that namespaces every key by the running
 *  Forgather server's identity.
 *
 *  The browser keys ``localStorage`` by origin (scheme + host + port).
 *  Two different Forgather servers reachable through the same loopback
 *  port — typical for SSH-forwarded multi-host setups — share the same
 *  origin and therefore cross-contaminate any persisted defaults that
 *  name on-disk paths (the MkDocs tool's ``mkdocs.yml`` was the
 *  motivating bug: a path saved while connected to server A surfaced
 *  as the default in a modal connected to server B, where the path
 *  doesn't exist).
 *
 *  Solution: mix the running server's identity (a stable hash of its
 *  ``forgather_repo_root()``) into every persisted key. Different
 *  servers naturally end up with different buckets and can never
 *  collide. The identity is fetched once at boot before React mounts,
 *  see ``main.tsx``.
 *
 *  Trade-off: switching to a new server install drops your
 *  preferences for that server's first session. That's the right
 *  default — preferences from a different install are usually wrong
 *  there anyway (paths point elsewhere, port choices may not apply).
 *  If a user wants to share prefs across multiple Forgather installs,
 *  they can copy the localStorage entries by hand. */

let _identity = "default";

/** Set the namespace used by all subsequent ``persistGet`` /
 *  ``persistSet`` / ``persistRemove`` calls. Called from ``main.tsx``
 *  before React mounts, with the value returned by
 *  ``GET /api/server-identity``. Falls back to ``"default"`` if the
 *  fetch fails — preserves prior single-bucket behaviour as a
 *  conservative fallback. */
export function setStorageNamespace(id: string): void {
  _identity = id || "default";
}

function nsKey(key: string): string {
  return `${_identity}:${key}`;
}

export function persistGet(key: string): string | null {
  try {
    return localStorage.getItem(nsKey(key));
  } catch {
    return null;
  }
}

export function persistSet(key: string, value: string): void {
  try {
    localStorage.setItem(nsKey(key), value);
  } catch {
    // Quota / private-browsing — silently fall back to in-memory
    // state for this session.
  }
}

export function persistRemove(key: string): void {
  try {
    localStorage.removeItem(nsKey(key));
  } catch {
    // see persistSet
  }
}
