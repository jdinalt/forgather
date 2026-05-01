/** Auth helpers for the webui.
 *
 * Two pieces:
 *
 * 1. ``installAuthFetch()`` monkey-patches ``window.fetch`` so every
 *    request automatically carries cookies and any 401 response fires a
 *    DOM event the shell listens for. This avoids touching the ~90
 *    existing ``fetch(...)`` call sites scattered through ``api.ts``.
 *
 * 2. The ``auth`` object wraps the ``/api/auth/*`` endpoints so the
 *    login gate can call them without going through the ``api`` module
 *    (which we want to keep dependency-free of auth state).
 */

export const AUTH_REQUIRED_EVENT = "forgather:auth-required";

export interface AuthStatus {
  authenticated: boolean;
  has_password: boolean;
  auth_disabled: boolean;
}

let _installed = false;

/** Wrap window.fetch so we can:
 *
 *  - send cookies on every request (needed when the SPA is served
 *    from a different origin than the API, e.g. a Vite dev server
 *    proxy that bypasses cookies by default);
 *  - notify the shell on 401 so it can swap in the login gate
 *    instead of letting individual queries surface raw errors.
 */
export function installAuthFetch(): void {
  if (_installed) return;
  _installed = true;
  const original = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const merged: RequestInit = { credentials: "include", ...(init ?? {}) };
    // Don't override an explicit ``credentials`` choice from the caller.
    if (init?.credentials === undefined) merged.credentials = "include";
    const r = await original(input, merged);
    if (r.status === 401) {
      // Don't fire for the auth endpoints themselves — login attempts
      // legitimately produce 401, and the LoginGate handles them
      // inline. Nothing else should be calling /api/auth/login.
      const url = typeof input === "string" ? input : input.toString();
      if (!url.includes("/api/auth/")) {
        window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT));
      }
    }
    return r;
  };
}

async function jsonOrThrow(r: Response): Promise<unknown> {
  if (!r.ok) {
    let detail: string = r.statusText;
    try {
      const body = await r.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      // keep statusText
    }
    throw new Error(detail);
  }
  return r.json();
}

export const authApi = {
  status: async (): Promise<AuthStatus> => {
    const r = await fetch("/api/auth/status");
    return (await jsonOrThrow(r)) as AuthStatus;
  },
  loginWithToken: async (token: string) => {
    const r = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    return (await jsonOrThrow(r)) as {
      ok: boolean;
      requires_password_setup: boolean;
    };
  },
  loginWithPassword: async (password: string) => {
    const r = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    return (await jsonOrThrow(r)) as {
      ok: boolean;
      requires_password_setup: boolean;
    };
  },
  setPassword: async (password: string) => {
    const r = await fetch("/api/auth/set-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    await jsonOrThrow(r);
  },
  logout: async () => {
    await fetch("/api/auth/logout", { method: "POST" });
  },
};

/** Pull a ``?token=...`` out of the current URL and strip it from the
 *  address bar. Returns the token (if any) so the caller can exchange
 *  it for a session cookie. We strip the query so the token doesn't
 *  end up in the user's history or shared screenshots. */
export function consumeUrlToken(): string | null {
  const params = new URLSearchParams(window.location.search);
  const tok = params.get("token");
  if (!tok) return null;
  params.delete("token");
  const qs = params.toString();
  const newUrl =
    window.location.pathname +
    (qs ? `?${qs}` : "") +
    window.location.hash;
  window.history.replaceState({}, "", newUrl);
  return tok;
}
