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
  /** Server is in read-only demo mode. Webui hides destructive
   *  controls and masks bearer-token fields. The backend rejects
   *  mutating requests with 403 regardless, so this flag is a UX
   *  signal rather than a security boundary. */
  demo_mode: boolean;
  /** Forgather package version (e.g. "0.1.0"). "unknown" when the
   *  server is an editable install without dist-info. */
  forgather_version: string;
}

let _installed = false;

/** Wrap window.fetch so we can:
 *
 *  - send cookies on every request (needed when the SPA is served
 *    from a different origin than the API, e.g. a Vite dev server
 *    proxy that bypasses cookies by default);
 *  - notify the shell on 401 so it can swap in the login gate
 *    instead of letting individual queries surface raw errors.
 *
 *  Reauth policy: fire ``AUTH_REQUIRED_EVENT`` on 401 by default.
 *  401 is HTTP's "session needs to authenticate" status and the only
 *  one the forgather AuthMiddleware emits when a session cookie is
 *  missing / expired / invalid (see ``auth.py``'s middleware path).
 *
 *  403 is "authenticated but not allowed" and in this codebase always
 *  carries an operational meaning — SSRF allowlist miss, fs-root
 *  policy refusal, demo-mode mutation block, upstream proxy refusal.
 *  None of those are recoverable by re-logging-in, so the default is
 *  to *not* fire reauth on 403. A backend that genuinely needs to
 *  force a re-login on a 403 can opt in with the response header
 *  ``X-Forgather-Reauth-Required: 1`` (no current route uses it; the
 *  hook exists for a future step-up-auth flow).
 *
 *  ``X-Upstream-Auth-Failed`` continues to suppress reauth on 401s
 *  from proxied upstreams (e.g. wrong inference bearer): the
 *  forgather session is fine, only the upstream rejected its own
 *  token.
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
    const reauthOptIn = r.headers.get("X-Forgather-Reauth-Required") === "1";
    const shouldReauth =
      r.status === 401 || (r.status === 403 && reauthOptIn);
    if (shouldReauth) {
      // Don't fire for the auth endpoints themselves — login attempts
      // legitimately produce 401, and the LoginGate handles them
      // inline. Nothing else should be calling /api/auth/login.
      const url = typeof input === "string" ? input : input.toString();
      // Suppress when an upstream proxy tagged the response: a 401
      // from an inference / dataset_server upstream means the upstream
      // rejected its own bearer, not that our session expired. The
      // middleware would have intercepted before any proxy code ran
      // if the session were truly expired.
      const upstreamAuthFailed =
        r.headers.get("X-Upstream-Auth-Failed") === "1";
      if (!url.includes("/api/auth/") && !upstreamAuthFailed) {
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
