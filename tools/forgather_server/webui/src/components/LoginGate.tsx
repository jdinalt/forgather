import { useEffect, useState } from "react";
import { authApi, AuthStatus, consumeUrlToken } from "../auth";

interface Props {
  status: AuthStatus;
  onAuthenticated: (opts: { promptPasswordSetup: boolean }) => void;
}

/** First-screen login. Accepts either the bearer token printed on the
 *  server console or the password the user set on a previous login. If
 *  the URL contains ``?token=...`` (the jupyter-style bootstrap link),
 *  we exchange it for a session automatically and skip the form. */
export function LoginGate({ status, onAuthenticated }: Props) {
  const [token, setToken] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Try the URL-bootstrap path on first paint.
  useEffect(() => {
    const urlToken = consumeUrlToken();
    if (!urlToken) {
      setBusy(false);
      return;
    }
    (async () => {
      try {
        const res = await authApi.loginWithToken(urlToken);
        onAuthenticated({ promptPasswordSetup: res.requires_password_setup });
      } catch (e) {
        setError(`URL token rejected: ${(e as Error).message}`);
        setBusy(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function submitToken(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await authApi.loginWithToken(token.trim());
      onAuthenticated({ promptPasswordSetup: res.requires_password_setup });
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  async function submitPassword(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await authApi.loginWithPassword(password);
      onAuthenticated({ promptPasswordSetup: res.requires_password_setup });
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="login-gate">
      <div className="login-card">
        <h1>Forgather</h1>
        <p className="login-blurb">
          Sign in with the auth token printed on the server console,
          {status.has_password ? " or the password you set previously." : "."}
        </p>

        <form onSubmit={submitToken} className="login-form">
          <label>
            Auth token
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="64-char hex from server console"
              autoFocus
            />
          </label>
          <button type="submit" disabled={busy || !token.trim()}>
            Sign in with token
          </button>
        </form>

        {status.has_password && (
          <form onSubmit={submitPassword} className="login-form">
            <label>
              Password
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>
            <button type="submit" disabled={busy || !password}>
              Sign in with password
            </button>
          </form>
        )}

        {error && <div className="login-error">{error}</div>}

        <p className="login-hint">
          The token lives at <code>~/.config/forgather/server/auth_token</code>.
          Run <code>cat ~/.config/forgather/server/auth_token</code> to print it.
        </p>
      </div>
    </div>
  );
}

interface SetPasswordProps {
  onDone: () => void;
  onSkip: () => void;
}

/** One-shot prompt shown right after a successful token login when the
 *  server reports no password is set yet. The user can skip and keep
 *  using the token, or set a password for future logins. */
export function SetPasswordPrompt({ onDone, onSkip }: SetPasswordProps) {
  const [pw1, setPw1] = useState("");
  const [pw2, setPw2] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (pw1 !== pw2) {
      setError("passwords don't match");
      return;
    }
    if (pw1.length < 4) {
      setError("password must be at least 4 characters");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await authApi.setPassword(pw1);
      onDone();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="login-gate">
      <div className="login-card">
        <h1>Set a password</h1>
        <p className="login-blurb">
          You're signed in. Set a password so you don't have to paste
          the token next time. (You can always sign in with the token
          again — passwords are an optional convenience.)
        </p>
        <form onSubmit={submit} className="login-form">
          <label>
            Password
            <input
              type="password"
              autoComplete="new-password"
              value={pw1}
              onChange={(e) => setPw1(e.target.value)}
              autoFocus
            />
          </label>
          <label>
            Confirm
            <input
              type="password"
              autoComplete="new-password"
              value={pw2}
              onChange={(e) => setPw2(e.target.value)}
            />
          </label>
          <div className="login-actions">
            <button type="button" onClick={onSkip} disabled={busy}>
              Skip
            </button>
            <button type="submit" disabled={busy || !pw1 || !pw2}>
              Save password
            </button>
          </div>
        </form>
        {error && <div className="login-error">{error}</div>}
      </div>
    </div>
  );
}
