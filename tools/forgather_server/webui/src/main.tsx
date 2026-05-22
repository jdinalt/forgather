import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import App from "./App";
import {
  AUTH_REQUIRED_EVENT,
  AuthStatus,
  authApi,
  installAuthFetch,
} from "./auth";
import { LoginGate, SetPasswordPrompt } from "./components/LoginGate";
import { setStorageNamespace } from "./persist";
import "./styles.css";

// Install the fetch wrapper before any module-level code makes a request.
installAuthFetch();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, staleTime: 5_000 },
  },
});

type GateState =
  | { kind: "loading" }
  | { kind: "login"; status: AuthStatus }
  | { kind: "set-password" }
  | { kind: "ready" };

function Root() {
  const [state, setState] = useState<GateState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await authApi.status();
        if (cancelled) return;
        if (status.authenticated) {
          // Bootstrap server identity *after* auth so the request
          // doesn't 401 (it would have triggered a forced login).
          await bootstrapIdentity();
          setState({ kind: "ready" });
        } else {
          setState({ kind: "login", status });
        }
      } catch {
        // If the status endpoint itself is unreachable, assume the
        // server is down rather than locking the user out — show the
        // login UI with a generic status so they can retry.
        if (!cancelled) {
          setState({
            kind: "login",
            status: {
              authenticated: false,
              has_password: false,
              auth_disabled: false,
              demo_mode: false,
              forgather_version: "unknown",
            },
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Any 401 from a regular API call (e.g. session expired mid-session)
  // bumps us back to the login gate.
  useEffect(() => {
    function onAuthRequired() {
      authApi.status().then(
        (status) => setState({ kind: "login", status }),
        () =>
          setState({
            kind: "login",
            status: {
              authenticated: false,
              has_password: false,
              auth_disabled: false,
              demo_mode: false,
              forgather_version: "unknown",
            },
          }),
      );
    }
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () =>
      window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, []);

  if (state.kind === "loading") {
    return <div className="login-gate" />;
  }
  if (state.kind === "login") {
    return (
      <LoginGate
        status={state.status}
        onAuthenticated={async ({ promptPasswordSetup }) => {
          await bootstrapIdentity();
          setState(
            promptPasswordSetup ? { kind: "set-password" } : { kind: "ready" },
          );
        }}
      />
    );
  }
  if (state.kind === "set-password") {
    return (
      <SetPasswordPrompt
        onDone={() => setState({ kind: "ready" })}
        onSkip={() => setState({ kind: "ready" })}
      />
    );
  }
  return (
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  );
}

/** Fetch the running server's identity hash and namespace localStorage
 *  on it. Best-effort: a failure leaves us on the ``"default"`` bucket,
 *  matching the prior single-bucket behaviour. */
async function bootstrapIdentity(): Promise<void> {
  try {
    const r = await fetch("/api/server-identity");
    if (r.ok) {
      const body = (await r.json()) as { identity?: string };
      if (body.identity) setStorageNamespace(body.identity);
    }
  } catch {
    // fall through with the "default" namespace
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
