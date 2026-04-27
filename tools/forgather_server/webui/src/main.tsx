import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import App from "./App";
import { setStorageNamespace } from "./persist";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, staleTime: 5_000 },
  },
});

/** Bootstrap: fetch the running server's identity hash and use it to
 *  namespace localStorage before any component mounts. Without this,
 *  ``useState(() => loadPersisted())`` calls inside modals would read
 *  the un-namespaced bucket and inherit values from a different
 *  Forgather server reachable at the same browser origin. See
 *  ``persist.ts`` for the rationale.
 *
 *  The fetch is best-effort: on failure (server down, request
 *  blocked, response malformed) we render with the ``"default"``
 *  fallback namespace, which is the prior single-bucket behaviour.
 *  Better to have a working app than to block on a nice-to-have. */
async function boot() {
  try {
    const r = await fetch("/api/server-identity");
    if (r.ok) {
      const body = (await r.json()) as { identity?: string };
      if (body.identity) setStorageNamespace(body.identity);
    }
  } catch {
    // Fall through with the "default" namespace.
  }
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </React.StrictMode>,
  );
}

void boot();
