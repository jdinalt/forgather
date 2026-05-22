/** Read-only demo-mode signal for the webui.
 *
 * The flag lives on the auth-status endpoint so we piggyback on the
 * existing query. Components call ``useDemoMode()`` and use the boolean
 * to disable mutating controls and mask bearer tokens. The backend
 * (auth.py + redact_sensitive_in_demo) is the real security boundary;
 * this hook only drives UX.
 */

import { useQuery } from "@tanstack/react-query";

import { authApi, AuthStatus } from "./auth";

const QUERY_KEY = ["auth-status"] as const;

/** Subscribe to auth status (cached app-wide via React Query). */
export function useAuthStatus(): AuthStatus | undefined {
  // The server returns this at SPA boot already, but components mount
  // before that data flows down; refetching once and caching is simpler
  // than threading status through props. ``staleTime: Infinity`` is
  // fine because both auth_disabled and demo_mode are CLI-time flags
  // — they don't flip during a session.
  const { data } = useQuery({
    queryKey: QUERY_KEY,
    queryFn: authApi.status,
    staleTime: Infinity,
  });
  return data;
}

/** True if the server is running with --demo. Defaults to false until
 *  the status query resolves so we never falsely hide controls. */
export function useDemoMode(): boolean {
  return useAuthStatus()?.demo_mode ?? false;
}

/** Forgather package version, or null until the status query resolves
 *  (so the header chip stays empty rather than flashing "unknown"). */
export function useServerVersion(): string | null {
  const v = useAuthStatus()?.forgather_version;
  if (!v || v === "unknown") return v ?? null;
  return v;
}

/** Convenience: standard disabled / tooltip props for a button that
 *  would trigger a mutation. Spread onto the button:
 *
 *      <button {...demoDisableProps(demo, "...optional override")}>
 *
 *  In non-demo mode returns empty props so the button keeps its
 *  normal behaviour. In demo mode sets ``disabled`` and a tooltip.
 */
export function demoDisableProps(
  demo: boolean,
  reason: string = "Read-only demo mode — mutations are disabled",
): { disabled?: true; title?: string } {
  return demo ? { disabled: true, title: reason } : {};
}
