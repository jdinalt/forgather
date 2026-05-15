import type { QueryClient } from "@tanstack/react-query";

import { api } from "./api";

/** Service type names accepted by the backend ``services:`` config. */
export type ServiceTypeName = "dataset" | "inference" | "tensorboard" | "mkdocs";

/** Shared flow behind every modal's "Create service…" button.
 *
 *  Prompts the operator for a service name, validates it locally to
 *  catch the obvious typos (the backend re-validates), then calls
 *  ``POST /api/services`` with ``enabled=true`` so the new entry
 *  auto-starts immediately (the API also runs an autostart pass). On
 *  success refreshes the cached services list so the sidebar updates
 *  without a page reload.
 *
 *  Returns ``true`` when an entry was written, ``false`` when the
 *  operator cancelled or the prompt was empty. Errors are surfaced via
 *  ``window.alert`` because none of the existing service modals carry
 *  a toast / banner system; failures here are rare (invalid name, file
 *  permissions) and a dialog mirrors the modal's existing UX. */
export async function promptAndCreateService(
  qc: QueryClient,
  type: ServiceTypeName,
  args: Record<string, unknown>,
  suggestedName: string = "",
): Promise<boolean> {
  const raw = window.prompt(
    `Create ${type} service.\nEnter a name (letters, digits, dash, underscore):`,
    suggestedName,
  );
  if (raw == null) return false;
  const name = raw.trim();
  if (!name) return false;
  if (!/^[A-Za-z0-9_-]+$/.test(name)) {
    window.alert(
      `Invalid service name: ${JSON.stringify(name)}. Only letters, digits, dash, and underscore are allowed.`,
    );
    return false;
  }
  try {
    await api.upsertService(type, name, true, args);
  } catch (e) {
    window.alert(
      `Could not create service ${type}:${name}: ${
        e instanceof Error ? e.message : String(e)
      }`,
    );
    return false;
  }
  qc.invalidateQueries({ queryKey: ["services"] });
  return true;
}
