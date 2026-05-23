import type { QueryClient } from "@tanstack/react-query";

import { api } from "./api";

/** Service type names accepted by the backend ``services:`` config. */
export type ServiceTypeName = "dataset" | "inference" | "tensorboard" | "mkdocs";

/** Coerce an arbitrary string into something the backend's name
 *  validator (``[A-Za-z0-9_-]+``) accepts. Anything that isn't a
 *  letter, digit, dash, or underscore becomes a dash; runs of dashes
 *  are collapsed; leading / trailing dashes are trimmed. Empty result
 *  → empty string (the prompt falls back to a blank default). */
export function sanitizeServiceName(raw: string): string {
  return raw
    .replace(/[^A-Za-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Low-level "write these args to the services config" wrapper used by
 *  both the create-and-prompt flow and the edit flow.
 *
 *  Posts to ``/api/services`` (upsert by ``<type, name>``), then
 *  invalidates the cached services list so the sidebar / panel update.
 *  Failures surface via ``window.alert`` because the modals currently
 *  carry no toast / banner system; errors here are rare (invalid
 *  payload, file-permissions). Returns ``true`` on success. */
export async function saveServiceArgs(
  qc: QueryClient,
  type: ServiceTypeName,
  name: string,
  enabled: boolean,
  args: Record<string, unknown>,
): Promise<boolean> {
  try {
    await api.upsertService(type, name, enabled, args);
  } catch (e) {
    window.alert(
      `Could not save service ${type}:${name}: ${
        e instanceof Error ? e.message : String(e)
      }`,
    );
    return false;
  }
  qc.invalidateQueries({ queryKey: ["services"] });
  return true;
}

/** Edit-mode save: when the service is currently running we have to
 *  stop the old instance before writing the new args. Otherwise the
 *  signature change (args → sha256) means the autostart pass treats
 *  the new entry as a brand-new service and enqueues a second instance
 *  while the original keeps running — the second one then fails to
 *  bind whatever port the first is holding.
 *
 *  Steps:
 *    1. If wasRunning: ``setEnabled(false)`` (backend aborts the live
 *       instance) and poll the services list until it's actually gone
 *       (bounded — a wedged process shouldn't lock the modal forever).
 *    2. ``upsertService`` with new args + the previously-effective
 *       ``enabled`` flag. When that flag is true the backend's autostart
 *       pass brings the service back up immediately. */
export async function saveServiceArgsAndMaybeRestart(
  qc: QueryClient,
  type: ServiceTypeName,
  name: string,
  wasRunning: boolean,
  enabled: boolean,
  args: Record<string, unknown>,
): Promise<boolean> {
  try {
    if (wasRunning) {
      await api.setServiceEnabled(type, name, false);
      qc.invalidateQueries({ queryKey: ["services"] });
      // Poll for the old instance to drain. The disable call kicks off
      // an abort but the process exit is asynchronous; without waiting,
      // step 2 would re-enable before the port is free.
      const deadline = Date.now() + 15_000;
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const statuses = await api.listServices();
        const cur = statuses.find(
          (s) => s.service.type === type && s.service.name === name,
        );
        if (!cur || (!cur.running && cur.status == null)) break;
        if (Date.now() > deadline) {
          // Best-effort: warn but proceed. The upsert will still write
          // the new args; autostart will get to it once the old
          // instance finishes dying.
          window.alert(
            `Stopping the running ${type}:${name} took longer than expected. ` +
              `Saving the new settings anyway; you may need to wait a moment ` +
              `for the new instance to come up.`,
          );
          break;
        }
        await new Promise((r) => setTimeout(r, 400));
      }
    }
    await api.upsertService(type, name, enabled, args);
  } catch (e) {
    window.alert(
      `Could not save service ${type}:${name}: ${
        e instanceof Error ? e.message : String(e)
      }`,
    );
    return false;
  }
  qc.invalidateQueries({ queryKey: ["services"] });
  return true;
}

/** Shared flow behind every modal's "Create service…" button.
 *
 *  Prompts the operator for a service name, validates it locally to
 *  catch the obvious typos (the backend re-validates), then calls
 *  ``saveServiceArgs`` with ``enabled=true`` so the new entry
 *  auto-starts immediately.
 *
 *  Returns ``true`` when an entry was written, ``false`` when the
 *  operator cancelled or the prompt was empty. */
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
  return saveServiceArgs(qc, type, name, true, args);
}
