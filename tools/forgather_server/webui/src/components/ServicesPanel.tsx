import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, ServiceStatus } from "../api";
import { ContextMenu } from "./ContextMenu";

/** Compact label shown above each entry — "<type>:<name>". Matches the
 *  service id used everywhere on the backend so operators see the same
 *  string in logs, API responses, and the sidebar. */
function serviceId(s: ServiceStatus): string {
  return `${s.service.type}:${s.service.name}`;
}

/** Sidebar list of configured auto-start services. Each entry is
 *  expandable to reveal its args; the row carries a red/green dot
 *  (running vs. not), a play/stop toggle wired to the enabled flag,
 *  and an X delete button. Polls every few seconds so the dot flips
 *  shortly after the service is actually up.
 *
 *  When ``filterType`` is provided only entries of that service type
 *  are rendered — used to fan out the list under each category's
 *  launcher button (Inference, Dataset, …). Empty omitted filter
 *  shows every configured service.
 *
 *  ``onSwitchView`` is used when a running instance's row is clicked
 *  for a type whose useful action is "go look at the matching view"
 *  rather than "open a URL" (inference / dataset). For mkdocs and
 *  tensorboard we open the spawned server's URL in a new tab
 *  directly. */
export function ServicesPanel({
  filterType,
  onSwitchView,
  onEditService,
}: {
  filterType?: string;
  onSwitchView?: (view: "inference" | "datasets") => void;
  /** Open the matching modal in edit mode, pre-populated from this
   *  service's persisted args. Called from the row's right-click menu
   *  ("Edit…") and from the inline pencil button. */
  onEditService?: (s: ServiceStatus) => void;
}) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["services"],
    queryFn: api.listServices,
    // Poll quickly enough that the red/green dot tracks reality without
    // hammering the file-backed config + queue / job records.
    refetchInterval: 4000,
  });

  const setEnabled = useMutation({
    mutationFn: ({
      type,
      name,
      enabled,
    }: {
      type: string;
      name: string;
      enabled: boolean;
    }) => api.setServiceEnabled(type, name, enabled),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["services"] }),
  });
  const del = useMutation({
    mutationFn: ({ type, name }: { type: string; name: string }) =>
      api.deleteService(type, name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["services"] }),
  });

  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const toggleOpen = (key: string) =>
    setExpanded((s) => ({ ...s, [key]: !s[key] }));

  // Right-click menu state. Keyed off the cursor position so the
  // ContextMenu component can clamp it to the viewport.
  const [menu, setMenu] = useState<
    { x: number; y: number; svc: ServiceStatus } | null
  >(null);

  /** Action triggered by clicking a running service's label. Returns
   *  the human-readable description of what we'll do, or ``null`` when
   *  the row is inert (not running, or a service type with no useful
   *  default action). The caller uses the description as the tooltip
   *  and disables the click when this returns null. */
  function describeActivate(s: ServiceStatus): {
    title: string;
    onClick: () => void;
  } | null {
    if (!s.running) return null;
    const t = s.service.type;
    const args = s.service.args;
    const port = typeof args.port === "number" ? args.port : null;
    const rawHost = typeof args.host === "string" ? args.host : "";
    // Wildcard / empty bind → use the host the browser is already
    // talking to. That's the only address we know is reachable from
    // here, and matches what the JobsPanel does via routable_host but
    // without needing a server-side stamp.
    const isWildcard = !rawHost || rawHost === "0.0.0.0" || rawHost === "::";
    const host = isWildcard
      ? window.location.hostname || "localhost"
      : rawHost;

    if (t === "tensorboard" && port != null) {
      // TensorBoard is launched with --path_prefix /api/tb/<queue_id>
      // by the scheduler so the forgather server's reverse proxy can
      // mount it at /api/tb/{queue_id}/. TB only answers under that
      // prefix — hitting :port/ returns 404. We don't have direct
      // access to ``job.path_prefix`` from the ServiceStatus, but the
      // prefix is deterministic from queue_id, so synthesize it.
      const prefix = s.queue_id ? `/api/tb/${s.queue_id}/` : "/";
      const url = `http://${host}:${port}${prefix}`;
      return {
        title: `Open TensorBoard at ${url} in a new tab`,
        onClick: () => window.open(url, "_blank", "noopener,noreferrer"),
      };
    }
    if (t === "mkdocs" && port != null) {
      const url = `http://${host}:${port}/`;
      return {
        title: `Open MkDocs at ${url} in a new tab`,
        onClick: () => window.open(url, "_blank", "noopener,noreferrer"),
      };
    }
    if (t === "inference" && onSwitchView) {
      return {
        title: "Open the Inference view to chat / complete against this server",
        onClick: () => onSwitchView("inference"),
      };
    }
    if (t === "dataset" && onSwitchView) {
      return {
        title: "Open the Datasets view to browse this server",
        onClick: () => onSwitchView("datasets"),
      };
    }
    return null;
  }

  if (q.isLoading) {
    return <div className="services-panel muted">Loading services…</div>;
  }
  if (q.error) {
    return (
      <div className="services-panel err">
        Failed to load services: {String(q.error)}
      </div>
    );
  }
  const all = q.data ?? [];
  const items = filterType
    ? all.filter((s) => s.service.type === filterType)
    : all;
  if (items.length === 0) {
    return (
      <div className="services-panel muted">
        {filterType
          ? `No ${filterType} services configured. Use “Create service…” to add one.`
          : "No services configured. Open Inference / Dataset / TensorBoard / MkDocs from above and use “Create service…” to add one."}
      </div>
    );
  }

  return (
    <div className="services-panel">
      {menu && onEditService && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
        >
          <div className="context-menu-header muted">
            {serviceId(menu.svc)}
          </div>
          <button
            className="context-menu-item"
            onClick={() => {
              const svc = menu.svc;
              setMenu(null);
              onEditService(svc);
            }}
          >
            Edit…
          </button>
        </ContextMenu>
      )}
      <ul className="services-list">
        {items.map((s) => {
          const key = serviceId(s);
          const isOpen = !!expanded[key];
          const argEntries = Object.entries(s.service.args);
          return (
            <li
              key={key}
              className={
                "service-row" + (s.service.enabled ? " enabled" : " disabled")
              }
              onContextMenu={(e) => {
                if (!onEditService) return;
                e.preventDefault();
                setMenu({ x: e.clientX, y: e.clientY, svc: s });
              }}
            >
              <div className="service-row-head">
                <button
                  className="service-disclosure"
                  onClick={() => toggleOpen(key)}
                  title={isOpen ? "Collapse" : "Expand"}
                  aria-label={isOpen ? "Collapse" : "Expand"}
                >
                  {isOpen ? "▾" : "▸"}
                </button>
                <span
                  className={
                    "service-dot " + (s.running ? "running" : "stopped")
                  }
                  title={
                    s.running
                      ? `Running (${s.status ?? "active"}; queue ${s.queue_id ?? "?"})`
                      : "Not running"
                  }
                />
                {(() => {
                  const action = describeActivate(s);
                  if (!action) {
                    return (
                      <span className="service-id" title={key}>
                        {key}
                      </span>
                    );
                  }
                  return (
                    <button
                      className="service-id service-id-active"
                      title={action.title}
                      onClick={action.onClick}
                    >
                      {key}
                    </button>
                  );
                })()}
                <button
                  className="service-action"
                  onClick={() =>
                    setEnabled.mutate({
                      type: s.service.type,
                      name: s.service.name,
                      enabled: !s.service.enabled,
                    })
                  }
                  disabled={setEnabled.isPending}
                  title={
                    s.service.enabled
                      ? "Disable + stop running instance"
                      : "Enable + start"
                  }
                  aria-label={s.service.enabled ? "Stop" : "Start"}
                >
                  {s.service.enabled ? "⏹" : "▶"}
                </button>
                {onEditService && (
                  <button
                    className="service-action"
                    onClick={() => onEditService(s)}
                    title="Edit args (opens the matching modal pre-filled; if running, stops + restarts to apply)"
                    aria-label="Edit"
                  >
                    ✎
                  </button>
                )}
                <button
                  className="service-action service-delete"
                  onClick={() => {
                    if (
                      window.confirm(
                        `Delete service ${key}? Any running instance will be aborted.`,
                      )
                    ) {
                      del.mutate({
                        type: s.service.type,
                        name: s.service.name,
                      });
                    }
                  }}
                  disabled={del.isPending}
                  title="Delete service entry"
                  aria-label="Delete"
                >
                  ×
                </button>
              </div>
              {isOpen && (
                <div className="service-row-body">
                  {argEntries.length === 0 ? (
                    <span className="muted">No args.</span>
                  ) : (
                    <ul className="service-args">
                      {argEntries.map(([k, v]) => (
                        <li key={k} className="service-arg">
                          <span className="service-arg-key">{k}</span>
                          <span className="service-arg-value">
                            {formatArgValue(v)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                  <div className="service-sig muted">
                    sig: <code>{s.service.signature}</code>
                  </div>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function formatArgValue(v: unknown): string {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
