import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, ServiceStatus } from "../api";

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
 *  shows every configured service. */
export function ServicesPanel({ filterType }: { filterType?: string }) {
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
                <span className="service-id" title={key}>
                  {key}
                </span>
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
