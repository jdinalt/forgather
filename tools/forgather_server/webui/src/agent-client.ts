/** Agent API client.
 *
 *  Streaming uses ``fetch`` + ``ReadableStream`` (the same shape as
 *  ``inference-client.ts``'s ``streamSse``), NOT ``EventSource`` — a plain
 *  fetch carries the forgather session cookie via the ``installAuthFetch``
 *  wrapper, so no query-string token is needed. Each SSE frame is one agent
 *  event dict from the backend loop; we parse and yield them verbatim.
 */

export interface AgentStatus {
  enabled: boolean;
  active_id?: string | null;
  label?: string | null;
  provider?: string | null;
  model?: string | null;
  base_url?: string | null;
  verify_tls?: boolean;
  has_imported_cert?: boolean;
}

/** A saved connection profile (credentials redacted to flags). */
export interface AgentProfile {
  id: string;
  label: string;
  provider: string;
  model: string;
  base_url: string;
  api_key_env: string;
  verify_tls: boolean;
  has_api_key: boolean;
  has_imported_cert: boolean;
  max_tokens: number;
  max_iterations: number;
}

/** Fields accepted when creating/updating a profile. Omitted fields are
 *  left unchanged; an empty string for api_key/ca_cert_pem clears it. */
export interface AgentProfileWrite {
  label?: string;
  provider?: string;
  model?: string;
  base_url?: string;
  api_key?: string;
  api_key_env?: string;
  verify_tls?: boolean;
  ca_cert_pem?: string;
  max_tokens?: number;
  max_iterations?: number;
}

export interface CertInfo {
  host: string;
  port: number;
  pem: string;
  sha256: string;
}

export interface ActionCard {
  action_id: string;
  risk: string;
  title: string;
  summary: string;
  path: string | null;
  before: string | null;
  after: string | null;
  pp_preview: string | null;
  extra: Record<string, unknown>;
}

/** One streamed event. ``type`` discriminates; other fields vary by type:
 *  session | text | tool_use | tool_result | action_card | awaiting_approval |
 *  action_resolved | recorded | usage | done | error. Kept loose on purpose —
 *  the hook narrows per type. */
export interface AgentEvent {
  type: string;
  [k: string]: unknown;
}

async function jsonReq<T>(url: string, method: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export async function getAgentStatus(): Promise<AgentStatus> {
  return jsonReq<AgentStatus>("/api/agent/status", "GET");
}

export async function getProfiles(): Promise<{ active_id: string | null; profiles: AgentProfile[] }> {
  return jsonReq("/api/agent/profiles", "GET");
}

export async function createProfile(body: AgentProfileWrite): Promise<AgentProfile> {
  return jsonReq("/api/agent/profiles", "POST", body);
}

export async function updateProfile(id: string, body: AgentProfileWrite): Promise<AgentProfile> {
  return jsonReq(`/api/agent/profiles/${id}`, "PUT", body);
}

export async function deleteProfile(id: string): Promise<{ removed: string; active_id: string | null }> {
  return jsonReq(`/api/agent/profiles/${id}`, "DELETE");
}

export async function activateProfile(id: string): Promise<{ active_id: string }> {
  return jsonReq(`/api/agent/profiles/${id}/activate`, "POST");
}

export interface ModelsQuery {
  profile_id?: string;
  provider?: string;
  base_url?: string;
  api_key?: string;
  api_key_env?: string;
  verify_tls?: boolean;
  ca_cert_pem?: string;
}

export interface ModelInfo {
  id: string;
  /** Server-reported context window (vLLM); null when not reported (Claude). */
  max_model_len: number | null;
}

export async function listAgentModels(q: ModelsQuery): Promise<ModelInfo[]> {
  const r = await jsonReq<{ models: ModelInfo[] }>("/api/agent/models", "POST", q);
  return r.models;
}

export async function fetchServerCert(base_url: string): Promise<CertInfo> {
  return jsonReq("/api/agent/fetch-cert", "POST", { base_url });
}

export interface SessionHistory {
  session_id: string;
  /** Canonical content-block messages (role + content[]); rebuilt into UI
   *  items on the client. */
  messages: Array<{ role: string; content: unknown }>;
  awaiting_approval: boolean;
  created_at: number;
  updated_at: number;
}

export async function getSession(id: string): Promise<SessionHistory> {
  return jsonReq(`/api/agent/sessions/${encodeURIComponent(id)}`, "GET");
}

async function* streamAgent(
  url: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
): AsyncIterable<AgentEvent> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    const detail = r.body ? await r.text() : "(no body)";
    throw new Error(`${r.status} ${r.statusText}: ${detail}`);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    while (true) {
      const sep = buffer.indexOf("\n\n");
      if (sep < 0) break;
      const event = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          yield JSON.parse(payload) as AgentEvent;
        } catch {
          // Ignore a truncated mid-chunk frame; the next read recovers.
        }
      }
    }
  }
}

export function streamMessage(
  message: string,
  sessionId: string | null,
  signal: AbortSignal,
): AsyncIterable<AgentEvent> {
  return streamAgent(
    "/api/agent/message",
    { message, session_id: sessionId },
    signal,
  );
}

export function streamDecision(
  actionId: string,
  approve: boolean,
  signal: AbortSignal,
): AsyncIterable<AgentEvent> {
  const url = approve ? "/api/agent/approve" : "/api/agent/reject";
  return streamAgent(url, { action_id: actionId }, signal);
}
