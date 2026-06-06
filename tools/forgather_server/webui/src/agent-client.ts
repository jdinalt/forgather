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
  provider: string | null;
  model: string | null;
  base_url: string | null;
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

export async function getAgentStatus(): Promise<AgentStatus> {
  const r = await fetch("/api/agent/status");
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return (await r.json()) as AgentStatus;
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
