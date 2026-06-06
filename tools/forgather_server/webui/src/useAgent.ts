/** Shared agent conversation controller.
 *
 *  Called once in App.tsx; the returned controller is handed to BOTH the
 *  right-sidebar panel and the full "Agent" view so they render the same
 *  live conversation (one session, one stream, one set of pending actions).
 *  An action proposed in the sidebar can therefore be approved in the full
 *  view and vice-versa.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ActionCard,
  AgentEvent,
  AgentStatus,
  getAgentStatus,
  streamDecision,
  streamMessage,
} from "./agent-client";

export type AgentItem =
  | { id: string; type: "user"; text: string }
  | { id: string; type: "assistant"; text: string }
  | {
      id: string;
      type: "tool";
      toolUseId: string;
      name: string;
      input: unknown;
      content?: string;
      isError?: boolean;
    }
  | {
      id: string;
      type: "action";
      card: ActionCard;
      status: "pending" | "approved" | "rejected" | "error";
      result?: string;
    }
  | { id: string; type: "error"; message: string };

// Distribute Omit across the union so each variant keeps its own keys
// (a plain ``Omit<AgentItem, "id">`` collapses to the common keys only).
type DistributiveOmit<T, K extends keyof any> = T extends unknown ? Omit<T, K> : never;
type AgentItemInput = DistributiveOmit<AgentItem, "id">;

export interface AgentController {
  status: AgentStatus | null;
  items: AgentItem[];
  busy: boolean;
  awaiting: boolean;
  sessionId: string | null;
  pendingActions: ActionCard[];
  send: (message: string) => void;
  decide: (actionId: string, approve: boolean) => void;
  stop: () => void;
  reset: () => void;
  refreshStatus: () => void;
}

export function useAgent(): AgentController {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [items, setItems] = useState<AgentItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [awaiting, setAwaiting] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const sessionIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const curAsstRef = useRef<string | null>(null);
  const idRef = useRef(0);
  const nextId = () => `it_${idRef.current++}`;

  const refreshStatus = useCallback(() => {
    getAgentStatus()
      .then(setStatus)
      .catch(() => setStatus({ enabled: false, provider: null, model: null, base_url: null }));
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  const addItem = useCallback((partial: AgentItemInput): string => {
    const id = nextId();
    setItems((prev) => [...prev, { id, ...partial } as AgentItem]);
    return id;
  }, []);

  const appendAssistant = useCallback((text: string) => {
    if (!curAsstRef.current) {
      const id = nextId();
      curAsstRef.current = id;
      setItems((prev) => [...prev, { id, type: "assistant", text }]);
    } else {
      const id = curAsstRef.current;
      setItems((prev) =>
        prev.map((it) =>
          it.id === id && it.type === "assistant" ? { ...it, text: it.text + text } : it,
        ),
      );
    }
  }, []);

  const applyEvent = useCallback(
    (ev: AgentEvent) => {
      switch (ev.type) {
        case "session":
          sessionIdRef.current = ev.session_id as string;
          setSessionId(ev.session_id as string);
          break;
        case "text":
          appendAssistant(ev.text as string);
          break;
        case "tool_use":
          curAsstRef.current = null;
          addItem({
            type: "tool",
            toolUseId: ev.id as string,
            name: ev.name as string,
            input: ev.input,
          });
          break;
        case "tool_result": {
          const tid = ev.tool_use_id as string;
          setItems((prev) =>
            prev.map((it) =>
              it.type === "tool" && it.toolUseId === tid
                ? { ...it, content: ev.content as string, isError: !!ev.is_error }
                : it,
            ),
          );
          break;
        }
        case "action_card":
          curAsstRef.current = null;
          addItem({
            type: "action",
            status: "pending",
            card: {
              action_id: ev.action_id as string,
              risk: ev.risk as string,
              title: ev.title as string,
              summary: ev.summary as string,
              path: (ev.path as string) ?? null,
              before: (ev.before as string) ?? null,
              after: (ev.after as string) ?? null,
              pp_preview: (ev.pp_preview as string) ?? null,
              extra: (ev.extra as Record<string, unknown>) ?? {},
            },
          });
          break;
        case "awaiting_approval":
        case "recorded":
          setAwaiting(true);
          break;
        case "action_resolved": {
          const aid = ev.action_id as string;
          const newStatus = ev.error ? "error" : ev.approved ? "approved" : "rejected";
          setItems((prev) =>
            prev.map((it) =>
              it.type === "action" && it.card.action_id === aid
                ? {
                    ...it,
                    status: newStatus,
                    result: (ev.error as string) || (ev.result as string) || undefined,
                  }
                : it,
            ),
          );
          break;
        }
        case "done":
          setAwaiting(false);
          break;
        case "error":
          addItem({ type: "error", message: ev.message as string });
          break;
        // "usage" intentionally ignored for now.
      }
    },
    [addItem, appendAssistant],
  );

  const consume = useCallback(
    async (stream: AsyncIterable<AgentEvent>) => {
      curAsstRef.current = null;
      setBusy(true);
      try {
        for await (const ev of stream) applyEvent(ev);
      } catch (e: any) {
        if (e?.name !== "AbortError") {
          addItem({ type: "error", message: String(e?.message ?? e) });
        }
      } finally {
        setBusy(false);
        abortRef.current = null;
      }
    },
    [addItem, applyEvent],
  );

  const send = useCallback(
    (message: string) => {
      const text = message.trim();
      if (!text || busy) return;
      addItem({ type: "user", text });
      setAwaiting(false);
      const ac = new AbortController();
      abortRef.current = ac;
      void consume(streamMessage(text, sessionIdRef.current, ac.signal));
    },
    [addItem, busy, consume],
  );

  const decide = useCallback(
    (actionId: string, approve: boolean) => {
      if (busy) return;
      const ac = new AbortController();
      abortRef.current = ac;
      void consume(streamDecision(actionId, approve, ac.signal));
    },
    [busy, consume],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    sessionIdRef.current = null;
    curAsstRef.current = null;
    setSessionId(null);
    setItems([]);
    setAwaiting(false);
    setBusy(false);
  }, []);

  const pendingActions = items
    .filter((it): it is Extract<AgentItem, { type: "action" }> => it.type === "action")
    .filter((it) => it.status === "pending")
    .map((it) => it.card);

  return {
    status,
    items,
    busy,
    awaiting,
    sessionId,
    pendingActions,
    send,
    decide,
    stop,
    reset,
    refreshStatus,
  };
}
