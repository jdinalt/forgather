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
  AgentProfile,
  AgentStatus,
  activateProfile,
  getAgentStatus,
  getProfiles,
  getSession,
  streamDecision,
  streamMessage,
} from "./agent-client";
import { persistGet, persistRemove, persistSet } from "./persist";

const STORAGE_KEY = "forgather-agent-conversation";

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

/** Rebuild the renderable item list from the backend's canonical
 *  content-block message log (used to rehydrate a conversation on load).
 *  Resolved tool calls (incl. approved/rejected authoring tools) render as
 *  tool items; the in-flight pending turn isn't in the log, so action cards
 *  aren't reconstructed here (a still-awaiting conversation keeps its cached
 *  items instead — see the load effect). */
function itemsFromMessages(messages: Array<{ role: string; content: unknown }>): AgentItem[] {
  const items: AgentItem[] = [];
  const toolIdx = new Map<string, number>();
  let n = 0;
  const nid = () => `r_${n++}`;
  for (const m of messages) {
    const role = m.role === "user" ? "user" : "assistant";
    const content = m.content;
    if (typeof content === "string") {
      if (content) items.push({ id: nid(), type: role, text: content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    for (const block of content as any[]) {
      if (!block || typeof block !== "object") continue;
      if (block.type === "text") {
        if (block.text) items.push({ id: nid(), type: role, text: String(block.text) });
      } else if (block.type === "tool_use") {
        items.push({ id: nid(), type: "tool", toolUseId: String(block.id), name: String(block.name), input: block.input });
        toolIdx.set(String(block.id), items.length - 1);
      } else if (block.type === "tool_result") {
        const idx = toolIdx.get(String(block.tool_use_id));
        const c = typeof block.content === "string" ? block.content : JSON.stringify(block.content);
        if (idx !== undefined) {
          const it = items[idx];
          if (it.type === "tool") items[idx] = { ...it, content: c, isError: !!block.is_error };
        }
      }
    }
  }
  return items;
}

export interface AgentController {
  status: AgentStatus | null;
  items: AgentItem[];
  busy: boolean;
  awaiting: boolean;
  sessionId: string | null;
  pendingActions: ActionCard[];
  profiles: AgentProfile[];
  activeProfileId: string | null;
  send: (message: string) => void;
  decide: (actionId: string, approve: boolean) => void;
  stop: () => void;
  reset: () => void;
  refreshStatus: () => void;
  refreshProfiles: () => void;
  activate: (profileId: string) => void;
}

export function useAgent(): AgentController {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [items, setItems] = useState<AgentItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [awaiting, setAwaiting] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);

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

  const refreshProfiles = useCallback(() => {
    getProfiles()
      .then((r) => {
        setProfiles(r.profiles);
        setActiveProfileId(r.active_id);
      })
      .catch(() => {
        setProfiles([]);
        setActiveProfileId(null);
      });
  }, []);

  const activate = useCallback(
    (profileId: string) => {
      activateProfile(profileId)
        .then(() => {
          refreshStatus();
          refreshProfiles();
        })
        .catch(() => {});
    },
    [refreshProfiles, refreshStatus],
  );

  useEffect(() => {
    refreshStatus();
    refreshProfiles();
  }, [refreshStatus, refreshProfiles]);

  // Restore the conversation on load: show the cached items immediately,
  // then rehydrate from the backend session log (authoritative — also picks
  // up turns made in another tab). If the conversation is still awaiting
  // approval, keep the cached items: the in-flight turn (and its action
  // card) isn't in the backend message log yet. If the backend session is
  // gone (e.g. server restart), keep the cached items for display.
  useEffect(() => {
    let raw: string | null = null;
    try {
      raw = persistGet(STORAGE_KEY);
    } catch {
      raw = null;
    }
    if (!raw) return;
    let cached: { sessionId?: string; items?: AgentItem[] } = {};
    try {
      cached = JSON.parse(raw);
    } catch {
      return;
    }
    if (cached.sessionId) {
      sessionIdRef.current = cached.sessionId;
      setSessionId(cached.sessionId);
    }
    if (Array.isArray(cached.items) && cached.items.length) {
      setItems(cached.items);
      idRef.current = cached.items.length; // continue ids past restored ones
    }
    if (!cached.sessionId) return;
    getSession(cached.sessionId)
      .then((h) => {
        if (h.awaiting_approval) {
          setAwaiting(true);
          return; // keep cached items (they include the pending action card)
        }
        const rebuilt = itemsFromMessages(h.messages);
        setItems(rebuilt);
        idRef.current = rebuilt.length;
      })
      .catch(() => {
        /* backend session gone — keep the cached items for display */
      });
    // Run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
          // A turn can error after awaiting_approval (e.g. the resumed turn
          // hits the iteration cap); clear awaiting so the UI doesn't stay
          // stuck showing "awaiting approval" with no actionable card.
          setAwaiting(false);
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
    try {
      persistRemove(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  // Persist the conversation so a reload / accidental navigation doesn't lose
  // it. Only write when idle (not mid-stream) to avoid a localStorage write
  // per streamed token; the final state of each turn is captured when busy
  // flips back to false.
  useEffect(() => {
    if (busy) return;
    try {
      if (sessionId || items.length) {
        persistSet(STORAGE_KEY, JSON.stringify({ sessionId, items }));
      }
    } catch {
      /* ignore quota / serialization errors */
    }
  }, [busy, items, sessionId]);

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
    profiles,
    activeProfileId,
    send,
    decide,
    stop,
    reset,
    refreshStatus,
    refreshProfiles,
    activate,
  };
}
