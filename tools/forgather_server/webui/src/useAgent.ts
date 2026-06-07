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
  importConversation,
  streamContinue,
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
        // tool_result content is usually a string, but the canonical form can
        // be a block array ([{type:'text',text:...}]); join its text so the
        // restored tool card matches what was shown live, not raw JSON.
        const raw = block.content;
        const c = typeof raw === "string"
          ? raw
          : Array.isArray(raw)
            ? raw.map((p: any) => (p && typeof p === "object" && "text" in p ? p.text : JSON.stringify(p))).join("")
            : JSON.stringify(raw);
        if (idx !== undefined) {
          const it = items[idx];
          if (it.type === "tool") items[idx] = { ...it, content: c, isError: !!block.is_error };
        }
      }
    }
  }
  return items;
}

export interface AgentUsage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  contextWindow: number | null;
}

/** Cumulative tokens billed across the whole session. An agentic loop re-sends
 *  the prefix on every request, so this sum is what reconciles with the
 *  provider's billing dashboard — typically many times the current context
 *  occupancy (AgentUsage). */
export interface AgentSessionCost {
  inputTokens: number; // fresh (uncached) input, summed over all requests
  cacheReadTokens: number; // prefix served from cache (~0.1x)
  cacheCreationTokens: number; // prefix written to cache (~1.25x)
  outputTokens: number;
  requests: number; // number of API round-trips
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
  /** Latest token accounting (context occupancy), or null until a turn runs. */
  usage: AgentUsage | null;
  /** Cumulative tokens billed across the session (sum over every API request).
   *  Reconciles with the provider billing dashboard; null until a turn runs. */
  sessionCost: AgentSessionCost | null;
  /** Set when the last turn ended truncated (max_tokens / iteration cap), so
   *  the UI can offer "Continue". Cleared when a new turn starts. */
  incompleteReason: string | null;
  /** Set when the agent just *created* a navigable artifact (an approved
   *  workspace / project / config commit). The app watches this to refresh
   *  the Projects tree and reveal the new item. ``nonce`` lets the same path
   *  fire a reveal more than once. */
  lastArtifact: { kind: string; path: string; nonce: number } | null;
  /** Set when the agent asks the UI to reveal a path (the reveal_in_ui tool),
   *  e.g. after locating a project the user asked about. ``where`` is
   *  "projects" or "files". The app routes it to the matching tree. */
  lastReveal: { path: string; where: string; nonce: number } | null;
  send: (message: string) => void;
  decide: (actionId: string, approve: boolean) => void;
  continueTurn: () => void;
  stop: () => void;
  reset: () => void;
  /** reset() guarded by a confirmation when a conversation exists, so a
   *  stray click on the "New conversation" control can't silently discard
   *  the current thread. */
  newConversation: () => void;
  refreshStatus: () => void;
  refreshProfiles: () => void;
  activate: (profileId: string) => void;
  /** Build a JSON-serializable dump of the conversation (incl. the backend
   *  message log) for diagnostics / context restore. */
  dumpConversation: () => Promise<Record<string, unknown>>;
  /** Load a dumped conversation: reseeds the backend session + the UI. */
  loadConversation: (data: Record<string, unknown>) => Promise<void>;
}

export function useAgent(): AgentController {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [items, setItems] = useState<AgentItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [awaiting, setAwaiting] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
  const [usage, setUsage] = useState<AgentUsage | null>(null);
  const [sessionCost, setSessionCost] = useState<AgentSessionCost | null>(null);
  const [incompleteReason, setIncompleteReason] = useState<string | null>(null);
  const [lastArtifact, setLastArtifact] = useState<
    { kind: string; path: string; nonce: number } | null
  >(null);
  const [lastReveal, setLastReveal] = useState<
    { path: string; where: string; nonce: number } | null
  >(null);

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
    let cached: {
      sessionId?: string;
      items?: AgentItem[];
      usage?: AgentUsage;
      sessionCost?: AgentSessionCost;
    } = {};
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
    if (cached.usage) setUsage(cached.usage);
    if (cached.sessionCost) setSessionCost(cached.sessionCost);
    if (!cached.sessionId) return;
    getSession(cached.sessionId)
      .then((h) => {
        if (h.awaiting_approval) {
          setAwaiting(true);
          return; // keep cached items (they include the pending action card)
        }
        const rebuilt = itemsFromMessages(h.messages);
        // Only replace the cached items when the backend log actually
        // reconstructs to something; an empty/non-renderable log (e.g. a
        // session recreated empty after a restart) must not wipe the cache
        // (which the idle persist effect would then make permanent).
        if (rebuilt.length) {
          setItems(rebuilt);
          idRef.current = rebuilt.length;
        }
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
          // A successful create commit: tell the app to refresh the Projects
          // tree and reveal the new workspace / project / config.
          if (ev.approved && !ev.error && ev.created_kind && ev.created_path) {
            setLastArtifact({
              kind: ev.created_kind as string,
              path: ev.created_path as string,
              nonce: Date.now(),
            });
          }
          break;
        }
        case "ui_directive":
          // The agent asked the UI to do something (reveal a path). No
          // conversation item — just surface it for the app to act on.
          if (ev.action === "reveal") {
            const p = (ev.payload as Record<string, unknown>) ?? {};
            if (typeof p.path === "string") {
              setLastReveal({
                path: p.path,
                where: (p.where as string) || "projects",
                nonce: Date.now(),
              });
            }
          }
          break;
        case "usage": {
          const inTok = (ev.input_tokens as number) ?? 0;
          const outTok = (ev.output_tokens as number) ?? 0;
          const cacheRead = (ev.cache_read_input_tokens as number) ?? 0;
          const cacheWrite = (ev.cache_creation_input_tokens as number) ?? 0;
          // Latest request = current context occupancy.
          setUsage({
            inputTokens: inTok,
            outputTokens: outTok,
            cacheReadTokens: cacheRead,
            cacheCreationTokens: cacheWrite,
            contextWindow: (ev.context_window as number) ?? null,
          });
          // Cumulative billed across the session: sum every request, since the
          // loop re-sends the prefix each round-trip.
          setSessionCost((prev) => ({
            inputTokens: (prev?.inputTokens ?? 0) + inTok,
            cacheReadTokens: (prev?.cacheReadTokens ?? 0) + cacheRead,
            cacheCreationTokens: (prev?.cacheCreationTokens ?? 0) + cacheWrite,
            outputTokens: (prev?.outputTokens ?? 0) + outTok,
            requests: (prev?.requests ?? 0) + 1,
          }));
          break;
        }
        case "done":
          setAwaiting(false);
          // A truncated turn (max_tokens / iteration cap) flags Continue.
          setIncompleteReason(ev.incomplete ? ((ev.reason as string) || "incomplete") : null);
          break;
        case "error":
          // A turn can error after awaiting_approval (e.g. the resumed turn
          // hits the iteration cap); clear awaiting so the UI doesn't stay
          // stuck showing "awaiting approval" with no actionable card.
          setAwaiting(false);
          addItem({ type: "error", message: ev.message as string });
          break;
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
      setIncompleteReason(null);
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

  const continueTurn = useCallback(() => {
    if (busy || !sessionIdRef.current) return;
    setIncompleteReason(null);
    const ac = new AbortController();
    abortRef.current = ac;
    void consume(streamContinue(sessionIdRef.current, ac.signal));
  }, [busy, consume]);

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
    setUsage(null);
    setSessionCost(null);
    setIncompleteReason(null);
    try {
      persistRemove(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  const newConversation = useCallback(() => {
    if (
      items.length > 0 &&
      !window.confirm("Start a new conversation? The current one will be cleared.")
    ) {
      return;
    }
    reset();
  }, [items.length, reset]);

  const dumpConversation = useCallback(async (): Promise<Record<string, unknown>> => {
    let messages: Array<{ role: string; content: unknown }> = [];
    const sid = sessionIdRef.current;
    if (sid) {
      try {
        messages = (await getSession(sid)).messages;
      } catch {
        /* backend session gone — export UI items only */
      }
    }
    return {
      version: 1,
      kind: "forgather-agent-conversation",
      exported_at: new Date().toISOString(),
      session_id: sid,
      messages,
      items,
      usage,
      sessionCost,
    };
  }, [items, usage, sessionCost]);

  const loadConversation = useCallback(
    async (data: Record<string, unknown>) => {
      abortRef.current?.abort();
      const messages = Array.isArray(data.messages)
        ? (data.messages as Array<{ role: string; content: unknown }>)
        : [];
      // Don't trust the file's session_id — it names a session on whatever
      // machine produced the dump, not this backend. Only adopt a session id
      // we just minted by reseeding here; otherwise leave it null so the next
      // message starts a fresh backend conversation (display-only restore).
      let sid: string | null = null;
      if (messages.length) {
        try {
          sid = (await importConversation(messages)).session_id;
        } catch {
          /* reseed failed (backend down/rejected) — display-only restore */
        }
      }
      const restored =
        Array.isArray(data.items) && data.items.length
          ? (data.items as AgentItem[])
          : itemsFromMessages(messages);
      sessionIdRef.current = sid;
      setSessionId(sid);
      setItems(restored);
      idRef.current = restored.length;
      setUsage((data.usage as AgentUsage) ?? null);
      // Loading mints a *new* backend session (messages are re-imported), so the
      // cumulative "billed this session" tally starts fresh — restoring the old
      // total would conflate two distinct billing sessions.
      setSessionCost(null);
      setAwaiting(false);
      setIncompleteReason(null);
    },
    [],
  );

  // Persist the conversation so a reload / accidental navigation doesn't lose
  // it. Only write when idle (not mid-stream) to avoid a localStorage write
  // per streamed token; the final state of each turn is captured when busy
  // flips back to false.
  useEffect(() => {
    if (busy) return;
    try {
      if (sessionId || items.length) {
        persistSet(
          STORAGE_KEY,
          JSON.stringify({ sessionId, items, usage, sessionCost }),
        );
      }
    } catch {
      /* ignore quota / serialization errors */
    }
  }, [busy, items, sessionId, usage, sessionCost]);

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
    usage,
    sessionCost,
    incompleteReason,
    lastArtifact,
    lastReveal,
    send,
    decide,
    continueTurn,
    stop,
    reset,
    newConversation,
    refreshStatus,
    refreshProfiles,
    activate,
    dumpConversation,
    loadConversation,
  };
}
