/** Browser client for the Forgather inference server's OpenAI-compatible
 *  endpoints, routed through forgather-server's same-origin proxy
 *  (``/api/inference/*``). This avoids the browser's cross-origin policy
 *  machinery (CORS, Private Network Access, extension blocking) entirely:
 *  the only endpoint the browser hits is the page's own origin; the
 *  forgather-server then forwards to whichever inference server URL the
 *  caller names.
 *
 *  Conventions:
 *  - `baseUrl` ends in `/v1` (matches the CLI client's default of
 *    `http://localhost:8137/v1`); the proxy appends `/models`,
 *    `/completions`, or strips `/v1` for `/health`.
 *  - Generation params are forwarded as-is. Blank / undefined fields are
 *    omitted by the caller so the server applies its per-model defaults.
 */

export interface ModelEntry {
  id: string;
  object?: string;
  owned_by?: string;
  created?: number;
}

export interface ModelsListResponse {
  object: string;
  data: ModelEntry[];
}

/** Generation parameters accepted by `/v1/completions`. Names match the
 *  server's flat body (no `extra_body` wrapper needed — the inference
 *  server accepts HuggingFace-extended fields directly alongside the
 *  OpenAI-named ones). All fields are optional; omit for server default. */
export interface GenerationParams {
  // OpenAI-named
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  echo?: boolean;
  seed?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  // HuggingFace extras — forwarded verbatim to GenerationConfig.
  // Unknown fields are dropped by the server's Pydantic models, so
  // adding a field here also needs wiring in
  // ``tools/inference_server/models/{chat,completion}.py`` and
  // ``tools/inference_server/service.py:build_generation_config``.
  top_k?: number;
  min_p?: number;
  typical_p?: number;
  epsilon_cutoff?: number;
  eta_cutoff?: number;
  repetition_penalty?: number;
  length_penalty?: number;
  no_repeat_ngram_size?: number;
  encoder_no_repeat_ngram_size?: number;
  min_length?: number;
  min_new_tokens?: number;
  num_beams?: number;
  num_beam_groups?: number;
  diversity_penalty?: number;
  guidance_scale?: number;
  penalty_alpha?: number;
  early_stopping?: boolean;
  do_sample?: boolean;
  ignore_eos?: boolean;
}

/** OpenAI-style chat message. The server is stateless: clients must
 *  send the full conversation history (optionally led by a single
 *  ``system`` message) on every request.
 *
 *  ``reasoning`` is a Forgather UI-only field: when a vLLM reasoning
 *  parser is active, the model's pre-answer thinking trace lands here
 *  during streaming so the panel can render it distinctly. It is
 *  stripped before any request is sent to the server (reasoning of a
 *  prior turn is per-turn scratch and should not be replayed). */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  reasoning?: string;
}

/** Strip UI-only fields before sending the conversation to the server.
 *  Currently just ``reasoning``; centralised so future UI-only fields
 *  can't accidentally leak through a forgotten request body. */
function stripUiFields(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}

/** Base URL of the same-origin proxy. Empty string intentional: it
 *  keeps the request relative, so the fetch hits whatever host the
 *  webui was served from without needing to know its port. */
const PROXY_PREFIX = "/api/inference";

function proxyUrl(
  path:
    | "models"
    | "completions"
    | "chat/completions"
    | "tokenize"
    | "health",
  baseUrl: string,
): string {
  return `${PROXY_PREFIX}/${path}?base=${encodeURIComponent(
    trimTrailingSlash(baseUrl),
  )}`;
}

/** Header the proxy reads to pin the upstream bearer token. The proxy
 *  converts this into a standard ``Authorization: Bearer <token>``
 *  header on the upstream request — i.e. what OpenAI / vLLM / etc.
 *  expect — so an OpenAI-compatible server sees nothing non-standard.
 *  We use a dedicated header rather than ``Authorization`` because the
 *  user's Authorization on these requests is the *forgather-server's*
 *  bearer (same-origin) and must not leak to the upstream. */
const TOKEN_HEADER = "X-Inference-Auth-Token";

function authHeaders(authToken?: string): Record<string, string> {
  return authToken ? { [TOKEN_HEADER]: authToken } : {};
}

export async function listModels(
  baseUrl: string,
  authToken?: string,
): Promise<ModelEntry[]> {
  const r = await fetch(proxyUrl("models", baseUrl), {
    headers: authHeaders(authToken),
  });
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  }
  const body = (await r.json()) as ModelsListResponse;
  return Array.isArray(body.data) ? body.data : [];
}

export async function checkHealth(
  baseUrl: string,
  authToken?: string,
): Promise<boolean> {
  const r = await fetch(proxyUrl("health", baseUrl), {
    headers: authHeaders(authToken),
  });
  return r.ok;
}

export type ServerCheckResult =
  | { kind: "ok" }
  | { kind: "auth-failed"; message: string }
  | { kind: "unreachable"; message: string };

/** Probe ``<base>/models`` to verify both reachability and auth.
 *
 *  ``/health`` on the inference server is intentionally open (so the
 *  proxy can probe it before the model finishes loading), which means a
 *  health check can't tell the user whether their token is valid.
 *  ``/models`` is auth-gated and cheap, so it makes a useful "is this
 *  server reachable AND does this token work?" probe. Distinguishes
 *  network/upstream errors (502 from the proxy) from auth rejections
 *  (401/403) so the UI can render a clear hint. */
export async function checkServer(
  baseUrl: string,
  authToken?: string,
): Promise<ServerCheckResult> {
  const r = await fetch(proxyUrl("models", baseUrl), {
    headers: authHeaders(authToken),
  });
  if (r.ok) return { kind: "ok" };
  if (r.status === 401 || r.status === 403) {
    return {
      kind: "auth-failed",
      message: `${r.status} ${r.statusText}`,
    };
  }
  // 502 from the proxy on connect-refused etc.; everything else funnels
  // here too (5xx upstream, 404 wrong path) — treat as "server side broke
  // somehow," distinct from auth.
  let detail = "";
  try {
    detail = await r.text();
  } catch {
    /* ignore */
  }
  return {
    kind: "unreachable",
    message: `${r.status} ${r.statusText}${detail ? `: ${detail}` : ""}`,
  };
}

/** Stream a completion. Yields text deltas as they arrive; the caller
 *  accumulates them however it wants. Raises on HTTP error. Respects
 *  `signal` — Stop button aborts the underlying fetch, which cancels
 *  the stream in-flight.
 *
 *  Wire format is OpenAI SSE:
 *    data: {"choices":[{"text":"foo"}], ...}\n\n
 *    data: [DONE]\n\n
 *  We split on the double-newline boundary to be tolerant of chunks that
 *  contain more than one event. */
/** One-shot completion (``stream: false``). Returns the full text after
 *  the server finishes generating. Needed for generation modes the HF
 *  streamer doesn't support, notably beam search. */
export async function runCompletion(
  baseUrl: string,
  model: string,
  prompt: string,
  params: GenerationParams,
  signal: AbortSignal,
  authToken?: string,
): Promise<string> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    prompt,
    stream: false,
    ...params,
  };
  const r = await fetch(proxyUrl("completions", baseUrl), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(authToken) },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  }
  const data = (await r.json()) as {
    choices?: Array<{ text?: string }>;
  };
  return data.choices?.[0]?.text ?? "";
}

export async function* streamCompletion(
  baseUrl: string,
  model: string,
  prompt: string,
  params: GenerationParams,
  signal: AbortSignal,
  authToken?: string,
): AsyncIterable<string> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    prompt,
    stream: true,
    ...params,
  };
  yield* streamSse<string>(
    proxyUrl("completions", baseUrl),
    body,
    signal,
    (frame) => {
      const t = frame?.choices?.[0]?.text;
      return typeof t === "string" && t.length > 0 ? t : undefined;
    },
    authToken,
  );
}

/** Options that ride alongside generation params on chat-completion
 *  requests but aren't part of the HF GenerationConfig surface.
 *  ``nextRole`` is a Forgather-specific extension on the inference
 *  server: when set to ``"user"``, the chat template renders with
 *  ``continue_final_message=True`` and a trailing empty user turn so
 *  the model generates in the user's voice — the "impersonate"
 *  feature. Default (omitted) is the standard "assistant" turn. */
export interface ChatRequestOptions {
  nextRole?: "assistant" | "user";
}

/** Tagged streaming delta. ``content`` is the final assistant reply;
 *  ``reasoning`` is the model's pre-answer thinking trace (emitted by
 *  vLLM when a reasoning parser is active, e.g. ``--reasoning-parser
 *  qwen3``). Consumers should render the two differently and only
 *  persist ``content`` in conversation history — replaying the
 *  thinking trace on subsequent turns is wasteful and the model
 *  typically discards it anyway. */
export type ChatDelta =
  | { kind: "content"; text: string }
  | { kind: "reasoning"; text: string };

/** Stream a chat completion. Same SSE framing as ``streamCompletion``,
 *  but the per-frame text lives at ``choices[0].delta.content`` (or
 *  ``delta.reasoning`` when a vLLM reasoning parser is active) and the
 *  body uses ``messages`` instead of ``prompt``. The first frame
 *  typically carries ``delta.role: "assistant"`` with no content; we
 *  ignore role-only frames and only yield non-empty content/reasoning
 *  deltas. */
export async function* streamChatCompletion(
  baseUrl: string,
  model: string,
  messages: ChatMessage[],
  params: GenerationParams,
  signal: AbortSignal,
  options?: ChatRequestOptions,
  authToken?: string,
): AsyncIterable<ChatDelta> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    messages: stripUiFields(messages),
    stream: true,
    ...params,
  };
  if (options?.nextRole) body.next_role = options.nextRole;
  yield* streamSse<ChatDelta>(
    proxyUrl("chat/completions", baseUrl),
    body,
    signal,
    (frame) => {
      const delta = frame?.choices?.[0]?.delta;
      if (!delta) return undefined;
      const r = delta.reasoning;
      if (typeof r === "string" && r.length > 0) {
        return { kind: "reasoning", text: r };
      }
      const c = delta.content;
      if (typeof c === "string" && c.length > 0) {
        return { kind: "content", text: c };
      }
      return undefined;
    },
    authToken,
  );
}

/** Per-token scoring result returned by ``scorePrompt``. Shape matches
 *  OpenAI legacy-completions ``choices[0].logprobs`` so the same client
 *  works against vLLM (or any compatible server) using
 *  ``echo=true, logprobs=K, max_tokens=0``.
 *
 *  ``tokens[i]`` is the decoded string for the i-th token. The first
 *  entry of ``token_logprobs`` / ``top_logprobs`` is ``null`` because a
 *  causal LM has no prediction for the first token. Per-token loss is
 *  ``-token_logprobs[i]``; perplexity is ``Math.exp(loss)``. */
export interface TokenScores {
  tokens: string[];
  token_logprobs: (number | null)[];
  top_logprobs: (Record<string, number> | null)[];
  text_offset: number[];
  /** Forgather extension: Shannon entropy (nats) of the full
   *  vocabulary distribution at each prediction position. Aligned
   *  with ``token_logprobs`` — index 0 is ``null``. Absent on OpenAI
   *  / vLLM responses (which only expose top-K logprobs, from which
   *  full-vocab entropy can't be reconstructed). Clients should
   *  treat as optional and fall back when missing. */
  token_entropies?: (number | null)[];
}

/** Score input text by running a single forward pass on the server and
 *  returning per-token logprobs + top-K alternatives. Uses the standard
 *  ``echo=true, logprobs=K, max_tokens=0`` shape — works against vLLM
 *  and our inference server identically. */
export async function scorePrompt(
  baseUrl: string,
  model: string,
  prompt: string,
  topK: number,
  signal: AbortSignal,
  authToken?: string,
): Promise<TokenScores> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    prompt,
    echo: true,
    logprobs: topK,
    max_tokens: 0,
    stream: false,
  };
  const r = await fetch(proxyUrl("completions", baseUrl), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(authToken) },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  }
  const data = (await r.json()) as {
    choices?: Array<{ logprobs?: TokenScores | null }>;
  };
  const lp = data.choices?.[0]?.logprobs;
  if (!lp || !Array.isArray(lp.tokens)) {
    throw new Error("Server did not return logprobs in response");
  }
  return lp;
}

/** vLLM-compatible /tokenize response. ``prompt`` is a Forgather
 *  extension carrying the rendered chat-template string — saves the
 *  caller a detokenize round trip when they want the text. */
export interface TokenizeResponse {
  count: number;
  max_model_len: number;
  tokens: number[];
  token_strs?: string[] | null;
  prompt?: string | null;
}

/** Render a chat conversation to its prompt string via the inference
 *  server's /tokenize endpoint. ``nextRole`` selects the impersonate
 *  path (matches the chat-completion field of the same name). The
 *  rendered text is returned in ``prompt``. */
export async function tokenizeChat(
  baseUrl: string,
  model: string,
  messages: ChatMessage[],
  options?: {
    nextRole?: "assistant" | "user";
    addGenerationPrompt?: boolean;
    continueFinalMessage?: boolean;
  },
  authToken?: string,
): Promise<TokenizeResponse> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    messages: stripUiFields(messages),
  };
  if (options?.nextRole) body.next_role = options.nextRole;
  if (typeof options?.addGenerationPrompt === "boolean") {
    body.add_generation_prompt = options.addGenerationPrompt;
  }
  if (typeof options?.continueFinalMessage === "boolean") {
    body.continue_final_message = options.continueFinalMessage;
  }
  const r = await fetch(proxyUrl("tokenize", baseUrl), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(authToken) },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  }
  return (await r.json()) as TokenizeResponse;
}

/** Result of a one-shot chat completion. ``content`` is the assistant's
 *  final reply (empty string if the model exhausted its token budget
 *  inside the reasoning trace). ``reasoning`` is the pre-answer
 *  thinking text when a vLLM reasoning parser is active, undefined
 *  otherwise. */
export interface ChatResult {
  content: string;
  reasoning?: string;
}

/** One-shot chat completion. Returns the assistant message text plus
 *  an optional reasoning trace. */
export async function runChatCompletion(
  baseUrl: string,
  model: string,
  messages: ChatMessage[],
  params: GenerationParams,
  signal: AbortSignal,
  options?: ChatRequestOptions,
  authToken?: string,
): Promise<ChatResult> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    messages: stripUiFields(messages),
    stream: false,
    ...params,
  };
  if (options?.nextRole) body.next_role = options.nextRole;
  const r = await fetch(proxyUrl("chat/completions", baseUrl), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(authToken) },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  }
  const data = (await r.json()) as {
    choices?: Array<{
      message?: { content?: string | null; reasoning?: string | null };
    }>;
  };
  const msg = data.choices?.[0]?.message;
  return {
    content: msg?.content ?? "",
    reasoning: msg?.reasoning ?? undefined,
  };
}

/** Shared SSE-stream consumer. Posts the JSON body, parses ``data: …``
 *  events terminated by a blank line, and yields whatever the supplied
 *  ``extract`` callback returns from each frame's parsed JSON. Stops on
 *  ``data: [DONE]`` or stream EOF.
 *
 *  Contract: ``extract`` returning ``undefined`` skips the frame; any
 *  other return value is yielded verbatim. Callers that want to drop
 *  empty strings (or other "no-op" values) must filter inside their
 *  own ``extract`` — this consumer does not enforce non-empty yields. */
async function* streamSse<T>(
  url: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
  extract: (frame: any) => T | undefined,
  authToken?: string,
): AsyncIterable<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(authToken) },
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
    // Consume every completed SSE event (terminated by blank line).
    while (true) {
      const sep = buffer.indexOf("\n\n");
      if (sep < 0) break;
      const event = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        if (payload === "[DONE]") return;
        try {
          const parsed = JSON.parse(payload);
          const item = extract(parsed);
          if (item !== undefined) {
            yield item;
          }
        } catch {
          // Ignore unparseable frames — the server never sends malformed
          // JSON on stream, but a truncated mid-chunk is possible if the
          // connection drops. Let the next loop iteration try to recover.
        }
      }
    }
  }
}

function trimTrailingSlash(s: string): string {
  return s.replace(/\/+$/, "");
}
