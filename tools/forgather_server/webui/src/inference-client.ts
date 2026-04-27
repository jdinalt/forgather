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
 *  ``system`` message) on every request. */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/** Base URL of the same-origin proxy. Empty string intentional: it
 *  keeps the request relative, so the fetch hits whatever host the
 *  webui was served from without needing to know its port. */
const PROXY_PREFIX = "/api/inference";

function proxyUrl(
  path: "models" | "completions" | "chat/completions" | "health",
  baseUrl: string,
): string {
  return `${PROXY_PREFIX}/${path}?base=${encodeURIComponent(
    trimTrailingSlash(baseUrl),
  )}`;
}

export async function listModels(baseUrl: string): Promise<ModelEntry[]> {
  const r = await fetch(proxyUrl("models", baseUrl));
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  }
  const body = (await r.json()) as ModelsListResponse;
  return Array.isArray(body.data) ? body.data : [];
}

export async function checkHealth(baseUrl: string): Promise<boolean> {
  const r = await fetch(proxyUrl("health", baseUrl));
  return r.ok;
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
): Promise<string> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    prompt,
    stream: false,
    ...params,
  };
  const r = await fetch(proxyUrl("completions", baseUrl), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
): AsyncIterable<string> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    prompt,
    stream: true,
    ...params,
  };
  yield* streamSse(
    proxyUrl("completions", baseUrl),
    body,
    signal,
    (frame) => frame?.choices?.[0]?.text,
  );
}

/** Stream a chat completion. Same SSE framing as ``streamCompletion``,
 *  but the per-frame text lives at ``choices[0].delta.content`` and the
 *  body uses ``messages`` instead of ``prompt``. The first frame
 *  typically carries ``delta.role: "assistant"`` with no content; we
 *  ignore role-only frames and only yield content deltas. */
export async function* streamChatCompletion(
  baseUrl: string,
  model: string,
  messages: ChatMessage[],
  params: GenerationParams,
  signal: AbortSignal,
): AsyncIterable<string> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    messages,
    stream: true,
    ...params,
  };
  yield* streamSse(
    proxyUrl("chat/completions", baseUrl),
    body,
    signal,
    (frame) => frame?.choices?.[0]?.delta?.content,
  );
}

/** One-shot chat completion. Returns the assistant message text. */
export async function runChatCompletion(
  baseUrl: string,
  model: string,
  messages: ChatMessage[],
  params: GenerationParams,
  signal: AbortSignal,
): Promise<string> {
  const body: Record<string, unknown> = {
    model: model || "inference-server",
    messages,
    stream: false,
    ...params,
  };
  const r = await fetch(proxyUrl("chat/completions", baseUrl), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  }
  const data = (await r.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  return data.choices?.[0]?.message?.content ?? "";
}

/** Shared SSE-stream consumer. Posts the JSON body, parses ``data: …``
 *  events terminated by a blank line, and yields whatever the supplied
 *  ``extract`` callback returns from each frame's parsed JSON. Stops on
 *  ``data: [DONE]`` or stream EOF. */
async function* streamSse(
  url: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
  extract: (frame: any) => string | undefined,
): AsyncIterable<string> {
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
          const text = extract(parsed);
          if (typeof text === "string" && text.length > 0) {
            yield text;
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
