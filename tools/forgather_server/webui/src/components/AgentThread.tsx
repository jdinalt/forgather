/** Renders the agent conversation: user/assistant text, tool activity,
 *  error notices, and action cards. Shared by the sidebar (compact) and the
 *  full Agent view. */

import { useEffect, useMemo, useRef } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { AgentController } from "../useAgent";
import { AgentActionCard } from "./AgentActionCard";

interface Props {
  agent: AgentController;
  compact?: boolean;
  onOpenFull?: () => void;
  /** Open a doc (absolute filesystem path) in the in-app Docs view. */
  onOpenDoc?: (absPath: string) => void;
  /** Repo root, used to resolve doc references the model emits (relative
   *  paths, /docs/... URL paths, or fabricated origin URLs) to an absolute
   *  path the Docs view understands. */
  repoRoot?: string;
}

function isDocPath(p: string): boolean {
  return /\.(md|markdown|ipynb)$/i.test(p);
}

/** Resolve a link href the agent emitted to an absolute doc path, or null if
 *  it isn't a doc reference. Handles: an absolute fs path under the repo, a
 *  URL-style absolute path (``/docs/x.md``), an http(s) URL (even one the
 *  model fabricated against the server origin), and a repo-relative path. */
function docAbsPath(href: string, repoRoot?: string): string | null {
  if (!href || !repoRoot) return null;
  const root = repoRoot.replace(/\/+$/, "");
  if (href.startsWith("/")) {
    if (href.startsWith(root + "/") && isDocPath(href)) return href; // /mnt/.../docs/x.md
    if (isDocPath(href) || href.includes("/docs/")) return root + href; // /docs/x.md
    return null;
  }
  if (/^https?:\/\//i.test(href)) {
    try {
      const u = new URL(href);
      if (isDocPath(u.pathname) || u.pathname.includes("/docs/")) return root + u.pathname;
    } catch {
      /* not a parseable URL */
    }
    return null;
  }
  if (isDocPath(href)) return root + "/" + href.replace(/^\.?\//, "");
  return null;
}

export function AgentThread({ agent, compact, onOpenFull, onOpenDoc, repoRoot }: Props) {
  const endRef = useRef<HTMLDivElement | null>(null);

  // Custom link handling so a link the agent emits never tears down the SPA:
  // doc references open in the in-app Docs view; everything else opens in a
  // new tab. (Plain markdown <a> would full-navigate the browser — e.g. to a
  // fabricated https://host:port/docs/... URL that 404s and loses the
  // conversation on Back.)
  const mdComponents = useMemo<Components>(
    () => ({
      a({ href, children, ...rest }) {
        if (!href) return <a {...rest}>{children}</a>;
        const doc = docAbsPath(href, repoRoot);
        if (doc && onOpenDoc) {
          return (
            <a
              href={href}
              onClick={(e) => {
                e.preventDefault();
                onOpenDoc(doc);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
            {children}
          </a>
        );
      },
    }),
    [onOpenDoc, repoRoot],
  );

  // Autoscroll to the latest content as the stream grows.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [agent.items]);

  return (
    <div className={"agent-thread" + (compact ? " compact" : "")}>
      {agent.items.length === 0 && (
        <div className="agent-empty muted">
          Ask about a project or config, search the docs, or request a change.
          Proposed changes are shown as a diff for you to approve.
        </div>
      )}
      {agent.items.map((it) => {
        if (it.type === "user") {
          return (
            <div key={it.id} className="agent-msg user">
              {it.text}
            </div>
          );
        }
        if (it.type === "assistant") {
          return (
            <div key={it.id} className="agent-msg assistant">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={mdComponents}
                urlTransform={(url) => url}
              >
                {it.text}
              </ReactMarkdown>
            </div>
          );
        }
        if (it.type === "tool") {
          return (
            <details key={it.id} className={"agent-tool" + (it.isError ? " error" : "")}>
              <summary>
                <span className="agent-tool-name">{it.name}</span>
                {it.isError && <span className="agent-tool-badge">error</span>}
              </summary>
              <pre className="agent-tool-input">{JSON.stringify(it.input, null, 2)}</pre>
              {it.content !== undefined && (
                <pre className="agent-tool-output">{it.content}</pre>
              )}
            </details>
          );
        }
        if (it.type === "action") {
          return (
            <AgentActionCard
              key={it.id}
              card={it.card}
              status={it.status}
              result={it.result}
              compact={compact}
              busy={agent.busy}
              onApprove={() => agent.decide(it.card.action_id, true)}
              onReject={() => agent.decide(it.card.action_id, false)}
              onOpenFull={onOpenFull}
            />
          );
        }
        return (
          <div key={it.id} className="agent-msg error">
            {it.message}
          </div>
        );
      })}
      {agent.busy && <div className="agent-typing muted">…</div>}
      <div ref={endRef} />
    </div>
  );
}
