import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";

import { api } from "../api";

/** Entry in the outline. Same shape as DocsPanel's. */
interface TocEntry {
  id: string;
  text: string;
  level: number;
}

interface Props {
  project_dir: string;
  enabled: boolean;
  /** When set, links to local .md / .ipynb / README files are intercepted and
   *  routed to the Docs view instead of opening in a new tab. */
  onOpenDoc?: (path: string) => void;
  /** When set, links to source / config files (.yaml, .py, etc.) open in the
   *  editor instead of streaming through the asset endpoint as a download. */
  onEditFile?: (path: string) => void;
}

const EDITABLE_SUFFIXES = [
  ".yaml", ".yml", ".py", ".jinja", ".j2",
  ".txt", ".json", ".toml", ".cfg", ".ini",
  ".sh", ".env",
];

function isEditableSource(path: string): boolean {
  const lower = path.toLowerCase();
  return EDITABLE_SUFFIXES.some((s) => lower.endsWith(s));
}

function isExternalUrl(href: string): boolean {
  return (
    href.startsWith("http://") ||
    href.startsWith("https://") ||
    href.startsWith("mailto:") ||
    href.startsWith("//") ||
    href.startsWith("data:")
  );
}

function isDocLike(path: string): boolean {
  const lower = path.toLowerCase();
  return (
    lower.endsWith(".md") ||
    lower.endsWith(".markdown") ||
    lower.endsWith(".ipynb")
  );
}

function joinAndNormalize(base: string, rel: string): string {
  const parts = (base + "/" + rel).split("/");
  const out: string[] = [];
  for (const p of parts) {
    if (p === "" || p === ".") continue;
    if (p === "..") {
      if (out.length > 0) out.pop();
      continue;
    }
    out.push(p);
  }
  return "/" + out.join("/");
}

function resolveAbsolute(projectDir: string, href: string): string | null {
  if (!href || href.startsWith("#")) return null;
  if (isExternalUrl(href)) return null;
  let clean = href;
  const hash = clean.indexOf("#");
  if (hash >= 0) clean = clean.slice(0, hash);
  const q = clean.indexOf("?");
  if (q >= 0) clean = clean.slice(0, q);
  if (clean.startsWith("/")) return clean;
  return joinAndNormalize(projectDir, clean);
}

export function InfoPane({ project_dir, enabled, onOpenDoc, onEditFile }: Props) {
  const readmeQ = useQuery({
    queryKey: ["project-readme", project_dir],
    queryFn: () => api.projectReadme(project_dir),
    enabled,
    retry: false,
  });

  // Outline / TOC. Same extraction pattern DocsPanel uses: query
  // the rendered DOM after the markdown commits, since rehype-slug
  // has already stamped ids on every h1–h6 we'd want to link to.
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [toc, setToc] = useState<TocEntry[]>([]);
  const refreshToc = useCallback(() => {
    const body = bodyRef.current;
    if (!body) {
      setToc([]);
      return;
    }
    const headings = body.querySelectorAll<HTMLElement>(
      ".info-pane-content h1, .info-pane-content h2, .info-pane-content h3",
    );
    const entries: TocEntry[] = [];
    headings.forEach((h) => {
      const id = h.id;
      if (!id) return;
      const text = (h.textContent || "").trim();
      if (!text) return;
      entries.push({ id, text, level: Number(h.tagName.slice(1)) });
    });
    setToc(entries);
  }, []);
  // Re-extract whenever the rendered project changes (project_dir
  // shift) or the README data refreshes. ``readmeQ.data`` is the
  // markdown source string — when it flips identity react-markdown
  // re-renders and we re-query.
  useEffect(() => {
    const id = requestAnimationFrame(refreshToc);
    return () => cancelAnimationFrame(id);
  }, [project_dir, readmeQ.data, refreshToc]);
  // Also catch DOM mutations inside the body — covers the case
  // where rehype plugins commit asynchronously after the first
  // render (rare here, but defensive and matches DocsPanel).
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const obs = new MutationObserver(() => refreshToc());
    obs.observe(el, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, [refreshToc]);

  const scrollToHeading = useCallback((id: string) => {
    const body = bodyRef.current;
    if (!body) return;
    const target = body.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, []);

  if (!enabled) return null;

  if (readmeQ.isLoading) {
    return <div className="pane-state">Loading...</div>;
  }

  if (readmeQ.isError) {
    const msg = String(readmeQ.error);
    if (msg.includes("404")) {
      return (
        <div className="pane-state muted">This project has no README.md.</div>
      );
    }
    return (
      <div className="pane-state err">
        <pre>{msg}</pre>
      </div>
    );
  }

  const components: Components = {
    img({ src, alt, ...rest }) {
      if (!src) return null;
      const isAbsolute =
        src.startsWith("http://") ||
        src.startsWith("https://") ||
        src.startsWith("//") ||
        src.startsWith("data:");
      const resolvedSrc = isAbsolute
        ? src
        : api.projectAssetUrl(
            project_dir,
            src.startsWith("./") ? src.slice(2) : src,
          );
      return <img src={resolvedSrc} alt={alt ?? ""} {...rest} />;
    },
    a({ href, children, ...rest }) {
      // In-page anchors. Info pane is a custom scroll container, so the
      // browser's default anchor scroll won't necessarily work — resolve
      // the target by id within the pane and scroll it explicitly.
      if (href && href.startsWith("#")) {
        return (
          <a
            href={href}
            onClick={(e) => {
              const id = decodeURIComponent(href.slice(1));
              if (!id) return;
              const pane = document.querySelector(".info-pane");
              const target = pane?.querySelector(
                `#${CSS.escape(id)}`,
              ) as HTMLElement | null;
              if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: "smooth", block: "start" });
              }
            }}
            {...rest}
          >
            {children}
          </a>
        );
      }
      if (href && !isExternalUrl(href)) {
        const abs = resolveAbsolute(project_dir, href);
        if (abs && isDocLike(abs) && onOpenDoc) {
          // Markdown / ipynb → Docs view.
          return (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                onOpenDoc(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
        if (abs && isEditableSource(abs) && onEditFile) {
          // Source / config file → open in the editor.
          return (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                onEditFile(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
        if (abs && onOpenDoc) {
          // Unknown extension — most likely a directory link
          // (GitHub-style "look in this folder"). Route to Docs;
          // the backend resolves a directory to its README.md.
          return (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                onOpenDoc(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
      }
      if (href) {
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
            {children}
          </a>
        );
      }
      return <a {...rest}>{children}</a>;
    },
  };

  // The TOC rides on the same flex shell pattern as DocsPanel's
  // ``.docs-pane-split`` — outline column on the left, scrollable
  // content on the right. Reusing the same ``.docs-pane-toc*``
  // class names keeps the styling consistent across the two
  // surfaces (one set of CSS for both outlines).
  //
  // Single inner content wrapper inside ``.info-pane``: that's
  // where max-width + centering live. Everything inside flows at
  // the wrapper's full width so headers, paragraphs, hr, lists,
  // tables all share the same left/right edges (instead of each
  // centering itself independently with its own max-width, which
  // produced the inconsistent-spacing rendering).
  return (
    <div className="info-pane-split">
      {toc.length > 1 && (
        <nav className="docs-pane-toc" aria-label="README outline">
          <div className="docs-pane-toc-title">On this page</div>
          <ul>
            {toc.map((e, i) => (
              <li
                key={`${e.id}-${i}`}
                className={`docs-toc-l${Math.min(e.level, 3)}`}
              >
                <button
                  type="button"
                  onClick={() => scrollToHeading(e.id)}
                  title={e.text}
                >
                  {e.text}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      )}
      <div className="info-pane" ref={bodyRef}>
        <div className="info-pane-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeSlug]}
            components={components}
          >
            {readmeQ.data ?? ""}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
