import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";

import { api, IpynbCell } from "../api";
import { ContextMenu } from "./ContextMenu";

interface Props {
  /** Absolute path of the doc to display, or null to load the root README. */
  path: string | null;
  /** Called when the panel decides to navigate to a different doc (e.g. user
   *  clicked an internal markdown link). The parent pushes ``path`` onto
   *  the history stack. */
  onNavigate: (path: string) => void;
  /** Switch to the editor view and open ``path``. Wired to the right-click
   *  "Edit" menu item and to source-file links. */
  onEdit: (path: string) => void;
  /** Whether the parent's history stack has anything to pop. Disables the
   *  Back button at the root of the navigation. */
  canGoBack: boolean;
  /** Pop the parent's docs-history stack. */
  onBack: () => void;
}

/** A single entry in the docs outline / TOC. ``level`` is the
 *  heading depth (h1=1, h2=2, h3=3) and is used purely for indent
 *  styling. */
interface TocEntry {
  id: string;
  text: string;
  level: number;
}

interface MenuState {
  x: number;
  y: number;
  path: string;
}

function dirname(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx <= 0 ? "/" : path.slice(0, idx);
}

function joinAndNormalize(base: string, rel: string): string {
  // ``base`` is an absolute directory; ``rel`` is a posix-style relative path.
  // Resolves ".." and "." segments. We don't handle absolute ``rel`` here —
  // the caller filters those out.
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

// Text source files we'd rather open in the editor than stream as a
// download. Markdown / ipynb are routed separately (Docs view), and
// raster/vector image formats keep their asset-URL behaviour so they
// preview inline in a new tab.
const EDITABLE_SUFFIXES = [
  ".yaml", ".yml", ".py", ".jinja", ".j2",
  ".txt", ".json", ".toml", ".cfg", ".ini",
  ".sh", ".env",
];

function isEditableSource(path: string): boolean {
  const lower = path.toLowerCase();
  return EDITABLE_SUFFIXES.some((s) => lower.endsWith(s));
}

// Asset-style binaries the user expects to preview / download in a
// new tab (the asset endpoint serves the bytes with a sensible
// Content-Type). Anything NOT in this set, NOT in the editable set,
// and NOT a doc gets routed through the Docs navigate path — so a
// link pointing at a directory hits the backend's README.md fallback.
const ASSET_SUFFIXES = [
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
  ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz",
  ".mp3", ".mp4", ".webm", ".wav", ".ogg",
];

function isAssetLike(path: string): boolean {
  const lower = path.toLowerCase();
  return ASSET_SUFFIXES.some((s) => lower.endsWith(s));
}

function resolveAbsolute(currentDocPath: string, href: string): string | null {
  // Returns an absolute filesystem path to ``href`` resolved against the
  // doc that contains it, or null if ``href`` can't be turned into one
  // (anchor-only, external URL, etc.).
  if (!href) return null;
  if (href.startsWith("#")) return null;
  if (isExternalUrl(href)) return null;
  // Strip query / fragment — they don't affect filesystem resolution.
  let clean = href;
  const hash = clean.indexOf("#");
  if (hash >= 0) clean = clean.slice(0, hash);
  const q = clean.indexOf("?");
  if (q >= 0) clean = clean.slice(0, q);
  if (clean.startsWith("/")) {
    // Absolute path on disk. Pass through unchanged.
    return clean;
  }
  return joinAndNormalize(dirname(currentDocPath), clean);
}

export function DocsPanel({ path, onNavigate, onEdit, canGoBack, onBack }: Props) {
  // Resolve the default landing page when the panel is opened without a
  // specific path. Once resolved, ``path`` should be set by the parent so
  // future navigations stick.
  const rootQ = useQuery({
    queryKey: ["docs-root"],
    queryFn: api.docsRoot,
    enabled: path === null,
    staleTime: Infinity,
  });

  const effectivePath = path ?? rootQ.data?.path ?? null;

  const fileQ = useQuery({
    queryKey: ["docs-file", effectivePath],
    queryFn: () => api.docsFile(effectivePath as string),
    enabled: !!effectivePath,
    retry: false,
  });

  const [menu, setMenu] = useState<MenuState | null>(null);

  // Outline / TOC. Extracted from the rendered DOM after each doc loads
  // — rehype-slug stamps id="..." on every h1-h6 so we can read both
  // the text and the anchor straight off the markup. Doing it from the
  // DOM rather than parsing the markdown source means we never go out
  // of sync with whatever the markdown engine actually produced (slug
  // normalisation, character escaping, ipynb-cell-by-cell rendering).
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [toc, setToc] = useState<TocEntry[]>([]);
  const refreshToc = useCallback(() => {
    const body = bodyRef.current;
    if (!body) {
      setToc([]);
      return;
    }
    const headings = body.querySelectorAll<HTMLElement>(
      ".docs-pane-content h1, .docs-pane-content h2, .docs-pane-content h3",
    );
    const entries: TocEntry[] = [];
    headings.forEach((h) => {
      const id = h.id;
      if (!id) return;
      const text = (h.textContent || "").trim();
      if (!text) return;
      const level = Number(h.tagName.slice(1)); // h2 -> 2
      entries.push({ id, text, level });
    });
    setToc(entries);
  }, []);

  // Reset scroll to top when switching documents so a fresh doc doesn't
  // inherit the previous one's scroll offset, and rebuild the TOC.
  useEffect(() => {
    const el = document.querySelector(".docs-pane-body");
    if (el) el.scrollTop = 0;
    // Defer to the next frame so react-markdown has actually committed
    // the new headings before we query for them.
    const id = requestAnimationFrame(refreshToc);
    return () => cancelAnimationFrame(id);
  }, [effectivePath, refreshToc]);

  // Rebuild when the rendered doc content itself changes (re-fetch,
  // ipynb cells loaded async, etc.). Lightweight observer over the
  // body subtree.
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const obs = new MutationObserver(() => {
      refreshToc();
    });
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

  // Render the header (with Back button) unconditionally. Loading and
  // error states render INSIDE the body so the user can always recover
  // by going back — without that, a 404 left the panel stuck with no
  // way out short of a page reload.
  const doc = fileQ.data ?? null;
  const headerPath = doc?.path ?? effectivePath ?? "";
  const docDir = doc ? dirname(doc.path) : "";

  let body: React.ReactNode;
  if (path === null && rootQ.isLoading) {
    body = <div className="pane-state">Loading...</div>;
  } else if (path === null && rootQ.data?.path == null) {
    body = (
      <div className="pane-state muted">
        No root README.md found in the Forgather repo.
      </div>
    );
  } else if (!effectivePath) {
    body = null;
  } else if (fileQ.isLoading) {
    body = <div className="pane-state">Loading...</div>;
  } else if (fileQ.isError) {
    const msg = String(fileQ.error);
    body = (
      <div className="pane-state err">
        <pre>{msg}</pre>
      </div>
    );
  } else if (doc) {
    body = (
      <div className="info-pane-content docs-pane-content">
        {doc.kind === "markdown" && (
          <DocMarkdown
            source={doc.content ?? ""}
            docDir={docDir}
            docPath={doc.path}
            onNavigate={onNavigate}
            onEdit={onEdit}
          />
        )}
        {doc.kind === "ipynb" && (
          <IpynbView
            cells={doc.cells ?? []}
            docDir={docDir}
            docPath={doc.path}
            onNavigate={onNavigate}
            onEdit={onEdit}
          />
        )}
      </div>
    );
  }

  return (
    <div
      className="docs-pane"
      onContextMenu={(e) => {
        // Show context menu only when the click target is inside the
        // rendered doc surface — not on the surrounding chrome.
        const target = e.target as HTMLElement;
        if (target.closest(".docs-pane-content") && doc) {
          e.preventDefault();
          setMenu({ x: e.clientX, y: e.clientY, path: doc.path });
        }
      }}
    >
      <div className="docs-pane-header">
        <button
          className="docs-pane-back"
          onClick={onBack}
          disabled={!canGoBack}
          title={canGoBack ? "Back" : "No previous doc"}
          aria-label="Back"
        >
          ←
        </button>
        <span className="docs-pane-path" title={headerPath}>
          {headerPath}
        </span>
      </div>
      <div className="docs-pane-split">
        {toc.length > 1 && (
          <nav className="docs-pane-toc" aria-label="Document outline">
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
        <div className="docs-pane-body info-pane" ref={bodyRef}>
          {body}
        </div>
      </div>
      {menu && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)}>
          <button
            onClick={() => {
              onEdit(menu.path);
              setMenu(null);
            }}
          >
            Edit "{menu.path.split("/").pop()}"
          </button>
        </ContextMenu>
      )}
    </div>
  );
}

interface MarkdownProps {
  source: string;
  docDir: string;
  docPath: string;
  onNavigate: (path: string) => void;
  onEdit: (path: string) => void;
}

function DocMarkdown({ source, docDir, docPath, onNavigate, onEdit }: MarkdownProps) {
  const components = useMemo<Components>(
    () => ({
      img({ src, alt, ...rest }) {
        if (!src) return null;
        if (isExternalUrl(src)) {
          return <img src={src} alt={alt ?? ""} {...rest} />;
        }
        const abs = src.startsWith("/")
          ? src
          : joinAndNormalize(docDir, src);
        return (
          <img src={api.docsAssetUrl(abs)} alt={alt ?? ""} {...rest} />
        );
      },
      a({ href, children, ...rest }) {
        if (!href) {
          return <a {...rest}>{children}</a>;
        }
        if (href.startsWith("#")) {
          // The Docs body is a custom scroll container (not the page),
          // so the browser's default anchor scroll is unreliable here.
          // Resolve the target by id within the doc body and scroll
          // it into view explicitly.
          return (
            <a
              href={href}
              onClick={(e) => {
                const id = decodeURIComponent(href.slice(1));
                if (!id) return;
                const body = document.querySelector(".docs-pane-body");
                const target = body?.querySelector(
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
        if (isExternalUrl(href)) {
          return (
            <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
              {children}
            </a>
          );
        }
        const abs = resolveAbsolute(docPath, href);
        if (!abs) {
          return (
            <a href={href} {...rest}>
              {children}
            </a>
          );
        }
        if (isDocLike(abs)) {
          return (
            <a
              href={api.docsAssetUrl(abs)}
              onClick={(e) => {
                e.preventDefault();
                onNavigate(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
        if (isEditableSource(abs)) {
          // Source / config file — open in the editor instead of
          // streaming as a download. Same gesture as the right-click
          // "Edit" item, just from a doc link.
          return (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                onEdit(abs);
              }}
              {...rest}
            >
              {children}
            </a>
          );
        }
        if (isAssetLike(abs)) {
          // Image / archive / media — let the browser handle it via
          // the asset endpoint (preview or download).
          return (
            <a
              href={api.docsAssetUrl(abs)}
              target="_blank"
              rel="noopener noreferrer"
              {...rest}
            >
              {children}
            </a>
          );
        }
        // No recognized extension — most often a directory link
        // (GitHub-style "look here" reference). Route through the
        // Docs view; the backend's /api/docs/file endpoint resolves
        // a directory to its README.md, or returns a clean 404 we
        // can recover from.
        return (
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault();
              onNavigate(abs);
            }}
            {...rest}
          >
            {children}
          </a>
        );
      },
    }),
    [docDir, docPath, onNavigate, onEdit],
  );

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeSlug]}
      components={components}
    >
      {source}
    </ReactMarkdown>
  );
}

interface IpynbProps {
  cells: IpynbCell[];
  docDir: string;
  docPath: string;
  onNavigate: (path: string) => void;
  onEdit: (path: string) => void;
}

function IpynbView({ cells, docDir, docPath, onNavigate, onEdit }: IpynbProps) {
  return (
    <div className="docs-ipynb">
      {cells.map((cell, i) => (
        <IpynbCellView
          key={i}
          cell={cell}
          docDir={docDir}
          docPath={docPath}
          onNavigate={onNavigate}
          onEdit={onEdit}
        />
      ))}
    </div>
  );
}

interface IpynbCellViewProps {
  cell: IpynbCell;
  docDir: string;
  docPath: string;
  onNavigate: (path: string) => void;
  onEdit: (path: string) => void;
}

function IpynbCellView({
  cell,
  docDir,
  docPath,
  onNavigate,
  onEdit,
}: IpynbCellViewProps) {
  if (cell.cell_type === "markdown") {
    return (
      <div className="docs-ipynb-cell docs-ipynb-md">
        <DocMarkdown
          source={cell.source}
          docDir={docDir}
          docPath={docPath}
          onNavigate={onNavigate}
          onEdit={onEdit}
        />
      </div>
    );
  }
  if (cell.cell_type === "code") {
    return (
      <div className="docs-ipynb-cell docs-ipynb-code">
        <pre className="docs-ipynb-source">
          <code>{cell.source}</code>
        </pre>
        {cell.outputs.length > 0 && (
          <div className="docs-ipynb-outputs">
            {cell.outputs.map((out, i) => (
              <IpynbOutput key={i} output={out} />
            ))}
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="docs-ipynb-cell docs-ipynb-raw">
      <pre>{cell.source}</pre>
    </div>
  );
}

function IpynbOutput({ output }: { output: Record<string, unknown> }) {
  const otype = output.output_type as string;
  if (otype === "stream") {
    const text = (output.text as string) ?? "";
    const name = (output.name as string) ?? "stdout";
    return (
      <pre
        className={
          "docs-ipynb-stream " +
          (name === "stderr" ? "docs-ipynb-stream-err" : "")
        }
      >
        {text}
      </pre>
    );
  }
  if (otype === "error") {
    const traceback = (output.traceback as string[]) ?? [];
    return (
      <pre className="docs-ipynb-stream-err">
        {traceback.join("\n") ||
          `${output.ename ?? "Error"}: ${output.evalue ?? ""}`}
      </pre>
    );
  }
  if (otype === "execute_result" || otype === "display_data") {
    const data = (output.data as Record<string, string>) ?? {};
    if (data["image/png"]) {
      return (
        <img
          className="docs-ipynb-image"
          src={`data:image/png;base64,${data["image/png"]}`}
          alt="output"
        />
      );
    }
    if (data["image/jpeg"]) {
      return (
        <img
          className="docs-ipynb-image"
          src={`data:image/jpeg;base64,${data["image/jpeg"]}`}
          alt="output"
        />
      );
    }
    if (data["image/svg+xml"]) {
      return (
        <div
          className="docs-ipynb-image"
          dangerouslySetInnerHTML={{ __html: data["image/svg+xml"] }}
        />
      );
    }
    if (data["text/html"]) {
      return (
        <div
          className="docs-ipynb-html"
          dangerouslySetInnerHTML={{ __html: data["text/html"] }}
        />
      );
    }
    if (data["text/plain"]) {
      return <pre className="docs-ipynb-stream">{data["text/plain"]}</pre>;
    }
  }
  return null;
}
