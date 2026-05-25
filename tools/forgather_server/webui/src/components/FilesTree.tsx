import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  FsEntry,
  FsListing,
  ProjectInfo,
  SearchRoot,
  WorkspaceCluster,
} from "../api";
import { ContextMenu } from "./ContextMenu";
import { InitWorkspaceModal } from "./InitWorkspaceModal";
import { NewProjectModal } from "./NewProjectModal";
import { NewTemplateModal } from "./NewTemplateModal";

interface Props {
  /** Open the file in the App-level editor view (which switches the
   *  main pane to "edit"). */
  onOpenFile: (path: string) => void;
  /** Drop ``path`` from the editor (every split, no dirty prompt) —
   *  invoked after a rename / move / delete makes the path stale. */
  onDropPath: (path: string) => void;
  /** Open the file in the Docs view. Surfaced as a context-menu item
   *  for ``.md`` / ``.markdown`` / ``.ipynb`` files. */
  onOpenDoc?: (path: string) => void;
  /** External "reveal this path" request from elsewhere in the app
   *  (typically the ProjectTree's right-click → "Reveal in Files"
   *  entry). The ``nonce`` lets the same path re-reveal — equality
   *  on the path alone wouldn't fire useEffect a second time. */
  revealRequest?: { path: string; nonce: number } | null;
}

/** Centralised tree state shared with every node. Selection drives the
 *  highlight; force-opened paths are the ancestor chain a reveal
 *  request expands so the target row is mounted and can be scrolled
 *  into view. */
export interface RevealState {
  selectedPath: string | null;
  forceOpen: Set<string>;
  /** Called when the user manually collapses a directory — clears the
   *  path from ``forceOpen`` so a subsequent reveal up-tree doesn't
   *  silently snap it back open. */
  onUnforce: (path: string) => void;
}

function isDocLike(path: string): boolean {
  const lower = path.toLowerCase();
  return (
    lower.endsWith(".md") ||
    lower.endsWith(".markdown") ||
    lower.endsWith(".ipynb")
  );
}

interface MenuTarget {
  x: number;
  y: number;
  path: string;
  parent: string; // dest_dir candidate for Paste-into-this-dir
  isDir: boolean;
  /** True only for the synthetic top-level row that *is* a search
   *  root. Used to tweak the menu (e.g. forbid Rename / Cut on roots). */
  isRoot: boolean;
}

interface Clipboard {
  path: string;
  isDir: boolean;
  mode: "cut" | "copy";
}

const basename = (p: string) => {
  const norm = p.replace(/\/+$/, "");
  const slash = norm.lastIndexOf("/");
  return slash >= 0 ? norm.slice(slash + 1) : norm;
};

/** Find the deepest ancestor of ``path`` (or ``path`` itself) that is a
 *  configured search root. Used by the New Workspace flow so a click on
 *  a subdirectory still resolves to the matching root for the
 *  ``parent_dir`` constraint. */
function enclosingSearchRoot(
  path: string,
  roots: SearchRoot[] | undefined,
): SearchRoot | null {
  if (!roots) return null;
  const norm = path.replace(/\/+$/, "");
  const matches = roots.filter((r) => {
    if (!r.exists) return false;
    const rp = r.path.replace(/\/+$/, "");
    return norm === rp || norm.startsWith(rp + "/");
  });
  matches.sort((a, b) => b.path.length - a.path.length);
  return matches[0] ?? null;
}

function enclosingWorkspace(
  path: string,
  clusters: WorkspaceCluster[] | undefined,
): WorkspaceCluster | null {
  if (!clusters) return null;
  const norm = path.replace(/\/+$/, "");
  const matches = clusters.filter((c) => {
    if (!c.workspace_root) return false;
    const wp = c.workspace_root.replace(/\/+$/, "");
    return norm === wp || norm.startsWith(wp + "/");
  });
  matches.sort((a, b) => b.workspace_root.length - a.workspace_root.length);
  return matches[0] ?? null;
}

/** Find the deepest enclosing project (or the project itself) for a
 *  clicked directory. Used by the New Config / New Template flow so a
 *  right-click inside a project's templates subtree opens the modal
 *  pre-targeted at the right project. */
function enclosingProject(
  path: string,
  clusters: WorkspaceCluster[] | undefined,
): ProjectInfo | null {
  if (!clusters) return null;
  const norm = path.replace(/\/+$/, "");
  let best: ProjectInfo | null = null;
  let bestLen = -1;
  for (const c of clusters) {
    for (const p of c.projects) {
      const pp = p.project_dir.replace(/\/+$/, "");
      if (norm === pp || norm.startsWith(pp + "/")) {
        if (pp.length > bestLen) {
          best = p;
          bestLen = pp.length;
        }
      }
    }
  }
  return best;
}

/** Compute relative path from ``base`` to ``path`` (assumes path is at
 *  or under base). Returns empty string when they're equal. */
function relPath(base: string, path: string): string {
  const b = base.replace(/\/+$/, "");
  const p = path.replace(/\/+$/, "");
  if (p === b) return "";
  return p.slice(b.length + 1);
}

export function FilesTree({
  onOpenFile,
  onDropPath,
  onOpenDoc,
  revealRequest,
}: Props) {
  const [showHidden, setShowHidden] = useState(false);
  const [menu, setMenu] = useState<MenuTarget | null>(null);
  const [clipboard, setClipboard] = useState<Clipboard | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  // Force-open set — paths whose ``<details>`` should be open regardless
  // of the local toggle state, populated by reveal-request expansion of
  // the target's ancestor chain.
  const [forceOpen, setForceOpen] = useState<Set<string>>(() => new Set());
  const containerRef = useRef<HTMLDivElement>(null);
  // Modal state for New Workspace / New Project pop-ups initiated from
  // the directory context menu.
  // The clicked directory becomes the workspace dir directly; there's
  // no parent / dir-name to pre-fill, so the state is just the path.
  const [initWorkspaceFor, setInitWorkspaceFor] = useState<string | null>(
    null,
  );
  const [newProjectFor, setNewProjectFor] = useState<{
    workspace: WorkspaceCluster;
    initialDirName: string;
  } | null>(null);
  const [newTemplateFor, setNewTemplateFor] = useState<{
    project: ProjectInfo;
    kind: "config" | "template";
    initialDirHint: string;
  } | null>(null);

  const rootsQ = useQuery({
    queryKey: ["search-roots"],
    queryFn: api.listSearchRoots,
  });

  const unforce = (p: string) => {
    setForceOpen((prev) => {
      if (!prev.has(p)) return prev;
      const next = new Set(prev);
      next.delete(p);
      return next;
    });
  };

  const reveal: RevealState = useMemo(
    () => ({ selectedPath, forceOpen, onUnforce: unforce }),
    [selectedPath, forceOpen],
  );

  // Apply incoming reveal requests: compute the ancestor chain up to
  // the deepest matching search root and force every step open, then
  // mark the target as selected. The scroll-into-view useEffect below
  // picks the row up once the listings (re-)render.
  //
  // Wait for ``rootsQ.data`` to land before deciding the target isn't
  // under any search root — otherwise an early reveal-request firing
  // before the query resolves will see ``roots = []`` and pop a
  // misleading "not under any search root" alert. The effect re-runs
  // when rootsQ.data flips from undefined to the loaded array.
  useEffect(() => {
    if (!revealRequest) return;
    if (rootsQ.data === undefined) return;
    const roots = rootsQ.data;
    const norm = revealRequest.path.replace(/\/+$/, "");
    let root: SearchRoot | null = null;
    let bestLen = -1;
    for (const r of roots) {
      if (!r.exists) continue;
      const rp = r.path.replace(/\/+$/, "");
      if ((norm === rp || norm.startsWith(rp + "/")) && rp.length > bestLen) {
        root = r;
        bestLen = rp.length;
      }
    }
    if (!root) {
      // Target isn't under any registered search root. Don't try to be
      // clever — surface the failure so the user knows why nothing
      // visibly happened.
      alert(
        `Cannot reveal:\n${revealRequest.path}\n\n` +
          `The path isn't under any configured search root.`,
      );
      return;
    }
    const rootPath = root.path.replace(/\/+$/, "");
    // Walk from the root toward the target, collecting every ancestor
    // directory that needs to be expanded. The target itself is the
    // selection — we don't force it open (a file has nothing to open;
    // a directory revealed at its own row stays as the user left it).
    const ancestors = new Set<string>([rootPath]);
    if (norm !== rootPath) {
      const rel = norm.slice(rootPath.length + 1).split("/");
      let cur = rootPath;
      // Skip the last segment — that's the target itself.
      for (let i = 0; i < rel.length - 1; i++) {
        cur = `${cur}/${rel[i]}`;
        ancestors.add(cur);
      }
    }
    setForceOpen((prev) => {
      const next = new Set(prev);
      for (const a of ancestors) next.add(a);
      return next;
    });
    setSelectedPath(norm);
  }, [revealRequest, rootsQ.data]);

  // Scroll the selected row into view once it (and any newly-expanded
  // ancestors) finish rendering. Listings load async per directory, so
  // we retry a few times before giving up — typical first try succeeds
  // when ancestors were already open, otherwise the second after the
  // listing query lands picks it up.
  //
  // The cleanup cancels any pending retry so unmounted / re-selected
  // components don't keep firing noop callbacks. ``cancelled`` is the
  // belt; ``clearTimeout`` is the suspenders — together they ensure
  // the loop stops both inside the callback and at the scheduler.
  useEffect(() => {
    if (!selectedPath || !containerRef.current) return;
    const container = containerRef.current;
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    const tryScroll = () => {
      if (cancelled) return;
      const el = container.querySelector(
        `[data-tree-path="${CSS.escape(selectedPath)}"]`,
      );
      if (el) {
        (el as HTMLElement).scrollIntoView({
          block: "nearest",
          behavior: "smooth",
        });
        return;
      }
      attempts += 1;
      // ~3 seconds total at 150ms cadence — long enough to cover a
      // cold listing fetch, short enough that a vanished path doesn't
      // hang a callback forever.
      if (attempts >= 20) return;
      timeoutId = setTimeout(tryScroll, 150);
    };
    // requestAnimationFrame so the first attempt runs after the
    // ancestor expansions have had a chance to mount.
    requestAnimationFrame(tryScroll);
    return () => {
      cancelled = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };
  }, [selectedPath, revealRequest]);
  // Workspaces drive New-Project enablement: we need the enclosing
  // workspace for the clicked directory to call POST /workspace/new-project.
  const projectsQ = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });

  if (rootsQ.isLoading) {
    return <div className="muted pad">Loading…</div>;
  }
  if (rootsQ.error) {
    return (
      <div className="err pad">
        <pre>{String(rootsQ.error)}</pre>
      </div>
    );
  }
  const roots = rootsQ.data ?? [];

  return (
    <div className="files-tree" ref={containerRef}>
      <label className="files-tree-show-hidden">
        <input
          type="checkbox"
          checked={showHidden}
          onChange={(e) => setShowHidden(e.target.checked)}
        />
        Show hidden
      </label>
      {roots.length === 0 && (
        <div className="muted pad">
          No search roots. Add one in the Projects → Search Roots section.
        </div>
      )}
      {roots.map((r) => (
        <RootNode
          key={r.path}
          path={r.path}
          exists={r.exists}
          showHidden={showHidden}
          clipboard={clipboard}
          setClipboard={setClipboard}
          openMenu={(t) => setMenu(t)}
          onOpenFile={onOpenFile}
          reveal={reveal}
        />
      ))}
      {menu && (
        <FilesContextMenu
          target={menu}
          clipboard={clipboard}
          enclosingRoot={enclosingSearchRoot(menu.path, rootsQ.data)}
          enclosingWs={enclosingWorkspace(menu.path, projectsQ.data)}
          enclosingProj={enclosingProject(
            menu.path,
            projectsQ.data,
          )}
          onClose={() => setMenu(null)}
          onOpenFile={onOpenFile}
          onOpenDoc={onOpenDoc}
          onDropPath={onDropPath}
          setClipboard={setClipboard}
          openInitWorkspace={(path) => setInitWorkspaceFor(path)}
          openNewProject={(p) => setNewProjectFor(p)}
          openNewTemplate={(p) => setNewTemplateFor(p)}
        />
      )}
      {initWorkspaceFor && (
        <InitWorkspaceModal
          workspaceDir={initWorkspaceFor}
          onCreated={() => {
            // ["projects"] is invalidated by the modal's mutation; the
            // tree refetches and shows the new forgather_workspace/
            // entry inside the directory.
          }}
          onClose={() => setInitWorkspaceFor(null)}
        />
      )}
      {newProjectFor && (
        <NewProjectModal
          workspace={newProjectFor.workspace}
          initialProjectDirName={newProjectFor.initialDirName}
          onCreated={() => {
            // Same — modal invalidates ["projects"].
          }}
          onClose={() => setNewProjectFor(null)}
        />
      )}
      {newTemplateFor && (
        <NewTemplateModal
          project={newTemplateFor.project}
          kind={newTemplateFor.kind}
          initialDirHint={newTemplateFor.initialDirHint}
          onCreated={(path) => onOpenFile(path)}
          onClose={() => setNewTemplateFor(null)}
        />
      )}
    </div>
  );
}

interface RootNodeProps {
  path: string;
  exists: boolean;
  showHidden: boolean;
  clipboard: Clipboard | null;
  setClipboard: (c: Clipboard | null) => void;
  openMenu: (t: MenuTarget) => void;
  onOpenFile: (path: string) => void;
  reveal: RevealState;
}

function RootNode({
  path,
  exists,
  showHidden,
  clipboard,
  setClipboard,
  openMenu,
  onOpenFile,
  reveal,
}: RootNodeProps) {
  const label = basename(path) || path;
  // Default-closed so we don't fetch every search root's top-level
  // listing before the user actually expands one. The DirChildren
  // mount is also gated below so the listing fetch only fires when
  // the node is actually visible.
  const [open, setOpen] = useState(false);
  const norm = path.replace(/\/+$/, "");
  const effectiveOpen = open || reveal.forceOpen.has(norm);
  const isSelected = reveal.selectedPath === norm;
  return (
    <details
      className="files-tree-root"
      open={effectiveOpen}
      onToggle={(e) => {
        if (e.target !== e.currentTarget) return;
        const nextOpen = (e.currentTarget as HTMLDetailsElement).open;
        setOpen(nextOpen);
        // If the user collapsed a force-opened root, drop it from the
        // force set so a future reveal upstream doesn't snap it open
        // again without explicit ancestor expansion.
        if (!nextOpen) reveal.onUnforce(norm);
      }}
    >
      <summary
        className={`${exists ? "" : "missing"}${isSelected ? " selected" : ""}`}
        data-tree-path={norm}
        title={path}
        onContextMenu={(e) => {
          if (!exists) return;
          e.preventDefault();
          e.stopPropagation();
          openMenu({
            x: e.clientX,
            y: e.clientY,
            path,
            parent: path, // root: paste-here means paste into the root itself
            isDir: true,
            isRoot: true,
          });
        }}
      >
        <span className="files-tree-root-label">{label}</span>
        {!exists && <span className="err-badge">missing</span>}
      </summary>
      {exists && effectiveOpen && (
        <DirChildren
          path={path}
          showHidden={showHidden}
          clipboard={clipboard}
          setClipboard={setClipboard}
          openMenu={openMenu}
          onOpenFile={onOpenFile}
          depth={0}
          reveal={reveal}
        />
      )}
    </details>
  );
}

interface DirChildrenProps {
  path: string;
  showHidden: boolean;
  clipboard: Clipboard | null;
  setClipboard: (c: Clipboard | null) => void;
  openMenu: (t: MenuTarget) => void;
  onOpenFile: (path: string) => void;
  depth: number;
  reveal: RevealState;
}

function DirChildren({
  path,
  showHidden,
  clipboard,
  setClipboard,
  openMenu,
  onOpenFile,
  depth,
  reveal,
}: DirChildrenProps) {
  const listingQ = useQuery({
    queryKey: ["fs-browse", path, showHidden, true],
    queryFn: () => api.fsBrowse(path, showHidden, true),
    // Modest stale time so refresh after rename/copy/move (which
    // invalidates) still feels snappy.
    staleTime: 30_000,
  });
  if (listingQ.isLoading) {
    return <div className="muted files-tree-status">Loading…</div>;
  }
  if (listingQ.error) {
    return (
      <div className="err files-tree-status">
        <pre>{String(listingQ.error)}</pre>
      </div>
    );
  }
  const data: FsListing | undefined = listingQ.data;
  if (!data || data.entries.length === 0) {
    return <div className="muted files-tree-status">(empty)</div>;
  }
  return (
    <ul className="files-tree-list">
      {data.entries.map((e) => (
        <FsNode
          key={e.path}
          entry={e}
          parent={path}
          showHidden={showHidden}
          clipboard={clipboard}
          setClipboard={setClipboard}
          openMenu={openMenu}
          onOpenFile={onOpenFile}
          depth={depth + 1}
          reveal={reveal}
        />
      ))}
    </ul>
  );
}

interface FsNodeProps {
  entry: FsEntry;
  parent: string;
  showHidden: boolean;
  clipboard: Clipboard | null;
  setClipboard: (c: Clipboard | null) => void;
  openMenu: (t: MenuTarget) => void;
  onOpenFile: (path: string) => void;
  depth: number;
  reveal: RevealState;
}

function FsNode({
  entry,
  parent,
  showHidden,
  clipboard,
  setClipboard,
  openMenu,
  onOpenFile,
  depth,
  reveal,
}: FsNodeProps) {
  const indent = { paddingLeft: `${Math.min(depth, 8) * 8 + 4}px` };
  if (entry.is_dir) {
    return (
      <DirNode
        entry={entry}
        parent={parent}
        showHidden={showHidden}
        clipboard={clipboard}
        setClipboard={setClipboard}
        openMenu={openMenu}
        onOpenFile={onOpenFile}
        depth={depth}
        indent={indent}
        reveal={reveal}
      />
    );
  }
  const norm = entry.path.replace(/\/+$/, "");
  const isSelected = reveal.selectedPath === norm;
  // Every file is click-to-open; the backend's text/binary check
  // refuses truly binary files with 415 and the editor surfaces the
  // error in-tab. Known extensions get language-specific highlighting
  // via languageFor; unknown ones fall back to plaintext.
  return (
    <li className="files-tree-file">
      <button
        type="button"
        className={`files-tree-file-btn${isSelected ? " selected" : ""}`}
        data-tree-path={norm}
        style={indent}
        onClick={() => onOpenFile(entry.path)}
        onContextMenu={(e) => {
          e.preventDefault();
          e.stopPropagation();
          openMenu({
            x: e.clientX,
            y: e.clientY,
            path: entry.path,
            parent,
            isDir: false,
            isRoot: false,
          });
        }}
        title={entry.path}
      >
        <span className="files-tree-name">{entry.name}</span>
      </button>
    </li>
  );
}

/** Directory variant of FsNode, broken out so we can give it React state
 *  for its own expanded/collapsed flag. The DirChildren mount is gated
 *  on this state — without that gate, the whole tree's listings would
 *  fetch eagerly because `<details>` keeps its content in the DOM
 *  regardless of open state. */
function DirNode({
  entry,
  parent,
  showHidden,
  clipboard,
  setClipboard,
  openMenu,
  onOpenFile,
  depth,
  indent,
  reveal,
}: FsNodeProps & { indent: React.CSSProperties }) {
  const [open, setOpen] = useState(false);
  const norm = entry.path.replace(/\/+$/, "");
  const effectiveOpen = open || reveal.forceOpen.has(norm);
  const isSelected = reveal.selectedPath === norm;
  return (
    <li className="files-tree-dir">
      <details
        open={effectiveOpen}
        onToggle={(e) => {
          if (e.target !== e.currentTarget) return;
          const nextOpen = (e.currentTarget as HTMLDetailsElement).open;
          setOpen(nextOpen);
          if (!nextOpen) reveal.onUnforce(norm);
        }}
      >
        <summary
          className={isSelected ? "selected" : undefined}
          data-tree-path={norm}
          style={indent}
          onContextMenu={(e) => {
            e.preventDefault();
            e.stopPropagation();
            openMenu({
              x: e.clientX,
              y: e.clientY,
              path: entry.path,
              parent,
              isDir: true,
              isRoot: false,
            });
          }}
        >
          <span className="files-tree-name">{entry.name}/</span>
        </summary>
        {effectiveOpen && (
          <DirChildren
            path={entry.path}
            showHidden={showHidden}
            clipboard={clipboard}
            setClipboard={setClipboard}
            openMenu={openMenu}
            onOpenFile={onOpenFile}
            depth={depth}
            reveal={reveal}
          />
        )}
      </details>
    </li>
  );
}

interface MenuProps {
  target: MenuTarget;
  clipboard: Clipboard | null;
  /** Search root that contains ``target.path`` (or ``target.path``
   *  itself if it IS a root). When non-null, the menu offers
   *  "New Workspace…" and pre-fills the modal accordingly. */
  enclosingRoot: SearchRoot | null;
  /** Workspace cluster that contains ``target.path`` (or matches it).
   *  When non-null, the menu offers "New Project…". */
  enclosingWs: WorkspaceCluster | null;
  /** Project that contains ``target.path`` (or matches it). When
   *  non-null, the menu offers "New Config…" / "New Template…",
   *  with the clicked directory passed through as a placement hint. */
  enclosingProj: ProjectInfo | null;
  onClose: () => void;
  onOpenFile: (path: string) => void;
  onOpenDoc?: (path: string) => void;
  onDropPath: (path: string) => void;
  setClipboard: (c: Clipboard | null) => void;
  /** Open the InitWorkspaceModal for the clicked directory — the path
   *  IS the workspace dir, no need to ask the user where it goes. */
  openInitWorkspace: (workspaceDir: string) => void;
  openNewProject: (p: {
    workspace: WorkspaceCluster;
    initialDirName: string;
  }) => void;
  openNewTemplate: (p: {
    project: ProjectInfo;
    kind: "config" | "template";
    initialDirHint: string;
  }) => void;
}

function FilesContextMenu({
  target,
  clipboard,
  enclosingRoot,
  enclosingWs,
  enclosingProj,
  onClose,
  onOpenFile,
  onOpenDoc,
  onDropPath,
  setClipboard,
  openInitWorkspace,
  openNewProject,
  openNewTemplate,
}: MenuProps) {
  const qc = useQueryClient();

  const invalidateAfter = (...affected: string[]) => {
    // Invalidate browse cache for every directory the operation
    // touched so both source and destination panes refresh.
    for (const p of affected) {
      qc.invalidateQueries({
        queryKey: ["fs-browse", p],
        // partial match on the path component — staleTime 30s would
        // otherwise hold the old listing.
        exact: false,
      });
    }
  };

  const doRename = async () => {
    onClose();
    const cur = basename(target.path);
    const next = window.prompt(`Rename:\n${target.path}\n\nNew name:`, cur);
    if (next == null) return;
    const trimmed = next.trim();
    if (!trimmed || trimmed === cur) return;
    try {
      await api.fsRename(target.path, trimmed);
      // Stale path: drop any open tab pointing at it.
      onDropPath(target.path);
      invalidateAfter(target.parent);
    } catch (e) {
      alert(`Rename failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const doDelete = async () => {
    onClose();
    const kind = target.isDir ? "directory (recursive)" : "file";
    if (
      !confirm(
        `Delete this ${kind} permanently?\n\n${target.path}\n\nThis cannot be undone.`,
      )
    ) {
      return;
    }
    try {
      if (target.isDir) {
        await api.deleteDir(target.path);
      } else {
        await api.deleteFile(target.path);
      }
      onDropPath(target.path);
      invalidateAfter(target.parent);
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const doCut = () => {
    onClose();
    setClipboard({ path: target.path, isDir: target.isDir, mode: "cut" });
  };

  const doCopy = () => {
    onClose();
    setClipboard({ path: target.path, isDir: target.isDir, mode: "copy" });
  };

  const doNewFile = async () => {
    onClose();
    const name = window.prompt(
      `New file under:\n${target.path}\n\nName (e.g. notes.md):`,
      "",
    );
    if (name == null) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      const r = await api.fsNewFile(target.path, trimmed);
      invalidateAfter(target.path);
      onOpenFile(r.path);
    } catch (e) {
      alert(`Create failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const doNewFolder = async () => {
    onClose();
    const name = window.prompt(
      `New folder under:\n${target.path}\n\nName:`,
      "",
    );
    if (name == null) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      await api.fsMkdir(target.path, trimmed);
      invalidateAfter(target.path);
    } catch (e) {
      alert(`Create failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const doPaste = async () => {
    onClose();
    if (!clipboard) return;
    try {
      if (clipboard.mode === "cut") {
        await api.fsMove(clipboard.path, target.path);
        // The moved path no longer exists at its old location.
        onDropPath(clipboard.path);
        setClipboard(null);
        const srcParent =
          clipboard.path.replace(/\/+$/, "").split("/").slice(0, -1).join("/") ||
          "/";
        invalidateAfter(srcParent, target.path);
      } else {
        // Copy-paste: silently auto-rename on collision rather than
        // erroring out. Matches the operator's "I want a duplicate"
        // intent when pasting back into the same dir, and avoids
        // having to manually clear a name conflict when pasting
        // somewhere that happens to already contain a file with the
        // same name.
        await api.fsCopy(clipboard.path, target.path, { autoRename: true });
        invalidateAfter(target.path);
      }
    } catch (e) {
      alert(`Paste failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const doInitWorkspace = () => {
    onClose();
    // The clicked directory IS the workspace dir. The modal collects
    // metadata only (name / description / forgather dir / libs); the
    // backend writes forgather_workspace/ inside this dir.
    openInitWorkspace(target.path);
  };

  const doNewProject = () => {
    onClose();
    if (!enclosingWs) return;
    // Project lives at <click-dir>/<new-name>; modal expects
    // workspace = the cluster and project_dir_name = rel path from
    // workspace_root + leaf.
    const rel = relPath(enclosingWs.workspace_root, target.path);
    openNewProject({
      workspace: enclosingWs,
      initialDirName: rel ? `${rel}/` : "",
    });
  };

  const doNewConfig = () => {
    onClose();
    if (!enclosingProj) return;
    openNewTemplate({
      project: enclosingProj,
      kind: "config",
      initialDirHint: target.path,
    });
  };

  const doNewTemplate = () => {
    onClose();
    if (!enclosingProj) return;
    openNewTemplate({
      project: enclosingProj,
      kind: "template",
      initialDirHint: target.path,
    });
  };

  const canPaste = target.isDir && clipboard != null;

  /** Right-click "Duplicate" — copies the clicked file or directory
   *  into its own parent with an auto-generated " (copy)" suffix.
   *  Skipped for roots (no sensible parent to copy into). */
  const doDuplicate = async () => {
    onClose();
    const parent =
      target.path.replace(/\/+$/, "").split("/").slice(0, -1).join("/") || "/";
    try {
      const r = await api.fsCopy(target.path, parent, { autoRename: true });
      invalidateAfter(parent);
      // Refresh selection cues / breadcrumbs if applicable. Nothing
      // to do here — the tree picks up the new entry from
      // invalidateAfter's query refresh.
      void r;
    } catch (e) {
      alert(`Duplicate failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <ContextMenu x={target.x} y={target.y} onClose={onClose}>
      <div className="context-menu-header muted">{basename(target.path)}</div>
      {!target.isDir && (
        <button
          className="context-menu-item"
          onClick={() => {
            onOpenFile(target.path);
            onClose();
          }}
        >
          ✎ Open
        </button>
      )}
      {!target.isDir && onOpenDoc && isDocLike(target.path) && (
        <button
          className="context-menu-item"
          onClick={() => {
            onOpenDoc(target.path);
            onClose();
          }}
        >
          📖 Open in Docs…
        </button>
      )}
      {!target.isDir && (
        <button
          className="context-menu-item"
          onClick={() => {
            api.downloadFile(target.path);
            onClose();
          }}
        >
          ⬇ Download…
        </button>
      )}
      {target.isDir && (
        <button className="context-menu-item" onClick={doNewFile}>
          ➕ New File…
        </button>
      )}
      {target.isDir && (
        <button className="context-menu-item" onClick={doNewFolder}>
          ➕ New Folder…
        </button>
      )}
      {target.isDir && enclosingRoot && (
        <button
          className="context-menu-item"
          onClick={doInitWorkspace}
          title={`Initialize workspace at ${target.path}`}
        >
          📁 New Workspace…
        </button>
      )}
      {target.isDir && enclosingWs && (
        <button
          className="context-menu-item"
          onClick={doNewProject}
          title={`Project will land under workspace ${enclosingWs.workspace_root}`}
        >
          📁 New Project…
        </button>
      )}
      {target.isDir && enclosingProj && (
        <button
          className="context-menu-item"
          onClick={doNewConfig}
          title={`New config inside project ${enclosingProj.name || enclosingProj.project_dir}`}
        >
          📄 New Config…
        </button>
      )}
      {target.isDir && enclosingProj && (
        <button
          className="context-menu-item"
          onClick={doNewTemplate}
          title={`New template inside project ${enclosingProj.name || enclosingProj.project_dir}`}
        >
          📄 New Template…
        </button>
      )}
      {!target.isRoot && (
        <button className="context-menu-item" onClick={doRename}>
          ✎ Rename…
        </button>
      )}
      {!target.isRoot && (
        <button className="context-menu-item" onClick={doCut}>
          ✂ Cut
        </button>
      )}
      <button className="context-menu-item" onClick={doCopy}>
        ❏ Copy
      </button>
      {!target.isRoot && (
        <button
          className="context-menu-item"
          onClick={doDuplicate}
          title={`Copy ${basename(target.path)} alongside itself with a "(copy)" suffix`}
        >
          ⎘ Duplicate
        </button>
      )}
      {canPaste && (
        <button
          className="context-menu-item"
          onClick={doPaste}
          title={`${clipboard!.mode === "cut" ? "Move" : "Copy"} ${clipboard!.path}`}
        >
          ⎘ Paste {clipboard!.mode === "cut" ? "(move)" : "(copy)"}
        </button>
      )}
      {!target.isRoot && (
        <button
          className="context-menu-item context-menu-destructive"
          onClick={doDelete}
        >
          🗑 Delete Permanently…
        </button>
      )}
    </ContextMenu>
  );
}
