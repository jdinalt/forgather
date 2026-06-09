# Forgather Server -- Web UI Features

> Part of the [Forgather Server](./README.md) docs. This page catalogs the
> web UI: every panel, view, and context menu, the right-sidebar AI-agent
> assistant, and the not-yet-implemented list. See the [README](./README.md)
> for CLI / config / security and running, and [ARCHITECTURE](./ARCHITECTURE.md)
> for internals.

## Implemented features

### App chrome

The left side of the window is a collapsible sidebar (`<aside
class="app-sidebar">`) that owns navigation and global actions. Top to
bottom:

- **Header** — "Forgather Server" title and a window/sidebar SVG
  toggle that collapses the sidebar. Right-click anywhere on the
  header opens a small context menu whose only entry today is
  **Help…**, routing to this reference document (rendered through
  MkDocs if a serve is alive, the built-in Docs viewer otherwise).
  (The Refresh and scheduler ▶/⏸ controls that used to live up here
  moved to the new footer — see below.)
- **Nodes** (cluster-only, sits above Views) — collapsible
  `<details>` listing every cluster peer by hostname with a tri-state
  health dot (green = reachable, yellow = reachable but a headline
  version is missing / diverges from the cluster majority, red =
  unreachable) and master/this-server tags. Clicking a peer mints a
  short-lived single-use SSO URL (`/api/cluster/peer_session`) and
  opens that peer's webui in a new tab with no login prompt — same
  trust model as cluster bearer access. On a single-node `default`
  cluster the list shows only this server.
- **Views** (collapsible `<details>`) — vertical tabs with icons:
  🖧 Cluster (cluster-only), 📁 Projects, ✎ Edit, 📚 Docs, 🖥 GPUs,
  📋 Queue, ⚙ Jobs, 🔮 Inference, 🗂 Datasets. Selecting anything
  in the project tree routes back to the Projects view
  automatically. The **Edit** view is the tabbed Monaco editor
  (formerly named "Files"); it was renamed to free the "Files" name
  for the new sidebar filesystem tree (see below). **GPUs** is
  always the local node's live `GpuPanel` (WS stream, kill, context
  menu), independent of cluster mode. **Cluster** is the
  cluster-wide surface — see
  [Cluster mode (multi-node, prototype)](./ARCHITECTURE.md#cluster-mode-multi-node-prototype).
- **Tools** (collapsible `<details>`) — one-shot model-manipulation
  utilities. Persisted to localStorage so the next open of each
  modal defaults to the last-committed values; `priority` resets
  each time since the right value depends on current queue state.
    - **📐 Evaluate…** — queues `forgather eval` against an arbitrary
      model directory.
    - **🔁 Convert Model…** — queues `forgather convert` against a
      pair of source/destination model paths. Direction
      (HF ↔ Forgather) is auto-detected unless `--reverse` is forced.
      Persisted under `forgather-global-convert-v1`. The footer
      carries a **Reset to defaults** button that clears the
      persisted blob.
    - **📦 Finalize Model…** — queues `forgather finalize` to package
      a trained Forgather output tree into a clean directory:
      tokenizer additions, chat template, generation config,
      root-copy / keep-optimizer toggles. Persisted under
      `forgather-global-finalize-v1`. Same **Reset to defaults**
      affordance.
    - **⬆️ Update Model…** — queues `forgather update` to migrate a
      saved Forgather model to the current source schema. Reads
      `forgather_arch` / `forgather_arch_version` from the source
      `config.json` and walks the per-arch migration chain; the
      modal exposes `--arch` / `--from-version` / `--to-version` /
      `--checkpoint` overrides plus dtype, device, strict / no-strict,
      safetensors, and dry-run toggles. Persisted under
      `forgather-global-update-v1`. Same **Reset to defaults**
      affordance.
- **Services** (collapsible `<details>`) — launchers for the four
  long-running spawned-process services: 🔮 Inference, 🗂 Dataset,
  📊 TensorBoard, 📖 MkDocs. Same persistence model as Tools. Each
  launcher carries a right-aligned running-count pill (same UI as
  Views → Jobs) and, when there are configured instances of that
  type, a chevron that expands a per-type list of saved services.
  Each saved-service row has a red/green dot reflecting actual
  running state (JobRecord `status == "running"`), a ▶/⏹ toggle
  that flips the `enabled` flag, an `×` delete (aborts the running
  instance first), and a clickable label that does the obvious
  thing for each type:

  - **Inference / Dataset** → switch to the matching view (chat
    or browse the running server).
  - **TensorBoard** → open `http://<host>:<port>/api/tb/<queue_id>/`
    in a new tab. The path prefix is the one the scheduler stamps
    onto the spawned TB via `--path_prefix`; TB only serves under
    that prefix.
  - **MkDocs** → open `http://<host>:<port>/` in a new tab.

  For wildcard binds (`0.0.0.0` / `::`), the URL substitutes
  `window.location.hostname` — the host the browser is already
  reaching the webui on, guaranteed to be reachable from there.

  Each service modal also has a **Create service…** button that
  prompts for a name (with a sensible default per type — model
  basename for inference, logdir basename for tensorboard, etc.)
  and persists the modal's current args into `server_config.yaml`
  via [`POST /api/services`](./API.md#services-auto-start). See
  [Auto-start services](./README.md#auto-start-services) for the boot
  semantics.
- **Project tree** — Search Roots + workspace-clustered projects
  (see below).

Below the scrolling section stack is a **sidebar footer** pinned
via `position: sticky; bottom: 0`. Four icon-only buttons (tooltips
explain each):

| Glyph | Action |
| --- | --- |
| ⟳ | **Refresh data** — invalidates the entire client query cache so disk edits to workspace metadata, templates, configs are picked up immediately. |
| ▶ / ⏸ | **Scheduler toggle** — flips the dispatcher loop on/off (green when running, muted when paused). Same mutation that backed the old header button. |
| ↺ | **Restart server** — confirms, then hits `POST /api/server/restart`. The process re-execs in place; running training / inference / dataset_server / mkdocs / tensorboard subprocesses survive across the exec via the standard PID-reattach path. Useful for picking up `server_config.yaml` changes without killing the terminal. |
| ⚙ | **Open server config** — opens the resolved `server_config.yaml` in the embedded editor. |

When collapsed, the sidebar shrinks to a 44-px strip showing only the
expand toggle and the icon-only view switcher. Both the collapsed
strip and the expanded layout stay mounted in the DOM (toggled via
`display:none`), so the project tree's expansion state — which
workspaces / projects / artifact groups are open — survives a
collapse/expand cycle.

Default ports for the spawned services match each tool's canonical
default — TensorBoard `6006`, inference `8137`, MkDocs `8000` — so
existing SSH port-forward configs keep working without per-host
rebinds. Inference picks `8137` rather than the more common `8000` so
it doesn't collide with MkDocs out of the box. Collisions on first
submit are easy to resolve in the dialog and the resolved port
persists for next time.

### Project / config discovery

- Walks each search root in two passes: first for ``forgather_workspace/``
  marker dirs (so empty workspaces seed empty clusters that still show
  in the tree), then for ``meta.yaml`` (projects, attached to whichever
  workspace_root MetaConfig resolves them to). Hierarchical workspaces
  nest under their enclosing parent. Both passes prune hidden directories,
  ``forgather_workspace/``, ``output_models/``, ``node_modules/``,
  ``__pycache__``, and ``.git`` to avoid slow or redundant subtree walks.
- Workspaces resolve display name + description from
  `forgather_workspace/workspace.yaml` → README title + first paragraph
  → directory basename. `forgather ws create` writes `workspace.yaml`
  alongside the existing files.
- Configs lazy-load `config_name`, `config_description`, and
  `config_class` from the materialized `meta` block when their project
  is expanded.
- **Per-config artifact sub-tree** — configs that have materialized
  outputs (runs, checkpoints, evaluations) expand to three sub-groups
  with live counts: **Logs**, **Checkpoints**, **Evaluations**. Leaves
  are clickable selection targets with their own detail panels in the
  right pane, and right-clickable for delete-permanently / delete-all
  (user-confirmed, guarded by `/api/fs/delete-dir`). Populated lazily
  via `/api/project/models` — two configs that materialize to the same
  `output_dir` show the same sub-nodes.
- **Refresh button** (⟳ in the sidebar footer) invalidates the
  entire client query cache so disk edits to workspace metadata,
  templates, configs are picked up immediately.

### Config inspection

A three-tab viewer for the selected config:

| Tab         | Content                                                                          |
| ----------- | -------------------------------------------------------------------------------- |
| `info`      | Project's `README.md` rendered as markdown (GFM tables, inline images).          |
| `pp`        | Jinja-rendered, fully preprocessed YAML.                                         |
| `templates` | Two browsing modes (mode bar at the top of the left pane): `trefs` shows the Graphviz-rendered template-dependency graph for the selected config; `tlist` shows every template on the project's search path, grouped by search-root category. Click a node / row to preview in the right pane. |

Monaco syntax-highlights these with a custom Monarch tokenizer
for Forgather's YAML + Jinja2 dialect (`--`/`<<`/`>>`/`==` line
statements, `[block]` / `[/block]`, `!call` / `!partial` / `!singleton`
/ etc., inline `#--- name ---` markers, anchors / aliases).

The `templates` tab's right pane displays the selected node's source
read-only. An **✎ Edit** button next to the path label hands the file
off to the Edit panel (see below) for actual editing.
Right-clicking any template — graph node in `trefs` mode, list row
in `tlist` mode — opens a context menu with **✎ Open in Editor**
that bypasses the preview and drops the file straight into the Files
panel.

The `tlist` view is backed by `GET /api/project/templates`, which
mirrors the interactive CLI's `edit` selector: groups labeled
"Project Templates" / "Workspace Templates" / "Base Templates" /
"Example Templates" / "<Subdir> Templates", with each template
attributed to the *first* matching search-path entry (Jinja's
first-match resolution). A synthetic **Meta** group is prepended,
containing the project's `meta.yaml` so it can be browsed and edited
alongside templates — `meta.yaml` lives outside any `templates/`
directory so `MetaConfig.find_templates()` doesn't yield it on its
own. The Meta group is inserted *after* the search-path attribution
loop runs so the project_dir search root (which contains every
project template) doesn't sweep them into Meta.

**Header**: shows the config's pretty name (from `config_name` in
the materialized meta block) bolded, with the yaml filename in muted
monospace next to it (omitted when the two would be identical), then
a small `config_class` chip, then the project label. Mirrors the
two-line label the project tree already uses.

**Auto-navigation**: clicking a project (expanding the tree node)
selects its `default_config` and switches to the `info` tab — so
browsing projects surfaces the README first.

**Tab tracking on config switch**: the `info` tab is project-scoped
(it's the README), so a click that's actively *choosing* a config
in the tree silently jumps to the `templates` tab. The two
config-scoped tabs (`pp`, `templates`) are left alone so the user
can iterate across configs while keeping the same lens — comparing
materialized YAML between configs is the entire point of `pp`, and
the `templates` view auto-updates its right pane (see below) so
re-clicking a config feels like flipping a slide.

**Right-pane follows the active config**: in either `trefs` or
`tlist` mode the read-only preview auto-resets to the active
config's own template every time the config changes, including the
initial mount. Manual deep-dives into parent templates (clicking a
node in `trefs` or a non-config row in `tlist`) override the
preview and aren't disturbed unless the user picks a different
config.

**tlist click promotes configs**: clicking a row in `tlist` whose
path matches one of the project's configs (i.e. lives under
`config_prefix`) promotes that config to the active selection,
updating the header chip, action buttons, dynamic-args form, and
the `trefs` graph that you'd see if you flipped modes. trefs nodes
are always *referenced* templates of the current config — never
sibling configs you'd want to switch to — so trefs clicks remain
preview-only.

**Class-aware actions**: configs marked `type.training_script*` get
**▶ Run**, **🔧 Overrides…**, **🗑 Clean Output…**, **📊 TensorBoard…**
buttons; when the config has checkpoints on disk, **🔮 Serve Inference…**
and **⚖ Evaluate…** also appear. Other classes (`type.model`,
`type.dataset`, etc.) only get **🔧 Overrides…**. Same filtering applies
to the right-click context menu on tree rows.

### Selection-driven detail panels

Clicking a leaf in the artifact sub-tree swaps the right pane to a
dedicated viewer — the tree is the single source of navigation truth:

- **Log** (`LogDetailPanel`) — tabs `TTY` (captured `tty.log`) and
  `Summary` (best loss, total steps, eval loss, perplexity, derived
  from `/api/run/summary`).
- **Checkpoint** (`CheckpointDetailPanel`) — step, size, world_size,
  saved timestamp, path, plus **🔮 Serve Inference…** and
  **⚖ Evaluate…** buttons pre-filled with this checkpoint's path.
- **Evaluation** (`EvalDetailPanel`) — results table (per-metric,
  per-sample if present) via `EvalResultTable`.

### Edit panel (tabbed editor)

Main-pane view that opens files for editing. Reached either by
clicking the ✎ Edit tab in the view switcher, the **✎ Edit** button
on a selected template in the Projects → templates view, the
**✎ Open** entry in the sidebar Files tree's right-click menu, or
the **📄 New Config…** / **📄 New Template…** flow under a project
context menu. All four routes hand the resulting absolute path to
`filesApi.openFile(path)` and switch the view to `edit`.

Per-buffer language is resolved by `webui/src/file-languages.ts`:
`.yaml` / `.yml` / `.jinja` / `.jinja2` use Forgather's custom
Monarch tokenizer; `.md` / `.markdown` use Monaco's built-in
markdown; `.py` uses built-in python; everything else falls back to
plaintext (so `.log`, `Makefile`, `LICENSE`, `.json`, `.toml`,
`.sh`, etc. all open and render — they just don't get
extension-specific syntax highlighting).

Click-to-open in the Files tree is *not* gated by extension.
`GET /api/template/source` does a binary-detection check on the
server (null-byte scan over the first 8 KiB plus a UTF-8 decode
attempt) and returns HTTP 415 for files that look binary. The
editor surfaces the 415's detail in-tab — clear "this isn't a
text file" instead of streaming garbage into Monaco.

State lives in `useFilesState` (`webui/src/files-state.ts`) — a single
hook owned by `App.tsx` so any caller can drop a file in regardless
of which view is currently visible. Buffers are keyed by absolute
path and shared across splits, so the same file open in two splits
stays in lock-step. The hook returns: `openFile`, `setContent`,
`saveFile`, `closeTab`, `closeOthers`, `closeAll`, `setActiveTab`,
`setActiveSplit`, `splitVertical`, `moveTab`, `isDirty`, and
`dropPath` (the last is a non-prompting close-everywhere used when
an external file op invalidates a path — rename / move / delete from
the Files tree).

**Render layout** (`components/FilesPanel.tsx`): a row of `SplitPane`s.
Each split has a tab bar (with one `FileTab` per open path, plus a ⊟
fork-vertical-split button) and a Monaco editor showing the active
buffer. Empty splits collapse automatically when their last tab moves
or closes (the layout always keeps at least one split). The dirty
indicator is the bullet next to the tab label.

**Save**: window-level Ctrl/Cmd+S handler installed during the panel's
`useEffect`, registered in capture phase so Monaco doesn't swallow the
key. Saves the active split's active tab via
`PUT /api/template/source` (atomic tmp+fsync+rename through
`_atomic.atomic_write_text`). The right-click context menu on a tab
or on the editor body offers Save / Close / Close Others / Close All —
Close-style actions confirm with `window.confirm` if any closing tab
is dirty.

**Drag/drop**: tabs are HTML5-draggable with the
`application/x-forgather-tab` MIME. Dropping on another tab inserts
before that tab; dropping on the spacer at the end of a tab bar
appends. Cross-split moves auto-collapse the source split if it
empties out and a peer is left.

**React-18 gotcha**: `setState(updater)` does not run the updater
synchronously. `openFile` decides whether to fire the
`api.templateSource(path)` fetch by reading `stateRef.current.buffers`
synchronously *before* calling `setState`, not by mutating a flag
inside the updater closure. (An earlier version did the latter and
the fetch never fired — the buffer appeared with `loading: true` and
stayed there.) Other places that need to read latest state from async
callbacks (`saveFile`) use the same `stateRef` snapshot.

**Backend**: `PUT /api/template/source` accepts `{path, content,
expected_mtime?}`, requires an absolute path to an *existing*
regular file (no create-new yet), and writes through
`_atomic.atomic_write_text`. Same trust posture as
`GET /api/template/source` — single-user localhost prototype, no
per-search-root containment check.

**Optimistic-concurrency: lost-update protection.** Every
`GET /api/template/source` returns the file's `os.path.getmtime`
as an `X-Mtime` response header. The editor stamps the buffer's
``baselineMtime`` from this header on load and after every
successful save. Save sends ``expected_mtime`` along with the
content; if the file's current on-disk mtime is newer (with a
1 µs tolerance for filesystem jitter), the server responds
**409** with `detail: {message, current_mtime, expected_mtime}`.
The client throws a typed `SaveConflictError`, the buffer keeps
its local content (no clobber), and `FilesPanel` opens a
`ConflictModal` showing the file path, both timestamps, and three
choices:

- **Overwrite** — `forceSaveFile(path)` re-PUTs without
  `expected_mtime` so the server skips the check.
- **Reload from disk** — `reloadFile(path)` re-GETs and replaces
  baseline + content + mtime; local edits are discarded.
- **Cancel** — `clearConflict(path)` dismisses the modal; the
  buffer stays dirty so the user can keep editing or retry.

`FileBuffer` carries `baselineMtime` and an optional
`conflict: {currentMtime}` flag; the modal watches every open
buffer and pops for the first conflicting one.

### Sidebar layout: top-level collapsible sections + footer bar

The sidebar's body below the header is a stack of independent
`<details>`-backed groups, all sharing the same chrome (uppercase
muted summary, custom `▸`/`▾` glyph via `::before`,
`::-webkit-details-marker { display: none }`) and all defaulting to
*closed* — first boot doesn't trigger any directory walks until
the user expands something. The bubbled `toggle` event is filtered
with `e.target === e.currentTarget` so nested `<details>` (project
rows, file-tree dirs) don't stomp on the outer section's open
state.

| Section | Component | Purpose |
| --- | --- | --- |
| **Views** | `<nav class="sidebar-views">` | The view switcher (📁 Projects, ✎ Edit, 🖥 GPUs, 📋 Queue, ⚙ Jobs, 🔮 Inference). |
| **Tools** | inline buttons | One-shot model-manipulation utilities: 📐 Evaluate, 🔁 Convert Model, 📦 Finalize Model, ⬆️ Update Model. |
| **Services** | inline buttons + `ServicesPanel` | Long-running spawned processes: 🔮 Inference, 🗂 Dataset, 📊 TensorBoard, 📖 MkDocs. Each launcher row carries a right-aligned running-count pill (same UI pattern as Views → Jobs) and, when there are configured instances of that type, a chevron that expands a per-type list of saved services with red/green dots and ▶/⏹/× controls. See [Auto-start services](./README.md#auto-start-services). |
| **Search Roots** | `SearchRootsPanel` | Root-list management: Browse… to add, × to remove, **📁 New Workspace…** for the dropdown-driven flow. Lifted out of `ProjectTree` so each group is its own top-level entry. |
| **Projects** | `ProjectTree` | The familiar workspace-clustered project forest. |
| **Files** | `FilesTree` | Hierarchical filesystem view of every search root. |

Below the scrolling section stack a **sidebar footer** is pinned via
`position: sticky; bottom: 0`. Four icon-only buttons:

- **⟳ Refresh data.** Invalidates the entire client query cache so
  disk edits to workspace metadata, templates, configs are picked
  up immediately. Moved here from the old sidebar header.
- **▶ / ⏸ Scheduler toggle.** Flips the dispatcher loop on/off
  (green when running, muted when paused). Same mutation that
  backed the old header button.
- **↺ Restart server.** Confirms, hits `POST /api/server/restart`,
  then polls `/api/health` and reloads the page once the rebooted
  server is responsive. Useful for picking up `server_config.yaml`
  changes without killing the terminal. Spawned jobs survive.
- **⚙ Open config.** Opens the loaded server config file
  (`server_config.yaml`) in the embedded editor. The path is
  surfaced by `GET /api/server-config-path`.

Earlier iterations had Tools and the view switcher visually
distinct from the rest (a horizontal rule above and below Tools, a
Tools-specific summary block). Those were dropped so the groups
read as a single uniform stack — easier to scan, no implicit
grouping where there isn't one. The Tools / Services split came
later to separate one-shot utilities (Evaluate / Convert / Finalize
/ Update) from persistent services (which gained the
configured-instance management above).

### Files tree (sidebar)

A hierarchical filesystem view of every configured search root,
letting users browse what's actually on disk and open files for
editing without knowing paths in advance. Component:
`webui/src/components/FilesTree.tsx`.

**Lazy loading.** Each root and each subdirectory is a *controlled*
`<details>` with React state (`useState(false)`) tracking open
state via `onToggle`, and the `<DirChildren>` listing pane is
**only rendered when the node is open**. Without this gate, React
mounts every `<details>`'s content regardless of the open
attribute, the inner `useQuery` fires immediately, and the entire
tree gets walked recursively on first paint. With the gate, opening
the Files section fetches only the search-roots list (one tiny
call); each root's listing fetches only when the user clicks it
open; the same applies to every nested directory.

Listings are cached under `["fs-browse", path, showHidden, true]`
(matching the same key the modal `DirectoryBrowser` uses, so cache
entries are shared and refreshes propagate). 30-second `staleTime`
keeps re-opens snappy.

Files render as clickable buttons regardless of extension — every
file gets a click-to-open. The backend's binary-detection in
`/api/template/source` (null-byte scan + UTF-8 decode check)
refuses truly binary files with HTTP 415 and the editor surfaces
the message in-tab, so the user gets a clear "this isn't a text
file" instead of garbage. Per-buffer language is resolved by
`languageFor(path)` (`file-languages.ts`): `.yaml`/`.yml`/`.jinja*`
→ Forgather Monarch tokenizer, `.md`/`.markdown` → Monaco markdown,
`.py` → Monaco python, everything else → plaintext (so `.log`,
`LICENSE`, `Makefile`, `.json`, `.toml`, `.sh` etc. all open fine).

A **Show hidden** checkbox at the top of the section toggles
dotfile visibility; the listing query key includes the flag so
toggling refetches.

**Right-click context menu** items (all conditional on the
target type):

| Item | Visible when | Action |
| --- | --- | --- |
| **✎ Open** | file | `filesApi.openFile(path)` + switch to Edit view |
| **➕ New File…** | dir | `POST /api/fs/new-file` (bare-name, refuses overwrite) — opens the new empty file in the editor |
| **➕ New Folder…** | dir | `POST /api/fs/mkdir` |
| **📁 New Workspace…** | dir under a search root | opens `InitWorkspaceModal` (see below) targeting the clicked dir |
| **📁 New Project…** | dir under an existing workspace | opens `NewProjectModal` with the enclosing workspace pre-resolved + the rel path from `workspace_root` pre-filled in `project_dir_name` |
| **✎ Rename…** | non-root | prompt for new bare basename → `POST /api/fs/rename` |
| **✂ Cut** | non-root | set in-memory clipboard `{path, mode: "cut"}` |
| **❏ Copy** | any | set clipboard `{path, mode: "copy"}` |
| **⎘ Paste** | dir, when clipboard set | `POST /api/fs/move` (cut, consumes clipboard) or `POST /api/fs/copy` with `auto_rename: true` (copy — collisions become `<stem> (copy)<ext>` siblings rather than 409 errors) |
| **⎘ Duplicate** | non-root | `POST /api/fs/copy` into the clicked node's parent with `auto_rename: true`; same "(copy)" suffix flow paste uses, no clipboard needed |
| **🗑 Delete Permanently…** | non-root | confirm + `POST /api/fs/delete-file` (file) or `POST /api/fs/delete-dir` (dir) |

The clipboard is in-memory (`useState` in `FilesTree`); no OS
clipboard interaction. Search roots themselves can't be Cut /
renamed / deleted via this menu — managing roots stays in the
**Search Roots** section.

After any rename / move / delete, the tree calls
`filesApi.dropPath(stale)` so any open editor tab pointing at the
now-stale path is dropped silently (no dirty-prompt). The user
saves before invoking the destructive op; if they didn't, the tab
is discarded without confirmation since the path is already gone
from disk.

**Init-workspace-here flow.** The Files-tree directory menu's
**📁 New Workspace…** opens a slimmer `InitWorkspaceModal` —
*not* the dropdown-driven `NewWorkspaceModal` from the Search-Roots
section — because the path is already determined by the
right-click target. The modal collects only metadata (name /
description / forgather dir / libs / additional search paths) and
the clicked dir becomes the workspace root directly. Backend
`POST /api/workspace/init-here` validates that the directory
exists, doesn't already contain `forgather_workspace/`, and lives
at-or-under a configured search root, then dispatches to
`ws_create_cmd` with the new `init_existing` flag — which skips
the original "must not exist" check + `os.makedirs(workspace_dir)`
and just writes the four metadata files into a new
`forgather_workspace/` subdir.

**Targeted cache invalidation.** Each create/rename/move/delete
op invalidates *only* the immediately-affected parent directory's
listing — keyed by `["fs-browse", parent]` with `exact: false`,
which prefix-matches just that path's variants (showHidden /
files_too). Sibling, ancestor, and unrelated subtrees aren't
touched. Combined with the lazy-mount above, creating a workspace
or project triggers exactly one listing refetch (the parent that
got the new entry), not a re-walk of everything currently visible.

**Backend** — endpoints under `routes/fs.py`, all sharing the same
safety posture as the existing `/fs/delete-file` (absolute path
required, no symlinks, ≥4 path components). None of these
non-destructive ops require a `confirmed` flag because each is
recoverable by reverse operation:

- `POST /api/fs/rename` `{path, new_name}` — `os.rename` to a
  bare basename; refuses overwrite (409).
- `POST /api/fs/copy` `{src, dest_dir, auto_rename?: bool, target_name?: string}`
  — `shutil.copy2` for files, `shutil.copytree` for directories.
  Without `auto_rename` a destination collision returns 409. With
  `auto_rename: true` the server picks a non-colliding sibling by
  appending ` (copy)` / ` (copy 2)` / … to the stem (used by paste
  and right-click Duplicate). `target_name` overrides the
  destination basename — single filename only, no path separators —
  used by the "Duplicate Config…" prompt to land the new file at
  the operator-chosen name.
- `POST /api/fs/move` `{src, dest_dir}` — `shutil.move` (so
  cross-device moves degrade to copy + unlink); refuses overwrite.
- `POST /api/fs/new-file` `{parent, name}` — `Path.touch()` an
  empty file; refuses overwrite.
- `POST /api/fs/mkdir` `{parent, name}` — single new directory
  (already existed; reused for + New Folder…).

### Markdown surfaces: Docs view + Project Info

Both the **Docs** view (`DocsPanel`) and the project tree's
**Info** tab (`InfoPane`) render markdown with `react-markdown +
remark-gfm + rehype-slug`. They share three behaviours worth
calling out:

- **Outline column.** A 220-px-wide nav rail to the left of the
  content lists every h1 / h2 / h3 by clicking the rendered DOM
  for `id`-stamped headings (rehype-slug stamps them) and
  rendering one entry per heading. Clicking smooth-scrolls the
  body to the matching anchor. Hidden entirely when the page has
  fewer than two headings.
- **Scroll restore.** The Docs view's Back button restores the
  scroll position of the page being returned to (the back-stack
  entry records `scrollTop` when pushed; the body re-applies it
  in a `requestAnimationFrame` after the content has rendered, so
  a saved offset doesn't get clamped to 0 by an empty body during
  a refetch). The Info tab applies the same trick across
  config-tab switches — it stays mounted with `display:none` so
  the scroll container survives, and its scrollTop is saved /
  restored from a ref.
- **Default landing page.** The Docs view lands on `docs/README.md`
  rather than the repo-root README — the docs index is the curated
  entry point with links to installation / tutorials / config / API,
  whereas the root README is closer to a project elevator pitch.
  Falls back to the root README if the docs index is missing, or
  the empty state if neither exists. Operators can override the
  default with `--docs-landing PATH` (or `args.docs_landing` in
  `server_config.yaml`); a missing override falls back to the
  built-in preference rather than failing hard.

**Pre-rendered API directives.** The Docs view serves raw markdown,
so `:::`-style `mkdocstrings` directives in `docs/api/*.md` would
otherwise appear unrendered. `forgather docs build` (see
`src/forgather/docs_build/`) walks `docs/`, expands directives via
griffe, and writes the result to `docs/.built/<rel>.md`. The
`/api/docs/file` endpoint prefers the built copy when one exists
and is not older than the source; otherwise it serves the raw
source unchanged, so the Docs view always works whether or not
the build step has been run. The cache is populated automatically
by `./build-webui.sh` (and therefore by the Docker post-build
step that runs it), and can be regenerated on demand with
`forgather docs build` or removed with `forgather docs clean`.
The reported response path is always the canonical source so
relative asset references resolve against the right directory.

`docs_hooks.py` is a MkDocs `on_page_markdown` hook (wired via
`mkdocs.yml: hooks:`) that rewrites relative markdown links on
pages whose source is a symlink. Many pages under `docs/` are
symlinks to canonical files elsewhere in the repo — e.g.
`docs/forgather-server.md → ../tools/forgather_server/README.md`.
MkDocs computes link paths from the docs_dir page location rather
than the source file's realpath, so relative links written from
the source author's perspective (`../../docs/foo.md`) come out
broken in the rendered site. The hook resolves each relative href
against the symlink target's realpath, then rewrites it as a path
relative to the docs_dir page; it also maintains a
`realpath → docs_dir alias` map so a link that lands on the
realpath of another docs symlink gets pointed at the in-tree alias
rather than ascending out of `docs_dir`.

### Persistent dynamic-args overrides

`/api/config/overrides` is a per-config JSON cache keyed by
`sha256(abspath(project_dir) + "\0" + config_name)`. Stored values are
layered as the *base* under any explicit kwargs and applied
automatically by `pp`, `output-dir`, `config/meta`, *and the trefs
graph* — so e.g. setting `--trainer-type=fsdp2` makes the trefs view
show `trainers/fsdp2_trainer.yaml` instead of the default. Submitting a
job auto-saves the values used; the **🔧 Overrides…** modal explicitly
sets/clears them.

### Submit / queue / scheduler

- ▶ **Run** opens a Submit modal with a generated form for the config's
  `[dynamic_args]` block. Schemas honor `type` (`int` / `str` / `float`
  / `bool` / `path`), `choices` (renders a dropdown), `action:
  store_true` / `store_false` (renders as a checkbox with concrete
  default), and `path` types (renders an inline file picker). The form
  pre-fills from the overrides cache.
- The Multi-node panel and the Dynamic arguments form are each in
  their own collapsible `<details>` block. With both open, neither
  takes more than 50% of the dialog body so a long Multi-node panel
  can't push the Dynamic args off-screen and vice versa. The
  participants table inside the Multi-node panel caps at ~9 rows
  and scrolls internally for the same reason.
- The form shows what `nproc_per_node` the config declares (`"gpu"` /
  fixed integer / `"cpu"` / `"auto"`) and warns when the user's GPU
  reservation count would mismatch a fixed worker count. These
  single-node-mode controls are hidden when the server is in cluster
  mode — the per-node GPUs column in the Multi-node panel takes their
  place. Priority stays visible across both modes.
- When cluster mode is active and the operator has only the local
  node selected (the implicit default), Submit goes through the
  regular single-node enqueue path and uses the panel's local-node
  GPUs value as the reservation count. Adding a peer flips Submit
  to the cluster fanout path; the button label changes to "Submit
  to N nodes" so the choice is explicit. The dialog refuses to
  submit if cluster mode is active and the operator has unselected
  every node.
- Last-used Multi-node settings (participants, per-node GPUs, iface,
  rdzv host/port, mismatch acknowledgement) persist alongside the
  dynamic-args overrides in the same per-config cache, so a config
  "opens where you left off" for both submit modes. **Reset to
  defaults** drops everything we cached for this config, including
  the Multi-node selection.
- The scheduler holds a JSON-backed queue + an in-memory dispatcher
  loop. **Enabled by default** so a freshly-restarted server resumes
  dispatch immediately. Pause anytime with the `▶`/`⏸` button in the
  sidebar header. The queue section at the top of the **Jobs** view
  shows the current `running` / `paused` state alongside the queued
  items (queued and running work share one view).
- Dispatch picks idle GPU indices that aren't excluded via
  `CUDA_VISIBLE_DEVICES`, sets the child's `CUDA_VISIBLE_DEVICES` to
  the assignment, and invokes `torchrun` directly (mirrors what
  `forgather train` does, minus the extra subprocess layer — lets the
  scheduler own the process group for clean abort).

**Ten job types** share the queue, scheduler, GPU accounting, and TTY
capture machinery. The non-CUDA-by-default types (`tensorboard`,
`mkdocs`, `convert`, `finalize`, `update`, `dataset`, `dataset_server`)
accept `requested_gpus == 0`; the others default to at least one GPU.
Convert / finalize / update will happily take a GPU if the user sets
`--device cuda…` and bumps the reservation.

| Type             | Spawned by                                                                             | Lifecycle                              |
| ---------------- | -------------------------------------------------------------------------------------- | -------------------------------------- |
| `training`       | ▶ Run (Submit modal)                                                                   | Terminal when trainer exits.           |
| `eval`           | ⚖ Evaluate… (EvalModal, from config or checkpoint)                                     | Terminal when `forgather eval` exits.  |
| `inference`      | 🔮 Inference… (InferenceModal, project-backed or ad-hoc; sidebar Services)             | Long-lived; kill/force-kill to stop.   |
| `dataset_server` | 🗂 Dataset… (DatasetServerModal, sidebar Services)                                     | Long-lived; kill to stop.              |
| `tensorboard`    | 📊 TensorBoard… (TensorBoardModal, sidebar Services or per-config/per-model)           | Long-lived; kill to stop.              |
| `mkdocs`         | 📖 MkDocs… (MkDocsModal, sidebar Services — picks an `mkdocs.yml` + host:port)         | Long-lived; kill to stop.              |
| `convert`        | 🔁 Convert Model… (ConvertModal, sidebar Tools)                                        | Terminal when `convert` exits.         |
| `finalize`       | 📦 Finalize Model… (FinalizeModal, sidebar Tools)                                      | Terminal when `finalize` exits.        |
| `update`         | ⬆️ Update Model… (UpdateModal, sidebar Tools or config / checkpoint right-click)        | Terminal when `update` exits.          |
| `model`          | Run on a model config (config_class `type.model`)                                      | Terminal when `forgather model` exits. |
| `dataset`        | Run on a dataset config (config_class `type.dataset`)                                  | Terminal when `forgather dataset` exits.|

Helpers live in `inference_ops.py`, `eval_ops.py`, `tensorboard_ops.py`,
`mkdocs_ops.py`, `convert_ops.py`, `finalize_ops.py`, `update_ops.py`,
`model_ops.py`, `dataset_ops.py`, `dataset_server_ops.py` (build argv)
and `launcher.spawn_*_process` (same sandbox as training but with the
right argv). The scheduler's dispatcher branches on `item.job_type` to
pick the spawn function; GPU accounting and re-attach logic are
unchanged. Long-lived web services (inference, tensorboard, mkdocs,
dataset_server) all surface their URL as a clickable link on the Jobs
card so the operator can jump straight to the running endpoint.

**Dataset-source selector**. Every job type whose subprocess pulls
training examples (`training`, `eval`, `model`, `dataset`) gains a
dropdown in its submit modal that picks where the loader fetches
from: **Local** (the in-process loader, default) or any
dataset_server the forgather_server knows about (spawned-locally
JobRecords + URLs registered under *Datasets → Servers → + Add
server*). The choice persists alongside the other overrides; if the
saved server has gone away by the time the modal re-opens it snaps
back to Local. Resolved server-side into `FORGATHER_DATASET_SERVER`
+ `FORGATHER_DATASET_SERVER_TOKEN` env vars and merged into the
spawn's `extra_env`. Cluster fanout applies the same env vars to
every peer (the master resolves once and broadcasts).

### Scheduling algorithm

Each scheduler tick (~2 s) runs this placement logic:

1. **Build the queue.** Read `queue.json`, sort items by priority
   descending, then by submission time ascending (so higher-priority
   jobs go first; FIFO within a priority band).

2. **Build the idle pool.** Start from every GPU and drop any that are:
   - **excluded** via `CUDA_VISIBLE_DEVICES` (set at server start);
   - **disabled** at runtime via the UI toggle (persists in
     `gpu_policy.json`);
   - already **reserved** for one of our `starting` / `running`
     JobRecords.

   External processes (the user's desktop compositor, an unrelated
   CUDA program, a hybrid C+G daemon like
   `gnome-remote-desktop-daemon`) are *not* consulted. Trying to
   classify arbitrary processes as "real compute work" vs "desktop
   rendering" turned out to be a tar pit: NVIDIA's proprietary driver
   routes graphics-with-CUDA-context daemons through the compute
   list, hybrid C+G processes show up there too, and any name-based
   allowlist is incomplete by construction. The escape valve for
   "I'm running unrelated work on this GPU and don't want Forgather
   touching it" is the disable button on the GPU card. Compute and
   graphics processes are still surfaced via NVML
   (`nvmlDeviceGetComputeRunningProcesses` /
   `nvmlDeviceGetGraphicsRunningProcesses`) for display in the UI
   and to gate the kill-process endpoint (which restricts itself to
   compute processes so it can't terminate the user's desktop).

3. **Per-item eligibility.** For each queue item, filter the idle pool
   to GPUs whose `min_priority` gate the item clears
   (`gpu.min_priority <= item.priority`). An item can't land on a
   reserved GPU unless it qualifies.

4. **Best-fit to threshold** is the key heuristic. Within the eligible
   set, prefer GPUs with the *highest* `min_priority` the item still
   clears. Tie-break by index ascending (determinism). Formally, sort
   eligible indices by `(-gpu.min_priority, gpu.index)`.

   Rationale: if a priority-10 job could run on either `gpu0` (no
   gate) or `gpu5` (gated `min_priority=10`), put it on `gpu5`. That
   leaves `gpu0` free for a priority-0 job that *can't* use `gpu5`.
   Without this bias, the high-priority job would happily grab `gpu0`
   and block the low-priority job behind it — defeating the whole
   purpose of having reserved the higher-threshold GPU.

5. **Skip, don't block.** If an item can't be placed (fewer eligible
   GPUs than it requested), skip it and continue with the next item.
   A head-of-queue item that's over-constrained (e.g. wants 8 GPUs
   when only 4 are idle) does not block items behind it that would
   fit. Item ordering is stable across ticks, so the skipped item is
   reconsidered every tick until its resources free up.

6. **Commit.** Take the first `requested_gpus` indices of the sorted
   eligible list, re-sort them by index for readability, remove them
   from the in-tick idle pool, and launch the item (moves it from
   `queue_store` to `job_records`, spawns `torchrun` with
   `CUDA_VISIBLE_DEVICES` set to the chosen indices).

What the algorithm intentionally does **not** do:

- **No preemption.** A running job keeps its GPU until it finishes.
  Raising a job's priority or setting a GPU's `min_priority` doesn't
  kick anyone off.
- **No backfill across priority bands.** If the head of the queue is a
  4-GPU job that can't fit, a 1-GPU job further down with lower priority
  *can* run ahead of it (because of "skip, don't block"). If they have
  the *same* priority, FIFO order is preserved. There's no attempt to
  reserve GPUs for the blocked high-priority item while smaller ones
  run — that would require pool-reservation bookkeeping that isn't in
  scope for the prototype.
- **No NUMA / PCIe-topology awareness.** Multi-GPU assignments are just
  the first N eligible indices after the best-fit sort.
- **No cross-node scheduling.** Every GPU is assumed to be on the same
  node. The `node` field on JobRecords / GpuInfo is set up so a future
  `NodeClient` abstraction can be slotted in without changing the
  dispatch logic.

### Jobs / TTY

- **Jobs tab** unifies two sources: JobRecords we launched (status
  `starting` / `running` / `done` / `failed` / `aborted`) and
  externally-discovered trainer endpoints from `~/.config/forgather/jobs/`.
  Merged by PID lineage, tagged with `source = record | merged |
  endpoint`.
- Training-job cards show live status pills (loss, lr, grad_norm, epoch,
  tok/s, tokens, peak mem) plus a progress bar derived from
  `global_step / max_steps`. Non-training job types show a compact row
  with their identifying params (model path, port, etc.).
- Per-job control buttons forward to the trainer's HTTP endpoint:
  Save checkpoint / Save & stop / Graceful stop / Abort. **Kill** sends
  SIGTERM to the local process group (works for our jobs even
  pre-correlation). **Force kill** (right-click → "☠ Force kill
  (SIGKILL)") sends SIGKILL to the process group as a last-resort
  escape hatch for hung torchrun groups that won't respond to SIGTERM.
  Eval / inference / tensorboard jobs have no trainer-control endpoint,
  so only Kill / Force kill apply.
- **Bulk cleanup**: a `🧹 Cleanup completed` button at the top of the
  Jobs tab sweeps every terminal record (`done` / `failed` / `aborted`)
  via `POST /api/jobs/cleanup`. Captured TTY files are kept until the
  record is removed, so per-job `🗑` on a finished row still works too.
- **Dead endpoint visibility**: by default the Jobs list filters out
  endpoint-only entries whose PID is dead/zombie/recycled — those are
  trainer-control directories left behind by an earlier Forgather
  server instance. Toggle **Include dead endpoints** on the panel
  header to see them; right-click → **✕ Remove stale endpoint**
  rmtree's the directory under `~/.config/forgather/jobs/` so the entry
  stops surfacing. Live endpoint-only entries (foreign trainers) are
  still shown but offer no actions — those aren't ours to evict.
  Zombie-PID detection respects ``STATUS_ZOMBIE`` properly; a
  process that has exited but hasn't been reaped is treated as
  dead, not running.
- **Split-pane TTY**: toggle "⊞ Show TTY" to split the Jobs view; click
  a job to route its TTY output to the bottom pane. Draggable handle
  resizes (persisted to `localStorage`); double-click to reset to 45%.
- TTY stream subscribes to `WS /api/jobs/{id}/tty` — backlog then poll-
  follow. The backlog is read in 1 MiB chunks so a large log doesn't
  OOM the server; the one-shot REST dump (`GET /api/jobs/{id}/tty`) caps
  at the trailing 32 MiB of the captured file. Imperative
  `appendChild(textNode)` so browser text selection survives new chunks
  streaming in (lets you copy log lines from a running job). Once the
  trainer registers `logs_dir`, the captured TTY is symlinked into
  `<logs_dir>/tty.log` for durability alongside the trainer's other
  artifacts.
- Per-card hide/restart aware: server restart marks orphaned-but-still-
  alive processes as re-attached and continues monitoring them.

### Inference panel

An in-browser replacement for `forgather inf client` that talks to
running inference-server jobs (or any OpenAI-compatible endpoint).
Four sub-tabs sharing the same `InferenceState` (base URL, model,
generation params — persisted to `localStorage`):

- **Model** — base URL entry with a reachability test, picker for
  **Running inference servers** (auto-fills URL from inference job
  params), model-list fetch against `/models`, a **Generation
  parameters** form covering the OpenAI-named fields plus a wide
  selection of HuggingFace `GenerationConfig` extensions (`min_p`,
  `penalty_alpha`, `num_beam_groups`, `epsilon_cutoff`, etc.) with an
  expandable Advanced section. Tri-state selects let the user override
  `do_sample` / `early_stopping` explicitly rather than being stuck with
  temperature-derived defaults.

  In **cluster mode** the picker queries `/api/cluster/inference_servers`
  (master-aggregated) so jobs running on any reachable cluster peer
  appear alongside local ones — see the "Cluster inference picker"
  subsection below. The picker waits for the cluster-mode gate to
  resolve before showing rows, so it never flickers from local-only
  to the full cluster set on first paint. Outside cluster mode the
  picker queries `/api/jobs` directly. Rows that came from the cluster
  endpoint also show a short peer-id badge and a ⚠ indicator when the
  master's health-poll reports the server unhealthy.
- **Completion** — textarea + Send/Stop/Clear. Streams via
  `POST /v1/completions` (SSE) with an async iterator; `stream`
  checkbox falls back to a one-shot `stream: false` POST so beam-search
  and other streamer-incompatible modes work. Status line reports
  tokens + elapsed seconds; abort cancels the underlying fetch.
- **Chat** — multi-turn chat against `/v1/chat/completions`. Stateless
  wire format (client sends full `messages[]` each turn). Collapsible
  system-message disclosure at the top, transcript with `ReactMarkdown`
  for assistant turns and preserved-whitespace monospace for user turns,
  multi-line compose with Ctrl/Cmd+Enter to send. Regenerate-last,
  per-message edit (truncate + re-run), per-message delete. History
  + system text persist under `forgather-inference-chat-v1`.
- **Analyze** — per-token causal-LM scoring + visualization. Sends
  `echo=true, logprobs=K, max_tokens=0` to `/v1/completions`, which
  runs a single forward pass and returns per-token logprobs + top-K
  alternatives — see the inference server's "Per-Token Scoring"
  section. Split layout: textarea on the left, color-coded token
  output on the right, with optional histogram below. Both splits
  have drag handles (double-click to reset); state persists under
  `forgather-analyze-prefs`.
  - **Metric**: `loss` (default) or `entropy`. Entropy is a Forgather
    extension that needs the full vocab distribution, so the server
    returns it as a non-standard `token_entropies` field; if a
    non-Forgather server (vLLM, OpenAI) is in use the panel falls
    back to loss for coloring and shows a one-line note.
  - **Colormap**: viridis (default), plasma, magma, inferno, turbo,
    or a fallback green→red HSL ramp. Implemented via degree-6
    polynomial fits in `colormaps.ts` — no LUT, no dep. Foreground
    text color is auto-picked per-token from background luminance.
  - **Scale**: `auto` (5th/95th percentile of the response's own
    values, outlier-robust) or `manual` (fixed `min`/`max` so the
    same color encodes the same value across runs).
  - **Smooth**: exponential moving average over the color-encoded
    signal, α in `[0, 1)`. 0 = off; higher = smoother. Tooltip values
    stay raw — smoothing is purely a coloring aid.
  - **Histogram**: SVG (resolution-independent, `viewBox` + `width:
    100%`), 30 bins over the raw metric values. Bars are colored
    using the same colormap and scale so the histogram doubles as a
    legend.
  - **Selection-aware**: if the user has a non-empty selection in
    the textarea, Analyze scores only the selected substring instead
    of the whole input. Button label changes to `Analyze selection
    (N ch)` so the mode is unambiguous. Lets the user probe a single
    paragraph out of a pasted novel without truncation.
  - **Tooltip on hover**: shows the actual token, loss, perplexity,
    entropy (if available), and the ranked top-K alternatives with
    their probabilities. Rendered as `position: fixed` with
    flip-above/below + viewport-edge clamping so it escapes the
    pane's `overflow: auto` clip and never gets cropped.

**Inference… (sidebar Services section)** — opens `InferenceModal`
in ad-hoc mode: the model path becomes a `PathField` instead of a
read-only summary, so the user can serve any on-disk directory without
a Forgather project. Ad-hoc settings (path, port, dtype, attention
impl, cache impl, compile flags, chat template, checkpoint path)
persist under `forgather-adhoc-inference-v1` — the next invocation
defaults to the last-submitted values. Requested GPUs and priority
stay fresh each invocation since the "right" value depends on current
queue occupancy.

**Multi-model UX (ad-hoc only).** A "+ Add model" button under the
model PathField adds another row. Each row carries a path picker and
an optional name override (auto-derived from the basename when blank);
the picker remembers the parent of the last-picked path under
`pathfield.last:inference.model` so each row reopens at the same
directory the previous one came from. With more than one row the rows
live in a scrollable container (`max(240px, 30vh)` cap) so a long list
doesn't overflow the modal. A `Keep all models on GPU (no CPU swap)`
checkbox surfaces in the same UI; it maps to `--keep-on-gpu` on the
spawned server. The specific-checkpoint-path field is hidden in
multi-model mode (the server rejects `-c <PATH>` with more than one
model — the boolean `from_checkpoint` toggle still applies globally
and loads each model's latest checkpoint).

The "Create service…" suggested name is the single model's basename
in one-model mode, or `host-port` in multi-model mode — concatenating
many model names produces ugly long names quickly.

### Cluster inference picker

Mirror of the dataset-server cluster inventory in
`cluster_dataset_inventory.py`, sized for inference. Implemented in
`cluster_inference_inventory.py`:

- `LocalInference` (per-peer enumeration): scans `job_records` for
  `job_type == "inference"` in the `starting` / `running` state.
  Single-model jobs derive a one-element `models[]` from
  `model_path`'s basename; multi-model jobs use the configured `-m`
  names. `0.0.0.0` binds are rewritten to the cluster identity's
  hostname; pure-loopback binds stay in the list but are flagged
  `loopback=true`.
- Master aggregation: two async loops self-gating on master role:
  - `master_collect_servers_loop` every 10 s pulls each peer's
    `/api/cluster/inference_servers_local` and merges into a master
    snapshot.
  - `master_health_loop` every 10 s probes each server's `/health`
    (unauthenticated, root-mounted on the inference server) and
    updates the `healthy` flag.
- Endpoints (all under `/api/cluster/`):
  - `GET /inference_servers_local` — per-peer view, tokens included,
    peer-mTLS allowed (the master polls this).
  - `GET /inference_servers` — master-aggregated browser-facing list.
    On non-master nodes the route proxies to the master so every
    webui sees the same set. **Includes the bearer token** — see the
    discussion in the route's docstring; the picker carries the
    token via `X-Inference-Auth-Token` so the proxy can dial off-host
    upstreams from any cluster node.
  - `POST /inference_servers/refresh` — wake hook called by the
    scheduler on inference-job spawn / reap / abort so the picker
    reflects the change in ~1s rather than waiting for the 10s
    collect tick. Also surfaced manually via the webui (not currently
    wired to a button, but reachable for operator-driven refreshes).

Auth token surface: `auth.py:_PEER_ALLOWED_PATHS` carves out
`inference_servers_local` and `inference_servers`;
`auth.py:_PEER_ALLOWED_MUTATIONS` carves out
`inference_servers/refresh`.

Token-stripping the browser response (so remote-peer tokens never
leave the master) is tracked as a follow-up; the current behavior
matches `/api/jobs`, which already ships per-job tokens to the
authenticated session.

### Cluster DiLoCo inventory

Same pattern as the dataset and inference cluster inventories, applied
to DiLoCo parameter servers so a server spawned on any cluster peer
surfaces in every other peer's `forgather diloco servers` output and
DiLoCo panel — no operator token copying. Implemented in
`cluster_diloco_inventory.py`:

- `LocalDiLoCo` (per-peer enumeration): two sources merged on
  `server_id` collision (JobRecord wins):
  1. `JobRecord(job_type == "diloco_server")` rows in `starting` /
     `running` state.
  2. The user-added registry at
     `<config>/server/diloco_server_registry.json` — the
     `forgather diloco register <url>` escape hatch for endpoints
     mDNS can't see (WAN, SSH tunnel).
  `0.0.0.0` binds are rewritten to the cluster identity's hostname
  (or the scheduler-stamped `routable_host`); loopback binds stay
  in the list flagged `loopback=true`.
- Master aggregation: two async loops self-gating on master role:
  - `master_collect_servers_loop` every 10 s pulls each peer's
    `/api/cluster/diloco_servers_local` and merges into the master
    snapshot.
  - `master_health_loop` every 10 s probes each server's `/health`
    (unauthenticated, root-mounted on the DiLoCo server, matching the
    inference convention).
- Endpoints (all under `/api/cluster/`):
  - `GET /diloco_servers_local` — per-peer view, tokens included,
    peer-mTLS allowed (the master polls this).
  - `GET /diloco_servers` — master-aggregated browser-facing list.
    On non-master nodes the route proxies to the master so every
    webui sees the same set. **Includes the bearer token** — the
    DiLoCo proxy at `routes/diloco.py` resolves it server-side from
    the master snapshot when forwarding requests upstream, so the
    browser never carries the upstream bearer.
  - `POST /diloco_servers/refresh` — wake hook called by the
    scheduler on diloco_server-job spawn / reap / abort and by the
    registry add/delete handlers so the panel reflects the change
    in ~1 s rather than waiting for the 10 s collect tick.

Auth token surface: `auth.py:_PEER_ALLOWED_PATHS` carves out
`diloco_servers_local` and `diloco_servers`;
`auth.py:_PEER_ALLOWED_MUTATIONS` carves out
`diloco_servers/refresh`. The escape-hatch registry endpoints
(`/api/diloco/registry`) remain authenticated browser-only — only
the aggregated inventory crosses peer boundaries.

See `docs/design/diloco-security.md#cross-node-discovery-cluster-inventory`
for the design rationale and end-to-end token-transit walkthrough.

**Operator note on SSRF posture.** Cluster-attested DiLoCo URLs are
implicitly added to the DiLoCo proxy's outbound allowlist, so any URL
a cluster peer attests to (via its `/diloco_servers_local`) becomes
reachable through this node's `/api/diloco/*` proxy. The dataset
proxy does not do this — DiLoCo widens the surface to make peer-
spawned servers inspectable from any node's webui without operator
URL-copying. Operators who don't trust every node that could join
the cluster's mTLS root should keep that root tight. The threat-model
deviation is laid out in `docs/design/diloco-security.md`'s
*Threat-model deviations* section.

**Generation presets** — save/load named JSON presets of the current
generation params. Served by `/api/generation-configs/*`, which merges
two layers: bundled examples under `<repo>/generation_config/` (read-
only: `greedy`, `precise`, `balanced`, `creative`, `beam_search`,
`contrastive`) and user presets under `~/.config/forgather/generation_config/`
(writable; shadows same-named bundled entries). Delete on a built-in
returns 403 with guidance; delete on a user shadow restores the
built-in.

**Browser → inference-server proxy** (`routes/inference_proxy.py`) —
the webui can't hit spawned inference servers directly without running
into CORS / Private Network Access / extension-blocking. Everything
routes through same-origin `/api/inference/*`; the proxy forwards to
whichever base URL the caller names, streaming byte-for-byte so the
SSE framing reaches the browser unchanged. The proxy accepts any
HTTP/HTTPS host the operator types into the panel — forgather is a
single-user research tool, the proxy is auth-gated by the same token
that gates training-job submission, and an authenticated attacker
already has full RCE on the host (a job can shell out and exfiltrate
anything). An SSRF guard on this endpoint adds friction without
adding security. The expected workflow is "vLLM on another box"; the
proxy is built around that. For operators who want stricter posture
(e.g. forgather behind a multi-user gate), pass
`--lock-inference-proxy` to `forgather server` to restrict the proxy
to `127.0.0.1` / `localhost` / `::1`. The scheme guard (http/https
only) is unconditional regardless of lock state.

### Datasets view

Top-level webui tab (sidebar 🗂 **Datasets**) for inspecting and
managing the dataset_servers a training run might pull from. Two
sub-tabs sharing the local + user-added server lists. The
cluster-wide *Cluster* sub-tab was moved to the
**Cluster view → datasets** tab — this surface is intentionally
per-node only:

- **Servers** — left list of **Spawned dataset servers** (locally-
  launched JobRecords, auto-discovered) and **User-added servers**
  (URLs registered via *+ Add server*). Add/delete dialog for user
  entries; **Copy bundle** on each alive spawned row emits a
  `forgather-dataset://host:port/?token=…` URI to the clipboard,
  and the *+ Add server* modal has a matching **Paste bundle**
  affordance for one-step cross-host transfer.

  Selecting a server reveals three typed renderers loaded
  concurrently, with a single **↻ Refresh** button that re-fetches
  all three at once:
  - **Status** — colored policy chips (auth required/disabled, HF
    cache enabled/disabled, paths off/allowed, downloads off/
    allowed) with tooltips explaining each setting.
  - **HF Cache** — sortable table with a horizontal stacked
    size-distribution bar above it. Each split name in the splits
    cell is a clickable link that opens that split in Explore.
  - **Local** — same shape (table + chart + per-split click-
    through). Registered `local/<name>` mappings are enriched
    server-side with split metadata so the webui shows the same
    row counts / features / size info HF cache entries get.
- **Explore** — hierarchical tree (server → HF cache / local →
  repo → config → split) with a paged preview table on the right
  for the selected split. Tree is lazily expanded; click-to-expand
  individual rows in the preview table bumps the per-cell
  truncation cap. The browse pane has a draggable vertical
  divider — drag to resize, double-click to reset, ←/→ to nudge
  (Shift for x4); width persists in localStorage. Pager elides
  the middle (`‹ Prev 1 … 42 43 44 … 588 Next ›`) with a 🎲 button
  that jumps to a random page and expands a random row on it
  (handy for sampling a large corpus); 25 / 50 / 100 rows-per-page
  selector plus a **Go to** input for jumping directly to a page.

  Row expand-on-click ignores drag-select gestures: if the user
  moves the cursor more than ~4 px between mousedown and mouseup,
  or releases with a non-empty selection, the row stays expanded
  so they can copy text out of it.

  Each cell has a right-click context menu with:
  - **Copy cell text** — writes the full underlying value (not the
    truncated displayed form) to the clipboard via `navigator.
    clipboard.writeText`. Non-string values are JSON-stringified.
  - **Analyze in Inference…** — jumps to *Inference > Analyze*
    with this cell's full text already loaded into the textarea and
    scoring kicked off, one click. Wired through an App-level
    `pendingAnalyze: {text, key}` slot (mirrors the existing
    Cluster→Datasets `pendingExplore` pattern); the key nonce
    dedups so flipping tabs back doesn't re-fire stale requests.

  Cross-view click-through: clicking a row in the *Cluster view →
  datasets* tab opens this Explore tab with the first healthy host's
  first config/split pre-resolved and selected. If the chosen server
  doesn't have the dataset cached (or has no enumerable splits yet),
  the right pane shows a yellow `couldn't resolve` hint instead of
  silently appearing empty.

**Dataset… (sidebar Services section)** — opens the
DatasetServerModal: host, port, no-auth toggle, loading-policy
flags (`--no-hf`, `--allow-paths`, `--allow-downloads`), a
repeatable Local-mapping form (`name=path`), and an optional
config-file path. Spawned dataset_servers join the regular Jobs
view with the same URL + token surfacing inference jobs get. The
generated bearer token is **persisted** across restarts (mirroring
`forgather server`'s `auth_token`) so peers keep working after a
server reboot; pass `--regen-token` to the underlying script (or
re-spawn from this modal after deleting the per-port `.token` file)
to rotate.

**Browser → dataset_server proxy** (`routes/dataset_server.py`) —
same-origin proxy for the `/v1/*` endpoints. Unlike the inference
proxy (localhost-default), this proxy's SSRF allowlist is the user
registry itself: loopback always, registered URLs always,
everything else 403 with a "register first" hint. The registration
step is the explicit operator consent. See the module docstring
for the threat-model details, including the small bearer-
amplification it acknowledges.

### DiLoCo server

DiLoCo follows the same model as dataset_server: per-port persisted
bearer token (`~/.config/forgather/diloco_server/<port>.token`,
mode `0o600`) generated on first run and reused across restarts;
`--regen-token` rotates. The DiLoCoPanel form for adding an external
server accepts `auth_token` (masked) and `verify_tls`, and the
`routes/diloco.py` proxy attaches `Authorization: Bearer …` via the
standard JobRecord-then-registry precedence.

Two distinct security planes:

* **Control** (`/register`, `/heartbeat`, `/control/*`, `/status`,
  `/info`, work-queue endpoints) — always TLS + bearer-required.
* **Bulk** (`/submit_pseudograd`, `/submit_fragment_pseudograd`,
  `/global_params`) — opt-in second listener via `--bulk-cleartext`
  (a single toggle, surfaced in the DiLoCo server modal). Always
  cleartext + no-auth on a server-picked ephemeral port; its only
  purpose is to bypass TLS for throughput on a trusted LAN. Workers
  learn the ephemeral port from the `X-Forgather-Bulk-Url` header on
  `/register` (delivered over the TLS control plane), so there's no
  port for the operator to choose or distribute. RCE protection is
  independent: every inbound tensor blob uses `weights_only=True`.

mTLS works the same as it does for `forgather server`: when TLS is
enabled with a cluster CA bundle present, a client presenting a
CA-signed cert at the handshake is treated as cluster-authenticated
and the bearer check is skipped. Full design notes:
`docs/design/diloco-security.md`. Operator setup:
`docs/operations/tls.md` (the "DiLoCo server" subsection).

### GPUs

- NVML-driven: per-card name, memory, util, temp, power, compute PIDs.
  Live updates via `WS /api/gpus/stream` (~2 s cadence, with REST
  prime).
- GPU↔job attribution: process chips on each GPU card map back to live
  jobs (chip turns blue + shows the config name when matched).
- **Three non-schedulable states**, visually distinct:
  - **Excluded** (red dashed border + `EXCLUDED` badge): filtered out
    via `CUDA_VISIBLE_DEVICES` at server start. Static.
  - **Disabled** (amber dashed border + `DISABLED` badge):
    runtime-toggled by the operator via the UI. Reversible, persists
    via `gpu_policy.json`.
  - **Priority-gated** (blue `≥N` pill): a minimum-priority threshold
    for scheduling. Only jobs with `priority >= N` get placed on the
    GPU. `0` means no gate.
- **Left-click a GPU card** toggles `disabled`. Excluded cards ignore
  clicks.
- **Right-click a GPU card** opens a context menu:
  - Enable/Disable GPU (same as left-click).
  - Set minimum priority… (prompt; integer validation).
  - Clear priority gate (shown when > 0).
  - ☠ Kill all N processes (SIGKILL) — last-resort cleanup for wedged
    ranks. Confirm dialog enumerates each PID and tags any that match
    one of our jobs (`pid 12345 (config_name)`). Hits **every** process
    on the GPU, including ones we didn't launch. Proceeds through
    `POST /api/gpus/{index}/kill` which requires `{confirmed: true}`.
- **Right-click a Job card** opens a context menu:
  - **☠ Force kill (SIGKILL)** for live server-launched jobs that
    aren't responding to SIGTERM — routes through a `force-kill`
    control action. Backend polls for the PID to actually exit
    (up to 2 s) and stamps the JobRecord's ``error`` field if it's
    still alive afterwards, so a stuck-in-CUDA process surfaces
    instead of silently leaving a phantom GPU consumer.
  - **✕ Remove stale endpoint** for endpoint-only entries whose
    PID is dead/zombie/recycled — backend rmtree's
    ``~/.config/forgather/jobs/job_<id>/`` so the entry stops showing up
    in the Jobs list. Live endpoint-only entries (foreign trainers
    we didn't launch) still show "No actions" — those aren't ours
    to evict. Toggle "include dead endpoints" on the Jobs panel
    header to see dead entries in the first place; the default
    view filters them out.

### Filesystem helpers

- Directory browser modal (used by Add Search Root, the `path`-type
  dynamic-args picker, and the New Workspace / New Project parent
  pickers) with quick-jump chips for Examples / Forgather repo / Home,
  supports show-hidden, navigate-by-double-click, click-to-pick on
  files, and a **+ New Folder** chip in the quick-row that calls
  `POST /api/fs/mkdir` on the current path and auto-navigates into
  the freshly-created directory. Bare-name validation server-side
  (no path separators, no `.`/`..`, no overwrite) keeps a single
  invocation to a single new directory.
- Asset endpoint with strict path-safety (resolved-target-must-stay-
  inside-project, `..` blocked, symlink containment check, 50 MiB cap)
  used to serve images embedded in the project README.

### Workspace creation

The **📁 New Workspace…** button in the Search Roots section
(alongside Browse…) opens `NewWorkspaceModal`, the in-app equivalent
of `forgather ws create`. Required: Parent (search-root dropdown,
auto-defaults to the first existing root), Name, Description, and
Forgather dir (auto-defaults to the bundled "Forgather repo"
quick-path). Optional: Workspace dir (relative to parent; nested
paths supported via `mkdir -p`; Browse… anchored to the chosen
parent lets the user pick an existing subdirectory and drops a
trailing-`/` relative path into the field), Libraries (newline-
separated, pre-filled with `base` + `examples` since every
workspace in the repo uses that pair), Additional search paths
(newline-separated absolute paths). The dropdown carries an extra
**+ Create new search root…** option that swaps in an inline
sub-form (existing parent dir + bare name); on submit the server
mkdirs the target and registers it as a search root in one shot
(`POST /api/search-roots {path, create: true}`), then auto-selects
it as the parent.

Submit calls `POST /api/workspace/new`, which validates that the
parent matches a configured search root exactly, slugifies the
workspace dir basename if not provided (CLI-matched: spaces -> `_`,
lowercased, dots stripped), splits and rejects any `..`/`.`
segments, runs an `os.path.commonpath` containment check against
the parent, then dispatches to `forgather.cli.workspace.ws_create_cmd`
via a `SimpleNamespace`. `os.makedirs` (called by the CLI) handles
intermediate-directory creation for nested paths.

Fresh workspaces appear in the project tree because discovery
walks for `forgather_workspace/` markers in addition to `meta.yaml`
projects (see "Project / config discovery" above) — empty
workspaces seed empty clusters that still render.

### Right-click context menus

The project tree exposes a different menu per node type:

- **Workspace row** — **📁 Create Project…** plus a trailing
  **🗑 Delete Workspace…**. Create-Project opens
  `NewProjectModal`, the in-app equivalent of
  `forgather project create`: required Name + Description, plus
  Config prefix (default `configs`), Default config (placeholder
  derived from the picked scaffold when applicable, otherwise
  `default.yaml`), Project dir (relative to workspace; may be
  nested with `mkdir -p` semantics; Browse… button anchored to
  `workspace_root` lets the user pick an existing subdirectory and
  drops the relative path back into the field with a trailing `/`
  for the leaf name), and a **Starting point** fieldset that
  controls how the project's first config gets seeded — tri-state:
  - **Blank** (default): server writes the built-in empty stub.
  - **Copy from existing file**: a `PathField` for a source config.
  - **Use a scaffold**: renders `MetaTemplatePicker` filtered to
    `target_kind: "config"` plus `MetaTemplateFields` for the
    selected scaffold; the user enters values and submission
    sends `meta_template` + `values` to the server, which renders
    the scaffold and uses the result as the seed.

  The Default-config filename placeholder is auto-derived from the
  scaffold's `CONFIG_NAME` value when one is picked, so the new
  project's first config is named meaningfully (e.g. `c4.yaml`)
  instead of always landing as `default.yaml`. Typing into the
  field overrides the auto-derivation. Submit calls `POST
  /api/workspace/new-project`, which dispatches into
  `forgather.cli.project.project_create_cmd` via a `SimpleNamespace`
  so we don't duplicate the CLI's project-skeleton logic — for the
  scaffold path the server renders the meta-template up front and
  passes the result as `seed_text` instead of going through the
  CLI's `--copy-from` file read. `copy_from` and `meta_template`
  are mutually exclusive (server enforces with 400). Tree refresh
  is via `["projects"]` invalidation. The
  synthetic "Unaffiliated" cluster (no `workspace_root`) doesn't
  receive the menu. Delete-Workspace recursively removes the
  workspace directory via `POST /api/fs/delete-dir`, with the same
  two-step gate as Delete-Project (standard `confirm()` plus a
  typed-token prompt requiring the workspace's directory basename),
  since deleting a workspace cascades to every project, config, and
  in-tree output_models within it.
- **Project row** — **📄 New Config…** / **📄 New Template…**.
  Both open a `NewTemplateModal` (shares the chrome with
  `CleanOutputModal` et al.). The modal is a two-step flow:
  1. **Pick a starting point.** A two-pane picker shows
     **Blank file** plus the meta-template tree (scaffolds shipped
     under `templatelib/meta/`, fetched via
     `GET /api/project/meta-templates`). Selecting a leaf reveals
     its title, description, and the list of fields it will ask
     for in step 2.
  2. **Configure.** The familiar name input + base-dir preview,
     and — when a scaffold was picked — one form field per variable
     declared in the scaffold's manifest (label, helper text,
     default pre-filled, required fields starred). Subdirectory
     creation under the configs / templates root is handled by
     typing a nested name (e.g. `experiments/foo.yaml`) — `mkdir
     -p` semantics on the server.
  
  The base path comes from `GET /api/project/template-paths`
  (`MetaConfig.searchpath[0]` for templates, plus `config_prefix`
  for configs). Submit calls `POST /api/project/new-template` —
  with `meta_template` + `values` when a scaffold was picked, or
  bare when "Blank file" was selected — invalidates the project
  tree and `project-templates` queries so the new file shows up
  in `tlist`, then hands the returned path to the Edit panel via
  the App-level `onEditTemplate` hook — the user lands directly
  on the editor for the new file (scaffold pre-filled, or blank). A trailing **🗑 Delete
  Project…** entry recursively removes the project directory via
  `POST /api/fs/delete-dir`; it's gated by both a standard
  `confirm()` and a typed-token prompt requiring the user to type
  `yes`, since the project tree often contains an `output_models/`
  subtree (runs / checkpoints) that the regular Clean Output flow
  won't touch. (Workspace delete keeps its stricter basename gate
  because it cascades to every project under it.) The confirm body
  spells out that outputs configured to live *outside* the project
  tree are not affected. After delete `["projects"]`,
  `["project-templates", dir]`, and `["project-models", dir]` are
  invalidated, and the active selection is dropped if it was
  pointing into the deleted project.
- **Config row** — Run / TensorBoard / Overrides plus, when the
  config has actually been run, Clean Output (gated on
  `configOutputDir`'s `output_dir_exists` — `output_dir` is
  per-config and can live anywhere on disk, so the menu polls the
  resolved path rather than guessing from `output_models/`).
  Serve Inference / Evaluate / Convert Model / Finalize Model
  surface when the config has checkpoints on disk. Convert and
  Finalize pre-fill the source path with the config's resolved
  `output_dir` while inheriting every other field from the global
  tool's persisted defaults; submit then writes everything
  (including the new source path) back, so the next opening —
  global tool or context-menu — reflects the last run. Items are
  filtered by `config_class` so non-training configs only show
  **Overrides**. **⎘ Duplicate Config…** prompts for the new
  filename (defaulting to `<stem> (copy)<ext>`) and copies the
  config file alongside the original via `POST /api/fs/copy` with
  `target_name`; the new entry appears in the tree immediately on
  `["projects"]` invalidation. A trailing **🗑 Delete Config…**
  entry unlinks just the config template file (via
  `POST /api/fs/delete-file`); it explicitly does *not* touch the
  config's `output_dir` / runs / checkpoints — those have their
  own Clean Output / Delete Permanently flows. After delete the
  `["projects"]` and `["project-templates", …]` queries are
  invalidated so the tree and the `tlist` view both refresh, and
  the active selection is cleared if it pointed at the deleted
  file.
- **Checkpoint leaf** — Serve Inference / Evaluate (both pre-fill the
  modal with this checkpoint's path), plus Delete Permanently.
- **Log leaf** / **Evaluation leaf** — Delete Permanently.
- **Logs / Checkpoints / Evaluations group header** — Delete All
  Permanently (atomic subdir deletion: one call to `/api/fs/delete-dir`
  on the parent directory rather than N per-leaf calls).

Destructive paths route through two sibling endpoints:
`POST /api/fs/delete-dir` (recursive directory removal, used by Clean
Output and the artifact-leaf / group menus) and
`POST /api/fs/delete-file` (single regular-file unlink, used by
Delete Config). Both require `confirmed: true`, reject symlinks,
require absolute paths, and enforce a ≥4-path-component depth floor;
the directory variant additionally checks against a denylist of
common system roots (`/`, `/home`, `/etc`, …) — the file variant
relies on the depth floor alone since you can't recursively wipe a
file.

## AI agent (right-sidebar assistant)

An in-process AI agent that helps inspect projects/configs, answer
questions about Forgather (with doc citations), and author configs and
templates. **Disabled until a profile is configured** — add one in the
webui (the agent sidebar's ⚙ button) or seed one from `server_config.yaml`.

### Provider

The agent uses the **Anthropic SDK as a single path** for both Claude and
local models. vLLM natively serves the Anthropic Messages API
(`/v1/messages`), so pointing `base_url` at a vLLM server reaches a local
model (e.g. Qwen3) through the same code path as Claude. A thin internal
`ChatProvider` seam (`agent/providers/base.py`) keeps the loop
provider-agnostic; an OpenAI-style adapter can be added later for
non-Anthropic-API providers.

For a local vLLM model, launch vLLM with
`--enable-auto-tool-choice --tool-call-parser <parser>` (e.g.
`qwen3_coder` for Qwen3.6, `hermes` for older Qwen3) and a
`--served-model-name` alias (the Messages API rejects model names
containing `/`).

The credential is sent as the right header automatically: a profile with
a `base_url` (a local vLLM server) sends `Authorization: Bearer <token>`
(what vLLM checks), while Claude (no `base_url`) uses the `x-api-key`
header. So a vLLM bearer goes in the same `api_key` / `api_key_env`
field.

### Profiles (multiple agents, hot-swappable)

A **profile** is one named way to reach a model (provider, model,
base_url, credentials, TLS posture). Manage several from the webui (agent
sidebar ⚙ → "Agent profiles") and switch the active one from the header
dropdown — changes take effect on the **next message, no server restart**
(the runtime rebuilds its loop whenever the profile store's revision
changes). Profiles persist at `<config>/server/agent_profiles.json` (mode
0600 — the file holds API keys / bearer tokens).

The `server_config.yaml` `agent:` block is only a **bootstrap**: on first
run, if no profiles exist, one is seeded from it. Edit profiles in the UI
thereafter.

```yaml
agent:
  provider: anthropic        # only adapter today
  model: claude-sonnet-4-6   # or a vLLM --served-model-name alias; blank = auto
  base_url: null             # e.g. https://kitt:8000 for local vLLM; null = Claude
  api_key_env: ANTHROPIC_API_KEY   # env var with the key (or vLLM bearer)
  # api_key: null            # explicit key/bearer (overrides api_key_env)
  # verify_tls: true         # see TLS below
  # max_tokens: 4096
  # max_iterations: 12       # tool-use loop cap per user message
  # prompt_caching: auto     # auto (on for Claude) | on | off — see below
```

### TLS for local (self-signed) servers

A vLLM box on a LAN typically serves TLS with a self-signed cert that the
system CA check rejects. Two per-profile options (same model as the
inference registry):

- **Verify TLS off** — accept any certificate. One toggle, but vulnerable
  to a man-in-the-middle on the path. Fine for a trusted LAN.
- **Import the certificate** (recommended) — the editor's "Import
  certificate…" button fetches the server's cert (trust-on-first-use),
  shows its SHA-256 fingerprint for you to confirm, and stores the PEM in
  the profile. The agent then verifies against exactly that cert, with
  hostname checking off (the chain check is the real boundary on a LAN —
  same posture as forgather's own private-CA peers).

Both model-listing and the chat connection use the profile's chosen
posture (we do not silently disable verification anywhere).

> **Security stance (intentional — please don't "harden" this away).**
> For a self-hosted model server on a LAN, verify-off is a *supported,
> warned* posture, and importing the cert is one click. We deliberately
> keep the friction low: a local LLM's bearer token has little resale
> value (these tokens are often plaintext in a config file anyway), and
> capturing it requires actively MITM-ing the LAN without being noticed —
> a high bar that even verify-off raises far above plain HTTP (which stops
> packet-sniffing outright). Making LAN TLS painful is counter-productive:
> it pushes people to plain HTTP, which is strictly worse. **Real Anthropic
> (Claude) keys are different** — high value, real theft market — so the
> agent never sends them over an unverified channel: Claude always goes
> over the SDK's verified TLS, and `ANTHROPIC_API_KEY` is never
> auto-forwarded to a custom `base_url` (see *Credential safety* below). A
> token is also only ever sent to the exact server its profile names; it is
> never resolved by URL match or borrowed from another panel.

### Credential safety

- Agent profiles are **self-contained**: a profile's token is sent only to
  that profile's own `base_url`. The agent never resolves a token by URL
  match or borrows one stored in another panel (e.g. the inference
  registry), so a blank key never silently authenticates from an unrelated
  config.
- `/api/agent/models` will not redirect a saved profile's token to a
  different `base_url` supplied in the request — an edited URL must carry
  its own key.
- `ANTHROPIC_API_KEY` (the well-known env var) is auto-read only for Claude
  (no `base_url`); it is never forwarded to a custom `base_url`. A local
  server needs an explicit key or a deliberately-named env var.
- Authoring tools enforce the filesystem-root allowlist at **propose** time
  (before any read), so `propose_edit_config` can't read a file outside the
  configured roots into the preview; writes are re-checked at commit.
- The profile store (`agent_profiles.json`, which holds API keys / bearer
  tokens) is written **mode 0600** and is in the startup perm-tightening
  set, so only the owner can read it (matching the other credential-bearing
  registries).

### Model selection (weakly bound)

The model is only weakly bound to a profile. The editor's "Load models"
button queries the server's live list (Claude via the SDK; vLLM/OpenAI via
`/v1/models`) and offers a picker. Your choice is remembered; if a saved
model no longer exists on the server, the first available is auto-selected.
Leaving the model blank means "auto — first available", resolved at
activation time. This suits vLLM, which serves one model at a time, so
swapping the model on the box needs no profile edit.

### Output budget (`max_tokens`)

`max_tokens` defaults to **auto** (leave the field blank). vLLM enforces
`prompt_tokens + max_tokens ≤ max_model_len`, and a verbose reasoning model
easily burns through a small cap, so a hardcoded number is the wrong
default. Auto:

- reads the model's context window (`max_model_len`) from the server's
  model card — the same query that backs the picker, so you never look it
  up per model;
- sizes the output budget to `min(context, 32768)` (32K is a sensible
  ceiling even on huge-context models — Gemma 256K, NVIDIA 1M — leaving the
  rest of the window for the prompt);
- **clamps per request** so the output budget always fits the remaining
  context as the conversation grows (a deliberately high prompt estimate
  biases toward a smaller, safe budget rather than a "context length
  exceeded" error).

Set a positive value to pin an explicit cap instead.

The model-list probe needs the server's bearer token. Credential
resolution for a profile is self-contained: the profile's own key → its
`api_key_env` env var. (Agent profiles do not silently borrow tokens from
other panels — enter the token in the API key field; for vLLM it's
`~/.config/vllm/api-key`.) **High-value-key guard:** the well-known
`ANTHROPIC_API_KEY` env var is only auto-read for Claude (no `base_url`) —
it is never forwarded to a custom `base_url`, so a blank-key local/vLLM
profile can't leak your Anthropic key to the local box; a local server
needs an explicit key or a deliberately-named env var. A 401 from "Load
models" means the token is
missing or wrong. The probe always skips TLS verification (it only returns
model ids, so a not-yet-imported self-signed cert never blocks discovery);
the actual chat connection still honors the profile's TLS setting.

### Prompt caching & token metering

An agentic loop re-sends the **whole prefix** — system prompt + every tool
schema (~7K tokens here: ~2K system + ~5K for ~29 tools) + the full
conversation — on *every* API round-trip. A turn that ends with a 35K-token
transcript but took ~25 tool round-trips bills as **Σ (prefix size) over all
requests** — easily ~600K input tokens, dwarfing the transcript. Output is
tiny by comparison (the model mostly emits short tool calls). This is expected,
not a billing bug: a single-snapshot tokenizer estimate of the final transcript
can't see the round-trips, the re-sent system+tools, or thinking tokens.

Two mechanisms make this visible and cheap:

- **Metering.** The Anthropic adapter captures the full per-request usage
  breakdown — `input` (fresh), `cache_read`, `cache_write`, `output` — logs it
  per request (`agent request usage: …`), and the webui accumulates a
  cumulative **session-billed** total across every round-trip. The header meter
  shows two numbers: *context occupancy* (latest request, the window bar) and
  *billed N* (cumulative, with a tooltip breaking down input/cache/output and
  the cache-hit ratio). The billed number is the one that reconciles with the
  provider's dashboard.
- **Prompt caching.** With `cache_control` breakpoints on the stable head
  (system, which covers tools) and the last message (the growing history), the
  re-sent prefix bills at ~0.1x instead of full rate — typically cutting the
  input bill by ~80-90% for these loops. Controlled per profile by
  **Prompt caching**: `auto` (default — on for Claude, off for a custom
  `base_url`, since vLLM does its own automatic prefix caching and may reject
  `cache_control`), `on`, or `off`. Watch `cache_read` climb in the meter to
  confirm it's working.
- **Cost estimate.** The meter also shows `~$N` next to the billed total — the
  cumulative token categories multiplied by a per-model price table
  (`agent_pricing.py`, USD per million tokens; cache read 0.1x / write 1.25x of
  the input rate). It is an **estimate**, not a billing source of truth: the
  per-message API never returns a dollar figure, and Anthropic's dashboard /
  Cost Admin API (`/v1/organizations/cost_report`, Admin key) is authoritative.
  A model not in the table (e.g. a self-hosted vLLM model) shows no cost. Prices
  match by longest model-id prefix and resolve in three layers — webui override
  file > server-config `agent.pricing` > built-in defaults (as of 2026-06). The
  built-in rates were captured from
  `platform.claude.com/docs/en/about-claude/pricing`.

  Edit the rates from **Profiles… → Edit price table…** (next to Prompt
  caching). That writes `<config>/server/agent_pricing.json` — a
  `{"model-id-prefix": [input, output]}` map — and hot-reloads it (no restart).
  For headless setups, the same overrides can be seeded from the server-config
  `agent.pricing` block (lowest-priority layer):

  ```yaml
  agent:
    pricing:
      claude-opus-4-8: [5.0, 25.0]    # [input, output] USD per Mtok
      claude-haiku-4: [1.0, 5.0]
  ```

### Interaction model — propose → approve → commit

Tools are classified by risk and the gate is enforced **server-side**, in
the agent loop, never in the browser:

- **`read`** — inspection / search tools run automatically.
- **`propose`** — authoring tools compute a *preview only* (a before/after
  diff) and return it; nothing is written. The turn pauses with a pending
  action. The actual write runs only when you click **Approve**, replaying
  the exact previewed content (the model cannot alter it after the fact).
  Reject feeds a rejection back so the model can adapt. After a write,
  the config is re-parsed and any error surfaces back into the chat.
- **`confirm`** — side-effecting tools without a file diff (enqueue a
  training job, start/stop a service, delete a path, query a model). They
  return a structured preview (a curated `extra` summary plus, for jobs, the
  reconstructed command) and pause for **Approve** the same way.

Every paused action card also shows **the arguments the agent actually passed**
in the tool call, in an "Agent-proposed arguments" block, regardless of what the
tool curated into its summary. This is the authoritative view of what you are
approving: a tool's `extra` summary may omit, reword, or default-fill arguments,
but this block is the raw tool-call dict. It lists only the keys the agent
specified — arguments left to defaults are absent, so a tool with a large
optional-argument surface stays readable — while keeping explicit `null` /
empty-string values visible (a passed-but-blank arg is still an input). An
over-long value (e.g. a multi-thousand-char prompt) is clipped with a
`… (+N more characters)` marker. The loop attaches the block generically at the
gate (`Proposal.to_card`), so a newly added propose/confirm tool can never
silently hide an argument it was given.

Because a paused turn owes a tool-result for every tool call it made, the
loop refuses to call the model again until every pending action is
resolved — so a change can never become permanent without approval.

### UI — two surfaces, one conversation

- **Right sidebar** (toggle with **Ctrl/Cmd+J**): the always-on compact
  thread. Condensed action cards with Approve/Reject and an "Open in Agent
  view" link for large diffs.
- **Full "Agent" view** (left-nav 🤖): the same conversation with a Monaco
  side-by-side diff for reviewing proposed changes, plus full history.

Both share one controller, so an action proposed in the sidebar can be
approved in the full view and vice-versa.

### Tools

Grouped by risk (the gate in `agent/registry.py`):

- **Read-only** (`read`, run automatically) — project/config inspection:
  `list_workspaces`, `list_projects`, `list_configs`, `inspect_config`,
  `check_config`, `render_config_pp`, `render_config_code`,
  `list_config_templates`, `config_template_refs`, `list_meta_templates`;
  filesystem: `read_file`, `list_directory`, `find_files`, `search_docs`;
  scheduler/jobs: `scheduler_status`, `list_jobs`, `read_job_output`,
  `wait_for_job`; datasets: `list_dataset_servers`, `dataset_info`; UI:
  `reveal_in_ui`.
- **Authoring** (`propose` → preview → commit) — `propose_edit_config`,
  `propose_new_config` (scaffold / copy / inline content),
  `propose_new_project`, `propose_new_workspace`.
- **Plain-file management** — `create_file` (`confirm`: touch an empty
  markdown/notes/scratch file) and `edit_file` (`propose`: overwrite an
  existing file, shown as a before/after diff), plus `stat_path`,
  `delete_path`, `move_path`, `copy_path` (`confirm`). The authoring tools
  above are for Forgather configs/templates (they scaffold and parse-check);
  these are for everything else, reusing the same crash-atomic write
  primitives and fs-root / no-clobber / mtime guards.
- **Run-as-job** (`confirm` → approve → enqueue) — submit a config to the
  scheduler: `run_dataset` (build/inspect a dataset split), `run_construct`
  (materialize a named target, e.g. the model or a tokenizer), `run_train`
  (train the model — long-running, reserves GPUs). All three go through the
  same validated path as the HTTP enqueue route (`queue_ops.validate_and_enqueue`).

The registry (`agent/registry.py`) is the single source of truth and is
designed to be re-exported later by a `forgather mcp` server for external
MCP clients (Claude Code / Desktop) without duplicating definitions.
Scheduler control (abort/kill) and DiLoCo coordination are planned later
phases.

## Not yet implemented

- Per-run metrics charts (loss curves, etc. — the data is already in
  `trainer_logs.json`; the UI just needs a renderer).
- Auto-rename or re-path of open editor buffers when the on-disk
  file is renamed / moved from the Files tree. Current behavior
  closes the stale tab silently — the user re-opens the new path
  from the tree.
- (CLI-only items mostly rolled into the UI: `forgather ws create`
  is now the New Workspace… button under Search Roots,
  `forgather project create` is the workspace context menu, and
  per-config / per-template creation is the project context menu.)
- Multi-node deployment. Today's design tags each GPU and JobRecord
  with a `node` identifier and concentrates the "this could be remote"
  surfaces in `gpu_monitor.py` / `launcher.py` / `scheduler.py`, so the
  future seam is a `NodeClient` abstraction in front of those modules.

---

