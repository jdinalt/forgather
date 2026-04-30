# Forgather Server (prototype)

A web frontend over the existing Forgather CLI. Single pane of glass for
discovering projects, inspecting configurations, queuing training / eval
/ inference / TensorBoard jobs across a GPU pool, watching their TTY
logs, controlling them, and talking to running inference servers from
the browser — wraps `MetaConfig`, `ConfigEnvironment`,
`TrainerControlClient`, and friends rather than re-implementing them.

**Prototype status.** Single-user, localhost-first. Binds to `127.0.0.1`
by default. No auth, no rate limiting. Do not expose the port on an
untrusted network.

---

## CLI access

The `forgather` CLI can talk to a running server directly — no browser needed. All commands accept `--server URL` or the `FORGATHER_SERVER_URL` environment variable; both default to `http://127.0.0.1:8765`.

For a workflow-oriented walkthrough with recipes, see
[guides/server-cli.md](guides/server-cli.md). The reference below is a
quick cheat-sheet.

**Submit jobs from the terminal:**

```bash
# Inside a project directory
forgather -t train.yaml train --enqueue
forgather -t train.yaml train --enqueue --priority 5 --requested-gpus 2
forgather eval test c4 -M output_models/my_model --enqueue
forgather tb --enqueue --port 6006
forgather inf server --enqueue -m output_models/my_model
forgather convert --enqueue --src output_models/my_model --dst /tmp/hf_export
forgather finalize --enqueue --source output_models/my_model --dest /tmp/final
forgather mkdocs -f docs/mkdocs.yml --enqueue
```

**Queue and scheduler:**

```bash
forgather sched status                   # enabled, queued/running counts, last tick
forgather sched list                     # table of all queued + active + recent jobs
forgather sched pause                    # stop dispatching new jobs
forgather sched resume
forgather sched cancel <queue_id>        # remove a queued or running job
forgather sched cleanup                  # bulk-remove terminal job records
forgather sched cleanup <job_id>         # remove one specific terminal record
```

**Per-job control and logs:**

```bash
forgather job status <id>               # trainer status dict (409 = still starting)
forgather job save <id>                 # trigger checkpoint
forgather job stop <id>                 # graceful stop (saves final checkpoint)
forgather job save-stop <id>
forgather job abort <id>                # immediate stop, no checkpoint
forgather job kill <id>                 # SIGTERM
forgather job force-kill --yes <id>     # SIGKILL
forgather job tail <id>                 # stream live TTY; Ctrl-C exits cleanly
forgather job dump <id>                 # write full captured log to stdout
forgather job dump <id> > log.txt
```

**GPU policy:**

```bash
forgather gpu status                    # table: util, mem, temp, power, disabled, min_priority, pids
forgather gpu disable <idx>             # mark GPU unavailable for scheduling
forgather gpu enable <idx>
forgather gpu priority <idx> <N>        # only dispatch jobs with priority >= N to this GPU
forgather gpu kill --yes <idx>          # SIGKILL all compute processes on the card
```

---

## Installation

### Python side

The server's runtime deps ship with Forgather: `fastapi`, `uvicorn`,
`websockets`, `psutil`, `pynvml`, `pydantic`, `pyyaml`. If you installed
Forgather with `pip install -e .`, you're done.

Notes:

- `websockets` is required for the live GPU stream and TTY tail. Without
  a WebSocket backend, uvicorn 404s on upgrade and those features
  silently degrade.
- `pynvml` provides full GPU info (utilization, power, temp, per-GPU
  PIDs). Without it, the server falls back to `torch.cuda` for name +
  memory only — and warns that indices may not match physical indices
  when `CUDA_VISIBLE_DEVICES` is set.
- `psutil` is used for liveness checks (job re-attach across restart,
  abort, the `alive` flag in `/api/jobs`).

### Web UI

Vite + React + TypeScript. Build once, then the running server serves
`webui/dist/` as static assets:

```bash
cd tools/forgather_server/webui
npm install          # one-time
npm run build        # produces webui/dist/
```

`node`/`npm` are only needed for the build step. The running server has
no Node dependency.

## Running

```bash
# Default: 127.0.0.1:8765
forgather server

# Custom bind / verbosity
forgather server -H 127.0.0.1 -p 8765 -l INFO
```

Open <http://127.0.0.1:8765/>. On first boot the server seeds its
search-roots list with `<repo>/examples`; add or remove roots via the
sidebar's **Browse…** button.

### Excluding misbehaving GPUs

Set `CUDA_VISIBLE_DEVICES` when starting the server to keep specific
GPUs out of the scheduler's allocation pool. Excluded cards still appear
in the GPUs view (telemetry stays live so you can monitor temperatures /
processes) but with a dashed red border and an `EXCLUDED` badge — the
scheduler refuses to assign them.

```bash
# Reserve GPU 2 (e.g. thermally suspect) — dispatcher won't pick it
CUDA_VISIBLE_DEVICES=0,1,3,4,5 forgather server -p 8765
```

The allow-list is parsed once at module import. Restart the server to
change it.

### Persistent state

Everything under `~/.forgather/server/` survives restarts:

| File / dir                  | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `search_roots.json`         | Project-discovery roots (seeded on first boot).        |
| `queue.json`                | Queue of items waiting for GPUs.                       |
| `job_records.json`          | Records for jobs the server has launched (any state). |
| `jobs/{queue_id}.tty`       | Captured stdout+stderr for each launched job.          |
| `overrides/{hash}.json`     | Per-config dynamic-args override cache.                |
| `gpu_policy.json`           | Per-GPU runtime policy: disabled + min_priority.       |

All state files are written crash-atomically via `_atomic.py`: tmp file
written in the target directory, `fsync` on the fd, then `os.replace`.
Power loss or SIGKILL mid-write never leaves the canonical file
partially written. Every reader tolerates a corrupt / truncated file by
falling back to empty state.

### Re-attach across restart

Training subprocesses are spawned with `start_new_session=True`, so they
keep running after the server exits. On startup the scheduler walks
every JobRecord still marked `running` / `starting` and:

- If the recorded PID is still alive (and `create_time()` matches, to
  guard against PID reuse): re-attach in the unified jobs list.
  Trainer-side control commands (Save / Stop / Save&Stop / Abort) and
  the local `Kill` keep working through the existing endpoint plus
  process-group SIGTERM.
- Otherwise: mark the record `failed` with a clear reason.

Reaping a re-attached job records `status="done"` with `exit_code=null`
since exit codes for non-child processes aren't recoverable from
outside.

### Dev mode (Vite + hot reload)

For rapid frontend iteration, run Vite separately from the API:

```bash
# Terminal 1 — API backend
forgather server -p 8765

# Terminal 2 — Vite dev server with hot reload
cd tools/forgather_server/webui
npm run dev
# opens http://localhost:5173, proxies /api → :8765 (REST + WebSocket)
```

---

## Implemented features

### App chrome

The left side of the window is a collapsible sidebar (`<aside
class="app-sidebar">`) that owns navigation and global actions. Top to
bottom:

- **Header** — "Forgather Server" · preview, then a `⟳ Refresh` button,
  a scheduler `▶`/`⏸` play/pause button (toggles the dispatcher; green
  when running, muted when paused), and a window/sidebar SVG toggle
  that collapses the sidebar.
- **Views** (collapsible `<details>`) — vertical tabs with icons:
  📁 Projects, ✎ Edit, 🖥 GPUs, 📋 Queue, ⚙ Jobs, 🔮 Inference.
  Selecting anything in the project tree routes back to the
  Projects view automatically. The **Edit** view is the tabbed
  Monaco editor (formerly named "Files"); it was renamed to free
  the "Files" name for the new sidebar filesystem tree (see below).
- **Tools** (collapsible `<details>`) — global actions that don't
  belong to a specific config. Each tool persists its last-submitted
  settings to localStorage so the next open defaults to those values
  (`priority` resets each time since the right value depends on
  current queue state). Current tools:
    - **🔮 Serve Inference…** — opens `InferenceModal` in ad-hoc mode
      so you can serve any on-disk directory without a Forgather
      project. Persisted under `forgather-adhoc-inference-v1`.
    - **📊 TensorBoard…** — `TensorBoardModal` in `global` mode against
      any logdir on disk. Persisted under `forgather-global-tensorboard-v1`.
    - **📖 MkDocs…** — queues `mkdocs serve` against an `mkdocs.yml`
      on disk. The picker defaults to `<Forgather repo>/mkdocs.yml`
      when no value has been persisted yet (resolved from
      `/api/fs/quick-paths`'s "Forgather repo" entry). Persisted under
      `forgather-global-mkdocs-v1`.
    - **🔁 Convert Model…** — queues `forgather convert` against a
      pair of source/destination model paths. Direction (HF↔Forgather)
      is auto-detected by the script unless `--reverse` is forced.
      Persisted under `forgather-global-convert-v1`. The footer carries
      a **Reset to defaults** button that clears the persisted blob.
    - **📦 Finalize Model…** — queues `forgather finalize` to package
      a trained Forgather output tree into a clean directory: tokenizer
      additions, chat template, generation config, root-copy /
      keep-optimizer toggles. Persisted under
      `forgather-global-finalize-v1`. Same **Reset to defaults**
      affordance as Convert.
- **Project tree** — Search Roots + workspace-clustered projects
  (see below).

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
- **Refresh button** invalidates the entire client query cache so disk
  edits to workspace metadata, templates, configs are picked up
  immediately.

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

### Sidebar layout: six top-level collapsible sections

The sidebar's body below the header is a stack of six independent
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
| **Tools** | inline buttons | Global actions: 🔮 Serve Inference, 📊 TensorBoard, 📖 MkDocs, 🔁 Convert Model, 📦 Finalize Model. |
| **Search Roots** | `SearchRootsPanel` | Root-list management: Browse… to add, × to remove, **📁 New Workspace…** for the dropdown-driven flow. Lifted out of `ProjectTree` so each group is its own top-level entry. |
| **Projects** | `ProjectTree` | The familiar workspace-clustered project forest. |
| **Files** | `FilesTree` | Hierarchical filesystem view of every search root. |

Earlier iterations had Tools and the view switcher visually
distinct from the rest (a horizontal rule above and below Tools, a
Tools-specific summary block). Those were dropped so the six
groups read as a single uniform stack — easier to scan, no
implicit grouping where there isn't one.

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
| **⎘ Paste** | dir, when clipboard set | `POST /api/fs/move` (cut, consumes clipboard) or `POST /api/fs/copy` (copy) |
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
- `POST /api/fs/copy` `{src, dest_dir}` — `shutil.copy2` for
  files, `shutil.copytree` for directories; refuses overwrite at
  destination.
- `POST /api/fs/move` `{src, dest_dir}` — `shutil.move` (so
  cross-device moves degrade to copy + unlink); refuses overwrite.
- `POST /api/fs/new-file` `{parent, name}` — `Path.touch()` an
  empty file; refuses overwrite.
- `POST /api/fs/mkdir` `{parent, name}` — single new directory
  (already existed; reused for + New Folder…).

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
- The form shows what `nproc_per_node` the config declares (`"gpu"` /
  fixed integer / `"cpu"` / `"auto"`) and warns when the user's GPU
  reservation count would mismatch a fixed worker count.
- The scheduler holds a JSON-backed queue + an in-memory dispatcher
  loop. **Enabled by default** so a freshly-restarted server resumes
  dispatch immediately. Pause anytime with the `▶`/`⏸` button in the
  sidebar header. The Queue view shows the current `running` /
  `paused` state.
- Dispatch picks idle GPU indices that aren't excluded via
  `CUDA_VISIBLE_DEVICES`, sets the child's `CUDA_VISIBLE_DEVICES` to
  the assignment, and invokes `torchrun` directly (mirrors what
  `forgather train` does, minus the extra subprocess layer — lets the
  scheduler own the process group for clean abort).

**Nine job types** share the queue, scheduler, GPU accounting, and TTY
capture machinery. The non-CUDA-by-default types (`tensorboard`,
`mkdocs`, `convert`, `finalize`, `dataset`) accept `requested_gpus == 0`; the
others default to at least one GPU. Convert / finalize will happily
take a GPU if the user sets `--device cuda…` and bumps the reservation.

| Type         | Spawned by                                                                             | Lifecycle                              |
| ------------ | -------------------------------------------------------------------------------------- | -------------------------------------- |
| `training`   | ▶ Run (Submit modal)                                                                   | Terminal when trainer exits.           |
| `eval`       | ⚖ Evaluate… (EvalModal, from config or checkpoint)                                     | Terminal when `forgather eval` exits.  |
| `inference`  | 🔮 Serve Inference… (InferenceModal, project-backed or ad-hoc)                         | Long-lived; kill/force-kill to stop.   |
| `tensorboard`| 📊 TensorBoard… (TensorBoardModal, per-config or per-model)                            | Long-lived; kill to stop.              |
| `mkdocs`     | 📖 MkDocs… (MkDocsModal, sidebar Tools — picks an `mkdocs.yml` + host:port)            | Long-lived; kill to stop.              |
| `convert`    | 🔁 Convert Model… (ConvertModal, sidebar Tools)                                        | Terminal when `convert` exits.         |
| `finalize`   | 📦 Finalize Model… (FinalizeModal, sidebar Tools)                                      | Terminal when `finalize` exits.        |
| `model`      | Run on a model config (config_class `type.model`)                                      | Terminal when `forgather model` exits. |
| `dataset`    | Run on a dataset config (config_class `type.dataset`)                                  | Terminal when `forgather dataset` exits.|

Helpers live in `inference_ops.py`, `eval_ops.py`, `tensorboard_ops.py`,
`mkdocs_ops.py`, `convert_ops.py`, `finalize_ops.py`, `model_ops.py`,
`dataset_ops.py` (build argv) and
`launcher.spawn_*_process` (same sandbox as training but with the right
argv). The scheduler's
dispatcher branches on `item.job_type` to pick the spawn function;
GPU accounting and re-attach logic are unchanged. Long-lived web
services (inference, tensorboard, mkdocs) all surface their URL as a
clickable link on the Jobs card so the operator can jump straight to
the running endpoint.

### Scheduling algorithm

Each scheduler tick (~2 s) runs this placement logic:

1. **Build the queue.** Read `queue.json`, sort items by priority
   descending, then by submission time ascending (so higher-priority
   jobs go first; FIFO within a priority band).

2. **Build the idle pool.** Start from every GPU and drop any that are:
   - running a compute process (NVML `nvmlDeviceGetComputeRunningProcesses`
     says busy — could be a job we launched, a job someone else
     launched, or stale-CUDA wedge);
   - **excluded** via `CUDA_VISIBLE_DEVICES` (set at server start);
   - **disabled** at runtime via the UI toggle (persists in
     `gpu_policy.json`);
   - already **reserved** for one of our `starting` / `running`
     JobRecords (defensive — these should already show as busy, but
     this handles the gap between spawn and first process report).

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
  externally-discovered trainer endpoints from `~/.forgather/jobs/`.
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
Three sub-tabs sharing the same `InferenceState` (base URL, model,
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

**Serve Inference… (sidebar Tools section)** — opens `InferenceModal`
in ad-hoc mode: the model path becomes a `PathField` instead of a
read-only summary, so the user can serve any on-disk directory without
a Forgather project. Ad-hoc settings (path, port, dtype, attention
impl, cache impl, compile flags, chat template, checkpoint path)
persist under `forgather-adhoc-inference-v1` — the next invocation
defaults to the last-submitted values. Requested GPUs and priority
stay fresh each invocation since the "right" value depends on current
queue occupancy.

**Generation presets** — save/load named JSON presets of the current
generation params. Served by `/api/generation-configs/*`, which merges
two layers: bundled examples under `<repo>/generation_config/` (read-
only: `greedy`, `precise`, `balanced`, `creative`, `beam_search`,
`contrastive`) and user presets under `~/.forgather/generation_config/`
(writable; shadows same-named bundled entries). Delete on a built-in
returns 403 with guidance; delete on a user shadow restores the
built-in.

**Browser → inference-server proxy** (`routes/inference_proxy.py`) —
the webui can't hit spawned inference servers directly without running
into CORS / Private Network Access / extension-blocking. Everything
routes through same-origin `/api/inference/*`; the proxy forwards to
whichever base URL the caller names, streaming byte-for-byte so the
SSE framing reaches the browser unchanged.

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
- **Right-click a Job card** opens a context menu with **☠ Force kill
  (SIGKILL)** for hung server-launched jobs that aren't responding to
  SIGTERM — routes through a new `force-kill` control action.

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
  Config prefix (default `configs`), Default config (default
  `default.yaml`), Project dir (relative to workspace; may be
  nested with `mkdir -p` semantics; Browse… button anchored to
  `workspace_root` lets the user pick an existing subdirectory and
  drops the relative path back into the field with a trailing `/`
  for the leaf name), and an optional Copy-from `PathField` for
  seeding the default config from an existing file. Submit calls
  `POST /api/workspace/new-project`, which dispatches into
  `forgather.cli.project.project_create_cmd` via a
  `SimpleNamespace` so we don't duplicate the CLI's project-skeleton
  logic. Tree refresh is via `["projects"]` invalidation. The
  synthetic "Unaffiliated" cluster (no `workspace_root`) doesn't
  receive the menu. Delete-Workspace recursively removes the
  workspace directory via `POST /api/fs/delete-dir`, with the same
  two-step gate as Delete-Project (standard `confirm()` plus a
  typed-token prompt requiring the workspace's directory basename),
  since deleting a workspace cascades to every project, config, and
  in-tree output_models within it.
- **Project row** — **📄 New Config…** / **📄 New Template…**.
  Both open a `NewTemplateModal` (shares the chrome with
  `CleanOutputModal` et al.) with project / kind / base-dir summary
  rows, an auto-focused name input, an inline hint about the
  `.yaml` default suffix and subdirectory support, and a live
  preview of the absolute target path. Subdirectory creation under
  the configs / templates root is handled by typing a nested name
  (e.g. `experiments/foo.yaml`) — `mkdir -p` semantics on the
  server. The base path comes from
  `GET /api/project/template-paths` (`MetaConfig.searchpath[0]` for
  templates, plus `config_prefix` for configs). Submit calls
  `POST /api/project/new-template`, invalidates the project tree
  and `project-templates` queries so the new file shows up in
  `tlist`, then hands the returned path to the Edit panel via
  the App-level `onEditTemplate` hook — the user lands directly
  on a blank editor for the new file. A trailing **🗑 Delete
  Project…** entry recursively removes the project directory via
  `POST /api/fs/delete-dir`; it's gated by both a standard
  `confirm()` and a typed-token prompt requiring the user to type
  the project's directory basename, since the project tree often
  contains an `output_models/` subtree (runs / checkpoints) that
  the regular Clean Output flow won't touch. The confirm body
  spells out that outputs configured to live *outside* the project
  tree are not affected. After delete `["projects"]`,
  `["project-templates", dir]`, and `["project-models", dir]` are
  invalidated, and the active selection is dropped if it was
  pointing into the deleted project.
- **Config row** — Run / TensorBoard / Overrides / Clean Output, plus
  Serve Inference / Evaluate / Convert Model / Finalize Model when
  the config has checkpoints on disk. Convert and Finalize pre-fill
  the source path with the config's resolved `output_dir` while
  inheriting every other field from the global tool's persisted
  defaults; submit then writes everything (including the new source
  path) back, so the next opening — global tool or context-menu —
  reflects the last run. Items are filtered by `config_class` so
  non-training configs only show **Overrides**. A trailing **🗑 Delete
  Config…** entry unlinks just the config template file (via
  `POST /api/fs/delete-file`); it explicitly does *not* touch the
  config's `output_dir` / runs / checkpoints — those have their own
  Clean Output / Delete Permanently flows. After delete the
  `["projects"]` and `["project-templates", …]` queries are
  invalidated so the tree and the `tlist` view both refresh, and
  the active selection is cleared if it pointed at the deleted file.
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

## Directory layout

```
src/forgather/cli/
├── server.py                  # CLI shim: `forgather server` → backend subprocess
└── wrappers_args.py           # CLI parser registration for `server`

generation_config/             # Bundled generation-parameter presets
│                              #   (read-only from the UI; shadowed by
│                              #    ~/.forgather/generation_config/)
├── greedy.json
├── precise.json
├── balanced.json
├── creative.json
├── beam_search.json
└── contrastive.json

tools/forgather_server/
├── server.py                  # uvicorn entry point
├── app.py                     # FastAPI app factory + lifespan (dispatcher loop)
├── paths.py                   # ~/.forgather/server/ state helpers
├── _atomic.py                 # Crash-atomic file-write helpers
│                              #   (tmp + fsync + os.replace)
├── search_roots.py            # JSON-backed search-root list, default seeding
├── discovery.py               # Walk roots → cluster projects by workspace
├── models_catalog.py          # Enumerate per-project output_dirs, runs,
│                              #   checkpoints, evaluations
├── config_ops.py              # Wrappers around ConfigEnvironment, with
│                              #   per-config overrides auto-applied
├── overrides_store.py         # Per-config dynamic-args override cache
├── queue_store.py             # Persistent FIFO queue (waiting items only)
├── job_records.py             # Persistent records of dispatched jobs
├── launcher.py                # Spawn training / eval / inference /
│                              #   tensorboard / mkdocs / convert /
│                              #   finalize / model / dataset processes;
│                              #   own process group
├── inference_ops.py           # Build inference-server argv
├── eval_ops.py                # Build `forgather eval` argv
├── tensorboard_ops.py         # Build tensorboard argv
├── mkdocs_ops.py              # Build `mkdocs serve` argv
├── convert_ops.py             # Build `forgather convert` argv
├── finalize_ops.py            # Build `forgather finalize` argv
├── model_ops.py               # Build `forgather model` argv
├── dataset_ops.py             # Build `forgather dataset` argv
├── scheduler.py               # Dispatcher loop, GPU allocation,
│                              #   per-job-type spawn, re-attach, reap, abort
├── gpu_monitor.py             # NVML / torch.cuda enumeration,
│                              #   CUDA_VISIBLE_DEVICES allow-list
├── gpu_policy.py              # Runtime per-GPU policy (disabled,
│                              #   min_priority) — persisted
├── routes/
│   ├── search_roots.py        # GET/POST/DELETE /api/search-roots
│   ├── projects.py            # /api/projects, /api/project/{readme,asset}
│   ├── configs.py             # /api/config/{raw,pp,trefs,meta,templates,
│   │                          #               overrides,output-dir} +
│   │                          #   /api/template/source
│   ├── models.py              # /api/project/models, /api/model/{runs,
│   │                          #   checkpoints,evaluations}, /api/run/{tty,
│   │                          #   summary}, /api/eval-configs
│   ├── fs.py                  # /api/fs/{browse,quick-paths,delete-dir}
│   ├── gpus.py                # /api/gpus + WS /api/gpus/stream + kill
│   ├── jobs.py                # /api/jobs (unified), control, TTY (REST + WS),
│   │                          #   cleanup
│   ├── queue.py               # /api/queue + /api/queue/scheduler +
│   │                          #   /api/config/dynamic-args
│   ├── inference_proxy.py     # /api/inference/{health,models,completions,
│   │                          #   chat/completions} — same-origin SSE proxy
│   └── generation_configs.py  # /api/generation-configs/{list,get,put,delete}
└── webui/
    ├── package.json           # Vite, React, TypeScript, Monaco, viz-js,
    │                          #   TanStack Query, react-markdown, remark-gfm
    ├── vite.config.ts         # dev-mode /api → :8765 proxy (REST + WS)
    └── src/
        ├── main.tsx           # React + QueryClientProvider bootstrap
        ├── App.tsx            # Collapsible sidebar (header, view
        │                      #   switcher, Tools, ProjectTree) + main
        │                      #   pane; owns view / selection / tab
        │                      #   state and the scheduler play/pause
        ├── api.ts             # Typed fetch wrappers for every endpoint
        ├── inference-client.ts# Browser client for /v1/* (via the proxy);
        │                      #   streamCompletion / streamChatCompletion /
        │                      #   runCompletion / runChatCompletion +
        │                      #   shared SSE loop
        ├── forgather-syntax.ts # Monaco Monarch tokenizer
        ├── file-languages.ts  # Extension -> Monaco language id;
        │                      #   plaintext fallback for unknown
        │                      #   types — every file is openable
        │                      #   subject to the backend binary check
        ├── files-state.ts     # useFilesState hook: open buffers, splits,
        │                      #   tabs, save (Ctrl+S), drag-drop reorder,
        │                      #   dropPath (silent close-everywhere)
        ├── styles.css
        └── components/
            ├── ProjectTree.tsx      # Sidebar tree + per-config artifact
            │                        #   sub-groups; context menus
            ├── DirectoryBrowser.tsx
            ├── PathField.tsx        # Text input + Browse… picker
            ├── ContextMenu.tsx      # Generic floating menu
            ├── ConfigViewer.tsx     # Tabs: info / pp / templates
            ├── InfoPane.tsx         # Markdown renderer (GFM + image proxy)
            ├── TemplatesView.tsx    # `templates` tab container: trefs/tlist
            │                        #   mode bar, shared right-pane preview,
            │                        #   right-click → Open in Editor
            ├── DynamicArgsForm.tsx  # Shared form for Submit + Overrides
            ├── SubmitModal.tsx      # Enqueue training job
            ├── OverridesModal.tsx   # Set/reset persistent dynamic-args
            ├── CleanOutputModal.tsx # Delete output_dir / models_dir
            ├── EvalModal.tsx        # Enqueue eval job
            ├── NewProjectModal.tsx  # forgather project create flow:
            │                        #   name/description + CLI-matched
            │                        #   defaults + copy-from picker;
            │                        #   nested project_dir via Browse…
            │                        #   anchored at the workspace root
            ├── NewWorkspaceModal.tsx# forgather ws create flow: parent
            │                        #   search-root dropdown (with
            │                        #   inline + Create new search
            │                        #   root… sub-form), nested
            │                        #   workspace dir, libs/search
            │                        #   paths textareas
            ├── InitWorkspaceModal.tsx# Init workspace in an existing
            │                        #   directory — slimmer modal for
            │                        #   the Files-tree right-click flow:
            │                        #   path is fixed, only metadata
            │                        #   is collected
            ├── NewTemplateModal.tsx # New Config / New Template prompt
            │                        #   with live target-path preview
            ├── SearchRootsPanel.tsx # Top-level Search Roots sidebar
            │                        #   group; root list + Browse… +
            │                        #   📁 New Workspace…
            ├── InferenceModal.tsx   # Enqueue inference-server job
            │                        #   (project-backed or ad-hoc)
            ├── TensorBoardModal.tsx # Enqueue tensorboard job
            │                        #   (config-backed; or `global`
            │                        #   from sidebar Tools)
            ├── MkDocsModal.tsx      # Enqueue `mkdocs serve` job
            │                        #   (sidebar Tools — global only)
            ├── ConvertModal.tsx     # Enqueue `forgather convert` job
            │                        #   (sidebar Tools — global only)
            ├── FinalizeModal.tsx    # Enqueue `forgather finalize` job
            │                        #   (sidebar Tools — global only)
            ├── LogDetailPanel.tsx   # Selection target for a run/log leaf
            ├── CheckpointDetailPanel.tsx # Selection target for a checkpoint
            ├── EvalDetailPanel.tsx  # Selection target for an evaluation
            ├── RunSummaryView.tsx   # Extracted from legacy models panel
            ├── EvalResultTable.tsx  # Extracted from legacy models panel
            ├── InferencePanel.tsx   # Inference view: model/completion/chat
            │                        #   sub-tabs (Serve Inference lives
            │                        #   in the sidebar Tools section)
            ├── InferenceModelPanel.tsx     # Base URL, params, presets
            ├── InferenceCompletionPanel.tsx# Textarea completion + Stream
            ├── InferenceChatPanel.tsx      # Multi-turn chat + markdown
            ├── GpuPanel.tsx         # Live GPU cards; PID→job attribution
            ├── JobsPanel.tsx        # Unified jobs list + split-pane TTY
            │                        #   + bulk cleanup
            ├── TtyViewer.tsx        # Imperative-append terminal
            ├── QueuePanel.tsx      # Queue list + scheduler status
            │                        #   (toggle lives in the sidebar)
            ├── FilesTree.tsx        # Sidebar filesystem tree per search
            │                        #   root; in-memory clipboard for
            │                        #   Cut / Copy / Paste; right-click
            │                        #   → Open / Rename / Delete
            └── FilesPanel.tsx       # Editor with tabbed splits, drag-drop
                                     #   reorder, Save / Close context menu;
                                     #   per-file Monaco language via
                                     #   file-languages.ts
```

## Architecture in one paragraph

The backend is a thin FastAPI app that wraps Forgather's existing Python
APIs — no re-implementation. Every endpoint ultimately calls into
`MetaConfig`, `ConfigEnvironment`, the `forgather.cli.trefs` renderers,
or `TrainerControlClient`. Config materialization respects per-config
override values pulled from a JSON cache, so `pp` / `trefs` /
`output-dir` / `config/meta` all reflect whatever the user has set in
the 🔧 Overrides modal. The scheduler dispatches nine job types —
training (`torchrun`), eval (`forgather eval`), inference
(`tools/inference_server/server.py`), TensorBoard (`tensorboard`),
MkDocs (`mkdocs serve`), convert (`forgather convert`), finalize
(`forgather finalize`), model, and dataset — all through a common
`launcher.spawn_*`
surface that owns its process group via `start_new_session=True` so
jobs survive server restart. Inference
servers spawned this way appear in the Inference panel's "Running
inference servers" picker; the browser talks to them through a
same-origin SSE proxy so CORS / PNA don't get in the way. The frontend
is a Vite/React SPA driven by TanStack Query for caching + background
refresh; persistent server state is plain JSON files under
`~/.forgather/server/` so it's inspectable with ordinary tools.

## API quick reference

All endpoints are under `/api`. JSON unless noted. Endpoints marked WS
are WebSockets.

### Discovery

| Endpoint                                       | Purpose                                             |
| ---------------------------------------------- | --------------------------------------------------- |
| `GET /api/health`                              | Liveness                                            |
| `GET /api/search-roots`                        | List search roots                                   |
| `POST /api/search-roots` `{path, create?: bool}`| Add a search root; with `create: true` the server `mkdir`s the path before registering (used by the New Workspace modal's inline create-root flow) |
| `DELETE /api/search-roots?path=`               | Remove a search root                                |
| `GET /api/projects`                            | Workspace-clustered project tree                    |
| `GET /api/project?project_dir=`                | Single-project detail                               |
| `GET /api/project/readme?project_dir=`         | README.md as markdown                               |
| `GET /api/project/asset?project_dir=&asset=`   | Image / file embedded in the README (path-guarded)  |
| `GET /api/project/templates?project_dir=`      | Every template on the project's search path, grouped by search-root category (with synthetic Meta group for `meta.yaml`) — backs the `tlist` view |
| `GET /api/project/template-paths?project_dir=` | Resolved `templates_dir` + `configs_dir` + `config_prefix` (for the New Config / New Template modal's path preview) |
| `POST /api/workspace/new-project` `{workspace_dir, name, description, config_prefix?, default_config?, project_dir_name?, copy_from?}` | Create a project under a workspace — wraps the CLI's `project_create_cmd`; nested `project_dir_name` (`a/b/c`) supported; refuses overwrite, returns absolute project_dir |
| `POST /api/workspace/new` `{parent_dir, name, description, workspace_dir_name?, forgather_dir, libs?, search_paths?}` | Create a workspace under a search root — wraps `ws_create_cmd`; parent must be a configured search root; nested `workspace_dir_name` supported; returns absolute workspace_dir |
| `POST /api/workspace/init-here` `{workspace_dir, name, description, forgather_dir, libs?, search_paths?}` | Initialize a workspace in an *existing* directory — used by the Files-tree right-click flow. Refuses if `forgather_workspace/` already exists; requires `workspace_dir` to live at-or-under a configured search root. |
| `POST /api/project/new-template` `{project_dir, kind: "config"\|"template", name}` | Create an empty file under the templates dir; refuses overwrite, `.yaml` auto-appended, returns absolute path |

### Config inspection

| Endpoint                                                                         | Purpose                                                        |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/config/raw?path=`                                                      | Raw config source                                              |
| `GET /api/config/pp?project_dir=&config=`                                        | Preprocessed YAML (overrides applied)                          |
| `GET /api/config/trefs?project_dir=&config=&format=json\|dot\|tree`              | Template dependency graph (overrides applied)                  |
| `GET /api/config/templates?project_dir=&config=`                                 | Flat list of consumed templates                                |
| `GET /api/config/meta?project_dir=&config=`                                      | `config_name` / `config_description` / `config_class`          |
| `GET /api/config/output-dir?project_dir=&config=`                                | Resolved `output_dir` + `models_dir`, sizes, `nproc_per_node`  |
| `GET /api/config/dynamic-args?project_dir=&config=`                              | Form schema for the submit / overrides UI                      |
| `GET /api/config/overrides?project_dir=&config=`                                 | Cached override values for this config                         |
| `POST /api/config/overrides` `{project_dir, config, values}`                     | Set / replace cached overrides                                 |
| `DELETE /api/config/overrides?project_dir=&config=`                              | Clear cached overrides                                         |
| `GET /api/template/source?path=`                                                 | Raw source of any template; `X-Mtime` response header carries the file's mtime so the editor can detect concurrent edits |
| `PUT /api/template/source` `{path, content, expected_mtime?}`                    | Write template content (atomic; path must exist). When `expected_mtime` is given, returns 409 with `{message, current_mtime, expected_mtime}` if the file is newer on disk; pass null/omit to force-overwrite. Successful response includes the new `mtime`. |

### Models / runs / checkpoints / evaluations

Populates the project-tree sub-groups and detail panels:

| Endpoint                                                 | Purpose                                                        |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/project/models?project_dir=`                   | Per-`output_dir` summary (configs, run/checkpoint/eval counts) |
| `GET /api/model/runs?output_dir=`                        | Run entries with timestamps and log paths                      |
| `GET /api/model/checkpoints?output_dir=`                 | Checkpoints (step, size, world_size, manifest)                 |
| `GET /api/model/evaluations?output_dir=`                 | Evaluations + results summary                                  |
| `GET /api/run/summary?run_dir=`                          | Trainer-log statistics (best loss, steps, perplexity, …)       |
| `GET /api/run/tty?run_dir=`                              | `tty.log` tail (one-shot)                                      |
| `GET /api/eval-configs`                                  | Discoverable eval configs for the EvalModal dropdown           |

### Filesystem

| Endpoint                                                | Purpose                                                  |
| ------------------------------------------------------- | -------------------------------------------------------- |
| `GET /api/fs/browse?path=&show_hidden=&files_too=`      | Directory listing (dirs only by default)                 |
| `GET /api/fs/quick-paths`                               | Named quick-jump shortcuts                               |
| `POST /api/fs/delete-dir` `{path, confirmed: true}`     | Delete a directory (multiple safety guards; see code)    |
| `POST /api/fs/delete-file` `{path, confirmed: true}`    | Delete a single regular file (depth floor + symlink reject; used by Delete Config) |
| `POST /api/fs/mkdir` `{parent, name}`                   | Create a single new directory under `parent`; bare-name (no separators), refuses overwrite — used by DirectoryBrowser's + New Folder chip |
| `POST /api/fs/rename` `{path, new_name}`                | Rename a file or directory in place (bare basename); refuses overwrite — used by the sidebar Files tree |
| `POST /api/fs/copy` `{src, dest_dir}`                   | Copy a file (`shutil.copy2`) or directory (`shutil.copytree`) to `dest_dir/basename(src)`; refuses overwrite — used by the Files tree's Paste-after-Copy |
| `POST /api/fs/move` `{src, dest_dir}`                   | Move a file or directory to `dest_dir/basename(src)` via `shutil.move`; refuses overwrite — used by the Files tree's Paste-after-Cut |
| `POST /api/fs/new-file` `{parent, name}`                | Create an empty file at `parent/name`; bare-name, refuses overwrite — used by the Files tree's New File… affordance |

### GPUs

| Endpoint                                                    | Purpose                                                                          |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `GET /api/gpus`                                             | One-shot snapshot                                                                |
| `WS /api/gpus/stream`                                       | Push updates every ~2 s                                                          |
| `GET /api/gpus/policy`                                      | All per-GPU runtime policies (`{index: {disabled, min_priority}}`)               |
| `POST /api/gpus/{index}/policy` `{disabled?, min_priority?}` | Upsert per-GPU policy; unset fields are left alone                              |
| `POST /api/gpus/{index}/kill` `{confirmed: true}`           | SIGKILL every compute process on the GPU (returns `{pids, killed, failed}`)      |

### Queue / scheduler

| Endpoint                                                  | Purpose                                                |
| --------------------------------------------------------- | ------------------------------------------------------ |
| `GET /api/queue`                                          | List queued items                                      |
| `POST /api/queue` `{project_dir, config, dynamic_args, requested_gpus, priority, job_type?, job_params?}` | Enqueue any job type (`training` / `eval` / `inference` / `tensorboard` / `mkdocs` / `convert` / `finalize` / `model` / `dataset`) |
| `DELETE /api/queue/{queue_id}`                            | Cancel a queued item (or abort if it's already running) |
| `GET /api/queue/scheduler`                                | Dispatcher on/off + counters                           |
| `POST /api/queue/scheduler` `{enabled}`                   | Enable / disable the dispatcher                        |

### Jobs (unified: launched + discovered)

| Endpoint                                                              | Purpose                                                    |
| --------------------------------------------------------------------- | ---------------------------------------------------------- |
| `GET /api/jobs?include_dead_endpoints=`                               | Merged list of JobRecords + endpoint discoveries           |
| `GET /api/jobs/{id}/status`                                           | Trainer-side `/status` proxy (step, loss, etc.)            |
| `POST /api/jobs/{id}/control/{save\|stop\|save-stop\|abort\|kill\|force-kill}` | Trainer control commands; `kill`=local SIGTERM, `force-kill`=local SIGKILL |
| `DELETE /api/jobs/{id}`                                               | Remove a terminal JobRecord from history                   |
| `POST /api/jobs/cleanup`                                              | Bulk-remove every terminal JobRecord (`done` / `failed` / `aborted`) |
| `GET /api/jobs/{id}/tty`                                              | Full captured TTY (one-shot)                               |
| `WS /api/jobs/{id}/tty?follow=`                                       | Backlog + follow-tail of captured TTY                      |

### Inference proxy

Same-origin forwarder so the browser can talk to inference-server jobs
without running into CORS / PNA issues.

| Endpoint                                              | Purpose                                                        |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/inference/health?base=`                     | Proxy `<base>/health`                                          |
| `GET /api/inference/models?base=`                     | Proxy `<base>/models`                                          |
| `POST /api/inference/completions?base=`               | Proxy `<base>/completions` (byte-for-byte SSE passthrough)     |
| `POST /api/inference/chat/completions?base=`          | Proxy `<base>/chat/completions` (byte-for-byte SSE passthrough) |

### Generation-parameter presets

Named JSON blobs consumed by the Inference panel's preset picker.
Read-only bundled examples at `<repo>/generation_config/` are merged
with user-writable presets at `~/.forgather/generation_config/`.

| Endpoint                                                  | Purpose                                                        |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/generation-configs`                             | List presets (`{name, builtin}[]`)                             |
| `GET /api/generation-configs/{name}`                      | Load one preset (user copy wins over bundled)                  |
| `PUT /api/generation-configs/{name}` `{…params…}`         | Save / overwrite — lands in `~/.forgather/generation_config/`  |
| `DELETE /api/generation-configs/{name}`                   | Delete a user preset (403 if it only exists as a bundled one)  |
