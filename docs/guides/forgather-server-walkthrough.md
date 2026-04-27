# Forgather server: end-to-end walkthrough

This guide walks you from a fresh install through training a small model
and chatting with it — all from the Forgather server's web UI. It's
intended as a first introduction to the UI; once you've done the round
trip you'll have seen most of the major panels and how they fit
together.

**What you'll do:**

1. [Install Forgather + build the web UI](#1-install-and-build)
2. [Start the server and connect](#2-start-the-server-and-connect)
3. [Serve the docs (optional)](#3-serve-the-docs)
4. [Find the Tiny Llama tutorial project](#4-find-the-tiny-llama-project)
5. [Inspect the configuration](#5-inspect-the-configuration)
6. [Queue and dispatch a training job](#6-queue-and-dispatch-a-training-job)
7. [Watch the run](#7-watch-the-run)
8. [Serve the trained model](#8-serve-the-trained-model)
9. [Generate text from the new model](#9-generate-text)

**What you'll need:**

- A Linux machine with at least one CUDA-capable GPU (the example uses
  two; one works too — see below).
- Python 3.12+ and Node.js / npm. See
  [Getting Started](../getting-started/README.md) for distro-specific
  install commands.
- A local browser. If your training host is remote, you'll set up SSH
  port forwarding in step 2.

The whole walkthrough takes ~10–15 minutes once everything is
installed, with most of that being the actual training run (~2 min on
an RTX 4090, longer on smaller GPUs).

---

## 1. Install and build

If you haven't already, follow [Getting Started](../getting-started/README.md)
through the install + cut-cross-entropy steps. That gets you a working
`forgather` CLI and its dependencies.

Then build the web UI:

```bash
cd tools/forgather_server/webui
npm install          # ~2 minutes on first run, fetches Vite + React
                     # + Monaco + viz-js (the WASM Graphviz)
npm run build        # produces webui/dist/
```

`npm install` pulls in a sizeable Node dependency tree the first time
through — go grab a coffee. The build itself is fast. You only need
to repeat this if you pull changes that touch `webui/src/`.

> **Heads-up:** if you start the server before the `dist/` directory
> exists, the API still works but the root URL returns 404 with no
> message. Easy to misdiagnose as a port-forwarding issue.

## 2. Start the server and connect

```bash
forgather server
```

Defaults to `http://127.0.0.1:8765/`. If your training host is local,
just open that URL in your browser.

If the host is remote, set up SSH port forwarding from your laptop —
forwarding the canonical ports for the Forgather server, inference
jobs, TensorBoard, and MkDocs all at once is convenient because every
spawned tool lives at a known port:

```bash
ssh -L 8765:localhost:8765 \
    -L 8137:localhost:8137 \
    -L 6006:localhost:6006 \
    -L 8000:localhost:8000 \
    user@dev-host
```

Then open <http://localhost:8765/> on the laptop. The
[Getting Started SSH section](../getting-started/README.md#accessing-a-remote-server-over-ssh)
has a `~/.ssh/config` snippet you can drop in to make this permanent.

> **Heads-up: prefer `localhost` over `127.0.0.1`.** They're not
> always interchangeable on the client side. Some browser
> environments — Chromebook in particular — resolve `localhost`
> through the SSH tunnel as expected, but `127.0.0.1` hits the
> Chromebook's own loopback (which has nothing listening) and
> fails to connect. macOS and most Linux desktops treat the two
> identically, but `localhost` is the safer default. Same goes
> for the spawned-tool URLs further down (TensorBoard, MkDocs,
> inference servers): if a job card's clickable link doesn't
> resolve, swap any `127.0.0.1` for `localhost` and try again.

The sidebar's five collapsible groups (Views, Tools, Search Roots,
Projects, Files) are all closed on first boot — expand the ones you
want.

![Forgather server first-load view, sidebar with all sections collapsed](screenshots/01-first-load.png)

## 3. Serve the docs

This step is optional but useful: the same docs you're reading now
can be served locally from the running server, which is handy for
flipping between the walkthrough and the live UI.

Open the **Tools** group in the sidebar and click **📖 MkDocs…**.
The modal pre-fills the right `mkdocs.yml` (the bundled one at the
repo root); leave the rest at defaults and submit.

![MkDocs… modal with default values](screenshots/02-mkdocs-modal.png)

> **Heads-up:** the *first* `mkdocs serve` build is slow — a couple
> of minutes typically — because it has to render all the example
> notebooks (`mkdocs-jupyter`). Subsequent rebuilds are quick.

Once the job's running, its card in the **Jobs** panel shows a
clickable URL (port 8000 by default). With the SSH forwards in place,
that link resolves transparently from the laptop. You now have these
docs at <http://localhost:8000/> alongside the UI you're using.

![Jobs panel showing the running mkdocs job with its clickable URL](screenshots/03-mkdocs-job-card.png)

## 4. Find the Tiny Llama project

Expand the **Projects** group in the sidebar. You should see a
workspace tree clustered by `forgather_workspace/` directory; the
bundled examples live under the `examples/` workspace. Drill into
**examples → tutorials → tiny_llama**.

![Projects panel with tiny_llama project highlighted](screenshots/04-projects-tree.png)

Clicking the project node selects its default config (`v2.yaml`) and
opens the project's README in the **info** tab. Take a moment to skim
it — the project trains a ~4M-parameter Llama on a subset of the
TinyStories dataset, and the README explains what's going on.

## 5. Inspect the configuration

The config viewer has three tabs: `info` (the README, currently shown),
`pp` (preprocessed YAML), and `templates` (template-dependency view).

Click **templates** to see the configuration's template graph. The
left panel shows the trefs view by default — every template that
contributes to `v2.yaml`'s materialized configuration, with arrows
showing inheritance and includes. Clicking a node loads its source
in the right panel.

![Templates tab showing the trefs graph for v2.yaml](screenshots/05-trefs-graph.png)

Switch the left-panel mode bar to **tlist** to see the same templates
listed alphabetically by category instead of as a graph. Both views
are useful — trefs for understanding *which* templates compose the
config, tlist for finding a specific template by name.

Click **pp** to see the fully preprocessed YAML — the same thing
`forgather pp` would print on the CLI. This is what the training
script actually receives. Worth a quick scroll-through to see how
much the templates expand into.

![pp tab showing the preprocessed v2.yaml](screenshots/06-pp-tab.png)

## 6. Queue and dispatch a training job

Before submitting, it's worth understanding the queue/scheduler
split: jobs are *enqueued* (added to the waiting queue) and then
*dispatched* (handed to a process and assigned GPUs). The dispatcher
runs on a 2-second tick and picks idle GPUs based on priority + GPU
policies.

You can pause dispatch independently of enqueueing — useful when you
want to inspect what's about to run before it actually starts. Click
the **▶/⏸** button in the sidebar header (next to ⟳ Refresh) to
toggle. **⏸** means dispatch is paused; new submissions sit in the
queue waiting.

For this walkthrough, pause the dispatcher first so you can see the
job in the queue panel before it kicks off:

![Sidebar header showing the scheduler paused (⏸ button)](screenshots/07-scheduler-paused.png)

If you have already run the Tiny Llama tutorial, clean the output artifacts by 
clicking on **Clean Output** first.

Now back to **Projects → examples → tutorials → tiny_llama**. The
config viewer's header has action buttons including **▶ Run**. Click
it to open the submit modal.

![tiny_llama config viewer with Run button highlighted](screenshots/08-run-button.png)

The submit modal exposes the config's dynamic args, requested GPU
count, and priority. The default `v2.yaml` config is set up to use however
many GPUs are assigned; if you have more than one GPU, change the **Requested GPUs** field to
the number of GPUs to use (the config will adapt — single-GPU training still works, just
without DDP). Leave the other fields at their defaults and submit.

![Submit modal with v2.yaml's dynamic args and GPU=2](screenshots/09-submit-modal.png)

Switch to the **Queue** view (📋 in the sidebar's Views group). The
job appears at the top of the list with status `pending`, waiting
for the dispatcher.

![Queue panel showing the queued tiny_llama job](screenshots/10-queue-pending.png)

Now click the **⏸/▶** button in the sidebar header to resume
dispatch. Within a tick or two the scheduler picks GPUs, marks the
job `starting`, and then `running`. The job moves out of the queue
and into the **Jobs** panel.

## 7. Watch the run

Switch to **Jobs** (⚙ in Views). Your training job is the first card,
showing live status pills (loss, lr, grad_norm, epoch, tok/s, peak
memory) plus a progress bar.

![Jobs panel with the training job card live-updating](screenshots/11-jobs-card-running.png)

Toggle **⊞ Show TTY** at the top of the panel. The view splits
horizontally; clicking the job card routes its captured stdout/stderr
to the bottom pane. Loss / lr lines stream in as the trainer reports
them — it's the same output `forgather train` would print in your
terminal, just captured server-side so you can scroll back through it.

![Jobs panel with TTY split-view showing training log](screenshots/12-jobs-tty-split.png)

Flip to the **GPUs** view (🖥) to see live utilization, memory, power,
and temperature. The GPUs assigned to your job glow blue and show a
process chip mapping back to the running job's config name; idle
GPUs are dimmed.

![GPUs panel showing the assigned GPUs busy with the training job](screenshots/13-gpus-panel.png)

Wait for the run to finish — about 2 minutes on an RTX 4090, longer
on smaller cards. When it does, the job card flips to `done`, the
GPUs go idle, and the loss should have come down to somewhere around
2.5 (TinyStories is friendly to small models).

If you return to the projects panel, you will see that the outputs have been associated with the training run.

![Project list with completed logs and checkpoint](screenshots/13.1-completed-artifacts.png)

You can summarize the run by clicking on the completed log and selecting the "summary" tab.

![Training summary](screenshots/13.2-logs-summary.png)

…or scroll the TTY pane in the UI to the bottom to see the trainer's
own summary line.

## 8. Serve the trained model

The trained checkpoint lands at
`examples/tutorials/tiny_llama/output_models/tiny_llama/`. To chat
with it (such as it is), spawn an inference server.

In **Projects**, with `v2.yaml` selected, the config viewer's header
now also shows **🔮 Serve Inference…** and **⚖ Evaluate…** buttons —
they appear once a config has at least one checkpoint on disk.

Click **🔮 Serve Inference…**. The modal pre-fills the model output
dir; leave the dtype / attention / cache impl at defaults and submit.

![Config context meu](screenshots/13.3-config-context-menu.png)


![Serve Inference modal with the trained tiny_llama model](screenshots/14-serve-inference-modal.png)

The inference job appears in the **Jobs** panel like the training
job did, but with a clickable URL on its card — port 8137 by default.
Wait for the job to finish loading the model (the TTY shows a "ready"
message); usually takes ~10 seconds for a 4M model.

![Jobs panel with the inference server running](screenshots/15-inference-job-card.png)

## 9. Generate text

Switch to **Inference** (🔮 in Views). The view has three sub-tabs:
**Model**, **Completion**, **Chat**.

Start in **Model**:

1. Click the **Running inference servers** picker — the inference
   job you just started appears as an option. Selecting it auto-fills
   the base URL.
2. Click **Fetch models** to discover the model id the server
   advertises (`tiny_llama` or similar). Pick it.
3. Optionally apply a generation preset from the picker — `creative`
   produces livelier outputs, `precise` is more deterministic. The
   `creative` preset is a good fit for TinyStories-style stories.

![Inference Model tab with running server selected and creative preset applied](screenshots/16-inference-model-tab.png)

Switch to **Completion**. In the textarea, type:

```
Once upon a time
```

…and click **Send**. The streamed output appears below. With a
4M-parameter model trained for two minutes, you should get a
reasonably coherent (if simple) short story.

![Completion tab with "Once upon a time" prompt and generated story](screenshots/17-completion-output.png)

The status line under the textarea reports tokens generated and
elapsed time. Try a few prompts to get a feel for how the model
behaves; flip back to **Model** to swap presets and see how the
distribution changes.

## What's next

You've now seen most of the major panels. Some directions for
follow-up:

- The [Tiny Llama tutorial](../tutorials/tiny_llama/README.md) covers
  the same project from the CLI side, with deeper notes on the
  config's structure, TensorBoard monitoring, loss plots, and
  programmatic model loading.
- The [Forgather server README](../forgather-server.md) is the
  reference for every panel, every endpoint, and every context
  menu — useful when you want to know "what does this button do"
  without reading source.
- Right-click context menus exist on workspaces, projects, configs,
  search roots, file-tree rows, GPU cards, and Job cards. Each has
  scope-appropriate actions (delete, rename, cut/copy/paste, force
  kill, etc.). Worth poking around once you've finished the basic
  flow.
- Try editing a config: right-click the project → **📄 New Config…**
  for a blank, or click ✎ Edit on a template node in the trefs view
  to open it in the **Edit** panel's tabbed Monaco editor with full
  syntax highlighting for Forgather's YAML+Jinja2 dialect.

Have fun.
