# Forgather Server Architecture

> Part of the [Forgather Server](./README.md) docs. Internals: cluster mode
> (multi-node), persistent on-disk state and garbage collection, dev mode,
> the directory layout, and the one-paragraph design summary. See the
> [API reference](./API.md) for endpoints and the [README](./README.md) for
> usage.

## Cluster mode (multi-node, prototype)

The server is always part of a peer-to-peer cluster of other forgather
servers on the same LAN. The cluster name defaults to `default`;
`--cluster <name>` overrides it. A single-node setup left on the
`default` name advertises on mDNS but only ever sees itself, so the
single-node experience is unchanged — the cluster machinery is simply
always available for the coordination features that depend on it.

```bash
# Default cluster — name "default", advertises on mDNS, peers with
# any other server also on "default" reachable on the LAN.
forgather server

# Named cluster: peer only with other servers using the same name.
# Bind to all interfaces so peers can reach the API across the network.
forgather server -H 0.0.0.0 --cluster lab
```

**Cluster name scoping.** Only servers running with the same
`--cluster NAME` see each other. Two unrelated clusters on the same
LAN will not auto-merge, and a host left on `default` will not merge
with a named cluster. The name is per-invocation (not persisted), so a
host can move between clusters by restarting with a different flag.

**Node identity.** Each host mints a stable UUID at first cluster
startup, persisted at `~/.config/forgather/cluster/node_id` (mode 0600).
The UUID survives hostname changes, NIC swaps, and cluster-name
changes. Master is selected deterministically as the lowest UUID
among reachable members; no election round-trip.

**Discovery.** mDNS / Zeroconf, advertising `_forgather._tcp` with
TXT records `cluster=<name>`, `node_id=<uuid>`, `version=<x.y.z>`,
`hostname=<host>`. Peers without a matching cluster TXT are ignored.

Address advertisement uses `psutil.net_if_addrs()` to enumerate real
LAN IPs — `socket.gethostname()` is unreliable on Linux because of
`/etc/hosts` artifacts like `127.0.1.1`. Common virtual interface
prefixes (`docker*`, `br-*`, `veth*`, `tun*`, `tap*`, `wg*`, etc.)
are filtered out because they share addresses across hosts and
typically don't carry inter-host traffic.

**When auto-detection fails:** if the server runs inside a container
whose network namespace hides the host's real interfaces, psutil may
see only loopback or only a container bridge. The auto-detector
falls back to `127.0.0.1` and emits a WARNING; peers on other hosts
will not be able to reach you in that state. Use `--cluster-address
<ip>` (repeatable) to specify the address(es) you want advertised:

```bash
# Inside a container without --network host: tell forgather what
# host-routable address to put in the mDNS record.
forgather server -H 0.0.0.0 --cluster lab --cluster-address 192.168.1.27
```

To diagnose what's happening on a running cluster, the server logs
which interface(s) it advertises at startup, and which local
interface it inferred for each incoming peer (matched by subnet
against the peer's advertised address). Look for
`mDNS peer <hostname> at <addr>:<port> via local iface <iface>`
in the log to confirm peers are showing up on the interface you
expect.

**Membership.** Every 5 s each node GETs `/api/cluster/members` from
every other known peer, merges the returned member tables, and marks
silent peers as unreachable after 15 s (two full peer-pull cycles at
the default 5 s cadence). Unreachable peers are kept in the table
(union-of-ever-seen view) — the user agreed model is to flag, not
delete. **Liveness is owned by the direct peer-pull alone**: mDNS
discovery and transitively-reported members are tagged identity-only
and never refresh `last_seen` / flip `reachable=True` on an existing
entry. New members coming in via discovery / peer_report start
`reachable=False` until a direct pull confirms them — otherwise a
stale mDNS cache or a third node restarting with an old member
table could resurrect a dead peer for one sweep window.

**Security.** Inter-node API calls authenticate via mTLS — every
peer presents a CA-signed client certificate during the TLS
handshake, and the auth middleware accepts the call without a
bearer token only for paths on a narrow allow-list. The threat
model assumes the cluster as a whole is trusted (consistent with
the torch.distributed assumption that already underpins multi-host
training — any peer can submit jobs, which is arbitrary code
execution). The carve-out is:

- GET on the read-only inter-node endpoints (members, self,
  master, gpus_local, bandwidth_local, training_status_local,
  dataset_servers_local, dataset_inventory, dataset_servers,
  diloco_servers_local, diloco_servers,
  dataset_router/resolve, issue_url_token) — see
  `auth._PEER_ALLOWED_PATHS`.
- POST on a smaller mutation allow-list (`gpu_policy_local`,
  `training_local`, `training_cancel_local`,
  `dataset_servers/refresh`) — see `auth._PEER_ALLOWED_MUTATIONS`.
- Per-node webui auth (bearer token / browser session) is unchanged
  — the mTLS carve-out applies only to inter-node traffic, not to
  browsers.

**Cross-node SSO.** Clicking a peer in the sidebar Nodes group calls
`POST /api/cluster/peer_session` on the local node. The local node
then GETs `/api/cluster/issue_url_token` on the target peer over
mTLS; the peer mints a 60 s **single-use** URL token (distinct from
its persistent bearer at `~/.config/forgather/server/auth_token`)
and returns it. The browser opens `https://peer:port/?token=…` in
a new tab, the peer's `LoginGate` consumes the token via
`/api/auth/login`, strips it from the address bar, and replaces it
with a session cookie. A leaked URL only exposes a 60 s single-use
window, not the long-lived bearer.

If you don't trust the operators of every node that could reach you
on the LAN, keep the server on loopback (`-H 127.0.0.1`, the default)
or give your cluster a private `--cluster NAME` so unrelated nodes on
`default` never peer with you.

**Cluster view.** A 🖧 **Cluster** entry appears in the sidebar. The
view is a Datasets-style tabbed panel with four
tabs, all kept mounted so scroll position and in-flight queries
survive switching:

- **jobs** — the Cluster Jobs card (multi-node training bundles);
  see *Cluster Jobs panel* below.
- **network** — pairwise latency + bandwidth probe. `Refresh`
  walks the peer list sequentially (so two simultaneous bulk
  transfers don't saturate the local NIC), per peer doing first a
  30-sample HTTP latency probe — min / median / max ms,
  warmup-trimmed — and then an adaptive parallel-stream **raw-TCP**
  bandwidth probe (4 streams in flight, sized for ~2 s of
  steady-state transfer per stream). The data channel is plain TCP
  via a one-shot ephemeral listener so Python's `ssl` module isn't
  the bottleneck on fast links; the control channel still flows
  over the authenticated mTLS HTTPS path. Each row in the table
  swaps its Latency / Throughput cells to "Measuring…" while that
  peer is in flight so the operator sees per-peer progress.
- **nodes** — per-peer rollup: hostname, master/peer/this-server
  tags, version chips (yellow on divergence), a collapsible
  **Interfaces** list, and a collapsible **GPUs (N · M idle)** list
  — one row per GPU with index/name/memory/util/temp/status. Click
  a GPU row to toggle disabled; mutations route through the master
  proxy.
- **datasets** — the master-aggregated dataset_server / dataset
  inventory previously under *Datasets → Cluster*. Click a dataset
  row to navigate to *Datasets → Explore* with the first healthy
  host's first split pre-selected (see *Cross-view click-through*
  below).

Peer right-click context menu (kill processes, set min-priority)
is intentionally absent in v1 — those mutations route through
future by-node proxy work.

**Sidebar Nodes group.** A second cluster-only surface in the
sidebar above Views lists every peer by hostname with a tri-state
health dot (green / yellow / red — see *Node health* below) and
hands one-click SSO to the peer's webui. Distinct from the
Cluster view in Views: this surface is about *navigating between
nodes*; the Cluster view is about the cluster's internal state.

**Node health.** Each peer's dot reflects three states:

- **green** — reachable and headline versions match the cluster
  majority.
- **yellow** — HTTP-reachable but at least one headline version
  (`forgather`, `torch`, `nccl`, `transformers`) is missing on this
  node or differs from the majority. Catches cases like a peer's
  nvml/driver glitch silently dropping its `nccl` version while the
  node otherwise stays up. The row tooltip lists the disagreements;
  click still works so the operator can SSO in and investigate.
- **red** — last peer-pull failed and `last_seen` exceeded the
  unreachable threshold (15 s by default — two full peer-pull
  cycles).

The dot reflects the live `member.reachable` flag, which is only
refreshed by a *direct* peer-pull GET to that node's
`/api/cluster/members`. Transitive entries reported by other peers
and mDNS-cached records are tagged as identity-only and never
vouch for liveness — so a third node restarting with a stale
member table can't resurrect a dead peer's dot to green.

**Pre-flight probe (Phase 2).** Each member entry carries a
``probe`` payload computed once at startup and propagated via
peer-pull:

- **Versions**: ``forgather``, ``torch`` + CUDA runtime + ``nccl``,
  ``transformers``, ``python``, platform string. Surfaced inline in
  every node's header as compact chips. When a node's value diverges
  from the cluster majority for any headline key, the chip turns
  yellow and tooltips with the divergence; the cluster header gets a
  "version mismatch" tag. Multi-node training is exquisitely
  sensitive to ``torch`` / ``nccl`` mismatches across hosts — the
  Samantha tutorial spends pages on this — so seeing it at a glance
  before launching anything matters.
- **Network interfaces**: every IPv4 interface with address, netmask,
  CIDR, link state, and link speed (when reported by the kernel).
  Collapsible per-node panel. Useful when picking
  ``NCCL_SOCKET_IFNAME`` for multi-node training, and as a quick
  sanity check that cluster-internal traffic is on the interface
  you expect.
- **CPU / RAM summary**: logical + physical core count and total
  RAM in GiB, shown in the node header next to the address.

**Network probe (Phase 2).** Lives on the **Cluster view → network**
tab. On-demand only, triggered by **Refresh** so the network stays
idle the rest of the time; sequential across peers because two
simultaneous bulk transfers would saturate the local NIC and
under-report each link.

For each peer the orchestrator runs two passes in order:

1. **Latency** — 30 keepalived round-trips to
   ``/api/cluster/latency_local`` (empty 200 over the mTLS HTTPS
   channel). First 3 samples discarded to skip TCP-connect /
   TLS-handshake / DNS spikes; report min / median / max ms.
2. **Bandwidth** — adaptive parallel-stream **raw TCP** transfer.
   Coordination over HTTPS: ``POST /api/cluster/bandwidth_prep``
   asks the peer to open a one-shot ``asyncio.start_server``
   listener on ``0.0.0.0:0`` and returns ``(port, 32-byte token)``.
   The local node then opens 4 concurrent plain TCP connections to
   that port, sends the token, and times the receive. The peer
   verifies the token before serving bytes; the listener self-closes
   after the first served connection (or 30 s timeout). Adaptive
   sizing: a single-stream probe estimates the rate, then each of
   the 4 streams pulls enough bytes to take ~2 s of steady-state
   transfer.

The raw-TCP data path bypasses Python's ``ssl`` module, which
otherwise capped single-stream throughput at ~2 Gbps even on a
10 Gbps wire. The bytes themselves are deterministic zero data with
no useful information content, so removing TLS from the data channel
adds no useful capability to an attacker who'd already need to be
inside the cluster LAN's trust boundary (and the 32-byte handshake
token prevents a coincidental port scan during a measurement from
poisoning the result).

Results cached for 1 hour. ``GET /api/cluster/bandwidth`` /
``/api/cluster/latency`` return cached entries;
``POST .../refresh`` re-runs across all peers;
``POST .../refresh_one/{node_id}`` re-runs against one peer (used
by the per-peer "Measuring…" progress feedback in the table).

**Multi-node training submit.** Multi-node submits are folded into
the regular Run dialog — the same dialog that opens from a config's
**▶ Run** action in the project tree or config viewer. A collapsible
**Multi-node** panel sits above the Dynamic arguments section. The
local node is pre-checked as the only participant by default, so a
webui that just clicks Submit gets identical single-node behaviour to
a single-host run. Adding peers turns it into a fanout.

In the panel, each row is a cluster member with five columns: a
**Use** checkbox, the node's hostname/address, a **GPUs** spinner
bounded by the node's actual hardware (with a `(N idle of M)` hint
matching the single-node dialog — wire format stays
``nproc_per_node`` because that's what torchrun expects, the local
scheduler translates it into nproc + ``CUDA_VISIBLE_DEVICES``), an
**NCCL iface** dropdown (or text field on nodes whose probe didn't
report any interfaces), and a **rdzv host** radio. The participant
table caps at ~9 rows then scrolls inside the panel so the
rdzv-port row, version warnings, and help line stay anchored even
with many cluster members.

Project + config come from the dialog itself (the config you
right-clicked Run on), and the dialog's existing dynamic-args + GPU
+ priority knobs flow through to every peer in the fanout. So
per-config overrides — dataset paths, `max_steps`, `lr`, etc. —
reach every node the same way they reach a single-node run.

When cluster mode is active, the dialog's single-node "GPUs"
spinner + nproc help text + gpuMismatch notice are **hidden**: the
panel's per-node GPUs column is the only knob, and showing both
got confusing. Priority stays visible because it applies to both
submit paths.

Last-used multi-node settings (participants, per-node GPUs, iface,
rdzv host/port, mismatch acknowledgement) persist in the same
per-config overrides cache as the dynamic-args, so a config "opens
where you left off" for both submit modes. Reset to defaults
clears multi-node state alongside the dynamic-args.

The **Cluster view → jobs** tab lists the running and
recently-finished bundles, with status, per-rank assignment, and a
Cancel action. There is no longer a "+ Multi-node training" button
on that panel — the submit flow is the regular Run dialog.

On submit, the master:

1. Validates participants are reachable and probe data shows
   matching ``forgather`` / ``torch`` / ``nccl`` / ``transformers``
   versions across the selected set; mismatches return HTTP 409
   unless ``allow_version_mismatch=true`` is passed.
2. Generates a unique ``rdzv_id`` and computes
   ``rdzv_endpoint = <rdzv_node.address>:<rdzv_port>``
   (default port ``29400``).
3. Assigns ``node_rank`` by request order — the rdzv host typically
   ends up rank 0 because the modal puts the master first.
4. Fans out a ``POST /api/cluster/training_local`` to each
   participant with that node's per-rank torchrun args
   (``--nnodes``, ``--node-rank``, ``--rdzv-backend=c10d``,
   ``--rdzv-endpoint``, ``--rdzv-id``, ``--nproc-per-node``,
   ``--rdzv-conf is_host=true|false``). Each peer also gets
   ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, and
   ``TP_SOCKET_IFNAME`` in ``extra_env``, all set to the same
   interface (NCCL for CUDA collectives, Gloo for CPU collectives,
   tensorpipe for RPC — each derives its advertised address
   independently and they must all be pinned together). The
   interface name comes from the operator's modal selection when
   set; otherwise the server auto-derives it by matching the
   member's advertised address against its probe's interface table
   (``_derive_iface_from_member`` in ``routes/cluster.py``). If no
   interface can be derived (probe missing, address mismatch) the
   submit fails with HTTP 422 rather than spawning a job that will
   deadlock in ``connectFullMesh``. The peer's local scheduler
   picks up the queue item and spawns torchrun in rendezvous mode
   (no ``--standalone``).
   The two ``/etc/hosts`` workarounds we have to apply explicitly:
   - ``is_host`` because torch's c10d backend autodetects "am I
     the rendezvous host?" by resolving ``socket.gethostname()``
     and comparing it to ``rdzv_endpoint``. On Debian/Ubuntu the
     system hostname resolves to ``127.0.1.1`` via ``/etc/hosts``,
     so the comparison silently fails on every node and *no* node
     binds the TCPStore.
   - ``GLOO_SOCKET_IFNAME`` (and ``TP_SOCKET_IFNAME``) because
     once the rendezvous succeeds, Gloo's ``connectFullMesh`` has
     each rank publish its own address — also via
     ``socket.gethostname()`` — so peers receive ``127.0.1.1`` and
     connect to their own loopback instead of each other.
5. Records a ClusterJob bundle linking the per-node queue ids back
   to a single ``cluster_job_id``. Listed via
   ``GET /api/cluster/jobs``; cancel via
   ``POST /api/cluster/jobs/{id}/cancel`` fans out a cancel to each
   participant. Bundle creation and cancellation are journaled via
   ``cluster_journal`` so Phase 4's replication seam covers
   multi-node lifecycle.

If a fanout step fails partway through, the master rolls back by
issuing cancels to the participants it already enqueued on, then
returns the original error. There's no half-submitted state.

**Status rollup.** ``GET /api/cluster/jobs`` and
``GET /api/cluster/jobs/{id}`` compute each bundle's live status by
fanning out to every member's ``GET /api/cluster/training_status_local``
(read-only; in the peer-allowed list). The master reads its own
participant's status directly from local job_records, queries every
remote peer in parallel, and rolls the per-rank statuses up via
priority order: ``failed > running > cancelled > queued > done``.
"done" requires *every* member to be terminal — partial completion
is ambiguous, not done. Once the rollup reaches a terminal state
the bundle's own ``status`` field is promoted in place
(``done`` / ``failed`` / ``cancelled``) so subsequent reads
short-circuit without fanning out. Slow or unreachable peers
contribute ``current_status="unknown"`` for that rank rather than
blocking the whole list.

**Non-master proxying.** Bundle records live on the master only. To
keep every webui in the cluster showing the same job list, non-master
nodes proxy ``GET /api/cluster/jobs`` to the master (which is in the
peer-allowed list, so no bearer is needed for the inter-node call).
Master-unreachable falls through to the local empty list rather than
erroring — the page must keep rendering during a master failover.

**Asymmetric topologies.** The fanout itself doesn't care whether
participants have matching ``nproc_per_node`` (the cluster of
operators we tested with had a 1-GPU box and a 2-GPU box).
Deeper, the trainer's per-node coordination groups (used by
``main_process_first`` for cached dataset preprocessing) discover
topology via an ``all_gather_object`` on hostnames rather than the
old ``world_size // local_world_size`` integer math, so heterogeneous
layouts produce correct local groups. Single-rank nodes skip
local-group creation but still participate in peer nodes' group
creation calls so the world-collective stays balanced.

**Limitations to be aware of in v1:**

- Project paths are assumed to resolve at the same location on every
  participant. There is no automatic config staging.
- Per-node TTY logs and job control still run through each peer's
  own webui — there's no cross-node log aggregation. Open the peer's
  webui in another tab to watch its rank's torchrun output.
- ``TrainerControlCallback`` registers only on rank 0 and binds its
  HTTP control endpoint to ``127.0.0.1`` — so live save/stop/abort
  commands have to be issued from the webui or CLI on whichever node
  hosts rank 0. The Cluster Jobs panel's Cancel button still works
  from any node because it routes through the JobRecord-level
  cancel-fanout, not the trainer-control HTTP layer.
- The version check is advisory at the headline-key level
  (``forgather`` / ``torch`` / ``nccl`` / ``transformers``). It
  doesn't compare CUDA toolkit, transformers patch versions, etc.;
  add those to ``cluster_probe.py`` if a real divergence bites.
- ``peak_hardware_flops`` for MFU is auto-detected from rank 0's GPU
  only and multiplied by world_size. For a homogeneous cluster this
  is correct; for a heterogeneous cluster (e.g. mixed 3090 + 4090,
  or pairing a Spark with a desktop GPU) the reported MFU is
  meaningless. Workaround: set
  ``peak_hardware_flops`` explicitly per-config, or stick to
  homogeneous training clusters until probe-driven aggregation
  lands.

**Operational notes for multi-node operation:**

- **Container PID 1 must reap orphan grandchildren.** Forgather's
  Python server doesn't see the worker subprocesses spawned by
  torchrun (those are torchrun's children, not ours), so when
  torchrun gets killed the workers re-parent to PID 1 of the
  container's pid namespace. If PID 1 is ``sleep infinity`` (the
  pre-init default of ``docker/run``) it doesn't call ``wait()``
  and the workers pile up as zombies. ``docker/run`` now passes
  ``--init`` so Docker's bundled ``tini`` becomes PID 1 and reaps
  orphans regardless of parentage. Existing containers need
  recreation to pick this up: ``docker/run --rm && docker/run``.

- **Diagnosing hangs with faulthandler.** ``train_script.py``
  enables Python's ``faulthandler`` at startup and registers
  ``SIGUSR1`` for live thread dumps:
  - On a crash (SIGSEGV / SIGFPE / SIGABRT / SIGBUS / SIGILL), every
    thread's Python stack is dumped to stderr — which torchrun
    routes to the per-rank TTY log. Silent rank deaths (CUDA driver
    assertions, OOM-kills, C++ exceptions in background threads)
    leave a trace where they used to leave nothing.
  - To inspect a hung rank live: ``kill -USR1 <pid>`` against the
    rank's worker process. Faulthandler dumps every thread's stack
    to the TTY log without killing the process. Same idiom as
    ``py-spy dump``, but works inside containers that strip
    ``CAP_SYS_PTRACE`` (which most production containers do, and our
    forgather-dev container in particular). The dump in the TTY
    log shows exactly which ``dist.*`` collective each rank is
    blocked in; matching them up across ranks gives you the
    deadlock site immediately.
  - The per-rank ``DistributedEnvironment(...)`` line includes
    ``host=<hostname>`` so you can correlate "rank N is hung" with
    the actual node it lives on without cross-referencing
    ``cluster_jobs``.

- **Kill verifies process exit.** ``abort`` and ``force-kill`` poll
  for the PID to actually exit (up to 2 s) after issuing the
  signal. If the process is still alive (e.g. stuck in an
  uninterruptible CUDA driver call), the JobRecord's ``error``
  field is populated with a message pointing at the lingering PID
  — the record stays visible in the UI instead of silently
  disappearing while the GPU is still pinned.

- **``force-kill`` reaps orphans behind terminal records.** A stuck
  process can outlive its JobRecord: the job is marked terminal
  (``done``/``failed``/``aborted``) and drops out of the UI while the PID
  lingers, holding a GPU. Soft ``kill`` (SIGTERM) keeps the terminal guard
  (terminal == "already gone"), but ``force-kill`` (SIGKILL, "do whatever it
  takes") still signals the process group when the record's PID is alive — so
  the operator can reap a leaked worker without a stale-but-live record getting
  in the way. The terminal record's status is left intact (we only reap its
  leaked process).

- **External-trainer promotion.** A trainer the server did not launch
  (e.g. a foreground ``forgather train`` with ``TrainerControlCallback``)
  writes a control endpoint but has no JobRecord. The scheduler tick
  promotes such a live endpoint into a JobRecord (``externally_launched``,
  ``gpu_indices=[]`` — GPU accounting deferred), so it appears in
  ``forgather job`` with first-class status and is reaped by PID liveness
  like a re-attached job. Scheduler-spawned trainers are excluded (they
  correlate to their own record by PID lineage, plus a one-tick grace), so
  they're never double-counted. An external job is controlled only through the
  graceful relay (``save``/``stop``/``save-stop``/``abort`` hit the trainer's
  own HTTP control endpoint); ``kill``/``force-kill`` are refused for it, since
  the server is not its session leader and signalling its process group would
  hit the operator's shell.

- **Stale endpoint cleanup.** A trainer-control endpoint directory
  (``~/.config/forgather/jobs/job_*/``) left behind by a crashed or
  SIGKILLed trainer can resurface as a phantom "running" job. The server
  treats an endpoint whose PID is dead/zombie/recycled as not-running (it
  drops out of the default Jobs list), and a periodic GC sweep
  (``_gc.sweep_dead_endpoint_dirs``, on the scheduler tick and at startup)
  removes the directory once it is older than the TTL
  (``FORGATHER_ORPHAN_JOB_DIR_TTL_SECONDS``, default 1h) — the reaper the
  removed ``forgather control cleanup`` used to provide. The Jobs panel's
  right-click **Remove stale endpoint** action still removes one on demand;
  toggle "include dead endpoints" to see them.

- **Single-writer checkpoints on shared FS.** When several ranks
  share a filesystem (NFS, the typical multi-node setup), only one
  rank globally writes the model shard files. The CheckpointManager
  honours ``save_on_each_node=False`` (the documented default for
  shared storage) by gating the shard-file save loop on
  ``_should_save_common``, so concurrent writers can't race on the
  same shard paths. Pipeline-parallel runs (``save_on_all_ranks=True``)
  still have every rank write its own non-overlapping shards as
  before — different stages own disjoint FQNs.

**State.** Cluster runtime state lives at `~/.config/forgather/cluster/`:

```
~/.config/forgather/cluster/
├── node_id              # persistent UUID (0600)
└── journal/
    └── events.jsonl     # append-only event log (Phase 4 seam)
```

The journal is a future-proofing seam: Phase 4 will route every
global-state mutation (queue, GPU policy, cluster jobs) through
append-only events so master/backup replication can be added later
without restructuring storage. v1 emits no events to the journal yet.

#### Multi-node dataset routing (`FORGATHER_DATASET_SERVER=auto`)

In cluster mode the master keeps a deduped inventory of every
`dataset_server` known to any peer (both spawned via the webui's
Tools menu and registered via the per-node user-registry). The
inventory drives a tiny router exposed at

```
GET /api/cluster/dataset_router/resolve?path=<dataset path>
```

which picks a healthy server at random across the candidate set
(crude load balance) and returns `{base_url, auth_token, server_id}`.
Master-only background loops, started from the lifespan and
self-gated on `cluster.is_self_master`, keep the inventory live:

| loop                                                    | interval               | what it does                                                                 |
|---------------------------------------------------------|------------------------|------------------------------------------------------------------------------|
| `cluster_dataset_inventory.master_collect_servers_loop` | 10 s                   | GET each peer's `/api/cluster/dataset_servers_local`, merge into the set     |
| `cluster_dataset_inventory.master_health_loop`          | 10 s                   | GET `/v1/health` on every dataset server, flip the per-server healthy flag   |
| `cluster_dataset_inventory.master_dataset_refresh_loop` | 10 s (warm-up) / 60 s  | GET `/v1/datasets` + `/v1/local`, rebuild the `local/<name>` routing index   |
| `cluster_inference_inventory.master_collect_servers_loop` | 10 s                 | GET each peer's `/api/cluster/inference_servers_local` for the picker        |
| `cluster_inference_inventory.master_health_loop`        | 10 s                   | GET `/health` on every inference server (root-mounted, not `/v1/health`)     |
| `cluster_diloco_inventory.master_collect_servers_loop`  | 10 s                   | GET each peer's `/api/cluster/diloco_servers_local` for the DiLoCo panel     |
| `cluster_diloco_inventory.master_health_loop`           | 10 s                   | GET `/health` on every DiLoCo server, flip the per-server healthy flag       |

On a master transition the new master clears its inventory and the
router returns `503 Retry-After: 5` until the first dataset-refresh
pass completes. `local/<name>` is a **global** key — two servers
advertising the same name are treated as interchangeable replicas
(intentional, gives operators a knob for redundancy/load-balance).
HF / path requests fall back to "any healthy server" and the
`dataset_server` loads on demand; the resilient client retries on
failure and re-routes to a different server on its next attempt.

To use the router from a training job:

```bash
FORGATHER_DATASET_SERVER=auto forgather train …                # CLI
forgather -p <proj> -t <cfg> submit --global --dataset-source auto …
```

Or pick `Auto (cluster routing)` in any submit modal. The CLI flag
and modal selector both encode `dataset_source={"kind":"auto"}` on
the job_params; the scheduler's `dataset_source.resolve_to_env`
expands that to `FORGATHER_DATASET_SERVER=auto` in the spawn env,
and the resilient client in
`forgather.ml.datasets.resilient_remote_backend` queries the local
forgather_server's resolve endpoint on every (re)connect — so a
peer that dies mid-iteration causes the next attempt to land on a
different healthy peer with no operator intervention.

Diagnostics: `forgather cluster datasets [-v]` prints the deduped
inventory; `forgather cluster resolve <path>` dry-runs the router;
`forgather cluster server <server_id> {status|list|cache|local}`
talks to any cluster server via the master-proxy without needing
the upstream bearer. The **Cluster view → datasets** tab in the
webui surfaces the same payload — server health, refresh ages,
per-server poll counters, and a deduped dataset table with hosts.
Clicking a dataset row navigates to *Datasets → Explore* with the
first healthy host's first split pre-selected.

**Known limits in v1.**

- No global scheduler — peer scheduling decisions are still
  independent. Cluster job submits use a static fanout at submit
  time; there is no live re-balancing or cross-node preemption.
- No file/log streaming through a by-node proxy — to inspect a peer's
  jobs / projects / files outside the Cluster Jobs panel, open that
  peer's webui directly. The "any node sees the same cluster job
  list" proxying covers `/api/cluster/jobs` only, not `/api/jobs` or
  the file/project endpoints.
- ``TrainerControlCallback`` registers only on rank 0 and binds its
  HTTP control endpoint to ``127.0.0.1`` — see "Operational notes"
  above.
- No automatic master failover — the master is whichever reachable
  member has the lowest UUID; if it goes down the cluster keeps
  running with a new master, but in-flight global state (queue
  mutations during the gap) is lost. Phase 4 + Phase 5 work.
- No cross-architecture training (e.g. ARM Spark + x86_64 desktop):
  the version probe surfaces a platform mismatch in the Cluster
  view's Nodes tab (and in the sidebar Nodes dot as yellow) and the
  multi-node submit refuses unless the operator acknowledges, but
  torch wheels and CUDA kernels won't actually interoperate across
  architectures. The check is advisory; the operator is on the
  hook for whether their cluster makes sense.

## Excluding misbehaving GPUs

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

## Persistent state

Everything under `~/.config/forgather/server/` survives restarts:

| File / dir                  | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `search_roots.json`         | Project-discovery roots (seeded on first boot).        |
| `queue.json`                | Queue of items waiting for GPUs.                       |
| `job_records.json`          | Records for jobs the server has launched (any state). |
| `jobs/{queue_id}.tty`       | Captured stdout+stderr for each launched job.          |
| `overrides/{hash}.json`     | Per-config dynamic-args override cache.                |
| `gpu_policy.json`           | Per-GPU runtime policy: disabled + min_priority.       |
| `auth_token`                | Bearer token shared with CLI clients (mode 0600).      |
| `password_hash`             | Optional pbkdf2_sha256 hash for browser logins (0600). |
| `sessions.json`             | Persisted browser sessions (0600). Present only when started with `--persist-sessions`. |
| `server_config.yaml`        | Operator-editable CLI defaults + auto-start services (0600). See [Server config file](./README.md#server-config-file-server_configyaml). |

All state files are written crash-atomically via `_atomic.py`: tmp file
written in the target directory, `fsync` on the fd, then `os.replace`.
Power loss or SIGKILL mid-write never leaves the canonical file
partially written. Every reader tolerates a corrupt / truncated file by
falling back to empty state.

## State directories and GC

Two sibling directories under `~/.config/forgather/` accumulate per-job files,
one per subsystem. They are independent — neither owns the other —
though the server reads the trainer-side directory to correlate
PID-lineage with running JobRecords.

#### `~/.config/forgather/server/jobs/q_*.tty` (server-owned)

The captured stdout/stderr of every job the server dispatches. For
training jobs the scheduler symlinks `q_<id>.tty` to
`<run>/logs/tty.log` once the trainer's `endpoint.json` is correlated,
so users can `tail -f logs/tty.log` from the run directory while the
job is live.

When a JobRecord transitions to a terminal status (`done` / `failed`
/ `aborted`), the scheduler **moves the captured TTY into the run's
`logs/tty.log`**, atomically replacing the symlink with the actual
file. After this the run directory is self-contained — the central
copy under `~/.config/forgather/server/jobs/` is gone. For non-training
jobs (eval, inference, tensorboard, …) there is no `logs_dir` to move
into; their TTY stays in the central directory until the JobRecord is
removed (`DELETE /api/jobs/{id}` or `POST /api/jobs/cleanup`), which
also unlinks it.

A periodic sweep (daily, plus once at server startup) deletes any
`q_*.tty` whose `queue_id` is not referenced by any record or
queued item, mtime older than `FORGATHER_ORPHAN_TTY_TTL_SECONDS`
(default `3600`). Run it on demand with:

```bash
forgather job gc
```

#### `~/.config/forgather/jobs/job_<ts>_<host>_<pid>/` (trainer-owned)

Each `TrainerControlCallback` (added to a Forgather Trainer via the
`callbacks=` argument; see the project-root `CLAUDE.md` for the
boilerplate) creates a per-job directory here on rank 0 and writes
`endpoint.json` with the host:port the trainer's HTTP control API
listens on. On a clean exit the callback both removes
`endpoint.json` and `rmdir`s the directory, so well-behaved runs
leave nothing behind. Crashed runs leak the directory.

The server's periodic GC sweep reaps both kinds of leftover:

- Directories whose `endpoint.json` points at a dead PID (or one that
  the kernel has recycled — verified against `psutil.Process.create_time()`).
- Directories with no `endpoint.json` and mtime older than the TTL
  (`FORGATHER_ORPHAN_JOB_DIR_TTL_SECONDS`, default 3600) — these are
  crash leftovers.

The same sweep that handles orphan TTY files (daily, plus once at
server startup) covers these directories; run it on demand with
`forgather job gc`.

## Re-attach across restart

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

## Dev mode (Vite + hot reload)

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

## Directory layout

```
src/forgather/cli/
├── server.py                  # CLI shim: `forgather server` → backend subprocess
└── wrappers_args.py           # CLI parser registration for `server`

generation_config/             # Bundled generation-parameter presets
│                              #   (read-only from the UI; shadowed by
│                              #    ~/.config/forgather/generation_config/)
├── greedy.json
├── precise.json
├── balanced.json
├── creative.json
├── beam_search.json
└── contrastive.json

tools/forgather_server/
├── server.py                  # uvicorn entry point
├── app.py                     # FastAPI app factory + lifespan (dispatcher loop)
├── paths.py                   # ~/.config/forgather/server/ state helpers
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
│                              #   finalize / update / model / dataset
│                              #   processes; own process group
├── inference_ops.py           # Build inference-server argv
├── eval_ops.py                # Build `forgather eval` argv
├── tensorboard_ops.py         # Build tensorboard argv
├── mkdocs_ops.py              # Build `mkdocs serve` argv
├── convert_ops.py             # Build `forgather convert` argv
├── finalize_ops.py            # Build `forgather finalize` argv
├── update_ops.py              # Build `forgather update` argv
├── model_ops.py               # Build `forgather model` argv
├── dataset_ops.py             # Build `forgather dataset` argv
├── scheduler.py               # Dispatcher loop, GPU allocation,
│                              #   per-job-type spawn, re-attach, reap, abort
├── gpu_monitor.py             # NVML / torch.cuda enumeration,
│                              #   CUDA_VISIBLE_DEVICES allow-list
├── gpu_policy.py              # Runtime per-GPU policy (disabled,
│                              #   min_priority) — persisted
├── agent_profiles_store.py    # Saved agent connection profiles + active
│                              #   selection (JSON, 0600); hot-swap revision
├── agent_tls.py               # Per-profile TLS verify, cert import (TOFU),
│                              #   model listing (Claude SDK / vLLM /v1/models)
├── agent/                     # In-process AI agent (right-sidebar assistant)
│   ├── providers/
│   │   ├── base.py            # ChatProvider seam + normalized events
│   │   └── anthropic.py       # Anthropic SDK adapter (Claude + vLLM via
│   │                          #   base_url); tool_use reassembly + tolerance
│   ├── registry.py            # ToolSpec + risk levels + Proposal (preview)
│   ├── loop.py                # Provider-agnostic loop + server-side gate
│   ├── session.py             # In-memory sessions + pending approvals
│   ├── runtime.py             # Config injection, factories, system prompt
│   ├── tools_readonly.py      # Phase 0 read tools
│   └── tools_authoring.py     # Phase 1 propose/commit tools
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
│   ├── agent.py               # /api/agent/{status,message,approve,reject,
│   │                          #   sessions} — SSE chat + approval gate
│   └── generation_configs.py  # /api/generation-configs/{list,get,put,delete}
└── webui/
    ├── package.json           # Vite, React, TypeScript, Monaco, viz-js,
    │                          #   TanStack Query, react-markdown, remark-gfm
    ├── vite.config.ts         # dev-mode /api → :8765 proxy (REST + WS)
    └── src/
        ├── main.tsx           # React + QueryClientProvider bootstrap
        ├── App.tsx            # Collapsible sidebar (header, Views,
        │                      #   Tools, Services, Search Roots,
        │                      #   ProjectTree, FilesTree, sticky
        │                      #   footer) + main pane; owns view /
        │                      #   selection / tab state and the
        │                      #   scheduler play/pause
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
            │                        #   from sidebar Services)
            ├── MkDocsModal.tsx      # Enqueue `mkdocs serve` job
            │                        #   (sidebar Services — global only)
            ├── ConvertModal.tsx     # Enqueue `forgather convert` job
            │                        #   (sidebar Tools or config / checkpoint
            │                        #   right-click)
            ├── FinalizeModal.tsx    # Enqueue `forgather finalize` job
            │                        #   (sidebar Tools or config / checkpoint
            │                        #   right-click)
            ├── UpdateModal.tsx      # Enqueue `forgather update` job
            │                        #   (sidebar Tools or config / checkpoint
            │                        #   right-click; pre-fills source path
            │                        #   and optional checkpoint)
            ├── ServicesPanel.tsx    # Configured-service rows in the
            │                        #   sidebar Services group (red/green
            │                        #   dots, ▶/⏹/× row controls,
            │                        #   click-through per type)
            ├── LogDetailPanel.tsx   # Selection target for a run/log leaf
            ├── CheckpointDetailPanel.tsx # Selection target for a checkpoint
            ├── EvalDetailPanel.tsx  # Selection target for an evaluation
            ├── RunSummaryView.tsx   # Extracted from legacy models panel
            ├── EvalResultTable.tsx  # Extracted from legacy models panel
            ├── InferencePanel.tsx   # Inference view: model/completion/chat
            │                        #   sub-tabs (Inference launcher lives
            │                        #   in the sidebar Services section)
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
the 🔧 Overrides modal. The scheduler dispatches ten job types —
training (`torchrun`), eval (`forgather eval`), inference
(`tools/inference_server/server.py`), TensorBoard (`tensorboard`),
MkDocs (`mkdocs serve`), convert (`forgather convert`), finalize
(`forgather finalize`), update (`forgather update`), model, and
dataset — all through a common `launcher.spawn_*`
surface that owns its process group via `start_new_session=True` so
jobs survive server restart. Inference
servers spawned this way appear in the Inference panel's "Running
inference servers" picker; the browser talks to them through a
same-origin SSE proxy so CORS / PNA don't get in the way. The frontend
is a Vite/React SPA driven by TanStack Query for caching + background
refresh; persistent server state is plain JSON files under
`~/.config/forgather/server/` so it's inspectable with ordinary tools.

