# Multi-node training with the Forgather server

This guide is for operators who want to run a training job across more
than one machine. It covers the practical setup, the submit flow,
common failure modes you should expect to see, and the diagnostic
tools available when something hangs.

If you've used the Forgather server in single-node mode (one host,
one or more GPUs), the multi-node flow is a small extension: a few
extra command-line flags at server startup, a new panel in the Run
dialog, and the same Cluster Jobs view to monitor what's happening.

**What you'll do:**

1. [Decide whether multi-node fits your workload](#when-to-use-multi-node)
2. [Set up the prerequisites](#prerequisites)
3. [Start the cluster](#starting-the-cluster)
4. [Submit a multi-node job](#submitting-a-multi-node-job)
5. [Monitor and diagnose](#monitoring-and-diagnosing)
6. [Operational notes](#operational-notes)

Companion references for when you need them:

- [Forgather Server reference](../forgather-server.md) — the cluster
  section there has the full feature/API reference; this guide
  extracts the operator-facing parts.
- [Pipeline Parallel](../trainers/pipeline-parallel.md) — when to
  pick PP over DDP/FSDP, how to size stages.
- [Checkpointing](../checkpointing/README.md) — distributed
  checkpoint behaviour, including the shared-FS single-writer rules
  multi-node depends on.

---

## When to use multi-node

Multi-node training across consumer Ethernet (1G or 2.5G) is **only
viable for pipeline parallel (PP)**. The rule of thumb:

| Strategy | Communication pattern | Practical inter-node bandwidth needed |
|---|---|---|
| **Pipeline Parallel (PP)** | activations between adjacent stages, once per microbatch | works on 1G+ Ethernet; the trainer Forgather just tested on |
| **DDP** | full gradient all-reduce every step | needs ~10G+ for non-trivial models; viable on 1G for ~1B-class |
| **FSDP** | parameter all-gather on every layer | needs NVLink or fast InfiniBand; near-unusable on Ethernet |
| **Tensor Parallel** | per-layer all-reduce | needs NVLink |

Forgather's multi-node infra is agnostic to the trainer choice — you
can submit any DDP/FSDP/PP config across the cluster — but on a
typical home-lab LAN, only PP will actually train at a useful rate.
DDP across two boxes on a 1G link will work, but the gradient
all-reduce will dominate and step time will be much longer than the
single-host equivalent.

The reference end-to-end run (Llama2-7B + Samantha + 1F1B PP across a
1+2 GPU layout on a 1Gbit link) achieves all-three-GPUs-at-100% util,
~28% link utilisation, ~40% MFU. That's a healthy regime: compute is
the bottleneck, network has headroom.

---

## Prerequisites

Multi-node operation has a small but firm list of preconditions. Skip
any of these and you'll hit a hard-to-diagnose failure later.

### Shared filesystem

**Every selected node must see identical absolute paths for**:

- The Forgather repo (so `forgather train` finds the same templates,
  scripts, generated model code).
- The project directory (`-p /path/to/project`).
- The dataset (whatever the project's dataset config resolves to —
  typically a HuggingFace cache or a local data root).
- The output directory (where checkpoints land).

The simplest setup is an NFS export mounted at the same path on
every node. Less popular but workable: a synchronised local layout
(rsync, syncthing) where the absolute paths happen to match.

The submit flow assumes shared paths and does not stage anything for
you. There is no automatic config replication or dataset proxy.

### Same Linux user / UID

Files written by one node need to be readable by the others. The
easiest way is to make sure all nodes run Forgather under the same
username + matching UID/GID. The example cluster used `dinalt`
(UID 1001) on every node.

### Matching software versions

Multi-node training is exquisitely sensitive to torch / NCCL version
mismatches. The cluster's pre-flight probe surfaces these as
"version mismatch" warnings in the Nodes view:

- `forgather` — should match exactly.
- `torch` — must match exactly. NCCL version is bundled with the
  torch wheel.
- `transformers` — minor mismatches usually work but produce subtle
  bugs.
- `python` — patch version differences are usually fine.

If you submit with mismatches, the server returns HTTP 409 unless
you explicitly acknowledge with the **"I understand — submit
anyway"** checkbox in the Multi-node panel.

### Trusted LAN

Inter-node API calls inside the cluster carve out auth for known
peers — they're identified by source IP and don't need a bearer
token. The threat model assumes the LAN is trusted (consistent with
torch.distributed's own assumption). If your LAN is not trusted, do
not enable cluster mode.

### Container PID 1

If you run the Forgather server inside a Docker container, **PID 1
must be a proper init process** that reaps orphan grandchildren.
Otherwise, when torchrun gets killed, its worker subprocesses
re-parent to PID 1, never get reaped, and pile up as zombies inside
the container.

The bundled `docker/run.sh` passes `--init` for exactly this
reason — Docker's `tini` becomes PID 1 and reaps orphans regardless
of parentage. If you bring your own container, make sure either you
add `--init` to the `docker run` invocation, or run a tini/dumb-init
process as the entrypoint.

To pick up the `--init` change on an existing forgather-dev
container:

```bash
docker/run.sh --rm
docker/run.sh
```

That recreates the container; existing zombies vanish with the old
PID 1.

---

## Starting the cluster

On every node, start the Forgather server with `--cluster <name>`
and bind to all interfaces so peers can reach the API:

```bash
forgather server -H 0.0.0.0 --cluster lab
```

The cluster name is per-invocation (not persisted). Two unrelated
clusters on the same LAN that use different names won't auto-merge.

Each node mints a stable UUID at first cluster startup (saved at
`~/.forgather/cluster/node_id`, mode 0600). Master selection is
deterministic — the lowest UUID among reachable members wins. No
election round-trip; if the master goes down, the survivors keep
running and a new master is selected within ~30 seconds.

### Verify cluster discovery

Open the webui on any node and click **Nodes** in the sidebar. You
should see one bounded box per cluster member, with:

- Hostname and IP address
- "this node" / "master" tags as appropriate
- GPU summary (one card per GPU)
- Version chips (forgather / torch / nccl / transformers) — yellow
  if a node diverges from the cluster majority
- A collapsible network interfaces panel

If a node is missing, check:

1. Is the server actually running there? `curl
   http://<peer-ip>:8765/api/cluster/self` should return its
   identity.
2. Is mDNS reaching across the LAN? Some VPN configs and managed
   switches block multicast.
3. Did both servers start with the same `--cluster` name?

### Container address advertisement

Inside a container without `--network host`, the server's psutil
view of network interfaces only sees the container's namespace —
not the host's real LAN interfaces. The auto-detector falls back to
`127.0.0.1` and emits a WARNING; peers on other hosts cannot reach
that address.

Tell the server which routable address to advertise:

```bash
forgather server -H 0.0.0.0 --cluster lab \
    --cluster-address 192.168.1.27
```

`--cluster-address` is repeatable if your node has multiple routable
addresses you want advertised.

You can also use `--network host` on the docker run command (the
default in `docker/run.sh`) so the container shares the host's
network stack and address advertisement Just Works.

### Optional: bandwidth probe

The Nodes view has a collapsible **Bandwidth** panel above the node
list with a **Refresh** button. It runs a single-stream HTTP
throughput measurement from the local node to every reachable peer,
sequentially. Results are cached for 1 hour.

This is a sanity check — if the measured bandwidth between two
hosts is much lower than your link's nominal speed, you have a
network problem to fix (wrong interface routing, broken cable,
duplex mismatch) before submitting any training jobs.

---

## Submitting a multi-node job

There is **no separate "Multi-node" submit button**. Multi-node
submits go through the same dialog as single-node submits — the
Run… action on any training config.

### Open the Run dialog

In the project tree on the left, navigate to a project that contains
a training config (e.g. `examples/finetune/samantha`). Click on a
config (e.g. `llama2_7b/2gpu_pp.yaml`) to open it in the right
pane, then click **▶ Run** at the top of the config viewer. Or
right-click the config in the tree and pick **▶ Run…**.

When the server is in cluster mode, the Run dialog shows two
collapsible sections in addition to the usual GPUs / Priority /
Dynamic args:

- **Multi-node** — present only when `--cluster` is active.
- **Dynamic arguments** — same as in single-node mode.

Both default to open. Each is capped at 50% of the dialog body when
both are open, so neither pushes the other off-screen.

### Pick participants

The Multi-node panel lists every cluster member as a row with five
columns:

| Column | Purpose |
|---|---|
| **Use** | checkbox — opt this node into the run |
| **Node** | hostname + address, with "this node" / "master" / "unreachable" tags |
| **GPUs** | how many GPUs this node contributes (= per-node `nproc_per_node`); bounded by the node's hardware count, with `(N idle of M)` hint |
| **NCCL iface** | dropdown of the node's up-and-running interfaces, defaulting to `(auto)` — see below |
| **rdzv host** | radio — which member hosts the rendezvous TCPStore |

By default, the local node is checked and every other node is
unchecked. Adding peers turns the submit into a fanout. The Submit
button label changes to **"Submit to N nodes"** so you know which
path you're taking.

If you leave only the local node checked, Submit goes through the
regular single-node enqueue path (no rendezvous overhead). If you
deselect every node, Submit is blocked.

### Pick the rendezvous host

The default is the cluster master. For a typical 2-node setup, you
can leave this alone. For larger clusters, prefer the node with the
lowest cross-node latency — the rdzv host is contacted by every
other node during initialization.

### Pick the network interface (or leave it on auto)

The **NCCL iface** dropdown shows interfaces from each node's
pre-flight probe. Pick the one you want NCCL/Gloo/tensorpipe to
use for cross-node traffic.

Leaving it on `(auto)` is the right default in most cases: the
server matches each member's cluster-advertised IP against its probe
interface table and picks the matching iface name automatically. You
only need to override this if:

- You have multiple routable interfaces on a node (e.g. a slow
  management interface and a fast training interface) and the
  cluster picked the wrong one for advertisement.
- You're testing with VLAN-tagged or specially-routed traffic.

If no interface can be derived (probe data missing, address
mismatch), the submit returns HTTP 422 rather than spawning a job
that would deadlock in `connectFullMesh`.

### Pick the rendezvous port

The **Rendezvous port** field at the top of the panel defaults to
`29400`. Change it if 29400 is in use (or if you're running multiple
multi-node submits side-by-side, which would clash on the same
port). Any free TCP port on the rdzv host works.

### Asymmetric topologies are fine

You can mix nodes with different GPU counts. The example reference
cluster ran 1 GPU on one node + 2 GPUs on another, world_size=3,
1F1B PP. The trainer's local-process-group machinery is
hostname-discovered, so heterogeneous layouts produce correct
groups.

The rule for picking per-node GPUs depends on the trainer:

- **PP**: pick the GPUs that maximize compute and minimize cross-node
  hops. World size = total GPUs across all nodes; stages = world
  size × `stages_per_rank` (default 1). Each rank owns one or more
  stages. Asymmetric layouts mean some nodes own more stages than
  others — the splitter's auto-balancer handles this fine for typical
  decoder-only LMs.
- **DDP**: pick the same number of GPUs per node when possible. Asymmetric
  DDP works but the small-rank-count node will be the straggler at
  every allreduce.

### Submit

Click **Submit to N nodes**. The master:

1. Validates participants are reachable and version-compatible.
2. Generates a rendezvous ID and computes the rdzv endpoint.
3. Fans out a per-rank training enqueue to each peer with their
   torchrun args (`--nnodes`, `--node-rank`, `--rdzv-backend=c10d`,
   etc.) and the right `NCCL_SOCKET_IFNAME` /
   `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` environment.
4. Records a ClusterJob bundle linking the per-rank queue ids back
   to a single `cluster_job_id`.

If a fanout step fails partway through, the master rolls back by
issuing cancels to the participants it already enqueued on. There's
no half-submitted state.

Last-used Multi-node settings (participants, per-node GPUs, iface,
rdzv host/port, mismatch acknowledgement) persist alongside the
config's dynamic-args overrides — so a config "opens where you left
off" for both single- and multi-node submits. **Reset to defaults**
in the dialog clears everything we cached for this config.

---

## Monitoring and diagnosing

### Cluster Jobs panel

Under **Nodes** in the sidebar, the **Cluster Jobs** panel lists
running and recently-finished bundles. Each row shows:

- Truncated bundle id
- Project / config
- Rendezvous endpoint + id
- Per-rank assignments (rank → hostname × GPUs, with current status)
- Rolled-up status — `running` / `done` / `failed` / `cancelled`
- Cancel button (active only while running)

The rolled-up status is computed from per-rank queue item statuses
across the cluster; partial completion shows up as `running`, and
`done` requires every rank to be terminal. Slow/unreachable peers
contribute `unknown` for their rank without blocking the list.

### Per-node TTY logs

To watch a specific rank's torchrun output, open that node's webui
and find its queue item in the Jobs panel. Each peer's webui shows
its own ranks; there is no cross-node TTY aggregation in v1.

The local TTY view supports live tail and full dump. WebSocket
streaming follows the file.

### Hang diagnosis with `kill -USR1`

Multi-node training hangs are common while you're getting a new
config working. The two failure modes that produce no Python
traceback are:

- **Silent rank death** — a worker exits from a CUDA driver
  assertion, kernel OOM-kill, or unhandled C++ exception in a
  background thread.
- **Distributed deadlock** — every rank is blocked in a `dist.*`
  collective, waiting for partners that aren't coming.

Forgather's `train_script.py` enables Python's `faulthandler` so
both cases produce useful diagnostic output:

- Crashes (SIGSEGV / SIGFPE / SIGABRT / SIGBUS / SIGILL) dump every
  thread's Python stack to stderr → per-rank TTY log → visible in
  the Jobs view.
- `kill -USR1 <pid>` against any rank's worker python dumps that
  rank's full thread stacks to its TTY without killing the process.
  Same idiom as `py-spy dump`, but works inside containers that
  strip `CAP_SYS_PTRACE`.

To find the worker PIDs: in the Jobs panel, click on the rank's
entry and the PID is shown in the metadata. Or from a shell on the
node:

```bash
ps -ef | grep "train_script.py" | grep -v grep
```

The `torchrun` parent and one or more worker python processes will
be listed. SIGUSR1 the workers, not torchrun.

After SIGUSR1, look at the rank's TTY log (`tail -200
~/.forgather/server/jobs/q_*.tty`). Each thread's stack ends with
the most recent frame, so you can read the deadlock site directly.

Common deadlock sites and what they mean:

| Stack tip | What's stuck | Common cause |
|---|---|---|
| `dist.new_group` in `_new_process_group_helper` | one rank is creating a process group that other ranks aren't | mismatched local_world_size assumptions; topology bug |
| `dist.all_gather` / `all_gather_object` | one rank hasn't reached the call yet | uneven progress between ranks |
| `dist.barrier` | one rank failed earlier; others now wait forever | scroll up in TTYs to find the silent failure |
| `_PipelineStage._receive` | upstream stage hasn't sent the activation | upstream rank crashed or is in a different scheduler op |
| NCCL `ncclSendRecv` | inter-stage activation/gradient transfer | NCCL_SOCKET_IFNAME wrong on one side; firewall |

### Topology log

When the cluster forms its first node-local process group, rank 0
logs the discovered topology:

```
[Rank 0] forgather.ml.distributed - INFO - get_local_process_group:
  discovered topology {'muthur': [0], 'wopr': [1, 2]}
```

This tells you, at a glance, which ranks live on which hosts. If
you ever see ranks distributed unexpectedly, the topology log is
the first place to look.

### Per-rank `host=` tag

Every rank's `DistributedEnvironment(...)` log line includes a
`host=<hostname>` field. So when SIGUSR1 dumps a stack, you can
correlate "rank N is hung at <site>" with "rank N is on
<hostname>" without cross-referencing cluster_jobs.

---

## Operational notes

### Cancelling a job

The Cluster Jobs panel's **Cancel** button fans out a cancel to
every peer of the bundle. Each peer's local scheduler then aborts
its rank's queue item. You can issue this from any node — non-master
nodes proxy the request to master.

If `Cancel` doesn't take effect within a few seconds (e.g. one rank
is stuck in an uninterruptible kernel call), you can use the
per-node Jobs panel's right-click context menu to **Force kill
(SIGKILL)** that specific rank. Backend polls for the PID to
actually exit (up to 2 seconds) and stamps the JobRecord's `error`
field if it's still alive afterwards — so a stuck-in-CUDA process
surfaces in the UI instead of silently leaving a phantom GPU
consumer.

### Cleaning up stale endpoints

If the Forgather server itself was restarted while a torchrun was
running, the in-flight trainer's endpoint files (under
`~/.forgather/jobs/job_*/`) can outlive their process. They surface
as endpoint-only entries in the Jobs panel — entries with no
`source: record` or `merged`, just `endpoint`.

By default these are filtered out of the Jobs list when their PID
is dead/zombie/recycled. To see them, toggle **"Include dead
endpoints"** on the Jobs panel header. Then right-click → **✕
Remove stale endpoint** to rmtree the directory and stop the entry
from showing up.

### Trainer control endpoint is local-only

`forgather control list` and the trainer's `save` / `stop` / `abort`
HTTP endpoint are bound to `127.0.0.1` on whichever node hosts rank
0. So:

- `forgather control list` shows the run on the rank-0 host only
  (e.g. muthur in the reference cluster); empty on other nodes.
- Save/stop/abort commands targeting rank 0 must be issued from a
  shell or webui on the rank-0 host.
- The Cluster Jobs panel's Cancel button still works from anywhere
  — it goes through the JobRecord-level cancel-fanout, not the
  per-trainer control endpoint.

This is a known v1 limitation. For a save/stop on the running run,
ssh to the node hosting rank 0 (look at the topology log to find
which one) and run the command there.

### MFU and heterogeneous clusters

The reported MFU number is correct for **homogeneous clusters**
(every rank's GPU is the same model). It's computed as
`achieved_aggregate_flops / (world_size × per_GPU_peak)`, where
per_GPU_peak is auto-detected from rank 0's device.

If your cluster is heterogeneous (e.g. mixed RTX 3090 + RTX 4090,
or pairing a Spark with a desktop GPU), the aggregate peak is
wrong and the MFU number is meaningless. Workaround: set
`peak_hardware_flops` explicitly per-config based on the
slowest-tier device, or stick to homogeneous training clusters
until probe-driven aggregation lands.

### Checkpoints on shared FS

When several ranks share a filesystem (NFS, the typical multi-node
setup), only one rank globally writes the model shard files. The
CheckpointManager honours `save_on_each_node=False` (the documented
default for shared storage) by gating the shard-file save on a
single-writer rank, so concurrent writers can't race on the same
shard paths.

For pipeline-parallel training (`save_on_all_ranks=True` in the
trainer), every rank writes its own non-overlapping shard files
based on which FQNs (parameter names) its stage owns — different
stages own disjoint FQN sets so there's no collision.

This means a multi-node PP run on shared NFS produces a complete,
correct checkpoint that any other Forgather process (e.g. the
inference server) can load as a normal model. The reference run
verified this end-to-end: Llama2-7B+Samantha trained 3-stage
across 1+2 GPUs, saved to NFS, loaded for inference, generated
coherent output.

---

## Known limitations and caveats

- **No global scheduler.** Per-peer scheduling decisions are
  independent. Cluster job submits use a static fanout at submit
  time; there is no live re-balancing or cross-node preemption.
  If you submit a multi-node job to a cluster where one peer is
  busy with another single-node job, the multi-node job will wait
  for that peer's GPUs to free up.
- **No automatic config staging.** Project paths must resolve to
  the same files on every participant.
- **No cross-node TTY aggregation.** Open each peer's webui to
  watch its rank's torchrun output. Cross-node Jobs / Files / TTY
  proxying through the master is on the roadmap (Phase 3.5) but
  not in v1.
- **No automatic master failover.** The master is whichever
  reachable member has the lowest UUID; if it goes down the
  cluster keeps running with a new master, but in-flight global
  state (queue mutations during the gap) is lost. Master failover
  with state replication is on the roadmap (Phase 4 + Phase 5).
- **No cross-architecture training.** The version probe surfaces a
  platform mismatch (e.g. ARM Spark + x86_64 desktop) but torch
  wheels and CUDA kernels won't actually interoperate across
  architectures. The check is advisory; the operator is on the
  hook for whether their cluster makes sense.

---

## Where to learn more

- **[Forgather Server reference](../forgather-server.md)** —
  full feature/API reference. The Cluster mode section has the
  authoritative threat model, mDNS discovery details, and API
  table.
- **[Pipeline Parallel](../trainers/pipeline-parallel.md)** —
  scheduling choices (1F1B, ZBV, GPipe, interleaved), stage
  layout, and the splitter API.
- **[Checkpointing](../checkpointing/README.md)** — distributed
  checkpoint behaviour, resume, sharing patterns.
- **[Training Performance Metrics](../trainers/training-performance-metrics.md)**
  — how MFU, tok/s, and FLOPs are computed.
- **[Forgather Server CLI](server-cli.md)** — `--enqueue`,
  `forgather sched`, `forgather job`, `forgather gpu` from the
  terminal.
