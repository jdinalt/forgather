# DiLoCo: Distributed Local-SGD Training

DiLoCo (Distributed Local SGD with Communication) enables distributed training
across multiple heterogeneous machines on a standard LAN. Unlike DDP, which
requires high-bandwidth interconnects (NVLink, InfiniBand), DiLoCo reduces
communication by ~500x, making 1 Gig Ethernet practical for multi-machine
training.

## How It Works

Each machine runs any existing Forgather trainer (single GPU, DDP, or pipeline)
as an independent "worker." Workers train locally for H steps using their inner
optimizer (e.g., AdamW), then synchronize with a central parameter server. The
server averages the workers' updates and applies an outer optimizer (SGD with
Nesterov momentum) to produce new global parameters that all workers adopt.

```
                    +-------------------+
                    |   DiLoCo Server   |
                    | (standalone proc) |
                    |                   |
                    | - Global params   |
                    | - Outer optimizer |
                    | - Worker registry |
                    +--------+----------+
                             |
                 HTTP over 1G Ethernet
                             |
         +-------------------+-------------------+
         |                   |                   |
   +-----+-----+      +-----+-----+      +-----+-----+
   |  Worker 0  |      |  Worker 1  |      |  Worker 2  |
   | (Machine A)|      | (Machine B)|      | (Machine C)|
   |            |      |            |      |            |
   | Pipeline   |      | Single GPU |      | DDP        |
   | Trainer    |      | Trainer    |      | Trainer    |
   | (4x 3090)  |      | (1x 4090)  |      | (2x A6000) |
   +------------+      +------------+      +------------+
```

### Sync Protocol

Each synchronization round follows these steps:

1. Workers train locally for `sync_every` optimizer steps (the "inner loop")
2. Each worker computes pseudo-gradients: `global_params - local_params`
3. Workers submit pseudo-gradients to the server over HTTP
4. Server waits until all workers have submitted (synchronous barrier)
5. Server averages the pseudo-gradients across all workers
6. Server applies the outer optimizer step (SGD with Nesterov momentum)
7. Updated global parameters are returned to all workers
8. Workers load the new parameters and begin the next inner loop

### Bandwidth Efficiency

Pseudo-gradients are optionally cast to bfloat16 before transmission, halving
bandwidth with minimal quality impact. With `sync_every=500`, a 1B parameter
model transfers ~2 GB every 500 training steps, achieving >97% compute
utilization on 1 Gig Ethernet.

| Model Size | BF16 Size | Transfer Time (1 Gbps) | H=500 steps @ 1s/step | Utilization |
|------------|-----------|------------------------|----------------------|-------------|
| 150M       | 300 MB    | 2.4s                   | 500s compute         | 99.5%       |
| 1B         | 2 GB      | 16s                    | 500s compute         | 97%         |
| 7B         | 14 GB     | 112s                   | 500s compute         | 82%         |

## Quick Start

### 1. Start the Server

The server is a standalone process that holds global model parameters. Start it
on any reachable machine (it does not need a GPU):

```bash
forgather diloco server \
    -m path/to/model \
    -n 2 \
    --port 8512
```

Arguments:
- `-m`: Path to a model directory (loaded via `AutoModelForCausalLM.from_pretrained`)
- `-n`: Number of workers expected (the barrier waits for all N)
- `--port`: Server port (default: 8512)

For Forgather checkpoint format, add `-c`:

```bash
forgather diloco server -c -m output_models/my_model/checkpoint-1000 -n 2
```

### 2. Start Workers

On each machine, launch a worker that wraps the normal training command:

```bash
# Machine A
forgather diloco worker \
    --server 192.168.1.100:8512 \
    --sync-every 500 \
    -p my_project -t train.yaml \
    train

# Machine B
forgather diloco worker \
    --server 192.168.1.100:8512 \
    --sync-every 500 \
    -d 0 \
    -p my_project -t train.yaml \
    train
```

Worker-specific arguments:
- `--server`: Server address as `host:port`
- `--sync-every`: Local steps between syncs (default: 500)
- `--worker-id`: Optional unique ID (auto-generated if omitted)
- `--no-bf16`: Send full-precision pseudo-gradients instead of bfloat16
- `-d`: CUDA visible devices

### 3. Monitor

```bash
forgather diloco status --server 192.168.1.100:8512
```

Shows sync round, registered workers, their hostnames, training speeds, and
pending sync submissions.

## Programmatic API

The DiLoCo system can also be used directly in Python, independent of the CLI.

### DiLoCoWorker

The worker is a composable wrapper that hooks into any optimizer via
`register_step_post_hook`. It works as a context manager:

```python
import torch
from forgather.ml.diloco import DiLoCoWorker

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

with DiLoCoWorker(
    model=model,
    optimizer=optimizer,
    server_addr="192.168.1.100:8512",
    sync_every=500,
    bf16_comm=True,
) as diloco:
    # Train normally - DiLoCo syncs happen automatically every 500 optimizer steps
    for batch in dataloader:
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Access sync metrics
    print(diloco.sync_metrics)
```

Key parameters:
- `model`: The model being trained
- `optimizer`: The inner optimizer (any `torch.optim.Optimizer`)
- `server_addr`: Server address as `"host:port"`
- `sync_every`: Steps between syncs (H in the DiLoCo paper)
- `bf16_comm`: Cast pseudo-gradients to bfloat16 (default: True)
- `worker_id`: Unique ID (auto-generated if None)

### DiLoCoServer

```python
from forgather.ml.diloco import DiLoCoServer

# Load initial model state
model = AutoModelForCausalLM.from_pretrained("my_model")
state_dict = model.state_dict()

# Start server (blocking)
server = DiLoCoServer(
    model_state_dict=state_dict,
    num_workers=3,
    port=8512,
    outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.7, momentum=0.9, nesterov=True),
)
server.run()

# Or start in background
server.start()
# ... do other things ...
server.stop()
```

### DiLoCoClient

Low-level client for direct server communication:

```python
from forgather.ml.diloco import DiLoCoClient

client = DiLoCoClient("192.168.1.100:8512")

# Register and get initial params
params = client.register("my_worker", {"hostname": "machine-a"})

# Submit pseudo-gradients (blocks until all workers sync)
new_params = client.submit_pseudogradients("my_worker", pseudograds)

# Other operations
status = client.get_status()
client.heartbeat("my_worker", steps_per_second=3.5)
client.deregister("my_worker")
```

## Server Configuration

### Outer Optimizer

The default outer optimizer is SGD with Nesterov momentum (lr=0.7, momentum=0.9),
following the DiLoCo paper. You can customize it via CLI flags or the factory
function:

```bash
# CLI
forgather diloco server -m model -n 2 --outer-lr 0.5 --outer-momentum 0.95

# Or disable Nesterov
forgather diloco server -m model -n 2 --no-nesterov
```

Any `torch.optim.Optimizer` can be used as the outer optimizer via the
programmatic API. The server wraps global parameters as `nn.Parameter` objects,
so standard optimizers work directly.

### Server State Persistence

The server can periodically save its state (global params + outer optimizer
state) for crash recovery:

```bash
forgather diloco server -m model -n 2 --save-dir /path/to/saves --save-every 10
```

To resume a server from saved state:

```bash
forgather diloco server -m model -n 2 --resume /path/to/saves/diloco_server_state_latest.pt
```

## How Pseudo-Gradients Work

The pseudo-gradient computation follows the TorchFt approach:

1. When a worker registers or completes a sync round, it saves a CPU snapshot
   of the model parameters (`_save_global_params_snapshot`)
2. The worker trains normally on GPU for `sync_every` steps
3. At sync time, the worker computes: `pseudo_grad = snapshot_cpu - model_params.cpu()`
4. The pseudo-gradient is optionally cast to bfloat16 and sent to the server
5. The server averages pseudo-gradients from all workers and applies the outer
   optimizer: `global_params -= lr * avg_pseudo_grad` (with momentum)

This design keeps the CPU snapshot in host memory without interfering with GPU
training, and the delta computation is done on CPU to avoid disrupting the
training computation graph.

## HTTP API Reference

The server exposes these HTTP endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register` | Worker registration; returns global params |
| POST | `/submit_pseudograd` | Submit pseudo-gradients; blocks until all workers sync; returns updated params |
| GET | `/global_params` | Fetch current global parameters |
| POST | `/heartbeat` | Worker heartbeat with training speed |
| POST | `/deregister` | Worker departure |
| GET | `/status` | Server status (workers, sync round, etc.) |

Tensor data is serialized using `torch.save` to `BytesIO` and sent as
`application/octet-stream`. The pseudo-gradient submission uses a
length-prefixed JSON header followed by the tensor payload.

## References

- Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models" (2024)
- Douillard et al., "DiPaCo: Distributed Path Composition" (2024)
- TorchFt (Meta) - fault-tolerant distributed training library
