# DiLoCo Distributed Training Example

This project demonstrates DiLoCo (Distributed Local-SGD) integration with the
Forgather trainer via `DiLoCoCallback`.

The instructions were written for testing on a single node with at least 2 GPUs, but 
it should be possible to adapt it for multinode training.

While we do make use of the forgather server, this set of instructions primarily covers
how to work with DiLoCo from the CLI. A separate tutorial for the webui is in-the-works.

All commands below assume you are in the project directory:

```bash
cd examples/tiny_experiments/diloco
```

### 1. Construct the Model (First Time Only)

The DiLoCo server needs a model with saved weights. Build and save weights using
a model project (not this training project):

```bash

# Create a freshly initialized model instance
forgather -p ../../models/llama -t small.yaml \
    model --device cpu --save-checkpoint --safetensors \
    --output-dir ../../../models/small_llama \
    construct
```

### 2. Start the Forgather Server

You can skip this step, if you already have configured and launched the Forgather server.

The DiLoCo server can optionally use the Forgather server for coordination. Without doing so,
you will have to do the coordination via the command line for things like discovery server discovery
and security settings.

We will want to use "cluster" mode, even if running on a single node, as this enables things like
auto-discovery of dataset servers. And you will need cluster mode when running on multiple nodes.

If you will be running on multiple nodes, it is strongly recommended that you complete the TLS setup
procedures first. TODO: add link to relevant instructions.

In a secondary terminal session...

```bash
# Note that you can add `cluster: demo` to the startup arguments in the server config instead.
forgather server --cluster demo
```

### 4. Start the dataset server

If you have already configured and started the dataset server, you can skip this step.

```bash
# Start the dataset server in the background on local-host.
forgather dataset-server start

# Start with the server bound to all interfaces
forgather dataset-server start -H 0.0.0.0
```

You can check the status of the server with:

```bash
forgather dataset-server status
service: forgather-dataset-server  version: 1.0.0
status:  ok
policy:
  auth_required:    True
  hf_cache_enabled: True
  allow_paths:      False
  allow_downloads:  False
  local_count:      0
```

### 4. Download, Build, and Index the datasets (if not already running and configured)

You will need to have any datasets you intend already cached on at least one cluster member.
If you have already loaded these via another project, they are already cached and you can skip this step.

You can check if the required datasets are in the cache like this:

```bash
forgather dataset-server cache
...
- HuggingFaceTB/smollm-corpus  (1.1 TB)
    fineweb-edu-dedup @ 0.0.0  -- train=190,168,005
...
- roneneldan/tiny_stories  (50.8 GB)
    default @ 0.0.0  -- train=2,119,719, validation=21,990
```

It's possible to have the dataset server download dataset automatically via `--allow-downloads`,
but with large datasets, I prefer doing to explicitly:

```python
from forgather.ml.datasets import fast_load_iterable_dataset

# Load, cache, and index Tiny Stories -- a few GBs
dataset = fast_load_iterable_dataset("roneneldan/TinyStories", revision="f54c09f", split="train")

# Load Fineweb EDU -- a bit short of 1 TB!
dataset = fast_load_iterable_dataset("HuggingFaceTB/smollm-corpus", name="fineweb-edu-dedup", split="train")
```

Alternatively, you can load and indirectly index them from thier project definitions.

```bash
# If you have already downloaded and indexed these datasets, these commands are very fast.
forgather -p ../../datasets/roneneldan/ -t fast-iter-packed.yaml dataset --target train_dataset_split
forgather -p ../../datasets/HuggingFaceTB -t smollm-corpus/fineweb-edu-packed.yaml dataset --target train_dataset_split
```

### 5. Start the DiLoCo Server

On any machine in the cluster (GPU not required):

```bash
forgather diloco server --output-dir ../../../models/small_llama --num-workers 2 -H 0.0.0.0
```

This configures the server to use the model we constructed in step #1, which will be automatically distributed to the workers.

We explicitly set the number of `--expected-workers` to 2. This is actually just a hint.

TODO: Elaborate more on what exactly `--expected-workers` impacts.

As we have security enabled, we have bound to all interfaces via `--host 0.0.0.0`. Leave out this argument to only bind to `localhost`.

#### Identify all DiLoCo servers in the cluster

```bash
forgather diloco servers
ID                         SOURCE      STATE              BASE_URL
------------------------------------------------------------------------------------------
local:q_1780289993506_f82969f3 local       alive              https://192.168.9.43:8512
```

#### Check the server status

```bash
forgather diloco status
DiLoCo Server Status
==================================================
  Status:        running
  Mode:          sync
  Sync round:    0
  Workers:       0/2
  Uptime:        0h 0m
  Parameters:    34,417,152 (131.3 MB)
  Outer opt:     SGD(lr=0.7, momentum=0.9)
  Save dir:      /mnt/rust/home/dinalt/rust/forgather/models/small_llama
  HB timeout:    120.0s (min workers: 1)
```

#### Check server logs

This command takes a job-id. Use the id found from `forgather diloco servers`

The `--follow` argument tails the logs. Without it, it just dumps them.

```bash
forgather diloco logs local:q_1780289993506_f82969f3 --follow
```

### 6. Start Workers

If running on a single node, you can quickly start N identical workers, with automatically assigned names, like this:

```bash
# We have disabled torch compile for faster startup, but for a real job, you probably want it enabled.
forgather diloco worker --count 2 --compile no
```

### 7. Monitor

#### Server Status 

You can watch the training overview like this:

```bash
forgather diloco status --queues --watch
forgather diloco status — https://192.168.9.43:8512 — 05:54:36 (every 2s, Ctrl-C to stop)

DiLoCo Server Status
==================================================
  Status:        running
  Mode:          sync
  Sync round:    2
  Workers:       2/2
  Uptime:        0h 4m
  Parameters:    34,417,152 (131.3 MB)
  Outer opt:     SGD(lr=0.7, momentum=0.9)
  Save dir:      /mnt/rust/home/dinalt/rust/forgather/models/small_llama
  HB timeout:    120.0s (min workers: 1)

Training stats (aggregate of 2 reporting):
  Total tokens:  64,396,308
  Total steps:   1,984
  Total FLOPs:   1.013e+16
  Throughput:    240,333 tok/s
  MFU:           16.9%
  Peak memory:   20.31 GB
  Grad norm:     0.582
  Train loss:    5.5406
  Eval loss:     5.4825 (@ step 963)

Workers (registered):
  ID                             Host            Round    Steps/s    Last HB
  ---------------------------------------------------------------------------
  glacial-chihuahua              hal9000         2        4.98       05:54:28
  brown-koa                      hal9000         2        4.91       05:54:29

Known workers: 2 (2 running)

Work-unit dispatch:
  HuggingFaceTB/smollm-corpus:fineweb-edu-dedup@train@0: 2/1024 issued (0% issued) — 190,158,005 rows
    dataset_id: 357183ce6248a323
    worker                           issued  completed
    brown-koa                             1          0
    glacial-chihuahua                     1          0
```

#### Watch worker logs

```bash
forgather diloco logs glacial-chihuahua --follow
...
INFO:forgather.ml.diloco.worker:DiLoCoWorker glacial-chihuahua: starting sync (round 3, after 500 local steps)
INFO:forgather.ml.diloco.worker:DiLoCoWorker glacial-chihuahua: sync round 3 complete. Sent 68.8 MB, received 137.7 MB, took 3.3s
2026-06-01 05:56:11      1,504   0.0001928   4.79387    0.6833    1.94e-04   1,038,760       48.8M     114,135   11.1%                         9.456 GiB
2026-06-01 05:56:17      1,536   0.0001973   4.76256    0.5926    1.98e-04   1,038,131       49.9M     170,232   16.6%                         9.456 GiB
2026-06-01 05:56:23      1,568   0.0002017   4.71522    0.4707    2.02e-04   1,034,890       50.9M     172,909   16.9%                         9.456 GiB
2026-06-01 05:56:29      1,600   0.0002061   4.63772    0.5049    2.06e-04   1,039,403       51.9M     168,540   16.5%                         9.456 GiB
2026-06-01 05:56:33      1,605  0.0    eval-loss: 4.66392
```

#### Monitor with TensorBoard

```
# Monitor all of the workers
tensorboard --bind_all --logdir output_models/ --port 6006 &

# Monitor the aggregate for the server
tensorboard --bind_all --logdir ../../../models/small_llama/ --port 6007 &
```
### 8. Stopping

To cleanly shutdown:

```bash
forgather diloco shutdown
Save & stop queued for 2 worker(s): glacial-chihuahua, brown-koa
Waiting up to 600s for workers to stop…
  stopped: brown-koa
  stopped: glacial-chihuahua
  2/2 stopped
All workers stopped.
Saving server checkpoint…
  server checkpoint saved.
Stopping server…
Done.
```

### 9. Resume Training

```bash
# Restart the server. It will resume from the latest checkpoint, remembering the config details.
forgather diloco server --output-dir ../../../models/small_llama --num-workers 2 -H 0.0.0.0

# Restart the workers. These are auto-resumed from the server's checkpoint, which remembers the names of the servers.
# Note that by removing `--compile no`, the workers will be restarted with Torch compile enabled.
forgather diloco worker --resume-workers
```

### 10. Cleanup

```bash
# Delete all worker logs and checkpoints
rm -rf output_models/

# Delete server checkpoints
rm -rf ../../../models/small_llama/checkpoints/

# Delete server logs
rm -rf ../../../models/small_llama/runs/
```
