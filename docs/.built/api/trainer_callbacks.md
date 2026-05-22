# Trainer Callbacks

Callbacks extend trainer behaviour at well-defined lifecycle events (step start/end,
epoch start/end, evaluation, checkpoint save/load, etc.) without modifying trainer
source. Pass a list of callback instances to any trainer's `callbacks` argument.

**Related documentation:**

- [Trainer Control](../trainers/trainer-control.md) — saving, stopping, and aborting running jobs
- [Checkpointing](../checkpointing/README.md) — stateful callbacks and checkpoint resume
- [Divergence Detection](../checkpointing/divergence_detection.md) — detecting and recovering from training instability
- [Log Analysis](../guides/logs-analysis.md) — working with logs produced by `JsonLogger`

---

## Base Class

### `TrainerCallback` {#forgather-ml-trainer-trainer_types-trainercallback}

`forgather.ml.trainer.trainer_types.TrainerCallback`

Base class for trainer event callbacks.

Subclasses implement only the event methods they need. Any method not
defined is simply never called for that callback. The trainer maintains
a lazy index mapping event names to the callbacks that define handlers,
so only relevant callbacks are invoked per event.

Available events (each receives args, state, control, **kwargs and
may return None or an updated TrainerControl):

    on_init_end          - After trainer initialization
    on_train_begin       - Before training loop starts
    on_train_end         - After training loop ends
    on_epoch_begin       - Before each epoch
    on_epoch_end         - After each epoch
    on_step_begin        - Before each training step
    on_step_end          - After each training step
    on_substep_end       - After each gradient-accumulation sub-step
    on_forward_backward_begin - Before each forward+backward micro-step
                           (inside gradient accumulation loop, after data
                           loading; fires once per micro-batch)
    on_forward_backward_end   - After each forward+backward micro-step
                           (before optimizer, grad clipping, LR scheduler;
                           fires once per micro-batch)
    on_optimizer_step    - After optimizer.step()
    on_pre_optimizer_step - Before optimizer.step()
    on_evaluate          - After evaluation
    on_predict           - After prediction (also receives metrics)
    on_prediction_step   - After each prediction batch
    on_save              - After checkpoint save
    on_log               - After metric logging (receives logs kwarg)

Forgather extensions (not in HF Trainer):

    on_log_step          - Called before on_log; receives the mutable logs dict
                           so callbacks can inject custom metrics before logging
    on_train_metrics     - Called each training step with per-step metrics
                           (loss, grad_norm, tokens, etc.) for fine-grained
                           monitoring or adaptive control

kwargs always include:
    model, processing_class, optimizer, lr_scheduler,
    train_dataloader, eval_dataloader, trainer

Compatible with HuggingFace TrainerCallback for easier porting.
See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

Identification:
    Each callback has a `name` property used for logging (e.g. when
    the trainer reports which callback requested an early stop).
    Subclasses inherit the default, which returns the class name, or
    can override `name` to provide a more descriptive label (useful
    when multiple instances of the same class are registered with
    different configurations).

**Attributes**

- `name` (str) — Human-readable identifier for this callback, used in log messages.

---

## Built-in Callbacks

Default callbacks included automatically by all trainers.

### `DefaultMetrics` {#forgather-ml-trainer-callbacks-default_callbacks-defaultmetrics}

`forgather.ml.trainer.callbacks.default_callbacks.DefaultMetrics`

Compute derived performance metrics and inject them into logs.

Runs via ``on_log_step`` (before ``on_log``), so computed values are
available to all downstream loggers (ProgressCallback, TBLogger, etc.).

Computed metrics:
    tok_per_sec   -- tokens processed per wall-clock second between log steps.
    mfu           -- Model FLOPs Utilization (requires *peak_hardware_flops*).
    peak_mem      -- per-rank peak CUDA memory allocated (list of bytes),
                     aliased from ``peak_mem_allocated`` for display
                     formatting (default reduction: max across ranks).

---

### `ProgressCallback` {#forgather-ml-trainer-callbacks-default_callbacks-progresscallback}

`forgather.ml.trainer.callbacks.default_callbacks.ProgressCallback`

A TQDM progress-bar callback class based upon:
https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

Controls which metrics are displayed in console logs during training
via configurable column specifications.  All metrics are still logged
to JsonLogger regardless of display settings.

Derived performance metrics (tok/s, MFU, peak_mem) are computed by
``DefaultMetrics`` via ``on_log_step`` and are available in the logs
dict by the time ``on_log`` fires.

---

### `InfoCallback` {#forgather-ml-trainer-callbacks-default_callbacks-infocallback}

`forgather.ml.trainer.callbacks.default_callbacks.InfoCallback`

_No documentation._

---

## Job Control

### `TrainerControlCallback` {#forgather-ml-trainer-callbacks-control_callback-trainercontrolcallback}

`forgather.ml.trainer.callbacks.control_callback.TrainerControlCallback`

Callback that enables external control of training jobs via HTTP API.

Features:
- Graceful stop: Stop training cleanly after current step
- Save checkpoint: Trigger checkpoint save (with evaluation if needed)
- Save and stop: Save checkpoint then stop training
- Status queries: Get current training status

Only rank 0 runs the HTTP server. Commands are broadcast to all ranks
via torch.distributed for coordination.

---

## Divergence Detection

### `DivergenceDetector` {#forgather-ml-trainer-callbacks-divergence_detector-divergencedetector}

`forgather.ml.trainer.callbacks.divergence_detector.DivergenceDetector`

Detects training divergence by comparing smoothed loss against its best observed value.

Maintains a smoothed loss (EMA) and tracks its running minimum. Triggers when
the smoothed loss exceeds the baseline minimum by a configurable threshold,
sustained for ``patience`` consecutive observations.

Supports absolute threshold (smoothed - best >= threshold), relative threshold
(smoothed >= best * factor), or both simultaneously (triggers on whichever
fires first).

Also detects NaN/Inf loss values immediately (no patience required).

Defaults are calibrated against real training runs where loss decreases from
~10 to ~3.8 then spikes to ~9.7 on divergence. With default settings
(smoothing=0.3, threshold=1.0, patience=3), divergence is detected within
3 log entries (~96 training steps at 32-step log intervals) of the spike,
with zero false positives on healthy runs.

**Examples**

```python
>>> detector = DivergenceDetector(
...     smoothing=0.3,        # EMA alpha (higher = more responsive)
...     threshold=1.0,        # Absolute: stop if smoothed - best >= 1.0
...     patience=3,           # Require 3 consecutive observations
...     action="abort",
... )
>>> trainer = Trainer(..., callbacks=[detector])
>>> trainer.train()
```

Using relative threshold (e.g., 50% increase from best):

```python
>>> detector = DivergenceDetector(
...     smoothing=0.3,
...     relative_threshold=1.5,  # Stop if smoothed >= 1.5 * best
...     patience=3,
...     action="stop",
... )
```

---

## Logging

### `JsonLogger` {#forgather-ml-trainer-callbacks-json_logger-jsonlogger}

`forgather.ml.trainer.callbacks.json_logger.JsonLogger`

A JSON logger callback that writes training metrics to a JSON file.

Writes a JSON record (with UTC timestamp, global_step, epoch, and all
reported metrics) each time ``on_log`` or ``on_evaluate`` is called.

Implements the ``Stateful`` protocol so that the log file path and last
written step are saved with checkpoints.  When training resumes from a
checkpoint, the logger reopens the original file, truncates any entries
recorded after the checkpoint step, and continues appending.

---

### `TBLogger` {#forgather-ml-trainer-callbacks-tb_logger-tblogger}

`forgather.ml.trainer.callbacks.tb_logger.TBLogger`

A Trainer callback that logs scalars to TensorBoard.

Scalars are configured as a dict mapping TensorBoard tags to spec
dicts with optional ``source`` and ``transform`` fields.  The dict
is merged with ``default_tb_scalars()`` so only deltas need to be
specified.  Set a key to ``None`` to erase a default scalar.

---

### `GradNormLogger` {#forgather-ml-trainer-callbacks-grad_logger-gradnormlogger}

`forgather.ml.trainer.callbacks.grad_logger.GradNormLogger`

Logs per-parameter gradient L2 norms to a JSON file.

Gradient norms are captured in ``on_pre_optimizer_step`` (after gradient
clipping, before optimizer step and zero_grad) and written to the log
file in ``on_evaluate``. This means gradient data is logged at eval
frequency, keeping overhead minimal.

The log file uses JSON array format with checkpoint resume support
via the Stateful protocol.

When ``fuse_optim_with_backward`` is enabled, gradients are consumed
during the backward pass and are not available for capture. The callback
detects this and disables itself with a warning.

---

### `ParameterNormLogger` {#forgather-ml-trainer-callbacks-parameter_norm_logger-parameternormlogger}

`forgather.ml.trainer.callbacks.parameter_norm_logger.ParameterNormLogger`

Logs per-parameter L2 norms and/or spectral norms to a JSON file.

Data is written on each evaluation step. The log file uses JSON array
format with checkpoint resume support via the Stateful protocol.

The existing ``WeightNormLogger`` continues to handle the total
parameter norm for TensorBoard/console logging. This callback provides
the per-parameter breakdown for diagnostic analysis and heatmap
visualization.

In pipeline-parallel training the model shell passed to callbacks
contains only meta-device tensors. This callback detects that case,
warns once, and skips logging for the remainder of training.

---

### `WeightNormLogger` {#forgather-ml-trainer-callbacks-weight_norm_logger-weightnormlogger}

`forgather.ml.trainer.callbacks.weight_norm_logger.WeightNormLogger`

Logs the total L2 norm of all model parameters to logs after each
evaluation step.

Computed identically to the gradient norm but using the weight tensors
themselves. A growing value over training indicates that weights are
increasing in magnitude, which usually means weight decay is too weak.
A stable or shrinking value while gradient norms rise points to a
different cause.

In pipeline-parallel training the model shell passed to callbacks contains
only meta-device tensors. This callback detects that case, warns once, and
skips logging for the remainder of training.

---

### `PeakMemory` {#forgather-ml-trainer-callbacks-peak_memory-peakmemory}

`forgather.ml.trainer.callbacks.peak_memory.PeakMemory`

PeakMemory is a TrainerCallback for monitoring and logging the peak CUDA memory usage during model training.
This callback is designed to help diagnose and optimize GPU memory consumption in PyTorch-based training loops,
especially when using distributed training. It records the maximum memory allocated on each GPU device throughout
the training process, and can optionally log detailed memory statistics and write them to TensorBoard for visualization.

IMPORTANT: Memory history recording is disabled by default to prevent memory leaks.
The torch.cuda.memory._record_memory_history feature can consume 1GB+ of memory during training.

Key Features:
- Tracks the peak CUDA memory allocated on each GPU during training.
- Supports both single-GPU and multi-GPU (distributed) training environments.
- Optionally logs detailed CUDA memory statistics for further analysis.
- Can write memory usage metrics to a TensorBoard SummaryWriter for visualization.
- Provides configurable logging frequency and verbosity.

**Parameters**

- `summary_writer` (SummaryWriter) — TensorBoard ``SummaryWriter`` instance for logging memory statistics.
- `show_details` (bool) — If ``True``, logs detailed CUDA memory statistics at each logging
step and at the end of training.
- `do_log` (bool) — If ``True``, logs peak memory usage at each logging step
(``on_log`` callback).
- `enable_memory_snapshot` (bool) — If ``True``, enables comprehensive CUDA memory history recording
and writes a pickled snapshot at end-of-training.
WARNING: This can consume 1 GB+ memory and cause memory leaks.
- `file_prefix` (str) — Filename prefix for the per-rank memory snapshot pickle.
Defaults to ``"memory_snapshot"``.

**Attributes**

- `rank` (int) — The process rank in distributed training.
- `world_size` (int) — The total number of processes in distributed training.
- `summary_writer` (SummaryWriter or None) — The TensorBoard ``SummaryWriter`` for logging.
- `enabled` (bool) — Whether CUDA is available and memory tracking is enabled.
- `show_details` (bool) — Whether to log detailed memory statistics.
- `do_log` (bool) — Whether to log memory usage on each log step.
- `enable_memory_snapshot` (bool) — Whether memory history recording and snapshot dumping is enabled.
- `max_allocated` (int) — The maximum CUDA memory allocated during training (in bytes).

---

## Text Generation

### `TextgenCallback` {#forgather-ml-trainer-callbacks-textgen_callback-textgencallback}

`forgather.ml.trainer.callbacks.textgen_callback.TextgenCallback`

Periodically generate and log text from a set of prompts for subjective model evaluation.

Automatically dispatches between single-rank generation (via model.generate()) and
pipeline-parallel generation (via trainer.pipeline_generate()) based on whether the
trainer exposes a pipeline_generate method. The same callback works unchanged with
SimpleTrainer, AccelTrainer/DDPTrainer, and PipelineTrainer.

---

## Advanced

### `ProfilerCallback` {#forgather-ml-trainer-callbacks-profiler_callback-profilercallback}

`forgather.ml.trainer.callbacks.profiler_callback.ProfilerCallback`

Profiles training steps and exports Chrome traces + summary tables.

---

### `DiLoCoCallback` {#forgather-ml-trainer-callbacks-diloco_callback-dilococallback}

`forgather.ml.trainer.callbacks.diloco_callback.DiLoCoCallback`

Trainer callback that manages a DiLoCoWorker for distributed local-SGD training.

Implements the Stateful protocol for checkpoint persistence. The checkpoint
manager auto-discovers Stateful callbacks and saves/restores their state.

When ``server_addr`` is empty (and DILOCO_SERVER is unset), all methods are
no-ops. This allows a single training configuration to work both with and
without a DiLoCo server.

**Parameters**

- `server_addr` (str) — DiLoCo server address (``"host:port"``). Falls back to
``DILOCO_SERVER`` env var.
- `sync_every` (int) — Local optimizer steps between syncs. Falls back to
``DILOCO_SYNC_EVERY`` env var. Default ``500``.
- `worker_id` (str) — Unique worker ID. Falls back to ``DILOCO_WORKER_ID`` env var.
Auto-generated if unset.
- `bf16_comm` (bool) — Cast pseudo-gradients to bfloat16. Falls back to
``DILOCO_BF16_COMM`` env var. Default ``True``.
- `dylu` (bool) — Enable Dynamic Local Updates. Falls back to ``DILOCO_DYLU`` env var.
Default ``False``.
- `heartbeat_interval` (float) — Seconds between heartbeats. Falls back to
``DILOCO_HEARTBEAT_INTERVAL`` env var. Default ``30.0``.
- `num_fragments` (int) — Number of streaming fragments. Falls back to
``DILOCO_NUM_FRAGMENTS`` env var. Default ``1`` (no streaming).
- `timeout` (float) — Client timeout in seconds. Default ``600``.
- `max_sync_retries` (int) — Max retries for sync failures. Default ``3``.

---

### `ResumableSummaryWriter` {#forgather-ml-trainer-callbacks-resumable_summary_writer-resumablesummarywriter}

`forgather.ml.trainer.callbacks.resumable_summary_writer.ResumableSummaryWriter`

A lazy, resumable wrapper around TensorBoard SummaryWriter.

When registered as a callback, it persists the active logging directory
in checkpoint metadata via the Stateful protocol. On resume from a
checkpoint, it redirects logging to the original directory and uses
SummaryWriter's ``purge_step`` to discard stale events recorded after
the checkpoint step.

When used as a SummaryWriter (passed to TBLogger, GradLogger, etc.),
it proxies method calls to the underlying writer, constructing it
lazily on first use.

**Parameters**

- `log_dir` (str) — Logging directory path (typically ``ns.logging_dir``
from the template system).
