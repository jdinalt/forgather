# AdamW + torch.compile Performance Gap

**Date**: 2026-02-23
**Status**: Partially resolved, remaining mystery documented
**Affects**: Any training using `torch.optim.AdamW` with `torch.compile` and DDP

## Summary

Training with `torch.optim.AdamW` + `torch.compile` + DDP runs significantly slower
than `Adafactor` + `torch.compile` + DDP on the same model, data pipeline, and hardware.
The gap is far larger than optimizer step time alone can explain: the compiled
forward+backward graph itself runs ~4x slower under AdamW.

**Environment**: 4x RTX 4090, ~162M parameter custom_deepone model, bf16 AMP, DDP

## Root Causes Found

### 1. `_count_batch_tokens` GPU-CPU synchronization (FIXED)

`BaseTrainer._count_batch_tokens()` called `.item()` on a GPU tensor after the compiled
forward+backward pass, forcing a full GPU-CPU synchronization every training step. This
blocked the CPU until all pending GPU work completed, serializing what should be
overlapped computation.

**Impact**: Fixing this improved AdamW from 41,831 to 58,461 tok/s (+40%).

**Fix**: Changed `_count_batch_tokens` to return a GPU `Tensor` instead of `int`,
deferring the `.item()` call to the logging step where synchronization is acceptable.

Files modified:
- `src/forgather/ml/trainer/trainer.py` — `_count_batch_tokens`, `_train_step`
- `src/forgather/ml/trainer/pipeline/pipeline_trainer.py` — `_count_batch_tokens`

### 2. `torch.optim.AdamW._get_value()` calls `.item()` per parameter (MITIGATED)

`torch.optim.adam._multi_tensor_adam` calls `_get_value()` twice per parameter per step
to read LR and beta values. Each `_get_value()` call invokes `.item()` on a CPU tensor.
For 179 parameters, this produces 358 `.item()` calls per step (716 `aten::item` events
total including the value extraction).

While these are CPU tensor operations (not GPU sync), they interact poorly with
`torch.compile`'s execution model and contribute to overhead.

**Mitigation**: Using `capturable=True` keeps step/LR tensors on GPU, eliminating these
calls. Combined with `reduce-overhead` mode, this yielded the best results.

**Note**: Adafactor produces zero `aten::item` events in profiler traces.

## Remaining Mystery: Compiled Graph Slowdown Under Memory Pressure

The most significant unexplained finding: the **same compiled forward+backward graph**
runs ~4x slower when AdamW optimizer state is present in GPU memory.

### Evidence

Profiling the compiled region (`CompiledFunction` / `CompiledFunctionBackward`) CUDA time:

| Configuration | Compiled region CUDA time/step |
|---|---|
| Adafactor | ~49.5 ms |
| AdamW | ~203 ms |

These are the same model, same compiled graph, same input data. The only difference is
that AdamW allocates ~1.2 GiB of additional GPU memory for momentum (m) and variance (v)
state tensors.

### What was ruled out

- **`.item()` synchronization**: Eliminated by `capturable=True` — compiled region still
  slower
- **GradScaler overhead**: With bf16 AMP, `self.scaler = None`; `unscale_()` is a no-op;
  `optimizer_step()` calls `optimizer.step()` directly
- **NCCL communication**: Only ~5ms total per step — negligible
- **Memory fragmentation**: Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  made performance worse
- **Optimizer step time**: The slowdown is in the forward+backward pass, not the optimizer
  step itself

### Hypotheses for further investigation

1. **CUDA memory allocator contention**: The additional 1.2 GiB of optimizer state may
   force the allocator into slower allocation patterns during the compiled graph execution.
   The allocator may need to search longer for free blocks or trigger more internal
   compaction.

2. **L2 cache pressure**: AdamW's m/v tensors consume GPU L2 cache lines, evicting
   activations and gradients that the compiled graph needs. This would cause more HBM
   round-trips during forward+backward.

3. **CUDA graph replay interference**: In `reduce-overhead` mode, CUDA graphs pre-capture
   the execution pattern. The additional memory from optimizer state may interfere with
   the graph replay mechanism or force different memory layouts.

4. **TLB (Translation Lookaside Buffer) pressure**: More allocated memory means more
   virtual-to-physical address translations, potentially causing TLB thrashing during
   kernel execution.

5. **PyTorch compiled graph re-tracing**: The presence of additional tensors in GPU memory
   may cause the compiler to make different optimization decisions, though this seems
   unlikely given the graph should be independent of optimizer state.

## All Configurations Tested

Tests run on 4x RTX 4090, ~162M parameter model, bf16 AMP, DDP, 50 steps.
Throughput measured at interval 2 (after warmup).

| Configuration | tok/s | vs Original | Notes |
|---|---|---|---|
| **Adafactor + compile** | 183,831 | — | Reference (best non-RO) |
| **Adafactor + reduce-overhead** | 188,560 | — | Reference (best overall) |
| AdamW compile (BEFORE fix) | 41,831 | baseline | 459 `.item()` calls/step |
| AdamW compile (AFTER fix) | 58,461 | +40% | `.item()` fix only |
| AdamW + capturable compile | 53,660 | +28% | capturable=True, default mode |
| AdamW + foreach=False fused=False | slower | — | Disabled multi-tensor |
| Forgather AdamW compile | 41,776 | ~0% | Pure Python loop, very slow |
| Forgather compiled AdamW | 17,370 | -58% | torch_compile on optimizer, terrible |
| **AdamW + reduce-overhead** | 82,481 | +97% | CUDA graphs help significantly |
| **AdamW + capturable + reduce-overhead** | **104,154** | **+149%** | **Best AdamW result** |
| Forgather compiled AdamW + RO | 17,370 | -58% | Optimizer compile + RO incompatible |

## Recommended Configuration

For AdamW + torch.compile + DDP, use:

```yaml
[trainer_args]
    default_dtype: "float32"
    mixed_precision: "bf16"
    torch_compile_mode: "reduce-overhead"
    torch_compile_dynamic: False

[optimizer]
optimizer: &optimizer !partial:torch.optim:AdamW
    lr: <your_lr>
    capturable: True
```

This achieves 104k tok/s vs the original 42k tok/s (2.5x improvement), though a 1.8x
gap to Adafactor (188k tok/s) remains due to the compiled graph slowdown under memory
pressure.

## Config Files

Test configurations are in `examples/pretrain/small-llm/templates/configs/`:

| File | Description |
|---|---|
| `adamw_amp.yaml` | Baseline AdamW + AMP (default compile) |
| `adamw_amp_capturable.yaml` | AdamW with capturable=True |
| `adamw_amp_capturable_ro.yaml` | AdamW + capturable + reduce-overhead (best) |
| `adamw_amp_reduce_overhead.yaml` | AdamW + reduce-overhead (no capturable) |
| `adamw_amp_nofused.yaml` | AdamW with foreach=False fused=False |
| `adamw_amp_forgather.yaml` | Forgather's custom AdamW |
| `adamw_amp_fg_compile.yaml` | Forgather AdamW with torch_compile |
| `adamw_amp_fg_compile_ro.yaml` | Forgather compiled AdamW + reduce-overhead |
| `amp_reduce_overhead.yaml` | Adafactor + reduce-overhead |

## Reproduction

All tests were run from the `examples/pretrain/small-llm` project directory on 4x RTX 4090.
The common dynamic args used for every run:

```bash
DYNAMIC='{"compile": true, "init_model": true, "attn_implementation": "flex_attention", "save_strategy": "no", "step_cadence": 0.8, "max_steps": 50}'
```

### Training throughput benchmarks

```bash
# Adafactor baseline (amp.yaml)
CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nproc_per_node=4 \
  scripts/train_script.py -p examples/pretrain/small-llm amp.yaml \
  --dynamic-args "$DYNAMIC"

# AdamW baseline
CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nproc_per_node=4 \
  scripts/train_script.py -p examples/pretrain/small-llm adamw_amp.yaml \
  --dynamic-args "$DYNAMIC"

# AdamW + capturable + reduce-overhead (best AdamW result)
CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nproc_per_node=4 \
  scripts/train_script.py -p examples/pretrain/small-llm adamw_amp_capturable_ro.yaml \
  --dynamic-args "$DYNAMIC"

# Substitute any config from the table above in place of the .yaml filename.
```

### Profiling (.item() counts and Chrome traces)

```bash
# AdamW profiling (produces Chrome trace + .item() summary)
CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nproc_per_node=4 \
  benchmarks/profile_real_training.py \
  --optimizer adamw --compile --max-steps 15 --profile-start 3 --profile-steps 5

# Adafactor profiling
CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nproc_per_node=4 \
  benchmarks/profile_real_training.py \
  --optimizer adafactor --compile --max-steps 15 --profile-start 3 --profile-steps 5
```

Adjust `CUDA_VISIBLE_DEVICES` to match available GPUs. The profiler script outputs
Chrome traces to `benchmarks/profiles/` by default (override with `--output-dir`).

## Potential Next Steps

1. **File a PyTorch issue** about compiled graph slowdown under optimizer memory pressure.
   Reproduce with a minimal script showing the same `torch.compile`d model running slower
   when additional tensors are allocated on GPU.

2. **Profile with Nsight Systems** to get GPU-side kernel timing and memory transaction
   counts, which would confirm or rule out L2 cache / TLB pressure hypotheses.

3. **Test with `torch.cuda.memory.CUDAPluggableAllocator`** or alternative allocation
   strategies to see if allocator behavior is the root cause.

4. **Test gradient checkpointing** to reduce activation memory and see if freeing memory
   for optimizer state eliminates the compiled graph slowdown.

5. **Compare with FSDP** where optimizer state is sharded across GPUs, reducing per-GPU
   memory pressure.
