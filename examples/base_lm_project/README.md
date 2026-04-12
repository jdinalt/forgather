# Base LM Project

A test harness for the base LM project template.

## Test LM Training Project Template

```bash
# With defaults  -- full float32 precision
forgather train

# Train in mixed-precision (bf16), with compile, and Ampere or later GPU
# Slower startup, but _much_ faster!
forgather train --compile true --mixed-precision bf16 --float32-matmul-precision high

# FSDP2 (fully_shard) -- parameters, gradients and optimizer state are sharded
# across the data-parallel mesh. 2 GPUs by default.
forgather -t fsdp2.yaml train
```

## Test Auto LR Project Template

Note that this configuration defaults to DDP. If you don't want to train on all GPUs, pass the `-d X` argument, where X is the set of GPU indices to use.

```bash
# With defaults
forgather -t auto_lr_project.yaml train

# Train with 8 gradient accumulation steps on GPU 0 only; scales LR automatically
forgather -t auto_lr_project.yaml train --gradient-accumulation-steps 8 -d 0

# With mixed-precision and only train on GPUs 0 and 1; as above, the lr will be scaled for the larger batch size
forgather -t auto_lr_project.yaml train --compile true --mixed-precision bf16 --float32-matmul-precision high -d 0,1
```

Beware that stopping DDP with ^C is not all that reliable; it can leave one or more workers stuck, spewing errors on the console. If this happens, you will need to track down the PIDs of the stuck workers, `nvidia-smi`, and kill them.

To cleanly shut this down, use:

```bash
# Find Forgather JOB_ID
forgather control list

# Stop it
forgather control stop JOB_ID
```