# Pipeline Parallel Training

Test harness for the Pipeline Parallel trainer, covering all supported PyTorch
pipeline schedules, checkpoint save/resume, and activation checkpointing.

For documentation on the PipelineTrainer module, see
[docs/trainers/pipeline-parallel.md](../../../docs/trainers/pipeline-parallel.md).

## Configurations

**Baseline (no pipeline):**

| Config | Description |
|--------|-------------|
| `control.yaml` | Standard trainer, single GPU -- baseline for comparison |

**Single-stage schedules** (`stages_per_rank=1`):

| Config | GPUs | Schedule | Description |
|--------|------|----------|-------------|
| `tiny_gpipe_2gpu.yaml` | 2 | GPipe | Quick debug test with tiny model |
| `gpipe_2gpu.yaml` | 2 | GPipe | GPipe on 2 GPUs |
| `gpipe_4gpu.yaml` | 4 | GPipe | GPipe on 4 GPUs |
| `1f1b_4gpu.yaml` | 4 | 1F1B | 1-forward-1-backward on 4 GPUs |

**Multi-stage schedules** (`stages_per_rank=2`):

| Config | GPUs | Schedule | Description |
|--------|------|----------|-------------|
| `interleaved_1f1b_4gpu.yaml` | 4 | Interleaved 1F1B | 2 stages per rank, interleaved |
| `looped_bfs_4gpu.yaml` | 4 | Looped BFS | Breadth-first scheduling |
| `zero_bubble_4gpu.yaml` | 4 | Interleaved Zero Bubble | Minimal pipeline bubble |
| `zbvz_4gpu.yaml` | 4 | ZBV Zero Bubble | V-pattern stage assignment |

**Activation checkpointing:**

| Config | Description |
|--------|-------------|
| `gpipe_4gpu_activation_checkpoint.yaml` | GPipe with gradient checkpointing enabled |

**Checkpoint save/resume:**

| Config | Description |
|--------|-------------|
| `checkpoint_test_train.yaml` | Train with frequent checkpoints (`save_steps: 50`) |
| `checkpoint_test_resume.yaml` | Resume from latest checkpoint |

**Cross-architecture:**

| Config | Description |
|--------|-------------|
| `tiny_llama_gpipe_3gpu.yaml` | Llama model (instead of default causal) on 3 GPUs |

## Usage

```bash
# Run a 4-GPU pipeline test
forgather -t gpipe_4gpu.yaml train -d 0,1,2,3

# Compare schedules
forgather -t 1f1b_4gpu.yaml train -d 0,1,2,3
forgather -t interleaved_1f1b_4gpu.yaml train -d 0,1,2,3

# Test checkpoint round-trip
forgather -t checkpoint_test_train.yaml train -d 0,1
forgather -t checkpoint_test_resume.yaml train -d 0,1
```

## References

- [PyTorch Pipeline Parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2410.06511)
- [TorchTitan](https://github.com/pytorch/torchtitan)
