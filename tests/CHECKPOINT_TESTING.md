# Checkpoint Integration Testing

Fast integration tests for checkpoint preservation and divergence detection without requiring actual model training.

## Quick Start

### Single Process (Fast)

```bash
python tests/test_checkpoint_integration.py
```

### Multi-Process (DDP Simulation)

```bash
# 2 GPUs
torchrun --nproc_per_node=2 tests/test_checkpoint_integration.py

# 4 GPUs
torchrun --nproc_per_node=4 tests/test_checkpoint_integration.py
```

## Test Scenarios

### Basic Scenario

Tests checkpoint preservation without divergence:
- 100 simulated training steps
- Gradually decreasing loss (normal training)
- Verifies best checkpoint preservation
- Verifies checkpoint count limits

```bash
python tests/test_checkpoint_integration.py --scenario basic
```

**Expected behavior:**
- Saves checkpoints at steps 25, 50, 75, 100
- Keeps 2 best checkpoints (lowest loss)
- Keeps 3 most recent checkpoints
- Total on disk: up to 5 checkpoints

### Spike Scenario

Tests divergence detection with injected loss spike:
- 150 simulated training steps
- Loss spike injected at step 75 (2.8 → 8.0)
- Verifies divergence detector triggers
- Verifies best checkpoints preserved before spike

```bash
python tests/test_checkpoint_integration.py --scenario spike
```

**Expected behavior:**
- Saves checkpoints at steps 25, 50, 75, 100
- Loss spike at step 75
- Divergence detector triggers around step 100-125
- Training stops before deleting good checkpoints
- Best checkpoints from before spike preserved

### All Scenarios

Run both scenarios:

```bash
python tests/test_checkpoint_integration.py --scenario all
```

## What Gets Tested

### 1. Checkpoint Preservation

- ✓ Best N checkpoints tracked correctly
- ✓ Best checkpoints never deleted
- ✓ save_total_limit respected
- ✓ Total checkpoints ≤ save_total_limit + preserve_n_best

### 2. Race Condition Fix

- ✓ Best checkpoint list updated BEFORE save
- ✓ Preserved list accurate during deletion
- ✓ No good checkpoints deleted

### 3. Divergence Detection

- ✓ Detector receives eval metrics
- ✓ Detector triggers on loss spike
- ✓ Training stops when divergence detected
- ✓ Stateful callback state preserved

### 4. DDP Coordination

- ✓ Only rank 0 logs
- ✓ All ranks coordinate via barriers
- ✓ Checkpoints visible to all ranks

### 5. Metrics Tracking

- ✓ Metrics shown with checkpoint list
- ✓ Best checkpoints sorted by metric
- ✓ Summary printed at end

## Verification

The test automatically verifies:

1. **Checkpoint count**: `len(best_checkpoints) <= preserve_n_best`
2. **Existence**: All best checkpoints exist on disk
3. **Total limit**: `total_checkpoints <= save_total_limit + preserve_n_best`
4. **Divergence**: Detector triggered if spike injected
5. **Sorting**: Best checkpoints sorted by metric value

Example output:

```
Verifying results...
Checkpoints on disk: 5
  checkpoint-25
  checkpoint-50
  checkpoint-75
  checkpoint-100

Best checkpoints tracked: 2
  checkpoint-25: loss=2.8234
  checkpoint-50: loss=2.7891

✓ Best checkpoints count: 2 <= preserve_n_best=2
✓ Best checkpoint exists: checkpoint-25
✓ Best checkpoint exists: checkpoint-50
✓ Total checkpoints: 5 <= 5
✓ Divergence detector triggered as expected
✓ Best checkpoints sorted correctly: [2.7891, 2.8234]

============================================================
✓ ALL ASSERTIONS PASSED
============================================================
```

## Options

```bash
python tests/test_checkpoint_integration.py --help
```

- `--scenario {basic,spike,all}`: Which scenario to run (default: all)
- `--output-dir PATH`: Custom output directory (default: temp dir)
- `--keep-outputs`: Keep checkpoint files after test (for inspection)

## Advanced Usage

### Keep Output for Inspection

```bash
python tests/test_checkpoint_integration.py --output-dir /tmp/checkpoint_test --keep-outputs
ls -la /tmp/checkpoint_test/*/checkpoint-*
```

### Run with Different Parameters

Edit the test script to customize:
- `preserve_n_best`: Number of best checkpoints
- `save_total_limit`: Max recent checkpoints
- `inject_spike_at_step`: When to inject spike
- `num_steps`: Total training steps
- `eval_interval` / `save_interval`: Checkpoint frequency

### Test Specific Bug Scenario

To reproduce the bug from the log:

```python
harness = CheckpointTestHarness(
    output_dir="/tmp/test",
    preserve_n_best=2,
    save_total_limit=3,
    inject_spike_at_step=75,  # Inject spike after 3 good checkpoints
    num_steps=150,
    eval_interval=25,
    save_interval=25,
)
```

This simulates:
- 3 good checkpoints saved (steps 25, 50, 75)
- Loss spike at step 75
- Divergence should trigger before step 125
- Checkpoints from steps 25 and 50 should be preserved

## CI Integration

Add to CI pipeline:

```yaml
- name: Checkpoint integration test
  run: |
    # Single process
    python tests/test_checkpoint_integration.py --scenario all

    # Multi-process (if GPUs available)
    torchrun --nproc_per_node=2 tests/test_checkpoint_integration.py --scenario all
```

## Troubleshooting

### Test Fails with "Best checkpoint missing"

This indicates the race condition bug - checkpoint was marked as best but then deleted.

**Fix**: Ensure `update_best_checkpoints()` is called BEFORE `save_checkpoint()`.

### Test Fails with "Divergence detector did not trigger"

This means the detector didn't receive eval metrics or threshold is too high.

**Fix**:
1. Verify `on_evaluate()` is implemented
2. Lower the threshold or increase spike magnitude

### Test Fails with Wrong Checkpoint Count

This means the trimming logic is incorrect.

**Fix**: Verify `preserve_n_best` slicing happens in `update_best_checkpoints()`.

### DDP Hangs

All ranks must hit barriers. Check for:
- Early returns that skip barriers
- Exceptions that prevent barrier calls

## Performance

Typical runtime:
- Single process: < 1 second
- 2 processes: < 2 seconds
- 4 processes: < 3 seconds

Compare to:
- Real training run: hours
- Energy cost: kWh vs milliwatt-seconds

## Next Steps

After all tests pass:
1. Run real training to verify
2. Monitor logs for single-rank logging
3. Verify checkpoint preservation in production
4. Check final summary shows correct best checkpoints
