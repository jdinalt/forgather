# Checkpoint State Debugging Tools

This directory contains tools for debugging InterleavedDataset checkpoint issues, particularly problems related to `examples_per_dataset` not being saved in checkpoints.

## Tools

### dump_checkpoint_state.py

Dumps and analyzes the complete dataset checkpoint state, showing all internal counters and state variables.

**Usage:**
```bash
# Basic usage - analyze checkpoint from rank 0
./dump_checkpoint_state.py /path/to/checkpoint-36621

# Analyze specific rank in multi-process checkpoint
./dump_checkpoint_state.py /path/to/checkpoint-36621 --rank 1

# JSON output for programmatic processing
./dump_checkpoint_state.py /path/to/checkpoint-36621 --format json

# Quick summary
./dump_checkpoint_state.py /path/to/checkpoint-36621 --format summary

# Skip trainer state display
./dump_checkpoint_state.py /path/to/checkpoint-36621 --no-trainer-state
```

**When to use:**
- When you need to see the raw checkpoint state structure
- To verify what state variables are saved/missing
- To understand the nesting structure of InterleavedDataset hierarchies
- To check packing ratios and length estimation states

**Key output sections:**
- **RAW CHECKPOINT STATE**: Complete JSON dump of checkpoint
- **ANALYSIS**: Structured view of InterleavedDataset state
  - Top-level state (current_example_count, probabilities, stopping_strategy)
  - Whether `examples_per_dataset` is saved (critical for soft_sequential)
  - Child dataset states (nested InterleavedDatasets or SimpleArrowIterableDataset)
  - Length estimation state (input_count, output_count, packing_ratio)
  - Shuffle state (seeds, epoch)
- **TRAINER STATE**: Training step, epoch, total steps

### analyze_checkpoint_probabilities.py

Analyzes how InterleavedDataset computes sampling probabilities with `stopping_strategy='soft_sequential'` and demonstrates the bug where missing `examples_per_dataset` causes incorrect probability calculations.

**Usage:**
```bash
# Basic usage
./analyze_checkpoint_probabilities.py /path/to/checkpoint-36621

# Analyze specific rank
./analyze_checkpoint_probabilities.py /path/to/checkpoint-36621 --rank 1

# JSON output
./analyze_checkpoint_probabilities.py /path/to/checkpoint-36621 --format json

# Quick summary
./analyze_checkpoint_probabilities.py /path/to/checkpoint-36621 --format summary
```

**When to use:**
- When you see unexpected training loss changes after checkpoint resume
- To verify soft_sequential probability calculations
- To understand the impact of missing `examples_per_dataset`
- To debug dataset sampling behavior

**Key output sections:**
- **DATASET STRUCTURE**: Lists all child datasets with:
  - Estimated length (based on packing ratio)
  - Examples already yielded
  - Percentage consumed
- **PROBABILITY ANALYSIS**:
  - **SCENARIO 1**: Correct probabilities (if examples_per_dataset was saved)
  - **SCENARIO 2**: Buggy probabilities (if examples_per_dataset was missing)
- **IMPACT**: Shows probability changes between scenarios
  - Highlights nearly-exhausted datasets that get "revived" after resume
  - Explains how this causes sudden training dynamics shifts

## Understanding the Output

### Critical Indicators

**dump_checkpoint_state.py**:
```
examples_per_dataset: NOT SAVED IN CHECKPOINT!
WARNING: This will cause incorrect probability calculations on resume
         for stopping_strategy='soft_sequential'
```
This indicates the checkpoint has the bug. On resume, probability calculations will be incorrect.

**analyze_checkpoint_probabilities.py**:
```
WARNING: This dataset was 99.78% consumed before resume
but will get 72.45% sampling probability after resume!
This can cause sudden shifts in training dynamics.
```
This shows a nearly-exhausted dataset will dominate sampling after resume, causing a sudden change in training loss.

### Interpreting Packing Ratios

The packing ratio shows how efficiently examples are packed into sequences:
- **Low ratio (~0.01-0.1)**: High packing efficiency (many short examples per sequence)
- **High ratio (~0.8-1.0)**: Low packing (one example per sequence, or long examples)

Example:
```
input_count: 5,000        # Files read
output_count: 200         # Sequences yielded
packing_ratio: 0.04       # 200 / 5000 = high packing
estimated_length: 40,000  # Estimated total sequences (based on ratio)
```

### Soft Sequential Probability Calculation

For `stopping_strategy='soft_sequential'`, probabilities are computed as:

1. For each dataset, compute remaining proportion:
   ```
   remaining = estimated_length - examples_yielded
   proportion = remaining / estimated_length
   ```

2. Compute weights sequentially:
   ```
   weight[0] = 1.0 * proportion[0]
   weight[1] = (1.0 - proportion[0]) * proportion[1]
   weight[2] = (1.0 - proportion[0]) * (1.0 - proportion[1]) * proportion[2]
   ...
   ```

3. Normalize weights to probabilities:
   ```
   probabilities = weights / sum(weights)
   ```

**The Bug**: If `examples_per_dataset` is not saved, it gets reset to `[0, 0, ...]` on resume. This makes all datasets appear to have 100% remaining, dramatically changing the probabilities.

## Common Scenarios

### Scenario 1: Training Loss Suddenly Drops After Resume

**Symptoms:**
- Training was progressing normally
- After checkpoint resume, loss suddenly drops
- Loss remains at the new lower level

**Diagnosis:**
```bash
./analyze_checkpoint_probabilities.py /path/to/checkpoint-36621
```

Look for:
- "examples_per_dataset: NOT SAVED IN CHECKPOINT!"
- Large probability changes in the IMPACT section
- A nearly-exhausted dataset (>95% consumed) that gets revived

**Explanation:** A nearly-exhausted "easy" dataset (like small stories) gets full sampling probability after resume, making training appear to make sudden progress.

### Scenario 2: Verifying a Checkpoint is Correct

**Check:**
```bash
./dump_checkpoint_state.py /path/to/checkpoint --format summary
```

**Expected output for correct checkpoint:**
```
Checkpoint: checkpoint-36621
  current_example_count: 146,486
  examples_per_dataset: SAVED
  num_children: 2
```

If you see "NOT SAVED", the checkpoint has the bug.

### Scenario 3: Understanding Dataset Consumption

**Check:**
```bash
./analyze_checkpoint_probabilities.py /path/to/checkpoint
```

This shows how much of each dataset has been consumed:
```
Dataset 0 (high packing - ratio 0.0415):
  Estimated length: 116,364
  Already yielded: 25,261
  Consumed: 21.71%

Dataset 1 (nested InterleavedDataset):
  Estimated length: 1,495,032
  Already yielded: 121,225
  Consumed: 8.11%
```

## Related Documentation

For detailed technical context on this bug and the fix, see:
- `../../CHECKPOINT_BUG_ANALYSIS.md` - Complete bug analysis and solution

For general checkpoint documentation:
- `../../docs/checkpointing/user_guide.md` - Checkpoint usage guide
- `../../docs/checkpointing/distributed_checkpoint_abstraction.md` - Technical details

## Installation

These scripts require PyTorch:
```bash
pip install torch
```

No other dependencies are needed beyond Python standard library.
