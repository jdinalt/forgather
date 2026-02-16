# Checkpoint Bug Analysis: Training Loss Discontinuity at Step 36,621

## Summary

Training resumed from checkpoint-36621 exhibited a large **drop** in training loss. Investigation revealed that `InterleavedDataset` does not save `examples_per_dataset` in its checkpoint state, causing incorrect probability calculations for the `soft_sequential` sampling strategy.

## The Bug

### What's Missing

The `InterleavedDataset.state_dict()` method (src/forgather/ml/datasets/interleaved.py:257-282) saves:
- ✅ `current_dataset_index`
- ✅ `current_example_count`
- ✅ `datasets_exhausted`
- ✅ `child_states` (for each child dataset)
- ❌ **`examples_per_dataset`** (NOT SAVED!)

### Why It Matters

The `soft_sequential` probability function computes sampling weights based on:

```python
remaining = len(dataset) - examples_per_dataset[i]
proportion = remaining / len(dataset)
```

When `examples_per_dataset` is reset to `[0, 0, ...]` on resume:
- All datasets appear to have 100% of their examples remaining
- The probability function thinks no progress has been made
- Earlier datasets (like Tiny Stories) get sampled with high probability again

### Impact on Your Training Run

**At checkpoint time (step 36,621):**

| Dataset | Estimated Length | Already Yielded | Remaining | Correct Probability |
|---------|-----------------|-----------------|-----------|---------------------|
| Tiny Stories | 25,317 | 25,261 (99.78%) | 56 | **0.22%** |
| Fineweb + Cosmopedia | 12,164,382 | 121,225 | 12,043,157 | **99.78%** |

**After resume (BUG):**

| Dataset | Estimated Length | Already Yielded (bug) | Remaining (bug) | Buggy Probability |
|---------|-----------------|------------------------|------------------|-------------------|
| Tiny Stories | 25,317 | 0 ❌ | 25,317 ❌ | **100.00%** ❌ |
| Fineweb + Cosmopedia | 12,164,382 | 0 ❌ | 12,164,382 ❌ | **0.00%** ❌ |

**Result:** The model suddenly starts sampling from Tiny Stories (which it had nearly exhausted) with 100% probability instead of 0.22%. Since Tiny Stories is much easier than the other datasets, training loss drops dramatically.

## Root Cause Details

### Dataset Structure

Your training configuration uses a 3-level dataset hierarchy:

```
InterleavedDataset (soft_sequential)
├── Dataset 0: Tiny Stories (SimpleArrowIterableDataset, packed)
└── Dataset 1: InterleavedDataset (soft_sequential, nested)
    ├── Child 0: Fineweb-Edu (SimpleArrowIterableDataset, packed)
    └── Child 1: Cosmopedia (SimpleArrowIterableDataset, packed)
```

### Checkpoint State Analysis

**Child datasets (SimpleArrowIterableDataset) correctly save:**
- `input_count`: 423,000 (99.78% of Tiny Stories consumed)
- `output_count`: 25,261 (after packing)
- These statistics correctly estimate remaining examples via `__len__()`

**InterleavedDataset fails to save:**
- `examples_per_dataset`: The running count of examples yielded from each child
- This count is used by `soft_sequential` to compute remaining examples

### Code Flow

1. **On checkpoint save:**
   ```python
   # interleaved.py:257-282
   def state_dict(self):
       return {
           "current_example_count": self._current_example_count,  # Total yielded
           # ... other state ...
           # Missing: examples_per_dataset is NOT saved!
       }
   ```

2. **On checkpoint resume:**
   ```python
   # interleaved.py:92-122
   def __iter__(self):
       examples_per_dataset = [0] * len(self.datasets)  # ❌ Reset to zeros!
       # ...
       while True:
           if self._probabilities_callable:
               current_probs = self._probabilities_fn(
                   step, self.datasets, examples_per_dataset, exhausted
               )
   ```

3. **Probability calculation:**
   ```python
   # soft_sequential.py:69-73
   total_length = len(dataset)  # ✅ Correct (uses restored input_count/output_count)
   remaining = max(0, total_length - count)  # ❌ count is 0 instead of 25,261!
   proportion = remaining / total_length  # ❌ 1.0 instead of 0.0022!
   ```

## The Fix

### Required Changes

**File:** `src/forgather/ml/datasets/interleaved.py`

#### 1. Save `examples_per_dataset` in `state_dict()` (line ~270)

```python
def state_dict(self) -> Dict[str, Any]:
    state = {
        "current_dataset_index": self._current_dataset_index,
        "current_example_count": self._current_example_count,
        "datasets_exhausted": self._datasets_exhausted.copy(),
        "probabilities": self.probabilities,
        "seed": self.seed,
        "stopping_strategy": self.stopping_strategy,
        "child_states": [],
    }

    # ADD THIS: Track how many examples yielded from each child
    # This is needed for correct probability calculations on resume
    if hasattr(self, '_examples_per_dataset_checkpoint'):
        state["examples_per_dataset"] = self._examples_per_dataset_checkpoint.copy()

    # Save state for each child dataset
    for i, dataset in enumerate(self.datasets):
        if hasattr(dataset, "state_dict"):
            state["child_states"].append(dataset.state_dict())
        else:
            state["child_states"].append(None)

    return state
```

#### 2. Track `examples_per_dataset` during iteration (line ~105)

```python
def __iter__(self):
    # Track examples per dataset (for dynamic probabilities and checkpointing)
    examples_per_dataset = [0] * len(self.datasets)

    # Restore from checkpoint if available
    if hasattr(self, '_restored_examples_per_dataset'):
        examples_per_dataset = self._restored_examples_per_dataset.copy()
        delattr(self, '_restored_examples_per_dataset')

    # ... rest of __iter__ ...

    while True:
        # ... sampling logic ...

        try:
            example = next(iterators[chosen_idx])
            examples_yielded += 1
            examples_per_dataset[chosen_idx] += 1

            # Save for checkpointing
            self._examples_per_dataset_checkpoint = examples_per_dataset.copy()

            yield example
```

#### 3. Restore `examples_per_dataset` in `load_state_dict()` (line ~295)

```python
def load_state_dict(self, state_dict: Dict[str, Any]):
    self._current_dataset_index = state_dict["current_dataset_index"]
    self._current_example_count = state_dict["current_example_count"]
    self._datasets_exhausted = state_dict.get(
        "datasets_exhausted", [False] * len(self.datasets)
    )

    # Restore examples_per_dataset if available
    if "examples_per_dataset" in state_dict:
        self._restored_examples_per_dataset = state_dict["examples_per_dataset"].copy()

    # Restore state for each child dataset
    child_states = state_dict.get("child_states", [])
    for i, (dataset, child_state) in enumerate(zip(self.datasets, child_states)):
        if child_state is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(child_state)
```

### Testing the Fix

After implementing the fix, verify with:

```bash
# 1. Run the analysis script on your old checkpoint
python analyze_checkpoint_probabilities.py /path/to/checkpoint-36621

# 2. Train for a few steps and save a new checkpoint
forgather -t config.yaml train --max_steps 100

# 3. Verify the new checkpoint includes examples_per_dataset
python dump_checkpoint_state.py /path/to/new-checkpoint

# Look for: "examples_per_dataset: [25261, 121225]" (not "NOT SAVED")
```

## Verification Scripts

Two scripts have been created to investigate this issue:

### 1. `dump_checkpoint_state.py`
Dumps raw checkpoint state and identifies missing fields.

```bash
python dump_checkpoint_state.py /path/to/checkpoint-36621 [rank]
```

### 2. `analyze_checkpoint_probabilities.py`
Computes soft_sequential probabilities with and without the bug.

```bash
python analyze_checkpoint_probabilities.py /path/to/checkpoint-36621 [rank]
```

## Impact Assessment

**Severity:** High

**Affected Use Cases:**
- ✅ Any training using `InterleavedDataset` with dynamic probability functions
- ✅ Specifically affects `soft_sequential` and `balance_remaining_examples`
- ❌ Does NOT affect static probabilities or round-robin interleaving

**Symptoms:**
- Training loss discontinuity on checkpoint resume
- Unexpected changes in dataset sampling distribution
- Loss may increase OR decrease depending on dataset difficulty ordering

## Additional Considerations

### Nested InterleavedDatasets

Your configuration has nested `InterleavedDataset` instances. The fix must be applied at **all levels**:
- Top-level InterleavedDataset (Tiny Stories + Combined)
- Nested InterleavedDataset (Fineweb-Edu + Cosmopedia)

### Backwards Compatibility

Old checkpoints (without `examples_per_dataset`) will continue to work with the fix, they'll just start with `[0, 0, ...]`. This means:

- ✅ You can apply the fix and continue training from old checkpoints
- ⚠️ The first few batches after resume will still have incorrect probabilities
- ✅ New checkpoints will save the correct state going forward

For your specific case at step 36,621:
- Tiny Stories is already ~100% consumed in the raw data
- Even with the bug, it will exhaust very quickly (56 examples remain)
- The training will self-correct within a few hundred steps

### Alternative Workaround

If you don't want to modify the code immediately, you can work around this by:

1. **Reset from a checkpoint before Tiny Stories was nearly exhausted**
2. **OR** accept the discontinuity and continue training (it will self-correct)
3. **OR** remove Tiny Stories from the configuration and resume

## Recommendation

**Implement the fix immediately** to prevent this issue in future training runs. For the current run at step 36,621, you can either:

1. **Continue training** - The bug will self-correct within ~500 steps as Tiny Stories exhausts
2. **Restart from an earlier checkpoint** - Before Tiny Stories was nearly exhausted
3. **Manual fix** - Remove Tiny Stories from config, manually adjust checkpoint to continue with just Fineweb+Cosmopedia

The loss discontinuity you observed is **expected behavior given the bug** and does not indicate a more serious problem with the model or training setup.
