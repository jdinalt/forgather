# Checkpoint Parameter Modification Tool

Safely modify optimizer, scheduler, and other component parameters in Forgather checkpoints without restarting training from scratch.

## Features

- **Safe modifications**: Atomic file operations with validation prevent corruption
- **Flexible**: Works with optimizer, scheduler, and other checkpoint components
- **Discovery**: List available parameters before modification
- **Distributed training support**: Handles PER_RANK, GLOBAL, REPLICATED patterns
- **Backup protection**: Optional backup creation with filesystem sync
- **Dry-run mode**: Preview changes before applying

## Quick Start

### Discover available components

```bash
# List all available components in a checkpoint
forgather checkpoint components checkpoint-92500/

# Shows: optimizer, scheduler, dataset, rng, trainer, etc.
```

### List modifiable parameters

```bash
# List optimizer parameters
forgather checkpoint list checkpoint-92500/ --component optimizer

# List scheduler parameters
forgather checkpoint list checkpoint-92500/ --component scheduler
```

### Modify parameters

```bash
# Change optimizer weight_decay
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-92500/ \
    --component optimizer \
    --set weight_decay=0.01

# Change learning rate
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-92500/ \
    --component optimizer \
    --set lr=0.0001

# Change multiple parameters at once
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-92500/ \
    --component optimizer \
    --set weight_decay=0.01 \
    --set lr=0.0001 \
    --set betas="(0.9,0.98)"

# Modify scheduler state
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-92500/ \
    --component scheduler \
    --set last_epoch=100
```

### Preview changes (dry-run)

```bash
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-92500/ \
    --component optimizer \
    --set weight_decay=0.01 \
    --dry-run
```

## Understanding Components

A **component** is a logical unit of checkpoint state identified by its file prefix. For example:

- `optimizer_state.pt` → component name: `optimizer`
- `scheduler_state.pt` → component name: `scheduler`
- `dataset_state.pt` → component name: `dataset`
- `rng_state_rank_0.pt` → component name: `rng`

Common components in Forgather checkpoints:
- **optimizer** - Optimizer state (learning rate, momentum buffers, etc.)
- **scheduler** - Learning rate scheduler state
- **model** - Model weights (usually not modifiable with this tool)
- **dataset** - Dataset iteration state
- **rng** - Random number generator state
- **trainer** - Training progress metadata

Use `forgather checkpoint components CHECKPOINT` to see what components are available in a specific checkpoint.

## Command Reference

### `components` - List available components

```bash
forgather checkpoint components CHECKPOINT_PATH [OPTIONS]
```

**Options:**
- `--verbose, -v` - Show individual file details for each component

**Example output:**
```
Checkpoint: /path/to/checkpoint-1000

Found 5 component(s):

  optimizer
    Files: 1
    Size: 786.75 KB

  scheduler
    Files: 1
    Size: 1.59 KB

  dataset
    Files: 5
    Size: 14.04 KB

  rng
    Files: 5
    Size: 34.28 KB

  trainer
    Files: 1
    Size: 1.39 KB
```

### `list` - List modifiable parameters

```bash
python tools/modify_checkpoint/modify_checkpoint.py list CHECKPOINT_PATH [OPTIONS]
```

**Options:**
- `--component COMPONENT` - Component to inspect (default: optimizer)
- `--verbose, -v` - Show value types and detailed information

**Example output:**
```
Component: optimizer
Files: optimizer_state.pt

Param group 0:
  lr: 0.001
  weight_decay: 0.0
  betas: (0.9, 0.999)
  eps: 1e-08
  amsgrad: False
```

### `modify` - Modify parameters

```bash
python tools/modify_checkpoint/modify_checkpoint.py modify CHECKPOINT_PATH [OPTIONS]
```

**Required:**
- `--component COMPONENT` - Component to modify (e.g., optimizer, scheduler)
- At least one modification:
  - `--set KEY=VALUE` - Set parameter to exact value
  - `--scale KEY=FACTOR` - Multiply parameter by factor

**Optional:**
- `--param-group INDEX` - For optimizer: target specific param group (default: all)
- `--dry-run` - Preview changes without saving
- `--no-backup` - Skip backup creation (still uses atomic operations)
- `--force` - Skip confirmation prompts
- `--verbose, -v` - Detailed logging
- `--quiet, -q` - Minimal output

## Examples

### Experiment with weight decay

```bash
# Start with no weight decay (0.0)
# Train for a while, save checkpoint at step 10000

# Change to weight_decay=0.01 and continue training
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-10000/ \
    --component optimizer \
    --set weight_decay=0.01

# Resume training from modified checkpoint
forgather -t config.yaml train --resume-from-checkpoint checkpoint-10000
```

### Scale learning rate

```bash
# Reduce learning rate by half
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-10000/ \
    --component optimizer \
    --scale lr=0.5
```

### Modify specific param group

```bash
# Some optimizers use different hyperparameters for different layers
# Target only param group 0
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-10000/ \
    --component optimizer \
    --param-group 0 \
    --set weight_decay=0.01
```

### Modify scheduler state

```bash
# Change scheduler's last_epoch (useful for resuming with different schedule)
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-10000/ \
    --component scheduler \
    --set last_epoch=5000
```

## Safety Features

### Atomic file operations

The tool uses atomic file operations to prevent corruption:

1. **Backup creation** (optional): Original file copied with fsync
2. **Temp file write**: Modified state written to `.tmp` file with fsync
3. **Validation**: Temp file loaded to verify it's not corrupted
4. **Atomic rename**: `os.rename()` atomically replaces original
5. **Directory sync**: Ensures metadata is persisted to disk

This prevents corruption even if the process crashes or power is lost during save.

### Validation

Before committing changes, the tool:
- Loads the temporary file to verify it's not corrupted
- Checks that state structure matches expected format
- If validation fails, deletes temp file and keeps original untouched

### Backup files

By default, the tool creates backup files (`.bak`) before modification. These are synced to disk with `fsync()` for safety.

To skip backup creation:
```bash
python tools/modify_checkpoint/modify_checkpoint.py modify \
    checkpoint-10000/ \
    --component optimizer \
    --set weight_decay=0.01 \
    --no-backup
```

## Distributed Training Support

The tool automatically handles different checkpoint file patterns:

- **GLOBAL/REPLICATED**: Single file (`optimizer_state.pt`)
- **PER_RANK**: Per-rank files (`optimizer_state_rank_0.pt`, `optimizer_state_rank_1.pt`, ...)
- **PER_GROUP**: Per-group files (`optimizer_state_group_*_grank_*_rank_*.pt`)
- **PER_NODE**: Per-node files (`optimizer_state_node_*_rank_*.pt`)

All matching files are modified consistently.

## Checkpoint Manifest

If `checkpoint_manifest.json` exists, the tool:
1. Updates `size_bytes` to reflect new file sizes
2. Clears `checksum` (no longer valid after modification)
3. Adds modification metadata (`modified_by`, `modified_at`)

The manifest is updated atomically using the same safety mechanisms as checkpoint files.

## Value Parsing

The tool supports various Python literal formats:

```bash
# Numbers
--set lr=0.01
--set lr=1e-4

# Booleans
--set amsgrad=True
--set amsgrad=false

# Tuples (for betas, etc.)
--set betas="(0.9,0.98)"
--set betas="(0.9,0.999)"

# Lists
--set base_lrs="[0.001,0.0001]"

# Strings
--set optimizer_name="'adam'"
```

## Troubleshooting

### Error: Checkpoint path not found

**Solution**: Verify the checkpoint directory exists and contains checkpoint files.

```bash
ls checkpoint-10000/
# Should show: optimizer_state.pt, model.safetensors, etc.
```

### Error: Parameter not found

**Solution**: Use the `list` command to see available parameters.

```bash
python tools/modify_checkpoint/modify_checkpoint.py list \
    checkpoint-10000/ --component optimizer
```

### Error: Modified checkpoint failed validation

**Solution**: This indicates corruption during save. The original file is unchanged, and backup is available.

```bash
# Restore from backup if needed
cp checkpoint-10000/optimizer_state.pt.bak checkpoint-10000/optimizer_state.pt
```

### Warning: Failed to update checkpoint manifest

**Cause**: Manifest update failed (rare).

**Impact**: Checkpoint is still modified successfully, but manifest may be inconsistent.

**Solution**: The backup manifest (`.bak`) is available if needed.

## Integration with Forgather

After modifying a checkpoint, resume training normally:

```bash
# Modify checkpoint
python tools/modify_checkpoint/modify_checkpoint.py modify \
    output_models/my_model/checkpoints/checkpoint-10000 \
    --component optimizer \
    --set weight_decay=0.01

# Resume training from modified checkpoint
forgather -t config.yaml train \
    --resume-from-checkpoint output_models/my_model/checkpoints/checkpoint-10000
```

The modified parameters will be used for continued training.

## Limitations

### Non-tensor values only

The tool only modifies non-tensor values (hyperparameters like `lr`, `weight_decay`).

It **cannot** modify:
- Optimizer state tensors (momentum buffers, variance estimates)
- Model weights
- RNG state

These are intentionally excluded to prevent corruption.

### Param groups structure

When modifying optimizer param groups, the tool assumes the parameter exists in all targeted groups. If you have different parameters in different groups, use `--param-group INDEX` to target specific groups.

## Advanced Usage

### Modify all checkpoints in a directory

```bash
# Find all checkpoints and modify them
for checkpoint in output_models/my_model/checkpoints/checkpoint-*/; do
    python tools/modify_checkpoint/modify_checkpoint.py modify \
        "$checkpoint" \
        --component optimizer \
        --set weight_decay=0.01 \
        --force  # Skip confirmation prompts
done
```

### Programmatic usage

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = Path("checkpoint-10000/optimizer_state.pt")
state_dict = torch.load(checkpoint_path, map_location='cpu')

# Modify parameter
state_dict["param_groups"][0]["weight_decay"] = 0.01

# Save with atomic operations (see modify_checkpoint.py for implementation)
from modify_checkpoint import save_checkpoint_atomically
save_checkpoint_atomically(checkpoint_path, state_dict, backup=True)
```

## Contributing

If you find bugs or have suggestions, please open an issue or PR in the Forgather repository.

## License

Same as Forgather (see repository LICENSE file).
