# Checkpoints

Forgather's checkpointing system saves and restores all trainer state — model weights,
optimizer, LR scheduler, dataset position, per-rank RNG, and training progress. Model
weights are written as standard HuggingFace Safetensors shards, readable by any
HF-compatible tool without a Forgather dependency.

**Related documentation:**

- [Checkpointing Overview](../checkpointing/README.md) — concepts and basic usage
- [User Guide](../checkpointing/user_guide.md) — practical patterns and troubleshooting
- [Distributed Abstraction](../checkpointing/distributed_checkpoint_abstraction.md) — state-sharing patterns (GLOBAL, PER_RANK, REPLICATED, etc.)
- [Migration Guide](../checkpointing/migration_guide.md) — implementing custom trainers
- [Sharded Checkpoint API](../checkpointing/sharded_checkpoint_api.md) — low-level shard API reference

---

## Checkpoint Management

### `CheckpointMeta` {#forgather-ml-sharded_checkpoint-checkpointmeta}

`forgather.ml.sharded_checkpoint.CheckpointMeta`

```python
class CheckpointMeta(file_name: str, is_index: bool, safetensors: bool)
```

_No documentation._

**Attributes**

- `file_name` (str)
- `is_index` (bool)
- `safetensors` (bool)

---

### `save_checkpoint` {#forgather-ml-sharded_checkpoint-save_checkpoint}

```python
def save_checkpoint(output_dir: str, module: StateDictLike, metadata: Optional[Dict] = None, safetensors: bool = False, max_shard_size: int = 2 ** 31, debug: bool = False, include_param_sharing: bool = True, param_sharing_metadata: Optional[SharingMetadataT] = None)
```

Save a sharded checkpoint for the whole model or a raw state dict.

**Parameters**

- `output_dir` (str) — Directory to write the checkpoint files into.
- `module` (nn.Module or Dict[str, Tensor]) — An nn.Module or a raw state dictionary to checkpoint.
- `metadata` (dict or None) — Additional metadata to embed in the shard index.
- `safetensors` (bool) — Save in safetensors format when True, PyTorch otherwise.
- `max_shard_size` (int) — Maximum bytes per shard file.
- `debug` (bool) — Enable debug-level logging of individual weights.
- `include_param_sharing` (bool) — If True and module is an nn.Module, detect and
include buffer sharing metadata automatically.
- `param_sharing_metadata` (list of list of str or None) — Explicit sharing metadata. When provided, skips
auto-detection even if module is an nn.Module.

---

### `load_checkpoint` {#forgather-ml-sharded_checkpoint-load_checkpoint}

```python
def load_checkpoint(model_dir: str, module: Optional[Module] = None, device: str = 'cpu', strict: bool = True, assign: bool = False, keys: Optional[Set[str]] = None)
```

Automatically detects checkpoint type and loads accordingly.

This should work for both sharded and normal checkpoint with either PyTorch
or safetensor formats.

**Parameters**

- `model_dir` (str) — Directory containing checkpoint files.
- `module` (nn.Module or None) — An nn.Module to load weights into. If None, returns a raw
Dict[str, Tensor] instead of loading into a module.
- `device` (str) — Device to map tensors to when loading.
- `strict` (bool) — Whether to require all module keys to be present in the checkpoint.
- `assign` (bool) — If True, assign loaded tensors rather than copying data.
- `keys` (set of str or None) — When module is None, optionally restrict which keys to load.
Ignored when module is provided.

> **Note**
>
> See `torch.nn.Module.load_state_dict
> <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict>`_
> for the semantics of the ``strict`` and ``assign`` flags.

> When the checkpoint is torchao-quantized, this function installs the
> matching quantized linear modules on ``module`` before
> ``load_state_dict`` runs and forces ``assign=True`` (``Tensor.copy_``
> does not handle quantized-to-quantized copies). In that branch the
> ``device`` argument is silently overridden to the module's existing
> device, so the ``assign``-rebound tensors don't migrate the model
> off the caller's compute device. Tied weights are restored
> post-load by the trainer's ``retie_parameters()`` step; eval /
> inference paths that don't re-tie still produce correct outputs
> because quantized inference doesn't grad-update tied tensors.

---

### `find_latest_checkpoint` {#forgather-ml-sharded_checkpoint-find_latest_checkpoint}

```python
def find_latest_checkpoint(model_dir: str)
```

Find the most recent valid checkpoint in the checkpoints directory.

Uses checkpoint_manifest.json timestamp when available, falling back to
filesystem modification time for legacy checkpoints.

---

### `next_checkpoint_path` {#forgather-ml-sharded_checkpoint-next_checkpoint_path}

```python
def next_checkpoint_path(model_dir: str, checkpoint_id: int | str)
```

Get path to save next checkpoint, given model directory and global_step

---

### `validate_checkpoint` {#forgather-ml-sharded_checkpoint-validate_checkpoint}

```python
def validate_checkpoint(checkpoint_path: str)
```

Validate that a checkpoint directory contains the necessary files.

---

## Protocols

Protocols that trainer components implement to participate in the checkpoint system.

### `CheckpointInterface` {#forgather-ml-trainer-trainer_types-checkpointinterface}

`forgather.ml.trainer.trainer_types.CheckpointInterface`

Protocol for checkpoint management.

Defines interface for saving/loading complete training state (model, optimizer,
scheduler, dataset position, RNG state, etc.) and standalone model weights.

Implementations:
- CheckpointManager: Standard implementation in src/forgather/ml/trainer/checkpoint_manager.py

Key responsibilities:
- Save complete training checkpoints with versioning and limits
- Load checkpoints for resuming training
- Track best checkpoint (for load_best_model_at_end)
- Save standalone model weights (HF Trainer compatibility)

**Methods**

#### `save_checkpoint` {#forgather-ml-trainer-trainer_types-checkpointinterface-save_checkpoint}

```python
def save_checkpoint(checkpoint_path: str | None = None, checkpoint_id: str | None = None)
```

Save complete training checkpoint.

Args:
    checkpoint_path: Specific path for checkpoint, or None for auto-generated
    checkpoint_id: Identifier for checkpoint (e.g., global_step), used if path is None

Returns:
    Path to saved checkpoint directory

#### `load_checkpoint` {#forgather-ml-trainer-trainer_types-checkpointinterface-load_checkpoint}

```python
def load_checkpoint(checkpoint_path: str | None = None)
```

Load checkpoint to resume training.

Args:
    checkpoint_path: Path to checkpoint, or None to load latest checkpoint

#### `save_model` {#forgather-ml-trainer-trainer_types-checkpointinterface-save_model}

```python
def save_model(output_dir: str | os.PathLike | None = None, overwrite_output_dir: bool = False)
```

Save only model weights (not full training state).

Args:
    output_dir: Directory to save model, or None for default
    overwrite_output_dir: Whether to overwrite existing model

#### `set_best_checkpoint` {#forgather-ml-trainer-trainer_types-checkpointinterface-set_best_checkpoint}

```python
def set_best_checkpoint(best_checkpoint: str)
```

Mark a checkpoint as the best model.

Args:
    best_checkpoint: Path to checkpoint to mark as best

#### `resolve_checkpoint_path` {#forgather-ml-trainer-trainer_types-checkpointinterface-resolve_checkpoint_path}

```python
def resolve_checkpoint_path(checkpoint_path: str | None)
```

Resolve checkpoint path (e.g., find latest if path is None).

Args:
    checkpoint_path: Explicit path or None for auto-resolution

Returns:
    Resolved checkpoint path or None if not found

---

### `StatefulProvider` {#forgather-ml-trainer-trainer_types-statefulprovider}

`forgather.ml.trainer.trainer_types.StatefulProvider`

Protocol for providing stateful objects for checkpointing.

Used by checkpoint managers to collect all components that need to be
saved/restored during checkpointing (optimizer, scheduler, dataset, etc.).

The protocol uses StateComponents which declare explicit sharing patterns
(GLOBAL, PER_RANK, REPLICATED, etc.) to enable automatic distributed
checkpoint coordination for hybrid parallelism strategies.

All implementations must provide:
- get_state_components(): Returns list of StateComponents with sharing patterns
- get_process_groups(): Returns named process groups (only if using PER_GROUP pattern)

**Methods**

#### `get_state_components` {#forgather-ml-trainer-trainer_types-statefulprovider-get_state_components}

```python
def get_state_components()
```

Get state components with explicit sharing patterns for distributed checkpointing.

This is the new preferred API for checkpoint coordination. Each StateComponent
declares its sharing pattern (GLOBAL, PER_RANK, REPLICATED, etc.), enabling
automatic distributed checkpoint coordination without manual rank checks.

Returns:
    List of StateComponent objects describing all checkpointable state

Example implementation for single-GPU trainer:
    def get_state_components(self):
        from forgather.ml.trainer.checkpoint_types import StateComponent, SharingPattern

        return [
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=self._get_dataset_sharing_pattern(),
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

Example for DDP trainer:
    def get_state_components(self):
        return [
            StateComponent(
                key="model",
                stateful=self.unwrapped_model(),
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=True,  # Verify DDP synchronization
            ),
            # ... other components
        ]

See: docs/checkpointing/migration_guide.md for full migration guide

#### `get_process_groups` {#forgather-ml-trainer-trainer_types-statefulprovider-get_process_groups}

```python
def get_process_groups()
```

Get named process groups for PER_GROUP sharing pattern.

Returns dictionary mapping group names to ProcessGroup objects.
Only needed if using PER_GROUP sharing pattern in state components.

Returns:
    Dictionary mapping process group names to ProcessGroup objects
    (e.g., {"dp_group": dp_pg, "pp_group": pp_pg})

Example:
    def get_process_groups(self):
        return {
            "dp_group": self.dp_process_group,
            "pp_group": self.pp_process_group,
        }
