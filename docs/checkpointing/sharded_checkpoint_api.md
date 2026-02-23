# Sharded Checkpoint API Reference

The `forgather.ml.sharded_checkpoint` module provides HuggingFace-compatible sharded checkpoint save/load functionality. Checkpoints use the same shard index format as HF `transformers`, making them interoperable with `.from_pretrained()` and other HF tooling.

## Overview

Sharded checkpoints split model weights across multiple files (shards) to limit peak memory during save and load. An index file (`model.safetensors.index.json` or `pytorch_model.bin.index.json`) maps each parameter name to its shard file.

The API supports two modes:

- **Module mode**: Pass an `nn.Module` directly (existing behavior).
- **Dict mode**: Pass a raw `Dict[str, Tensor]` state dictionary, useful when you don't have (or don't want) an `nn.Module` instance.

## Types

```python
from forgather.ml.sharded_checkpoint import StateDictLike, ShardIndex, SharingMetadataT

StateDictLike = Union[Module, Dict[str, Tensor]]
ShardIndex = Dict[str, Dict[str, str]]        # {"metadata": {...}, "weight_map": {...}}
SharingMetadataT = List[List[str]]             # Groups of tied parameter names
```

## Save API

### `save_checkpoint`

High-level save that creates both the shard index and shard files.

```python
def save_checkpoint(
    output_dir: str,
    module: StateDictLike,
    metadata: Optional[Dict] = None,
    safetensors: bool = False,
    max_shard_size: int = 2**31,
    debug: bool = False,
    include_param_sharing: bool = True,
    param_sharing_metadata: Optional[SharingMetadataT] = None,
) -> None
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `output_dir` | Directory to write checkpoint files into |
| `module` | An `nn.Module` or raw `Dict[str, Tensor]` |
| `metadata` | Extra metadata included in the index file |
| `safetensors` | Use safetensors format (default: PyTorch `.bin`) |
| `max_shard_size` | Maximum bytes per shard before splitting |
| `include_param_sharing` | Auto-detect tied parameters (only when `module` is `nn.Module`) |
| `param_sharing_metadata` | Explicit sharing metadata; skips auto-detection when provided |

**Module mode:**

```python
from forgather.ml.sharded_checkpoint import save_checkpoint

model = MyModel()
save_checkpoint("output/my_model", model, safetensors=True)
```

**Dict mode:**

```python
state_dict = {"layer.weight": tensor_a, "layer.bias": tensor_b}
save_checkpoint("output/my_model", state_dict, safetensors=True)
```

### `save_sharded_checkpoint`

Low-level save that writes shard files for a subset (or all) of the weights described by a shard index. Useful for distributed training where each rank saves its own shards.

```python
def save_sharded_checkpoint(
    output_dir: str,
    shard_index: ShardIndex,
    module: StateDictLike,
    safetensors: bool = False,
    debug: bool = False,
) -> None
```

### `save_shard_index`

Write a shard index to disk.

```python
def save_shard_index(
    shard_index: ShardIndex,
    output_dir: str,
    index_name: str,
) -> None
```

### `make_shard_index`

Construct a shard index from one or more state dictionaries. Each state dict gets its own set of shard files (no cross-dict file sharing).

```python
def make_shard_index(
    state_dictionaries: List[Dict[str, Tensor]],
    metadata: Optional[Dict] = None,
    safetensors: bool = False,
    max_shard_size: int = 2**32,
    param_sharing_metadata: Optional[List[List[str]]] = None,
) -> ShardIndex
```

## Load API

### `load_checkpoint`

High-level load that auto-detects checkpoint format and loads accordingly.

```python
# Module mode (existing behavior)
def load_checkpoint(model_dir, module: Module, device, ...) -> None

# Dict mode
def load_checkpoint(model_dir, module: None, device, ..., keys=None) -> Dict[str, Tensor]
```

**Full signature:**

```python
def load_checkpoint(
    model_dir: str,
    module: Optional[Module] = None,
    device: str = "cpu",
    strict: bool = True,
    assign: bool = False,
    keys: Optional[Set[str]] = None,
) -> Union[None, Dict[str, Tensor]]
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `module` | `nn.Module` to load into, or `None` to return a dict |
| `device` | Target device for loaded tensors |
| `keys` | When `module=None`, restrict which keys to load |

**Module mode:**

```python
model = MyModel()
model.to_empty(device="cuda")
load_checkpoint("output/my_model", model, device="cuda")
```

**Dict mode:**

```python
state_dict = load_checkpoint("output/my_model", module=None, device="cpu")

# Load only specific keys
subset = load_checkpoint(
    "output/my_model", module=None, device="cpu",
    keys={"layer.weight", "layer.bias"}
)
```

### `load_sharded_checkpoint`

Low-level load from a sharded checkpoint using a previously loaded shard index.

```python
# Module mode: returns Set[str] of unloaded keys
def load_sharded_checkpoint(model_dir, shard_index, module: Module, ...) -> Set[str]

# Dict mode: returns loaded tensors
def load_sharded_checkpoint(model_dir, shard_index, module: None, ..., keys=None) -> Dict[str, Tensor]
```

**Full signature:**

```python
def load_sharded_checkpoint(
    model_dir: str,
    shard_index: ShardIndex,
    module: Optional[Module] = None,
    device: str = "cpu",
    safetensors: bool = False,
    strict: bool = True,
    assign: bool = False,
    debug: bool = False,
    keys: Optional[Set[str]] = None,
) -> Union[Set[str], Dict[str, Tensor]]
```

### `load_shard_index`

Load a shard index file from disk.

```python
def load_shard_index(output_dir: str, index_name: str) -> ShardIndex
```

## Parameter Sharing

### `create_sharing_metadata`

Detect tied parameters in a module and return sharing metadata.

```python
def create_sharing_metadata(model: nn.Module) -> List[List[str]]
```

Returns a list of groups, where each group is a list of parameter names that share the same underlying storage.

### `retie_parameters`

Restore parameter tying after loading from a checkpoint where sharing was broken.

```python
def retie_parameters(module, sharing_metadata: List[List[str]]) -> None
```

## Checkpoint Management Utilities

### `find_latest_checkpoint`

Find the most recent valid checkpoint by modification time.

```python
def find_latest_checkpoint(model_dir: str) -> str | None
```

### `next_checkpoint_path`

Get the path for the next checkpoint.

```python
def next_checkpoint_path(model_dir: str, checkpoint_id: int | str) -> str
```

### `validate_checkpoint`

Check whether a directory contains a valid checkpoint.

```python
def validate_checkpoint(checkpoint_path: str) -> bool
```

### `get_checkpoint_metadata`

Detect checkpoint format and return metadata.

```python
def get_checkpoint_metadata(path: str) -> CheckpointMeta | None
```

Returns a `CheckpointMeta` dataclass with fields: `file_name`, `is_index`, `safetensors`.

### `maybe_delete_oldest_checkpoint`

Delete oldest checkpoints while preserving specified ones.

```python
def maybe_delete_oldest_checkpoint(
    model_dir: str,
    max_checkpoints: int,
    best_checkpoint: str | None = None,
    preserved_checkpoints: List[str] | None = None,
) -> None
```

### `save_checkpoint_metrics` / `load_checkpoint_metrics`

Save and load evaluation metrics alongside a checkpoint.

```python
def save_checkpoint_metrics(checkpoint_path: str, metrics: Dict[str, float]) -> None
def load_checkpoint_metrics(checkpoint_path: str) -> Dict[str, float] | None
```

### `create_pretrained_symlinks`

Create symlinks from the model root to the latest checkpoint, enabling HF `.from_pretrained()`.

```python
def create_pretrained_symlinks(
    model_dir: str,
    force_overwrite: bool = False,
    dry_run: bool = False,
) -> List[str]
```

## Examples

### Module-based save and load

```python
from forgather.ml.sharded_checkpoint import save_checkpoint, load_checkpoint

# Save
model = build_model()
save_checkpoint("output/my_model", model, safetensors=True)

# Load
model = build_model()
model.to_empty(device="cuda:0")
load_checkpoint("output/my_model", model, device="cuda:0")
```

### Dict-based save and load

```python
from forgather.ml.sharded_checkpoint import save_checkpoint, load_checkpoint

# Save raw state dict (e.g., aggregated parameters from a parameter server)
aggregated = {"layer.weight": averaged_weight, "layer.bias": averaged_bias}
save_checkpoint("output/aggregated", aggregated, safetensors=True)

# Load back as dict
state_dict = load_checkpoint("output/aggregated", module=None, device="cpu")

# Load subset of keys
subset = load_checkpoint(
    "output/aggregated", module=None, device="cpu",
    keys={"layer.weight"}
)
```

### Multi-GPU sharded checkpoint

```python
from forgather.ml.sharded_checkpoint import (
    make_shard_index, save_shard_index, save_sharded_checkpoint,
    load_shard_index, load_sharded_checkpoint, index_file_name,
)

# --- Save (each rank saves its shard) ---
# Construct index from all shards (typically on rank 0 or from meta-device model)
shard_index = make_shard_index(
    [shard.state_dict() for shard in model_shards],
    safetensors=True,
)

if local_rank == 0:
    save_shard_index(shard_index, output_dir, index_file_name(True))

# Each rank saves its own shard
save_sharded_checkpoint(output_dir, shard_index, model_shards[rank], safetensors=True)

# --- Load ---
shard_index = load_shard_index(output_dir, index_file_name(True))

# Load into module
model_shard.to_empty(device=device)
load_sharded_checkpoint(output_dir, shard_index, model_shard, device=device, safetensors=True)

# Or load as dict (e.g., for parameter server aggregation)
state_dict = load_sharded_checkpoint(
    output_dir, shard_index, module=None, device="cpu", safetensors=True
)
```
