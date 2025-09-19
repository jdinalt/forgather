import json
import os
import time
import glob
import shutil
from pprint import pp
import logging
from typing import Dict, List, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
import gc

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module

from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
This implements loading and saving sharded checkpoints

This is intended to be compatible with the Huggingface model conventions for
saving and loading model checkpoints, where the model weights are split
across multiple files (shards), as to limit the maximum memory requirements.

https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#sharded-checkpoints

Huggingface index files (json) are structured like this:

"metadata": {
    ...
  },
  "weight_map": {
    "lm_head.weight": "model-00002-of-00002.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    ...
}

Where the fully qualified parameter names are keys and the values contain the file name
with that parameter. By splitting the data across multiple files, the peak memory required
for loading the weights is reduced significantly.

This implementation has a somewhat different use-case, which is sharding model weights
across multiple processes / hosts, where it simplifies saving and loading. If the shard files
for each host are unique to that host, we can load and save in parallel and we sidestep the
complexity of multiple nodes saving data to the same file. By using an established checkpoint
format, the checkpoint is compatile with other libraries and tools.


Basic loading scenario:

    model = model_ctor().to(device)
    load_checkpoint(checkpoint_dir, model, device)

The primary use-case looks something like this:

    # Construct model with fake weight tensors, using the "meta" device.
    with torch.device("meta"):
        model = model_ctor()

    # Optionally, shard the model.
    shards = example_model_shard_function(model, rank=rank)

    # The shard for this rank
    model_shard = shards[rank] # 'shards' is a list of nn.Modules

    # Optional, assuming you will want to save a new checkpoint later.
    shard_index = make_shard_index([m.state_dict() for m in shards], metadata=dict(dtype=dtype))

    # Optionally, change weight dtype
    model_shard.to(dtype=dtype)

    # Move model to target device and allocate uninitialized memory for weights.
    model_shard.to_empty(device=device)

    # Load weights from checkpoint into model on device.
    load_checkpoint(checkpoint_dir, model_shard, device)


Basic sharded checkpoint creation:

    # Resulting checkpoint should be loadable with HF ".from_pretrained()"
    save_checkpoint(output_dir, model)

Create checkpoint from model shards:

    # Save one copy of the index per host.
    # Alternatively, if the hosts have a shared output directory, only save on rank == 0
    if local_rank == 0:
        # See sharded loading example for 'shard_index' creation.
        save_shard_index(shard_index, output_dir, index_file_name(use_safetensors), safetensors=use_safetensors)

    # Save only our shard
    save_sharded_checkpoint(output_dir, weight_map, model_shard, safetensors=use_safetensors)

"""

# File naming conventions used by Huggingface APIs
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"

ShardIndex = Dict[str, Dict[str, str]]


def id_to_fqn(module: nn.Module) -> Dict[int, Set[str]]:
    """
    Returns a dictionary mapping parameter ids to the set of FQNs which
    share the same storage (tied).
    """
    mapping = defaultdict(set)
    for name, p in module.state_dict(keep_vars=True).items():
        mapping[id(p)].add(name)
    for name, b in module.named_buffers(remove_duplicate=False):
        mapping[id(b)].add(name)
    return mapping


def get_all_fqns(module: nn.Module) -> Set[str]:
    """
    Get set of all FQN's in module
    The keys of the state dictionary exclude non-persistent buffers, this returns
    those in the set as well.
    """
    all_params = set(
        (name for name, _ in module.named_parameters(remove_duplicate=False))
    )
    all_buffers = set(
        (name for name, _ in module.named_buffers(remove_duplicate=False))
    )
    return all_params | all_buffers


def make_cannonical_names(
    fqns: Set[str], sharing_metadata: List[List[str]]
) -> Dict[str, List[str]]:
    """
    Given a set of FQN's in a module and parameter sharing meta-data,
    return a mapping of cannoical names (in fqns) to aliased names (other FQNs,
    which share the same data representation.
    """
    cnames = {}
    for parameter_names in sharing_metadata:
        parameter_names = list(set(parameter_names).intersection(fqns))
        if not len(parameter_names):
            continue
        cnames[parameter_names[0]] = parameter_names[1:]
    return cnames


def map_cannonical_names(cnames: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Invert the output from make_cannonical_names(), such that each key in an aliased FQN and
    """
    cname_map = {}
    for cname, fqns in cnames.items():
        for fqn in fqns:
            cname_map[fqn] = cname
    return cname_map


def create_sharing_metadata(model: nn.Module) -> List[List[str]]:
    """
    Create metadata about buffer sharing that can be stored in checkpoint index.
    Returns list of lists, where the sublist is a set of parameters which are tied.
    """
    # Convert to a format suitable for JSON serialization
    sharing_metadata = []
    for parameter_names in filter(lambda x: len(x) > 1, id_to_fqn(model).values()):
        sharing_metadata.append(list(parameter_names))
    return sharing_metadata


@torch.no_grad()
def retie_parameters(module, sharing_metadata: List[List[str]]) -> None:
    """
    Re-tie buffers across multiple modules based on sharing metadata.

    This restores buffer sharing after loading from sharded checkpoints
    where sharing was broken during the load process.

    Args:
        modules: List of modules (e.g., pipeline stages) to re-tie buffers across
        sharing_metadata: Buffer sharing metadata from checkpoint index
    """
    # Flatten shared parameters list and convert to set
    all_shared = set([item for sublist in sharing_metadata for item in sublist])

    # Get the set of all FQNs in the module
    all_fqn = get_all_fqns(module)

    # The intersection of these sets is the set of parameters we need to tie for this module.
    all_tied = get_all_fqns(module).intersection(all_shared)

    # If this module does not have tied parameters, return early
    if not len(all_tied):
        return

    # Convert the lists of shared FQNs into a mapping of
    # cannonical names and aliases. The choice of cannonical name is arbirary,
    # with the only requirement that it be a a name in 'module'
    cnames = make_cannonical_names(all_fqn, sharing_metadata)
    logger.debug(f"rank{os.getenv('RANK')} CNAMES: {cnames}")

    # Create a mapping of cname FQNs to tensors
    cname_tensors = {}
    for cname in cnames.keys():
        fqn_atoms = cname.split(".")
        # Navigate to cname
        sub_module = module
        for atom in fqn_atoms:
            sub_module = getattr(sub_module, atom)
        cname_tensors[cname] = sub_module

    # Create inverse mapping of aliases to cnames
    cname_map = map_cannonical_names(cnames)

    # Assign tensors from cname_tensors to modules in cname_map
    for aliased_name, cannonical_name in cname_map.items():
        logger.debug(
            f"rank{os.getenv('RANK')} Retie {aliased_name} to {cannonical_name}"
        )
        # Get the cannonical tensor
        canonical_tensor = cname_tensors[cannonical_name]

        # Navigate to parent module
        fqn_atoms = aliased_name.split(".")
        sub_module = module
        for atom in fqn_atoms[:-1]:
            sub_module = getattr(sub_module, atom)

        setattr(sub_module, fqn_atoms[-1], canonical_tensor)


def index_file_name(safetensors: bool) -> str:
    """
    Get the canonical name for the weight index file, which depends on if
    we are using safetensors.
    """
    if safetensors:
        return SAFE_WEIGHTS_INDEX_NAME
    else:
        return WEIGHTS_INDEX_NAME


def make_shard_index(
    state_dictionaries: List[Dict[str, Tensor]],
    metadata: Optional[Dict] = None,
    safetensors: bool = False,
    max_shard_size: int = 2**32,
    param_sharing_metadata: Optional[List[List[str]]] = None,
) -> ShardIndex:
    """
    Given a list of state dictionaries, construct a shard index

    This will ensure that no two dictionaries share the same shard files, resulting
    in a minimum of len(state_dictionaries) total shards.

    If a state_dict requires more than max_shard_size bytes, it will be split into
    multiple shards -- again, not shared with any other dictionary.

    state_dictionaries: A list of state dictionaries to map to shard files.
        The state dictionaries MAY be on the "meta" device. If so, size is still
        correctly computed.

    metadata: Any additonal meta data. Huggingface's modelling_utils.py seems
        to use "dtype" from the metadata, when loading a model.

    safetensors: This controls the naming convention for the shards, which matches
        that used by HF libraries.

    max_shard_size: Each dictionary will be split into multiple shards when this
        total is reached.

    """
    if not metadata:
        metadata = {}

    shard_list = []
    total_size = 0

    # Assign parameters to shards
    for state_dict in state_dictionaries:
        shard_bytes = 0
        weights = []

        # Partition each dictionary into unique shards
        for key, p in state_dict.items():
            weights.append(key)
            nbytes = p.untyped_storage().nbytes()
            shard_bytes += nbytes
            total_size += nbytes

            # Create new shard, when limit has been reached.
            if shard_bytes > max_shard_size:
                shard_list.append(weights)
                shard_bytes = 0
                weights = []

        # If we have a partial shard, add it to the list
        if len(weights):
            shard_list.append(weights)

    # Construct weight map from shard list
    weight_map = {}

    for shard_number, shard_weights in enumerate(shard_list):
        if safetensors:
            shard_file_name = (
                f"model-{shard_number + 1:05}-of-{len(shard_list):05}.safetensors"
            )
        else:
            shard_file_name = (
                f"pytorch_model-{shard_number + 1:05}-of-{len(shard_list):05}.bin"
            )
        for weight_name in shard_weights:
            weight_map[weight_name] = shard_file_name

    metadata["total_size"] = total_size

    # Add buffer sharing metadata if provided
    if param_sharing_metadata:
        metadata["param_sharing"] = param_sharing_metadata

    return {"metadata": metadata, "weight_map": weight_map}


def _intersect_weight_map(weight_map, state_dict):
    """
    Computes the intersection of a weight map and a module state dictionary
    """
    return set(weight_map.keys()).intersection(set(state_dict.keys()))


def _make_shard_dictionaries(
    weight_map: Dict[str, str], state_dict: Dict[str, Tensor]
) -> Dict[str, Dict[str, Tensor]]:
    """
    Given a weight_map (from an index file) and a module, create
    a map of file_name -> state_dict which only includes the weights
    actually in 'module'

    returns Dict[file_name: str, Dict[weight_name: str, weight: Tensor]]
    """

    intersection = _intersect_weight_map(weight_map, state_dict)
    file_map = {}
    for weight_name in intersection:
        file_name = weight_map[weight_name]
        weight = state_dict[weight_name]

        if file_name not in file_map:
            file_map[file_name] = {}
        file_map[file_name][weight_name] = weight
    return file_map


def save_checkpoint(
    output_dir: str,
    module: Module,
    metadata: Optional[Dict] = None,
    safetensors: bool = False,
    max_shard_size: int = 2**31,
    debug: bool = False,
    include_param_sharing: bool = True,
) -> None:
    """
    Save a sharded checkpoint for the whole model.

    Args:
        include_param_sharing: If True, detect and include buffer sharing metadata
    """
    # Detect buffer sharing if requested
    param_sharing_metadata = None
    if include_param_sharing:
        param_sharing_metadata = create_sharing_metadata(module)
        if param_sharing_metadata:
            logger.debug(f"Detected {len(param_sharing_metadata)} shared buffer groups")

    shard_index = make_shard_index(
        [module.state_dict()],
        metadata=metadata,
        safetensors=safetensors,
        max_shard_size=max_shard_size,
        param_sharing_metadata=param_sharing_metadata,
    )
    if safetensors:
        index_name = SAFE_WEIGHTS_INDEX_NAME
    else:
        index_name = WEIGHTS_INDEX_NAME
    save_shard_index(shard_index, output_dir, index_name)
    save_sharded_checkpoint(
        output_dir,
        shard_index,
        module,
        safetensors=safetensors,
        debug=debug,
    )


def save_sharded_checkpoint(
    output_dir: str,
    shard_index: ShardIndex,
    module: Module,
    safetensors: bool = False,
    debug: bool = False,
) -> None:
    """
    Save sharded checkpoint only for tensors in 'module'

    This is useful for saving sharded models, where 'weight_map' is the map for the complete
    model, and 'module' may only we a sub-set of those weights -- this is the use-case
    this was written for, but it can be used to save complete model as well.

    See "save_checkpoint" if you only wish to save the complete model.
    """
    weight_map = shard_index["weight_map"]

    os.makedirs(output_dir, exist_ok=True)
    shard_files = _make_shard_dictionaries(weight_map, module.state_dict())
    for shard_file_name, state_dict in shard_files.items():
        logger.info(f"Writing File: {shard_file_name}")
        total_size = 0
        for weight_name, p in state_dict.items():
            size = p.untyped_storage().nbytes()
            total_size += size
            logger.debug(f"{weight_name} : {p.shape=}, {p.dtype=}, {size=}")
        shard_file_path = os.path.join(output_dir, shard_file_name)
        if safetensors:
            safetensors_save(state_dict, shard_file_path)
        else:
            torch.save(state_dict, shard_file_path, _use_new_zipfile_serialization=True)
        logger.info(f"Wrote: {total_size // (1024 * 1024)} MiB")


def validate_output_dir(output_dir: str, overwrite: bool = False) -> None:
    """
    Check if a checkpoint already exists in output_dir. If so, raise
    an exception, if overwrite is False, otherwise warn.
    """
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise Exception(
            f"Something other than a directory already exists at the output path! {output_dir}"
        )
    checkpoint_meta = get_checkpoint_metadata(output_dir)
    if checkpoint_meta:
        if not overwrite:
            raise Exception(
                f"Checkpoint exists '{output_dir}' exists and 'overwrite' is False"
            )
        else:
            logger.warning(
                f"Checkpoint exists in '{output_dir}' and model may be overwritten!"
            )


def save_shard_index(
    shard_index: ShardIndex,
    output_dir: str,
    index_name: str,
) -> None:
    """
    Write a shard index file in json format
    """
    os.makedirs(output_dir, exist_ok=True)
    index_file_path = os.path.join(output_dir, index_name)
    with open(index_file_path, "w") as f:
        json.dump(shard_index, f, indent=4, ensure_ascii=True)


def load_shard_index(
    output_dir: str,
    index_name: str,
) -> ShardIndex:
    """
    Load a shard index file, returning the weight_map
    """
    index_file_path = os.path.join(output_dir, index_name)
    with open(index_file_path, "r") as f:
        shard_index = json.load(f)
    return shard_index


def load_checkpoint(
    model_dir: str,
    module: Module,
    device: str,
    strict: bool = True,
    assign: bool = False,
) -> None:
    """
    Automatically detects checkpoint type and loads accordingly.

    This should work for both sharded and normal checkpoint with either PyTorch
    or safetensor formats.

    See:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
    """
    checkpoint_meta = get_checkpoint_metadata(model_dir)

    if not checkpoint_meta:
        raise FileNotFoundError

    if checkpoint_meta.is_index:
        shard_index = load_shard_index(model_dir, checkpoint_meta.file_name)
        load_sharded_checkpoint(
            model_dir,
            shard_index,
            module,
            device=device,
            safetensors=checkpoint_meta.safetensors,
            strict=strict,
            assign=assign,
        )
        return

    state_dict_path = os.path.join(model_dir, checkpoint_meta.file_name)
    if checkpoint_meta.safetensors:
        state_dict = safetensors_load(
            state_dict_path, device=torch.device(device).index
        )
    else:
        state_dict = torch.load(
            state_dict_path, map_location=device, weights_only=True, mmap=True
        )
    # TODO: Properly handle strict, in this case?
    # We wish to ensure that all model weights were loaded, but ignore any other weights, like we do in load_sharded_checkpoint()
    module.load_state_dict(state_dict, strict=strict, assign=assign)


def load_sharded_checkpoint(
    model_dir: str,
    shard_index: ShardIndex,
    module: Module,
    device: str,
    safetensors: bool = False,
    strict: bool = True,
    assign: bool = False,
    debug: bool = False,
) -> Set[str]:
    """
    Load a sharded checkpoint

    The weight_map may be a super-set of the weights in weight_map, which is useful for
    loading only the relevant weights for a sharded model.
    """
    weight_map = shard_index["weight_map"]

    # Get intersection of weights in map and our state dictionary
    intersection = _intersect_weight_map(weight_map, module.state_dict())

    all_module_keys = set(module.state_dict().keys())
    missing_keys = all_module_keys - intersection
    if strict and len(missing_keys):
        raise Exception(
            f"Index file does not contain mappings for the following keys {missing_keys} "
        )

    shard_files = set()

    # Get uniqe fille names from intersection. All of this moduele's weights
    # should be in this sub-set.
    for weight_name in intersection:
        file_name = weight_map[weight_name]
        shard_files.add(file_name)

    # Load shards into module, one at a time
    for shard_file_name in shard_files:
        shard_file_path = os.path.join(model_dir, shard_file_name)
        if safetensors:
            state_dict = safetensors_load(
                shard_file_path, device=torch.device(device).index
            )
        else:
            state_dict = torch.load(
                shard_file_path, map_location=device, weights_only=True, mmap=True
            )

        # Keep track of which keys we have yet to load
        all_module_keys = all_module_keys - set(state_dict.keys())

        logger.debug(f"loading state_dict in '{shard_file_name}'")

        # Load state dictionary into model.
        module.load_state_dict(state_dict, strict=False, assign=assign)
        for weight_name, p in module.state_dict(keep_vars=True).items():
            logger.debug(f"{weight_name} : {p.shape=}, {p.dtype=}, {p.requires_grad=}")
        # Evict shard from memory
        state_dict = None
        gc.collect()

    if len(all_module_keys):
        msg = f"The following keys were not found in the shards {all_module_keys}"
        if strict:
            raise Exception(msg)
        else:
            logger.warning(msg)

    return all_module_keys


@dataclass
class CheckpointMeta:
    # The name of the index, if one exists, else, weights file
    file_name: str

    # The file name is an index file
    is_index: bool

    # The weights file uses safetensors, else PyTorch
    safetensors: bool


def get_checkpoint_metadata(
    path: str,
) -> CheckpointMeta | None:
    """
    Returns checkpoint metadata for ", if checkpoint exists, else None
    """
    torch_index_path = os.path.join(path, WEIGHTS_INDEX_NAME)
    safetensors_index_path = os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)
    torch_weights_path = os.path.join(path, WEIGHTS_NAME)
    safetensors_weights_path = os.path.join(path, SAFE_WEIGHTS_NAME)

    if os.path.exists(torch_index_path):
        return CheckpointMeta(WEIGHTS_INDEX_NAME, True, False)
    elif os.path.exists(safetensors_index_path):
        return CheckpointMeta(SAFE_WEIGHTS_INDEX_NAME, True, True)
    elif os.path.exists(torch_weights_path):
        return CheckpointMeta(WEIGHTS_NAME, False, False)
    elif os.path.exists(safetensors_weights_path):
        return CheckpointMeta(SAFE_WEIGHTS_NAME, False, True)
    else:
        return None


def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate that a checkpoint directory contains the necessary files."""
    if not os.path.isdir(checkpoint_path):
        return False

    # Check for at least one of the expected model files
    expected_model_files = [
        WEIGHTS_NAME,
        SAFE_WEIGHTS_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
    ]

    has_checkpoint = any(
        os.path.exists(os.path.join(checkpoint_path, filename))
        for filename in expected_model_files
    )

    if not has_checkpoint:
        return False

    return True


def find_latest_checkpoint(model_dir: str) -> str | None:
    """Find the most recent valid checkpoint in the checkpoints directory based on modification time."""
    checkpoints_dir = os.path.join(model_dir, "checkpoints")

    # If checkpoints directory does not exist, check the model directory
    if not os.path.exists(checkpoints_dir):
        logger.info(
            "No checkpoint directory found. Defaulting to main model directory."
        )
        if validate_checkpoint(model_dir):
            return model_dir
        else:
            return None

    checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    if not checkpoints:
        return None

    # Filter to only valid checkpoints and sort by modification time
    valid_checkpoints = [cp for cp in checkpoints if validate_checkpoint(cp)]

    if not valid_checkpoints:
        logger.warning("No valid checkpoints found in checkpoint directory")
        return None

    try:
        latest = max(valid_checkpoints, key=lambda path: os.path.getmtime(path))
        step_num = (
            os.path.basename(latest).split("-")[1]
            if "-" in os.path.basename(latest)
            else "unknown"
        )
        mtime = os.path.getmtime(latest)
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        logger.debug(
            f"Found latest valid checkpoint: {latest} (step {step_num}, modified {mtime_str})"
        )
        return latest
    except (OSError, IndexError) as e:
        logger.warning(f"Error finding latest checkpoint: {e}")
        return None


def next_checkpoint_path(model_dir: str, checkpoint_id: int | str) -> str:
    """Get path to save next checkpoint, given model directory and global_step"""
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint-{str(checkpoint_id)}")
    return checkpoint_path


def save_checkpoint_metrics(checkpoint_path: str, metrics: Dict[str, float]) -> None:
    """Save metrics to checkpoint directory in JSON format."""
    os.makedirs(checkpoint_path, exist_ok=True)
    metrics_path = os.path.join(checkpoint_path, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.debug(f"Saved metrics to {metrics_path}")


def load_checkpoint_metrics(checkpoint_path: str) -> Dict[str, float] | None:
    """Load metrics from checkpoint directory."""
    metrics_path = os.path.join(checkpoint_path, "eval_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def maybe_delete_oldest_checkpoint(
    model_dir: str, max_checkpoints: int, best_checkpoint: str | None = None
) -> None:
    """Delete oldest checkpoints, preserving the best checkpoint if specified."""
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        logger.debug(
            f"No checkpoints directory found at {checkpoints_dir}, skipping deletion"
        )
        return

    checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    if len(checkpoints) <= max_checkpoints:
        return

    # Never delete the best checkpoint if specified
    checkpoints_to_consider = checkpoints
    if best_checkpoint:
        checkpoints_to_consider = [cp for cp in checkpoints if cp != best_checkpoint]

    # Calculate how many to delete
    num_to_delete = len(checkpoints) - max_checkpoints
    # Ensure we don't delete more than available in checkpoints_to_consider
    if best_checkpoint and best_checkpoint in checkpoints:
        num_to_delete = min(num_to_delete, len(checkpoints_to_consider))

    if num_to_delete > 0:
        # Sort by modification time and delete the oldest
        checkpoints_to_consider.sort(key=lambda path: os.path.getmtime(path))
        for checkpoint_path in checkpoints_to_consider[:num_to_delete]:
            logger.info(f"Deleting checkpoint at {checkpoint_path}")
            shutil.rmtree(checkpoint_path)


def create_pretrained_symlinks(
    model_dir: str,
    force_overwrite: bool = False,
    dry_run: bool = False
) -> List[str]:
    """
    Create symlinks in model root directory pointing to latest checkpoint files.

    This enables Hugging Face .from_pretrained() to work with checkpointed models
    by making the latest checkpoint weights accessible from the model root directory.

    Args:
        model_dir: Path to the model directory containing checkpoints subdirectory
        force_overwrite: If True, overwrite existing real files. If False, only
                        overwrite existing symlinks or create new symlinks.
        dry_run: If True, only log what would be done without creating symlinks

    Returns:
        List of symlink paths that were created (or would be created in dry_run mode)

    Raises:
        FileNotFoundError: If no valid checkpoints are found
        FileExistsError: If target files exist and are not symlinks (when force_overwrite=False)
    """
    # Find latest checkpoint
    latest_checkpoint_dir = find_latest_checkpoint(model_dir)
    if not latest_checkpoint_dir:
        raise FileNotFoundError(f"No valid checkpoints found in {model_dir}")

    # Get checkpoint metadata to determine which files to link
    checkpoint_meta = get_checkpoint_metadata(latest_checkpoint_dir)
    if not checkpoint_meta:
        raise FileNotFoundError(f"Invalid checkpoint format in {latest_checkpoint_dir}")

    logger.info(f"Found latest checkpoint: {latest_checkpoint_dir}")
    logger.info(f"Checkpoint format: {'safetensors' if checkpoint_meta.safetensors else 'pytorch'}, "
                f"{'sharded' if checkpoint_meta.is_index else 'single file'}")

    symlinks_created = []
    files_to_link = []

    # Determine which files need to be symlinked
    if checkpoint_meta.is_index:
        # Sharded checkpoint - need to link index file and all shard files
        index_path = os.path.join(latest_checkpoint_dir, checkpoint_meta.file_name)
        files_to_link.append((checkpoint_meta.file_name, index_path))

        # Load index to find all shard files
        try:
            shard_index = load_shard_index(latest_checkpoint_dir, checkpoint_meta.file_name)
            weight_map = shard_index["weight_map"]

            # Get unique shard file names
            shard_files = set(weight_map.values())
            for shard_file in shard_files:
                shard_path = os.path.join(latest_checkpoint_dir, shard_file)
                if os.path.exists(shard_path):
                    files_to_link.append((shard_file, shard_path))
                else:
                    logger.warning(f"Shard file referenced in index but not found: {shard_path}")

        except Exception as e:
            logger.error(f"Failed to read shard index: {e}")
            raise
    else:
        # Single file checkpoint
        weight_file_path = os.path.join(latest_checkpoint_dir, checkpoint_meta.file_name)
        files_to_link.append((checkpoint_meta.file_name, weight_file_path))

    # Create symlinks
    for link_name, target_path in files_to_link:
        link_path = os.path.join(model_dir, link_name)

        # Check if target already exists
        if os.path.exists(link_path) or os.path.islink(link_path):
            is_symlink = os.path.islink(link_path)

            if not is_symlink and not force_overwrite:
                raise FileExistsError(
                    f"Target file {link_path} exists and is not a symlink. "
                    f"Use force_overwrite=True to replace real files."
                )

            if dry_run:
                action = "would overwrite" if is_symlink else "would replace real file with"
                logger.info(f"DRY RUN: {action} symlink {link_path} -> {target_path}")
            else:
                # Remove existing file/symlink
                if is_symlink:
                    logger.info(f"Replacing existing symlink {link_path}")
                else:
                    logger.warning(f"Replacing real file {link_path} with symlink (force_overwrite=True)")
                os.unlink(link_path)

        if dry_run:
            logger.info(f"DRY RUN: would create symlink {link_path} -> {target_path}")
        else:
            # Create relative symlink to make it more portable
            rel_target = os.path.relpath(target_path, model_dir)
            os.symlink(rel_target, link_path)
            logger.info(f"Created symlink {link_path} -> {rel_target}")

        symlinks_created.append(link_path)

    if dry_run:
        logger.info(f"DRY RUN: Would create {len(symlinks_created)} symlinks")
    else:
        logger.info(f"Successfully created {len(symlinks_created)} symlinks")

    return symlinks_created
