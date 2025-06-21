import json
import os
from pprint import pp
import logging
from typing import Any, Dict, List, Tuple
from types import NoneType

import torch
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
    metadata: Dict = None,
    safetensors: bool = False,
    max_shard_size: int = 2**32,
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
    metadata: Dict = None,
    safetensors: bool = False,
    max_shard_size: int = 2**31,
    debug: bool = False,
) -> None:
    """
    Save a sharded checkpoint for the whole model.
    """
    shard_index = make_shard_index(
        [module.state_dict()],
        metadata=metadata,
        safetensors=safetensors,
        max_shard_size=max_shard_size,
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
        if debug:
            print(f"Writing File: {shard_file_name}")
            total_size = 0
            for weight_name, p in state_dict.items():
                size = p.untyped_storage().nbytes()
                total_size += size
                print(f"{weight_name} : {p.shape=}, {p.dtype=}, {size=}")
        shard_file_path = os.path.join(output_dir, shard_file_name)
        if safetensors:
            safetensors_save(state_dict, shard_file_path)
        else:
            torch.save(state_dict, shard_file_path)
        if debug:
            print(f"Wrote: {total_size} bytes")


def validate_output_dir(output_dir: str, overwrite: bool = False) -> None:
    """
    Check if a checkpoint already exists in output_dir. If so, raise
    an exception, if overwrite is False, otherwise warn.
    """
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise Exception(
            f"Something other than a directory already exists at the output path! {output_dir}"
        )

    index_file_name, is_index, is_safetensors = checkpoint_exists(output_dir)
    if index_file_name:
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
    debug: bool = False,
) -> None:
    """
    Automatically detects checkpoint type and loads accordingly.

    This should work for both sharded and normal checkpoint with either PyTorch
    or safetensor formats.

    See:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
    """
    index_or_weights_name, is_index, is_safetensors = checkpoint_exists(model_dir)
    assert index_or_weights_name, f"No model weights found in {model_dir}"
    if is_index:
        shard_index = load_shard_index(model_dir, index_or_weights_name)
        load_sharded_checkpoint(
            model_dir,
            shard_index,
            module,
            device=device,
            safetensors=is_safetensors,
            strict=strict,
            assign=assign,
            debug=debug,
        )
        return

    state_dict_path = os.path.join(model_dir, index_or_weights_name)
    if is_safetensors:

        state_dict = safetensors_load(state_dict_path, device=device)
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
) -> None:
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
            state_dict = safetensors_load(shard_file_path, device=device)
        else:
            state_dict = torch.load(
                shard_file_path, map_location=device, weights_only=True, mmap=True
            )

        # Keep track of which keys we have yet to load
        all_module_keys = all_module_keys - set(state_dict.keys())

        if debug:
            print(f"loading state_dict in '{shard_file_name}'")
            for key in state_dict.keys():
                print(key)

        # Load state dictionary into model.
        module.load_state_dict(state_dict, strict=False, assign=assign)
        state_dict = None

    if len(all_module_keys):
        msg = f"The following keys were not found in the shards {all_module_keys}"
        if strict:
            raise Exception(msg)
        else:
            logger.warning(msg)

    return all_module_keys


def checkpoint_exists(
    model_dir: str,
) -> Tuple[str | NoneType, bool | NoneType, bool | NoneType]:
    """
    Returns Tuple[index_or_weights_name: Union[str, None], is_index: Bool, is_safetensor: Bool]

    The first element is None, if no checkpoint was found, otherwise, it's the name of
    either the index file or weights file (no index).

    The second element is True is this is a sharded checkpoint (has an index)

    The third element is True if the weights are safetensors.

    If the checkpoint does not exist, all elements are 'None'
    """
    torch_index_path = os.path.join(model_dir, WEIGHTS_INDEX_NAME)
    safetensors_index_path = os.path.join(model_dir, SAFE_WEIGHTS_INDEX_NAME)
    torch_weights_path = os.path.join(model_dir, WEIGHTS_NAME)
    safetensors_weights_path = os.path.join(model_dir, SAFE_WEIGHTS_NAME)

    if os.path.exists(torch_index_path):
        return (WEIGHTS_INDEX_NAME, True, False)
    elif os.path.exists(safetensors_index_path):
        return (SAFE_WEIGHTS_INDEX_NAME, True, True)
    elif os.path.exists(torch_weights_path):
        return (WEIGHTS_NAME, False, False)
    elif os.path.exists(safetensors_weights_path):
        return (SAFE_WEIGHTS_NAME, False, True)
    else:
        return (None, None, None)
