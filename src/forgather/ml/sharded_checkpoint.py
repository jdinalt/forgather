import gc
import glob
import itertools
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pprint import pp
from typing import Dict, List, Optional, Set, TypeAlias, Union, overload

import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save
from torch import Tensor, nn
from torch.nn import Module

from forgather.ml.distributed import prefix_logger_rank

logger = logging.getLogger(__name__)
# Show messages from every rank — this module's logs are the diagnostic
# story for "who actually wrote which file" during multi-rank /
# multi-node checkpointing, so the default rank-0-only filter would hide
# exactly the rows that matter when something goes wrong.
prefix_logger_rank(logger, show_all_ranks=True)

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

    # Record tied-parameter groups BEFORE to_empty() breaks the aliasing,
    # so they can be re-tied on the real device afterwards.
    sharing = create_sharing_metadata(model_shard)

    # Move model to target device and allocate uninitialized memory for weights.
    model_shard.to_empty(device=device)

    # Restore the parameter sharing that to_empty() severed.
    retie_parameters(model_shard, sharing)

    # Load weights from checkpoint into model on device. Loaded tensors are
    # flagged _is_hf_initialized so the step below leaves them untouched.
    load_checkpoint(checkpoint_dir, model_shard, device)

    # Recompute anything the checkpoint did NOT contain -- chiefly
    # non-persistent buffers such as RotaryEmbedding.inv_freq (never saved),
    # plus any genuinely missing parameters. Skips loaded tensors.
    initialize_missing_weights(model_shard)


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

SharingMetadataT: TypeAlias = List[List[str]]

StateDictLike = Union[Module, Dict[str, Tensor]]


def _resolve_state_dict(source: StateDictLike) -> Dict[str, Tensor]:
    """Resolve a Module or raw state dict to a state dict."""
    if isinstance(source, Module):
        return source.state_dict()
    return source


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

    Parameters
    ----------
    module : nn.Module
        Module containing parameters to re-tie.
    sharing_metadata : list of list of str
        Buffer sharing metadata from checkpoint index
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


def flag_loaded_tensors(module: nn.Module, loaded_keys) -> None:
    """Mark params/buffers filled by a checkpoint load with
    ``_is_hf_initialized = True``.

    This drives the HF v5 "initialize only the missing tensors" contract:
    after the trainer materializes a meta-constructed model and loads the
    checkpoint, a follow-up ``model.apply(model._init_weights)`` pass
    initializes only the *unflagged* tensors (e.g. non-persistent RoPE
    buffers, or params absent from a partial checkpoint) and leaves the
    loaded weights untouched. ``loaded_keys`` are the state-dict keys that
    were actually present in the checkpoint.
    """
    loaded = set(loaded_keys)
    for name, tensor in itertools.chain(
        module.named_parameters(), module.named_buffers()
    ):
        if name in loaded:
            # Attribute set on the underlying tensor; survives the
            # load (copy_ keeps the Parameter object; assign= rebinds but
            # we re-flag from the loaded key set either way).
            tensor._is_hf_initialized = True


@torch.no_grad()
def initialize_missing_weights(module: nn.Module) -> None:
    """Initialize the tensors a checkpoint load did NOT fill, skipping the
    ones it did (flagged ``_is_hf_initialized`` by ``flag_loaded_tensors``).

    This is the shared, HF-v5-standard "initialize only the missing
    tensors" pass used after constructing a model on meta and materializing
    it: it recomputes non-persistent derived buffers (e.g. RoPE
    ``inv_freq``) and initializes any params absent from the checkpoint,
    while leaving loaded weights untouched.

    Preferred path: ``module.apply(module._init_weights)`` — the model's own
    init, which both forgather models (``DynamicCasualLM._init_weights`` →
    flag-aware ``init_weights_by_regex`` / ``simple_weight_init``) and pure
    HF ``PreTrainedModel`` subclasses (their guarded ``_init_weights``)
    expose. Fallback for modules without an HF-style ``_init_weights`` (e.g.
    split pipeline-stage modules): ``reset_parameters()`` on any module that
    still owns a non-persistent buffer.

    Granularity caveat: the gate is per-MODULE, and ``reset_parameters`` /
    ``simple_weight_init`` are module-wide. A module that co-locates a
    *loaded* persistent parameter with an *unflagged* (not-in-checkpoint)
    tensor would have its loaded parameter re-initialized — unless the
    model's ``_init_weights`` is itself per-tensor flag-aware (HF's is;
    forgather's ``init_weights_by_regex`` is). Forgather avoids this by
    keeping derived buffers (RoPE ``inv_freq``) in dedicated buffer-only
    modules. Keep that pattern for any module on the pipeline fallback path.
    """
    init_fn = getattr(module, "_init_weights", None)
    if callable(init_fn):
        # HF-style models (forgather + pure HF): apply the model's own
        # init, but only to modules that still own an unflagged tensor.
        # Gating here (rather than trusting the model's _init_weights to be
        # flag-aware) keeps loaded weights safe even for models whose
        # bundled init code predates the _is_hf_initialized contract — a
        # fully-loaded module is never re-initialized, while a module with
        # an unflagged (e.g. non-persistent RoPE) tensor is recomputed.
        def _init_if_missing(m: nn.Module) -> None:
            tensors = list(m.parameters(recurse=False)) + list(m.buffers(recurse=False))
            if tensors and not all(
                getattr(t, "_is_hf_initialized", False) for t in tensors
            ):
                init_fn(m)

        module.apply(_init_if_missing)
        return
    # Fallback for modules without an HF-style _init_weights (e.g. split
    # pipeline-stage modules). Reset only modules that own a non-persistent
    # buffer — those are excluded from the state_dict, so they're never in a
    # checkpoint and always need recomputation (e.g. RoPE inv_freq). This is
    # safe without relying on _is_hf_initialized flagging: it never touches
    # ordinary (persistent, loaded) parameters.
    for m in module.modules():
        if getattr(m, "_non_persistent_buffers_set", None) and hasattr(
            m, "reset_parameters"
        ):
            m.reset_parameters()


def _tied_aliases_in_module(module: nn.Module, checkpoint_keys: Set[str]) -> Set[str]:
    """Return module FQNs whose absence from the checkpoint is explained by weight tying.

    Safetensors cannot represent shared storage, so HF's save deduplicates
    each tied group down to a single canonical key (e.g. only
    ``model.embed_tokens.weight`` lands on disk; the tied ``lm_head.weight``
    is omitted). At load time the in-memory module's ``state_dict`` still
    lists both names — they alias the same storage — so a strict missing-key
    check would falsely reject these checkpoints. The returned aliases are
    safe to ignore for missing-key purposes: ``load_state_dict``'s in-place
    ``copy_`` of the canonical key updates the shared storage in one shot,
    and the caller's ``tie_weights()`` step restores sharing if anything
    (e.g. ``assign=True``) broke it.

    Pipeline-parallel caveat: ``create_sharing_metadata`` groups by
    ``id()`` within the passed ``module`` only. When the caller iterates
    pipeline stage sub-modules separately (as ``checkpoint_manager`` does),
    a tied group whose two sides live on different stages is invisible to
    each per-stage call — the missing side would still be reported as a
    genuine missing key. This isn't reachable from current workflows
    (Forgather-native pipeline saves preserve both names in the
    checkpoint), but if an HF-deduped checkpoint is ever loaded into a
    pipeline-split model, consult ``shard_index["metadata"]["param_sharing"]``
    (already populated by ``make_shard_index``) for the global view.
    """
    sharing_metadata = create_sharing_metadata(module)
    aliases: Set[str] = set()
    for group in sharing_metadata:
        if any(k in checkpoint_keys for k in group):
            aliases.update(k for k in group if k not in checkpoint_keys)
    return aliases


def _synthesize_tied_aliases(
    module: nn.Module,
    state_dict: Dict[str, Tensor],
    sharing_metadata: Optional[List[List[str]]] = None,
) -> None:
    """In-place: add missing tied aliases to ``state_dict`` pointing at the canonical tensor.

    Mirrors HF safetensors load semantics: when a tied group has one
    canonical member present in the checkpoint and other members missing,
    aliasing the missing names to the same tensor lets a downstream
    ``module.load_state_dict(strict=True)`` succeed.

    With ``assign=True`` the resulting Parameters wrap the same underlying
    storage (data_ptr matches) but are no longer ``is``-identical; callers
    that care about identity (e.g. before re-saving) must invoke
    ``module.tie_weights()`` afterward. The built-in trainer's
    ``_load_model_from_checkpoint`` does this; non-trainer callers that
    use ``assign=True`` (``model_conversion/finalize.py``,
    ``tools/update_model/update.py``) should also call it before saving.

    ``sharing_metadata`` may be passed in to avoid re-walking the module
    when the caller has already computed it.
    """
    if sharing_metadata is None:
        sharing_metadata = create_sharing_metadata(module)
    if not sharing_metadata:
        return
    for group in sharing_metadata:
        present = [k for k in group if k in state_dict]
        if not present:
            continue
        tensor = state_dict[present[0]]
        for k in group:
            if k not in state_dict:
                state_dict[k] = tensor


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

    Weight order within each shard follows the input ``state_dict``'s
    iteration order (which is module-traversal order for an
    ``nn.Module.state_dict()``). Iterating ``state_dict`` directly —
    rather than the set returned by ``_intersect_weight_map`` — is
    load-bearing for DiLoCo, which keys its outer-optimizer momentum
    state by integer slot in ``_param_list``: a save-time order that
    disagrees with the load-time order silently misaligns slots and
    crashes the first ``optimizer.step()`` after restart (see #44 /
    #45 for the trace).

    returns Dict[file_name: str, Dict[weight_name: str, weight: Tensor]]
    """

    weight_map_keys = set(weight_map.keys())
    file_map = {}
    for weight_name, weight in state_dict.items():
        if weight_name not in weight_map_keys:
            continue
        file_name = weight_map[weight_name]

        if file_name not in file_map:
            file_map[file_name] = {}
        file_map[file_name][weight_name] = weight
    return file_map


def save_checkpoint(
    output_dir: str,
    module: StateDictLike,
    metadata: Optional[Dict] = None,
    safetensors: bool = False,
    max_shard_size: int = 2**31,
    debug: bool = False,
    include_param_sharing: bool = True,
    param_sharing_metadata: Optional[SharingMetadataT] = None,
) -> None:
    """
    Save a sharded checkpoint for the whole model or a raw state dict.

    Parameters
    ----------
    output_dir : str
        Directory to write the checkpoint files into.
    module : nn.Module or Dict[str, Tensor]
        An nn.Module or a raw state dictionary to checkpoint.
    metadata : dict or None, optional
        Additional metadata to embed in the shard index.
    safetensors : bool, optional
        Save in safetensors format when True, PyTorch otherwise.
    max_shard_size : int, optional
        Maximum bytes per shard file.
    debug : bool, optional
        Enable debug-level logging of individual weights.
    include_param_sharing : bool, optional
        If True and module is an nn.Module, detect and
        include buffer sharing metadata automatically.
    param_sharing_metadata : list of list of str or None, optional
        Explicit sharing metadata. When provided, skips
        auto-detection even if module is an nn.Module.
    """
    state_dict = _resolve_state_dict(module)

    # Detect buffer sharing if requested and not explicitly provided
    if (
        param_sharing_metadata is None
        and include_param_sharing
        and isinstance(module, Module)
    ):
        param_sharing_metadata = create_sharing_metadata(module)
        if param_sharing_metadata:
            logger.debug(f"Detected {len(param_sharing_metadata)} shared buffer groups")

    shard_index = make_shard_index(
        [state_dict],
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
        state_dict,
        safetensors=safetensors,
        debug=debug,
    )


def save_sharded_checkpoint(
    output_dir: str,
    shard_index: ShardIndex,
    module: StateDictLike,
    safetensors: bool = False,
    debug: bool = False,
) -> None:
    """
    Save sharded checkpoint only for tensors in 'module'

    This is useful for saving sharded models, where 'weight_map' is the map for the complete
    model, and 'module' may only we a sub-set of those weights -- this is the use-case
    this was written for, but it can be used to save complete model as well.

    Parameters
    ----------
    output_dir : str
        Directory to write the checkpoint shard files into.
    shard_index : ShardIndex
        Shard index mapping weight names to shard file names.
    module : nn.Module or Dict[str, Tensor]
        An nn.Module or a raw state dictionary; only weights present
        in both the shard index and this module are written.
    safetensors : bool, optional
        Save in safetensors format when True, PyTorch otherwise.
    debug : bool, optional
        Enable debug-level logging of individual weights.

    See "save_checkpoint" if you only wish to save the complete model.
    """
    weight_map = shard_index["weight_map"]
    state_dict = _resolve_state_dict(module)

    os.makedirs(output_dir, exist_ok=True)
    shard_files = _make_shard_dictionaries(weight_map, state_dict)
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


@overload
def load_checkpoint(
    model_dir: str,
    module: Module,
    device: str,
    strict: bool = True,
    assign: bool = False,
    keys: Optional[Set[str]] = None,
) -> None: ...


@overload
def load_checkpoint(
    model_dir: str,
    module: None,
    device: str,
    strict: bool = True,
    assign: bool = False,
    keys: Optional[Set[str]] = None,
) -> Dict[str, Tensor]: ...


def load_checkpoint(
    model_dir: str,
    module: Optional[Module] = None,
    device: str = "cpu",
    strict: bool = True,
    assign: bool = False,
    keys: Optional[Set[str]] = None,
) -> Union[None, Dict[str, Tensor]]:
    """
    Automatically detects checkpoint type and loads accordingly.

    This should work for both sharded and normal checkpoint with either PyTorch
    or safetensor formats.

    Parameters
    ----------
    model_dir : str
        Directory containing checkpoint files.
    module : nn.Module or None, optional
        An nn.Module to load weights into. If None, returns a raw
        Dict[str, Tensor] instead of loading into a module.
    device : str, optional
        Device to map tensors to when loading.
    strict : bool, optional
        Whether to require all module keys to be present in the checkpoint.
    assign : bool, optional
        If True, assign loaded tensors rather than copying data.
    keys : set of str or None, optional
        When module is None, optionally restrict which keys to load.
        Ignored when module is provided.

    Notes
    -----
    See `torch.nn.Module.load_state_dict
    <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict>`_
    for the semantics of the ``strict`` and ``assign`` flags.

    When the checkpoint is torchao-quantized, this function installs the
    matching quantized linear modules on ``module`` before
    ``load_state_dict`` runs and forces ``assign=True`` (``Tensor.copy_``
    does not handle quantized-to-quantized copies). In that branch the
    ``device`` argument is silently overridden to the module's existing
    device, so the ``assign``-rebound tensors don't migrate the model
    off the caller's compute device. Tied weights are restored
    post-load by the trainer's ``retie_parameters()`` step; eval /
    inference paths that don't re-tie still produce correct outputs
    because quantized inference doesn't grad-update tied tensors.
    """
    checkpoint_meta = get_checkpoint_metadata(model_dir)

    if not checkpoint_meta:
        raise FileNotFoundError

    if checkpoint_meta.is_index:
        shard_index = load_shard_index(model_dir, checkpoint_meta.file_name)
        if module is not None and _maybe_install_torchao_quantization(
            model_dir,
            module,
            shard_index=shard_index,
            safetensors=checkpoint_meta.safetensors,
            device=device,
        ):
            # Quantized weights are tensor subclasses; ``Tensor.copy_``
            # between two quantized subclasses fails with a metadata
            # mismatch. ``assign=True`` rebinds the Parameter directly,
            # bypassing copy_. The flip-side: assigned tensors keep
            # their map_location device, so we must load to wherever
            # the (already-constructed) module lives — not to the
            # caller-passed ``device`` (which may be a staging area
            # like CPU).
            assign = True
            device = _module_device(module, fallback=device)
        return load_sharded_checkpoint(
            model_dir,
            shard_index,
            module,
            device=device,
            safetensors=checkpoint_meta.safetensors,
            strict=strict,
            assign=assign,
            keys=keys,
        )

    state_dict_path = os.path.join(model_dir, checkpoint_meta.file_name)
    if checkpoint_meta.safetensors:
        state_dict = safetensors_load(
            state_dict_path, device=torch.device(device).index
        )
    else:
        state_dict = torch.load(
            state_dict_path, map_location=device, weights_only=True, mmap=True
        )

    if module is None:
        if keys is not None:
            return {k: v for k, v in state_dict.items() if k in keys}
        return state_dict

    if _maybe_install_torchao_quantization(model_dir, module, state_dict=state_dict):
        assign = True
        # Move the loaded tensors onto the module's existing device before
        # assigning them in (see the sharded branch above for the reason).
        target = _module_device(module, fallback=device)
        if str(target) != str(device):
            state_dict = {k: v.to(target) for k, v in state_dict.items()}

    # HF safetensors deduplicates tied weights on save: only one canonical
    # name per tied group lands in the file. Without this synthesis,
    # strict-mode load_state_dict would reject the checkpoint for missing
    # aliases like lm_head.weight on tied-embedding models (e.g. Gemma3).
    _synthesize_tied_aliases(module, state_dict)

    # TODO: Properly handle strict, in this case?
    # We wish to ensure that all model weights were loaded, but ignore any other weights, like we do in load_sharded_checkpoint()
    module.load_state_dict(state_dict, strict=strict, assign=assign)
    flag_loaded_tensors(module, state_dict.keys())
    return None


def _module_device(module: Module, *, fallback) -> "torch.device | str":
    """Return the device of the module's first parameter, or ``fallback``.

    Skips ``meta`` devices: a module constructed under ``torch.device("meta")``
    has no storage, and inheriting that device would silently produce a
    fully-meta model after ``load_state_dict(assign=True)``. In that
    case, defer to the caller-passed ``fallback`` instead.
    """
    for p in module.parameters():
        if p.device.type != "meta":
            return p.device
    for b in module.buffers():
        if b.device.type != "meta":
            return b.device
    return fallback


def _maybe_install_torchao_quantization(
    model_dir: str,
    module: Module,
    *,
    shard_index: "ShardIndex | None" = None,
    state_dict: "Dict[str, Tensor] | None" = None,
    safetensors: bool = False,
    device: str = "cpu",
) -> bool:
    """Detect torchao quantization on a checkpoint and install matching linear modules.

    Detection prefers ``<model_dir>/config.json``'s ``quantization_config``
    block (cheap, no shard load). Falls back to scanning the first shard
    (or the supplied single-file state_dict) for torchao tensor subclasses.
    Without this step, ``load_state_dict`` would try to copy quantized
    tensor subclasses into plain ``nn.Linear.weight`` slots and fail with
    ``'Parameter' object has no attribute 'tensor_data_names'``.

    Returns True if quantization was detected and installed; the caller
    should force ``assign=True`` on the subsequent ``load_state_dict``
    call to bypass ``Tensor.copy_`` (which doesn't handle
    quantized-to-quantized copies cleanly).
    """
    from forgather.ml.quantization_detect import (
        detect_torchao_quantization,
        install_torchao_quantization,
    )

    base_config = detect_torchao_quantization(model_dir=model_dir)
    if base_config is None:
        if state_dict is None and shard_index is not None:
            state_dict = _peek_first_shard(
                model_dir,
                shard_index,
                safetensors=safetensors,
                device=device,
            )
        if state_dict is not None:
            base_config = detect_torchao_quantization(state_dict=state_dict)
    if base_config is None:
        return False

    logger.info(
        "load_checkpoint: detected torchao quantization (%s); "
        "installing quantized linear modules before load_state_dict",
        type(base_config).__name__,
    )
    install_torchao_quantization(module, base_config)
    return True


def _peek_first_shard(
    model_dir: str,
    shard_index: "ShardIndex",
    *,
    safetensors: bool = False,
    device: str = "cpu",
) -> "Dict[str, Tensor]":
    """Load just one shard's state_dict for quantization detection.

    Eagerly register torchao tensor subclasses with PyTorch's
    ``add_safe_globals`` before loading, so ``weights_only=True`` accepts
    quantized shards. Registration is idempotent and cheap (a single
    import + dedup against PyTorch's allowlist), and importantly keeps
    the loader's safe-default posture: we never fall back to
    ``weights_only=False`` on a `.bin` shard, which would permit
    arbitrary pickled code execution.
    """
    from forgather.ml.quantization_detect import _register_torchao_safe_globals

    weight_map = shard_index["weight_map"]
    # Unique shard files; pick the first deterministically (sorted) so
    # tests are reproducible.
    shard_files = sorted(set(weight_map.values()))
    if not shard_files:
        return {}
    shard_file_path = os.path.join(model_dir, shard_files[0])
    if safetensors:
        return safetensors_load(shard_file_path, device=device)
    _register_torchao_safe_globals()
    return torch.load(
        shard_file_path, map_location=device, weights_only=True, mmap=True
    )


@overload
def load_sharded_checkpoint(
    model_dir: str,
    shard_index: ShardIndex,
    module: Module,
    device: str,
    safetensors: bool = False,
    strict: bool = True,
    assign: bool = False,
    debug: bool = False,
    keys: Optional[Set[str]] = None,
) -> Set[str]: ...


@overload
def load_sharded_checkpoint(
    model_dir: str,
    shard_index: ShardIndex,
    module: None,
    device: str,
    safetensors: bool = False,
    strict: bool = True,
    assign: bool = False,
    debug: bool = False,
    keys: Optional[Set[str]] = None,
) -> Dict[str, Tensor]: ...


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
) -> Union[Set[str], Dict[str, Tensor]]:
    """
    Load a sharded checkpoint.

    The weight_map may be a super-set of the weights in weight_map, which is useful for
    loading only the relevant weights for a sharded model.

    Parameters
    ----------
    model_dir : str
        Directory containing checkpoint shard files.
    shard_index : ShardIndex
        Shard index mapping weight names to shard file names.
    module : nn.Module or None, optional
        An nn.Module to load weights into. If None, returns a raw
        Dict[str, Tensor] instead of loading into a module.
    device : str, optional
        Device to map tensors to when loading.
    safetensors : bool, optional
        Load from safetensors format when True, PyTorch otherwise.
    strict : bool, optional
        Whether to require all module keys to be present in the checkpoint.
    assign : bool, optional
        If True, assign loaded tensors rather than copying data.
    debug : bool, optional
        Enable debug-level logging of individual weights.
    keys : set of str or None, optional
        When module is None, optionally restrict which keys to load
        from the weight map. Ignored when module is provided.

    Returns
    -------
    set of str or Dict[str, Tensor]
        When module is provided: set of str of keys that were NOT loaded (unloaded keys).
        When module is None: Dict[str, Tensor] of loaded tensors.
    """
    weight_map = shard_index["weight_map"]

    if module is not None:
        # Module mode: load into module (existing behavior)
        intersection = _intersect_weight_map(weight_map, module.state_dict())

        # HF safetensors deduplicates tied weights on save (e.g. only
        # model.embed_tokens.weight is stored; lm_head.weight is omitted).
        # Treat the missing aliases as already-covered: the in-place copy_
        # of the canonical tensor updates the shared storage they reference,
        # and the caller's tie_weights() step restores sharing afterward.
        tied_aliases = _tied_aliases_in_module(module, set(weight_map.keys()))

        all_module_keys = set(module.state_dict().keys()) - tied_aliases
        missing_keys = all_module_keys - intersection
        if strict and len(missing_keys):
            raise Exception(
                f"Index file does not contain mappings for the following keys {missing_keys} "
            )

        shard_files = set()
        for weight_name in intersection:
            shard_files.add(weight_map[weight_name])

        for shard_file_name in shard_files:
            shard_file_path = os.path.join(model_dir, shard_file_name)
            if safetensors:
                state_dict = safetensors_load(
                    shard_file_path,
                    device=device,
                )
            else:
                state_dict = torch.load(
                    shard_file_path, map_location=device, weights_only=True, mmap=True
                )

            all_module_keys = all_module_keys - set(state_dict.keys())
            logger.debug(f"loading state_dict in '{shard_file_name}'")

            module.load_state_dict(state_dict, strict=False, assign=assign)
            for weight_name, p in module.state_dict(keep_vars=True).items():
                logger.debug(
                    f"{weight_name} : {p.shape=}, {p.dtype=}, {p.requires_grad=}"
                )
            state_dict = None
            gc.collect()

        # Flag everything this module loaded once, after all its shards
        # (``intersection`` is the module's full loaded-key set), rather than
        # re-scanning the whole module per shard.
        flag_loaded_tensors(module, intersection)

        if len(all_module_keys):
            msg = f"The following keys were not found in the shards {all_module_keys}"
            if strict:
                raise Exception(msg)
            else:
                logger.warning(msg)

        return all_module_keys

    # Dict mode: accumulate tensors into a dict and return.
    # Shard files are iterated in sorted order so the returned dict's
    # key order is deterministic across runs (otherwise a multi-shard
    # checkpoint would expose hash-order cross-shard, mirroring the
    # write-side bug fixed by #44 on the per-shard level).
    target_keys = keys if keys is not None else set(weight_map.keys())
    loadable_keys = target_keys.intersection(set(weight_map.keys()))

    shard_files = sorted({weight_map[name] for name in loadable_keys})

    result: Dict[str, Tensor] = {}
    for shard_file_name in shard_files:
        shard_file_path = os.path.join(model_dir, shard_file_name)
        if safetensors:
            state_dict = safetensors_load(
                shard_file_path,
                device=device,
            )
        else:
            state_dict = torch.load(
                shard_file_path, map_location=device, weights_only=True, mmap=True
            )

        logger.debug(f"loading state_dict in '{shard_file_name}'")

        for key, tensor in state_dict.items():
            if key in loadable_keys:
                result[key] = tensor

        state_dict = None
        gc.collect()

    return result


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

    if os.path.exists(safetensors_index_path):
        return CheckpointMeta(SAFE_WEIGHTS_INDEX_NAME, True, True)
    elif os.path.exists(torch_index_path):
        return CheckpointMeta(WEIGHTS_INDEX_NAME, True, False)
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
        # A checkpoint with no model weights is valid ONLY when it is
        # explicitly marked model-less (the model-weights-external case, e.g.
        # DiLoCo: the parameter server owns the weights, so the worker
        # checkpoints only optimizer / scheduler / trainer / RNG). The marker
        # is written by the CheckpointManager when "model" is excluded. A
        # checkpoint missing model files WITHOUT the marker is a partial /
        # corrupt normal checkpoint and stays invalid, so discovery falls back
        # to an older complete one. See TrainingArguments.checkpoint_components.
        if os.path.exists(os.path.join(checkpoint_path, MODEL_EXCLUDED_MARKER)):
            return True
        return False

    return True


CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"

#: Sentinel file marking a checkpoint that intentionally carries no model
#: weights (model weights are supplied by an external authority, e.g. a DiLoCo
#: parameter server). Written by the CheckpointManager when "model" is
#: excluded from the active state components; consulted by validate_checkpoint
#: so such a checkpoint is still discoverable for resume while a model-less
#: *normal* checkpoint (missing weights, no marker) remains invalid.
MODEL_EXCLUDED_MARKER = ".forgather_no_model_weights"


def _get_checkpoint_timestamp(checkpoint_path: str) -> float:
    """Get effective timestamp for a checkpoint as epoch seconds.

    Prefers the timestamp from checkpoint_manifest.json (stable across file copies).
    Falls back to filesystem mtime if manifest is missing or corrupt.
    """
    manifest_path = os.path.join(checkpoint_path, CHECKPOINT_MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            ts = datetime.fromisoformat(data["timestamp"])
            return ts.timestamp()
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            logger.warning(
                f"Failed to read manifest timestamp from {manifest_path}: {e}. "
                "Falling back to filesystem mtime."
            )
    return os.path.getmtime(checkpoint_path)


def _checkpoint_sort_key(checkpoint_path: str) -> tuple[float, int]:
    """Sort key that tie-breaks by the integer suffix of ``checkpoint-<N>``.

    Filesystem mtime resolution can be too coarse to distinguish back-to-back
    checkpoints (notably on overlayfs), and manifest timestamps recorded with
    second-precision ``datetime.now().isoformat()`` collide too. Tie-breaking
    by the numeric suffix keeps oldest-first ordering stable so rotation
    deletes the genuinely oldest checkpoint.
    """
    ts = _get_checkpoint_timestamp(checkpoint_path)
    name = os.path.basename(checkpoint_path.rstrip(os.sep))
    suffix = name.split("checkpoint-", 1)[-1]
    try:
        index = int(suffix)
    except ValueError:
        index = 0
    return (ts, index)


def find_latest_checkpoint(model_dir: str) -> str | None:
    """Find the most recent valid checkpoint in the checkpoints directory.

    Uses checkpoint_manifest.json timestamp when available, falling back to
    filesystem modification time for legacy checkpoints.
    """
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
        latest = max(valid_checkpoints, key=_checkpoint_sort_key)
        step_num = (
            os.path.basename(latest).split("-")[1]
            if "-" in os.path.basename(latest)
            else "unknown"
        )
        ts = _get_checkpoint_timestamp(latest)
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        logger.debug(
            f"Found latest valid checkpoint: {latest} (step {step_num}, timestamp {ts_str})"
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
    model_dir: str,
    max_checkpoints: int,
    best_checkpoint: str | None = None,
    preserved_checkpoints: List[str] | None = None,
) -> None:
    """
    Delete oldest checkpoints, preserving specified checkpoints.

    Parameters
    ----------
    model_dir : str
        Model directory containing checkpoints subdirectory
    max_checkpoints : int
        Maximum number of checkpoints to keep
    best_checkpoint : str or None, optional
        (Deprecated) Single best checkpoint to preserve
    preserved_checkpoints : list of str or None, optional
        List of checkpoint paths to never delete
    """
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        logger.debug(
            f"No checkpoints directory found at {checkpoints_dir}, skipping deletion"
        )
        return

    checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    if len(checkpoints) <= max_checkpoints:
        return

    # Build set of preserved checkpoint paths
    preserved_set = set(preserved_checkpoints or [])
    if best_checkpoint:
        preserved_set.add(best_checkpoint)

    # Filter out preserved checkpoints from deletion candidates
    checkpoints_to_consider = [cp for cp in checkpoints if cp not in preserved_set]

    # Calculate how many to delete
    num_to_delete = len(checkpoints) - max_checkpoints
    # Ensure we don't delete more than available in checkpoints_to_consider
    num_to_delete = min(num_to_delete, len(checkpoints_to_consider))

    if num_to_delete > 0:
        # Sort by timestamp (manifest preferred, mtime fallback) and delete the oldest
        checkpoints_to_consider.sort(key=_checkpoint_sort_key)
        for checkpoint_path in checkpoints_to_consider[:num_to_delete]:
            logger.info(
                f"Deleting checkpoint at {checkpoint_path} (preserved: {preserved_set})"
            )
            shutil.rmtree(checkpoint_path)


def create_pretrained_symlinks(
    model_dir: str, force_overwrite: bool = False, dry_run: bool = False
) -> List[str]:
    """
    Create symlinks in model root directory pointing to latest checkpoint files.

    This enables Hugging Face .from_pretrained() to work with checkpointed models
    by making the latest checkpoint weights accessible from the model root directory.

    Parameters
    ----------
    model_dir : str
        Path to the model directory containing checkpoints subdirectory
    force_overwrite : bool, optional
        If True, overwrite existing real files. If False, only
        overwrite existing symlinks or create new symlinks.
    dry_run : bool, optional
        If True, only log what would be done without creating symlinks

    Returns
    -------
    list of str
        List of symlink paths that were created (or would be created in dry_run mode)

    Raises
    ------
    FileNotFoundError
        If no valid checkpoints are found
    FileExistsError
        If target files exist and are not symlinks (when force_overwrite=False)
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
    logger.info(
        f"Checkpoint format: {'safetensors' if checkpoint_meta.safetensors else 'pytorch'}, "
        f"{'sharded' if checkpoint_meta.is_index else 'single file'}"
    )

    symlinks_created = []
    files_to_link = []

    # Determine which files need to be symlinked
    if checkpoint_meta.is_index:
        # Sharded checkpoint - need to link index file and all shard files
        index_path = os.path.join(latest_checkpoint_dir, checkpoint_meta.file_name)
        files_to_link.append((checkpoint_meta.file_name, index_path))

        # Load index to find all shard files
        try:
            shard_index = load_shard_index(
                latest_checkpoint_dir, checkpoint_meta.file_name
            )
            weight_map = shard_index["weight_map"]

            # Get unique shard file names
            shard_files = set(weight_map.values())
            for shard_file in shard_files:
                shard_path = os.path.join(latest_checkpoint_dir, shard_file)
                if os.path.exists(shard_path):
                    files_to_link.append((shard_file, shard_path))
                else:
                    logger.warning(
                        f"Shard file referenced in index but not found: {shard_path}"
                    )

        except Exception as e:
            logger.error(f"Failed to read shard index: {e}")
            raise
    else:
        # Single file checkpoint
        weight_file_path = os.path.join(
            latest_checkpoint_dir, checkpoint_meta.file_name
        )
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
                action = (
                    "would overwrite" if is_symlink else "would replace real file with"
                )
                logger.info(f"DRY RUN: {action} symlink {link_path} -> {target_path}")
            else:
                # Remove existing file/symlink
                if is_symlink:
                    logger.info(f"Replacing existing symlink {link_path}")
                else:
                    logger.warning(
                        f"Replacing real file {link_path} with symlink (force_overwrite=True)"
                    )
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
