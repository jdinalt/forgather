"""
Workarounds for PyTorch pipeline parallelism bugs.

This module provides utilities to fix "zombie buffers" that are created by
PyTorch's pipeline splitting implementation. Zombie buffers exist as module
attributes but are not properly registered in the module's buffer registry,
causing failures when using to_empty() to materialize meta tensors.

The above issue appears to be triggered by including shared buffers in the model, for
example, sharing RoPE embeddings.

Another identified bug is that splitting a model sometimes results in unreferenced
copies of a module in the wrong sub-module. This results in duplicate FQNs,
with-respect-to the complete model, and this wreaks havoc with checkpointing,
where multiple ranks believe they own the same FQN, corrupting each others checkpoints.

This module contains functions for for identifying these "vestigial" modules
and removing them.

At some point, when I have the time, I would like to write a clean version of the
model splitter, which works correctly. I have a prototype, but have other priorities.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


def detect_shared_buffers(model: torch.nn.Module) -> Dict[int, List[str]]:
    """
    Detect buffers that share the same tensor instance.

    Args:
        model: The original model before pipeline splitting

    Returns:
        Dict mapping tensor ID to list of FQNs that share that tensor
    """
    buffer_id_to_names = defaultdict(list)

    # Check named_buffers for shared instances
    # IMPORTANT: Use remove_duplicate=False to detect shared buffers!
    for name, buf in model.named_buffers(remove_duplicate=False):
        if isinstance(buf, torch.Tensor):
            buffer_id_to_names[id(buf)].append(name)

    # Also check state_dict to catch any additional shared tensors
    # IMPORTANT: Use keep_vars=True to preserve tensor IDs for shared parameters
    for name, tensor in model.state_dict(keep_vars=True).items():
        if isinstance(tensor, torch.Tensor):
            buffer_id_to_names[id(tensor)].append(name)

    # Return only shared buffers (multiple names for same ID)
    shared_buffers = {}
    for buf_id, names in buffer_id_to_names.items():
        unique_names = list(set(names))  # Remove duplicates
        if len(unique_names) > 1:
            shared_buffers[buf_id] = unique_names

    return shared_buffers


def find_zombie_buffers(
    stage_module: torch.nn.Module, original_model: torch.nn.Module
) -> List[Tuple[torch.nn.Module, str, torch.Tensor, bool, str]]:
    """
    Find zombie buffers in a pipeline stage module.

    A zombie buffer is a tensor that:
    1. Exists as a module attribute
    2. Should be a buffer based on the original model
    3. Is not properly registered in the module's _buffers dict

    Args:
        stage_module: Pipeline stage module to scan
        original_model: Original model before splitting

    Returns:
        List of (module, attr_name, tensor, persistent, full_name) tuples
    """
    zombie_buffers = []

    def scan_module_recursively(mod: torch.nn.Module, prefix: str = ""):
        """Recursively scan a module for zombie buffers."""
        for name, attr in mod.__dict__.items():
            if isinstance(attr, torch.Tensor):
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if this tensor should be a buffer
                is_buffer_in_original = False
                buffer_persistent = True

                # Check by name first
                for orig_name, orig_buf in original_model.named_buffers():
                    if orig_name == full_name:
                        is_buffer_in_original = True
                        # Check persistence
                        try:
                            atoms = orig_name.split(".")
                            owner = original_model
                            for atom in atoms[:-1]:
                                owner = getattr(owner, atom)
                            final_name = atoms[-1]
                            buffer_persistent = (
                                final_name not in owner._non_persistent_buffers_set
                            )
                        except:
                            buffer_persistent = True
                        break

                # If not found by name, check by tensor ID (for shared buffers)
                if not is_buffer_in_original:
                    for orig_name, orig_buf in original_model.named_buffers():
                        if id(attr) == id(orig_buf):
                            is_buffer_in_original = True
                            buffer_persistent = True  # Default for shared buffers
                            break

                if is_buffer_in_original:
                    # Check if it's properly registered
                    is_properly_registered = name in mod._buffers

                    if not is_properly_registered:
                        zombie_buffers.append(
                            (mod, name, attr, buffer_persistent, full_name)
                        )
                        logger.debug(f"Found zombie buffer: {full_name}")

        # Recursively check named submodules (not __dict__ to avoid infinite recursion)
        for name, submod in mod.named_children():
            submod_prefix = f"{prefix}.{name}" if prefix else name
            scan_module_recursively(submod, submod_prefix)

    scan_module_recursively(stage_module)
    return zombie_buffers


def fix_zombie_buffers(
    stage_module: torch.nn.Module,
    original_model: torch.nn.Module,
    shared_buffers: Dict[int, List[str]],
) -> int:
    """
    Fix zombie buffers in a pipeline stage by properly registering them.

    Args:
        stage_module: Pipeline stage module to fix
        original_model: Original model before splitting
        shared_buffers: Dict of shared buffer groups from detect_shared_buffers

    Returns:
        Number of zombie buffers fixed
    """
    zombie_buffers = find_zombie_buffers(stage_module, original_model)

    if not zombie_buffers:
        return 0

    logger.debug(f"Fixing {len(zombie_buffers)} zombie buffers in pipeline stage")

    # Fix each zombie buffer
    for mod, name, tensor, persistent, full_name in zombie_buffers:
        try:
            # Remove the tensor as a regular attribute
            if hasattr(mod, name):
                delattr(mod, name)

            # Register it properly as a buffer
            mod.register_buffer(name, tensor, persistent=persistent)
            logger.debug(f"Fixed zombie buffer: {full_name} -> properly registered")

        except Exception as e:
            logger.warning(f"Failed to fix zombie buffer {full_name}: {e}")

    # Additional fix for shared buffers: ensure all shared FQNs are accessible
    _ensure_shared_buffer_access(stage_module, shared_buffers)

    return len(zombie_buffers)


def _ensure_shared_buffer_access(
    stage_module: torch.nn.Module, shared_buffers: Dict[int, List[str]]
) -> None:
    """
    Ensure all shared buffer FQNs are properly accessible in the stage.

    This handles cases where a shared buffer might be accessible under one FQN
    but not another that should also work.
    """
    for buf_id, shared_names in shared_buffers.items():
        # Find the canonical tensor in this stage
        canonical_tensor = None

        for name, buf in stage_module.named_buffers():
            if id(buf) == buf_id:
                canonical_tensor = buf
                break

        if canonical_tensor is None:
            continue

        # Ensure all shared FQNs are accessible
        for shared_name in shared_names:
            try:
                # Try to access it
                atoms = shared_name.split(".")
                obj = stage_module
                for atom in atoms[:-1]:
                    if not hasattr(obj, atom):
                        # Parent doesn't exist, skip this FQN
                        break
                    obj = getattr(obj, atom)
                else:
                    # All parents exist
                    final_name = atoms[-1]

                    # Check if it exists and is properly registered
                    if hasattr(obj, final_name):
                        existing_tensor = getattr(obj, final_name)
                        if id(existing_tensor) == buf_id:
                            # Check if it's properly registered
                            if final_name not in obj._buffers:
                                logger.debug(
                                    f"Re-registering shared buffer {shared_name}"
                                )
                                # Remove and re-register
                                delattr(obj, final_name)
                                obj.register_buffer(
                                    final_name, canonical_tensor, persistent=True
                                )
                    else:
                        # Buffer doesn't exist - register it
                        logger.debug(f"Adding missing shared buffer {shared_name}")
                        obj.register_buffer(
                            final_name, canonical_tensor, persistent=True
                        )

            except Exception as e:
                logger.debug(
                    f"Could not ensure access to shared buffer {shared_name}: {e}"
                )


def apply_pipeline_buffer_fix(
    pipeline_modules: List[torch.nn.Module], original_model: torch.nn.Module
) -> Dict[str, Any]:
    """
    Apply the buffer fix to all pipeline modules.

    This is the main entry point for the buffer fix workaround.

    Args:
        pipeline_modules: List of pipeline stage modules to fix
        original_model: Original model before pipeline splitting

    Returns:
        Dict with fix statistics and metadata
    """
    logger.info("Applying pipeline buffer fix workaround")

    # Detect shared buffers in the original model
    shared_buffers = detect_shared_buffers(original_model)

    fix_stats = {
        "shared_buffer_groups": len(shared_buffers),
        "stages_fixed": 0,
        "total_zombies_fixed": 0,
        "shared_buffer_details": shared_buffers,
    }

    if shared_buffers:
        logger.debug(f"Detected {len(shared_buffers)} shared buffer groups")
        for buf_id, names in shared_buffers.items():
            logger.debug(f"Shared buffer {buf_id}: {names}")

    # Apply fix to each pipeline module
    for i, stage_module in enumerate(pipeline_modules):
        logger.debug(f"Processing pipeline stage {i}")

        zombies_fixed = fix_zombie_buffers(stage_module, original_model, shared_buffers)

        if zombies_fixed > 0:
            fix_stats["stages_fixed"] += 1
            fix_stats["total_zombies_fixed"] += zombies_fixed
            logger.debug(f"Fixed {zombies_fixed} zombie buffers in stage {i}")

    if fix_stats["total_zombies_fixed"] > 0:
        logger.info(
            f"Pipeline buffer fix complete: "
            f"{fix_stats['total_zombies_fixed']} zombies fixed across "
            f"{fix_stats['stages_fixed']} stages"
        )
    else:
        logger.debug("No zombie buffers found - pipeline is clean")

    return fix_stats


def validate_pipeline_buffers(
    pipeline_modules: List[torch.nn.Module], original_model: torch.nn.Module
) -> bool:
    """
    Validate that pipeline modules have no zombie buffers.

    This can be used as a diagnostic tool to verify the fix worked.

    Args:
        pipeline_modules: List of pipeline stage modules to validate
        original_model: Original model before pipeline splitting

    Returns:
        True if all buffers are properly registered, False otherwise
    """
    all_clean = True

    for i, stage_module in enumerate(pipeline_modules):
        zombie_buffers = find_zombie_buffers(stage_module, original_model)

        if zombie_buffers:
            logger.error(
                f"Pipeline stage {i} has {len(zombie_buffers)} zombie buffers:"
            )
            for _, _, _, _, full_name in zombie_buffers:
                logger.error(f"  - {full_name}")
            all_clean = False
        else:
            logger.debug(f"Pipeline stage {i} is clean")

    return all_clean


def remove_vestigial_modules(all_pipeline_modules):
    """
    Remove vestigial submodules that don't have FX graphs.

    Vestigial modules are created during pipeline splitting but don't contain
    active computation paths. They can cause duplicate FQN conflicts during
    checkpoint saving if not removed.
    """
    from torch import fx

    for i, module in enumerate(all_pipeline_modules):
        logger.debug(f"Cleaning vestigial modules from pipeline stage {i}")
        modules_to_remove = []

        # Find all submodules and identify vestigial ones
        for name, submodule in module.named_modules():
            if name == "":  # Skip root module
                continue

            # Check if submodule has an FX graph (active) or not (vestigial)
            has_graph = hasattr(submodule, "graph")
            is_fx_graph = has_graph and isinstance(submodule.graph, fx.Graph)

            if not is_fx_graph:
                # This is a vestigial module - check if it can be safely removed
                parent_name = ".".join(name.split(".")[:-1])
                module_name = name.split(".")[-1]

                try:
                    if parent_name:
                        parent_module = module.get_submodule(parent_name)
                    else:
                        parent_module = module

                    # Only remove if it's not referenced in any FX graph
                    if is_module_unreferenced(module, name):
                        modules_to_remove.append((parent_module, module_name, name))
                        logger.debug(f"  Marking vestigial module for removal: {name}")
                except AttributeError:
                    logger.debug(f"  Could not access parent of {name}, skipping")

        # Remove vestigial modules
        for parent_module, module_name, full_name in modules_to_remove:
            try:
                delattr(parent_module, module_name)
                logger.debug(f"  Removed vestigial module: {full_name}")
            except AttributeError:
                logger.debug(f"  Failed to remove module: {full_name}")


def is_module_unreferenced(root_module, target_name):
    """
    Check if a module is unreferenced in any FX graph within the root module.
    Returns True if the module is safe to remove.
    """
    from torch import fx

    # Search all FX graphs for call_module operations referencing target_name
    for name, submodule in root_module.named_modules():
        if hasattr(submodule, "graph") and isinstance(submodule.graph, fx.Graph):
            for node in submodule.graph.nodes:
                if node.op == "call_module" and node.target == target_name:
                    return False  # Module is referenced, don't remove
    return True  # Module is not referenced, safe to remove


def assert_no_duplicate_fqns(state_dicts):
    """
    Assert that no FQN appears in multiple state dictionaries.

    This is a critical invariant for pipeline checkpoint saving - duplicate FQNs
    cause multiple processes to write to the same shard file, resulting in
    checkpoint corruption.
    """
    all_fqns = set()
    duplicate_fqns = set()

    for i, state_dict in enumerate(state_dicts):
        for fqn in state_dict.keys():
            if fqn in all_fqns:
                duplicate_fqns.add(fqn)
                logger.error(f"Duplicate FQN found: '{fqn}' in pipeline modules")
            all_fqns.add(fqn)

    if duplicate_fqns:
        # Show which modules contain each duplicate FQN for debugging
        for fqn in duplicate_fqns:
            modules_with_fqn = []
            for i, state_dict in enumerate(state_dicts):
                if fqn in state_dict:
                    modules_with_fqn.append(f"module_{i}")
            logger.error(f"FQN '{fqn}' appears in: {modules_with_fqn}")

        raise AssertionError(
            f"Duplicate FQNs detected across pipeline modules: {duplicate_fqns}. "
            f"This will cause checkpoint saving conflicts. Each FQN must appear "
            f"in exactly one pipeline module."
        )
