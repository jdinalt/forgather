from typing import List, Tuple
import logging
import re

from torch.fx import GraphModule
from torch.export.unflatten import InterpreterModule, UnflattenedModule
from torch.utils.checkpoint import checkpoint

from ..trainer import has_gradient_checkpointing_enable

logger = logging.getLogger(__name__)


def insert_activation_checkpoints(rank, module, targets):
    """
    Enable activation on graph-module (automatic-split)
    """
    targets_re = re.compile(targets)

    for name, submod in module.named_modules():
        if isinstance(
            submod,
            (
                GraphModule,
                InterpreterModule,
                UnflattenedModule,
            ),
        ):
            dirty = False
            logger.debug(f"Processing Checkpoint for: {name}, {type(submod)}")
            for node in submod.graph.nodes:
                if node.op == "call_module":
                    if not targets_re.match(node.target):
                        continue
                    dirty = True
                    logger.debug(f"fixing up node {node.target}")
                    node.args = (submod.get_submodule(node.target), *node.args)  # type: ignore
                    node.kwargs = dict(use_reentrant=False, **node.kwargs)
                    node.target = checkpoint
                    node.op = "call_function"

            if not dirty:
                continue

            if isinstance(submod, GraphModule):
                submod.graph.lint()
                submod.recompile()
            else:
                submod.graph.lint()
                # Recompiing appears to trigger a syntax error!? By default, these
                # are interpreted, so recompilation is not required, but why the error?


def missing_buffers(mod):
    """
    Generate the set of fully-qualified-names buffer names for buffers missing from the state dictionary.
    This can occur when mod.register_buffer(..., persistent=False)
    The option to not save these really does complicate things!
    """
    sd = mod.state_dict()
    bset = set()
    for name, buffer in mod.named_buffers():
        if not name in sd:
            bset.add(name)
    return bset


def persist_buffers(mod, bset, mod_fqn=""):
    """
    Walk module and all module's children recusively.

    If a buffer is in the set bset of fully-qualified-named (FQN), then convert the
    buffer to a persistent buffer.
    """
    # Convert buffers to persistent buffers.
    for name, buffer in mod.named_buffers(recurse=False):
        fqn = mod_fqn + "." + name
        if fqn in bset:
            logger.debug(
                f"Converting buffer non-persistent buffer {fqn} to persistent buffer"
            )
            mod.register_buffer(name, buffer.data)

    # And now for our children too...
    for name, child in mod.named_children():
        if len(mod_fqn):
            name = mod_fqn + "." + name
        persist_buffers(child, bset, name)


def set_parameter(mod, fqn, p):
    """
    Given a module, a FQN, and a paramm replace FQN in module with p

    This works with either buffers or parameters.
    """
    atoms = fqn.split(".")
    for atom in atoms[:-1]:
        mod = getattr(mod, atom)
    setattr(mod, atoms[-1], p)


def replace_parameters(to_mod, from_mod):
    """
    Replace the parameters in to_mod with those in from_mod

    IMPORTANT: Use remove_duplicate=False to ensure shared parameters are handled correctly
    """
    for name, p in to_mod.named_parameters(remove_duplicate=False):
        set_parameter(to_mod, name, from_mod.get_parameter(name))


def replace_buffers(to_mod, from_mod):
    """
    Replace the buffers in to_mod with those in from_mod

    IMPORTANT: Use remove_duplicate=False to ensure shared buffers are handled correctly
    """
    for name, p in to_mod.named_buffers(remove_duplicate=False):
        set_parameter(to_mod, name, from_mod.get_buffer(name))


def pipeline_stage_indices(
    pp_size, n_stages, style: str = "loop"
) -> List[Tuple[int, ...]]:
    """
    Get the stage indices for all ranks

    See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/pipeline.py#L194
    """
    stages_per_rank = n_stages // pp_size
    match style:
        case "loop":
            assert (
                n_stages % pp_size == 0
            ), f"n_stages {n_stages} must be divisible by pipeline size {pp_size}"

            stage_indices = list(
                tuple(rank + i * pp_size for i in range(stages_per_rank))
                for rank in range(pp_size)
            )
        case "v":
            # Sanity check that all of the computed indices are valid
            assert stages_per_rank == 2

            stage_indices = list(
                tuple(
                    x for x in zip(range(pp_size), range(n_stages - 1, pp_size - 1, -1))
                )
            )

        case _:
            raise Exception(f"Unrecognized indices styel {style}")

    return stage_indices
