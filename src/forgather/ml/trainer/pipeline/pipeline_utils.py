import logging
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def assert_no_duplicate_fqns(state_dicts):
    """
    Assert that no FQN appears in multiple state dictionaries.

    This is a critical invariant for pipeline checkpoint saving - duplicate FQNs
    cause multiple processes to write to the same shard file, resulting in
    checkpoint corruption.
    """
    all_fqns: Set[str] = set()
    duplicate_fqns: Set[str] = set()

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
