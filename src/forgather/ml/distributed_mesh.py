"""
Multi-axis device-mesh builder for DiLoCo + inner parallelism (issue #154).

DiLoCo's collective backend makes a group of **replicas** average their
pseudo-gradients every ``H`` steps; each replica can itself run inner parallelism
(pipeline / DDP / FSDP). The clean way to compose these — and the way
torchtitan's ``ParallelDims`` does it — is a single ``torch.distributed`` device
mesh with one named axis per parallelism type, so each collective runs on its own
sub-group.

This module owns the ``(diloco, inner)`` decomposition: a 2-axis mesh where

* ``diloco`` (dim 0, the outer/slow axis) is the **replicate** axis — the
  CollectiveBackend all-reduces pseudo-gradients across it; and
* ``inner`` (dim 1, the fast/contiguous axis) is the axis the *trainer*
  parallelizes over (``data_parallel`` for DDP/FSDP, ``pipeline_parallel`` for
  pipeline). Its per-step collectives span one replica's ranks only.

The degrees multiply to the torchrun world size (``diloco * inner == world_size``,
the torchtitan invariant). With ``inner == 1`` (Phase 1: N single-device
replicas) the mesh is effectively 1-D over ``diloco`` and the trainer sees
``world_size == 1``.

Rank ordering: with ``mesh_dim_names=("diloco", inner_axis)`` and shape
``(diloco, inner)``, dim 1 is contiguous in global rank — global ranks
``[k, inner+k, 2*inner+k, ...]`` form the ``diloco`` group at inner-position
``k`` (the replicas holding the *same* inner slice), and ``[0..inner-1]`` form
replica 0's inner group. That keeps inner (pipeline/DDP) ranks contiguous
(NCCL-friendly) while the DiLoCo all-reduce strides across replicas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

INNER_AXES = ("data_parallel", "pipeline_parallel")


@dataclass
class ForgatherParallelDims:
    """The ``(diloco, inner)`` device-mesh decomposition.

    ``diloco`` and ``inner`` are degrees that multiply to ``world_size``;
    ``inner_axis`` names the inner dimension (``"data_parallel"`` or
    ``"pipeline_parallel"``). The mesh is built lazily on first access (it needs
    an initialized process group), so a ``ForgatherParallelDims`` can be
    constructed and degree-validated without ``torch.distributed`` up.
    """

    diloco: int
    inner: int
    inner_axis: str
    world_size: int
    device_type: str = "cuda"

    def __post_init__(self):
        if self.diloco < 1 or self.inner < 1:
            raise ValueError(
                f"parallel degrees must be >= 1, got diloco={self.diloco}, "
                f"inner={self.inner}"
            )
        if self.inner_axis not in INNER_AXES:
            raise ValueError(
                f"inner_axis must be one of {INNER_AXES}, got {self.inner_axis!r}"
            )
        if self.diloco * self.inner != self.world_size:
            raise ValueError(
                f"diloco({self.diloco}) * inner({self.inner}) != "
                f"world_size({self.world_size})"
            )

    @cached_property
    def world_mesh(self):
        """Build (once) the 2-axis device mesh. Requires an initialized group."""
        from torch.distributed.device_mesh import init_device_mesh

        return init_device_mesh(
            self.device_type,
            (self.diloco, self.inner),
            mesh_dim_names=("diloco", self.inner_axis),
        )

    # ----- DiLoCo (replicate) axis -----------------------------------------

    def diloco_group(self):
        """The process group of replicas at this rank's inner position."""
        return self.world_mesh["diloco"].get_group()

    def diloco_rank(self) -> int:
        return self.world_mesh.get_local_rank("diloco")

    def diloco_size(self) -> int:
        return self.diloco

    # ----- inner (trainer) axis --------------------------------------------

    def inner_mesh(self):
        """The inner sub-mesh the trainer parallelizes over (DDP/FSDP/pipeline)."""
        return self.world_mesh[self.inner_axis]

    def inner_group(self):
        return self.world_mesh[self.inner_axis].get_group()

    def inner_rank(self) -> int:
        return self.world_mesh.get_local_rank(self.inner_axis)

    def inner_size(self) -> int:
        return self.inner
