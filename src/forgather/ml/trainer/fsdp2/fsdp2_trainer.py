"""
FSDP2 (fully_shard) trainer.

Parallels DDPTrainer but uses torch.distributed.fsdp.fully_shard to shard
parameters, gradients and optimizer state across the data-parallel mesh.

Design notes
------------
- Sharding is applied inside _prepare_model() (not _wrap()) because
  fully_shard() replaces parameters with DTensors in place; the optimizer
  must be created *after* that swap so its param groups hold the sharded
  parameters. The base Trainer._prepare() calls _prepare_model() before
  _init_optimizer(), so overriding _prepare_model() is the natural hook.
- Checkpoints are saved as per-rank shards using SharingPattern.PER_RANK.
  Each rank writes the local DTensor state via
  torch.distributed.checkpoint.state_dict.get_model_state_dict with default
  (sharded) StateDictOptions. Checkpoints are therefore tied to the world
  size they were saved at; resuming at a different world size is out of
  scope for this first cut (would require a DCP-based loader).
- Gradient-sync gating at accumulation boundaries uses FSDP2's
  set_requires_gradient_sync() instead of DDP's no_sync() context manager.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar, cast, override

import torch
from dacite import from_dict
from torch import Tensor
from torch import distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.nn import Module
from torch.utils.data import DataLoader

from forgather.ml.distributed import prefix_logger_rank
from forgather.ml.trainer import DataloaderDispatcher
from forgather.ml.trainer.checkpoint_manager import (
    CheckpointConfig,
    CheckpointManager,
    RNGState,
)
from forgather.ml.trainer.checkpoint_types import SharingPattern, StateComponent
from forgather.ml.trainer.synchronized_dataloader import SynchronizedDataLoader
from forgather.ml.trainer.trainer import Trainer, TrainingArguments
from forgather.ml.trainer.trainer_types import FusedLossFactoryT

logger = logging.getLogger(__name__)
prefix_logger_rank(logger, lambda rank: True)


_DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _parse_dtype(value: Optional[str]) -> Optional[torch.dtype]:
    if value is None:
        return None
    try:
        return _DTYPE_ALIASES[value.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dtype '{value}' for FSDP2 mixed precision; "
            f"expected one of {sorted(_DTYPE_ALIASES)}"
        ) from exc


def _resolve_attr_path(root: Any, path: str) -> Optional[Any]:
    """Walk a dotted attribute path, returning None if any hop is missing."""
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _iter_layer_modules(container: Any) -> List[torch.nn.Module]:
    """
    Iterate transformer-block modules out of a ModuleDict / ModuleList /
    plain nn.Module container. Returns an empty list if the container isn't
    iterable in a way that yields nn.Module children.
    """
    if isinstance(container, torch.nn.Module):
        children = list(container.children())
        if children and all(isinstance(c, torch.nn.Module) for c in children):
            return children
        return []
    raw: Any = container
    try:
        maybe = list(iter(raw))
    except TypeError:
        return []
    if maybe and all(isinstance(c, torch.nn.Module) for c in maybe):
        return maybe
    return []


@dataclass(kw_only=True)
class FSDP2Arguments:
    # Whether to re-shard parameters after forward. True behaves like ZeRO-3
    # (minimum memory); False behaves like ZeRO-2 (keeps params unsharded
    # between forward and backward, lower comm, higher memory). An int N
    # enables hybrid resharding across N ranks (ZeRO++ hpZ).
    reshard_after_forward: bool = True

    # MixedPrecisionPolicy dtypes. None disables FSDP-level mixed precision;
    # the trainer's existing AMP autocast still applies.
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    buffer_dtype: Optional[str] = None

    # Offload parameters (and gradients) to CPU between uses.
    cpu_offload: bool = False

    # Apply fully_shard layer-by-layer on the transformer blocks before the
    # root module. Disabling this falls back to a single root-level wrap,
    # which severely limits memory savings (no per-layer reshard).
    shard_transformer_layers: bool = True

    # Dotted attribute path on the unwrapped model that yields an iterable
    # of transformer blocks. The default matches Forgather's standard
    # causal-LM structure (an HF wrapper with `base_model_prefix="causal_lm"`
    # around a `CausalLM` whose `layer_stack.layers` is a `ModuleDict` /
    # `ModuleList`; see modelsrc/transformer/). Override for non-standard
    # models; set to "" or a path that does not resolve to disable
    # layer-wise sharding gracefully.
    transformer_layers_path: str = "causal_lm.layer_stack.layers"


@dataclass(kw_only=True)
class FSDP2TrainingArguments(TrainingArguments):
    # See DDPTrainingArguments.dispatch_batches -- same semantics here.
    dispatch_batches: bool = True

    fsdp2: FSDP2Arguments = field(default_factory=FSDP2Arguments)


TFSDP2TrainingArguments = TypeVar(
    "TFSDP2TrainingArguments", bound=FSDP2TrainingArguments
)


class _FSDP2ModelStateful(Stateful):
    """Stateful adapter that keeps FSDP2 model state sharded on save/load."""

    def __init__(self, model: Module):
        self._model = model

    def state_dict(self) -> Dict[str, Any]:
        return get_model_state_dict(self._model)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_model_state_dict(self._model, state_dict)


class _FSDP2OptimStateful(Stateful):
    """Stateful adapter for the optimizer under FSDP2 sharded state."""

    def __init__(self, model: Module, optimizer: torch.optim.Optimizer):
        self._model = model
        self._optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        return get_optimizer_state_dict(self._model, self._optimizer)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self._model, self._optimizer, optim_state_dict=state_dict
        )


class FSDP2Trainer(Trainer[TFSDP2TrainingArguments], Generic[TFSDP2TrainingArguments]):
    """
    Trainer that shards model, gradients and optimizer state via FSDP2
    (torch.distributed.fsdp.fully_shard).
    """

    args: TFSDP2TrainingArguments

    def __init__(
        self,
        *,
        args: TFSDP2TrainingArguments | dict,
        fused_loss_factory: Optional[FusedLossFactoryT] = None,
        **kwargs,
    ):
        if isinstance(args, dict):
            args = cast(
                TFSDP2TrainingArguments, from_dict(FSDP2TrainingArguments, args)
            )

        super().__init__(args=args, fused_loss_factory=fused_loss_factory, **kwargs)

        assert (
            not self.args.fuse_optim_with_backward
        ), "FSDP2Trainer does not support option fuse_optim_with_backward"

    @override
    def _init_distributed(self):
        assert (
            dist.is_available and dist.is_initialized()
        ) or self.dist.world_size == 1, (
            "FSDP2 trainer requires that torch.distributed has been initialized."
        )

        if self.dist.world_size == 1:
            return super()._init_distributed()

        self.is_local_process_zero = self.dist.local_rank == 0
        self.is_world_process_zero = self.dist.rank == 0
        self.num_processes = self.dist.world_size

        self.mesh = init_device_mesh(
            self.dist.device_type,
            (self.dist.world_size,),
            mesh_dim_names=("data_parallel",),
        )
        self.fsdp_group = self.mesh.get_group(0)

    def _build_mp_policy(self) -> MixedPrecisionPolicy:
        fsdp_args = self.args.fsdp2
        return MixedPrecisionPolicy(
            param_dtype=_parse_dtype(fsdp_args.param_dtype),
            reduce_dtype=_parse_dtype(fsdp_args.reduce_dtype),
            output_dtype=None,
            cast_forward_inputs=True,
        )

    @override
    def _prepare_model(self) -> None:
        super()._prepare_model()
        if self.dist.world_size == 1:
            return

        assert self.model is not None

        mp_policy = self._build_mp_policy()
        offload_policy: OffloadPolicy = (
            CPUOffloadPolicy() if self.args.fsdp2.cpu_offload else OffloadPolicy()
        )

        layer_iter: List[torch.nn.Module] = []
        if (
            self.args.fsdp2.shard_transformer_layers
            and self.args.fsdp2.transformer_layers_path
        ):
            container = _resolve_attr_path(
                self.model, self.args.fsdp2.transformer_layers_path
            )
            if container is not None:
                layer_iter = _iter_layer_modules(container)

        if layer_iter:
            for layer in layer_iter:
                fully_shard(
                    layer,
                    mesh=self.mesh,
                    reshard_after_forward=self.args.fsdp2.reshard_after_forward,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                )
            logger.info(
                f"Applied fully_shard to {len(layer_iter)} transformer blocks at "
                f"'{self.args.fsdp2.transformer_layers_path}'"
            )
        else:
            logger.warning(
                f"FSDP2 layer-wise sharding skipped: could not resolve "
                f"'{self.args.fsdp2.transformer_layers_path}' on model as an "
                "iterable of nn.Modules. Falling back to root-level "
                "fully_shard only; memory savings will be limited."
            )

        fully_shard(
            self.model,
            mesh=self.mesh,
            reshard_after_forward=self.args.fsdp2.reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )

    @override
    def _wrap(self) -> None:
        """Wrap dataloaders for DP; the model is already sharded."""
        if self.dist.world_size == 1:
            return super()._wrap()

        if self.args.dispatch_batches:
            if self.train_dataloader:
                self.train_dataloader = DataloaderDispatcher(
                    cast(DataLoader, self.train_dataloader),
                    self.mesh,
                    self.args.device,
                )
            if self.eval_dataloader:
                self.eval_dataloader = DataloaderDispatcher(
                    cast(DataLoader, self.eval_dataloader),
                    self.mesh,
                    self.args.device,
                )
        else:
            if self.train_dataloader:
                self.train_dataloader = SynchronizedDataLoader(
                    self.train_dataloader,
                    device=self.args.device,
                    process_group=self.fsdp_group,
                )
            if self.eval_dataloader:
                self.eval_dataloader = SynchronizedDataLoader(
                    self.eval_dataloader,
                    device=self.args.device,
                    process_group=self.fsdp_group,
                )

    @override
    def _init_checkpoint_manager(self) -> CheckpointManager:
        """
        FSDP2 model state is sharded DTensors — incompatible with the
        safetensors save path that CheckpointManager uses for the "model"
        component. We pass model_parts=[] and a dummy shard_index so
        ``_save_model`` is a no-op; the model is saved/loaded through the
        CheckpointCoordinator as the "fsdp2_model" PER_RANK StateComponent.
        """
        if self.dist.world_size == 1:
            return super()._init_checkpoint_manager()

        cp_config = CheckpointConfig(
            output_dir=self.args.output_dir,
            save_total_limit=self.args.save_total_limit,
            save_on_each_node=self.args.save_on_each_node,
            save_safetensors=self.args.save_safetensors,
        )

        checkpoint_manager = CheckpointManager(
            config=cp_config,
            dist=self.dist,
            model=self.unwrapped_model(),
            model_parts=[],
            model_preprocessor=self.processing_class,
            stateful_provider=self,
            shard_index={},
        )
        checkpoint_manager.trainer = self
        if hasattr(self.args, "preserve_n_best"):
            checkpoint_manager.preserve_n_best = self.args.preserve_n_best
        return checkpoint_manager

    @override
    def unwrapped_model(self) -> torch.nn.Module:
        # fully_shard mutates the module class in place; there's no .module
        # wrapper to peel off.
        assert self.model is not None
        return self.model

    @override
    def _distributed_loss(self, loss: Tensor) -> Tensor:
        if self.dist.world_size == 1:
            return super()._distributed_loss(loss)
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return loss

    @override
    def _distributed_tokens(self, tokens: Tensor) -> Tensor:
        if self.dist.world_size == 1:
            return super()._distributed_tokens(tokens)
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        return tokens

    @override
    def _distributed_peak_mem(self, local_peak: int) -> list[int]:
        if self.dist.world_size == 1:
            return super()._distributed_peak_mem(local_peak)
        value = torch.tensor(
            [int(local_peak)], dtype=torch.long, device=self.args.device
        )
        gathered = [torch.zeros_like(value) for _ in range(self.dist.world_size)]
        dist.all_gather(gathered, value)
        return [int(t.item()) for t in gathered]

    @override
    def _forward_backward_step(self, *args, **kwargs) -> Tensor:
        """
        Gate FSDP2 gradient reduction at accumulation boundaries.

        FSDP2 uses set_requires_gradient_sync() rather than DDP's no_sync()
        context manager. Setting False suppresses the reduce-scatter in the
        backward hook for the current backward; setting True re-enables it
        for the final micro-batch of the accumulation window.
        """
        if self.dist.world_size == 1:
            return super()._forward_backward_step(*args, **kwargs)

        sync = self._should_sync_gradients()
        cast(FSDPModule, self.model).set_requires_gradient_sync(sync)
        return super()._forward_backward_step(*args, **kwargs)

    def pipeline_generate(self, input_ids: Tensor, **kwargs) -> Tensor:
        """All-rank generate: FSDP2 forward pass needs every rank in the
        all_gather, so generation must run collectively. The
        ``TextgenCallback`` detects this method and uses a coordinated
        broadcast-then-generate flow (same path as PipelineTrainer)."""
        assert self.model is not None
        return self.model.generate(input_ids=input_ids, **kwargs)

    @override
    def get_state_components(self) -> List[StateComponent]:
        """
        State components for FSDP2.

        Model and optimizer state remain sharded: each rank saves and loads
        its own DTensor shard (PER_RANK). validate_replication is False for
        those components because the per-rank state is expected to differ by
        design. Scheduler, trainer progress, dataset and RNG mirror
        DDPTrainer.
        """
        if self.dist.world_size == 1:
            return super().get_state_components()

        assert self.model is not None

        components: List[StateComponent] = []

        # Key is "fsdp2_model" (not "model") so CheckpointManager routes it
        # through the CheckpointCoordinator's PER_RANK path rather than the
        # safetensors model-save path (which can't handle DTensors).
        components.append(
            StateComponent(
                key="fsdp2_model",
                stateful=_FSDP2ModelStateful(self.model),
                sharing_pattern=SharingPattern.PER_RANK,
                validate_replication=False,
                required=True,
            )
        )

        if self.optimizer is not None:
            components.append(
                StateComponent(
                    key="optimizer",
                    stateful=_FSDP2OptimStateful(
                        self.model, cast(torch.optim.Optimizer, self.optimizer)
                    ),
                    sharing_pattern=SharingPattern.PER_RANK,
                    validate_replication=False,
                    required=False,
                )
            )

        if self.lr_scheduler is not None:
            components.append(
                StateComponent(
                    key="scheduler",
                    stateful=cast(Stateful, self.lr_scheduler),
                    sharing_pattern=SharingPattern.REPLICATED,
                    required=False,
                )
            )

        components.append(
            StateComponent(
                key="trainer",
                stateful=self,
                sharing_pattern=SharingPattern.REPLICATED,
                required=False,
            )
        )

        if hasattr(self.train_dataloader, "state_dict"):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=cast(Stateful, self.train_dataloader),
                    sharing_pattern=self._get_dataset_sharing_pattern(),
                    required=False,
                )
            )

        components.append(
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=False,
            )
        )

        return components

    @override
    def _get_dataset_sharing_pattern(self) -> SharingPattern:
        if self.dist.world_size == 1:
            return super()._get_dataset_sharing_pattern()
        if self.args.dispatch_batches:
            return SharingPattern.GLOBAL
        return SharingPattern.PER_RANK

    @override
    def get_process_groups(self) -> Dict[str, Any]:
        if self.dist.world_size == 1:
            return super().get_process_groups()
        return {"fsdp_group": self.fsdp_group}
