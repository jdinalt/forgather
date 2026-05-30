# https://github.com/pytorch/pytorch/tree/main/torch/distributed/pipelining
import logging
import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
    override,
)

import torch
import torch.distributed as dist
from dacite import from_dict
from torch import Tensor, distributed
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import ScheduleGPipe
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.nn import Module
from torch.utils.data import DataLoader

from forgather.ml.construct import torch_dtype
from forgather.ml.loss import RescaleLoss
from forgather.ml.utils import default_dtype

from ...optim.opt_utils import build_parameter_groups
from ...sharded_checkpoint import (
    ShardIndex,
    SharingMetadataT,
    create_sharing_metadata,
    make_shard_index,
    retie_parameters,
)
from ..checkpoint_manager import CheckpointConfig, CheckpointManager, RNGState
from ..checkpoint_types import SharingPattern, StateComponent
from ..dataloader_dispatcher import DataloaderDispatcher
from ..trainer import Trainer, TrainingArguments, optimizer_hook
from ..trainer_types import LossFunctionT
from ._torch_patches import (  # noqa: F401  -- import also applies the stage_backward_input patch
    disable_compiled_backward_donated_buffers,
    is_zero_bubble_schedule,
)
from .model_splitter import ModelSplitter
from .pipeline_utils import (
    assert_no_duplicate_fqns,
    pipeline_stage_indices,
)

logger = logging.getLogger(__name__)


def log_level_for(level, prefix, modules: List[str]):
    for module_name in modules:
        logging.getLogger(prefix + module_name).setLevel(level)


# Enable debugging for various modules
log_level_for(
    logging.DEBUG,
    "torch.distributed.pipelining.",
    [
        # Add modules to enable logging on here.
    ],
)


class PipelineSchedulerT(Protocol):
    def step(self, *args, targets: torch.Tensor | None, losses: list, **kwargs): ...

    def eval(self, *args, **kwargs): ...


PipelineSchedulerFactorT: TypeAlias = Callable[
    [int, int, LossFunctionT, bool], PipelineSchedulerT
]


@dataclass(kw_only=True)
class PipelineTrainingArguments(TrainingArguments):
    """Training arguments for pipeline parallel training.

    Pipeline parallelism partitions a model across multiple GPUs, each handling
    one or more sequential stages. Input batches are split into microbatches that
    flow through the stages, allowing overlapped computation to keep all GPUs busy.

    See the PyTorch pipeline parallelism documentation for background:
    https://docs.pytorch.org/docs/stable/distributed.pipelining.html

    Parameters
    ----------
    n_microbatches : int, optional
        Number of microbatches to split each batch into. More microbatches
        improve pipeline efficiency (fewer bubbles) but increase memory usage.
        The batch size must be evenly divisible by ``n_microbatches``. Typical
        values are 4–16 depending on pipeline depth and memory constraints.
        Default is ``4``.
    stages_per_rank : int, optional
        Number of pipeline stages hosted on each GPU. Most schedulers use
        ``1``. Multi-stage schedulers (e.g. ``ScheduleZBVZeroBubble``) assign
        multiple stages per rank to reduce pipeline bubbles. Only set ``> 1``
        together with ``is_multistage=True``. Default is ``1``.
    pp_stage_type : str, optional
        Stage-to-rank assignment pattern. ``"loop"`` uses round-robin (e.g. 4
        stages on 2 ranks: rank0=[0,2], rank1=[1,3]). ``"v"`` uses the
        V-pattern required by ZeroBubble schedulers (see
        https://arxiv.org/pdf/2401.10241). Default is ``"loop"``.
    is_multistage : bool, optional
        Set ``True`` when the scheduler inherits from
        ``PipelineScheduleMulti`` (e.g. ``ScheduleZBVZeroBubble``). Leave
        ``False`` for single-stage schedulers such as ``ScheduleGPipe``.
        Default is ``False``.
    debug_pipeline : bool, optional
        Enable debug-level logging for the pipeline scheduler. Internal
        development flag. Default is ``False``.
    debug_split_model : bool, optional
        Log pipeline module details after splitting. Internal development
        flag. Default is ``False``.
    debug_model_params : bool, optional
        Log all parameter and buffer devices/dtypes after model construction.
        Internal development flag. Default is ``False``.
    debug_model_init : bool, optional
        Log every send/recv during parameter distribution from rank 0.
        Internal development flag. Default is ``False``.

    Notes
    -----
    ``model_splitter`` is passed to ``PipelineTrainer.__init__()`` rather than
    stored here because it is a callable, not a primitive serialisable type.
    """

    debug_pipeline: bool = False
    debug_split_model: bool = False
    debug_model_params: bool = False
    debug_model_init: bool = False
    n_microbatches: int = 4
    stages_per_rank: int = 1
    pp_stage_type: str = "loop"
    is_multistage: bool = False


TPipelineTrainingArguments = TypeVar(
    "TPipelineTrainingArguments", bound=PipelineTrainingArguments
)


class PipelineTrainer(
    Trainer[TPipelineTrainingArguments], Generic[TPipelineTrainingArguments]
):
    """Trainer for pipeline parallel training using PyTorch distributed pipelining.

    Partitions a model across multiple GPUs — each GPU hosts one or more
    sequential pipeline stages. Input batches are split into microbatches that
    flow through the stages with multiple microbatches in flight simultaneously,
    keeping all GPUs busy.

    This trainer is designed for environments where inter-GPU bandwidth is limited
    (consumer GPUs over PCIe, multi-node over Ethernet) where all-reduce–based DDP
    or FSDP would be communication-bound.

    Key differences from the single-device ``Trainer``:

    * Model is constructed on the meta device, then each stage is materialised
      on its assigned GPU — no full model ever lives on one GPU.
    * Rank 0 constructs a fully-initialised CPU model and distributes parameters
      to other ranks via point-to-point sends, avoiding N redundant initialisations.
    * All ranks receive the same batch (pure model parallelism); rank 0 loads data
      via ``DataloaderDispatcher`` and broadcasts it.
    * Gradient norm is all-reduced across ranks because each rank holds only a
      subset of the model's parameters.
    * Effective batch size does **not** scale with ``num_processes`` (the same
      batch flows through all stages; unlike DDP, there is no data replication).

    Parameters
    ----------
    args : PipelineTrainingArguments or dict
        Pipeline training configuration. Dicts are converted via
        ``dacite.from_dict``.
    model_splitter : ModelSplitter
        Callable that splits the model into pipeline stages and returns
        ``PipelineStage`` objects. See
        ``src/forgather/ml/trainer/pipeline/model_splitter.py`` for the
        expected signature.
    pipe_schedule_factory : callable, optional
        Factory for the pipeline scheduler (e.g. ``ScheduleGPipe``,
        ``ScheduleZBVZeroBubble``). Default is ``ScheduleGPipe``.
    **kwargs
        Additional arguments forwarded to the base ``Trainer``
        (``model_init``, ``train_dataset``, ``optimizer_factory``, etc.).

    Raises
    ------
    AssertionError
        If ``model`` is provided (pipeline training requires ``model_init``).
    AssertionError
        If ``model_init`` is not provided.
    AssertionError
        If batch size is not divisible by ``n_microbatches``.
    AssertionError
        If ``stages_per_rank > 1`` but ``is_multistage=False``.
    AssertionError
        If ``mixed_precision="fp16"`` (incompatible with pipeline schedulers).
    AssertionError
        If a zero-bubble schedule is used with ``torch_compile=True``.
    AssertionError
        If ``world_size == 1`` (pipeline parallelism requires multiple ranks).

    Examples
    --------
    >>> from torch.distributed.pipelining import ScheduleGPipe
    >>> args = PipelineTrainingArguments(
    ...     n_microbatches=8,
    ...     per_device_train_batch_size=64,
    ...     stages_per_rank=1,
    ... )
    >>> trainer = PipelineTrainer(
    ...     args=args,
    ...     model_init=model_factory,
    ...     model_splitter=my_splitter_fn,
    ...     pipe_schedule_factory=ScheduleGPipe,
    ...     train_dataset=train_dataset,
    ...     optimizer_factory=optimizer_factory,
    ... )
    >>> trainer.train()

    See Also
    --------
    ModelSplitter : Protocol for the model-splitting callable.

    References
    ----------
    PyTorch pipeline parallelism:
    https://docs.pytorch.org/docs/stable/distributed.pipelining.html
    """

    args: TPipelineTrainingArguments
    model_splitter: ModelSplitter
    pipe_schedule_factory: PipelineSchedulerFactorT
    pp_group: Any
    n_pipeline_stages: int
    scheduler: PipelineSchedulerT | None
    pipeline_modules: List[Module] | None
    sharing_metadata: SharingMetadataT | None
    shard_index: ShardIndex | None
    stage_indices: Tuple[int, ...] | None
    pp_has_last_stage: bool
    pp_has_first_stage: bool
    attention_mask_creator: Callable

    def __init__(
        self,
        *,
        args: TPipelineTrainingArguments | dict,
        model_splitter: ModelSplitter,  # Required: function to split model into pipeline stages
        pipe_schedule_factory: PipelineSchedulerFactorT = ScheduleGPipe,  # type: ignore[assignment]
        **kwargs,
    ):
        """Initialise the pipeline parallel trainer.

        Parameters
        ----------
        args : PipelineTrainingArguments or dict
            Pipeline training configuration. Dicts are converted via
            ``dacite.from_dict(PipelineTrainingArguments, args)``.
        model_splitter : ModelSplitter
            Callable that accepts the model on the meta device and returns
            all pipeline stage modules, the rank-local stage modules, and
            ``PipelineStage`` objects. See
            ``src/forgather/ml/trainer/pipeline/model_splitter.py`` for the
            full signature.
        pipe_schedule_factory : callable, optional
            Pipeline scheduler factory. ``ScheduleGPipe`` (default) uses simple
            GPipe scheduling. Pass ``ScheduleZBVZeroBubble`` or similar for
            zero-bubble schedules.
        **kwargs
            Forwarded to the base ``Trainer`` constructor (``model_init``,
            ``train_dataset``, ``optimizer_factory``, ``callbacks``, etc.).
        """
        if isinstance(args, dict):
            args = cast(
                TPipelineTrainingArguments, from_dict(PipelineTrainingArguments, args)
            )
        super().__init__(args=args, **kwargs)
        self.model_splitter = model_splitter
        self.pipe_schedule_factory = pipe_schedule_factory

        # Zero-bubble schedules split backward into input-grad and weight-grad
        # steps and call torch.autograd.grad(..., retain_graph=True). That
        # conflicts with donated buffers in torch.compile'd backward
        # (e.g. flex_attention). Disable the optimization before any compiled
        # backward has been captured.
        if is_zero_bubble_schedule(pipe_schedule_factory):
            disable_compiled_backward_donated_buffers()

            # torch.compile applied at stage granularity (see _compile_model)
            # collapses the stage interior into a single Python autograd.Function
            # (CompiledFunctionBackward). Zero-bubble's stage_backward_weight
            # then constructs GradientEdge(intermediate, 0) over those nodes and
            # passes them to torch.autograd.grad, which calls _make_grads ->
            # out.node._input_metadata. That attribute is unavailable on Python
            # autograd.Function nodes and the C++ binding raises:
            #   "Attribute '_input_metadata' is invalid for this instance of
            #    _C._FunctionBase. ... legacy access pattern that is no longer
            #    supported."
            # The crash is structural: AOTAutograd flattens the stage interior
            # so the I/W split has nothing to walk between intermediates and
            # parameters. Refuse the combination at init time with a clear
            # diagnostic instead of letting the user hit it mid-training.
            assert not self.args.torch_compile, (
                "PipelineTrainer does not support torch.compile with zero-bubble "
                "schedules (ScheduleZBVZeroBubble, ScheduleInterleavedZeroBubble). "
                "AOTAutograd wraps each compiled stage in a Python autograd.Function "
                "whose internal nodes do not expose _input_metadata, which the "
                "split-backward weight step requires. Use a non-zero-bubble schedule "
                "(e.g. ScheduleInterleaved1F1B) or disable torch_compile."
            )

        assert self.args.mixed_precision != "fp16", (
            "PipelineTrainer does not support fp16 mixed precision (GradScaler is incompatible "
            "with pipeline scheduler's internal backward). Use mixed_precision='bf16' instead."
        )

        if self.args.debug_pipeline:
            logger.setLevel(logging.DEBUG)

        assert (
            self.model is None
        ), "Pipeline trainer only support model_init=fn, where fn is a zero-args Callable, returning a model"
        assert self.model_init, "Pipeline trainer requires a model_init function"

        for batch_size in (
            self.args.per_device_train_batch_size,
            self.args.per_device_eval_batch_size,
        ):
            assert (
                batch_size % self.args.n_microbatches == 0
            ), f"Batch size ({batch_size}) must be evenly divisible by n_microbatches ({self.args.n_microbatches})"
        assert (
            self.args.is_multistage or self.args.stages_per_rank == 1
        ), "Only multistage schedulers may have more than one stages_per_rank"

        # The pipeline requires a fixed shape for the inputs
        self.args.dataloader_drop_last = True

    @override
    def _is_pipeline_parallel(self) -> bool:
        """Indicate that this trainer uses pipeline parallelism.

        Pipeline parallelism does **not** increase the effective batch size
        (unlike DDP): all microbatches belong to the same original batch and
        flow through the stages sequentially. The base trainer uses this flag
        to skip the DDP-style effective-batch-size scaling.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @override
    def _init_distributed(self):
        self.is_local_process_zero = self.dist.local_rank == 0
        self.is_world_process_zero = self.dist.rank == 0
        self.num_processes = self.dist.world_size

        # Pipeline parallelism is meaningless with a single rank, and downstream
        # code (DataloaderDispatcher pure-MP, pipeline scheduler) will fail in
        # confusing ways if we let this through. Catch it here with a clear
        # diagnostic.
        assert self.dist.world_size > 1, (
            f"PipelineTrainer requires world_size > 1, got {self.dist.world_size}. "
            "Launch with torchrun --nproc-per-node N (N > 1), or verify that any "
            "dynamic-arg override of nproc_per_node is being honored by the "
            "`forgather train` command."
        )

        # Calculate total number of pipeline stages
        self.n_pipeline_stages = self.args.stages_per_rank * self.dist.world_size

        # Create device mesh for pipeline parallel (pure MP - all ranks get same batch)
        # This mesh is used for batch distribution via DataloaderDispatcher
        self.mesh = init_device_mesh(
            self.dist.device_type,
            (self.dist.world_size,),
            mesh_dim_names=("pipeline_parallel",),
        )

        # Create pipeline parallel process group
        # For now, includes all ranks, but this allows future support for
        # hybrid parallelism where PP is a subset of ranks
        self.pp_group = self.mesh.get_group(0)

    def _print_modules(self, modules):
        if self.args.debug_model_params:
            for mod in modules:
                for name, p in mod.named_parameters(remove_duplicate=False):
                    logger.debug(
                        f"P {self.dist.rank} {name} : device {p.device}, dtype {p.dtype}"
                    )
                for name, p in mod.named_buffers(remove_duplicate=False):
                    logger.debug(
                        f"B {self.dist.rank} {name} : device {p.device}, dtype {p.dtype}"
                    )

    @override
    def _wrap(self) -> None:
        """Wrap dataloaders with ``DataloaderDispatcher`` for pipeline batch distribution.

        Pipeline parallelism requires all ranks to process the same batch (pure
        model-parallel mode). Rank 0 loads data from the underlying
        ``DataLoader`` and broadcasts it to all other ranks.

        ``DataloaderDispatcher`` is created with ``dp_mesh_dim=None`` to
        signal pure model-parallelism — no data-parallel dimension exists, so
        rank 0 broadcasts the full batch to every participant.
        """
        if self.train_dataloader:
            self.train_dataloader = DataloaderDispatcher(
                cast(DataLoader, self.train_dataloader),
                self.mesh,
                torch.device(self.dist.device),
                dp_mesh_dim=None,  # Pure MP: all ranks get same batch
            )

        if self.eval_dataloader:
            self.eval_dataloader = DataloaderDispatcher(
                cast(DataLoader, self.eval_dataloader),
                self.mesh,
                torch.device(self.dist.device),
                dp_mesh_dim=None,  # Pure MP: all ranks get same batch
            )

    @override
    def _prepare_model(self):
        """Construct and distribute the model across pipeline stages.

        This is the central setup method for pipeline parallel training. It
        performs the following steps in order:

        1. Construct the full model on the meta device (no memory allocation).
        2. Capture parameter-sharing metadata (tied weights, etc.).
        3. Split the model into pipeline stages via ``model_splitter``.
        4. Materialise each stage's parameters on its assigned device.
        5. Initialise parameters: rank 0 builds a full CPU model and sends
           each rank's stage parameters via point-to-point sends.
        6. Build the shard index for distributed checkpoint save/load.
        7. Construct the pipeline scheduler with the configured microbatch count.
        8. Set up the loss function (only the last stage computes loss).
        9. Enable gradient checkpointing on each stage when requested.

        After this method returns, ``self.pipeline_modules`` contains the
        materialised stage modules for this rank, and ``self.model`` holds the
        original meta-device model for shape/config queries.
        """
        # Reset -- this trainer always resets everything.
        self.scheduler = None
        self.model = None
        self.pipeline_modules = None
        self.optimizer = None
        self.lr_scheduler = None
        self.sharing_metadata = None
        # The pipeline trainer always constructs on the meta device, so the
        # base _prepare post-load step runs _initialize_missing_after_load()
        # (overridden below to init the materialized stages) after the load.
        self._constructed_on_meta = True

        assert self.train_dataloader or self.eval_dataloader

        # Construct model instance on the "meta" device; parameters have meta-data, but no actual data.
        # This allows us to construct a "huge" model, without having to have the memory for it.
        model = self._construct_model(device="meta")
        if self.dist.rank == 0:
            self._print_modules([model])

        # Get parameter sharing metadata
        self.sharing_metadata = create_sharing_metadata(model)

        # stage_indices : A List[Tuple[int]] with the assigned stage indices for each rank
        #   e.g. stage_indices[rank] would have the stage indices for "rank"
        stage_indices = pipeline_stage_indices(
            self.dist.world_size, self.n_pipeline_stages, style=self.args.pp_stage_type
        )

        last_stage_index = self.n_pipeline_stages - 1
        self.stage_indices = stage_indices[self.dist.rank]
        self.pp_has_last_stage = last_stage_index in self.stage_indices
        self.pp_has_first_stage = 0 in self.stage_indices

        # Split model into pipeline segments.
        if self.dist.rank == 0:
            logger.debug(f"All assigned pipeline indices {stage_indices}")
            logger.info("Splitting model...")
        all_pipeline_modules, pipeline_modules, pipeline_stages = self._split_model(
            model, stage_indices, train=True
        )
        # all_pipeline_modules : A list of all modules in the pipeline
        # pipeline_modules : A list of modules assigned to this rank
        # pipeline_stages : A list of pipeline stages assigned to this rank

        # Convert meta tensors to real tensor on assigned devices.
        for mod in pipeline_modules:
            mod.to_empty(device=self.dist.device)
            retie_parameters(mod, self.sharing_metadata)

        # Populate the materialized stages. Initialize from scratch ONLY for a
        # fresh, non-external run: rank-0 builds the full model on CPU just
        # long enough to initialize weights, then distributes each rank's
        # stage parameters (if this OOMs on a very large model, initialize
        # from a checkpoint instead). When resuming, or when weights are
        # external (DiLoCo), the stages stay materialized-empty here — the
        # weights are loaded later by the base _prepare via
        # _restore_from_checkpoint (a checkpoint load and/or the
        # on_load_model_weights hook), and initialize_missing_weights then
        # fills only the rest (e.g. RoPE buffers) per stage. Initializing here
        # would init before that load — the full-init the meta route avoids,
        # and (external) the expensive rank-0 build the parameter server
        # exists to replace.
        if not self.args.resume_from_checkpoint and not self._model_weights_external():
            if self.dist.rank == 0:
                logger.info(
                    "Constructing full model on CPU and distributing initialized parameters from rank0."
                )
            self._initialize_params(
                all_pipeline_modules, pipeline_modules, stage_indices, False
            )

        self._print_modules(pipeline_modules)

        if self.args.fp8_recipe:
            for i, mod in enumerate(pipeline_modules):
                pipeline_modules[i] = self._apply_fp8_training(mod)

        # Construct the pipeline scheduler.
        # Depending upon the class, it either takes a single stage (PipelineScheduleSingle) or a list of stages,
        # PipelineScheduleMulti. See: https://docs.pytorch.org/docs/stable/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle
        if self.args.is_multistage:
            stages_arg = pipeline_stages
        else:
            assert len(pipeline_stages) == 1
            stages_arg = pipeline_stages[0]

        # Make the shard index, which we will need for saving the distribued model.
        # First, assert no duplicate FQNs exist across pipeline modules
        state_dicts = [mod.state_dict() for mod in all_pipeline_modules]
        assert_no_duplicate_fqns(state_dicts)

        self.shard_index = make_shard_index(
            state_dicts,
            safetensors=self.args.save_safetensors,
            param_sharing_metadata=self.sharing_metadata,
        )

        # Only the last state needs to compute loss
        # TODO: Placing this here is not going to work for the auto-splitter. That will require more work...
        # Zero-bubble schedules are incompatible with Python autograd.Function-based
        # fused loss kernels (e.g. Apple CCE's LinearCrossEntropyFunction). Zero-bubble's
        # stage_backward_weight constructs GradientEdge over intermediate nodes and
        # calls torch.autograd.grad -> _make_grads -> out.node._input_metadata, which
        # is not exposed on Python autograd.Function nodes and raises the same legacy
        # accessor error documented for torch.compile above. Disable fused loss in this
        # case so the pipeline falls back to the standard (unfused) loss path.
        if self.pp_has_last_stage and not is_zero_bubble_schedule(
            self.pipe_schedule_factory
        ):
            self.loss_fn = self._maybe_get_fused_loss_fn(
                pipeline_modules[-1], self.loss_fn
            )
        elif self.pp_has_last_stage and self.fused_loss_factory is not None:
            logger.warning(
                "Zero-bubble pipeline schedules are incompatible with Python "
                "autograd.Function-based fused loss kernels. Falling back to the "
                "standard (unfused) loss path."
            )

        # Loss needs to be scaled by the number of micro-batches
        self.loss_fn = RescaleLoss(self.loss_fn, 1 / self.args.n_microbatches)

        # Only the outer wrapper will be disabled for eval
        self.loss_fn = RescaleLoss(
            self.loss_fn, 1 / self.args.gradient_accumulation_steps
        )

        # Note: scale_grads=True (default) rescales gradients in-place during each microbatch step.
        # This breaks gradient accumulation because it rescales the cumulative gradient repeatedly.
        # Instead, we manually scale the loss by 1/n_microbatches above, achieving correct scaling
        # without interfering with gradient accumulation. Set scale_grads=False to disable
        # scheduler's built-in (broken) scaling.
        # This mirrors the fix applied to TorchTitan: https://github.com/pytorch/torchtitan/pull/XXX
        self.scheduler = self.pipe_schedule_factory(
            stages_arg,
            self.args.n_microbatches,
            loss_fn=self.loss_fn,  # type: ignore[call-arg]
            scale_grads=False,
        )

        if self.args.gradient_checkpointing:
            if self.enable_activation_checkpoint_fn is None:
                if self.dist.rank == 0:
                    logger.warning(
                        f"Activation checkpointing requested, but no function defined!"
                    )
            else:
                # Enable activation checkpointing for all modules in the pipeline.
                for mod in pipeline_modules:
                    self.enable_activation_checkpoint_fn(self.dist.rank, mod)

        self.pipeline_modules = pipeline_modules

        for rank in range(len(stage_indices)):
            if last_stage_index in stage_indices[rank]:
                self.pp_last_stage_rank = rank
                break

        # Map every global stage index to the rank that owns it.
        self._stage_to_rank: dict[int, int] = {
            stage: rank_idx
            for rank_idx, stages in enumerate(stage_indices)
            for stage in stages
        }

        # Map global stage index → local pipeline_modules index (for this rank only).
        # pipeline_modules[j] corresponds to stage_indices[rank][j].
        self._stage_to_local_mod: dict[int, int] = {
            stage: idx for idx, stage in enumerate(self.stage_indices)
        }

        # Dtype for activations exchanged between ranks during generation.
        if self.args.mixed_precision == "bf16":
            self._generation_dtype = torch.bfloat16
        elif self.args.mixed_precision == "fp16":
            self._generation_dtype = torch.float16
        else:
            param = next((p for m in pipeline_modules for p in m.parameters()), None)
            self._generation_dtype = param.dtype if param is not None else torch.float32

        # We keep the original model on the meta-device. This model is obviously not functional, but
        # some trainer callbacks may wish to dump the layout.
        self.model = model

        # Compute FLOPs per token using the meta-device model.
        # p.numel() works correctly for meta tensors (shape is defined, data is not).
        self._flops_per_token = self._compute_flops_per_token()
        if self.dist.rank == 0:
            logger.info(f"Estimated FLOPs per token: {self._flops_per_token:.2e}")

    def _construct_model(self, device):
        # Construct model on device
        assert self.model_init
        with ExitStack() as exit_stack:
            exit_stack.enter_context(torch.device(device))
            if self.args.default_dtype:
                exit_stack.enter_context(
                    default_dtype(torch_dtype(self.args.default_dtype))
                )
            model = self.model_init()
        return model

    @override
    def _compile_model(self):
        """Compile each pipeline stage module assigned to this rank with ``torch.compile``.

        Unlike single-device training (where the entire model is compiled as one
        unit), each pipeline stage is compiled independently, which is necessary
        because each stage lives on a different GPU.
        """
        assert self.pipeline_modules
        for mod in self.pipeline_modules:
            mod.compile(
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
                dynamic=self.args.torch_compile_dynamic,
                fullgraph=self.args.torch_compile_full_graph,
            )

    def _split_model(
        self, model, stage_indices, train
    ) -> Tuple[List[Module], List[Module], List[_PipelineStageBase]]:
        """
        Split model into pipeline stages using the injected splitter.

        Delegates to self.model_splitter and captures the attention_mask_creator
        for later use in forward/backward passes.

        Returns modules on meta device - caller must materialize them.
        """
        rank = self.dist.rank

        # Call the injected splitter
        (
            all_pipeline_modules,
            pipeline_modules,
            pipeline_stages,
            attention_mask_creator,
        ) = self.model_splitter(
            model,
            stage_indices,
            train,
            device=self.dist.device,  # type: ignore[call-arg]
            rank=rank,
            pp_group=self.pp_group,
        )

        # Store attention mask creator for use in forward/backward steps
        # Will be None if splitter doesn't support external masks
        self.attention_mask_creator = attention_mask_creator

        if rank == 0 and self.args.debug_split_model:
            logger.debug("Pipeline modules created:")
            for i, mod in enumerate(all_pipeline_modules):
                logger.debug(f"  Stage {i}: {mod}")

        return all_pipeline_modules, pipeline_modules, list(pipeline_stages)

    @torch.no_grad()
    def _initialize_params(
        self, all_pipeline_modules, pipeline_modules, stage_indices, missing_buf_only
    ):
        """Distribute initialised parameters from rank 0 to all other ranks.

        Initialising each rank independently would require building N full models
        in CPU memory (one per rank). Instead, rank 0 constructs a single
        fully-initialised CPU model, copies its own stage parameters locally,
        and sends each other rank's parameters via NCCL point-to-point.

        Process
        -------
        1. Rank 0 constructs the full initialised model on CPU.
        2. Rank 0 copies parameters for its own stages directly to its GPU.
        3. Rank 0 streams each other rank's stage parameters to that rank's GPU
           via ``dist.send`` / ``dist.recv``.
        4. Non-rank-0 ranks call ``dist.recv`` to receive their stage parameters.

        Parameters
        ----------
        all_pipeline_modules : list of torch.nn.Module
            All pipeline stage modules across all ranks (on meta device).
        pipeline_modules : list of torch.nn.Module
            Stage modules assigned to the current rank (on this rank's device
            after ``to_empty``).
        stage_indices : list of tuple of int
            Stage index assignments per rank; ``stage_indices[r]`` is the tuple
            of global stage indices assigned to rank ``r``.
        missing_buf_only : bool
            When ``True``, transfer only non-persistent buffers (used when
            resuming from a checkpoint that omitted those buffers). When
            ``False``, transfer all parameters and buffers.

        Notes
        -----
        Each parameter is temporarily moved to the sender's GPU before calling
        ``dist.send``, because NCCL requires device tensors for transfers.
        """

        def make_state_dict(mod, missing_buf_only):
            """
            Build a state dictionary with /all/ the params/buffers,
            as non-persistent buffers are normally excluded from state_dict()

            missing_buf_only: When True, only include the buffers which are missing from
            the "actual" state_dict.
            """
            output_state_dict = {}
            if missing_buf_only:
                state_dict = mod.state_dict()
                for name, p in mod.named_buffers():
                    if name not in state_dict:
                        output_state_dict[name] = p.data
            else:
                # Include parameter alias names for shared parameters
                for name, p in mod.named_parameters(remove_duplicate=False):
                    output_state_dict[name] = p.data
                for name, p in mod.named_buffers(remove_duplicate=False):
                    output_state_dict[name] = p.data
            return output_state_dict

        if self.dist.rank == 0:
            # Construct a fully initialized model on the CPU, which we will use to distribute
            # initialized parameters.
            logger.debug("Constructing model on CPU")
            initialized_model = self._construct_model(device="cpu")
            init_state_dict = make_state_dict(initialized_model, missing_buf_only)
            # Initialize our own parameters first
            for mod in pipeline_modules:
                for name, p in make_state_dict(mod, missing_buf_only).items():
                    p.copy_(init_state_dict[name])

            logger.debug("Distributing params")
            # Send the initialized parameters for the other stages to their
            # respective processes.
            for dst_rank in range(1, self.dist.world_size):
                # Modules owned by dst_rank
                rank_indices = stage_indices[dst_rank]
                logger.debug(
                    f"rank0: Sending initialized params for stages {rank_indices} to rank{dst_rank}"
                )

                for stage_index in rank_indices:
                    mod = all_pipeline_modules[stage_index]

                    # All params and buffers in destination module
                    for name, _ in make_state_dict(mod, missing_buf_only).items():
                        # NCCL can't send between GPU and GPU, so copy each parameter to
                        # our GPU, send it, then free it. Kind of hack'ish, but it works.
                        # See: https://docs.pytorch.org/docs/stable/distributed.html
                        # I believe this will work for CPU to CPU, with gloo, but
                        # have yet to try it.
                        p = init_state_dict[name].to(self.dist.device)
                        if self.args.debug_model_init:
                            logger.debug(f"rank0: Sending {name} to rank{dst_rank}")
                        distributed.send(p, dst=dst_rank)
                        p = None
        else:
            # Load the parameters from rank 0
            rank_indices = stage_indices[self.dist.rank]

            logger.debug(
                f"rank{self.dist.rank}: Receiving initialized params for stages {rank_indices} from rank0"
            )
            for mod in pipeline_modules:
                for name, p in make_state_dict(mod, missing_buf_only).items():
                    if self.args.debug_model_init:
                        logger.debug(f"rank{self.dist.rank}: Receiving {name}")
                    distributed.recv(p, src=0)

    @override
    def _forward_backward_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Tensor:
        """Execute a combined forward and backward pass via the pipeline scheduler.

        The scheduler handles activation forwarding between stages and gradient
        backpropagation. Each rank's role depends on which stage(s) it hosts:

        * **First stage** — consumes ``input_ids`` and sends activations downstream.
        * **Middle stages** — receive activations, compute, and send downstream.
        * **Last stage** — receives activations, computes the loss, and initiates
          the backward pass.

        Attention masks and ``position_ids`` are constructed outside the pipeline
        and passed as kwargs rather than through the inter-stage activation stream.
        This is necessary because PyTorch's pipeline transport only handles tensors
        that require gradients; Python objects and non-differentiable tensors would
        raise errors if piped between stages.

        Parameters
        ----------
        input_dict : dict of str to Tensor
            Batch inputs. Must contain ``"input_ids"``; may contain
            ``"position_ids"`` when the splitter supports explicit positions.
        labels : Tensor
            Target token ids for loss computation (only consumed by the last stage).

        Returns
        -------
        Tensor
            Sum of per-microbatch losses on the last stage; ``0.0`` on all other
            stages. The caller broadcasts this to all ranks via
            ``_distributed_loss()``.
        """
        inputs = (input_dict["input_ids"],)

        # Create attention mask externally if supported by splitter
        # This follows TorchTitan pattern to avoid pipeline transport issues
        extra_kwargs = {}
        if self.use_fused_loss:
            extra_kwargs["return_hidden_states"] = True

        if self.attention_mask_creator is not None:
            attention_mask = self.attention_mask_creator(**input_dict)
            extra_kwargs["attention_mask"] = attention_mask
            if (position_ids := input_dict.get("position_ids", None)) is not None:
                extra_kwargs["position_ids"] = position_ids

        # See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py#L377
        targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
        assert self.scheduler
        with self.amp_context.autocast():
            if self.pp_has_first_stage:
                self.scheduler.step(
                    *inputs, **extra_kwargs, target=targets, losses=cast(list, losses)
                )
            else:
                self.scheduler.step(
                    **extra_kwargs, target=targets, losses=cast(list, losses)
                )

        if self.pp_has_last_stage:
            assert losses
            mean_loss = torch.stack([x.detach().float() for x in losses]).sum()
        else:
            mean_loss = torch.tensor(0.0, device=self.dist.device, dtype=torch.float32)
        return mean_loss

    @torch.no_grad()
    def _pipeline_step_for_generation(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        """Execute one forward pass through all pipeline stages for text generation.

        All ranks must call this method simultaneously. Works for any value of
        ``stages_per_rank`` and both ``"loop"`` and ``"v"`` stage assignment styles.

        Cross-rank activation transfers use ``dist.batch_isend_irecv`` with
        ``dist.P2POp``, the same primitive the pipeline scheduler uses internally.
        Using the batched async API avoids lazy creation of new per-pair NCCL
        sub-communicators on every call and ensures the two code paths share NCCL
        state cleanly.

        Parameters
        ----------
        input_ids : Tensor
            Token ids of shape ``[batch, seq]``, identical on all ranks.
        attention_mask : Tensor or None
            Optional attention mask created externally (bypasses pipeline transport
            limitations). ``None`` when the splitter does not support external masks.

        Returns
        -------
        Tensor or None
            Logits of shape ``[batch, seq, vocab]`` on the last-stage rank;
            ``None`` on all other ranks.

        Notes
        -----
        Inter-stage activations are assumed to have shape
        ``[batch, seq, model.config.hidden_size]``.
        """
        assert self.pipeline_modules

        forward_kwargs: dict = {}
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask

        hidden_states: Optional[Tensor] = None
        batch, seq = input_ids.shape

        with self.amp_context.autocast():
            for stage_k in range(self.n_pipeline_stages):
                rank_k = self._stage_to_rank[stage_k]

                # At each stage boundary, transfer activations if stages are on different ranks.
                if stage_k > 0:
                    rank_prev = self._stage_to_rank[stage_k - 1]
                    if rank_prev != rank_k:
                        if self.dist.rank == rank_prev:
                            assert hidden_states is not None
                            op = distributed.P2POp(
                                distributed.isend,
                                hidden_states.contiguous(),
                                peer=rank_k,
                                group=self.pp_group,
                            )
                            for w in distributed.batch_isend_irecv([op]):
                                w.wait()
                        elif self.dist.rank == rank_k:
                            recv_buf = torch.empty(
                                batch,
                                seq,
                                self.model.config.hidden_size,
                                dtype=self._generation_dtype,
                                device=self.dist.device,
                            )
                            op = distributed.P2POp(
                                distributed.irecv,
                                recv_buf,
                                peer=rank_prev,
                                group=self.pp_group,
                            )
                            for w in distributed.batch_isend_irecv([op]):
                                w.wait()
                            hidden_states = recv_buf
                        # Other ranks: not involved in this boundary, continue loop.

                # Run the module on its owning rank.
                if self.dist.rank == rank_k:
                    local_idx = self._stage_to_local_mod[stage_k]
                    mod = self.pipeline_modules[local_idx]
                    inp = input_ids if stage_k == 0 else hidden_states
                    hidden_states = mod(inp, **forward_kwargs)

        if self.pp_has_last_stage:
            return hidden_states  # logits tensor from last stage
        return None

    @torch.no_grad()
    def pipeline_generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """Generate text autoregressively through all pipeline stages.

        Bypasses the pipeline scheduler so input shapes are not constrained to
        the fixed training batch dimensions. All ranks must call this method
        simultaneously. The full generated sequence (prompt + new tokens) is
        returned on every rank.

        No KV caching is used; each decoding step reprocesses the entire
        sequence. This is acceptable for infrequent, qualitative generation
        checks (e.g. during a callback).

        Parameters
        ----------
        input_ids : Tensor
            Prompt token ids of shape ``[batch, prompt_len]``, same on all
            ranks.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        eos_token_id : int
            Token id that signals end of sequence. Once all sequences in the
            batch have emitted this token, generation stops early.
        pad_token_id : int
            Token id used to pad sequences that have already finished.
        do_sample : bool, optional
            If ``True``, sample from the probability distribution; if
            ``False``, use greedy (argmax) decoding. Default is ``True``.
        temperature : float, optional
            Softmax temperature applied before top-k filtering. Values ``< 1``
            sharpen the distribution; values ``> 1`` flatten it.
            Default is ``1.0``.
        top_k : int, optional
            When ``> 0``, restrict sampling to the top-k logits. ``0`` uses
            the full vocabulary. Default is ``0``.
        repetition_penalty : float, optional
            Multiplicative penalty applied to logits of tokens already present
            in the sequence. ``1.0`` disables the penalty. Default is ``1.0``.

        Returns
        -------
        Tensor
            Generated token ids of shape ``[batch, prompt_len + n_new_tokens]``
            as a ``LongTensor`` on the current device, identical on all ranks.
        """
        # Ensure all ranks reach this point before issuing any generation collectives.
        # This forces a clean synchronization fence after the trainer's eval phase, so
        # any pending scheduler ops on the same NCCL communicators are guaranteed to be
        # drained before our textgen ops start. Without this, our hand-rolled p2p can
        # get interleaved with leftover scheduler state and deadlock.
        distributed.barrier(group=self.pp_group)

        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.dist.device)

        # Temporarily bypass torch.compile on the pipeline stage modules for the
        # duration of generation. Compiled modules (e.g. flex_attention + max-autotune)
        # are specialized on the training shapes and fail when called with the varying
        # shapes used during autoregressive decoding. Mirrors the single-rank workaround
        # in TextgenCallback.generate().
        assert self.pipeline_modules
        saved_compiled_calls: list = []
        for mod in self.pipeline_modules:
            compiled_call = getattr(mod, "_compiled_call_impl", None)
            saved_compiled_calls.append(compiled_call)
            if compiled_call is not None:
                mod._compiled_call_impl = mod._call_impl

        try:
            for _ in range(max_new_tokens):
                attention_mask = None
                if self.attention_mask_creator is not None:
                    attention_mask = self.attention_mask_creator(
                        input_ids=generated_ids
                    )

                logits = self._pipeline_step_for_generation(
                    generated_ids, attention_mask
                )

                if self.pp_has_last_stage:
                    assert logits is not None
                    next_logits = logits[:, -1, :].float()  # [batch, vocab]

                    # Sanitize logits before sampling. Unstable model output (common
                    # early in training or with mixed precision) can produce NaN/Inf,
                    # which makes torch.multinomial fail the "probability tensor
                    # contains inf, nan or element < 0" assertion. Replace NaN with 0
                    # and clamp ±Inf to finite values. Top-k masking below uses its
                    # own float("-inf") after this step, so -inf handling here does
                    # not interfere with top-k.
                    next_logits = torch.nan_to_num(
                        next_logits, nan=0.0, posinf=1e4, neginf=-1e4
                    )

                    if repetition_penalty != 1.0:
                        for b in range(batch_size):
                            for tok in set(generated_ids[b].tolist()):
                                if next_logits[b, tok] > 0:
                                    next_logits[b, tok] /= repetition_penalty
                                else:
                                    next_logits[b, tok] *= repetition_penalty

                    if temperature != 1.0:
                        next_logits = next_logits / temperature

                    if top_k > 0:
                        top_values, _ = torch.topk(next_logits, top_k, dim=-1)
                        threshold = top_values[:, -1, None]
                        next_logits = next_logits.masked_fill(
                            next_logits < threshold, float("-inf")
                        )

                    if do_sample:
                        probs = torch.softmax(next_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, 1).squeeze(1)
                    else:
                        next_tokens = next_logits.argmax(dim=-1)
                else:
                    next_tokens = torch.zeros(
                        batch_size, dtype=torch.long, device=self.dist.device
                    )

                distributed.broadcast(next_tokens, src=self.pp_last_stage_rank)

                done = done | (next_tokens == eos_token_id)
                next_tokens = next_tokens.masked_fill(done, pad_token_id)
                generated_ids = torch.cat(
                    [generated_ids, next_tokens.unsqueeze(1)], dim=1
                )

                # Broadcast early-exit flag from last-stage rank to all ranks.
                if self.pp_has_last_stage:
                    stop = torch.tensor(
                        [int(done.all())], dtype=torch.long, device=self.dist.device
                    )
                else:
                    stop = torch.zeros(1, dtype=torch.long, device=self.dist.device)
                distributed.broadcast(stop, src=self.pp_last_stage_rank)
                if stop.item():
                    break
        finally:
            # Restore compiled forwards for training, even if generation raised.
            for mod, compiled_call in zip(self.pipeline_modules, saved_compiled_calls):
                if compiled_call is not None:
                    mod._compiled_call_impl = compiled_call

        return generated_ids

    @override
    def _init_optimizer(self):
        """Initialise the optimizer over all pipeline stage parameters on this rank.

        Collects parameters from every module in ``self.pipeline_modules`` (i.e.
        all stages assigned to this rank) and passes them to
        ``self.optimizer_factory`` to create a single optimizer instance. Also
        registers post-accumulate gradient hooks when
        ``args.fuse_optim_with_backward`` is enabled.
        """
        if self.optimizer is None:
            # Build a named-parameter generator for all of our modules
            def named_parameters(modules):
                for mod in modules:
                    for param in mod.named_parameters():
                        yield param

            assert self.pipeline_modules
            assert self.optimizer_factory
            if self.optimizer_groups is not None:
                opt_params = build_parameter_groups(
                    named_parameters(self.pipeline_modules),
                    self.optimizer_groups,
                    debug=self.args.debug_optimizer_groups,
                )
            else:
                opt_params = named_parameters(self.pipeline_modules)
            self.optimizer = self.optimizer_factory(opt_params)

            if self.args.fuse_optim_with_backward:
                self._total_grad_squared = torch.zeros(
                    1, device=self.args.device, dtype=torch.float32
                )

                for name, p in named_parameters(self.pipeline_modules):
                    if p.requires_grad:
                        hook = partial(
                            optimizer_hook,
                            self.optimizer,
                            self._total_grad_squared,
                            name,
                        )
                        p.register_post_accumulate_grad_hook(hook)

        if self.lr_scheduler is None and self.lr_scheduler_factory is not None:
            self.lr_scheduler = self.lr_scheduler_factory(self.optimizer)

    @contextmanager
    def _eval_schedule_context(self):
        """Temporarily strip backward actions from a pre-computed pipeline schedule.

        Workaround for PyTorch issue: schedules that inherit from
        _PipelineScheduleRuntime (e.g. ScheduleZBVZeroBubble) pre-compute
        pipeline_order_with_comms at __init__ time, including backward actions.
        During eval(), _has_backward is set to False and stage.backward_one_chunk
        early-returns, but the V-schedule special-case code still calls
        stage.get_local_bwd_output() which asserts has_backward, causing a crash.

        This context manager removes backward-related actions (BACKWARD_INPUT,
        BACKWARD_WEIGHT, FULL_BACKWARD, SEND_B, RECV_B, REDUCE_GRAD) from the
        schedule for the duration of eval, then restores the original.
        """
        from torch.distributed.pipelining.schedules import _ComputationType

        _BACKWARD_ACTIONS = frozenset(
            {
                _ComputationType.BACKWARD_INPUT,
                _ComputationType.BACKWARD_WEIGHT,
                _ComputationType.FULL_BACKWARD,
                _ComputationType.SEND_B,
                _ComputationType.RECV_B,
                _ComputationType.REDUCE_GRAD,
            }
        )

        scheduler = self.scheduler
        original = getattr(scheduler, "pipeline_order_with_comms", None)
        if original is not None:
            setattr(
                scheduler,
                "pipeline_order_with_comms",
                {
                    rank: [
                        a
                        for a in actions
                        if a.computation_type not in _BACKWARD_ACTIONS
                    ]
                    for rank, actions in original.items()
                },
            )
        try:
            yield
        finally:
            if original is not None:
                setattr(scheduler, "pipeline_order_with_comms", original)

    @override
    def _prediction_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Dict[str, Tensor | None]:
        """
        Execute evaluation forward pass through pipeline scheduler.

        Uses scheduler.eval() instead of scheduler.step() to disable gradient computation
        and skip backward pass. Loss is still computed on last stage for metrics.

        Similar to _forward_backward_step but without gradients. Creates attention masks
        externally for same reasons (PyTorch pipeline transport limitations).

        Parameters
        ----------
        input_dict : dict of str to Tensor
            Batch inputs with ``"input_ids"``.
        labels : Tensor
            Target labels for loss computation (only used by last stage).

        Returns
        -------
        dict
            Dictionary with ``"loss"`` (mean over microbatches), ``"logits"``
            (``None``), and ``"labels"`` (``None``).
        """
        inputs = (input_dict["input_ids"],)

        # Create attention mask externally if supported by splitter
        # This follows TorchTitan pattern to avoid pipeline transport issues
        extra_kwargs = {}
        if self.use_fused_loss:
            extra_kwargs["return_hidden_states"] = True
        if self.attention_mask_creator is not None:
            attention_mask = self.attention_mask_creator(
                input_ids=input_dict["input_ids"]
            )
            extra_kwargs["attention_mask"] = attention_mask

        targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
        assert self.scheduler
        loss_fn = self.loss_fn
        assert isinstance(loss_fn, RescaleLoss)
        with loss_fn.no_rescale(), self.amp_context.autocast():
            with self._eval_schedule_context():
                if self.pp_has_first_stage:
                    self.scheduler.eval(
                        *inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=cast(list, losses),
                    )
                else:
                    self.scheduler.eval(
                        **extra_kwargs,
                        target=targets,
                        losses=cast(list, losses),
                    )

        # Compute loss on last stage
        if self.pp_has_last_stage:
            assert losses
            mean_loss = torch.stack([x.detach().float() for x in losses]).sum()
        else:
            mean_loss = torch.tensor(0.0, device=self.dist.device, dtype=torch.float32)

        mean_loss = self._distributed_loss(mean_loss)
        return {
            "loss": mean_loss,
            "logits": None,
            "labels": None,
        }

    @override
    def _materialized_modules(self) -> List[torch.nn.Module]:
        """The on-device stages holding this rank's real parameters.

        ``self.model`` is the meta skeleton (kept only for shape/config
        queries); the materialized weights live in ``self.pipeline_modules``.
        Drives both the post-load init and the external-weights verification.

        Note for the init pass: split stage modules don't expose an HF-style
        ``_init_weights``, so ``initialize_missing_weights`` takes its
        fallback path — reset only modules that own a non-persistent buffer
        (e.g. RoPE ``inv_freq``, never in the checkpoint). That recompute is
        local and deterministic per module, so every rank produces the same
        values; loaded tensors are flagged so the fallback skips them.
        """
        return list(self.pipeline_modules or [])

    def _init_checkpoint_manager(self) -> CheckpointManager:
        """
        Initialize checkpoint manager for distributed pipeline parallel model.

        Unlike single-device trainer, pipeline trainer needs to save model shards
        across all ranks since each rank only has a portion of the model. The
        shard_index tracks which parameters belong to which rank for coordinated
        save/load operations.

        Sets save_on_all_ranks=True so all ranks participate in checkpointing,
        each saving their own pipeline stages.

        Returns
        -------
        CheckpointManager
            Configured for distributed pipeline model saving.
        """
        cp_config = CheckpointConfig(
            output_dir=self.args.output_dir,
            save_total_limit=self.args.save_total_limit,
            save_on_each_node=self.args.save_on_each_node,
            save_safetensors=self.args.save_safetensors,
            save_on_all_ranks=True,
        )

        assert self.model
        assert self.shard_index
        checkpoint_manager = CheckpointManager(
            config=cp_config,
            dist=self.dist,
            model=self.model,
            model_parts=self.pipeline_modules,
            model_preprocessor=self.processing_class,
            stateful_provider=self,
            shard_index=self.shard_index,
        )
        return checkpoint_manager

    @staticmethod
    def _all_reduce_norm(total_norm, norm_type):
        """All-reduce local gradient norms to compute the global norm.

        Each rank has computed a local norm over its pipeline stage parameters.
        This method combines those local norms into the true global gradient norm
        using the appropriate reduction for the requested norm type:

        * **L2** (``norm_type=2``) — sum squared local norms, then take sqrt.
        * **Lp** — sum ``p``-th powers, then take the ``1/p`` root.
        * **L-inf** — max across ranks.

        Parameters
        ----------
        total_norm : Tensor
            Local norm tensor on this rank (scalar).
        norm_type : float
            Exponent of the norm. Use ``2.0`` for L2, ``float("inf")`` for
            L-inf.

        Returns
        -------
        Tensor
            Global gradient norm after all-reduce, same value on all ranks.
        """
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX)
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
            total_norm **= 1.0 / norm_type
        return total_norm

    @override
    def _clip_grad_norm(self, max_grad_norm, norm_type=2.0) -> Tensor:
        """Compute and optionally clip the gradient norm across all pipeline stages.

        Unlike the single-device trainer, each rank holds gradients only for its
        own pipeline stages, so the local norms must be all-reduced to obtain the
        true global norm before clipping.

        Parameters
        ----------
        max_grad_norm : float or None
            Maximum allowed gradient norm. When ``None`` or ``0``, the norm is
            computed but no clipping is applied.
        norm_type : float, optional
            Type of norm. Default is ``2.0`` (L2 norm).

        Returns
        -------
        Tensor
            Global gradient norm (after all-reduce across pipeline ranks).
        """
        # If fused optimizer, we can't clip, but we can compute the value,
        # which we do from the tensor callacks
        if self.args.fuse_optim_with_backward:
            # Apply sqrt, as we accumulate the sum of the squares
            total_norm = self._total_grad_squared.sqrt()
            self._total_grad_squared -= self._total_grad_squared
            # Collective all-reduce with other ranks
            return self._all_reduce_norm(total_norm, norm_type)

        # Compute norm over all local trainable parameters
        assert self.pipeline_modules

        if False:
            sum = None
            for i, mod in enumerate(self.pipeline_modules):
                for name, p in mod.named_parameters():
                    if p.grad is not None:
                        grad = p.grad
                        norm = grad.square().sum().sqrt()
                        logger.info(f"r{self.dist.rank} m{i} {name} {norm}")

        parameters = [
            p
            for mod in self.pipeline_modules
            for p in mod.parameters()
            if p.grad is not None
        ]

        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type=norm_type, foreach=True
        )

        # All-reduce over all ranks
        total_norm = self._all_reduce_norm(total_norm, norm_type)

        if max_grad_norm is None or max_grad_norm == 0:
            return total_norm

        torch.nn.utils.clip_grads_with_norm_(
            parameters,
            max_grad_norm,
            total_norm,
            foreach=True,
        )

        return total_norm

    @override
    def _count_batch_tokens(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Tensor:
        """Count the number of tokens in the current batch for the first pipeline stage.

        Only the first stage receives ``input_ids`` and meaningful labels; all
        other stages return a zero tensor. ``_distributed_tokens()`` then sums
        across ranks to obtain the true batch token count.

        Parameters
        ----------
        input_dict : dict of str to Tensor
            Batch dictionary; ``"input_ids"`` is only populated on the first
            stage.
        labels : Tensor
            Target labels (meaningful only on the first stage).

        Returns
        -------
        Tensor
            Scalar token count on the first stage; zero tensor on other stages.
        """
        if not self.pp_has_first_stage:
            return torch.tensor(0, device=self.args.device, dtype=torch.int64)
        return super()._count_batch_tokens(input_dict, labels)

    @override
    def _distributed_tokens(self, tokens: Tensor) -> Tensor:
        """All-reduce token counts across pipeline stages.

        The first stage contributes the real count; all other stages contribute
        zero. Summing across ranks yields the total token count for the batch.

        Parameters
        ----------
        tokens : Tensor
            Local token count (non-zero only on the first stage).

        Returns
        -------
        Tensor
            Total token count summed across all pipeline ranks.
        """
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        return tokens

    @override
    def _distributed_loss(self, loss: Tensor):
        """Broadcast the loss from the last pipeline stage to all ranks.

        Only the last stage computes a meaningful loss (it alone has the labels).
        All other stages hold ``0.0``. Broadcasting from the last-stage rank
        ensures every rank can log the same loss value.

        Parameters
        ----------
        loss : Tensor
            Scalar loss tensor — meaningful only on the last stage, ``0.0``
            on all other stages.

        Returns
        -------
        Tensor
            Loss value from the last stage, same on all ranks after broadcast.
        """
        distributed.broadcast(loss, src=self.pp_last_stage_rank)
        return loss

    @override
    def _distributed_peak_mem(self, local_peak: int) -> list[int]:
        """All-gather per-rank peak CUDA memory across all pipeline ranks.

        Because each pipeline stage hosts a different subset of the model's
        layers, the memory footprint per rank can differ significantly. The
        per-rank peaks are therefore genuinely informative for capacity
        planning and imbalance detection.

        Parameters
        ----------
        local_peak : int
            Peak CUDA memory allocated on this rank, in bytes.

        Returns
        -------
        list of int
            Peak memory in bytes for each rank, indexed by rank.
        """
        value = torch.tensor(
            [int(local_peak)], dtype=torch.long, device=self.args.device
        )
        gathered = [torch.zeros_like(value) for _ in range(self.dist.world_size)]
        dist.all_gather(gathered, value)
        return [int(t.item()) for t in gathered]

    @override
    def get_state_components(self) -> List[StateComponent]:
        """Return state components for pipeline parallel training.

        Because the model is split across ranks, each rank saves only its own
        stage parameters. The sharing patterns reflect this:

        * ``"model"`` — PER_RANK (each rank holds different stages), required.
        * ``"optimizer"`` — PER_RANK (optimises different parameters), optional.
        * ``"scheduler"`` — REPLICATED (same LR schedule on all ranks), optional.
        * ``"trainer"`` — REPLICATED (same global step on all ranks), optional.
        * ``"dataset"`` — GLOBAL (``DataloaderDispatcher`` with
          ``dp_mesh_dim=None``; rank 0 loads and broadcasts), optional.
        * ``"rng"`` — PER_RANK (each stage may have different dropout), optional.

        Returns
        -------
        list of StateComponent
            All checkpointable state components with their sharing patterns.
        """
        components = []

        # Model - REQUIRED, PER_RANK (different pipeline stages per rank)
        # pipeline_modules contains the stage modules assigned to this rank
        if self.pipeline_modules:
            components.append(
                StateComponent(
                    key="model",
                    stateful=cast(Stateful, self.pipeline_modules),
                    sharing_pattern=SharingPattern.PER_RANK,
                    required=True,  # Model is always required
                )
            )

        # Optimizer - optional, PER_RANK (optimizes different parameters per rank)
        # Each rank's optimizer only contains parameters for its pipeline stages
        if self.optimizer:
            components.append(
                StateComponent(
                    key="optimizer",
                    stateful=self.optimizer,
                    sharing_pattern=SharingPattern.PER_RANK,
                    required=False,
                )
            )

        # LR Scheduler - optional, REPLICATED (same schedule across all ranks)
        # All ranks follow the same learning rate schedule
        if self.lr_scheduler:
            components.append(
                StateComponent(
                    key="scheduler",
                    stateful=self.lr_scheduler,
                    sharing_pattern=SharingPattern.REPLICATED,
                    required=False,
                )
            )

        # Trainer state - optional, REPLICATED (same global step across all ranks)
        # Training progress is synchronized across all pipeline stages
        components.append(
            StateComponent(
                key="trainer",
                stateful=self,
                sharing_pattern=SharingPattern.REPLICATED,
                required=False,
            )
        )

        # Dataset state - optional, GLOBAL (DataloaderDispatcher with pure MP mode)
        # Rank 0 loads data and broadcasts to all ranks (dp_mesh_dim=None)
        if hasattr(self.train_dataloader, "state_dict"):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=cast(Stateful, self.train_dataloader),
                    sharing_pattern=self._get_dataset_sharing_pattern(),
                    required=False,
                )
            )

        # RNG state - optional, PER_RANK (each rank needs different random numbers)
        # Different dropout patterns, etc. for different pipeline stages
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
        """Return the dataset sharing pattern for pipeline parallel training.

        ``PipelineTrainer`` uses ``DataloaderDispatcher`` with
        ``dp_mesh_dim=None`` (pure model-parallel mode). Rank 0 loads the data
        and broadcasts it to all other ranks, so only one dataloader state
        exists globally.

        Returns
        -------
        SharingPattern
            ``SharingPattern.GLOBAL``.
        """
        # Pure model parallelism: all ranks get same batch from rank 0
        # See _wrap() method where DataloaderDispatcher is created with dp_mesh_dim=None
        return SharingPattern.GLOBAL

    @override
    def get_process_groups(self) -> Dict[str, Any]:
        """Return named process groups for checkpoint coordination.

        The checkpoint manager uses this mapping to implement ``PER_GROUP``
        sharing patterns (e.g. saving one copy per pipeline-parallel group).

        Returns
        -------
        dict of str to ProcessGroup
            ``{"pp_group": self.pp_group}`` for pure pipeline parallelism.
        """
        return {
            "pp_group": self.pp_group,
        }
