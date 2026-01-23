# https://github.com/pytorch/pytorch/tree/main/torch/distributed/pipelining
import logging
import math
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Protocol,
    Tuple,
    TypeAlias,
    override,
)

import torch
import torch.distributed as dist
from torch import Tensor, distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import ScheduleGPipe
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.nn import Module

from forgather.ml.construct import torch_dtype
from forgather.ml.loss import RescaleLoss
from forgather.ml.utils import default_dtype

from ...sharded_checkpoint import (
    ShardIndex,
    SharingMetadataT,
    create_sharing_metadata,
    make_shard_index,
    retie_parameters,
)
from ..checkpoint_manager import CheckpointConfig, CheckpointManager
from ..dataloader_dispatcher import DataloaderDispatcher
from ..trainer import Trainer, TrainingArguments, optimizer_hook
from ..trainer_types import CheckpointInterface, LossFunctionT
from .model_splitter import ModelSplitter
from .pipeline_fixes import assert_no_duplicate_fqns
from .pipeline_utils import missing_buffers, pipeline_stage_indices

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
    def step(self, *args, targets: torch.Tensor | None, losses: list, **kwargs):
        pass


PipelineSchedulerFactorT: TypeAlias = Callable[
    [int, int, LossFunctionT, bool], PipelineSchedulerT
]


@dataclass(kw_only=True)
class PipelineTrainingArguments(TrainingArguments):
    """
    Training arguments for pipeline parallel training.

    Pipeline parallelism splits a model across multiple GPUs, with each GPU handling
    one or more stages. Batches are divided into microbatches that flow through the
    pipeline, enabling overlapped computation across stages to improve GPU utilization.

    See PyTorch pipeline parallelism docs:
    https://docs.pytorch.org/docs/stable/distributed.pipelining.html

    Args:
        n_microbatches: Number of microbatches to split each batch into. More microbatches
            improve pipeline efficiency by keeping all stages busy, but increase memory usage.
            Batch size must be evenly divisible by n_microbatches.
            Typical values: 4-16 depending on pipeline depth and memory constraints.

        stages_per_rank: Number of pipeline stages per GPU. Most schedulers use 1 stage per rank.
            Multi-stage schedulers (like ZeroBubble) can use multiple stages per rank to reduce
            pipeline bubbles. Only set > 1 when using is_multistage schedulers.

        pp_stage_type: Stage assignment pattern across ranks:
            - "loop": Round-robin assignment (e.g., 4 stages, 2 ranks: rank0=[0,2], rank1=[1,3])
            - "v": V-pattern for ZeroBubble scheduler (see https://arxiv.org/pdf/2401.10241)
            Default "loop" works for most cases.

        is_multistage: Whether using multi-stage scheduler API (multiple stages per rank).
            Set to True when using schedulers like ScheduleZBVZeroBubble that inherit from
            PipelineScheduleMulti. Leave False for single-stage schedulers (ScheduleGPipe, etc.).

        debug_pipeline: Internal development flag (not part of stable API).
        debug_split_model: Internal development flag (not part of stable API).
        debug_model_params: Internal development flag (not part of stable API).
        debug_model_init: Internal development flag (not part of stable API).

    Note: model_splitter is passed to PipelineTrainer.__init__(), not here, since
    it's a callable rather than a primitive type.
    """

    debug_pipeline: bool = False
    debug_split_model: bool = False
    debug_model_params: bool = False
    debug_model_init: bool = False
    n_microbatches: int = 4
    stages_per_rank: int = 1
    pp_stage_type: str = "loop"
    is_multistage: bool = False


class PipelineTrainer(Trainer):
    """
    Trainer for pipeline parallel training using PyTorch distributed pipelining.

    Pipeline parallelism splits a model across multiple GPUs where each GPU handles
    one or more stages (layers) of the model. Input batches are divided into microbatches
    that flow through the pipeline stages sequentially, with multiple microbatches
    in flight simultaneously to keep all GPUs busy and maximize utilization.

    Key differences from single-device Trainer:
    - Model construction on meta device, then materialized per-stage
    - Rank 0 initializes full model and distributes parameters to avoid redundant init
    - Rank 0 broadcasts batches to all ranks (only first/last stages need data)
    - Custom gradient computation and loss reduction across pipeline stages
    - Effective batch size doesn't scale with num_processes (same batch flows through pipeline)

    Pipeline stages must be defined by providing a ModelSplitter function that splits
    the model into stages and creates PipelineStage objects.

    Example usage:
        from torch.distributed.pipelining import ScheduleGPipe

        args = PipelineTrainingArguments(
            n_microbatches=8,
            per_device_train_batch_size=64,  # Must be divisible by n_microbatches
            stages_per_rank=1,
        )

        trainer = PipelineTrainer(
            args=args,
            model_init=model_factory,
            model_splitter=my_splitter_function,
            pipe_schedule_factory=ScheduleGPipe,
            train_dataset=train_dataset,
            optimizer_factory=optimizer_factory,
        )
        trainer.train()

    See:
    - PyTorch pipeline docs: https://docs.pytorch.org/docs/stable/distributed.pipelining.html
    - ModelSplitter signature: src/forgather/ml/trainer/pipeline/model_splitter.py
    """

    args: PipelineTrainingArguments
    model_splitter: ModelSplitter
    pipe_schedule_factory: PipelineSchedulerFactorT
    pp_group: Any
    n_pipeline_stages: int
    scheduler: PipelineSchedulerT | None
    pipeline_modules: List[Module] | None
    sharing_metadata: SharingMetadataT | None
    shard_index: ShardIndex | None
    stage_indices: None
    pp_has_last_stage: bool
    pp_has_first_stage: bool
    attention_mask_creator: Callable

    def __init__(
        self,
        *,
        args: PipelineTrainingArguments,
        model_splitter: ModelSplitter,  # Required: function to split model into pipeline stages
        pipe_schedule_factory: PipelineSchedulerT = ScheduleGPipe,
        **kwargs,
    ):
        """
        Initialize pipeline parallel trainer.

        Args:
            args: Pipeline training configuration
            model_splitter: Function to split model into pipeline stages. Must match ModelSplitter
                signature (see src/forgather/ml/trainer/pipeline/model_splitter.py).
                Takes model on meta device and returns stage modules and PipelineStage objects.
            pipe_schedule_factory: Pipeline scheduler factory (e.g., ScheduleGPipe, ScheduleZBVZeroBubble).
                Default ScheduleGPipe uses simple GPipe scheduling with gradient accumulation.
            **kwargs: Additional arguments passed to base Trainer (train_dataset, optimizer_factory, etc.)
        """
        assert isinstance(args, PipelineTrainingArguments)
        self.args = args  # For type checking hint
        self.model_splitter = model_splitter
        self.pipe_schedule_factory = pipe_schedule_factory
        super().__init__(args=args, **kwargs)

    @override
    def _is_pipeline_parallel(self) -> bool:
        """
        Indicate this trainer uses pipeline parallelism.

        Pipeline parallelism doesn't increase effective batch size (unlike DDP) because
        the same batch flows through all stages sequentially - different microbatches are
        in different stages at any given time, but they all belong to the same original batch.

        Returns:
            True to indicate pipeline parallel training
        """
        return True

    @override
    def _post_init(self) -> None:
        if self.args.debug_pipeline:
            logger.setLevel(logging.DEBUG)
        super()._post_init()
        assert self.model is None
        assert self.model_init

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
    def _init_distributed(self):
        self.is_local_process_zero = self.dist.local_rank == 0
        self.is_world_process_zero = self.dist.rank == 0
        self.num_processes = self.dist.world_size

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
        """
        Wrap dataloaders for pipeline parallel batch distribution.

        Pipeline parallelism requires all ranks to receive the same batch (pure MP mode),
        since different stages process different parts of the same batch. Rank 0 loads
        data and broadcasts to all other ranks.

        Uses DataloaderDispatcher with dp_mesh_dim=None for pure model-parallel mode,
        which is equivalent to the previous broadcast-based _dataloader_iter approach
        but with a more unified API consistent with other trainers.
        """
        if self.train_dataloader:
            self.train_dataloader = DataloaderDispatcher(
                self.train_dataloader,
                self.mesh,
                self.dist.device,
                dp_mesh_dim=None,  # Pure MP: all ranks get same batch
            )

        if self.eval_dataloader:
            self.eval_dataloader = DataloaderDispatcher(
                self.eval_dataloader,
                self.mesh,
                self.dist.device,
                dp_mesh_dim=None,  # Pure MP: all ranks get same batch
            )

    @override
    def _prepare_model(self):
        """
        Prepare model for pipeline parallel training.

        This is the main setup method that:
        1. Constructs full model on meta device (no memory allocation)
        2. Captures parameter sharing metadata (for tied weights)
        3. Splits model into pipeline stages using model_splitter
        4. Materializes each stage's parameters on its assigned device
        5. Initializes parameters (rank 0 broadcasts to other ranks)
        6. Creates pipeline scheduler with configured microbatches
        7. Sets up loss function (only on last stage)
        8. Enables gradient checkpointing if requested

        The model remains on meta device for reference; actual computation happens
        through pipeline_modules (the materialized stages).
        """
        # Reset -- this trainer always resets everything.
        self.scheduler = None
        self.model = None
        self.pipeline_modules = None
        self.optimizer = None
        self.lr_scheduler = None
        self.sharing_metadata = None

        assert self.train_dataloader or self.eval_dataloader

        # Construct model instance on the "meta" device; parameters have meta-data, but no actual data.
        # This allows us to construct a "huge" model, without having to have the memory for it.
        model = self._construct_model(device="meta")
        if self.dist.rank == 0:
            self._print_modules([model])

        # Get parameter sharing metadata
        self.sharing_metadata = create_sharing_metadata(model)

        # Get a micro-batch from the train_dataloader to use for tracing.
        dataloader = (
            self.train_dataloader if self.train_dataloader else self.eval_dataloader
        )
        example_args, example_kwargs = self._get_example(dataloader)

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
            model, example_args, example_kwargs, stage_indices, train=True
        )
        # all_pipeline_modules : A list of all modules in the pipeline
        # pipeline_modules : A list of modules assigned to this rank
        # pipeline_stages : A list of pipeline stages assigned to this rank

        # Convert meta tensors to real tensor on assigned devices.
        for mod in pipeline_modules:
            mod.to_empty(device=self.dist.device)
            retie_parameters(mod, self.sharing_metadata)

        # Load from checkpoint?
        if self.args.resume_from_checkpoint:
            missing_buffer_set = missing_buffers(model)
            if len(missing_buffer_set):
                if self.dist.rank == 0:
                    logger.warning(
                        f"The following buffers were not found in the model's state_dict: {missing_buffer_set}. "
                        "Forcing initialization of full model on CPU to construct missing buffers. "
                        "To avoid this, make sure all the model's buffers have 'persist' set to True."
                    )
                self._initialize_params(
                    all_pipeline_modules, pipeline_modules, stage_indices, True
                )
        else:
            if self.dist.rank == 0:
                # If this results in OOM (really large model), you will have to initialize the model from a checkpoint
                # which will likely entail some amount of work.
                logger.info(
                    "Constructing full model on CPU and distributing initialized parameters from rank0."
                )
            self._initialize_params(
                all_pipeline_modules, pipeline_modules, stage_indices, False
            )

        self._print_modules(pipeline_modules)

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
        if self.pp_has_last_stage:
            self.loss_fn = self._maybe_get_fused_loss_fn(
                pipeline_modules[-1], self.loss_fn
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
            loss_fn=self.loss_fn,
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

        # We keep the original model on the meta-device. This model is obviously not functional, but
        # some trainer callbacks may wish to dump the layout.
        self.model = model

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
        """
        Compile all pipeline stage modules assigned to this rank.

        Each pipeline stage is compiled independently with torch.compile().
        This is different from single-device training where the entire model
        is compiled as one unit.
        """
        assert self.pipeline_modules
        for mod in self.pipeline_modules:
            mod.compile(
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
                dynamic=self.args.torch_compile_dynamic,
                fullgraph=self.args.torch_compile_full_graph,
            )

    def _get_example(self, example_dataloader):
        """
        Get example microbatch for model tracing during pipeline stage creation.

        Pipeline parallel requires all batches to have identical shapes. This creates
        a meta-device tensor matching the shape of actual batches, then splits it into
        microbatches for tracing the model splitter.

        Note: Currently hardcoded to use "input_ids" as the main input tensor.

        Args:
            example_dataloader: Dataloader to extract shape information from

        Returns:
            Tuple of (example_args, example_kwargs) for a single microbatch on meta device
        """
        # Note that pipeline parallel requires all batches to have the same shape!
        # TODO: We have hard-coded "input_ids" This should be more flexible, as this is not always the case.
        example_batch = next(iter(example_dataloader))
        example_args = (torch.empty_like(example_batch["input_ids"], device="meta"),)
        example_kwargs = dict(
            #    input_ids=torch.empty_like(
            #        example_batch["input_ids"],
            #        device="meta"
            #    ),
        )

        # Split into microbatches
        split_args, split_kwargs = split_args_kwargs_into_chunks(
            example_args, example_kwargs, chunks=self.args.n_microbatches
        )

        # Return example micro-batches
        return split_args[0], split_kwargs[0]

    def _split_model(
        self, model, example_args, example_kwargs, stage_indices, train
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
            example_args,
            example_kwargs,
            stage_indices,
            train,
            device=self.dist.device,
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
        """
        Distribute initialized parameters from rank 0 to all other ranks.

        This is more efficient than each rank initializing the full model independently:
        - Memory: Avoids N copies of full model in CPU memory (one per rank)
        - Compute: Avoids redundant initialization computation on each rank
        - Simplicity: Each rank only needs to receive its stage's parameters

        Process:
        1. Rank 0: Constructs full initialized model on CPU
        2. Rank 0: Copies parameters for its own stages to device
        3. Rank 0: Sends each other rank's stage parameters directly to their device
        4. Other ranks: Receive and load their stage parameters

        Note: Uses point-to-point send/recv via NCCL. Each parameter is temporarily
        moved to GPU for transmission since NCCL requires device tensors.

        Args:
            all_pipeline_modules: All pipeline stage modules across all ranks
            pipeline_modules: Stage modules assigned to current rank
            stage_indices: Stage index assignments for all ranks
            missing_buf_only: If True, only initialize/transfer non-persistent buffers
                (optimization when loading from checkpoint that lacks some buffers)

        TODO: Optimize missing_buf_only case to only transfer missing buffers, not all params.
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
        """
        Execute forward and backward passes through the pipeline scheduler.

        The pipeline scheduler handles forwarding activations between stages and
        backpropagating gradients. Different ranks participate differently:
        - First stage: Receives input_ids, passes activations downstream
        - Middle stages: Receive activations, compute, pass downstream
        - Last stage: Receives activations, computes loss, backpropagates gradients

        Attention masks and position_ids are created externally (not passed through pipeline)
        because PyTorch pipeline can only transport tensors that require gradients. Non-gradient
        tensors and Python objects (like FlexAttention masks) would cause errors.
        See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py#L377

        Args:
            input_dict: Batch inputs with 'input_ids' and optionally 'position_ids'
            labels: Target labels for loss computation (only used by last stage)

        Returns:
            Mean loss summed over microbatches (0.0 on non-last stages, broadcast later)
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
        if self.pp_has_first_stage:
            self.scheduler.step(*inputs, **extra_kwargs, target=targets, losses=losses)
        else:
            self.scheduler.step(**extra_kwargs, target=targets, losses=losses)

        if self.pp_has_last_stage:
            assert losses
            mean_loss = torch.stack([x.detach().float() for x in losses]).sum()
        else:
            mean_loss = torch.tensor(0.0, device=self.dist.device, dtype=torch.float32)
        return mean_loss

    @override
    def _init_optimizer(self):
        """
        Initialize optimizer over parameters from all pipeline stages on this rank.

        Collects parameters from all pipeline_modules (stages) assigned to this rank
        and creates a single optimizer instance. Also sets up fused optimizer hooks
        if fuse_optim_with_backward is enabled.
        """
        if self.optimizer is None:
            # Build a named-parameter generator for all of our modules
            def named_parameters(modules):
                for mod in modules:
                    for param in mod.named_parameters():
                        yield param

            assert self.pipeline_modules
            assert self.optimizer_factory
            self.optimizer = self.optimizer_factory(
                named_parameters(self.pipeline_modules)
            )

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
            self.lr_scheduler = self.lr_scheduler_factory(
                optimizer=self.optimizer,
            )

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

        Args:
            input_dict: Batch inputs with 'input_ids'
            labels: Target labels for loss computation (only used by last stage)

        Returns:
            Dictionary with 'loss' (mean over microbatches), 'logits' (None), 'labels' (None)
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
        with self.loss_fn.no_rescale():
            if self.pp_has_first_stage:
                self.scheduler.eval(*inputs, **extra_kwargs)
            else:
                self.scheduler.eval(**extra_kwargs, target=targets, losses=losses)

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
    def _init_checkpoint_manager(self) -> CheckpointInterface:
        """
        Initialize checkpoint manager for distributed pipeline parallel model.

        Unlike single-device trainer, pipeline trainer needs to save model shards
        across all ranks since each rank only has a portion of the model. The
        shard_index tracks which parameters belong to which rank for coordinated
        save/load operations.

        Sets save_on_all_ranks=True so all ranks participate in checkpointing,
        each saving their own pipeline stages.

        Returns:
            CheckpointManager configured for distributed pipeline model saving
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
        """
        Compute global gradient norm across all pipeline stages using all-reduce.

        Each rank computes local norm over its stage parameters, then reduces across
        all ranks to get the true global norm for gradient clipping.

        For L2 norm (norm_type=2): Sum squared norms across ranks, then sqrt.
        For Lp norm: Sum p-norms across ranks, then take 1/p power.
        For Linf norm: Max across ranks.

        Args:
            total_norm: Local norm computed on this rank
            norm_type: Type of norm (2.0 for L2, inf for Linf, etc.)

        Returns:
            Global gradient norm across all pipeline stages
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
        """
        Compute and optionally clip gradient norm across all pipeline stages.

        Unlike single-device trainer, must all-reduce gradient norms across all ranks
        since each rank only has gradients for its pipeline stages. Global norm is needed
        for consistent gradient clipping.

        Args:
            max_grad_norm: Maximum norm for clipping (None = no clipping, just compute norm)
            norm_type: Type of norm (2.0 for L2 norm)

        Returns:
            Global gradient norm across all pipeline stages
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
    def _distributed_loss(self, loss: Tensor):
        """
        Broadcast loss from last pipeline stage to all other ranks for logging.

        Only the last stage computes the actual loss (has the labels). Other stages
        return 0.0. This broadcasts the real loss from last stage so all ranks can
        log the same loss value.

        Args:
            loss: Loss tensor (meaningful only on last stage, 0.0 on others)

        Returns:
            Broadcasted loss from last stage, same value on all ranks
        """
        distributed.broadcast(loss, src=self.pp_last_stage_rank)
        return loss
