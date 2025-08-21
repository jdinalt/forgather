# https://github.com/pytorch/pytorch/tree/main/torch/distributed/pipelining
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Tuple, Optional, Union, Iterable
from collections.abc import Sequence
from types import NoneType
import logging
import math
import copy
import os
from functools import partial
from contextlib import ExitStack
import re

import torch
from torch.utils.data import DataLoader
from torch.export.unflatten import InterpreterModule, UnflattenedModule
from torch.fx import GraphModule
from torch import Tensor
from torch import distributed
import torch.distributed as dist
from torch.distributed.pipelining import SplitPoint, ScheduleGPipe, PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    _Action,
    _ComputationType,
    _format_pipeline_order,
    _ScheduleForwardOnly,
)
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.distributed.pipelining import pipeline as build_pipeline
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
)

from ..trainer_types import TrainingArguments, TrainerState
from ..trainer import Trainer, optimzer_hook
from ...distributed import DistributedEnvironment, main_process_first
from ...sharded_checkpoint import (
    validate_output_dir,
    make_shard_index,
    save_shard_index,
    load_shard_index,
    load_sharded_checkpoint,
    save_sharded_checkpoint,
    load_checkpoint,
    index_file_name,
    create_sharing_metadata,
    retie_parameters,
    get_all_fqns,
)
from .pipeline_buffer_fix import apply_pipeline_buffer_fix, validate_pipeline_buffers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


@dataclass(kw_only=True)
class PipelineTrainingArguments(TrainingArguments):
    split_spec: dict
    unified_model: bool = False
    debug_pipeline: bool = False
    debug_split_model: bool = False
    debug_model_params: bool = False
    debug_model_init: bool = False
    pipeline_chunks: int = 4
    stages_per_rank: int = 1
    pp_stage_type: str = "loop"
    is_multistage: bool = False
    enable_activation_checkpoints: bool = False


def insert_activation_checkpoints(module, targets):
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
                    node.args = (submod.get_submodule(node.target), *node.args)
                    node.kwargs = dict(use_reentrant=False, **node.kwargs)
                    node.target = torch.utils.checkpoint.checkpoint
                    node.op = "call_function"

            if not dirty:
                continue

            if isinstance(submod, GraphModule):
                graph_module = submodule
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


def pipeline_stage_indices(pp_size, n_stages, style: str = "loop") -> List[Tuple[int]]:
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


class ScheduleMultiEval(PipelineScheduleMulti):
    """
    Multi-stage scheduler which only runs the forward pass

    We use this for our "eval" model
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        stage_indices: List[Tuple[int]],
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=None,
            output_merge_spec=output_merge_spec,
        )

        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}

        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank, stage_indices)
            self.pipeline_order[rank] = rank_ops

        logger.debug(f"Eval Pipeline:\n{_format_pipeline_order(self.pipeline_order)}")

    def _calculate_single_rank_operations(self, rank, stage_indices):
        rank_stage_indices = stage_indices[rank]
        rank_ops: List[Optional[_Action]] = [None for _ in range(rank)]

        for stage_index in rank_stage_indices:
            rank_ops.extend(
                _Action(stage_index, _ComputationType.FORWARD, mb_index)
                for mb_index in range(self._n_microbatches)
            )

        return rank_ops


class PipelineTrainer(Trainer):
    def __init__(
        self,
        *,
        args: PipelineTrainingArguments,
        distributed_env: DistributedEnvironment,
        loss_fn: Callable,
        pipe_schedule_factory: Callable = ScheduleGPipe,
        **kwargs,
    ):
        assert isinstance(args, PipelineTrainingArguments)
        self.denv = distributed_env
        self.loss_fn = loss_fn
        self.pipe_schedule_factory = pipe_schedule_factory
        super().__init__(args=args, **kwargs)

    # @override
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
                batch_size % self.args.pipeline_chunks == 0
            ), f"Batch size ({batch_size})must be evenly divisible by pipeline_chunks ({self.args.pipeline_chunks})"
        assert (
            self.args.is_multistage or self.args.stages_per_rank == 1
        ), "Only multistage schedulers may have more than one stages_per_rank"
        # TODO: Relax requirements to be at least as large as required.
        self.n_pipeline_stages = self.args.stages_per_rank * self.denv.world_size
        n_pipeline_stages = len(self.args.split_spec) + 1
        assert self.n_pipeline_stages == n_pipeline_stages, (
            f"stages_per_rank ({self.args.stages_per_rank}) * world_size "
            f"({self.denv.world_size}) != splits {n_pipeline_stages}"
        )

        # The pipeline requires a fixed shape for the inputs
        self.args.dataloader_drop_last = True

        self.is_local_process_zero = self.denv.local_rank == 0
        self.is_world_process_zero = self.denv.rank == 0
        self.num_processes = self.denv.world_size

        # Convert strings to enums in split-spec
        for key, value in self.args.split_spec.items():
            match value:
                case "beginning":
                    self.args.split_spec[key] = SplitPoint.BEGINNING
                case "end":
                    self.args.split_spec[key] = SplitPoint.END
                case _:
                    raise Exception(f"Unknown split-point type {value} for {key}")

    def _print_modules(self, modules):
        if self.args.debug_model_params:
            for mod in modules:
                for name, p in mod.named_parameters(remove_duplicate=False):
                    logger.debug(
                        f"P {self.denv.rank} {name} : device {p.device}, dtype {p.dtype}"
                    )
                for name, p in mod.named_buffers(remove_duplicate=False):
                    logger.debug(
                        f"B {self.denv.rank} {name} : device {p.device}, dtype {p.dtype}"
                    )

    # @override
    def _prepare_model(self):
        # Reset -- this trainer always resets everything.
        self.train_scheduler = None
        self.model = None
        self.pipeline_modules = None
        self.eval_pipeline_modules = None
        self.eval_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.sharing_metadata = None

        # Construct model instance on the "meta" device; parameters have meta-data, but no actual data.
        # This allows us to construct a "huge" model, without having to have the memory for it.
        model = self._construct_model(device="meta")
        if self.denv.rank == 0:
            self._print_modules([model])

        # Get parameter sharing metadata
        self.sharing_metadata = create_sharing_metadata(model)

        # Get a micro-batch from the train_dataloader to use for tracing.
        example_args, example_kwargs = self._get_example(self.train_dataloader)

        # stage_indices : A List[Tuple[int]] with the assigned stage indices for each rank
        #   e.g. stage_indices[rank] would have the stage indices for "rank"
        stage_indices = pipeline_stage_indices(
            self.denv.world_size, self.n_pipeline_stages, style=self.args.pp_stage_type
        )

        # Split model into pipeline segments.
        if self.denv.rank == 0:
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
            mod.to_empty(device=self.denv.device)
            retie_parameters(mod, self.sharing_metadata)

        # Load from checkpoint?
        if self.args.resume_from_checkpoint:
            missing_buffer_set = missing_buffers(model)
            if len(missing_buffer_set):
                if self.denv.rank == 0:
                    logger.warning(
                        f"The following buffers were not found in the model's state_dict: {missing_buffer_set}. "
                        "Forcing initialization of full model on CPU to construct missing buffers. "
                        "To avoid this, make sure all the model's buffers have 'persist' set to True."
                    )
                self._initialize_params(
                    all_pipeline_modules, pipeline_modules, stage_indices, True
                )
        else:
            if self.denv.rank == 0:
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
        self._assert_no_duplicate_fqns(state_dicts)

        self.shard_index = make_shard_index(
            state_dicts,
            safetensors=self.args.save_safetensors,
            param_sharing_metadata=self.sharing_metadata,
        )
        self.train_scheduler = self.pipe_schedule_factory(
            stages_arg, self.args.pipeline_chunks, loss_fn=self.loss_fn
        )

        if self.args.enable_activation_checkpoints:
            # Enable activation checkpointing for all modules in the pipeline.
            for mod in pipeline_modules:
                insert_activation_checkpoints(mod, r"^layers\.(\d+)$")

        self.pipeline_modules = pipeline_modules
        self.stage_indices = stage_indices[self.denv.rank]

        # Identify the rank of the last stage, as they need to broadcast the loss.
        last_stage_index = self.n_pipeline_stages - 1
        for rank in range(len(stage_indices)):
            if last_stage_index in stage_indices[rank]:
                self.pp_last_stage_rank = rank
                break

        # To simplify things...
        self.pp_has_last_stage = last_stage_index in self.stage_indices
        self.pp_has_first_stage = 0 in self.stage_indices

        # We keep the original model on the meta-device. This model is obviously not functional, but
        # some trainer callbacks may wish to dump the layout.
        self.model = model

        # Create a seperate eval pipeline
        # This does not generate gradients and MAY have a different batch size, as well as being traced in "eval" mode.
        if self.args.unified_model:
            self.eval_scheduler = None
        else:
            if self.denv.rank == 0:
                logger.info("Constructing and splitting eval model.")
            self._construct_eval_pipeline()

    def _construct_model(self, device):
        # Construct model on device
        with torch.device(device):
            model = self.model_init()

        return model

    def _compile_model(self):
        pipelines = [self.pipeline_modules]
        if not self.args.unified_model:
            pipelines.append(self.eval_pipeline_modules)
        for pipeline_modules in pipelines:
            for mod in pipeline_modules:
                mod.compile(
                    backend=self.args.torch_compile_backend,
                    mode=self.args.torch_compile_mode,
                    dynamic=self.args.torch_compile_dynamic,
                    fullgraph=self.args.torch_compile_full_graph,
                )

    def _get_example(self, example_dataloader):
        # Note that pipeline parallel requires all batches to have the same shape!
        # TODO: We have hard-coded "input_ids" This should be more flexible, as this is not always the case.
        example_batch = next(iter(example_dataloader))
        example_args = (torch.empty_like(example_batch["input_ids"], device="meta"),)
        exampke_kwargs = dict(
            #    input_ids=torch.empty_like(
            #        example_batch["input_ids"],
            #        device="meta"
            #    ),
        )

        # Split into microbatches
        split_args, split_kwargs = split_args_kwargs_into_chunks(
            example_args, exampke_kwargs, chunks=self.args.pipeline_chunks
        )

        # Return example micro-batches
        return split_args[0], split_kwargs[0]

    def _split_model(self, model, example_args, example_kwargs, stage_indices, train):
        rank = self.denv.rank
        # Trace model
        # The trace will use fake-tensors, so this does not perform any
        # actual computation. It just traces the path through the model
        # and records tensor geometery information.
        args = (model, example_args, example_kwargs)
        kwargs = dict(split_spec=self.args.split_spec)
        if train:
            model.train()
            pipe = build_pipeline(*args, **kwargs)
        else:
            model.eval()
            with torch.no_grad():
                pipe = build_pipeline(*args, **kwargs)

        if rank == 0:
            logger.debug(pipe)
            if self.args.debug_split_model:
                logger.debug(pipe.print_readable())

        all_pipeline_modules = [
            pipe.get_stage_module(i) for i in range(self.n_pipeline_stages)
        ]

        pipeline_modules = [
            all_pipeline_modules[i] for i in stage_indices[self.denv.rank]
        ]

        pipeline_stages = [
            pipe.build_stage(stage_index=i, device=self.denv.device)
            for i in stage_indices[self.denv.rank]
        ]

        # Apply buffer fix after pipeline split but before to_empty()
        # This fixes both zombie buffers and shared buffer accessibility issues
        apply_pipeline_buffer_fix(all_pipeline_modules, model)

        # Remove vestigial modules to prevent duplicate FQN conflicts during checkpoint saving
        self._remove_vestigial_modules(all_pipeline_modules)

        return all_pipeline_modules, pipeline_modules, pipeline_stages

    @torch.no_grad()
    def _initialize_params(
        self, all_pipeline_modules, pipeline_modules, stage_indices, missing_buf_only
    ):
        """
        Rank zero is expected to have initialized weights on the cpu,
        while all other ranks are expected to have uninitialized (empty)
        weights on their respective devices.

        Rank 0 sends an initialized copy of each other ranks's weights,
        which are loaded directly onto the target device.

        This avoids having to load N copies of the model into cpu memory, where most of
        the loaded data is thrown away.

        TODO: Optimize case where init is performed because of non-persistent buffers.
        In this case, we only need to transfer the non-persistent buffers.
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

        if self.denv.rank == 0:
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
            # Send the initialized paramters for the other stages to their
            # respective processes.
            for dst_rank in range(1, self.denv.world_size):
                # Modules owned by dst_rank
                rank_indices = stage_indices[dst_rank]
                logger.debug(
                    f"rank0: Sending initialized params for stages {rank_indices} to rank{dst_rank}"
                )

                for stage_index in rank_indices:
                    mod = all_pipeline_modules[stage_index]

                    # All params and buffes in destination module
                    for name, _ in make_state_dict(mod, missing_buf_only).items():
                        # NCCL can't send between GPU and GPU, so copy each parameter to
                        # our GPU, send it, then free it. Kind of hack'ish, but it works.
                        # See: https://docs.pytorch.org/docs/stable/distributed.html
                        # I believe this will work for CPU to CPU, with gloo, but
                        # have yet to try it.
                        p = init_state_dict[name].to(self.denv.device)
                        if self.args.debug_model_init:
                            logger.debug(f"rank0: Sending {name} to rank{dst_rank}")
                        distributed.send(p, dst=dst_rank)
                        p = None
        else:
            # Load the parameters from rank 0
            rank_indices = stage_indices[self.denv.rank]

            logger.debug(
                f"rank{self.denv.rank}: Receiving initialized params for stages {rank_indices} from rank0"
            )
            for mod in pipeline_modules:
                for name, p in make_state_dict(mod, missing_buf_only).items():
                    if self.args.debug_model_init:
                        logger.debug(f"rank{self.denv.rank}: Receiving {name}")
                    distributed.recv(p, src=0)

        distributed.barrier()

    def _dataloader_iter(self, dataloader: DataLoader) -> Iterable:
        """
        Asynchronous pipeline-parallel dataloader iterator.

        Rank 0: sends batches as soon as they are available.
        Other ranks: receive batches asynchronously using non-blocking communication,
        but only yield the batch after all tensors are fully received.
        """
        if self.denv.rank == 0:
            for batch in dataloader:
                # Signal start of new batch
                distributed.broadcast(torch.tensor(1, device=self.denv.device), src=0)
                # Send each tensor in the batch
                for key, value in batch.items():
                    assert isinstance(value, Tensor), (
                        f"Batch item {key} is not a Tensor, got {type(value)}. "
                        "Pipeline parallel requires all batch items to be Tensors."
                    )
                    distributed.broadcast(
                        value.to(device=self.denv.device), src=0, async_op=False
                    )
                yield batch
            # Signal end of dataloader
            distributed.broadcast(torch.tensor(0, device=self.denv.device), src=0)
        else:
            flag = torch.tensor(0, device=self.denv.device)
            reference_batch = next(iter(dataloader))
            batch = {}
            for key, value in reference_batch.items():
                assert isinstance(value, Tensor), (
                    f"Batch item {key} is not a Tensor, got {type(value)}. "
                    "Pipeline parallel requires all batch items to be Tensors."
                )
                batch[key] = torch.empty_like(value, device=self.denv.device)
            reference_batch = None

            while True:
                # Receive flag asynchronously
                req_flag = distributed.broadcast(flag, src=0, async_op=True)
                req_flag.wait()
                if flag.item() == 0:
                    break

                # Start async receives for all tensors
                requests = []
                for value in batch.values():
                    req = distributed.broadcast(value, src=0, async_op=True)
                    requests.append(req)

                # Wait for all tensors to be received before yielding
                for req in requests:
                    req.wait()
                yield copy.copy(batch)

    def _construct_eval_pipeline(self):
        # Construct an "eval" version of the pipelined model, which shares weights with the "train" version.
        if self.denv.rank:
            logger.debug("Constructing evaluation model")

        eval_model = self._construct_model(device="meta")
        example_args, example_kwargs = self._get_example(self.eval_dataloader)
        stage_indices = stage_indices = pipeline_stage_indices(
            self.denv.world_size, self.n_pipeline_stages, style=self.args.pp_stage_type
        )
        _, eval_pipeline_modules, eval_pipeline_stages = self._split_model(
            eval_model, example_args, example_kwargs, stage_indices, train=False
        )

        # Replace meta parameters on eval modules with parameters of train modules
        for eval_mod, train_mod in zip(eval_pipeline_modules, self.pipeline_modules):
            replace_parameters(eval_mod, train_mod)
            replace_buffers(eval_mod, train_mod)
            retie_parameters(eval_mod, self.sharing_metadata)

        if self.args.is_multistage:
            self.eval_scheduler = ScheduleMultiEval(
                eval_pipeline_stages,
                self.args.pipeline_chunks,
                stage_indices=stage_indices,
            )
        else:
            assert len(eval_pipeline_stages) == 1
            self.eval_scheduler = _ScheduleForwardOnly(
                eval_pipeline_stages[0], self.args.pipeline_chunks
            )
        self.eval_pipeline_modules = eval_pipeline_modules

    def _train_pipeline_step(self, batch: dict | tuple) -> Tensor:
        args, kwargs = self._prepare_batch(batch)
        # For now, we are hard-coding a specific input format.
        # There seems to be an issue with the scheduler implementation when passing kwargs,
        # so we convert to args for now.
        # TODO: Make this more flexible.
        inputs = (kwargs["input_ids"],)
        labels = kwargs["labels"]

        # See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py#L377
        targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)

        with ExitStack() as stack:
            if self.args.enable_activation_offloading:
                stack.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=True))
            if self.pp_has_first_stage:
                self.train_scheduler.step(*inputs, target=targets, losses=losses)
            else:
                self.train_scheduler.step(target=targets, losses=losses)

            if self.pp_has_last_stage:
                mean_loss = torch.stack([x.detach() for x in losses]).mean()
                mean_loss = mean_loss.float()
            else:
                mean_loss = torch.tensor(0.0, device=self.denv.device)

        return mean_loss

    def _eval_pipeline_step(self, batch: dict | tuple) -> Tensor:
        args, kwargs = self._prepare_batch(batch)
        inputs = (kwargs["input_ids"],)
        labels = kwargs["labels"]

        if self.pp_has_first_stage:
            outputs = self.eval_scheduler.step(*inputs)
        else:
            outputs = self.eval_scheduler.step()

        if self.pp_has_last_stage:
            loss = self.loss_fn(outputs, labels).detach()
            mean_loss = loss.float()
        else:
            mean_loss = torch.tensor(0.0, device=self.denv.device)

        return mean_loss

    # @override
    def _init_optimizer(self):
        if self.optimizer is None:
            # Build a named-parameter generator for all of our modules
            def named_parameters(modules):
                for mod in modules:
                    for param in mod.named_parameters():
                        yield param

            self.optimizer = self.optimizer_factory(
                named_parameters(self.pipeline_modules)
            )

            if self.args.fuse_optim_with_backward:
                hook = partial(optimzer_hook, self.optimizer)
                for name, p in named_parameters(self.pipeline_modules):
                    if p.requires_grad:
                        p.register_post_accumulate_grad_hook(hook)

        if self.lr_scheduler is None and self.lr_scheduler_factory is not None:
            self.lr_scheduler = self.lr_scheduler_factory(
                optimizer=self.optimizer,
            )

    # We don't want to automatically use no_grad(), as we may need to use the unified model.
    # @override
    def _eval_loop(self) -> Dict[str, float]:
        if not self.args.unified_model:
            super()._eval_loop()
        else:
            # Unified model for train and eval
            # This requires that we produce gradients.
            total_loss = torch.zeros(1, device=self.args.device)
            step = 0
            for step, batch in enumerate(self.eval_dataloader):
                outputs = self._unified_prediction_step(batch)
                loss = self._gather_reduce_loss(outputs["loss"])
                total_loss += loss
                self._dispatch_event("on_prediction_step")
            metrics = {"eval_loss": (total_loss / step).item()}
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    # @override
    def _train_step(self, batch: dict | tuple) -> Tensor:
        mean_loss = self._train_pipeline_step(batch)
        self._clip_grad_norm(self.args.max_grad_norm)
        if not self.args.fuse_optim_with_backward:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self._dispatch_event("on_optimizer_step")
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return mean_loss

    # @override
    def _prediction_step(self, batch: dict | tuple) -> Tensor:
        mean_loss = self._eval_pipeline_step(batch)
        return {
            "loss": mean_loss,
            "logits": None,
            "labels": None,
        }

    def _unified_prediction_step(self, batch: dict | tuple) -> Tensor:
        """
        Using the same traced model for both train and eval.
        """
        mean_loss = self._train_pipeline_step(batch)
        self.optimizer.zero_grad()
        return {
            "loss": mean_loss,
            "logits": None,
            "labels": None,
        }

    # @override
    def _clip_grad_norm(self, max_grad_norm, norm_type=2.0) -> Optional[Tensor]:
        if max_grad_norm is None or max_grad_norm == 0:
            return None

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
        # logger.debug(f"RANK{self.denv.rank}: {total_norm=}, {norm_type=}, {max_grad_norm=}")
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX)
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
            total_norm **= 1.0 / norm_type

        # if self.denv.rank == 0:
        #    logger.info(f"RANK{self.denv.rank}: Clipping gradients with total norm {total_norm:.4f} and max norm {max_grad_norm:.4f}")

        torch.nn.utils.clip_grads_with_norm_(
            parameters,
            max_grad_norm,
            total_norm,
            foreach=True,
        )

        return total_norm

    # @override
    def _gather_reduce_loss(self, loss: Tensor):
        distributed.broadcast(loss, src=self.pp_last_stage_rank)
        return loss

    # @override
    def _validate_dirs(self):
        validate_output_dir(
            self.args.output_dir, overwrite=self.args.overwrite_output_dir
        )

    # @override
    def _should_save_unique(self):
        """
        Should this process save a unique file?
        """
        return self.denv.rank == 0 or (
            self.args.save_on_each_node and self.denv.local_rank == 0
        )

    # @override
    def _barrier(self):
        distributed.barrier()

    # @override
    def _save_model(self, output_dir):
        shard_index = self.shard_index
        save_safetensors = self.args.save_safetensors

        # The primary process on each saves the common state
        if self._should_save_unique():
            # Save the shard index
            save_shard_index(shard_index, output_dir, index_file_name(save_safetensors))

        # All processes save their own pipeline stages
        for mod in self.pipeline_modules:
            save_sharded_checkpoint(
                output_dir,
                shard_index,
                mod,
                safetensors=save_safetensors,
                debug=self.args.debug_pipeline,
            )

    # @override
    def _save_training_state(self, output_dir: str) -> None:
        training_state = self._state_dict()
        if training_state:
            # Use rank-specific filename to avoid conflicts
            training_state_path = os.path.join(
                output_dir, f"training_state_rank_{self.denv.rank}.pt"
            )
            torch.save(training_state, training_state_path)
            logger.info(
                f"Rank {self.denv.rank}: Saved training state to {training_state_path}"
            )
        else:
            logger.warning("No training state saved!")

    # @override
    def _load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        for mod in self.pipeline_modules:
            load_checkpoint(
                checkpoint_path,
                mod,
                device=self.denv.device,
                strict=False,
            )

        self._barrier()

    # @override
    def _load_training_state(self, checkpoint_path: str) -> None:
        """Override to handle distributed optimizer/scheduler state loading."""
        # Each rank loads its own training state from rank-specific file
        training_state_path = os.path.join(
            checkpoint_path, f"training_state_rank_{self.denv.rank}.pt"
        )

        if not os.path.exists(training_state_path):
            logger.info(
                f"Rank {self.denv.rank}: No training state file found at: {training_state_path}"
            )
            return None
        try:
            training_state = torch.load(
                training_state_path, map_location=torch.device("cpu")
            )

        except Exception as e:
            logger.error(
                f"Rank {self.denv.rank}: Failed to load training state from {training_state_path}: {e}"
            )
        self._load_state_dict(training_state)

    def _remove_vestigial_modules(self, all_pipeline_modules):
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
                        if self._is_module_unreferenced(module, name):
                            modules_to_remove.append((parent_module, module_name, name))
                            logger.debug(
                                f"  Marking vestigial module for removal: {name}"
                            )
                    except AttributeError:
                        logger.debug(f"  Could not access parent of {name}, skipping")

            # Remove vestigial modules
            for parent_module, module_name, full_name in modules_to_remove:
                try:
                    delattr(parent_module, module_name)
                    logger.debug(f"  Removed vestigial module: {full_name}")
                except AttributeError:
                    logger.debug(f"  Failed to remove module: {full_name}")

    def _is_module_unreferenced(self, root_module, target_name):
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

    def _assert_no_duplicate_fqns(self, state_dicts):
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
