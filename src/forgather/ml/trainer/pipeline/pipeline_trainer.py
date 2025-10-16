# https://github.com/pytorch/pytorch/tree/main/torch/distributed/pipelining
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Tuple, Iterable
import logging
import math
import copy
from functools import partial

import torch
from torch.nn import Module
from torch import Tensor
from torch import distributed
import torch.distributed as dist
from torch.distributed.pipelining import SplitPoint, ScheduleGPipe
from torch.distributed.pipelining.schedules import (
    _ScheduleForwardOnly,
)
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.distributed.pipelining import pipeline as build_pipeline
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from ..trainer_types import CheckpointInterface
from ..checkpoint_manager import CheckpointManager, CheckpointConfig
from ..trainer import (
    Trainer,
    TrainingArguments,
    optimzer_hook,
    rescale_accumulated_loss,
)
from ...sharded_checkpoint import (
    make_shard_index,
    create_sharing_metadata,
    retie_parameters,
)
from .schedule_multi_eval import ScheduleMultiEval
from .pipeline_utils import (
    insert_activation_checkpoints,
    replace_parameters,
    replace_buffers,
    pipeline_stage_indices,
    missing_buffers,
)
from .pipeline_fixes import (
    apply_pipeline_buffer_fix,
    assert_no_duplicate_fqns,
    remove_vestigial_modules,
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


class PipelineTrainer(Trainer):
    def __init__(
        self,
        *,
        args: PipelineTrainingArguments,
        pipe_schedule_factory: Callable = ScheduleGPipe,
        **kwargs,
    ):
        assert isinstance(args, PipelineTrainingArguments)
        self.args = args  # For type checking hint
        self.pipe_schedule_factory = pipe_schedule_factory
        super().__init__(args=args, **kwargs)

    # @override
    def _is_pipeline_parallel(self) -> bool:
        """Pipeline parallelism doesn't increase effective batch size"""
        return True

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
            ), f"Batch size ({batch_size}) must be evenly divisible by pipeline_chunks ({self.args.pipeline_chunks})"
        assert (
            self.args.is_multistage or self.args.stages_per_rank == 1
        ), "Only multistage schedulers may have more than one stages_per_rank"
        # TODO: Relax requirements to be at least as large as required.
        self.n_pipeline_stages = self.args.stages_per_rank * self.dist.world_size
        n_pipeline_stages = len(self.args.split_spec) + 1
        assert self.n_pipeline_stages == n_pipeline_stages, (
            f"stages_per_rank ({self.args.stages_per_rank}) * world_size "
            f"({self.dist.world_size}) != splits {n_pipeline_stages}"
        )

        # The pipeline requires a fixed shape for the inputs
        self.args.dataloader_drop_last = True

        self.is_local_process_zero = self.dist.local_rank == 0
        self.is_world_process_zero = self.dist.rank == 0
        self.num_processes = self.dist.world_size

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
                        f"P {self.dist.rank} {name} : device {p.device}, dtype {p.dtype}"
                    )
                for name, p in mod.named_buffers(remove_duplicate=False):
                    logger.debug(
                        f"B {self.dist.rank} {name} : device {p.device}, dtype {p.dtype}"
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
        if self.dist.rank == 0:
            self._print_modules([model])

        # Get parameter sharing metadata
        self.sharing_metadata = create_sharing_metadata(model)

        # Get a micro-batch from the train_dataloader to use for tracing.
        example_args, example_kwargs = self._get_example(self.train_dataloader)

        # stage_indices : A List[Tuple[int]] with the assigned stage indices for each rank
        #   e.g. stage_indices[rank] would have the stage indices for "rank"
        stage_indices = pipeline_stage_indices(
            self.dist.world_size, self.n_pipeline_stages, style=self.args.pp_stage_type
        )

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

        # Note: scale_grads=True, the default, causes the pipeline to rescale the gradients by the number of microbatches
        # We handle this by applying this factor directly to the loss function, above. Without doing so,
        # gradient accumulation does not work properly, as the complete acculumlated gradient is rescaled
        # at each pipeline step.
        self.train_scheduler = self.pipe_schedule_factory(
            stages_arg,
            self.args.pipeline_chunks,
            loss_fn=self.train_loss_fn,
            scale_grads=False,
        )

        if self.args.gradient_checkpointing:
            if self.dist.rank == 0:
                logger.info("Applying activation checkpointing via FX graph")
            # Enable activation checkpointing for all modules in the pipeline.
            for mod in pipeline_modules:
                insert_activation_checkpoints(mod, r"^layers\.(\d+)$")

        self.pipeline_modules = pipeline_modules
        self.stage_indices = stage_indices[self.dist.rank]

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
            if self.dist.rank == 0:
                logger.info("Constructing and splitting eval model.")
            self._construct_eval_pipeline()

    # @override
    def _wrap_loss_fn(self):
        assert self.loss_fn
        self.eval_loss_fn = self.loss_fn
        # Rescale by product of accumulation-steps and mirco-batches (pipeline_chunks)
        self.train_loss_fn = rescale_accumulated_loss(
            self.loss_fn,
            self.args.gradient_accumulation_steps * self.args.pipeline_chunks,
        )

    def _construct_model(self, device):
        # Construct model on device
        assert self.model_init
        with torch.device(device):
            model = self.model_init()
        return model

    # @override
    def _compile_model(self):
        assert self.pipeline_modules
        pipelines = [self.pipeline_modules]
        if not self.args.unified_model:
            assert self.eval_pipeline_modules
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

    def _split_model(
        self, model, example_args, example_kwargs, stage_indices, train
    ) -> Tuple[List[Module], List[Module], List[_PipelineStageBase]]:
        rank = self.dist.rank
        # Trace model
        # The trace will use fake-tensors, so this does not perform any
        # actual computation. It just traces the path through the model
        # and records tensor geometery information.
        args = (model, example_args, example_kwargs)
        kwargs: Dict[str, Any] = dict(split_spec=self.args.split_spec)
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
            all_pipeline_modules[i] for i in stage_indices[self.dist.rank]
        ]

        pipeline_stages = [
            pipe.build_stage(stage_index=i, device=torch.device(self.dist.device))
            for i in stage_indices[self.dist.rank]
        ]

        # Apply buffer fix after pipeline split but before to_empty()
        # This fixes both zombie buffers and shared buffer accessibility issues
        apply_pipeline_buffer_fix(all_pipeline_modules, model)

        # Remove vestigial modules to prevent duplicate FQN conflicts during checkpoint saving
        remove_vestigial_modules(all_pipeline_modules)

        return all_pipeline_modules, pipeline_modules, list(pipeline_stages)

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
            # Send the initialized paramters for the other stages to their
            # respective processes.
            for dst_rank in range(1, self.dist.world_size):
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

        distributed.barrier()

    # @override
    def _dataloader_iter(
        self, dataloader: Iterable[Dict[str, Tensor]]
    ) -> Iterable[Dict[str, Tensor]]:
        """
        Asynchronous pipeline-parallel dataloader iterator.

        Rank 0: sends batches as soon as they are available.
        Other ranks: receive batches asynchronously using non-blocking communication,
        but only yield the batch after all tensors are fully received.
        """
        if self.dist.rank == 0:
            for batch in dataloader:
                # Signal start of new batch
                distributed.broadcast(torch.tensor(1, device=self.dist.device), src=0)
                # Send each tensor in the batch
                for key, value in batch.items():
                    assert isinstance(value, Tensor), (
                        f"Batch item {key} is not a Tensor, got {type(value)}. "
                        "Pipeline parallel requires all batch items to be Tensors."
                    )
                    distributed.broadcast(
                        value.to(device=self.dist.device), src=0, async_op=False
                    )
                yield batch
            # Signal end of dataloader
            distributed.broadcast(torch.tensor(0, device=self.dist.device), src=0)
        else:
            flag = torch.tensor(0, device=self.dist.device)
            reference_batch = next(iter(dataloader))
            batch = {}
            for key, value in reference_batch.items():
                assert isinstance(value, Tensor), (
                    f"Batch item {key} is not a Tensor, got {type(value)}. "
                    "Pipeline parallel requires all batch items to be Tensors."
                )
                batch[key] = torch.empty_like(value, device=self.dist.device)
            reference_batch = None

            while True:
                # Receive flag asynchronously
                req_flag = distributed.broadcast(flag, src=0, async_op=True)
                assert req_flag
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
        if self.dist.rank:
            logger.debug("Constructing evaluation model")

        eval_model = self._construct_model(device="meta")
        example_args, example_kwargs = self._get_example(self.eval_dataloader)
        stage_indices = stage_indices = pipeline_stage_indices(
            self.dist.world_size, self.n_pipeline_stages, style=self.args.pp_stage_type
        )
        _, eval_pipeline_modules, eval_pipeline_stages = self._split_model(
            eval_model, example_args, example_kwargs, stage_indices, train=False
        )

        # Replace meta parameters on eval modules with parameters of train modules
        assert self.pipeline_modules
        assert self.sharing_metadata is not None
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

    # @override
    def _forward_backward_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Tensor:
        inputs = (input_dict["input_ids"],)

        # See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py#L377
        targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
        assert self.train_scheduler
        if self.pp_has_first_stage:
            self.train_scheduler.step(*inputs, target=targets, losses=losses)
        else:
            self.train_scheduler.step(target=targets, losses=losses)

        if self.pp_has_last_stage:
            assert losses
            mean_loss = torch.stack([x.detach() for x in losses]).mean()
            mean_loss = mean_loss.float()
        else:
            mean_loss = torch.tensor(0.0, device=self.dist.device)
        return mean_loss

    def _eval_pipeline_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Tensor:
        inputs = (input_dict["input_ids"],)

        assert self.eval_scheduler
        if self.pp_has_first_stage:
            outputs = self.eval_scheduler.step(*inputs)
        else:
            outputs = self.eval_scheduler.step()

        if self.pp_has_last_stage:
            loss = self.eval_loss_fn(outputs, labels).detach()
            mean_loss = loss.float()
        else:
            mean_loss = torch.tensor(0.0, device=self.dist.device)
        return mean_loss

    # @override
    def _init_optimizer(self):
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
                hook = partial(optimzer_hook, self.optimizer, self._total_grad_squared)
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
            return super()._eval_loop()
        else:
            # Unified model for train and eval
            # This requires that we produce gradients.
            total_loss = torch.zeros(1, device=self.args.device)
            step = 0
            for step, batch in enumerate(self._dataloader_iter(self.eval_dataloader)):
                if self.args.max_eval_steps > 0 and step >= self.args.max_eval_steps:
                    break
                input_dict, labels = self._prepare_batch(batch)
                outputs = self._unified_prediction_step(input_dict, labels)
                loss = outputs["loss"]
                assert loss
                total_loss += loss
                self._dispatch_event("on_prediction_step")
            metrics = {"eval_loss": (total_loss / step).item()}
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    # @override
    def _prediction_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Dict[str, Tensor | None]:
        mean_loss = self._eval_pipeline_step(input_dict, labels)
        mean_loss = self._distributed_loss(mean_loss)
        return {
            "loss": mean_loss,
            "logits": None,
            "labels": None,
        }

    def _unified_prediction_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Dict[str, Tensor | None]:
        """
        Using the same traced model for both train and eval.
        """
        mean_loss = self._forward_backward_step(input_dict, labels)
        mean_loss = self._distributed_loss(mean_loss)
        assert self.optimizer
        self.optimizer.zero_grad()
        return {
            "loss": mean_loss,
            "logits": None,
            "labels": None,
        }

    # @override
    def _loss_post_scaler(self):
        return float(self.args.pipeline_chunks)

    # @override
    def _init_checkpoint_manager(self) -> CheckpointInterface:
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
        All-Reduce grad-norm from all ranks
        """
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX)
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
            total_norm **= 1.0 / norm_type
        return total_norm

    # @override
    def _clip_grad_norm(self, max_grad_norm, norm_type=2.0) -> Tensor:
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

    # @override
    def _distributed_loss(self, loss: Tensor):
        distributed.broadcast(loss, src=self.pp_last_stage_rank)
        return loss
