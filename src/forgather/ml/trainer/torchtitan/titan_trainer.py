# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# source: https://github.com/pytorch/torchtitan/

import importlib
import os
from typing import Optional, Callable

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.models.attention import init_attention_mask
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)

# Added
from torch.optim.optimizer import ParamsT
from torchtitan.train import Trainer
from torchtitan.protocols.train_spec import ParallelizeFunction, PipeliningFunction
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config.job_config import Parallelism
from torchtitan.components.ft.config import FaultTolerance as FTConfig

from forgather.ml.distributed import DistributedEnvInterface


# ExtendedParallelDims and ExtendedFTManager really should be exporting dp_degree and dp_rank
# as properites, as they are needed for constructing various objects, and coordinate better.
# This should work, but is relatively brittle. I could easily see the base-classes changing
# and breaking this hack.
# TODO: Work with torchtitan to clean this up
class ExtendedParallelDims(ParallelDims):
    # Factory method
    @staticmethod
    def from_config(
        distributed_env: DistributedEnvInterface, parallelism_config: Parallelism
    ):
        return ExtendedParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=distributed_env.world_size,
        )

    @property
    def dp_degree(self):
        if self.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            return dp_mesh.size()
        else:
            return 1

    @property
    def dp_rank(self):
        if self.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            return dp_mesh.get_local_rank()
        else:
            return 0


class ExtendedFTManager(FTManager):
    def __init__(
        self,
        ft_config: FTConfig,
        parallel_dims: ExtendedParallelDims,
    ) -> None:
        self.parallel_dims = parallel_dims
        super().__init__(ft_config)

    @property
    def dp_degree(self):
        dp_degree, _ = self.get_dp_info(
            self.parallel_dims.dp_degree, self.parallel_dims.dp_rank
        )
        return dp_degree

    @property
    def dp_rank(self):
        _, dp_rank = self.get_dp_info(
            self.parallel_dims.dp_degree, self.parallel_dims.dp_rank
        )
        return dp_rank


class Trainer(Trainer):
    @record
    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ExtendedParallelDims,
        distributed_env: DistributedEnvInterface,
        ft_manager: ExtendedFTManager,
        train_dataloader: BaseDataLoader,
        eval_dataloader: BaseDataLoader,
        tokenizer,  # TODO: What type is tokenizer?
        model_factory: Callable[[], torch.nn.Module],  # TODO: Define type
        optimizer_factory: Callable[
            [ParamsT], torch.optim.Optimizer
        ],  # TODO: Define type
        lr_scheduler_factory: Callable,
        parallelize_fn: ParallelizeFunction,
        pipelining_fn: Optional[PipeliningFunction],
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ],  # TODO: Define type
        state_dict_adapter: Optional[StateDictAdapter],
        model_args,
    ):

        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.dataloader = train_dataloader
        # TODO: Implement validation
        # self.eval_dataset = eval_dataloader
        self.tokenizer = tokenizer
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.parallelize_fn = parallelize_fn
        self.pipelining_fn = pipelining_fn
        self.loss_fn = loss_fn
        self.model_args = model_args

        # We deliberately avoid using train_spec, as everything this provides /should/ be
        # injected into the trainer as a dependency, rather than having the trainer build
        # them from data-arguments.
        #
        # At least at present, the only place the train spec shows up, outside of __init__
        # is in self.train(), but is wrapped with a hasattr() test, defaulting to None
        # As long as they don't add new references, this should work.
        self.train_spec = None

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        self.device = torch.device(distributed_env.device)

        parallelism_config = job_config.parallelism
        world_mesh = parallel_dims.world_mesh

        self.ft_manager = ft_manager  # FTManager(job_config.fault_tolerance)
        dp_degree, dp_rank = ft_manager.dp_degree, ft_manager.dp_rank

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )

        # build model (using meta init)
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            model = self.model_factory()

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        # TODO: Add factory argument
        build_metrics_processor_fn = build_metrics_processor
        self.metrics_processor = build_metrics_processor_fn(
            job_config, parallel_dims, None
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        model_param_count = sum(
            t.numel() if t.requires_grad else 0 for t in model.parameters()
        )

        # Bogus placeholder for now
        # TODO: Handle this properly
        self.metrics_processor.num_flops_per_token = 1 * job_config.training.seq_len

        logger.info(
            f"{color.blue}Model {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = utils.device_type
        else:
            init_device = utils.device_type
            buffer_device = None

        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {dp_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
        assert self.gradient_accumulation_steps > 0
        self.loss_fn = rescale_accumulated_loss(
            self.loss_fn, self.gradient_accumulation_steps
        )

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.pipelining_fn:
                raise RuntimeError(
                    "Pipeline Parallel is enabled but no pipeline_fn specified"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.pipelining_fn(
                model,
                parallel_dims,
                job_config,
                self.device,
                None,  # model_args
                self.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)

                # TODO: Note that this really should be contigent upon not loading from checkpoint!
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.parallelize_fn(model, parallel_dims, job_config)

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{utils.device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # Borrowed from pipeline_trainer
        # This will stuff all of the sub-module parameters into
        # the same optimizer instance, rather than creating an optimizer for each sub-module
        # and wrapping them in an optimizer container.
        def named_parameters(modules):
            for mod in modules:
                for param in mod.named_parameters():
                    yield param

        self.optimizers = self.optimizer_factory(named_parameters(self.model_parts))

        self.lr_schedulers = self.lr_scheduler_factory(
            optimizer=self.optimizers,
        )

        # Address a not-so-well encapsulated aspect of the torchtitan
        # scheduler container...
        if not hasattr(self.lr_schedulers, "schedulers"):
            self.lr_schedulers.schedulers = [self.lr_schedulers]

        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=state_dict_adapter,
            base_folder=job_config.job.dump_folder,
            ft_manager=self.ft_manager,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            utils.device_type,
        )

        # Build validator if validation is configured
        # TODO: Implement me!
        assert not job_config.validation.enable, "Validation is not implemented yet!"
        if job_config.validation.enable:
            assert self.train_spec.build_validator_fn is not None

            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = self.train_spec.build_validator_fn(
                job_config=job_config,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )
