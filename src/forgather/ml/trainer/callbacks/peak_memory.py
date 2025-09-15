import os
import torch
from pprint import pformat
import logging
import torch.distributed as dist

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
)
from ...utils import format_mapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PeakMemory(TrainerCallback):
    """
    PeakMemory is a TrainerCallback for monitoring and logging the peak CUDA memory usage during model training.
    This callback is designed to help diagnose and optimize GPU memory consumption in PyTorch-based training loops,
    especially when using distributed training. It records the maximum memory allocated on each GPU device throughout
    the training process, and can optionally log detailed memory statistics and write them to TensorBoard for visualization.

    IMPORTANT: Memory history recording is disabled by default to prevent memory leaks.
    The torch.cuda.memory._record_memory_history feature can consume 1GB+ of memory during training.

    Key Features:
    - Tracks the peak CUDA memory allocated on each GPU during training.
    - Supports both single-GPU and multi-GPU (distributed) training environments.
    - Optionally logs detailed CUDA memory statistics for further analysis.
    - Can write memory usage metrics to a TensorBoard SummaryWriter for visualization.
    - Provides configurable logging frequency and verbosity.
    Args:
        summary_writer (SummaryWriter, optional): TensorBoard SummaryWriter instance for logging memory statistics.
        show_details (bool, optional): If True, logs detailed CUDA memory statistics at each logging step and at the end of training.
        do_log (bool, optional): If True, logs peak memory usage at each logging step (on_log callback).
        enable_memory_history (bool, optional): If True, enables comprehensive CUDA memory history recording.
                                               WARNING: This can consume 1GB+ memory and cause memory leaks.
    Attributes:
        rank (int): The process rank in distributed training.
        world_size (int): The total number of processes in distributed training.
        summary_writer (SummaryWriter or None): The TensorBoard SummaryWriter for logging.
        enabled (bool): Whether CUDA is available and memory tracking is enabled.
        show_details (bool): Whether to log detailed memory statistics.
        do_log (bool): Whether to log memory usage on each log step.
        enable_memory_history (bool): Whether to enable comprehensive memory history recording.
        max_allocated (int): The maximum CUDA memory allocated during training (in bytes).
    Methods:
        on_train_begin: Initializes memory tracking at the start of training.
        on_log: Logs peak memory usage (and optionally details) at each logging step.
        on_train_end: Finalizes memory tracking and logs the maximum memory usage at the end of training.
        _format_peak_memory: Formats memory usage in human-readable GB units.
    """

    def __init__(
        self,
        summary_writer=None,
        show_details=False,
        do_log=False,
        enable_memory_snapshot=False,
        file_prefix="memory_snapshot",
    ):
        """
        :param summary_writer: Optional TensorBoard SummaryWriter to log peak memory
        :param show_details: Whether to log detailed memory stats
        :param do_log: Whether to log on each log step
        :param enable_memory_snapshot: Whether to enable CUDA memory history recording and snapshot
        """
        super().__init__()
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.summary_writer = summary_writer
        self.enabled = torch.cuda.is_available()
        self.show_details = show_details
        self.do_log = do_log
        self.enable_memory_snapshot = enable_memory_snapshot
        self.file_prefix = file_prefix

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        self.max_allocated = 0
        # This feature can consume 1GB+ of memory during training
        if self.enable_memory_snapshot:
            torch.cuda.memory._record_memory_history(enabled="all")

    @staticmethod
    def _format_peak_memory(max_allocated):
        """
        Format peak memory in GiB (binary GB, 1024 ** 3 bytes)
        """
        gib = 1024**3
        return f"{max_allocated / gib:.3f} GiB" if max_allocated else "0 GiB"

    @staticmethod
    def _mapping_as_markdown(mapping):
        """
        Format dictionary as markdown

        Tensorboard expects text to be in markdown format...
        """
        s = "```\n"
        s += format_mapping(mapping)
        s += "```"
        return s

    def on_log(self, args, state, control, logs, **kwargs):
        if not self.enabled or (not self.do_log and not self.summary_writer):
            return
        device = torch.cuda.current_device()
        if self.enable_memory_snapshot:
            # Decode at https://docs.pytorch.org/memory_viz
            output_file = f"{self.file_prefix}_rank{self.rank}.pickle"
            logger.info(f"Saving memory snapshot to {output_file}")
            try:
                torch.cuda.memory._dump_snapshot(output_file)
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {output_file}")
            torch.cuda.memory._record_memory_history(enabled=None)
            # Only take a single snapshot
            self.enable_memory_snapshot = False

        max_allocated = torch.cuda.max_memory_allocated(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
        self.max_allocated = max(self.max_allocated, max_allocated)
        if self.world_size > 1:
            max_allocated_tensor = torch.tensor(max_allocated, device=device)
            if self.rank == 0:
                gather_list = [
                    torch.zeros_like(max_allocated_tensor, device=device)
                    for i in range(self.world_size)
                ]
            else:
                gather_list = None
            dist.gather(max_allocated_tensor, gather_list, dst=0)
            if self.rank != 0:
                return
            # Only rank 0 will log the peak memory
            max_allocated_list = [i.item() for i in gather_list]
        else:
            max_allocated_list = [max_allocated]

        if self.summary_writer:
            for i, mem in enumerate(max_allocated_list):
                self.summary_writer.add_scalar(
                    f"peak_memory_rank{i}", mem, global_step=state.global_step
                )
            self.summary_writer.flush()
            if self.show_details:
                details = torch.cuda.memory_stats(device)
                self.summary_writer.add_text(
                    f"peak_memory_details",
                    self._mapping_as_markdown(details),
                    global_step=state.global_step,
                )
        if self.do_log:
            s = "Peak CUDA Memory Allocated: "
            for i, mem in enumerate(max_allocated_list):
                s += f"RANK{i} {self._format_peak_memory(mem)}, "
            logger.info(s)
            if self.show_details and not self.summary_writer:
                details = torch.cuda.memory_stats(device)
                logger.info(f"RANK{self.rank} Peak Memory Details: {pformat(details)}")

    def on_train_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.enabled:
            return
        max_allocated = torch.cuda.max_memory_allocated()
        self.max_allocated = max(self.max_allocated, max_allocated)
        if self.enable_memory_snapshot:
            torch.cuda.memory._record_memory_history(enabled=None)
        logger.info(
            f"RANK{self.rank} MAX CUDA MEMORY ALLOCATED: {self._format_peak_memory(self.max_allocated)}"
        )
        if self.show_details and not self.do_log:
            details = torch.cuda.memory_stats(torch.cuda.current_device())
            if self.summary_writer:
                self.summary_writer.add_text(
                    "peak_memory_details",
                    self._mapping_as_markdown(details),
                    global_step=state.global_step,
                )
            logger.info(f"RANK{self.rank}: {pformat(details)}")
