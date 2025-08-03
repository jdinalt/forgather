import os
import torch
from pprint import pformat
import logging
import torch.distributed as dist

from .trainer_types import (
    TrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeakMemory(TrainerCallback):
    """
    Records peak CUDA memory
    """

    def __init__(self, summary_writer, show_details=False, do_log=False):
        super().__init__()
        try:
            self.rank = os.environ["RANK"]
        except KeyError:
            self.rank = 0
        try:
            self.world_size = os.environ["WORLD_SIZE"]
        except KeyError:
            self.world_size = 1

        self.summary_writer = summary_writer
        self.enabled = torch.cuda.is_available()
        self.show_details = show_details
        self.do_log = do_log

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        self.max_allocated = 0
        torch.cuda.memory._record_memory_history(enabled="all")

    def on_log(self, args, state, control, logs, **kwargs):
        if not self.enabled or not self.do_log:
            return
        device = torch.cuda.current_device()
        max_allocated = torch.cuda.max_memory_allocated()
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
            max_allocated_list = [i.item() for i in gather_list]
        else:
            max_allocated_list = [max_allocated]
        self.summary_writer.add_scalars(
            "peak_memory",
            {f"rank{i}" for i, mem in enumerate(max_allocated_list)},
            state.global_step,
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.enabled:
            return
        max_allocated = torch.cuda.max_memory_allocated()
        self.max_allocated = max(self.max_allocated, max_allocated)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(
            f"RANK{self.rank} MAX CUDA MEMORY ALLOCATED: {self.max_allocated / 1000000000.:.3f} GB"
        )
        if self.show_details:
            details = torch.cuda.memory_stats(torch.cuda.current_device())
            print(f"RANK{self.rank}: {pformat(details)}")
