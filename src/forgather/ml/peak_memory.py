import os
import torch
from pprint import pformat
import logging

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
    def __init__(self, show_details=False):
        super().__init__()
        try:
            self.rank = os.environ['RANK']
        except KeyError:
            self.rank = 0
        self.enabled = torch.cuda.is_available()
        self.show_details = show_details

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        torch.cuda.memory._record_memory_history(enabled='all')
        
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
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"RANK{self.rank} MAX CUDA MEMORY ALLOCATED: {max_allocated / 1000000000.:.3f} GB")
        if self.show_details:
            details = torch.cuda.memory_stats(torch.cuda.current_device())
            print(f"RANK{self.rank}: {pformat(details)}")
