import math
import os
import re
from pprint import pformat
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..trainer_types import TrainerCallback


class GradLogger(TrainerCallback):
    def __init__(self, summary_writer, neg_re, pos_re, log_steps, **kwargs):
        super().__init__()
        self.summary_writer = summary_writer
        self.log_steps = log_steps
        self.neg_re = neg_re
        self.pos_re = pos_re

    @staticmethod
    @torch.no_grad()
    def rms(x):
        return x.square().mean().sqrt().item()

    def on_optimizer_step(self, args, state, control, /, model, **kwargs):
        if not state.is_world_process_zero:
            return

        if state.global_step % self.log_steps != 0:
            return

        for param_name, p in model.named_parameters():
            if p.requires_grad == False or re.search(self.neg_re, param_name):
                continue
            if not re.search(self.pos_re, param_name):
                continue

            self.summary_writer.add_scalar(
                f"weight:{param_name}", self.rms(p), global_step=state.global_step
            )
            self.summary_writer.add_scalar(
                f"grad:{param_name}", self.rms(p.grad), global_step=state.global_step
            )
