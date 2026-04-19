"""Per-parameter gradient norm logging callback."""

import logging
from collections import OrderedDict

import torch
from torch.distributed.checkpoint.stateful import Stateful

from ..trainer_types import TrainerCallback
from .json_log_writer import JsonLogWriter

logger = logging.getLogger(__name__)


class GradNormLogger(TrainerCallback, Stateful):
    """Logs per-parameter gradient L2 norms to a JSON file.

    Gradient norms are captured in ``on_pre_optimizer_step`` (after gradient
    clipping, before optimizer step and zero_grad) and written to the log
    file in ``on_evaluate``. This means gradient data is logged at eval
    frequency, keeping overhead minimal.

    The log file uses JSON array format with checkpoint resume support
    via the Stateful protocol.

    When ``fuse_optim_with_backward`` is enabled, gradients are consumed
    during the backward pass and are not available for capture. The callback
    detects this and disables itself with a warning.
    """

    LOG_FILENAME = "gradient_norms.json"

    def __init__(self):
        super().__init__()
        self._writer = JsonLogWriter(self.LOG_FILENAME)
        self._buffered_norms: dict[str, float] | None = None
        self._buffered_step: int = -1
        self._buffered_epoch: float = 0.0
        self._disabled = False
        self._warned_meta = False

    # -- Stateful protocol ----------------------------------------------------

    def state_dict(self) -> dict:
        return {"writer": self._writer.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        writer_state = state_dict.get("writer", {})
        self._writer.load_state_dict(writer_state)

    # -- Callback hooks -------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or args.logging_dir is None:
            return

        if getattr(args, "fuse_optim_with_backward", False):
            logger.warning(
                "GradNormLogger: fuse_optim_with_backward is enabled, "
                "gradients are consumed during backward. "
                "Gradient norm logging is disabled."
            )
            self._disabled = True
            return

        self._writer.open(args.logging_dir)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Capture per-parameter gradient norms before optimizer step."""
        if self._disabled or not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        if model is None:
            return

        # Pipeline-parallel guard
        try:
            first_param = next(model.parameters())
        except StopIteration:
            first_param = None

        if first_param is None or first_param.device.type == "meta":
            if not self._warned_meta:
                logger.warning(
                    "GradNormLogger: model parameters are on the meta "
                    "device (pipeline-parallel training). Logging disabled."
                )
                self._warned_meta = True
            return

        norms = OrderedDict()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    norms[name] = p.grad.float().norm().item()

        self._buffered_norms = norms
        self._buffered_step = state.global_step
        self._buffered_epoch = state.epoch

    def on_evaluate(self, args, state, control, **kwargs):
        """Write buffered gradient norms to log file."""
        if self._disabled or not state.is_world_process_zero:
            return
        if not self._writer.is_open or self._buffered_norms is None:
            return

        record = {"grad_norms": self._buffered_norms}
        self._writer.write_record(self._buffered_step, self._buffered_epoch, record)
        self._buffered_norms = None

    def on_train_end(self, args, state, control, **kwargs):
        self._writer.close()
