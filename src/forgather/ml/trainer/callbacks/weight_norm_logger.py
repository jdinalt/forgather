import logging
import math

import torch

from ..trainer_types import TrainerCallback

logger = logging.getLogger(__name__)


class WeightNormLogger(TrainerCallback):
    """
    Logs the total L2 norm of all model parameters to TensorBoard after each
    evaluation step.

    Computed identically to the gradient norm but using the weight tensors
    themselves. A growing value over training indicates that weights are
    increasing in magnitude, which usually means weight decay is too weak.
    A stable or shrinking value while gradient norms rise points to a
    different cause.

    In pipeline-parallel training the model shell passed to callbacks contains
    only meta-device tensors. This callback detects that case, warns once, and
    skips logging for the remainder of training.
    """

    def __init__(self, summary_writer, tag: str = "weight_norm"):
        """
        Args:
            summary_writer: TensorBoard SummaryWriter instance.
            tag: TensorBoard scalar tag. Defaults to "weight_norm".
        """
        super().__init__()
        self.summary_writer = summary_writer
        self.tag = tag
        self._warned_meta = False

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        if model is None:
            return

        # In pipeline-parallel training the model shell holds no real tensors;
        # parameters live on the meta device and actual stage weights are stored
        # elsewhere in the trainer. Detect this and bail out.
        try:
            first_param = next(model.parameters())
        except StopIteration:
            first_param = None

        if first_param is None or first_param.device.type == "meta":
            if not self._warned_meta:
                logger.warning(
                    "WeightNormLogger: model parameters are on the meta device "
                    "(pipeline-parallel training). Weight norm logging is disabled."
                )
                self._warned_meta = True
            return

        total_norm_sq = 0.0
        with torch.no_grad():
            for p in model.parameters():
                total_norm_sq += p.float().square().sum().item()

        self.summary_writer.add_scalar(
            self.tag, math.sqrt(total_norm_sq), global_step=state.global_step
        )
        self.summary_writer.flush()
