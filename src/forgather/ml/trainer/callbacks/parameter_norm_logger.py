"""Per-parameter weight norm and spectral norm logging callback."""

import logging
from collections import OrderedDict

import torch
from torch.distributed.checkpoint.stateful import Stateful

from ..trainer_types import TrainerCallback
from .json_log_writer import JsonLogWriter

logger = logging.getLogger(__name__)


def _spectral_norm_power_iter(weight, n_iters, u=None):
    """Estimate the largest singular value via power iteration.

    Args:
        weight: Parameter tensor (2D+ supported; reshaped to 2D internally).
        n_iters: Number of power iteration steps.
        u: Optional warm-start left singular vector from a previous call.

    Returns:
        (sigma, u) where sigma is the estimated spectral norm and u is the
        left singular vector for warm-starting the next call.
    """
    if weight.ndim < 2:
        # For 1D tensors (biases, layer norm weights), the spectral norm
        # is simply the largest absolute value.
        return weight.abs().max().item(), None

    h = weight.shape[0]
    weight_mat = weight.reshape(h, -1).float()

    if u is None:
        u = torch.randn(h, device=weight.device, dtype=torch.float32)
        u = u / u.norm()
    else:
        u = u.to(device=weight.device, dtype=torch.float32)

    v = weight_mat.t() @ u
    v = v / (v.norm() + 1e-12)

    with torch.no_grad():
        for _ in range(n_iters):
            u = weight_mat @ v
            u = u / (u.norm() + 1e-12)
            v = weight_mat.t() @ u
            v = v / (v.norm() + 1e-12)
        sigma = (u @ weight_mat @ v).item()

    return sigma, u


class ParameterNormLogger(TrainerCallback, Stateful):
    """Logs per-parameter L2 norms and/or spectral norms to a JSON file.

    Data is written on each evaluation step. The log file uses JSON array
    format with checkpoint resume support via the Stateful protocol.

    The existing ``WeightNormLogger`` continues to handle the total
    parameter norm for TensorBoard/console logging. This callback provides
    the per-parameter breakdown for diagnostic analysis and heatmap
    visualization.

    In pipeline-parallel training the model shell passed to callbacks
    contains only meta-device tensors. This callback detects that case,
    warns once, and skips logging for the remainder of training.
    """

    LOG_FILENAME = "parameter_norms.json"

    def __init__(
        self,
        log_norms: bool = True,
        log_spectral_norms: bool = True,
        power_iter_steps: int = 10,
    ):
        """
        Args:
            log_norms: Whether to log per-parameter L2 norms.
            log_spectral_norms: Whether to log per-parameter spectral norms.
            power_iter_steps: Number of power iteration steps for spectral
                norm estimation. First evaluation uses 2x this value for
                cold-start convergence.
        """
        super().__init__()
        self.log_norms = log_norms
        self.log_spectral_norms = log_spectral_norms
        self.power_iter_steps = power_iter_steps

        self._writer = JsonLogWriter(self.LOG_FILENAME)
        self._warned_meta = False
        # Cached direction vectors for power iteration warm-starting,
        # keyed by parameter FQN.
        self._u_vectors: dict[str, torch.Tensor] = {}
        self._first_eval = True

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
        self._writer.open(args.logging_dir)

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self._writer.is_open:
            return

        model = kwargs.get("model")
        if model is None:
            return

        # Pipeline-parallel guard: detect meta-device tensors.
        try:
            first_param = next(model.parameters())
        except StopIteration:
            first_param = None

        if first_param is None or first_param.device.type == "meta":
            if not self._warned_meta:
                logger.warning(
                    "ParameterNormLogger: model parameters are on the meta "
                    "device (pipeline-parallel training). Logging disabled."
                )
                self._warned_meta = True
            return

        record = {}

        n_iters = self.power_iter_steps
        if self._first_eval:
            n_iters *= 2
            self._first_eval = False

        with torch.no_grad():
            if self.log_norms:
                norms = OrderedDict()
                for name, p in model.named_parameters():
                    norms[name] = p.float().norm().item()
                record["norms"] = norms

            if self.log_spectral_norms:
                spectral_norms = OrderedDict()
                for name, p in model.named_parameters():
                    u = self._u_vectors.get(name)
                    sigma, u_new = _spectral_norm_power_iter(p.data, n_iters, u)
                    spectral_norms[name] = sigma
                    if u_new is not None:
                        self._u_vectors[name] = u_new
                record["spectral_norms"] = spectral_norms

        self._writer.write_record(state.global_step, state.epoch, record)

    def on_train_end(self, args, state, control, **kwargs):
        self._writer.close()
