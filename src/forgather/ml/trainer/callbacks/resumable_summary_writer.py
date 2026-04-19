"""
Resumable TensorBoard SummaryWriter wrapper.

Provides a lazy, checkpoint-aware wrapper around SummaryWriter that
continues logging into the original run directory when training resumes
from a checkpoint, preventing log fragmentation across directories.
"""

import logging
import os

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.tensorboard import SummaryWriter

from ..trainer_types import TrainerCallback

logger = logging.getLogger(__name__)


class ResumableSummaryWriter(TrainerCallback, Stateful):
    """
    A lazy, resumable wrapper around TensorBoard SummaryWriter.

    When registered as a callback, it persists the active logging directory
    in checkpoint metadata via the Stateful protocol. On resume from a
    checkpoint, it redirects logging to the original directory and uses
    SummaryWriter's ``purge_step`` to discard stale events recorded after
    the checkpoint step.

    When used as a SummaryWriter (passed to TBLogger, GradLogger, etc.),
    it proxies method calls to the underlying writer, constructing it
    lazily on first use.

    Args:
        log_dir: Logging directory path (typically ``ns.logging_dir``
                 from the template system).
    """

    def __init__(self, log_dir: str):
        super().__init__()
        self._new_log_dir = log_dir
        self._active_log_dir = log_dir
        self._writer: SummaryWriter | None = None
        self._purge_step: int | None = None
        self._resumed = False

    # -- Stateful protocol --------------------------------------------------

    def state_dict(self) -> dict:
        return {"log_dir": self._active_log_dir}

    def load_state_dict(self, state_dict: dict) -> None:
        original_dir = state_dict.get("log_dir")
        if original_dir and os.path.isdir(original_dir):
            logger.info(
                "ResumableSummaryWriter: resuming into original log dir: %s",
                original_dir,
            )
            self._active_log_dir = original_dir
            self._resumed = True
        else:
            logger.warning(
                "ResumableSummaryWriter: original log dir not found (%s), "
                "using new directory: %s",
                original_dir,
                self._new_log_dir,
            )

    # -- TrainerCallback protocol -------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        if self._resumed and state.global_step > 0:
            self._purge_step = state.global_step
            logger.info(
                "ResumableSummaryWriter: will purge TensorBoard events "
                "after step %d",
                self._purge_step,
            )

    # -- Lazy writer construction -------------------------------------------

    def _ensure_writer(self) -> SummaryWriter:
        if self._writer is None:
            kwargs: dict = {}
            if self._purge_step is not None:
                kwargs["purge_step"] = self._purge_step
            os.makedirs(self._active_log_dir, exist_ok=True)
            self._writer = SummaryWriter(self._active_log_dir, **kwargs)
            logger.info(
                "ResumableSummaryWriter: created SummaryWriter at %s",
                self._active_log_dir,
            )
        return self._writer

    # -- SummaryWriter method proxies ---------------------------------------

    def add_scalar(self, *args, **kwargs):
        return self._ensure_writer().add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        return self._ensure_writer().add_scalars(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        return self._ensure_writer().add_text(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        return self._ensure_writer().add_histogram(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        return self._ensure_writer().add_image(*args, **kwargs)

    def add_images(self, *args, **kwargs):
        return self._ensure_writer().add_images(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        return self._ensure_writer().add_figure(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        return self._ensure_writer().add_graph(*args, **kwargs)

    def flush(self):
        if self._writer is not None:
            self._writer.flush()

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __del__(self):
        self.close()
