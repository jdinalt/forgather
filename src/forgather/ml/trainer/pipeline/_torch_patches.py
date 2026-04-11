"""
Runtime patches for torch.distributed.pipelining to interoperate with
non-tensor forward kwargs (e.g., flex_attention BlockMask).

Applied at import time by pipeline_trainer so the fix is active whenever
the pipeline trainer is used.
"""

import logging

import torch
from torch.distributed.pipelining import _backward as _pp_backward
from torch.distributed.pipelining import stage as _pp_stage

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_forgather_non_tensor_input_patch"

# Names of zero-bubble schedule classes whose backward splits the input-gradient
# step from the weight-gradient step. These schedules call ``stage_backward_input``
# with ``retain_graph=True``, which is incompatible with donated buffers in
# torch.compile'd backward (used by flex_attention).
_ZERO_BUBBLE_SCHEDULE_NAMES = frozenset(
    {
        "ScheduleInterleavedZeroBubble",
        "ScheduleZBVZeroBubble",
    }
)


def _apply_stage_backward_input_patch() -> None:
    """
    Wrap ``stage_backward_input`` so that non-tensor entries in ``input_values``
    (e.g. flex_attention ``BlockMask`` objects passed through as forward kwargs)
    are filtered out before gradient-edge discovery.

    Zero-bubble schedules (``ScheduleInterleavedZeroBubble``,
    ``ScheduleZBVZeroBubble``) split the backward into an input-gradient step
    and a weight-gradient step. The input-gradient step calls
    ``stage_backward_input`` on every flattened forward argument, which
    unconditionally reads ``requires_grad`` — raising ``AttributeError`` when
    the arg is a non-tensor object like ``BlockMask``. Full-backward schedules
    (1F1B, GPipe) go through ``torch.autograd.backward`` directly and are
    unaffected.

    The fix: prefilter ``input_values`` to tensors. Non-tensor kwargs carry no
    autograd information, so dropping them is safe.
    """
    target = _pp_backward.stage_backward_input
    if getattr(target, _PATCH_ATTR, False):
        return

    def stage_backward_input_tensor_filter(
        stage_outputs_or_loss,
        output_grads,
        input_values,
        weights,
    ):
        filtered = [v for v in input_values if isinstance(v, torch.Tensor)]
        return target(stage_outputs_or_loss, output_grads, filtered, weights)

    setattr(stage_backward_input_tensor_filter, _PATCH_ATTR, True)

    _pp_backward.stage_backward_input = stage_backward_input_tensor_filter
    # stage.py binds stage_backward_input at import time, so patch the
    # re-exported reference as well.
    setattr(_pp_stage, "stage_backward_input", stage_backward_input_tensor_filter)

    logger.debug(
        "Patched torch.distributed.pipelining.stage_backward_input to tolerate "
        "non-tensor forward kwargs (needed for flex_attention BlockMask under "
        "zero-bubble schedules)."
    )


_apply_stage_backward_input_patch()


def is_zero_bubble_schedule(schedule_factory) -> bool:
    """Return True if the given schedule class/factory uses a zero-bubble
    split backward (I/W actions). Unwraps ``functools.partial`` so it works
    with the ``!partial:...`` bindings produced by Forgather config templates.
    """
    import functools

    target = schedule_factory
    while isinstance(target, functools.partial):
        target = target.func
    name = getattr(target, "__name__", None)
    return name in _ZERO_BUBBLE_SCHEDULE_NAMES


def disable_compiled_backward_donated_buffers() -> None:
    """Disable ``torch._functorch.config.donated_buffer``.

    Zero-bubble schedules call ``stage_backward_input`` with
    ``retain_graph=True`` so the later weight-gradient step can reuse the
    graph. This is incompatible with donated-buffer optimization used by
    torch.compile'd backward (most visibly, flex_attention), which raises:

        "This backward function was compiled with non-empty donated buffers
         which requires create_graph=False and retain_graph=False."

    Must be called before the first flex_attention (or other compiled
    backward) invocation, otherwise the compiled backward has already been
    captured with donated buffers and the runtime check will still fire.
    """
    import torch._functorch.config as _functorch_config

    if _functorch_config.donated_buffer:
        _functorch_config.donated_buffer = False
        logger.info(
            "Disabled torch._functorch.config.donated_buffer for zero-bubble "
            "pipeline schedule compatibility with compiled backward."
        )
