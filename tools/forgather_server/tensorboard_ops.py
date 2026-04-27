"""Server-side wrappers for launching a TensorBoard instance.

TensorBoard is a pure viewer over trainer_logs / TF event files — it needs
no GPUs, no forgather imports, and no special launch harness. The scheduler
treats it as a job type so it queues alongside training / eval / inference
and shows up in the unified Jobs view with TTY + kill controls; the only
actual difference is ``requested_gpus`` is typically zero.

The server shells out to the stock ``tensorboard`` CLI (same binary the
``forgather tb`` CLI wraps) rather than re-implementing anything from it.
"""

from __future__ import annotations

from typing import List, Optional


def build_tensorboard_command(
    *,
    logdir: str,
    port: int,
    host: Optional[str] = None,
    bind_all: bool = False,
    window_title: Optional[str] = None,
    reload_interval: Optional[int] = None,
    reload_multifile: bool = False,
    samples_per_plugin: Optional[str] = None,
) -> List[str]:
    """Build the argv for ``tensorboard``.

    Only ``logdir`` and ``port`` are required. ``bind_all`` and ``host``
    are mutually exclusive on the CLI — if both are passed, ``--bind_all``
    wins (matching the CLI's own precedence).

    The "advanced" knobs (``reload_interval``, ``reload_multifile``,
    ``samples_per_plugin``) surface the options a Forgather user
    actually tends to reach for during long-running experiments; the rest
    of TensorBoard's sprawling flag surface is left to anyone who needs
    it to run their own ``tensorboard`` directly.
    """
    cmd: List[str] = [
        "tensorboard",
        "--logdir",
        logdir,
        "--port",
        str(port),
    ]
    if bind_all:
        cmd.append("--bind_all")
    elif host:
        cmd.extend(["--host", host])
    if window_title:
        cmd.extend(["--window_title", window_title])
    if reload_interval is not None:
        cmd.extend(["--reload_interval", str(reload_interval)])
    if reload_multifile:
        cmd.extend(["--reload_multifile", "true"])
    if samples_per_plugin:
        cmd.extend(["--samples_per_plugin", samples_per_plugin])
    return cmd
