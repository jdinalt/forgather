"""Tests for the scheduler's idle-GPU decision.

The bug this protects against: on a workstation with a connected monitor,
the desktop's X server / Wayland compositor / window manager shows up as
a CUDA-using process on the GPU, which previously caused the scheduler
to refuse to dispatch any job to that card. The fix removes external-
process inspection from the dispatch rule entirely; the scheduler only
gates on operator/user controls (CUDA_VISIBLE_DEVICES exclusion and
runtime disable). External processes — desktop tools, hybrid C+G
daemons, unrelated user CUDA work — are surfaced to the UI but don't
gate dispatch.
"""

from unittest.mock import patch

import forgather_server.gpu_monitor as gpu_monitor
import forgather_server.scheduler as scheduler
from forgather_server.gpu_monitor import GpuInfo, GpuProcess


class TestSchedulerIdleGpuIndices:
    def _gpu(self, index, processes, **kw):
        defaults = dict(
            index=index,
            name="RTX 4090",
            total_mem_bytes=24 * 1024**3,
            used_mem_bytes=0,
            processes=processes,
            excluded=False,
            disabled=False,
            min_priority=0,
        )
        defaults.update(kw)
        return GpuInfo(**defaults)

    def test_gpu_with_only_graphics_processes_is_idle(self):
        # The exact bug: a workstation desktop GPU running the X server
        # plus a compositor must still be available for dispatch.
        gpus = [
            self._gpu(
                0,
                [
                    GpuProcess(
                        pid=1,
                        used_mem_bytes=120 * 1024**2,
                        name="Xorg",
                        kind="graphics",
                    ),
                    GpuProcess(
                        pid=2,
                        used_mem_bytes=80 * 1024**2,
                        name="gnome-shell",
                        kind="graphics",
                    ),
                ],
            ),
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0]

    def test_gpu_with_unknown_compute_process_is_still_idle(self):
        # Trade-off: external CUDA programs the user is running outside
        # Forgather no longer block dispatch. The user's escape valve is
        # the disable button; without that the scheduler is happy to
        # share the GPU. (The previous behaviour — refusing to dispatch
        # — was prone to false positives from desktop tools holding
        # CUDA contexts.)
        gpus = [
            self._gpu(
                0,
                [
                    GpuProcess(
                        pid=99,
                        used_mem_bytes=8 * 1024**3,
                        name="python",
                        kind="compute",
                    )
                ],
            ),
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0]

    def test_compositor_reported_in_compute_list_is_idle(self):
        # NVIDIA driver edge case: the compositor (or a hybrid C+G
        # daemon like gnome-remote-desktop-daemon) shows up in the
        # compute list. Pre-fix, this disqualified the GPU. Post-fix,
        # external-process kind doesn't matter at all.
        gpus = [
            self._gpu(
                0,
                [
                    GpuProcess(
                        pid=1,
                        used_mem_bytes=200 * 1024**2,
                        name="gnome-shell",
                        kind="compute",
                    ),
                    GpuProcess(
                        pid=2,
                        used_mem_bytes=260 * 1024**2,
                        name="gnome-remote-desktop-daemon",
                        kind="compute",
                    ),
                ],
            ),
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0]

    def test_disabled_gpu_never_idle(self):
        # The user's escape valve: clicking "disable" in the GPU panel
        # keeps Forgather off the card regardless of process state.
        gpus = [
            self._gpu(
                0,
                [
                    GpuProcess(
                        pid=1,
                        used_mem_bytes=200 * 1024**2,
                        name="Xorg",
                        kind="graphics",
                    )
                ],
                disabled=True,
            )
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == []

    def test_excluded_gpu_never_idle(self):
        # CUDA_VISIBLE_DEVICES exclusion at server start.
        gpus = [self._gpu(0, [], excluded=True)]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == []

    def test_clean_gpu_is_idle(self):
        gpus = [self._gpu(0, [])]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0]

    def test_multiple_gpus_filtered_independently(self):
        # GPU 0 has the desktop, GPU 1 is disabled, GPU 2 is excluded,
        # GPU 3 is clean. Only 0 and 3 should dispatch.
        gpus = [
            self._gpu(
                0,
                [
                    GpuProcess(
                        pid=1,
                        used_mem_bytes=200 * 1024**2,
                        name="Xorg",
                        kind="graphics",
                    )
                ],
            ),
            self._gpu(1, [], disabled=True),
            self._gpu(2, [], excluded=True),
            self._gpu(3, []),
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0, 3]
