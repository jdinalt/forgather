"""Tests for tools/forgather_server/gpu_monitor.py.

Covers the busy-process classifier that drives the scheduler's idle-GPU
decision. The bug this protects against: on a workstation with a connected
monitor, the desktop's X server / Wayland compositor / window manager shows
up as a CUDA-using process on the GPU, which previously caused the
scheduler to refuse to dispatch any job to that card. Graphics-only
processes must NOT block dispatch; real compute processes still must.
"""

from unittest.mock import patch

import forgather_server.gpu_monitor as gpu_monitor
import forgather_server.scheduler as scheduler
from forgather_server.gpu_monitor import GpuInfo, GpuProcess, is_blocking_process


class TestIsBlockingProcess:
    def test_compute_process_with_unknown_name_blocks(self):
        # Real training jobs end up here: kind=compute, name not on allowlist.
        p = GpuProcess(
            pid=1234, used_mem_bytes=8 * 1024**3, name="python", kind="compute"
        )
        assert is_blocking_process(p) is True

    def test_compute_process_with_no_name_blocks(self):
        # When name resolution fails, default to "blocks" — safer choice.
        p = GpuProcess(pid=1234, used_mem_bytes=8 * 1024**3, name=None, kind="compute")
        assert is_blocking_process(p) is True

    def test_graphics_process_does_not_block(self):
        # Pure graphics process (kind=graphics) — never blocks regardless of name.
        p = GpuProcess(
            pid=999, used_mem_bytes=200 * 1024**2, name="Xorg", kind="graphics"
        )
        assert is_blocking_process(p) is False

    def test_compute_process_with_desktop_name_does_not_block(self):
        # NVIDIA driver sometimes reports the compositor on the *compute*
        # list. Name-based filter must catch those too.
        for name in ("Xorg", "gnome-shell", "kwin_x11", "plasmashell", "mutter"):
            p = GpuProcess(
                pid=10, used_mem_bytes=300 * 1024**2, name=name, kind="compute"
            )
            assert is_blocking_process(p) is False, f"{name} should not block"


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

    def test_gpu_with_compute_process_is_busy(self):
        # A real training job (or anything compute-shaped) must still block.
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
            assert scheduler._idle_gpu_indices() == []

    def test_gpu_with_mixed_processes_is_busy(self):
        # Desktop compositor sharing the card with a real compute job
        # — compute job wins, GPU stays busy.
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
                        pid=99,
                        used_mem_bytes=8 * 1024**3,
                        name="python",
                        kind="compute",
                    ),
                ],
            ),
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == []

    def test_gpu_with_compositor_in_compute_list_is_idle(self):
        # NVIDIA driver edge case: the compositor shows up on the compute
        # list. Name-based allowlist must still let the GPU through.
        gpus = [
            self._gpu(
                0,
                [
                    GpuProcess(
                        pid=1,
                        used_mem_bytes=200 * 1024**2,
                        name="gnome-shell",
                        kind="compute",
                    )
                ],
            ),
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0]

    def test_disabled_gpu_never_idle_even_if_clean(self):
        # Sanity: the existing disabled/excluded gates still apply.
        gpus = [self._gpu(0, [], disabled=True)]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == []

    def test_excluded_gpu_never_idle_even_if_clean(self):
        gpus = [self._gpu(0, [], excluded=True)]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == []

    def test_clean_gpu_is_idle(self):
        gpus = [self._gpu(0, [])]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0]

    def test_multiple_gpus_filtered_independently(self):
        # GPU 0 has the desktop, GPU 1 has a real compute job, GPU 2 is
        # clean. Only GPU 0 and GPU 2 should dispatch.
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
            self._gpu(
                1,
                [
                    GpuProcess(
                        pid=99,
                        used_mem_bytes=8 * 1024**3,
                        name="python",
                        kind="compute",
                    )
                ],
            ),
            self._gpu(2, []),
        ]
        with patch.object(gpu_monitor, "snapshot", return_value=gpus):
            assert scheduler._idle_gpu_indices() == [0, 2]


class TestDesktopProcessAllowlist:
    def test_default_list_includes_common_compositors(self):
        names = gpu_monitor.desktop_graphics_processes()
        for expected in ("Xorg", "gnome-shell", "kwin_x11", "plasmashell", "mutter"):
            assert expected in names
