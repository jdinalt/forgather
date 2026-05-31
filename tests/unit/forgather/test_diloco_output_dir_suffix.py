"""Regression test for issue #103: the DiLoCo per-worker output-dir suffix
must honor the SAME worker-id precedence the callback uses at registration
(``diloco_callback.py``: ``worker_id or DILOCO_WORKER_ID``) — the
``--diloco-worker-id`` Jinja dynamic-arg wins, the ``DILOCO_WORKER_ID`` env
var (scheduler default = queue_id) is the fallback.

Before the fix the suffix read ``DILOCO_WORKER_ID`` *only*, so a worker-id
supplied via the dynamic-arg channel left the output dir un-suffixed (or
suffixed with the env value), desyncing it from the registered worker-id
and breaking the DiLoCo view's job correlation.

Exercises the real example templates (``lm_training_project.yaml`` and the
``finetune_v2.yaml`` child that re-applies the suffix on an overridden
``ns.output_dir``). Skipped when the bundled example project isn't present.
"""

import os
import re

import pytest

from forgather.project import Project

# Repo root is four levels up from tests/unit/forgather/<this file>.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DILOCO_PROJ = os.path.join(_REPO_ROOT, "examples", "tiny_experiments", "diloco")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_DILOCO_PROJ),
    reason="bundled examples/tiny_experiments/diloco project not present",
)

# Base (un-suffixed) model dir name the diloco default config resolves to.
_BASE = "tinyv2"


def _resolved_output_dir(**kwargs) -> str:
    """Render the diloco default config and return the basename of the
    resolved ``ns.output_dir`` (the value the trainer would write under)."""
    proj = Project("default.yaml", _DILOCO_PROJ, **kwargs)
    text = str(proj.pp_config)
    # The variable listing emits a stable, unambiguous line for ns.output_dir.
    m = re.search(r"#\s*ns\.output_dir:\s*\"([^\"]+)\"", text)
    assert m, "ns.output_dir not found in preprocessed config"
    return os.path.basename(m.group(1))


def test_no_worker_id_is_unsuffixed(monkeypatch):
    """Non-DiLoCo / no worker-id: the dir is the bare model name."""
    monkeypatch.delenv("DILOCO_WORKER_ID", raising=False)
    assert _resolved_output_dir() == _BASE


def test_env_worker_id_suffixes(monkeypatch):
    """Scheduler/cluster path: the env var suffixes the dir (unchanged behavior)."""
    monkeypatch.setenv("DILOCO_WORKER_ID", "w0")
    assert _resolved_output_dir() == f"{_BASE}_w0"


def test_dynamic_arg_worker_id_suffixes(monkeypatch):
    """The bug: a worker-id from the --diloco-worker-id dynamic-arg must now
    suffix the dir (previously left it un-suffixed)."""
    monkeypatch.delenv("DILOCO_WORKER_ID", raising=False)
    assert _resolved_output_dir(diloco_worker_id="custom123") == f"{_BASE}_custom123"


def test_dynamic_arg_wins_over_env(monkeypatch):
    """Precedence must match registration (worker_id or env): the Jinja
    dynamic-arg wins when both channels are set."""
    monkeypatch.setenv("DILOCO_WORKER_ID", "w0")
    assert _resolved_output_dir(diloco_worker_id="custom123") == f"{_BASE}_custom123"
