"""Regression guard: the CLI startup path must stay lightweight.

``forgather --help`` and ``forgather ls`` only build argument parsers and
render help/descriptions — they must never import heavy ML stacks (``torch``,
``numpy``, ``transformers``, ``datasets``). The heavy import is paid lazily by
the command *implementation* (imported per-``case`` in ``main.main``), not by
the parser-builder (``*_args``) modules that the subcommand registry imports
eagerly.

Historically this regressed when ``diloco_args`` did a top-level
``from forgather.ml.diloco.auth import add_auth_args``: importing that
stdlib-only submodule ran ``forgather.ml.diloco.__init__``, which eagerly
imported ``.client`` and pulled in ``torch`` — adding ~2.3s to every
``forgather --help`` / ``forgather ls``. These tests run a fresh interpreter
and assert the heavy modules never land in ``sys.modules``.
"""

import subprocess
import sys

import pytest

# Modules whose presence in a fresh `--help` / `ls`-equivalent interpreter
# signals a heavyweight import has leaked into the CLI startup path.
HEAVY_MODULES = ("torch", "numpy", "transformers", "datasets")

# Build the full subcommand registry and invoke every parser factory — this is
# exactly what `show_main_help` does for `forgather --help`, and is a superset
# of what any single subcommand (e.g. `ls`) imports. Then assert no heavy
# module was imported as a side effect.
_PROBE = """
import argparse, sys
from forgather.cli.main import get_subcommand_registry

registry = get_subcommand_registry()
dummy = argparse.Namespace(project_dir=".", config_template=None, no_dyn=True)
for name, factory in registry.items():
    try:
        factory(dummy)
    except Exception:
        # A factory that fails to build is a separate bug; this probe only
        # cares about import side effects, which already happened on import.
        pass

heavy = [m for m in {heavy!r} if m in sys.modules]
print("HEAVY:" + ",".join(heavy))
sys.exit(1 if heavy else 0)
"""


def _run_probe() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=HEAVY_MODULES)],
        capture_output=True,
        text=True,
    )


def test_building_parsers_does_not_import_heavy_modules():
    """The subcommand registry + all parser factories stay torch-free."""
    result = _run_probe()
    assert result.returncode == 0, (
        "Heavyweight import leaked into CLI parser construction "
        f"(slows `forgather --help` / `ls`):\n"
        f"  stdout: {result.stdout.strip()}\n"
        f"  stderr: {result.stderr.strip()}"
    )


@pytest.mark.parametrize("subcommand", ["--help", "ls"])
def test_cli_startup_does_not_import_torch(subcommand):
    """End-to-end: running the CLI command in a fresh process is torch-free."""
    probe = (
        "import sys, runpy\n"
        f"sys.argv = ['forgather', {subcommand!r}]\n"
        "try:\n"
        "    runpy.run_module('forgather.cli', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules, 'torch imported by forgather "
        f"{subcommand}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"`forgather {subcommand}` imported a heavy module:\n"
        f"  stderr: {result.stderr.strip()}"
    )
