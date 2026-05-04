"""Configure sys.path so the forgather_server package is importable.

This hook runs before pytest collects any test modules in this directory,
which means the import in each test file will already find forgather_server.
"""

import sys
from pathlib import Path


def pytest_configure(config):
    """Add tools/ to sys.path before any test module is imported."""
    tools_dir = str(Path(__file__).resolve().parents[4] / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
