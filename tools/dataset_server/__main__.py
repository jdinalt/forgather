"""``python -m tools.dataset_server`` entry point.

Defers to :func:`tools.dataset_server.server.main` so the same code
path serves both the standalone-script and module invocations.
"""

from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
