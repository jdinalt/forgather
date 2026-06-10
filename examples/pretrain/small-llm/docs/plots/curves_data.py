"""Read the committed experiment curves (``curves.csv``).

The render scripts plot from this committed data so they work on a clean
checkout with no ``output_models/`` present. Regenerate ``curves.csv`` from the
run logs with ``extract_curves.py``.
"""

import csv
from pathlib import Path

CSV = Path(__file__).resolve().parent / "curves.csv"


def load_curves(path=CSV):
    """Return {(run, metric): [(step, value), ...]} sorted by step."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data.setdefault((row["run"], row["metric"]), []).append(
                (int(row["step"]), float(row["value"]))
            )
    for pts in data.values():
        pts.sort()
    return data


def series(data, run, metric, x_max=None):
    """(steps, values) for one run+metric, optionally clipped to step <= x_max."""
    pts = data.get((run, metric), [])
    if x_max is not None:
        pts = [(s, v) for s, v in pts if s <= x_max]
    return [s for s, _ in pts], [v for _, v in pts]
