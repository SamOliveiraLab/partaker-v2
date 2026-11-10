"""Recordings saved as one TIFF per time point and named by the hour ("66.00h.tif", "66.25h.tif", ...).

Pure functions, so the import dialog and the tests read a folder the same way and nothing is ever renamed on disk."""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import numpy as np

HOUR_NAME = re.compile(r"^(?P<h>\d+(?:\.\d+)?)\s*h\.(tif|tiff)$", re.IGNORECASE)


def hour_of(path: str) -> Optional[float]:
    """The hour written in a file name like 66.25h.tif, else None."""
    m = HOUR_NAME.match(os.path.basename(path))
    return float(m.group("h")) if m else None


def hour_named_file_map(paths: Sequence[str]):
    """(file_map, hours) for a set of hour-named TIFFs: one position, one channel, time index in hour order.
    Hidden files are left out. None when a file does not carry an hour, two files carry the same hour, or there are
    fewer than two files."""
    paths = [p for p in paths if not os.path.basename(p).startswith(".")]      # macOS leaves "._name" files on some disks
    hours = [hour_of(p) for p in paths]
    if len(paths) < 2 or any(h is None for h in hours) or len(set(hours)) != len(hours):
        return None
    order = np.argsort(hours)
    file_map = {(0, t, 0): paths[i] for t, i in enumerate(order)}
    return file_map, [float(hours[i]) for i in order]


def interval_s(hours: Sequence[float]) -> float:
    """The time between frames in seconds: the median step of the hours."""
    return float(np.median(np.diff(np.asarray(hours, float))) * 3600.0)
