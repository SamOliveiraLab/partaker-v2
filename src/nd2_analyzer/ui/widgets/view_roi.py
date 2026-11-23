"""The view inside the ROI: once the chamber is drawn, the image panel shows that window of the frame, the outside black,
and clicks map back to full-frame coordinates. Plain numpy, used by the view area and its tests."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def roi_box(roi_mask: np.ndarray) -> Dict[str, int]:
    rr, cc = np.where(np.asarray(roi_mask) > 0)
    return dict(row0=int(rr.min()), row1=int(rr.max()) + 1, col0=int(cc.min()), col1=int(cc.max()) + 1)


def roi_window(image: np.ndarray, roi_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    """The frame cut to the ROI's box with everything outside the ROI set to black. Without an ROI, or when the mask does
    not match the frame, the frame comes back untouched (and the box is None)."""
    if roi_mask is None:
        return image, None
    roi = np.asarray(roi_mask) > 0
    if roi.shape != tuple(image.shape[:2]) or not roi.any():
        return image, None
    b = roi_box(roi)
    win = np.array(image[b["row0"]:b["row1"], b["col0"]:b["col1"]], copy=True)
    outside = ~roi[b["row0"]:b["row1"], b["col0"]:b["col1"]]
    win[outside] = 0
    return win, b


def window_to_frame(x: float, y: float, box: Optional[Dict[str, int]]) -> Tuple[int, int]:
    """A point in the shown window -> the same point in the full frame."""
    if box is None:
        return int(x), int(y)
    return int(x) + box["col0"], int(y) + box["row0"]


def inset_window(image, shown, box, margin=15):
    """Hide an excluded frame margin while preserving source coordinates."""
    height, width = image.shape[:2]
    if min(height, width) <= 2 * margin:
        return shown, box
    prior = box or dict(row0=0, row1=height, col0=0, col1=width)
    bounds = dict(row0=max(margin, prior["row0"]), row1=min(height - margin, prior["row1"]),
                  col0=max(margin, prior["col0"]), col1=min(width - margin, prior["col1"]))
    if bounds["row0"] >= bounds["row1"] or bounds["col0"] >= bounds["col1"]:
        return shown, box
    window = shown[bounds["row0"] - prior["row0"]:bounds["row1"] - prior["row0"],
                   bounds["col0"] - prior["col0"]:bounds["col1"] - prior["col0"]]
    return window, bounds
