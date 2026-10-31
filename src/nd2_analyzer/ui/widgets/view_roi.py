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
