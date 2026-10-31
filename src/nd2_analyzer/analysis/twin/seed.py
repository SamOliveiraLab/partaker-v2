"""The digital twin's seed: the cells of one frame, inside the ROI, as capsules in micrometres.

Port of DigitalTwin_rerun/export_frame0_seed.py and seed_v3.py (Oct 2026). A regionprops ellipse overstates a rod's length and
width by an amount that depends on its aspect ratio; the exact inversion below rasterises capsules once and solves every cell's
total length and width from its mask area and ellipse major axis, so the seed cells have the real cells' areas.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def roi_box(roi_mask: np.ndarray) -> Dict[str, int]:
    """Bounding box of the ROI as rows and columns (row1 and col1 exclusive)."""
    rr, cc = np.where(np.asarray(roi_mask) > 0)
    return dict(row0=int(rr.min()), row1=int(rr.max()) + 1, col0=int(cc.min()), col1=int(cc.max()) + 1)


class CapsuleTable:
    """Ellipse major axis and area of a capsule of total length k*w and width w, for k in [1, 8] (w = 1)."""

    def __init__(self, s: int = 40):
        from skimage.measure import regionprops
        self.ks = np.linspace(1.0, 8.0, 141)
        rows = []
        for k in self.ks:
            Lt = k; r = 0.5; H = int(2 * s); W = int((Lt + 1) * s)
            yy, xx = np.mgrid[:H, :W]
            x = (xx + 0.5) / s - (Lt + 1) / 2; y = (yy + 0.5) / s - 1.0
            dx = np.clip(np.abs(x) - (Lt - 2 * r) / 2, 0, None)
            m = (dx ** 2 + y ** 2) <= r * r
            p = regionprops(m.astype(int))[0]
            rows.append((p.major_axis_length / s, p.minor_axis_length / s, m.sum() / s / s))
        self.major, self.minor, self.area = np.array(rows).T
        self.shape = self.major / np.sqrt(self.area)          # dimensionless, monotonic in k

    def invert(self, major_um, area_um2) -> Tuple[np.ndarray, np.ndarray]:
        """(total length, width) of the capsule with this ellipse major axis and this area."""
        major_um = np.asarray(major_um, float); area_um2 = np.asarray(area_um2, float)
        q = major_um / np.sqrt(area_um2)
        k = np.interp(q, self.shape, self.ks)
        w = np.sqrt(area_um2 / np.interp(k, self.ks, self.area))
        return k * w, w

    def total_length_from_major(self, major_um: float, width_um: float) -> float:
        """Total capsule length whose ellipse major axis is major_um, at the given width."""
        k = np.interp(major_um / width_um, self.major, self.ks)
        return float(k * width_um)


_TABLE = None


def capsule_table() -> CapsuleTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = CapsuleTable()
    return _TABLE


def frame_seed(mask: np.ndarray, roi_mask: np.ndarray, pixel_um: float):
    """One row per cell of the labelled frame inside the ROI box: position in um from the box corner (y down), the ellipse
    axes, the area, the angle from the x axis, and the capsule sizes (length_total_um, width_um, length_seg_um)."""
    import pandas as pd
    from skimage.measure import regionprops
    box = roi_box(roi_mask)
    rows = []
    for p in regionprops(np.asarray(mask).astype(np.int32)):
        cy, cx = p.centroid
        if not (box["row0"] <= cy < box["row1"] and box["col0"] <= cx < box["col1"]):
            continue
        rows.append(dict(cell_id=int(p.label), x_px=float(cx), y_px=float(cy),
                         x_um=(cx - box["col0"]) * pixel_um, y_um=(cy - box["row0"]) * pixel_um,
                         major_um=p.major_axis_length * pixel_um, minor_um=p.minor_axis_length * pixel_um,
                         area_um2=p.area * pixel_um ** 2, angle_x_rad=float(np.pi / 2 - p.orientation)))
    df = pd.DataFrame(rows)
    if len(df):
        Lt, w = capsule_table().invert(df.major_um.values, df.area_um2.values)
        df["length_total_um"], df["width_um"] = Lt, w
        df["length_seg_um"] = np.maximum(Lt - w, 0.05)
    return df


def chamber(roi_mask: np.ndarray, pixel_um: float, frame_interval_s: float) -> Dict:
    box = roi_box(roi_mask)
    return dict(pixel_um=float(pixel_um), frame_interval_s=float(frame_interval_s), roi_box_px=box,
                width_um=float((box["col1"] - box["col0"]) * pixel_um), height_um=float((box["row1"] - box["row0"]) * pixel_um),
                roi_area_um2=float((np.asarray(roi_mask) > 0).sum() * pixel_um ** 2))
