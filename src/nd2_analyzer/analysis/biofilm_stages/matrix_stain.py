"""Matrix stain (Partaker 2, Figure 4), on the dye channel of a biofilm recording.

zero and flat   the dye image is zeroed with a frame taken before the dye was added (camera offset and medium) and
                flattened with a frame taken right after (free dye fills the chamber evenly, so its image is the shape of
                the light); both are the median over the positions of that file, smoothed (sigma 40 px)
free dye level  the median of the corrected image in open space (more than 1.5 um from any cell), on the frames where
                open space is more than a quarter of the frame; one level per position, the median over those frames
stain           the corrected image smoothed (sigma 2 px) minus the free dye level
matrix          stain above the cut-off (5,000 counts above free dye) that is at least 3 um wide (it survives an opening
                with a 1.5 um disc); narrower pieces above the cut-off are single stained cells
Plain Python (numpy, scipy, OpenCV), no Qt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class MatrixSettings:
    cut_off: float = 5000.0            # counts above the free dye level
    min_width_um: float = 3.0
    smooth_px: float = 2.0
    far_um: float = 1.5                # open space is further than this from any cell
    min_open_share: float = 0.25       # of the frame, for a frame to give a free dye level
    reference_sigma_px: float = 40.0


@dataclass
class DyeReference:
    zero: np.ndarray                   # the image with no dye
    flat: np.ndarray                   # the shape of the light, mean 1


def dye_reference(before: np.ndarray, after: np.ndarray, sigma_px: float = MatrixSettings.reference_sigma_px) -> DyeReference:
    """before, after: the dye channel of every position before the dye and right after it, (positions, y, x)."""
    zero = ndi.gaussian_filter(np.median(np.asarray(before, np.float32), axis=0), sigma_px)
    flat = ndi.gaussian_filter(np.median(np.asarray(after, np.float32), axis=0), sigma_px) - zero
    return DyeReference(zero, flat / flat.mean())


def read_dye_frames(path: str, channel: int) -> np.ndarray:
    """The dye channel of every position of an ND2 file of dye frames, (positions, y, x)."""
    import nd2
    with nd2.ND2File(str(path)) as f:
        data = f.asarray(); axes = "".join(f.sizes)
    data = np.asarray(data)
    for axis in ("T", "Z"):                                   # one time point, one plane
        if axis in axes:
            data = np.take(data, 0, axis=axes.index(axis)); axes = axes.replace(axis, "")
    if "C" in axes:
        data = np.take(data, int(channel), axis=axes.index("C")); axes = axes.replace("C", "")
    return data[None] if "P" not in axes else data


def correct(frame: np.ndarray, ref: DyeReference) -> np.ndarray:
    return (np.asarray(frame, np.float32) - ref.zero) / ref.flat


def open_space(labels: np.ndarray, pixel_um: float, far_um: float = MatrixSettings.far_um) -> np.ndarray:
    return ndi.distance_transform_edt(np.asarray(labels) == 0) * pixel_um > far_um


def free_dye_level(frames: Iterable[Tuple[np.ndarray, np.ndarray]], pixel_um: float, settings: MatrixSettings = MatrixSettings()) -> Tuple[float, int]:
    """frames: (corrected dye image, cell labels) of early frames of one position. Returns (level, frames used)."""
    levels = []
    for corrected, labels in frames:
        space = open_space(labels, pixel_um, settings.far_um)
        if space.mean() > settings.min_open_share:
            levels.append(float(np.median(np.asarray(corrected)[space])))
    return (float(np.median(levels)), len(levels)) if levels else (float("nan"), 0)


def stain(corrected: np.ndarray, level: float, settings: MatrixSettings = MatrixSettings()) -> np.ndarray:
    return cv2.GaussianBlur(np.asarray(corrected, np.float32), (0, 0), settings.smooth_px) - level


def matrix(stain_image: np.ndarray, pixel_um: float, settings: MatrixSettings = MatrixSettings()) -> Tuple[np.ndarray, np.ndarray]:
    """(matrix, single cells): what is above the cut-off and wide enough, and the narrower pieces above the cut-off."""
    r = int(round(settings.min_width_um / 2 / pixel_um)); disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    above = np.asarray(stain_image) > settings.cut_off
    counted = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, disc) > 0
    return counted, above & ~counted


def share(mask: np.ndarray) -> float:
    """Percent of the frame."""
    return 100.0 * float(np.asarray(mask).mean())


def microcolony_area(labels: np.ndarray, pixel_um: float, rule=None) -> np.ndarray:
    """The area of the microcolonies of a frame: the linked groups of 4 or more cells, each cell grown by half the gap."""
    from .microcolony_rules import MicrocolonySettings, drop_small_cells, linked_groups
    rule = rule or MicrocolonySettings()
    groups = linked_groups(drop_small_cells(labels, pixel_um, rule.min_cell_um2), pixel_um, rule.gap_um)
    return (groups.sizes[groups.group_map] >= rule.min_cells) & (groups.group_map > 0)


def inside_outside(counted: np.ndarray, labels: np.ndarray, pixel_um: float, rule=None) -> Tuple[float, float, float, float]:
    """(% of the microcolony area that is matrix, % of the rest of the frame that is matrix, % of the matrix that lies in
    microcolonies, % of the frame that is microcolony)."""
    colony = microcolony_area(labels, pixel_um, rule); counted = np.asarray(counted, bool)
    inside = 100.0 * float(counted[colony].mean()) if colony.any() else float("nan")
    outside = 100.0 * float(counted[~colony].mean()) if (~colony).any() else float("nan")
    return inside, outside, 100.0 * float((counted & colony).sum()) / max(int(counted.sum()), 1), 100.0 * float(colony.mean())
