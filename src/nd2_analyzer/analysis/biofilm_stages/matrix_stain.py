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

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

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


FIRST_STAIN_UM2 = 3.0          # a microcolony has matrix stain from the first frame with at least this much inside it


def per_microcolony(tracks, counted_at, pixel_um: float) -> list:
    """Matrix stain inside each followed microcolony, frame by frame. tracks: the followed microcolonies (with their
    gaps closed, microcolony_follow.close_gaps, when the pictures are); counted_at(frame): the matrix mask of the frame.
    Rows: time, hours, identity, area_um2, stain_um2, cover (stain / area)."""
    rows = []; a = pixel_um ** 2
    for t in tracks.frames:
        ids_map = tracks.identity_map(t); ids = np.unique(ids_map); ids = ids[ids > 0]
        if not ids.size:
            continue
        oy, ox = tracks.offsets[t]; counted = np.asarray(counted_at(t), bool); on_canvas = np.zeros(ids_map.shape, bool)
        on_canvas[oy:oy + counted.shape[0], ox:ox + counted.shape[1]] = counted
        area = ndi.sum(np.ones(ids_map.shape, np.float32), ids_map, ids) * a; stain_ = ndi.sum(on_canvas, ids_map, ids) * a
        for i, ar, st in zip(ids, area, stain_):
            rows.append({"time": int(t), "hours": tracks.hours[t], "identity": int(i), "area_um2": float(ar), "stain_um2": float(st), "cover": float(st / ar) if ar else 0.0})
    return rows


def microcolony_stain_summary(rows: list, first_stain_um2: float = FIRST_STAIN_UM2) -> dict:
    """identity -> first_stain_h (None when it never has matrix stain), most_cover_pct and the hour of it, followed from and to."""
    out = {}
    for r in rows:
        s = out.setdefault(r["identity"], {"first_stain_h": None, "most_cover_pct": 0.0, "most_cover_h": None, "from_h": r["hours"], "to_h": r["hours"]})
        s["from_h"], s["to_h"] = min(s["from_h"], r["hours"]), max(s["to_h"], r["hours"])
        if r["stain_um2"] >= first_stain_um2 and (s["first_stain_h"] is None or r["hours"] < s["first_stain_h"]):
            s["first_stain_h"] = r["hours"]
        if 100.0 * r["cover"] > s["most_cover_pct"] or s["most_cover_h"] is None:
            s["most_cover_pct"], s["most_cover_h"] = 100.0 * r["cover"], r["hours"]
    return out


@dataclass
class MicrocolonyStain:
    """The matrix stain of one tracked position: the matrix mask of every followed frame, kept packed (an eighth of a
    byte per pixel), and what it gives for each followed microcolony."""
    shape: tuple
    packed: Dict[int, bytes] = field(default_factory=dict, repr=False)     # frame -> the matrix mask of the frame
    rows: list = field(default_factory=list, repr=False)                   # per_microcolony: time, hours, identity, area, stain, cover
    summary: Dict[int, dict] = field(default_factory=dict)                 # microcolony_stain_summary

    def counted_at(self, t: int) -> np.ndarray:
        """The matrix mask of a followed frame, at once."""
        n = int(self.shape[0]) * int(self.shape[1])
        return np.unpackbits(np.frombuffer(self.packed[int(t)], np.uint8))[:n].reshape(self.shape).astype(bool)

    def series(self, identity: int) -> Tuple[np.ndarray, np.ndarray]:
        """(hours, % of the microcolony's area that is matrix stain) over the frames it is followed in."""
        mine = sorted((r for r in self.rows if r["identity"] == int(identity)), key=lambda r: r["time"])
        return np.array([r["hours"] for r in mine], float), np.array([100.0 * r["cover"] for r in mine], float)


def measure_microcolony_stain(tracks, counted_at: Callable[[int], np.ndarray], pixel_um: float,
                              progress: Optional[Callable[[int, int], None]] = None) -> MicrocolonyStain:
    """The matrix mask of every followed frame, worked out once and kept, and the stain inside each followed
    microcolony. counted_at(frame): the matrix mask of the frame (the slow part: about a third of a second a frame)."""
    out = MicrocolonyStain(shape=(0, 0)); n = len(tracks.frames)
    for k, t in enumerate(tracks.frames, 1):
        mask = np.asarray(counted_at(t), bool); out.shape = tuple(mask.shape); out.packed[int(t)] = np.packbits(mask).tobytes()
        if progress:
            progress(k, n)
    out.rows = per_microcolony(tracks, out.counted_at, pixel_um); out.summary = microcolony_stain_summary(out.rows)
    return out
