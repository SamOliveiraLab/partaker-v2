"""The microcolony metrics of Figure 3 (panels f to k, n, q), from the cell labels of a frame alone. Plain Python, no Qt.

They use the microcolony rule without its lasting test (close and enough: a linked group of 4 or more cells), so every
frame is measured by itself and any position can be measured at any step in time. The followed microcolonies of one
position (with their merges) are the follower's work, not this module's.

  f  cells per microcolony            cells in microcolonies / number of microcolonies
  g  cells in microcolonies           % of the cells; random_placement gives the same cells put down at random
  h  local cell density               other cells within 3 um of a cell, per 100 um2, for cells in microcolonies and single cells
  i  microcolony spacing              distance from a microcolony's centre to the nearest other one (median)
  j  cell alignment                   mean cos(2 x angle) between a cell and its neighbours within 3 um (1 parallel, 0 no order)
  k  largest microcolony              % of the cells in microcolonies that it holds
  n  density inside microcolonies     local density against the distance from the centre of the cell's own microcolony
  q  microcolony area                 the microcolonies (cells grown by half the gap), % of the frame
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .microcolony_rules import MicrocolonySettings, drop_small_cells, linked_groups

NEAR_UM = 3.0                 # "neighbours" are the cells whose centres are within this distance
MIN_ASPECT = 1.3              # a cell needs this length over width to have a direction
RING_UM, RING_MAX_UM = 2.0, 40.0
_U = np.random.default_rng(0).uniform(0, 1, (400, 2))
DISC = np.c_[np.sqrt(_U[:, 0]) * np.cos(2 * np.pi * _U[:, 1]), np.sqrt(_U[:, 0]) * np.sin(2 * np.pi * _U[:, 1])]   # 400 fixed points of the unit disc
SERIES = ("cells", "microcolonies", "cells_in_pct", "cells_per_microcolony", "largest_share_pct", "spacing_um", "density_in",
          "density_single", "alignment", "area_pct", "random_pct")


@dataclass
class FrameMetrics:
    cells: int
    microcolonies: int
    cells_in_microcolonies: int
    cells_in_pct: float                # g
    cells_per_microcolony: float       # f
    largest_share_pct: float           # k
    spacing_um: float                  # i
    density_in: float                  # h, cells in microcolonies
    density_single: float              # h, single cells
    alignment: float                   # j, cells in microcolonies
    area_pct: float                    # q
    ring_sum: np.ndarray               # n: the local densities of the cells of each ring, added up
    ring_n: np.ndarray                 # n: the cells of each ring


def cell_shapes(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(labels, centres as (row, column), direction in radians, length over width) of every cell, from the second
    moments of its pixels: the numbers skimage.measure.regionprops gives, in one pass over the frame."""
    labels = np.asarray(labels); width = labels.shape[1]
    at = np.flatnonzero(labels); cell = labels.ravel()[at]; r, c = np.divmod(at, width); r = r.astype(float); c = c.astype(float)
    n = np.bincount(cell).astype(float); found = np.flatnonzero(n)
    total = lambda w: np.bincount(cell, weights=w, minlength=n.size)[found]
    n = n[found]; rm, cm = total(r) / n, total(c) / n
    rr, cc, rc = total(r * r) / n - rm ** 2, total(c * c) / n - cm ** 2, total(r * c) / n - rm * cm
    a, b, d = cc, -rc, rr                                                      # the inertia tensor [[a, b], [b, d]] of scikit-image
    direction = np.where(a == d, np.where(b < 0, -np.pi / 4, np.pi / 4), 0.5 * np.arctan2(-2 * b, d - a))
    half, root = (a + d) / 2, np.sqrt(np.maximum(((a - d) / 2) ** 2 + b ** 2, 0))
    major, minor = 4 * np.sqrt(np.maximum(half + root, 0)), 4 * np.sqrt(np.maximum(half - root, 0))
    return found, np.c_[rm, cm], direction, major / np.maximum(minor, 1e-6)


def frame_metrics(labels: np.ndarray, pixel_um: float, settings: MicrocolonySettings = MicrocolonySettings(), near_um: float = NEAR_UM,
                  ring_um: float = RING_UM, ring_max_um: float = RING_MAX_UM) -> FrameMetrics:
    """The metrics of one frame."""
    edges = np.arange(0, ring_max_um + ring_um, ring_um); rings = len(edges) - 1
    ring_sum, ring_n = np.zeros(rings), np.zeros(rings, int); nan = float("nan")
    lab = drop_small_cells(labels, pixel_um, settings.min_cell_um2); height, width = lab.shape
    cell, centre, direction, aspect = cell_shapes(lab); n = int(cell.size)
    if n == 0:
        return FrameMetrics(0, 0, 0, 0.0, nan, nan, nan, nan, nan, nan, 0.0, ring_sum, ring_n)
    groups = linked_groups(lab, pixel_um, settings.gap_um)
    big = np.flatnonzero(groups.sizes >= settings.min_cells); big = big[big > 0]
    group = np.array([groups.cell_group[int(c)] for c in cell]); inside = groups.sizes[group] >= settings.min_cells; n_in = int(inside.sum())
    # h: the other cells within near_um, over the part of that disc that lies in the frame
    reach = near_um / pixel_um; near = cKDTree(centre).query_ball_point(centre, reach); disc = DISC * reach
    rows, cols = centre[:, None, 0] + disc[None, :, 0], centre[:, None, 1] + disc[None, :, 1]
    in_frame = ((rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)).mean(1)
    density = 100 * np.array([len(v) - 1 for v in near]) / (np.pi * near_um ** 2 * np.clip(in_frame, 0.2, 1))
    # j: how parallel a cell is to its neighbours
    order = np.full(n, np.nan)
    for i, v in enumerate(near):
        if aspect[i] >= MIN_ASPECT:
            v = [j for j in v if j != i and aspect[j] >= MIN_ASPECT]
            if v:
                order[i] = float(np.mean(np.cos(2 * (direction[i] - direction[v]))))
    # q, i: the area of the microcolonies and the distance between their centres
    is_big = np.zeros(groups.sizes.size, bool); is_big[big] = True
    area_pct = 100 * float(is_big[groups.group_map].mean())
    spacing = nan
    if big.size > 1:
        mid = np.array(ndi.center_of_mass(np.ones(lab.shape, np.uint8), groups.group_map, big))
        apart = np.hypot(mid[:, None, 0] - mid[None, :, 0], mid[:, None, 1] - mid[None, :, 1]); np.fill_diagonal(apart, np.inf)
        spacing = float(np.median(apart.min(1)) * pixel_um)
    # n: local density against the distance from the centre of the cell's own microcolony
    for k in np.unique(group[inside]):
        mine = group == k; mid = centre[mine].mean(0)
        ring = np.digitize(np.hypot(centre[mine, 0] - mid[0], centre[mine, 1] - mid[1]) * pixel_um, edges) - 1; ok = ring < rings
        np.add.at(ring_sum, ring[ok], density[mine][ok]); np.add.at(ring_n, ring[ok], 1)
    aligned = order[inside]
    return FrameMetrics(
        cells=n, microcolonies=int(big.size), cells_in_microcolonies=n_in, cells_in_pct=100 * n_in / n,
        cells_per_microcolony=n_in / big.size if big.size else nan,
        largest_share_pct=100 * float(groups.sizes[big].max()) / n_in if big.size else nan, spacing_um=spacing,
        density_in=float(density[inside].mean()) if n_in else nan, density_single=float(density[~inside].mean()) if n_in < n else nan,
        alignment=float(np.nanmean(aligned)) if np.isfinite(aligned).any() else nan, area_pct=area_pct, ring_sum=ring_sum, ring_n=ring_n)


# ---------------------------------------------------------------- random placement (the rings of panel g)
def random_placement(labels: np.ndarray, rng: np.random.Generator, tries: int = 200) -> np.ndarray:
    """The same cells, each turned by a random angle and put at a random place of the frame, no two overlapping."""
    labels = np.asarray(labels); out = np.zeros_like(labels); height, width = labels.shape
    boxes = ndi.find_objects(labels); cells = [i for i, box in enumerate(boxes, 1) if box is not None]
    cells.sort(key=lambda i: -(boxes[i - 1][0].stop - boxes[i - 1][0].start) * (boxes[i - 1][1].stop - boxes[i - 1][1].start))   # the largest first
    for i in cells:
        cell = np.pad((labels[boxes[i - 1]] == i).astype(np.uint8), 1)         # a margin, or turning shaves the pixels on the box's edge
        shape = ndi.rotate(cell, rng.uniform(0, 360), order=0, reshape=True) > 0
        ys, xs = np.nonzero(shape)
        if ys.size == 0:
            continue
        shape = shape[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; h, w = shape.shape
        for _ in range(tries):
            y, x = rng.integers(0, height - h + 1), rng.integers(0, width - w + 1)
            if not out[y:y + h, x:x + w][shape].any():
                out[y:y + h, x:x + w][shape] = i; break
    return out


def random_share(labels: np.ndarray, pixel_um: float, placements: int = 5, seed=0, settings: MicrocolonySettings = MicrocolonySettings()) -> float:
    """% of the cells that are in a linked group of min_cells or more when the same cells are placed at random (mean of
    the placements)."""
    lab = drop_small_cells(labels, pixel_um, settings.min_cell_um2); rng = np.random.default_rng(seed); shares = []
    for _ in range(placements):
        groups = linked_groups(random_placement(lab, rng), pixel_um, settings.gap_um)
        sizes = np.array([groups.sizes[g] for g in groups.cell_group.values()])
        shares.append(100 * float((sizes >= settings.min_cells).mean()) if sizes.size else 0.0)
    return float(np.mean(shares)) if shares else float("nan")


# ---------------------------------------------------------------- every position and frame
@dataclass
class MicrocolonyMetrics:
    positions: List[int]
    frames: Dict[int, List[int]]
    hours: Dict[int, np.ndarray]
    values: Dict[str, Dict[int, np.ndarray]]                 # SERIES name -> position -> one value per frame
    ring_edges: np.ndarray = field(default_factory=lambda: np.arange(0, RING_MAX_UM + RING_UM, RING_UM))
    ring_sum: Dict[int, np.ndarray] = field(default_factory=dict)    # position -> frames x rings
    ring_n: Dict[int, np.ndarray] = field(default_factory=dict)
    settings: MicrocolonySettings = MicrocolonySettings()

    def _shared(self, per_position: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """(the hours every position has, positions x hours)."""
        shared = sorted(set.intersection(*(set(np.round(self.hours[p], 6).tolist()) for p in per_position)))
        rows = np.array([[np.asarray(v)[int(np.argmin(np.abs(self.hours[p] - x)))] for x in shared] for p, v in per_position.items()], float)
        return np.array(shared), rows

    def mean_of(self, what: str, geometric: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(hours, mean, s.d.) over the positions that have a value. geometric: the mean of the logarithms, as the
        figure draws cells per microcolony; the s.d. is then the s.d. of log10."""
        hours, rows = self._shared(self.values[what])
        if geometric:
            rows = np.log10(np.where(rows > 0, rows, np.nan))
        have = np.isfinite(rows).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(rows, 0); sd = np.where(have > 1, np.nanstd(rows, 0, ddof=1), 0.0)
        return hours, (10 ** mean if geometric else mean), sd

    def have(self, what: str) -> Tuple[np.ndarray, np.ndarray]:
        """(hours, the number of positions that have a value)."""
        hours, rows = self._shared(self.values[what])
        return hours, np.isfinite(rows).sum(0)

    def radial(self, min_cells: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(hours, ring edges in um, hours x rings): the mean local density of the cells of each ring, all positions
        together; not a number where a ring holds fewer than min_cells cells."""
        hours, total = self._shared({p: np.arange(len(self.hours[p])) for p in self.positions})
        total_sum = np.zeros((len(hours), len(self.ring_edges) - 1)); total_n = np.zeros_like(total_sum)
        for row, p in zip(total.astype(int), self.positions):
            total_sum += self.ring_sum[p][row]; total_n += self.ring_n[p][row]
        return hours, self.ring_edges, np.where(total_n >= min_cells, total_sum / np.maximum(total_n, 1), np.nan)


def measure_metrics(labels_at: Callable[[int, int], np.ndarray], frames: Dict[int, Sequence[int]], pixel_um: float, hour_of: Callable[[int], float],
                    settings: MicrocolonySettings = MicrocolonySettings(), progress: Optional[Callable[[int, int], None]] = None,
                    random_hours: Sequence[float] = (), placements: int = 5, seed: int = 0) -> MicrocolonyMetrics:
    """labels_at(position, frame): the cell labels; frames: the frames to measure of every position; random_hours: the
    hours at which the cells are also placed at random (slow: about a second a placement on a full frame)."""
    positions = sorted(int(p) for p in frames); total = sum(len(frames[p]) for p in positions); done = 0
    out = MicrocolonyMetrics(positions=positions, frames={p: [int(t) for t in frames[p]] for p in positions}, hours={},
                             values={name: {} for name in SERIES}, settings=settings)
    for p in positions:
        ts = out.frames[p]; out.hours[p] = np.array([float(hour_of(t)) for t in ts]); rows = {name: [] for name in SERIES}; sums, counts = [], []
        for t, hour in zip(ts, out.hours[p]):
            lab = np.asarray(labels_at(p, t)); m = frame_metrics(lab, pixel_um, settings)
            for name in SERIES[:-1]:
                rows[name].append(getattr(m, name))
            at_random = any(abs(hour - h) < 1e-6 for h in random_hours)
            rows["random_pct"].append(random_share(lab, pixel_um, placements, [seed, p, t], settings) if at_random else np.nan)
            sums.append(m.ring_sum); counts.append(m.ring_n); done += 1
            if progress:
                progress(done, total)
        for name in SERIES:
            out.values[name][p] = np.array(rows[name], float)
        out.ring_sum[p] = np.array(sums).reshape(len(ts), -1); out.ring_n[p] = np.array(counts).reshape(len(ts), -1)
    return out


# ---------------------------------------------------------------- with the saved session
def save_metrics(metrics: MicrocolonyMetrics, path) -> str:
    """The metrics as a file of numbers and text (stage_files)."""
    from dataclasses import asdict
    from . import stage_files as SF
    header = {"positions": metrics.positions, "frames": metrics.frames, "settings": asdict(metrics.settings), "series": list(metrics.values)}
    arrays = {"ring_edges": np.asarray(metrics.ring_edges, float)}
    for p in metrics.positions:
        arrays[f"hours_{p}"] = metrics.hours[p]; arrays[f"ring_sum_{p}"] = metrics.ring_sum[p]; arrays[f"ring_n_{p}"] = metrics.ring_n[p]
        for name, per in metrics.values.items():
            arrays[f"{name}_{p}"] = per[p]
    return SF.write(path, "microcolony metrics", header, arrays)


def load_metrics(path) -> MicrocolonyMetrics:
    from . import stage_files as SF
    h, a = SF.read(path, "microcolony metrics"); positions = [int(p) for p in h["positions"]]
    return MicrocolonyMetrics(positions=positions, frames={int(p): [int(t) for t in f] for p, f in h["frames"].items()},
                              hours={p: a[f"hours_{p}"] for p in positions},
                              values={name: {p: a[f"{name}_{p}"] for p in positions} for name in h["series"]},
                              ring_edges=a["ring_edges"], ring_sum={p: a[f"ring_sum_{p}"] for p in positions},
                              ring_n={p: a[f"ring_n_{p}"] for p in positions}, settings=MicrocolonySettings(**h["settings"]))
