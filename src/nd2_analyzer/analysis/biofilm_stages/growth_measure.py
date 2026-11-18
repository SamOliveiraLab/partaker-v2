"""The growth-states measure of a recording: for every position and frame the cells are counted (growth_states) and, when
the dye frames are given, the matrix stain is measured (matrix_stain); then Th1 from the matrix stain and Th2 predicted
from the count. Plain Python: the Maturation page runs it in a worker thread, the tests run it directly."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import growth_states as GS
from . import matrix_stain as MX

EARLY_H = 12.0                 # the free dye level is measured on the frames of the first 12 hours


@dataclass
class GrowthStates:
    positions: List[int]
    frames: Dict[int, List[int]]
    hours: Dict[int, np.ndarray]
    cells: Dict[int, np.ndarray]                   # cells in the whole frame
    rate: Dict[int, np.ndarray]                    # cells per hour
    matrix_hours: Dict[int, np.ndarray] = field(default_factory=dict)  # the hours the matrix stain was measured at
    matrix: Dict[int, np.ndarray] = field(default_factory=dict)        # matrix stain, % of the frame
    single: Dict[int, np.ndarray] = field(default_factory=dict)        # single stained cells, % of the frame
    inside: Dict[int, np.ndarray] = field(default_factory=dict)        # matrix stain inside microcolonies, % of their area
    outside: Dict[int, np.ndarray] = field(default_factory=dict)       # matrix stain outside them, % of the rest
    colony_area: Dict[int, np.ndarray] = field(default_factory=dict)   # microcolonies, % of the frame
    free_dye: Dict[int, float] = field(default_factory=dict)           # free dye level of each position, counts
    th1: Optional[dict] = None
    th2: Optional[dict] = None
    settings: MX.MatrixSettings = MX.MatrixSettings()

    def mean_of(self, what: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(hours, mean, s.d.) over the positions of 'cells', 'rate', 'matrix', 'single', 'inside', 'outside' or 'colony_area'."""
        values = getattr(self, what); hours = self.hours if what in ("cells", "rate") else self.matrix_hours
        return GS.mean_curve({p: (hours[p], values[p]) for p in self.positions if p in values})

    def mean_cells(self):
        return self.mean_of("cells")


def measure_growth_states(labels_at: Callable[[int, int], np.ndarray], dye_at: Optional[Callable[[int, int], np.ndarray]],
                          frames: Dict[int, List[int]], pixel_um: float, hour_of: Callable[[int], float],
                          reference: Optional[MX.DyeReference], settings: MX.MatrixSettings = MX.MatrixSettings(),
                          progress: Optional[Callable[[int, int], None]] = None, resamples: int = 2000,
                          matrix_frames: Optional[Dict[int, List[int]]] = None) -> GrowthStates:
    """labels_at(position, frame): the cell labels; dye_at(position, frame): the raw dye channel (None: cells only);
    frames: the frames to measure of every position; reference: the zero and flat of the dye frames; matrix_frames: the
    frames the matrix stain is measured on (default: all of them; after a loss of focus the dye image is on another scale)."""
    positions = sorted(int(p) for p in frames); total = sum(len(v) for v in frames.values()); done = 0
    out = GrowthStates(positions=positions, frames={p: [int(t) for t in frames[p]] for p in positions}, hours={}, cells={}, rate={}, settings=settings)
    with_matrix = dye_at is not None and reference is not None
    for p in positions:
        ts = out.frames[p]; hours = np.array([float(hour_of(t)) for t in ts]); out.hours[p] = hours
        cells = []
        for t in ts:
            cells.append(GS.count_cells(np.asarray(labels_at(p, t)), pixel_um)); done += 1
            if progress:
                progress(done, total * (2 if with_matrix else 1))
        out.cells[p] = np.array(cells); out.rate[p] = GS.growth_rate(hours, out.cells[p]) if len(ts) > 1 else np.zeros(len(ts))
        if with_matrix:
            image = lambda t: MX.correct(dye_at(p, t), reference)             # one frame in memory at a time
            early = [t for t, h in zip(ts, hours) if h <= hours[0] + EARLY_H and (matrix_frames is None or t in set(matrix_frames.get(p, [])))]
            level, _used = MX.free_dye_level(((image(t), np.asarray(labels_at(p, t))) for t in early), pixel_um, settings); out.free_dye[p] = level
            rows = []; mts = [t for t in ts if matrix_frames is None or t in set(matrix_frames.get(p, []))]
            out.matrix_hours[p] = np.array([float(hour_of(t)) for t in mts])
            for t in mts:
                counted, single = MX.matrix(MX.stain(image(t), level, settings), pixel_um, settings)
                rows.append((MX.share(counted), MX.share(single), *MX.inside_outside(counted, np.asarray(labels_at(p, t)), pixel_um)))
                done += 1
                if progress:
                    progress(done, total * 2)
            a = np.array(rows, float).reshape(len(mts), 6)
            out.matrix[p], out.single[p], out.inside[p], out.outside[p], out.colony_area[p] = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 5]
    if with_matrix:
        out.th1 = GS.th1_of_positions({p: (out.matrix_hours[p], out.matrix[p]) for p in positions})
    try:
        out.th2 = GS.predict_th2({p: (out.hours[p], out.cells[p]) for p in positions}, resamples=resamples)
    except Exception:
        out.th2 = None                                         # too few frames, or a count no growth curve fits
    return out


# ---------------------------------------------------------------- with the saved session
PER_POSITION = ("hours", "cells", "rate", "matrix_hours", "matrix", "single", "inside", "outside", "colony_area")


def save_growth_states(gs: GrowthStates, path) -> str:
    """The growth-states measure as a file of numbers and text (stage_files)."""
    from dataclasses import asdict
    from . import stage_files as SF
    header = {"positions": gs.positions, "frames": gs.frames, "free_dye": gs.free_dye, "th1": gs.th1, "th2": gs.th2, "settings": asdict(gs.settings)}
    arrays = {f"{name}_{p}": np.asarray(values) for name in PER_POSITION for p, values in getattr(gs, name).items()}
    return SF.write(path, "growth states", header, arrays)


def load_growth_states(path) -> GrowthStates:
    from . import stage_files as SF
    h, a = SF.read(path, "growth states"); positions = [int(p) for p in h["positions"]]
    per = {name: {p: a[f"{name}_{p}"] for p in positions if f"{name}_{p}" in a} for name in PER_POSITION}
    th1, th2 = h.get("th1"), h.get("th2")
    if th1 is not None:
        th1["per_position_h"] = SF.int_keys(th1["per_position_h"])
    if th2 is not None:
        th2["range95_h"] = tuple(th2["range95_h"])
    return GrowthStates(positions=positions, frames={int(p): [int(t) for t in f] for p, f in h["frames"].items()},
                        free_dye={int(p): float(v) for p, v in h["free_dye"].items()}, th1=th1, th2=th2,
                        settings=MX.MatrixSettings(**h["settings"]), **per)
