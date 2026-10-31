"""Reading a twin run back: the cells of every frame inside the ROI, counts and areas, and the score against the real cells.
Port of DigitalTwin_rerun/compare_twin.py (Oct 2026). The run files are the pickles written by vivamunk_twin.py; they hold
process-bigraph objects we do not need, so unknown classes unpickle as stubs."""
from __future__ import annotations

import glob
import math
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except Exception:
            return _Stub


def load_run(path) -> dict:
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def load_runs(folder, rule: str) -> List[dict]:
    """Every seed's run of one rule in a run folder (seed*/montage_<rule>.pkl)."""
    return [load_run(p) for p in sorted(glob.glob(f"{folder}/seed*/montage_{rule}.pkl"))]


def cells_at(run: dict, k: int, frame_s: float) -> List[Tuple[str, float, float, float, float, float]]:
    """(id, x, y, total length, radius, angle) in um of the cells inside the ROI at frame k (positions relative to the ROI corner)."""
    res = run["results"]
    times = [r.get("time", i * 30.0) for i, r in enumerate(res)]
    i = int(np.argmin([abs(t - k * frame_s) for t in times]))
    roi = run["roi"]
    out = []
    for aid, ag in res[i]["agents"].items():
        x, y = ag["location"]
        if roi["x0"] <= x < roi["x0"] + roi["w"] and roi["y0"] <= y < roi["y0"] + roi["h"]:
            r = float(ag["radius"])
            out.append((aid, x - roi["x0"], y - roi["y0"], float(ag["length"]) + 2 * r, r, float(ag["angle"])))
    return out


def area_of(cells) -> float:
    return float(sum(max(L - 2 * r, 0) * 2 * r + math.pi * r * r for _, _, _, L, r, _ in cells))


def curves(runs: Sequence[dict], n_frames: int, frame_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per run and frame: cell count and total area (um2) inside the ROI."""
    per = [[cells_at(run, k, frame_s) for k in range(n_frames)] for run in runs]
    counts = np.array([[len(c) for c in row] for row in per], float)
    areas = np.array([[area_of(c) for c in row] for row in per], float)
    return counts, areas


def score(counts: np.ndarray, areas: np.ndarray, real_counts, real_areas, from_frame: int = 0) -> Dict[str, float]:
    rc = np.asarray(real_counts, float); ra = np.asarray(real_areas, float)
    m, am = counts.mean(0), areas.mean(0)
    k0 = int(from_frame)
    return dict(seeds=int(len(counts)), end_count_mean=float(m[-1]), end_count_sd=float(counts[:, -1].std(ddof=1)) if len(counts) > 1 else 0.0,
                real_end=float(rc[-1]), count_rel_err=float(np.abs(m - rc)[k0:].mean() / rc[k0:].mean()),
                area_rel_err=float(np.abs(am - ra)[k0:].mean() / ra[k0:].mean()))


def real_curves(masks_at, roi_mask: np.ndarray, n_frames: int, pixel_um: float) -> Tuple[np.ndarray, np.ndarray]:
    """The real cells' count and total area (um2) per frame, inside the ROI, from the tool's masks."""
    from nd2_analyzer.analysis.segmentation.segmentation_service import apply_roi_mask
    counts, areas = [], []
    for t in range(n_frames):
        m = apply_roi_mask(np.asarray(masks_at(t)), roi_mask)
        counts.append(len(np.unique(m)) - 1); areas.append(float((m > 0).sum() * pixel_um ** 2))
    return np.array(counts, float), np.array(areas, float)
