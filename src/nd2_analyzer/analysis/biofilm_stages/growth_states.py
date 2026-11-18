"""Growth states (Partaker 2, Figure 5). State I ends at Th1, when the matrix stain appears and stays; state II ends at
Th2, when growth stops. Growth is the number of cells in the whole frame (not their area), its rate the slope of the count
over three hours each way. Where the recording ends before growth stops, Th2 is predicted: a Gompertz curve is fitted to
the mean count of the positions, and Th2 is the first time after the fastest growth at which the growth rate of the curve
is down to a tenth of its peak; its range comes from resampling the positions. Plain Python, no Qt."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

MIN_CELL_UM2 = 1.0
RATE_HALF_WINDOW_H = 3.0
STOP_FRACTION = 0.10           # growth has stopped when its rate is down to this share of the peak
Curve = Tuple[np.ndarray, np.ndarray]       # (hours, cells)


def count_cells(labels: np.ndarray, pixel_um: float, min_cell_um2: float = MIN_CELL_UM2) -> int:
    """Cells of at least min_cell_um2 in the frame."""
    areas_um2 = np.bincount(np.asarray(labels).ravel())[1:] * pixel_um ** 2
    return int((areas_um2 >= min_cell_um2).sum())


def growth_rate(hours: Sequence[float], counts: Sequence[float], half_window_h: float = RATE_HALF_WINDOW_H) -> np.ndarray:
    """Cells per hour at each time: the slope of the count over half_window_h each way."""
    h, n = np.asarray(hours, float), np.asarray(counts, float); w = half_window_h + 0.01
    return np.array([np.polyfit(h[np.abs(h - x) <= w], n[np.abs(h - x) <= w], 1)[0] for x in h])


def mean_curve(per_position: Dict[int, Curve]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hours, mean count, s.d.) over the positions, at the hours every position has."""
    shared = sorted(set.intersection(*(set(np.round(h, 6).tolist()) for h, _n in per_position.values())))
    rows = np.array([[float(n[np.argmin(np.abs(h - x))]) for x in shared] for h, n in per_position.values()])
    return np.array(shared), rows.mean(0), (rows.std(0, ddof=1) if len(rows) > 1 else np.zeros(len(shared)))


def gompertz(t, K, r, t0):
    return K * np.exp(-np.exp(-r * (np.asarray(t, float) - t0)))


def fit_gompertz(hours: Sequence[float], counts: Sequence[float], start=(2600.0, 0.08, 35.0)) -> Tuple[float, float, float]:
    params, _ = curve_fit(gompertz, np.asarray(hours, float), np.asarray(counts, float), p0=start, maxfev=20000)
    return tuple(float(v) for v in params)


def th2_from_fit(params, fraction: float = STOP_FRACTION, first_h: float = 24.0, last_h: float = 200.0, step_h: float = 0.1) -> Tuple[float, float, float]:
    """(Th2, hour of the fastest growth, peak rate in cells per hour) of a fitted curve."""
    t = np.arange(first_h, last_h + step_h / 10, step_h); rate = np.gradient(gompertz(t, *params), t)
    i = int(rate.argmax()); j = i + int(np.argmax(rate[i:] <= fraction * rate[i]))
    return float(t[j]), float(t[i]), float(rate[i])


def predict_th2(per_position: Dict[int, Curve], resamples: int = 2000, seed: int = 0) -> dict:
    """Th2 from the mean count of the positions, with the 95 % range from resampling the positions."""
    hours, mean, _sd = mean_curve(per_position); params = fit_gompertz(hours, mean); th2, fastest, peak = th2_from_fit(params)
    positions = sorted(per_position); rows = {p: np.array([float(per_position[p][1][np.argmin(np.abs(per_position[p][0] - x))]) for x in hours]) for p in positions}
    rng = np.random.default_rng(seed); found = []
    for _ in range(int(resamples)):
        pick = rng.choice(positions, len(positions))
        try:
            found.append(th2_from_fit(fit_gompertz(hours, np.mean([rows[p] for p in pick], axis=0), start=params))[0])
        except Exception:
            continue                                         # a resample the curve cannot be fitted to
    low, high = (float(v) for v in np.percentile(found, [2.5, 97.5])) if found else (float("nan"), float("nan"))
    return {"th2_h": th2, "range95_h": (low, high), "fastest_h": fastest, "peak_rate": peak, "K": params[0], "r": params[1], "t0": params[2],
            "resamples": len(found), "last_counted_h": float(hours[-1]), "reached": bool(th2 <= hours[-1])}
