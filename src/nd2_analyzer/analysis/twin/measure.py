"""The rules measured from the tool's own tracks and masks, the way the Oct 2026 rerun measured them (measure_rules.py):
the adder (birth and division length, added length, interdivision time, growth rate) from the full generations, and the
position rule (growth rate falling with distance from the chamber opening)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from nd2_analyzer.analysis.tracking import family as F
from .seed import capsule_table, roi_box

LAG = dict(lag_end_min=75.0, lag_ramp_min=20.0, lag_factor=0.55)       # measured on the twin recording (RULE_EVIDENCE.md)
POS_SLOPE_MEASURED = 0.0129                                             # per um, the same


def _props(masks_at, by_id, i: int, t: int, pixel_um: float) -> Optional[Dict[str, float]]:
    """Ellipse major axis, area (um) and x (px) of track i's cell at frame t."""
    from skimage.measure import regionprops
    p = F.point_at(by_id, i, t)
    if p is None:
        return None
    m = np.asarray(masks_at(t))
    px, py = int(round(p[0])), int(round(p[1]))
    if not (0 <= py < m.shape[0] and 0 <= px < m.shape[1]) or m[py, px] == 0:
        return None
    lab = int(m[py, px])
    sub = (m == lab).astype(np.int32)
    rp = regionprops(sub)[0]
    return dict(major_um=rp.major_axis_length * pixel_um, area_um2=rp.area * pixel_um ** 2, x_px=float(rp.centroid[1]))


def measure_adder(tracks, masks_at, pixel_um: float, frame_min: float, width_um: float) -> Dict:
    """Birth and division sizes from the full generations (born and divided inside the recording), converted from ellipse
    majors to capsule total lengths at the median width; growth rate from the pair."""
    by_id = F.index_tracks(tracks); kids = F.children_of(by_id)
    Lb, Ld, tau = [], [], []
    for i, x in by_id.items():
        if F.parent_of(by_id, i) is None or not kids.get(i):
            continue
        t1, t2 = F.span(by_id, i)
        a = _props(masks_at, by_id, i, t1, pixel_um); b = _props(masks_at, by_id, i, t2, pixel_um)
        if a is None or b is None:
            continue
        Lb.append(a["major_um"]); Ld.append(b["major_um"]); tau.append((t2 - t1 + 1) * frame_min)
    if len(Lb) < 3:
        raise ValueError(f"only {len(Lb)} full generations in the tracks; the adder needs at least 3")
    tab = capsule_table()
    birth = tab.total_length_from_major(float(np.median(Lb)), width_um)
    division = tab.total_length_from_major(float(np.median(Ld)), width_um)
    tau_mean = float(np.mean(tau))
    added = np.array(Ld) - np.array(Lb)
    slope = float(np.polyfit(Lb, added, 1)[0]) if len(Lb) >= 5 else 0.0
    return dict(n_full_generations=len(Lb), birth_total_um=birth, division_total_um=division, added_um=division - birth,
                width_um=float(width_um), interdivision_min_mean=tau_mean, interdivision_min_sd=float(np.std(tau, ddof=1)),
                alpha_per_h=float(np.log(division / birth) / (tau_mean / 60.0)), added_vs_birth_slope=slope,
                rule="adder" if abs(slope) < 0.35 else ("sizer" if slope < 0 else "timer"))


def measure_position_slope(tracks, masks_at, roi_mask, pixel_um: float, frame_min: float, min_frames: int = 5) -> Dict:
    """Growth rate of each track (exponential fit of its area) against its distance from the chamber opening (the right
    edge of the ROI box). Returns the relative slope per um (rate = rate_opening * (1 - slope * distance))."""
    by_id = F.index_tracks(tracks); box = roi_box(roi_mask)
    rates, dists = [], []
    for i, x in by_id.items():
        ts = [int(v) for v in x["t"]]
        if len(ts) < min_frames:
            continue
        pr = [(_props(masks_at, by_id, i, t, pixel_um), t) for t in ts]
        pr = [(p, t) for p, t in pr if p is not None]
        if len(pr) < min_frames:
            continue
        a = np.log([p["area_um2"] for p, _ in pr]); tm = np.array([t * frame_min for _, t in pr], float)
        k = float(np.polyfit(tm, a, 1)[0])
        if -0.02 < k < 0.05:
            rates.append(k); dists.append(float(np.mean([(box["col1"] - p["x_px"]) * pixel_um for p, _ in pr])))
    if len(rates) < 10:
        return dict(n_tracks=len(rates), pos_slope=POS_SLOPE_MEASURED, measured=False)
    b, a0 = np.polyfit(dists, rates, 1)
    slope = float(-b / a0) if a0 > 0 else 0.0
    return dict(n_tracks=len(rates), pos_slope=float(max(slope, 0.0)), rate_at_opening_per_min=float(a0), measured=True)
