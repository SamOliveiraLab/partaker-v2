"""Seeding stage rules: the numbers of the Seeding page, measured from the tool's own cell rows and lineage tracks.

Pure Python (numpy, polars, scipy), no Qt, testable headless. Each rule is one small function that returns a Metric and
says honestly when the recording cannot give the number. Every threshold a paper left undefined is a field of
SeedingSettings with its default and its meaning; nothing is read from the literature at run time.

Sources: Lee 2018 (PNAS 115:4471) for arrivals, families and divisions; Lecuyer 2011 (Biophys J 100:341) for residence,
survival and motion under flow; Hubert 2024 (Nat Commun 15:6161) for dividers versus holders and the growth rate;
Conrad 2011 (Biophys J 100:1608) for the pose at landing.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ORIGINS = ("landed", "entered field", "born on surface", "present at start", "split fragment")
# a split fragment is the shorter piece of a mask flicker the tracker undid: not a cell, kept in per_track, never counted
FATES = ("detached", "left field", "divided", "present at end")
FLOW_AXES = {
    "left to right": (1.0, 0.0),
    "right to left": (-1.0, 0.0),
    "top to bottom": (0.0, 1.0),
    "bottom to top": (0.0, -1.0),
}


from nd2_analyzer.analysis.biofilm_stages.seeding_types import Metric, SeedingResults, SeedingSettings


def full_settings(settings) -> SeedingSettings:
    """The settings given, or the defaults when the caller passed None."""
    return settings if settings is not None else SeedingSettings()


@dataclass
class SeedingInputs:
    cells: pl.DataFrame            # MetricsService.df rows of one position (may be a subset of frames)
    tracks: list                   # lineage tracks of that position (may be empty: then tracking metrics are unavailable)
    dt_s: float                    # seconds per frame
    pixel_um: Optional[float]
    start_hour: float              # hours after inoculation at frame 0 of the recording
    n_frames: int                  # frames in the recording
    field_shape: tuple             # (Y, X) pixels
    flow: Optional[dict]           # the experiment's flow block or None
    settings: SeedingSettings = field(default_factory=SeedingSettings)
    frame_range: Optional[tuple] = None   # (first, last) frames the tool segmented; measure_seeding refuses to run without it


TRACK_SCHEMA = {
    "track_id": pl.Int64, "frame_start": pl.Int64, "frame_end": pl.Int64, "n_frames": pl.Int64,
    "t_start_h": pl.Float64, "t_end_h": pl.Float64, "residence_s": pl.Float64,
    "origin": pl.Utf8, "fate": pl.Utf8, "censored": pl.Boolean,
    "x_start": pl.Float64, "y_start": pl.Float64, "x_end": pl.Float64, "y_end": pl.Float64,
    "path_px": pl.Float64, "net_disp_px": pl.Float64, "path_um": pl.Float64, "net_disp_um": pl.Float64,
    "motile": pl.Boolean, "first_half_step_px": pl.Float64, "second_half_step_px": pl.Float64, "crawl_then_stop": pl.Boolean,
    "pose_first": pl.Utf8, "length_over_width_first": pl.Float64, "orientation_first": pl.Float64,
    "parent": pl.Int64, "n_children": pl.Int64, "generation": pl.Int64, "family_id": pl.Int64,
    "divider": pl.Boolean, "along_flow_um": pl.Float64,
}


# ----------------------------------------------------------------------------------------------------------------------
# small helpers

def hour_of_frame(frame, dt_s: float, start_hour: float):
    return start_hour + np.asarray(frame, dtype=float) * dt_s / 3600.0


def _unavailable(key, name, reason, source, group, unit="") -> Metric:
    return Metric(key=key, name=name, value=None, unit=unit, text="not available", available=False,
                  reason=reason, source=source, group=group)


def _pct(part, whole) -> float:
    return 100.0 * part / whole if whole else float("nan")


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def _near_edge(x: float, y: float, margin: int, shape) -> bool:
    ny, nx = int(shape[0]), int(shape[1])
    return x < margin or y < margin or x > nx - 1 - margin or y > ny - 1 - margin


def _normalise_tracks(tracks: list) -> list:
    """Tracks as plain records with sorted numpy arrays; tracks without points are dropped."""
    out = []
    for tr in tracks or []:
        t = np.asarray(tr.get("t", []), dtype=np.int64)
        if t.size == 0:
            continue
        order = np.argsort(t, kind="stable")
        parent = tr.get("parent")
        out.append({
            "id": int(tr["ID"]),
            "t": t[order],
            "x": np.asarray(tr["x"], dtype=float)[order],
            "y": np.asarray(tr["y"], dtype=float)[order],
            "parent": None if parent is None or int(parent) == int(tr["ID"]) else int(parent),
            "children": [int(c) for c in (tr.get("children") or [])],
            "fragment": bool(tr.get("fragment", False)),
            "cell_ids": (np.asarray(tr["cell_ids"], dtype=np.int64)[order]
                         if tr.get("cell_ids") is not None and len(tr["cell_ids"]) == t.size else None),
        })
    return out


def _frame_range(inp: SeedingInputs):
    """The frames analysed. The span cannot be read off the data: trailing empty frames leave no cell row and no track,
    and without them take-off and 'present at end' come out wrong, so the caller must say which frames were segmented."""
    if inp.frame_range is None:
        raise ValueError("frame_range is needed: which frames were segmented")
    first, last = int(inp.frame_range[0]), int(inp.frame_range[1])
    if last < first:
        raise ValueError(f"frame_range {inp.frame_range} ends before it starts")
    return first, last


# ----------------------------------------------------------------------------------------------------------------------
# joining cell rows to tracks

def attach_track_ids(cells: pl.DataFrame, tracks: list, max_px: float = 3.0) -> pl.DataFrame:
    """The cell rows with a track_id column. When every track carries the segmentation label of each of its points
    (cell_ids, as the mask tracker writes them) the join is exact on (time, cell_id); otherwise each track point claims
    the nearest cell row of its frame within max_px.

    A cell row claimed by two tracks keeps the nearer one. Rows no track claims get a null track_id."""
    if cells is None or cells.height == 0:
        return cells
    recs = tracks if tracks and isinstance(tracks[0], dict) and "id" in tracks[0] else _normalise_tracks(tracks)
    if recs and all(r.get("cell_ids") is not None for r in recs) and "cell_id" in cells.columns:
        return attach_track_ids_by_label(cells, recs)
    tid = np.full(cells.height, -1, dtype=np.int64)
    best = np.full(cells.height, np.inf)
    if not recs:
        return cells.with_columns(pl.lit(None, dtype=pl.Int64).alias("track_id"))
    points = pl.DataFrame({
        "track_id": np.concatenate([np.full(r["t"].size, r["id"], dtype=np.int64) for r in recs]),
        "time": np.concatenate([r["t"] for r in recs]),
        "x": np.concatenate([r["x"] for r in recs]),
        "y": np.concatenate([r["y"] for r in recs]),
    })
    by_frame = (cells.with_row_index("_row")
                .group_by("time").agg(pl.col("_row"), pl.col("centroid_x"), pl.col("centroid_y")))
    frames = {int(t): (np.asarray(r), np.column_stack([np.asarray(cx), np.asarray(cy)]))
              for t, r, cx, cy in by_frame.iter_rows()}
    for t, ids, xs, ys in points.group_by("time").agg(pl.col("track_id"), pl.col("x"), pl.col("y")).iter_rows():
        hit = frames.get(int(t))
        if hit is None:
            continue
        rows, xy = hit
        dist, idx = cKDTree(xy).query(np.column_stack([np.asarray(xs), np.asarray(ys)]))
        for d, i, tid_i in zip(np.atleast_1d(dist), np.atleast_1d(idx), ids):
            if d <= max_px and d < best[rows[i]]:
                best[rows[i]] = d
                tid[rows[i]] = tid_i
    col = pl.Series("track_id", tid, dtype=pl.Int64)
    return cells.with_columns(pl.when(col >= 0).then(col).otherwise(None).alias("track_id"))


def attach_track_ids_by_label(cells: pl.DataFrame, recs: list) -> pl.DataFrame:
    """The exact join: the track whose point at (time, cell_id) is this row. A label two tracks claim keeps the last."""
    keyed = pl.DataFrame({
        "time": np.concatenate([r["t"] for r in recs]),
        "cell_id": np.concatenate([r["cell_ids"] for r in recs]),
        "track_id": np.concatenate([np.full(r["t"].size, r["id"], dtype=np.int64) for r in recs]),
    }).unique(subset=["time", "cell_id"], keep="last")
    out = cells.drop("track_id") if "track_id" in cells.columns else cells
    return out.join(keyed.with_columns(pl.col("time").cast(out["time"].dtype), pl.col("cell_id").cast(out["cell_id"].dtype)),
                    on=["time", "cell_id"], how="left")


def _first_rows(cells: pl.DataFrame, recs: list) -> dict:
    """track id -> the cell row at the track's first frame, from the track_id column."""
    if cells is None or cells.height == 0 or "track_id" not in cells.columns:
        return {}
    first = {r["id"]: int(r["t"][0]) for r in recs}
    keyed = cells.filter(pl.col("track_id").is_not_null())
    out = {}
    for row in keyed.select(["track_id", "time", "major_axis_length", "minor_axis_length", "orientation"]).iter_rows(named=True):
        if first.get(row["track_id"]) == row["time"] and row["track_id"] not in out:
            out[row["track_id"]] = row
    return out


# ----------------------------------------------------------------------------------------------------------------------
# per frame counts and the per track table

def count_cells_per_frame(cells: pl.DataFrame, first: int, last: int, dt_s: float, start_hour: float) -> pl.DataFrame:
    """One row per frame of the analysed span: frame, hour, cells, mask_px, empty."""
    frames = np.arange(first, last + 1, dtype=np.int64)
    n = np.zeros(frames.size, dtype=np.int64)
    px = np.zeros(frames.size, dtype=float)
    if cells is not None and cells.height and "time" in cells.columns:
        agg = cells.group_by("time").agg(pl.len().alias("n"), pl.col("area").sum().alias("px"))
        for t, k, a in agg.iter_rows():
            if first <= t <= last:
                n[t - first] = k
                px[t - first] = a
    return pl.DataFrame({"frame": frames, "hour": hour_of_frame(frames, dt_s, start_hour), "cells": n,
                         "mask_px": px, "empty": n == 0})


def _generations(recs: list):
    """generation (root = 0) and family id (the root's id) per track, following parent links."""
    by_id = {r["id"]: r for r in recs}
    gen, fam = {}, {}

    def climb(tid):
        if tid in gen:
            return gen[tid], fam[tid]
        chain = []
        cur = tid
        while cur is not None and cur in by_id and cur not in gen and cur not in chain:
            chain.append(cur)
            cur = by_id[cur]["parent"]
            if cur is not None and cur not in by_id:
                cur = None
        g0, f0 = (gen[cur], fam[cur]) if cur in gen else (-1, chain[-1])
        for k, tid_k in enumerate(reversed(chain)):
            gen[tid_k] = g0 + 1 + k
            fam[tid_k] = f0
        return gen[tid], fam[tid]

    for r in recs:
        climb(r["id"])
    return gen, fam


def _steps_px(r: dict) -> np.ndarray:
    """Step length per frame between consecutive points (a gap of k frames counts as k)."""
    if r["t"].size < 2:
        return np.zeros(0)
    d = np.hypot(np.diff(r["x"]), np.diff(r["y"]))
    gaps = np.maximum(np.diff(r["t"]), 1)
    return d / gaps


def _crawl_halves(steps: np.ndarray):
    half = steps.size // 2
    if half == 0:
        return float("nan"), float("nan")
    return float(steps[:half].mean()), float(steps[half:].mean())


def measure_tracks(recs: list, first_rows: dict, inp: SeedingInputs, first: int, last: int,
                   minor_median_px: Optional[float]) -> pl.DataFrame:
    """One row per track with its origin, fate, residence, motion, pose, lineage and flow shift."""
    S = inp.settings
    gen, fam = _generations(recs)
    axis = FLOW_AXES.get((inp.flow or {}).get("direction")) if inp.flow and inp.flow.get("present") else None
    px = inp.pixel_um
    rows = {k: [] for k in TRACK_SCHEMA}
    for r in recs:
        t, x, y = r["t"], r["x"], r["y"]
        n_children = len(r["children"])
        origin = ("present at start" if t[0] == first else "split fragment" if r.get("fragment")
                  else "born on surface" if r["parent"] is not None
                  else "entered field" if _near_edge(x[0], y[0], S.edge_margin_px, inp.field_shape) else "landed")
        fate = ("divided" if n_children >= 2 else "present at end" if t[-1] == last
                else "left field" if _near_edge(x[-1], y[-1], S.edge_margin_px, inp.field_shape) else "detached")
        steps = _steps_px(r)
        path_px = float(np.hypot(np.diff(x), np.diff(y)).sum()) if t.size > 1 else 0.0
        net_px = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
        residence_s = float(t.size) * inp.dt_s
        motile = (net_px > S.motile_fraction_of_minor_axis * minor_median_px
                  if minor_median_px and t.size >= S.motile_min_frames else None)
        h1, h2 = _crawl_halves(steps)
        crawl = (h1 > S.crawl_step_px >= h2) if t.size >= S.crawl_min_frames else None
        fr = first_rows.get(r["id"])
        ratio = (fr["major_axis_length"] / fr["minor_axis_length"]
                 if fr and fr["minor_axis_length"] and fr["minor_axis_length"] > 0 else None)
        pose = "unknown" if ratio is None else ("on end" if ratio < S.on_end_length_over_width else "flat")
        if px is None or inp.dt_s > S.divider_max_dt_s:
            divider = None
        elif n_children >= 2:
            divider = True
        elif residence_s >= S.holder_min_track_s:
            divider = (net_px * px) ** 2 > S.divider_msd_um2
        else:
            divider = None
        along = (((x[-1] - x[0]) * axis[0] + (y[-1] - y[0]) * axis[1]) * px
                 if axis is not None and px is not None and t.size > 1 else None)
        vals = {
            "track_id": r["id"], "frame_start": int(t[0]), "frame_end": int(t[-1]), "n_frames": int(t.size),
            "t_start_h": float(hour_of_frame(t[0], inp.dt_s, inp.start_hour)),
            "t_end_h": float(hour_of_frame(t[-1], inp.dt_s, inp.start_hour)), "residence_s": residence_s,
            "origin": origin, "fate": fate, "censored": bool(t[0] == first or t[-1] == last),
            "x_start": float(x[0]), "y_start": float(y[0]), "x_end": float(x[-1]), "y_end": float(y[-1]),
            "path_px": path_px, "net_disp_px": net_px,
            "path_um": path_px * px if px is not None else None, "net_disp_um": net_px * px if px is not None else None,
            "motile": motile, "first_half_step_px": h1, "second_half_step_px": h2, "crawl_then_stop": crawl,
            "pose_first": pose, "length_over_width_first": ratio,
            "orientation_first": float(fr["orientation"]) if fr else None,
            "parent": r["parent"], "n_children": n_children, "generation": gen[r["id"]], "family_id": fam[r["id"]],
            "divider": divider, "along_flow_um": along,
        }
        for k in TRACK_SCHEMA:
            rows[k].append(vals[k])
    return pl.DataFrame(rows, schema=TRACK_SCHEMA)


# ----------------------------------------------------------------------------------------------------------------------
# group "arrivals"

def _needs_tracks(key, name, source, group, unit=""):
    return _unavailable(key, name, "no tracks for this position", source, group, unit)


def cells_on_surface(counts: pl.DataFrame) -> Metric:
    n = int(counts["cells"][-1]) if counts.height else 0
    return Metric("cells_on_surface", "Cells on the surface", float(n), "cells", _plural(n, "cell"), True, "",
                  "Lee 2018 Fig. 2B", "arrivals",
                  series={"hour": counts["hour"].to_list(), "cells": counts["cells"].to_list()}, n=counts.height)


def _counts_rise(counts: pl.DataFrame, frac: float) -> bool:
    k = max(1, int(round(counts.height * frac)))
    n = counts["cells"].to_numpy()
    return float(n[-k:].mean()) > float(n[:k].mean())


def _log_linear_fit(hours: np.ndarray, n: np.ndarray):
    """Least squares of ln N on hour over frames with N > 0: slope (1/h), intercept."""
    keep = n > 0
    if keep.sum() < 2:
        return None
    b, a = np.polyfit(hours[keep], np.log(n[keep]), 1)
    return float(b), float(a)


def _fit_lag_curve(hours: np.ndarray, n: np.ndarray, n0: float):
    """Least squares of N(t) = n0 exp((t - t_lag) / tau) on the counts themselves, as Lee fit Fig. 2B: (t_lag, tau) or
    None. n0 and t_lag only enter as one number, so n0 is held at the first frame's count and t_lag is the hour at which
    the fitted curve passes that count. The start values come from the log-linear fit."""
    from scipy.optimize import curve_fit
    start = _log_linear_fit(hours, n)
    if start is None or start[0] <= 0:
        return None
    b, a = start
    guess = [(math.log(n0) - a) / b, 1.0 / b]

    def curve(t, t_lag, tau):
        return n0 * np.exp((t - t_lag) / tau)

    try:
        (t_lag, tau), _ = curve_fit(curve, hours, n, p0=guess, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    if not np.isfinite(t_lag) or not np.isfinite(tau) or tau <= 0:
        return None
    return float(t_lag), float(tau)


def lag_time(counts: pl.DataFrame, S: SeedingSettings) -> Metric:
    """Lee Fig. 2B: N(t) = N0 exp((t - t_lag) / tau) fitted to the counts from the first frame with a cell on the surface.
    N0 is that frame's count; t_lag and tau are free."""
    key, name, src, grp = "lag_time", "Lag before growth", "Lee 2018 Fig. 2B", "arrivals"
    h0, h1 = (float(counts["hour"][0]), float(counts["hour"][-1])) if counts.height else (0.0, 0.0)
    if counts.height < S.min_fit_frames:
        return _unavailable(key, name, f"only {_plural(counts.height, 'frame')}, the fit needs {S.min_fit_frames}", src, grp,
                            "h")
    if not _counts_rise(counts, S.rising_fraction):
        return _unavailable(key, name, "the count never rises in these frames", src, grp, "h")
    seen = counts.filter(pl.col("cells") > 0)
    start = int(seen["frame"][0]) if seen.height else int(counts["frame"][0])
    fitted = counts.filter(pl.col("frame") >= start)          # from the first frame with a cell on the surface
    hours = fitted["hour"].to_numpy()
    n = fitted["cells"].to_numpy().astype(float)
    n0 = float(n[0]) if n[0] > 0 else 1.0
    fit = _fit_lag_curve(hours, n, n0)
    if fit is None:
        return _unavailable(key, name, "the count never rises in these frames", src, grp, "h")
    t_lag, tau_h = fit
    series = {"hour": hours.tolist(), "fit": (n0 * np.exp((hours - t_lag) / tau_h)).tolist(), "tau_h": tau_h, "n0": n0,
              "n0_hour": float(hours[0]), "t_lag_h": t_lag}
    outside = _lag_outside_frames(t_lag, tau_h, h0, h1, S)
    if outside:
        m = _unavailable(key, name, outside, src, grp, "h")
        m.series = series
        return m
    return Metric(key, name, t_lag, "h", f"Lag {t_lag:.1f} h", True, "", src, grp, series=series, n=int((n > 0).sum()))


def _lag_outside_frames(t_lag: float, tau_h: float, h0: float, h1: float, S: SeedingSettings) -> str:
    """Why a fitted lag cannot be trusted: Lee's recipe needs the counts to show the flat phase and the rise. With N0 held
    at the first count, a count that already rises at the first frame puts t_lag at or before h0, so the flat phase must
    last at least min_flat_phase_taus e-fold times inside the frames; a lag after the last frame means the rise is not
    in them."""
    if t_lag < h0 + S.min_flat_phase_taus * tau_h:
        return (f"the fitted lag {t_lag:.1f} h sits at or before the start of these frames ({h0:.1f} to {h1:.1f} h), "
                "the recording starts after the lag")
    if t_lag > h1:
        return f"the fitted lag {t_lag:.1f} h falls after these frames ({h0:.1f} to {h1:.1f} h), the rise is not in them"
    return ""


def take_off_hour(counts: pl.DataFrame) -> Metric:
    """Lee: the first hour after which no frame is empty again."""
    key, name, src, grp = "take_off_hour", "Take-off (surface never empty after)", "Lee 2018 Fig. 2B", "arrivals"
    empty = counts["empty"].to_numpy()
    if not empty.any():
        return _unavailable(key, name, "no frame was ever empty", src, grp, "h")
    if empty[-1]:
        return _unavailable(key, name, "the surface is still empty at the end", src, grp, "h")
    last_empty = int(np.flatnonzero(empty)[-1])
    h = float(counts["hour"][last_empty + 1])
    return Metric(key, name, h, "h", f"Take-off at {h:.1f} h", True, "", src, grp, n=counts.height)


def landings(pt: pl.DataFrame, counts: pl.DataFrame) -> Metric:
    key, name, src, grp = "landings", "Landings", "Lee 2018 Fig. 1B", "arrivals"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "cells")
    landed = pt.filter(pl.col("origin") == "landed")
    h0, h1 = float(counts["hour"][0]), float(counts["hour"][-1])
    edges = np.arange(math.floor(h0), math.floor(h1) + 2, 1.0)
    per_hour, _ = np.histogram(landed["t_start_h"].to_numpy(), bins=edges)
    return Metric(key, name, float(landed.height), "cells", _plural(landed.height, "landing"), True, "", src, grp,
                  series={"hour": edges[:-1].tolist(), "landings": per_hour.tolist()}, n=landed.height)


def _dt_too_coarse(dt_s: float, S: SeedingSettings) -> Optional[str]:
    limit = S.stay_window_s / 3.0
    if dt_s > limit:
        return f"frames {dt_s:g} s apart, the rule needs under {limit:.0f} s"
    return None


def _landed_with_known_end(pt: pl.DataFrame, S: SeedingSettings) -> pl.DataFrame:
    """Landed tracks whose stay against the window is known: every one with an end, plus those still there at the last
    frame but already seen longer than the window (they stayed). Only landings still under the window at the end are
    unknown and left out."""
    return pt.filter((pl.col("origin") == "landed") & ~_still_unknown_at_end(S))


def _still_unknown_at_end(S: SeedingSettings):
    return (pl.col("fate") == "present at end") & (pl.col("residence_s") <= S.stay_window_s)


def leave_within_window(pt: pl.DataFrame, dt_s: float, S: SeedingSettings) -> Metric:
    key, name, src, grp = "leave_within_30s", f"Leave within {S.stay_window_s:.0f} s", "Lee 2018 Fig. 1B", "arrivals"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    reason = _dt_too_coarse(dt_s, S)
    if reason:
        return _unavailable(key, name, reason, src, grp, "%")
    landed = _landed_with_known_end(pt, S)
    if landed.height == 0:
        return _unavailable(key, name, "no landing with a known end", src, grp, "%")
    left = int((landed["residence_s"] <= S.stay_window_s).sum())
    pct = _pct(left, landed.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} %", True, "", src, grp, n=landed.height)


def residence_median(pt: pl.DataFrame, dt_s: float) -> Metric:
    """Residence of the landed tracks that do not touch the last frame; log-spaced histogram to 40 min. Tracks that entered
    the field or were born on the surface are not landings and stay out, as in Lee Fig. 1B."""
    key, name, src, grp = "residence_median", "Median stay of a landing", "Lee 2018 Fig. 1B", "arrivals"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "s")
    kept = pt.filter((pl.col("origin") == "landed") & ~pl.col("censored"))
    if kept.height == 0:
        return _unavailable(key, name, "no landing with a known end", src, grp, "s")
    res = kept["residence_s"].to_numpy()
    hi = max(2400.0, dt_s * 10)
    edges = np.geomspace(dt_s, hi, 21)
    hist, _ = np.histogram(np.clip(res, dt_s, hi), bins=edges)
    med = float(np.median(res))
    return Metric(key, name, med, "s", f"Median stay {med:.0f} s", True, "", src, grp,
                  series={"edges_s": edges.tolist(), "counts": hist.tolist()}, n=kept.height)


def stay_by_landing_hour(pt: pl.DataFrame, dt_s: float, S: SeedingSettings) -> Metric:
    key, name, src, grp = ("stay_by_landing_hour", f"Still there after {S.stay_window_s:.0f} s", "Lee 2018 Fig. 1B",
                           "arrivals")
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    reason = _dt_too_coarse(dt_s, S)
    if reason:
        return _unavailable(key, name, reason, src, grp, "%")
    landed = _landed_with_known_end(pt, S)
    if landed.height == 0:
        return _unavailable(key, name, "no landing with a known end", src, grp, "%")
    stayed = landed.with_columns((pl.col("residence_s") > S.stay_window_s).alias("stayed"),
                                 pl.col("t_start_h").floor().alias("hour"))
    g = stayed.group_by("hour").agg(pl.col("stayed").mean().alias("p"), pl.len().alias("n")).sort("hour")
    p = g["p"].to_numpy()
    n = g["n"].to_numpy()
    se = 100.0 * np.sqrt(p * (1 - p) / n)
    pct_all = _pct(int(stayed["stayed"].sum()), stayed.height)
    return Metric(key, name, pct_all, "%", f"{pct_all:.1f} %", True, "",
                  src, grp, series={"hour": g["hour"].to_list(), "pct": (100 * p).tolist(), "se": se.tolist(),
                                    "n": n.tolist()}, n=stayed.height)


# ----------------------------------------------------------------------------------------------------------------------
# group "commitment": where reversible attachment turns irreversible

def _is_committed(S: SeedingSettings):
    """A landed cell attached for good when it divided on the surface, or was still there at the last frame after staying
    at least irreversible_stay_s. A long stay that ends in leaving is not irreversible: on the Lee recording 37 of the 60
    landings that stayed an hour left again later."""
    return (pl.col("fate") == "divided") | ((pl.col("fate") == "present at end") & (pl.col("residence_s") >= S.irreversible_stay_s))


# what happened to each track, for the colours of the window, the figures and the cells list
FATE_KEYS = ("left", "while", "end", "divided")


def fate_class_expr(S: SeedingSettings):
    """left: left within the stay window; end: still there at the last frame after irreversible_stay_s; divided; while:
    everything else (stayed a while, or still there at the end too briefly to know)."""
    return (pl.when(pl.col("fate") == "divided").then(pl.lit("divided"))
            .when((pl.col("fate") == "present at end") & (pl.col("residence_s") >= S.irreversible_stay_s)).then(pl.lit("end"))
            .when(pl.col("fate").is_in(["detached", "left field"]) & (pl.col("residence_s") <= S.stay_window_s)).then(pl.lit("left"))
            .otherwise(pl.lit("while")))


def fate_class_labels(S: SeedingSettings) -> dict:
    return {"left": f"left within {S.stay_window_s:g} s", "while": "stayed a while", "end": "stayed to the end",
            "divided": "divided"}


def irreversible_begins(pt: pl.DataFrame, S: SeedingSettings) -> Metric:
    key, name, src, grp = "irreversible_begins", "Irreversible attachment begins", "Partaker commitment rule", "commitment"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "h")
    landed = pt.filter(pl.col("origin") == "landed")
    if landed.height == 0:
        return _unavailable(key, name, "no landing in these frames", src, grp, "h")
    com = landed.filter(_is_committed(S)).sort("t_start_h")
    if com.height == 0:
        return _unavailable(key, name, f"no landed cell divided or stayed to the end ({_hours_words(S.irreversible_stay_s / 3600)} or more)",
                            src, grp, "h")
    first = com.row(0, named=True)
    hour = float(first["t_start_h"])
    by_hour = landed.with_columns(_is_committed(S).alias("committed"), pl.col("t_start_h").floor().alias("hour"))
    g = by_hour.group_by("hour").agg(pl.col("committed").mean().alias("p"), pl.len().alias("n")).sort("hour")
    p = g["p"].to_numpy(); n = g["n"].to_numpy()
    return Metric(key, name, hour, "h", f"Irreversible attachment begins at {hour:.1f} h", True, "", src, grp,
                  series={"first_hour": hour, "first_frame": int(first["frame_start"]), "first_track": int(first["track_id"]),
                          "hour": g["hour"].to_list(), "pct": (100 * p).tolist(),
                          "se": (100 * np.sqrt(p * (1 - p) / n)).tolist(), "n": n.tolist(),
                          "stay_s": float(S.irreversible_stay_s)}, n=landed.height)


def committed_share(pt: pl.DataFrame, S: SeedingSettings) -> Metric:
    key, name, src, grp = "committed_share", "Landings that attach for good", "Partaker commitment rule", "commitment"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    landed = pt.filter(pl.col("origin") == "landed")
    known = landed.filter(~((pl.col("fate") == "present at end") & (pl.col("residence_s") < S.irreversible_stay_s)))
    if known.height == 0:
        return _unavailable(key, name, "no landing with a known end", src, grp, "%")
    k = int(known.filter(_is_committed(S)).height)
    pct = _pct(k, known.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} % of landings attach for good ({k:,} of {known.height:,})", True, "",
                  src, grp, n=known.height)


def committed_at_end(pt: pl.DataFrame, S: SeedingSettings) -> Metric:
    key, name, src, grp = "committed_at_end", "Cells at the end that attached for good", "Partaker commitment rule", "commitment"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "cells")
    at_end = pt.filter(pl.col("fate") == "present at end")
    if at_end.height == 0:
        return _unavailable(key, name, "no cell is left at the last frame", src, grp, "cells")
    k = int(at_end.filter(pl.col("residence_s") >= S.irreversible_stay_s).height)
    words = _hours_words(S.irreversible_stay_s / 3600)
    return Metric(key, name, float(k), "cells", f"{k} of {at_end.height} cells at the end have been there over {words}",
                  True, "", src, grp, n=at_end.height)


def survival_curve(pt: pl.DataFrame, dt_s: float, S: SeedingSettings) -> Metric:
    """Lecuyer Eq. 3: f(t) = share of leavers still attached after t."""
    key, name, src, grp = "survival_curve", "Time by which half have left", "Lecuyer 2011 Eq. 3, Fig. 2B", "arrivals"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "s")
    leavers = pt.filter(~pl.col("censored") & pl.col("fate").is_in(["detached", "left field"]))
    if leavers.height <= S.min_survival_leavers:
        return _unavailable(key, name, f"only {_plural(leavers.height, 'leaver')}, the curve needs more than "
                                       f"{S.min_survival_leavers}", src, grp, "s")
    res = leavers["residence_s"].to_numpy()
    t = np.arange(0, res.max() + dt_s, dt_s)
    f = np.array([(res > ti).mean() for ti in t])
    below = np.flatnonzero(f <= 0.5)
    half = float(t[below[0]]) if below.size else float(t[-1])
    return Metric(key, name, half, "s", f"Half have left by {half:.0f} s", True, "", src, grp,
                  series={"t_s": t.tolist(), "f": f.tolist()}, n=leavers.height)


def fraction_present_at_end(pt: pl.DataFrame) -> Metric:
    key, name, src, grp = "fraction_present_at_end", "Still there at the end", "Lecuyer 2011 Methods", "arrivals"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    k = int((pt["fate"] == "present at end").sum())
    pct = _pct(k, pt.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} %", True, "", src, grp, n=pt.height)


# ----------------------------------------------------------------------------------------------------------------------
# group "movement"

def motile_share(pt: pl.DataFrame, minor_median_px: Optional[float], S: SeedingSettings) -> Metric:
    key, name, src, grp = "motile_share", "Moved while attached", "Lecuyer 2011 Fig. 3C", "movement"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    if not minor_median_px:
        return _unavailable(key, name, "no cell rows to give the cell width", src, grp, "%")
    judged = pt.filter(pl.col("motile").is_not_null())
    if judged.height == 0:
        return _unavailable(key, name, f"no track of {S.motile_min_frames} frames or more", src, grp, "%")
    pct = _pct(int(judged["motile"].sum()), judged.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} % moved", True, "", src, grp, n=judged.height,
                  series={"threshold_px": S.motile_fraction_of_minor_axis * minor_median_px})


def pose_at_landing(pt: pl.DataFrame, S: SeedingSettings) -> Metric:
    key, name, src, grp = "pose_at_landing", "Land on one pole", "Conrad 2011 Fig. 3A", "movement"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    landed = pt.filter((pl.col("origin") == "landed") & (pl.col("pose_first") != "unknown"))
    if landed.height == 0:
        return _unavailable(key, name, "no cell row matched a landing", src, grp, "%")
    k = int((landed["pose_first"] == "on end").sum())
    pct = _pct(k, landed.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} %", True, "", src, grp, n=landed.height,
                  series={"length_over_width": landed["length_over_width_first"].to_list()})


def crawl_then_stop(pt: pl.DataFrame, S: SeedingSettings) -> Metric:
    key, name, src, grp = "crawl_then_stop", "Crawl, then stop", "Lee 2018 Fig. 5B", "movement"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    judged = pt.filter(pl.col("crawl_then_stop").is_not_null())
    if judged.height == 0:
        return _unavailable(key, name, f"no track of {S.crawl_min_frames} frames or more", src, grp, "%")
    pct = _pct(int(judged["crawl_then_stop"].sum()), judged.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} %", True, "", src, grp, n=judged.height)


# ----------------------------------------------------------------------------------------------------------------------
# group "division"

def division_count(pt: pl.DataFrame) -> Metric:
    key, name, src, grp = "division_count", "Divisions", "Lee 2018 Fig. 2C", "division"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "")
    div = pt.filter(pl.col("n_children") >= 2)
    return Metric(key, name, float(div.height), "", _plural(div.height, "division"), True, "", src, grp, n=div.height,
                  series={"hour": div["t_end_h"].to_list()})


def division_time(pt: pl.DataFrame, S: SeedingSettings) -> Metric:
    """Hours between a mother's split and her daughter's split; gaps over max_division_time_h are missed divisions."""
    key, name, src, grp = "division_time_h", "Time between divisions", "Lee 2018 Fig. S1A", "division"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "h")
    end_h = dict(zip(pt["track_id"].to_list(), pt["t_end_h"].to_list()))
    dividers = pt.filter((pl.col("n_children") >= 2) & pl.col("parent").is_not_null())
    gaps = [row["t_end_h"] - end_h[row["parent"]] for row in dividers.iter_rows(named=True) if row["parent"] in end_h]
    gaps = [g for g in gaps if 0 < g <= S.max_division_time_h]
    if not gaps:
        return _unavailable(key, name, "no family with two divisions in a row", src, grp, "h")
    med = float(np.median(gaps))
    return Metric(key, name, med, "h", _hours_words(med), True, "", src, grp, n=len(gaps), series={"hours": gaps})


def _hours_words(h: float) -> str:
    """A time in hours as it is read: seconds under a minute, minutes under an hour."""
    if h * 3600 < 60:
        return f"{h * 3600:.0f} s"
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def division_outcome(pt: pl.DataFrame, S: SeedingSettings) -> Metric:
    """Lee Fig. 3B: after each division, how many daughters stay. A daughter stayed when she was seen at least
    daughter_stay_s, or when she divided herself (a cell that split never detached). A daughter that left the field, or
    was still there at the last frame before the window was up, voids the event."""
    key, name, src, grp = "division_outcome", "Daughters that stay after a division", "Lee 2018 Fig. 3B", "division"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "")
    by_id = {row["track_id"]: row for row in pt.iter_rows(named=True)}
    tally = {"two": 0, "one": 0, "zero": 0}
    voided = 0
    for mother in pt.filter(pl.col("n_children") >= 2).iter_rows(named=True):
        kids = [by_id[k] for k in pt.filter(pl.col("parent") == mother["track_id"])["track_id"].to_list() if k in by_id]
        if len(kids) != 2:
            voided += 1
            continue
        unknown = any(k["fate"] == "left field" or (k["fate"] == "present at end" and k["residence_s"] < S.daughter_stay_s)
                      for k in kids)
        if unknown:
            voided += 1
            continue
        stayed = sum(_daughter_stayed(k, S) for k in kids)
        tally[("zero", "one", "two")[stayed]] += 1
    n = sum(tally.values())
    series = dict(tally, voided=voided)
    if n < S.min_divisions_scored:
        m = _unavailable(key, name, f"only {_plural(n, 'division')} scored", src, grp, "")
        m.series = series
        m.n = n
        return m
    return Metric(key, name, float(tally["one"]), "", f"One daughter stays in {tally['one']} of {n} (both {tally['two']}, none {tally['zero']})", True, "", src, grp,
                  series=series, n=n)


def _daughter_stayed(kid: dict, S: SeedingSettings) -> bool:
    return kid["fate"] == "divided" or kid["residence_s"] >= S.daughter_stay_s


def generations_per_family(pt: pl.DataFrame) -> Metric:
    """Lee Fig. 2C: a family is a root track and its descendants; headline is the share of cells in one-generation families."""
    key, name, src, grp = "generations_per_family", "Cells in one-generation families", "Lee 2018 Fig. 2C", "division"
    if pt.height == 0:
        return _needs_tracks(key, name, src, grp, "%")
    fam = pt.group_by("family_id").agg((pl.col("generation").max() + 1).alias("generations"), pl.len().alias("cells"))
    gens = fam["generations"].to_numpy()
    hist = np.bincount(gens, minlength=int(gens.max()) + 1)[1:]
    single = int(fam.filter(pl.col("generations") == 1)["cells"].sum())
    pct = _pct(single, pt.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} %", True, "", src, grp,
                  series={"generations": list(range(1, hist.size + 1)), "families": hist.tolist()}, n=fam.height)


def _frame_has_microcolony(xy: np.ndarray, eqd: np.ndarray, gap_px: float, min_cells: int) -> bool:
    """A group of at least min_cells cells whose edges are within gap_px of each other (single linkage)."""
    n = xy.shape[0]
    if n < min_cells:
        return False
    reach = gap_px + float(eqd.max())
    pairs = cKDTree(xy).query_pairs(reach, output_type="ndarray")
    if pairs.size == 0:
        return False
    d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T) - (eqd[pairs[:, 0]] + eqd[pairs[:, 1]]) / 2
    close = pairs[d <= gap_px]
    if close.size == 0:
        return False
    g = coo_matrix((np.ones(close.shape[0]), (close[:, 0], close[:, 1])), shape=(n, n))
    k, labels = connected_components(g, directed=False)
    return int(np.bincount(labels).max()) >= min_cells


def stage_by_frame(cells: pl.DataFrame, counts: pl.DataFrame, pixel_um: Optional[float], S: SeedingSettings,
                   irreversible_frame: Optional[int] = None):
    """frame, hour, stage: 'seeding' (cells land and leave), 'irreversible' from the landing of the first cell that
    attached for good, 'microcolonies' from the first run of persist frames with a microcolony."""
    frames = counts["frame"].to_numpy()
    has = np.zeros(frames.size, dtype=bool)
    if pixel_um and cells is not None and cells.height:
        gap_px = S.microcolony_gap_um / pixel_um
        grouped = cells.group_by("time").agg(pl.col("centroid_x"), pl.col("centroid_y"), pl.col("equivalent_diameter"))
        first = int(frames[0])
        for t, cx, cy, eqd in grouped.iter_rows():
            i = int(t) - first
            if 0 <= i < frames.size:
                has[i] = _frame_has_microcolony(np.column_stack([cx, cy]), np.asarray(eqd, dtype=float), gap_px,
                                                S.microcolony_cells)
    stage = np.full(frames.size, "seeding", dtype=object)
    if irreversible_frame is not None and frames.size:
        i0 = int(irreversible_frame) - int(frames[0])
        if 0 <= i0 < frames.size:
            stage[i0:] = "irreversible"
    k = S.microcolony_persist_frames
    run = np.convolve(has.astype(int), np.ones(k, dtype=int), mode="valid") if frames.size >= k else np.zeros(0)
    hits = np.flatnonzero(run >= k)
    handover = int(hits[0]) if hits.size else None
    if handover is not None:
        stage[handover:] = "microcolonies"
    df = pl.DataFrame({"frame": frames, "hour": counts["hour"], "stage": pl.Series(stage.tolist(), dtype=pl.Utf8)})
    return df, handover


def handover_metric(stages: pl.DataFrame, handover: Optional[int], pixel_um: Optional[float], S: SeedingSettings) -> Metric:
    key, name, src, grp = "handover_hour", "Start of microcolonies", "Partaker handover rule", "division"
    if not pixel_um:
        return _unavailable(key, name, "no pixel size in this recording", src, grp, "h")
    if handover is None:
        return _unavailable(key, name, f"no group of {S.microcolony_cells} cells within {S.microcolony_gap_um:g} um held "
                                       f"for {S.microcolony_persist_frames} frames", src, grp, "h")
    h = float(stages["hour"][handover])
    return Metric(key, name, h, "h", f"Microcolonies from {h:.1f} h", True, "", src, grp, n=stages.height)


# ----------------------------------------------------------------------------------------------------------------------
# group "flow"

def _no_flow(key, name, source, unit=""):
    return _unavailable(key, name, "no flow in this recording", source, "flow", unit)


def along_flow_shift(pt: pl.DataFrame, inp: SeedingInputs) -> Metric:
    key, name, src = "along_flow_shift", "Shift along the flow", "Lecuyer 2011 Fig. S1"
    if pt.height == 0:
        return _needs_tracks(key, name, src, "flow", "um")
    if inp.pixel_um is None:
        return _unavailable(key, name, "no pixel size in this recording", src, "flow", "um")
    if (inp.flow or {}).get("direction") not in FLOW_AXES:
        return _unavailable(key, name, "flow direction not set", src, "flow", "um")
    vals = pt.filter(pl.col("along_flow_um").is_not_null())["along_flow_um"].to_numpy()
    if vals.size == 0:
        return _unavailable(key, name, "no track with two frames", src, "flow", "um")
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else float("nan")
    word = "downstream" if mean >= 0 else "upstream"
    return Metric(key, name, mean, "um", f"{abs(mean):.2f} um {word}", True, "", src, "flow", n=vals.size,
                  series={"se_um": se, "along_flow_um": vals.tolist()})


def orientation_vs_flow(pt: pl.DataFrame, inp: SeedingInputs) -> Metric:
    """Share of tracks whose cell, at its first frame, lies within align_angle_deg of the flow axis.

    regionprops gives the angle between the image rows (y axis) and the major axis, so a cell along a left-right flow
    has an orientation near plus or minus 90 degrees, and along a top-bottom flow near zero."""
    S = inp.settings
    key, name, src = "orientation_vs_flow", f"Within {S.align_angle_deg:.0f} degrees of the flow", "Lecuyer 2011 Methods"
    if pt.height == 0:
        return _needs_tracks(key, name, src, "flow", "%")
    axis = FLOW_AXES.get((inp.flow or {}).get("direction"))
    if axis is None:
        return _unavailable(key, name, "flow direction not set", src, "flow", "%")
    rows = pt.filter(pl.col("orientation_first").is_not_null())
    if rows.height == 0:
        return _unavailable(key, name, "no cell row matched a track start", src, "flow", "%")
    ang = np.degrees(np.abs(rows["orientation_first"].to_numpy()))
    off_axis = np.abs(90.0 - ang) if axis[0] != 0 else ang
    k = int((off_axis <= S.align_angle_deg).sum())
    pct = _pct(k, rows.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} %", True, "", src, "flow", n=rows.height,
                  series={"off_axis_deg": off_axis.tolist()})


def dividers_vs_holders(pt: pl.DataFrame, inp: SeedingInputs) -> Metric:
    key, name, src = "dividers_vs_holders", "Dividers among long stayers", "Hubert 2024 Fig. 4"
    S = inp.settings
    if pt.height == 0:
        return _needs_tracks(key, name, src, "flow", "%")
    if inp.dt_s > S.divider_max_dt_s:
        return _unavailable(key, name, f"frames {inp.dt_s:g} s apart, the rule needs under {S.divider_max_dt_s:.0f} s",
                            src, "flow", "%")
    if inp.pixel_um is None:
        return _unavailable(key, name, "no pixel size in this recording", src, "flow", "%")
    judged = pt.filter(pl.col("divider").is_not_null())
    if judged.height == 0:
        return _unavailable(key, name, f"no track lasted {S.holder_min_track_s / 60:.0f} min", src, "flow", "%")
    by_split = int((judged["divider"] & (judged["n_children"] >= 2)).sum())
    k = int(judged["divider"].sum())
    by_movement = k - by_split
    pct = _pct(k, judged.height)
    return Metric(key, name, pct, "%", f"{pct:.1f} %, {by_split} split, {by_movement} moved", True, "", src, "flow",
                  n=judged.height, series={"dividers": k, "by_split": by_split, "by_movement": by_movement,
                                           "holders": judged.height - k})


def growth_rate_eta(counts: pl.DataFrame, take_off: Metric, inp: SeedingInputs) -> Metric:
    """Hubert Fig. 3c: slope of ln N against hour after take-off, labelled with the shear when known."""
    key, name, src = "growth_rate_eta", "Growth rate", "Hubert 2024 Fig. 3c"
    S = inp.settings
    if not _counts_rise(counts, S.rising_fraction):
        return _unavailable(key, name, "the count never rises in these frames", src, "flow", "1/h")
    after = counts.filter(pl.col("hour") >= take_off.value) if take_off.available else counts
    after = after.filter(pl.col("cells") > 0)
    if after.height < S.min_fit_frames:
        return _unavailable(key, name, f"only {_plural(after.height, 'frame')} with cells after take-off, the fit needs "
                                       f"{S.min_fit_frames}", src, "flow", "1/h")
    fit = _log_linear_fit(after["hour"].to_numpy(), after["cells"].to_numpy().astype(float))
    if fit is None:
        return _unavailable(key, name, "the count never rises in these frames", src, "flow", "1/h")
    eta = fit[0]
    shear = (inp.flow or {}).get("shear_mpa")
    name = f"Growth rate at {shear:g} mPa" if shear is not None else "Growth rate (shear unknown)"
    return Metric(key, name, eta, "1/h", f"{eta:.2f} per h", True, "", src, "flow", n=after.height,
                  series={"hour": after["hour"].to_list(), "ln_cells": np.log(after["cells"].to_numpy()).tolist(),
                          "shear_mpa": shear})


def flow_group(pt: pl.DataFrame, counts: pl.DataFrame, take_off: Metric, inp: SeedingInputs) -> list:
    if not (inp.flow and inp.flow.get("present")):
        return [_no_flow("along_flow_shift", "Shift along the flow", "Lecuyer 2011 Fig. S1", "um"),
                _no_flow("orientation_vs_flow", f"Within {inp.settings.align_angle_deg:.0f} degrees of the flow",
                         "Lecuyer 2011 Methods", "%"),
                _no_flow("dividers_vs_holders", "Dividers among long stayers", "Hubert 2024 Fig. 4", "%"),
                _no_flow("growth_rate_eta", "Growth rate", "Hubert 2024 Fig. 3c", "1/h")]
    return [along_flow_shift(pt, inp), orientation_vs_flow(pt, inp), dividers_vs_holders(pt, inp),
            growth_rate_eta(counts, take_off, inp)]


# ----------------------------------------------------------------------------------------------------------------------
# candidates, fates, the whole run

CANDIDATE_SCHEMA = {"track_id": pl.Int64, "why": pl.Utf8, "score": pl.Float64, "frame_start": pl.Int64,
                    "frame_end": pl.Int64, "t_start_h": pl.Float64, "t_end_h": pl.Float64}


def _stay_words(residence_s: float) -> str:
    return f"stays {residence_s / 3600:.1f} h" if residence_s >= 3600 else f"stays {residence_s:.0f} s"


def candidates_for_film(results: SeedingResults, k: int = 8) -> pl.DataFrame:
    """Top k tracks worth following: longest landed stayers, crawl-then-stop tracks and dividers, taken in turns."""
    pt = real_tracks(results.per_track)
    if pt.height == 0:
        return pl.DataFrame(schema=CANDIDATE_SCHEMA)
    stayers = pt.filter(pl.col("origin") == "landed").sort("residence_s", descending=True)
    crawlers = pt.filter(pl.col("crawl_then_stop") == True).sort("first_half_step_px", descending=True)  # noqa: E712
    dividers = pt.filter(pl.col("n_children") >= 2).sort("residence_s", descending=True)
    lists = [
        (stayers, lambda r: _stay_words(r["residence_s"]), lambda r: r["residence_s"]),
        (crawlers, lambda r: "crawls, then stops", lambda r: r["first_half_step_px"]),
        (dividers, lambda r: f"divides at {r['t_end_h']:.1f} h", lambda r: r["residence_s"]),
    ]
    iters = [(df.iter_rows(named=True), why, score) for df, why, score in lists]
    seen, out = set(), []
    while len(out) < k and iters:
        alive = []
        for it, why, score in iters:
            row = next(it, None)
            while row is not None and row["track_id"] in seen:
                row = next(it, None)
            if row is None:
                continue
            seen.add(row["track_id"])
            out.append({"track_id": row["track_id"], "why": why(row), "score": float(score(row)),
                        "frame_start": row["frame_start"], "frame_end": row["frame_end"],
                        "t_start_h": row["t_start_h"], "t_end_h": row["t_end_h"]})
            alive.append((it, why, score))
            if len(out) >= k:
                break
        iters = alive
    return pl.DataFrame(out, schema=CANDIDATE_SCHEMA)


def frame_fates(results: SeedingResults, frame: int) -> dict:
    """track_id -> 'leave' | 'stay' | 'divides' for the tracks alive at this frame, from their fate."""
    pt = real_tracks(results.per_track)
    if pt.height == 0:
        return {}
    alive = pt.filter((pl.col("frame_start") <= frame) & (pl.col("frame_end") >= frame))
    words = {"detached": "leave", "left field": "leave", "present at end": "stay", "divided": "divides"}
    return {int(t): words[f] for t, f in zip(alive["track_id"].to_list(), alive["fate"].to_list())}


def real_tracks(pt: pl.DataFrame) -> pl.DataFrame:
    """The per track rows that are cells: split fragments (mask flicker the tracker undid) are left out of every rule,
    unless the fragment went on to divide, which proves it was a cell."""
    if pt.height == 0 or "origin" not in pt.columns:
        return pt
    return pt.filter((pl.col("origin") != "split fragment") | (pl.col("n_children") >= 2))


def _median_minor_axis_px(cells: pl.DataFrame) -> Optional[float]:
    if cells is None or cells.height == 0 or "minor_axis_length" not in cells.columns:
        return None
    v = cells.filter(pl.col("minor_axis_length") > 0)["minor_axis_length"]
    return float(v.median()) if v.len() else None


def _notes(inp: SeedingInputs, first: int, last: int, recs: list, cells: pl.DataFrame, pt: pl.DataFrame,
           metrics: list) -> list:
    hours = hour_of_frame([first, last], inp.dt_s, inp.start_hour)
    notes = [f"Frames {first:,} to {last:,} ({hours[0]:.1f} to {hours[1]:.1f} h after inoculation), {inp.dt_s:g} s apart."]
    n_rows = cells.height if cells is not None else 0
    notes.append(f"{n_rows:,} cell rows and {len(recs):,} tracks from the tool.")
    if recs and n_rows:
        joined = int(cells["track_id"].is_not_null().sum()) if "track_id" in cells.columns else 0
        if all(r.get("cell_ids") is not None for r in recs):
            notes.append(f"{joined:,} cell rows matched a track point by their segmentation label.")
        else:
            notes.append(f"{joined:,} cell rows matched a track point within {inp.settings.join_radius_px:g} px.")
    if pt.height:
        fragments = int((pt["origin"] == "split fragment").sum())
        if fragments:
            notes.append(f"{fragments:,} short pieces where one cell's mask briefly split in two are listed but not counted as "
                         "cells.")
        counts = pt.group_by("origin").len().sort("origin")
        notes.append("Origins: " + ", ".join(f"{n:,} {o}" for o, n in counts.iter_rows()) + ".")
        counts = pt.group_by("fate").len().sort("fate")
        notes.append("Fates: " + ", ".join(f"{n:,} {f}" for f, n in counts.iter_rows()) + ".")
        notes += _landing_notes(real_tracks(pt), inp.settings)
    notes += _fit_notes(metrics)
    skipped = [f"{m.name.lower()} ({m.reason})" for m in metrics if not m.available]
    if skipped:
        notes.append("Not available: " + "; ".join(skipped) + ".")
    if inp.pixel_um is None:
        notes.append("No pixel size: lengths stay in pixels.")
    return notes


def _landing_notes(pt: pl.DataFrame, S: SeedingSettings) -> list:
    """How the landings that have no clean end were handled."""
    landed = pt.filter(pl.col("origin") == "landed")
    if not landed.height:
        return []
    out = []
    unknown = int(landed.filter(_still_unknown_at_end(S)).height)
    if unknown:
        out.append(f"{_plural(unknown, 'landing')} still under {S.stay_window_s:.0f} s at the last frame left out of the "
                   f"{S.stay_window_s:.0f} s rules.")
    left_field = int((landed["fate"] == "left field").sum())
    if left_field:
        out.append(f"{_plural(left_field, 'landed track')} left the field and count as leaving at their exit.")
    return out


def _fit_notes(metrics: list) -> list:
    """The conventions behind the fitted numbers, where the reader can see them."""
    out = []
    for m in metrics:
        if m.key == "lag_time" and m.series and m.series.get("n0") is not None:
            out.append(f"Lag fit: N0 held at the {_plural(int(m.series['n0']), 'cell')} of the first frame with a cell "
                       f"({m.series['n0_hour']:.1f} h); a different N0 moves the lag by tau times the log of the ratio.")
        if m.key == "dividers_vs_holders" and m.available:
            out.append("Divider rule from Hubert 2024 assumes a cell that moves has divided; crawling cells also pass it.")
    return out


def not_seeding_signs(counts: pl.DataFrame, pt: pl.DataFrame, handover: Optional[Metric], S: SeedingSettings) -> list:
    """Signs, in plain words, that the frames do not show seeding (cells arriving on a clean surface). They hold for any
    species and frame rate: the surface already crowded in the first frame, microcolonies already there in the first frame,
    more cells born on the surface than landing on it. One sign alone happens in real seeding, so fewer than
    S.not_seeding_signs signs give an empty list."""
    signs = []
    hours = counts["hour"].to_numpy() if counts.height and "hour" in counts.columns else np.array([])
    if counts.height and "cells" in counts.columns:
        first, most = int(counts["cells"][0]), int(counts["cells"].max())
        if most and first >= S.not_seeding_start_cells and first >= S.not_seeding_start_fraction * most:
            signs.append(f"{first:,} cells are already on the surface in the first frame, {100 * first / most:.0f} % of the "
                         f"most seen; seeding starts on a nearly empty surface.")
    if handover is not None and handover.available and handover.value is not None and hours.size:
        half_frame = (hours[1] - hours[0]) / 2 if hours.size > 1 else 0.0
        if handover.value <= hours[0] + half_frame:
            signs.append(f"Microcolonies are already there in the first frame ({hours[0]:.1f} h).")
    if pt.height and "origin" in pt.columns:
        born = int((pt["origin"] == "born on surface").sum()); landed = int((pt["origin"] == "landed").sum())
        if born > landed:
            signs.append(f"More cells are born on the surface ({born:,}) than land on it ({landed:,}); in seeding, new cells "
                         f"come from the liquid.")
    return signs if len(signs) >= S.not_seeding_signs else []


def measure_seeding(inp: SeedingInputs, progress: Optional[Callable] = None) -> SeedingResults:
    """Run every seeding rule on one position's cell rows and tracks."""
    S = full_settings(inp.settings)
    if not inp.dt_s or inp.dt_s <= 0:
        raise ValueError("dt_s must be positive: the frame interval is not set")
    first, last = _frame_range(inp)

    def tell(pct, text):
        if progress is not None:
            progress(int(pct), text)

    tell(0, "reading tracks")
    recs = _normalise_tracks(inp.tracks)
    cells = inp.cells if inp.cells is not None else pl.DataFrame()
    if cells.height and "time" in cells.columns:
        cells = cells.filter((pl.col("time") >= first) & (pl.col("time") <= last))
    tell(5, "joining cell rows to tracks")
    if recs and cells.height and "track_id" not in cells.columns:
        cells = attach_track_ids(cells, recs, S.join_radius_px)
    tell(20, "counting cells per frame")
    counts = count_cells_per_frame(cells, first, last, inp.dt_s, inp.start_hour)
    minor_px = _median_minor_axis_px(cells)
    tell(30, "measuring tracks")
    inp = replace(inp, settings=S)
    pt_all = measure_tracks(recs, _first_rows(cells, recs), inp, first, last, minor_px)
    if pt_all.height:
        pt_all = pt_all.with_columns(fate_class_expr(S).alias("fate_class"))
    pt = real_tracks(pt_all)
    tell(55, "arrivals")
    take_off = take_off_hour(counts)
    metrics = [cells_on_surface(counts), lag_time(counts, S), take_off, landings(pt, counts),
               leave_within_window(pt, inp.dt_s, S), residence_median(pt, inp.dt_s),
               stay_by_landing_hour(pt, inp.dt_s, S), survival_curve(pt, inp.dt_s, S), fraction_present_at_end(pt)]
    irr = irreversible_begins(pt, S)
    metrics += [irr, committed_share(pt, S), committed_at_end(pt, S)]
    tell(65, "movement")
    metrics += [motile_share(pt, minor_px, S), pose_at_landing(pt, S), crawl_then_stop(pt, S)]
    tell(70, "division")
    metrics += [division_count(pt), division_time(pt, S), division_outcome(pt, S), generations_per_family(pt)]
    tell(75, "handover rule")
    stages, handover = stage_by_frame(cells, counts, inp.pixel_um, S,
                                      irreversible_frame=(irr.series or {}).get("first_frame") if irr.available else None)
    metrics.append(handover_metric(stages, handover, inp.pixel_um, S))
    tell(90, "flow")
    metrics += flow_group(pt, counts, take_off, inp)
    results = SeedingResults(metrics=metrics, per_track=pt_all, counts=counts, candidates=pl.DataFrame(schema=CANDIDATE_SCHEMA),
                             stage_by_frame=stages, settings=S, notes=[])
    results.candidates = candidates_for_film(results)
    results.notes = _notes(inp, first, last, recs, cells, pt_all, metrics)
    results.check = not_seeding_signs(counts, pt, results.metric("handover_hour"), S)
    if results.check:
        results.notes.insert(0, "Check: these frames may not show seeding. " + " ".join(results.check))
    tell(100, "done")
    return results


def metrics_as_records(results: SeedingResults) -> list:
    """The metrics as plain dicts (series included) for JSON export."""
    return [asdict(m) for m in results.metrics]
