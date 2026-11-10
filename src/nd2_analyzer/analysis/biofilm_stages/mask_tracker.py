"""Link the tool's per-cell rows frame to frame into tracks with divisions, without touching the images.

This is the tracking step of the seeding stage. It works from MetricsService rows (one row per cell per frame: time,
cell_id, centroid, bounding box, area), so a recording of 35,000 frames is tracked in seconds and nothing is stacked in
memory. The rule is the one Lee et al. 2018 describe for their own tracking: most overlap, then least displacement.

Output: tracks in the same dict format as the trackers in nd2_analyzer.analysis.tracking
    {"ID": int, "x": [...], "y": [...], "t": [frame ints], "parent": int or None, "children": [ints], "position": int,
     "cell_ids": [per frame segmentation label], "area": [px per frame]}
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class MaskTrackerSettings:
    max_shift_px: float = 25.0        # a cell is not linked to a cell further than this (centre to centre, per frame)
    min_overlap_frac: float = 0.0     # bounding boxes may just touch; raise to demand real overlap
    division_area_low: float = 0.55   # two daughters together must hold at least this share of the mother's area
    division_area_high: float = 1.7   # and at most this share
    max_gap_frames: int = 0           # frames a cell may be missing and still be the same track (0 = none)
    min_mother_s: float = 600.0       # a division needs a mother tracked at least this long (a cell cannot divide twice in 10 min)
    min_daughter_s: float = 15.0      # and both daughters seen at least this long; otherwise the split was mask flicker and is undone
    max_mother_area_jump: float = 0.5 # a mother whose area ever grew more than this share in one frame had merged with a
                                      # neighbour; its later split is a separation, not a division


def _boxes(df: pl.DataFrame) -> np.ndarray:
    """Bounding boxes as (y1, x1, y2, x2) from the metric rows."""
    return np.column_stack([df["y1"].to_numpy(), df["x1"].to_numpy(), df["y2"].to_numpy(), df["x2"].to_numpy()]).astype(float)


def _overlap_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise overlap area of boxes a (n, 4) and b (m, 4)."""
    y1 = np.maximum(a[:, None, 0], b[None, :, 0]); x1 = np.maximum(a[:, None, 1], b[None, :, 1])
    y2 = np.minimum(a[:, None, 2], b[None, :, 2]); x2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(y2 - y1, 0, None) * np.clip(x2 - x1, 0, None)


def track_rows(cells: pl.DataFrame, position: int = 0, settings: MaskTrackerSettings | None = None,
               progress=None, dt_s: float = 1.0) -> list[dict]:
    """Track one position's rows. `cells` needs time, cell_id, centroid_x, centroid_y, area, y1, x1, y2, x2.
    `dt_s` turns the time thresholds of the settings into frames."""
    s = settings or MaskTrackerSettings()
    s.dt_s = float(dt_s) if dt_s and dt_s > 0 else 1.0
    df = cells.filter(pl.col("position") == position) if "position" in cells.columns else cells
    df = df.sort(["time", "cell_id"])
    if df.is_empty():
        return []
    frames = df["time"].unique().sort().to_list()
    by_frame = {t: g for t, g in df.group_by("time", maintain_order=True)} if hasattr(df, "group_by") else {}
    by_frame = {int(k[0] if isinstance(k, tuple) else k): g for k, g in by_frame.items()}

    tracks: dict[int, dict] = {}
    next_id = 1

    def new_track(t, row_i, g, parent=None):
        nonlocal next_id
        tid = next_id; next_id += 1
        tracks[tid] = {"ID": tid, "x": [float(g["centroid_x"][row_i])], "y": [float(g["centroid_y"][row_i])],
                       "t": [int(t)], "parent": parent, "children": [], "position": int(position),
                       "cell_ids": [int(g["cell_id"][row_i])], "area": [float(g["area"][row_i])]}
        return tid

    def extend(tid, t, row_i, g):
        tr = tracks[tid]
        tr["x"].append(float(g["centroid_x"][row_i])); tr["y"].append(float(g["centroid_y"][row_i]))
        tr["t"].append(int(t)); tr["cell_ids"].append(int(g["cell_id"][row_i])); tr["area"].append(float(g["area"][row_i]))

    # live[i] = track id of row i of the previous frame
    prev_t = frames[0]
    prev = by_frame[prev_t]
    live = [new_track(prev_t, i, prev) for i in range(prev.height)]

    for k, t in enumerate(frames[1:], start=1):
        cur = by_frame[t]
        if progress and k % 1000 == 0:
            progress(k, len(frames))
        gap_ok = (t - prev_t) <= 1 + s.max_gap_frames
        if not gap_ok or prev.height == 0:
            live = [new_track(t, i, cur) for i in range(cur.height)]
            prev, prev_t = cur, t
            continue
        pa, ca = _boxes(prev), _boxes(cur)
        pc = np.column_stack([prev["centroid_x"].to_numpy(), prev["centroid_y"].to_numpy()])
        cc = np.column_stack([cur["centroid_x"].to_numpy(), cur["centroid_y"].to_numpy()])
        dist = np.linalg.norm(pc[:, None, :] - cc[None, :, :], axis=2)
        ov = _overlap_area(pa, ca)
        allowed = dist <= s.max_shift_px
        if s.min_overlap_frac > 0:
            prev_area = np.clip((pa[:, 2] - pa[:, 0]) * (pa[:, 3] - pa[:, 1]), 1, None)
            allowed &= (ov / prev_area[:, None]) >= s.min_overlap_frac
        # most overlap first, least displacement second: a single sortable cost
        cost = np.where(allowed, -ov * 1e6 + dist, np.inf)
        order = np.argsort(cost, axis=None)
        matched_prev, matched_cur = {}, {}
        for flat in order:
            i, j = divmod(int(flat), cur.height)
            if not np.isfinite(cost[i, j]):
                break
            if i in matched_prev or j in matched_cur:
                continue
            matched_prev[i] = j; matched_cur[j] = i
        # divisions: an unmatched current cell that overlaps a previous cell already matched to another current cell
        new_live = [None] * cur.height
        divided = set()
        for j in range(cur.height):
            if j in matched_cur:
                continue
            cand = [i for i in range(prev.height) if ov[i, j] > 0 and dist[i, j] <= s.max_shift_px and i in matched_prev]
            if not cand:
                continue
            i = min(cand, key=lambda ii: dist[ii, j])
            j2 = matched_prev[i]
            mother_area = float(prev["area"][i])
            together = float(cur["area"][j]) + float(cur["area"][j2])
            if s.division_area_low * mother_area <= together <= s.division_area_high * mother_area and i not in divided:
                mother = live[i]
                d1 = new_track(t, j, cur, parent=mother); d2 = new_track(t, j2, cur, parent=mother)
                tracks[mother]["children"] = [d1, d2]
                new_live[j] = d1; new_live[j2] = d2
                divided.add(i)
        for j in range(cur.height):
            if new_live[j] is not None:
                continue
            if j in matched_cur and matched_cur[j] not in divided:
                tid = live[matched_cur[j]]
                extend(tid, t, j, cur); new_live[j] = tid
            else:
                new_live[j] = new_track(t, j, cur)
        live = new_live
        prev, prev_t = cur, t
    if progress:
        progress(len(frames), len(frames))
    return clean_divisions(list(tracks.values()), s)


def clean_divisions(tracks: list[dict], s: MaskTrackerSettings) -> list[dict]:
    """Undo splits that are mask flicker, not division: the mother continues as the longer-lived daughter and the
    other piece becomes a track of its own with no parent, marked as a fragment."""
    by_id = {tr["ID"]: tr for tr in tracks}
    changed = True
    while changed:
        changed = False
        for tr in list(by_id.values()):
            kids = [by_id[k] for k in tr.get("children", []) if k in by_id]
            if len(kids) != 2:
                continue
            area = np.asarray(tr["area"], dtype=float)
            jumps = np.diff(area) / np.clip(area[:-1], 1, None) if len(area) > 1 else np.array([0.0])
            grew_gradually = float(jumps.max()) <= s.max_mother_area_jump
            dt = getattr(s, "dt_s", 1.0)
            min_mother = max(2, int(np.ceil(s.min_mother_s / dt)))
            min_daughter = max(2, int(np.ceil(s.min_daughter_s / dt)))
            ok = (len(tr["t"]) >= min_mother and grew_gradually
                  and all(len(k["t"]) >= min_daughter for k in kids))
            if ok:
                continue
            keep, drop = sorted(kids, key=lambda k: len(k["t"]), reverse=True)
            for key in ("x", "y", "t", "cell_ids", "area"):
                tr[key].extend(keep[key])
            tr["children"] = list(keep.get("children", []))
            for gk in tr["children"]:
                if gk in by_id:
                    by_id[gk]["parent"] = tr["ID"]
            drop["parent"] = None; drop["fragment"] = True
            del by_id[keep["ID"]]
            changed = True
            break
    return list(by_id.values())


def tracks_summary(tracks: list[dict], dt_s: float, field_shape, edge_margin_px: int = 10,
                   first_frame: int | None = None, last_frame: int | None = None) -> pl.DataFrame:
    """One row per track with origin and fate in plain words, the way the seeding rules need them."""
    if not tracks:
        return pl.DataFrame()
    Y, X = field_shape
    f0 = min(tr["t"][0] for tr in tracks) if first_frame is None else first_frame
    f1 = max(tr["t"][-1] for tr in tracks) if last_frame is None else last_frame
    rows = []
    for tr in tracks:
        xs, ys, ts = tr["x"], tr["y"], tr["t"]
        at_edge_start = xs[0] < edge_margin_px or ys[0] < edge_margin_px or xs[0] > X - 1 - edge_margin_px or ys[0] > Y - 1 - edge_margin_px
        at_edge_end = xs[-1] < edge_margin_px or ys[-1] < edge_margin_px or xs[-1] > X - 1 - edge_margin_px or ys[-1] > Y - 1 - edge_margin_px
        if ts[0] <= f0:
            origin = "present at start"
        elif tr.get("fragment"):
            origin = "split fragment"
        elif tr.get("parent") is not None:
            origin = "born on surface"
        elif at_edge_start:
            origin = "entered field"
        else:
            origin = "landed"
        if tr.get("children"):
            fate = "divided"
        elif ts[-1] >= f1:
            fate = "present at end"
        elif at_edge_end:
            fate = "left field"
        else:
            fate = "detached"
        steps = np.hypot(np.diff(xs), np.diff(ys)) if len(xs) > 1 else np.array([])
        rows.append({"track_id": tr["ID"], "parent": tr.get("parent"), "n_children": len(tr.get("children", [])),
                     "frame_start": ts[0], "frame_end": ts[-1], "n_frames": len(ts),
                     "residence_s": len(ts) * dt_s, "origin": origin, "fate": fate,
                     "path_px": float(steps.sum()), "net_disp_px": float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])),
                     "x_first": xs[0], "y_first": ys[0], "x_last": xs[-1], "y_last": ys[-1],
                     "area_first_px": tr["area"][0], "area_last_px": tr["area"][-1], "area_median_px": float(np.median(tr["area"]))})
    return pl.DataFrame(rows, infer_schema_length=None)
