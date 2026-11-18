"""The numbers behind the plots of the Microcolonies and Maturation pages as tables (a header and rows), and as CSV files:
what the Numbers button of each page writes. Plain Python, no Qt. A missing value is an empty cell."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import microcolony_plots as MP
from .growth_measure import GrowthStates
from .microcolony_follow import MicrocolonyTracks
from .microcolony_metrics import SERIES, MicrocolonyMetrics

Table = Tuple[List[str], List[list]]
ENDED = {"joined": "merged"}                 # the follower's word, and the word the page uses


def _cell(value):
    """A number as it goes into a table: plain Python, a missing one as an empty cell."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def microcolony_tables(tracks: MicrocolonyTracks, position: int) -> Dict[str, Table]:
    """The followed microcolonies of one position: their number over time, the merges, one row per microcolony, one row
    per microcolony and frame."""
    keep = tracks.lasting(); hours, count = MP.count_series(tracks); _h, share = MP.cells_series(tracks)
    most: Dict[int, int] = {}
    for r in tracks.rows:
        most[r["identity"]] = max(most.get(r["identity"], 0), int(r["cells"]))
    followed = [[i, tracks.hours[tracks.identities[i]["birth"]], MP.end_hour(tracks, i), tracks.identities[i]["born"],
                 ENDED.get(tracks.identities[i]["ended"], tracks.identities[i]["ended"]), _cell(tracks.identities[i]["into"]), most[i]] for i in sorted(keep)]
    tail = f"_p{int(position)}"
    return {
        "microcolonies_over_time" + tail: (["hours", "frame", "microcolonies", "cells_in_microcolonies_pct"],
                                           [[float(h), int(t), int(n), float(s)] for h, t, n, s in zip(hours, tracks.frames, count, share)]),
        "microcolony_merges" + tail: (["hours", "frame", "microcolony", "merges_into", "cells", "cells_of_the_other"],
                                      [[float(e["hours"]), int(e["time"]), int(e["identity"]), int(e["into"]), _cell(e.get("cells_before")),
                                        _cell(e.get("cells_other_before"))] for e in tracks.merges()]),
        "microcolonies_followed" + tail: (["microcolony", "first_h", "last_h", "began", "ended", "merges_into", "most_cells"], followed),
        "microcolony_frames" + tail: (["microcolony", "frame", "hours", "cells", "area_um2"],
                                      [[int(r["identity"]), int(r["time"]), float(r["hours"]), int(r["cells"]), _cell(r.get("area_um2"))]
                                       for r in sorted((r for r in tracks.rows if r["identity"] in keep), key=lambda r: (r["identity"], r["time"]))]),
    }


def stain_tables(stain, position: int) -> Dict[str, Table]:
    """The matrix stain of the followed microcolonies of one position (matrix_stain.MicrocolonyStain): one row per
    microcolony (when its stain begins, how much of it is covered at most) and one row per microcolony and frame."""
    tail = f"_p{int(position)}"
    each = [[int(i), _cell(s["first_stain_h"]), float(s["most_cover_pct"]), _cell(s["most_cover_h"]), float(s["from_h"]), float(s["to_h"])]
            for i, s in sorted(stain.summary.items())]
    frames = [[int(r["identity"]), int(r["time"]), float(r["hours"]), float(r["area_um2"]), float(r["stain_um2"]), 100.0 * float(r["cover"])]
              for r in sorted(stain.rows, key=lambda r: (r["identity"], r["time"]))]
    return {"microcolony_matrix_stain" + tail: (["microcolony", "first_stain_h", "most_covered_pct", "most_covered_at_h", "followed_from_h", "followed_to_h"], each),
            "microcolony_matrix_stain_frames" + tail: (["microcolony", "frame", "hours", "area_um2", "stain_um2", "covered_pct"], frames)}


def _means(names, mean_of, have) -> Table:
    """hours, positions, then the mean and s.d. over the positions of every series. names: a series, or (series, the
    name of its column)."""
    head = ["hours", "positions"]; columns = []; hours = None
    for name in names:
        series, column = (name, name) if isinstance(name, str) else name
        hours, mean, sd = mean_of(series); head += [f"{column}_mean", f"{column}_sd"]; columns += [mean, sd]
    rows = [[float(h), int(have)] + [_cell(c[k]) for c in columns] for k, h in enumerate(hours)] if hours is not None else []
    return head, rows


def metric_tables(metrics: MicrocolonyMetrics) -> Dict[str, Table]:
    """The metrics of every measured position, their mean over the positions, and the density against the distance from
    the centre of the cell's own microcolony (every ring that holds a cell, all positions together)."""
    every = [[p, t, float(h)] + [_cell(int(metrics.values[name][p][k]) if name in ("cells", "microcolonies") else metrics.values[name][p][k]) for name in SERIES]
             for p in metrics.positions for k, (t, h) in enumerate(zip(metrics.frames[p], metrics.hours[p]))]
    hours, edges, _density = metrics.radial(1); rings = []
    total_sum = {}; total_n = {}
    for p in metrics.positions:
        for k, h in enumerate(metrics.hours[p]):
            key = round(float(h), 6)
            total_sum[key] = total_sum.get(key, 0) + metrics.ring_sum[p][k]; total_n[key] = total_n.get(key, 0) + metrics.ring_n[p][k]
    for h in sorted(total_sum):
        for r in np.flatnonzero(total_n[h] > 0):
            rings.append([h, float(edges[r]), float(edges[r + 1]), int(total_n[h][r]), float(total_sum[h][r] / total_n[h][r])])
    return {"microcolony_metrics_every_position": (["position", "frame", "hours"] + list(SERIES), every),
            "microcolony_metrics_mean": _means(SERIES, metrics.mean_of, len(metrics.positions)),
            "microcolony_density_by_distance": (["hours", "from_um", "to_um", "cells", "cells_within_3um_per_100um2"], rings)}


def growth_tables(gs: GrowthStates) -> Dict[str, Table]:
    """The growth states: every position and frame, the mean over the positions, and the thresholds Th1 and Th2."""
    names = (("matrix", "matrix_pct"), ("single", "single_stained_cells_pct"), ("inside", "matrix_inside_microcolonies_pct"),
             ("outside", "matrix_outside_pct"), ("colony_area", "microcolony_area_pct"))
    every = []
    for p in gs.positions:
        at = {round(float(h), 6): k for k, h in enumerate(gs.matrix_hours.get(p, ()))}
        for k, (t, h) in enumerate(zip(gs.frames[p], gs.hours[p])):
            m = at.get(round(float(h), 6))
            every.append([p, int(t), float(h), int(gs.cells[p][k]), float(gs.rate[p][k])] +
                         [_cell(getattr(gs, attr)[p][m]) if m is not None and p in getattr(gs, attr) else "" for attr, _name in names])
    head, rows = _means(("cells", ("rate", "cells_per_h")), gs.mean_of, len(gs.positions))
    if gs.matrix:
        mhead, mrows = _means(names, gs.mean_of, len(gs.positions)); by_hour = {r[0]: r[2:] for r in mrows}
        head += mhead[2:]; rows = [r + by_hour.get(r[0], [""] * (len(mhead) - 2)) for r in rows]
    limits = []
    if gs.th1 is not None:
        limits.append(["Th1", _cell(gs.th1["th1_h"]), "", "", f"matrix stain appears and stays; mean of {gs.th1['positions']} positions, s.d. {gs.th1['sd_h']:.2f} h"])
        limits += [[f"Th1 position {p}", _cell(h), "", "", "" if h is not None else "no matrix stain that stays"] for p, h in sorted(gs.th1["per_position_h"].items())]
    if gs.th2 is not None:
        low, high = gs.th2["range95_h"]
        limits.append(["Th2", _cell(gs.th2["th2_h"]), _cell(low), _cell(high),
                       ("reached in the recording" if gs.th2["reached"] else "predicted, not reached in the recording") +
                       f": growth rate down to 10 % of its peak on a Gompertz curve fitted to the mean count; range from {gs.th2['resamples']} resamples of the positions"])
    else:
        limits.append(["Th2", "", "", "", "no growth curve fits the count"])
    return {"growth_states_every_position": (["position", "frame", "hours", "cells", "cells_per_h"] + [name for _a, name in names], every),
            "growth_states_mean": (head, rows), "growth_states_thresholds": (["threshold", "hours", "low_h", "high_h", "note"], limits)}


def write_tables(folder, tables: Dict[str, Table]) -> List[str]:
    """One CSV file per table in the folder (made when it is not there); the files written, in the tables' order."""
    out = Path(folder); out.mkdir(parents=True, exist_ok=True); written = []
    for name, (head, rows) in tables.items():
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(head); w.writerows(rows)
        written.append(str(out / f"{name}.csv"))
    return written
