"""Families in a set of tracks: who descends from whom, where a family sits in the frame, and the layout of its tree.

Plain Python on the tool's track dicts ({"ID", "t", "x", "y", "parent", "children", ...}); no Qt, no drawing. The lineage
window, the family film and the tests all use these functions, so the family shown on screen is the family that is exported.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def index_tracks(tracks: Iterable[dict]) -> Dict[int, dict]:
    """Tracks by ID."""
    return {int(x["ID"]): x for x in tracks}


def children_of(by_id: Dict[int, dict]) -> Dict[int, List[int]]:
    """Children of every track, from the 'children' lists when present, else from the 'parent' fields."""
    kids: Dict[int, List[int]] = {}
    for i, x in by_id.items():
        for c in x.get("children") or []:
            if int(c) in by_id:
                kids.setdefault(i, []).append(int(c))
    if not kids:
        for i, x in by_id.items():
            p = x.get("parent")
            if p not in (None, 0, -1) and int(p) in by_id:
                kids.setdefault(int(p), []).append(i)
    for v in kids.values():
        v.sort()
    return kids


def parent_of(by_id: Dict[int, dict], i: int) -> Optional[int]:
    p = by_id[i].get("parent")
    return int(p) if p not in (None, 0, -1) and int(p) in by_id else None


def root_of(by_id: Dict[int, dict], i: int) -> int:
    """The founder of the family that track i belongs to."""
    seen = {i}
    while True:
        p = parent_of(by_id, i)
        if p is None or p in seen:
            return i
        seen.add(p)
        i = p


def family(by_id: Dict[int, dict], root: int, kids: Optional[Dict[int, List[int]]] = None) -> List[int]:
    """root and all its descendants, depth first."""
    kids = children_of(by_id) if kids is None else kids
    out, stack = [], [int(root)]
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(reversed(kids.get(i, [])))
    return out


def families_by_size(by_id: Dict[int, dict], min_frames: int = 1) -> List[Tuple[int, int]]:
    """(root, family size) for every founder with at least min_frames frames, largest family first, then longest root."""
    kids = children_of(by_id)
    roots = [i for i, x in by_id.items() if parent_of(by_id, i) is None and len(x.get("t", [])) >= min_frames]
    sizes = [(i, len(family(by_id, i, kids))) for i in roots]
    sizes.sort(key=lambda r: (-r[1], -len(by_id[r[0]].get("t", [])), r[0]))
    return sizes


def tree_rows(by_id: Dict[int, dict], root: int, kids: Optional[Dict[int, List[int]]] = None) -> Dict[int, float]:
    """Row of every track in the tree: leaves one apart in tree order, a parent at the mean of its children."""
    kids = children_of(by_id) if kids is None else kids
    rows: Dict[int, float] = {}
    nxt = [0.0]

    def place(i):
        ch = kids.get(i, [])
        if not ch:
            rows[i] = nxt[0]
            nxt[0] += 1.0
        else:
            for c in ch:
                place(c)
            rows[i] = float(np.mean([rows[c] for c in ch]))

    place(int(root))
    return rows


def span(by_id: Dict[int, dict], i: int) -> Tuple[int, int]:
    ts = by_id[i]["t"]
    return int(ts[0]), int(ts[-1])


def family_span(by_id: Dict[int, dict], fam: Sequence[int]) -> Tuple[int, int]:
    s = [span(by_id, i) for i in fam]
    return min(a for a, _ in s), max(b for _, b in s)


def alive_at(by_id: Dict[int, dict], fam: Sequence[int], t: int) -> List[int]:
    return [i for i in fam if int(t) in {int(v) for v in by_id[i]["t"]}]


def point_at(by_id: Dict[int, dict], i: int, t: int) -> Optional[Tuple[float, float]]:
    """(x, y) of track i at frame t, or None when it is not there."""
    ts = [int(v) for v in by_id[i]["t"]]
    if int(t) not in ts:
        return None
    k = ts.index(int(t))
    return float(by_id[i]["x"][k]), float(by_id[i]["y"][k])


def trail(by_id: Dict[int, dict], i: int, t: int, n: int) -> List[Tuple[float, float]]:
    """The last n points of track i up to and including frame t (empty when the track is not at t)."""
    ts = [int(v) for v in by_id[i]["t"]]
    if int(t) not in ts:
        return []
    k = ts.index(int(t))
    x, y = by_id[i]["x"], by_id[i]["y"]
    return [(float(x[j]), float(y[j])) for j in range(max(0, k - n + 1), k + 1)]


def camera_crop(by_id: Dict[int, dict], fam: Sequence[int], t: int, shape: Tuple[int, int], half: int = 110) -> Tuple[slice, slice]:
    """A (rows, cols) window of 2*half centred on the family's cells at frame t, kept inside the frame."""
    pts = [p for p in (point_at(by_id, i, t) for i in fam) if p is not None]
    if not pts:
        # the family is not present: centre on its nearest frame in time
        best = None
        for i in fam:
            for k, tt in enumerate(by_id[i]["t"]):
                d = abs(int(tt) - int(t))
                if best is None or d < best[0]:
                    best = (d, float(by_id[i]["x"][k]), float(by_id[i]["y"][k]))
        pts = [(best[1], best[2])] if best else [(shape[1] / 2, shape[0] / 2)]
    cx = float(np.mean([p[0] for p in pts])); cy = float(np.mean([p[1] for p in pts]))
    h = min(int(half), shape[0] // 2, shape[1] // 2)
    y0 = int(np.clip(round(cy - h), 0, shape[0] - 2 * h)); x0 = int(np.clip(round(cx - h), 0, shape[1] - 2 * h))
    return slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h)


def owner_map(by_id: Dict[int, dict], mask: np.ndarray, t: int) -> Dict[int, int]:
    """Segmentation label -> track ID at frame t, from each track's point at t (the label under the point)."""
    own: Dict[int, int] = {}
    h, w = mask.shape
    for i, x in by_id.items():
        p = point_at(by_id, i, t)
        if p is None:
            continue
        px, py = int(round(p[0])), int(round(p[1]))
        if 0 <= px < w and 0 <= py < h:
            lab = int(mask[py, px])
            if lab > 0 and lab not in own:
                own[lab] = i
    return own


def track_at_point(by_id: Dict[int, dict], t: int, x: float, y: float, max_dist: float = 15.0) -> Optional[int]:
    """The track whose point at frame t is nearest to (x, y), within max_dist pixels; None when there is none."""
    best = None
    for i in by_id:
        p = point_at(by_id, i, t)
        if p is None:
            continue
        d = float(np.hypot(p[0] - x, p[1] - y))
        if d <= max_dist and (best is None or d < best[0]):
            best = (d, i)
    return None if best is None else best[1]


def divisions(by_id: Dict[int, dict], fam: Sequence[int], kids: Optional[Dict[int, List[int]]] = None) -> List[int]:
    """Frames at which a cell of the family divides (the last frame of each parent), sorted."""
    kids = children_of(by_id) if kids is None else kids
    return sorted({span(by_id, i)[1] for i in fam if kids.get(i)})


def generations(by_id: Dict[int, dict], root: int, fam: Sequence[int]) -> Dict[int, List[int]]:
    """Generation number (root = 0) -> tracks."""
    gens: Dict[int, List[int]] = {}
    for i in fam:
        g, p = 0, parent_of(by_id, i)
        while p is not None:
            g += 1
            p = parent_of(by_id, p)
        gens.setdefault(g, []).append(i)
    return gens


def interdivision_frames(by_id: Dict[int, dict], fam: Sequence[int], kids: Optional[Dict[int, List[int]]] = None) -> List[int]:
    """Frames from birth to division for the cells of the family that were born and divided inside the recording."""
    kids = children_of(by_id) if kids is None else kids
    out = []
    for i in fam:
        if kids.get(i) and parent_of(by_id, i) is not None:
            a, b = span(by_id, i)
            out.append(b - a + 1)
    return out
